"""Independent CPU verification of the frozen one-patient CDI extraction.
No target inference, feature extraction, fitting, AUC or significance testing.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "_reports/cvpr_u_pilot_v1_001"
MODULES = ("denoising_loss", "secmi", "pia", "gradient_masking", "multiple_loss", "noise_optim")
NAMES = (["Denoising Loss", "SecMI$_{stat}$", "PIA", "PIAN"]
         + [f"Gradient Masking_{i}" for i in range(10)]
         + [f"Multiple Loss_{i}" for i in range(10)]
         + ["Noise Optimization_0", "Noise Optimization_1"])
IMAGES = ["00014393_002.png", "00014393_006.png", "00014393_001.png", "00014393_004.png"]
SHAPE = (1, 4, 32, 32)
UNIT = np.finfo(np.float32).eps / 2
TINY = np.finfo(np.float32).tiny


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, label):
    if not condition:
        raise AssertionError(label)


def array(value, label, shape=None, dtype=None):
    require(isinstance(value, torch.Tensor), label + ": not a saved tensor")
    require(value.device.type == "cpu", label + ": CPU tensor required")
    if dtype is not None:
        require(value.dtype == dtype, label + ": dtype")
    a = value.detach().contiguous().numpy()
    if shape is not None:
        require(a.shape == tuple(shape), label + ": shape")
    require(np.isfinite(a).all(), label + ": nonfinite")
    return a


def norm(value, p=2):
    a = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(math.fsum(float(abs(x)) ** p for x in a) ** (1 / p))


def scalar(value, expected, label, dimensions=4096, double=False):
    value, expected = float(value), float(expected)
    require(math.isfinite(value) and math.isfinite(expected), label + ": nonfinite scalar")
    unit = np.finfo(np.float64).eps / 2 if double else UNIT
    gamma = (dimensions + 32) * unit / (1 - (dimensions + 32) * unit)
    bound = gamma * max(abs(expected), abs(value)) + 16 * (np.finfo(np.float64).tiny if double else TINY)
    require(abs(value - expected) <= bound, f"{label}: {value} != {expected}; bound={bound}")
    return {"reported": value, "cpu_float64": expected,
            "absolute_error": abs(value - expected), "rounding_bound": bound}


def linear_array(actual, expected, magnitude, label):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    require(actual.shape == expected.shape, label + ": shape")
    # Conservative fixed local FP32 arithmetic envelope; not an efficacy tolerance.
    bound = 32 * UNIT * np.maximum(np.asarray(magnitude, dtype=np.float64), TINY) + 16 * TINY
    require(np.all(np.abs(actual - expected) <= bound), label + ": FP32 arithmetic mismatch")


def same(a, b, label):
    require(np.array_equal(a, b), label + ": exact saved-value mismatch")


def noised(z, epsilon, alpha):
    return math.sqrt(alpha) * np.asarray(z, dtype=np.float64) + math.sqrt(1 - alpha) * np.asarray(epsilon, dtype=np.float64)


def check_noising(actual, z, epsilon, alpha, label):
    expected = noised(z, epsilon, alpha)
    magnitude = math.sqrt(alpha) * np.abs(z) + math.sqrt(1 - alpha) * np.abs(epsilon)
    linear_array(actual, expected, magnitude, label)


def check_prediction(record, timestep, alphas, use_grad, label):
    require(record["timestep"] == timestep, label + ": timestep")
    require(float(record["alpha"]) == float(alphas[timestep]), label + ": alpha")
    x = array(record["input"], label + ".input", SHAPE, torch.float32)
    eps = array(record["epsilon"], label + ".epsilon", SHAPE, torch.float32)
    pred = array(record["raw_prediction"], label + ".prediction", SHAPE, torch.float32)
    same(eps, pred, label + ": epsilon parameterization")
    require(record["grad_enabled"] is use_grad, label + ": grad mode")
    if use_grad:
        array(record["input_gradient"], label + ".input_gradient", SHAPE, torch.float32)
        array(record["epsilon_gradient"], label + ".epsilon_gradient", SHAPE, torch.float32)
    return x, eps


def check_loss_gradient(record, noise, label):
    pred = array(record["epsilon"], label + ".epsilon")
    error = pred.astype(np.float64) - np.asarray(noise, dtype=np.float64)
    size = norm(error)
    require(size > 0, label + ": zero residual gives unavailable normalized-gradient check")
    expected = error / size
    gradient = array(record["epsilon_gradient"], label + ".epsilon_gradient", SHAPE)
    # Norm reduction plus per-element subtraction/division, with absolute allowance near zero.
    bound = ((error.size + 64) * UNIT) * np.maximum(np.abs(expected), UNIT) + 32 * TINY
    require(np.all(np.abs(gradient.astype(np.float64) - expected) <= bound), label + ": L2 output gradient")
    return norm(array(record["input_gradient"], label + ".input_gradient"))


def recurrence(x, epsilon, alpha_in, alpha_out):
    denoised = (x.astype(np.float64) - math.sqrt(1 - alpha_in) * epsilon.astype(np.float64)) / math.sqrt(alpha_in)
    answer = math.sqrt(alpha_out) * denoised + math.sqrt(1 - alpha_out) * epsilon
    magnitude = (math.sqrt(alpha_out / alpha_in)
                 * (np.abs(x) + math.sqrt(1 - alpha_in) * np.abs(epsilon))
                 + math.sqrt(1 - alpha_out) * np.abs(epsilon))
    return answer, magnitude


def verify_image(row, raw, latent_cache, old_records, old_directory):
    iid = row["image_id"]
    require(row["patient_id"] == "14393" and row["eval_role"] == "fit", iid + ": cohort role")
    require(row["model"] == "model_1" and row["member"] == 1, iid + ": model/member")
    require(row["stream"] == raw["stream"] == "primary", iid + ": stream")
    require(raw["image_id"] == iid, iid + ": raw identity")
    require(row["prediction_type"] == "epsilon" and row["batch_size"] == 1, iid + ": backend")
    require(row["weights_unchanged"] is True and row["unet_training"] is False, iid + ": freeze/eval")
    is_e = iid in IMAGES[:2]
    require(row["scenario"] == ("E" if is_e else "U"), iid + ": scenario")
    require(row["record_role"] == ("train_candidate" if is_e else "U_observed"), iid + ": image role")
    require(row["image_member"] == int(is_e), iid + ": image membership")
    require(row["feature_names"] == NAMES and len(row["features"]) == 26, iid + ": feature order")
    require(set(raw["predictions"]) == set(MODULES), iid + ": prediction modules")
    require(set(raw["module_outputs"]) == set(MODULES), iid + ": output modules")
    require(set(row["modules"]) == set(MODULES), iid + ": result modules")
    z = array(raw["latent"], iid + ".latent", SHAPE, torch.float32)
    same(z, latent_cache.float().numpy(), iid + ": original cached latent")
    alphas = array(raw["alphas"], iid + ".alphas", (1000,), torch.float32)
    require(np.all((alphas > 0) & (alphas <= 1)), iid + ": schedule range")
    require(np.all(np.diff(alphas.astype(np.float64)) <= 0), iid + ": schedule monotonicity")
    packets, outputs = raw["predictions"], raw["module_outputs"]
    noises = {}
    for module in MODULES:
        draws = raw["noise_draws"][module]
        noises[module] = []
        for index, draw in enumerate(draws):
            eps = array(draw["noise"], f"{iid}.{module}.noise{index}", SHAPE, torch.float32)
            require(isinstance(draw["seed"], int) and 0 <= draw["seed"] < 2**63,
                    f"{iid}.{module}: seed metadata")
            payload = f"cdi-v1|260914|{iid}|primary|{module}|{index}".encode("utf-8")
            expected_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)
            require(draw["seed"] == expected_seed and draw["draw"] == index, f"{iid}.{module}: fixed seed")
            generator = torch.Generator(device="cpu").manual_seed(expected_seed)
            regenerated = torch.randn(SHAPE, generator=generator, device="cpu", dtype=torch.float32).numpy()
            same(eps, regenerated, f"{iid}.{module}: independently regenerated CPU noise")
            require(hashlib.sha256(eps.tobytes()).hexdigest() == draw["sha256"],
                    f"{iid}.{module}: noise hash")
            noises[module].append(eps)
        require(row["modules"][module]["noise_draw_count"] == len(draws), module + ": draw counter")
    require([len(noises[m]) for m in MODULES] == [5, 0, 0, 1, 1, 1], iid + ": noise sharing contract")

    features, checks = [], []
    module = "denoising_loss"
    require(len(packets[module]) == 5, iid + ": DL calls")
    dl = []
    for j, (record, noise) in enumerate(zip(packets[module], noises[module])):
        x, eps = check_prediction(record, 100, alphas, False, f"{iid}.DL{j}")
        check_noising(x, z, noise, float(alphas[100]), f"{iid}.DL{j}.noising")
        dl.append(norm(eps.astype(np.float64) - noise))
    out = array(outputs[module], iid + ".DL.output", (1, 5, 1))
    for j in range(5):
        checks.append(scalar(out[0, j, 0], dl[j], f"{iid}.DL{j}.norm"))
    features.append(math.fsum(dl) / 5)

    module = "secmi"
    stages = [(t, t + 10) for t in range(0, 100, 10)] + [(100, 110), (110, 100)]
    require(len(packets[module]) == 12, iid + ": SecMI calls")
    previous, last_expected, last_magnitude = z, None, None
    for j, ((t, target), record) in enumerate(zip(stages, packets[module])):
        x, eps = check_prediction(record, t, alphas, False, f"{iid}.SecMI{j}")
        if j == 0:
            same(x, previous, iid + ": SecMI starts from cache")
        else:
            linear_array(x, last_expected, last_magnitude, f"{iid}.SecMI{j}.recurrence")
        last_expected, last_magnitude = recurrence(x, eps, float(alphas[t]), float(alphas[target]))
    det = array(raw["secmi"]["z_det"], iid + ".z_det", SHAPE, torch.float32)
    recon = array(raw["secmi"]["z_recon"], iid + ".z_recon", SHAPE, torch.float32)
    same(det, array(packets[module][10]["input"], "SecMI stage100"), iid + ": zdet")
    linear_array(recon, last_expected, last_magnitude, iid + ": final SecMI recurrence")
    out = array(outputs[module], iid + ".SecMI.output", (1, 2, 4, 32, 32))
    same(out[:, 0], det, iid + ": SecMI output det")
    same(out[:, 1], recon, iid + ": SecMI output recon")
    secmi = norm(det.astype(np.float64) - recon)
    features.append(secmi)
    old = old_records[iid]
    require(old["model"] == "model_1" and old["patient_id"] == "14393" and old["eval_role"] == "fit",
            iid + ": prior SecMI metadata")
    require(old["scenario"] == row["scenario"] and old["score"] == -old["l2"], iid + ": prior polarity")
    old_path = old_directory / old["tensor_path"]
    require(digest(old_path) == old["tensor_sha256"], iid + ": prior SecMI tensor hash")
    prior_packet = torch.load(old_path, map_location="cpu", weights_only=True)
    # Existing SecMI states are recorded separately from this execution.
    if isinstance(prior_packet, dict):
        old_det = prior_packet.get("z_det")
        old_recon = prior_packet.get("z_recon", prior_packet.get("z_step"))
        require(old_det is not None and old_recon is not None, iid + ": prior state schema")
        old_det = array(old_det, iid + ".prior.zdet").reshape(SHAPE)
        old_recon = array(old_recon, iid + ".prior.zrecon").reshape(SHAPE)
    else:
        old_array = array(prior_packet, iid + ".prior.states")
        require(old_array.shape == (1, 2, 4, 32, 32), iid + ": prior state shape")
        old_det, old_recon = old_array[:, 0], old_array[:, 1]
    # Same frozen input/backend supports an exact deterministic-state crosscheck.
    same(det, old_det, iid + ": independently recorded SecMI zdet")
    same(recon, old_recon, iid + ": independently recorded SecMI zrecon")
    checks.append(scalar(old["l2"], secmi, iid + ": prior positive SecMI norm"))

    module = "pia"
    require(len(packets[module]) == 3, iid + ": PIA shared eps0 calls")
    x0, eps0 = check_prediction(packets[module][0], 0, alphas, False, iid + ".PIA0")
    same(x0, z, iid + ": PIA starts at clean latent")
    xp, predp = check_prediction(packets[module][1], 200, alphas, False, iid + ".PIA")
    check_noising(xp, z, eps0, float(alphas[200]), iid + ".PIA.noising")
    normalization = z.size * math.sqrt(math.pi / 2) / norm(eps0, 1)
    normalized = eps0.astype(np.float64) * normalization
    xn, predn = check_prediction(packets[module][2], 200, alphas, False, iid + ".PIAN")
    # Source normalization is performed in FP32, then used in the noising arithmetic.
    expected_n = noised(z, normalized, float(alphas[200]))
    magnitude_n = math.sqrt(float(alphas[200])) * np.abs(z) + math.sqrt(1-float(alphas[200])) * np.abs(normalized)
    # Sequential L1 summation envelope is also included for normalized eps0.
    linear_array(xn, expected_n, magnitude_n * (z.size + 32), iid + ".PIAN.normalization/noising")
    pia = norm(eps0.astype(np.float64) - predp, 5)
    pian = norm(normalized - predn, 5)
    out = array(outputs[module], iid + ".PIA.output", (1, 1, 2))
    checks += [scalar(out[0, 0, 0], pia, iid + ".PIA.output"),
               scalar(out[0, 0, 1], pian, iid + ".PIAN.output")]
    features.extend((pia, pian))

    module = "gradient_masking"
    require(len(packets[module]) == 20, iid + ": GM calls")
    gradients = array(raw["gm"]["gradients"], iid + ".GM.gradients", (1, 10, 4, 32, 32), torch.float32)
    masks = array(raw["gm"]["masks"], iid + ".GM.masks", (1, 10, 4, 32, 32), torch.bool)
    mixed = array(raw["gm"]["mixed_latents"], iid + ".GM.mixed", (1, 10, 4, 32, 32), torch.float32)
    noise = noises[module][0]
    out = array(outputs[module], iid + ".GM.output", (1, 10, 1))
    grad_norms = []
    for j, t in enumerate(range(0, 1000, 100)):
        record = packets[module][j]
        x, eps = check_prediction(record, t, alphas, True, f"{iid}.GM{j}.gradient")
        check_noising(x, z, noise, float(alphas[t]), f"{iid}.GM{j}.gradient.noising")
        gn = check_loss_gradient(record, noise, f"{iid}.GM{j}")
        require(gn > 0, f"{iid}.GM{j}: nonzero input gradient required")
        grad_norms.append(gn)
        same(gradients[:, j], array(record["input_gradient"], "GM recorded grad"),
             f"{iid}.GM{j}: saved gradient identity")
        mask = masks[:, j]
        require(int(mask.sum()) == int(z.size * .2), f"{iid}.GM{j}: top20 count")
        absgrad = np.abs(gradients[:, j])
        require(absgrad[mask].min() >= absgrad[~mask].max(), f"{iid}.GM{j}: top20 order")
        same(mixed[:, j], np.where(mask, noise, z), f"{iid}.GM{j}: clean-latent masking")
        xm, pm = check_prediction(packets[module][10+j], t, alphas, False, f"{iid}.GM{j}.masked")
        same(xm, mixed[:, j], f"{iid}.GM{j}: no second noising")
        value = norm((noise.astype(np.float64) - z - pm)[mask])
        checks.append(scalar(out[0, j, 0], value, f"{iid}.GM{j}.output", dimensions=int(mask.sum())))
        features.append(value)

    module = "multiple_loss"
    require(len(packets[module]) == 10, iid + ": ML calls")
    out = array(outputs[module], iid + ".ML.output", (1, 10, 1))
    noise = noises[module][0]
    for j, t in enumerate(range(0, 1000, 100)):
        x, eps = check_prediction(packets[module][j], t, alphas, False, f"{iid}.ML{j}")
        check_noising(x, z, noise, float(alphas[t]), f"{iid}.ML{j}.noising")
        value = norm(eps.astype(np.float64) - noise)
        checks.append(scalar(out[0, j, 0], value, f"{iid}.ML{j}.output"))
        features.append(value)

    module = "noise_optim"
    no, records = raw["no"], packets[module]
    trace, opt = no["objective_trace"], no["optimizer"]
    nfev = int(opt["nfev"])
    q = int(opt["objective_evaluations"])
    require(nfev == int(opt["njev"]) and nfev >= q == len(trace) >= 1, iid + ": NO evaluation counters")
    require(opt["memoized_evaluation_requests"] == nfev-q, iid + ": NO memoized requests")
    require(len(records) == q + 1, iid + ": NO objective plus exported prediction")
    require(0 <= int(opt["nit"]) <= 5, iid + ": NO iteration budget")
    require(isinstance(opt["success"], bool) and isinstance(opt["message"], str), iid + ": NO termination")
    noise = noises[module][0]
    zt = array(no["initial_noised_latent"], iid + ".NO.zt", SHAPE, torch.float32)
    check_noising(zt, z, noise, float(alphas[100]), iid + ".NO.initial.noising")
    delta = array(no["delta"], iid + ".NO.delta", SHAPE, torch.float64)
    x_trace, no_grad_norms = [], []
    for j, (entry, record) in enumerate(zip(trace, records[:-1])):
        xx = array(entry["x"], f"{iid}.NO{j}.x").reshape(SHAPE)
        gg = array(entry["gradient"], f"{iid}.NO{j}.gradient").reshape(SHAPE)
        if j == 0:
            require(np.all(xx == 0), iid + ": NO initial delta")
        x, eps = check_prediction(record, 100, alphas, True, f"{iid}.NO{j}")
        same(x, (zt.astype(np.float64) + xx).astype(np.float32), f"{iid}.NO{j}: direct objective input")
        value = norm(eps.astype(np.float64) - noise)
        checks.append(scalar(entry["loss"], value, f"{iid}.NO{j}.objective"))
        no_grad_norms.append(check_loss_gradient(record, noise, f"{iid}.NO{j}"))
        same(gg, array(record["input_gradient"], "NO input gradient").astype(gg.dtype),
             f"{iid}.NO{j}: optimizer gradient identity")
        x_trace.append(xx)
    require(max(no_grad_norms) > 0, iid + ": NO nonzero gradient evidence")
    # SciPy preserves the FP32 x0 dtype at objective calls but returns an FP64 x.
    # Match the actual evaluated coordinates, without pretending the exported FP64
    # delta itself was evaluated by the objective.
    matches = [j for j, xx in enumerate(x_trace) if np.array_equal(xx, delta.astype(xx.dtype))]
    require(bool(matches), iid + ": returned NO delta absent from objective trace")
    final_index = min(matches, key=lambda j: abs(float(trace[j]["loss"]) - float(opt["fun"])))
    scalar(opt["fun"], trace[final_index]["loss"], iid + ": optimizer.fun", double=True)
    same(array(opt["jac"], iid + ".NO.finaljac").reshape(SHAPE),
         array(trace[final_index]["gradient"], iid + ".NO.tracejac").reshape(SHAPE),
         iid + ": returned optimizer gradient")
    xf, pf = check_prediction(records[-1], 100, alphas, False, iid + ".NO.exported")
    check_noising(xf, (zt.astype(np.float64) + delta).astype(np.float32), noise,
                  float(alphas[100]), iid + ".NO.exported.second_noising")
    exported = norm(pf.astype(np.float64) - noise)
    perturbation = norm(delta)
    out = array(outputs[module], iid + ".NO.output", (1, 1, 2))
    checks += [scalar(out[0, 0, 0], exported, iid + ".NO.exported.loss"),
               scalar(out[0, 0, 1], perturbation, iid + ".NO.delta.norm", double=True)]
    features.extend((exported, perturbation))
    require(len(features) == 26, iid + ": independent feature length")
    for i, value in enumerate(features):
        checks.append(scalar(row["features"][i], value, f"{iid}.feature{i}",
                             double=i == 25))
    expected_counts = {"denoising_loss": (5, 0), "secmi": (12, 0), "pia": (3, 0),
                       "gradient_masking": (20, 10), "multiple_loss": (10, 0),
                       "noise_optim": (q+1, q)}
    for module, (f, b) in expected_counts.items():
        info = row["modules"][module]
        require(info["forward"] == f and info["backward"] == b, iid + "." + module + ": F/B")
        require(info["output_shape"] == list(outputs[module].shape), iid + "." + module + ": output shape")
        require(info["noise_seeds"] == [e["seed"] for e in raw["noise_draws"][module]]
                and info["noise_sha256"] == [e["sha256"] for e in raw["noise_draws"][module]],
                iid + "." + module + ": exported noise metadata")
        require(math.isfinite(info["seconds"]) and info["seconds"] >= 0, iid + ": timing")
    require(row["modules"]["noise_optim"]["optimizer"] == {k:v for k,v in opt.items() if k != "jac"},
            iid + ": exported optimizer metadata")
    exported_trace = row["modules"]["noise_optim"]["objective_trace"]
    require(len(exported_trace) == q, iid + ": exported trace length")
    for j, (summary, entry) in enumerate(zip(exported_trace, trace)):
        require(summary["evaluation"] == j and summary["loss"] == entry["loss"], iid + ": trace identity")
        g = array(entry["gradient"], iid + ": optimizer gradient summary")
        scalar(summary["gradient_l2"], norm(g), iid + ": gradient L2 summary", double=g.dtype == np.float64)
        require(summary["gradient_max_abs"] == float(np.max(np.abs(g))), iid + ": gradient max summary")
    require(row["forward"] == 51+q and row["backward"] == 10+q, iid + ": total F/B")
    return {"image_id": iid, "scenario": row["scenario"], "features_cpu_float64": features,
            "scalar_checks": checks, "noise_draws_checked": 8,
            "GM_gradient_norms": grad_norms, "NO_gradient_norms": no_grad_norms,
            "NO_nfev": nfev, "NO_objective_evaluations": q, "NO_memoized_requests": nfev-q,
            "NO_nit": opt["nit"], "NO_status": opt["status"],
            "NO_success": opt["success"], "NO_message": opt["message"],
            "NO_initial_objective": trace[0]["loss"], "NO_final_objective": opt["fun"],
            "NO_exported_second_noising_loss": exported, "NO_final_trace_index": final_index,
            "NO_final_objective_coordinate_dtype": str(x_trace[final_index].dtype),
            "NO_returned_vs_evaluated_delta_max_abs": float(np.max(np.abs(delta-x_trace[final_index]))),
            "prior_SecMI_states_exact": True, "forward": row["forward"], "backward": row["backward"]}


def verify(directory):
    begun = time.perf_counter()
    torch.set_num_threads(2)
    protocol, execution = load_json(directory/"protocol.json"), load_json(directory/"execution.json")
    results = load_json(directory/"results.json")
    require(execution["protocol_sha256"] == digest(directory/"protocol.json"), "protocol hash")
    require(execution["results_sha256"] == digest(directory/"results.json"), "results hash")
    require(execution["raw_sha256"] == digest(directory/"raw.pt"), "raw hash")
    contract = protocol["contract"]
    require(protocol["contract_sha256"] == digest(protocol["contract_path"]), "external contract hash")
    require(load_json(protocol["contract_path"]) == contract, "embedded contract")
    expected = dict(current_step=2, patient_id="14393", eval_role="fit", model="model_1",
                    checkpoint_step=1000, master_seed=260914, stream="primary", batch_size=1,
                    prediction_type="epsilon", dtype="float32", prompt="a frontal chest radiograph",
                    perform_fitting=False, perform_performance_evaluation=False,
                    expected_images=4, code_policy="released_code_literal")
    for key, value in expected.items():
        require(contract[key] == value, "contract." + key)
    require(contract["image_order"] == IMAGES, "contract image order")
    frozen = contract["frozen_sha256"]
    require(len(frozen) > 0, "missing frozen hashes")
    for filename, expected_hash in frozen.items():
        require(Path(filename).is_absolute() and digest(filename) == expected_hash, "frozen hash: " + filename)
    source = protocol["source_bindings"]
    require(source["repository"] == "https://github.com/sprintml/copyrighted_data_identification"
            and source["commit"] == "dcd62258b0b3fde05d52aaecfade3b5f4c09507a", "upstream identity")
    require(source["feature_names"] == NAMES, "source 26D order")
    source_root = Path(source["source_root"]).resolve()
    required_sources = {f"src/attacks/features_extraction/{m}.py" for m in MODULES}
    required_sources |= {"src/models/GeneralLatentDiffusionWrapper.py",
                         "src/attacks/scores_computation/secmi.py",
                         "src/attacks/scores_computation/denoising_loss.py",
                         "src/attacks/scores_computation/pia.py", "conf/attack/cdi.yaml"}
    required_sources |= {"src/models/CompVisLatentDiffusionWrapper.py",
                         "src/attacks/features_extraction/cdi.py"}
    required_sources |= {f"conf/attack/{m}.yaml" for m in
                         ("denoising_loss", "secmi_stat", "pia", "pian", "gradient_masking", "multiple_loss", "noise_optim")}
    require(len(required_sources) == 20 and required_sources == set(source["files_sha256"]),
            "incomplete/extra upstream binding")
    for relative, value in source["files_sha256"].items():
        filename = (source_root/relative).resolve()
        require(filename.is_relative_to(source_root) and digest(filename) == value, "upstream source hash: " + relative)
        require(frozen[str(filename)] == value, "upstream absent from frozen contract: " + relative)
    for required in (Path(__file__), ROOT/"u_patient_audit/cdi_adapter.py",
                     ROOT/"u_patient_audit/run_cdi_kernel.py", RUN/"cache/cache.pt",
                     RUN/"cohort/evaluation_images.csv", RUN/"training_coverage_v2/model_1/step_1000.pt"):
        require(str(required.resolve()) in frozen, "required frozen input absent: " + str(required))
    require(contract["checkpoint_sha256"] == digest(RUN/"training_coverage_v2/model_1/step_1000.pt"), "checkpoint")
    cohort = list(csv.DictReader((RUN/"cohort/evaluation_images.csv").open(encoding="utf-8-sig")))
    cohort_rows = {r["image_id"]: r for r in cohort if r["patient_id"] == "14393"}
    require(set(cohort_rows) == set(IMAGES), "actual cohort image set")
    train = list(csv.DictReader((RUN/"cohort/model_1_train.csv").open(encoding="utf-8-sig")))
    train_ids = {r["image_id"] for r in train}
    require(all(i in train_ids for i in IMAGES[:2]) and all(i not in train_ids for i in IMAGES[2:]),
            "actual E/U training manifest")
    checkpoint = torch.load(RUN/"training_coverage_v2/model_1/step_1000.pt", map_location="cpu", weights_only=True)
    exposures = checkpoint["exposures"]
    require(checkpoint["step"] == 1000 and set(exposures) == train_ids
            and sum(int(v) for v in exposures.values()) == 4000, "actual optimizer exposure ledger")
    for iid in IMAGES:
        require(cohort_rows[iid]["eval_role"] == "fit" and cohort_rows[iid]["assignment_group"] == "A", "cohort role")
    require([r["image_id"] for r in results] == IMAGES, "result order")
    require(len(results) == 4, "result count")
    raw = torch.load(directory/"raw.pt", map_location="cpu", weights_only=True)
    require(set(raw) == set(IMAGES), "raw image keys")
    cache = torch.load(RUN/"cache/cache.pt", map_location="cpu", weights_only=True)
    require(cache["audit_prompt"] == contract["prompt"], "cached generic prompt")
    hidden = cache["hidden"][cache["audit_prompt"]].detach().cpu().float().contiguous().numpy()
    hidden_sha = hashlib.sha256(hidden.tobytes()).hexdigest()
    for row in results:
        iid = row["image_id"]
        require(raw[iid]["hidden_sha256"] == hidden_sha, iid + ": cached conditioning tensor hash")
        actual = int(exposures.get(iid, 0))
        require(row["actual_training_exposures"] == actual
                and (actual > 0 if iid in IMAGES[:2] else actual == 0), iid + ": optimizer exposure")
    old_directory = RUN/"baseline_screen_20260914/secmi_v1"
    old_protocol = load_json(old_directory/"protocol.json")
    require(old_protocol["precision"] == "fp32" and old_protocol["batch_size"] == 1
            and old_protocol["generic_prompt"] == contract["prompt"], "prior SecMI compatible setting")
    require(old_protocol["checkpoint_sha256"]["model_1"] == contract["checkpoint_sha256"], "prior SecMI checkpoint")
    require(old_protocol["cache_sha256"] == digest(RUN/"cache/cache.pt"), "prior SecMI cache")
    old_records = {r["image_id"]: r for r in load_json(old_directory/"image_scores.json")
                   if r["model"] == "model_1" and str(r["patient_id"]) == "14393"}
    require(set(old_records) == set(IMAGES), "prior SecMI image set")
    checked = [verify_image(row, raw[row["image_id"]], cache["latents"][row["image_id"]],
                            old_records, old_directory) for row in results]
    for field in ("adapter_tensors_exact_equal", "parameter_versions_and_no_weight_gradients", "frozen_files_unchanged"):
        require(execution[field] is True, "execution." + field)
    require(execution["images"] == 4 and execution["patients"] == 1 and execution["model"] == "model_1", "execution cohort")
    require(execution["performance_evaluation"] is False and execution["attack_fitting"] is False, "scope")
    require(execution["forward"] == sum(r["forward"] for r in checked), "total forward")
    require(execution["backward"] == sum(r["backward"] for r in checked), "total backward")
    require(execution["vae_forward"] == execution["text_encoder_forward"] == 0, "cache-only encoding")
    return {"status": "PASS_SAVED_KERNEL_ARITHMETIC_AND_INTEGRITY", "current_step": 2,
            "independent_patients": 1, "images": 4, "features_checked": 104, "image_checks": checked,
            "forward": execution["forward"], "backward": execution["backward"],
            "frozen_files_checked": len(frozen),
            "independent_GPU_execution": False, "input_gradient_finite_difference_verified": False,
            "weight_invariance_scope": "producer pre/post tensor and parameter-version/gradient records, plus frozen file hashes",
            "noise_generation_scope": "32 CPU Gaussian draws independently regenerated from fixed module/image/stream seeds; exact values and hashes; no CUDA RNG",
            "no_fitting_or_performance_analysis": True, "attack_efficacy_or_causal_claim": False,
            "automatic_followup": False, "seconds_cpu": time.perf_counter()-begun,
            "protocol_sha256": digest(directory/"protocol.json"),
            "results_sha256": digest(directory/"results.json"), "raw_sha256": digest(directory/"raw.pt"),
            "verification_code_sha256": digest(__file__),
            "rounding_policy": "FP32 sequential norm-reduction gamma bound; fixed local linear-operation envelope; FP64 norm bound for NO delta"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=RUN/"baseline_screen_20260914/cdi_kernel_v1")
    parser.add_argument("--output-name", default="verification.json")
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    output = directory/args.output_name
    require(Path(args.output_name).name == args.output_name, "output must be a filename")
    if output.exists():
        raise FileExistsError(output)
    try:
        result = verify(directory)
    except Exception as exc:
        result = {"status": "FAIL_INDEPENDENT_KERNEL_VERIFICATION", "error": repr(exc),
                  "traceback": traceback.format_exc(), "independent_GPU_execution": False,
                  "attack_efficacy_or_causal_claim": False, "verification_code_sha256": digest(__file__)}
        with output.open("x", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
        raise
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k != "image_checks"},
                     ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
