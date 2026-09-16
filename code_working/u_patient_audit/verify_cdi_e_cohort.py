"""Independent CPU verification of E-only literal CDI packets.
Numeric functions copied from frozen verify_cdi_u_cohort.py; only E metadata
and exposure/reuse scope are changed. No producer imports, GPU or fitting.
"""
import argparse
import ast
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
KERNEL_HELPER_SHA256 = "57b7719c54fee4c32c2d92e29535ce44aceb5b784669b4be2fff43aa6dd7d342"
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
    require(row["eval_role"] in ("fit", "selection"), iid + ": cohort role")
    require(row["model"] in ("model_1", "model_2") and row["member"] in (0, 1), iid + ": model/member")
    require(row["stream"] == raw["stream"] == "primary", iid + ": stream")
    require(raw["image_id"] == iid, iid + ": raw identity")
    require(row["prediction_type"] == "epsilon" and row["batch_size"] == 1, iid + ": backend")
    require(row["weights_unchanged"] is True and row["unet_training"] is False, iid + ": freeze/eval")
    is_e = True
    require(row["scenario"] == ("E" if is_e else "U"), iid + ": scenario")
    require(row["record_role"] == ("train_candidate" if is_e else "U_observed"), iid + ": image role")
    require(row["image_member"] == row["member"], iid + ": image membership")
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
    prior_compared = iid in old_records
    if prior_compared:
        old = old_records[iid]
        require(old["model"] == row["model"] and old["patient_id"] == row["patient_id"] and old["eval_role"] == row["eval_role"],
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
            "scalar_checks": checks, "noise_draws_checked": 8, "model": row["model"], "patient_id": row["patient_id"], "eval_role": row["eval_role"],
            "GM_gradient_norms": grad_norms, "NO_gradient_norms": no_grad_norms,
            "NO_nfev": nfev, "NO_objective_evaluations": q, "NO_memoized_requests": nfev-q,
            "NO_nit": opt["nit"], "NO_status": opt["status"],
            "NO_success": opt["success"], "NO_message": opt["message"],
            "NO_initial_objective": trace[0]["loss"], "NO_final_objective": opt["fun"],
            "NO_exported_second_noising_loss": exported, "NO_final_trace_index": final_index,
            "NO_final_objective_coordinate_dtype": str(x_trace[final_index].dtype),
            "NO_returned_vs_evaluated_delta_max_abs": float(np.max(np.abs(delta-x_trace[final_index]))),
            "prior_SecMI_states_exact": True if prior_compared else None, "forward": row["forward"], "backward": row["backward"]}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def inside(directory, relative):
    path = (directory / relative).resolve()
    require(path.is_relative_to(directory.resolve()), "output path escapes run directory")
    return path


def exact_tree(actual, expected, label):
    if isinstance(expected, torch.Tensor):
        require(isinstance(actual, torch.Tensor) and actual.dtype == expected.dtype
                and actual.shape == expected.shape and torch.equal(actual, expected), label + ": tensor")
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + ": keys")
        for key in expected:
            exact_tree(actual[key], expected[key], label + "." + str(key))
    elif isinstance(expected, (list, tuple)):
        require(type(actual) is type(expected) and len(actual) == len(expected), label + ": sequence")
        for index, (a, b) in enumerate(zip(actual, expected)):
            exact_tree(a, b, label + f"[{index}]")
    else:
        require(type(actual) is type(expected) and actual == expected, label + ": value")


def frozen_inputs(contract):
    check_arithmetic_source()
    frozen = contract["frozen_sha256"]
    require(frozen and all(Path(p).is_absolute() for p in frozen), "absolute frozen input bindings")
    for path, value in frozen.items():
        require(digest(path) == value, "frozen input/source changed: " + path)
    require(digest(ROOT/"u_patient_audit/verify_cdi_kernel.py") == KERNEL_HELPER_SHA256,
            "arithmetic-copy source changed")
    required = [ROOT/"u_patient_audit/verify_cdi_kernel.py",
                ROOT/"u_patient_audit/run_cdi_e_cohort.py", ROOT/"u_patient_audit/cdi_adapter.py",
                RUN/"cache/cache.pt", RUN/"cache/summary.json", RUN/"cohort/lock.json",
                RUN/"cohort/evaluation_images.csv"]
    for model in ("model_1", "model_2"):
        required += [RUN/f"cohort/{model}_train.csv", RUN/f"training_coverage_v2/{model}/step_1000.pt"]
    for path in required:
        require(str(path.resolve()) in frozen, "required frozen dependency absent: " + str(path))
    return frozen


def original_cohort():
    rows = read_csv(RUN/"cohort/evaluation_images.csv")
    wanted = [r for r in rows if r["eval_role"] in ("fit", "selection") and r["record_role"] == "train_candidate"]
    require(len(wanted) == 240 and len({r["image_id"] for r in wanted}) == 240, "240 unique E images")
    patients = {}
    for row in wanted:
        pid = row["patient_id"]
        patients.setdefault(pid, []).append(row)
    require(len(patients) == 120 and all(len(v) == 2 for v in patients.values()), "120 patients with E2")
    for role, count in (("fit", 80), ("selection", 40)):
        sub = [rs for rs in patients.values() if rs[0]["eval_role"] == role]
        require(len(sub) == count, role + ": patient count")
        for group in ("A", "B"):
            require(sum(rs[0]["assignment_group"] == group for rs in sub) == count//2, role + ": A/B balance")
    return wanted, {r["image_id"]: r for r in wanted}, patients


def exposure_ledgers(contract):
    ledgers = {}
    for model in ("model_1", "model_2"):
        path = RUN/f"training_coverage_v2/{model}/step_1000.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        train = read_csv(RUN/f"cohort/{model}_train.csv")
        train_ids = {r["image_id"] for r in train}
        require(len(train_ids) == len(train), model + ": duplicate manifest image")
        exposures = checkpoint["exposures"]
        require(checkpoint["step"] == 1000 and set(exposures) == train_ids
                and sum(int(v) for v in exposures.values()) == 4000, model + ": actual exposure ledger")
        require(all(int(v) > 0 for v in exposures.values()), model + ": all actual training exposures positive")
        by_patient = {}
        for row in train:
            by_patient[row["patient_id"]] = by_patient.get(row["patient_id"], 0) + int(exposures[row["image_id"]])
        ledgers[model] = dict(images=exposures, patients=by_patient, sha256=digest(path), train_ids=train_ids)
        del checkpoint
    return ledgers


def prior_secmi(ledgers, cache_hash, frozen):
    directory = RUN/"baseline_screen_20260914/secmi_v1"
    for name in ("protocol.json", "image_scores.json"):
        require(str((directory/name).resolve()) in frozen, "unbound prior SecMI " + name)
    protocol = load_json(directory/"protocol.json")
    require(protocol["precision"] == "fp32" and protocol["batch_size"] == 1
            and protocol["generic_prompt"] == "a frontal chest radiograph", "prior SecMI backend")
    require(protocol["cache_sha256"] == cache_hash, "prior SecMI cache")
    records = {model: {} for model in ledgers}
    for row in load_json(directory/"image_scores.json"):
        if row["scenario"] != "E":
            continue
        model, iid = row["model"], row["image_id"]
        require(iid not in records[model], "duplicate prior SecMI E image")
        actual = int(ledgers[model]["images"].get(iid, 0))
        require(row["image_member"] == row["member"] == int(actual > 0)
                and row["actual_training_exposures"] == actual, "prior E actual label/exposure")
        records[model][iid] = row
    require(sum(map(len, records.values())) == 192, "prior SecMI exactly 192 eligible E records")
    for model in ledgers:
        require(protocol["checkpoint_sha256"][model] == ledgers[model]["sha256"], "prior SecMI checkpoint")
        values = list(records[model].values())
        require(len(values) == 96, model + ": prior E records")
        require(len({r["patient_id"] for r in values if r["eval_role"] == "fit"}) == 8,
                model + ": prior fit8 coverage")
        require(len({r["patient_id"] for r in values if r["eval_role"] == "selection"}) == 40,
                model + ": prior selection40 coverage")
    return directory, records


def kernel_reference(frozen):
    directory = RUN/"baseline_screen_20260914/cdi_kernel_v1"
    for name in ("protocol.json", "execution.json", "results.json", "raw.pt", "verification.json"):
        require(str((directory/name).resolve()) in frozen, "unbound original kernel " + name)
    execution = load_json(directory/"execution.json")
    require(execution["raw_sha256"] == digest(directory/"raw.pt"), "kernel raw hash")
    require(execution["results_sha256"] == digest(directory/"results.json"), "kernel results hash")
    require(execution["protocol_sha256"] == digest(directory/"protocol.json"), "kernel protocol hash")
    require(load_json(directory/"verification.json")["status"] == "PASS_SAVED_KERNEL_ARITHMETIC_AND_INTEGRITY",
            "kernel arithmetic verification")
    raw = torch.load(directory/"raw.pt", map_location="cpu", weights_only=True)
    rows = {r["image_id"]: r for r in load_json(directory/"results.json")}
    return directory, raw, rows, load_json(directory/"protocol.json")


def verify(directory):
    begun = time.perf_counter()
    torch.set_num_threads(2)
    protocol, execution = load_json(directory/"protocol.json"), load_json(directory/"execution.json")
    contract, contract_path = protocol["contract"], Path(protocol["contract_path"])
    require(contract_path.is_absolute(), "external contract absolute path")
    require(digest(contract_path) == protocol["contract_sha256"] == execution["contract_sha256"], "contract hash")
    require(load_json(contract_path) == contract, "embedded contract")
    require(digest(directory/"protocol.json") == execution["protocol_sha256"], "protocol hash")
    require(digest(directory/"results.json") == execution["results_sha256"], "results hash")
    expected = dict(schema="cdi-e-cohort-contract/v1", current_step=2,
        eval_roles=["fit", "selection"], scenario="E", models=["model_1", "model_2"],
        checkpoint_step=1000, master_seed=260914, stream="primary", batch_size=1,
        prediction_type="epsilon", dtype="float32", prompt="a frontal chest radiograph",
        code_policy="released_code_literal", perform_fitting=False,
        perform_performance_evaluation=False, expected_patients={"fit":80, "selection":40},
        expected_unique_images=240, expected_records=480, expected_reused_records=2, expected_new_records=478)
    for key, value in expected.items():
        require(contract[key] == value, "contract." + key)
    frozen = frozen_inputs(contract)
    lock = load_json(RUN/"cohort/lock.json")
    for name, value in lock["files"].items():
        require(digest(RUN/"cohort"/name) == value, "cohort lock: " + name)
    wanted, cohort, patients = original_cohort()
    image_order = []
    for role in ("fit", "selection"):
        ids = sorted((pid for pid, rows in patients.items() if rows[0]["eval_role"] == role), key=int)
        require(contract["patient_order"][role] == ids, role + ": original patient order")
        for pid in ids:
            image_order.extend(sorted(r["image_id"] for r in patients[pid]))
    require(contract["image_order"] == image_order, "original E image order")
    ledgers = exposure_ledgers(contract)
    require(contract["checkpoint_sha256"] == {m:d["sha256"] for m,d in ledgers.items()}, "checkpoint hashes")
    cache_hash, lock_hash = digest(RUN/"cache/cache.pt"), digest(RUN/"cohort/lock.json")
    cache_meta = load_json(RUN/"cache/summary.json")
    require(cache_meta["cache_sha256"] == cache_hash and cache_meta["cohort_lock_sha256"] == lock_hash, "cache/cohort binding")
    cache = torch.load(RUN/"cache/cache.pt", map_location="cpu", weights_only=True)
    require(cache["audit_prompt"] == contract["prompt"], "cache generic prompt")
    hidden = cache["hidden"][cache["audit_prompt"]].float().contiguous().numpy()
    hidden_sha = hashlib.sha256(hidden.tobytes()).hexdigest()
    old_directory, old_records = prior_secmi(ledgers, cache_hash, frozen)
    kernel_dir, kernel_raw, kernel_rows, kernel_protocol = kernel_reference(frozen)
    require(protocol["source_bindings"] == kernel_protocol["source_bindings"], "same pinned extractor source bindings")
    source = protocol["source_bindings"]
    require(source["feature_names"] == NAMES and len(source["files_sha256"]) == 20, "pinned 26D source order")
    for name, value in source["files_sha256"].items():
        path = (Path(source["source_root"])/name).resolve()
        require(str(path) in frozen and frozen[str(path)] == value, "upstream source binding: " + name)
    kernel_contract_path = Path(kernel_protocol["contract_path"])
    if not kernel_contract_path.is_absolute():
        kernel_contract_path = ROOT/kernel_contract_path
    require(digest(kernel_contract_path) == kernel_protocol["contract_sha256"], "original kernel contract")
    expected_alphas = kernel_raw["00014393_002.png"]["alphas"]
    results = load_json(directory/"results.json")
    require(len(results) == 480, "480 complete records")
    require([(r["model"], r["image_id"]) for r in results]
            == [(m, iid) for m in contract["models"] for iid in image_order], "full fixed record order")
    checked, raw_hashes, pair_hashes, reused = [], {}, {}, []
    plans = protocol["measurement_plan"]
    require(set(plans) == set(ledgers), "measurement-plan target set")
    for model in ledgers:
        require([r["image_id"] for r in plans[model]] == image_order, "measurement plan order")
    for index, row in enumerate(results):
        iid, model, pid = row["image_id"], row["model"], row["patient_id"]
        c = cohort[iid]
        member = int(c["assignment_group"] == ("A" if model == "model_1" else "B"))
        actual_patient = int(ledgers[model]["patients"].get(pid, 0))
        require(pid == c["patient_id"] and row["eval_role"] == c["eval_role"]
                and row["assignment_group"] == c["assignment_group"], iid + ": original cohort metadata")
        require(row["member"] == member == int(actual_patient > 0), iid + ": actual patient participation")
        require(row["patient_training_exposures"] == actual_patient, iid + ": patient exposure total")
        actual_image = int(ledgers[model]["images"].get(iid, 0))
        check_e_truth(row, member, actual_image, actual_patient, iid in ledgers[model]["train_ids"])
        for key, value in plans[model][index % 240].items():
            require(row[key] == value, iid + ": planned " + key)
        require(row["checkpoint_sha256"] == ledgers[model]["sha256"] and row["cache_sha256"] == cache_hash
                and row["cohort_lock_sha256"] == lock_hash and row["contract_sha256"] == protocol["contract_sha256"],
                iid + ": row input bindings")
        require(row["raw_path"] == f"{model}/{Path(iid).stem}.pt", iid + ": exact raw path")
        path = inside(directory, row["raw_path"])
        require(digest(path) == row["raw_sha256"], iid + ": raw hash")
        pair = path.with_suffix(".json")
        require(load_json(pair) == row, iid + ": immutable per-image JSON equality")
        raw_hashes[row["raw_path"]] = row["raw_sha256"]
        pair_hashes[str(pair.relative_to(directory))] = digest(pair)
        raw = torch.load(path, map_location="cpu", weights_only=True)
        require(raw["hidden_sha256"] == hidden_sha, iid + ": prompt tensor")
        require(torch.equal(raw["alphas"], expected_alphas), iid + ": pinned kernel alpha schedule")
        is_reuse = model == "model_1" and iid in ("00014393_002.png", "00014393_006.png")
        require(row["reused_from_kernel"] is is_reuse, iid + ": reuse eligibility")
        if is_reuse:
            exact_tree(raw, kernel_raw[iid], iid + ": reused raw packet")
            for key in ("features", "feature_names", "modules", "forward", "backward", "seconds"):
                require(row[key] == kernel_rows[iid][key], iid + ": reused " + key)
            provenance = row["reuse_provenance"]
            require(Path(provenance["directory"]).resolve() == kernel_dir.resolve()
                    and provenance["image_id"] == iid, iid + ": original reuse identity")
            for name in ("protocol", "results", "raw", "execution", "verification"):
                suffix = ".pt" if name == "raw" else ".json"
                require(provenance[name+"_sha256"] == digest(kernel_dir/(name+suffix)), iid + ": reuse provenance " + name)
            reused.append(row)
        else:
            require(row["reuse_provenance"] is None, iid + ": no fabricated reuse provenance")
        checked.append(verify_image(row, raw, cache["latents"][iid], old_records[model], old_directory))
        del raw
        if (index+1) % 40 == 0:
            print(json.dumps({"event":"verified_images", "records":index+1, "total":480}), flush=True)
    require(sum(r["image_member"] for r in results) == 240, "240 member and 240 nonmember E records")
    for model in ledgers:
        expected_prior = {r["image_id"] for r in results if r["model"] == model and r["image_id"] in old_records[model]}
        require(expected_prior == set(old_records[model]), "exact E prior-SecMI ID intersection")
    require(sum(r["prior_SecMI_states_exact"] is True for r in checked) == 192, "exact prior SecMI coverage")
    require(sum(r["prior_SecMI_states_exact"] is None for r in checked) == 288, "unmeasured prior SecMI not claimed")
    for key, value in dict(complete=True, current_step=2, records=480, unique_patients=120, unique_images=240,
        reused_records=2, new_records=478, frozen_files_unchanged=True, all_records_parameter_version_guards=True,
        vae_forward=0, text_encoder_forward=0, attack_fitting=False, performance_evaluation=False).items():
        require(execution[key] == value, "execution." + key)
    require(execution["status"] == "PASS_COHORT_EXTRACTION_PENDING_INDEPENDENT_VERIFICATION", "execution completion status")
    for key in ("forward", "backward"):
        total, reuse_cost = sum(r[key] for r in checked), sum(r[key] for r in reused)
        require(execution[key] == total and execution["reused_"+key] == reuse_cost
                and execution["new_"+key] == total-reuse_cost, key + ": total/reused/new cost")
        require(0 <= execution["current_attempt_"+key] <= total-reuse_cost, key + ": resumed-attempt cost bounds")
    require(execution["NO_objective_evaluations"] == sum(r["NO_objective_evaluations"] for r in checked), "NO actual q sum")
    require(execution["NO_nfev"] == sum(r["NO_nfev"] for r in checked), "NO SciPy request sum")
    require(len(execution["model_reports"]) == 2, "model report count")
    for report, model in zip(execution["model_reports"], ledgers):
        require(report["model"] == model and report["records"] == 240 and report["all_records_parameter_version_guards"] is True,
                "model freeze report")
        if report["new_this_attempt"]["records"]:
            require(report["current_session_adapter_values_exact_equal"] is True, "actual model adapter equality")
    for key in ("forward", "backward"):
        require(execution["current_attempt_"+key] == sum(r["new_this_attempt"][key] for r in execution["model_reports"]),
                "attempt model " + key + " reconciliation")
    for relative, value in {**raw_hashes, **pair_hashes}.items():
        require(digest(inside(directory, relative)) == value, "output changed during verification: " + relative)
    frozen_inputs(contract)
    return dict(status="PASS_SAVED_E_COHORT_ARITHMETIC_AND_INTEGRITY", current_step=2,
        records=480, unique_patients=120, unique_images=240, features_checked=12480,
        noise_draws_independently_regenerated=3840, prior_SecMI_exact_records=192,
        prior_SecMI_unavailable_records=288, reused_records=2, new_records=478,
        image_checks=checked, forward=execution["forward"], backward=execution["backward"],
        new_forward=execution["new_forward"], new_backward=execution["new_backward"],
        raw_sha256=raw_hashes, image_json_sha256=pair_hashes,
        protocol_sha256=digest(directory/"protocol.json"), results_sha256=digest(directory/"results.json"),
        execution_sha256=digest(directory/"execution.json"), verification_code_sha256=digest(__file__),
        arithmetic_source_sha256=KERNEL_HELPER_SHA256, E_copy_source_sha256=U_HELPER_SHA256,
        member_E_records=240, nonmember_E_records=240, independent_GPU_execution=False,
        actual_UNet_Jacobian_independently_recomputed=False, attack_efficacy_or_causal_claim=False,
        no_new_fitting=True, seconds_cpu=time.perf_counter()-begun)



U_HELPER_SHA256 = "da899f1efbc46f8571baebd2113d204038921b5b5374caba7ad22c7d58b64869"


def check_e_truth(row, member, image_exposures, patient_exposures, in_manifest):
    iid = row["image_id"]
    require(row["scenario"] == "E" and row["record_role"] == "train_candidate", iid + ": E-only image role")
    require(member in (0, 1) and row["member"] == member == int(patient_exposures > 0), iid + ": actual patient member")
    require(row["image_member"] == member == int(image_exposures > 0), iid + ": E image follows participation")
    require(bool(in_manifest) == bool(member), iid + ": E manifest membership")
    require(row["actual_training_exposures"] == image_exposures
            and row["patient_training_exposures"] == patient_exposures, iid + ": E actual exposure counts")


def check_arithmetic_source():
    source = ROOT/"u_patient_audit/verify_cdi_u_cohort.py"
    require(digest(source) == U_HELPER_SHA256, "frozen U arithmetic copy source")
    old_text = source.read_text(encoding="utf-8")
    # Exactly two metadata edits in verify_image; arithmetic/body remain unchanged.
    expected_text = old_text.replace("    is_e = False\n", "    is_e = True\n")
    expected_text = expected_text.replace('require(row["image_member"] == int(is_e),',
                                          'require(row["image_member"] == row["member"],')
    old_ast = ast.parse(expected_text)
    new_ast = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    names = ("array", "norm", "scalar", "linear_array", "same", "noised", "check_noising",
             "check_prediction", "check_loss_gradient", "recurrence", "verify_image", "exact_tree")
    previous = {n.name: n for n in old_ast.body if isinstance(n, ast.FunctionDef)}
    current = {n.name: n for n in new_ast.body if isinstance(n, ast.FunctionDef)}
    for name in names:
        require(ast.dump(previous[name], include_attributes=False) == ast.dump(current[name], include_attributes=False),
                "numeric function drift: " + name)
    return len(names)


def self_test():
    check_arithmetic_source()
    for model, group in (("model_1", "A"), ("model_2", "B")):
        for member in (0, 1):
            n = 4 if member else 0
            row = dict(image_id="synthetic", scenario="E", record_role="train_candidate",
                member=member, image_member=member, actual_training_exposures=n,
                patient_training_exposures=2*n)
            check_e_truth(row, member, n, 2*n, bool(member))
            wrong = dict(row, image_member=1-member)
            try:
                check_e_truth(wrong, member, n, 2*n, bool(member))
            except AssertionError:
                pass
            else:
                raise AssertionError("incorrect E image truth was accepted")
    # Existing kernel E packets plus a nonmember-E synthetic metadata regression.
    directory = RUN/"baseline_screen_20260914/cdi_kernel_v1"
    rows = {r["image_id"]:r for r in load_json(directory/"results.json")}
    raw = torch.load(directory/"raw.pt", map_location="cpu", weights_only=True)
    for iid in ("00014393_002.png", "00014393_006.png"):
        row = rows[iid]
        require(row["scenario"] == "E" and row["member"] == row["image_member"] == 1, "known E kernel metadata")
        check = verify_image(row, raw[iid], raw[iid]["latent"], {}, directory)
        require(check["prior_SecMI_states_exact"] is None, "no prior claim in synthetic helper test")
        synthetic = dict(row, member=0, image_member=0, actual_training_exposures=0, patient_training_exposures=0)
        # Arithmetic is label-independent; this is not a nonmember model evaluation.
        verify_image(synthetic, raw[iid], raw[iid]["latent"], {}, directory)
    print("PASS_E_MEMBER_NONMEMBER_TRUTH_AND_UNCHANGED_NUMERIC_KERNEL; no new model evaluation")


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=RUN/"baseline_screen_20260914/cdi_e_cohort_v1")
    parser.add_argument("--output-name", default="verification.json")
    parser.add_argument("--expected-code-sha256")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    require(bool(args.expected_code_sha256), "--expected-code-sha256 required")
    code_hash = digest(__file__)
    require(code_hash == args.expected_code_sha256.lower(), "externally frozen E verifier")
    directory = args.run_dir.resolve()
    require(Path(args.output_name).name == args.output_name and args.output_name.endswith(".json"), "simple JSON output name")
    output = directory/args.output_name
    protocol_path = output.with_name(output.stem + "_protocol.json")
    require(not output.exists() and not protocol_path.exists(), "immutable verification output")
    input_hashes = {name:digest(directory/name) for name in ("protocol.json", "execution.json", "results.json")}
    write_new(protocol_path, dict(schema="cdi-e-cohort-verification-protocol/v1",
        verification_source=str(Path(__file__).resolve()), expected_code_sha256=code_hash,
        arithmetic_copy_source=str(ROOT/"u_patient_audit/verify_cdi_u_cohort.py"),
        arithmetic_copy_source_sha256=U_HELPER_SHA256, original_kernel_arithmetic_sha256=KERNEL_HELPER_SHA256,
        input_sha256=input_hashes, E_truth="image_member equals patient member and actual individual image exposure>0",
        expected_prior_SecMI_records=192, expected_member_E_records=240, expected_nonmember_E_records=240,
        analysis_verification_scope="extraction only; four E/U analysis routes require separate independent verification",
        policy="independent saved-value arithmetic and metadata; no GPU/fitting/performance reads"))
    try:
        result = verify(directory)
        require(digest(__file__) == code_hash, "verifier changed during run")
        for name, value in input_hashes.items():
            require(digest(directory/name) == value, "verification input changed: " + name)
        result["verification_protocol_sha256"] = digest(protocol_path)
    except Exception as exc:
        write_new(output, dict(status="FAIL_INDEPENDENT_E_COHORT_VERIFICATION", error=repr(exc),
            traceback=traceback.format_exc(), verification_code_sha256=code_hash,
            verification_protocol_sha256=digest(protocol_path), independent_GPU_execution=False,
            attack_efficacy_or_causal_claim=False))
        raise
    write_new(output, result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("image_checks","raw_sha256","image_json_sha256")},
                     ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
