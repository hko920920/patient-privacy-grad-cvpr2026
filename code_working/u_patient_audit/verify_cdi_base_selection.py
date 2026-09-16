"""Independent CPU base-control verification; no GPU or fitting.
verify_base_image is copied from the frozen U verifier; only its target/member
metadata guard is changed to the explicitly checked base incremental label.
Numerical helpers remain imported from that hash-pinned independent verifier.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from safetensors import safe_open
from .verify_cdi_u_cohort import (
    ROOT,RUN,MODULES,NAMES,SHAPE,require,array,norm,scalar,linear_array,same,
    check_noising,check_prediction,check_loss_gradient,recurrence,digest,inside,
    read_csv,write_new,load_json,tight)
HELPER_SHA="da899f1efbc46f8571baebd2113d204038921b5b5374caba7ad22c7d58b64869"
ANALYZER_SHA="e1876c10d2533c10015bf9e24f00c43627dc311332bb162a2d26f28a1e471227"
TARGET=RUN/"baseline_screen_20260914/cdi_u_cohort_v1"
DEFAULT=RUN/"baseline_screen_20260914/cdi_base_selection_v1"

def verify_base_image(row, raw, latent_cache, old_records, old_directory):
    iid = row["image_id"]
    require(row["eval_role"] in ("fit", "selection"), iid + ": cohort role")
    require(row["model"] == "base" and row["member"] == row["incremental_patient_member"] == 0, iid + ": model/member")
    require(row["stream"] == raw["stream"] == "primary", iid + ": stream")
    require(raw["image_id"] == iid, iid + ": raw identity")
    require(row["prediction_type"] == "epsilon" and row["batch_size"] == 1, iid + ": backend")
    require(row["weights_unchanged"] is True and row["unet_training"] is False, iid + ": freeze/eval")
    is_e = False
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

def artifact_state(path):
    h=hashlib.sha256()
    with safe_open(str(path),framework="pt",device="cpu") as f:
        names=sorted(f.keys())
        for name in names:
            value=f.get_tensor(name)
            if value.is_floating_point():
                value=value.float()
            value=value.contiguous()
            require(bool(torch.isfinite(value).all()),"finite pretrained artifact")
            h.update(name.encode()+b"\0")
            h.update(str(value.dtype).encode("ascii"))
            h.update(json.dumps(list(value.shape)).encode("ascii"))
            h.update(value.numpy().tobytes())
    return dict(sha256=h.hexdigest(),state_tensors=len(names))

def verify_raw(directory):
    begun=time.perf_counter()
    torch.set_num_threads(2)
    protocol,execution=load_json(directory/"protocol.json"),load_json(directory/"execution.json")
    c=protocol["contract"]
    require(load_json(protocol["contract_path"])==c and digest(protocol["contract_path"])
            ==protocol["contract_sha256"]==execution["contract_sha256"],"external contract")
    require(execution["protocol_sha256"]==digest(directory/"protocol.json")
            and execution["results_sha256"]==digest(directory/"results.json"),"output bindings")
    expected=dict(schema="cdi-base-selection-contract/v1",current_step=2,model="base",
        eval_roles=["selection"],scenario="U",master_seed=260914,stream="primary",
        batch_size=1,prediction_type="epsilon",dtype="float32",prompt="a frontal chest radiograph",
        code_policy="released_code_literal",perform_fitting=False,perform_performance_evaluation=False,
        expected_patients=40,expected_unique_images=80,expected_records=80)
    for k,v in expected.items():require(c[k]==v,"contract."+k)
    require(c["analysis_policy"]["frozen_target_fitted_scorers"]==20
            and c["analysis_policy"]["no_new_fitting"] is True
            and c["analysis_policy"]["pretrained_membership"]=="unknown","analysis scope")
    frozen=c["frozen_sha256"]
    for p,h in frozen.items():require(digest(p)==h,"frozen input "+p)
    for p in [ROOT/"u_patient_audit/analyze_cdi_base_selection.py",
              RUN/"cache/cache.pt",RUN/"cohort/lock.json",Path(c["base_weight_path"])]:
        require(str(p.resolve()) in frozen,"required binding")
    require(digest(ROOT/"u_patient_audit/analyze_cdi_base_selection.py")==ANALYZER_SHA,"analyzer unchanged")
    require(digest(c["base_weight_path"])==c["base_weight_sha256"]==execution["base_weight_sha256"],"base weight digest")
    base_state=artifact_state(c["base_weight_path"])
    for key in ("sha256","state_tensors"):
        require(execution["initial_state"][key]==execution["final_state"][key]==base_state[key],"independent artifact-state."+key)
    require(execution["initial_state"]["artifact_tensors_exactly_checked"]==base_state["state_tensors"]
            and execution["final_state"]["artifact_tensors_exactly_checked"]==0,"producer actual base tensor checks")
    lock=load_json(RUN/"cohort/lock.json")
    for name,h in lock["files"].items():require(digest(RUN/"cohort"/name)==h,"cohort lock")
    expected_rows=[r for r in read_csv(RUN/"cohort/evaluation_images.csv")
                   if r["eval_role"]=="selection" and r["record_role"]=="U_observed"]
    expected_rows.sort(key=lambda r:(int(r["patient_id"]),r["image_id"]))
    ids=sorted({r["patient_id"] for r in expected_rows},key=int)
    require(len(expected_rows)==80 and len(ids)==40 and c["patient_order"]==ids
            and c["image_order"]==[r["image_id"] for r in expected_rows],"original cohort order")
    require(sum(r["assignment_group"]=="A" for r in expected_rows)==40,"A20/B20 U2")
    for m in ("model_1","model_2"):
        train={r["image_id"] for r in read_csv(RUN/"cohort"/(m+"_train.csv"))}
        require(not train&{r["image_id"] for r in expected_rows},"U absent from incremental training manifests")
    cache=torch.load(RUN/"cache/cache.pt",map_location="cpu",weights_only=True)
    cache_hash=digest(RUN/"cache/cache.pt");lock_hash=digest(RUN/"cohort/lock.json")
    meta=load_json(RUN/"cache/summary.json")
    require(meta["cache_sha256"]==cache_hash and meta["cohort_lock_sha256"]==lock_hash,"cache lock")
    require(cache["audit_prompt"]==c["prompt"],"same cached prompt")
    hidden_sha=hashlib.sha256(cache["hidden"][c["prompt"]].float().contiguous().numpy().tobytes()).hexdigest()
    kp=RUN/"baseline_screen_20260914/cdi_kernel_v1"
    kernel=torch.load(kp/"raw.pt",map_location="cpu",weights_only=True)
    alphas=next(iter(kernel.values()))["alphas"]
    target_protocol=load_json(TARGET/"protocol.json")
    require(protocol["source_bindings"]==target_protocol["source_bindings"],"same exact source bindings")
    require(protocol["source_bindings"]["feature_names"]==NAMES,"26 features")
    for p,h in target_protocol["contract"]["frozen_sha256"].items():
        require(frozen[p]==h,"all target bindings retained")
    rows=load_json(directory/"results.json")
    require(len(rows)==80 and len(protocol["measurement_plan"])==80,"80 base records/plans")
    checked,raw_hashes,pair_hashes=[],{},{}
    for row,old,plan in zip(rows,expected_rows,protocol["measurement_plan"]):
        iid=row["image_id"]
        require(iid==old["image_id"] and row["patient_id"]==old["patient_id"]
                and row["assignment_group"]==old["assignment_group"],"base original identity")
        for k,v in plan.items():require(row[k]==v,"planned metadata."+k)
        for k in ("member","incremental_patient_member","image_member","actual_training_exposures","patient_training_exposures"):
            require(row[k]==0,"incremental base zero "+k)
        require(row["model"]=="base" and row["pretrained_membership"]=="unknown"
                and row["assignment_group_is_base_membership"] is False
                and row["membership_label_scope"]=="NIH_incremental_training_only","base membership semantics")
        require(row["checkpoint_sha256"]==row["base_weight_sha256"]==c["base_weight_sha256"]
                and row["cache_sha256"]==cache_hash and row["cohort_lock_sha256"]==lock_hash
                and row["contract_sha256"]==protocol["contract_sha256"],"row input hashes")
        require(row["raw_path"]==Path(iid).stem+".pt" and row["reused_record"] is False,"fresh raw path")
        p=inside(directory,row["raw_path"])
        require(digest(p)==row["raw_sha256"],"raw digest")
        require(load_json(p.with_suffix(".json"))==row,"raw scalar pair exact")
        raw_hashes[row["raw_path"]]=row["raw_sha256"]
        pair_hashes[p.with_suffix(".json").name]=digest(p.with_suffix(".json"))
        raw=torch.load(p,map_location="cpu",weights_only=True)
        require(raw["hidden_sha256"]==hidden_sha and torch.equal(raw["alphas"],alphas),"prompt/schedule match target")
        checked.append(verify_base_image(row,raw,cache["latents"][iid],{},kp))
        del raw
    expected_execution=dict(status="PASS_BASE_SELECTION_EXTRACTION_PENDING_INDEPENDENT_VERIFICATION",
        complete=True,current_step=2,records=80,unique_patients=40,unique_images=80,
        model="base",eval_role="selection",scenario="U",source_variant="fp16",compute_dtype="float32",
        no_lora_modules=True,lora_parameter_count=0,state_values_exactly_unchanged=True,
        parameter_versions_and_no_weight_gradients=True,all_parameters_frozen=True,unet_training=False,
        gradient_checkpointing_enabled=False,frozen_files_unchanged=True,pretrained_membership="unknown",
        all_NIH_incremental_membership_labels=0,assignment_A_B_is_base_membership=False,
        attack_fitting=False,performance_evaluation=False,base_membership_AUC=False,
        vae_forward=0,text_encoder_forward=0,new_target_training=False)
    for k,v in expected_execution.items():require(execution[k]==v,"execution."+k)
    require(execution["parameter_versions_checked"]>0,"actual frozen parameter guards")
    for k in ("forward","backward","NO_objective_evaluations","NO_nfev"):
        require(execution[k]==sum(r[k] for r in checked),"reconciled "+k)
    require(all(r["prior_SecMI_states_exact"] is None for r in checked),"no fabricated base SecMI equivalence")
    for p,h in {**raw_hashes,**pair_hashes}.items():require(digest(directory/p)==h,"unchanged packet")
    for p,h in frozen.items():require(digest(p)==h,"unchanged frozen source/input")
    return dict(status="PASS_SAVED_BASE_ARITHMETIC_AND_INTEGRITY",records=80,patients=40,
        features_checked=2080,noise_draws_independently_regenerated=640,
        image_checks=checked,raw_sha256=raw_hashes,image_json_sha256=pair_hashes,
        base_artifact_FP32_state=base_state,independent_GPU_execution=False,
        actual_UNet_Jacobian_independently_recomputed=False,prior_SecMI_equivalence_claim=False,
        pretrained_membership="unknown",seconds_cpu=time.perf_counter()-begun)

def verify_stats(directory,tag):
    out=directory/tag
    prov=load_json(out/"provenance.json")
    require(prov["source_sha256"]==ANALYZER_SHA,"frozen analysis source")
    require(prov["protocol_sha256"]==digest(out/"protocol.json"),"analysis protocol")
    for p,h in prov["input_sha256"].items():require(digest(p)==h,"analysis input")
    for p,h in prov["output_sha256"].items():require(digest(out/p)==h,"analysis output")
    old=TARGET/"analysis_v1"
    contract=load_json(old/"contract_copy.json")
    patients=[p for p in contract["patient_rows"] if p["role"]=="selection"]
    ids=[p["patient_id"] for p in patients]
    methods={s["id"]:s for s in contract["methods"] if s["kind"]=="logistic_regression"}
    params={(p["model"],p["method"]):p for p in load_json(old/"fit_parameters.json")}
    source_pairs={(r["fit_target_scorer"],r["method"],r["patient_id"]):r
                  for r in load_json(old/"same_fitted_scorer_predictions.json")}
    base={r["image_id"]:r for r in load_json(directory/"results.json")}
    prediction_rows=load_json(out/"predictions.json")
    pred={(r["fit_target_scorer"],r["method"],r["patient_id"]):r for r in prediction_rows}
    require(len(pred)==len(prediction_rows)==800,"800 frozen-scorer predictions")
    pair_rows=load_json(out/"pairs.json")
    pair_map={(r["fit_target_scorer"],r["method"]):r["pairs"] for r in pair_rows}
    require(len(pair_rows)==len(pair_map)==20 and all(len(r)==400 for r in pair_map.values()),"8000 pair records")
    result=load_json(out/"analysis.json")
    summary={(r["fit_target_scorer"],r["method"]):r for r in result["summaries"]}
    require(len(summary)==20 and len(result["contrasts"])==4,"all specified summaries")
    boot=load_json(old/"bootstrap_resamples.json")
    require(boot["selection_patient_order"]==ids,"original bootstrap order")
    ix=np.asarray(boot["indices"])
    a=np.asarray([i for i,p in enumerate(patients) if p["group"]=="A"])
    b=np.asarray([i for i,p in enumerate(patients) if p["group"]=="B"])
    rng=np.random.default_rng(260914)
    require(np.array_equal(ix,[np.r_[rng.choice(a,20),rng.choice(b,20)] for _ in range(2000)]),"exact original bootstrap")
    points,boot_values={},{}
    def quant(v):return np.quantile(v,[.025,.975])
    for key,entry in params.items():
        model,method=key;spec=methods[method];par=entry["parameters"]
        require(par["classes"]==[0,1],"fixed class direction")
        base_scores=[]
        for p in patients:
            z=np.asarray([base[i]["features"]+[base[i]["modules"]["noise_optim"]["optimizer"]["fun"]] for i in p["images"]])
            rep=spec["representation"]
            if rep=="image26":design=z[:,:26]
            elif rep=="mean26":design=z[:,:26].mean(0)[None,:]
            else:
                require(rep in ("meanmax52","meanmax54"),"known representation")
                q=z[:,:26] if rep=="meanmax52" else z
                design=np.r_[q.mean(0),q.max(0)][None,:]
            transformed=(design-np.asarray(par["scaler_mean"]))/np.asarray(par["scaler_scale"])
            scores=expit(transformed.dot(np.asarray(par["coef"])[0])+par["intercept"][0])
            base_scores.append(float(scores.max() if rep=="image26" and spec["patient_probability_pool"]=="max" else scores.mean()))
        s1=np.asarray([source_pairs[(*key,p)]["score_model_1_features"] for p in ids])
        s2=np.asarray([source_pairs[(*key,p)]["score_model_2_features"] for p in ids])
        score_map={"base":np.asarray(base_scores),"own":s1 if model=="model_1" else s2,"common":(s1+s2)/2}
        pos,neg=(a,b) if model=="model_1" else (b,a)
        y=np.asarray([int(i in pos) for i in range(40)])
        for i,p in enumerate(patients):
            row=pred[(*key,p["patient_id"])]
            require(row["assignment_group"]==p["group"] and row["positive_assignment"]==int(y[i])
                    and row["incremental_patient_member"]==0 and row["pretrained_membership"]=="unknown","prediction labels")
            for n,field in (("base","score_base"),("own","score_own_target"),("common","score_common")):
                tight(row[field],score_map[n][i],"independent stored parameter prediction")
        credits={k:(v[pos,None]>v[None,neg]).astype(float)+.5*(v[pos,None]==v[None,neg]) for k,v in score_map.items()}
        av={k:float(roc_auc_score(y,v)) for k,v in score_map.items()}
        # Explicit repeated-row pair matrices are independent of producer count-weight einsum.
        local_pos=np.asarray([[int(np.where(pos==i)[0][0]) for i in draw if y[i]] for draw in ix])
        local_neg=np.asarray([[int(np.where(neg==i)[0][0]) for i in draw if not y[i]] for draw in ix])
        bs={k:v[local_pos[:,:,None],local_neg[:,None,:]].mean((1,2)) for k,v in credits.items()}
        points[key],boot_values[key]=av,bs
        row=summary[key]
        require(row["chosen_C"]==entry["chosen_C"] and row["positive_assignment"]==("A" if model=="model_1" else "B"),"frozen C/direction")
        for n,field in (("base","AUC_base_own_assignment"),("own","AUC_own_target"),("common","AUC_common_own_assignment")):
            tight(row[field],av[n],"independent sklearn assignment AUC")
            tight(row["AUC_CI95"][n],quant(bs[n]),"repeated-row bootstrap AUC")
        for n in ("own","common"):
            tight(row[n+"_minus_base_AUC"],av[n]-av["base"],"paired point difference")
            tight(row[n+"_minus_base_CI95"],quant(bs[n]-bs["base"]),"paired conditional interval")
            before,after=credits["base"],credits[n];delta=after-before
            change=row[n+"_vs_base_rank"]
            require(change["improved_pairs"]==int((delta>0).sum()) and change["worsened_pairs"]==int((delta<0).sum())
                    and change["unchanged_pairs"]==int((delta==0).sum())
                    and change["twice_credit_change_sum"]==int(2*delta.sum()),"rank change counts")
            tight(change["AUC_change"],delta.mean(),"rank/AUC identity")
            for f in range(3):
                for t in range(3):
                    require(change["transitions"][str(f)+"->"+str(t)]==int(((before==f/2)&(after==t/2)).sum()),"rank transition")
            ra,rb=rankdata(score_map["base"]),rankdata(score_map[n])
            corr=None if np.ptp(ra)==0 or np.ptp(rb)==0 else float(np.corrcoef(ra,rb)[0,1])
            if corr is None:require(row["Spearman_base_"+n] is None,"undefined correlation")
            else:tight(row["Spearman_base_"+n],corr,"independent rank correlation")
        for j,r in enumerate(pair_map[key]):
            pi,ni=pos[j//20],neg[j%20]
            require(r["positive_assignment_patient"]==ids[pi] and r["negative_assignment_patient"]==ids[ni],"pair ordering")
            for n,s in score_map.items():
                tight(r[n+"_margin"],s[pi]-s[ni],"raw pair margin")
                require(r[n+"_credit_twice"]==int(2*credits[n][j//20,j%20]),"exact half-tie pair")
    for r in result["contrasts"]:
        keya=(r["fit_target_scorer"],r["a"]);keyb=(r["fit_target_scorer"],r["b"])
        require(r["a"]=="patient_meanmax52_"+r["C_policy"] and r["b"]=="cdi_image_mean_"+r["C_policy"],"prespecified contrast")
        gain={n:points[keya][n]-points[keyb][n] for n in ("base","own","common")}
        bg={n:boot_values[keya][n]-boot_values[keyb][n] for n in gain}
        for n,field in (("base","base_assignment_AUC_gain"),("own","own_target_AUC_gain"),("common","common_assignment_AUC_gain")):
            tight(r[field],gain[n],"gain");tight(r["gain_CI95"][n],quant(bg[n]),"gain CI")
        tight(r["own_gain_minus_base_gain"],gain["own"]-gain["base"],"gain difference")
        tight(r["own_gain_minus_base_gain_CI95"],quant(bg["own"]-bg["base"]),"gain difference CI")
    require(result["new_fitting"] is False and result["target_inference_calls"]==0
            and result["original_primary_unchanged"] is True and result["success_gate"] is None,"analysis scope")
    return dict(status="PASS_FIXED_SCORER_BASE_ASSIGNMENT_STATISTICS",scorers=20,predictions=800,pair_records=8000,
        bootstrap=2000,independent_fitting=False,base_membership_AUC=False,causal_identification=False)

def self_test():
    a=np.asarray([0,1,1,3]);b=np.asarray([1,1,2,2])
    c=(a[:,None]>b[None,:])+.5*(a[:,None]==b[None,:])
    tight(c.mean(),roc_auc_score(np.r_[np.ones(4),np.zeros(4)],np.r_[a,b]),"synthetic ties")
    source=ROOT/"u_patient_audit/verify_cdi_u_cohort.py"
    require(digest(source)==HELPER_SHA,"frozen arithmetic helper")
    import ast,inspect
    from .verify_cdi_u_cohort import verify_image
    expected=inspect.getsource(verify_image).replace("def verify_image(","def verify_base_image(",1)
    expected=expected.replace('row["model"] in ("model_1", "model_2") and row["member"] in (0, 1)',
        'row["model"] == "base" and row["member"] == row["incremental_patient_member"] == 0')
    require(ast.dump(ast.parse(expected))==ast.dump(ast.parse(inspect.getsource(verify_base_image))),
            "only explicit base metadata guard differs; all arithmetic identical")
    require(not torch.cuda.is_initialized(),"CPU only")
    print("PASS_SYNTHETIC_TIES_AND_EXACT_FROZEN_ARITHMETIC_COPY")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-dir",type=Path,default=DEFAULT)
    p.add_argument("--analysis-tag")
    p.add_argument("--expected-code-sha256")
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:self_test();return
    require(a.expected_code_sha256 and digest(__file__)==a.expected_code_sha256.lower(),"externally frozen verifier")
    require(digest(ROOT/"u_patient_audit/verify_cdi_u_cohort.py")==HELPER_SHA,"frozen independent arithmetic helper")
    d=a.run_dir.resolve();out=d/"verification.json";vp=d/"verification_protocol.json"
    require(not out.exists() and not vp.exists(),"immutable verification")
    names=["protocol.json","execution.json","results.json"]
    if a.analysis_tag:
        require(Path(a.analysis_tag).name==a.analysis_tag and a.analysis_tag not in (".",".."),"simple analysis tag")
        names += [a.analysis_tag+"/"+n for n in ("protocol.json","analysis.json","predictions.json","pairs.json","provenance.json")]
    hashes={n:digest(d/n) for n in names}
    write_new(vp,dict(source_sha256=digest(__file__),helper_sha256=HELPER_SHA,input_sha256=hashes,
        analysis_tag=a.analysis_tag,scope="independent saved-value CPU checks; no GPU or fit"))
    try:
        result=verify_raw(d)
        if a.analysis_tag:result["analysis_verification"]=verify_stats(d,a.analysis_tag)
        require(digest(__file__)==a.expected_code_sha256.lower(),"verifier unchanged")
        for n,h in hashes.items():require(digest(d/n)==h,"input unchanged")
        result["verification_protocol_sha256"]=digest(vp)
        result["source_sha256"]=digest(__file__);result["helper_sha256"]=HELPER_SHA
    except Exception as e:
        write_new(out,dict(status="FAIL_INDEPENDENT_BASE_VERIFICATION",error=repr(e),traceback=traceback.format_exc()))
        raise
    write_new(out,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("image_checks","raw_sha256","image_json_sha256")},ensure_ascii=False))

if __name__=="__main__":
    main()

