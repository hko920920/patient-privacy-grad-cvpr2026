"""Construction only: P, two existing DP summaries and fixed synthetic outputs.
No V features/predictions or performance metrics are read in this process.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
from threadpoolctl import threadpool_limits
from common import Inputs, check, dump, sha, now, export_labels

def solve_labels(xp, xs, weights, targets):
    blocks, responses, scales = [], [], {}
    operators = {}
    for model in ("DINO", "ResNet18"):
        z = xs[model]
        op = np.linalg.solve(z.T @ z / 128 + .1*np.eye(z.shape[1]), z.T/128)
        operators[model] = op
        t = xp[model] @ targets[model]
        scale = float(np.sqrt(np.sum(weights * t*t)))
        check(scale > 1e-12, "Degenerate public target RMS")
        scales[model] = scale
        blocks.append(np.sqrt(.5*weights[:, None]) * (xp[model] @ op) / scale)
        responses.append(np.sqrt(.5*weights) * t / scale)
    A = np.vstack(blocks)
    t = np.concatenate(responses)
    eta = .001*np.sum(A*A)/128
    old = np.array([-1.]*64 + [1.]*64)
    design = np.vstack([A, np.sqrt(eta)*np.eye(128)])
    response = np.concatenate([t, np.sqrt(eta)*old])
    fit = lsq_linear(design, response, bounds=(-1,1), tol=1e-12, max_iter=500)
    check(fit.success and np.isfinite(fit.x).all(), "Bounded label solver failed")
    l = fit.x
    check(l.shape == (128,) and np.max(abs(l)) <= 1, "Invalid labels")
    gradient = design.T @ (design@l-response)
    kkt = float(np.max(abs(l - np.clip(l-gradient, -1,1))))
    check(kkt <= 1e-7, "Independent projected-gradient check failed")
    initial = float(np.sum((design@old-response)**2))
    final = float(np.sum((design@l-response)**2))
    check(final <= initial+1e-12, "Optimization worsened its own objective")
    info = {"public_target_RMS": scales, "regularizer": float(eta),
            "source_public_relative_RMSE": float(np.linalg.norm(blocks[0]@l-responses[0])*np.sqrt(2)),
            "receiver_public_relative_RMSE": float(np.linalg.norm(blocks[1]@l-responses[1])*np.sqrt(2)),
            "initial_full_objective": initial, "final_full_objective": final,
            "solver_status": int(fit.status), "solver_iterations": int(fit.nit),
            "solver_optimality": float(fit.optimality),
            "projected_gradient_max_abs": kkt,
            "labels_at_bound": int(np.sum(abs(l)>1-1e-6)),
            "labels_changed_sign": int(np.sum((l>0)!=(old>0)))}
    return l, info, operators

def run(out):
    started = time.monotonic()
    io = Inputs(out, "construct")
    check(not (out/"label_seal.json").exists(), "Already sealed; do not silently rerun")
    sys.path.insert(0, io.contract["historical_code_dir"])
    from geometry_diagnostic import moments
    from fixed_image_compilation import pw
    ps = io.npz("P_DINO")
    pr = io.npz("P_RN")
    for data in (ps, pr):
        check(set(data["roles"].tolist()) == {"P"}, "Only public P allowed")
    for key in ("labels", "patient_ids", "image_ids"):
        check(np.array_equal(ps[key].astype(str), pr[key].astype(str)), "Public alignment: "+key)
    check(ps["z"].shape == (4,813,16), "DINO source schema changed")
    check(np.array_equal(ps["condition_ids"], [0,1,2,3]), "Condition order changed")
    source = {**ps, "z": ps["z"].transpose(1,0,2).reshape(-1,64).astype(np.float64)}
    xp = {"DINO":source["z"], "ResNet18":pr["z"].astype(np.float64)}
    w = pw(source)
    check(np.max(abs(w-pw(pr))) < 1e-15 and abs(w.sum()-1)<1e-12, "Weight mismatch")
    hd,_ = moments(source, True)
    hr,_ = moments(pr, True)
    check(np.max(abs(hd-xp["DINO"].T@(w[:,None]*xp["DINO"])))<1e-12, "Moment mismatch")
    source_images = io.npz("synthetic_DINO")
    maps = io.npz("public_maps_and_DP_weights")
    saved_labels = io.npz("historical_MT_labels")
    prior = io.js("historical_artifact_manifest")
    calibration = io.js("public_clipping")
    check(calibration["bounds"] == io.contract["class_clipping_bounds"], "Public clipping changed")
    results, labels, target_weights = {}, {}, {}
    geometry = {}
    for draw in ("DP1","DP2"):
        check(time.monotonic()-started<600, "Numerical worker time cap")
        target_path = io.path(draw+"_target")
        check(sha(target_path) == prior["packets"][draw]["target_sha256"], "Wrong original DP release")
        with np.load(target_path, allow_pickle=False) as f:
            mu = f["target"].reshape(2,64).astype(np.float64)
        syn = io.npz(draw+"_synthetic_RN")
        check(np.array_equal(syn["labels"], [0]*64+[1]*64), "Synthetic RN row order")
        xs = {"DINO":source_images[draw], "ResNet18":syn["z"].astype(np.float64)}
        check(xs["DINO"].shape==(128,64) and xs["ResNet18"].shape==(128,128), "Synthetic schema")
        wd = np.linalg.solve(hd+.1*np.eye(64), .5*(mu[1]-mu[0]))
        q = xp["DINO"]@wd  # No sigmoid, clipping, hard-threshold, or temperature.
        rhs = xp["ResNet18"].T@(w*q)
        wk = np.linalg.solve(hr+.1*np.eye(128), rhs)
        # Independent weighted augmented least-squares implementation.
        aug = np.vstack([np.sqrt(w[:,None])*xp["ResNet18"], np.sqrt(.1)*np.eye(128)])
        response = np.concatenate([np.sqrt(w)*q, np.zeros(128)])
        independent = np.linalg.lstsq(aug, response, rcond=None)[0]
        kd_error = float(np.max(abs(wk-independent)))
        normal_error = float(np.max(abs((hr+.1*np.eye(128))@wk-rhs)))
        check(kd_error<1e-10 and normal_error<1e-10, "KD independent solve mismatch")
        transferred = np.stack([mu[c]@maps["ResNet18_T"+str(c)]+maps["ResNet18_b"+str(c)] for c in (0,1)])
        wm = np.linalg.solve(hr+.1*np.eye(128), .5*(transferred[1]-transferred[0]))
        # Exact public signed-label expression; uses all patient/class rows.
        ids = ps["patient_ids"].astype(str)
        groups = []
        for patient in sorted(set(ids)):
            for c in (0,1):
                ix = np.flatnonzero((ids==patient)&(ps["labels"]==c))
                if len(ix):
                    groups.append(ix)
        signed_rows = .5*(maps["ResNet18_"+draw+"_public_weights_c1"]-maps["ResNet18_"+draw+"_public_weights_c0"])
        check(len(groups)==len(signed_rows), "Signed weight row ordering")
        v = np.empty(len(w))
        for a, ix in zip(signed_rows, groups):
            v[ix] = a/len(ix)
        signed_labels = v/w
        wm_dual = np.linalg.solve(hr+.1*np.eye(128), xp["ResNet18"].T@(w*signed_labels))
        mt_weight_error = float(np.max(abs(wm-wm_dual)))
        check(mt_weight_error<1e-10, "Mean transport / public signed-label mismatch")
        # Reproduce old MT recipe only to verify the matched implementation.
        lm, mf, _ = solve_labels(xp, xs, w, {"DINO":wd, "ResNet18":wm})
        reproduction = float(np.max(abs(lm-saved_labels[draw])))
        check(reproduction<1e-8, "Historical label path changed")
        lk, kf, _ = solve_labels(xp, xs, w, {"DINO":wd, "ResNet18":wk})
        labels[draw] = lk
        target_weights.update({draw+"_DINO":wd, draw+"_KD_RN":wk, draw+"_MT_RN":wm})
        public_mt = xp["ResNet18"]@wm
        public_kd = xp["ResNet18"]@wk
        geometry[draw] = {
            "KD_augmented_lstsq_max_abs":kd_error, "KD_normal_equation_max_abs":normal_error,
            "MT_signed_public_labels_max_abs":mt_weight_error,
            "historical_MT_label_reproduction_max_abs":reproduction,
            "KD_MT_receiver_public_prediction_max_abs":float(np.max(abs(public_kd-public_mt))),
            "KD_MT_receiver_target_relative_L2":float(np.sqrt(np.sum(w*(public_kd-public_mt)**2)/np.sum(w*public_mt**2))),
            "teacher_score_clipped":False,
            "MT_and_KD_are_identical_at_1e-10":bool(np.max(abs(public_kd-public_mt))<1e-10)}
        results[draw] = {"KD":kf, "MT_reproduction":mf}
        print('{"phase":"LABELS_CONSTRUCTED","draw":"'+draw+'"}', flush=True)
    np.savez_compressed(out/"KD_soft_labels.npz", **labels)
    np.savez_compressed(out/"target_weights.npz", **target_weights)
    packets = {}
    (out/"label_packets").mkdir(exist_ok=True)
    for draw in ("DP1","DP2"):
        io.path(draw+"_PNG_manifest")
        io.path(draw+"_PNG_csv")
        packets[draw] = export_labels(io.contract["banks"][draw], labels[draw],
                                     out/"label_packets"/(draw+"_KD_soft_labels.csv"))
    io.verify_receipts()
    dump(out/"construction_results.json", {"scope":"Step1 KD target construction; no V efficacy read",
         "created":now(), "fits":results, "verification":geometry,
         "new_Q_access":0, "new_releases":0, "new_pixel_updates":0,
         "new_label_packets":2, "historical_reproduction_solves":2,
         "seconds":time.monotonic()-started})
    dump(out/"construction_inputs.json", io.receipts)
    files = ["KD_soft_labels.npz","target_weights.npz","construction_results.json","construction_inputs.json"]
    seal = {"status":"BOTH_KD_LABELS_FROZEN_BEFORE_EVALUATION", "sealed_utc":now(),
            "contract_sha256":sha(out/"contract.json"),
            "files":{n:sha(out/n) for n in files}, "packets":packets,
            "implementation":io.contract["implementation"]}
    dump(out/"label_seal.json", seal)
    print('{"status":"BOTH_LABELS_SEALED_NO_V_READ","new_labels":2}', flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a=p.parse_args()
    with threadpool_limits(limits=4):
        run(a.out)

