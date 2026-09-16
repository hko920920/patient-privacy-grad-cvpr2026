"""Frozen-contract CPU analysis of CDI-feature patient adaptations.

This is not the original CDI reference-set hypothesis test. No target inference,
calibration/test evaluation, retrospective direction choice, or cohort expansion.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "_reports/cvpr_u_pilot_v1_001"
DEFAULT_CONTRACT = ROOT.parent / "CVPR 주제 탐색/research_2026-09-10/spec_sources/cdi_u_analysis_contract.json"
CONTRACT_SHA256 = "6f0afa62cf2bbf26790370965c4716f3afc927e0ab2412f8c4e6befbe02efca3"
FEATURE_NAMES = (["Denoising Loss", "SecMI$_{stat}$", "PIA", "PIAN"]
                 + [f"Gradient Masking_{i}" for i in range(10)]
                 + [f"Multiple Loss_{i}" for i in range(10)]
                 + ["Noise Optimization_0", "Noise Optimization_1"])


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_contract(path):
    require(digest(path) == CONTRACT_SHA256, "Analysis contract changed")
    c = read_json(path)
    require(c["schema"] == "cdi-u-patient-analysis-contract/v1", "Contract schema")
    require(len(c["methods"]) == 18, "Expected exactly 18 computed methods")
    for bound, expected in c["frozen_sha256"].items():
        require(digest(bound) == expected, f"Changed bound input: {bound}")
    fit_ids = {r["patient_id"] for r in c["patient_rows"] if r["role"] == "fit"}
    all_validation = []
    for fold in c["folds"]:
        tr, va = set(fold["train_patient_ids"]), set(fold["validation_patient_ids"])
        require(not tr & va and tr | va == fit_ids, "Invalid patient fold partition")
        require(len(va) == 16 and len(tr) == 64, "Wrong fold size")
        all_validation.extend(va)
    require(len(all_validation) == len(set(all_validation)) == 80, "Fold validation overlap")
    return c


def load_data(run_dir, contract):
    """Validate metadata independently; tensor-level source conformance is separate."""
    execution = read_json(run_dir / "execution.json")
    require(execution.get("complete") is True, "Extraction must be complete")
    require(execution.get("status") in (
        "PASS_COHORT_EXTRACTION_PENDING_INDEPENDENT_VERIFICATION",
        "PASS_COHORT_EXTRACTION"), "Extraction status is not successful")
    require(execution["results_sha256"] == digest(run_dir / "results.json"), "Extraction result byte binding")
    require(execution["protocol_sha256"] == digest(run_dir / "protocol.json"), "Extraction protocol byte binding")
    require(execution["frozen_files_unchanged"] is True and execution["all_records_parameter_version_guards"] is True, "Extraction freeze/weight guard")
    protocol = read_json(run_dir / "protocol.json")
    require(protocol["contract_sha256"] == execution["contract_sha256"], "Extraction contract binding")
    bound = {str(Path(p).resolve()): sha for p, sha in protocol["contract"]["frozen_sha256"].items()}
    require(bound.get(str(DEFAULT_CONTRACT.resolve())) == CONTRACT_SHA256, "Extraction must bind predeclared analysis contract")
    rows = read_json(run_dir / "results.json")
    require(isinstance(rows, list) and len(rows) == 480, "Expected 480 image rows")
    indexed = {}
    patient_rows = {r["patient_id"]: r for r in contract["patient_rows"]}
    expected = {(model, p["patient_id"], image)
                for model in contract["models"] for p in patient_rows.values()
                for image in p["images"]}
    for row in rows:
        key = (row["model"], str(row["patient_id"]), row["image_id"])
        require(key in expected and key not in indexed, f"Unexpected/duplicate row {key}")
        patient = patient_rows[key[1]]
        member = int(patient["group"] == ("A" if key[0] == "model_1" else "B"))
        require(row["scenario"] == "U", f"Wrong scenario {key}")
        require(row["contract_sha256"] == execution["contract_sha256"], f"Row extraction contract {key}")
        require(row["eval_role"] == patient["role"] and row["assignment_group"] == patient["group"], f"Metadata {key}")
        require(int(row["member"]) == member, f"Patient label {key}")
        require(int(row["image_member"]) == 0 and int(row["actual_training_exposures"]) == 0, f"U exposure {key}")
        require((float(row["patient_training_exposures"]) > 0) == bool(member), f"Patient exposure {key}")
        require(row["feature_names"] == FEATURE_NAMES, f"26D order {key}")
        x = np.asarray(row["features"], dtype=np.float64)
        require(x.shape == (26,) and np.isfinite(x).all(), f"Invalid features {key}")
        objective = float(row["modules"]["noise_optim"]["optimizer"]["fun"])
        require(np.isfinite(objective), f"Invalid NO objective {key}")
        indexed[key] = np.append(x, objective)
    require(set(indexed) == expected, "Missing expected U images")
    result = {}
    for model in contract["models"]:
        result[model] = {}
        for role in ("fit", "selection"):
            patients = [p for p in contract["patient_rows"] if p["role"] == role]
            result[model][role] = {
                "patients": patients,
                "x": np.asarray([[indexed[(model, p["patient_id"], im)] for im in p["images"]] for p in patients]),
                "y": np.asarray([int(p["group"] == ("A" if model == "model_1" else "B")) for p in patients]),
            }
    return result, execution


def representation(x, name):
    """Input is patient x two images x 27; literal features occupy first 26."""
    if name == "image26":
        return x[:, :, :26].reshape(-1, 26)
    if name == "mean26":
        return x[:, :, :26].mean(axis=1)
    if name in ("meanmax52", "meanmax54"):
        z = x[:, :, :26] if name == "meanmax52" else x
        return np.concatenate([z.mean(axis=1), z.max(axis=1)], axis=1)
    raise ValueError(f"Unknown representation {name}")


def fit_lr(x, y, C, config):
    scaler = StandardScaler(with_mean=True, with_std=True)
    transformed = scaler.fit_transform(x)
    lr = LogisticRegression(C=C, solver=config["solver"], penalty="l2",
                            max_iter=config["max_iter"], random_state=config["random_state"],
                            class_weight=None)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", ConvergenceWarning)
        lr.fit(transformed, y)
    messages = [{"category": w.category.__name__, "message": str(w.message)} for w in recorded]
    require(np.isfinite(lr.coef_).all() and np.isfinite(lr.intercept_).all(), "Nonfinite fitted parameters")
    return scaler, lr, messages


def predict_patients(x, spec, scaler, lr):
    probabilities = lr.predict_proba(scaler.transform(representation(x, spec["representation"])))[:, 1]
    if spec["representation"] == "image26":
        probabilities = probabilities.reshape(len(x), 2)
        probabilities = (probabilities.mean(axis=1) if spec["patient_probability_pool"] == "mean"
                         else probabilities.max(axis=1))
    require(probabilities.shape == (len(x),) and np.isfinite(probabilities).all(), "Invalid patient predictions")
    return probabilities


def model_parameters(scaler, lr):
    return {"scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
            "scaler_var": scaler.var_.tolist(), "scaler_n_samples_seen": int(scaler.n_samples_seen_),
            "coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist(),
            "classes": lr.classes_.tolist(), "n_iter": lr.n_iter_.tolist()}


def fit_family(model, fit, selection, spec, contract):
    config = contract["logistic_regression"]
    ids = [p["patient_id"] for p in fit["patients"]]
    index = {p: i for i, p in enumerate(ids)}
    cv = []
    if spec["C_policy"] == "tuned":
        for C in config["grid_C"]:
            fold_results = []
            for fold in contract["folds"]:
                tr = np.asarray([index[p] for p in fold["train_patient_ids"]])
                va = np.asarray([index[p] for p in fold["validation_patient_ids"]])
                train_x = representation(fit["x"][tr], spec["representation"])
                train_y = np.repeat(fit["y"][tr], 2) if spec["representation"] == "image26" else fit["y"][tr]
                scaler, lr, messages = fit_lr(train_x, train_y, C, config)
                pred = predict_patients(fit["x"][va], spec, scaler, lr)
                fold_results.append({"fold": fold["fold"], "patient_log_loss": float(log_loss(fit["y"][va], pred, labels=[0, 1])),
                                     "validation_patient_ids": [ids[i] for i in va],
                                     "validation_probabilities": pred.tolist(), "warnings": messages,
                                     "n_iter": lr.n_iter_.tolist()})
            cv.append({"C": C, "mean_patient_log_loss": float(np.mean([f["patient_log_loss"] for f in fold_results])), "folds": fold_results})
        chosen = min(cv, key=lambda r: (r["mean_patient_log_loss"], r["C"]))["C"]
    else:
        chosen = config["fixed_C"]
    train_x = representation(fit["x"], spec["representation"])
    train_y = np.repeat(fit["y"], 2) if spec["representation"] == "image26" else fit["y"]
    scaler, lr, messages = fit_lr(train_x, train_y, chosen, config)
    predictions = {"fit": predict_patients(fit["x"], spec, scaler, lr),
                   "selection": predict_patients(selection["x"], spec, scaler, lr)}
    params = {"model": model, "method": spec["id"], "chosen_C": chosen,
              "C_policy": spec["C_policy"], "representation": spec["representation"],
              "fitting_patient_ids": ids, "fitting_rows": len(train_y), "warnings": messages,
              "parameters": model_parameters(scaler, lr)}
    cv_out = {"model": model, "method": spec["id"], "chosen_C": chosen, "candidates": cv}
    return predictions, params, cv_out, scaler, lr


def bootstrap_indices(patients, n, seed):
    a = np.asarray([i for i, p in enumerate(patients) if p["group"] == "A"])
    b = np.asarray([i for i, p in enumerate(patients) if p["group"] == "B"])
    require(len(a) == len(b) == 20, "Selection must have 20 patients per assignment")
    rng = np.random.default_rng(seed)
    return np.asarray([np.concatenate([rng.choice(a, size=len(a), replace=True),
                                     rng.choice(b, size=len(b), replace=True)]) for _ in range(n)])


def bootstrap_auc(scores, labels, indices):
    # Same paired patients across models and methods. Matrix credit includes ties.
    bs, by = scores[indices], labels[indices]
    positive = bs[by == 1].reshape(len(indices), 20)
    negative = bs[by == 0].reshape(len(indices), 20)
    return ((positive[:, :, None] > negative[:, None, :]).sum(axis=(1, 2))
            + .5 * (positive[:, :, None] == negative[:, None, :]).sum(axis=(1, 2))) / 400.


def ci(values):
    return np.quantile(values, [.025, .975], method="linear").tolist()


def run_analysis(run_dir, contract_path, output_tag, expected_code_sha256):
    started = time.perf_counter()
    code_sha = digest(__file__)
    require(code_sha == expected_code_sha256.lower(), "Analysis source is not the externally frozen SHA256")
    contract = validate_contract(contract_path)
    require(output_tag and Path(output_tag).name == output_tag and output_tag not in (".", ".."), "Output tag must be a simple directory name")
    output = run_dir / output_tag
    require(not output.exists(), f"Refusing to overwrite {output}")
    data, extraction = load_data(run_dir, contract)
    dino_rows = read_json(contract["reused_control"]["scores_path"])
    dino = {}
    for r in dino_rows:
        key = (r["model"], str(r["patient_id"]))
        require(key not in dino, "Duplicate reused DINO row")
        dino[key] = r
    scores, prediction_rows, parameters, cv_results = {}, [], [], []
    same_scorer_rows, same_scorer_deltas = [], []
    warning_count = 0
    for model in contract["models"]:
        fit, selection = data[model]["fit"], data[model]["selection"]
        scores[model] = {}
        for spec in contract["methods"]:
            if spec["kind"] == "logistic_regression":
                pred, params, cv_out, scaler, lr = fit_family(model, fit, selection, spec, contract)
                parameters.append(params); cv_results.append(cv_out)
                warning_count += len(params["warnings"]) + sum(len(f["warnings"]) for c in cv_out["candidates"] for f in c["folds"])
                other_model = "model_2" if model == "model_1" else "model_1"
                other_selection = data[other_model]["selection"]
                require([p["patient_id"] for p in selection["patients"]] == [p["patient_id"] for p in other_selection["patients"]], "Cross-target patient ordering")
                both = {model: pred["selection"], other_model: predict_patients(other_selection["x"], spec, scaler, lr)}
                aligned = np.where(data["model_1"]["selection"]["y"] == 1,
                                   both["model_1"] - both["model_2"], both["model_2"] - both["model_1"])
                same_scorer_deltas.append({"fit_target_scorer": model, "method": spec["id"], "deltas": aligned})
                for i, p in enumerate(selection["patients"]):
                    same_scorer_rows.append({"fit_target_scorer": model, "method": spec["id"],
                                             "patient_id": p["patient_id"], "group": p["group"],
                                             "score_model_1_features": float(both["model_1"][i]),
                                             "score_model_2_features": float(both["model_2"][i]),
                                             "participation_aligned_delta": float(aligned[i])})
            else:
                pred = {}
                for role in ("fit", "selection"):
                    image_scores = spec["direction"] * data[model][role]["x"][:, :, spec["feature_index"]]
                    pred[role] = (image_scores.mean(axis=1) if spec["pool_after_direction"] == "mean" else image_scores.max(axis=1))
            scores[model][spec["id"]] = pred
        reused = {}
        for role in ("fit", "selection"):
            values = []
            for p, member in zip(data[model][role]["patients"], data[model][role]["y"]):
                old = dino[(model, p["patient_id"])]
                require(old["role"] == role and old["group"] == p["group"] and int(old["member"]) == int(member), "DINO metadata mismatch")
                values.append(float(old["score"]))
            reused[role] = np.asarray(values)
            require(np.isfinite(reused[role]).all(), "DINO nonfinite")
        scores[model]["dino_existing"] = reused
        for method, pred in scores[model].items():
            for role in ("fit", "selection"):
                for p, member, score in zip(data[model][role]["patients"], data[model][role]["y"], pred[role]):
                    prediction_rows.append({"model": model, "method": method, "patient_id": p["patient_id"],
                                            "role": role, "group": p["group"], "member": int(member), "score": float(score)})
        print(f"Fitted fixed/tuned methods on {model}: 18 computed + DINO reuse; no selection fitting", flush=True)
    patients = data["model_1"]["selection"]["patients"]
    indices = bootstrap_indices(patients, contract["bootstrap"]["resamples"], contract["bootstrap"]["seed"])
    same_scorer_summary = []
    for entry in same_scorer_deltas:
        delta = entry["deltas"]
        same_scorer_summary.append({"fit_target_scorer": entry["fit_target_scorer"], "method": entry["method"],
                                    "mean_participation_aligned_delta": float(delta.mean()),
                                    "mean_delta_CI95": ci(delta[indices].mean(axis=1)),
                                    "positive_count": int((delta > 0).sum()), "zero_count": int((delta == 0).sum()),
                                    "negative_count": int((delta < 0).sum()),
                                    "assignment_group_mean_delta": {g: float(delta[[p["group"] == g for p in patients]].mean()) for g in ("A", "B")}})
    results, contrasts, distributions = [], [], {}
    for model in contract["models"]:
        distributions[model] = {}
        for method, pred in scores[model].items():
            y = data[model]["selection"]["y"]
            auc = float(roc_auc_score(y, pred["selection"]))
            boot = bootstrap_auc(pred["selection"], y, indices)
            distributions[model][method] = boot
            results.append({"model": model, "method": method, "selection_patient_AUC": auc,
                            "selection_patient_AUC_CI95": ci(boot), "selection_patients": 40,
                            "selection_members": 20, "selection_nonmembers": 20,
                            "fit_patient_AUC_descriptive_only": float(roc_auc_score(data[model]["fit"]["y"], pred["fit"]))})
        point = {r["method"]: r["selection_patient_AUC"] for r in results if r["model"] == model}
        for comparison in contract["contrasts"]:
            a, b = comparison["a"], comparison["b"]
            contrasts.append({"model": model, **comparison, "selection_delta_AUC": point[a] - point[b],
                              "paired_CI95": ci(distributions[model][a] - distributions[model][b])})
    # Mean of paired targets is descriptive, with common patient resampling; not n=2 replicates.
    joint_results, joint_contrasts = [], []
    for method in scores["model_1"]:
        points = [r["selection_patient_AUC"] for r in results if r["method"] == method]
        boot = (distributions["model_1"][method] + distributions["model_2"][method]) / 2
        joint_results.append({"method": method, "mean_target_AUC_descriptive": float(np.mean(points)), "paired_patient_CI95": ci(boot)})
    for comparison in contract["contrasts"]:
        a, b = comparison["a"], comparison["b"]
        boot = sum(distributions[m][a] - distributions[m][b] for m in contract["models"]) / 2
        points = [r["selection_delta_AUC"] for r in contrasts if r["id"] == comparison["id"]]
        joint_contrasts.append({**comparison, "mean_target_delta_AUC_descriptive": float(np.mean(points)), "paired_patient_CI95": ci(boot)})
    result = {"schema": "cdi-u-patient-analysis/v1", "status": "COMPUTED_PENDING_INDEPENDENT_ANALYSIS_VERIFICATION",
              "research_stage": 2, "computed_methods": 18, "reused_controls": 1,
              "results": results, "contrasts": contrasts,
              "paired_target_descriptive_results": joint_results,
              "paired_target_descriptive_contrasts": joint_contrasts,
              "same_fitted_scorer_diagnostic": {"contract": contract["same_fitted_scorer_diagnostic"], "summaries": same_scorer_summary},
              "warning_count": warning_count, "overall_stage2_gate": None,
              "interval_interpretation": "Unadjusted exploratory intervals for predeclared methods and contrasts; not family-wise corrected confirmatory tests",
              "scope_limits": contract["scope_limits"], "bootstrap": contract["bootstrap"],
              "extraction_status": extraction["status"], "seconds_cpu_analysis": time.perf_counter() - started}
    input_paths = [run_dir / "results.json", run_dir / "execution.json", run_dir / "protocol.json", Path(contract_path)]
    # No tensor-level re-verification is implied by these byte bindings.
    provenance = {"source_path": str(Path(__file__).resolve()), "source_sha256": code_sha,
                  "analysis_contract_sha256": CONTRACT_SHA256,
                  "input_sha256": {str(p.resolve()): digest(p) for p in input_paths},
                  "numpy": np.__version__, "sklearn": sklearn.__version__, "python": platform.python_version(),
                  "tensor_conformance": "Separate extraction verifier; this analyzer validates metadata, features finite/order, fitting and statistics",
                  "new_GPU_calls": 0, "calibration_test_features_read": False,
                  "selection_used_for_fitting_tuning_or_direction": False}
    require(digest(__file__) == code_sha and digest(contract_path) == CONTRACT_SHA256, "Frozen analysis changed while running")
    output.mkdir()
    save_json(output / "contract_copy.json", contract)
    save_json(output / "analysis.json", result)
    save_json(output / "predictions.json", prediction_rows)
    save_json(output / "fit_parameters.json", parameters)
    save_json(output / "cv_results.json", cv_results)
    save_json(output / "same_fitted_scorer_predictions.json", same_scorer_rows)
    save_json(output / "bootstrap_resamples.json", {"selection_patient_order": [p["patient_id"] for p in patients], "indices": indices.tolist()})
    provenance["output_sha256"] = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
    save_json(output / "provenance.json", provenance)
    return result, output


def self_test():
    """Synthetic only: check patient pooling, actual fit-only scaling and bootstrap ties."""
    x = np.zeros((4, 2, 27), dtype=np.float64)
    x[:, 0, 0] = [1, 3, 7, 9]; x[:, 1, 0] = [2, 4, 8, 10]
    require(np.array_equal(representation(x, "meanmax52")[:, [0, 26]], [[1.5, 2], [3.5, 4], [7.5, 8], [9.5, 10]]), "Patient feature pooling")
    require(np.array_equal((-x[:, :, 0]).max(axis=1), [-1, -3, -7, -9]), "Scalar negation before max")
    config = {"solver": "liblinear", "max_iter": 1000, "random_state": 0}
    scaler, lr, _ = fit_lr(representation(x[:2], "image26"), [0, 0, 1, 1], 1., config)
    require(scaler.mean_[0] == 2.5, "Scaler leaked held-out examples")
    spec = {"representation": "image26", "patient_probability_pool": "mean"}
    direct = lr.predict_proba(scaler.transform(representation(x, "image26")))[:, 1].reshape(4, 2)
    require(np.array_equal(predict_patients(x, spec, scaler, lr), direct.mean(axis=1)), "Image probability pooling")
    patients = [{"group": "A" if i < 20 else "B"} for i in range(40)]
    labels = np.r_[np.ones(20, dtype=int), np.zeros(20, dtype=int)]
    indices = bootstrap_indices(patients, 13, 260914)
    scores = (np.arange(40) % 7).astype(float)
    vectorized = bootstrap_auc(scores, labels, indices)
    independent = np.asarray([roc_auc_score(labels[ix], scores[ix]) for ix in indices])
    require(np.allclose(vectorized, independent, rtol=0, atol=2e-16), "Bootstrap half-tie AUC mismatch")
    require(np.allclose(bootstrap_auc(scores, 1-labels, indices), 1-vectorized, rtol=0, atol=2e-16), "Complementary target pairing")
    print("PASS_SYNTHETIC_CPU_CHECKS; no actual cohort features read")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=RUN / "baseline_screen_20260914/cdi_u_cohort_v1")
    p.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--output-tag", default="analysis_v1")
    p.add_argument("--expected-code-sha256")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test(); return
    p.error("--expected-code-sha256 is required for actual analysis") if not args.expected_code_sha256 else None
    result, output = run_analysis(args.run_dir, args.contract, args.output_tag, args.expected_code_sha256)
    print(json.dumps({"status": result["status"], "output": str(output), "seconds": result["seconds_cpu_analysis"], "warning_count": result["warning_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
