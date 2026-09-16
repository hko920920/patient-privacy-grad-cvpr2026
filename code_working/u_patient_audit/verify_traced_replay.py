"""Independent CPU-only provenance/arithmetic verification of the traced U8 run."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import hashlib
import json
import math
import re
import statistics
import traceback
from pathlib import Path
import torch
from .common import RUN, SALT, digest, read_csv, verify_inputs, write_json
from .verify_numerical_diagnostic import (
    ROUNDING_FACTOR, check_cells, check_embedding_summary, rounding_check,
    scalar_close, vector,
)

OLD = RUN / "probe_smoke/training_coverage_v2/U_step_1000_n8"
FEATURES = ("response", "loss_mean", "loss_max", "base_difference_mean",
            "projected_conditioning_gradient_dot", "projected_conditioning_gradient_cosine")
CONTEXT = {"stage": "initial"}
COUNTS = {"forward": 0, "backward": 0, "support_steps": 0, "query_cells": 0}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def need(condition, message):
    if not condition:
        raise AssertionError(message)


def check_query(delta, image_id):
    cells = delta["cells"]
    expected = [(t, draw) for t in (50, 250, 500, 750, 950) for draw in (0, 1)]
    need([(c["timestep"], c["draw"]) for c in cells] == expected, "query timestep/draw ordering mismatch")
    base, target = 0.0, 0.0
    for cell in cells:
        need(cell["phase"] == "query" and cell["gradient_computed"] is False, "query gradient/phase mismatch")
        seed_text = "|".join(map(str, (SALT, "audit-noise", "query", image_id, cell["timestep"], cell["draw"])))
        need(cell["noise_seed"] == int(hashlib.sha256(seed_text.encode()).hexdigest()[:15], 16),
             "query seed does not match image, phase, timestep and draw")
        need(cell["base_loss"] >= 0 and cell["target_loss"] >= 0, "negative MSE")
        scalar_close(cell["value"], cell["base_loss"] - cell["target_loss"])
        need(all(cell[k] is None for k in ("gradient", "base_gradient", "target_gradient")),
             "no-gradient query unexpectedly contains gradients")
        base += cell["base_loss"] / len(cells)
        target += cell["target_loss"] / len(cells)
    scalar_close(delta["base_loss"], base)
    scalar_close(delta["target_loss"], target)
    scalar_close(delta["value"], base - target)
    need(torch.equal(vector(delta["gradient"]), torch.zeros(8)), "no-gradient delta aggregate is not zero")
    COUNTS["forward"] += 2 * len(cells)
    COUNTS["query_cells"] += len(cells)
    return base - target, base, target


def check_trace(trace, image_id, active_dimensions):
    need(trace["image_id"] == image_id and trace["precision"] == "fp16", "support identity/precision mismatch")
    need(trace["support_timesteps"] == [250, 500, 750] and trace["draws_per_timestep"] == 1,
         "support timestep/draw contract changed")
    need((trace["step_size"], trace["radius"], trace["gradient_scale"]) == (.01, .05, 1024.),
         "support step/radius/scale changed")
    need(trace["initial_coefficient"] == [0.] * 8 and len(trace["steps"]) == 6,
         "support initial coefficient/step count changed")
    need(trace["weights_unchanged"] is True and trace["seconds"] > 0, "support execution metadata invalid")
    final = trace["final_evaluation"]
    fb, ft, fg = check_cells(final["cells"], image_id, [250, 500, 750], False)
    scalar_close(final["base_loss"], fb)
    scalar_close(final["target_loss"], ft)
    scalar_close(final["value"], fb - ft)
    rounding_check(final["gradient"], fg)
    COUNTS["forward"] += 2 * len(final["cells"])
    previous = vector(trace["initial_coefficient"])
    maximum_cpu_error = 0.0
    for index, row in enumerate(trace["steps"]):
        need(row["step"] == index, "support step ordering mismatch")
        before, after = vector(row["coefficient_before"]), vector(row["coefficient_after"])
        need(torch.equal(before, previous), "support coefficient chain is broken")
        base, target, gradient = check_cells(row["cells"], image_id, [250, 500, 750], True)
        scalar_close(row["base_loss_before"], base)
        scalar_close(row["target_loss_before"], target)
        scalar_close(row["objective_before"], base - target)
        rounding_check(row["gradient"], gradient)
        g = vector(row["gradient"])
        requested = .01 * g / (g.norm() + 1e-12)
        raw = before + requested
        factor = min(float(.05 / (raw.norm() + 1e-12)), 1.)
        errors = [
            rounding_check(row["gradient_norm"], g.norm()),
            rounding_check(row["requested_step"], requested),
            rounding_check(row["raw_coefficient_norm"], raw.norm()),
            rounding_check(row["projection_factor"], factor),
            rounding_check(after, raw * factor),
            rounding_check(row["coefficient_norm_before"], before.norm()),
            rounding_check(row["coefficient_norm_after"], after.norm()),
            rounding_check(row["actual_step"], after - before),
            rounding_check(row["actual_step_norm"], (after - before).norm()),
            rounding_check(row["gradient_dot_actual_step"], torch.dot(g, after - before)),
        ]
        maximum_cpu_error = max(maximum_cpu_error, *errors)
        need(row["projection_applied"] == (row["raw_coefficient_norm"] > .05),
             "projection flag disagrees with recorded raw norm")
        need(float(after.norm()) <= .05 * (1 + ROUNDING_FACTOR), "coefficient exceeds radius beyond FP32 roundoff")
        objective_after = trace["steps"][index + 1]["objective_before"] if index < 5 else fb - ft
        scalar_close(row["objective_after"], objective_after)
        scalar_close(row["objective_change"], objective_after - row["objective_before"])
        embedding = row["quantized_embedding_step"]
        check_embedding_summary(embedding, "fp16")
        need(embedding["changed_count"] <= active_dimensions, "embedding change exceeds masked active dimensions")
        if embedding["changed_count"]:
            need(embedding["l2"] >= embedding["max_abs"] * (1 - ROUNDING_FACTOR), "embedding L2 below maximum element")
            need(embedding["l2"] <= math.sqrt(embedding["changed_count"]) * embedding["max_abs"] * (1 + ROUNDING_FACTOR),
                 "embedding L2 exceeds count-based upper bound")
        COUNTS["forward"] += 2 * len(row["cells"])
        COUNTS["backward"] += 2 * len(row["cells"])
        COUNTS["support_steps"] += 1
        previous = after
    need(torch.equal(vector(trace["final_coefficient"]), previous), "final coefficient differs from last update")
    scalar_close(trace["initial_objective"], trace["steps"][0]["objective_before"])
    scalar_close(trace["final_objective"], fb - ft)
    scalar_close(trace["objective_gain"], trace["final_objective"] - trace["initial_objective"])
    need((trace["forward"], trace["backward"], trace["extra_final_objective_forward"]) == (42, 36, 6),
         "support counter summary disagrees with cells")
    return maximum_cpu_error


def verify_data(path):
    torch.set_num_threads(2)
    verify_inputs()
    report, protocol = load(path / "report.json"), load(path / "protocol.json")
    need(report["status"] == "COMPLETED_TRACE_AND_REPRODUCTION_MEASUREMENTS", "replay is not completed")
    need(protocol["schema"] == "u-traced-replay/v1", "unexpected replay schema")
    need(protocol["precision"] == report["precision"] == "fp16", "precision contract mismatch")
    need(protocol["scenario"] == report["scenario"] == "U", "scenario mismatch")
    need(protocol["models"] == ["model_1", "model_2"], "model list mismatch")
    need(protocol["new_patients"] == report["new_patients"] == 0, "new patients entered replay")
    need(protocol["fitting"] is False and report["score_direction_or_threshold_fitted"] is False,
         "this verifier expects no score fitting")
    need(report["calibration_test_patients_queried"] is False and report["medical_performance_or_novelty_claim"] is False,
         "unexpected scientific success claim")
    need(protocol["reference_query_gradients_computed"] is False and report["reference_query_gradients_computed"] is False,
         "unexpected reference-gradient claim")
    need(protocol["support_objective_improvement_is_measurement_not_pass_requirement"] is True,
         "optimization outcomes must remain measurements")
    need((report["step"], report["training_dir"]) == (1000, "training_coverage_v2"), "checkpoint contract changed")
    bindings = protocol["source_bindings"]
    need(bindings == report["source_bindings"], "protocol/report source bindings differ")
    CONTEXT["stage"] = "source and checkpoint hashes"
    for key, source in (
        ("legacy_results_sha256", OLD / "results.json"), ("legacy_report_sha256", OLD / "report.json"),
        ("legacy_verification_sha256", OLD / "verification.json"),
        ("cohort_lock_sha256", RUN / "cohort/lock.json"),
        ("cache_summary_sha256", RUN / "cache/summary.json"),
    ):
        need(bindings[key] == digest(source), "hash mismatch: " + str(source))
    code = Path(__file__).parent
    for name, expected in bindings["code_sha256"].items():
        need(Path(name).name == name, "code binding must be a basename")
        need(digest(code / name) == expected, "producer dependency changed: " + name)
    need(set(bindings["code_sha256"]) == {
        "run_traced_replay.py", "probe_verified.py", "probe.py", "models.py", "common.py", "run_probe_smoke.py"},
        "source-code binding set differs from contract")
    cache_path = RUN / "cache/cache.pt"
    need(digest(cache_path) == load(RUN / "cache/summary.json")["cache_sha256"], "cache content hash mismatch")
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    active_dimensions = int(cache["token_mask"].sum()) * cache["hidden"][cache["audit_prompt"]].shape[-1]
    legacy = load(OLD / "results.json")
    old_verification = load(OLD / "verification.json")
    need(old_verification["status"] == "PASS", "legacy verification is not PASS")
    need(old_verification["results_sha256"] == digest(OLD / "results.json"), "legacy verification does not bind source")
    old_pairs = [(r["model"], r["patient_id"]) for r in legacy]
    expected_pairs = [(m, p) for m in ("model_1", "model_2") for p in protocol["patient_order"]]
    need(len(protocol["patient_order"]) == len(set(protocol["patient_order"])) == 8, "patient count/order duplicated")
    need(old_pairs == expected_pairs and len(set(old_pairs)) == 16, "legacy original ordering differs")
    old_by_pair = {(r["model"], r["patient_id"]): r for r in legacy}
    evaluation = read_csv(RUN / "cohort/evaluation_images.csv")
    auxiliary = read_csv(RUN / "cohort/auxiliary_images.csv")
    image_rows = {r["image_id"]: r for r in evaluation + auxiliary}
    exposures, training = {}, {}
    for model in ("model_1", "model_2"):
        checkpoint = RUN / "training_coverage_v2" / model / "step_1000.pt"
        need(digest(checkpoint) == bindings["checkpoint_sha256"][model], "checkpoint hash mismatch: " + model)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        exposures[model] = state["exposures"]
        del state
        training[model] = {r["image_id"] for r in read_csv(RUN / "cohort" / (model + "_train.csv"))}
    results = load(path / "results.json")
    need(report["results_sha256"] == digest(path / "results.json"), "results hash mismatch")
    need(results == load(path / "partial_results.json"), "final partial results differ")
    need([(r["model"], r["patient_id"]) for r in results] == old_pairs, "replay ordering differs from original")
    support_summaries, detail_hashes = [], {}
    maximum_cpu_error = 0.0
    absolute_differences = {name: [] for name in FEATURES}
    exact_flags = []
    for result in results:
        model, patient = result["model"], result["patient_id"]
        CONTEXT["stage"] = model + "/patient " + patient
        old = old_by_pair[(model, patient)]
        need(result["eval_role"] == "fit" and result["scenario"] == "U", "wrong evaluation role")
        need(result["checkpoint_sha256"] == old["checkpoint_sha256"] == bindings["checkpoint_sha256"][model],
             "patient target checkpoint differs")
        patient_rows = [r for r in evaluation if r["patient_id"] == patient]
        member_manifest = int(any(r["image_id"] in training[model] for r in patient_rows))
        member_exposure = int(any(exposures[model].get(r["image_id"], 0) > 0 for r in patient_rows))
        need(result["declared_member"] == old["declared_member"] == member_manifest,
             "declared membership disagrees with manifest")
        need(result["realized_member"] == old["realized_member"] == member_exposure == member_manifest,
             "realized membership disagrees with exposures")
        expected_detail = Path("patients") / model / (patient + ".json")
        need(result["detail_path"] == expected_detail.as_posix(), "unexpected detail path")
        detail_path = path / expected_detail
        need(digest(detail_path) == result["detail_sha256"], "detail hash mismatch")
        detail_hashes[expected_detail.as_posix()] = result["detail_sha256"]
        detail = load(detail_path)
        need(detail["scalar_result"] == {k: v for k, v in result.items() if k != "detail_sha256"},
             "detail scalar snapshot differs from summary")
        for field in ("model", "patient_id", "scenario", "eval_role", "declared_member", "realized_member", "checkpoint_sha256"):
            need(detail[field] == result[field], "detail identity mismatch: " + field)
        need(detail["reference_query_gradients_computed"] is False, "detail claims reference gradients")
        need(len(detail["folds"]) == len(result["folds"]) == 2, "fold count differs")
        need(detail["image_order"] == [old["folds"][0]["support"], old["folds"][0]["query"]], "image order differs")
        before_counts = dict(COUNTS)
        reconstructed_folds = []
        initial_gradients = []
        for index, fold in enumerate(detail["folds"]):
            CONTEXT["stage"] = model + "/patient " + patient + "/fold " + str(index)
            own_support, own_query = fold["support"], fold["query"]
            need(own_support != own_query, "same image used as support and query")
            for image_id in (own_support, own_query):
                row = image_rows[image_id]
                need(row["patient_id"] == patient and row["eval_role"] == "fit" and row["record_role"] == "U_observed",
                     "support/query is not original patient's fit U image")
                need(image_id not in training[model] and exposures[model].get(image_id, 0) == 0,
                     "U image entered target training")
            need((own_support, own_query) == (old["folds"][index]["support"], old["folds"][index]["query"]),
                 "fold support/query order differs from original")
            trace = fold["support_trace"]
            maximum_cpu_error = max(maximum_cpu_error, check_trace(trace, own_support, active_dimensions))
            initial_gradients.append(vector(trace["steps"][0]["gradient"]))
            rounded_norm = vector(trace["final_coefficient"]).norm()
            rounding_check(fold["coefficient_norm"], rounded_norm)
            own_shift, _, _ = check_query(fold["query_evaluations"]["shifted"], own_query)
            own_zero, base_zero, target_zero = check_query(fold["query_evaluations"]["zero"], own_query)
            refs = fold["reference_evaluations"]
            need([r["image_id"] for r in refs] == cache["reference_matches"][own_query], "reference matching changed")
            need(len(refs) == len({r["patient_id"] for r in refs}) == 2, "reference patient distinctness violated")
            gains = []
            for ref in refs:
                rid = ref["image_id"]
                need(image_rows[rid]["eval_role"] == "reference", "reference image has wrong role")
                need(image_rows[rid]["patient_id"] == ref["patient_id"] != patient, "reference patient identity mismatch")
                need(rid not in training[model] and exposures[model].get(rid, 0) == 0, "reference exposed to target")
                shift, _, _ = check_query(ref["shifted"], rid)
                zero, _, _ = check_query(ref["zero"], rid)
                scalar_close(ref["gain"], shift - zero)
                gains.append(shift - zero)
            own_gain = own_shift - own_zero
            reference_gain = sum(gains) / 2
            recomputed = {"support": own_support, "query": own_query,
                          "coefficient_norm": fold["coefficient_norm"],
                          "own_gain": own_gain, "reference_gain": reference_gain,
                          "response": own_gain - reference_gain,
                          "base_loss": base_zero, "target_loss": target_zero, "base_minus_target": own_zero}
            for key, value in recomputed.items():
                if isinstance(value, str):
                    need(fold[key] == result["folds"][index][key] == value, "fold identity summary differs")
                else:
                    scalar_close(fold[key], value)
                    scalar_close(result["folds"][index][key], value)
            need(fold["reference_query_gradients_computed"] is False, "fold claims reference gradients")
            reconstructed_folds.append(recomputed)
            support_summaries.append({
                "model": model, "patient_id": patient, "support": own_support,
                "initial_objective": trace["initial_objective"], "final_objective": trace["final_objective"],
                "objective_gain": trace["objective_gain"], "final_coefficient_norm": fold["coefficient_norm"],
                "decreasing_steps": [r["step"] for r in trace["steps"] if r["objective_change"] < 0],
                "step_objective_changes": [r["objective_change"] for r in trace["steps"]],
            })
        need(reconstructed_folds[0]["support"] == reconstructed_folds[1]["query"] and
             reconstructed_folds[0]["query"] == reconstructed_folds[1]["support"], "folds are not swapped")
        f0, f1 = reconstructed_folds
        dot = float(torch.dot(initial_gradients[0], initial_gradients[1]))
        denominator = max(float(initial_gradients[0].norm() * initial_gradients[1].norm()), 1e-12)
        reconstructed = {
            "response": (f0["response"] + f1["response"]) / 2,
            "loss_mean": -(f0["target_loss"] + f1["target_loss"]) / 2,
            "loss_max": max(-f0["target_loss"], -f1["target_loss"]),
            "base_difference_mean": (f0["base_minus_target"] + f1["base_minus_target"]) / 2,
            "projected_conditioning_gradient_dot": dot,
            "projected_conditioning_gradient_cosine": dot / denominator,
        }
        for key in FEATURES:
            if key.startswith("projected_conditioning"):
                rounding_check(result[key], reconstructed[key])
            else:
                scalar_close(result[key], reconstructed[key])
            difference = result[key] - old[key]
            expected_difference = {"old": old[key], "new": result[key], "difference": difference,
                                   "absolute_difference": abs(difference), "exact_equal": result[key] == old[key]}
            need(result["legacy_scalar_differences"][key] == expected_difference, "legacy difference summary incorrect")
            absolute_differences[key].append(abs(difference))
            exact_flags.append(expected_difference["exact_equal"])
        need(COUNTS["forward"] - before_counts["forward"] == result["forward"] == 324, "patient forward count mismatch")
        need(COUNTS["backward"] - before_counts["backward"] == result["backward"] == 72, "patient backward count mismatch")
        need(result["extra_final_support_forward"] == 12 and result["weights_unchanged"] is True,
             "patient runtime-attestation summary inconsistent")
        need(result["seconds"] > 0 and result["max_cuda_GiB"] > 0, "invalid runtime/memory measurement")
    CONTEXT["stage"] = "aggregate replay report"
    need(report["support_diagnostics_sha256"] == digest(path / "support_diagnostics.json"), "support summary hash mismatch")
    need(load(path / "support_diagnostics.json") == support_summaries, "support summary differs from raw traces")
    need((report["unique_fit_patients"], report["model_patient_evaluations"]) == (8, 16), "aggregate sample counts wrong")
    need((protocol["expected_forward_per_patient"], protocol["expected_backward_per_patient"]) == (324, 72),
         "protocol costs wrong")
    need((report["forward_per_patient"], report["backward_per_patient"], report["extra_final_support_forward_per_patient"])
         == (324, 72, 12), "report per-patient costs wrong")
    need((COUNTS["forward"], COUNTS["backward"]) == (5184, 1152), "derived cell costs wrong")
    need((report["total_forward"], report["total_backward"]) == (COUNTS["forward"], COUNTS["backward"]),
         "report total costs differ")
    need(report["all_legacy_scalars_exact"] == all(exact_flags), "aggregate exact reproduction flag incorrect")
    need(report["max_absolute_legacy_scalar_difference"] == {k: max(v) for k, v in absolute_differences.items()},
         "maximum scalar-difference summary incorrect")
    need(report["support_folds"] == 32, "support fold count wrong")
    need(report["support_final_gain_positive_count"] == sum(r["objective_gain"] > 0 for r in support_summaries),
         "support positive count wrong")
    need(report["support_final_gain_negative_count"] == sum(r["objective_gain"] < 0 for r in support_summaries),
         "support negative count wrong")
    need(report["support_folds_with_decreasing_steps"] == sum(bool(r["decreasing_steps"]) for r in support_summaries),
         "support decreasing-step count wrong")
    scalar_close(report["support_final_gain_min"], min(r["objective_gain"] for r in support_summaries))
    scalar_close(report["support_final_gain_max"], max(r["objective_gain"] for r in support_summaries))
    scalar_close(report["median_seconds_per_patient"], statistics.median(r["seconds"] for r in results))
    scalar_close(report["max_seconds_per_patient"], max(r["seconds"] for r in results))
    need(report["elapsed_seconds"] >= sum(r["seconds"] for r in results), "elapsed time below sum of patient times")
    need(report["all_weights_unchanged"] is True and len(report["model_reports"]) == 2, "model runtime summaries invalid")
    for model_report, model in zip(report["model_reports"], ("model_1", "model_2")):
        need(model_report["model"] == model and model_report["checkpoint_sha256"] == bindings["checkpoint_sha256"][model],
             "model report binding mismatch")
        need(model_report["adapter_tensor_values_exactly_unchanged"] is True and
             model_report["parameter_versions_and_gradients_unchanged"] is True, "weight attestation inconsistent")
        need((model_report["forward"], model_report["backward"]) == (2592, 576), "model cost summary wrong")
    progress = load(path / "progress.json")
    need(progress["status"] == "COMPLETE" and progress["completed"] == progress["total"] == 16,
         "progress does not indicate completed16")
    return {
        "status": "PASS_TRACE_METADATA_AND_ARITHMETIC_ONLY",
        "failures": [], "GPU_executions_by_verifier": 0, "new_patients": 0,
        "unique_fit_patients": 8, "model_patient_evaluations": 16,
        "independently_reconstructed_counts": dict(COUNTS),
        "maximum_CPU_projection_arithmetic_discrepancy": maximum_cpu_error,
        "all_legacy_scalars_exact": all(exact_flags),
        "max_absolute_legacy_scalar_difference": {k: max(v) for k, v in absolute_differences.items()},
        "support_final_gain_positive_count": sum(r["objective_gain"] > 0 for r in support_summaries),
        "support_final_gain_negative_count": sum(r["objective_gain"] < 0 for r in support_summaries),
        "support_folds_with_decreasing_steps": sum(bool(r["decreasing_steps"]) for r in support_summaries),
        "detail_sha256": detail_hashes,
        "input_sha256": {p.name: digest(p) for p in (
            path / "protocol.json", path / "report.json", path / "results.json", path / "support_diagnostics.json")},
        "arithmetic_roundoff_envelope": "64 times FP32 epsilon times local maximum absolute magnitude, floor1e-8; arithmetic only",
        "no_membership_or_gradient_accuracy_pass_threshold": True,
        "limitations": [
            "GPU predictions and gradients are saved producer evaluations, not independently recomputed by this CPU verifier.",
            "Parameter-version, absent-gradient and exact adapter-equality checks remain runtime attestations.",
            "Embedding-change bounds are checked; raw GPU embeddings were not retained for independent equality.",
            "Reference query gradients were not computed; saved data do not establish the matched first-order baseline.",
            "Optimization improvement and exact reproduction do not establish patient membership information or attack superiority.",
        ],
    }


def verify(path, verification_name="verification"):
    global COUNTS
    path = Path(path)
    if not (path / "report.json").exists():
        raise FileNotFoundError("completed replay report.json is required; no verification output written")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", verification_name):
        raise ValueError("verification name must be a simple name without extension")
    output = path / (verification_name + ".json")
    if output.exists():
        raise FileExistsError("verification output already exists; use a new --verification-name")
    COUNTS = {"forward": 0, "backward": 0, "support_steps": 0, "query_cells": 0}
    CONTEXT["stage"] = "initial provenance"
    try:
        result = verify_data(path)
    except Exception as error:
        result = {
            "status": "FAIL_TRACE_VERIFICATION",
            "failures": [{"context": CONTEXT["stage"], "exception": type(error).__name__,
                          "message": str(error), "traceback": traceback.format_exc(limit=5)}],
            "partial_reconstructed_counts": dict(COUNTS),
            "GPU_executions_by_verifier": 0, "no_membership_success_declared": True,
        }
    result["verifier_sha256"] = digest(Path(__file__))
    result["independent_numeric_helper_sha256"] = digest(Path(__file__).with_name("verify_numerical_diagnostic.py"))
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-tag", default="traced_U8_v1")
    parser.add_argument("--verification-name", default="verification")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.output_tag):
        raise ValueError("invalid output tag")
    path = RUN / "verification_20260914" / args.output_tag
    result = verify(path, args.verification_name)
    print(json.dumps(result, indent=2))
    if result["status"].startswith("FAIL"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

