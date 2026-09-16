"""Replay only the existing U8 pair with numerical traces, without fitting.

This is a new versioned measurement run. Frozen original code/results are read,
never overwritten. Reference query gradients are NOT computed in this replay;
saved coefficients and support gradients do not constitute a matched first-order
reference-corrected baseline.
"""
import argparse
import gc
import json
import re
import statistics
import time
from pathlib import Path

import torch

from .common import RUN, digest, read_csv, verify_inputs, write_json
from .models import adapter_state, load_cache, load_unet, scheduler, setup
from .probe import QUERY
from .probe_verified import DiagnosticProbe
from .run_probe_smoke import selection

FEATURES = (
    "response", "loss_mean", "loss_max", "base_difference_mean",
    "projected_conditioning_gradient_dot", "projected_conditioning_gradient_cosine",
)
LEGACY = RUN / "probe_smoke/training_coverage_v2/U_step_1000_n8"


def serial(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


def scalar_differences(result, old):
    return {
        k: {"old": old[k], "new": result[k], "difference": result[k] - old[k],
            "absolute_difference": abs(result[k] - old[k]),
            "exact_equal": result[k] == old[k]}
        for k in FEATURES
    }


def replay_patient(probe, image_ids, patient_for_image):
    if len(image_ids) != 2 or image_ids[0] == image_ids[1]:
        raise ValueError("exactly two distinct original U images required")
    started = time.perf_counter()
    before_f, before_b = probe.forward, probe.backward
    fold_summaries, fold_details, initial_gradients = [], [], []
    for support_id, query_id in (image_ids, image_ids[::-1]):
        support_trace = probe.trace_support(support_id)
        coefficient = torch.tensor(support_trace["final_coefficient"], device="cuda", dtype=torch.float32)
        initial_gradients.append(torch.tensor(
            support_trace["steps"][0]["gradient"], device="cuda", dtype=torch.float32))
        shifted = probe.delta_cells(query_id, coefficient, "query", QUERY, draws=2, gradient=False)
        zero = probe.delta_cells(query_id, torch.zeros_like(coefficient), "query", QUERY, draws=2, gradient=False)
        own_gain = shifted["value"] - zero["value"]
        references = probe.cache["reference_matches"][query_id]
        if len(references) != 2:
            raise AssertionError("original reference count changed")
        reference_evaluations = []
        for reference_id in references:
            shifted_reference = probe.delta_cells(
                reference_id, coefficient, "query", QUERY, draws=2, gradient=False)
            zero_reference = probe.delta_cells(
                reference_id, torch.zeros_like(coefficient), "query", QUERY, draws=2, gradient=False)
            reference_evaluations.append({
                "image_id": reference_id,
                "patient_id": patient_for_image[reference_id],
                "shifted": serial(shifted_reference), "zero": serial(zero_reference),
                "gain": shifted_reference["value"] - zero_reference["value"],
            })
        reference_gain = sum(r["gain"] for r in reference_evaluations) / len(reference_evaluations)
        fold = {
            "support": support_id, "query": query_id,
            "coefficient_norm": float(coefficient.norm()),
            "own_gain": own_gain, "reference_gain": reference_gain,
            "response": own_gain - reference_gain,
            "base_loss": zero["base_loss"], "target_loss": zero["target_loss"],
            "base_minus_target": zero["value"],
        }
        fold_summaries.append(fold)
        fold_details.append({
            **fold, "support_trace": support_trace,
            "query_evaluations": {"shifted": serial(shifted), "zero": serial(zero)},
            "reference_evaluations": reference_evaluations,
            "reference_query_gradients_computed": False,
        })
    torch.cuda.synchronize()
    probe.assert_weights_unchanged()
    dot = float(torch.dot(initial_gradients[0], initial_gradients[1]))
    denominator = max(float(initial_gradients[0].norm() * initial_gradients[1].norm()), 1e-12)
    result = {
        "response": sum(f["response"] for f in fold_summaries) / 2,
        "loss_mean": -sum(f["target_loss"] for f in fold_summaries) / 2,
        "loss_max": max(-f["target_loss"] for f in fold_summaries),
        "base_difference_mean": sum(f["base_minus_target"] for f in fold_summaries) / 2,
        "projected_conditioning_gradient_dot": dot,
        "projected_conditioning_gradient_cosine": dot / denominator,
        "folds": fold_summaries,
        "forward": probe.forward - before_f, "backward": probe.backward - before_b,
        "extra_final_support_forward": 12,
        "seconds": time.perf_counter() - started, "weights_unchanged": True,
    }
    if result["forward"] != 324 or result["backward"] != 72:
        raise AssertionError("unexpected traced-replay model-example counts")
    return result, fold_details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-tag", default="traced_U8_v1")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.output_tag):
        raise ValueError("output-tag must be a simple alphanumeric/underscore/hyphen name")
    out = RUN / "verification_20260914" / args.output_tag
    if out.exists():
        raise FileExistsError("immutable replay output already exists; use a new output-tag")

    setup()
    verify_inputs()
    old_results = json.loads((LEGACY / "results.json").read_text(encoding="utf-8"))
    old_report = json.loads((LEGACY / "report.json").read_text(encoding="utf-8"))
    old_verification = json.loads((LEGACY / "verification.json").read_text(encoding="utf-8"))
    if digest(LEGACY / "results.json") != old_verification["results_sha256"]:
        raise AssertionError("legacy results changed since independent verification")
    if digest(LEGACY / "report.json") != old_verification["report_sha256"]:
        raise AssertionError("legacy report changed since independent verification")
    if old_report["unique_fit_patients"] != 8 or len(old_results) != 16:
        raise AssertionError("this replay is restricted to original U8 x two models")
    selected = selection("U")
    selected_by_patient = dict(selected)
    selected_ids = [p for p, _ in selected]
    expected_order = [(m, p) for m in ("model_1", "model_2") for p in selected_ids]
    if [(r["model"], r["patient_id"]) for r in old_results] != expected_order:
        raise AssertionError("original selection or source ordering changed")
    old_by_pair = {(r["model"], r["patient_id"]): r for r in old_results}
    evaluation = read_csv(RUN / "cohort/evaluation_images.csv")
    auxiliary = read_csv(RUN / "cohort/auxiliary_images.csv")
    patient_for_image = {r["image_id"]: r["patient_id"] for r in evaluation + auxiliary}
    all_eval_for_patient = {
        p: [r for r in evaluation if r["patient_id"] == p] for p in selected_ids
    }
    cache = load_cache()
    checkpoints = {
        model: RUN / "training_coverage_v2" / model / "step_1000.pt"
        for model in ("model_1", "model_2")
    }
    checkpoint_hashes = {m: digest(p) for m, p in checkpoints.items()}
    code = Path(__file__).parent
    source_bindings = {
        "legacy_results_sha256": digest(LEGACY / "results.json"),
        "legacy_report_sha256": digest(LEGACY / "report.json"),
        "legacy_verification_sha256": digest(LEGACY / "verification.json"),
        "cohort_lock_sha256": digest(RUN / "cohort/lock.json"),
        "cache_summary_sha256": digest(RUN / "cache/summary.json"),
        "checkpoint_sha256": checkpoint_hashes,
        "code_sha256": {name: digest(code / name) for name in (
            "run_traced_replay.py", "probe_verified.py", "probe.py",
            "models.py", "common.py", "run_probe_smoke.py")},
    }
    for old in old_results:
        if old["checkpoint_sha256"] != checkpoint_hashes[old["model"]]:
            raise AssertionError("target checkpoint differs from original U8")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "protocol.json", {
        "schema": "u-traced-replay/v1", "precision": "fp16", "scenario": "U",
        "models": ["model_1", "model_2"], "patient_order": selected_ids,
        "expected_forward_per_patient": 324, "expected_backward_per_patient": 72,
        "source_bindings": source_bindings, "new_patients": 0, "fitting": False,
        "support_objective_improvement_is_measurement_not_pass_requirement": True,
        "reference_query_gradients_computed": False,
    })
    started = time.perf_counter()
    results, support_diagnostics, model_reports = [], [], []
    for model, group in (("model_1", "A"), ("model_2", "B")):
        state = torch.load(checkpoints[model], map_location="cpu", weights_only=True)
        exposure = state["exposures"]
        del state
        unet = load_unet(checkpoints[model], training=False)
        probe = DiagnosticProbe(unet, scheduler(), cache, precision="fp16")
        before_adapter = adapter_state(unet)
        torch.cuda.reset_peak_memory_stats()
        for p in selected_ids:
            rows = selected_by_patient[p]
            if not str(p).isdigit():
                raise AssertionError("unexpected patient identifier format")
            ids = sorted(r["image_id"] for r in rows)
            old = old_by_pair[(model, p)]
            if [old["folds"][0]["support"], old["folds"][0]["query"]] != ids:
                raise AssertionError("original image order changed")
            if any(r["record_role"] != "U_observed" or r["eval_role"] != "fit" for r in rows):
                raise AssertionError("replay candidate is not original fit-role U data")
            declared = int(rows[0]["assignment_group"] == group)
            realized = int(any(exposure.get(r["image_id"], 0) > 0 for r in all_eval_for_patient[p]))
            if any(exposure.get(image_id, 0) > 0 for image_id in ids):
                raise AssertionError("a U image was exposed during target training")
            if declared != realized or declared != old["declared_member"] or realized != old["realized_member"]:
                raise AssertionError("original participation labels changed")
            result, details = replay_patient(probe, ids, patient_for_image)
            detail_path = Path("patients") / model / (str(p) + ".json")
            result.update({
                "model": model, "patient_id": p, "eval_role": "fit", "scenario": "U",
                "declared_member": declared, "realized_member": realized,
                "checkpoint_sha256": checkpoint_hashes[model],
                "max_cuda_GiB": torch.cuda.max_memory_allocated() / 2**30,
                "detail_path": detail_path.as_posix(),
            })
            result["legacy_scalar_differences"] = scalar_differences(result, old)
            for fold in details:
                trace = fold["support_trace"]
                support_diagnostics.append({
                    "model": model, "patient_id": p, "support": fold["support"],
                    "initial_objective": trace["initial_objective"],
                    "final_objective": trace["final_objective"],
                    "objective_gain": trace["objective_gain"],
                    "final_coefficient_norm": fold["coefficient_norm"],
                    "decreasing_steps": [r["step"] for r in trace["steps"] if r["objective_change"] < 0],
                    "step_objective_changes": [r["objective_change"] for r in trace["steps"]],
                })
            write_json(out / detail_path, {
                "model": model, "patient_id": p, "scenario": "U", "eval_role": "fit",
                "declared_member": declared, "realized_member": realized,
                "checkpoint_sha256": checkpoint_hashes[model], "image_order": ids,
                "scalar_result": result, "folds": details,
                "reference_query_gradients_computed": False,
            })
            result["detail_sha256"] = digest(out / detail_path)
            results.append(result)
            write_json(out / "partial_results.json", results)
            progress = {
                "status": "RUNNING", "completed": len(results), "total": 16,
                "model": model, "patient_id": p,
                "elapsed_seconds": time.perf_counter() - started,
                "last_patient_seconds": result["seconds"],
                "all_legacy_scalars_exact_this_patient": all(
                    d["exact_equal"] for d in result["legacy_scalar_differences"].values()),
            }
            write_json(out / "progress.json", progress)
            print(json.dumps(progress), flush=True)
        probe.assert_weights_unchanged()
        after_adapter = adapter_state(unet)
        if before_adapter.keys() != after_adapter.keys() or not all(
                torch.equal(before_adapter[k], after_adapter[k]) for k in before_adapter):
            raise AssertionError("adapter tensor values changed during replay")
        model_reports.append({
            "model": model, "checkpoint_sha256": checkpoint_hashes[model],
            "adapter_tensor_values_exactly_unchanged": True,
            "parameter_versions_and_gradients_unchanged": True,
            "forward": probe.forward, "backward": probe.backward,
        })
        del probe, unet, before_adapter, after_adapter
        gc.collect()
        torch.cuda.empty_cache()
    write_json(out / "results.json", results)
    write_json(out / "support_diagnostics.json", support_diagnostics)
    report = {
        "status": "COMPLETED_TRACE_AND_REPRODUCTION_MEASUREMENTS",
        "training_dir": "training_coverage_v2", "step": 1000, "scenario": "U",
        "precision": "fp16", "unique_fit_patients": 8, "model_patient_evaluations": 16,
        "forward_per_patient": 324, "backward_per_patient": 72,
        "extra_final_support_forward_per_patient": 12,
        "total_forward": sum(r["forward"] for r in results),
        "total_backward": sum(r["backward"] for r in results),
        "median_seconds_per_patient": statistics.median(r["seconds"] for r in results),
        "max_seconds_per_patient": max(r["seconds"] for r in results),
        "elapsed_seconds": time.perf_counter() - started,
        "all_weights_unchanged": True,
        "all_legacy_scalars_exact": all(
            d["exact_equal"] for r in results for d in r["legacy_scalar_differences"].values()),
        "max_absolute_legacy_scalar_difference": {
            k: max(r["legacy_scalar_differences"][k]["absolute_difference"] for r in results) for k in FEATURES},
        "support_folds": len(support_diagnostics),
        "support_final_gain_positive_count": sum(r["objective_gain"] > 0 for r in support_diagnostics),
        "support_final_gain_negative_count": sum(r["objective_gain"] < 0 for r in support_diagnostics),
        "support_folds_with_decreasing_steps": sum(bool(r["decreasing_steps"]) for r in support_diagnostics),
        "support_final_gain_min": min(r["objective_gain"] for r in support_diagnostics),
        "support_final_gain_max": max(r["objective_gain"] for r in support_diagnostics),
        "source_bindings": source_bindings, "model_reports": model_reports,
        "results_sha256": digest(out / "results.json"),
        "support_diagnostics_sha256": digest(out / "support_diagnostics.json"),
        "score_direction_or_threshold_fitted": False,
        "calibration_test_patients_queried": False, "new_patients": 0,
        "reference_query_gradients_computed": False,
        "medical_performance_or_novelty_claim": False,
    }
    write_json(out / "report.json", report)
    write_json(out / "progress.json", {
        "status": "COMPLETE", "completed": 16, "total": 16,
        "elapsed_seconds": report["elapsed_seconds"],
    })
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

