"""CPU-only arithmetic and provenance verification of actual numerical probes.

Uses saved model evaluations; never executes a target model or fits an attack.
Floating-point arithmetic integrity and scientific gradient adequacy are separate.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics as st

import torch

from .common import ROOT, RUN, SALT, digest, read_csv, verify_inputs, write_json

OLD = RUN / "probe_smoke/training_coverage_v2/U_step_1000_n8"
PRECISIONS = ("fp16", "fp32")
ROUNDING_FACTOR = 64 * torch.finfo(torch.float32).eps


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def scalar_close(value, expected):
    assert math.isfinite(float(value)) and math.isfinite(float(expected))
    assert math.isclose(float(value), float(expected), rel_tol=1e-11, abs_tol=1e-14), (value, expected)


def vector(value):
    result = torch.tensor(value, dtype=torch.float32, device="cpu")
    assert result.shape == (8,) and bool(torch.isfinite(result).all())
    return result


def rounding_check(observed, expected):
    """Fixed FP32 roundoff envelope, not a gradient-quality acceptance rule."""
    a = torch.as_tensor(observed, dtype=torch.float32)
    b = torch.as_tensor(expected, dtype=torch.float32)
    assert a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
    scale = max(float(a.abs().max()), float(b.abs().max()), 1e-8)
    difference = float((a - b).abs().max())
    assert difference <= ROUNDING_FACTOR * scale, (difference, scale, observed, expected)
    return difference


def vector_comparison(first, second):
    a, b = vector(first), vector(second)
    difference = a - b
    na, nb = float(a.norm()), float(b.norm())
    return {
        "max_abs": float(difference.abs().max()),
        "relative_l2": float(difference.norm() / max(na, 1e-12)),
        "relative_l2_denominator": "first vector norm, floored at 1e-12",
        "exact_equal": bool(torch.equal(a, b)),
        "first_norm": na, "second_norm": nb,
        "cosine": float(torch.dot(a.double(), b.double()) / (a.double().norm() * b.double().norm())) if na and nb else None,
        "component_sign_disagreements": int(((a > 0).int() - (a < 0).int() != (b > 0).int() - (b < 0).int()).sum()),
    }


def check_cells(cells, image_id, timesteps, gradient):
    assert [(r["timestep"], r["draw"]) for r in cells] == [(t, 0) for t in timesteps]
    count = len(cells)
    base_sum = target_sum = 0.0
    base_gradient = torch.zeros(8)
    target_gradient = torch.zeros(8)
    for cell in cells:
        assert cell["phase"] == "support" and cell["gradient_computed"] is gradient
        text = "|".join(map(str, (SALT, "audit-noise", "support", image_id, cell["timestep"], 0)))
        assert cell["noise_seed"] == int(hashlib.sha256(text.encode()).hexdigest()[:15], 16)
        assert cell["base_loss"] >= 0 and cell["target_loss"] >= 0
        scalar_close(cell["value"], cell["base_loss"] - cell["target_loss"])
        base_sum += cell["base_loss"] / count
        target_sum += cell["target_loss"] / count
        if gradient:
            bg, tg = vector(cell["base_gradient"]), vector(cell["target_gradient"])
            rounding_check(cell["gradient"], bg - tg)
            base_gradient += bg / count
            target_gradient += tg / count
        else:
            assert all(cell[k] is None for k in ("base_gradient", "target_gradient", "gradient"))
    return base_sum, target_sum, base_gradient - target_gradient


def check_delta(delta, image_id, timesteps, gradient):
    base, target, grad = check_cells(delta["cells"], image_id, timesteps, gradient)
    scalar_close(delta["base_loss"], base)
    scalar_close(delta["target_loss"], target)
    scalar_close(delta["value"], base - target)
    rounding_check(delta["gradient"], grad)
    return base - target, grad


def check_embedding_summary(value, precision):
    assert value["precision"] == precision
    assert 0 <= value["changed_count"] <= value["element_count"]
    assert value["element_count"] == 77 * 1024
    assert math.isfinite(value["l2"]) and math.isfinite(value["max_abs"])
    assert value["l2"] >= 0 and value["max_abs"] >= 0
    if value["changed_count"] == 0:
        assert value["l2"] == value["max_abs"] == 0
    else:
        assert value["l2"] > 0 and value["max_abs"] > 0


def compare_previous_phase(previous, current):
    """Compare every retained old measurement; omit runtime/memory observations."""
    checked = 0
    differences = []

    def walk(old, new, path):
        nonlocal checked
        if isinstance(old, dict):
            for key, value in old.items():
                if key in {"seconds", "peak_cuda_GiB"}:
                    continue
                assert key in new, path + "/" + key
                walk(value, new[key], path + "/" + key)
        elif isinstance(old, list):
            assert isinstance(new, list) and len(old) == len(new), path
            for index, (left, right) in enumerate(zip(old, new)):
                walk(left, right, path + "/" + str(index))
        else:
            checked += 1
            if old != new:
                differences.append(dict(path=path, previous=old, current=new))

    walk(previous, current, "")
    return dict(scalar_measurements_compared=checked, exact_reproduction=not differences,
                difference_count=len(differences), first_differences=differences[:100],
                excluded_fields=["seconds", "peak_cuda_GiB"],
                newly_added_raw_comparison_fields="not present in v1; independently checked in current run")


def check_phase(phase, precision, image_id):
    assert phase["status"] == "NUMERICAL_MEASUREMENTS_COMPLETE"
    assert phase["precision"] == precision and phase["weights_unchanged"] is True
    trace = phase["trace"]
    assert trace["precision"] == precision and trace["image_id"] == image_id
    assert trace["support_timesteps"] == [250, 500, 750]
    assert trace["draws_per_timestep"] == 1 and trace["gradient_scale"] == 1024.
    assert trace["step_size"] == .01 and trace["radius"] == .05
    assert trace["initial_coefficient"] == [0.] * 8
    assert trace["weights_unchanged"] is True and len(trace["steps"]) == 6
    final_value, _ = check_delta(trace["final_evaluation"], image_id, [250, 500, 750], False)
    cpu_update_discrepancies = []
    previous = vector(trace["initial_coefficient"])
    for index, row in enumerate(trace["steps"]):
        assert row["step"] == index
        before, after = vector(row["coefficient_before"]), vector(row["coefficient_after"])
        assert torch.equal(before, previous)
        base, target, gradient = check_cells(row["cells"], image_id, [250, 500, 750], True)
        scalar_close(row["base_loss_before"], base)
        scalar_close(row["target_loss_before"], target)
        scalar_close(row["objective_before"], base - target)
        rounding_check(row["gradient"], gradient)
        # Recompute from stored gradient without calling the producer update helper.
        g = vector(row["gradient"])
        requested = .01 * g / (g.norm() + 1e-12)
        raw = before + requested
        factor = min(float(.05 / (raw.norm() + 1e-12)), 1.0)
        expected_after = raw * factor
        errors = [rounding_check(row["requested_step"], requested),
                  rounding_check(row["raw_coefficient_norm"], raw.norm()),
                  rounding_check(row["projection_factor"], factor),
                  rounding_check(after, expected_after),
                  rounding_check(row["gradient_norm"], g.norm()),
                  rounding_check(row["coefficient_norm_before"], before.norm()),
                  rounding_check(row["coefficient_norm_after"], after.norm()),
                  rounding_check(row["actual_step"], after - before),
                  rounding_check(row["actual_step_norm"], (after - before).norm()),
                  rounding_check(row["gradient_dot_actual_step"], torch.dot(g, after - before))]
        assert row["projection_applied"] == (row["raw_coefficient_norm"] > .05)
        assert float(after.norm()) <= .05 * (1 + ROUNDING_FACTOR)
        expected_objective_after = trace["steps"][index + 1]["objective_before"] if index < 5 else final_value
        scalar_close(row["objective_after"], expected_objective_after)
        scalar_close(row["objective_change"], expected_objective_after - row["objective_before"])
        check_embedding_summary(row["quantized_embedding_step"], precision)
        cpu_update_discrepancies.append(max(errors))
        previous = after
    assert torch.equal(vector(trace["final_coefficient"]), previous)
    scalar_close(trace["initial_objective"], trace["steps"][0]["objective_before"])
    scalar_close(trace["final_objective"], final_value)
    scalar_close(trace["objective_gain"], final_value - trace["initial_objective"])
    assert trace["forward"] == 42 and trace["backward"] == 36
    assert trace["extra_final_objective_forward"] == 6

    cell0 = next(c for c in trace["steps"][0]["cells"] if c["timestep"] == 500)
    ad_gradient = vector(cell0["gradient"])
    rounding_check(phase["initial_gradient_norm"], ad_gradient.norm())
    checkpoint = phase["checkpointing_check"]
    scalar_close(checkpoint["on_value"], cell0["value"])
    scalar_close(checkpoint["loss_absolute_difference"], abs(checkpoint["on_value"] - checkpoint["off_value"]))
    comparison = checkpoint["gradient"]
    assert comparison["max_abs"] >= 0 and comparison["relative_l2"] >= 0
    assert comparison["exact_equal"] == (comparison["max_abs"] == 0)
    if comparison["exact_equal"]:
        assert comparison["relative_l2"] == 0
    raw_checkpoint_comparison = None
    if phase.get("raw_checkpointing") is not None:
        raw = phase["raw_checkpointing"]
        on_value, on_gradient = check_delta(raw["on"], image_id, [500], True)
        off_value, off_gradient = check_delta(raw["off"], image_id, [500], True)
        scalar_close(on_value, checkpoint["on_value"])
        scalar_close(off_value, checkpoint["off_value"])
        assert torch.equal(on_gradient, ad_gradient)
        raw_checkpoint_comparison = vector_comparison(on_gradient.tolist(), off_gradient.tolist())
        for name in ("max_abs", "relative_l2"):
            rounding_check(comparison[name], raw_checkpoint_comparison[name])
        assert comparison["exact_equal"] == raw_checkpoint_comparison["exact_equal"]
    repeats = phase["repeated_values"]
    assert len(repeats) == 2 and all(math.isfinite(v) for v in repeats)
    scalar_close(phase["repeat_absolute_range"], max(repeats) - min(repeats))
    raw_legacy_comparison = None
    if precision == "fp16":
        legacy = phase["legacy_cell_reproduction"]
        assert legacy is not None and legacy["gradient"]["exact_equal"] is True
        assert all(legacy[k] == 0 for k in ("value_absolute_difference", "base_loss_absolute_difference", "target_loss_absolute_difference"))
        assert legacy["gradient"]["max_abs"] == legacy["gradient"]["relative_l2"] == 0
        assert phase["legacy_extra_forward"] == phase["legacy_extra_backward"] == 2
        if phase.get("raw_legacy") is not None:
            raw_legacy = phase["raw_legacy"]
            scalar_close(raw_legacy["value"], raw_legacy["base_loss"] - raw_legacy["target_loss"])
            for name, cell_name in (("value", "value"), ("base_loss", "base_loss"), ("target_loss", "target_loss")):
                scalar_close(legacy[name + "_absolute_difference"], abs(raw_legacy[name] - cell0[cell_name]))
            raw_legacy_comparison = vector_comparison(raw_legacy["gradient"], ad_gradient.tolist())
            for name in ("max_abs", "relative_l2"):
                rounding_check(legacy["gradient"][name], raw_legacy_comparison[name])
            assert legacy["gradient"]["exact_equal"] == raw_legacy_comparison["exact_equal"]
    else:
        assert phase["legacy_cell_reproduction"] is None
        assert phase["legacy_extra_forward"] == phase["legacy_extra_backward"] == 0

    expected_keys = {("gradient_direction", h) for h in (.005, .001, .0002)}
    if precision == "fp32":
        expected_keys |= {(f"coordinate_{i}", h) for i in range(8) for h in (.001, .0002)}
    records = phase["finite_differences"]
    assert len(records) == len(expected_keys)
    assert {(r["direction"], r["h"]) for r in records} == expected_keys
    errors = []
    for record in records:
        direction = record["direction"]
        if direction == "gradient_direction":
            v = ad_gradient / ad_gradient.norm()
        else:
            v = torch.zeros(8)
            v[int(direction.split("_")[-1])] = 1
        rounding_check(record["autograd"], torch.dot(ad_gradient, v))
        minus, _ = check_delta(record["minus"], image_id, [500], False)
        plus, _ = check_delta(record["plus"], image_id, [500], False)
        fd = (plus - minus) / (2 * record["h"])
        scalar_close(record["finite_difference"], fd)
        absolute = abs(fd - record["autograd"])
        relative = absolute / max(abs(fd), abs(record["autograd"]), 1e-12)
        scalar_close(record["absolute_error"], absolute)
        scalar_close(record["symmetric_relative_error"], relative)
        check_embedding_summary(record["embedding_change"], precision)
        errors.append({k: record[k] for k in ("direction", "h", "autograd", "finite_difference", "absolute_error", "symmetric_relative_error")})
    assert phase["forward"] == 42 + 4 + 2 + 4 * len(records)
    assert phase["backward"] == 38
    coordinate_vectors = {}
    if precision == "fp32":
        for h in (.001, .0002):
            fd_vector = [next(r["finite_difference"] for r in records if r["direction"] == f"coordinate_{i}" and r["h"] == h) for i in range(8)]
            coordinate_vectors[str(h)] = {"finite_difference_vector": fd_vector,
                                          "autograd_vector": ad_gradient.tolist(),
                                          "comparison": vector_comparison(ad_gradient.tolist(), fd_vector)}
    direction_rows = sorted([r for r in errors if r["direction"] == "gradient_direction"], key=lambda r: -r["h"])
    convergence = []
    for first, second in zip(direction_rows, direction_rows[1:]):
        convergence.append({"larger_h": first["h"], "smaller_h": second["h"],
                            "absolute_error_change": second["absolute_error"] - first["absolute_error"],
                            "absolute_error_decreased": second["absolute_error"] < first["absolute_error"],
                            "error_ratio_small_over_large": second["absolute_error"] / first["absolute_error"] if first["absolute_error"] else None})
    return {
        "precision": precision, "objective_gain": trace["objective_gain"],
        "objective_change_each_update": [r["objective_change"] for r in trace["steps"]],
        "objective_increased_updates": sum(r["objective_change"] > 0 for r in trace["steps"]),
        "maximum_CPU_projection_arithmetic_discrepancy": max(cpu_update_discrepancies),
        "gradient_direction_finite_differences": direction_rows,
        "successive_h_error_comparison": convergence,
        "coordinate_finite_difference_vectors": coordinate_vectors,
        "all_finite_difference_errors": errors,
        "checkpointing_runtime_comparison": checkpoint,
        "checkpointing_raw_vector_comparison": raw_checkpoint_comparison,
        "repeat_range": phase["repeat_absolute_range"],
        "repeat_minus_initial_value": [v - cell0["value"] for v in repeats],
        "legacy_runtime_comparison": phase["legacy_cell_reproduction"],
        "legacy_raw_vector_comparison": raw_legacy_comparison,
        "gradient_accuracy_pass_threshold_used": False,
    }


def verify(path):
    torch.set_num_threads(2)
    verify_inputs()
    report = load(path / "report.json")
    protocol = load(path / "protocol.json")
    assert report["protocol_sha256"] == digest(path / "protocol.json")
    assert protocol["schema"] in {"cvpr-u-actual-numerical/v1", "cvpr-u-actual-numerical/v2"}
    assert protocol["new_patients"] == protocol["new_training_steps"] == 0
    assert report["new_patients"] == report["new_training_steps"] == 0
    assert report["no_attack_performance_claim"] is True
    assert protocol["FP16_and_FP32_control"] is True
    assert protocol["finite_steps"] == [.005, .001, .0002]
    assert protocol["FP32_coordinate_steps"] == [.001, .0002]
    assert protocol["source_results_sha256"] == digest(OLD / "results.json")
    old_verification = load(OLD / "verification.json")
    assert old_verification["status"] == "PASS"
    assert old_verification["results_sha256"] == protocol["source_results_sha256"]
    first = next(r for r in load(OLD / "results.json") if r["model"] == "model_1")
    image_id = first["folds"][0]["support"]
    assert protocol["image_id"] == image_id
    assert protocol["original_patient_id"] == first["patient_id"]
    assert protocol["checkpoint_sha256"] == first["checkpoint_sha256"] == report["checkpoint_sha256"]
    checkpoint = RUN / "training_coverage_v2/model_1/step_1000.pt"
    assert digest(checkpoint) == report["checkpoint_sha256"]
    assert protocol["cache_sha256"] == digest(RUN / "cache/cache.pt")
    assert load(RUN / "cache/summary.json")["cache_sha256"] == protocol["cache_sha256"]
    for name, expected in protocol["code_sha256"].items():
        assert Path(name).name == name
        assert digest(Path(__file__).with_name(name)) == expected
    assert {"probe_verified.py", "probe.py", "models.py"}.issubset(protocol["code_sha256"])
    runner_names = set(protocol["code_sha256"]) - {"probe_verified.py", "probe.py", "models.py"}
    assert runner_names in ({"run_numerical_diagnostic.py"}, {"run_numerical_diagnostic_v2.py"})
    manifest = {r["image_id"]: r for r in read_csv(RUN / "cohort/evaluation_images.csv")}
    assert manifest[image_id]["patient_id"] == first["patient_id"]
    assert manifest[image_id]["eval_role"] == "fit" and manifest[image_id]["record_role"] == "U_observed"
    for model in ("model_1", "model_2"):
        assert image_id not in {r["image_id"] for r in read_csv(RUN / "cohort" / (model + "_train.csv"))}
    phases = {}
    summaries = {}
    for precision in PRECISIONS:
        assert report["phase_hashes"][precision] == digest(path / (precision + ".json"))
        phases[precision] = load(path / (precision + ".json"))
        summaries[precision] = check_phase(phases[precision], precision, image_id)
        scalar_close(report["objective_gains"][precision], summaries[precision]["objective_gain"])
    total_forward = sum(r["forward"] + r["legacy_extra_forward"] for r in phases.values())
    total_backward = sum(r["backward"] + r["legacy_extra_backward"] for r in phases.values())
    assert report["totals"] == dict(forward=total_forward, backward=total_backward)
    traces = {key: value["trace"] for key, value in phases.items()}
    coefficient = vector_comparison(traces["fp16"]["final_coefficient"], traces["fp32"]["final_coefficient"])
    stored = report["final_coefficient_precision_comparison"]
    rounding_check(stored["max_abs"], coefficient["max_abs"])
    rounding_check(stored["relative_l2"], coefficient["relative_l2"])
    assert stored["exact_equal"] == coefficient["exact_equal"]
    first_gradients = {p: traces[p]["steps"][0]["gradient"] for p in PRECISIONS}
    first_cells = {p: next(r["gradient"] for r in traces[p]["steps"][0]["cells"] if r["timestep"] == 500) for p in PRECISIONS}
    raw_comparisons_verified = all(summaries[p]["checkpointing_raw_vector_comparison"] is not None for p in PRECISIONS) and summaries["fp16"]["legacy_raw_vector_comparison"] is not None
    if protocol["schema"] == "cvpr-u-actual-numerical/v2":
        assert protocol["raw_comparison_evidence"] is True and raw_comparisons_verified
    summary = {
        "scope": "one existing fit image, one target checkpoint, fixed support noise; numerical diagnostic only",
        "unique_patients": 1, "new_patients": 0, "new_training_steps": 0,
        "phases": summaries,
        "FP16_vs_FP32": {
            "initial_three_timestep_gradient": vector_comparison(first_gradients["fp16"], first_gradients["fp32"]),
            "initial_t500_gradient": vector_comparison(first_cells["fp16"], first_cells["fp32"]),
            "final_coefficient": coefficient,
            "objective_gain_difference_fp16_minus_fp32": traces["fp16"]["objective_gain"] - traces["fp32"]["objective_gain"],
            "control_definition": "FP32 promotes original FP16 pretrained tensors and fixed noisy inputs/targets; it is not a separately recovered full-precision model.",
        },
        "arithmetic_correctness": "saved cell aggregation, update/projection and FD arithmetic independently checked on CPU",
        "no_gradient_quality_threshold_or_membership_success_declared": True,
        "limitations": [
            ("Checkpoint-on/off and FP16 legacy raw gradient vectors were retained and their equality/discrepancy summaries independently recomputed."
             if raw_comparisons_verified else
             "Checkpoint-off and/or legacy raw gradient vectors were not retained. Missing raw-vector comparisons remain runtime attestations checked for consistency, not independently reconstructed vectors."),
            "Embedding-change summaries are checked for finite values and shape bounds; raw GPU embeddings were not retained.",
            "All eight FP32 coordinate derivatives are checked at two h values and gradient direction at three; finite differences are local to a=0, t=500, one noise/image/checkpoint.",
            "Finite-difference disagreement may include finite-step truncation, floating-point roundoff and FP16 quantization; no universal all-gradients-proven claim.",
            "FP16 and FP32 optimization paths diverge after the first update; later trajectory gradients are not compared as if taken at identical coefficients.",
            "Numerical correctness does not establish patient membership information, attack superiority, or clinical utility.",
        ],
    }
    previous_directory = RUN / "verification_20260914/numerical_v1"
    if path.resolve() != previous_directory.resolve() and (previous_directory / "report.json").exists():
        previous_report = load(previous_directory / "report.json")
        previous_protocol = load(previous_directory / "protocol.json")
        assert digest(previous_directory / "protocol.json") == previous_report["protocol_sha256"]
        for key in ("source_results_sha256", "checkpoint_sha256", "cache_sha256", "image_id"):
            assert previous_protocol[key] == protocol[key]
        previous_comparison = {}
        for precision in PRECISIONS:
            prior_path = previous_directory / (precision + ".json")
            assert digest(prior_path) == previous_report["phase_hashes"][precision]
            previous_comparison[precision] = compare_previous_phase(load(prior_path), phases[precision])
        summary["prior_numerical_v1_reproduction"] = previous_comparison
    verification = {
        "status": "PASS_NUMERIC_INTEGRITY_ONLY",
        "cell_aggregation_projection_FD_arithmetic_checked": True,
        "source_checkpoint_protocol_phase_hashes_checked": True,
        "precision_phases": list(PRECISIONS), "trace_updates_checked": 12,
        "finite_difference_records_checked": sum(len(r["finite_differences"]) for r in phases.values()),
        "runtime_checkpoint_legacy_attestations_consistent": True,
        "independently_recomputed_checkpoint_legacy_gradient_vectors": raw_comparisons_verified,
        "GPU_executions_by_verifier": 0, "new_patients": 0,
        "arithmetic_roundoff_envelope": "64 times FP32 epsilon times local maximum absolute magnitude, floor 1e-8; arithmetic-only check, not gradient-accuracy threshold",
        "input_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in
                         [path / "protocol.json", path / "report.json", path / "fp16.json", path / "fp32.json",
                          OLD / "results.json", OLD / "verification.json", RUN / "cohort/evaluation_images.csv"]},
        "verifier_sha256": digest(Path(__file__)),
    }
    write_json(path / "interpretation.json", summary)
    verification["interpretation_sha256"] = digest(path / "interpretation.json")
    write_json(path / "verification.json", verification)
    return verification


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--directory", type=Path)
    group.add_argument("--output-tag", default=None)
    args = parser.parse_args()
    if args.output_tag is not None:
        assert args.output_tag.replace("_", "").isalnum()
    directory = args.directory or RUN / "verification_20260914" / (args.output_tag or "numerical_v1")
    print(json.dumps(verify(directory), indent=2))


if __name__ == "__main__":
    main()
