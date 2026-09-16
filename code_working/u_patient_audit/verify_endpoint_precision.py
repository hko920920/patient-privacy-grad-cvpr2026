"""Independent CPU integrity/aggregation audit of fixed-coefficient FP32 endpoints."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from .common import ROOT, RUN, SALT, digest, read_csv, verify_inputs, write_json

MODELS = ("model_1", "model_2")
TS = (50, 250, 500, 750, 950)
COMPONENTS = ("own_gain", "reference_gain", "response",
              "base_loss", "target_loss", "base_minus_target")
LEGACY = RUN / "probe_smoke/training_coverage_v2/U_step_1000_n8"
REPLAY = RUN / "verification_20260914/traced_U8_v1"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def close(a, b):
    assert math.isfinite(float(a)) and math.isfinite(float(b))
    assert math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12), (a, b)


def sign(value):
    return int(value > 0) - int(value < 0)


def check_delta(result, image_id):
    cells = result["cells"]
    assert len(cells) == 10
    assert [(c["timestep"], c["draw"]) for c in cells] == [
        (t, d) for t in TS for d in range(2)]
    for cell in cells:
        assert cell["phase"] == "query" and cell["gradient_computed"] is False
        text = "|".join(map(str, (SALT, "audit-noise", "query",
                                 image_id, cell["timestep"], cell["draw"])))
        assert cell["noise_seed"] == int(hashlib.sha256(text.encode()).hexdigest()[:15], 16)
        for name in ("base_loss", "target_loss"):
            assert math.isfinite(cell[name]) and cell[name] >= 0
        close(cell["value"], cell["base_loss"]-cell["target_loss"])
        for key in ("base_gradient", "target_gradient", "gradient"):
            assert cell[key] is None
    base = math.fsum(c["base_loss"] for c in cells)/10
    target = math.fsum(c["target_loss"] for c in cells)/10
    close(result["base_loss"], base)
    close(result["target_loss"], target)
    close(result["value"], base-target)
    assert len(result["gradient"]) == 8 and all(v == 0 for v in result["gradient"])
    return dict(base_loss=base, target_loss=target, value=base-target)


def verify(directory):
    started = time.perf_counter()
    torch.set_num_threads(2)
    verify_inputs()
    protocol = load(directory/"protocol.json")
    report = load(directory/"report.json")
    assert report["protocol_sha256"] == digest(directory/"protocol.json")
    expected = {
        "schema": "u-endpoint-precision/v1", "scenario": "U", "patient_id": "14393",
        "models": list(MODELS), "evaluation_precision": "fp32",
        "coefficient_precision_source": "saved original FP16 traced run",
        "audit_prompt": "a frontal chest radiograph", "coefficients_reoptimized": False,
        "same_coefficients_and_zero_used_for_both_precisions": True,
        "query_timesteps": list(TS), "draws": 2, "noise_phase": "query",
        "reference_count_per_fold": 2, "forward_per_model": 240,
        "total_forward": 480, "total_backward": 0,
        "new_patients": 0, "new_training_steps": 0, "score_fitting": False,
    }
    for key, value in expected.items():
        assert protocol[key] == value, key
    for filename, expected_hash in protocol["input_sha256"].items():
        source = (ROOT/filename).resolve()
        assert source.is_relative_to(ROOT.resolve())
        assert digest(source) == expected_hash, filename
    for filename, expected_hash in protocol["code_sha256"].items():
        source = (ROOT/"u_patient_audit"/filename).resolve()
        assert source.is_relative_to((ROOT/"u_patient_audit").resolve())
        assert digest(source) == expected_hash, filename
    assert set(protocol["code_sha256"]) == {
        "run_endpoint_precision.py", "probe_verified.py", "probe.py",
        "models.py", "common.py", "run_traced_replay.py"}
    cache_path = RUN/"cache/cache.pt"
    assert digest(cache_path) == protocol["cache_sha256"]
    cache_summary = load(RUN/"cache/summary.json")
    assert cache_summary["cache_sha256"] == protocol["cache_sha256"]
    assert cache_summary["cohort_lock_sha256"] == digest(RUN/"cohort/lock.json")
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    assert cache["audit_prompt"] == protocol["audit_prompt"]
    references = cache["reference_matches"]
    del cache
    original = load(LEGACY/"results.json")
    original_check = load(LEGACY/"verification.json")
    assert original_check["status"] == "PASS"
    assert original_check["results_sha256"] == digest(LEGACY/"results.json")
    assert original_check["report_sha256"] == digest(LEGACY/"report.json")
    first = next(r for r in original if r["model"] == "model_1")
    patient = str(first["patient_id"])
    assert patient == protocol["patient_id"] == report["patient_id"]
    replay = load(REPLAY/"results.json")
    replay_report = load(REPLAY/"report.json")
    assert replay_report["results_sha256"] == digest(REPLAY/"results.json")
    assert replay_report["status"] == "COMPLETED_TRACE_AND_REPRODUCTION_MEASUREMENTS"
    assert replay_report["precision"] == "fp16" and replay_report["all_legacy_scalars_exact"] is True
    assert replay_report["unique_fit_patients"] == 8
    assert replay_report["model_patient_evaluations"] == 16
    manifests = read_csv(RUN/"cohort/evaluation_images.csv") + read_csv(RUN/"cohort/auxiliary_images.csv")
    images = {r["image_id"]: r for r in manifests}
    patient_rows = [r for r in manifests if r["patient_id"] == patient]
    assert len(patient_rows) == 4 and all(r["eval_role"] == "fit" for r in patient_rows)
    u_ids = sorted(r["image_id"] for r in patient_rows if r["record_role"] == "U_observed")
    assert len(u_ids) == 2
    output_values = {}
    comparison = {}
    paired_cells_checked = 0
    source_files = []
    for model in MODELS:
        path = directory/(model+".json")
        assert report["model_output_sha256"][model] == digest(path)
        result = load(path)
        old = next(r for r in original if r["model"] == model and str(r["patient_id"]) == patient)
        replay_row = next(r for r in replay if r["model"] == model and str(r["patient_id"]) == patient)
        detail_path = (REPLAY/replay_row["detail_path"]).resolve()
        assert detail_path.is_relative_to(REPLAY.resolve())
        assert digest(detail_path) == replay_row["detail_sha256"] == result["detail_source_sha256"]
        detail = load(detail_path)
        source_files.append(detail_path)
        assert detail["image_order"] == u_ids
        assert detail["model"] == result["model"] == model
        assert str(detail["patient_id"]) == str(result["patient_id"]) == patient
        assert detail["eval_role"] == result["eval_role"] == "fit"
        assert detail["scenario"] == result["scenario"] == "U"
        close(detail["scalar_result"]["response"], replay_row["response"])
        close(old["response"], replay_row["response"])
        checkpoint = RUN/"training_coverage_v2"/model/"step_1000.pt"
        actual_hash = digest(checkpoint)
        assert actual_hash == protocol["checkpoint_sha256"][model] == result["checkpoint_sha256"]
        assert actual_hash == old["checkpoint_sha256"] == detail["checkpoint_sha256"]
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert state["step"] == 1000
        assert state["contract"] == load(checkpoint.parent.parent/"protocol.json")
        exposure = state["exposures"]
        train = {r["image_id"] for r in read_csv(RUN/"cohort"/(model+"_train.csv"))}
        assert set(exposure) == train and min(exposure.values()) >= 1
        assert sum(exposure.values()) == 4000
        member = int(any(exposure.get(r["image_id"], 0) > 0 for r in patient_rows))
        assert member == result["realized_member"] == result["declared_member"] == old["realized_member"]
        assert member == int(patient_rows[0]["assignment_group"] == ("A" if model == "model_1" else "B"))
        assert all(exposure.get(i, 0) == 0 for i in u_ids)
        assert len(result["folds"]) == len(detail["folds"]) == 2
        assert [(f["support"], f["query"]) for f in result["folds"]] == [
            tuple(u_ids), tuple(reversed(u_ids))]
        fold_comparison = []
        for fold, old_fold in zip(result["folds"], detail["folds"]):
            assert fold["support"] == old_fold["support"] and fold["query"] == old_fold["query"]
            coeff = torch.tensor(fold["coefficient"], dtype=torch.float32)
            source_coeff = torch.tensor(old_fold["support_trace"]["final_coefficient"], dtype=torch.float32)
            assert coeff.shape == (8,) and torch.isfinite(coeff).all()
            assert torch.equal(coeff, source_coeff)
            assert float(coeff.norm()) <= .050001
            close(fold["coefficient_norm"], float(coeff.norm()))
            assert fold["coefficient_float32_sha256"] == hashlib.sha256(coeff.numpy().tobytes()).hexdigest()
            assert fold["coefficient_unchanged"] is True
            query = fold["query"]
            q_shift = check_delta(fold["query_evaluations"]["shifted"], query)
            q_zero = check_delta(fold["query_evaluations"]["zero"], query)
            old_q_shift = check_delta(old_fold["query_evaluations"]["shifted"], query)
            old_q_zero = check_delta(old_fold["query_evaluations"]["zero"], query)
            refs = fold["reference_evaluations"]
            assert [r["image_id"] for r in refs] == references[query]
            assert [r["image_id"] for r in refs] == [r["image_id"] for r in old_fold["reference_evaluations"]]
            assert len(refs) == 2 and len({images[r["image_id"]]["patient_id"] for r in refs}) == 2
            ref_gains = []
            old_ref_gains = []
            for ref, old_ref in zip(refs, old_fold["reference_evaluations"]):
                image_id = ref["image_id"]
                assert images[image_id]["eval_role"] == "reference"
                assert ref["patient_id"] == images[image_id]["patient_id"] != patient
                assert image_id not in train and exposure.get(image_id, 0) == 0
                shifted = check_delta(ref["shifted"], image_id)
                zero = check_delta(ref["zero"], image_id)
                gain = shifted["value"]-zero["value"]
                close(ref["gain"], gain)
                close(ref["fp16_gain"], old_ref["gain"])
                close(ref["gain_precision_difference"], gain-old_ref["gain"])
                ref_gains.append(gain)
                old_shift = check_delta(old_ref["shifted"], image_id)
                old_zero = check_delta(old_ref["zero"], image_id)
                old_gain = old_shift["value"]-old_zero["value"]
                close(old_ref["gain"], old_gain)
                old_ref_gains.append(old_gain)
            own = q_shift["value"]-q_zero["value"]
            reference = math.fsum(ref_gains)/2
            current = dict(own_gain=own, reference_gain=reference, response=own-reference,
                           base_loss=q_zero["base_loss"], target_loss=q_zero["target_loss"],
                           base_minus_target=q_zero["value"])
            old_own = old_q_shift["value"]-old_q_zero["value"]
            old_reference = math.fsum(old_ref_gains)/2
            old_recomputed = dict(own_gain=old_own, reference_gain=old_reference,
                                  response=old_own-old_reference,
                                  base_loss=old_q_zero["base_loss"],
                                  target_loss=old_q_zero["target_loss"],
                                  base_minus_target=old_q_zero["value"])
            for name in COMPONENTS:
                close(old_fold[name], old_recomputed[name])
                close(fold["fp32_components"][name], current[name])
                close(fold["fp16_components"][name], old_fold[name])
                change = fold["precision_differences"][name]
                close(change["fp16"], old_fold[name])
                close(change["fp32"], current[name])
                close(change["difference"], current[name]-old_fold[name])
                close(change["absolute_difference"], abs(current[name]-old_fold[name]))
            fold_comparison.append({"fp32": current, "fp16": fold["fp16_components"]})
            paired_cells_checked += 60
        response = math.fsum(f["fp32"]["response"] for f in fold_comparison)/2
        response16 = old["response"]
        close(result["fp16_response"], response16)
        close(result["fp32_response_at_fp16_coefficients"], response)
        close(result["response_precision_difference"], response-response16)
        close(result["response_precision_absolute_difference"], abs(response-response16))
        assert result["response_zero_sign_changed"] == (sign(response) != sign(response16))
        assert result["response_zero_sign_is_not_a_membership_threshold"] is True
        assert result["forward"] == 240 and result["backward"] == result["coefficient_updates"] == 0
        for flag in ("weights_unchanged", "adapter_tensor_values_exactly_unchanged",
                     "all_parameters_frozen", "parameter_versions_and_gradients_unchanged"):
            assert result[flag] is True, flag
        assert result["adapter_tensors_checked"] == 256
        assert result["parameter_versions_checked"] == 942
        output_values[model] = {"member": member, "fp16": response16, "fp32": response}
        comparison[model] = {
            "fp16_response": response16, "fp32_response_at_fixed_fp16_coefficients": response,
            "precision_difference": response-response16,
            "zero_sign_changed": sign(response) != sign(response16), "fold_components": fold_comparison}
        del state
    assert paired_cells_checked == 240
    assert Counter(v["member"] for v in output_values.values()) == {0: 1, 1: 1}
    member_model = next(m for m, v in output_values.items() if v["member"])
    nonmember_model = next(m for m, v in output_values.items() if not v["member"])
    d16 = output_values[member_model]["fp16"]-output_values[nonmember_model]["fp16"]
    d32 = output_values[member_model]["fp32"]-output_values[nonmember_model]["fp32"]
    paired = report["paired_diagnostic_analyst_only"]
    assert paired["member_model"] == member_model and paired["nonmember_model"] == nonmember_model
    close(paired["FP16_member_minus_nonmember"], d16)
    close(paired["FP32_member_minus_nonmember_at_FP16_coefficients"], d32)
    close(paired["paired_delta_precision_difference"], d32-d16)
    close(paired["paired_delta_absolute_difference"], abs(d32-d16))
    assert paired["paired_delta_sign_changed"] == (sign(d32) != sign(d16))
    assert paired["not_an_auditor_input_or_single_patient_causal_estimate"] is True
    assert report["status"] == "FIXED_COEFFICIENT_ENDPOINT_MEASUREMENTS_COMPLETE_REVIEW_REQUIRED"
    assert report["unique_patients"] == 1 and report["model_patient_evaluations"] == 2
    assert report["total_forward"] == 480 and report["total_backward"] == 0
    for key in ("new_patients", "new_training_steps", "coefficient_updates"):
        assert report[key] == 0
    assert report["all_weights_and_coefficients_unchanged"] is True
    assert report["fitting_or_low_FPR_estimation"] is False
    assert report["numerical_equivalence_or_attack_success_declared"] is False
    analysis = {
        "status": "DESCRIPTIVE_FIXED_COEFFICIENT_PRECISION_COMPARISON",
        "unique_patients": 1, "independent_training_seeds": 1,
        "per_model": comparison, "paired_diagnostic_analyst_only": paired,
        "coefficient_reoptimization_performed": False, "membership_threshold_fitted": False,
        "precision_equivalence_declared": False, "attack_efficacy_declared": False,
        "interpretation": "Numerical differences are measured at fixed saved coefficients; they alone identify neither a code bug nor membership efficacy.",
        "limitations": [
            "One preselected existing patient and two correlated target models.",
            "Does not validate FP32 reoptimization or all-patient numerical robustness.",
            "Zero response is not a calibrated membership threshold.",
        ],
    }
    write_json(directory/"analysis.json", analysis)
    verification = {
        "status": "PASS_INTEGRITY_ONLY", "paired_noise_cells_recomputed": paired_cells_checked,
        "source_FP16_endpoint_noise_cells_also_recomputed": paired_cells_checked,
        "forward_evaluations_accounted": 2*paired_cells_checked, "backward": 0,
        "model_outputs_checked": 2, "folds_recomputed": 4,
        "source_coefficients_hashes_and_exposures_verified": True,
        "protocol_input_and_code_hashes_verified": True,
        "gpu_execution_by_verifier": False,
        "freeze_checks_are_report_cross_checks_not_independent_GPU_reinspection": True,
        "precision_difference_is_not_automatically_a_bug": True,
        "paired_delta_sign_changed": paired["paired_delta_sign_changed"],
        "input_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in [
            directory/"protocol.json", directory/"report.json",
            directory/"model_1.json", directory/"model_2.json", *source_files]},
        "analysis_sha256": digest(directory/"analysis.json"),
        "verifier_sha256": digest(Path(__file__)),
        "seconds_cpu": time.perf_counter()-started,
    }
    write_json(directory/"verification.json", verification)
    return verification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path,
                        default=RUN/"verification_20260914/endpoint_precision_v1")
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), indent=2))


if __name__ == "__main__":
    main()
