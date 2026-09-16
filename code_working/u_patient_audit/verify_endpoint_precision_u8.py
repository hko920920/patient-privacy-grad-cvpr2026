"""Independent CPU verification of all-U8 fixed-coefficient FP32 endpoint corrections."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import torch

from .common import ROOT, RUN, digest, read_csv, verify_inputs, write_json
from .verify_endpoint_precision import check_delta, close, sign

MODELS = ("model_1", "model_2")
SCORES = ("response", "loss_mean", "loss_max", "base_difference_mean")
COMPONENTS = ("own_gain", "reference_gain", "response",
              "base_loss", "target_loss", "base_minus_target")
LEGACY = RUN / "probe_smoke/training_coverage_v2/U_step_1000_n8"
REPLAY = RUN / "verification_20260914/traced_U8_v1"
SINGLE = RUN / "verification_20260914/endpoint_precision_v1"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def score_from_folds(folds):
    return {
        "response": math.fsum(f["response"] for f in folds)/2,
        "loss_mean": -math.fsum(f["target_loss"] for f in folds)/2,
        "loss_max": max(-f["target_loss"] for f in folds),
        "base_difference_mean": math.fsum(f["base_minus_target"] for f in folds)/2,
    }


def recompute_fold(fold, source, references, images, patient, training, exposures):
    assert fold["support"] == source["support"] and fold["query"] == source["query"]
    coefficient = torch.tensor(fold["coefficient"], dtype=torch.float32)
    old_coefficient = torch.tensor(source["support_trace"]["final_coefficient"], dtype=torch.float32)
    assert coefficient.shape == (8,) and torch.isfinite(coefficient).all()
    assert torch.equal(coefficient, old_coefficient)
    assert float(coefficient.norm()) <= .050001
    close(fold["coefficient_norm"], float(coefficient.norm()))
    assert fold["coefficient_unchanged"] is True
    assert fold["coefficient_float32_sha256"] == hashlib.sha256(coefficient.numpy().tobytes()).hexdigest()
    query = fold["query"]
    values = {}
    for precision, container in (("fp32", fold), ("fp16", source)):
        shifted = check_delta(container["query_evaluations"]["shifted"], query)
        zero = check_delta(container["query_evaluations"]["zero"], query)
        ref_rows = container["reference_evaluations"]
        assert len(ref_rows) == 2
        assert [r["image_id"] for r in ref_rows] == references[query]
        assert len({images[r["image_id"]]["patient_id"] for r in ref_rows}) == 2
        gains = []
        for ref in ref_rows:
            image = images[ref["image_id"]]
            assert image["eval_role"] == "reference" and image["patient_id"] != patient
            assert ref["patient_id"] == image["patient_id"]
            assert ref["image_id"] not in training and exposures.get(ref["image_id"], 0) == 0
            shift = check_delta(ref["shifted"], ref["image_id"])
            base = check_delta(ref["zero"], ref["image_id"])
            gain = shift["value"]-base["value"]
            close(ref["gain"], gain)
            gains.append(gain)
        own = shifted["value"]-zero["value"]
        reference = math.fsum(gains)/2
        values[precision] = {
            "own_gain": own, "reference_gain": reference, "response": own-reference,
            "base_loss": zero["base_loss"], "target_loss": zero["target_loss"],
            "base_minus_target": zero["value"],
        }
    for name in COMPONENTS:
        close(source[name], values["fp16"][name])
        close(fold["fp16_components"][name], values["fp16"][name])
        close(fold["fp32_components"][name], values["fp32"][name])
        change = fold["precision_differences"][name]
        close(change["fp16"], values["fp16"][name])
        close(change["fp32"], values["fp32"][name])
        close(change["difference"], values["fp32"][name]-values["fp16"][name])
        close(change["absolute_difference"], abs(values["fp32"][name]-values["fp16"][name]))
    for now, old in zip(fold["reference_evaluations"], source["reference_evaluations"]):
        close(now["fp16_gain"], old["gain"])
        close(now["gain_precision_difference"], now["gain"]-old["gain"])
    return values


def auc(records, score):
    positive = [r[score] for r in records if r["member"]]
    negative = [r[score] for r in records if not r["member"]]
    assert len(positive) == len(negative) == 4
    return math.fsum(1.0 if p > n else .5 if p == n else 0.0
                     for p in positive for n in negative)/16


def verify(directory):
    started = time.perf_counter()
    torch.set_num_threads(2)
    verify_inputs()
    protocol = load(directory/"protocol.json")
    report = load(directory/"report.json")
    results = load(directory/"results.json")
    assert protocol["schema"] == "u-endpoint-precision-u8/v1"
    expected_protocol = {
        "models": list(MODELS), "precision": "fp32",
        "coefficient_source": "unchanged saved FP16 traced_U8_v1 coefficients",
        "batch_size": 1, "query_timesteps": [50, 250, 500, 750, 950],
        "draws": 2, "noise_phase": "query", "reference_count_per_fold": 2,
        "forward_per_patient": 240, "total_forward": 3840, "total_backward": 0,
        "new_patients": 0, "new_training_steps": 0, "coefficient_updates": 0,
        "precision_policy": "same FP32 endpoint arithmetic for every original patient; no reoptimization or fitting",
        "no_performance_or_numerical_equivalence_pass_threshold": True,
    }
    for key, expected in expected_protocol.items():
        assert protocol[key] == expected, key
    assert protocol["policy_sha256"] == digest(RUN/"verification_20260914/endpoint_precision_policy.json")
    assert report["protocol_sha256"] == digest(directory/"protocol.json")
    assert report["results_sha256"] == digest(directory/"results.json")
    for filename, expected in protocol["input_sha256"].items():
        source = (ROOT/filename).resolve()
        assert source.is_relative_to(ROOT.resolve()) and digest(source) == expected, filename
    for filename, expected in protocol["code_sha256"].items():
        source = (ROOT/"u_patient_audit"/filename).resolve()
        assert source.is_relative_to((ROOT/"u_patient_audit").resolve())
        assert digest(source) == expected, filename
    assert set(protocol["code_sha256"]) == {
        "run_endpoint_precision_u8.py", "run_endpoint_precision.py", "probe_verified.py",
        "probe.py", "models.py", "common.py", "run_traced_replay.py"}
    assert digest(RUN/"cache/cache.pt") == protocol["cache_sha256"]
    cache_meta = load(RUN/"cache/summary.json")
    assert cache_meta["cache_sha256"] == protocol["cache_sha256"]
    assert cache_meta["cohort_lock_sha256"] == digest(RUN/"cohort/lock.json")
    cache = torch.load(RUN/"cache/cache.pt", map_location="cpu", weights_only=True)
    references = cache["reference_matches"]
    assert cache["audit_prompt"] == protocol["audit_prompt"] == "a frontal chest radiograph"
    del cache
    original = load(LEGACY/"results.json")
    old_verification = load(LEGACY/"verification.json")
    assert old_verification["status"] == "PASS"
    assert old_verification["results_sha256"] == digest(LEGACY/"results.json")
    assert old_verification["report_sha256"] == digest(LEGACY/"report.json")
    original_pairs = [(r["model"], str(r["patient_id"])) for r in original]
    assert len(original_pairs) == len(set(original_pairs)) == 16
    assert [(r["model"], str(r["patient_id"])) for r in results] == original_pairs
    selected = sorted({p for _, p in original_pairs})
    assert len(selected) == 8
    original_patient_order = [p for model, p in original_pairs if model == "model_1"]
    assert protocol["patient_order"] == report["patient_order"] == original_patient_order
    assert original_pairs == [(model, patient) for model in MODELS for patient in original_patient_order]
    old_map = {(r["model"], str(r["patient_id"])): r for r in original}
    replay_report = load(REPLAY/"report.json")
    replay_rows = load(REPLAY/"results.json")
    assert replay_report["results_sha256"] == digest(REPLAY/"results.json")
    assert replay_report["all_legacy_scalars_exact"] is True and replay_report["precision"] == "fp16"
    replay_map = {(r["model"], str(r["patient_id"])): r for r in replay_rows}
    assert set(replay_map) == set(old_map)
    images = {r["image_id"]: r for r in
              read_csv(RUN/"cohort/evaluation_images.csv") + read_csv(RUN/"cohort/auxiliary_images.csv")}
    by_patient = defaultdict(list)
    for row in images.values():
        by_patient[row["patient_id"]].append(row)
    exposures = {}
    training = {}
    checkpoint_hashes = {}
    for model in MODELS:
        checkpoint = RUN/"training_coverage_v2"/model/"step_1000.pt"
        checkpoint_hashes[model] = digest(checkpoint)
        assert checkpoint_hashes[model] == protocol["checkpoint_sha256"][model]
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert state["step"] == 1000
        assert state["contract"] == load(checkpoint.parent.parent/"protocol.json")
        exposures[model] = dict(state["exposures"])
        training[model] = {r["image_id"] for r in read_csv(RUN/"cohort"/(model+"_train.csv"))}
        assert set(exposures[model]) == training[model]
        assert min(exposures[model].values()) >= 1 and sum(exposures[model].values()) == 4000
        del state
    computed = {}
    details = {}
    detail_paths = []
    source_paths = []
    for row in results:
        model, patient = row["model"], str(row["patient_id"])
        detail_path = (directory/row["detail_path"]).resolve()
        assert detail_path.is_relative_to(directory.resolve())
        assert digest(detail_path) == row["detail_sha256"]
        detail = load(detail_path)
        detail_scalar = detail["scalar_result"]
        assert detail_scalar == {k: v for k, v in row.items() if k != "detail_sha256"}
        details[model, patient] = detail
        detail_paths.append(detail_path)
        source_row = replay_map[model, patient]
        source_path = (REPLAY/source_row["detail_path"]).resolve()
        assert source_path.is_relative_to(REPLAY.resolve())
        assert digest(source_path) == source_row["detail_sha256"] == row["detail_source_sha256"]
        source = load(source_path)
        source_paths.append(source_path)
        for item in (row, detail, source):
            assert item["model"] == model and str(item["patient_id"]) == patient
            assert item["scenario"] == "U" and item["eval_role"] == "fit"
            assert item["checkpoint_sha256"] == checkpoint_hashes[model]
        patient_rows = by_patient[patient]
        assert len(patient_rows) == 4 and all(r["eval_role"] == "fit" for r in patient_rows)
        member = int(any(exposures[model].get(r["image_id"], 0) > 0 for r in patient_rows))
        group = "A" if model == "model_1" else "B"
        assert member == int(patient_rows[0]["assignment_group"] == group)
        assert member == old_map[model, patient]["realized_member"]
        for item in (row, detail):
            assert item["declared_member"] == item["realized_member"] == member
        ids = sorted(r["image_id"] for r in patient_rows if r["record_role"] == "U_observed")
        assert len(ids) == 2 and all(exposures[model].get(i, 0) == 0 for i in ids)
        assert source["image_order"] == ids
        assert [(f["support"], f["query"]) for f in detail["folds"]] == [tuple(ids), tuple(reversed(ids))]
        assert len(detail["folds"]) == len(source["folds"]) == 2
        folds = [recompute_fold(f, s, references, images, patient, training[model], exposures[model])
                 for f, s in zip(detail["folds"], source["folds"])]
        values = {precision: score_from_folds([f[precision] for f in folds])
                  for precision in ("fp16", "fp32")}
        for score in SCORES:
            close(values["fp16"][score], old_map[model, patient][score])
            for item in (row, detail_scalar):
                close(item["fp16_scores"][score], values["fp16"][score])
                close(item["fp32_scores"][score], values["fp32"][score])
                difference = item["score_precision_differences"][score]
                close(difference["fp16"], values["fp16"][score])
                close(difference["fp32"], values["fp32"][score])
                close(difference["difference"], values["fp32"][score]-values["fp16"][score])
                close(difference["absolute_difference"], abs(values["fp32"][score]-values["fp16"][score]))
        assert row["folds"] == [{k: f[k] for k in (
            "support", "query", "coefficient", "coefficient_float32_sha256", "coefficient_unchanged",
            "fp16_components", "fp32_components")} for f in detail["folds"]]
        computed[model, patient] = dict(member=member, **values)
        assert row["forward"] == detail_scalar["forward"] == 240
        assert row["backward"] == detail_scalar["backward"] == 0
        assert row["coefficient_updates"] == 0 and row["weights_and_coefficients_unchanged"] is True
        assert math.isfinite(row["seconds"]) and row["seconds"] > 0
    # The first original patient was already evaluated in a separate immutable run.
    first_patient = original_pairs[0][1]
    single_report = load(SINGLE/"report.json")
    assert single_report["patient_id"] == first_patient
    single_checks = {}
    for model in MODELS:
        previous = load(SINGLE/(model+".json"))
        assert single_report["model_output_sha256"][model] == digest(SINGLE/(model+".json"))
        now = details[model, first_patient]
        equal_cells = 0
        for before, after in zip(previous["folds"], now["folds"]):
            assert before["coefficient"] == after["coefficient"]
            assert before["coefficient_float32_sha256"] == after["coefficient_float32_sha256"]
            for arm in ("shifted", "zero"):
                assert before["query_evaluations"][arm]["cells"] == after["query_evaluations"][arm]["cells"]
                equal_cells += 10
            for old_ref, new_ref in zip(before["reference_evaluations"], after["reference_evaluations"]):
                assert old_ref["image_id"] == new_ref["image_id"]
                for arm in ("shifted", "zero"):
                    assert old_ref[arm]["cells"] == new_ref[arm]["cells"]
                    equal_cells += 10
        assert equal_cells == 120
        close(previous["fp32_response_at_fp16_coefficients"], computed[model, first_patient]["fp32"]["response"])
        single_checks[model] = dict(exact_paired_noise_cells=equal_cells, coefficients_exact=True)
    metrics = {}
    for score in SCORES:
        per_model = {}
        for model in MODELS:
            old_records = [dict(member=computed[model, p]["member"], **computed[model, p]["fp16"]) for p in selected]
            new_records = [dict(member=computed[model, p]["member"], **computed[model, p]["fp32"]) for p in selected]
            ranks = []
            member_ranks = []
            for p, q in itertools.combinations(selected, 2):
                old_diff = computed[model, p]["fp16"][score]-computed[model, q]["fp16"][score]
                new_diff = computed[model, p]["fp32"][score]-computed[model, q]["fp32"][score]
                if sign(old_diff) != sign(new_diff):
                    changed = dict(patient_1=p, patient_2=q, old_sign=sign(old_diff), new_sign=sign(new_diff))
                    ranks.append(changed)
                    if computed[model, p]["member"] != computed[model, q]["member"]:
                        member_ranks.append(changed)
            per_model[model] = {
                "members": 4, "nonmembers": 4,
                "fp16_auc": auc(old_records, score), "fp32_auc": auc(new_records, score),
                "all_patient_pair_count": 28, "changed_patient_pair_count": len(ranks),
                "member_nonmember_pair_count": 16, "changed_member_nonmember_pair_count": len(member_ranks),
                "rank_changes": ranks, "membership_pair_rank_changes": member_ranks,
            }
        paired = []
        for patient in selected:
            member_model = next(m for m in MODELS if computed[m, patient]["member"])
            nonmember_model = next(m for m in MODELS if not computed[m, patient]["member"])
            old_delta = computed[member_model, patient]["fp16"][score]-computed[nonmember_model, patient]["fp16"][score]
            new_delta = computed[member_model, patient]["fp32"][score]-computed[nonmember_model, patient]["fp32"][score]
            paired.append(dict(patient_id=patient, member_model=member_model, nonmember_model=nonmember_model,
                               fp16_delta=old_delta, fp32_delta=new_delta,
                               precision_difference=new_delta-old_delta,
                               sign_changed=sign(old_delta) != sign(new_delta)))
        metrics[score] = {
            "per_model": per_model, "paired_analyst_only": paired,
            "fp16_paired_positive": sum(p["fp16_delta"] > 0 for p in paired),
            "fp32_paired_positive": sum(p["fp32_delta"] > 0 for p in paired),
            "paired_sign_changes": sum(p["sign_changed"] for p in paired),
            "fp16_paired_mean": math.fsum(p["fp16_delta"] for p in paired)/8,
            "fp32_paired_mean": math.fsum(p["fp32_delta"] for p in paired)/8,
            "fp16_macro_auc": math.fsum(v["fp16_auc"] for v in per_model.values())/2,
            "fp32_macro_auc": math.fsum(v["fp32_auc"] for v in per_model.values())/2,
        }
    assert report["total_forward"] == sum(r["forward"] for r in results) == 3840
    assert report["total_backward"] == sum(r["backward"] for r in results) == 0
    assert report["status"] == "FIXED_COEFFICIENT_U8_ENDPOINT_COMPLETE_REVIEW_REQUIRED"
    assert report["unique_patients"] == 8 and report["model_patient_evaluations"] == 16
    assert report["all_weights_and_coefficients_unchanged"] is True
    assert report["score_sign_fitting"] is False and report["low_FPR_or_performance_success_claim"] is False
    for key in ("new_patients", "new_training_steps", "coefficient_updates"):
        assert report[key] == 0
    assert len(report["model_reports"]) == 2
    assert [m["model"] for m in report["model_reports"]] == list(MODELS)
    for model_report in report["model_reports"]:
        assert model_report["checkpoint_sha256"] == checkpoint_hashes[model_report["model"]]
        assert model_report["forward"] == 1920 and model_report["backward"] == 0
        assert model_report["adapter_tensors_checked"] == 256
        assert model_report["parameter_versions_checked"] == 942
        for key in ("adapter_tensor_values_exactly_unchanged", "parameter_versions_and_gradients_unchanged", "all_parameters_frozen"):
            assert model_report[key] is True
        for key in ("peak_cuda_GiB", "seconds"):
            assert math.isfinite(model_report[key]) and model_report[key] > 0
    assert sum(m["forward"] for m in report["model_reports"]) == report["total_forward"]
    analysis = {
        "status": "EXPLORATORY_FIXED_COEFFICIENT_FP32_ENDPOINT_CORRECTION",
        "unique_patients": 8, "model_patient_evaluations": 16,
        "independent_training_seed_count": 1,
        "metrics": metrics, "single_case_reproduction": single_checks,
        "same_endpoint_precision_given_to_candidate_and_raw_loss_baselines": True,
        "coefficient_reoptimization_performed": False, "score_sign_fitted": False,
        "threshold_fitted": False, "low_fpr_performance_estimated": False,
        "calibration_or_test_used": False, "paired_comparisons_are_analyst_only": True,
        "limitations": [
            "FP32 endpoint evaluation at frozen FP16-optimized coefficients, not FP32 reoptimization.",
            "Eight fit patients and a correlated inclusion-swap target pair.",
            "Numerical change alone establishes neither implementation bug nor membership efficacy.",
            "Original scores remain intact; corrected estimates are reported for all four scores equally.",
        ],
    }
    write_json(directory/"analysis.json", analysis)
    inputs = [directory/"protocol.json", directory/"report.json", directory/"results.json",
              LEGACY/"results.json", REPLAY/"results.json", SINGLE/"report.json",
              SINGLE/"model_1.json", SINGLE/"model_2.json", *detail_paths, *source_paths]
    verification = {
        "status": "PASS_INTEGRITY_ONLY", "unique_patients": 8,
        "paired_FP32_noise_cells_recomputed": 1920, "source_FP16_endpoint_cells_recomputed": 1920,
        "forward_accounted": 3840, "backward": 0, "folds_recomputed": 32,
        "patient_scores_recomputed": 64, "AUCs_recomputed": 16,
        "original_result_order_preserved": True,
        "stored_coefficients_labels_actual_exposures_and_reference_isolation_verified": True,
        "producer_protocol_and_freeze_counter_declarations_cross_checked": True,
        "freeze_checks_are_not_independent_GPU_reinspection": True,
        "single_case_FP32_raw_cells_exact": True,
        "gpu_execution_by_verifier": False,
        "input_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in inputs},
        "verifier_sha256": digest(Path(__file__)),
        "shared_CPU_arithmetic_checker_sha256": digest(Path(__file__).with_name("verify_endpoint_precision.py")),
        "analysis_sha256": digest(directory/"analysis.json"),
        "seconds_cpu": time.perf_counter()-started,
    }
    write_json(directory/"verification.json", verification)
    return verification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path,
                        default=RUN/"verification_20260914/endpoint_precision_U8_v1")
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), indent=2))


if __name__ == "__main__":
    main()
