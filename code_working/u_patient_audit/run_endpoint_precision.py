"""Fixed-coefficient FP32 endpoint control for the first original U8 patient.

This new run preserves all executed files. It evaluates saved FP16 coefficients;
there is no new optimization, no new patient, no score/sign fitting and no claim
that a single patient establishes performance or universal precision robustness.
"""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import time

import torch

from .common import ROOT, RUN, digest, read_csv, verify_inputs, write_json
from .models import adapter_state, load_cache, load_unet, scheduler, setup
from .probe import QUERY
from .probe_verified import DiagnosticProbe

MODELS = ("model_1", "model_2")
LEGACY = RUN / "probe_smoke/training_coverage_v2/U_step_1000_n8"
REPLAY = RUN / "verification_20260914/traced_U8_v1"
COMPONENTS = ("own_gain", "reference_gain", "response", "base_loss", "target_loss", "base_minus_target")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def serial(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


def sign(value):
    return int(value > 0) - int(value < 0)


def changes(old, new):
    return {name: dict(fp16=old[name], fp32=new[name],
                       difference=new[name] - old[name],
                       absolute_difference=abs(new[name] - old[name])) for name in COMPONENTS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-tag", default="endpoint_precision_v1")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.output_tag):
        raise ValueError("output-tag must be a simple local directory name")
    out = RUN / "verification_20260914" / args.output_tag
    if out.exists():
        raise FileExistsError("immutable endpoint output exists; inspect it and use a new tag")
    verify_inputs()
    old_rows = load(LEGACY / "results.json")
    old_report = load(LEGACY / "report.json")
    old_verification = load(LEGACY / "verification.json")
    assert old_verification["status"] == "PASS"
    assert digest(LEGACY / "results.json") == old_verification["results_sha256"]
    assert digest(LEGACY / "report.json") == old_verification["report_sha256"]
    assert len(old_rows) == 16 and old_report["unique_fit_patients"] == 8
    first = next(r for r in old_rows if r["model"] == "model_1")
    patient = str(first["patient_id"])
    assert patient == "14393", "predeclared first-original-patient selection changed"
    selected_old = {m: next(r for r in old_rows if r["model"] == m and str(r["patient_id"]) == patient) for m in MODELS}
    replay_report = load(REPLAY / "report.json")
    replay_rows = load(REPLAY / "results.json")
    assert replay_report["status"] == "COMPLETED_TRACE_AND_REPRODUCTION_MEASUREMENTS"
    assert replay_report["results_sha256"] == digest(REPLAY / "results.json")
    assert replay_report["unique_fit_patients"] == 8 and replay_report["model_patient_evaluations"] == 16
    assert replay_report["precision"] == "fp16" and replay_report["all_legacy_scalars_exact"]
    selected_replay = {m: next(r for r in replay_rows if r["model"] == m and str(r["patient_id"]) == patient) for m in MODELS}
    details = {}
    detail_paths = {}
    for model in MODELS:
        row = selected_replay[model]
        detail_path = (REPLAY / row["detail_path"]).resolve()
        assert detail_path.is_relative_to(REPLAY.resolve())
        assert digest(detail_path) == row["detail_sha256"]
        detail_paths[model] = detail_path
        details[model] = load(detail_path)
        assert details[model]["model"] == model and str(details[model]["patient_id"]) == patient
        assert details[model]["eval_role"] == "fit" and details[model]["scenario"] == "U"
        assert details[model]["scalar_result"]["response"] == row["response"] == selected_old[model]["response"]
        assert len(details[model]["folds"]) == 2
    checkpoints = {m: RUN / "training_coverage_v2" / m / "step_1000.pt" for m in MODELS}
    checkpoint_hashes = {m: digest(p) for m, p in checkpoints.items()}
    for model in MODELS:
        assert checkpoint_hashes[model] == selected_old[model]["checkpoint_sha256"] == details[model]["checkpoint_sha256"]
    cache = load_cache()
    evaluation = read_csv(RUN / "cohort/evaluation_images.csv")
    auxiliary = read_csv(RUN / "cohort/auxiliary_images.csv")
    images = {r["image_id"]: r for r in evaluation + auxiliary}
    patient_rows = [r for r in evaluation if r["patient_id"] == patient]
    u_ids = sorted(r["image_id"] for r in patient_rows if r["record_role"] == "U_observed")
    assert len(u_ids) == 2 and all(images[i]["eval_role"] == "fit" for i in u_ids)
    for model in MODELS:
        assert details[model]["image_order"] == u_ids
        assert [(f["support"], f["query"]) for f in details[model]["folds"]] == [tuple(u_ids), tuple(reversed(u_ids))]
    files = [LEGACY / "results.json", LEGACY / "report.json", LEGACY / "verification.json",
             REPLAY / "protocol.json", REPLAY / "report.json", REPLAY / "results.json",
             *detail_paths.values(), RUN / "cohort/lock.json", RUN / "cohort/evaluation_images.csv",
             RUN / "cohort/auxiliary_images.csv", RUN / "cache/summary.json"]
    files += [RUN / "cohort" / (m + "_train.csv") for m in MODELS]
    code_names = ("run_endpoint_precision.py", "probe_verified.py", "probe.py", "models.py", "common.py", "run_traced_replay.py")
    protocol = {
        "schema": "u-endpoint-precision/v1", "scenario": "U", "patient_id": patient,
        "selection_rule": "first model_1 record in original U8 results order; selected before endpoint measurement, not by score difference",
        "models": list(MODELS), "evaluation_precision": "fp32", "coefficient_precision_source": "saved original FP16 traced run",
        "audit_prompt": cache["audit_prompt"],
        "coefficients_reoptimized": False, "same_coefficients_and_zero_used_for_both_precisions": True,
        "query_timesteps": list(QUERY), "draws": 2, "noise_phase": "query", "reference_count_per_fold": 2,
        "forward_per_model": 240, "total_forward": 480, "total_backward": 0,
        "new_patients": 0, "new_training_steps": 0, "score_fitting": False,
        "checkpoint_sha256": checkpoint_hashes, "cache_sha256": digest(RUN / "cache/cache.pt"),
        "input_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in files},
        "code_sha256": {name: digest(Path(__file__).with_name(name)) for name in code_names},
        "precision_control_definition": "Promote original FP16 model tensors and the legacy fixed FP16 noisy inputs/targets into FP32; compare endpoint computation at identical saved coefficients.",
        "decision_rule": "Record all endpoint and paired-delta changes; no evidence-selected passing threshold or automatic membership-success verdict.",
    }
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "protocol.json", protocol)
    setup()
    started = time.perf_counter()
    results = []
    for model in MODELS:
        model_started = time.perf_counter()
        state = torch.load(checkpoints[model], map_location="cpu", weights_only=True)
        exposure = state["exposures"]
        del state
        train_ids = {r["image_id"] for r in read_csv(RUN / "cohort" / (model + "_train.csv"))}
        declared = int(any(r["image_id"] in train_ids for r in patient_rows))
        realized = int(any(exposure.get(r["image_id"], 0) > 0 for r in patient_rows))
        assert declared == realized == selected_old[model]["realized_member"]
        assert all(exposure.get(i, 0) == 0 for i in u_ids)
        unet = load_unet(checkpoints[model], training=False)
        probe = DiagnosticProbe(unet, scheduler(), cache, precision="fp32")
        before_adapter = adapter_state(unet)
        torch.cuda.reset_peak_memory_stats()
        fold_results = []
        for source_fold in details[model]["folds"]:
            source_coefficient = torch.tensor(source_fold["support_trace"]["final_coefficient"], dtype=torch.float32)
            assert source_coefficient.shape == (8,) and torch.isfinite(source_coefficient).all()
            assert float(source_coefficient.norm()) <= .050001
            coefficient = source_coefficient.clone().to("cuda")
            zero_coefficient = torch.zeros_like(coefficient)
            query_id = source_fold["query"]
            refs = cache["reference_matches"][query_id]
            assert len(refs) == 2 and len({images[r]["patient_id"] for r in refs}) == 2
            assert refs == [r["image_id"] for r in source_fold["reference_evaluations"]]
            assert all(images[r]["patient_id"] != patient and images[r]["eval_role"] == "reference" for r in refs)
            assert all(r not in train_ids and exposure.get(r, 0) == 0 for r in refs)
            shifted = probe.delta_cells(query_id, coefficient, "query", QUERY, draws=2, gradient=False)
            zero = probe.delta_cells(query_id, zero_coefficient, "query", QUERY, draws=2, gradient=False)
            reference_values = []
            for ref, prior in zip(refs, source_fold["reference_evaluations"]):
                ref_shifted = probe.delta_cells(ref, coefficient, "query", QUERY, draws=2, gradient=False)
                ref_zero = probe.delta_cells(ref, zero_coefficient, "query", QUERY, draws=2, gradient=False)
                gain = ref_shifted["value"] - ref_zero["value"]
                reference_values.append({"image_id": ref, "patient_id": images[ref]["patient_id"],
                                         "shifted": serial(ref_shifted), "zero": serial(ref_zero),
                                         "gain": gain, "fp16_gain": prior["gain"],
                                         "gain_precision_difference": gain - prior["gain"]})
            own = shifted["value"] - zero["value"]
            reference = sum(r["gain"] for r in reference_values) / 2
            current = dict(own_gain=own, reference_gain=reference, response=own - reference,
                           base_loss=zero["base_loss"], target_loss=zero["target_loss"], base_minus_target=zero["value"])
            assert all(math.isfinite(v) for v in current.values())
            assert torch.equal(coefficient.detach().cpu(), source_coefficient)
            assert torch.count_nonzero(zero_coefficient) == 0
            fold_results.append({
                "support": source_fold["support"], "query": query_id,
                "coefficient": source_coefficient.tolist(),
                "coefficient_float32_sha256": hashlib.sha256(source_coefficient.numpy().tobytes()).hexdigest(),
                "coefficient_norm": float(source_coefficient.norm()), "coefficient_unchanged": True,
                "fp32_components": current, "fp16_components": {k: source_fold[k] for k in COMPONENTS},
                "precision_differences": changes(source_fold, current),
                "query_evaluations": {"shifted": serial(shifted), "zero": serial(zero)},
                "reference_evaluations": reference_values,
            })
        response = sum(f["fp32_components"]["response"] for f in fold_results) / 2
        fp16_response = selected_old[model]["response"]
        probe.assert_weights_unchanged()
        after_adapter = adapter_state(unet)
        assert before_adapter.keys() == after_adapter.keys()
        assert all(torch.equal(before_adapter[k], after_adapter[k]) for k in before_adapter)
        assert probe.forward == 240 and probe.backward == 0
        torch.cuda.synchronize()
        result = {
            "model": model, "patient_id": patient, "scenario": "U", "eval_role": "fit",
            "declared_member": declared, "realized_member": realized,
            "checkpoint_sha256": checkpoint_hashes[model], "detail_source_sha256": digest(detail_paths[model]),
            "fp16_response": fp16_response, "fp32_response_at_fp16_coefficients": response,
            "response_precision_difference": response - fp16_response,
            "response_precision_absolute_difference": abs(response - fp16_response),
            "response_zero_sign_changed": sign(response) != sign(fp16_response),
            "response_zero_sign_is_not_a_membership_threshold": True,
            "folds": fold_results, "forward": probe.forward, "backward": probe.backward,
            "coefficient_updates": 0, "weights_unchanged": True,
            "adapter_tensor_values_exactly_unchanged": True,
            "adapter_tensors_checked": len(before_adapter),
            "parameter_versions_checked": len(probe.initial_versions),
            "all_parameters_frozen": all(not p.requires_grad for p in unet.parameters()),
            "parameter_versions_and_gradients_unchanged": True,
            "max_cuda_GiB": torch.cuda.max_memory_allocated() / 2**30,
            "seconds_including_model_load": time.perf_counter() - model_started,
        }
        write_json(out / (model + ".json"), result)
        results.append(result)
        print(json.dumps({k: result[k] for k in ("model", "fp16_response", "fp32_response_at_fp16_coefficients", "response_precision_difference", "forward", "backward", "seconds_including_model_load")}), flush=True)
        del probe, unet, before_adapter, after_adapter
        gc.collect()
        torch.cuda.empty_cache()
    member = next(r for r in results if r["realized_member"])
    nonmember = next(r for r in results if not r["realized_member"])
    delta16 = member["fp16_response"] - nonmember["fp16_response"]
    delta32 = member["fp32_response_at_fp16_coefficients"] - nonmember["fp32_response_at_fp16_coefficients"]
    report = {
        "status": "FIXED_COEFFICIENT_ENDPOINT_MEASUREMENTS_COMPLETE_REVIEW_REQUIRED",
        "protocol_sha256": digest(out / "protocol.json"),
        "model_output_sha256": {m: digest(out / (m + ".json")) for m in MODELS},
        "patient_id": patient, "unique_patients": 1, "model_patient_evaluations": 2,
        "new_patients": 0, "new_training_steps": 0, "coefficient_updates": 0,
        "total_forward": sum(r["forward"] for r in results), "total_backward": sum(r["backward"] for r in results),
        "all_weights_and_coefficients_unchanged": True,
        "paired_diagnostic_analyst_only": {
            "member_model": member["model"], "nonmember_model": nonmember["model"],
            "FP16_member_minus_nonmember": delta16, "FP32_member_minus_nonmember_at_FP16_coefficients": delta32,
            "paired_delta_precision_difference": delta32 - delta16,
            "paired_delta_absolute_difference": abs(delta32 - delta16),
            "paired_delta_sign_changed": sign(delta32) != sign(delta16),
            "not_an_auditor_input_or_single_patient_causal_estimate": True,
        },
        "seconds_total": time.perf_counter() - started,
        "fitting_or_low_FPR_estimation": False, "numerical_equivalence_or_attack_success_declared": False,
        "scope": "one preselected original patient, both target models, frozen FP16 coefficients; endpoint precision only",
        "limitations": ["Does not check FP32 reoptimization trajectories or establish precision robustness for all eight patients.",
                        "Pair changes many training patients jointly; paired contrast is descriptive and analyst-only.",
                        "Finite precision sensitivity alone neither identifies a coding bug nor establishes membership inference efficacy."],
    }
    assert report["total_forward"] == 480 and report["total_backward"] == 0
    write_json(out / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
