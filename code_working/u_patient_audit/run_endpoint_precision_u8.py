"""Versioned FP32 endpoint measurement for all original U8 x two targets.

Uses saved FP16 coefficients unchanged. No optimization, fitting, new patient,
batching, or result-selected patient exclusion is performed.
"""
import argparse
import gc
import hashlib
import math
from pathlib import Path
import re
import time
import json

import torch

from .common import ROOT, RUN, digest, read_csv, verify_inputs, write_json
from .models import adapter_state, load_cache, load_unet, scheduler, setup
from .probe import QUERY
from .probe_verified import DiagnosticProbe
from .run_endpoint_precision import load, serial, changes, COMPONENTS

MODELS = ("model_1", "model_2")
SCORES = ("response", "loss_mean", "loss_max", "base_difference_mean")
LEGACY = RUN / "probe_smoke/training_coverage_v2/U_step_1000_n8"
REPLAY = RUN / "verification_20260914/traced_U8_v1"
SINGLE = RUN / "verification_20260914/endpoint_precision_v1"
POLICY = RUN / "verification_20260914/endpoint_precision_policy.json"


def evaluate(probe, detail, cache, images, train_ids, exposure):
    patient = str(detail["patient_id"])
    before_f, before_b = probe.forward, probe.backward
    folds = []
    for prior in detail["folds"]:
        source_a = torch.tensor(prior["support_trace"]["final_coefficient"], dtype=torch.float32)
        assert source_a.shape == (8,) and torch.isfinite(source_a).all() and float(source_a.norm()) <= .050001
        coefficient = source_a.clone().to("cuda")
        zero_coefficient = torch.zeros_like(coefficient)
        query_id = prior["query"]
        refs = cache["reference_matches"][query_id]
        assert len(refs) == 2 and len({images[r]["patient_id"] for r in refs}) == 2
        assert refs == [r["image_id"] for r in prior["reference_evaluations"]]
        assert all(images[r]["patient_id"] != patient and images[r]["eval_role"] == "reference" for r in refs)
        assert all(r not in train_ids and exposure.get(r, 0) == 0 for r in refs)
        shifted = probe.delta_cells(query_id, coefficient, "query", QUERY, draws=2, gradient=False)
        zero = probe.delta_cells(query_id, zero_coefficient, "query", QUERY, draws=2, gradient=False)
        references = []
        for ref, old_ref in zip(refs, prior["reference_evaluations"]):
            ref_shifted = probe.delta_cells(ref, coefficient, "query", QUERY, draws=2, gradient=False)
            ref_zero = probe.delta_cells(ref, zero_coefficient, "query", QUERY, draws=2, gradient=False)
            gain = ref_shifted["value"] - ref_zero["value"]
            references.append(dict(image_id=ref, patient_id=images[ref]["patient_id"],
                                   shifted=serial(ref_shifted), zero=serial(ref_zero), gain=gain,
                                   fp16_gain=old_ref["gain"], gain_precision_difference=gain-old_ref["gain"]))
        own = shifted["value"] - zero["value"]
        reference = sum(r["gain"] for r in references) / 2
        components = dict(own_gain=own, reference_gain=reference, response=own-reference,
                          base_loss=zero["base_loss"], target_loss=zero["target_loss"], base_minus_target=zero["value"])
        assert all(math.isfinite(v) for v in components.values())
        assert torch.equal(coefficient.detach().cpu(), source_a) and torch.count_nonzero(zero_coefficient) == 0
        folds.append(dict(support=prior["support"], query=query_id, coefficient=source_a.tolist(),
                          coefficient_float32_sha256=hashlib.sha256(source_a.numpy().tobytes()).hexdigest(),
                          coefficient_norm=float(source_a.norm()), coefficient_unchanged=True,
                          fp32_components=components, fp16_components={k: prior[k] for k in COMPONENTS},
                          precision_differences=changes(prior, components),
                          query_evaluations=dict(shifted=serial(shifted), zero=serial(zero)),
                          reference_evaluations=references))
    assert len(folds) == 2
    scores = dict(response=sum(f["fp32_components"]["response"] for f in folds)/2,
                  loss_mean=-sum(f["fp32_components"]["target_loss"] for f in folds)/2,
                  loss_max=max(-f["fp32_components"]["target_loss"] for f in folds),
                  base_difference_mean=sum(f["fp32_components"]["base_minus_target"] for f in folds)/2)
    probe.assert_weights_unchanged()
    assert probe.forward-before_f == 240 and probe.backward-before_b == 0
    return scores, folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-tag", default="endpoint_precision_U8_v1")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.output_tag):
        raise ValueError("simple local output-tag required")
    out = RUN / "verification_20260914" / args.output_tag
    if out.exists():
        raise FileExistsError("immutable endpoint output exists; use a new reviewed tag")
    verify_inputs()
    old_rows, old_report, old_check = [load(LEGACY/name) for name in ("results.json", "report.json", "verification.json")]
    assert old_check["status"] == "PASS" and digest(LEGACY/"results.json") == old_check["results_sha256"]
    assert digest(LEGACY/"report.json") == old_check["report_sha256"]
    assert len(old_rows) == 16 and old_report["unique_fit_patients"] == 8
    order = [str(r["patient_id"]) for r in old_rows if r["model"] == "model_1"]
    assert len(order) == len(set(order)) == 8 and order[0] == "14393"
    assert [(r["model"], str(r["patient_id"])) for r in old_rows] == [(m,p) for m in MODELS for p in order]
    replay_report, replay_rows = load(REPLAY/"report.json"), load(REPLAY/"results.json")
    assert replay_report["status"] == "COMPLETED_TRACE_AND_REPRODUCTION_MEASUREMENTS"
    assert digest(REPLAY/"results.json") == replay_report["results_sha256"]
    assert replay_report["precision"] == "fp16" and replay_report["all_legacy_scalars_exact"]
    assert [(r["model"], str(r["patient_id"])) for r in replay_rows] == [(r["model"], str(r["patient_id"])) for r in old_rows]
    old_by_pair = {(r["model"], str(r["patient_id"])): r for r in old_rows}
    details, detail_paths = {}, {}
    for r in replay_rows:
        key = (r["model"], str(r["patient_id"]))
        path = (REPLAY/r["detail_path"]).resolve()
        assert path.is_relative_to(REPLAY.resolve()) and digest(path) == r["detail_sha256"]
        detail = load(path)
        assert detail["model"] == key[0] and str(detail["patient_id"]) == key[1]
        assert detail["eval_role"] == "fit" and detail["scenario"] == "U"
        assert all(detail["scalar_result"][k] == old_by_pair[key][k] for k in SCORES)
        details[key], detail_paths[key] = detail, path
    evaluation = read_csv(RUN/"cohort/evaluation_images.csv")
    auxiliary = read_csv(RUN/"cohort/auxiliary_images.csv")
    images = {r["image_id"]: r for r in evaluation+auxiliary}
    patient_rows = {p: [r for r in evaluation if r["patient_id"] == p] for p in order}
    checkpoints = {m: RUN/"training_coverage_v2"/m/"step_1000.pt" for m in MODELS}
    checkpoint_hashes = {m: digest(p) for m,p in checkpoints.items()}
    for key, detail in details.items():
        assert detail["checkpoint_sha256"] == old_by_pair[key]["checkpoint_sha256"] == checkpoint_hashes[key[0]]
        ids = sorted(r["image_id"] for r in patient_rows[key[1]] if r["record_role"] == "U_observed")
        assert len(ids) == 2 and all(images[i]["eval_role"] == "fit" for i in ids)
        assert detail["image_order"] == ids
        assert [(f["support"],f["query"]) for f in detail["folds"]] == [tuple(ids),tuple(reversed(ids))]
    cache = load_cache()
    single_report = load(SINGLE/"report.json")
    assert single_report["patient_id"] == order[0]
    for m in MODELS:
        assert digest(SINGLE/(m+".json")) == single_report["model_output_sha256"][m]
    load(POLICY)
    paths = [LEGACY/name for name in ("results.json","report.json","verification.json")]
    paths += [REPLAY/name for name in ("protocol.json","report.json","results.json")]
    paths += list(detail_paths.values())+[POLICY, SINGLE/"report.json", SINGLE/"model_1.json", SINGLE/"model_2.json"]
    paths += [RUN/"cohort"/name for name in ("lock.json","evaluation_images.csv","auxiliary_images.csv","model_1_train.csv","model_2_train.csv")]
    paths += [RUN/"cache/summary.json"]
    code_names = ("run_endpoint_precision_u8.py","run_endpoint_precision.py","probe_verified.py","probe.py","models.py","common.py","run_traced_replay.py")
    protocol = dict(schema="u-endpoint-precision-u8/v1", models=list(MODELS), patient_order=order,
        selection_rule="all original eight patients and two models in original order; no score-dependent exclusion",
        precision="fp32",coefficient_source="unchanged saved FP16 traced_U8_v1 coefficients",batch_size=1,
        query_timesteps=list(QUERY),draws=2,noise_phase="query",reference_count_per_fold=2,audit_prompt=cache["audit_prompt"],
        forward_per_patient=240,total_forward=3840,total_backward=0,new_patients=0,new_training_steps=0,coefficient_updates=0,
        checkpoint_sha256=checkpoint_hashes,cache_sha256=digest(RUN/"cache/cache.pt"),policy_sha256=digest(POLICY),
        input_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in paths},
        code_sha256={n:digest(Path(__file__).with_name(n)) for n in code_names},
        precision_policy="same FP32 endpoint arithmetic for every original patient; no reoptimization or fitting",
        no_performance_or_numerical_equivalence_pass_threshold=True)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/"protocol.json",protocol)
    setup()
    started=time.perf_counter()
    results, model_reports = [], []
    for model in MODELS:
        model_started=time.perf_counter()
        state=torch.load(checkpoints[model],map_location="cpu",weights_only=True)
        exposure=state["exposures"]
        del state
        train_ids={r["image_id"] for r in read_csv(RUN/"cohort"/(model+"_train.csv"))}
        unet=load_unet(checkpoints[model],training=False)
        probe=DiagnosticProbe(unet,scheduler(),cache,precision="fp32")
        before_adapter=adapter_state(unet)
        torch.cuda.reset_peak_memory_stats()
        for patient in order:
            patient_started=time.perf_counter()
            key=(model,patient);detail=details[key];old=old_by_pair[key]
            declared=int(any(r["image_id"] in train_ids for r in patient_rows[patient]))
            realized=int(any(exposure.get(r["image_id"],0)>0 for r in patient_rows[patient]))
            assert declared == realized == old["realized_member"] == old["declared_member"]
            assert all(exposure.get(i,0)==0 for i in detail["image_order"])
            scores,folds=evaluate(probe,detail,cache,images,train_ids,exposure)
            fp16={k:old[k] for k in SCORES}
            result=dict(model=model,patient_id=patient,scenario="U",eval_role="fit",declared_member=declared,
                realized_member=realized,checkpoint_sha256=checkpoint_hashes[model],detail_source_sha256=digest(detail_paths[key]),
                fp16_scores=fp16,fp32_scores=scores,
                score_precision_differences={k:dict(fp16=fp16[k],fp32=scores[k],difference=scores[k]-fp16[k],absolute_difference=abs(scores[k]-fp16[k])) for k in SCORES},
                folds=[{k:f[k] for k in ("support","query","coefficient","coefficient_float32_sha256","coefficient_unchanged","fp16_components","fp32_components")} for f in folds],
                forward=240,backward=0,coefficient_updates=0,weights_and_coefficients_unchanged=True,
                seconds=time.perf_counter()-patient_started)
            relative=Path("patients")/model/(patient+".json")
            result["detail_path"]=relative.as_posix()
            write_json(out/relative,dict(model=model,patient_id=patient,scenario="U",eval_role="fit",declared_member=declared,
                realized_member=realized,checkpoint_sha256=checkpoint_hashes[model],scalar_result=result,folds=folds))
            result["detail_sha256"]=digest(out/relative)
            results.append(result)
            write_json(out/"partial_results.json",results)
            progress=dict(completed=len(results),total=16,model=model,patient_id=patient,
                          last_patient_seconds=result["seconds"],elapsed_seconds=time.perf_counter()-started)
            write_json(out/"progress.json",progress)
            print(json.dumps(progress),flush=True)
        probe.assert_weights_unchanged()
        after_adapter=adapter_state(unet)
        assert before_adapter.keys()==after_adapter.keys() and all(torch.equal(before_adapter[k],after_adapter[k]) for k in before_adapter)
        assert probe.forward==1920 and probe.backward==0
        model_reports.append(dict(model=model,checkpoint_sha256=checkpoint_hashes[model],forward=probe.forward,backward=probe.backward,
            adapter_tensors_checked=len(before_adapter),adapter_tensor_values_exactly_unchanged=True,
            parameter_versions_checked=len(probe.initial_versions),parameter_versions_and_gradients_unchanged=True,
            all_parameters_frozen=all(not p.requires_grad for p in unet.parameters()),
            peak_cuda_GiB=torch.cuda.max_memory_allocated()/2**30,seconds=time.perf_counter()-model_started))
        del probe,unet,before_adapter,after_adapter
        gc.collect();torch.cuda.empty_cache()
    write_json(out/"results.json",results)
    report=dict(status="FIXED_COEFFICIENT_U8_ENDPOINT_COMPLETE_REVIEW_REQUIRED",protocol_sha256=digest(out/"protocol.json"),
        results_sha256=digest(out/"results.json"),unique_patients=8,model_patient_evaluations=16,patient_order=order,
        new_patients=0,new_training_steps=0,coefficient_updates=0,total_forward=sum(r["forward"] for r in results),
        total_backward=sum(r["backward"] for r in results),model_reports=model_reports,all_weights_and_coefficients_unchanged=True,
        elapsed_seconds=time.perf_counter()-started,score_sign_fitting=False,low_FPR_or_performance_success_claim=False,
        scope="all original U8 patients; frozen FP16 coefficients, FP32 endpoint measurement; no new fitting or reoptimization")
    assert report["total_forward"]==3840 and report["total_backward"]==0
    write_json(out/"report.json",report)
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
