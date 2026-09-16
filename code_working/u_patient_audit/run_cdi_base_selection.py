"""Literal CDI features on the frozen pre-NIH base; selection U only.

A/B is a later fixed-scorer assignment diagnostic, not base membership truth.
This producer performs no attack fitting, performance analysis or downloading.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time
import traceback
import uuid

import torch
from safetensors import safe_open

from .common import ROOT, RUN, PROMPT, MODEL_ID, MODEL_REVISION, digest, read_csv, write_json
from .models import setup, load_cache, snapshot, UNet2DConditionModel, DDPMScheduler
from .cdi_adapter import CDIAdapter, FEATURE_NAMES, source_bindings, tensor_digest

OUTPUT = RUN / "baseline_screen_20260914/cdi_base_selection_v1"
TARGET = RUN / "baseline_screen_20260914/cdi_u_cohort_v1"
KERNEL = RUN / "baseline_screen_20260914/cdi_kernel_v1"
ADAPTER_SHA256 = "70d91b15397394d1405c8e930d2ec94ff4dcb0d3bbddd7ace6c8bdd58b0da8ce"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def now():
    return datetime.now(timezone.utc).isoformat()


def hashes_match(expected):
    for path, value in expected.items():
        assert digest(Path(path)) == value, "Frozen file changed: " + str(path)


def new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), "Immutable output exists: " + str(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".pending")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.rename(path)


def validate(contract):
    required = dict(schema="cdi-base-selection-contract/v1", current_step=2,
        model="base", eval_roles=["selection"], scenario="U", master_seed=260914,
        stream="primary", batch_size=1, prediction_type="epsilon", dtype="float32",
        prompt=PROMPT, code_policy="released_code_literal", perform_fitting=False,
        perform_performance_evaluation=False, expected_patients=40,
        expected_unique_images=80, expected_records=80)
    for key, value in required.items():
        assert contract[key] == value, "Contract mismatch: " + key
    assert Path(contract["output_directory"]).resolve() == OUTPUT.resolve()
    snap = snapshot().resolve()  # local-only pinned revision; no model construction.
    assert snap.name == MODEL_REVISION and Path(contract["base_snapshot_path"]).resolve() == snap
    weight = snap / "unet/diffusion_pytorch_model.fp16.safetensors"
    assert Path(contract["base_weight_path"]).resolve() == weight
    frozen = {str(Path(p).resolve()): h for p, h in contract["frozen_sha256"].items()}
    essential = [Path(__file__), ROOT / "u_patient_audit/cdi_adapter.py",
        ROOT / "u_patient_audit/common.py", ROOT / "u_patient_audit/models.py",
        weight, snap / "unet/config.json", snap / "scheduler/scheduler_config.json",
        RUN / "cache/cache.pt", RUN / "cache/summary.json", RUN / "cohort/lock.json",
        RUN / "cohort/evaluation_images.csv", RUN / "cohort/model_1_train.csv",
        RUN / "cohort/model_2_train.csv"]
    essential += [KERNEL / name for name in ("protocol.json", "execution.json", "results.json", "raw.pt", "verification.json")]
    essential += [TARGET / name for name in ("protocol.json", "execution.json", "results.json", "verification.json")]
    essential += [TARGET / "analysis_v1" / name for name in (
        "analysis.json", "fit_parameters.json", "cv_results.json", "predictions.json",
        "same_fitted_scorer_predictions.json", "provenance.json", "contract_copy.json")]
    for path in essential:
        assert str(path.resolve()) in frozen, "Missing frozen binding: " + str(path)
    assert frozen[str(weight)] == contract["base_weight_sha256"]
    historical_weight = load(RUN / "cache/summary.json")["model_hashes"]["unet/diffusion_pytorch_model.fp16.safetensors"]
    assert historical_weight["status"] == "PASS"
    assert historical_weight["expected"].lower() == historical_weight["actual"].lower() == contract["base_weight_sha256"].lower()
    assert frozen[str((ROOT / "u_patient_audit/cdi_adapter.py").resolve())] == ADAPTER_SHA256
    hashes_match(frozen)
    target_protocol = load(TARGET / "protocol.json")
    target_contract = target_protocol["contract"]
    for key in ("master_seed", "stream", "batch_size", "prediction_type", "dtype", "prompt", "code_policy"):
        assert target_contract[key] == contract[key], "Different target/base protocol: " + key
    for path, value in target_contract["frozen_sha256"].items():
        assert frozen[str(Path(path).resolve())] == value, "Target binding missing/changed: " + path
    source = source_bindings()
    assert source == target_protocol["source_bindings"]
    previous = load(TARGET / "execution.json")
    verification = load(TARGET / "verification.json")
    assert previous["complete"] and previous["records"] == 480
    assert previous["status"] == "PASS_COHORT_EXTRACTION_PENDING_INDEPENDENT_VERIFICATION"
    assert verification["status"] == "PASS_SAVED_U_COHORT_ARITHMETIC_AND_INTEGRITY"
    assert verification["analysis_verification"]["status"] == "PASS_SAVED_ANALYSIS_PREDICTIONS_AND_STATISTICS"
    assert verification["analysis_verification"]["provenance_sha256"] == digest(TARGET / "analysis_v1/provenance.json")
    for name, expected in load(TARGET / "analysis_v1/provenance.json")["output_sha256"].items():
        assert digest(TARGET / "analysis_v1" / name) == expected
    assert digest(TARGET / "results.json") == previous["results_sha256"]
    assert digest(TARGET / "protocol.json") == previous["protocol_sha256"]
    lock = load(RUN / "cohort/lock.json")
    for name, expected in lock["files"].items():
        assert digest(RUN / "cohort" / name) == expected
    evaluation = read_csv(RUN / "cohort/evaluation_images.csv")
    by_patient = defaultdict(list)
    for row in evaluation:
        if row["eval_role"] == "selection":
            by_patient[row["patient_id"]].append(row)
    order = sorted(by_patient, key=int)
    assert order == contract["patient_order"] and len(order) == 40
    assert Counter(rows[0]["assignment_group"] for rows in by_patient.values()) == {"A": 20, "B": 20}
    rows = []
    for patient in order:
        candidate = by_patient[patient]
        assert len(candidate) == 4 and len({r["assignment_group"] for r in candidate}) == 1
        assert Counter(r["record_role"] for r in candidate) == {"train_candidate": 2, "U_observed": 2}
        for row in sorted((r for r in candidate if r["record_role"] == "U_observed"), key=lambda r: r["image_id"]):
            rows.append(dict(model="base", patient_id=patient, image_id=row["image_id"],
                eval_role="selection", scenario="U", record_role="U_observed",
                assignment_group=row["assignment_group"], member=0,
                incremental_patient_member=0, image_member=0, actual_training_exposures=0,
                patient_training_exposures=0, pretrained_membership="unknown",
                membership_label_scope="NIH_incremental_training_only",
                assignment_group_is_base_membership=False,
                base_weight_sha256=contract["base_weight_sha256"],
                checkpoint_sha256=contract["base_weight_sha256"]))
    assert len(rows) == len({r["image_id"] for r in rows}) == 80
    assert [r["image_id"] for r in rows] == contract["image_order"]
    prior_rows = {(r["model"], r["image_id"]): r for r in load(TARGET / "results.json")}
    for model in ("model_1", "model_2"):
        manifest = {r["image_id"] for r in read_csv(RUN / "cohort" / (model + "_train.csv"))}
        for row in rows:
            old = prior_rows[(model, row["image_id"])]
            assert all(old[k] == row[k] for k in ("patient_id", "eval_role", "scenario", "record_role", "assignment_group"))
            assert old["image_member"] == old["actual_training_exposures"] == 0
            assert row["image_id"] not in manifest
    cache = load_cache()
    assert cache["audit_prompt"] == PROMPT
    hidden = cache["hidden"][PROMPT].float()
    assert tuple(hidden.shape) == (1, 77, 1024) and bool(torch.isfinite(hidden).all())
    sched = DDPMScheduler.from_pretrained(snap / "scheduler", local_files_only=True)
    assert sched.config.prediction_type == "epsilon" and sched.config.num_train_timesteps == 1000
    alphas = sched.alphas_cumprod.detach().cpu().float()
    original_raw = torch.load(KERNEL / "raw.pt", map_location="cpu", weights_only=True)
    for packet in original_raw.values():
        assert torch.equal(packet["alphas"], alphas)
        assert packet["hidden_sha256"] == tensor_digest(hidden)
    for row in rows:
        z = cache["latents"][row["image_id"]]
        assert tuple(z.shape) == (1, 4, 32, 32) and bool(torch.isfinite(z).all())
    del original_raw
    return dict(rows=rows, frozen=frozen, snapshot=snap, weight=weight, cache=cache,
                scheduler=sched, alphas=alphas, hidden=hidden, source_bindings=source)


def state_fingerprint(unet, weight_path=None):
    """Bound actual FP32 values; optional exact comparison to the fp16 artifact."""
    state = unet.state_dict()
    combined = hashlib.sha256()
    checked = 0
    reader = safe_open(str(weight_path), framework="pt", device="cpu") if weight_path else None
    try:
        if reader is not None:
            assert set(reader.keys()) == set(state), "Loaded state differs from base artifact keys"
        for name in sorted(state):
            value = state[name].detach().cpu().contiguous()
            assert bool(torch.isfinite(value).all()), "Nonfinite base state: " + name
            if value.is_floating_point():
                assert value.dtype == torch.float32
            if reader is not None:
                expected = reader.get_tensor(name).to(value.dtype)
                assert torch.equal(expected, value), "Loaded base artifact mismatch: " + name
                checked += 1
                del expected
            combined.update(name.encode("utf-8") + b"\0")
            combined.update(str(value.dtype).encode("ascii"))
            combined.update(json.dumps(list(value.shape)).encode("ascii"))
            combined.update(value.numpy().tobytes())
            del value
    finally:
        if reader is not None:
            del reader
    return dict(sha256=combined.hexdigest(), state_tensors=len(state),
                artifact_tensors_exactly_checked=checked)


def save_pair(expected, measured, raw):
    iid = expected["image_id"]
    assert Path(iid).name == iid and Path(iid).stem
    raw_path = OUTPUT / (Path(iid).stem + ".pt")
    json_path = raw_path.with_suffix(".json")
    assert not raw_path.exists() and not json_path.exists(), "Immutable image pair already exists"
    temporary = raw_path.with_name(raw_path.name + "." + uuid.uuid4().hex + ".pending")
    with temporary.open("xb") as handle:
        torch.save(raw, handle)
    raw_hash = digest(temporary)
    temporary.rename(raw_path)
    for key in set(expected).intersection(measured):
        assert measured[key] == expected[key]
    row = {**measured, **expected, "raw_path": raw_path.relative_to(OUTPUT).as_posix(),
           "raw_sha256": raw_hash, "reused_record": False}
    new_json(json_path, row)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.contract = args.contract.resolve()
    contract = load(args.contract)
    contract_hash = digest(args.contract)
    context = validate(contract)
    if args.dry_run:
        assert not torch.cuda.is_initialized(), "CPU preflight initialized CUDA"
        print(json.dumps(dict(status="PASS_CPU_BASE_SELECTION_PREFLIGHT", records=80,
            patients=40, scenario="U", eval_role="selection", new_GPU_calls=0,
            new_fitting=False, base_weight_sha256=contract["base_weight_sha256"],
            contract_sha256=contract_hash), ensure_ascii=False), flush=True)
        return
    assert not OUTPUT.exists(), "Output exists; preserve immutable packets and review separately"
    OUTPUT.mkdir(parents=True, exist_ok=False)
    protocol = dict(schema="cdi-base-selection-execution/v1", current_step=2,
        started_utc=now(), contract_path=str(args.contract), contract_sha256=contract_hash,
        contract=contract, source_bindings=context["source_bindings"], measurement_plan=context["rows"],
        model_policy="fresh pinned fp16 safetensors base promoted to FP32; no LoRA modules loaded or added",
        noise_policy="identical CDI per-image/module/draw master seed and primary stream as target run",
        environment=dict(python=platform.python_version(), **{n: version(n) for n in ("torch", "diffusers", "peft", "numpy", "scipy", "safetensors")}),
        scope="selection40 U2 base feature extraction only; pretraining membership unknown; A/B is assignment metadata")
    new_json(OUTPUT / "protocol.json", protocol)
    protocol_hash = digest(OUTPUT / "protocol.json")
    begun = time.perf_counter()
    rows = []
    unet = adapter = None
    cheap = {str(args.contract): contract_hash, **{p: h for p, h in context["frozen"].items() if Path(p).suffix == ".py"}}
    try:
        setup()
        torch.manual_seed(contract["master_seed"])
        torch.cuda.manual_seed_all(contract["master_seed"])
        unet = UNet2DConditionModel.from_pretrained(context["snapshot"] / "unet",
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True, local_files_only=True)
        assert not any("lora" in name.lower() for name, _ in unet.named_parameters())
        assert not any("lora" in type(module).__name__.lower() for module in unet.modules())
        assert not getattr(unet, "peft_config", None)
        unet.eval().requires_grad_(False).to(device="cuda", dtype=torch.float32)
        before = state_fingerprint(unet, context["weight"])
        adapter = CDIAdapter(unet, context["scheduler"], context["hidden"], contract["master_seed"])
        assert not any(p.requires_grad or p.grad is not None for p in unet.parameters())
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        print(json.dumps(dict(event="base_loaded", model="base", remaining_images=80,
            seconds=time.perf_counter()-begun, base_weight_sha256=contract["base_weight_sha256"],
            no_lora=True, artifact_tensors_exactly_checked=before["artifact_tensors_exactly_checked"])), flush=True)
        for metadata in context["rows"]:
            iid = metadata["image_id"]
            measured = adapter.score(context["cache"]["latents"][iid], iid, stream=contract["stream"])
            raw = measured.pop("raw")
            assert torch.equal(raw["alphas"], context["alphas"])
            assert raw["hidden_sha256"] == tensor_digest(context["hidden"])
            assert measured["feature_names"] == FEATURE_NAMES and len(measured["features"]) == 26
            expected = {**metadata, "contract_sha256": contract_hash,
                "cache_sha256": context["frozen"][str((RUN / "cache/cache.pt").resolve())],
                "cohort_lock_sha256": context["frozen"][str((RUN / "cohort/lock.json").resolve())]}
            rows.append(save_pair(expected, measured, raw))
            del raw, measured
            if len(rows) % 10 == 0:
                hashes_match(cheap)
                progress = dict(event="progress", complete=False, records=len(rows),
                    expected_records=80, model="base", seconds=time.perf_counter()-begun,
                    forward=adapter.forward, backward=adapter.backward, contract_sha256=contract_hash)
                write_json(OUTPUT / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        assert len(rows) == len({r["image_id"] for r in rows}) == 80
        assert [r["image_id"] for r in rows] == contract["image_order"]
        assert all(r["member"] == r["image_member"] == r["actual_training_exposures"] == 0 for r in rows)
        assert adapter.assert_weights_unchanged()
        after = state_fingerprint(unet)
        assert before["sha256"] == after["sha256"], "Base state values changed"
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        total_f, total_b = sum(r["forward"] for r in rows), sum(r["backward"] for r in rows)
        assert total_f == adapter.forward and total_b == adapter.backward
        for row in rows:
            assert digest(OUTPUT / row["raw_path"]) == row["raw_sha256"]
            saved = load((OUTPUT / row["raw_path"]).with_suffix(".json"))
            assert saved == row
        hashes_match(context["frozen"])
        hashes_match({str(args.contract): contract_hash})
        new_json(OUTPUT / "results.json", rows)
        summary = dict(schema="cdi-base-selection-result/v1",
            status="PASS_BASE_SELECTION_EXTRACTION_PENDING_INDEPENDENT_VERIFICATION",
            complete=True, current_step=2, ended_utc=now(), records=80,
            unique_patients=40, unique_images=80, model="base", eval_role="selection", scenario="U",
            forward=total_f, backward=total_b, runner_seconds=time.perf_counter()-begun,
            score_seconds=sum(r["seconds"] for r in rows), peak_cuda_allocated_bytes=peak,
            NO_objective_evaluations=sum(r["modules"]["noise_optim"]["optimizer"]["objective_evaluations"] for r in rows),
            NO_nfev=sum(r["modules"]["noise_optim"]["optimizer"]["nfev"] for r in rows),
            NO_termination_status_counts=dict(Counter(str(r["modules"]["noise_optim"]["optimizer"]["status"]) for r in rows)),
            base_weight_path=str(context["weight"]), base_weight_sha256=contract["base_weight_sha256"],
            base_model_id=MODEL_ID, base_model_revision=MODEL_REVISION,
            source_variant="fp16", compute_dtype="float32", no_lora_modules=True, lora_parameter_count=0,
            initial_state=before, final_state=after, state_values_exactly_unchanged=True,
            parameter_versions_checked=len(adapter.initial_versions), parameter_versions_and_no_weight_gradients=True,
            all_parameters_frozen=True, unet_training=False, gradient_checkpointing_enabled=False,
            frozen_files_unchanged=True, protocol_sha256=protocol_hash, contract_sha256=contract_hash,
            results_sha256=digest(OUTPUT / "results.json"),
            pretrained_membership="unknown", all_NIH_incremental_membership_labels=0,
            assignment_A_B_is_base_membership=False, attack_fitting=False,
            performance_evaluation=False, base_membership_AUC=False,
            vae_forward=0, text_encoder_forward=0, new_target_training=False,
            scope="base response control; selection is prior development data; no privacy absence or causal identification claim")
        new_json(OUTPUT / "execution.json", summary)
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
    except BaseException as exc:
        new_json(OUTPUT / "failure.json", dict(status="FAILED_INCOMPLETE_BASE_SELECTION", complete=False,
            ended_utc=now(), records_saved=len(rows), seconds=time.perf_counter()-begun,
            error=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        if adapter is not None:
            adapter._forward_handle.remove()
        adapter = unet = None
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
