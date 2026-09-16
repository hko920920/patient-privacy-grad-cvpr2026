#!/usr/bin/env python3
"""Bounded exact-resume preflight using the real pinned SD 2.1 LoRA state.

Only K5 ``public_development`` records are loaded.  For every frozen arm
semantics, four uninterrupted optimizer steps are compared bit-for-bit with
two steps, an encrypted DPAPI checkpoint, state destruction/reload, and two
more steps.  This script never touches ``private_train`` and never creates the
full K5 run roots.
"""

from __future__ import annotations

import gc
import importlib.metadata
import json
import secrets
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from diffusers import DDPMScheduler

from dp_training import k5_full_runner as core
from dp_training.secure_resume import load_envelope, save_envelope_atomic


SCHEMA = "nih-cxr14-k5-actual-sd21-resume-preflight/v1"
OUTPUT_NAME = "nih_cxr14_k5_actual_sd21_resume_preflight_v1_002"
PROTOCOL_NAME = "nih_cxr14_k5_actual_sd21_resume_preflight_protocol_v1_002"
PREFLIGHT_PROTOCOL_SCHEMA = "nih-cxr14-k5-actual-sd21-resume-preflight-protocol/v1"
SOURCE_FILES = (
    "dp_training/k5_full_runner.py",
    "dp_training/run_k5_full_training.py",
    "dp_training/run_k5_actual_sd21_resume_preflight.py",
    "dp_training/mechanism.py",
    "dp_training/secure_resume.py",
    "dp_protocol/calibrate_xray_public_clip_norms.py",
    "data_pipeline/nih_cxr14_model_input.py",
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def public_development_fixture(records: list[Any]) -> list[Any]:
    by_patient: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        by_patient[record.patient_id].append(record)
    for values in by_patient.values():
        values.sort(key=lambda record: (record.cap_rank, record.image_id))
    eligible = [patient for patient in sorted(by_patient) if len(by_patient[patient]) >= 2]
    core.require(len(eligible) >= 8, "public-development fixture lacks eight multi-image patients")
    selected = [record for patient in eligible[:8] for record in by_patient[patient][:2]]
    core.require(len(selected) == 16 and len({record.patient_id for record in selected}) == 8, "fixture boundary")
    return selected


def run_steps(
    state: dict[str, Any],
    schedule: list[list[dict[str, Any]]],
    count: int,
    *,
    latents: dict[str, Any],
    hidden: dict[str, Any],
    model_input: Any,
    scheduler: Any,
) -> None:
    for _ in range(count):
        index = int(state["step"])
        core.require(index < len(schedule), "preflight step overflow")
        core.run_one_step(
            state=state,
            step_units=schedule[index],
            latent_by_image=latents,
            hidden_by_prompt=hidden,
            model_input=model_input,
            scheduler=scheduler,
        )


def run_arm(
    arm: str,
    *,
    snapshot: Path,
    protocol: dict[str, Any],
    experiment_root: bytes,
    schedule: list[list[dict[str, Any]]],
    schedule_meta: dict[str, Any],
    config: Any | None,
    latents: dict[str, Any],
    hidden: dict[str, Any],
    model_input: Any,
    scheduler: Any,
    frozen_digests: dict[str, Any],
) -> dict[str, Any]:
    print(f"ACTUAL_SD21_RESUME_PREFLIGHT_START {arm}", flush=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    baseline_state = core.create_training_state(
        snapshot, protocol, arm, experiment_root, config, schedule_meta
    )
    initial_digest = baseline_state["initial_adapter_sha256"]
    run_steps(
        baseline_state,
        schedule,
        4,
        latents=latents,
        hidden=hidden,
        model_input=model_input,
        scheduler=scheduler,
    )
    uninterrupted = core.capture_resume_payload(
        baseline_state,
        experiment_root=experiment_root,
        schedule_meta=schedule_meta,
        frozen_digests=frozen_digests,
    )
    core.destroy_training_state(baseline_state)
    del baseline_state

    interrupted_state = core.create_training_state(
        snapshot, protocol, arm, experiment_root, config, schedule_meta
    )
    core.require(
        interrupted_state["initial_adapter_sha256"] == initial_digest,
        "paired initial LoRA mismatch",
    )
    with tempfile.TemporaryDirectory(prefix=f"k5-actual-sd21-{arm}-") as directory:
        temporary_root = Path(directory)
        current = temporary_root / "resume.dpapi"
        step0 = core.capture_resume_payload(
            interrupted_state,
            experiment_root=experiment_root,
            schedule_meta=schedule_meta,
            frozen_digests=frozen_digests,
        )
        rng_before = {
            "cpu": torch.get_rng_state().clone(),
            "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()],
            "gaussian": None
            if interrupted_state["gaussian"] is None
            else interrupted_state["gaussian"].get_state().clone(),
        }
        step0_envelope = save_envelope_atomic(current, step0)
        rng_after = {
            "cpu": torch.get_rng_state().clone(),
            "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()],
            "gaussian": None
            if interrupted_state["gaussian"] is None
            else interrupted_state["gaussian"].get_state().clone(),
        }
        core.require(core.exact_equal(rng_before, rng_after), "DPAPI save consumed torch RNG")
        core.require(experiment_root not in current.read_bytes(), "root appears in ciphertext")
        run_steps(
            interrupted_state,
            schedule,
            2,
            latents=latents,
            hidden=hidden,
            model_input=model_input,
            scheduler=scheduler,
        )
        step2 = core.capture_resume_payload(
            interrupted_state,
            experiment_root=experiment_root,
            schedule_meta=schedule_meta,
            frozen_digests=frozen_digests,
        )
        step2_envelope = save_envelope_atomic(current, step2)
        previous = temporary_root / "resume.previous.dpapi"
        core.require(previous.is_file(), "step-0 previous envelope missing")
        core.require(load_envelope(previous)["committed_step"] == 0, "previous is not step 0")
        core.require(load_envelope(current)["committed_step"] == 2, "current is not step 2")
        core.destroy_training_state(interrupted_state)
        del interrupted_state, step0, step2
        gc.collect()
        torch.rand(17)
        torch.rand(17, device="cuda")

        loaded = load_envelope(current)
        resumed_state = core.restore_training_state(
            loaded,
            snapshot=snapshot,
            protocol=protocol,
            arm=arm,
            experiment_root=experiment_root,
            config=config,
            schedule_meta=schedule_meta,
            frozen_digests=frozen_digests,
        )
        run_steps(
            resumed_state,
            schedule,
            2,
            latents=latents,
            hidden=hidden,
            model_input=model_input,
            scheduler=scheduler,
        )
        resumed = core.capture_resume_payload(
            resumed_state,
            experiment_root=experiment_root,
            schedule_meta=schedule_meta,
            frozen_digests=frozen_digests,
        )
        checks = core.compare_resume_payloads(uninterrupted, resumed)
        final_digest = core.named_tensor_digest(resumed_state["trainable"])
        core.require(final_digest != initial_digest, "actual SD2.1 adapter did not update")
        core.require(int(resumed["committed_step"]) == 4, "resumed preflight did not finish")
        core.destroy_training_state(resumed_state)
        del resumed_state, loaded

    elapsed = time.perf_counter() - started
    del uninterrupted, resumed
    gc.collect()
    torch.cuda.empty_cache()
    report = {
        "status": "PASS",
        "exact_checks": checks,
        "initial_adapter_sha256": initial_digest,
        "final_adapter_sha256": final_digest,
        "schedule_commitment": schedule_meta["commitment"],
        "trace_events": 4,
        "step0_ciphertext_bytes": int(step0_envelope["ciphertext_bytes"]),
        "step2_ciphertext_bytes": int(step2_envelope["ciphertext_bytes"]),
        "current_plus_previous_rotation": bool(step2_envelope["previous_retained"]),
        "plaintext_resume_file_created": False,
        "DPAPI_save_consumed_torch_RNG": False,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    print(f"ACTUAL_SD21_RESUME_PREFLIGHT_PASS {arm} {elapsed:.3f}s", flush=True)
    return report


def scan_public(value: Any) -> None:
    forbidden = {
        "experiment_root",
        "generator_states",
        "schedule_generator_end_states",
        "dp_gaussian_generator_state",
        "global_cpu_rng",
        "global_cuda_rng_all",
        "noise_seed",
        "selected_ids",
        "patient_id",
        "image_id",
    }
    if isinstance(value, dict):
        core.require(not (set(value) & forbidden), "public preflight contains secret field")
        for child in value.values():
            scan_public(child)
    elif isinstance(value, list):
        for child in value:
            scan_public(child)
    elif isinstance(value, bytes):
        raise RuntimeError("public preflight contains bytes")


def main() -> int:
    core.configure_deterministic_cuda()
    paths = core.default_paths()
    output = paths["root"] / "_reports" / OUTPUT_NAME
    preflight_protocol_path = (
        paths["root"] / "_reports" / PROTOCOL_NAME / "protocol.json"
    )
    core.require(not output.exists(), "refusing to overwrite actual-SD2.1 resume preflight")
    core.require(not paths["run_root"].exists(), "full restricted K5 run root already exists")
    core.require(not paths["report_root"].exists(), "full public K5 report root already exists")
    core.require(core.sha256_file(paths["manifest"]) == core.K5_SHA256, "K5 manifest hash")
    core.require(
        core.sha256_file(paths["preprocessing"]) == core.PREPROCESSING_SHA256,
        "preprocessing hash",
    )
    core.require(preflight_protocol_path.is_file(), "actual-SD2.1 preflight protocol missing")
    preflight_protocol = json.loads(preflight_protocol_path.read_text(encoding="utf-8"))
    core.require(
        preflight_protocol["schema"] == PREFLIGHT_PROTOCOL_SCHEMA
        and preflight_protocol["status"]
        == "FROZEN_BEFORE_ACTUAL_SD21_RESUME_EXECUTION",
        "actual-SD2.1 preflight protocol status",
    )
    core.require(
        preflight_protocol["source_hashes"]
        == {
            relative: core.sha256_file(paths["root"] / Path(relative))
            for relative in SOURCE_FILES
        },
        "actual-SD2.1 preflight source drift",
    )
    protocol = core.load_frozen_protocol(paths["protocol"])
    model_input = core.load_model_input_module()
    public_records = model_input.read_manifest(
        paths["manifest"], partitions={"public_development"}
    )
    fixture = public_development_fixture(public_records)
    core.require(not any(record.partition == "private_train" for record in fixture), "private record reached preflight")
    snapshot, model_hashes = core.resolve_and_verify_snapshot(protocol)
    scheduler = DDPMScheduler.from_pretrained(snapshot / "scheduler", local_files_only=True)
    experiment_root = secrets.token_bytes(32)

    schedules = {}
    metas = {}
    configs = {}
    for arm in core.ARMS:
        schedule, meta, config = core.build_arm_schedule(
            fixture,
            protocol,
            arm,
            experiment_root,
            int(scheduler.config.num_train_timesteps),
            steps=4,
            fixture=True,
        )
        schedules[arm] = schedule
        metas[arm] = meta
        configs[arm] = config
    core.require(
        metas["M1-I8"]["commitment"] == metas["M1-G8"]["commitment"],
        "M1 preflight schedule is not shared",
    )
    all_units = [unit for arm in core.ARMS for step in schedules[arm] for unit in step]
    latents, hidden, cache_summary = core.prepare_arm_cache(
        snapshot,
        [all_units],
        paths["image_root"],
        model_input,
    )
    core.require(cache_summary["unique_images"] == 16, "fixture cache image count")

    source_hashes = {
        relative: core.sha256_file(paths["root"] / Path(relative))
        for relative in SOURCE_FILES
    }
    frozen_digests = core.frozen_digest_bundle(
        environment_manifest_sha256="ACTUAL_SD21_PREFLIGHT_BEFORE_ENVIRONMENT_FREEZE",
        source_hashes=source_hashes,
    )
    arm_reports = {}
    full_started = time.perf_counter()
    for arm in core.ARMS:
        arm_reports[arm] = run_arm(
            arm,
            snapshot=snapshot,
            protocol=protocol,
            experiment_root=experiment_root,
            schedule=schedules[arm],
            schedule_meta=metas[arm],
            config=configs[arm],
            latents=latents,
            hidden=hidden,
            model_input=model_input,
            scheduler=scheduler,
            frozen_digests=frozen_digests,
        )
    core.require(len({value["initial_adapter_sha256"] for value in arm_reports.values()}) == 1, "arm initial LoRA mismatch")
    report = {
        "schema": SCHEMA,
        "status": "PASS_ACTUAL_SD21_EXACT_ENCRYPTED_RESUME_ALL_FOUR_ARMS",
        "scope": "PUBLIC_DEVELOPMENT_BOUNDED_FOUR_STEP_PREFLIGHT_NOT_FULL_TRAINING_NOT_PRIVACY_NOT_UTILITY",
        "run_id_reserved_for_later": core.RUN_ID,
        "protocol_sha256": core.PROTOCOL_SHA256,
        "actual_resume_preflight_protocol_sha256": core.sha256_file(
            preflight_protocol_path
        ),
        "source_hashes": source_hashes,
        "model": {
            "id": core.MODEL_ID,
            "revision": core.MODEL_REVISION,
            "critical_hashes_verified": model_hashes,
            "real_UNet_and_rank8_fp32_LoRA": True,
            "trainable_parameters": core.TRAINABLE_PARAMETERS,
        },
        "fixture": {
            "partition": "public_development",
            "images": 16,
            "patients": 8,
            "images_per_patient": 2,
            "P256_deterministic_VAE_mode": True,
            "private_train_records_loaded": False,
        },
        "comparison": "four uninterrupted versus two plus DPAPI checkpoint, destruction/reload, plus two",
        "arms": arm_reports,
        "all_exact_checks_pass": all(
            all(report["exact_checks"].values()) for report in arm_reports.values()
        ),
        "M1_shared_schedule_commitment": arm_reports["M1-I8"]["schedule_commitment"],
        "all_arms_identical_initial_LoRA": True,
        "full_elapsed_seconds": time.perf_counter() - full_started,
        "environment": {
            "python": __import__("platform").python_version(),
            "torch": torch.__version__,
            "torch_cuda": str(torch.version.cuda),
            "gpu": torch.cuda.get_device_name(0),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("diffusers", "peft", "transformers", "safetensors", "accelerate")
            },
        },
        "artifact_boundary": {
            "encrypted_preflight_envelopes_retained": False,
            "plaintext_resume_file_created": False,
            "adapter_or_optimizer_retained": False,
            "generated_image_retained": False,
            "full_run_roots_created": False,
            "full_K5_optimizer_execution_started": False,
        },
        "next": "independently freeze and verify the full runner/environment; report before B0 or M0",
    }
    scan_public(report)
    output.mkdir(parents=True)
    report_path = output / "public_report.json"
    write_json(report_path, report)
    core.require(not paths["run_root"].exists(), "preflight created restricted full-run root")
    core.require(not paths["report_root"].exists(), "preflight created public full-run root")
    print(json.dumps({"status": report["status"], "report": str(report_path), "sha256": core.sha256_file(report_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
