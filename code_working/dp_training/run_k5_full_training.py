#!/usr/bin/env python3
"""Fail-closed command-line lifecycle for the frozen K5 training matrix.

The default ``validate`` command is read-only.  Creating the restricted run
requires the independently verified gate hash, an already completed B0
generation manifest, and the explicit ``--start-full-run`` acknowledgement.
Each long invocation runs exactly one arm and enforces the frozen order.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import secrets
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator

from dp_training import k5_full_runner as core
from dp_training.secure_resume import load_envelope, save_envelope_atomic


LAUNCH_AUTHORITY_SCHEMA = "nih-cxr14-k5-full-runner-independent-launch-authority/v1"
B0_SCHEMA = "nih-cxr14-k5-b0-generation-manifest/v1"
SOURCE_FILES = (
    "dp_training/k5_full_runner.py",
    "dp_training/run_k5_full_training.py",
    "dp_training/mechanism.py",
    "dp_training/secure_resume.py",
    "dp_protocol/calibrate_xray_public_clip_norms.py",
    "data_pipeline/nih_cxr14_model_input.py",
)


def parse_args() -> argparse.Namespace:
    paths = core.default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("validate", "initialize", "run-arm"), nargs="?", default="validate"
    )
    parser.add_argument("--arm", choices=core.ARMS)
    parser.add_argument("--protocol", type=Path, default=paths["protocol"])
    parser.add_argument("--manifest", type=Path, default=paths["manifest"])
    parser.add_argument("--image-root", type=Path, default=paths["image_root"])
    parser.add_argument("--preprocessing", type=Path, default=paths["preprocessing"])
    parser.add_argument("--run-root", type=Path, default=paths["run_root"])
    parser.add_argument("--report-root", type=Path, default=paths["report_root"])
    parser.add_argument("--launch-authority", type=Path, default=paths["launch_authority"])
    parser.add_argument("--b0-manifest", type=Path)
    parser.add_argument("--acknowledge-gate-sha256")
    parser.add_argument("--start-full-run", action="store_true")
    return parser.parse_args()


def load_authority(path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    path = path.resolve()
    core.require(path.is_file(), "independent launch authority is missing")
    authority_sha = core.sha256_file(path)
    authority = json.loads(path.read_text(encoding="utf-8"))
    core.require(authority["schema"] == LAUNCH_AUTHORITY_SCHEMA, "launch-authority schema")
    core.require(
        authority["status"] == "PASS_K5_FULL_RUNNER_GATE_NO_FULL_TRAINING_STARTED",
        "launch authority is not PASS",
    )
    core.require(authority["run_id"] == core.RUN_ID, "launch-authority run ID")
    core.require(authority["full_K5_optimizer_execution_started"] is False, "invalid launch boundary")
    environment_path = path.parent / authority["environment_manifest_file"]
    core.require(environment_path.is_file(), "environment manifest is missing")
    core.require(
        core.sha256_file(environment_path) == authority["environment_manifest_sha256"],
        "environment manifest hash mismatch",
    )
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    return authority, environment, authority_sha


def validate_sources(authority: dict[str, Any]) -> dict[str, str]:
    root = core.project_root()
    expected = authority["source_hashes"]
    core.require(set(expected) == set(SOURCE_FILES), "launch-authority source set drift")
    actual = {}
    for relative in SOURCE_FILES:
        path = root / Path(relative)
        core.require(path.is_file(), f"frozen source missing: {relative}")
        actual[relative] = core.sha256_file(path)
        core.require(actual[relative] == expected[relative], f"frozen source drift: {relative}")
    return actual


def validate_environment(environment: dict[str, Any]) -> dict[str, Any]:
    import torch

    current_packages = {
        name: importlib.metadata.version(name)
        for name in environment["packages"]
    }
    core.require(current_packages == environment["packages"], "Python package environment drift")
    core.require(platform.python_version() == environment["python"]["version"], "Python version drift")
    core.require(torch.__version__ == environment["torch"]["version"], "torch version drift")
    core.require(str(torch.version.cuda) == environment["torch"]["cuda"], "torch CUDA drift")
    core.require(torch.cuda.get_device_name(0) == environment["cuda_device"]["name"], "GPU identity drift")
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(0),
    }


def nvidia_state() -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
    core.require(len(query) == 1, "exactly one CUDA GPU is required")
    name, driver, total, free, used, temperature, utilization = [
        value.strip() for value in query[0].split(",")
    ]
    return {
        "name": name,
        "driver": driver,
        "memory_total_mib": int(total),
        "memory_free_mib": int(free),
        "memory_used_mib": int(used),
        "temperature_c": int(temperature),
        "utilization_percent": int(utilization),
        "idle_detection": "WDDM-safe aggregate utilization and free-memory thresholds",
    }


def require_idle_compatible_gpu(environment: dict[str, Any]) -> dict[str, Any]:
    state = nvidia_state()
    frozen = environment["cuda_device"]
    core.require(state["name"] == frozen["name"], "GPU name changed after environment freeze")
    core.require(state["driver"] == frozen["driver"], "NVIDIA driver changed after freeze")
    core.require(state["memory_total_mib"] == frozen["memory_total_mib"], "GPU memory drift")
    core.require(state["memory_free_mib"] >= 6_500, "insufficient idle GPU memory")
    core.require(state["temperature_c"] < 80, "GPU is too hot to start an arm")
    core.require(state["utilization_percent"] <= 10, "GPU utilization indicates an active workload")
    return state


def default_b0_manifest(report_root: Path) -> Path:
    return report_root / "B0" / "generation_manifest.json"


def validate_b0(path: Path, authority: dict[str, Any]) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    core.require(path.is_file(), "B0 generation manifest is required before initialization")
    digest = core.sha256_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    core.require(value["schema"] == B0_SCHEMA, "B0 manifest schema")
    core.require(value["status"] == "PASS_B0_FROZEN_BEFORE_PRIVATE_TRAINING", "B0 status")
    core.require(value["run_id"] == core.RUN_ID and value["model"] == "B0", "B0 identity")
    core.require(int(value["image_count"]) == 448, "B0 image count")
    core.require(value["generation_tasks_sha256"] == authority["generation_tasks_sha256"], "B0 task drift")
    core.require(value["source_gate_sha256"] == authority["primary_gate_sha256"], "B0 source gate drift")
    return value, digest


@contextlib.contextmanager
def exclusive_run_lock(run_root: Path) -> Iterator[None]:
    import msvcrt

    run_root.mkdir(parents=True, exist_ok=True)
    lock_path = run_root / "matrix.lock"
    handle = lock_path.open("a+b")
    try:
        if lock_path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError("another K5 matrix process holds the exclusive lock") from error
        yield
    finally:
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            handle.close()


def validate_static_inputs(args: argparse.Namespace) -> dict[str, Any]:
    protocol = core.load_frozen_protocol(args.protocol)
    core.require(core.sha256_file(args.manifest.resolve()) == core.K5_SHA256, "K5 manifest hash")
    core.require(
        core.sha256_file(args.preprocessing.resolve()) == core.PREPROCESSING_SHA256,
        "preprocessing hash",
    )
    core.require(args.image_root.resolve().is_dir(), "NIH image root is missing")
    authority, environment, authority_sha = load_authority(args.launch_authority)
    sources = validate_sources(authority)
    core.configure_deterministic_cuda()
    runtime_environment = validate_environment(environment)
    return {
        "protocol": protocol,
        "authority": authority,
        "environment": environment,
        "authority_sha": authority_sha,
        "sources": sources,
        "runtime_environment": runtime_environment,
    }


def initialize(args: argparse.Namespace, context: dict[str, Any]) -> dict[str, Any]:
    core.require(args.start_full_run, "initialize requires --start-full-run")
    core.require(
        args.acknowledge_gate_sha256 == context["authority_sha"],
        "exact independently verified gate SHA-256 acknowledgement is required",
    )
    run_root = args.run_root.resolve()
    report_root = args.report_root.resolve()
    b0_path = (args.b0_manifest or default_b0_manifest(report_root)).resolve()
    _, b0_sha = validate_b0(b0_path, context["authority"])
    core.require(not run_root.exists(), "restricted K5 run root already exists; refuse reinitialization")
    free_gib = core.require_free_space(run_root, 15.0)
    gpu = require_idle_compatible_gpu(context["environment"])
    run_root.mkdir(parents=True)
    report_root.mkdir(parents=True, exist_ok=True)
    with exclusive_run_lock(run_root):
        experiment_root = secrets.token_bytes(32)
        root_commitment = core.sha256_bytes(
            b"nih-cxr14-k5-matrix-root-v1\0" + experiment_root
        )
        matrix_payload = {
            "schema": core.MATRIX_SCHEMA,
            "run_id": core.RUN_ID,
            "experiment_root": experiment_root,
            "root_commitment": root_commitment,
            "protocol_sha256": core.PROTOCOL_SHA256,
            "launch_authority_sha256": context["authority_sha"],
            "environment_manifest_sha256": context["authority"]["environment_manifest_sha256"],
            "b0_manifest_sha256": b0_sha,
            "arms": list(core.ARMS),
        }
        envelope = save_envelope_atomic(run_root / "matrix.dpapi", matrix_payload)
        public = {
            "schema": "nih-cxr14-k5-matrix-initialization/v1",
            "status": "INITIALIZED_NO_PRIVATE_OPTIMIZER_STEP_YET",
            "run_id": core.RUN_ID,
            "root_commitment": root_commitment,
            "protocol_sha256": core.PROTOCOL_SHA256,
            "launch_authority_sha256": context["authority_sha"],
            "b0_manifest_sha256": b0_sha,
            "arm_order": list(core.ARMS),
            "free_space_gib_at_initialization": free_gib,
            "gpu": {key: gpu[key] for key in ("name", "driver", "memory_total_mib", "temperature_c")},
            "encrypted_matrix_envelope_bytes": envelope["ciphertext_bytes"],
            "private_seed_reported": False,
            "full_optimizer_steps_completed": 0,
        }
        core.write_json_atomic(report_root / "matrix_initialized.json", public)
    return public


def load_matrix(
    run_root: Path, context: dict[str, Any], b0_sha: str
) -> dict[str, Any]:
    payload = load_envelope(run_root / "matrix.dpapi")
    core.require(payload["schema"] == core.MATRIX_SCHEMA, "matrix envelope schema")
    core.require(payload["run_id"] == core.RUN_ID, "matrix run ID")
    core.require(payload["protocol_sha256"] == core.PROTOCOL_SHA256, "matrix protocol drift")
    core.require(payload["launch_authority_sha256"] == context["authority_sha"], "matrix gate drift")
    core.require(payload["b0_manifest_sha256"] == b0_sha, "matrix B0 drift")
    core.require(len(payload["experiment_root"]) == 32, "matrix root length")
    expected = core.sha256_bytes(
        b"nih-cxr14-k5-matrix-root-v1\0" + bytes(payload["experiment_root"])
    )
    core.require(payload["root_commitment"] == expected, "matrix root commitment")
    return payload


def completion_path(report_root: Path, arm: str) -> Path:
    return report_root / "arms" / arm / "completion.json"


def enforce_arm_order(report_root: Path, arm: str) -> None:
    index = core.ARMS.index(arm)
    for previous in core.ARMS[:index]:
        path = completion_path(report_root, previous)
        core.require(path.is_file(), f"previous arm is incomplete: {previous}")
        value = json.loads(path.read_text(encoding="utf-8"))
        core.require(value["status"] == "PASS_4000_FIXED_STEPS", f"previous arm failed: {previous}")
    for later in core.ARMS[index + 1 :]:
        core.require(not completion_path(report_root, later).exists(), f"later arm already exists: {later}")


def replay_accounting(protocol: dict[str, Any], arm: str) -> dict[str, Any] | None:
    if arm == "M0":
        return None
    from dp_accounting import dp_event
    from dp_accounting.privacy_accountant import NeighboringRelation
    from dp_accounting.rdp import RdpAccountant
    from opacus.accountants.analysis import rdp

    orders = [round(1.0 + index / 10.0, 1) for index in range(1, 100)]
    orders += list(range(12, 64)) + [64, 128, 256, 512]
    spec = protocol["DP_arms"][arm]
    target = spec["ideal_4000_step_bound"]
    delta_key = "delta_patient" if arm == "M2-P8" else "delta_image"
    epsilon_key = "epsilon_patient" if arm == "M2-P8" else "epsilon_image"
    q = float(spec["q"])
    sigma = float(spec["sigma"])
    delta = float(target[delta_key])
    opacus_rdp = rdp.compute_rdp(q=q, noise_multiplier=sigma, steps=core.FULL_STEPS, orders=orders)
    opacus_epsilon, opacus_order = rdp.get_privacy_spent(orders=orders, rdp=opacus_rdp, delta=delta)
    google = RdpAccountant(orders=orders, neighboring_relation=NeighboringRelation.ADD_OR_REMOVE_ONE)
    google.compose(dp_event.PoissonSampledDpEvent(q, dp_event.GaussianDpEvent(sigma)), count=core.FULL_STEPS)
    google_epsilon, google_order = google.get_epsilon_and_optimal_order(delta)
    conservative = max(float(opacus_epsilon), float(google_epsilon))
    core.require(conservative <= float(target[epsilon_key]) + 1e-10, "accounting exceeds frozen target")
    return {
        "privacy_unit": spec["privacy_unit"],
        "adjacency": spec["adjacency"],
        "events": core.FULL_STEPS,
        "q": q,
        "sigma": sigma,
        "delta": delta,
        "frozen_epsilon_bound": float(target[epsilon_key]),
        "opacus_epsilon": float(opacus_epsilon),
        "opacus_optimal_order": float(opacus_order),
        "google_epsilon": float(google_epsilon),
        "google_optimal_order": float(google_order),
        "conservative_epsilon": conservative,
        "research_PRNG_not_release_authority": True,
    }


def run_arm(args: argparse.Namespace, context: dict[str, Any]) -> dict[str, Any]:
    import torch
    from diffusers import DDPMScheduler

    core.require(args.arm in core.ARMS, "run-arm requires --arm")
    core.require(args.start_full_run, "run-arm requires --start-full-run")
    core.require(
        args.acknowledge_gate_sha256 == context["authority_sha"],
        "exact launch gate acknowledgement is required",
    )
    run_root = args.run_root.resolve()
    report_root = args.report_root.resolve()
    core.require(run_root.is_dir(), "matrix must be initialized first")
    b0_path = (args.b0_manifest or default_b0_manifest(report_root)).resolve()
    _, b0_sha = validate_b0(b0_path, context["authority"])
    core.require_free_space(run_root, 15.0)
    gpu_start = require_idle_compatible_gpu(context["environment"])
    enforce_arm_order(report_root, args.arm)
    core.require(not completion_path(report_root, args.arm).exists(), "arm is already complete")

    with exclusive_run_lock(run_root):
        matrix = load_matrix(run_root, context, b0_sha)
        experiment_root = bytes(matrix["experiment_root"])
        model_input = core.load_model_input_module()
        records = model_input.read_manifest(args.manifest.resolve(), partitions={"private_train"})
        core.validate_full_records(records)
        snapshot, _ = core.resolve_and_verify_snapshot(context["protocol"])
        scheduler = DDPMScheduler.from_pretrained(snapshot / "scheduler", local_files_only=True)
        schedule, schedule_meta, config = core.build_arm_schedule(
            records,
            context["protocol"],
            args.arm,
            experiment_root,
            int(scheduler.config.num_train_timesteps),
        )
        arm_root = run_root / "arms" / args.arm
        resume_path = arm_root / "resume.dpapi"
        source_hashes = context["sources"]
        frozen_digests = core.frozen_digest_bundle(
            environment_manifest_sha256=context["authority"]["environment_manifest_sha256"],
            source_hashes=source_hashes,
        )
        restored_payload = None
        if resume_path.exists():
            restored_payload = load_envelope(resume_path)
            committed_step = int(restored_payload["committed_step"])
        else:
            committed_step = 0
        core.require(0 <= committed_step <= core.FULL_STEPS, "invalid resume step")
        latents, hidden, cache_summary = core.prepare_arm_cache(
            snapshot,
            schedule,
            args.image_root.resolve(),
            model_input,
            first_step_index=committed_step,
        )
        state = None
        started = time.perf_counter()
        try:
            if restored_payload is None:
                arm_root.mkdir(parents=True, exist_ok=False)
                state = core.create_training_state(
                    snapshot,
                    context["protocol"],
                    args.arm,
                    experiment_root,
                    config,
                    schedule_meta,
                )
                initial_payload = core.capture_resume_payload(
                    state,
                    experiment_root=experiment_root,
                    schedule_meta=schedule_meta,
                    frozen_digests=frozen_digests,
                )
                save_envelope_atomic(resume_path, initial_payload)
            else:
                state = core.restore_training_state(
                    restored_payload,
                    snapshot=snapshot,
                    protocol=context["protocol"],
                    arm=args.arm,
                    experiment_root=experiment_root,
                    config=config,
                    schedule_meta=schedule_meta,
                    frozen_digests=frozen_digests,
                )
            initial_digest = state["initial_adapter_sha256"]
            for previous in core.ARMS[: core.ARMS.index(args.arm)]:
                prior = json.loads(completion_path(report_root, previous).read_text(encoding="utf-8"))
                core.require(prior["initial_adapter_sha256"] == initial_digest, "initial LoRA differs across arms")
            if args.arm == "M1-G8":
                prior = json.loads(completion_path(report_root, "M1-I8").read_text(encoding="utf-8"))
                core.require(prior["schedule_commitment"] == schedule_meta["commitment"], "M1 shared schedule mismatch")

            torch.cuda.reset_peak_memory_stats()
            while int(state["step"]) < core.FULL_STEPS:
                index = int(state["step"])
                core.run_one_step(
                    state=state,
                    step_units=schedule[index],
                    latent_by_image=latents,
                    hidden_by_prompt=hidden,
                    model_input=model_input,
                    scheduler=scheduler,
                )
                step = int(state["step"])
                if step % core.RESUME_INTERVAL == 0:
                    payload = core.capture_resume_payload(
                        state,
                        experiment_root=experiment_root,
                        schedule_meta=schedule_meta,
                        frozen_digests=frozen_digests,
                    )
                    save_envelope_atomic(resume_path, payload)
                    progress = {
                        "schema": "nih-cxr14-k5-arm-progress/v1",
                        "run_id": core.RUN_ID,
                        "arm": args.arm,
                        "committed_step": step,
                        "maximum_step": core.FULL_STEPS,
                        "trace_event_count": state["trace_event_count"],
                        "trace_head_sha256": state["trace_head"],
                        "adapter_sha256": core.named_tensor_digest(state["trainable"]),
                        "private_realization_reported": False,
                    }
                    core.write_json_atomic(arm_root / "committed_progress.json", progress)
                    if step in core.ADAPTER_STEPS:
                        core.save_adapter_snapshot(state, arm_root, frozen_digests)

            elapsed = time.perf_counter() - started
            final_digest = core.named_tensor_digest(state["trainable"])
            core.require(state["trace_event_count"] == core.FULL_STEPS, "final trace event count")
            accounting = replay_accounting(context["protocol"], args.arm)
            restricted_summary = {
                "schema": "nih-cxr14-k5-arm-restricted-summary/v1",
                "classification": "LOCAL_RESTRICTED_DO_NOT_RELEASE",
                "run_id": core.RUN_ID,
                "arm": args.arm,
                "scheduled_units": schedule_meta["scheduled_units"],
                "scheduled_raw_images": schedule_meta["scheduled_raw_images"],
                "unique_images_cached_in_this_process": cache_summary["unique_images"],
                "identifiers_serialized": False,
                "seed_values_serialized_outside_DPAPI": False,
            }
            core.write_json_atomic(arm_root / "restricted_summary.json", restricted_summary)
            public = {
                "schema": "nih-cxr14-k5-arm-completion/v1",
                "status": "PASS_4000_FIXED_STEPS",
                "scope": "ONE_SEED_K5_RESEARCH_FEASIBILITY_NOT_RELEASE",
                "run_id": core.RUN_ID,
                "arm": args.arm,
                "completed_steps": core.FULL_STEPS,
                "initial_adapter_sha256": initial_digest,
                "final_adapter_sha256": final_digest,
                "schedule_commitment": schedule_meta["commitment"],
                "trace_event_count": state["trace_event_count"],
                "trace_head_sha256": state["trace_head"],
                "accounting": accounting,
                "elapsed_seconds_this_invocation": elapsed,
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "frozen_digests": frozen_digests,
                "gpu_at_start": {key: gpu_start[key] for key in ("name", "driver", "memory_total_mib", "temperature_c")},
                "identifiers_realized_counts_losses_norms_seeds_reported": False,
                "adapter_snapshot_steps": list(core.ADAPTER_STEPS),
                "quantitative_evaluation_checkpoint": core.FULL_STEPS,
                "research_PRNG_release_eligible": False,
            }
            core.write_json_atomic(completion_path(report_root, args.arm), public)
            return public
        finally:
            core.destroy_training_state(state)


def main() -> int:
    args = parse_args()
    context = validate_static_inputs(args)
    validation = {
        "status": "PASS_READ_ONLY_FULL_RUNNER_VALIDATION",
        "run_id": core.RUN_ID,
        "protocol_sha256": core.PROTOCOL_SHA256,
        "launch_authority_sha256": context["authority_sha"],
        "source_hashes": context["sources"],
        "runtime_environment": context["runtime_environment"],
        "optimizer_steps_started": False,
    }
    if args.command == "validate":
        result = validation
    elif args.command == "initialize":
        result = initialize(args, context)
    else:
        result = run_arm(args, context)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
