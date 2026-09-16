#!/usr/bin/env python3
"""Freeze the full K5 runner, environment, and launch contract after preflights."""

from __future__ import annotations

import datetime as dt
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from dp_training import k5_full_runner as core


GATE_NAME = "nih_cxr14_k5_full_runner_gate_v1_001"
ENV_SCHEMA = "nih-cxr14-k5-full-runner-environment/v1"
PRIMARY_SCHEMA = "nih-cxr14-k5-full-runner-primary-gate/v1"
RUNNER_SOURCE_FILES = (
    "dp_training/k5_full_runner.py",
    "dp_training/run_k5_full_training.py",
    "dp_training/mechanism.py",
    "dp_training/secure_resume.py",
    "dp_protocol/calibrate_xray_public_clip_norms.py",
    "data_pipeline/nih_cxr14_model_input.py",
)
AUDIT_SOURCE_FILES = (
    "dp_training/run_k5_actual_sd21_resume_preflight.py",
    "dp_training/build_k5_actual_sd21_resume_preflight_protocol.py",
    "dp_training/test_k5_full_runner.py",
    "dp_training/freeze_k5_full_runner_gate.py",
    "dp_training/verify_k5_full_runner_gate_independent.py",
)
PACKAGE_NAMES = (
    "torch",
    "torchvision",
    "diffusers",
    "peft",
    "transformers",
    "safetensors",
    "accelerate",
    "huggingface-hub",
    "opacus",
    "dp-accounting",
    "numpy",
    "Pillow",
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    core.require(path.is_file(), f"required gate artifact missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def nvidia_state() -> dict[str, Any]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
    core.require(len(output) == 1, "one NVIDIA GPU is required")
    name, driver, total, free, used, temperature, utilization = [
        item.strip() for item in output[0].split(",")
    ]
    return {
        "name": name,
        "driver": driver,
        "memory_total_mib": int(total),
        "memory_free_mib_at_freeze": int(free),
        "memory_used_mib_at_freeze": int(used),
        "temperature_c_at_freeze": int(temperature),
        "utilization_percent_at_freeze": int(utilization),
        "idle_detection": "WDDM-safe aggregate utilization and free-memory thresholds",
    }


def main() -> int:
    paths = core.default_paths()
    root = paths["root"]
    output = root / "_reports" / GATE_NAME
    core.require(not output.exists(), "refusing to overwrite full-runner gate")
    core.require(not paths["run_root"].exists(), "restricted full-run root already exists")
    core.require(not paths["report_root"].exists(), "public full-run root already exists")
    protocol = core.load_frozen_protocol(paths["protocol"])
    core.require(core.sha256_file(paths["manifest"]) == core.K5_SHA256, "K5 manifest hash")
    core.require(core.sha256_file(paths["preprocessing"]) == core.PREPROCESSING_SHA256, "preprocessing hash")
    snapshot, model_hashes = core.resolve_and_verify_snapshot(protocol)

    evidence_paths = {
        "feasibility_protocol": paths["protocol"],
        "generation_tasks": paths["protocol"].parent / "generation_tasks.csv",
        "real_reference": paths["protocol"].parent / "real_reference_local.csv",
        "evaluator_primary": root
        / "_reports"
        / "nih_cxr14_k5_evaluator_preflight_v1_001"
        / "public_report.json",
        "evaluator_independent": root
        / "_reports"
        / "nih_cxr14_k5_evaluator_preflight_gate_v1_001"
        / "independent_verification.json",
        "synthetic_resume_primary": root
        / "_reports"
        / "nih_cxr14_k5_encrypted_resume_preflight_v1_001"
        / "public_report.json",
        "synthetic_resume_independent": root
        / "_reports"
        / "nih_cxr14_k5_encrypted_resume_preflight_gate_v1_001"
        / "independent_verification.json",
        "actual_resume_protocol": root
        / "_reports"
        / "nih_cxr14_k5_actual_sd21_resume_preflight_protocol_v1_002"
        / "protocol.json",
        "actual_resume_result": root
        / "_reports"
        / "nih_cxr14_k5_actual_sd21_resume_preflight_v1_002"
        / "public_report.json",
    }
    evidence = {name: load_json(path) if path.suffix == ".json" else None for name, path in evidence_paths.items()}
    core.require(
        evidence["evaluator_primary"]["status"] == "PASS_K5_EVALUATOR_PREFLIGHT",
        "evaluator primary status",
    )
    core.require(
        evidence["evaluator_independent"]["status"] == "PASS_K5_EVALUATOR_PREFLIGHT_INDEPENDENT",
        "evaluator independent status",
    )
    core.require(
        evidence["synthetic_resume_primary"]["status"]
        == "PASS_EXACT_ENCRYPTED_RESUME_EQUIVALENCE",
        "synthetic resume status",
    )
    core.require(
        evidence["synthetic_resume_independent"]["status"]
        == "PASS_EXACT_ENCRYPTED_RESUME_INDEPENDENT",
        "synthetic resume independent status",
    )
    actual = evidence["actual_resume_result"]
    core.require(
        actual["status"] == "PASS_ACTUAL_SD21_EXACT_ENCRYPTED_RESUME_ALL_FOUR_ARMS"
        and actual["all_exact_checks_pass"],
        "actual SD2.1 resume result",
    )
    core.require(actual["artifact_boundary"]["full_K5_optimizer_execution_started"] is False, "actual preflight boundary")

    runner_source_hashes = {
        relative: core.sha256_file(root / Path(relative)) for relative in RUNNER_SOURCE_FILES
    }
    audit_source_hashes = {
        relative: core.sha256_file(root / Path(relative)) for relative in AUDIT_SOURCE_FILES
    }
    core.require(
        all(actual["source_hashes"].get(relative) == digest for relative, digest in runner_source_hashes.items()),
        "runner source changed after actual-SD2.1 preflight",
    )
    core.require(
        evidence["actual_resume_protocol"]["source_hashes"] == actual["source_hashes"],
        "actual resume protocol/result source mismatch",
    )

    regression = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "dp_training", "-p", "test_*.py", "-v"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    regression_text = regression.stdout + regression.stderr
    core.require(regression.returncode == 0 and "Ran 32 tests" in regression_text and "OK" in regression_text, "32-test regression suite")

    packages = {name: importlib.metadata.version(name) for name in PACKAGE_NAMES}
    gpu = nvidia_state()
    free_gib = core.require_free_space(root, 15.0)
    environment = {
        "schema": ENV_SCHEMA,
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": core.RUN_ID,
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "executable_sha256": core.sha256_file(Path(sys.executable).resolve()),
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "packages": packages,
        "torch": {
            "version": torch.__version__,
            "cuda": str(torch.version.cuda),
            "cudnn": int(torch.backends.cudnn.version()),
        },
        "cuda_device": gpu,
        "determinism": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "matmul_TF32": False,
            "cudnn_TF32": False,
            "deterministic_algorithms": True,
        },
        "runner_source_hashes": runner_source_hashes,
        "audit_source_hashes": audit_source_hashes,
        "frozen_inputs": {
            "protocol_sha256": core.PROTOCOL_SHA256,
            "k5_manifest_sha256": core.K5_SHA256,
            "preprocessing_sha256": core.PREPROCESSING_SHA256,
            "generation_tasks_sha256": core.sha256_file(evidence_paths["generation_tasks"]),
            "real_reference_sha256": core.sha256_file(evidence_paths["real_reference"]),
            "model_snapshot": str(snapshot),
            "model_id": core.MODEL_ID,
            "model_revision": core.MODEL_REVISION,
            "critical_model_hashes": model_hashes,
        },
        "preflight_evidence_sha256": {
            name: core.sha256_file(path) for name, path in evidence_paths.items()
        },
        "regression": {
            "tests": 32,
            "status": "PASS",
            "command": f'"{sys.executable}" -m unittest discover -s dp_training -p test_*.py -v',
            "combined_output_sha256": core.sha256_bytes(regression_text.encode("utf-8")),
        },
        "resource_snapshot": {
            "free_space_gib": free_gib,
            "minimum_before_each_arm_gib": 15.0,
            "full_runner_roots_absent": True,
        },
    }
    output.mkdir(parents=True)
    environment_path = output / "environment_manifest.json"
    write_json(environment_path, environment)
    environment_sha = core.sha256_file(environment_path)
    primary = {
        "schema": PRIMARY_SCHEMA,
        "status": "FROZEN_PENDING_INDEPENDENT_FULL_RUNNER_VERIFICATION",
        "run_id": core.RUN_ID,
        "scope": "SOURCE_ENVIRONMENT_LAUNCH_FREEZE_NO_B0_NO_FULL_OPTIMIZER_EXECUTION",
        "protocol_sha256": core.PROTOCOL_SHA256,
        "source_hashes": runner_source_hashes,
        "audit_source_hashes": audit_source_hashes,
        "environment_manifest_file": environment_path.name,
        "environment_manifest_sha256": environment_sha,
        "preflight_evidence_sha256": environment["preflight_evidence_sha256"],
        "actual_SD21_resume": {
            "status": actual["status"],
            "all_four_arms_exact": actual["all_exact_checks_pass"],
            "report_sha256": core.sha256_file(evidence_paths["actual_resume_result"]),
            "full_elapsed_seconds": actual["full_elapsed_seconds"],
            "minimum_step2_ciphertext_bytes": min(
                value["step2_ciphertext_bytes"] for value in actual["arms"].values()
            ),
        },
        "scale_rationale": {
            "images": 18_393,
            "patients": 8_476,
            "steps_per_arm": 4_000,
            "M0_and_M1_expected_image_exposures": 32_000,
            "M0_and_M1_expected_exposures_per_image": 32_000 / 18_393,
            "M2_expected_patient_unit_exposures": 16_000,
            "M2_expected_exposures_per_patient": 16_000 / 8_476,
            "interpretation": "reasonable one-seed rank-8 LoRA feasibility budget; not confirmatory superiority evidence",
        },
        "launch_contract": {
            "default_command": "validate (read-only)",
            "B0_before_initialize": True,
            "explicit_flags": ["--start-full-run", "--acknowledge-gate-sha256 EXACT_AUTHORITY_SHA256"],
            "arm_order": list(core.ARMS),
            "one_arm_per_invocation": True,
            "resume_interval_steps": core.RESUME_INTERVAL,
            "adapter_snapshot_steps": list(core.ADAPTER_STEPS),
            "commands": {
                "validate": f'"{sys.executable}" -m dp_training.run_k5_full_training validate',
                "initialize_after_B0_and_report": f'"{sys.executable}" -m dp_training.run_k5_full_training initialize --start-full-run --acknowledge-gate-sha256 EXACT_AUTHORITY_SHA256',
                "run_one_arm": f'"{sys.executable}" -m dp_training.run_k5_full_training run-arm --arm ARM --start-full-run --acknowledge-gate-sha256 EXACT_AUTHORITY_SHA256',
            },
        },
        "fail_closed": {
            "minimum_free_space_gib_before_arm": 15,
            "WDDM_idle_gate": "ABORT if aggregate free memory is below 6500 MiB or utilization exceeds 10 percent",
            "GPU_temperature_at_or_above_80C": "ABORT",
            "hash_environment_or_protocol_drift": "ABORT",
            "corrupt_or_missing_required_DPAPI_state": "ABORT",
            "nonfinite_state_or_resource_realization_limit": "ABORT_WITHOUT_RESAMPLING",
        },
        "measured_planning_envelope": protocol["resource_and_failure_policy"]["planning_envelope"],
        "full_K5_optimizer_execution_started": False,
        "full_run_roots_created": False,
        "next": "independent verification must create the launch authority; then report this gate and generate/hash B0 before any M0 update",
    }
    primary_path = output / "primary_gate.json"
    write_json(primary_path, primary)
    print(
        json.dumps(
            {
                "status": primary["status"],
                "primary_gate": str(primary_path),
                "primary_gate_sha256": core.sha256_file(primary_path),
                "environment_sha256": environment_sha,
                "full_K5_optimizer_execution_started": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
