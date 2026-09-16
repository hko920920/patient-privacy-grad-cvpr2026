#!/usr/bin/env python3
"""Freeze the B0 generation contract before creating any B0 image."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dp_training import k5_generation as core


OUTPUT_NAME = "nih_cxr14_k5_b0_generation_protocol_v1_002"
BASE_SNAPSHOT_MANIFEST_SHA256 = "E38A5F8FE1695745EDD60FB6E2EEDFD35A01A65B5E981ACA28EEF026F4C02638"
BASE_GATE_RESULT_SHA256 = "0B9BF8031D0C182D4E695EA8995D6E1F27A1F6B509B38E99A5E25DEBA21EEB35"
GENERATOR_SOURCE_FILES = (
    "dp_training/k5_generation.py",
    "dp_training/run_k5_b0_generation.py",
)
AUDIT_SOURCE_FILES = (
    "dp_training/build_k5_b0_generation_protocol.py",
    "dp_training/test_k5_generation.py",
    "dp_training/verify_k5_b0_generation_independent.py",
)


def load_json(path: Path) -> dict[str, Any]:
    core.require(path.is_file(), f"missing required JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_exclusive(path: Path, value: Any) -> None:
    core.require(not path.exists(), f"refusing to overwrite: {path}")
    temporary = path.with_name(path.name + ".new")
    core.require(not temporary.exists(), f"stale protocol temporary requires audit: {temporary}")
    body = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    load_json(temporary)
    os.replace(temporary, path)


def main() -> int:
    paths = core.default_paths()
    root = paths["root"]
    output = root / "_reports" / OUTPUT_NAME
    core.require(not output.exists(), "refusing to overwrite B0 generation protocol")
    core.require(not paths["staging_root"].exists(), "B0 staging exists before protocol freeze")
    core.require(not paths["full_run_root"].exists(), "full K5 run exists before B0 protocol freeze")
    core.require(not paths["public_report_root"].exists(), "public K5 run root exists before B0 protocol freeze")

    upstream_paths = {
        "feasibility_protocol": paths["feasibility_protocol"],
        "generation_tasks": paths["generation_tasks"],
        "primary_gate": paths["primary_gate"],
        "launch_authority": paths["launch_authority"],
        "environment_manifest": paths["environment_manifest"],
        "base_snapshot_manifest": paths["base_snapshot_manifest"],
        "base_gate_result": paths["base_gate_result"],
    }
    upstream = {name: core.sha256_file(path) for name, path in upstream_paths.items()}
    expected_upstream = {
        "feasibility_protocol": core.FEASIBILITY_PROTOCOL_SHA256,
        "generation_tasks": core.GENERATION_TASKS_SHA256,
        "primary_gate": core.PRIMARY_GATE_SHA256,
        "launch_authority": core.LAUNCH_AUTHORITY_SHA256,
        "environment_manifest": core.ENVIRONMENT_MANIFEST_SHA256,
        "base_snapshot_manifest": BASE_SNAPSHOT_MANIFEST_SHA256,
        "base_gate_result": BASE_GATE_RESULT_SHA256,
    }
    core.require(upstream == expected_upstream, "upstream evidence changed before B0 freeze")

    tasks = core.load_and_validate_tasks(paths["generation_tasks"])
    main_protocol = load_json(paths["feasibility_protocol"])
    task_contract = main_protocol["evaluation"]["generation_tasks"]
    core.require(task_contract["images_per_model"] == 448, "main protocol B0 count")
    core.require(task_contract["sampler"] == "DDIM", "main protocol sampler")
    core.require(task_contract["inference_steps"] == 50, "main protocol steps")
    core.require(task_contract["guidance_scale"] == 7.5, "main protocol guidance")
    core.require(task_contract["negative_prompt"] is None, "main protocol negative prompt")
    core.require(task_contract["batch_size"] == 4, "main protocol batch")
    core.require(task_contract["B0_timing"] == "generate and hash before any full K5 private optimizer update", "main protocol B0 timing")

    launch = load_json(paths["launch_authority"])
    core.require(launch["status"] == "PASS_K5_FULL_RUNNER_GATE_NO_FULL_TRAINING_STARTED", "launch authority status")
    core.require(launch["full_K5_optimizer_execution_started"] is False, "launch optimizer boundary")
    environment = load_json(paths["environment_manifest"])
    snapshot = Path(environment["frozen_inputs"]["model_snapshot"]).resolve()
    core.require(snapshot.is_dir() and snapshot.name == core.MODEL_REVISION, "model snapshot")
    for relative, check in environment["frozen_inputs"]["critical_model_hashes"].items():
        core.require(check["actual"] == check["expected"] == core.sha256_file(snapshot / relative), f"model hash: {relative}")

    generator_hashes = {
        relative: core.sha256_file(root / Path(relative)) for relative in GENERATOR_SOURCE_FILES
    }
    audit_hashes = {relative: core.sha256_file(root / Path(relative)) for relative in AUDIT_SOURCE_FILES}
    generation = {
        "sampler": "DDIM",
        "inference_steps": 50,
        "guidance_scale": 7.5,
        "negative_prompt": None,
        "batch_size": 4,
        "height": 256,
        "width": 256,
        "output_type": "np",
        "uint8_quantization": "clip(x * 255.0, 0.0, 255.0).astype(uint8); truncation toward zero",
        "png": {
            "format": "PNG",
            "mode": "RGB",
            "compress_level": 9,
            "optimize": False,
            "metadata": "none",
        },
    }
    protocol = {
        "schema": core.PROTOCOL_SCHEMA,
        "status": "FROZEN_BEFORE_ANY_B0_OUTPUT",
        "run_id": core.RUN_ID,
        "model": "B0",
        "role": "hash-pinned unadapted SD2.1 comparison baseline",
        "expected_images": len(tasks),
        "conditions": [condition for condition, _, _ in core.CONDITIONS],
        "images_per_condition": 64,
        "generation": generation,
        "pipeline_loading": {
            "class": "diffusers.StableDiffusionPipeline",
            "model_id": core.MODEL_ID,
            "revision": core.MODEL_REVISION,
            "variant": core.MODEL_VARIANT,
            "torch_dtype": "float16",
            "use_safetensors": True,
            "local_files_only": True,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "broken_inherited_onnxruntime_masked_from_diffusers_feature_detection": True,
        },
        "determinism": {
            "per_image_generator": "torch.Generator(device='cuda').manual_seed(task seed)",
            "generator_list_order": "the four consecutive CSV rows in each frozen batch",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "matmul_TF32": False,
            "cudnn_TF32": False,
        },
        "resumption": {
            "unit": "one frozen consecutive four-task batch",
            "prepare_before_publish": "record each encoded PNG SHA-256 and pixel statistics in atomic progress JSON before its final path appears",
            "committed_file_rule": "existing files are never overwritten and must match their prepared record exactly",
            "prepared_without_file_rule": "regenerate the entire same four-task batch and require the encoded SHA-256 to match before commit",
            "untracked_or_extra_file_rule": "hard fail for audit",
            "stale_dot_new_rule": "hard fail for audit",
            "selection_prohibited": "all 448 fixed rows must be retained; no reroll, rejection, replacement, or result-dependent seed change",
        },
        "pixel_sanity": {
            "grayscale_conversion": "PIL RGB.convert('L') 8-bit ITU-R 601 luma",
            "low_contrast_image": "population standard deviation of L/255 in float64 is < 0.02",
            "saturated_image": "more than 98% of L pixels are <=1 or >=254",
            "exact_duplicate_fraction": "(448 - number of unique encoded-PNG SHA-256 values) / 448",
            "required_decode_rate": 1.0,
            "maximum_exact_duplicate_fraction": 0.01,
            "maximum_low_contrast_fraction": 0.01,
            "maximum_saturation_fraction": 0.01,
            "failure_action": "retain evidence, publish no PASS manifest, report before any M0 decision",
        },
        "independent_verification": {
            "all_tasks_reconstructed_from_salt": True,
            "all_448_PNGs_redecoded_and_rehashed": True,
            "all_pixel_statistics_recomputed": True,
            "sentinel_method": "reload the pinned pipeline independently and regenerate the exact frozen batches",
            "sentinel_batch_indices": [0, 111],
            "sentinel_images": 8,
            "publication": "atomically rename B0.new only after independent report and full-runner-compatible manifest are complete",
        },
        "sentinel_batch_indices": [0, 111],
        "resource_gate": {
            "minimum_free_disk_gib": 15.0,
            "minimum_idle_gpu_memory_mib": 6500,
            "maximum_start_gpu_utilization_percent": 10,
            "maximum_start_temperature_c_exclusive": 80,
            "runtime_thermal_abort_c_exclusive": 88,
            "gpu_name": environment["cuda_device"]["name"],
            "driver": environment["cuda_device"]["driver"],
            "memory_total_mib": environment["cuda_device"]["memory_total_mib"],
        },
        "paths": {
            "restricted_B0_staging": "code_working/_restricted_generation/nih_cxr14_k5_feasibility_v1_001/B0",
            "full_private_run_root_must_remain_absent": "code_working/_restricted_runs/nih_cxr14_k5_feasibility_v1_001",
            "public_manifest_after_independent_PASS": "code_working/_reports/nih_cxr14_k5_feasibility_v1_001/B0/generation_manifest.json",
        },
        "boundary": {
            "B0_must_precede": "matrix initialization and every private optimizer update",
            "primary_generator_can_publish_PASS": False,
            "independent_verifier_can_publish_PASS_after_all_checks": True,
            "generated_images_release_eligible": False,
            "B0_does_not_establish": [
                "radiographic fidelity",
                "clinical utility",
                "privacy",
                "near-copy safety",
                "release eligibility",
            ],
        },
        "commands": {
            "read_only": "base_gate/.venv/Scripts/python.exe -m dp_training.run_k5_b0_generation validate",
            "independent_preflight": "base_gate/.venv/Scripts/python.exe -m dp_training.verify_k5_b0_generation_independent preflight",
            "generate": "base_gate/.venv/Scripts/python.exe -m dp_training.run_k5_b0_generation generate --start-b0 --acknowledge-protocol-sha256 EXACT_PROTOCOL_SHA256",
            "independent_final": "base_gate/.venv/Scripts/python.exe -m dp_training.verify_k5_b0_generation_independent final",
        },
        "upstream_sha256": upstream,
        "generator_source_sha256": generator_hashes,
        "audit_source_sha256": audit_hashes,
        "environment": {
            "python": environment["python"],
            "packages": environment["packages"],
            "critical_model_hashes": environment["frozen_inputs"]["critical_model_hashes"],
        },
    }

    output.mkdir(parents=True)
    protocol_path = output / "protocol.json"
    write_json_exclusive(protocol_path, protocol)
    report = {
        "schema": "nih-cxr14-k5-b0-generation-protocol-build/v1",
        "status": "FROZEN_BEFORE_ANY_B0_OUTPUT",
        "run_id": core.RUN_ID,
        "protocol_file": "protocol.json",
        "protocol_sha256": core.sha256_file(protocol_path),
        "expected_images": len(tasks),
        "generator_source_sha256": generator_hashes,
        "audit_source_sha256": audit_hashes,
        "B0_staging_created": False,
        "full_K5_run_initialized": False,
        "optimizer_steps_started": 0,
        "next": "independent preflight, then wait for an idle GPU before explicit B0 generation",
    }
    write_json_exclusive(output / "build_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
