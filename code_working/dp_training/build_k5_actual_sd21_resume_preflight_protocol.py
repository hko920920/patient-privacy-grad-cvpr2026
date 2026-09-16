#!/usr/bin/env python3
"""Freeze the real-SD2.1 exact-resume preflight before observing its output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dp_training import k5_full_runner as core


SCHEMA = "nih-cxr14-k5-actual-sd21-resume-preflight-protocol/v1"
OUTPUT_NAME = "nih_cxr14_k5_actual_sd21_resume_preflight_protocol_v1_002"
RESULT_NAME = "nih_cxr14_k5_actual_sd21_resume_preflight_v1_002"
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


def main() -> int:
    paths = core.default_paths()
    root = paths["root"]
    output = root / "_reports" / OUTPUT_NAME
    result = root / "_reports" / RESULT_NAME
    core.require(not output.exists(), "refusing to overwrite actual-SD2.1 preflight protocol")
    core.require(not result.exists(), "actual-SD2.1 result already exists")
    core.require(not paths["run_root"].exists(), "full restricted K5 run root already exists")
    core.require(not paths["report_root"].exists(), "full K5 report root already exists")
    protocol = core.load_frozen_protocol(paths["protocol"])
    core.require(core.sha256_file(paths["manifest"]) == core.K5_SHA256, "K5 manifest hash")
    core.require(core.sha256_file(paths["preprocessing"]) == core.PREPROCESSING_SHA256, "preprocessing hash")
    source_hashes = {
        relative: core.sha256_file(root / Path(relative)) for relative in SOURCE_FILES
    }
    value = {
        "schema": SCHEMA,
        "status": "FROZEN_BEFORE_ACTUAL_SD21_RESUME_EXECUTION",
        "scope": "BOUNDED_PUBLIC_DEVELOPMENT_CHECKPOINT_EQUIVALENCE_NOT_K5_TRAINING_NOT_PRIVACY_NOT_UTILITY",
        "upstream_protocol_sha256": core.PROTOCOL_SHA256,
        "source_hashes": source_hashes,
        "model": {
            "id": core.MODEL_ID,
            "revision": core.MODEL_REVISION,
            "variant": core.MODEL_VARIANT,
            "profile": "P256",
            "LoRA_rank": core.LORA_RANK,
            "LoRA_targets": list(core.LORA_TARGETS),
            "trainable_parameters": core.TRAINABLE_PARAMETERS,
            "base_dtype": "float16",
            "trainable_dtype": "float32",
        },
        "fixture": {
            "source_manifest_sha256": core.K5_SHA256,
            "partition": "public_development",
            "selection": "lexicographically first eight patients having at least two records; within patient sort by cap_rank then image_id and take two",
            "patients": 8,
            "images_per_patient": 2,
            "images": 16,
            "private_train_allowed": False,
            "persistent_image_or_latent_cache": False,
        },
        "comparison": {
            "arms": list(core.ARMS),
            "steps": 4,
            "uninterrupted": "fresh identical LoRA then four optimizer steps",
            "resumed": "fresh identical LoRA, encrypted step-0 save, two steps, atomic encrypted step-2 rotation, destroy state, decrypt/restore into a fresh real UNet, two steps",
            "exact_fields": [
                "all named LoRA tensors",
                "complete AdamW state",
                "schedule RNG end states",
                "arm-specific DP Gaussian RNG state where applicable",
                "global CPU and all CUDA RNG states",
                "public trace head and event count",
                "schedule commitment, root, committed step, and frozen digests",
            ],
            "M1_control": "M1-I8 and M1-G8 must have one identical keyed schedule commitment",
            "M2_boundary": "two same-patient images are averaged in one loss before one outer patient gradient clip",
            "M0_boundary": "eight-image mean-loss batch with neither clipping nor DP noise",
        },
        "DPAPI": {
            "scope": "Windows CurrentUser",
            "plaintext_file": "PROHIBITED",
            "rotation": "encrypted .new + fsync + decrypt/hash/schema verification + current/previous os.replace",
            "expected_real_state": "1,659,904 fp32 LoRA values plus complete AdamW first/second moments after step 2",
        },
        "pass_rule": "every exact field passes for all four arms, adapters change and stay finite, initial LoRA matches across arms, M1 commitment matches, ciphertext-only rotation passes, and full run roots remain absent",
        "failure_rule": "retain a public failure description, do not weaken equality, and block source/environment freeze until a separately recorded correction",
        "artifact_boundary": {
            "preflight_envelopes": "temporary and deleted",
            "adapter_optimizer_or_generated_image": "not retained",
            "result_directory": f"code_working/_reports/{RESULT_NAME}",
            "full_run_roots": "must remain absent",
        },
        "revision_note": "v1_002 refreezes the unchanged optimizer/resume core together with a WDDM-safe aggregate GPU-idle parser after the v1_001 full-gate verifier observed [N/A] per-process memory fields",
        "main_protocol_matrix": protocol["matrix"],
    }
    output.mkdir(parents=True)
    path = output / "protocol.json"
    write_json(path, value)
    build = {
        "schema": "nih-cxr14-k5-actual-sd21-resume-preflight-protocol-build/v1",
        "status": value["status"],
        "protocol_sha256": core.sha256_file(path),
        "full_K5_optimizer_execution_started": False,
    }
    write_json(output / "build_report.json", build)
    print(json.dumps(build, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
