#!/usr/bin/env python3
"""Run the frozen public-development RAD-DINO/BioViL-T evaluator preflight."""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from dp_training.k5_evaluator import (
    aggregate_by_label,
    effective_rank,
    load_real_p256_grayscale,
    require,
    sha256_file,
    stratified_bootstrap_mean,
    tree_sha256,
)


SCHEMA = "nih-cxr14-k5-evaluator-preflight-result/v1"
PROTOCOL_SHA256 = "C1283BD37077258F1997BE60FDEE3F49F8C782B0AF3110FFA226EACF11D8F2C9"
TASKS_SHA256 = "F82C4513EB64FAC2C17B31686BA18EAC4650C7A2EAA5871BE3C5D0C7CC3889FA"
RAD_REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"
BIO_REVISION = "692f09e9be1bfe5fdd5f3efdd0e1eca7d2c10b23"
RAD_WEIGHT_SHA256 = "DBFB9F54459C38773505DE64A6AB7807BDCB392610FE1E697166342E43FB91AE"
BIO_TEXT_SHA256 = "70BED344872E0A4F4DCA2352564C44F44CB7A9CFCAC3D878BE2DD08A5FFF9ABA"
BIO_IMAGE_SHA256 = "B2399D73DC2A68B9F3A1950E864AE0ECD24093FB07AA459D7E65807EBDC0FB77"
RAD_CONFIG_SHA256 = "89DAF9751D9576D586DEDF9543C1083211611FA3A36908DB7A799B3CE7C68EDE"
RAD_PROCESSOR_SHA256 = "C537FC995C30E2353F07253899618D60E9EAE3D5F82473778602C007C6523B56"
BIO_CONFIG_SHA256 = "58A8EAD5BDBE8B71E396CD094746779DC929954841282F94B6C100F0FE009A88"
BIO_CONFIGURATION_CODE_SHA256 = "857366A973291D09EA8E19CFE749FC801E81FAE9AE9CE0C9B462C7C26CB67E1F"
BIO_MODELING_CODE_SHA256 = "88F0F0D1FCF6B5C91FD3BEBEEC0C3BFEE271103DB7C13C5229E3F4807CB17AC2"
EXPECTED_VERSIONS = {
    "torch": "2.6.0+cu124",
    "torchvision": "0.21.0",
    "transformers": "4.56.2",
    "hi-ml-multimodal": "0.2.2",
    "numpy": "1.26.4",
    "scipy": "1.16.1",
    "scikit-learn": "1.3.0",
    "Pillow": "11.3.0",
}
PROMPT_ORDER = [
    "no_finding",
    "pneumothorax",
    "pneumonia",
    "consolidation",
    "pleural_effusion",
    "mass_opacity",
    "nodule_opacity",
]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def batches(values: list[dict[str, str]], size: int) -> Iterable[list[dict[str, str]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def load_tasks(path: Path) -> list[dict[str, str]]:
    require(sha256_file(path) == TASKS_SHA256, "evaluator tasks hash drift")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_fields = [
            "eval_index",
            "reference_stratum",
            "image_id",
            "patient_id",
            "image_filename",
            "selection_key_sha256",
            "matched_condition",
            "matched_prompt",
            "deranged_donor_patient_id",
            "deranged_condition",
            "deranged_prompt",
        ]
        require(reader.fieldnames == expected_fields, "evaluator task schema drift")
        rows = list(reader)
    require(len(rows) == 448, "evaluator task count")
    require([int(row["eval_index"]) for row in rows] == list(range(448)), "evaluator task order")
    require(len({row["image_id"] for row in rows}) == 448, "image uniqueness")
    require(len({row["patient_id"] for row in rows}) == 448, "patient uniqueness")
    require(
        all(row["matched_condition"] != row["deranged_condition"] for row in rows),
        "derangement condition equality",
    )
    require(
        all(row["patient_id"] != row["deranged_donor_patient_id"] for row in rows),
        "derangement patient equality",
    )
    return rows


def configure_torch() -> Any:
    import torch

    require(torch.cuda.is_available(), "CUDA is required")
    require("RTX 3070" in torch.cuda.get_device_name(0), "target GPU mismatch")
    torch.manual_seed(260_903)
    torch.cuda.manual_seed_all(260_903)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def run_rad_dino(
    tasks: list[dict[str, str]], image_root: Path, snapshot: Path, torch: Any
) -> tuple[np.ndarray, dict[str, Any]]:
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True, use_fast=False)
    model = AutoModel.from_pretrained(snapshot, local_files_only=True, use_safetensors=True).eval().cuda()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for group in batches(tasks, 2):
            images = [load_real_p256_grayscale(image_root / row["image_filename"]).convert("RGB") for row in group]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {name: value.cuda(non_blocking=False) for name, value in inputs.items()}
            output = model(**inputs).pooler_output.detach().float().cpu().numpy()
            chunks.append(output)
    features = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = int(torch.cuda.max_memory_allocated())
    require(features.shape == (448, 768), "RAD-DINO output shape")
    require(bool(np.isfinite(features).all()), "RAD-DINO non-finite feature")
    norms = np.linalg.norm(features.astype(np.float64), axis=1)
    require(bool(np.all(norms > 0.0)), "RAD-DINO zero row")

    replay_images = [
        load_real_p256_grayscale(image_root / row["image_filename"]).convert("RGB") for row in tasks[:8]
    ]
    replay_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start_index in range(0, 8, 2):
            inputs = processor(images=replay_images[start_index : start_index + 2], return_tensors="pt")
            inputs = {name: value.cuda(non_blocking=False) for name, value in inputs.items()}
            replay_chunks.append(model(**inputs).pooler_output.detach().float().cpu().numpy())
    replay = np.concatenate(replay_chunks, axis=0)
    replay_max_abs = float(np.max(np.abs(replay - features[:8])))
    rank = effective_rank(features)
    require(replay_max_abs <= 1e-5, "RAD-DINO replay drift")
    require(rank > 2.0, "RAD-DINO effective-rank gate")
    report = {
        "shape": list(features.shape),
        "finite": True,
        "row_norm_min": float(norms.min()),
        "row_norm_max": float(norms.max()),
        "total_feature_variance": float(np.var(features.astype(np.float64), axis=0, ddof=1).sum()),
        "effective_rank": rank,
        "first_eight_replay_max_abs": replay_max_abs,
        "seconds": elapsed,
        "peak_cuda_bytes": peak,
        "status": "PASS",
    }
    del model, processor, replay, replay_chunks, chunks
    gc.collect()
    torch.cuda.empty_cache()
    return features, report


def run_biovil_image(
    tasks: list[dict[str, str]], image_root: Path, weights: Path, torch: Any
) -> tuple[np.ndarray, dict[str, Any]]:
    from health_multimodal.image import ImageEncoderType, ImageModel
    from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference

    transform = create_chest_xray_transform_for_inference(resize=512, center_crop_size=448)
    model = ImageModel(
        img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE,
        joint_feature_size=128,
        pretrained_model_path=weights,
    ).eval().cuda()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for group in batches(tasks, 8):
            tensors = [transform(load_real_p256_grayscale(image_root / row["image_filename"])) for row in group]
            batch = torch.stack(tensors, dim=0).cuda(non_blocking=False)
            output = model(batch).projected_global_embedding
            output = torch.nn.functional.normalize(output, dim=-1)
            chunks.append(output.detach().float().cpu().numpy())
    features = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = int(torch.cuda.max_memory_allocated())
    require(features.shape == (448, 128), "BioViL-T image output shape")
    require(bool(np.isfinite(features).all()), "BioViL-T image non-finite")
    norms = np.linalg.norm(features.astype(np.float64), axis=1)
    require(float(np.max(np.abs(norms - 1.0))) <= 1e-5, "BioViL-T image norm")

    replay_tensors = [
        transform(load_real_p256_grayscale(image_root / row["image_filename"])) for row in tasks[:8]
    ]
    with torch.inference_mode():
        batch = torch.stack(replay_tensors, dim=0).cuda(non_blocking=False)
        replay_output = model(batch).projected_global_embedding
        replay = torch.nn.functional.normalize(replay_output, dim=-1).detach().float().cpu().numpy()
    replay_max_abs = float(np.max(np.abs(replay - features[:8])))
    require(replay_max_abs <= 1e-5, "BioViL-T image replay drift")
    report = {
        "shape": list(features.shape),
        "finite": True,
        "unit_norm_max_abs_error": float(np.max(np.abs(norms - 1.0))),
        "first_eight_replay_max_abs": replay_max_abs,
        "seconds": elapsed,
        "peak_cuda_bytes": peak,
        "status": "PASS",
    }
    del model, transform, replay, replay_output, replay_tensors, chunks
    gc.collect()
    torch.cuda.empty_cache()
    return features, report


def run_biovil_text(tasks: list[dict[str, str]], snapshot: Path, torch: Any) -> tuple[np.ndarray, dict[str, Any]]:
    from transformers import AutoModel, AutoTokenizer

    prompt_by_condition: dict[str, str] = {}
    for row in tasks:
        existing = prompt_by_condition.setdefault(row["matched_condition"], row["matched_prompt"])
        require(existing == row["matched_prompt"], "matched prompt drift")
        existing_deranged = prompt_by_condition.setdefault(row["deranged_condition"], row["deranged_prompt"])
        require(existing_deranged == row["deranged_prompt"], "deranged prompt drift")
    require(set(prompt_by_condition) == set(PROMPT_ORDER), "prompt condition set")
    prompts = [prompt_by_condition[condition] for condition in PROMPT_ORDER]
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, use_fast=False)
    model, loading_info = AutoModel.from_pretrained(
        snapshot,
        trust_remote_code=True,
        local_files_only=True,
        use_safetensors=True,
        output_loading_info=True,
    )
    model = model.eval().cuda()
    tokenized = tokenizer(prompts, padding=True, return_tensors="pt")
    start = time.perf_counter()
    with torch.inference_mode():
        embeddings = model.get_projected_text_embeddings(
            tokenized.input_ids.cuda(non_blocking=False),
            tokenized.attention_mask.cuda(non_blocking=False),
        ).detach().float().cpu().numpy()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    require(embeddings.shape == (7, 128), "BioViL-T text output shape")
    require(bool(np.isfinite(embeddings).all()), "BioViL-T text non-finite")
    norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
    require(float(np.max(np.abs(norms - 1.0))) <= 1e-5, "BioViL-T text norm")
    unused = sorted(loading_info.get("unexpected_keys", []))
    require(
        unused == ["bert.pooler.dense.bias", "bert.pooler.dense.weight"],
        f"unexpected BioViL-T load boundary: {unused}",
    )
    require(not loading_info.get("missing_keys"), "BioViL-T missing text weights")
    report = {
        "shape": list(embeddings.shape),
        "finite": True,
        "unit_norm_max_abs_error": float(np.max(np.abs(norms - 1.0))),
        "unused_checkpoint_keys": unused,
        "missing_checkpoint_keys": [],
        "seconds": elapsed,
        "status": "PASS",
    }
    del model, tokenizer, tokenized
    gc.collect()
    torch.cuda.empty_cache()
    return embeddings, report


def scan_public_report(value: Any) -> None:
    forbidden_keys = {
        "patient_id",
        "image_id",
        "image_filename",
        "selection_key_sha256",
        "deranged_donor_patient_id",
        "embedding",
        "per_image_score",
    }

    def walk(item: Any) -> None:
        if isinstance(item, dict):
            require(not (set(item) & forbidden_keys), "public report forbidden key")
            for child in item.values():
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)
        elif isinstance(item, str):
            require(".png" not in item.lower(), "public report filename leak")

    walk(value)


def main() -> int:
    total_start = time.perf_counter()
    root = Path(__file__).resolve().parent.parent
    protocol_dir = root / "_reports" / "nih_cxr14_k5_evaluator_preflight_protocol_v1_001"
    output_dir = root / "_reports" / "nih_cxr14_k5_evaluator_preflight_v1_001"
    require(not output_dir.exists(), "refusing to overwrite evaluator preflight result")
    protocol_path = protocol_dir / "protocol.json"
    task_path = protocol_dir / "real_evaluator_tasks_local.csv"
    require(sha256_file(protocol_path) == PROTOCOL_SHA256, "evaluator protocol hash drift")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    require(protocol["status"] == "FROZEN_BEFORE_REAL_ENCODER_OUTPUT", "protocol status")
    tasks = load_tasks(task_path)
    image_root = root / "_data" / "raw" / "nih_cxr14_pa_k10_plus_census_v1" / "images"
    rad_snapshot = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--microsoft--rad-dino"
        / "snapshots"
        / RAD_REVISION
    )
    bio_snapshot = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--microsoft--BiomedVLP-BioViL-T"
        / "snapshots"
        / BIO_REVISION
    )
    exact_files = {
        "rad_weights": (rad_snapshot / "model.safetensors", RAD_WEIGHT_SHA256),
        "rad_config": (rad_snapshot / "config.json", RAD_CONFIG_SHA256),
        "rad_processor": (rad_snapshot / "preprocessor_config.json", RAD_PROCESSOR_SHA256),
        "biovil_text_weights": (bio_snapshot / "model.safetensors", BIO_TEXT_SHA256),
        "biovil_image_weights": (
            bio_snapshot / "biovil_t_image_model_proj_size_128.pt",
            BIO_IMAGE_SHA256,
        ),
        "biovil_config": (bio_snapshot / "config.json", BIO_CONFIG_SHA256),
        "biovil_configuration_code": (
            bio_snapshot / "configuration_cxrbert.py",
            BIO_CONFIGURATION_CODE_SHA256,
        ),
        "biovil_modeling_code": (bio_snapshot / "modeling_cxrbert.py", BIO_MODELING_CODE_SHA256),
    }
    for name, (path, digest) in exact_files.items():
        require(path.is_file() and sha256_file(path) == digest, f"exact encoder file drift: {name}")
    versions = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    require(versions == EXPECTED_VERSIONS, f"dependency drift: {versions}")
    require(all((image_root / row["image_filename"]).is_file() for row in tasks), "missing real input")

    torch = configure_torch()
    rad_features, rad_report = run_rad_dino(tasks, image_root, rad_snapshot, torch)
    biovil_image, biovil_image_report = run_biovil_image(
        tasks, image_root, bio_snapshot / "biovil_t_image_model_proj_size_128.pt", torch
    )
    biovil_text, biovil_text_report = run_biovil_text(tasks, bio_snapshot, torch)

    prompt_index = {condition: index for index, condition in enumerate(PROMPT_ORDER)}
    similarities = biovil_image.astype(np.float64) @ biovil_text.astype(np.float64).T
    matched = np.array(
        [similarities[index, prompt_index[row["matched_condition"]]] for index, row in enumerate(tasks)],
        dtype=np.float64,
    )
    deranged = np.array(
        [similarities[index, prompt_index[row["deranged_condition"]]] for index, row in enumerate(tasks)],
        dtype=np.float64,
    )
    difference = matched - deranged
    strata = [row["reference_stratum"] for row in tasks]
    conditions = [row["matched_condition"] for row in tasks]
    bootstrap = stratified_bootstrap_mean(
        difference,
        strata,
        seed=int(protocol["bootstrap"]["seed"]),
        replicates=int(protocol["bootstrap"]["replicates"]),
    )
    alignment_pass = (
        bootstrap["mean_difference"] > 0.0 and bootstrap["lower_95_percentile"] > 0.0
    )
    alignment_report = {
        "matched_cosine": {
            "mean": float(matched.mean()),
            "standard_deviation": float(matched.std(ddof=1)),
        },
        "deranged_cosine": {
            "mean": float(deranged.mean()),
            "standard_deviation": float(deranged.std(ddof=1)),
        },
        "matched_minus_deranged": bootstrap,
        "difference_by_matched_condition": aggregate_by_label(difference, conditions),
        "difference_by_reference_stratum": aggregate_by_label(difference, strata),
        "decision": "PASS" if alignment_pass else "FAIL_BLOCK_FULL_K5",
    }

    import health_multimodal

    health_root = Path(health_multimodal.__file__).resolve().parent
    health_source_hash = tree_sha256(health_root.rglob("*.py"), health_root)
    del rad_features, biovil_image, biovil_text, similarities, matched, deranged, difference
    gc.collect()
    torch.cuda.empty_cache()
    final_status = "PASS_K5_EVALUATOR_PREFLIGHT" if alignment_pass else "FAIL_K5_EVALUATOR_PREFLIGHT_BLOCK_FULL_K5"
    report = {
        "schema": SCHEMA,
        "status": final_status,
        "scope": protocol["scope"],
        "source_hashes": {
            "protocol": PROTOCOL_SHA256,
            "local_tasks": TASKS_SHA256,
            "runner": sha256_file(Path(__file__).resolve()),
            "metric_core": sha256_file(root / "dp_training" / "k5_evaluator.py"),
            "health_multimodal_python_tree": health_source_hash,
            **{name: digest for name, (_, digest) in exact_files.items()},
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": versions,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "input": {
            "images_decoded": 448,
            "unique_patients": 448,
            "matched_condition_counts": dict(Counter(conditions)),
            "local_identifiers_in_public_report": False,
        },
        "RAD_DINO": rad_report,
        "BioViL_T_image": biovil_image_report,
        "BioViL_T_text": biovil_text_report,
        "BioViL_T_real_alignment_validity": alignment_report,
        "artifact_policy": {
            "per_image_features_persisted": False,
            "per_image_scores_persisted": False,
            "resized_or_raw_images_persisted": False,
            "model_or_optimizer_created": False,
            "generated_images_created": False,
        },
        "wall_time_seconds": time.perf_counter() - total_start,
        "full_K5_optimizer_execution_started": False,
        "next": (
            "independently verify this evaluator result, then implement exact encrypted restart equivalence; "
            "do not start the 4000-step K5 matrix"
            if alignment_pass
            else "retain the failed validity result and block full K5 pending a separately frozen evaluator redesign"
        ),
    }
    scan_public_report(report)
    output_dir.mkdir(parents=True)
    report_path = output_dir / "public_report.json"
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"public_report_sha256={sha256_file(report_path)}")
    return 0 if alignment_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
