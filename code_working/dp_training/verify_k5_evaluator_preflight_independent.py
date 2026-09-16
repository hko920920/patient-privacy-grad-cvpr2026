#!/usr/bin/env python3
"""Independent full recomputation of the K5 public-development evaluator preflight."""

from __future__ import annotations

import csv
import ctypes
import gc
import hashlib
import json
import math
import os
import platform
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


SCHEMA = "nih-cxr14-k5-evaluator-preflight-independent-verification/v1"
PROTOCOL_SHA256 = "C1283BD37077258F1997BE60FDEE3F49F8C782B0AF3110FFA226EACF11D8F2C9"
TASKS_SHA256 = "F82C4513EB64FAC2C17B31686BA18EAC4650C7A2EAA5871BE3C5D0C7CC3889FA"
REFERENCE_SHA256 = "99972FC6BAB402F620D2F6B142348CF6A3F16B60BFD63F41EB877FFDDAAA898E"
PUBLIC_REPORT_SHA256 = "914002B52C2EF48C551FFD9A5CD0260BA7708EA6492F5296AEF98BF6D0B4F5B5"
RUNNER_SHA256 = "1E68C93EBD6DE31B6805EA8F9B4FF7B24F28FAC847EFC872C5E674A8027DE63E"
CORE_SHA256 = "D95DEE5CBBB217F7847ED76DA26EEED68421E53E50139957903AA7A477F64EE7"
RAD_REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"
BIO_REVISION = "692f09e9be1bfe5fdd5f3efdd0e1eca7d2c10b23"
WEIGHT_HASHES = {
    "rad_weights": "DBFB9F54459C38773505DE64A6AB7807BDCB392610FE1E697166342E43FB91AE",
    "biovil_text_weights": "70BED344872E0A4F4DCA2352564C44F44CB7A9CFCAC3D878BE2DD08A5FFF9ABA",
    "biovil_image_weights": "B2399D73DC2A68B9F3A1950E864AE0ECD24093FB07AA459D7E65807EBDC0FB77",
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
PROMPTS = {
    "no_finding": "a frontal posteroanterior chest radiograph with no labeled finding",
    "pneumothorax": "a frontal posteroanterior chest radiograph with radiographic findings of pneumothorax",
    "pneumonia": "a frontal posteroanterior chest radiograph with radiographic findings of pneumonia",
    "consolidation": "a frontal posteroanterior chest radiograph with radiographic findings of consolidation",
    "pleural_effusion": "a frontal posteroanterior chest radiograph with radiographic findings of pleural effusion",
    "mass_opacity": "a frontal posteroanterior chest radiograph with radiographic findings of mass opacity",
    "nodule_opacity": "a frontal posteroanterior chest radiograph with radiographic findings of nodule opacity",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def condition(row: dict[str, str]) -> str:
    labels = set(row["finding_labels"].split("|"))
    if row["reference_stratum"] == "pneumonia_or_consolidation":
        return "pneumonia" if "Pneumonia" in labels else "consolidation"
    if row["reference_stratum"] == "mass_or_nodule":
        return "mass_opacity" if "Mass" in labels else "nodule_opacity"
    return row["reference_stratum"]


def reconstruct_tasks(reference_path: Path, task_path: Path) -> list[dict[str, str]]:
    with reference_path.open("r", encoding="utf-8", newline="") as handle:
        references = list(csv.DictReader(handle))
    enriched = [
        {**row, "matched_condition": condition(row), "matched_prompt": PROMPTS[condition(row)]}
        for row in references
    ]
    enriched.sort(key=lambda row: (row["matched_condition"], row["selection_key_sha256"], row["image_id"]))
    shift = max(Counter(row["matched_condition"] for row in enriched).values())
    require(shift == 89, "independent derangement shift")
    expected: list[dict[str, str]] = []
    for index, row in enumerate(enriched):
        donor = enriched[(index + shift) % len(enriched)]
        expected.append(
            {
                "eval_index": str(index),
                "reference_stratum": row["reference_stratum"],
                "image_id": row["image_id"],
                "patient_id": row["patient_id"],
                "image_filename": row["image_filename"],
                "selection_key_sha256": row["selection_key_sha256"],
                "matched_condition": row["matched_condition"],
                "matched_prompt": row["matched_prompt"],
                "deranged_donor_patient_id": donor["patient_id"],
                "deranged_condition": donor["matched_condition"],
                "deranged_prompt": donor["matched_prompt"],
            }
        )
    with task_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        actual = list(reader)
    require(reader.fieldnames == list(expected[0]), "task fields")
    require(actual == expected, "task reconstruction mismatch")
    require(all(row["patient_id"] != row["deranged_donor_patient_id"] for row in actual), "patient derangement")
    require(all(row["matched_condition"] != row["deranged_condition"] for row in actual), "condition derangement")
    return actual


def preprocess(path: Path) -> Image.Image:
    with Image.open(path) as source:
        require(source.size == (1024, 1024), "source geometry")
        if source.mode == "L":
            gray = source.copy()
        elif source.mode == "RGBA":
            array = np.asarray(source, dtype=np.uint8)
            require(array.shape == (1024, 1024, 4), "RGBA shape")
            require(bool(np.all(array[..., 0] == array[..., 1])), "RGBA R/G")
            require(bool(np.all(array[..., 0] == array[..., 2])), "RGBA R/B")
            require(bool(np.all(array[..., 3] == 255)), "RGBA alpha")
            gray = Image.fromarray(array[..., 0])
        else:
            raise RuntimeError(f"unexpected source mode: {source.mode}")
    return gray.resize((256, 256), resample=Image.Resampling.LANCZOS, reducing_gap=None)


def effective_rank(features: np.ndarray) -> float:
    covariance = np.cov(np.asarray(features, dtype=np.float64), rowvar=False, ddof=1)
    values = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    probabilities = values[values > 0] / values.sum()
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def rad_features(tasks: list[dict[str, str]], image_root: Path, snapshot: Path) -> np.ndarray:
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True, use_fast=False)
    model = AutoModel.from_pretrained(snapshot, local_files_only=True, use_safetensors=True).eval().cuda()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(tasks), 2):
            images = [
                preprocess(image_root / row["image_filename"]).convert("RGB")
                for row in tasks[start : start + 2]
            ]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {name: value.cuda() for name, value in inputs.items()}
            output.append(model(**inputs).pooler_output.detach().float().cpu().numpy())
    result = np.concatenate(output, axis=0)
    del model, processor, output
    gc.collect()
    torch.cuda.empty_cache()
    return result


def biovil_image_features(tasks: list[dict[str, str]], image_root: Path, weights: Path) -> np.ndarray:
    from health_multimodal.image import ImageEncoderType, ImageModel
    from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference

    transform = create_chest_xray_transform_for_inference(resize=512, center_crop_size=448)
    model = ImageModel(
        img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE,
        joint_feature_size=128,
        pretrained_model_path=weights,
    ).eval().cuda()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(tasks), 8):
            values = [
                transform(preprocess(image_root / row["image_filename"]))
                for row in tasks[start : start + 8]
            ]
            tensor = torch.stack(values).cuda()
            projected = model(tensor).projected_global_embedding
            projected = torch.nn.functional.normalize(projected, dim=-1)
            output.append(projected.detach().float().cpu().numpy())
    result = np.concatenate(output, axis=0)
    del model, transform, output
    gc.collect()
    torch.cuda.empty_cache()
    return result


def biovil_text_features(snapshot: Path) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, use_fast=False)
    model, loading = AutoModel.from_pretrained(
        snapshot,
        trust_remote_code=True,
        local_files_only=True,
        use_safetensors=True,
        output_loading_info=True,
    )
    require(not loading.get("missing_keys"), "text missing keys")
    require(
        sorted(loading.get("unexpected_keys", []))
        == ["bert.pooler.dense.bias", "bert.pooler.dense.weight"],
        "text unused-key boundary",
    )
    model = model.eval().cuda()
    tokens = tokenizer([PROMPTS[name] for name in PROMPT_ORDER], padding=True, return_tensors="pt")
    with torch.inference_mode():
        output = model.get_projected_text_embeddings(
            tokens.input_ids.cuda(), tokens.attention_mask.cuda()
        ).detach().float().cpu().numpy()
    del model, tokenizer, tokens
    gc.collect()
    torch.cuda.empty_cache()
    return output


def bootstrap(difference: np.ndarray, strata: list[str], seed: int) -> dict[str, float]:
    labels = np.asarray(strata, dtype=object)
    groups = [np.flatnonzero(labels == name) for name in sorted(set(strata))]
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty(2_000, dtype=np.float64)
    for replicate in range(2_000):
        sampled_parts = [rng.choice(group, size=len(group), replace=True) for group in groups]
        draws[replicate] = float(np.concatenate([difference[index] for index in sampled_parts]).mean())
    lower, upper = np.quantile(draws, [0.025, 0.975], method="linear")
    return {"mean": float(difference.mean()), "lower": float(lower), "upper": float(upper)}


def close(actual: float, expected: float) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=1e-8, abs_tol=1e-10)


def scan_report(report: Any, image_ids: set[str]) -> None:
    forbidden = {"patient_id", "image_id", "image_filename", "selection_key_sha256", "embedding"}
    if isinstance(report, dict):
        require(not (set(report) & forbidden), "public identifier key")
        for child in report.values():
            scan_report(child, image_ids)
    elif isinstance(report, list):
        for child in report:
            scan_report(child, image_ids)
    elif isinstance(report, str):
        require(report not in image_ids and ".png" not in report.lower(), "public image identifier")


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    protocol_dir = root / "_reports" / "nih_cxr14_k5_evaluator_preflight_protocol_v1_001"
    result_dir = root / "_reports" / "nih_cxr14_k5_evaluator_preflight_v1_001"
    gate_dir = root / "_reports" / "nih_cxr14_k5_evaluator_preflight_gate_v1_001"
    require(not gate_dir.exists(), "refusing to overwrite evaluator independent gate")
    paths = {
        "protocol": protocol_dir / "protocol.json",
        "tasks": protocol_dir / "real_evaluator_tasks_local.csv",
        "reference": root
        / "_reports"
        / "nih_cxr14_k5_feasibility_protocol_v1_001"
        / "real_reference_local.csv",
        "public_report": result_dir / "public_report.json",
        "runner": root / "dp_training" / "run_k5_evaluator_preflight.py",
        "core": root / "dp_training" / "k5_evaluator.py",
    }
    expected = {
        "protocol": PROTOCOL_SHA256,
        "tasks": TASKS_SHA256,
        "reference": REFERENCE_SHA256,
        "public_report": PUBLIC_REPORT_SHA256,
        "runner": RUNNER_SHA256,
        "core": CORE_SHA256,
    }
    for name, digest in expected.items():
        require(paths[name].is_file() and sha256_file(paths[name]) == digest, f"hash drift: {name}")
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    public = json.loads(paths["public_report"].read_text(encoding="utf-8"))
    require(public["status"] == "PASS_K5_EVALUATOR_PREFLIGHT", "public evaluator status")
    tasks = reconstruct_tasks(paths["reference"], paths["tasks"])
    scan_report(public, {row["image_id"] for row in tasks})
    image_root = root / "_data" / "raw" / "nih_cxr14_pa_k10_plus_census_v1" / "images"
    rad_snapshot = Path.home() / ".cache/huggingface/hub/models--microsoft--rad-dino/snapshots" / RAD_REVISION
    bio_snapshot = Path.home() / ".cache/huggingface/hub/models--microsoft--BiomedVLP-BioViL-T/snapshots" / BIO_REVISION
    weight_paths = {
        "rad_weights": rad_snapshot / "model.safetensors",
        "biovil_text_weights": bio_snapshot / "model.safetensors",
        "biovil_image_weights": bio_snapshot / "biovil_t_image_model_proj_size_128.pt",
    }
    for name, path in weight_paths.items():
        require(sha256_file(path) == WEIGHT_HASHES[name], f"weight drift: {name}")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rad = rad_features(tasks, image_root, rad_snapshot)
    rad_rank = effective_rank(rad)
    require(rad.shape == (448, 768) and bool(np.isfinite(rad).all()), "independent RAD-DINO features")
    require(close(rad_rank, public["RAD_DINO"]["effective_rank"]), "RAD-DINO rank mismatch")
    bio_image = biovil_image_features(
        tasks, image_root, weight_paths["biovil_image_weights"]
    )
    bio_text = biovil_text_features(bio_snapshot)
    require(bio_image.shape == (448, 128) and bio_text.shape == (7, 128), "BioViL shapes")
    prompt_index = {name: index for index, name in enumerate(PROMPT_ORDER)}
    similarity = bio_image.astype(np.float64) @ bio_text.astype(np.float64).T
    matched = np.array(
        [similarity[index, prompt_index[row["matched_condition"]]] for index, row in enumerate(tasks)]
    )
    deranged = np.array(
        [similarity[index, prompt_index[row["deranged_condition"]]] for index, row in enumerate(tasks)]
    )
    difference = matched - deranged
    recomputed = bootstrap(difference, [row["reference_stratum"] for row in tasks], int(protocol["bootstrap"]["seed"]))
    frozen = public["BioViL_T_real_alignment_validity"]["matched_minus_deranged"]
    require(close(recomputed["mean"], frozen["mean_difference"]), "alignment mean mismatch")
    require(close(recomputed["lower"], frozen["lower_95_percentile"]), "alignment lower mismatch")
    require(close(recomputed["upper"], frozen["upper_95_percentile"]), "alignment upper mismatch")
    require(recomputed["mean"] > 0 and recomputed["lower"] > 0, "independent alignment gate")
    condition_means = {
        name: float(difference[np.array([row["matched_condition"] == name for row in tasks])].mean())
        for name in PROMPT_ORDER
    }
    for name, value in condition_means.items():
        require(
            close(
                value,
                public["BioViL_T_real_alignment_validity"]["difference_by_matched_condition"][name]["mean"],
            ),
            f"condition alignment mismatch: {name}",
        )
    negative = sorted(name for name, value in condition_means.items() if value < 0)
    del rad, bio_image, bio_text, similarity, matched, deranged, difference
    gc.collect()
    torch.cuda.empty_cache()

    report = {
        "schema": SCHEMA,
        "status": "PASS_K5_EVALUATOR_PREFLIGHT_INDEPENDENT",
        "source_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "task_reconstruction": {
            "rows": 448,
            "unique_patients": 448,
            "derangement_shift": 89,
            "all_rows_exact": True,
            "status": "PASS",
        },
        "RAD_DINO": {
            "shape": [448, 768],
            "effective_rank": rad_rank,
            "matches_primary_report": True,
            "status": "PASS",
        },
        "BioViL_T": {
            "image_shape": [448, 128],
            "text_shape": [7, 128],
            "mean_difference": recomputed["mean"],
            "lower_95_percentile": recomputed["lower"],
            "upper_95_percentile": recomputed["upper"],
            "overall_gate": "PASS",
            "negative_descriptive_condition_means": negative,
            "boundary": "overall weak-label alignment validity only; negative condition summaries prohibit a seven-condition clinical-validity interpretation",
        },
        "public_boundary": {
            "identifier_or_filename_found": False,
            "features_or_per_image_scores_persisted_by_verifier": False,
            "full_K5_optimizer_execution_started": False,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    gate_dir.mkdir(parents=True)
    report_path = gate_dir / "independent_verification.json"
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"independent_report_sha256={sha256_file(report_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
