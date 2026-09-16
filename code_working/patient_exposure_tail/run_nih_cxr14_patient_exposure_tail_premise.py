#!/usr/bin/env python3
"""Run the frozen public NIH patient-exposure-tail endpoint diagnostic."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA = "nih-cxr14-patient-exposure-tail-premise/v1"
PROTOCOL_SHA256 = "DF95C8A154509BF782E87D38B31D002D6FB65DB19E446F9B4E3CAD5D1792BCD2"
K10_SHA256 = "2D749FB7B70823114A69FD55921277B0D0FF2F9AEC8FECD239159C553F3A0C06"
DINO_SHA256 = "0B8B82F85DE91B424ADED121C7E1DCC2B7BC6D0ADEEA651BF73A13307FAD8C73"
RAD_REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"
RAD_WEIGHT_SHA256 = "DBFB9F54459C38773505DE64A6AB7807BDCB392610FE1E697166342E43FB91AE"
RAD_CONFIG_SHA256 = "89DAF9751D9576D586DEDF9543C1083211611FA3A36908DB7A799B3CE7C68EDE"
RAD_PROCESSOR_SHA256 = "C537FC995C30E2353F07253899618D60E9EAE3D5F82473778602C007C6523B56"
PAIR_ORDER_SALT = "nih-cxr14-patient-exposure-tail-v1-order"
NEGATIVE_SALT = "nih-cxr14-patient-exposure-tail-v1-negative"
BOOTSTRAP_SALT = "nih-cxr14-patient-exposure-tail-v1-bootstrap"
EXPECTED_PUBLIC_IMAGES = 4_831
EXPECTED_PUBLIC_PATIENTS = 1_816
EXPECTED_ELIGIBLE_PATIENTS = 886
EXPECTED_RECORD_COUNTS = {2: 290, 3: 171, 4: 117, 5: 79, 6: 39, 7: 27, 8: 42, 9: 24, 10: 97}
DINO_BATCH = 4
RAD_BATCH = 2
BOOTSTRAP_REPLICATES = 5_000
MARGIN_THRESHOLDS = (-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05)
MEMORY_LIMIT_BYTES = int(7.5 * 1024**3)


@dataclass(frozen=True)
class Record:
    image_id: str
    patient_id: str
    partition: str
    view: str
    followup_no: int
    patient_sex: str
    target_patient: int
    finding_labels: str
    image_filename: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest().upper()


def hash_seed(*parts: object) -> int:
    return int(stable_hash(*parts)[:16], 16)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def default_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parent.parent
    rad_snapshot = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--microsoft--rad-dino"
        / "snapshots"
        / RAD_REVISION
    )
    return {
        "root": root,
        "manifest": root / "_data" / "derived" / "nih_cxr14_pa_target_enriched_v1" / "k10_private.csv",
        "image_root": root / "_data" / "raw" / "nih_cxr14_pa_k10_plus_census_v1" / "images",
        "protocol": root / "patient_exposure_tail" / "NIH_CXR14_PATIENT_EXPOSURE_TAIL_PREMISE_PROTOCOL_V1.md",
        "dino_weights": Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "dinov2_vitb14_pretrain.pth",
        "rad_snapshot": rad_snapshot,
        "output": root / "_reports" / "nih_cxr14_patient_exposure_tail_premise_v1_001",
    }


def parse_args() -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser()
    for name in ("manifest", "image_root", "protocol", "dino_weights", "rad_snapshot"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=paths[name])
    parser.add_argument("--output-dir", type=Path, default=paths["output"])
    return parser.parse_args()


def read_manifest(path: Path) -> list[Record]:
    records: list[Record] = []
    seen_images: set[str] = set()
    patient_metadata: dict[str, tuple[str, int]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["partition"] != "public_development":
                continue
            record = Record(
                image_id=row["image_id"],
                patient_id=row["patient_id"],
                partition=row["partition"],
                view=row["view"],
                followup_no=int(row["followup_no"]),
                patient_sex=row["patient_sex"],
                target_patient=int(row["target_patient"]),
                finding_labels=row["finding_labels"],
                image_filename=row["image_filename"],
            )
            require(record.view == "PA", "non-PA record in PA manifest")
            require(record.image_id not in seen_images, "duplicate image ID")
            seen_images.add(record.image_id)
            metadata = (record.patient_sex, record.target_patient)
            require(
                patient_metadata.setdefault(record.patient_id, metadata) == metadata,
                "patient metadata changes across records",
            )
            records.append(record)
    require(len(records) == EXPECTED_PUBLIC_IMAGES, "public image count drift")
    require(len(patient_metadata) == EXPECTED_PUBLIC_PATIENTS, "public patient count drift")
    return records


def record_count_bucket(count: int) -> str:
    require(2 <= count <= 10, "record count outside registered K10 range")
    if count == 2:
        return "2"
    if count == 3:
        return "3"
    if count <= 5:
        return "4-5"
    return "6-10"


def select_units(records: list[Record]) -> list[dict[str, Any]]:
    by_patient: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_patient[record.patient_id].append(record)
    eligible = {patient_id: values for patient_id, values in by_patient.items() if len(values) >= 2}
    require(len(eligible) == EXPECTED_ELIGIBLE_PATIENTS, "eligible patient count drift")
    counts = Counter(len(values) for values in eligible.values())
    require(dict(sorted(counts.items())) == EXPECTED_RECORD_COUNTS, "record-count distribution drift")

    units: list[dict[str, Any]] = []
    for patient_id in sorted(
        eligible, key=lambda value: (stable_hash(PAIR_ORDER_SALT, value), value)
    ):
        ordered = sorted(
            eligible[patient_id], key=lambda record: (record.followup_no, record.image_id)
        )
        query = ordered[0]
        gallery = ordered[-1]
        require(query.image_id != gallery.image_id, "query and gallery are identical")
        require(all(record.patient_sex == query.patient_sex for record in ordered), "sex drift")
        require(all(record.target_patient == query.target_patient for record in ordered), "target drift")
        units.append(
            {
                "patient_id": patient_id,
                "patient_sex": query.patient_sex,
                "target_patient": query.target_patient,
                "record_count": len(ordered),
                "record_count_bucket": record_count_bucket(len(ordered)),
                "query": query,
                "gallery": gallery,
                "followup_order_gap": gallery.followup_no - query.followup_no,
            }
        )
    require(len(units) == EXPECTED_ELIGIBLE_PATIENTS, "unit count mismatch")
    require(len({unit["patient_id"] for unit in units}) == len(units), "duplicate patient unit")
    return units


def subgroup_key(unit: dict[str, Any]) -> str:
    return (
        f"sex={unit['patient_sex']}|target={unit['target_patient']}|"
        f"n={unit['record_count_bucket']}"
    )


def matched_negative_indices(units: list[dict[str, Any]]) -> list[int]:
    cells: dict[str, list[int]] = defaultdict(list)
    for index, unit in enumerate(units):
        cells[subgroup_key(unit)].append(index)
    require(len(cells) == 16, "registered subgroup-cell count drift")
    mapping = [-1] * len(units)
    for key, indices in cells.items():
        require(len(indices) >= 2, f"insufficient derangement cell: {key}")
        ordered = sorted(
            indices,
            key=lambda index: (
                stable_hash(NEGATIVE_SALT, units[index]["patient_id"]),
                units[index]["patient_id"],
            ),
        )
        for position, index in enumerate(ordered):
            mapping[index] = ordered[(position + 1) % len(ordered)]
    require(all(index >= 0 for index in mapping), "incomplete derangement")
    require(all(index != mapped for index, mapped in enumerate(mapping)), "derangement fixed point")
    require(len(set(mapping)) == len(units), "derangement is not a permutation")
    for index, mapped in enumerate(mapping):
        require(subgroup_key(units[index]) == subgroup_key(units[mapped]), "stratum mismatch")
        require(units[index]["patient_id"] != units[mapped]["patient_id"], "same-patient negative")
    return mapping


def write_selection(path: Path, units: list[dict[str, Any]], negative_indices: list[int]) -> None:
    fields = (
        "patient_id",
        "patient_sex",
        "target_patient",
        "record_count",
        "record_count_bucket",
        "query_image_id",
        "gallery_image_id",
        "negative_patient_id",
        "negative_gallery_image_id",
        "query_followup_no",
        "gallery_followup_no",
        "followup_order_gap",
        "query_finding_labels",
        "gallery_finding_labels",
        "selection_commitment",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, unit in enumerate(units):
            negative = units[negative_indices[index]]
            writer.writerow(
                {
                    "patient_id": unit["patient_id"],
                    "patient_sex": unit["patient_sex"],
                    "target_patient": unit["target_patient"],
                    "record_count": unit["record_count"],
                    "record_count_bucket": unit["record_count_bucket"],
                    "query_image_id": unit["query"].image_id,
                    "gallery_image_id": unit["gallery"].image_id,
                    "negative_patient_id": negative["patient_id"],
                    "negative_gallery_image_id": negative["gallery"].image_id,
                    "query_followup_no": unit["query"].followup_no,
                    "gallery_followup_no": unit["gallery"].followup_no,
                    "followup_order_gap": unit["followup_order_gap"],
                    "query_finding_labels": unit["query"].finding_labels,
                    "gallery_finding_labels": unit["gallery"].finding_labels,
                    "selection_commitment": stable_hash(
                        PAIR_ORDER_SALT,
                        unit["patient_id"],
                        unit["query"].image_id,
                        unit["gallery"].image_id,
                        negative["patient_id"],
                        negative["gallery"].image_id,
                    ),
                }
            )


def configure_torch() -> Any:
    import torch

    require(torch.cuda.is_available(), "CUDA is required")
    torch.manual_seed(260_903_397_624)
    torch.cuda.manual_seed_all(260_903_397_624)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def batches(values: list[Record], size: int) -> Iterable[list[Record]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def extract_dino(
    records: list[Record], image_root: Path, weights: Path, torch: Any
) -> tuple[np.ndarray, dict[str, Any]]:
    import timm
    import torch.nn.functional as functional

    patient_set_dir = Path(__file__).resolve().parent.parent / "patient_set_diffusion"
    sys.path.insert(0, str(patient_set_dir))
    import run_nih_cxr14_patient_set_premise as premise

    model = timm.create_model("vit_base_patch14_dinov2", pretrained=False, num_classes=0)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    require(not incompatible.missing_keys, "DINO missing keys")
    require(incompatible.unexpected_keys == ["mask_token"], "DINO unexpected keys")
    model.eval().cuda()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for group in batches(records, DINO_BATCH):
            tensor = torch.stack(
                [premise.preprocess_image(image_root / record.image_filename) for record in group]
            ).cuda()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(tensor)
            chunks.append(functional.normalize(output.float(), dim=1).cpu().numpy())
    features = np.concatenate(chunks, axis=0).astype(np.float64)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated())
    replay_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for group in batches(records[:8], DINO_BATCH):
            tensor = torch.stack(
                [premise.preprocess_image(image_root / record.image_filename) for record in group]
            ).cuda()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(tensor)
            replay_chunks.append(functional.normalize(output.float(), dim=1).cpu().numpy())
    replay = np.concatenate(replay_chunks, axis=0).astype(np.float64)
    replay_error = float(np.max(np.abs(replay - features[:8])))
    norms = np.linalg.norm(features, axis=1)
    require(features.shape == (len(records), 768), "DINO feature shape")
    require(np.all(np.isfinite(features)), "DINO non-finite")
    require(float(np.max(np.abs(norms - 1.0))) <= 1e-5, "DINO norm")
    require(replay_error <= 1e-5, "DINO replay drift")
    metadata = {
        "architecture": "vit_base_patch14_dinov2",
        "shape": list(features.shape),
        "unit_norm_max_abs_error": float(np.max(np.abs(norms - 1.0))),
        "first_eight_replay_max_abs": replay_error,
        "seconds": elapsed,
        "peak_cuda_bytes": peak,
        "timm_version": timm.__version__,
    }
    del model, state, chunks, replay_chunks, replay
    gc.collect()
    torch.cuda.empty_cache()
    return features, metadata


def extract_rad_dino(
    records: list[Record], image_root: Path, snapshot: Path, torch: Any
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch.nn.functional as functional
    from transformers import AutoImageProcessor, AutoModel

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from dp_training.k5_evaluator import load_real_p256_grayscale

    processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True, use_fast=False)
    model = AutoModel.from_pretrained(
        snapshot, local_files_only=True, use_safetensors=True
    ).eval().cuda()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for group in batches(records, RAD_BATCH):
            images = [
                load_real_p256_grayscale(image_root / record.image_filename).convert("RGB")
                for record in group
            ]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {name: value.cuda(non_blocking=False) for name, value in inputs.items()}
            output = model(**inputs).pooler_output
            chunks.append(functional.normalize(output.float(), dim=1).cpu().numpy())
    features = np.concatenate(chunks, axis=0).astype(np.float64)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated())
    replay_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for group in batches(records[:8], RAD_BATCH):
            images = [
                load_real_p256_grayscale(image_root / record.image_filename).convert("RGB")
                for record in group
            ]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {name: value.cuda(non_blocking=False) for name, value in inputs.items()}
            output = model(**inputs).pooler_output
            replay_chunks.append(functional.normalize(output.float(), dim=1).cpu().numpy())
    replay = np.concatenate(replay_chunks, axis=0).astype(np.float64)
    replay_error = float(np.max(np.abs(replay - features[:8])))
    norms = np.linalg.norm(features, axis=1)
    require(features.shape == (len(records), 768), "RAD-DINO feature shape")
    require(np.all(np.isfinite(features)), "RAD-DINO non-finite")
    require(float(np.max(np.abs(norms - 1.0))) <= 1e-5, "RAD-DINO norm")
    require(replay_error <= 1e-5, "RAD-DINO replay drift")
    metadata = {
        "architecture": "microsoft/rad-dino",
        "revision": RAD_REVISION,
        "shape": list(features.shape),
        "unit_norm_max_abs_error": float(np.max(np.abs(norms - 1.0))),
        "first_eight_replay_max_abs": replay_error,
        "seconds": elapsed,
        "peak_cuda_bytes": peak,
        "transformers_version": importlib.metadata.version("transformers"),
        "same_source_pretraining_limit": (
            "RAD-DINO was pretrained on NIH ChestXray14 and cannot support the gate alone"
        ),
    }
    del model, processor, chunks, replay_chunks, replay
    gc.collect()
    torch.cuda.empty_cache()
    return features, metadata


def rank_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    from scipy.stats import rankdata

    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    require(positive.ndim == negative.ndim == 1, "AUC inputs must be vectors")
    require(len(positive) > 0 and len(negative) > 0, "empty AUC input")
    scores = np.concatenate((negative, positive))
    ranks = rankdata(scores, method="average")
    n_negative = len(negative)
    n_positive = len(positive)
    positive_rank_sum = float(np.sum(ranks[n_negative:]))
    return (positive_rank_sum - n_positive * (n_positive + 1) / 2) / (
        n_positive * n_negative
    )


def spearman(value_a: np.ndarray, value_b: np.ndarray) -> float:
    from scipy.stats import rankdata

    rank_a = rankdata(np.asarray(value_a, dtype=np.float64), method="average")
    rank_b = rankdata(np.asarray(value_b, dtype=np.float64), method="average")
    require(float(np.std(rank_a)) > 0 and float(np.std(rank_b)) > 0, "constant Spearman input")
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def interval(values: np.ndarray) -> list[float]:
    return [
        float(np.quantile(values, 0.025, method="linear")),
        float(np.quantile(values, 0.975, method="linear")),
    ]


def bootstrap_encoder(
    positive: np.ndarray,
    negative: np.ndarray,
    ranks: np.ndarray,
    encoder: str,
) -> dict[str, list[float]]:
    n = len(positive)
    rng = np.random.default_rng(hash_seed(BOOTSTRAP_SALT, encoder))
    auc_values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    recall_values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    difficult_values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, n, size=n)
        auc_values[replicate] = rank_auc(positive[sampled], negative[sampled])
        recall_values[replicate] = float(np.mean(ranks[sampled] == 1))
        difficult_values[replicate] = float(np.mean(ranks[sampled] > 10))
    return {
        "auc": interval(auc_values),
        "recall_at_1": interval(recall_values),
        "rank_greater_than_10_fraction": interval(difficult_values),
    }


def bootstrap_spearman(margin_a: np.ndarray, margin_b: np.ndarray) -> list[float]:
    n = len(margin_a)
    rng = np.random.default_rng(hash_seed(BOOTSTRAP_SALT, "cross-encoder-spearman"))
    values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, n, size=n)
        values[replicate] = spearman(margin_a[sampled], margin_b[sampled])
    return interval(values)


def encoder_metrics(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    negative_indices: list[int],
    units: list[dict[str, Any]],
    encoder: str,
) -> tuple[dict[str, Any], np.ndarray]:
    require(query_features.shape == gallery_features.shape, "query/gallery feature shape mismatch")
    n = len(units)
    require(query_features.shape[0] == n, "feature/unit count mismatch")
    similarity = query_features @ gallery_features.T
    positive = np.diag(similarity).copy()
    negative = similarity[np.arange(n), np.asarray(negative_indices, dtype=np.int64)]
    ranks = np.empty(n, dtype=np.int64)
    margins = np.empty(n, dtype=np.float64)
    for index in range(n):
        impostors = np.delete(similarity[index], index)
        ranks[index] = 1 + int(np.sum(impostors >= positive[index]))
        margins[index] = positive[index] - float(np.max(impostors))
    boot = bootstrap_encoder(positive, negative, ranks, encoder)
    margin_quantiles = {
        f"p{percentile:02d}": float(np.quantile(margins, percentile / 100, method="linear"))
        for percentile in (1, 5, 10, 25, 50, 75, 90, 95, 99)
    }
    subgroup_results: dict[str, Any] = {}
    for key in sorted({subgroup_key(unit) for unit in units}):
        indices = np.asarray(
            [index for index, unit in enumerate(units) if subgroup_key(unit) == key],
            dtype=np.int64,
        )
        subgroup_results[key] = {
            "patients": int(len(indices)),
            "pair_auc": rank_auc(positive[indices], negative[indices]),
            "recall_at_1": float(np.mean(ranks[indices] == 1)),
            "recall_at_10": float(np.mean(ranks[indices] <= 10)),
            "rank_greater_than_10_fraction": float(np.mean(ranks[indices] > 10)),
            "median_rank": float(np.median(ranks[indices])),
            "mean_margin": float(np.mean(margins[indices])),
        }
    result = {
        "patients": n,
        "true_pair_similarity": {
            "mean": float(np.mean(positive)),
            "standard_deviation": float(np.std(positive, ddof=1)),
        },
        "matched_negative_similarity": {
            "mean": float(np.mean(negative)),
            "standard_deviation": float(np.std(negative, ddof=1)),
        },
        "true_vs_matched_negative_auc": rank_auc(positive, negative),
        "recall_at_1": float(np.mean(ranks == 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "rank_greater_than_10_fraction": float(np.mean(ranks > 10)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "margin_quantiles": margin_quantiles,
        "margin_survival_fraction_at_least": {
            f"{threshold:+.2f}": float(np.mean(margins >= threshold))
            for threshold in MARGIN_THRESHOLDS
        },
        "bootstrap_95_percentile_intervals": boot,
        "subgroups_descriptive_only": subgroup_results,
    }
    return result, margins


def decision_from_results(
    encoders: dict[str, dict[str, Any]],
    cross_encoder: dict[str, Any],
    peak_cuda_bytes: int,
) -> tuple[dict[str, bool], dict[str, dict[str, bool]]]:
    per_encoder: dict[str, dict[str, bool]] = {}
    for name, result in encoders.items():
        per_encoder[name] = {
            "pair_auc_at_least_0_80": result["true_vs_matched_negative_auc"] >= 0.80,
            "pair_auc_bootstrap_lower_above_0_75": (
                result["bootstrap_95_percentile_intervals"]["auc"][0] > 0.75
            ),
            "recall_at_1_at_least_0_15": result["recall_at_1"] >= 0.15,
            "recall_at_1_bootstrap_lower_above_0_10": (
                result["bootstrap_95_percentile_intervals"]["recall_at_1"][0] > 0.10
            ),
            "rank_greater_than_10_fraction_at_least_0_10": (
                result["rank_greater_than_10_fraction"] >= 0.10
            ),
            "margin_p90_positive": result["margin_quantiles"]["p90"] > 0.0,
            "margin_p10_negative": result["margin_quantiles"]["p10"] < 0.0,
        }
    cross_checks = {
        "cross_encoder_margin_spearman_at_least_0_20": cross_encoder["margin_spearman"] >= 0.20,
        "cross_encoder_spearman_bootstrap_lower_above_0_10": (
            cross_encoder["bootstrap_95_percentile_interval"][0] > 0.10
        ),
        "peak_memory_below_7_5_gib": peak_cuda_bytes < MEMORY_LIMIT_BYTES,
    }
    return cross_checks, per_encoder


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    source_path = Path(__file__).resolve()
    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), "refusing to overwrite diagnostic output")
    require(sha256_file(args.protocol.resolve()) == PROTOCOL_SHA256, "protocol hash mismatch")
    require(sha256_file(args.manifest.resolve()) == K10_SHA256, "manifest hash mismatch")
    require(sha256_file(args.dino_weights.resolve()) == DINO_SHA256, "DINO weight hash mismatch")
    rad_snapshot = args.rad_snapshot.resolve()
    require(sha256_file(rad_snapshot / "model.safetensors") == RAD_WEIGHT_SHA256, "RAD weight hash")
    require(sha256_file(rad_snapshot / "config.json") == RAD_CONFIG_SHA256, "RAD config hash")
    require(
        sha256_file(rad_snapshot / "preprocessor_config.json") == RAD_PROCESSOR_SHA256,
        "RAD processor hash",
    )

    records = read_manifest(args.manifest.resolve())
    units = select_units(records)
    negative_indices = matched_negative_indices(units)
    query_records = [unit["query"] for unit in units]
    gallery_records = [unit["gallery"] for unit in units]
    feature_records = query_records + gallery_records
    image_root = args.image_root.resolve()
    require(all((image_root / record.image_filename).is_file() for record in feature_records), "missing image")

    torch = configure_torch()
    started = time.perf_counter()
    dino_features, dino_execution = extract_dino(
        feature_records, image_root, args.dino_weights.resolve(), torch
    )
    dino_result, dino_margins = encoder_metrics(
        dino_features[: len(units)],
        dino_features[len(units) :],
        negative_indices,
        units,
        "DINOv2",
    )
    del dino_features
    gc.collect()
    torch.cuda.empty_cache()

    rad_features, rad_execution = extract_rad_dino(
        feature_records, image_root, rad_snapshot, torch
    )
    rad_result, rad_margins = encoder_metrics(
        rad_features[: len(units)],
        rad_features[len(units) :],
        negative_indices,
        units,
        "RAD-DINO",
    )
    del rad_features
    gc.collect()
    torch.cuda.empty_cache()
    elapsed = time.perf_counter() - started

    cross_encoder = {
        "margin_spearman": spearman(dino_margins, rad_margins),
        "bootstrap_95_percentile_interval": bootstrap_spearman(dino_margins, rad_margins),
    }
    peak_cuda_bytes = max(dino_execution["peak_cuda_bytes"], rad_execution["peak_cuda_bytes"])
    encoder_results = {"DINOv2": dino_result, "RAD-DINO": rad_result}
    cross_checks, per_encoder_checks = decision_from_results(
        encoder_results, cross_encoder, peak_cuda_bytes
    )
    all_checks = all(cross_checks.values()) and all(
        value for checks in per_encoder_checks.values() for value in checks.values()
    )
    status = (
        "PASS_PATIENT_EXPOSURE_TAIL_ENDPOINT_PREMISE"
        if all_checks
        else "FAIL_PATIENT_EXPOSURE_TAIL_ENDPOINT_PREMISE"
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    selection_path = output_dir / "selected_public_query_gallery_private.csv"
    write_selection(selection_path, units, negative_indices)
    report = {
        "schema": SCHEMA,
        "status": status,
        "scope": "PUBLIC_NONDP_FEATURE_ENDPOINT_NOT_GENERATOR_NOT_MEMBERSHIP_AUC_NOT_PRIVACY_NOT_CLINICAL",
        "frozen_inputs": {
            "protocol_sha256": PROTOCOL_SHA256,
            "implementation_sha256": sha256_file(source_path),
            "manifest_sha256": K10_SHA256,
            "dino_weights_sha256": DINO_SHA256,
            "rad_revision": RAD_REVISION,
            "rad_weight_sha256": RAD_WEIGHT_SHA256,
            "rad_config_sha256": RAD_CONFIG_SHA256,
            "rad_processor_sha256": RAD_PROCESSOR_SHA256,
            "pair_order_salt": PAIR_ORDER_SALT,
            "negative_salt": NEGATIVE_SALT,
            "bootstrap_salt": BOOTSTRAP_SALT,
        },
        "selection": {
            "public_images": len(records),
            "public_patients": len({record.patient_id for record in records}),
            "eligible_multi_record_patients": len(units),
            "query_images": len(query_records),
            "true_gallery_images": len(gallery_records),
            "record_count_distribution": dict(sorted(Counter(unit["record_count"] for unit in units).items())),
            "registered_subgroup_counts": dict(sorted(Counter(subgroup_key(unit) for unit in units).items())),
            "followup_order_gap": {
                "minimum": min(unit["followup_order_gap"] for unit in units),
                "median": float(np.median([unit["followup_order_gap"] for unit in units])),
                "maximum": max(unit["followup_order_gap"] for unit in units),
            },
            "selection_file": selection_path.name,
            "selection_sha256": sha256_file(selection_path),
            "identifier_boundary": "selection file is local-only; report contains aggregates only",
        },
        "encoders": {
            "DINOv2": {"execution": dino_execution, "analysis": dino_result},
            "RAD-DINO": {"execution": rad_execution, "analysis": rad_result},
        },
        "cross_encoder": cross_encoder,
        "gate_checks": {
            "per_encoder": per_encoder_checks,
            "cross_and_resource": cross_checks,
            "all_conjunctive": all_checks,
        },
        "execution": {
            "elapsed_feature_and_analysis_seconds": elapsed,
            "peak_cuda_memory_bytes": peak_cuda_bytes,
            "peak_cuda_memory_gib": peak_cuda_bytes / 1024**3,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "feature_retained": False,
            "encoder_checkpoint_written": False,
            "optimizer_created": False,
            "gradient_retained": False,
            "latent_retained": False,
            "generated_image_written": False,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "scipy_version": importlib.metadata.version("scipy"),
        },
        "known_evidence_boundary": (
            "Earlier small public probes already showed same-patient feature similarity. This run is a full-cohort, "
            "equal-opportunity endpoint-feasibility audit, not an independent discovery of CXR re-identification."
        ),
        "interpretation_limit": (
            "A pass licenses only a separately frozen generator biometric-exposure endpoint addendum. It does not "
            "authorize K5 training, establish generator leakage, estimate individual-patient membership AUC, prove "
            "privacy, or establish a novel CVPR method."
        ),
    }
    write_json(output_dir / "report.json", report)
    print(status)
    print(json.dumps(report["gate_checks"], sort_keys=True))
    for encoder in ("DINOv2", "RAD-DINO"):
        analysis = report["encoders"][encoder]["analysis"]
        print(
            encoder,
            "AUC=", analysis["true_vs_matched_negative_auc"],
            "R1=", analysis["recall_at_1"],
            "R10=", analysis["recall_at_10"],
            "rank>10=", analysis["rank_greater_than_10_fraction"],
        )
    print("margin_spearman=", cross_encoder["margin_spearman"])
    return 0 if status.startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
