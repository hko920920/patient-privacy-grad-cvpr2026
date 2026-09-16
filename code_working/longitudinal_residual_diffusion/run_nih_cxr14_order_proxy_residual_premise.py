#!/usr/bin/env python3
"""Run the frozen NIH order-proxy longitudinal residual premise diagnostic."""

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


SCHEMA = "nih-cxr14-order-proxy-residual-premise/v1"
PROTOCOL_SHA256 = "92B0B0D031255E6C606074F76FCCD081688AFC429779C53456C94DDAE995B14D"
K10_SHA256 = "2D749FB7B70823114A69FD55921277B0D0FF2F9AEC8FECD239159C553F3A0C06"
DINO_SHA256 = "0B8B82F85DE91B424ADED121C7E1DCC2B7BC6D0ADEEA651BF73A13307FAD8C73"
RAD_REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"
RAD_WEIGHT_SHA256 = "DBFB9F54459C38773505DE64A6AB7807BDCB392610FE1E697166342E43FB91AE"
RAD_CONFIG_SHA256 = "89DAF9751D9576D586DEDF9543C1083211611FA3A36908DB7A799B3CE7C68EDE"
RAD_PROCESSOR_SHA256 = "C537FC995C30E2353F07253899618D60E9EAE3D5F82473778602C007C6523B56"
EXCLUSION_HASHES = {
    "clip_calibration": "CB7E845F7B1D627BF52973ADBA9E7F3E6207F2081441DBEFBADE7E40AE261F6E",
    "ucan": "13CB2054DE2A25208972590BA023585C16555D7ECE260D4FA571633CD9681AF0",
    "patient_set_v1": "87B356A6175493E309DE2D84E8D0769B4764005BEE809AA8EB5CF5D5EF544AB0",
    "patient_set_v2": "E1ABC33EA68082F69086CDFB752B558580710FAD35FB70640BDAD590DFC2C6F8",
    "context_v1": "B6951897C7270CC7A462496F82A5EC1CDDC65D7F37A15EA6E2D629809604E319",
    "cross_context_v2": "9145E378DAFD976ACAECEE6B78ED8F8B28D73BCDB13A466111DECBB648D038C0",
}
EXPECTED_EXCLUDED_PATIENTS = 528
PAIR_SALT = "nih-cxr14-order-proxy-residual-premise-v1-pair"
PATIENT_SALTS = {
    "no_change": "nih-cxr14-order-proxy-residual-premise-v1-patient-no-change",
    "change": "nih-cxr14-order-proxy-residual-premise-v1-patient-change",
}
TRAIN_PER_CLASS = 96
VALIDATION_PER_CLASS = 48
TOTAL_PATIENTS = 2 * (TRAIN_PER_CLASS + VALIDATION_PER_CLASS)
TOTAL_IMAGES = 2 * TOTAL_PATIENTS
IMAGE_SIZE = 518
DINO_BATCH = 4
RAD_BATCH = 2
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
CV_FOLDS = 5
BOOTSTRAP_REPLICATES = 5_000
MEMORY_LIMIT_BYTES = int(7.5 * 1024**3)
LABELS = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
    "No Finding",
)


@dataclass(frozen=True)
class Record:
    image_id: str
    patient_id: str
    partition: str
    view: str
    followup_no: int
    patient_sex: str
    finding_labels: tuple[str, ...]
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
    reports = root / "_reports"
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
        "protocol": root / "longitudinal_residual_diffusion" / "NIH_CXR14_ORDER_PROXY_RESIDUAL_PREMISE_PROTOCOL_V1.md",
        "dino_weights": Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "dinov2_vitb14_pretrain.pth",
        "rad_snapshot": rad_snapshot,
        "clip_calibration": reports / "nih_cxr14_public_clip_calibration_v1_001" / "selected_public_units_private.csv",
        "ucan": reports / "nih_cxr14_ucan_public_diagnostic_v1_001" / "selected_public_patients_private.csv",
        "patient_set_v1": reports / "nih_cxr14_patient_set_premise_v1_001" / "selected_public_patient_sets_private.csv",
        "patient_set_v2": reports / "nih_cxr14_patient_set_premise_confirmation_v2_001" / "selected_public_patient_sets_private.csv",
        "context_v1": reports / "nih_cxr14_setadapter_context_signal_v1_001" / "selected_public_context_cohort_private.csv",
        "cross_context_v2": reports / "nih_cxr14_cross_record_context_confirmation_v2_001" / "selected_public_cross_record_cohort_private.csv",
        "output": reports / "nih_cxr14_order_proxy_residual_premise_v1_001",
    }


def parse_args() -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser()
    for name in (
        "manifest",
        "image_root",
        "protocol",
        "dino_weights",
        "rad_snapshot",
        *EXCLUSION_HASHES.keys(),
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=paths[name])
    parser.add_argument("--output-dir", type=Path, default=paths["output"])
    return parser.parse_args()


def parse_labels(value: str) -> tuple[str, ...]:
    labels = tuple(item.strip() for item in value.split("|") if item.strip())
    require(bool(labels), "empty finding label")
    require(not (set(labels) - set(LABELS)), f"unknown labels: {set(labels) - set(LABELS)}")
    require(not ("No Finding" in labels and len(labels) != 1), "No Finding mixed with pathology")
    return labels


def canonical_labels(record: Record) -> str:
    return "|".join(record.finding_labels)


def read_manifest(path: Path) -> list[Record]:
    rows: list[Record] = []
    seen: set[str] = set()
    patient_partition: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["partition"] != "public_development":
                continue
            require(row["view"] == "PA", "non-PA row in PA manifest")
            require(row["image_id"] not in seen, "duplicate image")
            seen.add(row["image_id"])
            previous = patient_partition.setdefault(row["patient_id"], row["partition"])
            require(previous == row["partition"], "patient crosses partitions")
            rows.append(
                Record(
                    image_id=row["image_id"],
                    patient_id=row["patient_id"],
                    partition=row["partition"],
                    view=row["view"],
                    followup_no=int(row["followup_no"]),
                    patient_sex=row["patient_sex"],
                    finding_labels=parse_labels(row["finding_labels"]),
                    image_filename=row["image_filename"],
                )
            )
    require(bool(rows), "empty public-development manifest")
    return rows


def read_patient_ids(path: Path, expected_hash: str) -> set[str]:
    require(sha256_file(path) == expected_hash, f"selection hash mismatch: {path.name}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["patient_id"] for row in csv.DictReader(handle)}


def pair_class(previous: Record, current: Record) -> str:
    return "no_change" if previous.finding_labels == current.finding_labels else "change"


def candidate_pairs(
    records: list[Record], excluded: set[str]
) -> dict[str, dict[str, tuple[Record, Record]]]:
    by_patient: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        if record.patient_id not in excluded:
            by_patient[record.patient_id].append(record)
    choices: dict[str, dict[str, tuple[Record, Record]]] = {"no_change": {}, "change": {}}
    for patient_id, patient_records in by_patient.items():
        ordered = sorted(patient_records, key=lambda item: (item.followup_no, item.image_id))
        per_class: dict[str, list[tuple[Record, Record]]] = defaultdict(list)
        for previous, current in zip(ordered, ordered[1:]):
            if current.followup_no - previous.followup_no == 1:
                per_class[pair_class(previous, current)].append((previous, current))
        for name, pairs in per_class.items():
            choices[name][patient_id] = min(
                pairs,
                key=lambda pair: (
                    stable_hash(PAIR_SALT, patient_id, pair[0].image_id, pair[1].image_id),
                    pair[0].image_id,
                    pair[1].image_id,
                ),
            )
    return choices


def select_cohort(
    records: list[Record], excluded: set[str]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    choices = candidate_pairs(records, excluded)
    per_class = TRAIN_PER_CLASS + VALIDATION_PER_CLASS
    no_change_ranked = sorted(
        choices["no_change"],
        key=lambda patient_id: (stable_hash(PATIENT_SALTS["no_change"], patient_id), patient_id),
    )
    require(len(no_change_ranked) >= per_class, "insufficient no-change patients")
    no_change_selected = no_change_ranked[:per_class]
    occupied = set(no_change_selected)
    change_ranked = sorted(
        (patient_id for patient_id in choices["change"] if patient_id not in occupied),
        key=lambda patient_id: (stable_hash(PATIENT_SALTS["change"], patient_id), patient_id),
    )
    require(len(change_ranked) >= per_class, "insufficient disjoint change patients")
    selected = {"no_change": no_change_selected, "change": change_ranked[:per_class]}

    units: list[dict[str, Any]] = []
    for name in ("no_change", "change"):
        for rank, patient_id in enumerate(selected[name]):
            previous, current = choices[name][patient_id]
            units.append(
                {
                    "patient_id": patient_id,
                    "transition_class": name,
                    "split": "train" if rank < TRAIN_PER_CLASS else "validation",
                    "class_rank": rank,
                    "previous": previous,
                    "current": current,
                    "selection_commitment": stable_hash(
                        PATIENT_SALTS[name], patient_id, previous.image_id, current.image_id
                    ),
                }
            )

    patient_ids = [unit["patient_id"] for unit in units]
    image_ids = [
        record.image_id for unit in units for record in (unit["previous"], unit["current"])
    ]
    require(len(units) == TOTAL_PATIENTS, "cohort patient count")
    require(len(set(patient_ids)) == TOTAL_PATIENTS, "patient overlap")
    require(set(patient_ids).isdisjoint(excluded), "excluded patient selected")
    require(len(image_ids) == TOTAL_IMAGES and len(set(image_ids)) == TOTAL_IMAGES, "image overlap")
    counts = Counter((unit["split"], unit["transition_class"]) for unit in units)
    require(counts[("train", "no_change")] == TRAIN_PER_CLASS, "train no-change count")
    require(counts[("train", "change")] == TRAIN_PER_CLASS, "train change count")
    require(counts[("validation", "no_change")] == VALIDATION_PER_CLASS, "val no-change count")
    require(counts[("validation", "change")] == VALIDATION_PER_CLASS, "val change count")
    return units, {
        "eligible_no_change_patients_after_exclusion": len(choices["no_change"]),
        "eligible_change_patients_after_exclusion": len(choices["change"]),
        "eligible_change_after_no_change_selection": len(change_ranked),
    }


def write_selection(path: Path, units: list[dict[str, Any]]) -> None:
    fields = (
        "patient_id",
        "split",
        "transition_class",
        "class_rank",
        "previous_image_id",
        "current_image_id",
        "previous_followup_no",
        "current_followup_no",
        "patient_sex",
        "previous_labels",
        "current_labels",
        "selection_commitment",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for unit in units:
            previous, current = unit["previous"], unit["current"]
            writer.writerow(
                {
                    "patient_id": unit["patient_id"],
                    "split": unit["split"],
                    "transition_class": unit["transition_class"],
                    "class_rank": unit["class_rank"],
                    "previous_image_id": previous.image_id,
                    "current_image_id": current.image_id,
                    "previous_followup_no": previous.followup_no,
                    "current_followup_no": current.followup_no,
                    "patient_sex": previous.patient_sex,
                    "previous_labels": canonical_labels(previous),
                    "current_labels": canonical_labels(current),
                    "selection_commitment": unit["selection_commitment"],
                }
            )


def transition_vector(unit: dict[str, Any]) -> np.ndarray:
    previous = set(unit["previous"].finding_labels)
    current = set(unit["current"].finding_labels)
    additions = [float(label in current and label not in previous) for label in LABELS]
    removals = [float(label in previous and label not in current) for label in LABELS]
    return np.asarray(additions + removals, dtype=np.float64)


def fold_ids(units: list[dict[str, Any]]) -> np.ndarray:
    order = sorted(
        range(len(units)),
        key=lambda index: (stable_hash("order-proxy-residual-cv-fold", units[index]["patient_id"]), index),
    )
    result = np.empty(len(units), dtype=np.int64)
    for rank, index in enumerate(order):
        result[index] = rank % CV_FOLDS
    return result


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, np.ndarray | float]:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    require(x64.ndim == 2 and y64.ndim == 2 and len(x64) == len(y64), "ridge shapes")
    mean = x64.mean(axis=0)
    std = x64.std(axis=0, ddof=0)
    std = np.where(std > 0.0, std, 1.0)
    standardized = (x64 - mean) / std
    y_mean = y64.mean(axis=0)
    gram = standardized.T @ standardized
    beta = np.linalg.solve(
        gram + float(alpha) * np.eye(standardized.shape[1], dtype=np.float64),
        standardized.T @ (y64 - y_mean),
    )
    return {"x_mean": mean, "x_std": std, "y_mean": y_mean, "beta": beta, "alpha": alpha}


def predict_ridge(model: dict[str, np.ndarray | float], x: np.ndarray) -> np.ndarray:
    standardized = (np.asarray(x, dtype=np.float64) - model["x_mean"]) / model["x_std"]
    return standardized @ model["beta"] + model["y_mean"]


def choose_ridge(
    x: np.ndarray, y: np.ndarray, units: list[dict[str, Any]]
) -> tuple[dict[str, np.ndarray | float], dict[str, Any]]:
    folds = fold_ids(units)
    fold_counts = Counter(int(item) for item in folds)
    require(set(fold_counts) == set(range(CV_FOLDS)), "missing ridge fold")
    errors: dict[float, float] = {}
    for alpha in RIDGE_ALPHAS:
        per_pair: list[float] = []
        for fold in range(CV_FOLDS):
            train = folds != fold
            test = folds == fold
            model = fit_ridge(x[train], y[train], alpha)
            prediction = predict_ridge(model, x[test])
            per_pair.extend(np.mean((y[test] - prediction) ** 2, axis=1).tolist())
        errors[alpha] = float(np.mean(per_pair))
    best = min(RIDGE_ALPHAS, key=lambda alpha: (errors[alpha], -alpha))
    return fit_ridge(x, y, best), {
        "selected_alpha": best,
        "cv_mean_mse_by_alpha": {str(alpha): errors[alpha] for alpha in RIDGE_ALPHAS},
        "fold_counts": {str(key): fold_counts[key] for key in sorted(fold_counts)},
    }


def derange_transition_rows(
    x: np.ndarray, units: list[dict[str, Any]]
) -> tuple[np.ndarray, dict[str, Any]]:
    output = np.empty_like(x)
    report: dict[str, Any] = {}
    for name in ("no_change", "change"):
        indices = [index for index, unit in enumerate(units) if unit["transition_class"] == name]
        ordered = sorted(
            indices,
            key=lambda index: (stable_hash("order-proxy-residual-derangement", units[index]["patient_id"]), index),
        )
        require(len(ordered) >= 2, "derangement class too small")
        candidates = []
        for shift in range(1, len(ordered)):
            exact_matches = sum(
                bool(np.array_equal(x[index], x[ordered[(rank + shift) % len(ordered)]]))
                for rank, index in enumerate(ordered)
            )
            candidates.append((exact_matches, shift))
        exact_matches, shift = min(candidates)
        for rank, index in enumerate(ordered):
            donor = ordered[(rank + shift) % len(ordered)]
            require(donor != index, "derangement fixed point")
            output[index] = x[donor]
        report[name] = {
            "rows": len(ordered),
            "cyclic_shift": shift,
            "fixed_points": 0,
            "exact_transition_vector_matches": exact_matches,
        }
    return output, report


def label_jaccard(left: Record, right: Record) -> float:
    first, second = set(left.finding_labels), set(right.finding_labels)
    return len(first & second) / len(first | second)


def choose_shuffled_prior(
    index: int, validation_indices: list[int], units: list[dict[str, Any]]
) -> tuple[int, dict[str, Any]]:
    unit = units[index]
    candidates = [
        other
        for other in validation_indices
        if other != index and units[other]["transition_class"] == unit["transition_class"]
    ]
    require(bool(candidates), "no shuffled-prior candidate")

    def properties(other: int) -> tuple[int, float]:
        donor = units[other]
        same_sex = int(donor["previous"].patient_sex == unit["previous"].patient_sex)
        similarity = label_jaccard(donor["previous"], unit["previous"]) + label_jaccard(
            donor["current"], unit["current"]
        )
        return same_sex, similarity

    chosen = min(
        candidates,
        key=lambda other: (
            -properties(other)[0],
            -properties(other)[1],
            stable_hash(
                "order-proxy-residual-shuffled-prior",
                unit["patient_id"],
                units[other]["patient_id"],
            ),
        ),
    )
    donor = units[chosen]
    return chosen, {
        "same_sex": donor["previous"].patient_sex == unit["previous"].patient_sex,
        "previous_exact_label": donor["previous"].finding_labels == unit["previous"].finding_labels,
        "future_exact_label": donor["current"].finding_labels == unit["current"].finding_labels,
        "previous_label_jaccard": label_jaccard(donor["previous"], unit["previous"]),
        "future_label_jaccard": label_jaccard(donor["current"], unit["current"]),
    }


def describe(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    require(array.size > 0 and np.all(np.isfinite(array)), "invalid statistic")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "minimum": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def bootstrap_ratio(
    numerator: np.ndarray, denominator: np.ndarray, seed: int
) -> tuple[float, list[float]]:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    require(numerator.shape == denominator.shape and numerator.ndim == 1, "bootstrap shapes")
    require(np.all(numerator >= 0.0) and np.all(denominator > 0.0), "bootstrap errors")
    ratio = float(numerator.mean() / denominator.mean())
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(numerator), size=(BOOTSTRAP_REPLICATES, len(numerator)))
    values = numerator[indices].mean(axis=1) / denominator[indices].mean(axis=1)
    interval = np.quantile(values, [0.025, 0.975], method="linear")
    return ratio, [float(interval[0]), float(interval[1])]


def contrast(
    numerator: np.ndarray,
    denominator: np.ndarray,
    encoder: str,
    name: str,
) -> dict[str, Any]:
    ratio, interval = bootstrap_ratio(
        numerator, denominator, hash_seed("order-proxy-residual-bootstrap", encoder, name)
    )
    wins = int(np.sum(numerator < denominator))
    ties = int(np.sum(numerator == denominator))
    return {
        "ratio_of_mean_errors": ratio,
        "bootstrap_95_percentile_interval": interval,
        "wins": wins,
        "win_fraction": wins / len(numerator),
        "ties": ties,
        "cells": len(numerator),
    }


def configure_torch() -> Any:
    import torch

    require(torch.cuda.is_available(), "CUDA is required")
    torch.manual_seed(260_903_397_623)
    torch.cuda.manual_seed_all(260_903_397_623)
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

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "patient_set_diffusion"))
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
    chunks = []
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
    replay_chunks = []
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
    require(features.shape == (TOTAL_IMAGES, 768), "DINO feature shape")
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
    del model, state, chunks, replay_chunks, replay, tensor, output
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
    chunks = []
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
    replay_chunks = []
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
    require(features.shape == (TOTAL_IMAGES, 768), "RAD-DINO feature shape")
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
        "same_source_pretraining_limit": "RAD-DINO was pretrained on NIH ChestXray14; require generic DINO gate too",
    }
    del model, processor, chunks, replay_chunks, replay, images, inputs, output
    gc.collect()
    torch.cuda.empty_cache()
    return features, metadata


def analyze_encoder(
    features: np.ndarray, units: list[dict[str, Any]], encoder: str
) -> tuple[dict[str, Any], dict[str, bool]]:
    transitions = np.stack([transition_vector(unit) for unit in units], axis=0)
    deltas = np.stack(
        [features[2 * index + 1] - features[2 * index] for index in range(len(units))], axis=0
    )
    train_indices = [index for index, unit in enumerate(units) if unit["split"] == "train"]
    validation_indices = [index for index, unit in enumerate(units) if unit["split"] == "validation"]
    train_units = [units[index] for index in train_indices]
    x_train = transitions[train_indices]
    y_train = deltas[train_indices]
    true_model, true_cv = choose_ridge(x_train, y_train, train_units)
    deranged_x, derangement = derange_transition_rows(x_train, train_units)
    deranged_model, deranged_cv = choose_ridge(deranged_x, y_train, train_units)

    x_validation = transitions[validation_indices]
    y_validation = deltas[validation_indices]
    true_prediction = predict_ridge(true_model, x_validation)
    deranged_prediction = predict_ridge(deranged_model, x_validation)
    copy_error = np.mean(y_validation**2, axis=1)
    true_error = np.mean((y_validation - true_prediction) ** 2, axis=1)
    deranged_error = np.mean((y_validation - deranged_prediction) ** 2, axis=1)
    shuffled_error = np.empty_like(true_error)
    matching = []
    for local_index, unit_index in enumerate(validation_indices):
        donor_index, match = choose_shuffled_prior(unit_index, validation_indices, units)
        predicted_future = features[2 * donor_index] + true_prediction[local_index]
        shuffled_error[local_index] = np.mean(
            (features[2 * unit_index + 1] - predicted_future) ** 2
        )
        matching.append(match)

    class_local = {
        name: np.asarray(
            [
                local
                for local, index in enumerate(validation_indices)
                if units[index]["transition_class"] == name
            ],
            dtype=np.int64,
        )
        for name in ("change", "no_change")
    }
    require(all(len(indices) == VALIDATION_PER_CLASS for indices in class_local.values()), "validation classes")

    report: dict[str, Any] = {
        "true_transition_ridge": true_cv,
        "deranged_training_ridge": deranged_cv,
        "derangement": derangement,
        "classes": {},
    }
    gates: dict[str, bool] = {}
    for name, indices in class_local.items():
        true_copy = contrast(true_error[indices], copy_error[indices], encoder, f"{name}-true-copy")
        true_deranged = contrast(
            true_error[indices], deranged_error[indices], encoder, f"{name}-true-deranged"
        )
        correct_shuffled = contrast(
            true_error[indices], shuffled_error[indices], encoder, f"{name}-correct-shuffled"
        )
        class_matches = [matching[index] for index in indices]
        report["classes"][name] = {
            "errors": {
                "copy": describe(copy_error[indices]),
                "true_transition": describe(true_error[indices]),
                "deranged_training": describe(deranged_error[indices]),
                "shuffled_prior": describe(shuffled_error[indices]),
            },
            "true_transition_over_copy": true_copy,
            "true_transition_over_deranged_training": true_deranged,
            "correct_over_shuffled_prior": correct_shuffled,
            "shuffled_matching": {
                "same_sex_fraction": float(np.mean([item["same_sex"] for item in class_matches])),
                "previous_exact_label_fraction": float(
                    np.mean([item["previous_exact_label"] for item in class_matches])
                ),
                "future_exact_label_fraction": float(
                    np.mean([item["future_exact_label"] for item in class_matches])
                ),
                "previous_label_jaccard_mean": float(
                    np.mean([item["previous_label_jaccard"] for item in class_matches])
                ),
                "future_label_jaccard_mean": float(
                    np.mean([item["future_label_jaccard"] for item in class_matches])
                ),
            },
        }

    change = report["classes"]["change"]
    no_change = report["classes"]["no_change"]
    first = change["true_transition_over_copy"]
    second = change["true_transition_over_deranged_training"]
    third = change["correct_over_shuffled_prior"]
    fourth = no_change["true_transition_over_copy"]
    gates.update(
        {
            "change_true_over_copy_ratio_at_most_0_995": first["ratio_of_mean_errors"] <= 0.995,
            "change_true_over_copy_upper_ci_below_1": first["bootstrap_95_percentile_interval"][1] < 1.0,
            "change_true_over_copy_wins_at_least_0_60": first["win_fraction"] >= 0.60,
            "change_true_over_deranged_ratio_at_most_0_995": second["ratio_of_mean_errors"] <= 0.995,
            "change_true_over_deranged_upper_ci_below_1": second["bootstrap_95_percentile_interval"][1] < 1.0,
            "change_true_over_deranged_wins_at_least_0_60": second["win_fraction"] >= 0.60,
            "change_correct_over_shuffled_ratio_at_most_0_95": third["ratio_of_mean_errors"] <= 0.95,
            "change_correct_over_shuffled_upper_ci_below_1": third["bootstrap_95_percentile_interval"][1] < 1.0,
            "change_correct_over_shuffled_wins_at_least_0_75": third["win_fraction"] >= 0.75,
            "no_change_true_over_copy_at_most_1_01": fourth["ratio_of_mean_errors"] <= 1.01,
        }
    )
    report["gate_checks"] = gates
    report["gate_pass"] = all(gates.values())
    return report, gates


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

    exclusion_paths = {name: getattr(args, name).resolve() for name in EXCLUSION_HASHES}
    exclusion_sets = {
        name: read_patient_ids(exclusion_paths[name], expected)
        for name, expected in EXCLUSION_HASHES.items()
    }
    excluded: set[str] = set().union(*exclusion_sets.values())
    require(len(excluded) == EXPECTED_EXCLUDED_PATIENTS, "exclusion union count drift")
    records = read_manifest(args.manifest.resolve())
    units, eligibility = select_cohort(records, excluded)
    flat_records = [record for unit in units for record in (unit["previous"], unit["current"])]
    image_root = args.image_root.resolve()
    require(all((image_root / record.image_filename).is_file() for record in flat_records), "missing image")

    torch = configure_torch()
    total_started = time.perf_counter()
    dino_features, dino_metadata = extract_dino(
        flat_records, image_root, args.dino_weights.resolve(), torch
    )
    dino_result, dino_gates = analyze_encoder(dino_features, units, "DINOv2")
    del dino_features
    gc.collect()
    torch.cuda.empty_cache()
    rad_features, rad_metadata = extract_rad_dino(flat_records, image_root, rad_snapshot, torch)
    rad_result, rad_gates = analyze_encoder(rad_features, units, "RAD-DINO")
    del rad_features
    gc.collect()
    torch.cuda.empty_cache()
    elapsed = time.perf_counter() - total_started

    all_substantive = all(dino_gates.values()) and all(rad_gates.values())
    resource_pass = max(dino_metadata["peak_cuda_bytes"], rad_metadata["peak_cuda_bytes"]) < MEMORY_LIMIT_BYTES
    status = (
        "PASS_NIH_ORDER_PROXY_RESIDUAL_PREMISE"
        if all_substantive and resource_pass
        else "FAIL_NIH_ORDER_PROXY_RESIDUAL_PREMISE"
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    selection_path = output_dir / "selected_public_ordered_pairs_private.csv"
    write_selection(selection_path, units)
    report = {
        "schema": SCHEMA,
        "status": status,
        "scope": "PUBLIC_NONDP_FEATURE_PREMISE_NOT_GENERATION_NOT_TIME_NOT_PRIVACY_NOT_CLINICAL",
        "frozen_inputs": {
            "protocol_sha256": PROTOCOL_SHA256,
            "implementation_sha256": sha256_file(source_path),
            "manifest_sha256": K10_SHA256,
            "dino_weights_sha256": DINO_SHA256,
            "rad_revision": RAD_REVISION,
            "rad_weight_sha256": RAD_WEIGHT_SHA256,
            "rad_config_sha256": RAD_CONFIG_SHA256,
            "rad_processor_sha256": RAD_PROCESSOR_SHA256,
            "selection_salts": {
                "pair": PAIR_SALT,
                "patient": PATIENT_SALTS,
            },
            "exclusion_selection_sha256": EXCLUSION_HASHES,
        },
        "selection": {
            "excluded_unique_patients": len(excluded),
            "exclusion_source_patient_counts": {
                name: len(values) for name, values in exclusion_sets.items()
            },
            **eligibility,
            "patients": TOTAL_PATIENTS,
            "images": TOTAL_IMAGES,
            "pairs": TOTAL_PATIENTS,
            "class_split_counts": {
                f"{split}_{name}": sum(
                    unit["split"] == split and unit["transition_class"] == name for unit in units
                )
                for split in ("train", "validation")
                for name in ("no_change", "change")
            },
            "selection_file": selection_path.name,
            "selection_sha256": sha256_file(selection_path),
        },
        "encoders": {
            "DINOv2": {"execution": dino_metadata, "analysis": dino_result},
            "RAD-DINO": {"execution": rad_metadata, "analysis": rad_result},
        },
        "gate_checks": {
            "DINOv2_all_substantive": all(dino_gates.values()),
            "RAD_DINO_all_substantive": all(rad_gates.values()),
            "peak_memory_below_7_5_gib": resource_pass,
            "all_conjunctive": all_substantive and resource_pass,
        },
        "execution": {
            "elapsed_feature_and_analysis_seconds": elapsed,
            "peak_cuda_memory_bytes": max(
                dino_metadata["peak_cuda_bytes"], rad_metadata["peak_cuda_bytes"]
            ),
            "peak_cuda_memory_gib": max(
                dino_metadata["peak_cuda_bytes"], rad_metadata["peak_cuda_bytes"]
            )
            / 1024**3,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "feature_retained": False,
            "regressor_retained": False,
            "checkpoint_written": False,
            "latent_retained": False,
            "gradient_retained": False,
            "optimizer_created": False,
            "generated_image_written": False,
        },
        "interpretation_limit": (
            "A pass licenses only a timestamp/report data-access gate. A failure closes the present NIH "
            "weak-label order-proxy PPLRD route, not every richer-EHR longitudinal model."
        ),
    }
    write_json(output_dir / "report.json", report)
    print(status)
    print(json.dumps(report["gate_checks"], sort_keys=True))
    for encoder in ("DINOv2", "RAD-DINO"):
        change = report["encoders"][encoder]["analysis"]["classes"]["change"]
        print(
            encoder,
            "true/copy=",
            change["true_transition_over_copy"]["ratio_of_mean_errors"],
            "true/deranged=",
            change["true_transition_over_deranged_training"]["ratio_of_mean_errors"],
            "correct/shuffled=",
            change["correct_over_shuffled_prior"]["ratio_of_mean_errors"],
        )
    return 0 if status.startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
