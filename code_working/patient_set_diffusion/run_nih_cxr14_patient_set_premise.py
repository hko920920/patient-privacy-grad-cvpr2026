#!/usr/bin/env python3
"""Evaluate the frozen public-data premise for patient-set X-ray diffusion."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "nih-cxr14-patient-set-premise/v1"
SELECTION_SALT = "nih-cxr14-patient-set-premise-v1"
PROTOCOL_SHA256 = "02B291BA61F90E1B0E5011AAC6A687BCC6448B949520C9D12C2ECE49A440BBA4"
DINO_SHA256 = "0B8B82F85DE91B424ADED121C7E1DCC2B7BC6D0ADEEA651BF73A13307FAD8C73"
CALIBRATION_SELECTION_SHA256 = "CB7E845F7B1D627BF52973ADBA9E7F3E6207F2081441DBEFBADE7E40AE261F6E"
UCAN_SELECTION_SHA256 = "13CB2054DE2A25208972590BA023585C16555D7ECE260D4FA571633CD9681AF0"
PATIENTS_PER_TARGET = 40
RECORDS_PER_PATIENT = 4
IMAGE_SIZE = 518
BATCH_SIZE = 4


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def stable_hash(*parts: object) -> str:
    import hashlib

    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest().upper()


def default_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parent.parent
    return {
        "root": root,
        "manifest": root / "_data" / "derived" / "nih_cxr14_pa_target_enriched_v1" / "k10_private.csv",
        "image_root": root / "_data" / "raw" / "nih_cxr14_pa_k10_plus_census_v1" / "images",
        "calibration_selection": root / "_reports" / "nih_cxr14_public_clip_calibration_v1_001" / "selected_public_units_private.csv",
        "ucan_selection": root / "_reports" / "nih_cxr14_ucan_public_diagnostic_v1_001" / "selected_public_patients_private.csv",
        "protocol": root / "patient_set_diffusion" / "NIH_CXR14_PATIENT_SET_PREMISE_PROTOCOL_V1.md",
        "dino_weights": Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "dinov2_vitb14_pretrain.pth",
        "output": root / "_reports" / "nih_cxr14_patient_set_premise_v1_001",
    }


def parse_args() -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser()
    for name in ("manifest", "image_root", "calibration_selection", "ucan_selection", "protocol", "dino_weights"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=paths[name])
    parser.add_argument("--output-dir", type=Path, default=paths["output"])
    return parser.parse_args()


def read_patient_ids(path: Path, expected_sha256: str, sha256_file: Any) -> set[str]:
    require(sha256_file(path) == expected_sha256, f"selection hash mismatch: {path.name}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["patient_id"] for row in csv.DictReader(handle)}


def select_units(records: list[Any], excluded: set[str]) -> list[dict[str, Any]]:
    by_patient: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        by_patient[str(record.patient_id)].append(record)

    units: list[dict[str, Any]] = []
    for target in (0, 1):
        eligible = [
            patient_id
            for patient_id, patient_records in by_patient.items()
            if patient_id not in excluded
            and len(patient_records) >= RECORDS_PER_PATIENT
            and int(patient_records[0].target_patient) == target
        ]
        ranked = sorted(
            eligible,
            key=lambda patient_id: (stable_hash(SELECTION_SALT, "patient", target, patient_id), patient_id),
        )
        require(len(ranked) >= PATIENTS_PER_TARGET, f"only {len(ranked)} eligible target={target} patients")
        for patient_id in ranked[:PATIENTS_PER_TARGET]:
            all_records = by_patient[patient_id]
            require(all(int(record.target_patient) == target for record in all_records), "target flag changes within patient")
            chosen = sorted(
                all_records,
                key=lambda record: (
                    stable_hash(SELECTION_SALT, "record", patient_id, record.image_id),
                    record.image_id,
                ),
            )[:RECORDS_PER_PATIENT]
            units.append(
                {
                    "patient_id": patient_id,
                    "target_patient": target,
                    "eligible_image_count": len(all_records),
                    "records": chosen,
                }
            )
    require(len(units) == 80 and len({unit["patient_id"] for unit in units}) == 80, "cohort size mismatch")
    require({unit["patient_id"] for unit in units}.isdisjoint(excluded), "excluded patient selected")
    return units


def write_selection(path: Path, units: list[dict[str, Any]]) -> None:
    fields = [
        "patient_id", "target_patient", "eligible_image_count", "image_ids",
        "finding_labels", "followup_numbers", "selection_commitment",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for unit in units:
            records = unit["records"]
            writer.writerow(
                {
                    "patient_id": unit["patient_id"],
                    "target_patient": unit["target_patient"],
                    "eligible_image_count": unit["eligible_image_count"],
                    "image_ids": "|".join(record.image_id for record in records),
                    "finding_labels": "||".join(canonical_labels(record) for record in records),
                    "followup_numbers": "|".join(str(followup_number(record)) for record in records),
                    "selection_commitment": stable_hash(
                        SELECTION_SALT, unit["patient_id"], *(record.image_id for record in records)
                    ),
                }
            )


def load_dino(weights_path: Path) -> tuple[Any, dict[str, Any]]:
    import timm
    import torch

    model = timm.create_model("vit_base_patch14_dinov2", pretrained=False, num_classes=0)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    require(not incompatible.missing_keys, f"DINO missing keys: {incompatible.missing_keys}")
    require(incompatible.unexpected_keys == ["mask_token"], f"DINO unexpected keys: {incompatible.unexpected_keys}")
    model.eval().to("cuda")
    return model, {
        "architecture": "vit_base_patch14_dinov2",
        "feature_dim": int(model.num_features),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "timm_version": timm.__version__,
    }


def preprocess_image(path: Path) -> Any:
    import torch
    from PIL import Image

    with Image.open(path) as source:
        grayscale = source.convert("L")
        resized = grayscale.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.LANCZOS, reducing_gap=None)
        array = np.asarray(resized, dtype=np.float32) / 255.0
    rgb = np.stack([array, array, array], axis=0)
    tensor = torch.from_numpy(rgb)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
    return ((tensor - mean) / std).contiguous()


def extract_features(model: Any, records: list[Any], image_root: Path) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    output = []
    with torch.inference_mode():
        for start in range(0, len(records), BATCH_SIZE):
            batch_records = records[start : start + BATCH_SIZE]
            batch = torch.stack(
                [preprocess_image(image_root / record.image_filename) for record in batch_records]
            ).to("cuda")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                feature = model(batch)
            feature = functional.normalize(feature.float(), dim=1)
            output.append(feature.cpu())
    matrix = torch.cat(output, dim=0).numpy().astype(np.float64)
    require(matrix.shape == (320, 768), f"unexpected feature shape: {matrix.shape}")
    require(np.all(np.isfinite(matrix)), "non-finite DINO features")
    require(float(np.max(np.abs(np.linalg.norm(matrix, axis=1) - 1.0))) < 1e-5, "features are not normalized")
    return matrix


def describe(values: list[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    require(array.size > 0, "empty statistic")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
    }


def canonical_labels(record: Any) -> str:
    return "|".join(str(item) for item in record.finding_labels)


def followup_number(record: Any) -> int:
    return int(Path(record.image_id).stem.rsplit("_", 1)[1])


def label_set(value: str) -> set[str]:
    return {item for item in value.split("|") if item}


def jaccard_distance(left: str, right: str) -> float:
    first, second = label_set(left), label_set(right)
    union = first | second
    return 0.0 if not union else 1.0 - len(first & second) / len(union)


def rank_auc(positive: list[float], negative: list[float]) -> float:
    from scipy.stats import rankdata

    scores = np.asarray(positive + negative, dtype=np.float64)
    ranks = rankdata(scores, method="average")
    n_positive = len(positive)
    n_negative = len(negative)
    rank_sum = float(np.sum(ranks[:n_positive]))
    return (rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "dp_protocol"))
    import calibrate_xray_public_clip_norms as calibration

    require(calibration.sha256_file(args.protocol.resolve()) == PROTOCOL_SHA256, "protocol hash mismatch")
    require(calibration.sha256_file(args.manifest.resolve()) == calibration.K10_SHA256, "manifest hash mismatch")
    require(calibration.sha256_file(args.dino_weights.resolve()) == DINO_SHA256, "DINO weight hash mismatch")

    import torch

    require(torch.cuda.is_available(), "CUDA is required for the DINO premise diagnostic")
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    sys.path.insert(0, str(root / "data_pipeline"))
    import nih_cxr14_model_input as model_input

    records = model_input.read_manifest(args.manifest.resolve(), partitions={"public_development"})
    excluded = read_patient_ids(args.calibration_selection.resolve(), CALIBRATION_SELECTION_SHA256, calibration.sha256_file)
    excluded |= read_patient_ids(args.ucan_selection.resolve(), UCAN_SELECTION_SHA256, calibration.sha256_file)
    units = select_units(records, excluded)
    flat_records = [record for unit in units for record in unit["records"]]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "selected_public_patient_sets_private.csv"
    write_selection(selection_path, units)

    model, model_metadata = load_dino(args.dino_weights.resolve())
    started = time.perf_counter()
    features = extract_features(model, flat_records, args.image_root.resolve())
    elapsed = time.perf_counter() - started
    del model
    torch.cuda.empty_cache()

    similarity = features @ features.T
    patient_ids = [str(record.patient_id) for record in flat_records]
    targets = [int(record.target_patient) for record in flat_records]
    labels = [canonical_labels(record) for record in flat_records]

    within_pairs: list[tuple[int, int]] = []
    cross_same_target: list[tuple[int, int]] = []
    cross_exact_label: list[tuple[int, int]] = []
    within_same_label: list[tuple[int, int]] = []
    within_different_label: list[tuple[int, int]] = []
    for left, right in itertools.combinations(range(len(flat_records)), 2):
        same_patient = patient_ids[left] == patient_ids[right]
        if same_patient:
            within_pairs.append((left, right))
            if labels[left] == labels[right]:
                within_same_label.append((left, right))
            else:
                within_different_label.append((left, right))
        elif targets[left] == targets[right]:
            cross_same_target.append((left, right))
            if labels[left] == labels[right]:
                cross_exact_label.append((left, right))
    require(len(within_pairs) == 480, "within-patient pair count mismatch")

    def scores(pairs: list[tuple[int, int]]) -> list[float]:
        return [float(similarity[left, right]) for left, right in pairs]

    ranked_negatives = sorted(
        cross_same_target,
        key=lambda pair: (
            stable_hash(SELECTION_SALT, "balanced_negative", flat_records[pair[0]].image_id, flat_records[pair[1]].image_id),
            pair,
        ),
    )[: len(within_pairs)]
    within_scores = scores(within_pairs)
    negative_scores = scores(ranked_negatives)

    recall_1 = 0
    recall_5 = 0
    for index in range(len(flat_records)):
        row = similarity[index].copy()
        row[index] = -np.inf
        order = np.argsort(-row)
        recall_1 += patient_ids[order[0]] == patient_ids[index]
        recall_5 += any(patient_ids[candidate] == patient_ids[index] for candidate in order[:5])
    recall_1 /= len(flat_records)
    recall_5 /= len(flat_records)
    random_r1 = 3 / 319
    random_r5 = 1.0 - math.comb(316, 5) / math.comb(319, 5)

    distinct_label_counts = []
    patient_jaccard = []
    changed_patients = 0
    followup_spans = []
    for unit in units:
        unit_labels = [canonical_labels(record) for record in unit["records"]]
        distinct = len(set(unit_labels))
        distinct_label_counts.append(distinct)
        changed_patients += distinct > 1
        patient_jaccard.extend(
            jaccard_distance(left, right) for left, right in itertools.combinations(unit_labels, 2)
        )
        followups = [followup_number(record) for record in unit["records"]]
        followup_spans.append(max(followups) - min(followups))

    within_summary = describe(within_scores)
    cross_summary = describe(scores(cross_same_target))
    auc = rank_auc(within_scores, negative_scores)
    changed_fraction = changed_patients / len(units)
    mean_jaccard = float(np.mean(patient_jaccard))
    gate_checks = {
        "within_minus_cross_mean_at_least_0_05": within_summary["mean"] - cross_summary["mean"] >= 0.05,
        "balanced_similarity_auc_at_least_0_75": auc >= 0.75,
        "recall_at_1_at_least_five_times_chance": recall_1 >= 5.0 * random_r1,
        "changed_label_patient_fraction_at_least_0_30": changed_fraction >= 0.30,
        "mean_label_jaccard_distance_at_least_0_10": mean_jaccard >= 0.10,
    }
    passed = all(gate_checks.values())
    status = "PASS_PATIENT_SET_DATA_PREMISE" if passed else "FAIL_PATIENT_SET_DATA_PREMISE"
    report = {
        "schema": SCHEMA,
        "status": status,
        "scope": "PUBLIC_DATA_GEOMETRY_ONLY_NOT_TRAINING_NOT_DP_NOT_GENERATION_NOT_CLINICAL",
        "frozen_inputs": {
            "protocol_sha256": PROTOCOL_SHA256,
            "manifest_sha256": calibration.K10_SHA256,
            "dino_weights_sha256": DINO_SHA256,
            "selection_salt": SELECTION_SALT,
            "excluded_patient_count": len(excluded),
            "patients_per_target": PATIENTS_PER_TARGET,
            "records_per_patient": RECORDS_PER_PATIENT,
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
        },
        "selection": {
            "patients": len(units),
            "images": len(flat_records),
            "target_patients": sum(unit["target_patient"] == 1 for unit in units),
            "control_patients": sum(unit["target_patient"] == 0 for unit in units),
            "selection_file": selection_path.name,
            "selection_sha256": calibration.sha256_file(selection_path),
        },
        "probe": {
            **model_metadata,
            "preprocessing": "native grayscale; full-field 518x518 Lanczos; replicate RGB; ImageNet normalization",
            "feature": "L2-normalized CLS embedding",
            "feature_vectors_retained": False,
            "elapsed_feature_seconds": elapsed,
        },
        "visual_geometry": {
            "within_patient": within_summary,
            "cross_patient_same_target": cross_summary,
            "within_minus_cross_mean": float(within_summary["mean"] - cross_summary["mean"]),
            "within_patient_same_exact_label": describe(scores(within_same_label)),
            "within_patient_different_exact_label": describe(scores(within_different_label)),
            "cross_patient_same_target_same_exact_label": describe(scores(cross_exact_label)),
            "balanced_negative": describe(negative_scores),
            "balanced_similarity_auc": auc,
            "retrieval_recall_at_1": recall_1,
            "retrieval_recall_at_5": recall_5,
            "random_candidate_recall_at_1": random_r1,
            "random_candidate_recall_at_5": random_r5,
            "recall_at_1_chance_multiple": recall_1 / random_r1,
        },
        "record_variation": {
            "patients_with_multiple_exact_label_sets": changed_patients,
            "fraction_with_multiple_exact_label_sets": changed_fraction,
            "distinct_exact_label_sets_per_patient": describe(distinct_label_counts),
            "within_patient_label_jaccard_distance": describe(patient_jaccard),
            "followup_number_span": describe(followup_spans),
        },
        "gate_checks": gate_checks,
        "interpretation_limit": (
            "A pass supports only the data premise for designing a patient-set generator. DINO similarity may include "
            "acquisition artifacts and is not a clinical identity measure; no generator or patient-DP utility was tested."
        ),
    }
    report_path = output_dir / "report.json"
    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "status": status,
        "within_minus_cross_mean": report["visual_geometry"]["within_minus_cross_mean"],
        "balanced_similarity_auc": auc,
        "recall_at_1": recall_1,
        "recall_at_1_chance_multiple": recall_1 / random_r1,
        "changed_label_patient_fraction": changed_fraction,
        "mean_label_jaccard_distance": mean_jaccard,
        "elapsed_feature_seconds": elapsed,
        "report": str(report_path),
    }, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
