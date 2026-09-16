#!/usr/bin/env python3
"""Run the frozen independent scale-free patient-set premise confirmation."""

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

import run_nih_cxr14_patient_set_premise as premise_v1


SCHEMA = "nih-cxr14-patient-set-premise-confirmation/v2"
SELECTION_SALT = "nih-cxr14-patient-set-premise-confirmation-v2"
PROTOCOL_SHA256 = "1ABE0C90DCF91D9FC83635358D1533381DA115994C68959976274C792240E212"
DINO_SHA256 = premise_v1.DINO_SHA256
CALIBRATION_SELECTION_SHA256 = premise_v1.CALIBRATION_SELECTION_SHA256
UCAN_SELECTION_SHA256 = premise_v1.UCAN_SELECTION_SHA256
PREMISE_V1_SELECTION_SHA256 = "87B356A6175493E309DE2D84E8D0769B4764005BEE809AA8EB5CF5D5EF544AB0"
PATIENTS_PER_CELL = 40
RECORD_COUNTS = (2, 3)
TARGETS = (0, 1)
TOTAL_PATIENTS = PATIENTS_PER_CELL * len(RECORD_COUNTS) * len(TARGETS)
TOTAL_IMAGES = PATIENTS_PER_CELL * sum(RECORD_COUNTS) * len(TARGETS)
IMAGE_SIZE = 518
BATCH_SIZE = 4


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def default_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parent.parent
    return {
        "root": root,
        "manifest": root / "_data" / "derived" / "nih_cxr14_pa_target_enriched_v1" / "k10_private.csv",
        "image_root": root / "_data" / "raw" / "nih_cxr14_pa_k10_plus_census_v1" / "images",
        "calibration_selection": root / "_reports" / "nih_cxr14_public_clip_calibration_v1_001" / "selected_public_units_private.csv",
        "ucan_selection": root / "_reports" / "nih_cxr14_ucan_public_diagnostic_v1_001" / "selected_public_patients_private.csv",
        "premise_v1_selection": root / "_reports" / "nih_cxr14_patient_set_premise_v1_001" / "selected_public_patient_sets_private.csv",
        "protocol": root / "patient_set_diffusion" / "NIH_CXR14_PATIENT_SET_PREMISE_CONFIRMATION_PROTOCOL_V2.md",
        "dino_weights": premise_v1.default_paths()["dino_weights"],
        "output": root / "_reports" / "nih_cxr14_patient_set_premise_confirmation_v2_001",
    }


def parse_args() -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser()
    for name in (
        "manifest",
        "image_root",
        "calibration_selection",
        "ucan_selection",
        "premise_v1_selection",
        "protocol",
        "dino_weights",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=paths[name])
    parser.add_argument("--output-dir", type=Path, default=paths["output"])
    return parser.parse_args()


def read_patient_ids(path: Path, expected_sha256: str, sha256_file: Any) -> set[str]:
    require(sha256_file(path) == expected_sha256, f"selection hash mismatch: {path.name}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["patient_id"] for row in csv.DictReader(handle)}


def cell_key(target: int, record_count: int) -> str:
    return f"target{target}_n{record_count}"


def select_units(records: list[Any], excluded: set[str]) -> list[dict[str, Any]]:
    by_patient: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        by_patient[str(record.patient_id)].append(record)

    units: list[dict[str, Any]] = []
    for target in TARGETS:
        for record_count in RECORD_COUNTS:
            eligible = [
                patient_id
                for patient_id, patient_records in by_patient.items()
                if patient_id not in excluded
                and len(patient_records) == record_count
                and int(patient_records[0].target_patient) == target
            ]
            ranked = sorted(
                eligible,
                key=lambda patient_id: (
                    premise_v1.stable_hash(SELECTION_SALT, "patient", target, record_count, patient_id),
                    patient_id,
                ),
            )
            require(
                len(ranked) >= PATIENTS_PER_CELL,
                f"only {len(ranked)} eligible target={target}, n={record_count} patients",
            )
            for patient_id in ranked[:PATIENTS_PER_CELL]:
                patient_records = sorted(by_patient[patient_id], key=lambda record: record.image_id)
                require(len(patient_records) == record_count, "record-count stratum changed")
                require(
                    all(int(record.target_patient) == target for record in patient_records),
                    "target flag changes within patient",
                )
                units.append(
                    {
                        "patient_id": patient_id,
                        "target_patient": target,
                        "record_count": record_count,
                        "records": patient_records,
                    }
                )

    require(len(units) == TOTAL_PATIENTS, "cohort patient count mismatch")
    require(sum(unit["record_count"] for unit in units) == TOTAL_IMAGES, "cohort image count mismatch")
    require(len({unit["patient_id"] for unit in units}) == TOTAL_PATIENTS, "duplicate patient")
    require({unit["patient_id"] for unit in units}.isdisjoint(excluded), "excluded patient selected")
    observed = defaultdict(int)
    for unit in units:
        observed[(unit["target_patient"], unit["record_count"])] += 1
    require(
        all(observed[(target, count)] == PATIENTS_PER_CELL for target in TARGETS for count in RECORD_COUNTS),
        "cell balance mismatch",
    )
    return units


def write_selection(path: Path, units: list[dict[str, Any]]) -> None:
    fields = [
        "patient_id",
        "target_patient",
        "record_count",
        "image_ids",
        "finding_labels",
        "followup_numbers",
        "selection_commitment",
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
                    "record_count": unit["record_count"],
                    "image_ids": "|".join(record.image_id for record in records),
                    "finding_labels": "||".join(premise_v1.canonical_labels(record) for record in records),
                    "followup_numbers": "|".join(
                        str(premise_v1.followup_number(record)) for record in records
                    ),
                    "selection_commitment": premise_v1.stable_hash(
                        SELECTION_SALT, unit["patient_id"], *(record.image_id for record in records)
                    ),
                }
            )


def extract_features(model: Any, records: list[Any], image_root: Path) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    output = []
    with torch.inference_mode():
        for start in range(0, len(records), BATCH_SIZE):
            batch_records = records[start : start + BATCH_SIZE]
            batch = torch.stack(
                [premise_v1.preprocess_image(image_root / record.image_filename) for record in batch_records]
            ).to("cuda")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                feature = model(batch)
            output.append(functional.normalize(feature.float(), dim=1).cpu())
    matrix = torch.cat(output, dim=0).numpy().astype(np.float64)
    require(matrix.shape == (TOTAL_IMAGES, 768), f"unexpected feature shape: {matrix.shape}")
    require(np.all(np.isfinite(matrix)), "non-finite DINO features")
    require(float(np.max(np.abs(np.linalg.norm(matrix, axis=1) - 1.0))) < 1e-5, "features not normalized")
    return matrix


def balanced_auc(
    similarity: np.ndarray,
    positives: list[tuple[int, int]],
    negatives: list[tuple[int, int]],
    records: list[Any],
    salt_tag: str,
) -> tuple[float, list[tuple[int, int]]]:
    require(positives and len(negatives) >= len(positives), f"insufficient pairs for {salt_tag}")
    selected = sorted(
        negatives,
        key=lambda pair: (
            premise_v1.stable_hash(
                SELECTION_SALT,
                salt_tag,
                records[pair[0]].image_id,
                records[pair[1]].image_id,
            ),
            pair,
        ),
    )[: len(positives)]
    positive_scores = [float(similarity[left, right]) for left, right in positives]
    negative_scores = [float(similarity[left, right]) for left, right in selected]
    return premise_v1.rank_auc(positive_scores, negative_scores), selected


def retrieval_metrics(
    similarity: np.ndarray,
    candidate_indices: list[int],
    patient_ids: list[str],
) -> dict[str, float | int]:
    hits_1 = 0
    hits_5 = 0
    random_r1_terms = []
    random_r5_terms = []
    candidate_count = len(candidate_indices)
    require(candidate_count > 5, "too few retrieval candidates")
    for index in candidate_indices:
        peers = [candidate for candidate in candidate_indices if patient_ids[candidate] == patient_ids[index]]
        same_count = len(peers) - 1
        require(same_count >= 1, "singleton entered retrieval")
        candidates = [candidate for candidate in candidate_indices if candidate != index]
        order = sorted(candidates, key=lambda candidate: (-similarity[index, candidate], candidate))
        hits_1 += patient_ids[order[0]] == patient_ids[index]
        hits_5 += any(patient_ids[candidate] == patient_ids[index] for candidate in order[:5])
        random_r1_terms.append(same_count / (candidate_count - 1))
        random_r5_terms.append(
            1.0 - math.comb(candidate_count - 1 - same_count, 5) / math.comb(candidate_count - 1, 5)
        )
    recall_1 = hits_1 / candidate_count
    recall_5 = hits_5 / candidate_count
    random_r1 = float(np.mean(random_r1_terms))
    random_r5 = float(np.mean(random_r5_terms))
    return {
        "queries": candidate_count,
        "recall_at_1": recall_1,
        "recall_at_5": recall_5,
        "random_candidate_recall_at_1": random_r1,
        "random_candidate_recall_at_5": random_r5,
        "recall_at_1_chance_multiple": recall_1 / random_r1,
    }


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "dp_protocol"))
    import calibrate_xray_public_clip_norms as calibration

    require(calibration.sha256_file(args.protocol.resolve()) == PROTOCOL_SHA256, "protocol hash mismatch")
    require(calibration.sha256_file(args.manifest.resolve()) == calibration.K10_SHA256, "manifest hash mismatch")
    require(calibration.sha256_file(args.dino_weights.resolve()) == DINO_SHA256, "DINO weight hash mismatch")

    import torch

    require(torch.cuda.is_available(), "CUDA is required for the DINO confirmation")
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    sys.path.insert(0, str(root / "data_pipeline"))
    import nih_cxr14_model_input as model_input

    records = model_input.read_manifest(args.manifest.resolve(), partitions={"public_development"})
    exclusion_sources = {
        "clip_calibration": (args.calibration_selection.resolve(), CALIBRATION_SELECTION_SHA256),
        "ucan": (args.ucan_selection.resolve(), UCAN_SELECTION_SHA256),
        "premise_v1": (args.premise_v1_selection.resolve(), PREMISE_V1_SELECTION_SHA256),
    }
    excluded_parts = {
        name: read_patient_ids(path, expected_hash, calibration.sha256_file)
        for name, (path, expected_hash) in exclusion_sources.items()
    }
    excluded = set().union(*excluded_parts.values())
    require(len(excluded) == 240, f"expected 240 unique exclusions, got {len(excluded)}")
    units = select_units(records, excluded)

    flat_records: list[Any] = []
    for unit in units:
        unit["indices"] = list(range(len(flat_records), len(flat_records) + unit["record_count"]))
        flat_records.extend(unit["records"])

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    selection_path = output_dir / "selected_public_patient_sets_private.csv"
    write_selection(selection_path, units)

    model, model_metadata = premise_v1.load_dino(args.dino_weights.resolve())
    started = time.perf_counter()
    features = extract_features(model, flat_records, args.image_root.resolve())
    elapsed = time.perf_counter() - started
    del model
    torch.cuda.empty_cache()

    similarity = features @ features.T
    del features
    patient_ids = [str(record.patient_id) for record in flat_records]
    labels = [premise_v1.canonical_labels(record) for record in flat_records]
    index_to_cell: dict[int, str] = {}
    units_by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        key = cell_key(unit["target_patient"], unit["record_count"])
        units_by_cell[key].append(unit)
        for index in unit["indices"]:
            index_to_cell[index] = key

    positives_by_cell: dict[str, list[tuple[int, int]]] = defaultdict(list)
    negatives_by_cell: dict[str, list[tuple[int, int]]] = defaultdict(list)
    within_same_label: list[tuple[int, int]] = []
    within_different_label: list[tuple[int, int]] = []
    cross_same_exact_label: list[tuple[int, int]] = []
    for key, cell_units in units_by_cell.items():
        cell_indices = [index for unit in cell_units for index in unit["indices"]]
        for unit in cell_units:
            for pair in itertools.combinations(unit["indices"], 2):
                positives_by_cell[key].append(pair)
                if labels[pair[0]] == labels[pair[1]]:
                    within_same_label.append(pair)
                else:
                    within_different_label.append(pair)
        for left, right in itertools.combinations(cell_indices, 2):
            if patient_ids[left] != patient_ids[right]:
                negatives_by_cell[key].append((left, right))
                if labels[left] == labels[right]:
                    cross_same_exact_label.append((left, right))

    cell_reports: dict[str, Any] = {}
    pooled_positives: list[tuple[int, int]] = []
    pooled_selected_negatives: list[tuple[int, int]] = []
    patient_wins = []
    for key in sorted(units_by_cell):
        auc, selected_negatives = balanced_auc(
            similarity,
            positives_by_cell[key],
            negatives_by_cell[key],
            flat_records,
            f"balanced_negative_{key}",
        )
        pooled_positives.extend(positives_by_cell[key])
        pooled_selected_negatives.extend(selected_negatives)

        wins = []
        margins = []
        cell_indices = [index for unit in units_by_cell[key] for index in unit["indices"]]
        for unit in units_by_cell[key]:
            within = [
                float(similarity[left, right]) for left, right in itertools.combinations(unit["indices"], 2)
            ]
            cross = [
                float(similarity[left, other])
                for left in unit["indices"]
                for other in cell_indices
                if patient_ids[other] != unit["patient_id"]
            ]
            margin = float(np.mean(within) - np.mean(cross))
            margins.append(margin)
            wins.append(margin > 0.0)
        patient_wins.extend(wins)
        retrieval = retrieval_metrics(similarity, cell_indices, patient_ids)
        cell_reports[key] = {
            "patients": len(units_by_cell[key]),
            "images": len(cell_indices),
            "positive_pairs": len(positives_by_cell[key]),
            "available_cross_pairs": len(negatives_by_cell[key]),
            "balanced_similarity_auc": auc,
            "patient_ordering_win_rate": float(np.mean(wins)),
            "patient_within_minus_cross_margin": premise_v1.describe(margins),
            "retrieval": retrieval,
        }

    pooled_positive_scores = [float(similarity[left, right]) for left, right in pooled_positives]
    pooled_negative_scores = [float(similarity[left, right]) for left, right in pooled_selected_negatives]
    pooled_auc = premise_v1.rank_auc(pooled_positive_scores, pooled_negative_scores)
    global_retrieval = retrieval_metrics(similarity, list(range(TOTAL_IMAGES)), patient_ids)

    distinct_label_counts = []
    label_jaccard = []
    changed_patients = 0
    followup_spans = []
    for unit in units:
        unit_labels = [premise_v1.canonical_labels(record) for record in unit["records"]]
        distinct = len(set(unit_labels))
        distinct_label_counts.append(distinct)
        changed_patients += distinct > 1
        label_jaccard.extend(
            premise_v1.jaccard_distance(left, right)
            for left, right in itertools.combinations(unit_labels, 2)
        )
        followups = [premise_v1.followup_number(record) for record in unit["records"]]
        followup_spans.append(max(followups) - min(followups))

    changed_fraction = changed_patients / TOTAL_PATIENTS
    mean_jaccard = float(np.mean(label_jaccard))
    overall_win_rate = float(np.mean(patient_wins))
    gate_checks = {
        "pooled_balanced_auc_at_least_0_75": pooled_auc >= 0.75,
        "every_cell_auc_at_least_0_65": all(
            report["balanced_similarity_auc"] >= 0.65 for report in cell_reports.values()
        ),
        "patient_win_rate_overall_0_75_and_every_cell_0_60": (
            overall_win_rate >= 0.75
            and all(report["patient_ordering_win_rate"] >= 0.60 for report in cell_reports.values())
        ),
        "retrieval_global_10x_and_every_cell_5x_chance": (
            global_retrieval["recall_at_1_chance_multiple"] >= 10.0
            and all(
                report["retrieval"]["recall_at_1_chance_multiple"] >= 5.0
                for report in cell_reports.values()
            )
        ),
        "record_variation_fraction_0_30_and_jaccard_0_10": (
            changed_fraction >= 0.30 and mean_jaccard >= 0.10
        ),
    }
    passed = all(gate_checks.values())
    status = "PASS_PATIENT_SET_DATA_PREMISE_CONFIRMATION" if passed else "FAIL_PATIENT_SET_DATA_PREMISE_CONFIRMATION"

    def score_summary(pairs: list[tuple[int, int]]) -> dict[str, float | int] | None:
        if not pairs:
            return None
        return premise_v1.describe([float(similarity[left, right]) for left, right in pairs])

    report = {
        "schema": SCHEMA,
        "status": status,
        "scope": "PUBLIC_DATA_GEOMETRY_ONLY_NOT_TRAINING_NOT_DP_NOT_GENERATION_NOT_CLINICAL",
        "frozen_inputs": {
            "protocol_sha256": PROTOCOL_SHA256,
            "manifest_sha256": calibration.K10_SHA256,
            "dino_weights_sha256": DINO_SHA256,
            "selection_salt": SELECTION_SALT,
            "excluded_unique_patients": len(excluded),
            "exclusion_counts": {name: len(ids) for name, ids in excluded_parts.items()},
            "patients_per_cell": PATIENTS_PER_CELL,
            "targets": list(TARGETS),
            "record_counts": list(RECORD_COUNTS),
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
        },
        "selection": {
            "patients": len(units),
            "images": len(flat_records),
            "cells": {
                key: {"patients": len(cell_units), "images": sum(unit["record_count"] for unit in cell_units)}
                for key, cell_units in sorted(units_by_cell.items())
            },
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
        "scale_free_geometry": {
            "pooled_positive_pairs": len(pooled_positives),
            "pooled_balanced_negative_pairs": len(pooled_selected_negatives),
            "pooled_balanced_similarity_auc": pooled_auc,
            "overall_patient_ordering_win_rate": overall_win_rate,
            "global_retrieval": global_retrieval,
            "cells": cell_reports,
        },
        "label_conditioned_secondary": {
            "within_same_exact_label": score_summary(within_same_label),
            "within_different_exact_label": score_summary(within_different_label),
            "cross_same_cell_same_exact_label": score_summary(cross_same_exact_label),
        },
        "record_variation": {
            "patients_with_multiple_exact_label_sets": changed_patients,
            "fraction_with_multiple_exact_label_sets": changed_fraction,
            "distinct_exact_label_sets_per_patient": premise_v1.describe(distinct_label_counts),
            "within_patient_label_jaccard_distance": premise_v1.describe(label_jaccard),
            "followup_number_span": premise_v1.describe(followup_spans),
        },
        "gate_checks": gate_checks,
        "interpretation_limit": (
            "A pass licenses only a bounded public Q=2 architecture smoke. DINO may encode acquisition artifacts; "
            "no generator, DP utility, clinical coherence, or unlinkability was evaluated."
        ),
    }
    report_path = output_dir / "report.json"
    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        json.dumps(
            {
                "status": status,
                "pooled_balanced_similarity_auc": pooled_auc,
                "cell_auc": {
                    key: value["balanced_similarity_auc"] for key, value in cell_reports.items()
                },
                "overall_patient_ordering_win_rate": overall_win_rate,
                "cell_patient_win_rate": {
                    key: value["patient_ordering_win_rate"] for key, value in cell_reports.items()
                },
                "global_recall_at_1": global_retrieval["recall_at_1"],
                "global_recall_at_1_chance_multiple": global_retrieval["recall_at_1_chance_multiple"],
                "cell_recall_at_1_chance_multiple": {
                    key: value["retrieval"]["recall_at_1_chance_multiple"]
                    for key, value in cell_reports.items()
                },
                "changed_label_patient_fraction": changed_fraction,
                "mean_label_jaccard_distance": mean_jaccard,
                "gate_checks": gate_checks,
                "elapsed_feature_seconds": elapsed,
                "report": str(report_path),
            },
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
