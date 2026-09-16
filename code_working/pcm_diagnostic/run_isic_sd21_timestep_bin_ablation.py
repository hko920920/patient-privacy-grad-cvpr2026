#!/usr/bin/env python3
"""Exact ablation of true timestep bins versus generic perturbation diversity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from run_isic_sd21_multipatient_diagnostic import bootstrap_ratio


SCHEMA = "pcm-isic-sd21-timestep-bin-ablation/v1"
PROTOCOL = "ISIC_SD21_TIMESTEP_BIN_ABLATION_PROTOCOL_V1.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def pair_partitions(values: tuple[int, ...]) -> Iterator[tuple[tuple[int, int], ...]]:
    if not values:
        yield ()
        return
    first = values[0]
    for index in range(1, len(values)):
        second = values[index]
        remainder = values[1:index] + values[index + 1 :]
        for tail in pair_partitions(remainder):
            yield ((first, second),) + tail


def enumerate_group_balanced_weights(
    records: int, perturbations: int, groups: tuple[tuple[int, ...], ...]
) -> np.ndarray:
    if len(groups) != 4:
        raise ValueError("Q=4 requires four perturbation groups")
    flattened = [value for group in groups for value in group]
    if sorted(flattened) != list(range(perturbations)):
        raise ValueError("groups must partition all perturbations")
    rows: list[np.ndarray] = []
    for record_choice in itertools.combinations(range(records), 4):
        for group_assignment in itertools.permutations(groups):
            for perturbation_assignment in itertools.product(*group_assignment):
                weights = np.zeros(records * perturbations, dtype=np.float64)
                for record, perturbation in zip(record_choice, perturbation_assignment):
                    weights[record * perturbations + perturbation] = 0.25
                rows.append(weights)
    return np.stack(rows)


def enumerate_distinct_perturbation_weights(records: int, perturbations: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    for record_choice in itertools.combinations(range(records), 4):
        for perturbation_choice in itertools.combinations(range(perturbations), 4):
            for perturbation_assignment in itertools.permutations(perturbation_choice):
                weights = np.zeros(records * perturbations, dtype=np.float64)
                for record, perturbation in zip(record_choice, perturbation_assignment):
                    weights[record * perturbations + perturbation] = 0.25
                rows.append(weights)
    return np.stack(rows)


def weight_covariance(weights: np.ndarray) -> np.ndarray:
    reference = np.full(weights.shape[1], 1.0 / weights.shape[1])
    centered = weights - reference
    return centered.T @ centered / weights.shape[0]


def mse_from_covariance(covariance: np.ndarray, gram: np.ndarray, dimension: int) -> float:
    return float(np.sum(covariance * gram) / dimension)


def geometric_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.exp(np.mean(np.log(array))))


def build_markdown(result: dict[str, Any], patients: list[dict[str, Any]]) -> str:
    primary = result["primary"]
    summary = result["summary"]
    lines = [
        "# ISIC SD 2.1 timestep-bin mechanism ablation",
        "",
        f"- Execution status: **{result['status']}**",
        f"- Incremental decision: **{result['decision']}**",
        "- Scope: exact analysis of stored confirmatory Gram matrices",
        "",
        "## Frozen incremental gate",
        "",
        f"- True-bin/distinct-only geometric MSE ratio: `{primary['geometric_mean_ratio']:.4f}`",
        f"- Patient bootstrap 95% interval: `[{primary['bootstrap_ci95_lower']:.4f}, "
        f"{primary['bootstrap_ci95_upper']:.4f}]`",
        f"- True-bin wins: `{primary['true_bin_wins']}/16`",
        "",
    ]
    for condition in result["conditions"]:
        lines.append(
            f"- `{condition['name']}`: **{condition['status']}** "
            f"(observed={condition['observed']}, required={condition['required']})"
        )
    lines.extend(
        [
            "",
            "## Mechanism decomposition",
            "",
            f"- Distinct-only/replacement geometric ratio: "
            f"`{summary['distinct_to_replacement_geometric_ratio']:.4f}`",
            f"- True-bin/replacement geometric ratio: "
            f"`{summary['true_bin_to_replacement_geometric_ratio']:.4f}`",
            f"- True chronological grouping median rank among 105 pairings: "
            f"`{summary['true_grouping_median_rank']:.2f}`",
            f"- True-bin/mean-pairing geometric ratio: "
            f"`{summary['true_to_mean_pairing_geometric_ratio']:.4f}`",
            "",
            "## Patient results",
            "",
            "| Patient | True/distinct | Distinct/replacement | True grouping rank / 105 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in patients:
        lines.append(
            f"| {row['patient_id']} | {row['true_to_distinct_ratio']:.4f} | "
            f"{row['distinct_to_replacement_ratio']:.4f} | {row['true_grouping_rank']} |"
        )
    lines.extend(
        [
            "",
            "This ablation separates generic finite-population diversity from chronological "
            "timestep-bin structure. It does not establish DP utility, training quality, novelty, "
            "or venue suitability.",
            "",
        ]
    )
    return "\n".join(lines)


def run(report_dir: Path) -> dict[str, Any]:
    report_dir = report_dir.resolve()
    source = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads((report_dir / "secondary_audit.json").read_text(encoding="utf-8"))
    specs = read_csv(report_dir / "perturbation_specs.csv")
    source_metrics = read_csv(report_dir / "patient_preclip_metrics.csv")
    metric_map = {
        (row["patient_id"], row["method"]): float(row["preclip_mse"])
        for row in source_metrics
    }
    with np.load(report_dir / "patient_gradient_grams_local_only.npz") as archive:
        grams = {name: np.asarray(archive[name], dtype=np.float64) for name in archive.files}
    patients = source["selection"]["selected"]
    dimension = int(source["model"]["parameter_dimension"])

    true_groups = tuple(
        tuple(
            int(row["perturbation_index"])
            for row in specs
            if int(row["bin_index"]) == bin_index
        )
        for bin_index in range(4)
    )
    all_pairings = list(pair_partitions(tuple(range(8))))
    distinct_weights = enumerate_distinct_perturbation_weights(5, 8)
    true_weights = enumerate_group_balanced_weights(5, 8, true_groups)
    distinct_covariance = weight_covariance(distinct_weights)
    grouping_covariances = [
        weight_covariance(enumerate_group_balanced_weights(5, 8, groups))
        for groups in all_pairings
    ]
    true_index = all_pairings.index(true_groups)
    true_covariance = grouping_covariances[true_index]
    reference = np.full(40, 1.0 / 40.0)
    maximum_expected_weight_error = max(
        float(np.max(np.abs(np.mean(distinct_weights, axis=0) - reference))),
        float(np.max(np.abs(np.mean(true_weights, axis=0) - reference))),
    )

    patient_rows: list[dict[str, Any]] = []
    maximum_source_relative_error = 0.0
    for patient in patients:
        patient_id = patient["patient_id"]
        gram = grams[patient_id]
        replacement = metric_map[(patient_id, "replacement_uniform_m4")]
        source_true = metric_map[(patient_id, "bin_balanced_m4")]
        distinct = mse_from_covariance(distinct_covariance, gram, dimension)
        grouping_values = [
            mse_from_covariance(covariance, gram, dimension)
            for covariance in grouping_covariances
        ]
        true_value = grouping_values[true_index]
        maximum_source_relative_error = max(
            maximum_source_relative_error,
            abs(true_value - source_true) / max(source_true, 1e-30),
        )
        true_rank = 1 + sum(value < true_value for value in grouping_values)
        mean_pairing = float(np.mean(grouping_values))
        patient_rows.append(
            {
                "patient_id": patient_id,
                "site_group": patient["site_group"],
                "replacement_mse": replacement,
                "distinct_mse": distinct,
                "true_bin_mse": true_value,
                "mean_pairing_mse": mean_pairing,
                "true_to_distinct_ratio": true_value / distinct,
                "distinct_to_replacement_ratio": distinct / replacement,
                "true_to_replacement_ratio": true_value / replacement,
                "true_to_mean_pairing_ratio": true_value / mean_pairing,
                "true_grouping_rank": true_rank,
            }
        )
    write_csv(report_dir / "timestep_bin_ablation_patients.csv", patient_rows)

    ratios = [row["true_to_distinct_ratio"] for row in patient_rows]
    bootstrap = bootstrap_ratio(np.asarray(ratios, dtype=np.float64))
    true_bin_wins = int(sum(value < 1.0 for value in ratios))
    conditions = [
        {
            "name": "true_bin_to_distinct_geometric_ratio_le_0.95",
            "status": "PASS"
            if bootstrap["geometric_mean_ratio"] <= 0.95
            else "FAIL",
            "observed": bootstrap["geometric_mean_ratio"],
            "required": "<=0.95",
        },
        {
            "name": "bootstrap_upper_lt_1.00",
            "status": "PASS"
            if bootstrap["bootstrap_ci95_upper"] < 1.0
            else "FAIL",
            "observed": bootstrap["bootstrap_ci95_upper"],
            "required": "<1.00",
        },
        {
            "name": "true_bin_wins_ge_10_of_16",
            "status": "PASS" if true_bin_wins >= 10 else "FAIL",
            "observed": true_bin_wins,
            "required": ">=10",
        },
    ]
    decision = (
        "TIMESTEP_BIN_INCREMENT_RETAINS_SIGNAL"
        if all(row["status"] == "PASS" for row in conditions)
        else "TIMESTEP_BIN_INCREMENT_NOT_SUPPORTED"
    )
    execution_checks = [
        {
            "name": "source_and_secondary_audit_pass",
            "status": "PASS"
            if source["status"] == audit["status"] == "PASS"
            else "FAIL",
        },
        {
            "name": "exact_105_pair_partitions",
            "status": "PASS"
            if len(all_pairings) == 105 and len(set(all_pairings)) == 105
            else "FAIL",
        },
        {
            "name": "expected_state_counts",
            "status": "PASS"
            if distinct_weights.shape == (8400, 40)
            and true_weights.shape == (1920, 40)
            else "FAIL",
        },
        {
            "name": "estimators_unbiased",
            "status": "PASS"
            if maximum_expected_weight_error <= 1e-12
            else "FAIL",
            "maximum_expected_weight_error": maximum_expected_weight_error,
        },
        {
            "name": "source_true_bin_mse_reproduced",
            "status": "PASS"
            if maximum_source_relative_error <= 1e-10
            else "FAIL",
            "maximum_relative_error": maximum_source_relative_error,
        },
    ]
    status = (
        "PASS"
        if all(row["status"] == "PASS" for row in execution_checks)
        else "FAIL"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "decision": decision,
        "protocol": {
            "file": PROTOCOL,
            "sha256": sha256_file(Path(__file__).with_name(PROTOCOL)),
        },
        "boundary": (
            "Mechanism ablation on stored finite-bank gradients; no training, DP, generation, "
            "clinical, novelty, or venue claim."
        ),
        "execution_checks": execution_checks,
        "conditions": conditions,
        "primary": {
            **bootstrap,
            "true_bin_wins": true_bin_wins,
            "patients": len(patient_rows),
        },
        "summary": {
            "distinct_to_replacement_geometric_ratio": geometric_mean(
                [row["distinct_to_replacement_ratio"] for row in patient_rows]
            ),
            "true_bin_to_replacement_geometric_ratio": geometric_mean(
                [row["true_to_replacement_ratio"] for row in patient_rows]
            ),
            "true_to_mean_pairing_geometric_ratio": geometric_mean(
                [row["true_to_mean_pairing_ratio"] for row in patient_rows]
            ),
            "true_grouping_median_rank": float(
                np.median([row["true_grouping_rank"] for row in patient_rows])
            ),
            "true_grouping_top_quartile_patients": int(
                sum(row["true_grouping_rank"] <= 27 for row in patient_rows)
            ),
            "pair_partitions": len(all_pairings),
        },
        "artifacts": {
            "patient_results": "timestep_bin_ablation_patients.csv",
            "summary": "timestep_bin_ablation.md",
        },
    }
    (report_dir / "timestep_bin_ablation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (report_dir / "timestep_bin_ablation.md").write_text(
        build_markdown(result, patient_rows), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.report_dir)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
