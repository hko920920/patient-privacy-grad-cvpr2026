#!/usr/bin/env python3
"""Independent primary recomputation audit for actual B=2 joint-batch locality."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "pcm-isic-sd21-joint-batch-locality-secondary-audit/v1"
PROTOCOL = "ISIC_SD21_JOINT_BATCH_LOCALITY_PROTOCOL_V1.md"
SPEC_FILE = "ISIC_SD21_JOINT_BATCH_BANKS_V1.csv"
TAGS = tuple(f"joint_{index:02d}" for index in range(1, 6))
C = 0.20
BOOTSTRAP_SEED = 26090305
BOOTSTRAP_REPLICATES = 10_000


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def geom(values: np.ndarray | list[float]) -> float:
    return float(np.exp(np.mean(np.log(np.asarray(values, dtype=np.float64)))))


def relerr(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), 1e-30)


def complement(values: tuple[int, ...]) -> tuple[int, ...]:
    selected = set(values)
    return tuple(value for value in range(8) if value not in selected)


def independent_weights(values: tuple[int, ...]) -> np.ndarray:
    rows = []
    for records in itertools.combinations(range(5), 4):
        for assignment in itertools.permutations(values):
            row = np.zeros(40, dtype=np.float64)
            for record, perturbation in zip(records, assignment):
                row[record * 8 + perturbation] = 0.25
            rows.append(row)
    return np.asarray(rows)


def independent_patient_stats(
    gram: np.ndarray, subsets: tuple[tuple[int, ...], ...], weights: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = np.full(40, 0.025, dtype=np.float64)
    reference_norm = math.sqrt(max(0.0, float(reference @ gram @ reference)))
    reference_scale = min(1.0, C / reference_norm) if reference_norm > 0 else 1.0
    means = np.empty((len(subsets), 40), dtype=np.float64)
    seconds = np.empty(len(subsets), dtype=np.float64)
    rates = np.empty(len(subsets), dtype=np.float64)
    for index, states in enumerate(weights):
        norms = np.sqrt(
            np.maximum(
                0.0, np.sum((states @ gram) * states, axis=1, dtype=np.float64)
            )
        )
        scales = np.minimum(
            1.0, np.divide(C, norms, out=np.ones_like(norms), where=norms > 0)
        )
        differences = scales[:, None] * states - reference_scale * reference
        quadratic = np.sum(
            (differences @ gram) * differences, axis=1, dtype=np.float64
        )
        means[index] = differences.mean(axis=0)
        seconds[index] = float(np.mean(np.maximum(0.0, quadratic)))
        rates[index] = float(np.mean(norms > C))
    return means, seconds, rates


def independent_pair(
    left: tuple[np.ndarray, np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray, np.ndarray],
    cross: np.ndarray,
    first_indices: np.ndarray,
    second_indices: np.ndarray,
    dimension: int,
) -> dict[str, float]:
    left_means, left_seconds, _ = left
    right_means, right_seconds, _ = right
    cross_values = np.sum(
        (left_means[first_indices] @ cross) * right_means[second_indices],
        axis=1,
        dtype=np.float64,
    )
    within = 0.25 * float(
        np.mean(left_seconds[first_indices] + right_seconds[second_indices])
    ) / dimension
    cross_component = 0.5 * float(np.mean(cross_values)) / dimension
    return {
        "joint_mse": max(0.0, within + cross_component),
        "within_component_mse": within,
        "cross_component_mse": cross_component,
    }


def independent_bootstrap(ratios: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        chosen = rng.choice(5, size=5, replace=True)
        draws[index] = geom(ratios[chosen])
    return {
        "geometric_mean_ratio": geom(ratios),
        "bootstrap_ci95_lower": float(np.quantile(draws, 0.025)),
        "bootstrap_ci95_upper": float(np.quantile(draws, 0.975)),
    }


def run(report_dir: Path) -> dict[str, Any]:
    report_dir = report_dir.resolve()
    source = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    pair_csv = read_csv(report_dir / "pair_clip_metrics.csv")
    marginal_csv = read_csv(report_dir / "patient_marginal_metrics.csv")
    pair_map = {
        (
            row["bank_tag"],
            row["left_patient_id"],
            row["right_patient_id"],
            float(row["clip_norm"]),
        ): row
        for row in pair_csv
    }
    marginal_map = {
        (row["bank_tag"], row["patient_id"], float(row["clip_norm"])): row
        for row in marginal_csv
    }
    subsets = tuple(itertools.combinations(range(8), 4))
    subset_index = {subset: index for index, subset in enumerate(subsets)}
    weights = [independent_weights(subset) for subset in subsets]
    global_first = np.arange(70, dtype=np.int64)
    global_second = np.asarray(
        [subset_index[complement(subset)] for subset in subsets], dtype=np.int64
    )
    spec_rows = read_csv(Path(__file__).with_name(SPEC_FILE))
    patients: list[str] | None = None
    site_groups: dict[str, str] = {}
    recomputed_pairs: list[dict[str, Any]] = []
    recomputed_marginals: list[dict[str, Any]] = []
    maximum_pair_relative_error = 0.0
    maximum_marginal_relative_error = 0.0
    crossgram_hash_match = True
    dimension = int(source["configuration"]["parameter_dimension"])

    for source_bank in source["source_crossgram"]["banks"]:
        tag = source_bank["bank_tag"]
        gram_path = Path(source_bank["gram_file"])
        metadata_path = Path(source_bank["metadata_file"])
        crossgram_hash_match = crossgram_hash_match and (
            source_bank["gram_sha256"] == sha256_file(gram_path)
            and source_bank["metadata_sha256"] == sha256_file(metadata_path)
        )
        crossgram_summary_path = Path(source["source_crossgram"]["directory"])
        index_rows = read_csv(crossgram_summary_path / "gradient_index.csv")
        current_patients: list[str] = []
        for row in index_rows:
            if row["patient_id"] not in current_patients:
                current_patients.append(row["patient_id"])
            site_groups[row["patient_id"]] = row["site_group"]
        if patients is None:
            patients = current_patients
        elif patients != current_patients:
            raise ValueError("patient order changed between source banks")
        gram = np.load(gram_path, allow_pickle=False)
        stats = []
        for patient_index, patient_id in enumerate(current_patients):
            block = slice(patient_index * 40, (patient_index + 1) * 40)
            patient_stats = independent_patient_stats(
                np.asarray(gram[block, block], dtype=np.float64), subsets, weights
            )
            stats.append(patient_stats)

        selected_specs = [row for row in spec_rows if row["bank_tag"] == tag]
        groups = tuple(
            tuple(
                int(row["perturbation_index"])
                for row in selected_specs
                if int(row["bin_index"]) == bin_index
            )
            for bin_index in range(4)
        )
        local_choices = tuple(
            tuple(sorted(choice)) for choice in itertools.product(*groups)
        )
        local_first = np.asarray(
            [subset_index[choice] for choice in local_choices], dtype=np.int64
        )
        local_second = np.asarray(
            [subset_index[complement(choice)] for choice in local_choices],
            dtype=np.int64,
        )
        for patient_index, patient_id in enumerate(current_patients):
            _, seconds, rates = stats[patient_index]
            global_mse = float(np.mean(seconds[global_first]) / dimension)
            local_mse = float(np.mean(seconds[local_first]) / dimension)
            row = {
                "bank_tag": tag,
                "patient_id": patient_id,
                "global_marginal_mse": global_mse,
                "unit_local_marginal_mse": local_mse,
                "unit_local_to_global_ratio": local_mse / global_mse,
                "global_clip_rate": float(np.mean(rates[global_first])),
                "unit_local_clip_rate": float(np.mean(rates[local_first])),
            }
            recomputed_marginals.append(row)
            observed = marginal_map[(tag, patient_id, C)]
            for name in row:
                if name not in ("bank_tag", "patient_id"):
                    maximum_marginal_relative_error = max(
                        maximum_marginal_relative_error,
                        relerr(float(row[name]), float(observed[name])),
                    )

        for left_index, right_index in itertools.combinations(range(16), 2):
            left_id = current_patients[left_index]
            right_id = current_patients[right_index]
            left_block = slice(left_index * 40, (left_index + 1) * 40)
            right_block = slice(right_index * 40, (right_index + 1) * 40)
            cross = np.asarray(gram[left_block, right_block], dtype=np.float64)
            global_metric = independent_pair(
                stats[left_index],
                stats[right_index],
                cross,
                global_first,
                global_second,
                dimension,
            )
            local_metric = independent_pair(
                stats[left_index],
                stats[right_index],
                cross,
                local_first,
                local_second,
                dimension,
            )
            row = {
                "bank_tag": tag,
                "left_patient_id": left_id,
                "right_patient_id": right_id,
                "global_joint_mse": global_metric["joint_mse"],
                "unit_local_joint_mse": local_metric["joint_mse"],
                "unit_local_to_global_ratio": local_metric["joint_mse"]
                / global_metric["joint_mse"],
                "global_within_component_mse": global_metric[
                    "within_component_mse"
                ],
                "unit_local_within_component_mse": local_metric[
                    "within_component_mse"
                ],
                "global_cross_component_mse": global_metric["cross_component_mse"],
                "unit_local_cross_component_mse": local_metric[
                    "cross_component_mse"
                ],
            }
            recomputed_pairs.append(row)
            observed = pair_map[(tag, left_id, right_id, C)]
            for name in row:
                if name not in ("bank_tag", "left_patient_id", "right_patient_id"):
                    maximum_pair_relative_error = max(
                        maximum_pair_relative_error,
                        relerr(float(row[name]), float(observed[name])),
                    )
        del gram

    if patients is None:
        raise RuntimeError("no patients audited")
    bank_rows: list[dict[str, Any]] = []
    for tag in TAGS:
        pairs = [row for row in recomputed_pairs if row["bank_tag"] == tag]
        marginals = [row for row in recomputed_marginals if row["bank_tag"] == tag]
        global_mean = float(np.mean([row["global_joint_mse"] for row in pairs]))
        local_mean = float(np.mean([row["unit_local_joint_mse"] for row in pairs]))
        global_marginal = float(
            np.mean([row["global_marginal_mse"] for row in marginals])
        )
        local_marginal = float(
            np.mean([row["unit_local_marginal_mse"] for row in marginals])
        )
        bank_rows.append(
            {
                "bank_tag": tag,
                "global_joint_mse_mean": global_mean,
                "unit_local_joint_mse_mean": local_mean,
                "joint_ratio_of_means": local_mean / global_mean,
                "unit_local_pair_wins": int(
                    sum(row["unit_local_to_global_ratio"] < 1.0 for row in pairs)
                ),
                "marginal_ratio_of_means": local_marginal / global_marginal,
            }
        )
    ratios = np.asarray([row["joint_ratio_of_means"] for row in bank_rows])
    bootstrap = independent_bootstrap(ratios)
    leaveouts = []
    for patient_id in patients:
        rows = [
            row
            for row in recomputed_pairs
            if row["left_patient_id"] != patient_id
            and row["right_patient_id"] != patient_id
        ]
        global_mean = float(np.mean([row["global_joint_mse"] for row in rows]))
        local_mean = float(np.mean([row["unit_local_joint_mse"] for row in rows]))
        leaveouts.append(local_mean / global_mean)

    marginal_ratios = np.asarray([row["marginal_ratio_of_means"] for row in bank_rows])
    global_clip_rate = float(
        np.mean([row["global_clip_rate"] for row in recomputed_marginals])
    )
    local_clip_rate = float(
        np.mean([row["unit_local_clip_rate"] for row in recomputed_marginals])
    )
    counts = {
        "banks_below_one": int(np.sum(ratios < 1.0)),
        "banks_with_84_pair_wins": int(
            sum(row["unit_local_pair_wins"] >= 84 for row in bank_rows)
        ),
        "leave_one_patient_out_below_one": int(sum(value < 1.0 for value in leaveouts)),
        "marginal_banks_below_one": int(np.sum(marginal_ratios < 1.0)),
    }
    marginal_equal_bank_ratio = geom(marginal_ratios)
    conditions_pass = (
        counts["banks_below_one"] == 5
        and bootstrap["geometric_mean_ratio"] <= 0.95
        and bootstrap["bootstrap_ci95_upper"] < 1.0
        and counts["banks_with_84_pair_wins"] == 5
        and counts["leave_one_patient_out_below_one"] == 16
        and counts["marginal_banks_below_one"] == 5
        and marginal_equal_bank_ratio <= 0.95
        and 0.05 <= global_clip_rate <= 0.95
        and 0.05 <= local_clip_rate <= 0.95
    )
    expected_decision = (
        "JOINT_BATCH_LOCALITY_SIGNAL_CONFIRMED_FOR_XRAY_METHOD_GATE"
        if conditions_pass
        else "UNIT_LOCAL_JOINT_BATCH_INCREMENT_NOT_SUPPORTED"
    )
    primary = source["primary"]
    primary_match = all(
        math.isclose(float(primary[name]), float(bootstrap[name]), rel_tol=0, abs_tol=1e-14)
        for name in (
            "geometric_mean_ratio",
            "bootstrap_ci95_lower",
            "bootstrap_ci95_upper",
        )
    ) and all(int(primary[name]) == value for name, value in counts.items())
    primary_match = primary_match and all(
        math.isclose(float(primary[name]), value, rel_tol=0, abs_tol=1e-14)
        for name, value in (
            ("marginal_equal_bank_geometric_ratio", marginal_equal_bank_ratio),
            ("mean_global_clip_rate", global_clip_rate),
            ("mean_unit_local_clip_rate", local_clip_rate),
        )
    )
    checks = [
        {"name": "source_execution_pass", "pass": source["status"] == "PASS"},
        {
            "name": "protocol_hash_match",
            "pass": source["protocol"]["sha256"]
            == sha256_file(Path(__file__).with_name(PROTOCOL)),
        },
        {
            "name": "bank_spec_hash_match",
            "pass": source["bank_spec"]["sha256"]
            == sha256_file(Path(__file__).with_name(SPEC_FILE)),
        },
        {"name": "crossgram_source_hashes_match", "pass": crossgram_hash_match},
        {
            "name": "complete_primary_cells",
            "pass": len(recomputed_pairs) == 600
            and len(recomputed_marginals) == 80
            and len(pair_map) == 3000
            and len(marginal_map) == 400,
        },
        {
            "name": "primary_pair_metrics_reproduced",
            "pass": maximum_pair_relative_error <= 2e-12,
            "maximum_relative_error": maximum_pair_relative_error,
        },
        {
            "name": "primary_marginal_metrics_reproduced",
            "pass": maximum_marginal_relative_error <= 2e-12,
            "maximum_relative_error": maximum_marginal_relative_error,
        },
        {"name": "primary_summary_reproduced", "pass": primary_match},
        {
            "name": "decision_reproduced",
            "pass": source["decision"] == expected_decision,
            "expected": expected_decision,
        },
    ]
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS" if all(row["pass"] for row in checks) else "FAIL",
        "decision": source["decision"],
        "checks": checks,
        "primary_recomputation": {
            **bootstrap,
            **counts,
            "marginal_equal_bank_geometric_ratio": marginal_equal_bank_ratio,
            "mean_global_clip_rate": global_clip_rate,
            "mean_unit_local_clip_rate": local_clip_rate,
        },
        "source_artifact_sha256": {
            name: sha256_file(report_dir / name)
            for name in (
                "summary.json",
                "pair_clip_metrics.csv",
                "patient_marginal_metrics.csv",
                "bank_clip_results.csv",
                "leave_one_patient_out.csv",
            )
        },
        "analysis_boundary": "Independent primary recomputation from frozen full cross-Grams.",
    }
    (report_dir / "secondary_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
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
