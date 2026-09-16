#!/usr/bin/env python3
"""Exact five-bank patient-unit-local versus batch-global clipping proxy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from run_isic_sd21_multipatient_diagnostic import CLIP_NORMS, exact_estimator_metrics
from run_isic_sd21_timestep_bin_ablation import (
    enumerate_distinct_perturbation_weights,
    enumerate_group_balanced_weights,
)


SCHEMA = "pcm-isic-sd21-unit-local-clipping-proxy/v1"
PROTOCOL = "ISIC_SD21_UNIT_LOCAL_CLIPPING_PROXY_PROTOCOL_V1.md"
EXPECTED_TAGS = tuple(f"repl_{index:02d}" for index in range(1, 6))
PRIMARY_CLIP_NORM = 0.20
BOOTSTRAP_SEED = 26090302
BOOTSTRAP_REPLICATES = 10_000


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
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


def geometric_mean(values: np.ndarray | list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or np.any(array <= 0) or not np.all(np.isfinite(array)):
        raise ValueError("geometric mean requires finite positive values")
    return float(np.exp(np.mean(np.log(array))))


def hierarchical_bootstrap(
    ratios: np.ndarray,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    ratios = np.asarray(ratios, dtype=np.float64)
    if ratios.ndim != 2 or np.any(ratios <= 0) or not np.all(np.isfinite(ratios)):
        raise ValueError("ratios must be a finite positive bank-by-patient matrix")
    banks, patients = ratios.shape
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        bank_indices = rng.integers(0, banks, size=banks)
        patient_indices = rng.integers(0, patients, size=patients)
        draws[index] = geometric_mean(ratios[np.ix_(bank_indices, patient_indices)])
    return {
        "geometric_mean_ratio": geometric_mean(ratios),
        "bootstrap_ci95_lower": float(np.quantile(draws, 0.025)),
        "bootstrap_ci95_upper": float(np.quantile(draws, 0.975)),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def b2_global_allocations(perturbations: int = 8, q: int = 4) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """All globally balanced B=2 allocations; patient B receives the complement."""
    universe = tuple(range(perturbations))
    allocations = []
    for first in itertools.combinations(universe, q):
        first_set = set(first)
        second = tuple(value for value in universe if value not in first_set)
        allocations.append((first, second))
    return allocations


def b2_unit_local_allocations(
    groups: tuple[tuple[int, ...], ...]
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """All complementary B=2 assignments with one perturbation per bin and patient."""
    if any(len(group) != 2 for group in groups):
        raise ValueError("B=2 complementary construction requires two perturbations per group")
    allocations = []
    for first in itertools.product(*groups):
        second = tuple(
            next(value for value in group if value != chosen)
            for group, chosen in zip(groups, first)
        )
        allocations.append((tuple(first), second))
    return allocations


def true_groups_from_specs(specs: list[dict[str, str]]) -> tuple[tuple[int, ...], ...]:
    groups = tuple(
        tuple(
            int(row["perturbation_index"])
            for row in specs
            if int(row["bin_index"]) == bin_index
        )
        for bin_index in range(4)
    )
    if tuple(len(group) for group in groups) != (2, 2, 2, 2):
        raise ValueError("expected exactly two perturbations in each of four bins")
    if sorted(value for group in groups for value in group) != list(range(8)):
        raise ValueError("timestep groups must partition perturbations 0..7")
    return groups


def metrics_by_clip(rows: list[dict[str, Any]]) -> dict[float, dict[str, Any]]:
    return {float(row["clip_norm"]): row for row in rows}


def build_markdown(
    result: dict[str, Any],
    bank_rows: list[dict[str, Any]],
    clip_rows: list[dict[str, Any]],
) -> str:
    primary = result["primary"]
    lines = [
        "# ISIC SD 2.1 patient-unit-local clipping proxy",
        "",
        f"- Execution status: **{result['status']}**",
        f"- Incremental decision: **{result['decision']}**",
        "- Comparison: same-compute B=2 global-only distinct allocation versus "
        "patient-unit-local true-bin allocation",
        "- Scope: exact patient-marginal analysis of five stored replication Gram banks",
        "",
        "## Frozen primary at C=0.20",
        "",
        f"- Unit-local/global-only geometric post-clipping MSE ratio: "
        f"`{primary['geometric_mean_ratio']:.4f}`",
        f"- Hierarchical bootstrap 95% interval: "
        f"`[{primary['bootstrap_ci95_lower']:.4f}, "
        f"{primary['bootstrap_ci95_upper']:.4f}]`",
        f"- Banks below one: `{primary['banks_below_one']}/5`",
        f"- Banks with >=10/16 patient wins: `{primary['banks_with_ten_wins']}/5`",
        f"- Global-only / unit-local mean clip rates: "
        f"`{primary['mean_global_clip_rate']:.4f}` / "
        f"`{primary['mean_unit_local_clip_rate']:.4f}`",
        "",
        "## Frozen gate",
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
            "## Bank results at C=0.20",
            "",
            "| Bank | Postclip local/global | Preclip local/global | Wins / 16 | "
            "Global clip rate | Local clip rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in bank_rows:
        lines.append(
            f"| {row['bank_tag']} | {row['postclip_geometric_ratio']:.4f} | "
            f"{row['preclip_geometric_ratio']:.4f} | {row['unit_local_wins']}/16 | "
            f"{row['mean_global_clip_rate']:.4f} | "
            f"{row['mean_unit_local_clip_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Secondary clipping grid",
            "",
            "| C | Local/global ratio | Hierarchical 95% CI | Wins / 80 | "
            "Global clip rate | Local clip rate |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in clip_rows:
        lines.append(
            f"| {row['clip_norm']:.2f} | {row['geometric_mean_ratio']:.4f} | "
            f"[{row['bootstrap_ci95_lower']:.4f}, {row['bootstrap_ci95_upper']:.4f}] | "
            f"{row['unit_local_wins']}/80 | {row['mean_global_clip_rate']:.4f} | "
            f"{row['mean_unit_local_clip_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "This PASS/FAIL is a locality-clipping proxy, not an actual joint-batch update test. "
            "Cross-patient Gram blocks, DP noise, training, generation, clinical validity, "
            "novelty, and venue suitability are outside this analysis.",
            "",
        ]
    )
    return "\n".join(lines)


def run(report_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    if len(report_dirs) != 5:
        raise ValueError("exactly five replication report directories are required")
    protocol_path = Path(__file__).with_name(PROTOCOL)
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol_hash = sha256_file(protocol_path)

    distinct_weights = enumerate_distinct_perturbation_weights(5, 8)
    global_allocations = b2_global_allocations()
    reference_weights = np.full(40, 1.0 / 40.0, dtype=np.float64)
    maximum_expected_weight_error = float(
        np.max(np.abs(np.mean(distinct_weights, axis=0) - reference_weights))
    )

    patient_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    execution_checks: list[dict[str, Any]] = []
    patient_order: list[str] | None = None
    parameter_dimension: int | None = None
    maximum_preclip_relative_error = 0.0
    bank_tags: list[str] = []

    for raw_report_dir in report_dirs:
        report_dir = raw_report_dir.resolve()
        source = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
        source_audit = json.loads(
            (report_dir / "secondary_audit.json").read_text(encoding="utf-8")
        )
        ablation = json.loads(
            (report_dir / "timestep_bin_ablation.json").read_text(encoding="utf-8")
        )
        specs = read_csv(report_dir / "perturbation_specs.csv")
        ablation_patients = read_csv(report_dir / "timestep_bin_ablation_patients.csv")
        ablation_map = {row["patient_id"]: row for row in ablation_patients}
        with np.load(report_dir / "patient_gradient_grams_local_only.npz") as archive:
            grams = {
                name: np.asarray(archive[name], dtype=np.float64) for name in archive.files
            }

        bank_tag = str(source["configuration"]["bank_tag"])
        bank_tags.append(bank_tag)
        selected = source["selection"]["selected"]
        current_patient_order = [str(row["patient_id"]) for row in selected]
        cohort_match = patient_order is None or current_patient_order == patient_order
        if patient_order is None:
            patient_order = current_patient_order
        current_dimension = int(source["model"]["parameter_dimension"])
        dimension_match = parameter_dimension is None or current_dimension == parameter_dimension
        if parameter_dimension is None:
            parameter_dimension = current_dimension

        groups = true_groups_from_specs(specs)
        local_weights = enumerate_group_balanced_weights(5, 8, groups)
        local_allocations = b2_unit_local_allocations(groups)
        local_expected_weight_error = float(
            np.max(np.abs(np.mean(local_weights, axis=0) - reference_weights))
        )
        maximum_expected_weight_error = max(
            maximum_expected_weight_error, local_expected_weight_error
        )
        gram_checks = all(
            patient_id in grams
            and grams[patient_id].shape == (40, 40)
            and np.all(np.isfinite(grams[patient_id]))
            for patient_id in current_patient_order
        )
        source_integrity = (
            source["status"] == "PASS"
            and source_audit["status"] == "PASS"
            and ablation["status"] == "PASS"
            and cohort_match
            and dimension_match
            and len(selected) == 16
            and len(specs) == 8
            and len(ablation_patients) == 16
            and gram_checks
        )
        execution_checks.append(
            {
                "name": f"{bank_tag}_source_integrity",
                "status": "PASS" if source_integrity else "FAIL",
            }
        )
        source_rows.append(
            {
                "bank_tag": bank_tag,
                "report_dir": str(report_dir),
                "summary_sha256": sha256_file(report_dir / "summary.json"),
                "source_audit_sha256": sha256_file(report_dir / "secondary_audit.json"),
                "ablation_sha256": sha256_file(report_dir / "timestep_bin_ablation.json"),
                "gram_sha256": sha256_file(
                    report_dir / "patient_gradient_grams_local_only.npz"
                ),
            }
        )

        for selected_patient in selected:
            patient_id = str(selected_patient["patient_id"])
            gram = grams[patient_id]
            global_primary, global_clipping = exact_estimator_metrics(
                distinct_weights, gram, current_dimension, CLIP_NORMS
            )
            local_primary, local_clipping = exact_estimator_metrics(
                local_weights, gram, current_dimension, CLIP_NORMS
            )
            source_patient = ablation_map[patient_id]
            source_distinct = float(source_patient["distinct_mse"])
            source_local = float(source_patient["true_bin_mse"])
            maximum_preclip_relative_error = max(
                maximum_preclip_relative_error,
                abs(global_primary["preclip_mse"] - source_distinct)
                / max(source_distinct, 1e-30),
                abs(local_primary["preclip_mse"] - source_local)
                / max(source_local, 1e-30),
            )
            global_by_clip = metrics_by_clip(global_clipping)
            local_by_clip = metrics_by_clip(local_clipping)
            for clip_norm in CLIP_NORMS:
                global_metric = global_by_clip[float(clip_norm)]
                local_metric = local_by_clip[float(clip_norm)]
                patient_rows.append(
                    {
                        "bank_tag": bank_tag,
                        "patient_id": patient_id,
                        "site_group": selected_patient["site_group"],
                        "clip_norm": float(clip_norm),
                        "global_postclip_mse": global_metric["postclip_mse"],
                        "unit_local_postclip_mse": local_metric["postclip_mse"],
                        "unit_local_to_global_ratio": local_metric["postclip_mse"]
                        / global_metric["postclip_mse"],
                        "global_clip_rate": global_metric["clip_rate"],
                        "unit_local_clip_rate": local_metric["clip_rate"],
                        "global_clip_displacement": global_metric["clip_displacement"],
                        "unit_local_clip_displacement": local_metric["clip_displacement"],
                        "reference_clipped": global_metric["reference_clipped"],
                        "global_preclip_mse": global_primary["preclip_mse"],
                        "unit_local_preclip_mse": local_primary["preclip_mse"],
                        "unit_local_to_global_preclip_ratio": local_primary["preclip_mse"]
                        / global_primary["preclip_mse"],
                    }
                )

        execution_checks.append(
            {
                "name": f"{bank_tag}_b2_allocation_structure",
                "status": "PASS"
                if len(global_allocations) == 70
                and len(local_allocations) == 16
                and all(
                    sorted(first + second) == list(range(8))
                    for first, second in global_allocations + local_allocations
                )
                and all(
                    all(len(set(first) & set(group)) == 1 for group in groups)
                    and all(len(set(second) & set(group)) == 1 for group in groups)
                    for first, second in local_allocations
                )
                else "FAIL",
            }
        )

    execution_checks.extend(
        [
            {
                "name": "exact_preregistered_bank_tags",
                "status": "PASS" if tuple(bank_tags) == EXPECTED_TAGS else "FAIL",
                "observed": bank_tags,
            },
            {
                "name": "expected_exact_state_counts",
                "status": "PASS"
                if distinct_weights.shape == (8400, 40)
                and len(patient_rows) == 5 * 16 * len(CLIP_NORMS)
                else "FAIL",
            },
            {
                "name": "patient_marginal_estimators_unbiased",
                "status": "PASS" if maximum_expected_weight_error <= 1e-12 else "FAIL",
                "maximum_expected_weight_error": maximum_expected_weight_error,
            },
            {
                "name": "source_preclip_metrics_reproduced",
                "status": "PASS"
                if maximum_preclip_relative_error <= 1e-10
                else "FAIL",
                "maximum_relative_error": maximum_preclip_relative_error,
            },
        ]
    )

    if patient_order is None or parameter_dimension is None:
        raise RuntimeError("no patients loaded")
    primary_rows = [
        row for row in patient_rows if float(row["clip_norm"]) == PRIMARY_CLIP_NORM
    ]
    primary_map = {
        (row["bank_tag"], row["patient_id"]): row for row in primary_rows
    }
    primary_matrix = np.asarray(
        [
            [
                primary_map[(bank_tag, patient_id)]["unit_local_to_global_ratio"]
                for patient_id in patient_order
            ]
            for bank_tag in bank_tags
        ],
        dtype=np.float64,
    )
    bootstrap = hierarchical_bootstrap(primary_matrix)

    bank_rows: list[dict[str, Any]] = []
    for bank_tag in bank_tags:
        rows = [row for row in primary_rows if row["bank_tag"] == bank_tag]
        bank_rows.append(
            {
                "bank_tag": bank_tag,
                "postclip_geometric_ratio": geometric_mean(
                    [row["unit_local_to_global_ratio"] for row in rows]
                ),
                "preclip_geometric_ratio": geometric_mean(
                    [row["unit_local_to_global_preclip_ratio"] for row in rows]
                ),
                "unit_local_wins": int(
                    sum(row["unit_local_to_global_ratio"] < 1.0 for row in rows)
                ),
                "mean_global_clip_rate": float(
                    np.mean([row["global_clip_rate"] for row in rows])
                ),
                "mean_unit_local_clip_rate": float(
                    np.mean([row["unit_local_clip_rate"] for row in rows])
                ),
            }
        )

    clip_rows: list[dict[str, Any]] = []
    for clip_norm in CLIP_NORMS:
        rows = [row for row in patient_rows if float(row["clip_norm"]) == clip_norm]
        ratio_matrix = np.asarray(
            [
                [
                    next(
                        row["unit_local_to_global_ratio"]
                        for row in rows
                        if row["bank_tag"] == bank_tag
                        and row["patient_id"] == patient_id
                    )
                    for patient_id in patient_order
                ]
                for bank_tag in bank_tags
            ],
            dtype=np.float64,
        )
        clip_bootstrap = hierarchical_bootstrap(ratio_matrix)
        clip_rows.append(
            {
                "clip_norm": float(clip_norm),
                **clip_bootstrap,
                "unit_local_wins": int(np.sum(ratio_matrix < 1.0)),
                "banks_below_one": int(
                    sum(
                        geometric_mean(ratio_matrix[index]) < 1.0
                        for index in range(ratio_matrix.shape[0])
                    )
                ),
                "mean_global_clip_rate": float(
                    np.mean([row["global_clip_rate"] for row in rows])
                ),
                "mean_unit_local_clip_rate": float(
                    np.mean([row["unit_local_clip_rate"] for row in rows])
                ),
            }
        )

    banks_below_one = int(
        sum(row["postclip_geometric_ratio"] < 1.0 for row in bank_rows)
    )
    banks_with_ten_wins = int(
        sum(row["unit_local_wins"] >= 10 for row in bank_rows)
    )
    mean_global_clip_rate = float(
        np.mean([row["global_clip_rate"] for row in primary_rows])
    )
    mean_unit_local_clip_rate = float(
        np.mean([row["unit_local_clip_rate"] for row in primary_rows])
    )
    conditions = [
        {
            "name": "all_five_bank_ratios_lt_1",
            "status": "PASS" if banks_below_one == 5 else "FAIL",
            "observed": banks_below_one,
            "required": "5/5",
        },
        {
            "name": "all_five_banks_win_ge_10_of_16",
            "status": "PASS" if banks_with_ten_wins == 5 else "FAIL",
            "observed": banks_with_ten_wins,
            "required": "5/5",
        },
        {
            "name": "equal_bank_geometric_ratio_le_0.95",
            "status": "PASS" if bootstrap["geometric_mean_ratio"] <= 0.95 else "FAIL",
            "observed": bootstrap["geometric_mean_ratio"],
            "required": "<=0.95",
        },
        {
            "name": "hierarchical_bootstrap_upper_lt_1",
            "status": "PASS" if bootstrap["bootstrap_ci95_upper"] < 1.0 else "FAIL",
            "observed": bootstrap["bootstrap_ci95_upper"],
            "required": "<1.00",
        },
        {
            "name": "both_mean_clip_rates_between_0.05_and_0.95",
            "status": "PASS"
            if 0.05 <= mean_global_clip_rate <= 0.95
            and 0.05 <= mean_unit_local_clip_rate <= 0.95
            else "FAIL",
            "observed": {
                "global_only": mean_global_clip_rate,
                "unit_local": mean_unit_local_clip_rate,
            },
            "required": "both in [0.05,0.95]",
        },
    ]
    status = (
        "PASS"
        if all(row["status"] == "PASS" for row in execution_checks)
        else "FAIL"
    )
    decision = (
        "UNIT_LOCAL_CLIPPING_PROXY_PASSES_FOR_JOINT_BATCH_TEST"
        if status == "PASS" and all(row["status"] == "PASS" for row in conditions)
        else "UNIT_LOCAL_CLIPPING_INCREMENT_NOT_SUPPORTED"
    )

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "patient_bank_clip_metrics.csv", patient_rows)
    write_csv(output_dir / "bank_primary_results.csv", bank_rows)
    write_csv(output_dir / "clip_norm_summary.csv", clip_rows)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "decision": decision,
        "protocol": {"file": PROTOCOL, "sha256": protocol_hash},
        "configuration": {
            "banks": list(bank_tags),
            "patients": len(patient_order),
            "records_per_patient": 5,
            "perturbations": 8,
            "q_per_patient": 4,
            "conceptual_batch_patients": 2,
            "primary_clip_norm": PRIMARY_CLIP_NORM,
            "secondary_clip_norms": list(CLIP_NORMS),
            "parameter_dimension": parameter_dimension,
            "global_marginal_states": int(distinct_weights.shape[0]),
            "unit_local_marginal_states_per_bank": 1920,
        },
        "execution_checks": execution_checks,
        "conditions": conditions,
        "primary": {
            **bootstrap,
            "banks_below_one": banks_below_one,
            "banks_with_ten_wins": banks_with_ten_wins,
            "unit_local_wins": int(np.sum(primary_matrix < 1.0)),
            "cells": int(primary_matrix.size),
            "mean_global_clip_rate": mean_global_clip_rate,
            "mean_unit_local_clip_rate": mean_unit_local_clip_rate,
        },
        "bank_results": bank_rows,
        "clip_norm_summary": clip_rows,
        "source_banks": source_rows,
        "artifacts": {
            "patient_bank_clip_metrics": "patient_bank_clip_metrics.csv",
            "bank_primary_results": "bank_primary_results.csv",
            "clip_norm_summary": "clip_norm_summary.csv",
            "summary": "summary.md",
        },
        "analysis_boundary": (
            "Exact patient-marginal clipping proxy from stored within-patient Grams; no "
            "cross-patient batch covariance, DP noise, training, generation, clinical, "
            "novelty, or venue claim."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        build_markdown(result, bank_rows, clip_rows), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.report_dir, args.output_dir)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
