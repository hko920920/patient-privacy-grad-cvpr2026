#!/usr/bin/env python3
"""Exact finite-bank diagnostic for record/perturbation two-axis balance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from run_isic_sd21_multipatient_diagnostic import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CLIP_NORMS,
    bootstrap_ratio,
    exact_estimator_metrics,
)


SCHEMA = "pcm-isic-sd21-two-axis-balance/v1"
PROTOCOL = "ISIC_SD21_TWO_AXIS_BALANCE_PROTOCOL_V1.md"


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


def enumerate_two_axis_weights(records: int, perturbations: int, m: int, r: int) -> np.ndarray:
    if m * r != perturbations:
        raise ValueError("two-axis balance requires m*r equal to perturbation count")
    rows: list[np.ndarray] = []
    for record_choice in itertools.combinations(range(records), m):
        unique_assignments: dict[tuple[float, ...], np.ndarray] = {}
        for assignment in itertools.permutations(range(perturbations)):
            weights = np.zeros(records * perturbations, dtype=np.float64)
            cursor = 0
            for record in record_choice:
                for _ in range(r):
                    perturbation = assignment[cursor]
                    cursor += 1
                    weights[record * perturbations + perturbation] += 1.0 / perturbations
            unique_assignments.setdefault(tuple(weights), weights)
        rows.extend(unique_assignments.values())
    return np.stack(rows)


def geometric_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.exp(np.mean(np.log(array))))


def build_markdown(result: dict[str, Any], patients: list[dict[str, Any]]) -> str:
    primary = result["primary"]
    lines = [
        "# ISIC SD 2.1 two-axis balanced perturbation diagnostic",
        "",
        f"- Execution status: **{result['status']}**",
        f"- Frozen decision: **{result['decision']}**",
        "- Scope: existing finite gradient bank; no optimizer update or DP noise",
        "",
        "## Primary balanced m4 versus replacement uniform m4",
        "",
        f"- Geometric MSE ratio: `{primary['geometric_mean_ratio']:.4f}`",
        f"- Patient bootstrap 95% interval: `[{primary['bootstrap_ci95_lower']:.4f}, "
        f"{primary['bootstrap_ci95_upper']:.4f}]`",
        f"- Balanced wins: `{primary['balanced_wins']}/16`",
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
            "## Allocation summary",
            "",
            "| Method | Replacement comparator | Geometric MSE ratio | Balanced wins |",
            "|---|---|---:|---:|",
        ]
    )
    for row in result["method_summary"]:
        lines.append(
            f"| {row['balanced_method']} | {row['replacement_method']} | "
            f"{row['geometric_mean_ratio']:.4f} | {row['balanced_wins']}/16 |"
        )
    lines.extend(
        [
            "",
            "## Patient-paired primary ratios",
            "",
            "| Patient | Site group | Balanced/uniform m4 MSE ratio |",
            "|---|---|---:|",
        ]
    )
    for row in patients:
        lines.append(
            f"| {row['patient_id']} | {row['site_group']} | "
            f"{row['balanced_to_uniform_m4_ratio']:.4f} |"
        )
    lines.extend(
        [
            "",
            "This finite-bank screen jointly changes timestep-bin and frozen-noise coverage. It "
            "does not yet isolate their effects or establish method novelty, DP utility, or "
            "generation quality.",
            "",
        ]
    )
    return "\n".join(lines)


def run(report_dir: Path) -> dict[str, Any]:
    report_dir = report_dir.resolve()
    source = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    secondary = json.loads(
        (report_dir / "secondary_audit.json").read_text(encoding="utf-8")
    )
    primary_rows = read_csv(report_dir / "patient_preclip_metrics.csv")
    clipping_source = read_csv(report_dir / "patient_clipping_metrics.csv")
    source_preclip = {
        (row["patient_id"], row["method"]): float(row["preclip_mse"])
        for row in primary_rows
    }
    source_postclip = {
        (row["patient_id"], row["method"], float(row["clip_norm"])): float(
            row["postclip_mse"]
        )
        for row in clipping_source
    }
    with np.load(report_dir / "patient_gradient_grams_local_only.npz") as archive:
        grams = {name: np.asarray(archive[name], dtype=np.float64) for name in archive.files}

    patients = source["selection"]["selected"]
    parameter_dimension = int(source["model"]["parameter_dimension"])
    configurations = [
        ("balanced_m1_r4", "uniform_m1_r4", 1, 4),
        ("balanced_m2_r2", "uniform_m2_r2", 2, 2),
        ("balanced_m4_r1", "uniform_m4_r1", 4, 1),
    ]
    weight_banks = {
        balanced: enumerate_two_axis_weights(5, 4, m, r)
        for balanced, _, m, r in configurations
    }
    maximum_expected_weight_error = max(
        float(np.max(np.abs(np.mean(weights, axis=0) - np.full(20, 1.0 / 20.0))))
        for weights in weight_banks.values()
    )
    maximum_row_sum_error = max(
        float(np.max(np.abs(np.sum(weights, axis=1) - 1.0)))
        for weights in weight_banks.values()
    )

    balanced_rows: list[dict[str, Any]] = []
    balanced_clipping_rows: list[dict[str, Any]] = []
    for patient in patients:
        patient_id = patient["patient_id"]
        gram = grams[patient_id]
        for balanced_method, replacement_method, _, _ in configurations:
            metrics, clipping = exact_estimator_metrics(
                weight_banks[balanced_method],
                gram,
                parameter_dimension,
                CLIP_NORMS,
            )
            balanced_rows.append(
                {
                    "patient_id": patient_id,
                    "site_group": patient["site_group"],
                    "balanced_method": balanced_method,
                    "replacement_method": replacement_method,
                    **metrics,
                }
            )
            for row in clipping:
                balanced_clipping_rows.append(
                    {
                        "patient_id": patient_id,
                        "site_group": patient["site_group"],
                        "balanced_method": balanced_method,
                        "replacement_method": replacement_method,
                        **row,
                    }
                )

    balanced_preclip = {
        (row["patient_id"], row["balanced_method"]): float(row["preclip_mse"])
        for row in balanced_rows
    }
    method_summary: list[dict[str, Any]] = []
    for balanced_method, replacement_method, _, _ in configurations:
        ratios = [
            balanced_preclip[(patient["patient_id"], balanced_method)]
            / source_preclip[(patient["patient_id"], replacement_method)]
            for patient in patients
        ]
        method_summary.append(
            {
                "balanced_method": balanced_method,
                "replacement_method": replacement_method,
                "geometric_mean_ratio": geometric_mean(ratios),
                "median_ratio": float(np.median(ratios)),
                "balanced_wins": int(sum(value < 1.0 for value in ratios)),
            }
        )

    patient_rows: list[dict[str, Any]] = []
    primary_ratios: list[float] = []
    for patient in patients:
        patient_id = patient["patient_id"]
        ratio = (
            balanced_preclip[(patient_id, "balanced_m4_r1")]
            / source_preclip[(patient_id, "uniform_m4_r1")]
        )
        primary_ratios.append(ratio)
        patient_rows.append(
            {
                "patient_id": patient_id,
                "site_group": patient["site_group"],
                "uniform_m4_preclip_mse": source_preclip[
                    (patient_id, "uniform_m4_r1")
                ],
                "balanced_m4_preclip_mse": balanced_preclip[
                    (patient_id, "balanced_m4_r1")
                ],
                "balanced_to_uniform_m4_ratio": ratio,
            }
        )
    bootstrap = bootstrap_ratio(np.asarray(primary_ratios, dtype=np.float64))
    balanced_wins = int(sum(value < 1.0 for value in primary_ratios))
    conditions = [
        {
            "name": "geometric_mean_ratio_le_0.90",
            "status": "PASS"
            if bootstrap["geometric_mean_ratio"] <= 0.90
            else "FAIL",
            "observed": bootstrap["geometric_mean_ratio"],
            "required": "<=0.90",
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
            "name": "balanced_wins_ge_12_of_16",
            "status": "PASS" if balanced_wins >= 12 else "FAIL",
            "observed": balanced_wins,
            "required": ">=12",
        },
    ]
    decision = (
        "CONTINUE_TWO_AXIS_BALANCE"
        if all(row["status"] == "PASS" for row in conditions)
        else "RETIRE_CURRENT_TWO_AXIS_BALANCE"
    )

    clipping_summary: list[dict[str, Any]] = []
    for clip_norm in CLIP_NORMS:
        balanced_map = {
            row["patient_id"]: float(row["postclip_mse"])
            for row in balanced_clipping_rows
            if row["balanced_method"] == "balanced_m4_r1"
            and float(row["clip_norm"]) == clip_norm
        }
        ratios = [
            balanced_map[patient["patient_id"]]
            / source_postclip[(patient["patient_id"], "uniform_m4_r1", clip_norm)]
            for patient in patients
        ]
        clipping_summary.append(
            {
                "clip_norm": clip_norm,
                "geometric_mean_ratio": geometric_mean(ratios),
                "balanced_wins": int(sum(value < 1.0 for value in ratios)),
            }
        )

    site_summary: list[dict[str, Any]] = []
    for site_group in ("low_site", "high_site"):
        values = [
            row["balanced_to_uniform_m4_ratio"]
            for row in patient_rows
            if row["site_group"] == site_group
        ]
        site_summary.append(
            {
                "site_group": site_group,
                "geometric_mean_ratio": geometric_mean(values),
                "balanced_wins": int(sum(value < 1.0 for value in values)),
                "patients": len(values),
            }
        )

    execution_checks = [
        {
            "name": "source_and_secondary_audit_pass",
            "status": "PASS"
            if source["status"] == secondary["status"] == "PASS"
            else "FAIL",
        },
        {
            "name": "expected_enumeration_sizes",
            "status": "PASS"
            if {name: weights.shape[0] for name, weights in weight_banks.items()}
            == {"balanced_m1_r4": 5, "balanced_m2_r2": 60, "balanced_m4_r1": 120}
            else "FAIL",
        },
        {
            "name": "balanced_estimators_unbiased",
            "status": "PASS"
            if maximum_expected_weight_error <= 1e-12
            and maximum_row_sum_error <= 1e-12
            else "FAIL",
            "maximum_expected_weight_error": maximum_expected_weight_error,
            "maximum_row_sum_error": maximum_row_sum_error,
        },
        {
            "name": "sixteen_patient_results",
            "status": "PASS" if len(patient_rows) == 16 else "FAIL",
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
            "Finite-bank development diagnostic only; perturbation index jointly changes "
            "timestep bin and noise seed. No training, DP, generation, utility, or novelty claim."
        ),
        "execution_checks": execution_checks,
        "enumerated_states": {
            name: int(weights.shape[0]) for name, weights in weight_banks.items()
        },
        "primary": {
            **bootstrap,
            "balanced_wins": balanced_wins,
            "patients": len(primary_ratios),
        },
        "conditions": conditions,
        "method_summary": method_summary,
        "site_group_summary": site_summary,
        "clipping_summary": clipping_summary,
        "bootstrap_contract": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "artifacts": {
            "balanced_patient_metrics": "two_axis_patient_metrics.csv",
            "balanced_clipping_metrics": "two_axis_clipping_metrics.csv",
            "patient_primary": "two_axis_patient_primary.csv",
            "summary": "two_axis_balance.md",
        },
    }
    write_csv(report_dir / "two_axis_patient_metrics.csv", balanced_rows)
    write_csv(report_dir / "two_axis_clipping_metrics.csv", balanced_clipping_rows)
    write_csv(report_dir / "two_axis_patient_primary.csv", patient_rows)
    (report_dir / "two_axis_balance.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (report_dir / "two_axis_balance.md").write_text(
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
