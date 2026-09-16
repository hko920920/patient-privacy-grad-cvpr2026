#!/usr/bin/env python3
"""Aggregate five preregistered timestep-balance replication banks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "pcm-isic-sd21-timestep-multibank-aggregate/v1"
PROTOCOL = "ISIC_SD21_TIMESTEP_MULTIBANK_REPLICATION_PROTOCOL_V1.md"
EXPECTED_TAGS = tuple(f"repl_{index:02d}" for index in range(1, 6))
BOOTSTRAP_SEED = 26090205
BOOTSTRAP_REPLICATES = 10_000


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


def geometric_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if np.any(values <= 0):
        raise ValueError("geometric mean requires positive values")
    return float(np.exp(np.mean(np.log(values))))


def hierarchical_bootstrap(
    ratios: np.ndarray,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    ratios = np.asarray(ratios, dtype=np.float64)
    if ratios.ndim != 2 or np.any(ratios <= 0):
        raise ValueError("ratios must be a positive bank-by-patient matrix")
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    banks, patients = ratios.shape
    for index in range(replicates):
        bank_indices = rng.integers(0, banks, size=banks)
        patient_indices = rng.integers(0, patients, size=patients)
        values[index] = geometric_mean(ratios[np.ix_(bank_indices, patient_indices)])
    return {
        "geometric_mean_ratio": geometric_mean(ratios),
        "bootstrap_ci95_lower": float(np.quantile(values, 0.025)),
        "bootstrap_ci95_upper": float(np.quantile(values, 0.975)),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def build_markdown(result: dict[str, Any], bank_rows: list[dict[str, Any]]) -> str:
    primary = result["primary"]
    lines = [
        "# ISIC SD 2.1 timestep-balance multi-bank replication",
        "",
        f"- Execution status: **{result['status']}**",
        f"- Replication decision: **{result['decision']}**",
        "- Scope: five preregistered new perturbation banks on the frozen 16-patient cohort",
        "",
        "## Frozen multi-bank primary",
        "",
        f"- Equal-bank true-bin/distinct geometric ratio: "
        f"`{primary['geometric_mean_ratio']:.4f}`",
        f"- Hierarchical bootstrap 95% interval: "
        f"`[{primary['bootstrap_ci95_lower']:.4f}, "
        f"{primary['bootstrap_ci95_upper']:.4f}]`",
        f"- Banks with ratio < 1: `{primary['banks_below_one']}/5`",
        f"- Banks with >=10/16 patient wins: "
        f"`{primary['banks_with_ten_wins']}/5`",
        f"- One-sided 5/5 sign probability under median null: "
        f"`{primary['five_of_five_sign_probability']:.5f}`",
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
            "## Bank results",
            "",
            "| Bank | True/distinct | True/replacement | Wins / 16 | Median rank / 105 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in bank_rows:
        lines.append(
            f"| {row['bank_tag']} | {row['true_to_distinct_ratio']:.4f} | "
            f"{row['true_to_replacement_ratio']:.4f} | "
            f"{row['true_bin_wins']}/16 | {row['median_true_grouping_rank']:.1f} |"
        )
    lines.extend(
        [
            "",
            "A PASS only addresses perturbation-bank replication of a stored-gradient "
            "diagnostic. It does not establish training utility, DP utility, clinical validity, "
            "novelty, or venue suitability.",
            "",
        ]
    )
    return "\n".join(lines)


def run(report_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    if len(report_dirs) != 5:
        raise ValueError("exactly five replication report directories are required")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = Path(__file__).with_name(PROTOCOL)
    expected_protocol_hash = sha256_file(protocol_path)

    bank_rows: list[dict[str, Any]] = []
    patient_order: list[str] | None = None
    ratio_rows: list[list[float]] = []
    all_noise_seeds: list[int] = []
    execution_checks: list[dict[str, Any]] = []

    for report_dir in report_dirs:
        report_dir = report_dir.resolve()
        summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
        audit = json.loads(
            (report_dir / "secondary_audit.json").read_text(encoding="utf-8")
        )
        ablation = json.loads(
            (report_dir / "timestep_bin_ablation.json").read_text(encoding="utf-8")
        )
        patients = read_csv(report_dir / "timestep_bin_ablation_patients.csv")
        specs = read_csv(report_dir / "perturbation_specs.csv")
        bank_tag = str(summary["configuration"]["bank_tag"])

        ids = [row["patient_id"] for row in patients]
        if patient_order is None:
            patient_order = ids
        cohort_match = ids == patient_order
        ratios = [float(row["true_to_distinct_ratio"]) for row in patients]
        ranks = [int(row["true_grouping_rank"]) for row in patients]
        all_noise_seeds.extend(int(row["noise_seed"]) for row in specs)
        ratio_rows.append(ratios)
        bank_rows.append(
            {
                "bank_tag": bank_tag,
                "true_to_distinct_ratio": float(
                    ablation["primary"]["geometric_mean_ratio"]
                ),
                "true_to_replacement_ratio": float(
                    ablation["summary"]["true_bin_to_replacement_geometric_ratio"]
                ),
                "true_bin_wins": int(ablation["primary"]["true_bin_wins"]),
                "median_true_grouping_rank": float(np.median(ranks)),
                "rank_one_patients": int(sum(rank_value == 1 for rank_value in ranks)),
                "report_dir": str(report_dir),
            }
        )
        execution_checks.append(
            {
                "name": f"{bank_tag}_source_integrity",
                "status": "PASS"
                if summary["status"] == audit["status"] == ablation["status"] == "PASS"
                and summary["protocol"]["sha256"] == expected_protocol_hash
                and cohort_match
                and len(patients) == 16
                and len(specs) == 8
                else "FAIL",
            }
        )

    tags = tuple(row["bank_tag"] for row in bank_rows)
    execution_checks.extend(
        [
            {
                "name": "exact_preregistered_bank_tags",
                "status": "PASS" if tags == EXPECTED_TAGS else "FAIL",
                "observed": list(tags),
            },
            {
                "name": "all_40_noise_seeds_unique",
                "status": "PASS"
                if len(all_noise_seeds) == len(set(all_noise_seeds)) == 40
                else "FAIL",
            },
        ]
    )

    ratio_matrix = np.asarray(ratio_rows, dtype=np.float64)
    bootstrap = hierarchical_bootstrap(ratio_matrix)
    bank_ratios = np.asarray(
        [row["true_to_distinct_ratio"] for row in bank_rows], dtype=np.float64
    )
    banks_below_one = int(np.sum(bank_ratios < 1.0))
    banks_with_ten_wins = int(
        sum(row["true_bin_wins"] >= 10 for row in bank_rows)
    )
    top_quartile_banks = int(
        sum(row["median_true_grouping_rank"] <= 27 for row in bank_rows)
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
            "status": "PASS"
            if bootstrap["geometric_mean_ratio"] <= 0.95
            else "FAIL",
            "observed": bootstrap["geometric_mean_ratio"],
            "required": "<=0.95",
        },
        {
            "name": "hierarchical_bootstrap_upper_lt_1",
            "status": "PASS"
            if bootstrap["bootstrap_ci95_upper"] < 1.0
            else "FAIL",
            "observed": bootstrap["bootstrap_ci95_upper"],
            "required": "<1.00",
        },
        {
            "name": "true_grouping_top_quartile_in_ge_4_banks",
            "status": "PASS" if top_quartile_banks >= 4 else "FAIL",
            "observed": top_quartile_banks,
            "required": ">=4/5",
        },
    ]
    status = (
        "PASS"
        if all(row["status"] == "PASS" for row in execution_checks)
        else "FAIL"
    )
    decision = (
        "MULTIBANK_TIMESTEP_SIGNAL_REPLICATED_FOR_LOCALITY_TEST"
        if status == "PASS" and all(row["status"] == "PASS" for row in conditions)
        else "MULTIBANK_TIMESTEP_SIGNAL_NOT_REPLICATED"
    )
    write_csv(output_dir / "bank_results.csv", bank_rows)
    patient_bank_rows = [
        {
            "bank_tag": bank_rows[bank_index]["bank_tag"],
            "patient_id": patient_order[patient_index] if patient_order else "",
            "true_to_distinct_ratio": ratio_matrix[bank_index, patient_index],
        }
        for bank_index in range(ratio_matrix.shape[0])
        for patient_index in range(ratio_matrix.shape[1])
    ]
    write_csv(output_dir / "patient_bank_ratios.csv", patient_bank_rows)

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "decision": decision,
        "protocol": {
            "file": PROTOCOL,
            "sha256": expected_protocol_hash,
        },
        "execution_checks": execution_checks,
        "conditions": conditions,
        "primary": {
            **bootstrap,
            "banks": 5,
            "patients": 16,
            "banks_below_one": banks_below_one,
            "banks_with_ten_wins": banks_with_ten_wins,
            "top_quartile_grouping_banks": top_quartile_banks,
            "five_of_five_sign_probability": 1.0 / 32.0,
        },
        "bank_results": bank_rows,
        "artifacts": {
            "bank_results": "bank_results.csv",
            "patient_bank_ratios": "patient_bank_ratios.csv",
            "summary": "summary.md",
        },
        "analysis_boundary": (
            "Five-bank finite-gradient replication only; no optimizer update, DP noise, "
            "generation, clinical, novelty, or venue claim."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        build_markdown(result, bank_rows), encoding="utf-8"
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
