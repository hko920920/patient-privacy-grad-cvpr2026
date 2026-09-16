#!/usr/bin/env python3
"""Independent recomputation audit for the five-bank aggregate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "pcm-isic-sd21-timestep-multibank-secondary-audit/v1"
EXPECTED_TAGS = tuple(f"repl_{index:02d}" for index in range(1, 6))
BOOTSTRAP_SEED = 26090205
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


def geom(values: np.ndarray) -> float:
    return float(np.exp(np.mean(np.log(np.asarray(values, dtype=np.float64)))))


def run(report_dir: Path) -> dict[str, Any]:
    report_dir = report_dir.resolve()
    source = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    banks = read_csv(report_dir / "bank_results.csv")
    cells = read_csv(report_dir / "patient_bank_ratios.csv")
    tags = tuple(row["bank_tag"] for row in banks)
    patient_ids: list[str] = []
    for row in cells:
        if row["patient_id"] not in patient_ids:
            patient_ids.append(row["patient_id"])
    cell_map = {
        (row["bank_tag"], row["patient_id"]): float(row["true_to_distinct_ratio"])
        for row in cells
    }
    matrix = np.asarray(
        [[cell_map[(tag, patient_id)] for patient_id in patient_ids] for tag in tags],
        dtype=np.float64,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        sampled_banks = rng.choice(5, size=5, replace=True)
        sampled_patients = rng.choice(16, size=16, replace=True)
        draws[index] = geom(matrix[sampled_banks][:, sampled_patients])
    recomputed = {
        "geometric_mean_ratio": geom(matrix),
        "bootstrap_ci95_lower": float(np.quantile(draws, 0.025)),
        "bootstrap_ci95_upper": float(np.quantile(draws, 0.975)),
        "banks_below_one": int(sum(float(row["true_to_distinct_ratio"]) < 1 for row in banks)),
        "banks_with_ten_wins": int(sum(int(row["true_bin_wins"]) >= 10 for row in banks)),
        "top_quartile_grouping_banks": int(
            sum(float(row["median_true_grouping_rank"]) <= 27 for row in banks)
        ),
    }
    primary = source["primary"]
    numeric_match = all(
        math.isclose(float(recomputed[name]), float(primary[name]), rel_tol=0.0, abs_tol=1e-14)
        for name in (
            "geometric_mean_ratio",
            "bootstrap_ci95_lower",
            "bootstrap_ci95_upper",
        )
    )
    count_match = all(
        int(recomputed[name]) == int(primary[name])
        for name in (
            "banks_below_one",
            "banks_with_ten_wins",
            "top_quartile_grouping_banks",
        )
    )
    checks = [
        {"name": "source_execution_pass", "pass": source["status"] == "PASS"},
        {"name": "exact_five_tags", "pass": tags == EXPECTED_TAGS},
        {
            "name": "complete_five_by_sixteen_matrix",
            "pass": bool(
                matrix.shape == (5, 16)
                and len(cell_map) == 80
                and np.all(matrix > 0)
            ),
        },
        {"name": "primary_numeric_reproduced", "pass": numeric_match},
        {"name": "primary_counts_reproduced", "pass": count_match},
        {
            "name": "decision_reproduced",
            "pass": source["decision"]
            == "MULTIBANK_TIMESTEP_SIGNAL_REPLICATED_FOR_LOCALITY_TEST"
            and recomputed["banks_below_one"] == 5
            and recomputed["banks_with_ten_wins"] == 5
            and recomputed["geometric_mean_ratio"] <= 0.95
            and recomputed["bootstrap_ci95_upper"] < 1.0
            and recomputed["top_quartile_grouping_banks"] >= 4,
        },
    ]
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS" if all(row["pass"] for row in checks) else "FAIL",
        "decision": source["decision"],
        "checks": checks,
        "primary_recomputation": {
            **recomputed,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "source_artifact_sha256": {
            name: sha256_file(report_dir / name)
            for name in ("summary.json", "bank_results.csv", "patient_bank_ratios.csv")
        },
        "analysis_boundary": (
            "Independent aggregate recomputation only; no training, DP, generation, clinical, "
            "novelty, or venue claim."
        ),
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
