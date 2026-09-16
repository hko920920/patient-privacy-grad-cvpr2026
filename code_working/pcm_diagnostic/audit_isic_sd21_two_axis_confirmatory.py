#!/usr/bin/env python3
"""Independent recomputation and artifact lock for the confirmatory diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from run_isic_sd21_multipatient_diagnostic import bootstrap_ratio
from run_isic_sd21_two_axis_confirmatory import EXCLUDED_PATIENTS


SCHEMA = "pcm-isic-sd21-two-axis-confirmatory-secondary-audit/v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def geometric_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.exp(np.mean(np.log(array))))


def build_markdown(result: dict[str, Any]) -> str:
    primary = result["primary_recomputation"]
    lines = [
        "# ISIC SD 2.1 two-axis confirmatory secondary audit",
        "",
        f"- Audit status: **{result['status']}**",
        f"- Frozen decision reproduced: **{result['decision']}**",
        f"- Geometric MSE ratio: `{primary['geometric_mean_ratio']:.6f}`",
        f"- Bootstrap 95% interval: `[{primary['bootstrap_ci95_lower']:.6f}, "
        f"{primary['bootstrap_ci95_upper']:.6f}]`",
        f"- Balanced wins: `{primary['balanced_wins']}/16`",
        "",
        "## Checks",
        "",
    ]
    for check in result["checks"]:
        lines.append(f"- `{check['name']}`: **{'PASS' if check['pass'] else 'FAIL'}**")
    lines.extend(
        [
            "",
            "This is an independent recomputation of the frozen finite-bank result, not a DP, "
            "training-utility, generation, clinical, or novelty claim.",
            "",
        ]
    )
    return "\n".join(lines)


def run(report_dir: Path) -> dict[str, Any]:
    report_dir = report_dir.resolve()
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    selected = read_csv(report_dir / "selected_patient_records.csv")
    gradients = read_csv(report_dir / "gradient_evaluations.csv")
    metrics = read_csv(report_dir / "patient_preclip_metrics.csv")
    patients = read_csv(report_dir / "patient_primary_comparison.csv")
    metric_map = {
        (row["patient_id"], row["method"]): float(row["preclip_mse"])
        for row in metrics
    }
    ratios = [
        metric_map[(row["patient_id"], "bin_balanced_m4")]
        / metric_map[(row["patient_id"], "replacement_uniform_m4")]
        for row in patients
    ]
    bootstrap = bootstrap_ratio(np.asarray(ratios, dtype=np.float64))
    balanced_wins = int(sum(value < 1.0 for value in ratios))
    decision = (
        "TWO_AXIS_BALANCE_CONFIRMED_FOR_METHOD_REVIEW"
        if bootstrap["geometric_mean_ratio"] <= 0.90
        and bootstrap["bootstrap_ci95_upper"] < 1.0
        and balanced_wins >= 12
        else "RETIRE_CURRENT_TWO_AXIS_BALANCE"
    )
    patient_counts: dict[str, int] = {}
    for row in selected:
        patient_counts[row["patient_id"]] = patient_counts.get(row["patient_id"], 0) + 1
    checks = [
        {"name": "source_execution_pass", "pass": summary["status"] == "PASS"},
        {"name": "decision_reproduced", "pass": decision == summary["decision"]},
        {
            "name": "geometric_ratio_reproduced",
            "pass": math.isclose(
                float(bootstrap["geometric_mean_ratio"]),
                float(summary["primary"]["geometric_mean_ratio"]),
                rel_tol=0.0,
                abs_tol=1e-14,
            ),
        },
        {
            "name": "bootstrap_interval_reproduced",
            "pass": math.isclose(
                float(bootstrap["bootstrap_ci95_lower"]),
                float(summary["primary"]["bootstrap_ci95_lower"]),
                rel_tol=0.0,
                abs_tol=1e-14,
            )
            and math.isclose(
                float(bootstrap["bootstrap_ci95_upper"]),
                float(summary["primary"]["bootstrap_ci95_upper"]),
                rel_tol=0.0,
                abs_tol=1e-14,
            ),
        },
        {
            "name": "sixteen_new_patients_five_records_each",
            "pass": len(patient_counts) == 16
            and len(selected) == 80
            and set(patient_counts.values()) == {5}
            and not (set(patient_counts) & EXCLUDED_PATIENTS),
        },
        {
            "name": "six_hundred_forty_finite_gradients",
            "pass": len(gradients) == 640
            and all(row["finite"].lower() == "true" for row in gradients),
        },
        {
            "name": "all_source_execution_checks_pass",
            "pass": all(row["status"] == "PASS" for row in summary["execution_checks"]),
        },
    ]
    locked_names = [
        "summary.json",
        "selected_patient_records.csv",
        "gradient_evaluations.csv",
        "perturbation_specs.csv",
        "patient_gradient_grams_local_only.npz",
        "patient_preclip_metrics.csv",
        "patient_clipping_metrics.csv",
        "patient_primary_comparison.csv",
    ]
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS" if all(row["pass"] for row in checks) else "FAIL",
        "decision": decision,
        "checks": checks,
        "primary_recomputation": {
            **bootstrap,
            "balanced_wins": balanced_wins,
            "median_ratio": float(np.median(ratios)),
            "minimum_ratio": float(min(ratios)),
            "maximum_ratio": float(max(ratios)),
        },
        "source_artifact_sha256": {
            name: sha256_file(report_dir / name) for name in locked_names
        },
        "analysis_boundary": (
            "Independent recomputation of a finite-bank development diagnostic; no DP, "
            "training, generation, clinical, novelty, or venue claim."
        ),
    }
    (report_dir / "secondary_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (report_dir / "secondary_audit.md").write_text(
        build_markdown(result), encoding="utf-8"
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
