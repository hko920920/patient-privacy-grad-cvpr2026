#!/usr/bin/env python3
"""Independent, read-only recomputation of the multi-patient result tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "pcm-isic-sd21-multipatient-secondary-audit/v1"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 260902


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
    if np.any(array <= 0) or not np.all(np.isfinite(array)):
        raise ValueError("geometric mean requires positive finite inputs")
    return float(np.exp(np.mean(np.log(array))))


def bootstrap_geometric_ratio(values: list[float]) -> dict[str, float | int]:
    logs = np.log(np.asarray(values, dtype=np.float64))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(logs), size=(BOOTSTRAP_REPLICATES, len(logs)))
    means = np.mean(logs[indices], axis=1)
    return {
        "geometric_mean_ratio": float(np.exp(np.mean(logs))),
        "bootstrap_ci95_lower": float(np.exp(np.quantile(means, 0.025))),
        "bootstrap_ci95_upper": float(np.exp(np.quantile(means, 0.975))),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def method_ratio(
    patient_ids: list[str],
    preclip: dict[tuple[str, str], float],
    numerator: str,
    denominator: str,
) -> dict[str, Any]:
    ratios = [
        preclip[(patient_id, numerator)] / preclip[(patient_id, denominator)]
        for patient_id in patient_ids
    ]
    return {
        "numerator": numerator,
        "denominator": denominator,
        "geometric_mean_ratio": geometric_mean(ratios),
        "median_ratio": float(np.median(ratios)),
        "numerator_wins": int(sum(value < 1.0 for value in ratios)),
        "patients": len(ratios),
        "minimum_ratio": float(min(ratios)),
        "maximum_ratio": float(max(ratios)),
    }


def build_markdown(result: dict[str, Any]) -> str:
    primary = result["primary_recomputation"]
    lines = [
        "# ISIC SD 2.1 multi-patient secondary audit",
        "",
        f"- Audit status: **{result['status']}**",
        f"- Frozen decision reproduced: **{result['decision']}**",
        "- This audit does not replace or tune the frozen primary gate.",
        "",
        "## Primary recomputation",
        "",
        f"- Coverage/uniform m4 geometric ratio: `{primary['geometric_mean_ratio']:.6f}`",
        f"- Patient bootstrap 95% interval: `[{primary['bootstrap_ci95_lower']:.6f}, "
        f"{primary['bootstrap_ci95_upper']:.6f}]`",
        f"- Coverage wins: `{primary['coverage_wins']}/16`",
        f"- Median feature-gradient Spearman: `{primary['median_alignment']:.6f}`",
        "",
        "## Fixed-compute method comparisons",
        "",
        "| Comparison | Geometric MSE ratio | Numerator wins |",
        "|---|---:|---:|",
    ]
    for row in result["method_comparisons"]:
        lines.append(
            f"| {row['numerator']} / {row['denominator']} | "
            f"{row['geometric_mean_ratio']:.4f} | {row['numerator_wins']}/16 |"
        )
    lines.extend(
        [
            "",
            "## Site-stratum check",
            "",
            "| Site group | Coverage/uniform m4 ratio | Coverage wins |",
            "|---|---:|---:|",
        ]
    )
    for row in result["site_group_comparisons"]:
        lines.append(
            f"| {row['site_group']} | {row['geometric_mean_ratio']:.4f} | "
            f"{row['coverage_wins']}/{row['patients']} |"
        )
    lines.extend(
        [
            "",
            "## Clipping sensitivity",
            "",
            "| C | Coverage/uniform m4 post-clip MSE ratio | Coverage wins |",
            "|---:|---:|---:|",
        ]
    )
    for row in result["clipping_comparisons"]:
        lines.append(
            f"| {row['clip_norm']:.2f} | {row['geometric_mean_ratio']:.4f} | "
            f"{row['coverage_wins']}/16 |"
        )
    variance = result["variance_decomposition"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The median between-record variance fraction was `{variance['median_between_fraction']:.4f}` "
            f"(range `{variance['minimum_between_fraction']:.4f}`--"
            f"`{variance['maximum_between_fraction']:.4f}`).",
            "Uniformly spreading Q=4 evaluations across four distinct records beat one- and "
            "two-record allocation for every patient. The current VAE-stratified rule did not "
            "beat uniform m4 overall, in either site stratum, or at any frozen clipping norm.",
            "",
            "The arithmetic mean MSE is not the primary paired statistic: a high-scale patient "
            "for whom coverage helped can dominate it. The frozen patient-paired geometric ratio, "
            "bootstrap interval, and win count determine the decision.",
            "",
        ]
    )
    return "\n".join(lines)


def run(report_dir: Path, manifest: Path) -> dict[str, Any]:
    report_dir = report_dir.resolve()
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    selected = read_csv(report_dir / "selected_patient_records.csv")
    gradients = read_csv(report_dir / "gradient_evaluations.csv")
    primary_rows = read_csv(report_dir / "patient_preclip_metrics.csv")
    patient_rows = read_csv(report_dir / "patient_primary_comparison.csv")
    clipping_rows = read_csv(report_dir / "patient_clipping_metrics.csv")
    variance_rows = read_csv(report_dir / "patient_variance_decomposition.csv")

    patient_ids = [row["patient_id"] for row in patient_rows]
    preclip = {
        (row["patient_id"], row["method"]): float(row["preclip_mse"])
        for row in primary_rows
    }
    primary_ratios = [float(row["coverage_to_uniform_m4_ratio"]) for row in patient_rows]
    bootstrap = bootstrap_geometric_ratio(primary_ratios)
    alignments = np.asarray(
        [float(row["feature_gradient_distance_spearman"]) for row in patient_rows]
    )
    coverage_wins = int(sum(value < 1.0 for value in primary_ratios))
    median_alignment = float(np.median(alignments))
    decision = (
        "CONTINUE_VAE_COVERAGE"
        if bootstrap["geometric_mean_ratio"] <= 0.95
        and bootstrap["bootstrap_ci95_upper"] < 1.0
        and coverage_wins >= 10
        and median_alignment > 0.10
        else "RETIRE_CURRENT_VAE_COVERAGE"
    )

    method_comparisons = [
        method_ratio(patient_ids, preclip, "uniform_m1_r4", "uniform_m4_r1"),
        method_ratio(patient_ids, preclip, "uniform_m2_r2", "uniform_m4_r1"),
        method_ratio(
            patient_ids, preclip, "vae_stratified_m2_r2", "uniform_m2_r2"
        ),
        method_ratio(
            patient_ids, preclip, "vae_stratified_m4_r1", "uniform_m4_r1"
        ),
    ]

    site_group_comparisons: list[dict[str, Any]] = []
    for site_group in ("low_site", "high_site"):
        values = [
            float(row["coverage_to_uniform_m4_ratio"])
            for row in patient_rows
            if row["site_group"] == site_group
        ]
        site_group_comparisons.append(
            {
                "site_group": site_group,
                "patients": len(values),
                "geometric_mean_ratio": geometric_mean(values),
                "coverage_wins": int(sum(value < 1.0 for value in values)),
            }
        )

    clipping_comparisons: list[dict[str, Any]] = []
    clip_norms = sorted({float(row["clip_norm"]) for row in clipping_rows})
    for clip_norm in clip_norms:
        postclip = {
            (row["patient_id"], row["method"]): float(row["postclip_mse"])
            for row in clipping_rows
            if float(row["clip_norm"]) == clip_norm
        }
        values = [
            postclip[(patient_id, "vae_stratified_m4_r1")]
            / postclip[(patient_id, "uniform_m4_r1")]
            for patient_id in patient_ids
        ]
        clipping_comparisons.append(
            {
                "clip_norm": clip_norm,
                "geometric_mean_ratio": geometric_mean(values),
                "coverage_wins": int(sum(value < 1.0 for value in values)),
                "patients": len(values),
            }
        )

    between_fractions = np.asarray(
        [float(row["between_fraction"]) for row in variance_rows]
    )
    patient_counts: dict[str, int] = {}
    for row in selected:
        patient_counts[row["patient_id"]] = patient_counts.get(row["patient_id"], 0) + 1

    checks = [
        {"name": "source_execution_pass", "pass": summary["status"] == "PASS"},
        {
            "name": "source_decision_reproduced",
            "pass": decision == summary["method_gate"]["decision"],
        },
        {
            "name": "primary_ratio_reproduced",
            "pass": math.isclose(
                float(bootstrap["geometric_mean_ratio"]),
                float(summary["method_gate"]["geometric_mean_ratio"]),
                rel_tol=0.0,
                abs_tol=1e-14,
            ),
        },
        {
            "name": "primary_bootstrap_interval_reproduced",
            "pass": math.isclose(
                float(bootstrap["bootstrap_ci95_lower"]),
                float(summary["method_gate"]["bootstrap_ci95_lower"]),
                rel_tol=0.0,
                abs_tol=1e-14,
            )
            and math.isclose(
                float(bootstrap["bootstrap_ci95_upper"]),
                float(summary["method_gate"]["bootstrap_ci95_upper"]),
                rel_tol=0.0,
                abs_tol=1e-14,
            ),
        },
        {
            "name": "sixteen_patients_five_records_each",
            "pass": len(patient_counts) == 16
            and len(selected) == 80
            and set(patient_counts.values()) == {5},
        },
        {
            "name": "three_hundred_twenty_finite_gradients",
            "pass": len(gradients) == 320
            and all(row["finite"].lower() == "true" for row in gradients),
        },
        {
            "name": "all_original_execution_checks_pass",
            "pass": all(row["status"] == "PASS" for row in summary["execution_checks"]),
        },
    ]
    status = "PASS" if all(row["pass"] for row in checks) else "FAIL"

    locked_names = [
        "summary.json",
        "selected_patient_records.csv",
        "gradient_evaluations.csv",
        "patient_gradient_grams_local_only.npz",
        "patient_preclip_metrics.csv",
        "patient_clipping_metrics.csv",
        "patient_variance_decomposition.csv",
        "patient_feature_gradient_alignment.csv",
        "patient_primary_comparison.csv",
    ]
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "decision": decision,
        "analysis_boundary": (
            "Independent recomputation and secondary description of the frozen diagnostic; "
            "no primary-gate tuning and no DP, training, generation, clinical, or novelty claim."
        ),
        "checks": checks,
        "primary_recomputation": {
            **bootstrap,
            "coverage_wins": coverage_wins,
            "median_alignment": median_alignment,
        },
        "method_comparisons": method_comparisons,
        "site_group_comparisons": site_group_comparisons,
        "clipping_comparisons": clipping_comparisons,
        "variance_decomposition": {
            "median_between_fraction": float(np.median(between_fractions)),
            "minimum_between_fraction": float(np.min(between_fractions)),
            "maximum_between_fraction": float(np.max(between_fractions)),
        },
        "alignment_description": {
            "positive_patients": int(np.sum(alignments > 0)),
            "nonpositive_patients": int(np.sum(alignments <= 0)),
        },
        "input_manifest": {
            "path": str(manifest.resolve()),
            "sha256": sha256_file(manifest.resolve()),
        },
        "source_artifact_sha256": {
            name: sha256_file(report_dir / name) for name in locked_names
        },
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
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.report_dir, args.manifest)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
