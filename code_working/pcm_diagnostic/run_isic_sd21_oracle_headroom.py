#!/usr/bin/env python3
"""Exact oracle-headroom analysis over all ten paired-strata partitions."""

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

from run_isic_sd21_multipatient_diagnostic import (
    enumerate_stratified_weights,
    exact_estimator_metrics,
)


SCHEMA = "pcm-isic-sd21-paired-strata-oracle-headroom/v1"
PROTOCOL = "ISIC_SD21_ORACLE_HEADROOM_PROTOCOL_V1.md"


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


def geometric_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.exp(np.mean(np.log(array))))


def labels_for_pair(pair: tuple[int, int]) -> np.ndarray:
    labels = np.arange(5, dtype=np.int64)
    labels[pair[1]] = labels[pair[0]]
    return labels


def build_markdown(result: dict[str, Any], patients: list[dict[str, Any]]) -> str:
    summary = result["summary"]
    lines = [
        "# ISIC SD 2.1 paired-strata oracle headroom",
        "",
        f"- Execution status: **{result['status']}**",
        f"- Headroom decision: **{result['decision']}**",
        "- Boundary: outcome-informed oracle ceiling only; not an implementable method",
        "",
        "## Aggregate result",
        "",
        f"- Oracle/uniform m4 geometric MSE ratio: `{summary['oracle_to_uniform_geometric_ratio']:.4f}`",
        f"- Oracle wins: `{summary['oracle_wins']}/16`",
        f"- Frozen VAE/oracle geometric regret: `{summary['vae_to_oracle_geometric_ratio']:.4f}`",
        f"- Frozen VAE median rank among 10 pairs: `{summary['vae_pair_median_rank']:.2f}`",
        f"- Random-pair/uniform m4 geometric ratio: `{summary['random_pair_to_uniform_geometric_ratio']:.4f}`",
        "",
        "## Frozen headroom gate",
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
            "## Patient results",
            "",
            "| Patient | VAE pair | VAE rank | Oracle pair | Oracle/uniform | VAE/oracle |",
            "|---|---|---:|---|---:|---:|",
        ]
    )
    for row in patients:
        lines.append(
            f"| {row['patient_id']} | {row['vae_pair']} | {row['vae_pair_rank']} | "
            f"{row['oracle_pair']} | {row['oracle_to_uniform_ratio']:.4f} | "
            f"{row['vae_to_oracle_ratio']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Passing this gate only permits a separate proxy-search stage. The oracle uses the "
            "same reference gradients that define the evaluation target and cannot be reported as "
            "a deployable allocation result.",
            "",
        ]
    )
    return "\n".join(lines)


def run(report_dir: Path) -> dict[str, Any]:
    report_dir = report_dir.resolve()
    source_summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    secondary = json.loads(
        (report_dir / "secondary_audit.json").read_text(encoding="utf-8")
    )
    strata_rows = read_csv(report_dir / "coverage_strata.csv")
    primary_rows = read_csv(report_dir / "patient_preclip_metrics.csv")
    source_preclip = {
        (row["patient_id"], row["method"]): float(row["preclip_mse"])
        for row in primary_rows
    }
    with np.load(report_dir / "patient_gradient_grams_local_only.npz") as archive:
        grams = {name: np.asarray(archive[name], dtype=np.float64) for name in archive.files}

    patient_ids = [row["patient_id"] for row in source_summary["selection"]["selected"]]
    parameter_dimension = int(source_summary["model"]["parameter_dimension"])
    pair_rows: list[dict[str, Any]] = []
    patient_rows: list[dict[str, Any]] = []
    maximum_expected_weight_error = 0.0
    maximum_vae_mse_relative_error = 0.0
    matrices_valid = True

    for patient_id in patient_ids:
        gram = grams[patient_id]
        matrices_valid = matrices_valid and (
            gram.shape == (20, 20)
            and bool(np.isfinite(gram).all())
            and bool(np.allclose(gram, gram.T, rtol=0.0, atol=1e-10))
        )
        patient_strata = sorted(
            (row for row in strata_rows if row["patient_id"] == patient_id),
            key=lambda row: int(row["record_index"]),
        )
        labels = np.asarray([int(row["stratum_m4"]) for row in patient_strata])
        groups = [tuple(np.flatnonzero(labels == label)) for label in np.unique(labels)]
        paired_groups = [group for group in groups if len(group) == 2]
        if len(paired_groups) != 1:
            raise RuntimeError(f"{patient_id} does not have exactly one VAE pair")
        vae_pair = tuple(int(value) for value in paired_groups[0])

        patient_pair_rows: list[dict[str, Any]] = []
        for pair in itertools.combinations(range(5), 2):
            weights = enumerate_stratified_weights(labels_for_pair(pair), 4, 1)
            expected_error = float(
                np.max(np.abs(np.mean(weights, axis=0) - np.full(20, 1.0 / 20.0)))
            )
            maximum_expected_weight_error = max(
                maximum_expected_weight_error, expected_error
            )
            metrics, _ = exact_estimator_metrics(
                weights, gram, parameter_dimension, ()
            )
            row = {
                "patient_id": patient_id,
                "pair_left": pair[0],
                "pair_right": pair[1],
                "is_frozen_vae_pair": int(pair == vae_pair),
                "preclip_mse": float(metrics["preclip_mse"]),
                "max_expected_weight_error": expected_error,
            }
            patient_pair_rows.append(row)
            pair_rows.append(row)

        ordered = sorted(
            patient_pair_rows,
            key=lambda row: (row["preclip_mse"], row["pair_left"], row["pair_right"]),
        )
        oracle = ordered[0]
        anti_oracle = ordered[-1]
        vae = next(row for row in patient_pair_rows if row["is_frozen_vae_pair"] == 1)
        vae_rank = 1 + sum(
            row["preclip_mse"] < vae["preclip_mse"] for row in patient_pair_rows
        )
        uniform = source_preclip[(patient_id, "uniform_m4_r1")]
        source_vae = source_preclip[(patient_id, "vae_stratified_m4_r1")]
        relative_error = abs(vae["preclip_mse"] - source_vae) / max(source_vae, 1e-30)
        maximum_vae_mse_relative_error = max(
            maximum_vae_mse_relative_error, relative_error
        )
        patient_rows.append(
            {
                "patient_id": patient_id,
                "site_group": next(
                    row["site_group"]
                    for row in source_summary["selection"]["selected"]
                    if row["patient_id"] == patient_id
                ),
                "uniform_m4_mse": uniform,
                "vae_pair": f"{vae_pair[0]}-{vae_pair[1]}",
                "vae_pair_rank": vae_rank,
                "vae_mse": vae["preclip_mse"],
                "random_pair_mean_mse": float(
                    np.mean([row["preclip_mse"] for row in patient_pair_rows])
                ),
                "oracle_pair": f"{oracle['pair_left']}-{oracle['pair_right']}",
                "oracle_mse": oracle["preclip_mse"],
                "anti_oracle_pair": f"{anti_oracle['pair_left']}-{anti_oracle['pair_right']}",
                "anti_oracle_mse": anti_oracle["preclip_mse"],
                "oracle_to_uniform_ratio": oracle["preclip_mse"] / uniform,
                "vae_to_oracle_ratio": vae["preclip_mse"] / oracle["preclip_mse"],
                "random_pair_to_uniform_ratio": float(
                    np.mean([row["preclip_mse"] for row in patient_pair_rows]) / uniform
                ),
            }
        )

    oracle_ratios = [row["oracle_to_uniform_ratio"] for row in patient_rows]
    oracle_wins = int(sum(value < 1.0 for value in oracle_ratios))
    oracle_geometric = geometric_mean(oracle_ratios)
    conditions = [
        {
            "name": "oracle_geometric_ratio_le_0.90",
            "status": "PASS" if oracle_geometric <= 0.90 else "FAIL",
            "observed": oracle_geometric,
            "required": "<=0.90",
        },
        {
            "name": "oracle_wins_ge_12_of_16",
            "status": "PASS" if oracle_wins >= 12 else "FAIL",
            "observed": oracle_wins,
            "required": ">=12",
        },
    ]
    decision = (
        "CONTINUE_PAIR_PROXY_SEARCH"
        if all(row["status"] == "PASS" for row in conditions)
        else "RETIRE_PAIR_STRATIFIED_M4_FAMILY"
    )
    execution_checks = [
        {
            "name": "source_and_secondary_audit_pass",
            "status": "PASS"
            if source_summary["status"] == secondary["status"] == "PASS"
            else "FAIL",
        },
        {
            "name": "sixteen_gram_matrices_valid",
            "status": "PASS"
            if len(grams) == 16 and matrices_valid
            else "FAIL",
        },
        {
            "name": "ten_pairs_per_patient",
            "status": "PASS" if len(pair_rows) == 160 else "FAIL",
        },
        {
            "name": "all_pair_estimators_unbiased",
            "status": "PASS"
            if maximum_expected_weight_error <= 1e-12
            else "FAIL",
            "maximum_expected_weight_error": maximum_expected_weight_error,
        },
        {
            "name": "frozen_vae_mse_reproduced",
            "status": "PASS"
            if maximum_vae_mse_relative_error <= 1e-10
            else "FAIL",
            "maximum_relative_error": maximum_vae_mse_relative_error,
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
            "Outcome-informed reference-gradient oracle ceiling only; not an implementable "
            "allocation method, efficacy result, DP result, or novelty claim."
        ),
        "execution_checks": execution_checks,
        "conditions": conditions,
        "summary": {
            "patients": len(patient_rows),
            "partitions_per_patient": 10,
            "oracle_to_uniform_geometric_ratio": oracle_geometric,
            "oracle_wins": oracle_wins,
            "vae_to_oracle_geometric_ratio": geometric_mean(
                [row["vae_to_oracle_ratio"] for row in patient_rows]
            ),
            "vae_pair_median_rank": float(
                np.median([row["vae_pair_rank"] for row in patient_rows])
            ),
            "random_pair_to_uniform_geometric_ratio": geometric_mean(
                [row["random_pair_to_uniform_ratio"] for row in patient_rows]
            ),
            "anti_oracle_to_uniform_geometric_ratio": geometric_mean(
                [row["anti_oracle_mse"] / row["uniform_m4_mse"] for row in patient_rows]
            ),
        },
        "artifacts": {
            "pair_results": "oracle_pair_results.csv",
            "patient_results": "oracle_patient_results.csv",
            "summary": "oracle_headroom.md",
        },
    }
    write_csv(report_dir / "oracle_pair_results.csv", pair_rows)
    write_csv(report_dir / "oracle_patient_results.csv", patient_rows)
    (report_dir / "oracle_headroom.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (report_dir / "oracle_headroom.md").write_text(
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
