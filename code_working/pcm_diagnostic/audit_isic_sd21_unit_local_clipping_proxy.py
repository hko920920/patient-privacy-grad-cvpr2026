#!/usr/bin/env python3
"""Independent raw-Gram audit of the patient-unit-local clipping proxy."""

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


SCHEMA = "pcm-isic-sd21-unit-local-clipping-proxy-secondary-audit/v1"
PROTOCOL = "ISIC_SD21_UNIT_LOCAL_CLIPPING_PROXY_PROTOCOL_V1.md"
EXPECTED_TAGS = tuple(f"repl_{index:02d}" for index in range(1, 6))
CLIP_NORMS = (0.05, 0.10, 0.20, 0.50, 1.00)
PRIMARY_CLIP_NORM = 0.20
BOOTSTRAP_SEED = 26090302
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
    array = np.asarray(values, dtype=np.float64)
    return float(np.exp(np.mean(np.log(array))))


def independent_distinct_weights() -> np.ndarray:
    rows: list[np.ndarray] = []
    for records in itertools.combinations(range(5), 4):
        for perturbations in itertools.combinations(range(8), 4):
            for assigned in itertools.permutations(perturbations):
                row = np.zeros(40, dtype=np.float64)
                for record, perturbation in zip(records, assigned):
                    row[record * 8 + perturbation] = 0.25
                rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def independent_local_weights(
    groups: tuple[tuple[int, ...], ...]
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for records in itertools.combinations(range(5), 4):
        for ordered_groups in itertools.permutations(groups):
            for assigned in itertools.product(*ordered_groups):
                row = np.zeros(40, dtype=np.float64)
                for record, perturbation in zip(records, assigned):
                    row[record * 8 + perturbation] = 0.25
                rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def groups_from_specs(rows: list[dict[str, str]]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(
            int(row["perturbation_index"])
            for row in rows
            if int(row["bin_index"]) == bin_index
        )
        for bin_index in range(4)
    )


def independent_metrics(
    weights: np.ndarray, gram: np.ndarray, dimension: int
) -> tuple[float, dict[float, dict[str, float | int]]]:
    reference = np.full(40, 1.0 / 40.0, dtype=np.float64)
    differences = weights - reference
    preclip_quadratic = np.einsum(
        "bi,ij,bj->b", differences, gram, differences, optimize=True
    )
    preclip = float(np.mean(np.maximum(0.0, preclip_quadratic)) / dimension)
    norm_squared = np.maximum(
        0.0, np.einsum("bi,ij,bj->b", weights, gram, weights, optimize=True)
    )
    norms = np.sqrt(norm_squared)
    reference_norm = math.sqrt(max(0.0, float(reference @ gram @ reference)))
    output: dict[float, dict[str, float | int]] = {}
    for clip_norm in CLIP_NORMS:
        scales = np.minimum(
            1.0,
            np.divide(
                clip_norm,
                norms,
                out=np.ones_like(norms),
                where=norms > 0,
            ),
        )
        reference_scale = (
            min(1.0, clip_norm / reference_norm) if reference_norm > 0 else 1.0
        )
        post_difference = scales[:, None] * weights - reference_scale * reference
        post_quadratic = np.einsum(
            "bi,ij,bj->b", post_difference, gram, post_difference, optimize=True
        )
        displacement = (scales - 1.0)[:, None] * weights
        displacement_quadratic = np.einsum(
            "bi,ij,bj->b", displacement, gram, displacement, optimize=True
        )
        output[float(clip_norm)] = {
            "postclip_mse": float(
                np.mean(np.maximum(0.0, post_quadratic)) / dimension
            ),
            "clip_displacement": float(
                np.mean(np.maximum(0.0, displacement_quadratic)) / dimension
            ),
            "clip_rate": float(np.mean(norms > clip_norm)),
            "reference_clipped": int(reference_norm > clip_norm),
        }
    return preclip, output


def independent_bootstrap(ratios: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        banks = rng.choice(ratios.shape[0], size=ratios.shape[0], replace=True)
        patients = rng.choice(ratios.shape[1], size=ratios.shape[1], replace=True)
        draws[index] = geom(ratios[banks][:, patients])
    return {
        "geometric_mean_ratio": geom(ratios),
        "bootstrap_ci95_lower": float(np.quantile(draws, 0.025)),
        "bootstrap_ci95_upper": float(np.quantile(draws, 0.975)),
    }


def relative_error(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), 1e-30)


def run(report_dir: Path) -> dict[str, Any]:
    report_dir = report_dir.resolve()
    source = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    cells = read_csv(report_dir / "patient_bank_clip_metrics.csv")
    bank_csv = read_csv(report_dir / "bank_primary_results.csv")
    clip_csv = read_csv(report_dir / "clip_norm_summary.csv")
    cell_map = {
        (row["bank_tag"], row["patient_id"], float(row["clip_norm"])): row
        for row in cells
    }
    distinct_weights = independent_distinct_weights()
    bank_tags = [str(row["bank_tag"]) for row in source["source_banks"]]
    patient_order: list[str] | None = None
    dimension = int(source["configuration"]["parameter_dimension"])
    recomputed: dict[tuple[str, str, float], dict[str, float | int]] = {}
    maximum_cell_relative_error = 0.0
    reference_flag_match = True
    source_hash_match = True

    for source_bank in source["source_banks"]:
        bank_tag = str(source_bank["bank_tag"])
        source_dir = Path(source_bank["report_dir"])
        bank_summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
        current_patients = [
            str(row["patient_id"]) for row in bank_summary["selection"]["selected"]
        ]
        if patient_order is None:
            patient_order = current_patients
        elif current_patients != patient_order:
            raise ValueError("patient order differs across banks")
        source_hash_match = source_hash_match and all(
            source_bank[name] == sha256_file(source_dir / filename)
            for name, filename in (
                ("summary_sha256", "summary.json"),
                ("source_audit_sha256", "secondary_audit.json"),
                ("ablation_sha256", "timestep_bin_ablation.json"),
                ("gram_sha256", "patient_gradient_grams_local_only.npz"),
            )
        )
        specs = read_csv(source_dir / "perturbation_specs.csv")
        groups = groups_from_specs(specs)
        local_weights = independent_local_weights(groups)
        with np.load(source_dir / "patient_gradient_grams_local_only.npz") as archive:
            for patient_id in current_patients:
                gram = np.asarray(archive[patient_id], dtype=np.float64)
                global_preclip, global_metrics = independent_metrics(
                    distinct_weights, gram, dimension
                )
                local_preclip, local_metrics = independent_metrics(
                    local_weights, gram, dimension
                )
                for clip_norm in CLIP_NORMS:
                    global_metric = global_metrics[float(clip_norm)]
                    local_metric = local_metrics[float(clip_norm)]
                    values: dict[str, float | int] = {
                        "global_postclip_mse": float(global_metric["postclip_mse"]),
                        "unit_local_postclip_mse": float(local_metric["postclip_mse"]),
                        "unit_local_to_global_ratio": float(
                            local_metric["postclip_mse"]
                        )
                        / float(global_metric["postclip_mse"]),
                        "global_clip_rate": float(global_metric["clip_rate"]),
                        "unit_local_clip_rate": float(local_metric["clip_rate"]),
                        "global_clip_displacement": float(
                            global_metric["clip_displacement"]
                        ),
                        "unit_local_clip_displacement": float(
                            local_metric["clip_displacement"]
                        ),
                        "reference_clipped": int(global_metric["reference_clipped"]),
                        "global_preclip_mse": global_preclip,
                        "unit_local_preclip_mse": local_preclip,
                        "unit_local_to_global_preclip_ratio": local_preclip
                        / global_preclip,
                    }
                    key = (bank_tag, patient_id, float(clip_norm))
                    recomputed[key] = values
                    observed = cell_map[key]
                    for name, value in values.items():
                        if name == "reference_clipped":
                            reference_flag_match = reference_flag_match and int(
                                observed[name]
                            ) == int(value)
                        else:
                            maximum_cell_relative_error = max(
                                maximum_cell_relative_error,
                                relative_error(float(observed[name]), float(value)),
                            )

    if patient_order is None:
        raise RuntimeError("no patients audited")
    primary_matrix = np.asarray(
        [
            [
                recomputed[(bank_tag, patient_id, PRIMARY_CLIP_NORM)][
                    "unit_local_to_global_ratio"
                ]
                for patient_id in patient_order
            ]
            for bank_tag in bank_tags
        ],
        dtype=np.float64,
    )
    bootstrap = independent_bootstrap(primary_matrix)
    recomputed_bank_rows: list[dict[str, float | int | str]] = []
    for bank_index, bank_tag in enumerate(bank_tags):
        ratios = primary_matrix[bank_index]
        recomputed_bank_rows.append(
            {
                "bank_tag": bank_tag,
                "postclip_geometric_ratio": geom(ratios),
                "preclip_geometric_ratio": geom(
                    [
                        recomputed[(bank_tag, patient_id, PRIMARY_CLIP_NORM)][
                            "unit_local_to_global_preclip_ratio"
                        ]
                        for patient_id in patient_order
                    ]
                ),
                "unit_local_wins": int(np.sum(ratios < 1.0)),
                "mean_global_clip_rate": float(
                    np.mean(
                        [
                            recomputed[(bank_tag, patient_id, PRIMARY_CLIP_NORM)][
                                "global_clip_rate"
                            ]
                            for patient_id in patient_order
                        ]
                    )
                ),
                "mean_unit_local_clip_rate": float(
                    np.mean(
                        [
                            recomputed[(bank_tag, patient_id, PRIMARY_CLIP_NORM)][
                                "unit_local_clip_rate"
                            ]
                            for patient_id in patient_order
                        ]
                    )
                ),
            }
        )

    bank_csv_map = {row["bank_tag"]: row for row in bank_csv}
    maximum_bank_relative_error = 0.0
    bank_count_match = True
    for row in recomputed_bank_rows:
        observed = bank_csv_map[str(row["bank_tag"])]
        bank_count_match = bank_count_match and int(observed["unit_local_wins"]) == int(
            row["unit_local_wins"]
        )
        for name in (
            "postclip_geometric_ratio",
            "preclip_geometric_ratio",
            "mean_global_clip_rate",
            "mean_unit_local_clip_rate",
        ):
            maximum_bank_relative_error = max(
                maximum_bank_relative_error,
                relative_error(float(observed[name]), float(row[name])),
            )

    primary_cells = [
        recomputed[(bank_tag, patient_id, PRIMARY_CLIP_NORM)]
        for bank_tag in bank_tags
        for patient_id in patient_order
    ]
    banks_below_one = int(
        sum(float(row["postclip_geometric_ratio"]) < 1.0 for row in recomputed_bank_rows)
    )
    banks_with_ten_wins = int(
        sum(int(row["unit_local_wins"]) >= 10 for row in recomputed_bank_rows)
    )
    mean_global_clip_rate = float(
        np.mean([row["global_clip_rate"] for row in primary_cells])
    )
    mean_unit_local_clip_rate = float(
        np.mean([row["unit_local_clip_rate"] for row in primary_cells])
    )
    conditions_pass = (
        banks_below_one == 5
        and banks_with_ten_wins == 5
        and bootstrap["geometric_mean_ratio"] <= 0.95
        and bootstrap["bootstrap_ci95_upper"] < 1.0
        and 0.05 <= mean_global_clip_rate <= 0.95
        and 0.05 <= mean_unit_local_clip_rate <= 0.95
    )
    expected_decision = (
        "UNIT_LOCAL_CLIPPING_PROXY_PASSES_FOR_JOINT_BATCH_TEST"
        if conditions_pass
        else "UNIT_LOCAL_CLIPPING_INCREMENT_NOT_SUPPORTED"
    )
    primary = source["primary"]
    primary_numeric_match = all(
        math.isclose(
            float(primary[name]), float(bootstrap[name]), rel_tol=0.0, abs_tol=1e-14
        )
        for name in (
            "geometric_mean_ratio",
            "bootstrap_ci95_lower",
            "bootstrap_ci95_upper",
        )
    ) and all(
        math.isclose(
            float(primary[name]), value, rel_tol=0.0, abs_tol=1e-14
        )
        for name, value in (
            ("mean_global_clip_rate", mean_global_clip_rate),
            ("mean_unit_local_clip_rate", mean_unit_local_clip_rate),
        )
    )
    primary_count_match = (
        int(primary["banks_below_one"]) == banks_below_one
        and int(primary["banks_with_ten_wins"]) == banks_with_ten_wins
        and int(primary["unit_local_wins"]) == int(np.sum(primary_matrix < 1.0))
    )

    clip_csv_map = {float(row["clip_norm"]): row for row in clip_csv}
    maximum_clip_summary_relative_error = 0.0
    clip_summary_counts_match = True
    for clip_norm in CLIP_NORMS:
        matrix = np.asarray(
            [
                [
                    recomputed[(bank_tag, patient_id, float(clip_norm))][
                        "unit_local_to_global_ratio"
                    ]
                    for patient_id in patient_order
                ]
                for bank_tag in bank_tags
            ],
            dtype=np.float64,
        )
        boot = independent_bootstrap(matrix)
        observed = clip_csv_map[float(clip_norm)]
        rates_global = [
            recomputed[(bank_tag, patient_id, float(clip_norm))]["global_clip_rate"]
            for bank_tag in bank_tags
            for patient_id in patient_order
        ]
        rates_local = [
            recomputed[(bank_tag, patient_id, float(clip_norm))]["unit_local_clip_rate"]
            for bank_tag in bank_tags
            for patient_id in patient_order
        ]
        expected_values = {
            **boot,
            "mean_global_clip_rate": float(np.mean(rates_global)),
            "mean_unit_local_clip_rate": float(np.mean(rates_local)),
        }
        for name, value in expected_values.items():
            maximum_clip_summary_relative_error = max(
                maximum_clip_summary_relative_error,
                relative_error(float(observed[name]), float(value)),
            )
        clip_summary_counts_match = clip_summary_counts_match and (
            int(observed["unit_local_wins"]) == int(np.sum(matrix < 1.0))
            and int(observed["banks_below_one"])
            == int(sum(geom(matrix[index]) < 1.0 for index in range(5)))
        )

    protocol_hash_match = (
        source["protocol"]["sha256"]
        == sha256_file(Path(__file__).with_name(PROTOCOL))
    )
    checks = [
        {"name": "source_execution_pass", "pass": source["status"] == "PASS"},
        {"name": "protocol_hash_match", "pass": protocol_hash_match},
        {"name": "source_bank_hashes_match", "pass": source_hash_match},
        {"name": "exact_bank_tags", "pass": tuple(bank_tags) == EXPECTED_TAGS},
        {
            "name": "complete_unique_five_by_sixteen_by_five_cells",
            "pass": len(cells) == len(cell_map) == 400 and len(recomputed) == 400,
        },
        {
            "name": "independent_state_counts",
            "pass": distinct_weights.shape == (8400, 40),
        },
        {
            "name": "all_cell_metrics_reproduced",
            "pass": maximum_cell_relative_error <= 2e-12 and reference_flag_match,
            "maximum_relative_error": maximum_cell_relative_error,
        },
        {
            "name": "bank_summaries_reproduced",
            "pass": maximum_bank_relative_error <= 2e-12 and bank_count_match,
            "maximum_relative_error": maximum_bank_relative_error,
        },
        {
            "name": "clip_grid_summaries_reproduced",
            "pass": maximum_clip_summary_relative_error <= 2e-12
            and clip_summary_counts_match,
            "maximum_relative_error": maximum_clip_summary_relative_error,
        },
        {
            "name": "primary_summary_reproduced",
            "pass": primary_numeric_match and primary_count_match,
        },
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
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "banks_below_one": banks_below_one,
            "banks_with_ten_wins": banks_with_ten_wins,
            "unit_local_wins": int(np.sum(primary_matrix < 1.0)),
            "mean_global_clip_rate": mean_global_clip_rate,
            "mean_unit_local_clip_rate": mean_unit_local_clip_rate,
        },
        "source_artifact_sha256": {
            name: sha256_file(report_dir / name)
            for name in (
                "summary.json",
                "patient_bank_clip_metrics.csv",
                "bank_primary_results.csv",
                "clip_norm_summary.csv",
            )
        },
        "analysis_boundary": (
            "Independent raw-Gram recomputation of the patient-marginal clipping proxy only."
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
