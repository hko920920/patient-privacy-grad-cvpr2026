#!/usr/bin/env python3
"""Exact B=2 clipped joint-batch locality analysis from full patient cross-Grams."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "pcm-isic-sd21-joint-batch-locality/v1"
PROTOCOL = "ISIC_SD21_JOINT_BATCH_LOCALITY_PROTOCOL_V1.md"
SPEC_FILE = "ISIC_SD21_JOINT_BATCH_BANKS_V1.csv"
EXPECTED_TAGS = tuple(f"joint_{index:02d}" for index in range(1, 6))
CLIP_NORMS = (0.05, 0.10, 0.20, 0.50, 1.00)
PRIMARY_CLIP_NORM = 0.20
BOOTSTRAP_SEED = 26090305
BOOTSTRAP_REPLICATES = 10_000
PATIENTS = 16
RECORDS = 5
PERTURBATIONS = 8
GRADIENTS_PER_PATIENT = RECORDS * PERTURBATIONS


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


def bank_bootstrap(
    ratios: np.ndarray,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    ratios = np.asarray(ratios, dtype=np.float64)
    if ratios.shape != (5,) or np.any(ratios <= 0):
        raise ValueError("expected five positive bank ratios")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(ratios), size=(replicates, len(ratios)))
    draws = np.exp(np.mean(np.log(ratios)[indices], axis=1))
    return {
        "geometric_mean_ratio": geometric_mean(ratios),
        "bootstrap_ci95_lower": float(np.quantile(draws, 0.025)),
        "bootstrap_ci95_upper": float(np.quantile(draws, 0.975)),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def all_subsets() -> tuple[tuple[int, ...], ...]:
    return tuple(itertools.combinations(range(PERTURBATIONS), 4))


def complement(subset: tuple[int, ...]) -> tuple[int, ...]:
    chosen = set(subset)
    return tuple(index for index in range(PERTURBATIONS) if index not in chosen)


def fixed_subset_weights(subset: tuple[int, ...]) -> np.ndarray:
    if len(subset) != 4 or len(set(subset)) != 4:
        raise ValueError("subset must contain four distinct perturbations")
    rows: list[np.ndarray] = []
    for record_choice in itertools.combinations(range(RECORDS), 4):
        for perturbation_assignment in itertools.permutations(subset):
            weights = np.zeros(RECORDS * PERTURBATIONS, dtype=np.float64)
            for record, perturbation in zip(record_choice, perturbation_assignment):
                weights[record * PERTURBATIONS + perturbation] = 0.25
            rows.append(weights)
    return np.stack(rows)


def groups_for_bank(spec_rows: list[dict[str, str]], bank_tag: str) -> tuple[tuple[int, ...], ...]:
    selected = [row for row in spec_rows if row["bank_tag"] == bank_tag]
    groups = tuple(
        tuple(
            int(row["perturbation_index"])
            for row in selected
            if int(row["bin_index"]) == bin_index
        )
        for bin_index in range(4)
    )
    if tuple(len(group) for group in groups) != (2, 2, 2, 2):
        raise ValueError(f"invalid frozen groups for {bank_tag}: {groups}")
    return groups


def local_subsets(groups: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(sorted(values)) for values in itertools.product(*groups))


def allocation_statistics(
    gram: np.ndarray,
    subset_weights: list[np.ndarray],
    clip_norms: tuple[float, ...] = CLIP_NORMS,
) -> dict[str, np.ndarray]:
    """Exact clipped error moments for every perturbation subset."""
    reference = np.full(RECORDS * PERTURBATIONS, 1.0 / (RECORDS * PERTURBATIONS))
    reference_norm = float(np.sqrt(max(0.0, reference @ gram @ reference)))
    subsets = len(subset_weights)
    clips = len(clip_norms)
    mean_differences = np.empty((clips, subsets, 40), dtype=np.float64)
    mean_squared_norms = np.empty((clips, subsets), dtype=np.float64)
    clip_rates = np.empty((clips, subsets), dtype=np.float64)
    for subset_index, weights in enumerate(subset_weights):
        norm_squared = np.maximum(
            0.0, np.einsum("bi,ij,bj->b", weights, gram, weights, optimize=True)
        )
        norms = np.sqrt(norm_squared)
        for clip_index, clip_norm in enumerate(clip_norms):
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
            differences = scales[:, None] * weights - reference_scale * reference
            squared = np.maximum(
                0.0,
                np.einsum(
                    "bi,ij,bj->b", differences, gram, differences, optimize=True
                ),
            )
            mean_differences[clip_index, subset_index] = np.mean(differences, axis=0)
            mean_squared_norms[clip_index, subset_index] = float(np.mean(squared))
            clip_rates[clip_index, subset_index] = float(np.mean(norms > clip_norm))
    return {
        "mean_differences": mean_differences,
        "mean_squared_norms": mean_squared_norms,
        "clip_rates": clip_rates,
        "reference_norm": np.asarray(reference_norm),
    }


def exact_joint_pair_metric(
    left_stats: dict[str, np.ndarray],
    right_stats: dict[str, np.ndarray],
    cross_gram: np.ndarray,
    clip_index: int,
    first_subset_indices: np.ndarray,
    second_subset_indices: np.ndarray,
    dimension: int,
) -> dict[str, float]:
    left_mean = left_stats["mean_differences"][clip_index, first_subset_indices]
    right_mean = right_stats["mean_differences"][clip_index, second_subset_indices]
    cross_values = np.einsum(
        "bi,ij,bj->b", left_mean, cross_gram, right_mean, optimize=True
    )
    left_squared = left_stats["mean_squared_norms"][
        clip_index, first_subset_indices
    ]
    right_squared = right_stats["mean_squared_norms"][
        clip_index, second_subset_indices
    ]
    within_component = 0.25 * float(np.mean(left_squared + right_squared)) / dimension
    cross_component = 0.50 * float(np.mean(cross_values)) / dimension
    joint_mse = within_component + cross_component
    if joint_mse < -1e-18:
        raise RuntimeError(f"negative joint MSE beyond tolerance: {joint_mse}")
    return {
        "joint_mse": max(0.0, joint_mse),
        "within_component_mse": within_component,
        "cross_component_mse": cross_component,
    }


def build_markdown(
    result: dict[str, Any], bank_rows: list[dict[str, Any]], leaveout_rows: list[dict[str, Any]]
) -> str:
    primary = result["primary"]
    lines = [
        "# ISIC SD 2.1 actual B=2 joint-batch locality",
        "",
        f"- Execution status: **{result['status']}**",
        f"- Decision: **{result['decision']}**",
        "- Scope: five new full cross-patient Gram banks and all 120 patient pairs",
        "",
        "## Frozen primary at C=0.20",
        "",
        f"- Equal-bank ratio-of-pair-means: `{primary['geometric_mean_ratio']:.4f}`",
        f"- Bank bootstrap 95% interval: `[{primary['bootstrap_ci95_lower']:.4f}, "
        f"{primary['bootstrap_ci95_upper']:.4f}]`",
        f"- Banks below one: `{primary['banks_below_one']}/5`",
        f"- Patient leave-one-out ratios below one: "
        f"`{primary['leave_one_patient_out_below_one']}/16`",
        f"- New-bank marginal local/global ratio: "
        f"`{primary['marginal_equal_bank_geometric_ratio']:.4f}`",
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
            "## Bank results",
            "",
            "| Bank | Joint local/global | Pair wins / 120 | Marginal local/global | "
            "Global MSE | Local MSE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in bank_rows:
        lines.append(
            f"| {row['bank_tag']} | {row['joint_ratio_of_means']:.4f} | "
            f"{row['unit_local_pair_wins']}/120 | {row['marginal_ratio_of_means']:.4f} | "
            f"{row['global_joint_mse_mean']:.4e} | "
            f"{row['unit_local_joint_mse_mean']:.4e} |"
        )
    lines.extend(
        [
            "",
            "## Leave-one-patient-out",
            "",
            "| Excluded patient | Joint local/global |",
            "|---|---:|",
        ]
    )
    for row in leaveout_rows:
        lines.append(
            f"| {row['excluded_patient_id']} | {row['joint_ratio_of_means']:.4f} |"
        )
    lines.extend(
        [
            "",
            "This is an exact finite-bank B=2 gradient diagnostic before DP noise. It does not "
            "establish DP training utility, privacy accounting, generation quality, clinical "
            "validity, novelty, or venue suitability.",
            "",
        ]
    )
    return "\n".join(lines)


def run(crossgram_dir: Path, output_dir: Path) -> dict[str, Any]:
    crossgram_dir = crossgram_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = Path(__file__).with_name(PROTOCOL)
    spec_path = Path(__file__).with_name(SPEC_FILE)
    crossgram_summary = json.loads(
        (crossgram_dir / "crossgram_summary.json").read_text(encoding="utf-8")
    )
    gradient_index = read_csv(crossgram_dir / "gradient_index.csv")
    spec_rows = read_csv(spec_path)
    patient_order = list(crossgram_summary["selection"]["patient_order"])
    patient_site_groups: dict[str, str] = {}
    for row in gradient_index:
        patient_site_groups[row["patient_id"]] = row["site_group"]
    dimension = int(crossgram_summary["model"]["parameter_dimension"])

    subsets = all_subsets()
    subset_to_index = {subset: index for index, subset in enumerate(subsets)}
    subset_weights = [fixed_subset_weights(subset) for subset in subsets]
    global_first_indices = np.arange(len(subsets), dtype=np.int64)
    global_second_indices = np.asarray(
        [subset_to_index[complement(subset)] for subset in subsets], dtype=np.int64
    )
    expected_reference = np.full(40, 1.0 / 40.0)
    expected_weight_error = float(
        np.max(
            np.abs(
                np.mean(np.concatenate(subset_weights, axis=0), axis=0)
                - expected_reference
            )
        )
    )

    pair_rows: list[dict[str, Any]] = []
    marginal_rows: list[dict[str, Any]] = []
    bank_clip_rows: list[dict[str, Any]] = []
    execution_checks: list[dict[str, Any]] = []
    source_bank_rows: list[dict[str, Any]] = []

    for bank_tag in EXPECTED_TAGS:
        bank_metadata_path = crossgram_dir / f"{bank_tag}_crossgram.json"
        bank_metadata = json.loads(bank_metadata_path.read_text(encoding="utf-8"))
        gram_path = crossgram_dir / f"{bank_tag}_crossgram_local_only.npy"
        gram = np.load(gram_path, allow_pickle=False)
        source_hash_match = (
            bank_metadata["artifacts_sha256"][gram_path.name] == sha256_file(gram_path)
        )
        execution_checks.append(
            {
                "name": f"{bank_tag}_crossgram_source_integrity",
                "status": "PASS"
                if bank_metadata["status"] == "PASS"
                and source_hash_match
                and gram.shape == (640, 640)
                and np.all(np.isfinite(gram))
                else "FAIL",
            }
        )
        source_bank_rows.append(
            {
                "bank_tag": bank_tag,
                "gram_file": str(gram_path),
                "gram_sha256": sha256_file(gram_path),
                "metadata_file": str(bank_metadata_path),
                "metadata_sha256": sha256_file(bank_metadata_path),
            }
        )
        local_choices = local_subsets(groups_for_bank(spec_rows, bank_tag))
        local_first_indices = np.asarray(
            [subset_to_index[subset] for subset in local_choices], dtype=np.int64
        )
        local_second_indices = np.asarray(
            [subset_to_index[complement(subset)] for subset in local_choices],
            dtype=np.int64,
        )
        patient_stats: list[dict[str, np.ndarray]] = []
        for patient_index, patient_id in enumerate(patient_order):
            patient_slice = slice(
                patient_index * GRADIENTS_PER_PATIENT,
                (patient_index + 1) * GRADIENTS_PER_PATIENT,
            )
            within_gram = np.asarray(gram[patient_slice, patient_slice], dtype=np.float64)
            stats = allocation_statistics(within_gram, subset_weights)
            patient_stats.append(stats)
            for clip_index, clip_norm in enumerate(CLIP_NORMS):
                global_mse = float(
                    np.mean(stats["mean_squared_norms"][clip_index, global_first_indices])
                    / dimension
                )
                local_mse = float(
                    np.mean(stats["mean_squared_norms"][clip_index, local_first_indices])
                    / dimension
                )
                marginal_rows.append(
                    {
                        "bank_tag": bank_tag,
                        "patient_id": patient_id,
                        "site_group": patient_site_groups[patient_id],
                        "clip_norm": float(clip_norm),
                        "global_marginal_mse": global_mse,
                        "unit_local_marginal_mse": local_mse,
                        "unit_local_to_global_ratio": local_mse / global_mse,
                        "global_clip_rate": float(
                            np.mean(stats["clip_rates"][clip_index, global_first_indices])
                        ),
                        "unit_local_clip_rate": float(
                            np.mean(stats["clip_rates"][clip_index, local_first_indices])
                        ),
                    }
                )

        for left_index, right_index in itertools.combinations(range(PATIENTS), 2):
            left_id = patient_order[left_index]
            right_id = patient_order[right_index]
            left_slice = slice(
                left_index * GRADIENTS_PER_PATIENT,
                (left_index + 1) * GRADIENTS_PER_PATIENT,
            )
            right_slice = slice(
                right_index * GRADIENTS_PER_PATIENT,
                (right_index + 1) * GRADIENTS_PER_PATIENT,
            )
            cross_gram = np.asarray(gram[left_slice, right_slice], dtype=np.float64)
            for clip_index, clip_norm in enumerate(CLIP_NORMS):
                global_metric = exact_joint_pair_metric(
                    patient_stats[left_index],
                    patient_stats[right_index],
                    cross_gram,
                    clip_index,
                    global_first_indices,
                    global_second_indices,
                    dimension,
                )
                local_metric = exact_joint_pair_metric(
                    patient_stats[left_index],
                    patient_stats[right_index],
                    cross_gram,
                    clip_index,
                    local_first_indices,
                    local_second_indices,
                    dimension,
                )
                pair_rows.append(
                    {
                        "bank_tag": bank_tag,
                        "left_patient_id": left_id,
                        "right_patient_id": right_id,
                        "left_site_group": patient_site_groups[left_id],
                        "right_site_group": patient_site_groups[right_id],
                        "clip_norm": float(clip_norm),
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
                        "global_cross_component_mse": global_metric[
                            "cross_component_mse"
                        ],
                        "unit_local_cross_component_mse": local_metric[
                            "cross_component_mse"
                        ],
                    }
                )

        for clip_norm in CLIP_NORMS:
            selected_pairs = [
                row
                for row in pair_rows
                if row["bank_tag"] == bank_tag and row["clip_norm"] == clip_norm
            ]
            selected_marginals = [
                row
                for row in marginal_rows
                if row["bank_tag"] == bank_tag and row["clip_norm"] == clip_norm
            ]
            global_mean = float(
                np.mean([row["global_joint_mse"] for row in selected_pairs])
            )
            local_mean = float(
                np.mean([row["unit_local_joint_mse"] for row in selected_pairs])
            )
            global_marginal_mean = float(
                np.mean([row["global_marginal_mse"] for row in selected_marginals])
            )
            local_marginal_mean = float(
                np.mean([row["unit_local_marginal_mse"] for row in selected_marginals])
            )
            bank_clip_rows.append(
                {
                    "bank_tag": bank_tag,
                    "clip_norm": float(clip_norm),
                    "global_joint_mse_mean": global_mean,
                    "unit_local_joint_mse_mean": local_mean,
                    "joint_ratio_of_means": local_mean / global_mean,
                    "joint_geometric_mean_pair_ratio": geometric_mean(
                        [row["unit_local_to_global_ratio"] for row in selected_pairs]
                    ),
                    "unit_local_pair_wins": int(
                        sum(row["unit_local_to_global_ratio"] < 1.0 for row in selected_pairs)
                    ),
                    "global_cross_component_mse_mean": float(
                        np.mean(
                            [row["global_cross_component_mse"] for row in selected_pairs]
                        )
                    ),
                    "unit_local_cross_component_mse_mean": float(
                        np.mean(
                            [
                                row["unit_local_cross_component_mse"]
                                for row in selected_pairs
                            ]
                        )
                    ),
                    "global_marginal_mse_mean": global_marginal_mean,
                    "unit_local_marginal_mse_mean": local_marginal_mean,
                    "marginal_ratio_of_means": local_marginal_mean
                    / global_marginal_mean,
                    "mean_global_clip_rate": float(
                        np.mean([row["global_clip_rate"] for row in selected_marginals])
                    ),
                    "mean_unit_local_clip_rate": float(
                        np.mean(
                            [row["unit_local_clip_rate"] for row in selected_marginals]
                        )
                    ),
                }
            )
        del gram

    primary_bank_rows = [
        row for row in bank_clip_rows if row["clip_norm"] == PRIMARY_CLIP_NORM
    ]
    bank_ratios = np.asarray(
        [row["joint_ratio_of_means"] for row in primary_bank_rows], dtype=np.float64
    )
    bootstrap = bank_bootstrap(bank_ratios)
    primary_pairs = [
        row for row in pair_rows if row["clip_norm"] == PRIMARY_CLIP_NORM
    ]
    primary_marginals = [
        row for row in marginal_rows if row["clip_norm"] == PRIMARY_CLIP_NORM
    ]
    leaveout_rows: list[dict[str, Any]] = []
    for patient_id in patient_order:
        rows = [
            row
            for row in primary_pairs
            if row["left_patient_id"] != patient_id
            and row["right_patient_id"] != patient_id
        ]
        global_mean = float(np.mean([row["global_joint_mse"] for row in rows]))
        local_mean = float(np.mean([row["unit_local_joint_mse"] for row in rows]))
        leaveout_rows.append(
            {
                "excluded_patient_id": patient_id,
                "remaining_pairs_per_bank": 105,
                "banks": 5,
                "global_joint_mse_mean": global_mean,
                "unit_local_joint_mse_mean": local_mean,
                "joint_ratio_of_means": local_mean / global_mean,
            }
        )

    banks_below_one = int(np.sum(bank_ratios < 1.0))
    banks_with_84_pair_wins = int(
        sum(int(row["unit_local_pair_wins"]) >= 84 for row in primary_bank_rows)
    )
    leaveouts_below_one = int(
        sum(row["joint_ratio_of_means"] < 1.0 for row in leaveout_rows)
    )
    marginal_bank_ratios = np.asarray(
        [row["marginal_ratio_of_means"] for row in primary_bank_rows],
        dtype=np.float64,
    )
    marginal_banks_below_one = int(np.sum(marginal_bank_ratios < 1.0))
    marginal_equal_bank_ratio = geometric_mean(marginal_bank_ratios)
    mean_global_clip_rate = float(
        np.mean([row["global_clip_rate"] for row in primary_marginals])
    )
    mean_local_clip_rate = float(
        np.mean([row["unit_local_clip_rate"] for row in primary_marginals])
    )
    conditions = [
        {
            "name": "all_five_joint_bank_ratios_lt_1",
            "status": "PASS" if banks_below_one == 5 else "FAIL",
            "observed": banks_below_one,
            "required": "5/5",
        },
        {
            "name": "equal_bank_joint_ratio_le_0.95",
            "status": "PASS" if bootstrap["geometric_mean_ratio"] <= 0.95 else "FAIL",
            "observed": bootstrap["geometric_mean_ratio"],
            "required": "<=0.95",
        },
        {
            "name": "bank_bootstrap_upper_lt_1",
            "status": "PASS" if bootstrap["bootstrap_ci95_upper"] < 1.0 else "FAIL",
            "observed": bootstrap["bootstrap_ci95_upper"],
            "required": "<1.00",
        },
        {
            "name": "all_five_banks_win_ge_84_of_120_pairs",
            "status": "PASS" if banks_with_84_pair_wins == 5 else "FAIL",
            "observed": banks_with_84_pair_wins,
            "required": "5/5",
        },
        {
            "name": "all_16_leave_one_patient_out_ratios_lt_1",
            "status": "PASS" if leaveouts_below_one == 16 else "FAIL",
            "observed": leaveouts_below_one,
            "required": "16/16",
        },
        {
            "name": "new_bank_marginal_replication",
            "status": "PASS"
            if marginal_banks_below_one == 5 and marginal_equal_bank_ratio <= 0.95
            else "FAIL",
            "observed": {
                "banks_below_one": marginal_banks_below_one,
                "equal_bank_ratio": marginal_equal_bank_ratio,
            },
            "required": "5/5 and <=0.95",
        },
        {
            "name": "both_mean_clip_rates_between_0.05_and_0.95",
            "status": "PASS"
            if 0.05 <= mean_global_clip_rate <= 0.95
            and 0.05 <= mean_local_clip_rate <= 0.95
            else "FAIL",
            "observed": {
                "global_only": mean_global_clip_rate,
                "unit_local": mean_local_clip_rate,
            },
            "required": "both in [0.05,0.95]",
        },
    ]
    execution_checks.extend(
        [
            {
                "name": "crossgram_aggregate_pass",
                "status": "PASS" if crossgram_summary["status"] == "PASS" else "FAIL",
            },
            {
                "name": "protocol_and_bank_spec_hash_match",
                "status": "PASS"
                if crossgram_summary["protocol"]["sha256"] == sha256_file(protocol_path)
                and crossgram_summary["bank_spec"]["sha256"] == sha256_file(spec_path)
                else "FAIL",
            },
            {
                "name": "exact_global_and_local_state_structures",
                "status": "PASS"
                if len(subsets) == 70
                and all(weights.shape == (120, 40) for weights in subset_weights)
                and len(pair_rows) == 5 * 120 * len(CLIP_NORMS)
                else "FAIL",
            },
            {
                "name": "global_patient_marginal_unbiased",
                "status": "PASS" if expected_weight_error <= 1e-12 else "FAIL",
                "maximum_expected_weight_error": expected_weight_error,
            },
            {
                "name": "all_joint_mses_finite_positive",
                "status": "PASS"
                if all(
                    np.isfinite(row["global_joint_mse"])
                    and np.isfinite(row["unit_local_joint_mse"])
                    and row["global_joint_mse"] > 0
                    and row["unit_local_joint_mse"] > 0
                    for row in pair_rows
                )
                else "FAIL",
            },
        ]
    )
    status = (
        "PASS" if all(row["status"] == "PASS" for row in execution_checks) else "FAIL"
    )
    decision = (
        "JOINT_BATCH_LOCALITY_SIGNAL_CONFIRMED_FOR_XRAY_METHOD_GATE"
        if status == "PASS" and all(row["status"] == "PASS" for row in conditions)
        else "UNIT_LOCAL_JOINT_BATCH_INCREMENT_NOT_SUPPORTED"
    )

    clip_summary_rows: list[dict[str, Any]] = []
    for clip_norm in CLIP_NORMS:
        rows = [row for row in bank_clip_rows if row["clip_norm"] == clip_norm]
        ratios = np.asarray([row["joint_ratio_of_means"] for row in rows])
        boot = bank_bootstrap(ratios)
        clip_summary_rows.append(
            {
                "clip_norm": float(clip_norm),
                **boot,
                "banks_below_one": int(np.sum(ratios < 1.0)),
                "unit_local_pair_wins": int(
                    sum(int(row["unit_local_pair_wins"]) for row in rows)
                ),
                "marginal_equal_bank_geometric_ratio": geometric_mean(
                    [row["marginal_ratio_of_means"] for row in rows]
                ),
                "mean_global_clip_rate": float(
                    np.mean([row["mean_global_clip_rate"] for row in rows])
                ),
                "mean_unit_local_clip_rate": float(
                    np.mean([row["mean_unit_local_clip_rate"] for row in rows])
                ),
            }
        )

    write_csv(output_dir / "pair_clip_metrics.csv", pair_rows)
    write_csv(output_dir / "patient_marginal_metrics.csv", marginal_rows)
    write_csv(output_dir / "bank_clip_results.csv", bank_clip_rows)
    write_csv(output_dir / "leave_one_patient_out.csv", leaveout_rows)
    write_csv(output_dir / "clip_norm_summary.csv", clip_summary_rows)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "decision": decision,
        "protocol": {"file": PROTOCOL, "sha256": sha256_file(protocol_path)},
        "bank_spec": {"file": SPEC_FILE, "sha256": sha256_file(spec_path)},
        "configuration": {
            "banks": list(EXPECTED_TAGS),
            "patients": PATIENTS,
            "unordered_patient_pairs": 120,
            "records_per_patient": RECORDS,
            "perturbations": PERTURBATIONS,
            "q_per_patient": 4,
            "primary_clip_norm": PRIMARY_CLIP_NORM,
            "secondary_clip_norms": list(CLIP_NORMS),
            "global_perturbation_allocations": 70,
            "local_perturbation_allocations": 16,
            "record_states_per_fixed_subset_per_patient": 120,
            "global_joint_states_per_pair": 1_008_000,
            "local_joint_states_per_pair": 230_400,
            "parameter_dimension": dimension,
        },
        "execution_checks": execution_checks,
        "conditions": conditions,
        "primary": {
            **bootstrap,
            "banks_below_one": banks_below_one,
            "banks_with_84_pair_wins": banks_with_84_pair_wins,
            "leave_one_patient_out_below_one": leaveouts_below_one,
            "marginal_banks_below_one": marginal_banks_below_one,
            "marginal_equal_bank_geometric_ratio": marginal_equal_bank_ratio,
            "mean_global_clip_rate": mean_global_clip_rate,
            "mean_unit_local_clip_rate": mean_local_clip_rate,
            "five_of_five_sign_probability": 1.0 / 32.0,
        },
        "bank_primary_results": primary_bank_rows,
        "leave_one_patient_out": leaveout_rows,
        "clip_norm_summary": clip_summary_rows,
        "source_crossgram": {
            "directory": str(crossgram_dir),
            "summary_sha256": sha256_file(crossgram_dir / "crossgram_summary.json"),
            "banks": source_bank_rows,
        },
        "artifacts": {
            "pair_clip_metrics": "pair_clip_metrics.csv",
            "patient_marginal_metrics": "patient_marginal_metrics.csv",
            "bank_clip_results": "bank_clip_results.csv",
            "leave_one_patient_out": "leave_one_patient_out.csv",
            "clip_norm_summary": "clip_norm_summary.csv",
            "summary": "summary.md",
        },
        "analysis_boundary": (
            "Exact B=2 finite-bank clipped update diagnostic over a frozen 16-patient cohort; "
            "no DP noise, optimizer training, generalization, privacy, clinical, novelty, or "
            "venue claim."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        build_markdown(result, primary_bank_rows, leaveout_rows), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crossgram-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.crossgram_dir, args.output_dir)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
