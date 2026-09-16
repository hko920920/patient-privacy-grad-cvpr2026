#!/usr/bin/env python3
"""Post-hoc rank-template discovery from frozen full cross-patient Grams."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

import run_isic_sd21_joint_batch_locality as joint


SCHEMA = "pcm-isic-sd21-covariance-allocation-discovery/v1"
PROTOCOL = "ISIC_SD21_COVARIANCE_ALLOCATION_DISCOVERY_PROTOCOL_V1.md"
EXPECTED_PROTOCOL_SHA256 = (
    "45930814495B582FF859B65B8B732294FBCDED78488A594DAB0EE776D3C38445"
)
EXPECTED_CROSSGRAM_SUMMARY_SHA256 = (
    "9208B089AD1C93EEB41630820AE54DCD9A4F37369885775CA366AC43AEA19463"
)
EXPECTED_JOINT_SUMMARY_SHA256 = (
    "2A06856BFC37CEAE2D2194DE350D01C45C310B57D0D6BF0ECBFAC0A150A37286"
)
PRIMARY_CLIP_NORM = 0.20
BOOTSTRAP_SEED = 26090306
BOOTSTRAP_REPLICATES = 10_000


def canonical_template(subset: tuple[int, ...]) -> tuple[int, ...]:
    """Canonicalize a four-token subset modulo complement orientation."""
    selected = tuple(sorted(subset))
    if len(selected) != 4 or len(set(selected)) != 4:
        raise ValueError("template subset must contain four distinct rank tokens")
    if min(selected) < 0 or max(selected) >= 8:
        raise ValueError("rank token outside [0, 8)")
    other = tuple(token for token in range(8) if token not in set(selected))
    return min(selected, other)


def all_rank_templates() -> tuple[tuple[int, ...], ...]:
    templates = {
        canonical_template(tuple(values))
        for values in itertools.combinations(range(8), 4)
    }
    result = tuple(sorted(templates))
    if len(result) != 35:
        raise RuntimeError(f"expected 35 templates, found {len(result)}")
    return result


def template_id(template: tuple[int, ...]) -> str:
    return "rt_" + "".join(str(token) for token in template)


def template_shape(template: tuple[int, ...]) -> str:
    counts = [sum(token // 2 == bin_index for token in template) for bin_index in range(4)]
    signature = tuple(sorted(counts, reverse=True))
    labels = {
        (1, 1, 1, 1): "1111",
        (2, 1, 1, 0): "2110",
        (2, 2, 0, 0): "2200",
    }
    if signature not in labels:
        raise RuntimeError(f"unexpected template shape: {counts}")
    return labels[signature]


def rank_token_map(
    spec_rows: list[dict[str, str]], bank_tag: str
) -> tuple[int, ...]:
    """Map public (bin, within-bin timestep rank) tokens to perturbation indices."""
    mapping: list[int] = []
    for bin_index in range(4):
        rows = [
            row
            for row in spec_rows
            if row["bank_tag"] == bank_tag and int(row["bin_index"]) == bin_index
        ]
        if len(rows) != 2:
            raise ValueError(f"{bank_tag} bin {bin_index} does not have two rows")
        rows.sort(key=lambda row: (int(row["timestep"]), int(row["perturbation_index"])))
        mapping.extend(int(row["perturbation_index"]) for row in rows)
    if sorted(mapping) != list(range(8)):
        raise ValueError(f"invalid rank-token mapping for {bank_tag}: {mapping}")
    return tuple(mapping)


def mapped_subset(
    template: tuple[int, ...], token_map: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(sorted(token_map[token] for token in template))


def symmetric_pair_components(
    left_stats: dict[str, np.ndarray],
    right_stats: dict[str, np.ndarray],
    cross_gram: np.ndarray,
    clip_index: int,
    subset_indices: np.ndarray,
    complement_indices: np.ndarray,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return joint, within and cross MSE for 50:50 template orientations."""
    left_means = left_stats["mean_differences"][clip_index]
    right_means = right_stats["mean_differences"][clip_index]
    left_seconds = left_stats["mean_squared_norms"][clip_index]
    right_seconds = right_stats["mean_squared_norms"][clip_index]

    cross_forward = np.einsum(
        "bi,ij,bj->b",
        left_means[subset_indices],
        cross_gram,
        right_means[complement_indices],
        optimize=True,
    )
    cross_reverse = np.einsum(
        "bi,ij,bj->b",
        left_means[complement_indices],
        cross_gram,
        right_means[subset_indices],
        optimize=True,
    )
    within_forward = (
        0.25
        * (left_seconds[subset_indices] + right_seconds[complement_indices])
        / dimension
    )
    within_reverse = (
        0.25
        * (left_seconds[complement_indices] + right_seconds[subset_indices])
        / dimension
    )
    within = 0.5 * (within_forward + within_reverse)
    cross = 0.25 * (cross_forward + cross_reverse) / dimension
    total = within + cross
    if np.min(total) < -1e-18:
        raise RuntimeError(f"negative template joint MSE: {float(np.min(total))}")
    return np.maximum(0.0, total), within, cross


def patient_template_components(
    stats: dict[str, np.ndarray],
    clip_index: int,
    subset_indices: np.ndarray,
    complement_indices: np.ndarray,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    seconds = stats["mean_squared_norms"][clip_index]
    rates = stats["clip_rates"][clip_index]
    marginal_mse = 0.5 * (
        seconds[subset_indices] + seconds[complement_indices]
    ) / dimension
    clip_rate = 0.5 * (rates[subset_indices] + rates[complement_indices])
    return marginal_mse, clip_rate


def relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-300)


def select_training_template(
    bank_template_ratios: np.ndarray,
    training_bank_indices: list[int],
) -> tuple[int, np.ndarray]:
    scores = np.exp(
        np.mean(np.log(bank_template_ratios[training_bank_indices]), axis=0)
    )
    selected = int(np.argmin(scores))
    return selected, scores


def bank_bootstrap(ratios: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(ratios, dtype=np.float64)
    if values.shape != (5,) or np.any(values <= 0):
        raise ValueError("expected five positive held-out ratios")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    choices = rng.integers(0, 5, size=(BOOTSTRAP_REPLICATES, 5))
    draws = np.exp(np.mean(np.log(values)[choices], axis=1))
    return {
        "geometric_mean_ratio": joint.geometric_mean(values),
        "bootstrap_ci95_lower": float(np.quantile(draws, 0.025)),
        "bootstrap_ci95_upper": float(np.quantile(draws, 0.975)),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def build_markdown(result: dict[str, Any]) -> str:
    primary = result["primary_discovery"]
    lines = [
        "# ISIC SD 2.1 covariance-aware allocation discovery",
        "",
        f"- Execution status: **{result['status']}**",
        f"- Discovery decision: **{result['decision']}**",
        "- Status: post-hoc discovery; not confirmatory method evidence",
        "",
        "## Leave-one-bank-out primary",
        "",
        f"- Equal-bank held-out ratio: `{primary['geometric_mean_ratio']:.6f}`",
        f"- Bank bootstrap 95% CI: `[{primary['bootstrap_ci95_lower']:.6f}, "
        f"{primary['bootstrap_ci95_upper']:.6f}]`",
        f"- Held-out banks below one: `{primary['banks_below_one']}/5`",
        f"- Banks with at least 84/120 pair wins: "
        f"`{primary['banks_with_84_pair_wins']}/5`",
        f"- Nested patient leaveouts below one: "
        f"`{primary['nested_patient_leaveouts_below_one']}/16`",
        "",
        "## Frozen discovery gate",
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
            "## Held-out folds",
            "",
            "| Held-out bank | Selected rank template | Shape | Ratio | Pair wins |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in result["lobo_results"]:
        lines.append(
            f"| {row['heldout_bank']} | {row['selected_template_id']} | "
            f"{row['selected_template_shape']} | {row['heldout_ratio']:.6f} | "
            f"{row['pair_wins']}/120 |"
        )
    headroom = result["descriptive_headroom"]
    lines.extend(
        [
            "",
            "## Descriptive headroom",
            "",
            f"- Pair-specific hindsight oracle ratio: "
            f"`{headroom['pair_specific_oracle_geometric_ratio']:.6f}`",
            f"- Bank-shared hindsight oracle ratio: "
            f"`{headroom['bank_shared_oracle_geometric_ratio']:.6f}`",
            f"- Pooled fixed hindsight template: "
            f"`{headroom['pooled_fixed_template_id']}` "
            f"(`{headroom['pooled_fixed_geometric_ratio']:.6f}`)",
            "",
            "Hindsight oracles are upper-bound diagnostics only. The selected rule must pass "
            "fresh banks before any efficacy claim.",
            "",
        ]
    )
    return "\n".join(lines)


def run(crossgram_dir: Path, joint_report_dir: Path, output_dir: Path) -> dict[str, Any]:
    crossgram_dir = crossgram_dir.resolve()
    joint_report_dir = joint_report_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = Path(__file__).with_name(PROTOCOL)
    spec_path = Path(__file__).with_name(joint.SPEC_FILE)
    crossgram_summary_path = crossgram_dir / "crossgram_summary.json"
    joint_summary_path = joint_report_dir / "summary.json"
    crossgram_summary = json.loads(crossgram_summary_path.read_text(encoding="utf-8"))
    joint_summary = json.loads(joint_summary_path.read_text(encoding="utf-8"))
    spec_rows = joint.read_csv(spec_path)

    templates = all_rank_templates()
    template_ids = tuple(template_id(value) for value in templates)
    shapes = tuple(template_shape(value) for value in templates)
    shape_counts = {shape: shapes.count(shape) for shape in ("1111", "2110", "2200")}
    patient_order = list(crossgram_summary["selection"]["patient_order"])
    if len(patient_order) != joint.PATIENTS:
        raise ValueError("source does not contain the expected 16 patients")
    pair_indices = tuple(itertools.combinations(range(joint.PATIENTS), 2))
    if len(pair_indices) != 120:
        raise RuntimeError("patient-pair enumeration is incomplete")
    dimension = int(crossgram_summary["model"]["parameter_dimension"])
    clip_index = joint.CLIP_NORMS.index(PRIMARY_CLIP_NORM)

    all_subsets = joint.all_subsets()
    subset_to_index = {subset: index for index, subset in enumerate(all_subsets)}
    subset_weights = [joint.fixed_subset_weights(subset) for subset in all_subsets]
    reference = np.full(40, 1.0 / 40.0)
    maximum_expected_weight_error = 0.0
    for template in templates:
        first = joint.fixed_subset_weights(template)
        second = joint.fixed_subset_weights(joint.complement(template))
        expected = np.mean(np.concatenate((first, second), axis=0), axis=0)
        maximum_expected_weight_error = max(
            maximum_expected_weight_error,
            float(np.max(np.abs(expected - reference))),
        )

    banks = joint.EXPECTED_TAGS
    bank_count = len(banks)
    template_count = len(templates)
    pair_count = len(pair_indices)
    pair_mse = np.empty((bank_count, pair_count, template_count), dtype=np.float64)
    pair_within = np.empty_like(pair_mse)
    pair_cross = np.empty_like(pair_mse)
    patient_mse = np.empty(
        (bank_count, joint.PATIENTS, template_count), dtype=np.float64
    )
    patient_clip_rate = np.empty_like(patient_mse)
    source_bank_integrity: list[dict[str, Any]] = []

    bank_metadata_by_tag = {
        row["bank_tag"]: row
        for row in crossgram_summary["gradient_execution"]["bank_results"]
    }
    joint_bank_by_tag = {
        row["bank_tag"]: row for row in joint_summary["bank_primary_results"]
    }
    maximum_source_reproduction_error = 0.0

    for bank_index, bank_tag in enumerate(banks):
        gram_path = crossgram_dir / f"{bank_tag}_crossgram_local_only.npy"
        metadata_path = crossgram_dir / f"{bank_tag}_crossgram.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        gram_hash = joint.sha256_file(gram_path)
        expected_gram_hash = bank_metadata_by_tag[bank_tag]["artifacts_sha256"][
            gram_path.name
        ]
        gram = np.load(gram_path, allow_pickle=False)
        integrity = (
            metadata["status"] == "PASS"
            and gram_hash == expected_gram_hash
            and gram.shape == (640, 640)
            and np.all(np.isfinite(gram))
        )
        source_bank_integrity.append(
            {
                "bank_tag": bank_tag,
                "status": "PASS" if integrity else "FAIL",
                "gram_sha256": gram_hash,
                "metadata_sha256": joint.sha256_file(metadata_path),
            }
        )
        token_map = rank_token_map(spec_rows, bank_tag)
        actual_subsets = tuple(mapped_subset(template, token_map) for template in templates)
        actual_complements = tuple(joint.complement(subset) for subset in actual_subsets)
        first_indices = np.asarray(
            [subset_to_index[subset] for subset in actual_subsets], dtype=np.int64
        )
        second_indices = np.asarray(
            [subset_to_index[subset] for subset in actual_complements], dtype=np.int64
        )

        patient_stats: list[dict[str, np.ndarray]] = []
        for patient_index in range(joint.PATIENTS):
            block = slice(
                patient_index * joint.GRADIENTS_PER_PATIENT,
                (patient_index + 1) * joint.GRADIENTS_PER_PATIENT,
            )
            stats = joint.allocation_statistics(
                np.asarray(gram[block, block], dtype=np.float64), subset_weights
            )
            patient_stats.append(stats)
            current_mse, current_rate = patient_template_components(
                stats,
                clip_index,
                first_indices,
                second_indices,
                dimension,
            )
            patient_mse[bank_index, patient_index] = current_mse
            patient_clip_rate[bank_index, patient_index] = current_rate

        for pair_index, (left_index, right_index) in enumerate(pair_indices):
            left_block = slice(
                left_index * joint.GRADIENTS_PER_PATIENT,
                (left_index + 1) * joint.GRADIENTS_PER_PATIENT,
            )
            right_block = slice(
                right_index * joint.GRADIENTS_PER_PATIENT,
                (right_index + 1) * joint.GRADIENTS_PER_PATIENT,
            )
            current_total, current_within, current_cross = symmetric_pair_components(
                patient_stats[left_index],
                patient_stats[right_index],
                np.asarray(gram[left_block, right_block], dtype=np.float64),
                clip_index,
                first_indices,
                second_indices,
                dimension,
            )
            pair_mse[bank_index, pair_index] = current_total
            pair_within[bank_index, pair_index] = current_within
            pair_cross[bank_index, pair_index] = current_cross

        global_pair = np.mean(pair_mse[bank_index], axis=1)
        local_template_indices = np.asarray(
            [index for index, shape in enumerate(shapes) if shape == "1111"],
            dtype=np.int64,
        )
        computed_global = float(np.mean(global_pair))
        computed_local = float(
            np.mean(np.mean(pair_mse[bank_index, :, local_template_indices], axis=1))
        )
        source_bank = joint_bank_by_tag[bank_tag]
        maximum_source_reproduction_error = max(
            maximum_source_reproduction_error,
            relative_error(computed_global, float(source_bank["global_joint_mse_mean"])),
            relative_error(computed_local, float(source_bank["unit_local_joint_mse_mean"])),
        )
        del gram

    global_pair_mse = np.mean(pair_mse, axis=2)
    global_patient_mse = np.mean(patient_mse, axis=2)
    global_patient_clip_rate = np.mean(patient_clip_rate, axis=2)
    bank_global_means = np.mean(global_pair_mse, axis=1)
    bank_template_means = np.mean(pair_mse, axis=1)
    bank_template_ratios = bank_template_means / bank_global_means[:, None]

    template_catalog_rows = [
        {
            "template_id": template_ids[index],
            "rank_tokens": " ".join(str(token) for token in template),
            "complement_rank_tokens": " ".join(
                str(token) for token in joint.complement(template)
            ),
            "shape": shapes[index],
        }
        for index, template in enumerate(templates)
    ]
    template_pair_rows: list[dict[str, Any]] = []
    template_patient_rows: list[dict[str, Any]] = []
    template_bank_rows: list[dict[str, Any]] = []
    shape_bank_rows: list[dict[str, Any]] = []

    for bank_index, bank_tag in enumerate(banks):
        for pair_index, (left_index, right_index) in enumerate(pair_indices):
            baseline = float(global_pair_mse[bank_index, pair_index])
            for template_index in range(template_count):
                value = float(pair_mse[bank_index, pair_index, template_index])
                template_pair_rows.append(
                    {
                        "bank_tag": bank_tag,
                        "left_patient_id": patient_order[left_index],
                        "right_patient_id": patient_order[right_index],
                        "template_id": template_ids[template_index],
                        "shape": shapes[template_index],
                        "joint_mse": value,
                        "ratio_to_pair_global_uniform": value / baseline,
                        "within_component_mse": float(
                            pair_within[bank_index, pair_index, template_index]
                        ),
                        "cross_component_mse": float(
                            pair_cross[bank_index, pair_index, template_index]
                        ),
                    }
                )
        for patient_index, patient_id in enumerate(patient_order):
            baseline = float(global_patient_mse[bank_index, patient_index])
            for template_index in range(template_count):
                value = float(patient_mse[bank_index, patient_index, template_index])
                template_patient_rows.append(
                    {
                        "bank_tag": bank_tag,
                        "patient_id": patient_id,
                        "template_id": template_ids[template_index],
                        "shape": shapes[template_index],
                        "marginal_mse": value,
                        "ratio_to_patient_global_uniform": value / baseline,
                        "clip_rate": float(
                            patient_clip_rate[
                                bank_index, patient_index, template_index
                            ]
                        ),
                    }
                )
        for template_index in range(template_count):
            template_bank_rows.append(
                {
                    "bank_tag": bank_tag,
                    "template_id": template_ids[template_index],
                    "shape": shapes[template_index],
                    "joint_mse_mean": float(
                        bank_template_means[bank_index, template_index]
                    ),
                    "ratio_to_global_uniform": float(
                        bank_template_ratios[bank_index, template_index]
                    ),
                    "pair_wins": int(
                        np.sum(
                            pair_mse[bank_index, :, template_index]
                            < global_pair_mse[bank_index]
                        )
                    ),
                    "within_component_mse_mean": float(
                        np.mean(pair_within[bank_index, :, template_index])
                    ),
                    "cross_component_mse_mean": float(
                        np.mean(pair_cross[bank_index, :, template_index])
                    ),
                    "marginal_mse_mean": float(
                        np.mean(patient_mse[bank_index, :, template_index])
                    ),
                    "marginal_ratio_to_global_uniform": float(
                        np.mean(patient_mse[bank_index, :, template_index])
                        / np.mean(global_patient_mse[bank_index])
                    ),
                    "clip_rate_mean": float(
                        np.mean(patient_clip_rate[bank_index, :, template_index])
                    ),
                }
            )
        for shape in ("1111", "2110", "2200"):
            indices = np.asarray(
                [index for index, value in enumerate(shapes) if value == shape],
                dtype=np.int64,
            )
            shape_pair = np.mean(pair_mse[bank_index][:, indices], axis=1)
            shape_bank_rows.append(
                {
                    "bank_tag": bank_tag,
                    "shape": shape,
                    "templates": len(indices),
                    "joint_mse_mean": float(np.mean(shape_pair)),
                    "ratio_to_global_uniform": float(
                        np.mean(shape_pair) / bank_global_means[bank_index]
                    ),
                    "pair_wins": int(
                        np.sum(shape_pair < global_pair_mse[bank_index])
                    ),
                    "within_component_mse_mean": float(
                        np.mean(pair_within[bank_index, :, indices])
                    ),
                    "cross_component_mse_mean": float(
                        np.mean(pair_cross[bank_index, :, indices])
                    ),
                    "marginal_mse_mean": float(
                        np.mean(patient_mse[bank_index, :, indices])
                    ),
                    "clip_rate_mean": float(
                        np.mean(patient_clip_rate[bank_index, :, indices])
                    ),
                }
            )

    lobo_rows: list[dict[str, Any]] = []
    selected_template_indices: list[int] = []
    for heldout_index, heldout_bank in enumerate(banks):
        training_indices = [index for index in range(bank_count) if index != heldout_index]
        selected, training_scores = select_training_template(
            bank_template_ratios, training_indices
        )
        selected_template_indices.append(selected)
        current_pair_mse = pair_mse[heldout_index, :, selected]
        lobo_rows.append(
            {
                "heldout_bank": heldout_bank,
                "training_banks": " ".join(banks[index] for index in training_indices),
                "selected_template_id": template_ids[selected],
                "selected_template_rank_tokens": " ".join(
                    str(token) for token in templates[selected]
                ),
                "selected_template_shape": shapes[selected],
                "training_geometric_ratio": float(training_scores[selected]),
                "heldout_joint_mse_mean": float(np.mean(current_pair_mse)),
                "heldout_global_uniform_mse_mean": float(
                    bank_global_means[heldout_index]
                ),
                "heldout_ratio": float(bank_template_ratios[heldout_index, selected]),
                "pair_wins": int(
                    np.sum(current_pair_mse < global_pair_mse[heldout_index])
                ),
                "within_component_mse_mean": float(
                    np.mean(pair_within[heldout_index, :, selected])
                ),
                "cross_component_mse_mean": float(
                    np.mean(pair_cross[heldout_index, :, selected])
                ),
                "marginal_mse_mean": float(
                    np.mean(patient_mse[heldout_index, :, selected])
                ),
                "marginal_ratio_to_global_uniform": float(
                    np.mean(patient_mse[heldout_index, :, selected])
                    / np.mean(global_patient_mse[heldout_index])
                ),
                "selected_clip_rate_mean": float(
                    np.mean(patient_clip_rate[heldout_index, :, selected])
                ),
                "global_uniform_clip_rate_mean": float(
                    np.mean(global_patient_clip_rate[heldout_index])
                ),
            }
        )

    nested_rows: list[dict[str, Any]] = []
    nested_summary_rows: list[dict[str, Any]] = []
    for excluded_patient_index, excluded_patient_id in enumerate(patient_order):
        pair_mask = np.asarray(
            [
                excluded_patient_index not in (left, right)
                for left, right in pair_indices
            ],
            dtype=bool,
        )
        if int(np.sum(pair_mask)) != 105:
            raise RuntimeError("nested patient leaveout did not retain 105 pairs")
        patient_heldout_ratios: list[float] = []
        for heldout_index, heldout_bank in enumerate(banks):
            training_indices = [
                index for index in range(bank_count) if index != heldout_index
            ]
            leaveout_bank_ratios = np.empty(
                (bank_count, template_count), dtype=np.float64
            )
            for bank_index in range(bank_count):
                denominator = float(
                    np.mean(global_pair_mse[bank_index, pair_mask])
                )
                leaveout_bank_ratios[bank_index] = (
                    np.mean(pair_mse[bank_index, pair_mask, :], axis=0) / denominator
                )
            selected, training_scores = select_training_template(
                leaveout_bank_ratios, training_indices
            )
            heldout_ratio = float(leaveout_bank_ratios[heldout_index, selected])
            patient_heldout_ratios.append(heldout_ratio)
            nested_rows.append(
                {
                    "excluded_patient_id": excluded_patient_id,
                    "heldout_bank": heldout_bank,
                    "selected_template_id": template_ids[selected],
                    "selected_template_shape": shapes[selected],
                    "training_geometric_ratio": float(training_scores[selected]),
                    "heldout_ratio": heldout_ratio,
                    "remaining_pairs": int(np.sum(pair_mask)),
                    "pair_wins": int(
                        np.sum(
                            pair_mse[heldout_index, pair_mask, selected]
                            < global_pair_mse[heldout_index, pair_mask]
                        )
                    ),
                }
            )
        nested_ratio = joint.geometric_mean(np.asarray(patient_heldout_ratios))
        nested_summary_rows.append(
            {
                "excluded_patient_id": excluded_patient_id,
                "heldout_banks": 5,
                "remaining_pairs_per_bank": 105,
                "equal_bank_geometric_ratio": nested_ratio,
                "banks_below_one": int(
                    sum(value < 1.0 for value in patient_heldout_ratios)
                ),
            }
        )

    pair_oracle_ratios = np.asarray(
        [
            float(np.mean(np.min(pair_mse[index], axis=1)) / bank_global_means[index])
            for index in range(bank_count)
        ],
        dtype=np.float64,
    )
    bank_best_indices = np.argmin(bank_template_ratios, axis=1)
    bank_best_ratios = np.asarray(
        [bank_template_ratios[index, bank_best_indices[index]] for index in range(bank_count)]
    )
    pooled_scores = np.exp(np.mean(np.log(bank_template_ratios), axis=0))
    pooled_best_index = int(np.argmin(pooled_scores))
    lobo_ratios = np.asarray([float(row["heldout_ratio"]) for row in lobo_rows])
    bootstrap = bank_bootstrap(lobo_ratios)
    banks_below_one = int(np.sum(lobo_ratios < 1.0))
    banks_with_84_pair_wins = int(
        sum(int(row["pair_wins"]) >= 84 for row in lobo_rows)
    )
    nested_below_one = int(
        sum(float(row["equal_bank_geometric_ratio"]) < 1.0 for row in nested_summary_rows)
    )
    selected_clip_rate = float(
        np.mean([float(row["selected_clip_rate_mean"]) for row in lobo_rows])
    )
    global_clip_rate = float(
        np.mean([float(row["global_uniform_clip_rate_mean"]) for row in lobo_rows])
    )

    execution_checks = [
        {
            "name": "frozen_protocol_hash",
            "status": "PASS"
            if joint.sha256_file(protocol_path) == EXPECTED_PROTOCOL_SHA256
            else "FAIL",
        },
        {
            "name": "frozen_crossgram_summary_hash",
            "status": "PASS"
            if joint.sha256_file(crossgram_summary_path)
            == EXPECTED_CROSSGRAM_SUMMARY_SHA256
            else "FAIL",
        },
        {
            "name": "frozen_joint_summary_hash",
            "status": "PASS"
            if joint.sha256_file(joint_summary_path) == EXPECTED_JOINT_SUMMARY_SHA256
            else "FAIL",
        },
        {
            "name": "all_five_source_banks_integral",
            "status": "PASS"
            if all(row["status"] == "PASS" for row in source_bank_integrity)
            else "FAIL",
        },
        {
            "name": "complete_35_orientation_symmetric_templates",
            "status": "PASS"
            if template_count == 35
            and shape_counts == {"1111": 8, "2110": 24, "2200": 3}
            else "FAIL",
            "shape_counts": shape_counts,
        },
        {
            "name": "all_template_patient_cell_weights_unbiased",
            "status": "PASS" if maximum_expected_weight_error <= 1e-15 else "FAIL",
            "maximum_expected_weight_error": maximum_expected_weight_error,
        },
        {
            "name": "global_and_local_primary_reproduced",
            "status": "PASS" if maximum_source_reproduction_error <= 2e-12 else "FAIL",
            "maximum_relative_error": maximum_source_reproduction_error,
        },
        {
            "name": "all_template_metrics_finite_positive",
            "status": "PASS"
            if np.all(np.isfinite(pair_mse))
            and np.all(pair_mse > 0)
            and np.all(np.isfinite(patient_mse))
            and np.all(patient_mse > 0)
            else "FAIL",
        },
    ]
    integrity_pass = all(row["status"] == "PASS" for row in execution_checks)
    conditions = [
        {
            "name": "all_five_lobo_bank_ratios_lt_1",
            "status": "PASS" if banks_below_one == 5 else "FAIL",
            "observed": banks_below_one,
            "required": "5/5",
        },
        {
            "name": "lobo_equal_bank_ratio_le_0.95",
            "status": "PASS"
            if float(bootstrap["geometric_mean_ratio"]) <= 0.95
            else "FAIL",
            "observed": bootstrap["geometric_mean_ratio"],
            "required": "<=0.95",
        },
        {
            "name": "lobo_bank_bootstrap_upper_lt_1",
            "status": "PASS"
            if float(bootstrap["bootstrap_ci95_upper"]) < 1.0
            else "FAIL",
            "observed": bootstrap["bootstrap_ci95_upper"],
            "required": "<1.00",
        },
        {
            "name": "all_five_lobo_banks_win_ge_84_pairs",
            "status": "PASS" if banks_with_84_pair_wins == 5 else "FAIL",
            "observed": banks_with_84_pair_wins,
            "required": "5/5",
        },
        {
            "name": "all_16_nested_patient_leaveouts_lt_1",
            "status": "PASS" if nested_below_one == 16 else "FAIL",
            "observed": nested_below_one,
            "required": "16/16",
        },
        {
            "name": "selected_and_global_clip_rates_nondegenerate",
            "status": "PASS"
            if 0.05 <= selected_clip_rate <= 0.95
            and 0.05 <= global_clip_rate <= 0.95
            else "FAIL",
            "observed": {
                "selected": selected_clip_rate,
                "global_uniform": global_clip_rate,
            },
            "required": "both in [0.05,0.95]",
        },
        {
            "name": "source_and_enumeration_integrity",
            "status": "PASS" if integrity_pass else "FAIL",
            "observed": sum(row["status"] == "PASS" for row in execution_checks),
            "required": f"{len(execution_checks)}/{len(execution_checks)}",
        },
    ]
    all_conditions_pass = all(row["status"] == "PASS" for row in conditions)
    decision = (
        "RANK_TEMPLATE_HEADROOM_FOUND_FOR_FRESH_CONFIRMATION"
        if all_conditions_pass
        else "COVARIANCE_ALLOCATION_BRANCH_NOT_SUPPORTED_BY_DISCOVERY"
    )

    primary = {
        **bootstrap,
        "banks_below_one": banks_below_one,
        "banks_with_84_pair_wins": banks_with_84_pair_wins,
        "nested_patient_leaveouts_below_one": nested_below_one,
        "mean_selected_clip_rate": selected_clip_rate,
        "mean_global_uniform_clip_rate": global_clip_rate,
    }
    descriptive_headroom = {
        "pair_specific_oracle_bank_ratios": pair_oracle_ratios.tolist(),
        "pair_specific_oracle_geometric_ratio": joint.geometric_mean(
            pair_oracle_ratios
        ),
        "bank_shared_oracle_templates": [
            template_ids[index] for index in bank_best_indices
        ],
        "bank_shared_oracle_bank_ratios": bank_best_ratios.tolist(),
        "bank_shared_oracle_geometric_ratio": joint.geometric_mean(bank_best_ratios),
        "pooled_fixed_template_id": template_ids[pooled_best_index],
        "pooled_fixed_template_rank_tokens": list(templates[pooled_best_index]),
        "pooled_fixed_template_shape": shapes[pooled_best_index],
        "pooled_fixed_bank_ratios": bank_template_ratios[
            :, pooled_best_index
        ].tolist(),
        "pooled_fixed_geometric_ratio": float(pooled_scores[pooled_best_index]),
    }

    artifact_rows = {
        "template_catalog.csv": template_catalog_rows,
        "template_pair_metrics.csv": template_pair_rows,
        "template_patient_metrics.csv": template_patient_rows,
        "template_bank_metrics.csv": template_bank_rows,
        "shape_bank_metrics.csv": shape_bank_rows,
        "lobo_results.csv": lobo_rows,
        "nested_patient_bank_leaveout.csv": nested_rows,
        "nested_patient_leaveout_summary.csv": nested_summary_rows,
    }
    for name, rows in artifact_rows.items():
        joint.write_csv(output_dir / name, rows)

    artifact_hashes = {
        name: joint.sha256_file(output_dir / name) for name in artifact_rows
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS" if integrity_pass else "FAIL",
        "decision": decision,
        "study_status": "POST_HOC_DISCOVERY_REQUIRES_FRESH_CONFIRMATION",
        "protocol": {
            "file": PROTOCOL,
            "sha256": joint.sha256_file(protocol_path),
        },
        "source": {
            "crossgram_directory": str(crossgram_dir),
            "crossgram_summary_sha256": joint.sha256_file(crossgram_summary_path),
            "joint_report_directory": str(joint_report_dir),
            "joint_summary_sha256": joint.sha256_file(joint_summary_path),
            "bank_integrity": source_bank_integrity,
        },
        "configuration": {
            "banks": list(banks),
            "patients": len(patient_order),
            "patient_pairs": pair_count,
            "rank_templates": template_count,
            "shape_counts": shape_counts,
            "primary_clip_norm": PRIMARY_CLIP_NORM,
            "selection": "leave_one_bank_out_minimum_training_geometric_ratio",
            "template_representation": "public_within_bin_timestep_rank",
            "orientation": "symmetric_50_50",
            "parameter_dimension": dimension,
        },
        "execution_checks": execution_checks,
        "conditions": conditions,
        "primary_discovery": primary,
        "lobo_results": lobo_rows,
        "nested_patient_leaveout_summary": nested_summary_rows,
        "descriptive_headroom": descriptive_headroom,
        "shape_bank_results": shape_bank_rows,
        "artifacts_sha256": artifact_hashes,
        "analysis_boundary": (
            "Post-hoc finite-bank B=2 rank-template discovery from already observed ISIC "
            "cross-Grams; no fresh confirmation, optimizer update, DP noise, privacy, "
            "generation, medical, novelty, or venue claim."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(build_markdown(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crossgram-dir", type=Path, required=True)
    parser.add_argument("--joint-report-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.crossgram_dir, args.joint_report_dir, args.output_dir)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
