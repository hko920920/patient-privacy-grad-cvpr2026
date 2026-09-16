#!/usr/bin/env python3
"""Independent raw-Gram audit of covariance-allocation discovery results."""

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


SCHEMA = "pcm-isic-sd21-covariance-allocation-discovery-audit/v1"
PROTOCOL = "ISIC_SD21_COVARIANCE_ALLOCATION_DISCOVERY_PROTOCOL_V1.md"
EXPECTED_PROTOCOL_SHA256 = (
    "45930814495B582FF859B65B8B732294FBCDED78488A594DAB0EE776D3C38445"
)
TAGS = tuple(f"joint_{index:02d}" for index in range(1, 6))
C = 0.20
BOOTSTRAP_SEED = 26090306
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


def complement(subset: tuple[int, ...]) -> tuple[int, ...]:
    selected = set(subset)
    return tuple(index for index in range(8) if index not in selected)


def canonical(subset: tuple[int, ...]) -> tuple[int, ...]:
    selected = tuple(sorted(subset))
    other = complement(selected)
    return min(selected, other)


def templates() -> tuple[tuple[int, ...], ...]:
    return tuple(
        sorted(
            {
                canonical(tuple(values))
                for values in itertools.combinations(range(8), 4)
            }
        )
    )


def template_name(template: tuple[int, ...]) -> str:
    return "rt_" + "".join(str(value) for value in template)


def rank_map(spec_rows: list[dict[str, str]], tag: str) -> tuple[int, ...]:
    result: list[int] = []
    for bin_index in range(4):
        rows = [
            row
            for row in spec_rows
            if row["bank_tag"] == tag and int(row["bin_index"]) == bin_index
        ]
        rows = sorted(
            rows,
            key=lambda row: (int(row["timestep"]), int(row["perturbation_index"])),
        )
        if len(rows) != 2:
            raise ValueError("invalid two-per-bin bank specification")
        result.extend(int(row["perturbation_index"]) for row in rows)
    if sorted(result) != list(range(8)):
        raise ValueError("invalid perturbation map")
    return tuple(result)


def state_weights(subset: tuple[int, ...]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for records in itertools.combinations(range(5), 4):
        for assignment in itertools.permutations(subset):
            row = np.zeros(40, dtype=np.float64)
            for record, perturbation in zip(records, assignment):
                row[record * 8 + perturbation] = 0.25
            rows.append(row)
    result = np.stack(rows)
    if result.shape != (120, 40):
        raise RuntimeError("independent state enumeration failed")
    return result


def patient_moments(
    gram: np.ndarray,
    subsets: tuple[tuple[int, ...], ...],
    weights: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = np.full(40, 1.0 / 40.0)
    reference_norm = math.sqrt(max(0.0, float(reference @ gram @ reference)))
    reference_scale = min(1.0, C / reference_norm) if reference_norm > 0 else 1.0
    means = np.empty((70, 40), dtype=np.float64)
    seconds = np.empty(70, dtype=np.float64)
    rates = np.empty(70, dtype=np.float64)
    for index, _ in enumerate(subsets):
        states = weights[index]
        squared_norms = np.sum((states @ gram) * states, axis=1, dtype=np.float64)
        norms = np.sqrt(np.maximum(0.0, squared_norms))
        scales = np.minimum(
            1.0,
            np.divide(C, norms, out=np.ones_like(norms), where=norms > 0),
        )
        differences = scales[:, None] * states - reference_scale * reference
        squared_errors = np.sum(
            (differences @ gram) * differences, axis=1, dtype=np.float64
        )
        means[index] = np.mean(differences, axis=0)
        seconds[index] = float(np.mean(np.maximum(0.0, squared_errors)))
        rates[index] = float(np.mean(norms > C))
    return means, seconds, rates


def symmetric_pair(
    left: tuple[np.ndarray, np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray, np.ndarray],
    cross: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_means, left_seconds, _ = left
    right_means, right_seconds, _ = right
    forward_dot = np.sum(
        (left_means[first] @ cross) * right_means[second],
        axis=1,
        dtype=np.float64,
    )
    reverse_dot = np.sum(
        (left_means[second] @ cross) * right_means[first],
        axis=1,
        dtype=np.float64,
    )
    within = 0.125 * (
        left_seconds[first]
        + right_seconds[second]
        + left_seconds[second]
        + right_seconds[first]
    ) / dimension
    cross_component = 0.25 * (forward_dot + reverse_dot) / dimension
    total = np.maximum(0.0, within + cross_component)
    return total, within, cross_component


def geom(values: np.ndarray | list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.exp(np.mean(np.log(array))))


def select_template(ratios: np.ndarray, training: list[int]) -> tuple[int, np.ndarray]:
    scores = np.exp(np.mean(np.log(ratios[training]), axis=0))
    return int(np.argmin(scores)), scores


def independent_bootstrap(ratios: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        chosen = rng.choice(5, size=5, replace=True)
        draws[index] = geom(ratios[chosen])
    return {
        "geometric_mean_ratio": geom(ratios),
        "bootstrap_ci95_lower": float(np.quantile(draws, 0.025)),
        "bootstrap_ci95_upper": float(np.quantile(draws, 0.975)),
    }


def relerr(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-300)


def run(report_dir: Path) -> dict[str, Any]:
    report_dir = report_dir.resolve()
    summary_path = report_dir / "summary.json"
    source = json.loads(summary_path.read_text(encoding="utf-8"))
    source_summary_hash = sha256_file(summary_path)
    protocol_path = Path(__file__).with_name(PROTOCOL)
    spec_path = Path(__file__).with_name("ISIC_SD21_JOINT_BATCH_BANKS_V1.csv")
    spec_rows = read_csv(spec_path)
    template_values = templates()
    template_ids = tuple(template_name(value) for value in template_values)
    all_subsets = tuple(itertools.combinations(range(8), 4))
    subset_index = {subset: index for index, subset in enumerate(all_subsets)}
    weights = [state_weights(subset) for subset in all_subsets]
    patient_order: list[str] | None = None
    pair_indices = tuple(itertools.combinations(range(16), 2))
    dimension = int(source["configuration"]["parameter_dimension"])

    pair_mse = np.empty((5, 120, 35), dtype=np.float64)
    pair_within = np.empty_like(pair_mse)
    pair_cross = np.empty_like(pair_mse)
    patient_mse = np.empty((5, 16, 35), dtype=np.float64)
    patient_rates = np.empty_like(patient_mse)
    source_hashes_match = True

    for bank_index, bank_source in enumerate(source["source"]["bank_integrity"]):
        tag = bank_source["bank_tag"]
        if tag != TAGS[bank_index]:
            raise ValueError("bank order mismatch")
        crossgram_dir = Path(source["source"]["crossgram_directory"])
        gram_path = crossgram_dir / f"{tag}_crossgram_local_only.npy"
        metadata_path = crossgram_dir / f"{tag}_crossgram.json"
        source_hashes_match = source_hashes_match and (
            sha256_file(gram_path) == bank_source["gram_sha256"]
            and sha256_file(metadata_path) == bank_source["metadata_sha256"]
        )
        index_rows = read_csv(crossgram_dir / "gradient_index.csv")
        current_patients: list[str] = []
        for row in index_rows:
            if row["patient_id"] not in current_patients:
                current_patients.append(row["patient_id"])
        if patient_order is None:
            patient_order = current_patients
        elif patient_order != current_patients:
            raise ValueError("patient order differs across banks")

        token_map = rank_map(spec_rows, tag)
        mapped = tuple(
            tuple(sorted(token_map[token] for token in template))
            for template in template_values
        )
        first = np.asarray([subset_index[value] for value in mapped], dtype=np.int64)
        second = np.asarray(
            [subset_index[complement(value)] for value in mapped], dtype=np.int64
        )
        gram = np.load(gram_path, allow_pickle=False)
        moments: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for patient_index in range(16):
            block = slice(patient_index * 40, (patient_index + 1) * 40)
            current = patient_moments(
                np.asarray(gram[block, block], dtype=np.float64),
                all_subsets,
                weights,
            )
            moments.append(current)
            _, seconds, rates = current
            patient_mse[bank_index, patient_index] = 0.5 * (
                seconds[first] + seconds[second]
            ) / dimension
            patient_rates[bank_index, patient_index] = 0.5 * (
                rates[first] + rates[second]
            )
        for pair_index, (left_index, right_index) in enumerate(pair_indices):
            left_block = slice(left_index * 40, (left_index + 1) * 40)
            right_block = slice(right_index * 40, (right_index + 1) * 40)
            values = symmetric_pair(
                moments[left_index],
                moments[right_index],
                np.asarray(gram[left_block, right_block], dtype=np.float64),
                first,
                second,
                dimension,
            )
            pair_mse[bank_index, pair_index] = values[0]
            pair_within[bank_index, pair_index] = values[1]
            pair_cross[bank_index, pair_index] = values[2]
        del gram

    if patient_order is None:
        raise RuntimeError("no patient order")
    global_pair = np.mean(pair_mse, axis=2)
    global_patient = np.mean(patient_mse, axis=2)
    global_rate = np.mean(patient_rates, axis=2)
    bank_global = np.mean(global_pair, axis=1)
    bank_ratios = np.mean(pair_mse, axis=1) / bank_global[:, None]

    observed_pair_rows = read_csv(report_dir / "template_pair_metrics.csv")
    observed_patient_rows = read_csv(report_dir / "template_patient_metrics.csv")
    observed_bank_rows = read_csv(report_dir / "template_bank_metrics.csv")
    observed_lobo_rows = read_csv(report_dir / "lobo_results.csv")
    observed_nested_rows = read_csv(report_dir / "nested_patient_bank_leaveout.csv")
    observed_nested_summary = read_csv(
        report_dir / "nested_patient_leaveout_summary.csv"
    )
    pair_lookup = {
        (row["bank_tag"], row["left_patient_id"], row["right_patient_id"], row["template_id"]): row
        for row in observed_pair_rows
    }
    patient_lookup = {
        (row["bank_tag"], row["patient_id"], row["template_id"]): row
        for row in observed_patient_rows
    }
    bank_lookup = {
        (row["bank_tag"], row["template_id"]): row for row in observed_bank_rows
    }
    maximum_pair_error = 0.0
    maximum_patient_error = 0.0
    maximum_bank_error = 0.0
    for bank_index, tag in enumerate(TAGS):
        for pair_index, (left_index, right_index) in enumerate(pair_indices):
            for template_index, current_id in enumerate(template_ids):
                observed = pair_lookup[
                    (tag, patient_order[left_index], patient_order[right_index], current_id)
                ]
                for field, value in (
                    ("joint_mse", pair_mse[bank_index, pair_index, template_index]),
                    ("within_component_mse", pair_within[bank_index, pair_index, template_index]),
                    ("cross_component_mse", pair_cross[bank_index, pair_index, template_index]),
                ):
                    maximum_pair_error = max(
                        maximum_pair_error, relerr(float(value), float(observed[field]))
                    )
        for patient_index, patient_id in enumerate(patient_order):
            for template_index, current_id in enumerate(template_ids):
                observed = patient_lookup[(tag, patient_id, current_id)]
                maximum_patient_error = max(
                    maximum_patient_error,
                    relerr(
                        float(patient_mse[bank_index, patient_index, template_index]),
                        float(observed["marginal_mse"]),
                    ),
                    relerr(
                        float(patient_rates[bank_index, patient_index, template_index]),
                        float(observed["clip_rate"]),
                    ),
                )
        for template_index, current_id in enumerate(template_ids):
            observed = bank_lookup[(tag, current_id)]
            expected_ratio = float(bank_ratios[bank_index, template_index])
            expected_wins = int(
                np.sum(pair_mse[bank_index, :, template_index] < global_pair[bank_index])
            )
            maximum_bank_error = max(
                maximum_bank_error,
                relerr(expected_ratio, float(observed["ratio_to_global_uniform"])),
            )
            if expected_wins != int(observed["pair_wins"]):
                maximum_bank_error = math.inf

    lobo_rows: list[dict[str, Any]] = []
    for heldout_index, tag in enumerate(TAGS):
        training = [index for index in range(5) if index != heldout_index]
        selected, scores = select_template(bank_ratios, training)
        lobo_rows.append(
            {
                "heldout_bank": tag,
                "selected_template_id": template_ids[selected],
                "training_geometric_ratio": float(scores[selected]),
                "heldout_ratio": float(bank_ratios[heldout_index, selected]),
                "pair_wins": int(
                    np.sum(pair_mse[heldout_index, :, selected] < global_pair[heldout_index])
                ),
                "selected_clip_rate_mean": float(
                    np.mean(patient_rates[heldout_index, :, selected])
                ),
                "global_uniform_clip_rate_mean": float(np.mean(global_rate[heldout_index])),
            }
        )
    maximum_lobo_error = 0.0
    lobo_identity_match = len(observed_lobo_rows) == 5
    for expected, observed in zip(lobo_rows, observed_lobo_rows):
        lobo_identity_match = lobo_identity_match and (
            expected["heldout_bank"] == observed["heldout_bank"]
            and expected["selected_template_id"] == observed["selected_template_id"]
            and expected["pair_wins"] == int(observed["pair_wins"])
        )
        for field in (
            "training_geometric_ratio",
            "heldout_ratio",
            "selected_clip_rate_mean",
            "global_uniform_clip_rate_mean",
        ):
            maximum_lobo_error = max(
                maximum_lobo_error,
                relerr(float(expected[field]), float(observed[field])),
            )

    nested_rows: list[dict[str, Any]] = []
    nested_summary: list[dict[str, Any]] = []
    for patient_index, patient_id in enumerate(patient_order):
        mask = np.asarray(
            [patient_index not in pair for pair in pair_indices], dtype=bool
        )
        fold_ratios: list[float] = []
        for heldout_index, tag in enumerate(TAGS):
            training = [index for index in range(5) if index != heldout_index]
            ratios = np.empty((5, 35), dtype=np.float64)
            for bank_index in range(5):
                ratios[bank_index] = np.mean(
                    pair_mse[bank_index, mask, :], axis=0
                ) / np.mean(global_pair[bank_index, mask])
            selected, scores = select_template(ratios, training)
            heldout_ratio = float(ratios[heldout_index, selected])
            fold_ratios.append(heldout_ratio)
            nested_rows.append(
                {
                    "excluded_patient_id": patient_id,
                    "heldout_bank": tag,
                    "selected_template_id": template_ids[selected],
                    "training_geometric_ratio": float(scores[selected]),
                    "heldout_ratio": heldout_ratio,
                    "pair_wins": int(
                        np.sum(
                            pair_mse[heldout_index, mask, selected]
                            < global_pair[heldout_index, mask]
                        )
                    ),
                }
            )
        nested_summary.append(
            {
                "excluded_patient_id": patient_id,
                "equal_bank_geometric_ratio": geom(fold_ratios),
                "banks_below_one": int(sum(value < 1.0 for value in fold_ratios)),
            }
        )
    maximum_nested_error = 0.0
    nested_identity_match = len(nested_rows) == len(observed_nested_rows) == 80
    for expected, observed in zip(nested_rows, observed_nested_rows):
        nested_identity_match = nested_identity_match and (
            expected["excluded_patient_id"] == observed["excluded_patient_id"]
            and expected["heldout_bank"] == observed["heldout_bank"]
            and expected["selected_template_id"] == observed["selected_template_id"]
            and expected["pair_wins"] == int(observed["pair_wins"])
        )
        for field in ("training_geometric_ratio", "heldout_ratio"):
            maximum_nested_error = max(
                maximum_nested_error,
                relerr(float(expected[field]), float(observed[field])),
            )
    nested_summary_match = len(nested_summary) == len(observed_nested_summary) == 16
    for expected, observed in zip(nested_summary, observed_nested_summary):
        nested_summary_match = nested_summary_match and (
            expected["excluded_patient_id"] == observed["excluded_patient_id"]
            and expected["banks_below_one"] == int(observed["banks_below_one"])
            and relerr(
                float(expected["equal_bank_geometric_ratio"]),
                float(observed["equal_bank_geometric_ratio"]),
            )
            <= 2e-12
        )

    lobo_ratios = np.asarray([float(row["heldout_ratio"]) for row in lobo_rows])
    bootstrap = independent_bootstrap(lobo_ratios)
    banks_below = int(np.sum(lobo_ratios < 1.0))
    banks_pair_gate = int(sum(int(row["pair_wins"]) >= 84 for row in lobo_rows))
    nested_below = int(
        sum(float(row["equal_bank_geometric_ratio"]) < 1.0 for row in nested_summary)
    )
    selected_rate = float(
        np.mean([float(row["selected_clip_rate_mean"]) for row in lobo_rows])
    )
    baseline_rate = float(
        np.mean([float(row["global_uniform_clip_rate_mean"]) for row in lobo_rows])
    )
    conditions_pass = (
        banks_below == 5
        and bootstrap["geometric_mean_ratio"] <= 0.95
        and bootstrap["bootstrap_ci95_upper"] < 1.0
        and banks_pair_gate == 5
        and nested_below == 16
        and 0.05 <= selected_rate <= 0.95
        and 0.05 <= baseline_rate <= 0.95
    )
    expected_decision = (
        "RANK_TEMPLATE_HEADROOM_FOUND_FOR_FRESH_CONFIRMATION"
        if conditions_pass
        else "COVARIANCE_ALLOCATION_BRANCH_NOT_SUPPORTED_BY_DISCOVERY"
    )
    observed_primary = source["primary_discovery"]
    primary_match = all(
        math.isclose(
            float(observed_primary[field]), float(value), rel_tol=0, abs_tol=2e-14
        )
        for field, value in (
            ("geometric_mean_ratio", bootstrap["geometric_mean_ratio"]),
            ("bootstrap_ci95_lower", bootstrap["bootstrap_ci95_lower"]),
            ("bootstrap_ci95_upper", bootstrap["bootstrap_ci95_upper"]),
            ("mean_selected_clip_rate", selected_rate),
            ("mean_global_uniform_clip_rate", baseline_rate),
        )
    ) and all(
        int(observed_primary[field]) == value
        for field, value in (
            ("banks_below_one", banks_below),
            ("banks_with_84_pair_wins", banks_pair_gate),
            ("nested_patient_leaveouts_below_one", nested_below),
        )
    )
    artifact_hashes_match = all(
        sha256_file(report_dir / name) == digest
        for name, digest in source["artifacts_sha256"].items()
    )

    checks = [
        {
            "name": "protocol_hash_match",
            "pass": sha256_file(protocol_path) == EXPECTED_PROTOCOL_SHA256
            == source["protocol"]["sha256"],
        },
        {"name": "source_crossgram_hashes_match", "pass": source_hashes_match},
        {"name": "reported_artifact_hashes_match", "pass": artifact_hashes_match},
        {
            "name": "complete_template_pair_cells",
            "pass": len(pair_lookup) == 5 * 120 * 35,
        },
        {
            "name": "complete_template_patient_cells",
            "pass": len(patient_lookup) == 5 * 16 * 35,
        },
        {
            "name": "raw_gram_pair_metrics_reproduced",
            "pass": maximum_pair_error <= 2e-12,
            "maximum_relative_error": maximum_pair_error,
        },
        {
            "name": "raw_gram_patient_metrics_reproduced",
            "pass": maximum_patient_error <= 2e-12,
            "maximum_relative_error": maximum_patient_error,
        },
        {
            "name": "bank_metrics_reproduced",
            "pass": maximum_bank_error <= 2e-12,
            "maximum_relative_error": maximum_bank_error,
        },
        {
            "name": "lobo_selection_reproduced",
            "pass": lobo_identity_match and maximum_lobo_error <= 2e-12,
            "maximum_relative_error": maximum_lobo_error,
        },
        {
            "name": "nested_bank_patient_leaveout_reproduced",
            "pass": nested_identity_match
            and nested_summary_match
            and maximum_nested_error <= 2e-12,
            "maximum_relative_error": maximum_nested_error,
        },
        {"name": "primary_summary_reproduced", "pass": primary_match},
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
            "banks_below_one": banks_below,
            "banks_with_84_pair_wins": banks_pair_gate,
            "nested_patient_leaveouts_below_one": nested_below,
            "mean_selected_clip_rate": selected_rate,
            "mean_global_uniform_clip_rate": baseline_rate,
        },
        "source_summary_sha256": source_summary_hash,
        "analysis_boundary": (
            "Independent raw-Gram re-enumeration of post-hoc discovery primary; "
            "not fresh confirmation."
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
