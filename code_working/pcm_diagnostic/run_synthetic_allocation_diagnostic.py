#!/usr/bin/env python3
"""Synthetic controls for fixed-compute multi-record gradient allocation.

This is a measurement-harness test, not a DP mechanism or a model-training run.
It deliberately uses equal-size strata so that both uniform and stratified
estimators are unbiased without private, data-dependent reweighting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA = "pcm-synthetic-allocation-diagnostic/v1"


@dataclass(frozen=True)
class Config:
    seeds: tuple[int, ...] = (260902, 260903, 260904, 260905, 260906)
    entities: int = 16
    records_per_entity: int = 8
    strata: int = 4
    dimension: int = 32
    trials_per_entity: int = 64
    q_values: tuple[int, ...] = (2, 4, 8)
    entity_norm: float = 0.9
    record_mode_norm: float = 0.8
    within_mode_norm: float = 0.08
    perturbation_norm: float = 0.7
    clipping_norm: float = 1.0


@dataclass
class MetricAccumulator:
    count: int = 0
    preclip_mse_sum: float = 0.0
    postclip_mse_sum: float = 0.0
    clip_displacement_sum: float = 0.0
    cosine_error_sum: float = 0.0
    clipped_sum: int = 0

    def add(self, estimate: np.ndarray, reference: np.ndarray, clip_norm: float) -> None:
        clipped_estimate = clip_vector(estimate, clip_norm)
        clipped_reference = clip_vector(reference, clip_norm)
        self.count += 1
        self.preclip_mse_sum += float(np.mean(np.square(estimate - reference)))
        self.postclip_mse_sum += float(
            np.mean(np.square(clipped_estimate - clipped_reference))
        )
        self.clip_displacement_sum += float(
            np.mean(np.square(clipped_estimate - estimate))
        )
        denominator = float(np.linalg.norm(estimate) * np.linalg.norm(reference))
        cosine = float(np.dot(estimate, reference) / denominator) if denominator > 0 else 1.0
        self.cosine_error_sum += 1.0 - max(-1.0, min(1.0, cosine))
        self.clipped_sum += int(np.linalg.norm(estimate) > clip_norm)

    def means(self) -> dict[str, float]:
        if self.count == 0:
            raise RuntimeError("empty metric accumulator")
        scale = 1.0 / self.count
        return {
            "samples": self.count,
            "preclip_mse": self.preclip_mse_sum * scale,
            "postclip_mse": self.postclip_mse_sum * scale,
            "clip_displacement": self.clip_displacement_sum * scale,
            "cosine_error": self.cosine_error_sum * scale,
            "clip_rate": self.clipped_sum * scale,
        }


def clip_vector(vector: np.ndarray, clip_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= clip_norm or norm == 0.0:
        return vector.copy()
    return vector * (clip_norm / norm)


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def normalize_rows(values: np.ndarray, target_norm: float) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    norms = np.maximum(norms, np.finfo(np.float64).eps)
    return values * (target_norm / norms)


def make_population(
    config: Config,
    seed: int,
    record_heterogeneity: bool,
    aligned_strata: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return record means, sampling-strata labels and exact entity references."""

    rng = np.random.default_rng(stable_seed("population", seed))
    base = normalize_rows(
        rng.normal(size=(config.entities, config.dimension)), config.entity_norm
    )
    labels = np.tile(
        np.arange(config.strata, dtype=np.int64),
        config.records_per_entity // config.strata,
    )

    if not record_heterogeneity:
        record_means = np.repeat(base[:, None, :], config.records_per_entity, axis=1)
    else:
        modes = rng.normal(size=(config.entities, config.strata, config.dimension))
        modes -= np.mean(modes, axis=1, keepdims=True)
        flat_modes = modes.reshape(-1, config.dimension)
        flat_modes = normalize_rows(flat_modes, config.record_mode_norm)
        modes = flat_modes.reshape(config.entities, config.strata, config.dimension)
        jitter = normalize_rows(
            rng.normal(
                size=(config.entities, config.records_per_entity, config.dimension)
            ).reshape(-1, config.dimension),
            config.within_mode_norm,
        ).reshape(config.entities, config.records_per_entity, config.dimension)
        record_means = base[:, None, :] + modes[:, labels, :] + jitter

    sampling_labels = np.repeat(labels[None, :], config.entities, axis=0)
    if record_heterogeneity and not aligned_strata:
        for entity in range(config.entities):
            sampling_labels[entity] = rng.permutation(sampling_labels[entity])

    references = np.mean(record_means, axis=1)
    return record_means, sampling_labels, references


def allowed_record_counts(q: int, records_per_entity: int) -> tuple[int, ...]:
    return tuple(
        m for m in range(1, min(q, records_per_entity) + 1) if q % m == 0
    )


def select_record_indices(
    rng: np.random.Generator,
    labels: np.ndarray,
    m: int,
    strategy: str,
) -> np.ndarray:
    n = int(labels.shape[0])
    if m < 1 or m > n:
        raise ValueError(f"m must be in [1, {n}], got {m}")
    if strategy == "uniform":
        return np.asarray(rng.choice(n, size=m, replace=False), dtype=np.int64)
    if strategy != "stratified":
        raise ValueError(f"unknown strategy: {strategy}")

    strata = np.unique(labels)
    groups = {int(label): np.flatnonzero(labels == label) for label in strata}
    if len({len(indices) for indices in groups.values()}) != 1:
        raise ValueError("v1 synthetic diagnostic requires equal-size strata")

    allocation = {int(label): m // len(strata) for label in strata}
    remainder = m % len(strata)
    if remainder:
        extra = rng.choice(strata, size=remainder, replace=False)
        for label in extra:
            allocation[int(label)] += 1

    chosen: list[int] = []
    for label in strata:
        count = allocation[int(label)]
        if count > len(groups[int(label)]):
            raise ValueError("requested stratified allocation exceeds stratum capacity")
        if count:
            chosen.extend(
                int(index)
                for index in rng.choice(groups[int(label)], size=count, replace=False)
            )
    if len(chosen) != m or len(set(chosen)) != m:
        raise AssertionError("stratified selector did not return m distinct records")
    return np.asarray(chosen, dtype=np.int64)


def scenario_specs() -> tuple[tuple[str, bool, bool], ...]:
    return (
        ("no_record_heterogeneity", False, True),
        ("aligned_strata", True, True),
        ("misaligned_strata", True, False),
    )


def run_seed(config: Config, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    perturbation_scale = config.perturbation_norm / math.sqrt(config.dimension)

    for scenario, heterogeneous, aligned in scenario_specs():
        record_means, labels, references = make_population(
            config, seed, heterogeneous, aligned
        )
        for q in config.q_values:
            noise_rng = np.random.default_rng(stable_seed("noise", seed, scenario, q))
            common_noise = noise_rng.normal(
                scale=perturbation_scale,
                size=(config.entities, config.trials_per_entity, q, config.dimension),
            ).mean(axis=2)

            for strategy in ("uniform", "stratified"):
                for m in allowed_record_counts(q, config.records_per_entity):
                    r = q // m
                    selector_rng = np.random.default_rng(
                        stable_seed("selector", seed, scenario, q, strategy, m)
                    )
                    metrics = MetricAccumulator()
                    for entity in range(config.entities):
                        for trial in range(config.trials_per_entity):
                            indices = select_record_indices(
                                selector_rng, labels[entity], m, strategy
                            )
                            record_component = np.mean(
                                record_means[entity, indices], axis=0
                            )
                            estimate = record_component + common_noise[entity, trial]
                            metrics.add(
                                estimate, references[entity], config.clipping_norm
                            )
                    row = {
                        "seed": seed,
                        "scenario": scenario,
                        "q": q,
                        "strategy": strategy,
                        "m_records": m,
                        "r_perturbations_per_record": r,
                    }
                    row.update(metrics.means())
                    rows.append(row)
    return rows


def aggregate_seed_rows(seed_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = (
        "scenario",
        "q",
        "strategy",
        "m_records",
        "r_perturbations_per_record",
    )
    metric_names = (
        "preclip_mse",
        "postclip_mse",
        "clip_displacement",
        "cosine_error",
        "clip_rate",
    )
    for row in seed_rows:
        grouped[tuple(row[key] for key in keys)].append(row)

    aggregate: list[dict[str, Any]] = []
    for group_key, rows in sorted(grouped.items()):
        output = dict(zip(keys, group_key))
        output["seeds"] = len(rows)
        output["samples_per_seed"] = rows[0]["samples"]
        for metric in metric_names:
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            output[metric] = float(np.mean(values))
            output[f"{metric}_seed_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        aggregate.append(output)

    baselines = {
        (row["scenario"], row["q"]): row["preclip_mse"]
        for row in aggregate
        if row["strategy"] == "uniform" and row["m_records"] == 1
    }
    for row in aggregate:
        baseline = baselines[(row["scenario"], row["q"])]
        row["preclip_mse_ratio_to_noise_only"] = row["preclip_mse"] / baseline
    return aggregate


def find_row(
    rows: Iterable[dict[str, Any]],
    scenario: str,
    q: int,
    strategy: str,
    m: int,
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["scenario"] == scenario
        and row["q"] == q
        and row["strategy"] == strategy
        and row["m_records"] == m
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one diagnostic row, found {len(matches)}")
    return matches[0]


def build_checks(config: Config, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for q in config.q_values:
        negative = [
            row
            for row in rows
            if row["scenario"] == "no_record_heterogeneity" and row["q"] == q
        ]
        values = [row["preclip_mse"] for row in negative]
        spread = max(values) - min(values)
        checks.append(
            {
                "name": f"negative_control_allocation_invariance_q{q}",
                "status": "PASS" if spread <= 1e-12 else "FAIL",
                "observed_absolute_spread": spread,
                "threshold": 1e-12,
            }
        )

    aligned_noise = find_row(rows, "aligned_strata", 4, "uniform", 1)
    aligned_uniform = find_row(rows, "aligned_strata", 4, "uniform", 4)
    aligned_stratified = find_row(rows, "aligned_strata", 4, "stratified", 4)
    checks.extend(
        [
            {
                "name": "aligned_distinct_beats_noise_only_q4",
                "status": (
                    "PASS"
                    if aligned_uniform["preclip_mse"] < aligned_noise["preclip_mse"]
                    else "FAIL"
                ),
                "noise_only_preclip_mse": aligned_noise["preclip_mse"],
                "uniform_distinct_preclip_mse": aligned_uniform["preclip_mse"],
            },
            {
                "name": "aligned_stratification_beats_uniform_q4",
                "status": (
                    "PASS"
                    if aligned_stratified["preclip_mse"]
                    < aligned_uniform["preclip_mse"]
                    else "FAIL"
                ),
                "uniform_distinct_preclip_mse": aligned_uniform["preclip_mse"],
                "stratified_preclip_mse": aligned_stratified["preclip_mse"],
            },
        ]
    )
    return checks


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_summary_markdown(
    config: Config, rows: list[dict[str, Any]], checks: list[dict[str, Any]], status: str
) -> str:
    lines = [
        "# PCM synthetic allocation diagnostic v1",
        "",
        f"- Status: **{status}**",
        "- Scope: measurement-harness controls only; not medical evidence, DP training, or novelty evidence",
        f"- Seeds: {len(config.seeds)}",
        f"- Entities per seed: {config.entities}",
        f"- Trials per entity: {config.trials_per_entity}",
        "",
        "## Internal controls",
        "",
    ]
    for check in checks:
        lines.append(f"- `{check['name']}`: **{check['status']}**")

    lines.extend(
        [
            "",
            "## Q=4 key comparison",
            "",
            "| Scenario | Allocation | Preclip MSE | Postclip MSE | Clip rate | Ratio to noise-only |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    selected = [
        row
        for row in rows
        if row["q"] == 4
        and (
            (row["strategy"] == "uniform" and row["m_records"] in (1, 4))
            or (row["strategy"] == "stratified" and row["m_records"] == 4)
        )
    ]
    for row in selected:
        allocation = (
            "noise-only m=1,r=4"
            if row["strategy"] == "uniform" and row["m_records"] == 1
            else f"{row['strategy']} m={row['m_records']},r={row['r_perturbations_per_record']}"
        )
        lines.append(
            f"| {row['scenario']} | {allocation} | {row['preclip_mse']:.8f} | "
            f"{row['postclip_mse']:.8f} | {row['clip_rate']:.4f} | "
            f"{row['preclip_mse_ratio_to_noise_only']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A PASS only verifies expected synthetic negative and positive controls. The next gate must use frozen",
            "real-image latents and actual SD 2.1 LoRA gradients. A method claim is forbidden until a",
            "compute-matched real-gradient diagnostic and a later DP utility pilot both pass.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config: Config, output_dir: Path) -> dict[str, Any]:
    if config.records_per_entity % config.strata != 0:
        raise ValueError("records_per_entity must be divisible by strata")
    if any(q < 1 for q in config.q_values):
        raise ValueError("all Q values must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        seed_rows.extend(run_seed(config, seed))
    aggregate_rows = aggregate_seed_rows(seed_rows)
    checks = build_checks(config, aggregate_rows)
    status = "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL"

    config_payload = {
        "schema": SCHEMA,
        "config": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8"
    )
    write_csv(output_dir / "metrics_by_seed.csv", seed_rows)
    write_csv(output_dir / "metrics.csv", aggregate_rows)

    result = {
        "schema": SCHEMA,
        "status": status,
        "checks": checks,
        "artifacts": {
            "config": "config.json",
            "metrics_by_seed": "metrics_by_seed.csv",
            "metrics": "metrics.csv",
            "summary": "summary.md",
        },
        "interpretation_limit": (
            "Synthetic measurement controls only; not a medical-image result, "
            "DP guarantee, utility result, or novelty result."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        build_summary_markdown(config, aggregate_rows, checks, status),
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config()
    if args.quick:
        config = Config(seeds=(260902, 260903), entities=8, trials_per_entity=32)
    result = run(config, args.output_dir.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
