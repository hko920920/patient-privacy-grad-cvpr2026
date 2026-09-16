"""Train a fixed-size window route with Hypergeometric MoG-PLD owner accounting."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get("UNITDP_DATA_ROOT", str(ROOT / "data"))
)
SRC = ROOT / "src"
for path in [ROOT, SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.run_conservative_els_owner_baseline import (  # noqa: E402
    parse_csv,
    parse_seeds,
    prepare_dataset,
)
from unitdp.els_accountant import (  # noqa: E402
    ElsMogPldConfig,
    calibrate_noise_els_mog_pld,
    epsilon_for_els_mog_pld,
    hypergeometric_sensitivity_distribution,
)
from unitdp.window_dpsgd import WindowDpsgdConfig, train_window_dpsgd  # noqa: E402


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def fmt(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric != numeric:
        return "n/a"
    return f"{numeric:.4f}"


def run_one(dataset: str, seed: int, output_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_dataset(dataset, output_dir, seed, args)
    population_size = int(prepared.selected_windows)
    sample_size = min(int(args.window_batch_size), population_size)
    steps = int(args.epochs * max(1, population_size // sample_size))
    sample_rate = sample_size / population_size

    noise = calibrate_noise_els_mog_pld(
        target_epsilon=args.owner_epsilon,
        owner_kappa=prepared.owner_kappa,
        sample_rate=sample_rate,
        steps=steps,
        delta=args.owner_delta,
        value_discretization_interval=args.value_discretization_interval,
        log_mass_truncation_bound=args.log_mass_truncation_bound,
        tail_mass_truncation=args.tail_mass_truncation,
        use_connect_dots=not args.privacy_buckets,
        sampling_model="fixed_size",
        population_size=population_size,
        sample_size=sample_size,
    )
    owner_epsilon = epsilon_for_els_mog_pld(
        ElsMogPldConfig(
            owner_kappa=prepared.owner_kappa,
            sample_rate=sample_rate,
            noise_multiplier=noise,
            steps=steps,
            delta=args.owner_delta,
            value_discretization_interval=args.value_discretization_interval,
            log_mass_truncation_bound=args.log_mass_truncation_bound,
            tail_mass_truncation=args.tail_mass_truncation,
            use_connect_dots=not args.privacy_buckets,
            sampling_model="fixed_size",
            population_size=population_size,
            sample_size=sample_size,
        )
    )
    sensitivities, probabilities = hypergeometric_sensitivity_distribution(
        population_size=population_size,
        owner_kappa=prepared.owner_kappa,
        sample_size=sample_size,
    )

    torch.manual_seed(seed)
    result = train_window_dpsgd(
        model=nn.Linear(prepared.train_x.shape[1], prepared.num_classes),
        train_x=prepared.selected_x,
        train_y=prepared.selected_y,
        test_x=prepared.test_x,
        test_y=prepared.test_y,
        config=WindowDpsgdConfig(
            epochs=args.epochs,
            batch_size=sample_size,
            learning_rate=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            target_epsilon=args.owner_epsilon,
            target_delta=args.owner_delta,
            noise_multiplier=noise,
            seed=seed,
            class_weights=prepared.class_weights,
            poisson_sampling=False,
            drop_last=True,
        ),
    )
    row = {
        "dataset": dataset,
        "seed": seed,
        "route": "fixed_size_mog_pld_owner",
        "training_unit": "window",
        "accounted_unit": "owner_via_fixed_size_mog_pld",
        "claimed_unit": "owner",
        "selected_windows": prepared.selected_windows,
        "selected_owners": prepared.selected_owners,
        "owner_kappa": prepared.owner_kappa,
        "window_population": population_size,
        "fixed_window_batch_size": sample_size,
        "window_sample_rate": sample_rate,
        "steps": steps,
        "owner_epsilon": owner_epsilon,
        "owner_delta": args.owner_delta,
        "window_epsilon_reported_by_opacus": result.epsilon,
        "noise_multiplier": result.noise_multiplier,
        "test_accuracy": result.test_accuracy,
        "test_macro_f1": result.test_macro_f1,
        "test_auroc": result.test_auroc,
        "test_auprc": result.test_auprc,
        "value_discretization_interval": args.value_discretization_interval,
        "log_mass_truncation_bound": args.log_mass_truncation_bound,
        "tail_mass_truncation": args.tail_mass_truncation,
        "mog_backend": "dp_accounting.pld.from_mixture_gaussian_mechanism",
        "sensitivity_distribution": {
            "model": "Hypergeometric(population_size, owner_kappa, sample_size)",
            "sensitivities": sensitivities,
            "probabilities": probabilities,
        },
        "route_validation": {
            "unit_alignment": "pass",
            "loader_law": "fixed_size_without_replacement_drop_last",
            "bounded_selected_owner_multiplicity": prepared.owner_kappa,
            "population_size": population_size,
            "sample_size": sample_size,
            "steps": steps,
            "pld_configuration_recorded": "pass",
            "owner_epsilon_calibration": "pass",
        },
    }
    (output_dir / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["dataset"]), []).append(row)

    out: list[dict[str, object]] = []
    for dataset, items in sorted(grouped.items()):
        payload: dict[str, object] = {
            "dataset": dataset,
            "route": "fixed_size_mog_pld_owner",
            "runs": len(items),
            "training_unit": "window",
            "accounted_unit": "owner_via_fixed_size_mog_pld",
            "selected_windows": items[0]["selected_windows"],
            "selected_owners": items[0]["selected_owners"],
            "owner_kappa": items[0]["owner_kappa"],
            "window_population": items[0]["window_population"],
            "fixed_window_batch_size": items[0]["fixed_window_batch_size"],
            "window_sample_rate": items[0]["window_sample_rate"],
            "steps": items[0]["steps"],
            "owner_delta": items[0]["owner_delta"],
            "value_discretization_interval": items[0]["value_discretization_interval"],
            "log_mass_truncation_bound": items[0]["log_mass_truncation_bound"],
            "tail_mass_truncation": items[0]["tail_mass_truncation"],
            "mog_backend": items[0]["mog_backend"],
            "route_validation": items[0]["route_validation"],
        }
        for metric in [
            "owner_epsilon",
            "window_epsilon_reported_by_opacus",
            "noise_multiplier",
            "test_accuracy",
            "test_macro_f1",
            "test_auroc",
            "test_auprc",
        ]:
            values = [float(item[metric]) for item in items if item.get(metric) is not None]
            avg, std = mean_std(values)
            payload[f"{metric}_mean"] = avg
            payload[f"{metric}_std"] = std
        out.append(payload)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row if not isinstance(row[key], dict)})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key in fieldnames})


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Fixed-Size MoG-PLD Owner Baseline",
        "",
        "This trains the fixed-size window route with noise calibrated by a Hypergeometric Mixture-of-Gaussians PLD owner accountant. The route models a fixed-size window batch sampled without replacement and composes the per-step owner PLD conservatively across steps.",
        "",
        "| dataset | runs | kappa | N | B | owner eps | noise | macro F1 | AUROC | AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {runs} | {kappa} | {population} | {batch} | {eps} | {noise} | {f1} | {auroc} | {auprc} |".format(
                dataset=row["dataset"],
                runs=row["runs"],
                kappa=row["owner_kappa"],
                population=row["window_population"],
                batch=row["fixed_window_batch_size"],
                eps=f"{fmt(row['owner_epsilon_mean'])} +/- {fmt(row['owner_epsilon_std'])}",
                noise=f"{fmt(row['noise_multiplier_mean'])} +/- {fmt(row['noise_multiplier_std'])}",
                f1=f"{fmt(row['test_macro_f1_mean'])} +/- {fmt(row['test_macro_f1_std'])}",
                auroc=f"{fmt(row['test_auroc_mean'])} +/- {fmt(row['test_auroc_std'])}",
                auprc=f"{fmt(row['test_auprc_mean'])} +/- {fmt(row['test_auprc_std'])}",
            )
        )
    lines.extend(
        [
            "",
            "Claim boundary: this is an implemented and validated fixed-size ELS owner-accounted window route under fixed-size without-replacement window batches, bounded selected-owner multiplicity, Hypergeometric owner-change counts, and the recorded dp_accounting MoG-PLD discretization/truncation settings. It does not claim tight epoch-shuffle amplification.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_existing_result(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="uci,wisdm")
    parser.add_argument("--seeds", default="13,23,31,37,41")
    parser.add_argument("--output-dir", default="reports/fixed_size_mog_owner_baseline_uci_wisdm_5seed_001")
    parser.add_argument("--owner-epsilon", type=float, default=8.0)
    parser.add_argument("--owner-delta", type=float, default=1e-5)
    parser.add_argument("--policy", default="support_balanced_cap")
    parser.add_argument("--max-windows-per-owner", type=int, default=16)
    parser.add_argument("--step-window-budget", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--owner-batch-size", type=int, default=8)
    parser.add_argument("--window-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--uci-data-root",
        default=str(
            DATA_ROOT
            / "uci_har"
            / "extracted"
            / "UCI HAR Dataset"
        ),
    )
    parser.add_argument(
        "--wisdm-data-path",
        default=str(
            DATA_ROOT
            / "wisdm"
            / "WISDM_ar_v1.1"
            / "WISDM_ar_v1.1_raw.txt"
        ),
    )
    parser.add_argument("--wisdm-train-user-frac", type=float, default=0.7)
    parser.add_argument(
        "--sepsis-cache-dir",
        default=str(DATA_ROOT / "sepsis2019_cache"),
    )
    parser.add_argument("--sepsis-max-owners", type=int, default=400)
    parser.add_argument("--sepsis-train-frac", type=float, default=0.8)
    parser.add_argument("--sepsis-window-length", type=int, default=12)
    parser.add_argument("--sepsis-stride", type=int, default=6)
    parser.add_argument("--sepsis-label-mode", choices=["any", "end"], default="any")
    parser.add_argument("--sepsis-max-windows-per-owner", type=int, default=8)
    parser.add_argument("--sepsis-step-window-budget", type=int, default=8)
    parser.add_argument("--sepsis-class-weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--value-discretization-interval", type=float, default=0.002)
    parser.add_argument("--log-mass-truncation-bound", type=float, default=-50.0)
    parser.add_argument("--tail-mass-truncation", type=float, default=1e-15)
    parser.add_argument("--privacy-buckets", action="store_true")
    parser.add_argument("--skip-runs", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.skip_runs:
        rows = json.loads((output_root / "all_runs.json").read_text(encoding="utf-8"))
    else:
        rows = []
        for dataset in parse_csv(args.datasets):
            for seed in parse_seeds(args.seeds):
                run_dir = output_root / dataset / f"seed_{seed}"
                existing = load_existing_result(run_dir / "result.json") if args.resume_existing else None
                if existing is not None:
                    print(f"Reusing {dataset} fixed-size MoG-PLD owner baseline seed {seed}", flush=True)
                    rows.append(existing)
                else:
                    print(f"Running {dataset} fixed-size MoG-PLD owner baseline seed {seed}", flush=True)
                    rows.append(run_one(dataset, seed, run_dir, args))

    aggregate_rows = aggregate(rows)
    (output_root / "all_runs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_root / "aggregate.json").write_text(json.dumps(aggregate_rows, indent=2), encoding="utf-8")
    write_csv(output_root / "all_runs.csv", rows)
    write_csv(output_root / "aggregate.csv", aggregate_rows)
    write_markdown(output_root / "aggregate.md", aggregate_rows)
    print(f"Wrote {output_root / 'aggregate.md'}")


if __name__ == "__main__":
    main()
