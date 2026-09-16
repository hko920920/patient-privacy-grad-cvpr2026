"""Ablate owner-gradient aggregation choices inside OWA-DPSGD."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get("UNITDP_DATA_ROOT", str(ROOT / "data"))
)
SRC = ROOT / "src"
for path in [ROOT, SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import run_uci_comparison as uci  # noqa: E402
from scripts import run_wisdm_owa as wisdm  # noqa: E402
from unitdp.compiler import compile_contract  # noqa: E402
from unitdp.contract import load_contract  # noqa: E402
from unitdp.models import build_classifier  # noqa: E402
from unitdp.owa_dpsgd import OwaDpsgdConfig, train_owa_dpsgd  # noqa: E402


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_seeds(text: str) -> list[int]:
    return [int(item) for item in parse_csv(text)]


def contract_args(args: argparse.Namespace, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        policy=args.policy,
        max_windows_per_owner=args.max_windows_per_owner,
        step_window_budget=args.step_window_budget,
        seed=seed,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        epochs=args.epochs,
        owner_batch_size=args.owner_batch_size,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
    )


def prepare_uci(output_dir: Path, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    train_x_raw, train_y, train_subjects = uci.load_split(Path(args.uci_data_root), "train")
    test_x_raw, test_y, _ = uci.load_split(Path(args.uci_data_root), "test")
    preprocessor = uci.load_registered_preprocessor(
        getattr(
            args,
            "uci_preprocessing_artifact",
            uci.DEFAULT_UCI_PREPROCESSING_ARTIFACT,
        ),
        getattr(
            args,
            "uci_preprocessing_artifact_sha256",
            uci.DEFAULT_UCI_PREPROCESSING_SHA256,
        ),
    )
    preprocessor.verify_reference_file(
        Path(args.uci_data_root) / "train" / "X_train.txt"
    )
    train_x = preprocessor.transform(train_x_raw)
    test_x = preprocessor.transform(test_x_raw)
    mapping_path = output_dir / "uci_train_mapping.csv"
    uci.write_mapping(uci.build_records(train_subjects, train_y, "train"), mapping_path)
    contract_path = output_dir / "owner_contract.yaml"
    uci.write_contract(
        contract_path,
        mapping_path.resolve(),
        contract_args(args, seed),
        preprocessor.contract_binding(),
    )
    route = compile_contract(load_contract(contract_path), base_dir=ROOT)
    return {
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
        "route": route,
        "num_classes": int(max(train_y.max(), test_y.max()) + 1),
    }


def prepare_wisdm(output_dir: Path, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    rows_by_user = wisdm.parse_raw_wisdm(Path(args.wisdm_data_path))
    train_owners, test_owners = wisdm.fixed_owner_split(
        rows_by_user,
        args.wisdm_train_user_frac,
    )
    train_records, train_x_raw, train_label_raw = wisdm.generate_windows(
        rows_by_user, train_owners, "train", "wisdm_train_overlap"
    )
    _test_records, test_x_raw, test_label_raw = wisdm.generate_windows(
        rows_by_user, test_owners, "test", "wisdm_test_overlap"
    )
    train_y = wisdm.encode_activity_labels(train_label_raw)
    test_y = wisdm.encode_activity_labels(test_label_raw)
    preprocessor = wisdm.load_registered_preprocessor(
        getattr(
            args,
            "wisdm_preprocessing_artifact",
            wisdm.DEFAULT_WISDM_PREPROCESSING_ARTIFACT,
        ),
        getattr(
            args,
            "wisdm_preprocessing_artifact_sha256",
            wisdm.DEFAULT_WISDM_PREPROCESSING_SHA256,
        ),
    )
    preprocessor.verify_reference_file(Path(args.wisdm_data_path))
    train_x = preprocessor.transform(train_x_raw)
    test_x = preprocessor.transform(test_x_raw)
    mapping_path = output_dir / "wisdm_train_mapping.csv"
    wisdm.write_mapping(train_records, mapping_path)
    contract_path = output_dir / "owner_contract.yaml"
    local_args = contract_args(args, seed)
    wisdm.write_contract(
        contract_path,
        mapping_path.resolve(),
        local_args,
        args.step_window_budget,
        preprocessor.contract_binding(),
    )
    route = compile_contract(load_contract(contract_path), base_dir=ROOT)
    return {
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
        "route": route,
        "num_classes": int(max(train_y.max(), test_y.max()) + 1),
    }


def prepare_dataset(dataset: str, output_dir: Path, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    if dataset == "uci":
        return prepare_uci(output_dir, seed, args)
    if dataset == "wisdm":
        return prepare_wisdm(output_dir, seed, args)
    raise ValueError(f"Unsupported dataset: {dataset}")


def run_one(dataset: str, mode: str, seed: int, output_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_dataset(dataset, output_dir, seed, args)
    route = prepared["route"]
    torch.manual_seed(seed)
    result = train_owa_dpsgd(
        model=build_classifier(prepared["train_x"].shape[1], prepared["num_classes"], "linear"),
        train_x=prepared["train_x"],
        train_y=prepared["train_y"],
        test_x=prepared["test_x"],
        test_y=prepared["test_y"],
        route=route,
        config=OwaDpsgdConfig(
            epochs=args.epochs,
            owner_batch_size=args.owner_batch_size,
            learning_rate=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            target_epsilon=args.target_epsilon,
            target_delta=args.target_delta,
            seed=seed,
            owner_aggregation=mode,
            fixed_window_normalizer=(
                args.fixed_window_normalizer
                if args.fixed_window_normalizer is not None
                else args.step_window_budget
            ),
        ),
    )
    stats = route.selected_mapping.stats()
    row = {
        "dataset": dataset,
        "seed": seed,
        "route": "owa_dpsgd",
        "owner_aggregation": mode,
        "accounted_unit": "owner",
        "selected_windows": result.selected_windows,
        "selected_owners": result.selected_owners,
        "owner_kappa": int(stats["owner_kappa"]),
        "step_window_budget": args.step_window_budget,
        "fixed_window_normalizer": (
            args.fixed_window_normalizer
            if args.fixed_window_normalizer is not None
            else args.step_window_budget
        ),
        "epsilon": result.epsilon,
        "noise_multiplier": result.noise_multiplier,
        "test_accuracy": result.test_accuracy,
        "test_macro_f1": result.test_macro_f1,
        "test_auroc": result.test_auroc,
        "test_auprc": result.test_auprc,
    }
    (output_dir / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["dataset"]), str(row["owner_aggregation"])), []).append(row)
    out: list[dict[str, object]] = []
    mode_order = {"mean": 0, "fixed_normalizer": 1, "sum": 2}
    for (dataset, mode), items in sorted(
        groups.items(), key=lambda item: (item[0][0], mode_order.get(item[0][1], 99))
    ):
        payload: dict[str, object] = {
            "dataset": dataset,
            "owner_aggregation": mode,
            "runs": len(items),
            "selected_windows": items[0]["selected_windows"],
            "selected_owners": items[0]["selected_owners"],
            "owner_kappa": items[0]["owner_kappa"],
            "step_window_budget": items[0]["step_window_budget"],
            "fixed_window_normalizer": items[0]["fixed_window_normalizer"],
        }
        for metric in [
            "epsilon",
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


def fmt(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "n/a"
    return f"{number:.4f}"


def pm(row: dict[str, object], metric: str) -> str:
    return f"{fmt(row[f'{metric}_mean'])} +/- {fmt(row[f'{metric}_std'])}"


def tex_pm(row: dict[str, object], metric: str) -> str:
    return f"${float(row[f'{metric}_mean']):.4f}\\pm {float(row[f'{metric}_std']):.4f}$"


def dataset_label(dataset: str) -> str:
    return {"uci": "UCI", "wisdm": "WISDM"}.get(dataset, dataset)


def mode_label(mode: str) -> str:
    return {
        "mean": "mean",
        "fixed_normalizer": "fixed-$b$",
        "sum": "sum",
    }.get(mode, mode)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# OWA Owner Aggregation Ablation",
        "",
        "| dataset | aggregation | runs | epsilon | noise | macro F1 | AUROC | AUPRC |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {mode} | {runs} | {eps} | {noise} | {f1} | {auroc} | {auprc} |".format(
                dataset=row["dataset"],
                mode=row["owner_aggregation"],
                runs=row["runs"],
                eps=pm(row, "epsilon"),
                noise=pm(row, "noise_multiplier"),
                f1=pm(row, "test_macro_f1"),
                auroc=pm(row, "test_auroc"),
                auprc=pm(row, "test_auprc"),
            )
        )
    lines.extend(
        [
            "",
            "All rows use the same owner sampling, owner clipping/noising, and owner accountant. The aggregation choice changes only the pre-clipping owner gradient estimator.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Dataset & Owner aggregate & Noise & Macro F1 & AUROC & AUPRC \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            "{dataset} & {mode} & {noise:.3f} & {f1} & {auroc} & {auprc} \\\\".format(
                dataset=dataset_label(str(row["dataset"])),
                mode=mode_label(str(row["owner_aggregation"])),
                noise=float(row["noise_multiplier_mean"]),
                f1=tex_pm(row, "test_macro_f1"),
                auroc=tex_pm(row, "test_auroc"),
                auprc=tex_pm(row, "test_auprc"),
            )
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{OWA-DPSGD owner-aggregation ablation at target owner $\\varepsilon=8$. Mean aggregation is the default: the owner gradient is formed from the mean loss over the sampled within-owner windows. Fixed-$b$ aggregation divides the summed selected-window loss by the public per-step budget $b$ even when fewer windows are available. Sum aggregation uses the unnormalized selected-window loss before owner-level clipping. All variants still clip and noise once per sampled owner, so the privacy route is unchanged; the table measures the utility effect of owner-gradient scaling.}",
            "\\label{tab:aggregation-ablation}",
            "\\end{table*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="uci,wisdm")
    parser.add_argument("--modes", default="mean,fixed_normalizer,sum")
    parser.add_argument("--seeds", default="13,23,31,37,41")
    parser.add_argument("--output-dir", default="reports/aggregation_ablation_5seed_001")
    parser.add_argument("--policy", default="support_balanced_cap")
    parser.add_argument("--max-windows-per-owner", type=int, default=16)
    parser.add_argument("--step-window-budget", type=int, default=8)
    parser.add_argument("--fixed-window-normalizer", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--owner-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-epsilon", type=float, default=8.0)
    parser.add_argument("--target-delta", type=float, default=1e-5)
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
        "--uci-preprocessing-artifact",
        default=str(uci.DEFAULT_UCI_PREPROCESSING_ARTIFACT),
    )
    parser.add_argument(
        "--uci-preprocessing-artifact-sha256",
        default=uci.DEFAULT_UCI_PREPROCESSING_SHA256,
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
    parser.add_argument(
        "--wisdm-preprocessing-artifact",
        default=str(wisdm.DEFAULT_WISDM_PREPROCESSING_ARTIFACT),
    )
    parser.add_argument(
        "--wisdm-preprocessing-artifact-sha256",
        default=wisdm.DEFAULT_WISDM_PREPROCESSING_SHA256,
    )
    parser.add_argument("--wisdm-train-user-frac", type=float, default=0.7)
    parser.add_argument("--paper-table", default="")
    parser.add_argument("--skip-runs", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.skip_runs:
        rows = json.loads((output_root / "all_runs.json").read_text(encoding="utf-8"))
    else:
        rows = []
        for dataset in parse_csv(args.datasets):
            for mode in parse_csv(args.modes):
                for seed in parse_seeds(args.seeds):
                    print(f"Running dataset={dataset} aggregation={mode} seed={seed}", flush=True)
                    rows.append(
                        run_one(dataset, mode, seed, output_root / dataset / mode / f"seed_{seed}", args)
                    )
    aggregate_rows = aggregate(rows)
    (output_root / "all_runs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_root / "aggregate.json").write_text(json.dumps(aggregate_rows, indent=2), encoding="utf-8")
    write_csv(output_root / "all_runs.csv", rows)
    write_csv(output_root / "aggregate.csv", aggregate_rows)
    write_markdown(output_root / "aggregate.md", aggregate_rows)
    write_tex(output_root / "aggregation_ablation.tex", aggregate_rows)
    if args.paper_table:
        write_tex(Path(args.paper_table), aggregate_rows)
    print(f"Wrote {output_root / 'aggregate.md'}")


if __name__ == "__main__":
    main()
