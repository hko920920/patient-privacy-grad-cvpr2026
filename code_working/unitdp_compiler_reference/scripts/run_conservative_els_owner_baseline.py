"""Run window DP-SGD calibrated to owner privacy by conservative group conversion."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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

from scripts import run_sepsis_owa as sepsis
from scripts import run_uci_comparison as uci
from scripts import run_wisdm_owa as wisdm
from unitdp.accountant import group_privacy_conversion, required_base_delta_for_group
from unitdp.compiler import compile_contract
from unitdp.contract import load_contract
from unitdp.window_dpsgd import WindowDpsgdConfig, train_window_dpsgd


@dataclass(frozen=True)
class PreparedDataset:
    train_x: np.ndarray
    train_y: np.ndarray
    selected_x: np.ndarray
    selected_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    num_classes: int
    owner_kappa: int
    selected_windows: int
    selected_owners: int
    class_weights: list[float] | None = None


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_seeds(text: str) -> list[int]:
    return [int(item) for item in parse_csv(text)]


def contract_args(args: argparse.Namespace, seed: int, max_windows: int, step_budget: int) -> SimpleNamespace:
    return SimpleNamespace(
        policy=args.policy,
        max_windows_per_owner=max_windows,
        step_window_budget=step_budget,
        seed=seed,
        target_epsilon=args.owner_epsilon,
        target_delta=args.owner_delta,
        epochs=args.epochs,
        owner_batch_size=args.owner_batch_size,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        class_weight=args.sepsis_class_weight,
    )


def prepare_uci(output_dir: Path, seed: int, args: argparse.Namespace) -> PreparedDataset:
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
    local_args = contract_args(args, seed, args.max_windows_per_owner, args.step_window_budget)
    contract_path = output_dir / "owner_contract.yaml"
    uci.write_contract(
        contract_path,
        mapping_path.resolve(),
        local_args,
        preprocessor.contract_binding(),
    )
    route = compile_contract(load_contract(contract_path), base_dir=ROOT)
    selected_indices = [record.row_index for record in route.selected_mapping.records]
    num_classes = int(max(train_y.max(), test_y.max()) + 1)
    return PreparedDataset(
        train_x=train_x,
        train_y=train_y,
        selected_x=train_x[selected_indices],
        selected_y=train_y[selected_indices],
        test_x=test_x,
        test_y=test_y,
        num_classes=num_classes,
        owner_kappa=int(route.selected_mapping.stats()["owner_kappa"]),
        selected_windows=len(selected_indices),
        selected_owners=int(route.selected_mapping.stats()["num_owners"]),
    )


def prepare_wisdm(output_dir: Path, seed: int, args: argparse.Namespace) -> PreparedDataset:
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
    local_args = contract_args(args, seed, args.max_windows_per_owner, args.step_window_budget)
    contract_path = output_dir / "owner_contract.yaml"
    wisdm.write_contract(
        contract_path,
        mapping_path.resolve(),
        local_args,
        args.step_window_budget,
        preprocessor.contract_binding(),
    )
    route = compile_contract(load_contract(contract_path), base_dir=ROOT)
    selected_indices = [record.row_index for record in route.selected_mapping.records]
    num_classes = int(max(train_y.max(), test_y.max()) + 1)
    return PreparedDataset(
        train_x=train_x,
        train_y=train_y,
        selected_x=train_x[selected_indices],
        selected_y=train_y[selected_indices],
        test_x=test_x,
        test_y=test_y,
        num_classes=num_classes,
        owner_kappa=int(route.selected_mapping.stats()["owner_kappa"]),
        selected_windows=len(selected_indices),
        selected_owners=int(route.selected_mapping.stats()["num_owners"]),
    )


def prepare_sepsis(output_dir: Path, seed: int, args: argparse.Namespace) -> PreparedDataset:
    protocol_args = SimpleNamespace(
        max_owners=args.sepsis_max_owners,
        train_frac=args.sepsis_train_frac,
        window_length=args.sepsis_window_length,
        stride=args.sepsis_stride,
        label_mode=args.sepsis_label_mode,
        feature_mode=getattr(args, "sepsis_feature_mode", "basic"),
        policy=args.policy,
        max_windows_per_owner=args.sepsis_max_windows_per_owner,
        step_window_budget=args.sepsis_step_window_budget,
    )
    sepsis.validate_registered_protocol_args(protocol_args)
    protocol = sepsis.load_registered_public_protocol(
        args.sepsis_cache_dir,
        getattr(
            args,
            "sepsis_preprocessing_artifact",
            sepsis.DEFAULT_SEPSIS_PREPROCESSING_ARTIFACT,
        ),
        getattr(
            args,
            "sepsis_preprocessing_artifact_sha256",
            sepsis.DEFAULT_SEPSIS_PREPROCESSING_SHA256,
        ),
    )
    train_files = list(protocol.train_files)
    test_files = list(protocol.test_files)
    train_records, train_x_raw, train_y = sepsis.generate_windows(
        train_files,
        "train",
        args.sepsis_window_length,
        args.sepsis_stride,
        args.sepsis_label_mode,
        getattr(args, "sepsis_feature_mode", "basic"),
    )
    _test_records, test_x_raw, test_y = sepsis.generate_windows(
        test_files,
        "test",
        args.sepsis_window_length,
        args.sepsis_stride,
        args.sepsis_label_mode,
        getattr(args, "sepsis_feature_mode", "basic"),
    )
    if train_x_raw.shape != (1601, 240) or test_x_raw.shape != (398, 240):
        raise ValueError(
            "Sepsis generated feature tables do not match the registered protocol"
        )
    train_x = protocol.preprocessor.transform(train_x_raw)
    test_x = protocol.preprocessor.transform(test_x_raw)

    mapping_path = output_dir / "sepsis_train_mapping.csv"
    sepsis.write_mapping(train_records, mapping_path)
    local_args = contract_args(
        args,
        seed,
        args.sepsis_max_windows_per_owner,
        args.sepsis_step_window_budget,
    )
    contract_path = output_dir / "owner_contract.yaml"
    class_weights = (
        list(protocol.fixed_class_weights)
        if args.sepsis_class_weight == "balanced"
        else None
    )
    sepsis.write_contract(
        contract_path,
        mapping_path.resolve(),
        local_args,
        args.sepsis_step_window_budget,
        protocol,
        class_weights,
    )
    route = compile_contract(load_contract(contract_path), base_dir=ROOT)
    selected_indices = [record.row_index for record in route.selected_mapping.records]
    selected_y = train_y[selected_indices]
    return PreparedDataset(
        train_x=train_x,
        train_y=train_y,
        selected_x=train_x[selected_indices],
        selected_y=selected_y,
        test_x=test_x,
        test_y=test_y,
        num_classes=2,
        owner_kappa=int(route.selected_mapping.stats()["owner_kappa"]),
        selected_windows=len(selected_indices),
        selected_owners=int(route.selected_mapping.stats()["num_owners"]),
        class_weights=class_weights,
    )


def prepare_dataset(dataset: str, output_dir: Path, seed: int, args: argparse.Namespace) -> PreparedDataset:
    if dataset == "uci":
        return prepare_uci(output_dir, seed, args)
    if dataset == "wisdm":
        return prepare_wisdm(output_dir, seed, args)
    if dataset == "sepsis":
        return prepare_sepsis(output_dir, seed, args)
    raise ValueError(f"Unsupported dataset: {dataset}")


def run_one(dataset: str, seed: int, output_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_dataset(dataset, output_dir, seed, args)
    target_window_epsilon = args.owner_epsilon / prepared.owner_kappa
    target_window_delta = required_base_delta_for_group(
        args.owner_delta,
        target_window_epsilon,
        prepared.owner_kappa,
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
            batch_size=args.window_batch_size,
            learning_rate=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            target_epsilon=target_window_epsilon,
            target_delta=target_window_delta,
            seed=seed,
            class_weights=prepared.class_weights,
        ),
    )
    converted = group_privacy_conversion(result.epsilon, result.delta, prepared.owner_kappa)
    row = {
        "dataset": dataset,
        "seed": seed,
        "route": "conservative_els_owner",
        "training_unit": "window",
        "accounted_unit": "window",
        "claimed_unit": "owner",
        "selected_windows": result.selected_windows,
        "selected_owners": prepared.selected_owners,
        "owner_kappa": prepared.owner_kappa,
        "target_owner_epsilon": args.owner_epsilon,
        "target_owner_delta": args.owner_delta,
        "target_window_epsilon": target_window_epsilon,
        "target_window_delta": target_window_delta,
        "observed_window_epsilon": result.epsilon,
        "observed_window_delta": result.delta,
        "converted_owner_epsilon": converted["epsilon"],
        "converted_owner_delta": converted["delta"],
        "converted_owner_vacuous": converted["vacuous"],
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
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["dataset"]), []).append(row)

    out: list[dict[str, object]] = []
    for dataset, items in sorted(grouped.items()):
        payload: dict[str, object] = {
            "dataset": dataset,
            "route": "conservative_els_owner",
            "runs": len(items),
            "owner_kappa": items[0]["owner_kappa"],
            "selected_windows": items[0]["selected_windows"],
            "selected_owners": items[0]["selected_owners"],
            "target_owner_epsilon": items[0]["target_owner_epsilon"],
            "target_owner_delta": items[0]["target_owner_delta"],
            "target_window_epsilon": items[0]["target_window_epsilon"],
            "target_window_delta": items[0]["target_window_delta"],
        }
        for metric in [
            "observed_window_epsilon",
            "converted_owner_epsilon",
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
        deltas = [item["converted_owner_delta"] for item in items]
        payload["converted_owner_delta"] = deltas[0] if len(set(map(str, deltas))) == 1 else str(deltas)
        payload["converted_owner_vacuous"] = any(bool(item["converted_owner_vacuous"]) for item in items)
        out.append(payload)
    return out


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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Conservative ELS-Owner Baseline",
        "",
        "This baseline trains ordinary window DP-SGD, but calibrates the base window epsilon and delta so conservative group conversion can meet the requested owner target.",
        "",
        "| dataset | runs | kappa | target window eps | target window delta | owner eps | noise | macro F1 | AUROC | AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {runs} | {kappa} | {eps_w} | {delta_w:.2e} | {eps_o} | {noise} | {f1} | {auroc} | {auprc} |".format(
                dataset=row["dataset"],
                runs=row["runs"],
                kappa=row["owner_kappa"],
                eps_w=fmt(row["target_window_epsilon"]),
                delta_w=float(row["target_window_delta"]),
                eps_o=f"{fmt(row['converted_owner_epsilon_mean'])} +/- {fmt(row['converted_owner_epsilon_std'])}",
                noise=f"{fmt(row['noise_multiplier_mean'])} +/- {fmt(row['noise_multiplier_std'])}",
                f1=f"{fmt(row['test_macro_f1_mean'])} +/- {fmt(row['test_macro_f1_std'])}",
                auroc=f"{fmt(row['test_auroc_mean'])} +/- {fmt(row['test_auroc_std'])}",
                auprc=f"{fmt(row['test_auprc_mean'])} +/- {fmt(row['test_auprc_std'])}",
            )
        )
    lines.extend(
        [
            "",
            "This is not a tight ELS/MoG owner accountant. It is the conservative fallback made into a valid owner-level training baseline.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataset_label(dataset: str) -> str:
    labels = {"uci": "UCI", "wisdm": "WISDM", "sepsis": "Sepsis"}
    return labels.get(dataset, dataset)


def dataset_order(dataset: str) -> int:
    return {"uci": 0, "wisdm": 1, "sepsis": 2}.get(dataset, 99)


def pm(mean: object, std: object) -> str:
    return f"${float(mean):.4f}\\pm {float(std):.4f}$"


def load_owa_rows(path: str | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(row["dataset"]): row for row in rows if row.get("route") == "owa_dpsgd"}


def write_tex(path: Path, conservative_rows: list[dict[str, object]], owa_rows: dict[str, dict[str, object]]) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Dataset & Route & noise & Macro F1 & AUROC & AUPRC \\\\",
        "\\midrule",
    ]
    for row in sorted(conservative_rows, key=lambda item: dataset_order(str(item["dataset"]))):
        dataset = str(row["dataset"])
        lines.append(
            "{dataset} & Conservative ELS-owner & {noise} & {f1} & {auroc} & {auprc} \\\\".format(
                dataset=dataset_label(dataset),
                noise=pm(row["noise_multiplier_mean"], row["noise_multiplier_std"]),
                f1=pm(row["test_macro_f1_mean"], row["test_macro_f1_std"]),
                auroc=pm(row["test_auroc_mean"], row["test_auroc_std"]),
                auprc=pm(row["test_auprc_mean"], row["test_auprc_std"]),
            )
        )
        if dataset in owa_rows:
            owa = owa_rows[dataset]
            lines.append(
                "{dataset} & OWA-DPSGD & {noise} & {f1} & {auroc} & {auprc} \\\\".format(
                    dataset=dataset_label(dataset),
                    noise=pm(owa["noise_multiplier_mean"], owa["noise_multiplier_std"]),
                    f1=pm(owa["test_macro_f1_mean"], owa["test_macro_f1_std"]),
                    auroc=pm(owa["test_auroc_mean"], owa["test_auroc_std"]),
                    auprc=pm(owa["test_auprc_mean"], owa["test_auprc_std"]),
                )
            )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Five-seed owner-level comparison between a conservative ELS-style fallback and the native owner route. Conservative ELS-owner trains ordinary window DP-SGD with base window $\\varepsilon_w=\\varepsilon_o/\\kappa_{\\mathrm{owner}}$ and a reduced base $\\delta_w$ so that standard group privacy yields owner $\\varepsilon_o\\le 8$ and $\\delta_o\\le 10^{-5}$. It is valid but not a tight ELS/MoG accountant.}",
            "\\label{tab:conservative-els-owner}",
            "\\end{table*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="uci,wisdm,sepsis")
    parser.add_argument("--seeds", default="13,23,31")
    parser.add_argument("--output-dir", default="reports/conservative_els_owner_001")
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
    parser.add_argument(
        "--sepsis-cache-dir",
        default=str(DATA_ROOT / "sepsis2019_cache"),
    )
    parser.add_argument(
        "--sepsis-preprocessing-artifact",
        default=str(sepsis.DEFAULT_SEPSIS_PREPROCESSING_ARTIFACT),
    )
    parser.add_argument(
        "--sepsis-preprocessing-artifact-sha256",
        default=sepsis.DEFAULT_SEPSIS_PREPROCESSING_SHA256,
    )
    parser.add_argument("--sepsis-max-owners", type=int, default=400)
    parser.add_argument("--sepsis-train-frac", type=float, default=0.8)
    parser.add_argument("--sepsis-window-length", type=int, default=12)
    parser.add_argument("--sepsis-stride", type=int, default=6)
    parser.add_argument("--sepsis-label-mode", choices=["any", "end"], default="any")
    parser.add_argument("--sepsis-max-windows-per-owner", type=int, default=8)
    parser.add_argument("--sepsis-step-window-budget", type=int, default=8)
    parser.add_argument("--sepsis-class-weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--owa-aggregate-json", default=None)
    parser.add_argument("--paper-table", default=None)
    parser.add_argument("--skip-runs", action="store_true")
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
                print(f"Running {dataset} seed {seed}", flush=True)
                rows.append(run_one(dataset, seed, run_dir, args))

    aggregate_rows = aggregate(rows)
    (output_root / "all_runs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_root / "aggregate.json").write_text(json.dumps(aggregate_rows, indent=2), encoding="utf-8")
    write_csv(output_root / "all_runs.csv", rows)
    write_csv(output_root / "aggregate.csv", aggregate_rows)
    write_markdown(output_root / "aggregate.md", aggregate_rows)
    if args.paper_table is not None:
        write_tex(Path(args.paper_table), aggregate_rows, load_owa_rows(args.owa_aggregate_json))
    print(f"Wrote {output_root / 'aggregate.md'}")


if __name__ == "__main__":
    main()
