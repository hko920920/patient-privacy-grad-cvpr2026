"""Auxiliary owner-level membership-inference probe.

This is not a privacy proof. It scores train owners and held-out owners by mean
negative cross-entropy loss, then reports owner-level membership AUC.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get("UNITDP_DATA_ROOT", str(ROOT / "data"))
)
SRC = ROOT / "src"
for path in [ROOT, SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import run_uci_comparison as uci  # noqa: E402
from scripts.run_aggregation_ablation import contract_args  # noqa: E402
from unitdp.compiler import compile_contract  # noqa: E402
from unitdp.contract import load_contract  # noqa: E402
from unitdp.models import build_classifier  # noqa: E402
from unitdp.nonprivate import NonPrivateConfig, train_nonprivate  # noqa: E402
from unitdp.owa_dpsgd import OwaDpsgdConfig, train_owa_dpsgd  # noqa: E402


def parse_seeds(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def prepare_uci(output_dir: Path, seed: int, args: argparse.Namespace) -> dict[str, object]:
    train_x_raw, train_y, train_subjects = uci.load_split(Path(args.uci_data_root), "train")
    test_x_raw, test_y, test_subjects = uci.load_split(Path(args.uci_data_root), "test")
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
    local_args = SimpleNamespace(
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
    contract_path = output_dir / "owner_contract.yaml"
    uci.write_contract(
        contract_path,
        mapping_path.resolve(),
        contract_args(local_args, seed),
        preprocessor.contract_binding(),
    )
    route = compile_contract(load_contract(contract_path), base_dir=ROOT)
    return {
        "train_x": train_x,
        "train_y": train_y.astype(np.int64),
        "train_subjects": train_subjects,
        "test_x": test_x,
        "test_y": test_y.astype(np.int64),
        "test_subjects": test_subjects,
        "route": route,
        "num_classes": int(max(train_y.max(), test_y.max()) + 1),
    }


def owner_loss_scores(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_subjects: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    test_subjects: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    criterion = nn.CrossEntropyLoss(reduction="none")
    labels: list[int] = []
    scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for member, x, y, subjects in [
            (1, train_x, train_y, train_subjects),
            (0, test_x, test_y, test_subjects),
        ]:
            xt = torch.tensor(x, dtype=torch.float32)
            yt = torch.tensor(y, dtype=torch.long)
            losses = criterion(model(xt), yt).detach().cpu().numpy()
            for owner in sorted(set(subjects.tolist())):
                mask = subjects == owner
                labels.append(member)
                scores.append(float(-losses[mask].mean()))
    return np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)


def run_one(route_name: str, seed: int, output_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_uci(output_dir, seed, args)
    torch.manual_seed(seed)
    model = build_classifier(int(data["train_x"].shape[1]), int(data["num_classes"]), "linear")
    if route_name == "nonprivate":
        result = train_nonprivate(
            model=model,
            train_x=data["train_x"],
            train_y=data["train_y"],
            test_x=data["test_x"],
            test_y=data["test_y"],
            config=NonPrivateConfig(
                epochs=args.epochs,
                batch_size=args.window_batch_size,
                learning_rate=args.learning_rate,
                seed=seed,
            ),
        ).to_dict()
        epsilon = None
        noise = 0.0
    elif route_name == "owa":
        result_obj = train_owa_dpsgd(
            model=model,
            train_x=data["train_x"],
            train_y=data["train_y"],
            test_x=data["test_x"],
            test_y=data["test_y"],
            route=data["route"],
            config=OwaDpsgdConfig(
                epochs=args.epochs,
                owner_batch_size=args.owner_batch_size,
                learning_rate=args.learning_rate,
                max_grad_norm=args.max_grad_norm,
                target_epsilon=args.target_epsilon,
                target_delta=args.target_delta,
                seed=seed,
            ),
        )
        result = result_obj.to_dict()
        epsilon = result_obj.epsilon
        noise = result_obj.noise_multiplier
    else:
        raise ValueError(route_name)

    y_true, scores = owner_loss_scores(
        model,
        data["train_x"],
        data["train_y"],
        data["train_subjects"],
        data["test_x"],
        data["test_y"],
        data["test_subjects"],
    )
    auc = float(roc_auc_score(y_true, scores))
    row = {
        "dataset": "uci",
        "route": route_name,
        "seed": seed,
        "owner_membership_auc": auc,
        "member_owners": int(y_true.sum()),
        "nonmember_owners": int((1 - y_true).sum()),
        "epsilon": epsilon,
        "noise_multiplier": noise,
        "test_macro_f1": result["test_macro_f1"],
        "test_auroc": result.get("test_auroc"),
        "test_auprc": result.get("test_auprc"),
    }
    (output_dir / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row["route"]), []).append(row)
    out: list[dict[str, object]] = []
    for route_name, items in sorted(groups.items()):
        payload: dict[str, object] = {
            "dataset": "uci",
            "route": route_name,
            "runs": len(items),
            "member_owners": items[0]["member_owners"],
            "nonmember_owners": items[0]["nonmember_owners"],
        }
        for metric in [
            "owner_membership_auc",
            "epsilon",
            "noise_multiplier",
            "test_macro_f1",
            "test_auroc",
            "test_auprc",
        ]:
            values = [float(item[metric]) for item in items if item.get(metric) is not None]
            if values:
                avg, std = mean_std(values)
            else:
                avg, std = float("nan"), float("nan")
            payload[f"{metric}_mean"] = avg
            payload[f"{metric}_std"] = std
        out.append(payload)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
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


def route_label(route_name: str) -> str:
    return {"nonprivate": "Non-private", "owa": "OWA-DPSGD"}.get(route_name, route_name)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Owner Membership Probe",
        "",
        "| route | runs | owners in/out | epsilon | noise | owner MIA AUC | macro F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {route} | {runs} | {members}/{nonmembers} | {eps} | {noise} | {auc} | {f1} |".format(
                route=route_label(str(row["route"])),
                runs=row["runs"],
                members=row["member_owners"],
                nonmembers=row["nonmember_owners"],
                eps=pm(row, "epsilon"),
                noise=pm(row, "noise_multiplier"),
                auc=pm(row, "owner_membership_auc"),
                f1=pm(row, "test_macro_f1"),
            )
        )
    lines.append("")
    lines.append("The attack score is negative mean cross-entropy loss per owner. This is an empirical probe only; the privacy claim is the formal DP certificate.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Route & $\\varepsilon$ & Owner MIA AUC & Macro F1 \\\\",
        "\\midrule",
    ]
    for row in rows:
        eps = "n/a" if str(row["route"]) == "nonprivate" else f"{float(row['epsilon_mean']):.2f}"
        lines.append(
            "{route} & {eps} & {auc} & {f1} \\\\".format(
                route=route_label(str(row["route"])),
                eps=eps,
                auc=tex_pm(row, "owner_membership_auc"),
                f1=tex_pm(row, "test_macro_f1"),
            )
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\caption{Auxiliary owner-level membership probe on UCI HAR. The attack scores each owner by negative mean cross-entropy loss and distinguishes training owners from held-out owners. This probe is not a privacy proof; it checks whether the formal owner-level route also reduces a simple owner-level loss attack relative to non-private training.}",
            "\\label{tab:owner-membership-probe}",
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", default="nonprivate,owa")
    parser.add_argument("--seeds", default="13,23,31,37,41")
    parser.add_argument("--output-dir", default="reports/owner_membership_probe_uci_5seed_001")
    parser.add_argument("--policy", default="support_balanced_cap")
    parser.add_argument("--max-windows-per-owner", type=int, default=16)
    parser.add_argument("--step-window-budget", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--owner-batch-size", type=int, default=8)
    parser.add_argument("--window-batch-size", type=int, default=64)
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
    parser.add_argument("--paper-table", default="")
    parser.add_argument("--skip-runs", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    if args.skip_runs:
        rows = json.loads((output_root / "all_runs.json").read_text(encoding="utf-8"))
    else:
        rows = []
        for route_name in [item.strip() for item in args.routes.split(",") if item.strip()]:
            for seed in parse_seeds(args.seeds):
                print(f"Running owner MIA route={route_name} seed={seed}", flush=True)
                rows.append(run_one(route_name, seed, output_root / route_name / f"seed_{seed}", args))
    aggregate_rows = aggregate(rows)
    (output_root / "all_runs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_root / "aggregate.json").write_text(json.dumps(aggregate_rows, indent=2), encoding="utf-8")
    write_csv(output_root / "all_runs.csv", rows)
    write_csv(output_root / "aggregate.csv", aggregate_rows)
    write_markdown(output_root / "aggregate.md", aggregate_rows)
    write_tex(output_root / "owner_membership_probe.tex", aggregate_rows)
    if args.paper_table:
        table_path = Path(args.paper_table)
        if not table_path.is_absolute():
            table_path = ROOT / table_path
        write_tex(table_path, aggregate_rows)
    print(f"Wrote {output_root / 'aggregate.md'}")


if __name__ == "__main__":
    main()
