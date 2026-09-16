"""Run Owner-Window Aligned DP-SGD on UCI HAR feature windows."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get("UNITDP_DATA_ROOT", str(ROOT / "data"))
)
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unitdp.certificate import build_certificate, write_certificate
from unitdp.compiler import compile_contract
from unitdp.contract import load_contract
from unitdp.owa_dpsgd import OwaDpsgdConfig, train_owa_dpsgd
from unitdp.preprocessing import load_fixed_affine_preprocessor


WINDOW_LENGTH = 128
WINDOW_STEP = 64
DEFAULT_UCI_PREPROCESSING_ARTIFACT = (
    ROOT
    / "configs"
    / "preprocessing"
    / "uci_har_published_train_standard_scaler_v1.json"
)
DEFAULT_UCI_PREPROCESSING_SHA256 = (
    "774713e3bef48c784113f4097642a59741481e7495f858ed41cdb1788ae8fadf"
)


def load_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_dir = root / split
    x = np.loadtxt(split_dir / f"X_{split}.txt").astype(np.float32)
    y = np.loadtxt(split_dir / f"y_{split}.txt", dtype=int).astype(np.int64) - 1
    subjects = np.loadtxt(split_dir / f"subject_{split}.txt", dtype=int)
    return x, y, subjects


def build_records(subjects: np.ndarray, labels: np.ndarray, split: str) -> list[dict[str, object]]:
    per_subject_counts: dict[int, int] = defaultdict(int)
    records: list[dict[str, object]] = []
    for row_idx, subject_raw in enumerate(subjects):
        subject = int(subject_raw)
        local_idx = per_subject_counts[subject]
        per_subject_counts[subject] += 1
        start = local_idx * WINDOW_STEP
        records.append(
            {
                "scenario": f"uci_{split}",
                "window_id": f"{split}_row_{row_idx}",
                "owner_id": str(subject),
                "owner_ids": str(subject),
                "start": start,
                "end": start + WINDOW_LENGTH,
                "row_index": row_idx,
                "label": str(int(labels[row_idx])),
            }
        )
    return records


def write_mapping(records: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "window_id",
                "owner_id",
                "owner_ids",
                "start",
                "end",
                "row_index",
                "label",
            ],
        )
        writer.writeheader()
        writer.writerows(records)


def parse_budget_list(text: str) -> list[int | None]:
    out: list[int | None] = []
    for item in text.split(","):
        token = item.strip().lower()
        if not token:
            continue
        out.append(None if token in {"all", "none"} else int(token))
    return out


def write_contract(
    path: Path,
    mapping_path: Path,
    args: argparse.Namespace,
    step_budget: int | None,
    preprocessing_binding: dict[str, object],
) -> None:
    owner_policy = {
        "type": args.policy,
        "max_windows_per_owner": args.max_windows_per_owner,
        "seed": args.seed,
    }
    if step_budget is not None:
        owner_policy["windows_per_owner_per_step"] = step_budget
    contract = {
        "privacy_unit": "owner",
        "mapping": str(mapping_path),
        "schedule_mode": "fixed",
        "adjacency": "replace_all_owner_contributions",
        "owner_policy": owner_policy,
        "mechanism": {
            "sampling_unit": "owner",
            "clipping_unit": "owner",
            "noising_unit": "owner",
        },
        "accountant": {
            "backend": "rdp",
            "unit": "owner",
            "target_epsilon": args.target_epsilon,
            "delta": args.target_delta,
        },
        "training": {
            "epochs": args.epochs,
            "owner_batch_size": args.owner_batch_size,
            "learning_rate": args.learning_rate,
            "max_grad_norm": args.max_grad_norm,
        },
        "data": {
            "dataset": "uci_har",
            "input_representation": "published_561_feature_vectors",
            "preprocessing": preprocessing_binding,
        },
    }
    path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")


def fmt_metric(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# UCI HAR OWA-DPSGD Run",
        "",
        "| step window budget | selected windows | owners | epsilon | noise | test acc | macro F1 | AUROC | AUPRC | owner kappa | label TV | min label owner coverage | group fallback delta |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {step_window_budget} | {selected_windows} | {selected_owners} | {epsilon:.4f} | "
            "{noise_multiplier:.4f} | {test_accuracy:.4f} | {test_macro_f1:.4f} | "
            "{test_auroc} | {test_auprc} | "
            "{owner_kappa} | {label_tv} | {min_label_owner_coverage} | {fallback_delta} |".format(
                step_window_budget=row["step_window_budget"],
                selected_windows=row["selected_windows"],
                selected_owners=row["selected_owners"],
                epsilon=row["epsilon"],
                noise_multiplier=row["noise_multiplier"],
                test_accuracy=row["test_accuracy"],
                test_macro_f1=row["test_macro_f1"],
                test_auroc=fmt_metric(row.get("test_auroc")),
                test_auprc=fmt_metric(row.get("test_auprc")),
                owner_kappa=row["selected_mapping_stats"]["owner_kappa"],
                label_tv=fmt_metric(row["selection_diagnostics"]["label_total_variation"]),
                min_label_owner_coverage=fmt_metric(
                    row["selection_diagnostics"]["min_label_owner_coverage"]
                ),
                fallback_delta=row["group_fallback"]["delta"],
            )
        )
    lines.extend(
        [
            "",
            "This is a first real-data owner-aligned smoke run. The UCI HAR feature rows are already generated windows; subjects are treated as owners.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default=str(
            DATA_ROOT
            / "uci_har"
            / "extracted"
            / "UCI HAR Dataset"
        ),
    )
    parser.add_argument(
        "--preprocessing-artifact",
        default=str(DEFAULT_UCI_PREPROCESSING_ARTIFACT),
    )
    parser.add_argument(
        "--preprocessing-artifact-sha256",
        default=DEFAULT_UCI_PREPROCESSING_SHA256,
    )
    parser.add_argument("--output-dir", default="reports/uci_owa_demo")
    parser.add_argument("--policy", default="support_balanced_cap")
    parser.add_argument("--max-windows-per-owner", type=int, default=32)
    parser.add_argument("--step-window-budgets", default="4,8,all")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--owner-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-epsilon", type=float, default=8.0)
    parser.add_argument("--target-delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_x_raw, train_y, train_subjects = load_split(data_root, "train")
    test_x_raw, test_y, _test_subjects = load_split(data_root, "test")
    preprocessor = load_fixed_affine_preprocessor(
        args.preprocessing_artifact,
        expected_payload_sha256=args.preprocessing_artifact_sha256,
    )
    preprocessor.verify_reference_file(data_root / "train" / "X_train.txt")
    train_x = preprocessor.transform(train_x_raw)
    test_x = preprocessor.transform(test_x_raw)

    mapping_path = output_dir / "uci_train_mapping.csv"
    write_mapping(build_records(train_subjects, train_y, "train"), mapping_path)

    rows: list[dict[str, object]] = []
    for step_budget in parse_budget_list(args.step_window_budgets):
        budget_name = "all" if step_budget is None else str(step_budget)
        run_dir = output_dir / f"budget_{budget_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        contract_path = run_dir / "contract.yaml"
        write_contract(
            contract_path,
            mapping_path.resolve(),
            args,
            step_budget,
            preprocessor.contract_binding(),
        )
        route = compile_contract(load_contract(contract_path), base_dir=ROOT)

        torch.manual_seed(args.seed)
        model = nn.Linear(train_x.shape[1], int(max(train_y.max(), test_y.max()) + 1))
        result = train_owa_dpsgd(
            model=model,
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
            route=route,
            config=OwaDpsgdConfig(
                epochs=args.epochs,
                owner_batch_size=args.owner_batch_size,
                learning_rate=args.learning_rate,
                max_grad_norm=args.max_grad_norm,
                target_epsilon=args.target_epsilon,
                target_delta=args.target_delta,
                seed=args.seed,
                device="cpu",
            ),
        )
        certificate = build_certificate(route, result.to_dict())
        write_certificate(certificate, run_dir / "certificate.json")
        write_certificate(certificate, run_dir / "certificate.md")

        payload = result.to_dict()
        payload["step_window_budget"] = budget_name
        payload["policy"] = args.policy
        payload["max_windows_per_owner"] = args.max_windows_per_owner
        payload["selected_mapping_stats"] = certificate.selected_mapping_stats
        payload["selection_diagnostics"] = certificate.selection_diagnostics
        payload["group_fallback"] = certificate.group_fallback
        (run_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        rows.append(payload)

    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_summary(output_dir / "summary.md", rows)
    print(f"Wrote {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
