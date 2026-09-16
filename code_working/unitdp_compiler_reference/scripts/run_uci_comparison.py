"""Compare window-level DP-SGD with OWA-DPSGD on UCI HAR."""

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

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get("UNITDP_DATA_ROOT", str(ROOT / "data"))
)
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unitdp.accountant import group_privacy_conversion
from unitdp.certificate import build_certificate, write_certificate
from unitdp.compiler import compile_contract
from unitdp.contract import load_contract
from unitdp.models import build_classifier
from unitdp.owa_dpsgd import OwaDpsgdConfig, train_owa_dpsgd
from unitdp.preprocessing import (
    FixedAffinePreprocessor,
    load_fixed_affine_preprocessor,
)
from unitdp.window_dpsgd import WindowDpsgdConfig, train_window_dpsgd


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


def load_registered_preprocessor(
    artifact_path: str | Path = DEFAULT_UCI_PREPROCESSING_ARTIFACT,
    expected_sha256: str = DEFAULT_UCI_PREPROCESSING_SHA256,
) -> FixedAffinePreprocessor:
    """Load the approved UCI public-fixed preprocessing artifact."""

    return load_fixed_affine_preprocessor(
        artifact_path,
        expected_payload_sha256=expected_sha256,
    )


def write_contract(
    path: Path,
    mapping_path: Path,
    args: argparse.Namespace,
    preprocessing_binding: dict[str, object] | None = None,
) -> None:
    contract = {
        "privacy_unit": "owner",
        "mapping": str(mapping_path),
        "schedule_mode": "fixed",
        "adjacency": "replace_all_owner_contributions",
        "owner_policy": {
            "type": args.policy,
            "max_windows_per_owner": args.max_windows_per_owner,
            "windows_per_owner_per_step": args.step_window_budget,
            "seed": args.seed,
        },
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
    }
    if preprocessing_binding is not None:
        contract["data"] = {
            "dataset": "uci_har",
            "input_representation": "published_561_feature_vectors",
            "preprocessing": preprocessing_binding,
        }
    path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")


def fmt_metric(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# UCI HAR Window vs Owner-Aligned DP Comparison",
        "",
        "| route | accounted unit | selected windows | epsilon | noise | test acc | macro F1 | AUROC | AUPRC | owner kappa | owner group delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {route} | {accounted_unit} | {selected_windows} | {epsilon:.4f} | {noise_multiplier:.4f} | "
            "{test_accuracy:.4f} | {test_macro_f1:.4f} | {test_auroc} | {test_auprc} | "
            "{owner_kappa} | {owner_group_delta} |".format(
                **{
                    **row,
                    "test_auroc": fmt_metric(row.get("test_auroc")),
                    "test_auprc": fmt_metric(row.get("test_auprc")),
                }
            )
        )
    lines.extend(
        [
            "",
            "The window route reports window-level DP. The owner group delta column shows the conservative owner-level group fallback for that window epsilon.",
            "The OWA route reports native owner-level DP because sampling, clipping, noising, and accounting are owner-aligned.",
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
    parser.add_argument("--output-dir", default="reports/uci_comparison_001")
    parser.add_argument("--policy", default="support_balanced_cap")
    parser.add_argument("--max-windows-per-owner", type=int, default=16)
    parser.add_argument("--step-window-budget", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--owner-batch-size", type=int, default=8)
    parser.add_argument("--window-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-epsilon", type=float, default=8.0)
    parser.add_argument("--target-delta", type=float, default=1e-5)
    parser.add_argument("--model-type", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_x_raw, train_y, train_subjects = load_split(Path(args.data_root), "train")
    test_x_raw, test_y, _ = load_split(Path(args.data_root), "test")
    preprocessor = load_registered_preprocessor(
        args.preprocessing_artifact,
        args.preprocessing_artifact_sha256,
    )
    preprocessor.verify_reference_file(
        Path(args.data_root) / "train" / "X_train.txt"
    )
    train_x = preprocessor.transform(train_x_raw)
    test_x = preprocessor.transform(test_x_raw)

    mapping_path = output_dir / "uci_train_mapping.csv"
    write_mapping(build_records(train_subjects, train_y, "train"), mapping_path)
    contract_path = output_dir / "owner_contract.yaml"
    write_contract(
        contract_path,
        mapping_path.resolve(),
        args,
        preprocessor.contract_binding(),
    )
    route = compile_contract(load_contract(contract_path), base_dir=ROOT)
    selected_indices = [record.row_index for record in route.selected_mapping.records]
    selected_x = train_x[selected_indices]
    selected_y = train_y[selected_indices]
    num_classes = int(max(train_y.max(), test_y.max()) + 1)

    torch.manual_seed(args.seed)
    window_result = train_window_dpsgd(
        model=build_classifier(train_x.shape[1], num_classes, args.model_type, args.hidden_dim),
        train_x=selected_x,
        train_y=selected_y,
        test_x=test_x,
        test_y=test_y,
        config=WindowDpsgdConfig(
            epochs=args.epochs,
            batch_size=args.window_batch_size,
            learning_rate=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            target_epsilon=args.target_epsilon,
            target_delta=args.target_delta,
            seed=args.seed,
        ),
    )

    torch.manual_seed(args.seed)
    owner_result = train_owa_dpsgd(
        model=build_classifier(train_x.shape[1], num_classes, args.model_type, args.hidden_dim),
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
        ),
    )
    certificate = build_certificate(route, owner_result.to_dict())
    write_certificate(certificate, output_dir / "owner_certificate.json")
    write_certificate(certificate, output_dir / "owner_certificate.md")

    owner_kappa = int(route.selected_mapping.stats()["owner_kappa"])
    window_group = group_privacy_conversion(
        window_result.epsilon,
        window_result.delta,
        owner_kappa,
    )
    rows = [
        {
            "route": "window_dpsgd",
            "model_type": args.model_type,
            "hidden_dim": args.hidden_dim,
            "accounted_unit": "window",
            "selected_windows": window_result.selected_windows,
            "epsilon": window_result.epsilon,
            "noise_multiplier": window_result.noise_multiplier,
            "test_accuracy": window_result.test_accuracy,
            "test_macro_f1": window_result.test_macro_f1,
            "test_auroc": window_result.test_auroc,
            "test_auprc": window_result.test_auprc,
            "owner_kappa": owner_kappa,
            "owner_group_delta": window_group["delta"],
        },
        {
            "route": "owa_dpsgd",
            "model_type": args.model_type,
            "hidden_dim": args.hidden_dim,
            "accounted_unit": "owner",
            "selected_windows": owner_result.selected_windows,
            "epsilon": owner_result.epsilon,
            "noise_multiplier": owner_result.noise_multiplier,
            "test_accuracy": owner_result.test_accuracy,
            "test_macro_f1": owner_result.test_macro_f1,
            "test_auroc": owner_result.test_auroc,
            "test_auprc": owner_result.test_auprc,
            "owner_kappa": owner_kappa,
            "owner_group_delta": "not needed",
        },
    ]
    (output_dir / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_summary(output_dir / "comparison.md", rows)
    print(f"Wrote {output_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
