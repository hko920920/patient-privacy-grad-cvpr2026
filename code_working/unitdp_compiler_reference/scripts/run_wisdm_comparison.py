"""Compare window-level DP-SGD with OWA-DPSGD on WISDM."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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

from scripts import run_wisdm_owa as wisdm
from unitdp.accountant import group_privacy_conversion
from unitdp.certificate import build_certificate, write_certificate
from unitdp.compiler import compile_contract
from unitdp.contract import load_contract
from unitdp.models import build_classifier
from unitdp.owa_dpsgd import OwaDpsgdConfig, train_owa_dpsgd
from unitdp.window_dpsgd import WindowDpsgdConfig, train_window_dpsgd


def fmt_metric(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# WISDM Window vs Owner-Aligned DP Comparison",
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
        "--data-path",
        default=str(
            DATA_ROOT
            / "wisdm"
            / "WISDM_ar_v1.1"
            / "WISDM_ar_v1.1_raw.txt"
        ),
    )
    parser.add_argument(
        "--preprocessing-artifact",
        default=str(wisdm.DEFAULT_WISDM_PREPROCESSING_ARTIFACT),
    )
    parser.add_argument(
        "--preprocessing-artifact-sha256",
        default=wisdm.DEFAULT_WISDM_PREPROCESSING_SHA256,
    )
    parser.add_argument("--output-dir", default="reports/wisdm_comparison_001")
    parser.add_argument("--train-user-frac", type=float, default=0.7)
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
    parser.add_argument("--model-type", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_user = wisdm.parse_raw_wisdm(Path(args.data_path))
    train_owners, test_owners = wisdm.fixed_owner_split(
        rows_by_user,
        args.train_user_frac,
    )
    train_records, train_x_raw, train_label_raw = wisdm.generate_windows(
        rows_by_user, train_owners, "train", "wisdm_train_overlap"
    )
    test_records, test_x_raw, test_label_raw = wisdm.generate_windows(
        rows_by_user, test_owners, "test", "wisdm_test_overlap"
    )

    train_y = wisdm.encode_activity_labels(train_label_raw)
    test_y = wisdm.encode_activity_labels(test_label_raw)
    preprocessor = wisdm.load_registered_preprocessor(
        args.preprocessing_artifact,
        args.preprocessing_artifact_sha256,
    )
    preprocessor.verify_reference_file(Path(args.data_path))
    train_x = preprocessor.transform(train_x_raw)
    test_x = preprocessor.transform(test_x_raw)

    mapping_path = output_dir / "wisdm_train_mapping.csv"
    wisdm.write_mapping(train_records, mapping_path)
    wisdm.write_mapping(test_records, output_dir / "wisdm_test_mapping.csv")
    contract_path = output_dir / "owner_contract.yaml"
    wisdm.write_contract(
        contract_path,
        mapping_path.resolve(),
        args,
        args.step_window_budget,
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
    window_group = group_privacy_conversion(window_result.epsilon, window_result.delta, owner_kappa)
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
