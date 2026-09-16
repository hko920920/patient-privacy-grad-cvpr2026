"""Compare window-level DP-SGD with OWA-DPSGD on PhysioNet Sepsis windows."""

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

from scripts import run_sepsis_owa as sepsis
from unitdp.accountant import group_privacy_conversion
from unitdp.certificate import build_certificate, write_certificate
from unitdp.compiler import compile_contract
from unitdp.contract import load_contract
from unitdp.models import build_classifier
from unitdp.owa_dpsgd import OwaDpsgdConfig, train_owa_dpsgd
from unitdp.window_dpsgd import WindowDpsgdConfig, train_window_dpsgd


def fmt_metric(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def write_summary(
    path: Path,
    rows: list[dict[str, object]],
    train_positive_rate: float,
    test_positive_rate: float,
) -> None:
    lines = [
        "# Sepsis Window vs Owner-Aligned DP Comparison",
        "",
        f"- train positive window rate: `{train_positive_rate:.4f}`",
        f"- test positive window rate: `{test_positive_rate:.4f}`",
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
            "The OWA route reports native patient/owner-level DP because sampling, clipping, noising, and accounting are owner-aligned.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        default=str(DATA_ROOT / "sepsis2019_cache"),
    )
    parser.add_argument(
        "--preprocessing-artifact",
        default=str(sepsis.DEFAULT_SEPSIS_PREPROCESSING_ARTIFACT),
    )
    parser.add_argument(
        "--preprocessing-artifact-sha256",
        default=sepsis.DEFAULT_SEPSIS_PREPROCESSING_SHA256,
    )
    parser.add_argument("--output-dir", default="reports/sepsis_comparison_001")
    parser.add_argument("--max-owners", type=int, default=400)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--window-length", type=int, default=12)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--label-mode", choices=["any", "end"], default="any")
    parser.add_argument("--feature-mode", choices=["basic", "trend"], default="basic")
    parser.add_argument("--policy", default="support_balanced_cap")
    parser.add_argument("--max-windows-per-owner", type=int, default=8)
    parser.add_argument("--step-window-budget", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--owner-batch-size", type=int, default=32)
    parser.add_argument("--window-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-epsilon", type=float, default=8.0)
    parser.add_argument("--target-delta", type=float, default=1e-5)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--model-type", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sepsis.validate_registered_protocol_args(args)
    protocol = sepsis.load_registered_public_protocol(
        args.cache_dir,
        args.preprocessing_artifact,
        args.preprocessing_artifact_sha256,
    )
    train_files = list(protocol.train_files)
    test_files = list(protocol.test_files)
    train_records, train_x_raw, train_y = sepsis.generate_windows(
        train_files, "train", args.window_length, args.stride, args.label_mode, args.feature_mode
    )
    test_records, test_x_raw, test_y = sepsis.generate_windows(
        test_files, "test", args.window_length, args.stride, args.label_mode, args.feature_mode
    )
    if train_x_raw.shape != (1601, 240) or test_x_raw.shape != (398, 240):
        raise ValueError(
            "Sepsis generated feature tables do not match the registered protocol"
        )
    train_x = protocol.preprocessor.transform(train_x_raw)
    test_x = protocol.preprocessor.transform(test_x_raw)

    mapping_path = output_dir / "sepsis_train_mapping.csv"
    sepsis.write_mapping(train_records, mapping_path)
    sepsis.write_mapping(test_records, output_dir / "sepsis_test_mapping.csv")
    contract_path = output_dir / "owner_contract.yaml"
    class_weights = (
        list(protocol.fixed_class_weights)
        if args.class_weight == "balanced"
        else None
    )
    sepsis.write_contract(
        contract_path,
        mapping_path.resolve(),
        args,
        args.step_window_budget,
        protocol,
        class_weights,
    )
    route = compile_contract(load_contract(contract_path), base_dir=ROOT)
    selected_indices = [record.row_index for record in route.selected_mapping.records]
    selected_x = train_x[selected_indices]
    selected_y = train_y[selected_indices]
    selected_class_weights = class_weights

    torch.manual_seed(args.seed)
    window_result = train_window_dpsgd(
        model=build_classifier(train_x.shape[1], 2, args.model_type, args.hidden_dim),
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
            class_weights=selected_class_weights,
        ),
    )

    torch.manual_seed(args.seed)
    owner_result = train_owa_dpsgd(
        model=build_classifier(train_x.shape[1], 2, args.model_type, args.hidden_dim),
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
            class_weights=class_weights,
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
            "feature_mode": args.feature_mode,
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
            "feature_mode": args.feature_mode,
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
    write_summary(
        output_dir / "comparison.md",
        rows,
        train_positive_rate=float(np.mean(train_y)),
        test_positive_rate=float(np.mean(test_y)),
    )
    print(f"Wrote {output_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
