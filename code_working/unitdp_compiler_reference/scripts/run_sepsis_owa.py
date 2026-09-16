"""Run Owner-Window Aligned DP-SGD on PhysioNet/CinC 2019 Sepsis windows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

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
from unitdp.models import build_classifier
from unitdp.owa_dpsgd import OwaDpsgdConfig, train_owa_dpsgd
from unitdp.preprocessing import (
    FixedAffinePreprocessor,
    file_sha256,
    load_fixed_affine_preprocessor,
)


DEFAULT_SEPSIS_PREPROCESSING_ARTIFACT = (
    ROOT
    / "configs"
    / "preprocessing"
    / "physionet2019_setA_first400_basic12x6_public_scaler_v1.json"
)
DEFAULT_SEPSIS_PREPROCESSING_SHA256 = (
    "e2ddd1cbb651e5ef5629bdb3ba0b885e5c9cead7c6b8afd9d5d9490e463b210b"
)
SEPSIS_PROTOCOL_ID = "physionet2019_setA_first400_basic12x6_v1"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SepsisPublicProtocol:
    """Materialized public cohort, split, class weights, and preprocessor."""

    protocol_id: str
    preprocessor: FixedAffinePreprocessor
    train_files: tuple[Path, ...]
    test_files: tuple[Path, ...]
    fixed_class_weights: tuple[float, float]
    patient_manifest_sha256: str

    def contract_binding(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "patient_manifest_sha256": self.patient_manifest_sha256,
            "train_owners": len(self.train_files),
            "test_owners": len(self.test_files),
            "fixed_class_weights": list(self.fixed_class_weights),
            "preprocessing": self.preprocessor.contract_binding(),
        }


def load_registered_public_protocol(
    cache_dir: str | Path,
    artifact_path: str | Path = DEFAULT_SEPSIS_PREPROCESSING_ARTIFACT,
    expected_sha256: str = DEFAULT_SEPSIS_PREPROCESSING_SHA256,
) -> SepsisPublicProtocol:
    """Load and verify every materialized file in the public Sepsis protocol."""

    preprocessor = load_fixed_affine_preprocessor(
        artifact_path,
        expected_payload_sha256=expected_sha256,
    )
    raw = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    protocol = raw.get("dataset_protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Sepsis artifact is missing dataset_protocol")
    if protocol.get("protocol_id") != SEPSIS_PROTOCOL_ID:
        raise ValueError(
            f"Unsupported Sepsis protocol: {protocol.get('protocol_id')!r}"
        )
    expected_fields = {
        "eligible_min_rows": 12,
        "window_length": 12,
        "window_step": 6,
        "label_mode": "any",
        "feature_mode": "basic",
        "feature_scope": "record_local_window",
        "feature_dimension": 240,
        "train_window_count": 1601,
        "test_window_count": 398,
    }
    for key, expected in expected_fields.items():
        if protocol.get(key) != expected:
            raise ValueError(
                f"Sepsis protocol field {key} must be {expected!r}, "
                f"got {protocol.get(key)!r}"
            )
    selection = protocol.get("selection_policy")
    if not isinstance(selection, dict) or selection != {
        "type": "support_balanced_cap",
        "max_windows_per_owner": 8,
        "windows_per_owner_per_step": 8,
        "selected_window_count": 1507,
    }:
        raise ValueError("Sepsis selection policy does not match the registered protocol")

    manifest = protocol.get("patient_manifest")
    if not isinstance(manifest, list) or len(manifest) != 400:
        raise ValueError("Sepsis patient manifest must contain exactly 400 files")
    manifest_sha256 = _canonical_sha256(manifest)
    if manifest_sha256 != protocol.get("patient_manifest_sha256"):
        raise ValueError("Sepsis patient manifest digest mismatch")
    if manifest_sha256 != preprocessor.source_sha256:
        raise ValueError(
            "Sepsis patient manifest does not match preprocessing source digest"
        )

    root = Path(cache_dir).resolve()
    by_owner: dict[str, Path] = {}
    eligible_owners: set[str] = set()
    for item in manifest:
        if not isinstance(item, dict):
            raise ValueError("Sepsis patient manifest entries must be mappings")
        relative = Path(str(item.get("relative_path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe Sepsis manifest path: {relative}")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Sepsis manifest path escapes cache root: {path}") from exc
        expected_file_sha256 = str(item.get("sha256", ""))
        if not path.is_file() or file_sha256(path) != expected_file_sha256:
            raise ValueError(f"Sepsis public patient file mismatch: {relative}")
        owner = str(item.get("owner_id", ""))
        if not owner or owner in by_owner:
            raise ValueError(f"Invalid or duplicate Sepsis owner id: {owner!r}")
        by_owner[owner] = path
        if item.get("eligible") is True:
            eligible_owners.add(owner)

    train_owner_ids = protocol.get("train_owner_ids")
    test_owner_ids = protocol.get("test_owner_ids")
    if not isinstance(train_owner_ids, list) or not isinstance(test_owner_ids, list):
        raise ValueError("Sepsis materialized train/test owner lists are required")
    train_set = set(map(str, train_owner_ids))
    test_set = set(map(str, test_owner_ids))
    if len(train_owner_ids) != 312 or len(test_owner_ids) != 78:
        raise ValueError("Sepsis registered split must contain 312/78 owners")
    if train_set & test_set or train_set | test_set != eligible_owners:
        raise ValueError(
            "Sepsis train/test lists must be a disjoint partition of eligible owners"
        )

    weights = np.asarray(protocol.get("fixed_class_weights"), dtype=np.float64)
    if weights.shape != (2,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Sepsis fixed class weights must be two positive finite values")
    if preprocessor.input_dim != 240:
        raise ValueError("Sepsis preprocessor must have input_dim=240")

    return SepsisPublicProtocol(
        protocol_id=SEPSIS_PROTOCOL_ID,
        preprocessor=preprocessor,
        train_files=tuple(by_owner[str(owner)] for owner in train_owner_ids),
        test_files=tuple(by_owner[str(owner)] for owner in test_owner_ids),
        fixed_class_weights=(float(weights[0]), float(weights[1])),
        patient_manifest_sha256=manifest_sha256,
    )


def validate_registered_protocol_args(args: argparse.Namespace) -> None:
    """Reject runtime choices that would change the registered data protocol."""

    required = {
        "max_owners": 400,
        "train_frac": 0.8,
        "window_length": 12,
        "stride": 6,
        "label_mode": "any",
        "feature_mode": "basic",
        "policy": "support_balanced_cap",
        "max_windows_per_owner": 8,
    }
    for key, expected in required.items():
        observed = getattr(args, key)
        if isinstance(expected, float):
            matches = abs(float(observed) - expected) <= 1e-12
        else:
            matches = observed == expected
        if not matches:
            raise ValueError(
                f"Registered Sepsis protocol requires {key}={expected!r}, "
                f"got {observed!r}"
            )
    if hasattr(args, "step_window_budget") and int(args.step_window_budget) != 8:
        raise ValueError(
            "Registered Sepsis protocol requires step_window_budget=8"
        )


def patient_files(cache_dir: Path, max_owners: int | None) -> list[Path]:
    files: list[Path] = []
    for split in ["training_setA", "training_setB"]:
        split_files = sorted((cache_dir / split).glob("*.psv"))
        files.extend(split_files)
    return files[:max_owners] if max_owners is not None else files


def load_patient(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    frame = pd.read_csv(path, sep="|")
    if "SepsisLabel" not in frame.columns:
        raise ValueError(f"{path} missing SepsisLabel")
    labels = frame["SepsisLabel"].fillna(0).astype(int).to_numpy()
    feature_frame = frame.drop(columns=["SepsisLabel"])
    columns = list(feature_frame.columns)
    values = feature_frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    return values, labels, columns


def _observed_endpoint(values: np.ndarray, first: bool) -> np.ndarray:
    valid = ~np.isnan(values)
    filled = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.zeros(values.shape[1], dtype=np.float32)
    positions = range(values.shape[0]) if first else range(values.shape[0] - 1, -1, -1)
    seen = np.zeros(values.shape[1], dtype=bool)
    for pos in positions:
        take = valid[pos] & ~seen
        out[take] = filled[pos, take]
        seen[take] = True
    return out


def _nan_slope(values: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(values)
    filled = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    time = np.arange(values.shape[0], dtype=np.float32)
    slopes = np.zeros(values.shape[1], dtype=np.float32)
    for col in range(values.shape[1]):
        mask = valid[:, col]
        if int(mask.sum()) < 2:
            continue
        t = time[mask]
        y = filled[mask, col]
        t_centered = t - t.mean()
        denom = float(np.sum(t_centered * t_centered))
        if denom > 0:
            slopes[col] = float(np.sum(t_centered * (y - y.mean())) / denom)
    return slopes


def window_features(values: np.ndarray, feature_mode: str = "basic") -> np.ndarray:
    missing = np.isnan(values).mean(axis=0)
    filled = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    valid = ~np.isnan(values)
    count = valid.sum(axis=0)
    safe = np.where(valid, values, 0.0)
    mean = safe.sum(axis=0) / np.maximum(count, 1)
    centered = np.where(valid, values - mean, 0.0)
    std = np.sqrt((centered * centered).sum(axis=0) / np.maximum(count, 1))
    min_value = np.where(count > 0, np.nanmin(np.where(valid, values, np.inf), axis=0), 0.0)
    max_value = np.where(count > 0, np.nanmax(np.where(valid, values, -np.inf), axis=0), 0.0)
    last = filled[-1]
    pieces = [mean, std, min_value, max_value, last, missing]
    if feature_mode == "trend":
        first_observed = _observed_endpoint(values, first=True)
        last_observed = _observed_endpoint(values, first=False)
        delta = last_observed - first_observed
        slope = _nan_slope(values)
        observed_fraction = count / max(1, values.shape[0])
        pieces.extend([first_observed, last_observed, delta, slope, observed_fraction])
    elif feature_mode != "basic":
        raise ValueError(f"Unsupported feature mode: {feature_mode}")
    return np.nan_to_num(np.concatenate(pieces), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def patient_has_positive(path: Path) -> int:
    try:
        _values, labels, _columns = load_patient(path)
    except Exception:
        return 0
    return int(np.any(labels > 0))


def stratified_owner_split(files: list[Path], train_frac: float, seed: int) -> tuple[list[Path], list[Path]]:
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[Path]] = defaultdict(list)
    for path in files:
        by_label[patient_has_positive(path)].append(path)
    train: list[Path] = []
    test: list[Path] = []
    for label, items in by_label.items():
        shuffled = list(items)
        rng.shuffle(shuffled)
        split_at = max(1, int(round(len(shuffled) * train_frac))) if len(shuffled) > 1 else len(shuffled)
        train.extend(shuffled[:split_at])
        test.extend(shuffled[split_at:])
    return sorted(train), sorted(test)


def generate_windows(
    files: list[Path],
    split_name: str,
    window_length: int,
    stride: int,
    label_mode: str,
    feature_mode: str = "basic",
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    records: list[dict[str, object]] = []
    features: list[np.ndarray] = []
    labels_out: list[int] = []
    for path in files:
        values, labels, _columns = load_patient(path)
        if values.shape[0] < window_length:
            continue
        owner_id = f"{path.parent.name}:{path.stem}"
        for ordinal, start in enumerate(range(0, values.shape[0] - window_length + 1, stride)):
            end = start + window_length
            window_label_values = labels[start:end]
            if label_mode == "end":
                label = int(window_label_values[-1] > 0)
            elif label_mode == "any":
                label = int(np.any(window_label_values > 0))
            else:
                raise ValueError(f"Unsupported label mode: {label_mode}")
            row_index = len(features)
            features.append(window_features(values[start:end], feature_mode))
            labels_out.append(label)
            records.append(
                {
                    "scenario": f"sepsis_{split_name}",
                    "window_id": f"{split_name}_{path.parent.name}_{path.stem}_w{ordinal}",
                    "owner_id": owner_id,
                    "owner_ids": owner_id,
                    "start": start,
                    "end": end,
                    "row_index": row_index,
                    "label": str(label),
                    "patient_file": path.name,
                }
            )
    if not features:
        raise ValueError("No windows generated")
    return records, np.stack(features).astype(np.float32), np.asarray(labels_out, dtype=np.int64)


def write_mapping(records: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "window_id",
        "owner_id",
        "owner_ids",
        "start",
        "end",
        "row_index",
        "label",
        "patient_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def parse_budget_list(text: str) -> list[int | None]:
    values: list[int | None] = []
    for item in text.split(","):
        token = item.strip().lower()
        if not token:
            continue
        values.append(None if token in {"all", "none"} else int(token))
    return values


def write_contract(
    path: Path,
    mapping_path: Path,
    args: argparse.Namespace,
    step_budget: int | None,
    protocol: SepsisPublicProtocol,
    class_weights: list[float] | None,
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
            "class_weight": args.class_weight,
            "class_weights": class_weights,
        },
        "data": {
            "dataset": "physionet2019_sepsis",
            "input_representation": "record_local_basic_window_features_240_v1",
            "public_protocol": {
                "protocol_id": protocol.protocol_id,
                "patient_manifest_sha256": protocol.patient_manifest_sha256,
                "train_owners": len(protocol.train_files),
                "test_owners": len(protocol.test_files),
                "fixed_class_weights": list(protocol.fixed_class_weights),
            },
            "preprocessing": protocol.preprocessor.contract_binding(),
        },
    }
    path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")


def fmt_metric(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def write_summary(
    path: Path,
    rows: list[dict[str, object]],
    train_owners: int,
    test_owners: int,
    train_positive_rate: float,
    test_positive_rate: float,
) -> None:
    lines = [
        "# Sepsis OWA-DPSGD Run",
        "",
        f"- train owners: `{train_owners}`",
        f"- test owners: `{test_owners}`",
        f"- train positive window rate: `{train_positive_rate:.4f}`",
        f"- test positive window rate: `{test_positive_rate:.4f}`",
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
            "This is a large-owner clinical smoke run. It uses patient files as owners and contiguous hourly windows as generated training examples.",
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
        default=str(DEFAULT_SEPSIS_PREPROCESSING_ARTIFACT),
    )
    parser.add_argument(
        "--preprocessing-artifact-sha256",
        default=DEFAULT_SEPSIS_PREPROCESSING_SHA256,
    )
    parser.add_argument("--output-dir", default="reports/sepsis_owa_demo")
    parser.add_argument("--max-owners", type=int, default=400)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--window-length", type=int, default=12)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--label-mode", choices=["any", "end"], default="any")
    parser.add_argument("--feature-mode", choices=["basic", "trend"], default="basic")
    parser.add_argument("--policy", default="support_balanced_cap")
    parser.add_argument("--max-windows-per-owner", type=int, default=8)
    parser.add_argument("--step-window-budgets", default="8")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--owner-batch-size", type=int, default=32)
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
    validate_registered_protocol_args(args)
    step_budgets = parse_budget_list(args.step_window_budgets)
    if step_budgets != [8]:
        raise ValueError(
            "Registered Sepsis protocol requires --step-window-budgets 8"
        )
    protocol = load_registered_public_protocol(
        args.cache_dir,
        args.preprocessing_artifact,
        args.preprocessing_artifact_sha256,
    )
    train_files = list(protocol.train_files)
    test_files = list(protocol.test_files)
    train_records, train_x_raw, train_y = generate_windows(
        train_files, "train", args.window_length, args.stride, args.label_mode, args.feature_mode
    )
    test_records, test_x_raw, test_y = generate_windows(
        test_files, "test", args.window_length, args.stride, args.label_mode, args.feature_mode
    )

    if train_x_raw.shape != (1601, 240) or test_x_raw.shape != (398, 240):
        raise ValueError(
            "Sepsis generated feature tables do not match the registered protocol"
        )
    train_x = protocol.preprocessor.transform(train_x_raw)
    test_x = protocol.preprocessor.transform(test_x_raw)
    class_weights = None
    if args.class_weight == "balanced":
        class_weights = list(protocol.fixed_class_weights)

    mapping_path = output_dir / "sepsis_train_mapping.csv"
    write_mapping(train_records, mapping_path)
    write_mapping(test_records, output_dir / "sepsis_test_mapping.csv")

    rows: list[dict[str, object]] = []
    for step_budget in step_budgets:
        budget_name = "all" if step_budget is None else str(step_budget)
        run_dir = output_dir / f"budget_{budget_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        contract_path = run_dir / "contract.yaml"
        write_contract(
            contract_path,
            mapping_path.resolve(),
            args,
            step_budget,
            protocol,
            class_weights,
        )
        route = compile_contract(load_contract(contract_path), base_dir=ROOT)

        torch.manual_seed(args.seed)
        model = build_classifier(train_x.shape[1], 2, args.model_type, args.hidden_dim)
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
                class_weights=class_weights,
            ),
        )
        certificate = build_certificate(route, result.to_dict())
        write_certificate(certificate, run_dir / "certificate.json")
        write_certificate(certificate, run_dir / "certificate.md")
        payload = result.to_dict()
        payload["step_window_budget"] = budget_name
        payload["policy"] = args.policy
        payload["model_type"] = args.model_type
        payload["feature_mode"] = args.feature_mode
        payload["hidden_dim"] = args.hidden_dim
        payload["max_windows_per_owner"] = args.max_windows_per_owner
        payload["selected_mapping_stats"] = certificate.selected_mapping_stats
        payload["selection_diagnostics"] = certificate.selection_diagnostics
        payload["group_fallback"] = certificate.group_fallback
        (run_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        rows.append(payload)

    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_summary(
        output_dir / "summary.md",
        rows,
        train_owners=len(train_files),
        test_owners=len(test_files),
        train_positive_rate=float(np.mean(train_y)),
        test_positive_rate=float(np.mean(test_y)),
    )
    print(f"Wrote {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
