"""Run Owner-Window Aligned DP-SGD on WISDM feature windows."""

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
from unitdp.preprocessing import (
    FixedAffinePreprocessor,
    file_sha256,
    load_fixed_affine_preprocessor,
)


WINDOW_LENGTH = 200
WINDOW_STEP = 100
WISDM_RAW_SHA256 = "9aba4b1eaece56ab6d9187b16fc010c176e464750a4ec44c9d833514ecc1fa2e"
WISDM_USER_IDS = tuple(range(1, 37))
WISDM_TRAIN_OWNER_IDS = tuple(range(1, 26))
WISDM_TEST_OWNER_IDS = tuple(range(26, 37))
WISDM_ACTIVITY_LABELS = (
    "Downstairs",
    "Jogging",
    "Sitting",
    "Standing",
    "Upstairs",
    "Walking",
)
WISDM_ACTIVITY_TO_INDEX = {
    label: index for index, label in enumerate(WISDM_ACTIVITY_LABELS)
}
WISDM_PARSED_RECORDS = 1_098_208
WISDM_MALFORMED_RECORDS = 1
DEFAULT_WISDM_PREPROCESSING_ARTIFACT = (
    ROOT
    / "configs"
    / "preprocessing"
    / "wisdm_v1_1_train_owners_1_25_stats24_standard_scaler_v1.json"
)
DEFAULT_WISDM_PREPROCESSING_SHA256 = (
    "707ee3abf755a9601263d16e5f1ce8e5e9096f956f42da4d42cf57e29e6bf32f"
)


def parse_raw_wisdm(
    path: Path,
    *,
    strict_public_protocol: bool = True,
) -> dict[int, list[dict[str, object]]]:
    """Parse semicolon-delimited WISDM records in exact source order."""

    if strict_public_protocol:
        observed_sha256 = file_sha256(path)
        if observed_sha256 != WISDM_RAW_SHA256:
            raise ValueError(
                "WISDM public source digest mismatch: "
                f"expected {WISDM_RAW_SHA256}, got {observed_sha256}"
            )

    rows_by_user: dict[int, list[dict[str, object]]] = defaultdict(list)
    source_record_idx = 0
    malformed_records = 0
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            # Semicolon is the documented record delimiter. Some public-file
            # lines contain two records, and some valid records end in ",;".
            for raw_record in line.split(";"):
                record = raw_record.strip().rstrip(",").strip()
                if not record:
                    continue
                parts = [part.strip() for part in record.split(",")]
                if len(parts) != 6:
                    malformed_records += 1
                    source_record_idx += 1
                    continue
                try:
                    user_id = int(parts[0])
                    activity = parts[1]
                    timestamp = int(parts[2])
                    xyz = (float(parts[3]), float(parts[4]), float(parts[5]))
                except ValueError:
                    malformed_records += 1
                    source_record_idx += 1
                    continue
                rows_by_user[user_id].append(
                    {
                        "line_idx": source_record_idx,
                        "activity": activity,
                        "timestamp": timestamp,
                        "xyz": xyz,
                    }
                )
                source_record_idx += 1
    for user_id in rows_by_user:
        rows_by_user[user_id].sort(key=lambda row: int(row["line_idx"]))
        for event_idx, row in enumerate(rows_by_user[user_id]):
            row["event_idx"] = event_idx
    parsed = sum(len(rows) for rows in rows_by_user.values())
    observed_users = tuple(sorted(rows_by_user))
    observed_activities = {
        str(row["activity"])
        for rows in rows_by_user.values()
        for row in rows
    }
    if strict_public_protocol:
        if parsed != WISDM_PARSED_RECORDS:
            raise ValueError(
                f"WISDM parsed-record mismatch: expected {WISDM_PARSED_RECORDS}, got {parsed}"
            )
        if malformed_records != WISDM_MALFORMED_RECORDS:
            raise ValueError(
                "WISDM malformed-record mismatch: "
                f"expected {WISDM_MALFORMED_RECORDS}, got {malformed_records}"
            )
        if observed_users != WISDM_USER_IDS:
            raise ValueError(
                f"WISDM owner vocabulary mismatch: expected {WISDM_USER_IDS}, "
                f"got {observed_users}"
            )
        if observed_activities != set(WISDM_ACTIVITY_LABELS):
            raise ValueError(
                "WISDM activity vocabulary mismatch: "
                f"expected {WISDM_ACTIVITY_LABELS}, got {sorted(observed_activities)}"
            )
    return dict(rows_by_user)


def fixed_owner_split(
    rows_by_user: dict[int, list[dict[str, object]]],
    train_user_frac: float = 0.7,
) -> tuple[list[int], list[int]]:
    """Return the registered public owner split, never an observed-size split."""

    if abs(float(train_user_frac) - 0.7) > 1e-12:
        raise ValueError(
            "The registered WISDM protocol fixes train owners 1..25 and "
            "test owners 26..36; train_user_frac must remain 0.7"
        )
    observed_users = tuple(sorted(rows_by_user))
    if observed_users != WISDM_USER_IDS:
        raise ValueError(
            f"WISDM owner vocabulary mismatch: expected {WISDM_USER_IDS}, "
            f"got {observed_users}"
        )
    return list(WISDM_TRAIN_OWNER_IDS), list(WISDM_TEST_OWNER_IDS)


def encode_activity_labels(labels: np.ndarray) -> np.ndarray:
    """Encode the registered public activity vocabulary without fitting."""

    values = [str(label) for label in np.asarray(labels).tolist()]
    unknown = sorted(set(values).difference(WISDM_ACTIVITY_TO_INDEX))
    if unknown:
        raise ValueError(f"Unknown WISDM activity labels: {unknown}")
    return np.asarray(
        [WISDM_ACTIVITY_TO_INDEX[label] for label in values],
        dtype=np.int64,
    )


def load_registered_preprocessor(
    artifact_path: str | Path = DEFAULT_WISDM_PREPROCESSING_ARTIFACT,
    expected_sha256: str = DEFAULT_WISDM_PREPROCESSING_SHA256,
) -> FixedAffinePreprocessor:
    """Load the registered public-fixed WISDM preprocessing artifact."""

    return load_fixed_affine_preprocessor(
        artifact_path,
        expected_payload_sha256=expected_sha256,
    )


def user_segments(rows: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    segments: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    last_activity: str | None = None
    for row in rows:
        activity = str(row["activity"])
        if last_activity is not None and activity != last_activity:
            if current:
                segments.append(current)
            current = []
        current.append(row)
        last_activity = activity
    if current:
        segments.append(current)
    return segments


def featurize(values: list[tuple[float, float, float]]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    mag = np.sqrt(np.sum(arr * arr, axis=1, keepdims=True))
    full = np.concatenate([arr, mag], axis=1)
    pieces = [
        full.mean(axis=0),
        full.std(axis=0),
        full.min(axis=0),
        full.max(axis=0),
        np.percentile(full, 25, axis=0),
        np.percentile(full, 75, axis=0),
    ]
    return np.concatenate(pieces).astype(np.float32)


def generate_windows(
    rows_by_user: dict[int, list[dict[str, object]]],
    owners: list[int],
    split: str,
    scenario: str,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    records: list[dict[str, object]] = []
    features: list[np.ndarray] = []
    labels: list[str] = []
    for owner_id in owners:
        rows = rows_by_user[owner_id]
        for segment_idx, segment in enumerate(user_segments(rows)):
            if len(segment) < WINDOW_LENGTH:
                continue
            activity = str(segment[0]["activity"])
            for ordinal, offset in enumerate(range(0, len(segment) - WINDOW_LENGTH + 1, WINDOW_STEP)):
                window = segment[offset : offset + WINDOW_LENGTH]
                start = int(window[0]["event_idx"])
                end = int(window[-1]["event_idx"]) + 1
                row_index = len(features)
                values = [row["xyz"] for row in window]  # type: ignore[index]
                features.append(featurize(values))
                labels.append(activity)
                records.append(
                    {
                        "scenario": scenario,
                        "window_id": f"{split}_{scenario}_u{owner_id}_s{segment_idx}_w{ordinal}",
                        "owner_id": str(owner_id),
                        "owner_ids": str(owner_id),
                        "start": start,
                        "end": end,
                        "row_index": row_index,
                        "label": activity,
                    }
                )
    return records, np.asarray(features, dtype=np.float32), np.asarray(labels)


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
            "dataset": "wisdm_v1_1",
            "input_representation": "record_local_stats24_v1",
            "preprocessing": preprocessing_binding,
        },
    }
    path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")


def fmt_metric(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def write_summary(path: Path, rows: list[dict[str, object]], train_owners: int, test_owners: int) -> None:
    lines = [
        "# WISDM OWA-DPSGD Run",
        "",
        f"- train owners: `{train_owners}`",
        f"- test owners: `{test_owners}`",
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
            "This run uses WISDM feature windows. Users are owners; generated windows are the training units.",
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
        default=str(DEFAULT_WISDM_PREPROCESSING_ARTIFACT),
    )
    parser.add_argument(
        "--preprocessing-artifact-sha256",
        default=DEFAULT_WISDM_PREPROCESSING_SHA256,
    )
    parser.add_argument("--output-dir", default="reports/wisdm_owa_demo")
    parser.add_argument("--train-user-frac", type=float, default=0.7)
    parser.add_argument("--policy", default="support_balanced_cap")
    parser.add_argument("--max-windows-per-owner", type=int, default=32)
    parser.add_argument("--step-window-budgets", default="2,8,all")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--owner-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-epsilon", type=float, default=8.0)
    parser.add_argument("--target-delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_user = parse_raw_wisdm(Path(args.data_path))
    train_owners, test_owners = fixed_owner_split(
        rows_by_user,
        args.train_user_frac,
    )
    train_records, train_x_raw, train_label_raw = generate_windows(
        rows_by_user, train_owners, "train", "wisdm_train_overlap"
    )
    test_records, test_x_raw, test_label_raw = generate_windows(
        rows_by_user, test_owners, "test", "wisdm_test_overlap"
    )
    train_y = encode_activity_labels(train_label_raw)
    test_y = encode_activity_labels(test_label_raw)
    preprocessor = load_registered_preprocessor(
        args.preprocessing_artifact,
        args.preprocessing_artifact_sha256,
    )
    preprocessor.verify_reference_file(Path(args.data_path))
    train_x = preprocessor.transform(train_x_raw)
    test_x = preprocessor.transform(test_x_raw)

    mapping_path = output_dir / "wisdm_train_mapping.csv"
    write_mapping(train_records, mapping_path)
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
    write_summary(output_dir / "summary.md", rows, len(train_owners), len(test_owners))
    print(f"Wrote {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
