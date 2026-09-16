"""Build the registered PhysioNet 2019 Sepsis public protocol artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get("UNITDP_DATA_ROOT", str(ROOT / "data"))
)
SRC = ROOT / "src"
for path in [ROOT, SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import run_sepsis_owa as sepsis  # noqa: E402
from unitdp.mapping import WindowMapping, WindowRecord  # noqa: E402
from unitdp.policies import ContributionPolicy  # noqa: E402
from unitdp.preprocessing import (  # noqa: E402
    FIXED_AFFINE_SCHEMA,
    PUBLIC_FIXED_SCOPE,
    file_sha256,
    preprocessing_payload_sha256,
)


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def to_window_record(raw: dict[str, object]) -> WindowRecord:
    return WindowRecord(
        window_id=str(raw["window_id"]),
        owner_ids=(str(raw["owner_id"]),),
        start=int(raw["start"]),
        end=int(raw["end"]),
        row_index=int(raw["row_index"]),
        label=str(raw["label"]),
        scenario=str(raw["scenario"]),
    )


def owner_id(path: Path) -> str:
    return f"{path.parent.name}:{path.stem}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        default=str(DATA_ROOT / "sepsis2019_cache"),
    )
    parser.add_argument(
        "--output",
        default=(
            "configs/preprocessing/"
            "physionet2019_setA_first400_basic12x6_public_scaler_v1.json"
        ),
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cohort = sepsis.patient_files(cache_dir, 400)
    if len(cohort) != 400:
        raise ValueError(f"Expected 400 public cohort files, got {len(cohort)}")

    patient_manifest: list[dict[str, object]] = []
    eligible_files: list[Path] = []
    feature_columns: list[str] | None = None
    for path in cohort:
        values, labels, columns = sepsis.load_patient(path)
        if feature_columns is None:
            feature_columns = columns
        elif columns != feature_columns:
            raise ValueError(f"Patient feature schema mismatch in {path}")
        eligible = bool(values.shape[0] >= 12)
        if eligible:
            eligible_files.append(path)
        patient_manifest.append(
            {
                "relative_path": path.relative_to(cache_dir).as_posix(),
                "owner_id": owner_id(path),
                "sha256": file_sha256(path),
                "rows": int(values.shape[0]),
                "positive_owner": int(np.any(labels > 0)),
                "eligible": eligible,
            }
        )

    train_files, test_files = sepsis.stratified_owner_split(
        eligible_files,
        train_frac=0.8,
        seed=13,
    )
    train_records, train_x, train_y = sepsis.generate_windows(
        train_files,
        "train",
        12,
        6,
        "any",
        "basic",
    )
    test_records, test_x, test_y = sepsis.generate_windows(
        test_files,
        "test",
        12,
        6,
        "any",
        "basic",
    )
    if train_x.shape != (1601, 240) or test_x.shape != (398, 240):
        raise ValueError(
            "Unexpected registered Sepsis feature shapes: "
            f"train={train_x.shape}, test={test_x.shape}"
        )
    if not np.isfinite(train_x).all() or not np.isfinite(test_x).all():
        raise ValueError("Registered Sepsis features contain non-finite values")

    policy = ContributionPolicy(
        policy_type="support_balanced_cap",
        max_windows_per_owner=8,
        windows_per_owner_per_step=8,
        seed=0,
    )
    selected = policy.select_mapping(
        WindowMapping([to_window_record(record) for record in train_records])
    ).records
    selected_indices = [record.row_index for record in selected]
    selected_y = train_y[selected_indices]
    selected_counts = np.bincount(selected_y, minlength=2).astype(np.int64)
    fixed_class_weights = (
        len(selected_y) / np.maximum(selected_counts, 1) / 2.0
    ).tolist()

    scaler = StandardScaler().fit(train_x.astype(np.float32, copy=False))
    patient_manifest_sha256 = canonical_sha256(patient_manifest)
    payload = {
        "schema": FIXED_AFFINE_SCHEMA,
        "artifact_id": (
            "physionet2019_setA_first400_basic12x6_public_scaler_v1"
        ),
        "scope": PUBLIC_FIXED_SCOPE,
        "dataset": "PhysioNet/CinC Challenge 2019 Sepsis",
        "representation": "record_local_basic_window_features_240_v1",
        "input_dim": 240,
        "transform": "standard_scaler",
        "mean": [float(value) for value in scaler.mean_],
        "scale": [float(value) for value in scaler.scale_],
        "fit_on_protected_data": False,
        "generation": {
            "method": "sklearn.preprocessing.StandardScaler.fit",
            "input_dtype": "float32",
            "sample_count": int(train_x.shape[0]),
            "sklearn_version": sklearn.__version__,
        },
        "source": {
            "reference": (
                "sepsis2019_cache/training_setA/p000001.psv..p000400.psv"
            ),
            "sha256": patient_manifest_sha256,
            "is_public_reference": True,
        },
        "dataset_protocol": {
            "protocol_id": "physionet2019_setA_first400_basic12x6_v1",
            "cohort_selection": {
                "ordered_source_sets": ["training_setA", "training_setB"],
                "first_files": 400,
            },
            "patient_manifest_sha256": patient_manifest_sha256,
            "patient_manifest": patient_manifest,
            "eligible_min_rows": 12,
            "eligible_owner_ids": [owner_id(path) for path in eligible_files],
            "train_owner_ids": [owner_id(path) for path in train_files],
            "test_owner_ids": [owner_id(path) for path in test_files],
            "split_generation": {
                "type": "stratified_by_public_owner_label",
                "train_fraction": 0.8,
                "seed": 13,
                "materialized": True,
            },
            "feature_columns": feature_columns,
            "window_length": 12,
            "window_step": 6,
            "label_mode": "any",
            "feature_mode": "basic",
            "feature_scope": "record_local_window",
            "feature_dimension": 240,
            "train_window_count": int(train_x.shape[0]),
            "test_window_count": int(test_x.shape[0]),
            "selection_policy": {
                "type": "support_balanced_cap",
                "max_windows_per_owner": 8,
                "windows_per_owner_per_step": 8,
                "selected_window_count": len(selected_indices),
            },
            "fixed_class_weights": [
                float(weight) for weight in fixed_class_weights
            ],
            "selected_class_counts": [
                int(count) for count in selected_counts.tolist()
            ],
        },
        "deployment_constraint": (
            "Treat the cohort, patient eligibility, split, feature schema, "
            "class weights, and affine parameters as immutable public side "
            "information. Do not regenerate them from protected patients."
        ),
    }
    payload["payload_sha256"] = preprocessing_payload_sha256(payload)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(payload["payload_sha256"])


if __name__ == "__main__":
    main()
