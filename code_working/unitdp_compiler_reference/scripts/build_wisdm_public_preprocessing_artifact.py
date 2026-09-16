"""Build the registered WISDM v1.1 public preprocessing artifact."""

from __future__ import annotations

import argparse
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

from scripts import run_wisdm_owa as wisdm  # noqa: E402
from unitdp.preprocessing import (  # noqa: E402
    FIXED_AFFINE_SCHEMA,
    PUBLIC_FIXED_SCOPE,
    file_sha256,
    preprocessing_payload_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-data",
        default=str(
            DATA_ROOT
            / "wisdm"
            / "WISDM_ar_v1.1"
            / "WISDM_ar_v1.1_raw.txt"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "configs/preprocessing/"
            "wisdm_v1_1_train_owners_1_25_stats24_standard_scaler_v1.json"
        ),
    )
    args = parser.parse_args()

    raw_path = Path(args.raw_data)
    rows_by_user = wisdm.parse_raw_wisdm(raw_path)
    train_owners, test_owners = wisdm.fixed_owner_split(rows_by_user)
    records, train_x, train_labels = wisdm.generate_windows(
        rows_by_user,
        train_owners,
        "train",
        "wisdm_train_overlap",
    )
    encoded_labels = wisdm.encode_activity_labels(train_labels)
    if train_x.shape != (7049, 24):
        raise ValueError(f"Unexpected registered WISDM feature shape: {train_x.shape}")
    if len(records) != train_x.shape[0] or encoded_labels.shape[0] != train_x.shape[0]:
        raise ValueError("WISDM records, features, and labels are not aligned")
    if not np.isfinite(train_x).all():
        raise ValueError("WISDM public feature matrix contains non-finite values")

    scaler = StandardScaler().fit(train_x.astype(np.float32, copy=False))
    payload = {
        "schema": FIXED_AFFINE_SCHEMA,
        "artifact_id": (
            "wisdm_v1_1_train_owners_1_25_stats24_standard_scaler_v1"
        ),
        "scope": PUBLIC_FIXED_SCOPE,
        "dataset": "WISDM Activity Prediction Dataset v1.1",
        "representation": "record_local_stats24_v1",
        "input_dim": 24,
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
            "reference": "WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt",
            "sha256": file_sha256(raw_path),
            "is_public_reference": True,
        },
        "dataset_protocol": {
            "protocol_id": "wisdm_v1_1_stats24_public_split_v1",
            "parser": "semicolon_records_strip_optional_trailing_comma_v1",
            "parsed_records": wisdm.WISDM_PARSED_RECORDS,
            "malformed_records_skipped": wisdm.WISDM_MALFORMED_RECORDS,
            "owner_ids": list(wisdm.WISDM_USER_IDS),
            "train_owner_ids": train_owners,
            "test_owner_ids": test_owners,
            "activity_to_index": wisdm.WISDM_ACTIVITY_TO_INDEX,
            "window_length": wisdm.WINDOW_LENGTH,
            "window_step": wisdm.WINDOW_STEP,
            "segment_boundary": "within_owner_activity_change",
            "feature_channels": ["x", "y", "z", "magnitude"],
            "feature_statistics": ["mean", "std", "min", "max", "p25", "p75"],
            "feature_scope": "record_local_window",
        },
        "deployment_constraint": (
            "Treat the parser protocol, owner split, label vocabulary, and "
            "affine parameters as immutable public side information. Do not "
            "regenerate them from a protected deployment dataset."
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
