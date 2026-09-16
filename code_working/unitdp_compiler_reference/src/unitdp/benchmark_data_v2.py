"""Registered full-data preparers for the three AAAI-27 V2 benchmarks."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np

from unitdp.mapping import WindowMapping, WindowRecord
from unitdp.preprocessing import (
    FixedAffinePreprocessor,
    load_fixed_affine_preprocessor,
)


class BenchmarkDataV2Error(ValueError):
    """Raised when a public benchmark input differs from its registration."""


@lru_cache(maxsize=3)
def _load_audited_script_module(filename: str) -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    source = repo_root / "scripts" / filename
    module_name = f"_unitdp_audited_{source.stem}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise BenchmarkDataV2Error(
            f"Could not create import spec for {source}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


@dataclass(frozen=True)
class PreparedBenchmarkV2:
    profile_id: str
    raw_train_x: np.ndarray
    train_y: np.ndarray
    raw_test_x: np.ndarray
    test_y: np.ndarray
    train_records: tuple[dict[str, object], ...]
    test_records: tuple[dict[str, object], ...]
    preprocessor: FixedAffinePreprocessor
    source_files: tuple[Path, ...]
    data_preparation_implementation_id: str

    def transformed_train_sha256(self) -> str:
        return array_bytes_sha256(
            self.preprocessor.transform(self.raw_train_x)
        )

    def transformed_test_sha256(self) -> str:
        return array_bytes_sha256(
            self.preprocessor.transform(self.raw_test_x)
        )

    def train_labels_sha256(self) -> str:
        return array_bytes_sha256(self.train_y)

    def test_labels_sha256(self) -> str:
        return array_bytes_sha256(self.test_y)

    def train_mapping_canonical_sha256(self) -> str:
        return records_to_mapping(self.train_records).canonical_hash()

    def test_mapping_canonical_sha256(self) -> str:
        return records_to_mapping(self.test_records).canonical_hash()

    def full_data_conformance_payload(self) -> dict[str, object]:
        return {
            "train_rows": int(self.raw_train_x.shape[0]),
            "test_rows": int(self.raw_test_x.shape[0]),
            "train_transformed_sha256": (
                self.transformed_train_sha256()
            ),
            "test_transformed_sha256": self.transformed_test_sha256(),
            "train_labels_sha256": self.train_labels_sha256(),
            "test_labels_sha256": self.test_labels_sha256(),
            "train_mapping_canonical_sha256": (
                self.train_mapping_canonical_sha256()
            ),
            "test_mapping_canonical_sha256": (
                self.test_mapping_canonical_sha256()
            ),
        }

    def full_data_conformance_sha256(self) -> str:
        encoded = json.dumps(
            self.full_data_conformance_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def assert_registered_full_conformance(self) -> None:
        from unitdp.benchmark_registry_v2 import get_benchmark_profile_v2

        profile = get_benchmark_profile_v2(self.profile_id)
        if (
            self.data_preparation_implementation_id
            != profile.data_preparation_implementation_id
        ):
            raise BenchmarkDataV2Error(
                "Prepared data implementation id does not match the profile"
            )
        if self.preprocessor.contract_binding() != (
            profile.expected_contract_binding()["data"]["preprocessing"]
        ):
            raise BenchmarkDataV2Error(
                "Prepared data preprocessor does not match the profile"
            )
        observed = self.full_data_conformance_payload()
        expected = profile.full_data_conformance_payload()
        if observed != expected:
            differences = {
                key: {
                    "expected": expected[key],
                    "observed": observed.get(key),
                }
                for key in expected
                if observed.get(key) != expected[key]
            }
            raise BenchmarkDataV2Error(
                "Prepared full benchmark differs from the registered "
                f"conformance payload: {differences}"
            )
        if (
            self.full_data_conformance_sha256()
            != profile.full_data_conformance_sha256
        ):
            raise BenchmarkDataV2Error(
                "Prepared full-data conformance digest is inconsistent"
            )


def array_bytes_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def records_to_mapping(
    records: tuple[dict[str, object], ...] | list[dict[str, object]],
) -> WindowMapping:
    parsed: list[WindowRecord] = []
    for record in records:
        owner_id = str(record["owner_id"])
        owner_ids = str(record["owner_ids"])
        if owner_ids != owner_id or ";" in owner_ids:
            raise BenchmarkDataV2Error(
                "Prepared V2 mapping must use one matching owner"
            )
        parsed.append(
            WindowRecord(
                scenario=str(record["scenario"]),
                window_id=str(record["window_id"]),
                owner_ids=(owner_id,),
                start=int(record["start"]),
                end=int(record["end"]),
                row_index=int(record["row_index"]),
                label=str(record["label"]),
            )
        )
    return WindowMapping(parsed)


def _freeze_array(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value)
    array.setflags(write=False)
    return array


def _validate_prepared(
    prepared: PreparedBenchmarkV2,
    *,
    input_dim: int,
    num_classes: int,
) -> PreparedBenchmarkV2:
    for name, features, labels, records in (
        (
            "train",
            prepared.raw_train_x,
            prepared.train_y,
            prepared.train_records,
        ),
        (
            "test",
            prepared.raw_test_x,
            prepared.test_y,
            prepared.test_records,
        ),
    ):
        if (
            features.ndim != 2
            or features.shape[0] <= 0
            or features.shape[1] != input_dim
            or not np.issubdtype(features.dtype, np.number)
            or not np.isfinite(features).all()
        ):
            raise BenchmarkDataV2Error(
                f"Prepared {name} features are invalid"
            )
        if (
            labels.ndim != 1
            or labels.shape[0] != features.shape[0]
            or not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels < 0)
            or np.any(labels >= num_classes)
        ):
            raise BenchmarkDataV2Error(
                f"Prepared {name} labels are invalid"
            )
        if len(records) != features.shape[0]:
            raise BenchmarkDataV2Error(
                f"Prepared {name} mapping length is invalid"
            )
        mapping = records_to_mapping(records)
        if sorted(mapping.row_indices()) != list(range(features.shape[0])):
            raise BenchmarkDataV2Error(
                f"Prepared {name} row indices are not contiguous"
            )
    prepared.raw_train_x.setflags(write=False)
    prepared.train_y.setflags(write=False)
    prepared.raw_test_x.setflags(write=False)
    prepared.test_y.setflags(write=False)
    return prepared


def prepare_uci_har_v2(
    *,
    dataset_root: str | Path,
    preprocessing_artifact_path: str | Path,
) -> PreparedBenchmarkV2:
    """Prepare the published 561-feature UCI HAR owner split."""

    audited = _load_audited_script_module("run_uci_owa.py")
    build_records = audited.build_records
    load_split = audited.load_split

    implementation_id = (
        "unitdp.benchmark_data_v2.prepare_uci_har_v2"
    )
    root = Path(dataset_root)
    preprocessor = load_fixed_affine_preprocessor(
        preprocessing_artifact_path,
        expected_payload_sha256=(
            "774713e3bef48c784113f4097642a59741481e7495f858ed41cdb1788ae8fadf"
        ),
    )
    reference = root / "train" / "X_train.txt"
    preprocessor.verify_reference_file(reference)
    train_x, train_y, train_subjects = load_split(root, "train")
    test_x, test_y, test_subjects = load_split(root, "test")
    train_records = tuple(
        build_records(train_subjects, train_y, "train")
    )
    test_records = tuple(
        build_records(test_subjects, test_y, "test")
    )
    if (
        train_x.shape != (7352, 561)
        or test_x.shape != (2947, 561)
        or len(set(map(int, train_subjects))) != 21
        or len(set(map(int, test_subjects))) != 9
    ):
        raise BenchmarkDataV2Error(
            "UCI HAR public split does not match the registered shape"
        )
    prepared = PreparedBenchmarkV2(
        profile_id="uci_har_aaai27_v1",
        raw_train_x=_freeze_array(train_x.astype(np.float32, copy=False)),
        train_y=_freeze_array(train_y.astype(np.int64, copy=False)),
        raw_test_x=_freeze_array(test_x.astype(np.float32, copy=False)),
        test_y=_freeze_array(test_y.astype(np.int64, copy=False)),
        train_records=train_records,
        test_records=test_records,
        preprocessor=preprocessor,
        source_files=(
            reference,
            root / "train" / "y_train.txt",
            root / "train" / "subject_train.txt",
            root / "test" / "X_test.txt",
            root / "test" / "y_test.txt",
            root / "test" / "subject_test.txt",
        ),
        data_preparation_implementation_id=implementation_id,
    )
    prepared = _validate_prepared(
        prepared,
        input_dim=561,
        num_classes=6,
    )
    prepared.assert_registered_full_conformance()
    return prepared


def prepare_wisdm_v2(
    *,
    raw_path: str | Path,
    preprocessing_artifact_path: str | Path,
) -> PreparedBenchmarkV2:
    """Prepare the corrected semicolon-parsed WISDM public split."""

    audited = _load_audited_script_module("run_wisdm_owa.py")
    encode_activity_labels = audited.encode_activity_labels
    fixed_owner_split = audited.fixed_owner_split
    generate_windows = audited.generate_windows
    parse_raw_wisdm = audited.parse_raw_wisdm

    implementation_id = (
        "unitdp.benchmark_data_v2.prepare_wisdm_v2"
    )
    source = Path(raw_path)
    preprocessor = load_fixed_affine_preprocessor(
        preprocessing_artifact_path,
        expected_payload_sha256=(
            "707ee3abf755a9601263d16e5f1ce8e5e9096f956f42da4d42cf57e29e6bf32f"
        ),
    )
    preprocessor.verify_reference_file(source)
    rows_by_user = parse_raw_wisdm(
        source,
        strict_public_protocol=True,
    )
    train_owners, test_owners = fixed_owner_split(
        rows_by_user,
        0.7,
    )
    train_records_raw, train_x, train_label_names = generate_windows(
        rows_by_user,
        train_owners,
        "train",
        "wisdm_train_overlap",
    )
    test_records_raw, test_x, test_label_names = generate_windows(
        rows_by_user,
        test_owners,
        "test",
        "wisdm_test_overlap",
    )
    train_y = encode_activity_labels(train_label_names)
    test_y = encode_activity_labels(test_label_names)
    if train_x.shape != (7049, 24) or test_x.shape != (3331, 24):
        raise BenchmarkDataV2Error(
            "WISDM public split does not match the registered shape"
        )
    prepared = PreparedBenchmarkV2(
        profile_id="wisdm_aaai27_v1",
        raw_train_x=_freeze_array(train_x.astype(np.float32, copy=False)),
        train_y=_freeze_array(train_y.astype(np.int64, copy=False)),
        raw_test_x=_freeze_array(test_x.astype(np.float32, copy=False)),
        test_y=_freeze_array(test_y.astype(np.int64, copy=False)),
        train_records=tuple(train_records_raw),
        test_records=tuple(test_records_raw),
        preprocessor=preprocessor,
        source_files=(source,),
        data_preparation_implementation_id=implementation_id,
    )
    prepared = _validate_prepared(
        prepared,
        input_dim=24,
        num_classes=6,
    )
    prepared.assert_registered_full_conformance()
    return prepared


def prepare_sepsis_v2(
    *,
    cache_dir: str | Path,
    preprocessing_artifact_path: str | Path,
) -> PreparedBenchmarkV2:
    """Prepare the materialized 312/78 Sepsis public protocol."""

    audited = _load_audited_script_module("run_sepsis_owa.py")
    generate_windows = audited.generate_windows
    load_registered_public_protocol = (
        audited.load_registered_public_protocol
    )

    implementation_id = (
        "unitdp.benchmark_data_v2.prepare_sepsis_v2"
    )
    protocol = load_registered_public_protocol(
        cache_dir,
        preprocessing_artifact_path,
        (
            "e2ddd1cbb651e5ef5629bdb3ba0b885e5c9cead7c6b8afd9d5d9490e463b210b"
        ),
    )
    train_records_raw, train_x, train_y = generate_windows(
        list(protocol.train_files),
        "train",
        12,
        6,
        "any",
        "basic",
    )
    test_records_raw, test_x, test_y = generate_windows(
        list(protocol.test_files),
        "test",
        12,
        6,
        "any",
        "basic",
    )
    if train_x.shape != (1601, 240) or test_x.shape != (398, 240):
        raise BenchmarkDataV2Error(
            "Sepsis public split does not match the registered shape"
        )
    prepared = PreparedBenchmarkV2(
        profile_id="sepsis_aaai27_v1",
        raw_train_x=_freeze_array(train_x.astype(np.float32, copy=False)),
        train_y=_freeze_array(train_y.astype(np.int64, copy=False)),
        raw_test_x=_freeze_array(test_x.astype(np.float32, copy=False)),
        test_y=_freeze_array(test_y.astype(np.int64, copy=False)),
        train_records=tuple(train_records_raw),
        test_records=tuple(test_records_raw),
        preprocessor=protocol.preprocessor,
        source_files=tuple(protocol.train_files + protocol.test_files),
        data_preparation_implementation_id=implementation_id,
    )
    prepared = _validate_prepared(
        prepared,
        input_dim=240,
        num_classes=2,
    )
    prepared.assert_registered_full_conformance()
    return prepared


DATA_PREPARERS_V2: dict[str, Callable[..., PreparedBenchmarkV2]] = {
    "unitdp.benchmark_data_v2.prepare_uci_har_v2": prepare_uci_har_v2,
    "unitdp.benchmark_data_v2.prepare_wisdm_v2": prepare_wisdm_v2,
    "unitdp.benchmark_data_v2.prepare_sepsis_v2": prepare_sepsis_v2,
}


def get_data_preparer_v2(
    implementation_id: str,
) -> Callable[..., PreparedBenchmarkV2]:
    try:
        return DATA_PREPARERS_V2[implementation_id]
    except KeyError as exc:
        raise BenchmarkDataV2Error(
            f"Unregistered V2 data preparer: {implementation_id!r}"
        ) from exc


def write_prepared_mapping_v2(
    prepared: PreparedBenchmarkV2,
    path: str | Path,
    *,
    split: str = "train",
) -> Path:
    """Write the exact registered columns for strict compiler ingestion."""

    prepared.assert_registered_full_conformance()
    if split == "train":
        records = prepared.train_records
    elif split == "test":
        records = prepared.test_records
    else:
        raise BenchmarkDataV2Error("split must be train or test")
    columns = list(records[0])
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    return target
