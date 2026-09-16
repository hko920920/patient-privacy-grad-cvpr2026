#!/usr/bin/env python3
"""Deterministic NIH ChestXray14 input adapter for SD 2.1.

The adapter deliberately performs no augmentation, histogram equalization,
per-image standardization, or private-data-derived normalization.  It keeps
the complete 1024x1024 PA field of view, resizes a single grayscale channel,
then replicates that channel to the three-channel SD input interface.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


SCHEMA = "nih-cxr14-sd21-model-input/v1"
PROMPT_POLICY = "nih-cxr14-weak-label-prompt/v1"
SOURCE_SIZE = (1024, 1024)
ALLOWED_PARTITIONS = {
    "private_train",
    "public_development",
    "privacy_attack_holdout",
    "final_test",
}
NIH_LABEL_ORDER = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
)
LABEL_TEXT = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Effusion": "pleural effusion",
    "Infiltration": "infiltration",
    "Mass": "mass opacity",
    "Nodule": "nodule opacity",
    "Pneumonia": "pneumonia",
    "Pneumothorax": "pneumothorax",
    "Consolidation": "consolidation",
    "Edema": "edema",
    "Emphysema": "emphysema",
    "Fibrosis": "fibrosis",
    "Pleural_Thickening": "pleural thickening",
    "Hernia": "hernia",
}


@dataclass(frozen=True)
class CxrRecord:
    image_id: str
    patient_id: str
    partition: str
    official_source_split: str
    view: str
    finding_labels: tuple[str, ...]
    primary_groups: tuple[str, ...]
    target_patient: int
    cap_rank: int
    cap_key_sha256: str
    image_filename: str


def _required(row: dict[str, str], field: str) -> str:
    value = row.get(field, "").strip()
    if not value:
        raise ValueError(f"manifest field {field!r} is missing")
    return value


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in value.split("|") if token.strip())


def read_manifest(
    manifest_path: Path | str,
    partitions: Iterable[str] | None = None,
) -> list[CxrRecord]:
    """Read the frozen capped manifest and enforce patient isolation."""

    path = Path(manifest_path)
    selected = set(partitions) if partitions is not None else None
    if selected is not None:
        unknown = selected - ALLOWED_PARTITIONS
        if unknown:
            raise ValueError(f"unknown partitions requested: {sorted(unknown)}")

    records: list[CxrRecord] = []
    seen_images: set[str] = set()
    patient_partition: dict[str, str] = {}
    allowed_labels = set(NIH_LABEL_ORDER) | {"No Finding"}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            partition = _required(row, "partition")
            if partition not in ALLOWED_PARTITIONS:
                raise ValueError(f"invalid partition: {partition}")
            if selected is not None and partition not in selected:
                continue
            image_id = _required(row, "image_id")
            patient_id = _required(row, "patient_id")
            if image_id in seen_images:
                raise ValueError(f"duplicate image_id: {image_id}")
            seen_images.add(image_id)
            previous = patient_partition.setdefault(patient_id, partition)
            if previous != partition:
                raise ValueError(f"patient {patient_id} crosses partitions")
            labels = _tokens(_required(row, "finding_labels"))
            unknown_labels = set(labels) - allowed_labels
            if unknown_labels:
                raise ValueError(
                    f"unknown labels for {image_id}: {sorted(unknown_labels)}"
                )
            if "No Finding" in labels and len(labels) != 1:
                raise ValueError(f"No Finding is mixed with positive labels: {image_id}")
            target_patient = int(_required(row, "target_patient"))
            if target_patient not in (0, 1):
                raise ValueError(f"invalid target_patient for {image_id}")
            record = CxrRecord(
                image_id=image_id,
                patient_id=patient_id,
                partition=partition,
                official_source_split=_required(row, "official_source_split"),
                view=_required(row, "view"),
                finding_labels=labels,
                primary_groups=_tokens(row.get("primary_groups", "")),
                target_patient=target_patient,
                cap_rank=int(_required(row, "cap_rank")),
                cap_key_sha256=_required(row, "cap_key_sha256"),
                image_filename=_required(row, "image_filename"),
            )
            if record.view != "PA":
                raise ValueError(f"non-PA record reached model input: {image_id}")
            if record.cap_rank < 1:
                raise ValueError(f"invalid cap rank for {image_id}")
            records.append(record)
    if not records:
        raise ValueError(f"manifest selection is empty: {path}")
    return records


def prompt_for_record(record: CxrRecord) -> str:
    """Map weak NIH labels to a fixed, non-diagnostic text condition."""

    prefix = "a frontal posteroanterior chest radiograph"
    labels = set(record.finding_labels)
    if labels == {"No Finding"}:
        return prefix + " with no labeled finding"
    ordered = [label for label in NIH_LABEL_ORDER if label in labels]
    if len(ordered) != len(labels):
        raise ValueError(f"prompt label mismatch for {record.image_id}")
    descriptions = [LABEL_TEXT[label] for label in ordered]
    if len(descriptions) == 1:
        joined = descriptions[0]
    elif len(descriptions) == 2:
        joined = " and ".join(descriptions)
    else:
        joined = ", ".join(descriptions[:-1]) + ", and " + descriptions[-1]
    return prefix + " with radiographic findings of " + joined


def load_native_grayscale(path: Path | str) -> tuple[Any, dict[str, Any]]:
    """Decode an original NIH PNG without modifying it.

    Native ``L`` is retained.  ``RGBA`` is accepted only if R=G=B for every
    pixel and alpha is exactly 255, then the shared intensity channel is used.
    All other modes, geometry, color, or transparency fail closed.
    """

    import numpy as np
    from PIL import Image

    image_path = Path(path)
    with Image.open(image_path) as source:
        source.load()
        native_format = source.format
        native_mode = source.mode
        native_size = source.size
        if native_format != "PNG":
            raise ValueError(f"expected PNG, got {native_format}: {image_path}")
        if native_size != SOURCE_SIZE:
            raise ValueError(
                f"expected source size {SOURCE_SIZE}, got {native_size}: {image_path}"
            )
        if native_mode == "L":
            grayscale = source.copy()
            conversion = "native_L_identity"
        elif native_mode == "RGBA":
            pixels = np.asarray(source, dtype=np.uint8)
            if pixels.shape != (SOURCE_SIZE[1], SOURCE_SIZE[0], 4):
                raise ValueError(f"unexpected RGBA array shape: {pixels.shape}")
            if not (
                np.array_equal(pixels[:, :, 0], pixels[:, :, 1])
                and np.array_equal(pixels[:, :, 0], pixels[:, :, 2])
            ):
                raise ValueError(f"RGBA is not grayscale-equivalent: {image_path}")
            if not np.all(pixels[:, :, 3] == 255):
                raise ValueError(f"RGBA alpha is not fully opaque: {image_path}")
            grayscale = Image.fromarray(np.ascontiguousarray(pixels[:, :, 0]))
            conversion = "opaque_grayscale_RGBA_shared_channel"
        else:
            raise ValueError(f"unsupported native mode {native_mode}: {image_path}")
    return grayscale, {
        "native_format": native_format,
        "native_mode": native_mode,
        "native_size": list(native_size),
        "native_conversion": conversion,
    }


def preprocess_path(path: Path | str, image_size: int) -> tuple[Any, dict[str, Any]]:
    """Return an RGB model image using full-field grayscale Lanczos resize."""

    from PIL import Image

    if image_size not in (256, 512):
        raise ValueError("frozen profiles permit only 256 or 512 pixels")
    grayscale, metadata = load_native_grayscale(path)
    resized = grayscale.resize(
        (image_size, image_size),
        resample=Image.Resampling.LANCZOS,
        reducing_gap=None,
    )
    rgb = Image.merge("RGB", (resized, resized, resized))
    metadata.update(
        {
            "geometry_policy": "full_field_square_resize_no_crop_no_pad",
            "resampler": "PIL.Image.Resampling.LANCZOS",
            "reducing_gap": None,
            "output_size": [image_size, image_size],
            "channel_mapping": "resize_one_grayscale_channel_then_replicate_RGB",
            "augmentation": "none",
            "intensity_transform": "uint8_identity_then_x_over_127.5_minus_1",
        }
    )
    return rgb, metadata


def pil_to_normalized_tensor(image: Any) -> Any:
    """Convert exact uint8 RGB pixels to contiguous CHW float32 [-1, 1]."""

    import numpy as np
    import torch

    pixels = np.asarray(image, dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB image, got {pixels.shape}")
    if not (
        np.array_equal(pixels[:, :, 0], pixels[:, :, 1])
        and np.array_equal(pixels[:, :, 0], pixels[:, :, 2])
    ):
        raise ValueError("model RGB channels are not identical")
    values = pixels.astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(values).permute(2, 0, 1).contiguous()
    if not torch.isfinite(tensor).all():
        raise ValueError("preprocessed tensor contains non-finite values")
    return tensor


def pil_to_unit_float_array(image: Any) -> Any:
    """Convert the exact model pixels to the PP-Mark HWC float32 [0, 1] API."""

    import numpy as np

    pixels = np.asarray(image, dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB image, got {pixels.shape}")
    return pixels.astype(np.float32) / 255.0


class CxrManifestDataset:
    """Image-unit view shared by non-DP and image-DP implementations."""

    def __init__(
        self,
        manifest_path: Path | str,
        image_root: Path | str,
        partitions: Iterable[str] | None = None,
        image_size: int = 256,
        require_all_images: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_root = Path(image_root)
        self.records = read_manifest(self.manifest_path, partitions)
        self.image_size = image_size
        if image_size not in (256, 512):
            raise ValueError("frozen profiles permit only 256 or 512 pixels")
        if require_all_images:
            missing = [
                record.image_filename
                for record in self.records
                if not (self.image_root / record.image_filename).is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} manifest images are missing; first={missing[0]}"
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image, metadata = preprocess_path(
            self.image_root / record.image_filename, self.image_size
        )
        return {
            "pixel_values": pil_to_normalized_tensor(image),
            "prompt": prompt_for_record(record),
            "image_id": record.image_id,
            "patient_id": record.patient_id,
            "partition": record.partition,
            "finding_labels": record.finding_labels,
            "primary_groups": record.primary_groups,
            "target_patient": record.target_patient,
            "cap_rank": record.cap_rank,
            "preprocessing": metadata,
        }


class PatientGroupedView:
    """Patient-unit index; DP aggregation/noise is intentionally out of scope."""

    def __init__(self, dataset: CxrManifestDataset) -> None:
        self.dataset = dataset
        indices: dict[str, list[int]] = defaultdict(list)
        partitions: dict[str, str] = {}
        for index, record in enumerate(dataset.records):
            indices[record.patient_id].append(index)
            previous = partitions.setdefault(record.patient_id, record.partition)
            if previous != record.partition:
                raise ValueError(f"patient {record.patient_id} crosses partitions")
        self.patient_ids = tuple(sorted(indices, key=lambda value: int(value)))
        self.indices_by_patient = {
            patient_id: tuple(
                sorted(indices[patient_id], key=lambda i: dataset.records[i].cap_rank)
            )
            for patient_id in self.patient_ids
        }

    def __len__(self) -> int:
        return len(self.patient_ids)

    def record_indices(self, patient_id: str) -> tuple[int, ...]:
        return self.indices_by_patient[patient_id]

    def iter_patient_records(self) -> Iterator[tuple[str, tuple[CxrRecord, ...]]]:
        for patient_id in self.patient_ids:
            yield patient_id, tuple(
                self.dataset.records[index]
                for index in self.indices_by_patient[patient_id]
            )

    def validate_cap(self, cap: int) -> None:
        if cap <= 0:
            raise ValueError("cap must be positive")
        offenders = [
            patient_id
            for patient_id, indices in self.indices_by_patient.items()
            if len(indices) > cap
        ]
        if offenders:
            raise ValueError(f"{len(offenders)} patients exceed K={cap}")
