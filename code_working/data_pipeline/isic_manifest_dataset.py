#!/usr/bin/env python3
"""Manifest-backed ISIC loader with image- and patient-unit views.

The module intentionally keeps manifest validation independent of PyTorch.
Pillow and PyTorch are imported only when pixels are requested, so split and
privacy-unit checks can run in a minimal audit environment.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


ALLOWED_PARTITIONS = {
    "private_train",
    "public_development",
    "privacy_attack_holdout",
    "final_test",
}


@dataclass(frozen=True)
class IsicRecord:
    image_id: str
    patient_id: str
    lesion_id: str
    partition: str
    target: int
    benign_malignant: str
    diagnosis: str
    sex: str
    age_approx: str
    anatom_site: str
    patient_image_count_dedup: int
    patient_target_count_dedup: int
    cap_rank: int
    cap_key_sha256: str
    image_filename: str


def _required(row: dict[str, str], field: str) -> str:
    value = row.get(field, "").strip()
    if not value:
        raise ValueError(f"manifest field {field!r} is missing")
    return value


def read_manifest(
    manifest_path: Path | str,
    partitions: Iterable[str] | None = None,
) -> list[IsicRecord]:
    path = Path(manifest_path)
    selected_partitions = set(partitions) if partitions is not None else None
    if selected_partitions is not None:
        unknown = selected_partitions - ALLOWED_PARTITIONS
        if unknown:
            raise ValueError(f"unknown partitions requested: {sorted(unknown)}")

    records: list[IsicRecord] = []
    seen_images: set[str] = set()
    patient_partition: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            partition = _required(row, "partition")
            if partition not in ALLOWED_PARTITIONS:
                raise ValueError(f"invalid partition in manifest: {partition}")
            if selected_partitions is not None and partition not in selected_partitions:
                continue

            image_id = _required(row, "image_id")
            patient_id = _required(row, "patient_id")
            if image_id in seen_images:
                raise ValueError(f"duplicate image_id in manifest: {image_id}")
            seen_images.add(image_id)
            previous = patient_partition.setdefault(patient_id, partition)
            if previous != partition:
                raise ValueError(f"patient {patient_id} crosses manifest partitions")

            target = int(_required(row, "target"))
            if target not in (0, 1):
                raise ValueError(f"invalid target for {image_id}: {target}")
            record = IsicRecord(
                image_id=image_id,
                patient_id=patient_id,
                lesion_id=_required(row, "lesion_id"),
                partition=partition,
                target=target,
                benign_malignant=row.get("benign_malignant", "").strip(),
                diagnosis=row.get("diagnosis", "").strip(),
                sex=row.get("sex", "").strip(),
                age_approx=row.get("age_approx", "").strip(),
                anatom_site=row.get("anatom_site", "").strip(),
                patient_image_count_dedup=int(
                    _required(row, "patient_image_count_dedup")
                ),
                patient_target_count_dedup=int(
                    _required(row, "patient_target_count_dedup")
                ),
                cap_rank=int(_required(row, "cap_rank")),
                cap_key_sha256=_required(row, "cap_key_sha256"),
                image_filename=_required(row, "image_filename"),
            )
            if record.cap_rank < 1:
                raise ValueError(f"invalid cap rank for {image_id}")
            records.append(record)

    if not records:
        raise ValueError(f"manifest selection is empty: {path}")
    return records


def prompt_for_record(record: IsicRecord) -> str:
    if record.target == 1:
        return "a clinical dermoscopic image of malignant melanoma"
    return "a clinical dermoscopic image of a benign skin lesion"


def preprocess_pil(image: Any, image_size: int, policy: str = "center_crop") -> Any:
    """Return a deterministic RGB square without performing augmentation.

    `center_crop` is the locked smoke/default path because the ISIC lesions are
    centered and it avoids training a synthetic padding border.  `fit_pad` is
    exposed only for a preregistered preprocessing sensitivity check.
    """

    from PIL import Image, ImageOps

    if image_size <= 0 or image_size % 8 != 0:
        raise ValueError("image_size must be a positive multiple of 8")
    image = ImageOps.exif_transpose(image).convert("RGB")
    if policy == "center_crop":
        return ImageOps.fit(
            image,
            (image_size, image_size),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
    if policy == "fit_pad":
        return ImageOps.pad(
            image,
            (image_size, image_size),
            method=Image.Resampling.LANCZOS,
            color=(0, 0, 0),
            centering=(0.5, 0.5),
        )
    raise ValueError(f"unknown preprocessing policy: {policy}")


def pil_to_normalized_tensor(image: Any) -> Any:
    import numpy as np
    import torch

    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected HxWx3 image array, got {array.shape}")
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    if not torch.isfinite(tensor).all():
        raise ValueError("preprocessed tensor contains non-finite values")
    return tensor


class IsicManifestDataset:
    """Image-unit view used by non-DP and image-DP training paths."""

    def __init__(
        self,
        manifest_path: Path | str,
        image_root: Path | str,
        partitions: Iterable[str] | None = None,
        image_size: int = 256,
        preprocessing_policy: str = "center_crop",
        require_all_images: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_root = Path(image_root)
        self.records = read_manifest(self.manifest_path, partitions)
        self.image_size = image_size
        self.preprocessing_policy = preprocessing_policy
        if require_all_images:
            missing = [
                record.image_filename
                for record in self.records
                if not (self.image_root / record.image_filename).is_file()
            ]
            if missing:
                preview = ", ".join(missing[:5])
                raise FileNotFoundError(
                    f"{len(missing)} manifest images are missing under {self.image_root}: {preview}"
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image

        record = self.records[index]
        path = self.image_root / record.image_filename
        with Image.open(path) as source:
            image = preprocess_pil(source, self.image_size, self.preprocessing_policy)
        tensor = pil_to_normalized_tensor(image)
        return {
            "pixel_values": tensor,
            "prompt": prompt_for_record(record),
            "image_id": record.image_id,
            "patient_id": record.patient_id,
            "partition": record.partition,
            "target": record.target,
            "cap_rank": record.cap_rank,
        }


class PatientGroupedView:
    """Patient-unit index for patient sampling and within-patient aggregation.

    This class does not itself claim DP.  The training mechanism must compute
    and clip one aggregate gradient per sampled patient before adding noise.
    """

    def __init__(self, dataset: IsicManifestDataset) -> None:
        self.dataset = dataset
        indices: dict[str, list[int]] = defaultdict(list)
        partition_by_patient: dict[str, str] = {}
        for index, record in enumerate(dataset.records):
            indices[record.patient_id].append(index)
            previous = partition_by_patient.setdefault(record.patient_id, record.partition)
            if previous != record.partition:
                raise ValueError(f"patient {record.patient_id} crosses partitions")
        self.patient_ids = tuple(sorted(indices))
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

    def iter_patient_records(self) -> Iterator[tuple[str, tuple[IsicRecord, ...]]]:
        for patient_id in self.patient_ids:
            yield patient_id, tuple(
                self.dataset.records[index]
                for index in self.indices_by_patient[patient_id]
            )

    def contribution_counts(self) -> list[int]:
        return [len(self.indices_by_patient[patient]) for patient in self.patient_ids]

    def validate_cap(self, cap: int) -> None:
        if cap <= 0:
            raise ValueError("cap must be positive")
        over = [
            patient
            for patient, indices in self.indices_by_patient.items()
            if len(indices) > cap
        ]
        if over:
            raise ValueError(f"{len(over)} patients exceed K={cap}")


def collate_image_records(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Minimal image-unit collator; patient aggregation belongs in its own trainer."""

    import torch

    if not batch:
        raise ValueError("cannot collate an empty batch")
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "prompt": [item["prompt"] for item in batch],
        "image_id": [item["image_id"] for item in batch],
        "patient_id": [item["patient_id"] for item in batch],
        "target": torch.tensor([item["target"] for item in batch], dtype=torch.long),
    }
