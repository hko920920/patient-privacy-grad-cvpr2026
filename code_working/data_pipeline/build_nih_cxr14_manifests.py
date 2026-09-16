#!/usr/bin/env python3
"""Build deterministic PA chest-X-ray patient manifests before model training.

The source authority is the exact NIH ChestXray14 metadata and official split
lists.  The study cohort enriches clinically consequential finding patients
against an equal number of non-target patients.  It is a method-evaluation
cohort, not an estimate of hospital prevalence.

Detailed patient/image mappings are local-only.  The aggregate report contains
counts, policies, and cryptographic commitments suitable for public evidence.
No model result is read by this program.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RELEASE_ID = "NIH-ChestXray14-2017-v2020"
VIEW_POLICY_ID = "posterior-anterior-only-v1"
COHORT_POLICY_ID = "all-target-patients-plus-hash-matched-nontarget-patients-v1"
PARTITION_POLICY_ID = "official-test-plus-stratified-70-15-15-train-patients-v1"
CAP_POLICY_ID = "within-patient-label-independent-sha256-rank-v1"

# Fixed before generator training.  The first evaluated seed was retained.
SEED = "unified-private-image-assurance|nih-cxr14-pa-target-enriched-v1|2026-09-02"

CAPS = (2, 5, 10)
CAP_ROLES = {2: "feasibility_pilot", 5: "primary_main", 10: "unit_stress"}
PARTITION_ORDER = {
    "private_train": 0,
    "public_development": 1,
    "privacy_attack_holdout": 2,
    "final_test": 3,
}

SOURCE_LOCKS = {
    "Data_Entry_2017_v2020.csv": {
        "bytes": 9_003_496,
        "sha1": "48A9F849A8F100A0F1721B33BDBD209767656111",
        "sha256": "C69A6DACA3549AF707CA9CACBDF9F9A7B6A9188E8C61157DF653520BB72D8EB1",
        "box_file_id": "219760887468",
        "role": "image, patient, view, demographic, and weak-finding authority",
    },
    "BBox_List_2017.csv": {
        "bytes": 92_416,
        "sha1": "C567CEB25277C9883C5A7FE4C2F70C2EF94B02BE",
        "sha256": "0BBFEA9D4C4E9771481B3023B1BC9F0DF9DEA924453B12986BEB29B0C4D0C95B",
        "box_file_id": "219760940956",
        "role": "optional localization evidence; not used for cohort selection",
    },
    "train_val_list.txt": {
        "bytes": 1_470_907,
        "sha1": "E3E1B677C01D28481777F3D84E10FCDAAC05694C",
        "sha256": "61FBE896321C1C1C8B75F3E4F3A08E4FEF6486D95EF8A667C31D4D60DCA6CB81",
        "box_file_id": "256056636701",
        "role": "official train/validation membership authority",
    },
    "test_list.txt": {
        "bytes": 435_131,
        "sha1": "41B85E218ABEC560A2F5999ACBCF333B0F2FA495",
        "sha256": "38CA5EF7F756092946F57C1A59FACA882ED589A1AB1F72590B45DC06C6D5E1CC",
        "box_file_id": "256055473534",
        "role": "official held-out test membership authority",
    },
}

EXPECTED_HEADER = (
    "Image Index",
    "Finding Labels",
    "Follow-up #",
    "Patient ID",
    "Patient Age",
    "Patient Sex",
    "View Position",
    "OriginalImage[Width",
    "Height]",
    "OriginalImagePixelSpacing[x",
    "y]",
)
EXPECTED_SOURCE_IMAGES = 112_120
EXPECTED_SOURCE_PATIENTS = 30_805
EXPECTED_PA_IMAGES = 67_310
EXPECTED_PA_PATIENTS = 28_868
EXPECTED_PA_TRAIN_PATIENTS = 26_221
EXPECTED_PA_TEST_PATIENTS = 2_647
EXPECTED_PA_TEST_IMAGES = 11_096
EXPECTED_TRAIN_TARGET_PATIENTS = 6_054
EXPECTED_TRAIN_NONTARGET_PATIENTS = 20_167
EXPECTED_TEST_TARGET_PATIENTS = 909
EXPECTED_TEST_NONTARGET_PATIENTS = 1_738

EXPECTED_PARTITION_PATIENTS = {
    "private_train": 8_476,
    "public_development": 1_816,
    "privacy_attack_holdout": 1_816,
    "final_test": 1_818,
}
EXPECTED_CAP_IMAGES = {
    2: {
        "private_train": 12_467,
        "public_development": 2_702,
        "privacy_attack_holdout": 2_681,
        "final_test": 2_837,
    },
    5: {
        "private_train": 18_393,
        "public_development": 4_031,
        "privacy_attack_holdout": 4_012,
        "final_test": 5_015,
    },
    10: {
        "private_train": 21_756,
        "public_development": 4_831,
        "privacy_attack_holdout": 4_740,
        "final_test": 7_165,
    },
}

ALLOWED_LABELS = {
    "No Finding",
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
}

PRIMARY_GROUPS: dict[str, frozenset[str]] = {
    "pneumothorax": frozenset({"Pneumothorax"}),
    "pneumonia_or_consolidation": frozenset({"Pneumonia", "Consolidation"}),
    "pleural_effusion": frozenset({"Effusion"}),
    "mass_or_nodule": frozenset({"Mass", "Nodule"}),
}
TARGET_LABELS = frozenset().union(*PRIMARY_GROUPS.values())

# Fixed metadata adequacy gates.  Counts may exceed these values but must not be
# tuned against model quality or privacy-attack outcomes.
K5_MINIMUMS = {
    "private_train": {
        "pneumothorax": (750, 500),
        "pneumonia_or_consolidation": (800, 750),
        "pleural_effusion": (2_400, 1_700),
        "mass_or_nodule": (3_000, 2_200),
    },
    "privacy_attack_holdout": {
        "pneumothorax": (180, 125),
        "pneumonia_or_consolidation": (170, 140),
        "pleural_effusion": (500, 350),
        "mass_or_nodule": (680, 500),
    },
    "final_test": {
        "pneumothorax": (450, 225),
        "pneumonia_or_consolidation": (250, 220),
        "pleural_effusion": (750, 450),
        "mass_or_nodule": (750, 430),
    },
}

PATIENT_FIELDS = (
    "patient_id",
    "partition",
    "official_source_split",
    "target_patient",
    "pa_image_count",
    "cohort_key_sha256",
    "partition_key_sha256",
)
IMAGE_FIELDS = (
    "image_id",
    "patient_id",
    "partition",
    "official_source_split",
    "view",
    "followup_no",
    "patient_age",
    "patient_sex",
    "finding_labels",
    "primary_groups",
    "target_patient",
    "patient_pa_image_count",
    "cap_rank",
    "cap_key_sha256",
    "selected_k2",
    "selected_k5",
    "selected_k10",
    "image_filename",
)
CENSUS_FIELDS = (
    "image_id",
    "patient_id",
    "official_source_split",
    "view",
    "followup_no",
    "patient_age",
    "patient_sex",
    "finding_labels",
    "primary_groups",
    "target_patient",
    "patient_pa_image_count",
    "image_filename",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def tagged_hash(tag: str, *values: object) -> str:
    # Keep the exact serialization used by the first, pre-training cohort
    # evaluation.  Changing even this delimiter changes the selected controls
    # and patient partitions, so it is part of the frozen policy.
    payload = "|".join((SEED, tag, *(str(value) for value in values)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def canonical_csv_bytes(
    fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    return buffer.getvalue().encode("utf-8")


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def round_ratio(count: int, numerator: int, denominator: int = 100) -> int:
    return (count * numerator + denominator // 2) // denominator


def validate_source_files(source_dir: Path) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for name, lock in SOURCE_LOCKS.items():
        path = source_dir / name
        require(path.is_file(), f"required NIH source file missing: {path}")
        actual = {
            "file": name,
            "bytes": path.stat().st_size,
            "sha1": digest_file(path, "sha1"),
            "sha256": digest_file(path, "sha256"),
            "box_file_id": lock["box_file_id"],
            "role": lock["role"],
        }
        for field in ("bytes", "sha1", "sha256"):
            require(
                actual[field] == lock[field],
                f"source lock mismatch for {name} {field}: {actual[field]}",
            )
        evidence[name] = actual
    return evidence


def read_metadata(source_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_evidence = validate_source_files(source_dir)
    metadata_path = source_dir / "Data_Entry_2017_v2020.csv"
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames or ()) == EXPECTED_HEADER, "NIH header changed")
        raw_rows = list(reader)

    require(len(raw_rows) == EXPECTED_SOURCE_IMAGES, "NIH image count changed")
    train_files = set((source_dir / "train_val_list.txt").read_text().splitlines())
    test_files = set((source_dir / "test_list.txt").read_text().splitlines())
    require(len(train_files) == 86_524, "official train/val count changed")
    require(len(test_files) == 25_596, "official test count changed")
    require(not (train_files & test_files), "official image lists overlap")

    rows: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    for raw in raw_rows:
        image_id = raw["Image Index"].strip()
        require(image_id.endswith(".png"), f"unexpected image name: {image_id}")
        require(image_id not in seen_images, f"duplicate image ID: {image_id}")
        seen_images.add(image_id)
        patient_id = int(raw["Patient ID"])
        require(int(image_id[:8]) == patient_id, f"filename/patient mismatch: {image_id}")
        labels = frozenset(part.strip() for part in raw["Finding Labels"].split("|"))
        require(labels and labels <= ALLOWED_LABELS, f"unexpected labels: {labels}")
        require(
            labels == {"No Finding"} or "No Finding" not in labels,
            f"No Finding co-occurs with another label: {image_id}",
        )
        require(raw["View Position"] in {"PA", "AP"}, f"unexpected view: {image_id}")
        require(image_id in train_files or image_id in test_files, f"unlisted image: {image_id}")
        rows.append(
            {
                "image_id": image_id,
                "patient_id": patient_id,
                "labels": labels,
                "finding_labels": "|".join(sorted(labels)),
                "followup_no": int(raw["Follow-up #"]),
                "patient_age": int(raw["Patient Age"]),
                "patient_sex": raw["Patient Sex"].strip(),
                "view": raw["View Position"],
                "official_source_split": "train_val" if image_id in train_files else "test",
            }
        )

    require(seen_images == train_files | test_files, "metadata/list populations differ")
    require(
        len({row["patient_id"] for row in rows}) == EXPECTED_SOURCE_PATIENTS,
        "NIH patient count changed",
    )
    train_patients = {row["patient_id"] for row in rows if row["official_source_split"] == "train_val"}
    test_patients = {row["patient_id"] for row in rows if row["official_source_split"] == "test"}
    require(not train_patients & test_patients, "official split crosses patients")

    pa_rows = [row for row in rows if row["view"] == "PA"]
    require(len(pa_rows) == EXPECTED_PA_IMAGES, "PA image count changed")
    require(
        len({row["patient_id"] for row in pa_rows}) == EXPECTED_PA_PATIENTS,
        "PA patient count changed",
    )
    return pa_rows, {
        "files": source_evidence,
        "release": RELEASE_ID,
        "official_box": "https://nihcc.app.box.com/v/ChestXray-NIHCC",
        "source_images": len(rows),
        "source_patients": len({row["patient_id"] for row in rows}),
        "official_train_images": len(train_files),
        "official_test_images": len(test_files),
        "official_patient_overlap": 0,
        "pa_images": len(pa_rows),
        "pa_patients": len({row["patient_id"] for row in pa_rows}),
    }


def select_patients(
    pa_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]], dict[str, Any]]:
    by_patient: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in pa_rows:
        by_patient[int(row["patient_id"])].append(dict(row))

    def is_target(patient_id: int) -> bool:
        return any(row["labels"] & TARGET_LABELS for row in by_patient[patient_id])

    train_patients = sorted(
        patient_id
        for patient_id, patient_rows in by_patient.items()
        if patient_rows[0]["official_source_split"] == "train_val"
    )
    test_patients = sorted(
        patient_id
        for patient_id, patient_rows in by_patient.items()
        if patient_rows[0]["official_source_split"] == "test"
    )
    require(len(train_patients) == EXPECTED_PA_TRAIN_PATIENTS, "PA train patients changed")
    require(len(test_patients) == EXPECTED_PA_TEST_PATIENTS, "PA test patients changed")

    train_target = [patient_id for patient_id in train_patients if is_target(patient_id)]
    train_nontarget = [patient_id for patient_id in train_patients if not is_target(patient_id)]
    test_target = [patient_id for patient_id in test_patients if is_target(patient_id)]
    test_nontarget = [patient_id for patient_id in test_patients if not is_target(patient_id)]
    require(len(train_target) == EXPECTED_TRAIN_TARGET_PATIENTS, "train target count changed")
    require(len(train_nontarget) == EXPECTED_TRAIN_NONTARGET_PATIENTS, "train control count changed")
    require(len(test_target) == EXPECTED_TEST_TARGET_PATIENTS, "test target count changed")
    require(len(test_nontarget) == EXPECTED_TEST_NONTARGET_PATIENTS, "test control count changed")

    train_nontarget_selected = sorted(
        train_nontarget, key=lambda patient_id: tagged_hash("train-control", patient_id)
    )[: len(train_target)]
    test_nontarget_selected = sorted(
        test_nontarget, key=lambda patient_id: tagged_hash("test-control", patient_id)
    )[: len(test_target)]

    patient_records: dict[int, dict[str, Any]] = {}
    for stratum, patient_ids in (
        ("target", train_target),
        ("control", train_nontarget_selected),
    ):
        ranked = sorted(
            patient_ids,
            key=lambda patient_id: tagged_hash(f"partition-{stratum}", patient_id),
        )
        private_end = round_ratio(len(ranked), 70)
        development_end = round_ratio(len(ranked), 85)
        assignments = (
            ("private_train", ranked[:private_end]),
            ("public_development", ranked[private_end:development_end]),
            ("privacy_attack_holdout", ranked[development_end:]),
        )
        for partition, assigned in assignments:
            for patient_id in assigned:
                patient_records[patient_id] = {
                    "patient_id": patient_id,
                    "partition": partition,
                    "official_source_split": "train_val",
                    "target_patient": int(is_target(patient_id)),
                    "pa_image_count": len(by_patient[patient_id]),
                    "cohort_key_sha256": tagged_hash(
                        "train-target-all" if stratum == "target" else "train-control",
                        patient_id,
                    ),
                    "partition_key_sha256": tagged_hash(f"partition-{stratum}", patient_id),
                }

    for stratum, patient_ids in (
        ("target", test_target),
        ("nontarget", test_nontarget_selected),
    ):
        for patient_id in patient_ids:
            patient_records[patient_id] = {
                "patient_id": patient_id,
                "partition": "final_test",
                "official_source_split": "test",
                "target_patient": int(is_target(patient_id)),
                "pa_image_count": len(by_patient[patient_id]),
                "cohort_key_sha256": tagged_hash(
                    "test-target-all" if stratum == "target" else "test-control",
                    patient_id,
                ),
                "partition_key_sha256": "OFFICIAL_TEST",
            }

    partition_counts = Counter(record["partition"] for record in patient_records.values())
    require(dict(partition_counts) == EXPECTED_PARTITION_PATIENTS, "partition patient counts changed")
    for partition, expected in EXPECTED_PARTITION_PATIENTS.items():
        records = [record for record in patient_records.values() if record["partition"] == partition]
        require(len(records) == expected, f"partition count changed: {partition}")
        require(
            sum(record["target_patient"] for record in records) * 2 == expected,
            f"target/control patient balance changed: {partition}",
        )

    selection_evidence = {
        "available_pa_patients": {
            "train_target": len(train_target),
            "train_nontarget": len(train_nontarget),
            "test_target": len(test_target),
            "test_nontarget": len(test_nontarget),
        },
        "selected_patients": dict(sorted(partition_counts.items(), key=lambda item: PARTITION_ORDER[item[0]])),
        "target_patient_definition": (
            "At least one PA image contains one or more primary target labels."
        ),
        "prevalence_boundary": (
            "Each partition is target-patient enriched 1:1 with non-target patients; "
            "it is not representative of hospital prevalence."
        ),
    }
    return patient_records, by_patient, selection_evidence


def materialize_images(
    patient_records: Mapping[int, Mapping[str, Any]],
    by_patient: Mapping[int, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    patient_rows = sorted(
        (dict(record) for record in patient_records.values()),
        key=lambda row: (PARTITION_ORDER[str(row["partition"])], int(row["patient_id"])),
    )
    image_rows: list[dict[str, Any]] = []
    for patient in patient_rows:
        patient_id = int(patient["patient_id"])
        ranked = sorted(
            by_patient[patient_id],
            key=lambda row: (tagged_hash("image-cap", row["image_id"]), row["image_id"]),
        )
        for cap_rank, source in enumerate(ranked, start=1):
            groups = sorted(
                name for name, labels in PRIMARY_GROUPS.items() if source["labels"] & labels
            )
            image_rows.append(
                {
                    "image_id": source["image_id"],
                    "patient_id": patient_id,
                    "partition": patient["partition"],
                    "official_source_split": source["official_source_split"],
                    "view": source["view"],
                    "followup_no": source["followup_no"],
                    "patient_age": source["patient_age"],
                    "patient_sex": source["patient_sex"],
                    "finding_labels": source["finding_labels"],
                    "primary_groups": "|".join(groups),
                    "target_patient": patient["target_patient"],
                    "patient_pa_image_count": patient["pa_image_count"],
                    "cap_rank": cap_rank,
                    "cap_key_sha256": tagged_hash("image-cap", source["image_id"]),
                    "selected_k2": int(cap_rank <= 2),
                    "selected_k5": int(cap_rank <= 5),
                    "selected_k10": int(cap_rank <= 10),
                    "image_filename": source["image_id"],
                }
            )
    image_rows.sort(
        key=lambda row: (
            PARTITION_ORDER[str(row["partition"])],
            int(row["patient_id"]),
            int(row["cap_rank"]),
            str(row["image_id"]),
        )
    )
    require(len({row["image_id"] for row in image_rows}) == len(image_rows), "duplicate image row")
    return patient_rows, image_rows


def materialize_official_pa_test_census(
    pa_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the full PA official-test population as a distribution sensitivity set.

    The enriched final-test subset is useful for finding-stratified method evaluation,
    but it cannot support prevalence-like claims.  This census is never a training,
    tuning, or attack-threshold-selection input, and the two overlapping test views
    must not be analyzed as independent samples.
    """
    test_rows = [row for row in pa_rows if row["official_source_split"] == "test"]
    by_patient: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in test_rows:
        by_patient[int(row["patient_id"])].append(row)
    target_patients = {
        patient_id
        for patient_id, rows in by_patient.items()
        if any(row["labels"] & TARGET_LABELS for row in rows)
    }
    census_rows: list[dict[str, Any]] = []
    for source in test_rows:
        patient_id = int(source["patient_id"])
        groups = sorted(
            name for name, labels in PRIMARY_GROUPS.items() if source["labels"] & labels
        )
        census_rows.append(
            {
                "image_id": source["image_id"],
                "patient_id": patient_id,
                "official_source_split": source["official_source_split"],
                "view": source["view"],
                "followup_no": source["followup_no"],
                "patient_age": source["patient_age"],
                "patient_sex": source["patient_sex"],
                "finding_labels": source["finding_labels"],
                "primary_groups": "|".join(groups),
                "target_patient": int(patient_id in target_patients),
                "patient_pa_image_count": len(by_patient[patient_id]),
                "image_filename": source["image_id"],
            }
        )
    census_rows.sort(
        key=lambda row: (int(row["patient_id"]), int(row["followup_no"]), str(row["image_id"]))
    )
    require(len(census_rows) == EXPECTED_PA_TEST_IMAGES, "PA official-test image count changed")
    require(
        len({int(row["patient_id"]) for row in census_rows}) == EXPECTED_PA_TEST_PATIENTS,
        "PA official-test census patient count changed",
    )
    require(
        len({str(row["image_id"]) for row in census_rows}) == len(census_rows),
        "duplicate PA official-test census image",
    )
    return census_rows


def group_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    label_counts: Counter[str] = Counter()
    for row in rows:
        label_counts.update(str(row["finding_labels"]).split("|"))
    return {
        "patients": len({int(row["patient_id"]) for row in rows}),
        "images": len(rows),
        "target_patients": len(
            {int(row["patient_id"]) for row in rows if int(row["target_patient"]) == 1}
        ),
        "nontarget_patients": len(
            {int(row["patient_id"]) for row in rows if int(row["target_patient"]) == 0}
        ),
        "no_finding_images": sum(row["finding_labels"] == "No Finding" for row in rows),
        "multi_label_images": sum("|" in str(row["finding_labels"]) for row in rows),
        "labels": dict(sorted(label_counts.items())),
        "primary_groups": {
            name: {
                "images": sum(
                    name in str(row["primary_groups"]).split("|") for row in rows
                ),
                "patients": len(
                    {
                        int(row["patient_id"])
                        for row in rows
                        if name in str(row["primary_groups"]).split("|")
                    }
                ),
            }
            for name in PRIMARY_GROUPS
        },
    }


def build_outputs(source_dir: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    pa_rows, source_evidence = read_metadata(source_dir)
    patient_records, by_patient, selection_evidence = select_patients(pa_rows)
    patient_rows, image_rows = materialize_images(patient_records, by_patient)
    census_rows = materialize_official_pa_test_census(pa_rows)

    outputs: dict[str, bytes] = {
        "patients_private.csv": canonical_csv_bytes(PATIENT_FIELDS, patient_rows),
        "all_selected_pa_images_private.csv": canonical_csv_bytes(IMAGE_FIELDS, image_rows),
        "official_pa_test_census_private.csv": canonical_csv_bytes(CENSUS_FIELDS, census_rows),
    }
    cap_report: dict[str, Any] = {}
    for cap in CAPS:
        field = f"selected_k{cap}"
        cap_rows = [row for row in image_rows if int(row[field]) == 1]
        outputs[f"k{cap}_private.csv"] = canonical_csv_bytes(IMAGE_FIELDS, cap_rows)
        partition_report: dict[str, Any] = {}
        for partition in PARTITION_ORDER:
            partition_rows = [row for row in cap_rows if row["partition"] == partition]
            stats = group_stats(partition_rows)
            require(
                stats["patients"] == EXPECTED_PARTITION_PATIENTS[partition],
                f"patient disappeared under K{cap}: {partition}",
            )
            require(
                stats["images"] == EXPECTED_CAP_IMAGES[cap][partition],
                f"K{cap} image count changed: {partition}",
            )
            partition_report[partition] = stats
        cap_report[str(cap)] = {
            "role": CAP_ROLES[cap],
            "partitions": partition_report,
            "total_images": len(cap_rows),
            "total_patients": len({int(row["patient_id"]) for row in cap_rows}),
        }

    # Nested caps are mandatory and use one label-independent order per patient.
    sets = {
        cap: {
            row["image_id"]
            for row in image_rows
            if int(row[f"selected_k{cap}"]) == 1
        }
        for cap in CAPS
    }
    require(sets[2] <= sets[5] <= sets[10], "contribution-cap manifests are not nested")
    census_ids = {str(row["image_id"]) for row in census_rows}
    k10_final_ids = {
        str(row["image_id"])
        for row in image_rows
        if int(row["selected_k10"]) == 1 and row["partition"] == "final_test"
    }
    require(k10_final_ids <= census_ids, "K10 final test is not contained in PA test census")
    acquisition_union_images = len(sets[10] | census_ids)
    require(acquisition_union_images == 42_423, "K10/census acquisition union changed")

    for partition, minima in K5_MINIMUMS.items():
        stats = cap_report["5"]["partitions"][partition]["primary_groups"]
        for group, (min_images, min_patients) in minima.items():
            require(stats[group]["images"] >= min_images, f"K5 {partition} {group} image gate failed")
            require(
                stats[group]["patients"] >= min_patients,
                f"K5 {partition} {group} patient gate failed",
            )

    commitments = {
        name: {"bytes": len(data), "sha256": sha256_bytes(data)}
        for name, data in sorted(outputs.items())
    }
    manifest_lock_payload = {
        "release": RELEASE_ID,
        "seed": SEED,
        "policies": {
            "view": VIEW_POLICY_ID,
            "cohort": COHORT_POLICY_ID,
            "partition": PARTITION_POLICY_ID,
            "cap": CAP_POLICY_ID,
        },
        "source_sha256": {
            name: evidence["sha256"] for name, evidence in source_evidence["files"].items()
        },
        "output_sha256": {name: item["sha256"] for name, item in commitments.items()},
    }
    report = {
        "schema": "nih-cxr14-pa-patient-manifest-report/v1",
        "status": "PASS",
        "decision_date": "2026-09-02",
        "claim_limit": (
            "Metadata, cohort, patient split, view policy and contribution caps only; "
            "not image acquisition, diagnostic validity, model utility, DP, attack resistance, "
            "PP-Mark robustness or release authorization."
        ),
        "source": source_evidence,
        "policies": manifest_lock_payload["policies"],
        "seed": SEED,
        "finding_scope": {
            "primary_groups": {name: sorted(labels) for name, labels in PRIMARY_GROUPS.items()},
            "label_semantics": (
                "Weak radiographic findings mined from reports; Mass/Nodule is not cancer, "
                "and No Finding is not adjudicated clinical normality."
            ),
            "all_source_labels_retained_in_manifests": True,
            "prompt_conditioning_policy_status": "NOT_YET_FROZEN",
            "groups_are_overlapping_evaluation_strata": True,
        },
        "selection": selection_evidence,
        "caps": cap_report,
        "official_pa_test_census": {
            **group_stats(census_rows),
            "role": "secondary_distribution_sensitivity_only",
            "overlaps_enriched_final_test": True,
            "independent_test_sample": False,
            "may_select_hyperparameters_or_attack_thresholds": False,
            "k10_final_subset": True,
        },
        "nesting": {
            "k2_subset_k5": True,
            "k5_subset_k10": True,
            "k10_final_subset_official_pa_test_census": True,
        },
        "selective_acquisition": {
            "k10_plus_official_pa_test_census_unique_images": acquisition_union_images,
            "additional_census_images_beyond_k10": acquisition_union_images - len(sets[10]),
        },
        "adequacy_gate": {
            "cap": 5,
            "minimums": K5_MINIMUMS,
            "status": "PASS",
        },
        "commitments": commitments,
        "manifest_lock_sha256": sha256_bytes(canonical_json_bytes(manifest_lock_payload)),
        "privacy_boundary": (
            "Patient is the privacy unit; cap selection is label-independent within a patient."
        ),
        "prevalence_boundary": selection_evidence["prevalence_boundary"],
    }
    return outputs, report


def write_or_verify(
    outputs: Mapping[str, bytes],
    report: Mapping[str, Any],
    output_dir: Path,
    report_dir: Path,
    verify: bool,
) -> None:
    all_outputs = dict(outputs)
    all_outputs["report.json"] = canonical_json_bytes(report)
    destinations = {
        name: (report_dir / name if name == "report.json" else output_dir / name)
        for name in all_outputs
    }
    if verify:
        for name, expected in all_outputs.items():
            path = destinations[name]
            require(path.is_file(), f"verification output missing: {path}")
            actual = path.read_bytes()
            require(actual == expected, f"deterministic regeneration mismatch: {path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, data in all_outputs.items():
        destinations[name].write_bytes(data)


def default_paths() -> tuple[Path, Path, Path]:
    working = Path(__file__).resolve().parents[1]
    return (
        working / "_data" / "intake" / "nih_chestxray14_metadata",
        working / "_data" / "derived" / "nih_cxr14_pa_target_enriched_v1",
        working / "_reports" / "nih_cxr14_pa_split_v1_001",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    source, output, report = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=source)
    parser.add_argument("--output-dir", type=Path, default=output)
    parser.add_argument("--report-dir", type=Path, default=report)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        outputs, report = build_outputs(args.source_dir)
        write_or_verify(
            outputs,
            report,
            args.output_dir,
            args.report_dir,
            args.verify,
        )
    except (OSError, ValueError, csv.Error, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    action = "verified" if args.verify else "materialized"
    print(
        json.dumps(
            {
                "status": "PASS",
                "action": action,
                "manifest_lock_sha256": report["manifest_lock_sha256"],
                "k2_images": report["caps"]["2"]["total_images"],
                "k5_images": report["caps"]["5"]["total_images"],
                "k10_images": report["caps"]["10"]["total_images"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
