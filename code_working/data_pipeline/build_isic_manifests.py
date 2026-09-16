#!/usr/bin/env python3
"""Build deterministic patient splits and contribution-cap manifests for ISIC 2020.

This program uses the fixed v2 challenge CSV as the lineage authority.  The
current ISIC API metadata is checked for image/patient agreement but is never
allowed to replace the fixed v2 lesion mapping.  No model result is consulted.

The detailed patient/image mappings are local evidence and must not be
published for a genuinely restricted cohort.  The report directory contains
only aggregate counts, algorithms, seeds, and cryptographic commitments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RELEASE_ID = "SIIM-ISIC-2020-fixed-v2"
DEDUP_POLICY_ID = "official-425-pairs-keep-lexicographically-smaller-v1"
SPLIT_POLICY_ID = "patient-hash-threshold-70-10-10-10-v1"
CAP_POLICY_ID = "within-patient-sha256-rank-v1"

# These human-readable strings are the public seeds.  They were fixed before
# generator training and are not selected from model utility or attack results.
SPLIT_SEED = (
    "doi:10.34970/2020-ds01|unified-private-image-lifecycle|"
    "patient-split|v1"
)
CAP_SEED = (
    "doi:10.34970/2020-ds01|unified-private-image-lifecycle|"
    "within-patient-cap|v1"
)

PARTITIONS: tuple[tuple[str, int], ...] = (
    ("private_train", 70),
    ("public_development", 10),
    ("privacy_attack_holdout", 10),
    ("final_test", 10),
)
PARTITION_ORDER = {name: index for index, (name, _) in enumerate(PARTITIONS)}
CAPS = (2, 5, 10)

EXPECTED_RAW_IMAGES = 33_126
EXPECTED_DUPLICATE_PAIRS = 425
EXPECTED_DEDUP_IMAGES = 32_701
EXPECTED_PATIENTS = 2_056
EXPECTED_UNIQUE_LESIONS = 32_693
EXPECTED_REPEATED_LESIONS = 8
EXPECTED_CAP_COUNTS = {2: 4_112, 5: 9_495, 10: 15_924}

IMAGE_FIELDS = (
    "image_id",
    "patient_id",
    "lesion_id",
    "partition",
    "target",
    "benign_malignant",
    "diagnosis",
    "sex",
    "age_approx",
    "anatom_site",
    "patient_image_count_dedup",
    "patient_target_count_dedup",
    "cap_rank",
    "cap_key_sha256",
    "selected_k2",
    "selected_k5",
    "selected_k10",
    "image_filename",
)
PATIENT_FIELDS = (
    "patient_id",
    "partition",
    "image_count_dedup",
    "target_count_dedup",
    "has_malignant",
    "split_key_sha256",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def tagged_hash(seed: str, *values: str) -> str:
    payload = "\x1f".join((seed, *values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def partition_for_patient(patient_id: str) -> tuple[str, str]:
    digest = tagged_hash(SPLIT_SEED, patient_id)
    # The first 64 bits define a point in [0, 2^64).  Integer thresholds avoid
    # platform-dependent floating-point behavior.
    value = int(digest[:16], 16)
    scale = 1 << 64
    cumulative = 0
    for name, percentage in PARTITIONS:
        cumulative += percentage
        if value < (scale * cumulative) // 100:
            return name, digest
    raise AssertionError("partition thresholds must cover the complete hash range")


def normalize_target(value: str) -> int:
    normalized = value.strip()
    require(normalized in {"0", "1"}, f"invalid target value: {value!r}")
    return int(normalized)


def validate_and_join(
    source_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fixed_path = source_dir / "ground_truth_v2.csv"
    api_path = source_dir / "metadata.csv"
    duplicate_path = source_dir / "duplicates.csv"

    for path in (fixed_path, api_path, duplicate_path):
        require(path.is_file(), f"required source file is missing: {path}")

    fixed_rows = read_csv(fixed_path)
    api_rows = read_csv(api_path)
    duplicate_rows = read_csv(duplicate_path)

    require(
        len(fixed_rows) == EXPECTED_RAW_IMAGES,
        f"fixed v2 row count changed: {len(fixed_rows)}",
    )
    require(
        len(api_rows) == EXPECTED_RAW_IMAGES,
        f"API metadata row count changed: {len(api_rows)}",
    )
    require(
        len(duplicate_rows) == EXPECTED_DUPLICATE_PAIRS,
        f"duplicate-pair count changed: {len(duplicate_rows)}",
    )

    fixed_by_id: dict[str, dict[str, str]] = {}
    for row in fixed_rows:
        image_id = row["image_name"].strip()
        require(image_id and image_id not in fixed_by_id, f"duplicate fixed id: {image_id}")
        require(row["patient_id"].strip(), f"missing patient_id: {image_id}")
        require(row["lesion_id"].strip(), f"missing lesion_id: {image_id}")
        fixed_by_id[image_id] = row

    api_by_id: dict[str, dict[str, str]] = {}
    for row in api_rows:
        image_id = row["isic_id"].strip()
        require(image_id and image_id not in api_by_id, f"duplicate API id: {image_id}")
        api_by_id[image_id] = row

    require(
        set(fixed_by_id) == set(api_by_id),
        "fixed v2 and API image-id populations differ",
    )

    patient_mismatches = [
        image_id
        for image_id, row in fixed_by_id.items()
        if row["patient_id"].strip() != api_by_id[image_id]["patient_id"].strip()
    ]
    lesion_mismatches = [
        image_id
        for image_id, row in fixed_by_id.items()
        if row["lesion_id"].strip() != api_by_id[image_id]["lesion_id"].strip()
    ]
    require(not patient_mismatches, "fixed v2/API patient IDs differ")

    excluded_ids: set[str] = set()
    duplicate_pair_keys: set[tuple[str, str]] = set()
    for row in duplicate_rows:
        left = row["image_name_1"].strip()
        right = row["image_name_2"].strip()
        require(left in fixed_by_id and right in fixed_by_id, "unknown duplicate image ID")
        require(left != right, f"self duplicate: {left}")
        pair = tuple(sorted((left, right)))
        require(pair not in duplicate_pair_keys, f"repeated duplicate pair: {pair}")
        duplicate_pair_keys.add(pair)
        excluded_ids.add(max(left, right))

    require(
        len(excluded_ids) == EXPECTED_DUPLICATE_PAIRS,
        "duplicate graph does not yield one distinct exclusion per official pair",
    )

    joined: list[dict[str, Any]] = []
    for image_id in sorted(fixed_by_id):
        if image_id in excluded_ids:
            continue
        row = fixed_by_id[image_id]
        target = normalize_target(row["target"])
        joined.append(
            {
                "image_id": image_id,
                "patient_id": row["patient_id"].strip(),
                "lesion_id": row["lesion_id"].strip(),
                "target": target,
                "benign_malignant": row["benign_malignant"].strip(),
                "diagnosis": row["diagnosis"].strip(),
                "sex": row["sex"].strip(),
                "age_approx": row["age_approx"].strip(),
                "anatom_site": row["anatom_site_general_challenge"].strip(),
            }
        )

    require(len(joined) == EXPECTED_DEDUP_IMAGES, "deduplicated image count changed")
    require(
        len({row["patient_id"] for row in joined}) == EXPECTED_PATIENTS,
        "deduplicated patient count changed",
    )
    lesion_to_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        lesion_to_rows[str(row["lesion_id"])].append(row)
    repeated_lesions = {
        lesion_id: rows for lesion_id, rows in lesion_to_rows.items() if len(rows) > 1
    }
    require(
        len(lesion_to_rows) == EXPECTED_UNIQUE_LESIONS,
        f"unique lesion count changed: {len(lesion_to_rows)}",
    )
    require(
        len(repeated_lesions) == EXPECTED_REPEATED_LESIONS,
        f"repeated lesion count changed: {len(repeated_lesions)}",
    )
    require(
        all(len(rows) == 2 for rows in repeated_lesions.values()),
        "a retained lesion has more than two images",
    )
    require(
        all(len({str(row['patient_id']) for row in rows}) == 1 for rows in repeated_lesions.values()),
        "a lesion ID crosses patients",
    )

    source_evidence = {
        "fixed_v2": {
            "file": fixed_path.name,
            "bytes": fixed_path.stat().st_size,
            "sha256": sha256_file(fixed_path),
            "role": "sole lineage/split authority",
        },
        "api_metadata": {
            "file": api_path.name,
            "bytes": api_path.stat().st_size,
            "sha256": sha256_file(api_path),
            "role": "image/patient cross-check and attribution only",
        },
        "official_duplicates": {
            "file": duplicate_path.name,
            "bytes": duplicate_path.stat().st_size,
            "sha256": sha256_file(duplicate_path),
            "role": "deduplication authority",
        },
        "api_vs_fixed": {
            "image_id_mismatches": 0,
            "patient_id_mismatches": len(patient_mismatches),
            "lesion_id_mismatches": len(lesion_mismatches),
            "lesion_mismatch_ids_sha256": sha256_bytes(
                ("\n".join(sorted(lesion_mismatches)) + "\n").encode("utf-8")
            ),
        },
        "retained_lesion_structure": {
            "unique_lesion_ids": len(lesion_to_rows),
            "repeated_lesion_ids": len(repeated_lesions),
            "maximum_images_per_lesion": max(len(rows) for rows in lesion_to_rows.values()),
            "cross_patient_lesion_ids": 0,
            "policy": (
                "Retain these records because they are not in the official 425-pair "
                "duplicate list; patient_id remains the privacy unit."
            ),
        },
    }
    return joined, source_evidence


def materialize_rows(
    joined: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_row in joined:
        by_patient[str(source_row["patient_id"])].append(dict(source_row))

    patient_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []

    for patient_id in sorted(by_patient):
        partition, split_digest = partition_for_patient(patient_id)
        patient_images = by_patient[patient_id]
        target_count = sum(int(row["target"]) for row in patient_images)
        ranked: list[tuple[str, str, dict[str, Any]]] = []
        for row in patient_images:
            cap_digest = tagged_hash(CAP_SEED, patient_id, str(row["image_id"]))
            ranked.append((cap_digest, str(row["image_id"]), row))
        ranked.sort(key=lambda item: (item[0], item[1]))

        patient_rows.append(
            {
                "patient_id": patient_id,
                "partition": partition,
                "image_count_dedup": len(ranked),
                "target_count_dedup": target_count,
                "has_malignant": int(target_count > 0),
                "split_key_sha256": split_digest,
            }
        )

        for rank, (cap_digest, image_id, row) in enumerate(ranked, start=1):
            image_rows.append(
                {
                    **row,
                    "partition": partition,
                    "patient_image_count_dedup": len(ranked),
                    "patient_target_count_dedup": target_count,
                    "cap_rank": rank,
                    "cap_key_sha256": cap_digest,
                    "selected_k2": int(rank <= 2),
                    "selected_k5": int(rank <= 5),
                    "selected_k10": int(rank <= 10),
                    "image_filename": f"{image_id}.jpg",
                }
            )

    patient_rows.sort(
        key=lambda row: (PARTITION_ORDER[str(row["partition"])], str(row["patient_id"]))
    )
    image_rows.sort(
        key=lambda row: (
            PARTITION_ORDER[str(row["partition"])],
            str(row["patient_id"]),
            int(row["cap_rank"]),
            str(row["image_id"]),
        )
    )
    return patient_rows, image_rows


def summarize_subset(
    image_rows: Sequence[Mapping[str, Any]],
    patient_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    patient_ids = {str(row["patient_id"]) for row in image_rows}
    malignant_patients = {
        str(row["patient_id"]) for row in image_rows if int(row["target"]) == 1
    }
    contributions = Counter(str(row["patient_id"]) for row in image_rows)
    counts = sorted(contributions.values())
    patient_lookup = {str(row["patient_id"]): row for row in patient_rows}
    return {
        "images": len(image_rows),
        "patients": len(patient_ids),
        "malignant_images": sum(int(row["target"]) for row in image_rows),
        "malignant_bearing_patients_after_selection": len(malignant_patients),
        "malignant_bearing_patients_before_selection": sum(
            int(patient_lookup[patient_id]["has_malignant"]) for patient_id in patient_ids
        ),
        "contribution": {
            "min": min(counts) if counts else 0,
            "median": statistics.median(counts) if counts else 0,
            "mean": round(statistics.fmean(counts), 6) if counts else 0,
            "max": max(counts) if counts else 0,
        },
    }


def aggregate_summary(
    patient_rows: Sequence[Mapping[str, Any]],
    image_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    subsets: dict[str, Sequence[Mapping[str, Any]]] = {"full": image_rows}
    for cap in CAPS:
        subsets[f"k{cap}"] = [
            row for row in image_rows if int(row[f"selected_k{cap}"]) == 1
        ]

    summary: dict[str, Any] = {
        "full": summarize_subset(image_rows, patient_rows),
        "conditions": {},
    }
    for condition, rows in subsets.items():
        condition_summary: dict[str, Any] = {
            "all_partitions": summarize_subset(rows, patient_rows),
            "partitions": {},
        }
        for partition, _ in PARTITIONS:
            selected = [row for row in rows if row["partition"] == partition]
            condition_summary["partitions"][partition] = summarize_subset(
                selected, patient_rows
            )
        summary["conditions"][condition] = condition_summary
    return summary


def validate_materialization(
    patient_rows: Sequence[Mapping[str, Any]],
    image_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    require(len(patient_rows) == EXPECTED_PATIENTS, "patient manifest count changed")
    require(len(image_rows) == EXPECTED_DEDUP_IMAGES, "image manifest count changed")

    image_ids = [str(row["image_id"]) for row in image_rows]
    require(len(image_ids) == len(set(image_ids)), "image IDs are not unique")

    patient_partitions: dict[str, set[str]] = defaultdict(set)
    for row in image_rows:
        patient_partitions[str(row["patient_id"])].add(str(row["partition"]))
    crossing = [patient for patient, parts in patient_partitions.items() if len(parts) != 1]
    require(not crossing, "a patient crosses partitions")

    for cap, expected in EXPECTED_CAP_COUNTS.items():
        actual = int(summary["conditions"][f"k{cap}"]["all_partitions"]["images"])
        require(actual == expected, f"K{cap} count changed: {actual} != {expected}")
        require(
            int(summary["conditions"][f"k{cap}"]["all_partitions"]["contribution"]["max"])
            <= cap,
            f"K{cap} contribution bound violated",
        )

    # Predeclared metadata-only adequacy gates.  They prevent freezing a split
    # with an unusably small rare-class holdout; they do not select a model or
    # optimize a reported result.
    full_partitions = summary["conditions"]["full"]["partitions"]
    require(full_partitions["private_train"]["patients"] >= 1_350, "train patients < 1350")
    require(
        full_partitions["private_train"]["malignant_bearing_patients_before_selection"]
        >= 275,
        "train malignant-bearing patients < 275",
    )
    for partition in ("public_development", "privacy_attack_holdout", "final_test"):
        require(full_partitions[partition]["patients"] >= 160, f"{partition} patients < 160")
        require(
            full_partitions[partition]["malignant_bearing_patients_before_selection"] >= 25,
            f"{partition} malignant-bearing patients < 25",
        )


def summary_markdown(summary: Mapping[str, Any], lock: Mapping[str, Any]) -> str:
    lines = [
        "# ISIC 2020 patient split and contribution-cap lock",
        "",
        "- Status: **FROZEN_BEFORE_MODEL_TRAINING**",
        f"- Release: `{RELEASE_ID}`",
        f"- Split: `{SPLIT_POLICY_ID}`",
        f"- Cap: `{CAP_POLICY_ID}`",
        "- Detailed patient/image mappings: local-only under `_data/derived/`",
        "",
        "## Aggregate counts",
        "",
        "| Condition | Partition | Patients | Images | Malignant images | Malignant-bearing patients after selection | Max images/patient |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for condition in ("full", "k2", "k5", "k10"):
        for partition, _ in PARTITIONS:
            values = summary["conditions"][condition]["partitions"][partition]
            lines.append(
                "| {condition} | {partition} | {patients} | {images} | "
                "{malignant_images} | {malignant_patients} | {maximum} |".format(
                    condition=condition.upper(),
                    partition=partition,
                    patients=values["patients"],
                    images=values["images"],
                    malignant_images=values["malignant_images"],
                    malignant_patients=values[
                        "malignant_bearing_patients_after_selection"
                    ],
                    maximum=values["contribution"]["max"],
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation lock",
            "",
            "K10 is the controlled main condition because the dissertation tests the effect of "
            "a bounded per-patient contribution on image-level versus patient-level privacy. "
            "It is not a response to insufficient source data. K2 and K5 are unit-sensitivity "
            "conditions. The uncapped population is a separately labelled realism/scale "
            "extension and must not be substituted post hoc to obtain a preferred ordering.",
            "",
            "Generation quality remains a mandatory utility gate, but the primary novelty is the "
            "privacy-unit comparison followed by audit, model-bound receipt, and PP-Mark "
            "provenance on outputs from the exact trained checkpoint.",
            "",
            "## Public commitments",
            "",
        ]
    )
    for item in lock["private_manifest_commitments"]:
        lines.append(
            f"- `{item['file']}`: `{item['sha256']}` ({item['bytes']} bytes)"
        )
    return "\n".join(lines) + "\n"


def build_artifacts(
    source_dir: Path,
) -> tuple[dict[str, bytes], dict[str, Any], dict[str, Any]]:
    joined, source_evidence = validate_and_join(source_dir)
    patient_rows, image_rows = materialize_rows(joined)
    summary = aggregate_summary(patient_rows, image_rows)
    validate_materialization(patient_rows, image_rows, summary)

    rendered: dict[str, bytes] = {
        "patient_split_private.csv": canonical_csv_bytes(PATIENT_FIELDS, patient_rows),
        "deduplicated_images_private.csv": canonical_csv_bytes(IMAGE_FIELDS, image_rows),
    }
    for cap in CAPS:
        selected = [row for row in image_rows if int(row[f"selected_k{cap}"]) == 1]
        rendered[f"k{cap}_private.csv"] = canonical_csv_bytes(IMAGE_FIELDS, selected)

    smoke_ids = {
        path.stem for path in (source_dir / "smoke_images").glob("*.jpg") if path.is_file()
    }
    smoke_rows = [row for row in image_rows if str(row["image_id"]) in smoke_ids]
    rendered["smoke_subset_private.csv"] = canonical_csv_bytes(IMAGE_FIELDS, smoke_rows)

    commitments = [
        {
            "file": name,
            "bytes": len(data),
            "sha256": sha256_bytes(data),
        }
        for name, data in sorted(rendered.items())
    ]
    lock: dict[str, Any] = {
        "schema": "isic-patient-split-lock/v1",
        "status": "FROZEN_BEFORE_MODEL_TRAINING",
        "frozen_date": "2026-09-02",
        "release_id": RELEASE_ID,
        "dedup_policy": {
            "id": DEDUP_POLICY_ID,
            "rule": "For each official pair retain min(image_id), exclude max(image_id).",
        },
        "split_policy": {
            "id": SPLIT_POLICY_ID,
            "seed_utf8": SPLIT_SEED,
            "hash": "SHA-256(seed US patient_id), first 64 bits as unsigned integer",
            "threshold_percentages": {name: percent for name, percent in PARTITIONS},
            "label_independent": True,
        },
        "cap_policy": {
            "id": CAP_POLICY_ID,
            "seed_utf8": CAP_SEED,
            "hash": "SHA-256(seed US patient_id US image_id), rank ascending",
            "caps": list(CAPS),
            "label_independent": True,
        },
        "source_evidence": source_evidence,
        "private_manifest_commitments": commitments,
        "disclosure": {
            "private_manifests": "LOCAL_ONLY",
            "public_report": "aggregate counts and cryptographic commitments only",
        },
    }
    return rendered, summary, lock


def write_or_verify(
    private_dir: Path,
    report_dir: Path,
    rendered: Mapping[str, bytes],
    summary: Mapping[str, Any],
    lock: Mapping[str, Any],
    verify: bool,
) -> None:
    report_files = {
        "manifest_summary.json": canonical_json_bytes(summary),
        "manifest_lock.json": canonical_json_bytes(lock),
        "manifest_summary.md": summary_markdown(summary, lock).encode("utf-8"),
    }
    private_files = {**rendered, "manifest_lock.json": report_files["manifest_lock.json"]}

    if verify:
        failures: list[str] = []
        for directory, files in ((private_dir, private_files), (report_dir, report_files)):
            for name, expected in files.items():
                path = directory / name
                if not path.is_file():
                    failures.append(f"missing: {path}")
                elif path.read_bytes() != expected:
                    failures.append(f"content mismatch: {path}")
        require(not failures, "verification failed:\n" + "\n".join(failures))
        return

    private_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, data in private_files.items():
        (private_dir / name).write_bytes(data)
    for name, data in report_files.items():
        (report_dir / name).write_bytes(data)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    working_root = script_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=working_root / "_data" / "intake" / "isic2020_collection70",
    )
    parser.add_argument(
        "--private-dir",
        type=Path,
        default=working_root / "_data" / "derived" / "isic2020_v2_split_v1",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=working_root / "_reports" / "isic2020_split_v1_001",
    )
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rendered, summary, lock = build_artifacts(args.source_dir.resolve())
        write_or_verify(
            args.private_dir.resolve(),
            args.report_dir.resolve(),
            rendered,
            summary,
            lock,
            args.verify,
        )
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    mode = "VERIFY" if args.verify else "BUILD"
    k10 = summary["conditions"]["k10"]["partitions"]
    print(f"{mode}: PASS")
    for partition, _ in PARTITIONS:
        row = k10[partition]
        print(
            f"K10 {partition}: {row['patients']} patients, {row['images']} images, "
            f"{row['malignant_images']} malignant images"
        )
    print("lock_sha256=" + sha256_bytes(canonical_json_bytes(lock)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
