#!/usr/bin/env python3
"""Acquire exactly the frozen NIH PA K10-plus-test-census image union.

Each of the 12 exact NIH archives is downloaded and provider-hash verified one
at a time.  The archive is fully scanned before selected members are extracted,
its selected contribution and complete source-name list are atomically
committed, and the verified temporary archive is then removed by default.  The
script is restartable from per-archive commits and never trains a model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageChops, ImageStat

import smoke_nih_cxr14_official_archive as source_gate


SOURCE_GATE_SCRIPT_SHA256 = "A80652110001FF832EFD7D7E448E80ECE7550FA41F63B538E83141277BAFE9F4"
EXPECTED_ARCHIVES = 12
EXPECTED_SOURCE_IMAGES = 112_120
EXPECTED_UNION_IMAGES = 42_423
DEFAULT_RESERVE_GIB = 8.0
STATE_SCHEMA = "nih-cxr14-selective-archive-commit/v1"
REPORT_SCHEMA = "nih-cxr14-pa-k10-plus-census-acquisition-report/v1"
ARCHIVE_PATTERN = re.compile(r"^images_(\d{3})\.tar\.gz$")

INVENTORY_FIELDS = (
    "image_id",
    "patient_id",
    "partition",
    "target_patient",
    "finding_labels",
    "primary_groups",
    "in_k10",
    "in_official_pa_test_census",
    "archive_name",
    "archive_member",
    "bytes",
    "sha256",
    "width",
    "height",
    "mode",
    "format",
    "grayscale_equivalent",
    "alpha_min",
    "alpha_max",
    "pixel_min",
    "pixel_max",
    "pixel_mean",
)

INTEGER_FIELDS = {
    "target_patient",
    "in_k10",
    "in_official_pa_test_census",
    "bytes",
    "width",
    "height",
    "grayscale_equivalent",
    "alpha_min",
    "alpha_max",
    "pixel_min",
    "pixel_max",
}


def default_paths() -> dict[str, Path]:
    working = Path(__file__).resolve().parents[1]
    smoke_paths = source_gate.default_paths()
    raw = working / "_data" / "raw" / "nih_cxr14_pa_k10_plus_census_v1"
    return {
        "catalog": smoke_paths["catalog"],
        "metadata": smoke_paths["metadata"],
        "k10": smoke_paths["k10"],
        "census": smoke_paths["census"],
        "staging": smoke_paths["staging"],
        "images": raw / "images",
        "state": raw / "_acquisition_state_private",
        "inventory": raw / "content_inventory_private.csv",
        "report": working / "_reports" / "nih_cxr14_union_acquisition_v1_001" / "report.json",
    }


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f"{path.name}.tmp"
    if temporary.exists():
        source_gate.require(temporary.is_file(), f"non-file atomic temporary path: {temporary}")
        temporary.unlink()
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def canonical_name_bytes(names: Sequence[str]) -> bytes:
    source_gate.require(list(names) == sorted(names), "source names are not sorted")
    source_gate.require(len(names) == len(set(names)), "source names are not unique")
    return ("\n".join(names) + "\n").encode("ascii")


def digest_name_set(names: Sequence[str]) -> str:
    return hashlib.sha256(canonical_name_bytes(names)).hexdigest().upper()


def commit_paths(state_dir: Path, archive_name: str) -> dict[str, Path]:
    match = ARCHIVE_PATTERN.fullmatch(archive_name)
    source_gate.require(match is not None, f"unexpected archive name: {archive_name}")
    stem = f"archive_{match.group(1)}"
    return {
        "inventory": state_dir / f"{stem}_selected_inventory_private.csv",
        "names": state_dir / f"{stem}_all_png_names_private.txt",
        "state": state_dir / f"{stem}_commit.json",
    }


def normalize_inventory_row(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for field in INVENTORY_FIELDS:
        source_gate.require(field in row, f"inventory field missing: {field}")
        value = row[field]
        if field in INTEGER_FIELDS:
            normalized[field] = int(value)
        elif field == "pixel_mean":
            normalized[field] = float(value)
        else:
            normalized[field] = str(value)
    return normalized


def inspect_native_png(path: Path) -> dict[str, Any]:
    """Validate an exact PNG while preserving its native channel encoding.

    NIH contains both native ``L`` PNGs and PNGs stored as opaque ``RGBA``
    even though R, G and B are exactly equal.  Raw acquisition records that
    distinction; it does not silently convert the file.  Any actual color or
    non-opaque alpha fails closed.
    """

    source_gate.require(path.stat().st_size > 0, f"empty PNG: {path}")
    with Image.open(path) as image:
        source_gate.require((image.format or "").upper() == "PNG", f"unexpected image format: {path}")
        image.load()
        width, height = image.size
        native_mode = image.mode
        alpha_min = -1
        alpha_max = -1
        if native_mode == "L":
            grayscale = image
        elif native_mode == "RGB":
            red, green, blue = image.split()
            source_gate.require(
                ImageChops.difference(red, green).getbbox() is None
                and ImageChops.difference(red, blue).getbbox() is None,
                f"actual color information in RGB PNG: {path.name}",
            )
            grayscale = red
        elif native_mode in {"LA", "RGBA"}:
            channels = image.split()
            if native_mode == "LA":
                grayscale, alpha = channels
            else:
                red, green, blue, alpha = channels
                source_gate.require(
                    ImageChops.difference(red, green).getbbox() is None
                    and ImageChops.difference(red, blue).getbbox() is None,
                    f"actual color information in RGBA PNG: {path.name}",
                )
                grayscale = red
            alpha_min, alpha_max = alpha.getextrema()
            source_gate.require(
                (alpha_min, alpha_max) == (255, 255),
                f"non-opaque alpha in PNG: {path.name}: {(alpha_min, alpha_max)}",
            )
        else:
            raise ValueError(f"unsupported native PNG mode: {path.name}: {native_mode}")
        stats = ImageStat.Stat(grayscale)
        extrema = grayscale.getextrema()
    source_gate.require((width, height) == (1024, 1024), f"unexpected PNG dimensions: {path.name}")
    source_gate.require(isinstance(extrema, tuple) and len(extrema) == 2, f"unexpected extrema: {path.name}")
    return {
        "width": width,
        "height": height,
        "mode": native_mode,
        "format": "PNG",
        "grayscale_equivalent": 1,
        "alpha_min": int(alpha_min),
        "alpha_max": int(alpha_max),
        "pixel_min": int(extrema[0]),
        "pixel_max": int(extrema[1]),
        "pixel_mean": round(float(stats.mean[0]), 6),
    }


def validate_inventory_row(
    row: Mapping[str, Any],
    union: Mapping[str, Mapping[str, Any]],
    archive_name: str,
) -> None:
    image_id = str(row["image_id"])
    source_gate.require(image_id in union, f"committed image outside frozen union: {image_id}")
    expected = union[image_id]
    for field in (
        "patient_id",
        "partition",
        "target_patient",
        "finding_labels",
        "primary_groups",
        "in_k10",
        "in_official_pa_test_census",
    ):
        source_gate.require(
            str(row[field]) == str(expected[field]),
            f"committed manifest mapping mismatch: {image_id}: {field}",
        )
    source_gate.require(row["archive_name"] == archive_name, f"archive mismatch: {image_id}")
    source_gate.require(row["format"] == "PNG", f"format mismatch: {image_id}")
    source_gate.require(row["mode"] in {"L", "LA", "RGB", "RGBA"}, f"mode mismatch: {image_id}")
    source_gate.require(int(row["grayscale_equivalent"]) == 1, f"non-grayscale image: {image_id}")
    if row["mode"] in {"LA", "RGBA"}:
        source_gate.require(
            int(row["alpha_min"]) == 255 and int(row["alpha_max"]) == 255,
            f"alpha mismatch: {image_id}",
        )
    else:
        source_gate.require(
            int(row["alpha_min"]) == -1 and int(row["alpha_max"]) == -1,
            f"unexpected alpha marker: {image_id}",
        )
    source_gate.require(
        int(row["width"]) == 1024 and int(row["height"]) == 1024,
        f"dimension mismatch: {image_id}",
    )


def verify_materialized_file(row: Mapping[str, Any], image_dir: Path, decode: bool) -> None:
    image_id = str(row["image_id"])
    path = image_dir / image_id
    source_gate.require(path.is_file(), f"committed PNG missing: {path}")
    source_gate.require(path.stat().st_size == int(row["bytes"]), f"PNG byte mismatch: {image_id}")
    source_gate.require(
        source_gate.digest_file(path, "sha256") == str(row["sha256"]),
        f"PNG SHA-256 mismatch: {image_id}",
    )
    if decode:
        observed = inspect_native_png(path)
        for field in (
            "width",
            "height",
            "mode",
            "format",
            "grayscale_equivalent",
            "alpha_min",
            "alpha_max",
            "pixel_min",
            "pixel_max",
        ):
            source_gate.require(str(observed[field]) == str(row[field]), f"PNG {field} mismatch: {image_id}")
        source_gate.require(
            float(observed["pixel_mean"]) == float(row["pixel_mean"]),
            f"PNG pixel mean mismatch: {image_id}",
        )


def archive_identity(archive: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": str(archive["name"]),
        "box_file_id": str(archive["box_file_id"]),
        "box_file_version_id": str(archive["box_file_version_id"]),
        "bytes": int(archive["bytes"]),
        "provider_sha1": str(archive["sha1"]).upper(),
    }


def write_archive_commit(
    state_dir: Path,
    archive: Mapping[str, Any],
    archive_sha1: str,
    archive_sha256: str,
    tar_stats: Mapping[str, Any],
    all_names: Sequence[str],
    inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    archive_name = str(archive["name"])
    paths = commit_paths(state_dir, archive_name)
    ordered_inventory = sorted(
        (normalize_inventory_row(row) for row in inventory), key=lambda row: row["image_id"]
    )
    inventory_bytes = source_gate.canonical_csv_bytes(INVENTORY_FIELDS, ordered_inventory)
    names_bytes = canonical_name_bytes(all_names)
    state = {
        "schema": STATE_SCHEMA,
        "archive": {
            **archive_identity(archive),
            "verified_sha1": archive_sha1,
            "local_sha256": archive_sha256,
        },
        "tar": dict(tar_stats),
        "all_png_names": {
            "count": len(all_names),
            "sha256": hashlib.sha256(names_bytes).hexdigest().upper(),
        },
        "selected": {
            "images": len(ordered_inventory),
            "patients": len({row["patient_id"] for row in ordered_inventory}),
            "bytes": sum(int(row["bytes"]) for row in ordered_inventory),
            "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest().upper(),
            "ordered_content_set_sha256": source_gate.content_set_sha256(ordered_inventory),
        },
    }
    atomic_write_bytes(paths["inventory"], inventory_bytes)
    atomic_write_bytes(paths["names"], names_bytes)
    atomic_write_bytes(paths["state"], source_gate.canonical_json_bytes(state))
    return paths


def load_archive_commit(
    state_dir: Path,
    archive: Mapping[str, Any],
    union: Mapping[str, Mapping[str, Any]],
    image_dir: Path,
    verify_files: bool,
    decode_files: bool = False,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    archive_name = str(archive["name"])
    paths = commit_paths(state_dir, archive_name)
    source_gate.require(paths["state"].is_file(), f"archive commit missing: {archive_name}")
    source_gate.require(paths["inventory"].is_file(), f"archive inventory missing: {archive_name}")
    source_gate.require(paths["names"].is_file(), f"archive name list missing: {archive_name}")
    state = json.loads(paths["state"].read_text(encoding="utf-8"))
    source_gate.require(state.get("schema") == STATE_SCHEMA, f"archive state schema mismatch: {archive_name}")
    expected_identity = archive_identity(archive)
    for field, expected in expected_identity.items():
        state_key = "provider_sha1" if field == "provider_sha1" else field
        source_gate.require(
            str(state["archive"][state_key]) == str(expected),
            f"archive state identity mismatch: {archive_name}: {field}",
        )
    source_gate.require(
        state["archive"]["verified_sha1"] == expected_identity["provider_sha1"],
        f"archive verified SHA-1 mismatch: {archive_name}",
    )

    inventory_bytes = paths["inventory"].read_bytes()
    source_gate.require(
        hashlib.sha256(inventory_bytes).hexdigest().upper() == state["selected"]["inventory_sha256"],
        f"archive inventory commitment mismatch: {archive_name}",
    )
    with io.StringIO(inventory_bytes.decode("utf-8"), newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = [normalize_inventory_row(row) for row in raw_rows]
    source_gate.require(
        source_gate.canonical_csv_bytes(INVENTORY_FIELDS, rows) == inventory_bytes,
        f"archive inventory is not canonical: {archive_name}",
    )
    source_gate.require(len(rows) == int(state["selected"]["images"]), f"selected count mismatch: {archive_name}")
    source_gate.require(
        len({row["image_id"] for row in rows}) == len(rows), f"duplicate selected image: {archive_name}"
    )
    source_gate.require(
        source_gate.content_set_sha256(rows) == state["selected"]["ordered_content_set_sha256"],
        f"selected content commitment mismatch: {archive_name}",
    )
    source_gate.require(
        sum(int(row["bytes"]) for row in rows) == int(state["selected"]["bytes"]),
        f"selected byte sum mismatch: {archive_name}",
    )

    names_bytes = paths["names"].read_bytes()
    source_gate.require(
        hashlib.sha256(names_bytes).hexdigest().upper() == state["all_png_names"]["sha256"],
        f"archive source-name commitment mismatch: {archive_name}",
    )
    names = names_bytes.decode("ascii").splitlines()
    source_gate.require(canonical_name_bytes(names) == names_bytes, f"source names are not canonical: {archive_name}")
    source_gate.require(len(names) == int(state["all_png_names"]["count"]), f"source-name count mismatch: {archive_name}")
    source_gate.require(set(row["image_id"] for row in rows) <= set(names), f"selected name absent from tar list: {archive_name}")

    for row in rows:
        validate_inventory_row(row, union, archive_name)
        if verify_files:
            verify_materialized_file(row, image_dir, decode=decode_files)
    return state, names, rows


def remove_verified_archive(path: Path, staging_dir: Path, catalog_names: set[str]) -> None:
    if not path.exists():
        return
    resolved = path.resolve()
    resolved_staging = staging_dir.resolve()
    source_gate.require(resolved.is_file(), f"archive cleanup target is not a file: {resolved}")
    source_gate.require(resolved.parent == resolved_staging, f"archive cleanup escaped staging: {resolved}")
    source_gate.require(resolved.name in catalog_names, f"archive cleanup name is not catalogued: {resolved.name}")
    resolved.unlink()
    source_gate.require(not resolved.exists(), f"verified temporary archive was not removed: {resolved}")


def require_disk_capacity(staging_dir: Path, archive_bytes: int, reserve_gib: float) -> None:
    staging_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(staging_dir).free
    required = archive_bytes + int(reserve_gib * 1024**3)
    source_gate.require(
        free >= required,
        f"insufficient free space: {free / 1024**3:.3f} GiB free; "
        f"need archive plus reserve {required / 1024**3:.3f} GiB",
    )


def process_archive(
    catalog: Mapping[str, Any],
    archive: Mapping[str, Any],
    union: Mapping[str, Mapping[str, Any]],
    source_image_ids: set[str],
    staging_dir: Path,
    image_dir: Path,
    state_dir: Path,
    retries: int,
    timeout: float,
    reserve_gib: float,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]], Path]:
    require_disk_capacity(staging_dir, int(archive["bytes"]), reserve_gib)
    archive_path, archive_sha1, archive_sha256 = source_gate.download_archive(
        catalog, archive, staging_dir, retries, timeout, verify_only=False
    )
    print(f"{archive['name']}: provider identity verified; scanning tar", flush=True)
    tar_stats, member_map, candidates = source_gate.inspect_tar(archive_path, source_image_ids, union)
    all_names = sorted(member_map)
    selected_ids = sorted(name for name in all_names if name in union)
    source_gate.require(
        len(selected_ids) == int(tar_stats["union_candidate_images"]),
        f"union candidate count mismatch: {archive['name']}",
    )
    source_gate.require(bool(selected_ids), f"archive has no frozen-union contribution: {archive['name']}")
    selected_metadata = [dict(union[image_id]) for image_id in selected_ids]
    extracted = source_gate.extract_selected(archive_path, member_map, selected_metadata, image_dir)
    inventory: list[dict[str, Any]] = []
    for index, metadata in enumerate(selected_metadata, start=1):
        image_id = metadata["image_id"]
        inventory.append(
            {
                **metadata,
                "archive_name": archive["name"],
                **extracted[image_id],
                **inspect_native_png(image_dir / image_id),
            }
        )
        if index % 500 == 0 or index == len(selected_metadata):
            print(f"{archive['name']}: decoded {index}/{len(selected_metadata)} selected PNGs", flush=True)
    write_archive_commit(
        state_dir,
        archive,
        archive_sha1,
        archive_sha256,
        tar_stats,
        all_names,
        inventory,
    )
    state, committed_names, committed_rows = load_archive_commit(
        state_dir,
        archive,
        union,
        image_dir,
        verify_files=True,
        decode_files=False,
    )
    return state, committed_names, committed_rows, archive_path


def state_commit_set_sha256(state_dir: Path, archives: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for archive in archives:
        path = commit_paths(state_dir, str(archive["name"]))["state"]
        digest.update(str(archive["name"]).encode("ascii"))
        digest.update(b"\x00")
        digest.update(source_gate.digest_file(path, "sha256").encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest().upper()


def build_report(
    catalog: Mapping[str, Any],
    states: Mapping[str, Mapping[str, Any]],
    all_source_names: Sequence[str],
    inventory: Sequence[Mapping[str, Any]],
    inventory_sha256: str,
    commit_set_sha256: str,
) -> dict[str, Any]:
    partition_image_counts = Counter(str(row["partition"]) for row in inventory)
    partition_patients: dict[str, set[str]] = {}
    for row in inventory:
        partition_patients.setdefault(str(row["partition"]), set()).add(str(row["patient_id"]))
    group_counts = {
        group: sum(group in str(row["primary_groups"]).split("|") for row in inventory)
        for group in source_gate.PRIMARY_GROUPS
    }
    archive_records = []
    for archive in catalog["archives"]:
        state = states[str(archive["name"])]
        archive_records.append(
            {
                "name": archive["name"],
                "box_file_id": archive["box_file_id"],
                "box_file_version_id": archive["box_file_version_id"],
                "bytes": int(archive["bytes"]),
                "provider_sha1": str(archive["sha1"]).upper(),
                "verified_sha1": state["archive"]["verified_sha1"],
                "local_sha256": state["archive"]["local_sha256"],
                "png_files": int(state["tar"]["png_files"]),
                "selected_images": int(state["selected"]["images"]),
                "selected_bytes": int(state["selected"]["bytes"]),
                "all_png_name_set_sha256": state["all_png_names"]["sha256"],
                "selected_content_set_sha256": state["selected"]["ordered_content_set_sha256"],
            }
        )
    total_bytes = sum(int(row["bytes"]) for row in inventory)
    return {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "decision_date": "2026-09-02",
        "claim_limit": (
            "Exact official-source acquisition, frozen-manifest linkage, PNG integrity and aggregate "
            "image statistics only; not clinical adjudication, generator utility, differential "
            "privacy, attack resistance, PP-Mark robustness or release approval."
        ),
        "source": {
            "official_folder_url": catalog["official_folder_url"],
            "catalog_sha256": source_gate.CATALOG_SHA256,
            "archives": EXPECTED_ARCHIVES,
            "compressed_archive_bytes": int(catalog["total_archive_bytes"]),
            "source_pngs": len(all_source_names),
            "source_png_name_set_sha256": digest_name_set(all_source_names),
            "source_metadata_sha256": source_gate.SOURCE_METADATA_SHA256,
            "complete_metadata_filename_coverage": True,
            "archive_records": archive_records,
        },
        "frozen_population": {
            "k10_images": source_gate.EXPECTED_K10_IMAGES,
            "official_pa_test_census_images": source_gate.EXPECTED_CENSUS_IMAGES,
            "union_images": EXPECTED_UNION_IMAGES,
            "k10_manifest_sha256": source_gate.K10_MANIFEST_SHA256,
            "census_manifest_sha256": source_gate.CENSUS_MANIFEST_SHA256,
        },
        "materialized": {
            "images": len(inventory),
            "unique_patients": len({str(row["patient_id"]) for row in inventory}),
            "total_bytes": total_bytes,
            "total_gib": round(total_bytes / 1024**3, 6),
            "inventory_sha256": inventory_sha256,
            "ordered_content_set_sha256": source_gate.content_set_sha256(inventory),
            "archive_commit_set_sha256": commit_set_sha256,
            "partitions_images": dict(sorted(partition_image_counts.items())),
            "partitions_patients": {
                key: len(value) for key, value in sorted(partition_patients.items())
            },
            "target_patient_images": sum(int(row["target_patient"]) for row in inventory),
            "nontarget_patient_images": sum(1 - int(row["target_patient"]) for row in inventory),
            "primary_group_images": group_counts,
            "no_finding_images": sum(row["finding_labels"] == "No Finding" for row in inventory),
            "bytes": source_gate.distribution([int(row["bytes"]) for row in inventory]),
            "pixel_mean": source_gate.distribution([float(row["pixel_mean"]) for row in inventory]),
            "pixel_min": source_gate.distribution([int(row["pixel_min"]) for row in inventory]),
            "pixel_max": source_gate.distribution([int(row["pixel_max"]) for row in inventory]),
            "formats": dict(Counter(str(row["format"]) for row in inventory)),
            "native_modes": dict(Counter(str(row["mode"]) for row in inventory)),
            "grayscale_equivalent_images": sum(
                int(row["grayscale_equivalent"]) for row in inventory
            ),
            "opaque_alpha_images": sum(
                row["mode"] in {"LA", "RGBA"}
                and int(row["alpha_min"]) == 255
                and int(row["alpha_max"]) == 255
                for row in inventory
            ),
            "dimensions": dict(Counter(f"{row['width']}x{row['height']}" for row in inventory)),
        },
        "operational_boundary": (
            "Each verified temporary archive is removed only after its full source-name list, local "
            "SHA-256 and selected contribution are atomically committed; it remains recoverable by "
            "re-downloading the locked official Box file."
        ),
        "privacy_boundary": "Detailed filenames, labels, patient IDs and per-image hashes remain local-only.",
        "next_gate": (
            "Independently re-hash and decode all 42,423 files, then freeze intensity, crop/padding, "
            "grayscale channel mapping, resolution, prompt policy and the exact SD 2.1/PP-Mark image interface."
        ),
    }


def finalize(
    args: argparse.Namespace,
    catalog: Mapping[str, Any],
    source_image_ids: set[str],
    union: Mapping[str, Mapping[str, Any]],
    committed: Mapping[str, tuple[dict[str, Any], list[str], list[dict[str, Any]]]],
) -> tuple[dict[str, Any], bytes, bytes]:
    source_names: list[str] = []
    inventory: list[dict[str, Any]] = []
    states: dict[str, dict[str, Any]] = {}
    for archive in catalog["archives"]:
        name = str(archive["name"])
        source_gate.require(name in committed, f"archive is not committed: {name}")
        state, names, rows = committed[name]
        states[name] = state
        source_names.extend(names)
        inventory.extend(rows)
    source_gate.require(len(source_names) == EXPECTED_SOURCE_IMAGES, "full source member count changed")
    source_gate.require(len(set(source_names)) == len(source_names), "PNG basename overlaps across archives")
    source_names.sort()
    source_gate.require(set(source_names) == source_image_ids, "archive PNG set differs from exact metadata")

    inventory.sort(key=lambda row: row["image_id"])
    source_gate.require(len(inventory) == EXPECTED_UNION_IMAGES, "materialized union count changed")
    source_gate.require(len({row["image_id"] for row in inventory}) == len(inventory), "duplicate union image")
    source_gate.require(set(row["image_id"] for row in inventory) == set(union), "materialized union coverage mismatch")
    actual_pngs = {path.name for path in args.image_dir.glob("*.png") if path.is_file()}
    source_gate.require(actual_pngs == set(union), "image directory has missing or extra PNG files")
    source_gate.require(not list(args.image_dir.glob("*.part")), "partial PNG remains in image directory")

    inventory_bytes = source_gate.canonical_csv_bytes(INVENTORY_FIELDS, inventory)
    inventory_sha256 = hashlib.sha256(inventory_bytes).hexdigest().upper()
    commit_sha256 = state_commit_set_sha256(args.state_dir, catalog["archives"])
    report = build_report(
        catalog,
        states,
        source_names,
        inventory,
        inventory_sha256,
        commit_sha256,
    )
    return report, inventory_bytes, source_gate.canonical_json_bytes(report)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=paths["catalog"])
    parser.add_argument("--metadata", type=Path, default=paths["metadata"])
    parser.add_argument("--k10-manifest", type=Path, default=paths["k10"])
    parser.add_argument("--census-manifest", type=Path, default=paths["census"])
    parser.add_argument("--staging-dir", type=Path, default=paths["staging"])
    parser.add_argument("--image-dir", type=Path, default=paths["images"])
    parser.add_argument("--state-dir", type=Path, default=paths["state"])
    parser.add_argument("--inventory", type=Path, default=paths["inventory"])
    parser.add_argument("--report", type=Path, default=paths["report"])
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--min-free-reserve-gib", type=float, default=DEFAULT_RESERVE_GIB)
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        source_script = Path(source_gate.__file__).resolve()
        source_gate.require(
            source_gate.digest_file(source_script, "sha256") == SOURCE_GATE_SCRIPT_SHA256,
            "source-smoke dependency SHA-256 changed",
        )
        source_gate.require(args.min_free_reserve_gib >= 0, "free-space reserve cannot be negative")
        catalog = source_gate.load_catalog(args.catalog)
        source_gate.require(len(catalog["archives"]) == EXPECTED_ARCHIVES, "archive count changed")
        union = source_gate.load_union(args.k10_manifest, args.census_manifest)
        source_image_ids = source_gate.load_source_image_ids(args.metadata)
        source_gate.require(len(union) == EXPECTED_UNION_IMAGES, "frozen union changed")
        catalog_names = {str(archive["name"]) for archive in catalog["archives"]}
        args.image_dir.mkdir(parents=True, exist_ok=True)
        args.state_dir.mkdir(parents=True, exist_ok=True)
        args.staging_dir.mkdir(parents=True, exist_ok=True)
        existing_pngs = {path.name for path in args.image_dir.glob("*.png") if path.is_file()}
        source_gate.require(existing_pngs <= set(union), "image directory contains PNG outside frozen union")

        committed: dict[str, tuple[dict[str, Any], list[str], list[dict[str, Any]]]] = {}
        seen_source_names: set[str] = set()
        seen_selected_names: set[str] = set()

        for archive in catalog["archives"]:
            name = str(archive["name"])
            paths = commit_paths(args.state_dir, name)
            archive_path = args.staging_dir / name
            if paths["state"].is_file():
                print(f"{name}: validating existing archive commit", flush=True)
                item = load_archive_commit(
                    args.state_dir,
                    archive,
                    union,
                    args.image_dir,
                    verify_files=True,
                    decode_files=False,
                )
                state, names, rows = item
                source_gate.require(not (set(names) & seen_source_names), f"source overlap before {name}")
                source_gate.require(
                    not ({row["image_id"] for row in rows} & seen_selected_names),
                    f"selected overlap before {name}",
                )
                committed[name] = item
                seen_source_names.update(names)
                seen_selected_names.update(row["image_id"] for row in rows)
                if archive_path.exists() and not args.keep_archives and not args.verify_only:
                    remove_verified_archive(archive_path, args.staging_dir, catalog_names)
                    print(f"{name}: removed already-committed verified temporary archive", flush=True)
                continue
            if args.verify_only:
                raise FileNotFoundError(f"archive commit missing in verify-only mode: {name}")

            print(f"{name}: starting exact download/acquisition", flush=True)
            state, names, rows, archive_path = process_archive(
                catalog,
                archive,
                union,
                source_image_ids,
                args.staging_dir,
                args.image_dir,
                args.state_dir,
                args.retries,
                args.timeout,
                args.min_free_reserve_gib,
            )
            source_gate.require(not (set(names) & seen_source_names), f"source overlap at {name}")
            source_gate.require(
                not ({row["image_id"] for row in rows} & seen_selected_names),
                f"selected overlap at {name}",
            )
            item = (state, names, rows)
            committed[name] = item
            seen_source_names.update(names)
            seen_selected_names.update(row["image_id"] for row in rows)
            if not args.keep_archives:
                remove_verified_archive(archive_path, args.staging_dir, catalog_names)
                print(
                    f"{name}: committed {len(rows)} selected PNGs; removed verified temporary archive",
                    flush=True,
                )
            else:
                print(f"{name}: committed {len(rows)} selected PNGs; retained archive by request", flush=True)

        report, inventory_bytes, report_bytes = finalize(
            args, catalog, source_image_ids, union, committed
        )
        if args.verify_only:
            source_gate.require(args.inventory.is_file(), f"final inventory missing: {args.inventory}")
            source_gate.require(args.report.is_file(), f"final report missing: {args.report}")
            source_gate.require(args.inventory.read_bytes() == inventory_bytes, "final inventory mismatch")
            source_gate.require(args.report.read_bytes() == report_bytes, "final report mismatch")
            action = "verified"
        else:
            atomic_write_bytes(args.inventory, inventory_bytes)
            atomic_write_bytes(args.report, report_bytes)
            action = "materialized"
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "action": action,
                    "archives": EXPECTED_ARCHIVES,
                    "source_pngs": EXPECTED_SOURCE_IMAGES,
                    "materialized_images": report["materialized"]["images"],
                    "materialized_gib": report["materialized"]["total_gib"],
                    "inventory_sha256": report["materialized"]["inventory_sha256"],
                    "content_set_sha256": report["materialized"]["ordered_content_set_sha256"],
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except (OSError, ValueError, KeyError, csv.Error, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
