#!/usr/bin/env python3
"""Download one exact NIH archive and validate a deterministic official-PNG sample.

This is a source/interface smoke gate, not the full selective acquisition.  The
archive is byte/SHA-1 locked to the NIH Box file record, tar paths are handled
without ``extractall``, and only a metadata-selected sample from the frozen
K10-plus-official-PA-test-census union is materialized.  Detailed image/patient
mappings stay local under ``_data``; the report is aggregate only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import statistics
import sys
import tarfile
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

from PIL import Image, ImageStat


CATALOG_SHA256 = "3CD87825C2F7B604FF1E599A988A68D7D4B7C27AB8DB984B94F3139837572B41"
K10_MANIFEST_SHA256 = "2D749FB7B70823114A69FD55921277B0D0FF2F9AEC8FECD239159C553F3A0C06"
CENSUS_MANIFEST_SHA256 = "7896B9FF931040CEC319EF755BADF098962B008EBDD6E531B5EEE33AD7AA74FE"
SOURCE_METADATA_SHA256 = "C69A6DACA3549AF707CA9CACBDF9F9A7B6A9188E8C61157DF653520BB72D8EB1"
EXPECTED_UNION_IMAGES = 42_423
EXPECTED_K10_IMAGES = 38_492
EXPECTED_CENSUS_IMAGES = 11_096
SMOKE_SEED = "unified-private-image-assurance|nih-cxr14-official-png-smoke-v1|2026-09-02"
PARTITIONS = (
    "private_train",
    "public_development",
    "privacy_attack_holdout",
    "final_test",
)
PRIMARY_GROUPS = (
    "pneumothorax",
    "pneumonia_or_consolidation",
    "pleural_effusion",
    "mass_or_nodule",
)
INVENTORY_FIELDS = (
    "image_id",
    "patient_id",
    "partition",
    "target_patient",
    "finding_labels",
    "primary_groups",
    "in_k10",
    "in_official_pa_test_census",
    "selection_bucket",
    "archive_name",
    "archive_member",
    "bytes",
    "sha256",
    "width",
    "height",
    "mode",
    "format",
    "pixel_min",
    "pixel_max",
    "pixel_mean",
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


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def canonical_csv_bytes(fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    return buffer.getvalue().encode("utf-8")


def tagged_hash(tag: str, value: str) -> str:
    return hashlib.sha256(f"{SMOKE_SEED}|{tag}|{value}".encode("utf-8")).hexdigest().upper()


def default_paths() -> dict[str, Path]:
    working = Path(__file__).resolve().parents[1]
    intake = working / "_data" / "intake" / "nih_chestxray14_metadata"
    raw = working / "_data" / "raw" / "nih_cxr14_pa_k10_plus_census_v1"
    return {
        "catalog": intake / "official_image_archives_v1.json",
        "metadata": intake / "Data_Entry_2017_v2020.csv",
        "k10": working / "_data" / "derived" / "nih_cxr14_pa_target_enriched_v1" / "k10_private.csv",
        "census": working
        / "_data"
        / "derived"
        / "nih_cxr14_pa_target_enriched_v1"
        / "official_pa_test_census_private.csv",
        "staging": working / "_data" / "staging" / "nih_cxr14_official_archives_v1",
        "images": raw / "images",
        "inventory": raw / "smoke_inventory_private.csv",
        "report": working / "_reports" / "nih_cxr14_official_png_smoke_v1_001" / "report.json",
    }


def load_csv_locked(path: Path, expected_sha256: str, expected_rows: int) -> list[dict[str, str]]:
    require(path.is_file(), f"required manifest missing: {path}")
    require(digest_file(path, "sha256") == expected_sha256, f"manifest lock mismatch: {path.name}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == expected_rows, f"manifest row count changed: {path.name}")
    return rows


def load_union(k10_path: Path, census_path: Path) -> dict[str, dict[str, Any]]:
    k10_rows = load_csv_locked(k10_path, K10_MANIFEST_SHA256, EXPECTED_K10_IMAGES)
    census_rows = load_csv_locked(census_path, CENSUS_MANIFEST_SHA256, EXPECTED_CENSUS_IMAGES)
    union: dict[str, dict[str, Any]] = {}
    for row in k10_rows:
        image_id = row["image_id"].strip()
        require(image_id.endswith(".png"), f"unexpected K10 image ID: {image_id}")
        require(row["view"].strip() == "PA", f"non-PA K10 row: {image_id}")
        require(row["selected_k10"].strip() == "1", f"unselected K10 row: {image_id}")
        require(image_id not in union, f"duplicate K10 image: {image_id}")
        union[image_id] = {
            "image_id": image_id,
            "patient_id": row["patient_id"].strip(),
            "partition": row["partition"].strip(),
            "target_patient": int(row["target_patient"]),
            "finding_labels": row["finding_labels"].strip(),
            "primary_groups": row["primary_groups"].strip(),
            "in_k10": 1,
            "in_official_pa_test_census": 0,
        }
    for row in census_rows:
        image_id = row["image_id"].strip()
        require(image_id.endswith(".png"), f"unexpected census image ID: {image_id}")
        require(row["view"].strip() == "PA", f"non-PA census row: {image_id}")
        if image_id in union:
            record = union[image_id]
            require(record["patient_id"] == row["patient_id"].strip(), f"patient mismatch: {image_id}")
            require(record["finding_labels"] == row["finding_labels"].strip(), f"label mismatch: {image_id}")
            record["in_official_pa_test_census"] = 1
        else:
            union[image_id] = {
                "image_id": image_id,
                "patient_id": row["patient_id"].strip(),
                "partition": "official_pa_test_census_only",
                "target_patient": int(row["target_patient"]),
                "finding_labels": row["finding_labels"].strip(),
                "primary_groups": row["primary_groups"].strip(),
                "in_k10": 0,
                "in_official_pa_test_census": 1,
            }
    require(len(union) == EXPECTED_UNION_IMAGES, f"K10/census union changed: {len(union)}")
    return union


def load_source_image_ids(metadata_path: Path) -> set[str]:
    require(metadata_path.is_file(), f"source metadata missing: {metadata_path}")
    require(
        digest_file(metadata_path, "sha256") == SOURCE_METADATA_SHA256,
        "source metadata SHA-256 changed",
    )
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    image_ids = {row["Image Index"].strip() for row in rows}
    require(len(rows) == 112_120 and len(image_ids) == 112_120, "source image population changed")
    return image_ids


def load_catalog(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"archive catalog missing: {path}")
    require(digest_file(path, "sha256") == CATALOG_SHA256, "archive catalog SHA-256 changed")
    catalog = json.loads(path.read_text(encoding="utf-8"))
    archives = catalog.get("archives")
    require(isinstance(archives, list) and len(archives) == 12, "archive catalog count changed")
    require(sum(int(item["bytes"]) for item in archives) == 45_079_862_784, "archive byte sum changed")
    require(len({item["name"] for item in archives}) == 12, "duplicate archive name")
    return catalog


def archive_download_url(catalog: Mapping[str, Any], archive: Mapping[str, Any]) -> str:
    return str(catalog["download_url_template"]).format(
        shared_name=catalog["shared_name"], box_file_id=archive["box_file_id"]
    )


def download_archive(
    catalog: Mapping[str, Any],
    archive: Mapping[str, Any],
    staging_dir: Path,
    retries: int,
    timeout: float,
    verify_only: bool,
) -> tuple[Path, str, str]:
    staging_dir.mkdir(parents=True, exist_ok=True)
    destination = staging_dir / str(archive["name"])
    partial = staging_dir / f"{archive['name']}.part"
    expected_bytes = int(archive["bytes"])
    expected_sha1 = str(archive["sha1"]).upper()

    def verify(path: Path) -> tuple[str, str]:
        require(path.stat().st_size == expected_bytes, f"archive size mismatch: {path}")
        actual_sha1 = digest_file(path, "sha1")
        require(actual_sha1 == expected_sha1, f"archive SHA-1 mismatch: {actual_sha1}")
        return actual_sha1, digest_file(path, "sha256")

    if destination.is_file():
        sha1, sha256 = verify(destination)
        return destination, sha1, sha256
    if verify_only:
        raise FileNotFoundError(f"verified archive missing: {destination}")
    if partial.exists() and partial.stat().st_size > expected_bytes:
        raise ValueError(f"partial archive exceeds expected size: {partial}")

    url = archive_download_url(catalog, archive)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        resume_from = partial.stat().st_size if partial.exists() else 0
        headers = {
            "User-Agent": "unified-private-image-dissertation-acquisition/1.0",
            "Accept": "application/octet-stream",
        }
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", response.getcode())
                require(status in {200, 206}, f"unexpected archive HTTP status: {status}")
                append = bool(resume_from and status == 206)
                if append:
                    content_range = response.headers.get("Content-Range") or ""
                    require(
                        content_range.startswith(f"bytes {resume_from}-"),
                        f"invalid resume Content-Range: {content_range}",
                    )
                mode = "ab" if append else "wb"
                written = resume_from if append else 0
                next_notice = ((written // (256 * 1024 * 1024)) + 1) * (256 * 1024 * 1024)
                with partial.open(mode) as handle:
                    while True:
                        block = response.read(8 * 1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
                        written += len(block)
                        if written >= next_notice:
                            print(
                                f"downloaded {written / (1024 ** 3):.3f}/{expected_bytes / (1024 ** 3):.3f} GiB",
                                flush=True,
                            )
                            next_notice += 256 * 1024 * 1024
                    handle.flush()
                    os.fsync(handle.fileno())
            require(partial.stat().st_size == expected_bytes, "archive response ended before expected size")
            sha1, sha256 = verify(partial)
            os.replace(partial, destination)
            return destination, sha1, sha256
        except (OSError, ValueError, urllib.error.URLError, tarfile.TarError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            print(f"download attempt {attempt + 1} failed: {exc}; resuming", flush=True)
            time.sleep(min(2**attempt, 8))
    assert last_error is not None
    raise last_error


def inspect_tar(
    archive_path: Path, source_image_ids: set[str], union: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, tarfile.TarInfo], list[dict[str, Any]]]:
    png_members: dict[str, tarfile.TarInfo] = {}
    regular_files = 0
    non_png_files: list[str] = []
    directories = 0
    other_members = 0
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            require(not path.is_absolute() and ".." not in path.parts, f"unsafe tar path: {member.name}")
            if member.isdir():
                directories += 1
                continue
            if not member.isfile():
                other_members += 1
                continue
            regular_files += 1
            basename = path.name
            if not basename.lower().endswith(".png"):
                non_png_files.append(member.name)
                continue
            require(basename not in png_members, f"duplicate PNG basename in archive: {basename}")
            require(basename in source_image_ids, f"archive PNG absent from exact metadata: {basename}")
            png_members[basename] = member
    candidates = [dict(union[name]) for name in png_members if name in union]
    candidates.sort(key=lambda row: (tagged_hash("candidate", row["image_id"]), row["image_id"]))
    stats = {
        "regular_files": regular_files,
        "png_files": len(png_members),
        "directories": directories,
        "other_members": other_members,
        "non_png_regular_files": len(non_png_files),
        "all_pngs_in_exact_metadata": True,
        "duplicate_png_basenames": 0,
        "union_candidate_images": len(candidates),
    }
    return stats, png_members, candidates


def choose_smoke_sample(candidates: Sequence[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    require(sample_size >= 16, "smoke sample must contain at least 16 images")
    chosen: dict[str, dict[str, Any]] = {}

    def take(bucket: str, predicate: Callable[[Mapping[str, Any]], bool], count: int = 1) -> None:
        eligible = sorted(
            (row for row in candidates if predicate(row) and row["image_id"] not in chosen),
            key=lambda row: (tagged_hash(bucket, row["image_id"]), row["image_id"]),
        )
        require(len(eligible) >= count, f"archive lacks required smoke bucket: {bucket}")
        for row in eligible[:count]:
            selected = dict(row)
            selected["selection_bucket"] = bucket
            chosen[selected["image_id"]] = selected

    for partition in PARTITIONS:
        for target in (0, 1):
            take(
                f"partition:{partition}:target:{target}",
                lambda row, partition=partition, target=target: row["partition"] == partition
                and int(row["target_patient"]) == target,
            )
    for group in PRIMARY_GROUPS:
        take(
            f"primary_group:{group}",
            lambda row, group=group: group in str(row["primary_groups"]).split("|"),
        )
    take("no_finding", lambda row: row["finding_labels"] == "No Finding", count=2)
    take("census_only", lambda row: int(row["in_k10"]) == 0, count=2)

    remaining = sorted(
        (row for row in candidates if row["image_id"] not in chosen),
        key=lambda row: (tagged_hash("fill", row["image_id"]), row["image_id"]),
    )
    require(len(chosen) <= sample_size, "required diversity buckets exceed requested sample size")
    require(len(remaining) >= sample_size - len(chosen), "not enough candidates to fill smoke sample")
    for row in remaining[: sample_size - len(chosen)]:
        selected = dict(row)
        selected["selection_bucket"] = "hash_fill"
        chosen[selected["image_id"]] = selected
    sample = sorted(chosen.values(), key=lambda row: row["image_id"])
    require(len(sample) == sample_size, "smoke sample size changed")
    return sample


def extract_selected(
    archive_path: Path,
    member_map: Mapping[str, tarfile.TarInfo],
    sample: Sequence[Mapping[str, Any]],
    image_dir: Path,
) -> dict[str, dict[str, Any]]:
    image_dir.mkdir(parents=True, exist_ok=True)
    selected = {str(row["image_id"]): row for row in sample}
    results: dict[str, dict[str, Any]] = {}
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            basename = PurePosixPath(member.name).name
            if basename not in selected:
                continue
            require(member.isfile(), f"selected tar member is not regular: {member.name}")
            source = archive.extractfile(member)
            require(source is not None, f"could not read tar member: {member.name}")
            destination = image_dir / basename
            partial = image_dir / f"{basename}.part"
            digest = hashlib.sha256()
            size = 0
            with partial.open("wb") as handle:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    handle.write(block)
                    digest.update(block)
                    size += len(block)
                handle.flush()
                os.fsync(handle.fileno())
            require(size == member.size, f"tar member size mismatch: {basename}")
            if destination.exists():
                require(destination.stat().st_size == size, f"existing PNG size mismatch: {basename}")
                require(
                    digest_file(destination, "sha256") == digest.hexdigest().upper(),
                    f"existing PNG content mismatch: {basename}",
                )
                partial.unlink()
            else:
                os.replace(partial, destination)
            results[basename] = {
                "archive_member": member.name,
                "bytes": size,
                "sha256": digest.hexdigest().upper(),
            }
    require(set(results) == set(selected), "not every selected PNG was extracted")
    require(not list(image_dir.glob("*.part")), "partial PNG remains after extraction")
    return results


def inspect_png(path: Path) -> dict[str, Any]:
    require(path.stat().st_size > 0, f"empty PNG: {path}")
    with Image.open(path) as image:
        require((image.format or "").upper() == "PNG", f"unexpected image format: {path}")
        image.load()
        width, height = image.size
        mode = image.mode
        stats = ImageStat.Stat(image)
        extrema = image.getextrema()
    require((width, height) == (1024, 1024), f"unexpected PNG dimensions: {path.name}")
    require(mode == "L", f"unexpected PNG mode: {path.name}: {mode}")
    require(isinstance(extrema, tuple) and len(extrema) == 2, f"unexpected extrema: {path.name}")
    return {
        "width": width,
        "height": height,
        "mode": mode,
        "format": "PNG",
        "pixel_min": int(extrema[0]),
        "pixel_max": int(extrema[1]),
        "pixel_mean": round(float(stats.mean[0]), 6),
    }


def distribution(values: Sequence[float | int]) -> dict[str, float | int]:
    ordered = sorted(values)
    require(bool(ordered), "empty distribution")
    return {
        "min": ordered[0],
        "median": round(float(statistics.median(ordered)), 6),
        "mean": round(float(statistics.fmean(ordered)), 6),
        "max": ordered[-1],
    }


def content_set_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item["image_id"])):
        digest.update(str(row["image_id"]).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(row["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest().upper()


def build_report(
    catalog: Mapping[str, Any],
    archive: Mapping[str, Any],
    archive_sha1: str,
    archive_sha256: str,
    tar_stats: Mapping[str, Any],
    inventory: Sequence[Mapping[str, Any]],
    sample_size: int,
) -> dict[str, Any]:
    partition_counts = Counter(str(row["partition"]) for row in inventory)
    group_counts = {
        group: sum(group in str(row["primary_groups"]).split("|") for row in inventory)
        for group in PRIMARY_GROUPS
    }
    report = {
        "schema": "nih-cxr14-official-png-smoke-report/v1",
        "status": "PASS",
        "decision_date": "2026-09-02",
        "claim_limit": (
            "Exact NIH archive identity, safe tar structure, frozen-manifest linkage and a "
            "deterministic official-PNG decode sample only; not full acquisition, generator "
            "utility, DP, attack resistance, PP-Mark robustness, diagnosis or release approval."
        ),
        "official_source": {
            "folder_url": catalog["official_folder_url"],
            "catalog_sha256": CATALOG_SHA256,
            "archive_name": archive["name"],
            "box_file_id": archive["box_file_id"],
            "box_file_version_id": archive["box_file_version_id"],
            "bytes": int(archive["bytes"]),
            "provider_sha1": str(archive["sha1"]).upper(),
            "verified_sha1": archive_sha1,
            "local_sha256": archive_sha256,
        },
        "frozen_population": {
            "k10_images": EXPECTED_K10_IMAGES,
            "official_pa_test_census_images": EXPECTED_CENSUS_IMAGES,
            "union_images": EXPECTED_UNION_IMAGES,
            "k10_manifest_sha256": K10_MANIFEST_SHA256,
            "census_manifest_sha256": CENSUS_MANIFEST_SHA256,
        },
        "tar": dict(tar_stats),
        "sample": {
            "policy": "metadata-only deterministic diversity buckets plus SHA-256 fill",
            "seed": SMOKE_SEED,
            "images": sample_size,
            "unique_patients": len({str(row["patient_id"]) for row in inventory}),
            "partitions": dict(sorted(partition_counts.items())),
            "target_patient_images": sum(int(row["target_patient"]) for row in inventory),
            "nontarget_patient_images": sum(1 - int(row["target_patient"]) for row in inventory),
            "primary_group_images": group_counts,
            "no_finding_images": sum(row["finding_labels"] == "No Finding" for row in inventory),
            "census_only_images": sum(int(row["in_k10"]) == 0 for row in inventory),
            "bytes": distribution([int(row["bytes"]) for row in inventory]),
            "pixel_mean": distribution([float(row["pixel_mean"]) for row in inventory]),
            "pixel_min": distribution([int(row["pixel_min"]) for row in inventory]),
            "pixel_max": distribution([int(row["pixel_max"]) for row in inventory]),
            "formats": dict(Counter(str(row["format"]) for row in inventory)),
            "modes": dict(Counter(str(row["mode"]) for row in inventory)),
            "dimensions": dict(
                Counter(f"{row['width']}x{row['height']}" for row in inventory)
            ),
            "ordered_content_set_sha256": content_set_sha256(inventory),
        },
        "privacy_boundary": "Detailed filenames, labels, patient IDs and per-image hashes remain local-only.",
        "next_gate": (
            "Only after this PASS: process all 12 locked archives one at a time, materialize the "
            "42,423-image union once, independently verify it, then freeze preprocessing and the "
            "SD 2.1/PP-Mark exact-image interface."
        ),
    }
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-name", default="images_001.tar.gz")
    parser.add_argument("--sample-size", type=int, default=24)
    parser.add_argument("--catalog", type=Path, default=paths["catalog"])
    parser.add_argument("--metadata", type=Path, default=paths["metadata"])
    parser.add_argument("--k10-manifest", type=Path, default=paths["k10"])
    parser.add_argument("--census-manifest", type=Path, default=paths["census"])
    parser.add_argument("--staging-dir", type=Path, default=paths["staging"])
    parser.add_argument("--image-dir", type=Path, default=paths["images"])
    parser.add_argument("--inventory", type=Path, default=paths["inventory"])
    parser.add_argument("--report", type=Path, default=paths["report"])
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        require(args.sample_size > 0, "sample size must be positive")
        catalog = load_catalog(args.catalog)
        matching = [item for item in catalog["archives"] if item["name"] == args.archive_name]
        require(len(matching) == 1, f"archive name absent or duplicated: {args.archive_name}")
        archive = matching[0]
        union = load_union(args.k10_manifest, args.census_manifest)
        source_image_ids = load_source_image_ids(args.metadata)
        archive_path, archive_sha1, archive_sha256 = download_archive(
            catalog,
            archive,
            args.staging_dir,
            args.retries,
            args.timeout,
            args.verify_only,
        )
        print("archive identity verified; scanning tar structure", flush=True)
        tar_stats, member_map, candidates = inspect_tar(archive_path, source_image_ids, union)
        sample = choose_smoke_sample(candidates, args.sample_size)
        extracted = extract_selected(archive_path, member_map, sample, args.image_dir)
        inventory: list[dict[str, Any]] = []
        for row in sample:
            image_id = str(row["image_id"])
            item = {
                **row,
                "archive_name": archive["name"],
                **extracted[image_id],
                **inspect_png(args.image_dir / image_id),
            }
            inventory.append(item)
        inventory.sort(key=lambda row: str(row["image_id"]))
        inventory_bytes = canonical_csv_bytes(INVENTORY_FIELDS, inventory)
        report = build_report(
            catalog,
            archive,
            archive_sha1,
            archive_sha256,
            tar_stats,
            inventory,
            args.sample_size,
        )
        report["local_only_inventory_sha256"] = hashlib.sha256(inventory_bytes).hexdigest().upper()
        report_bytes = canonical_json_bytes(report)
        if args.verify_only:
            require(args.inventory.is_file(), f"inventory missing: {args.inventory}")
            require(args.report.is_file(), f"report missing: {args.report}")
            require(args.inventory.read_bytes() == inventory_bytes, "inventory regeneration mismatch")
            require(args.report.read_bytes() == report_bytes, "report regeneration mismatch")
            action = "verified"
        else:
            args.inventory.parent.mkdir(parents=True, exist_ok=True)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.inventory.write_bytes(inventory_bytes)
            args.report.write_bytes(report_bytes)
            action = "materialized"
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "action": action,
                    "archive": archive["name"],
                    "archive_sha256": archive_sha256,
                    "tar_pngs": tar_stats["png_files"],
                    "union_candidates": tar_stats["union_candidate_images"],
                    "sample_images": args.sample_size,
                    "sample_content_set_sha256": report["sample"]["ordered_content_set_sha256"],
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except (OSError, ValueError, KeyError, csv.Error, json.JSONDecodeError, tarfile.TarError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
