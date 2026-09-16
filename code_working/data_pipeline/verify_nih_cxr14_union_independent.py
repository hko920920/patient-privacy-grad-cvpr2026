#!/usr/bin/env python3
"""Independently verify the complete frozen NIH X-ray acquisition.

This verifier intentionally does not import the acquisition or source-smoke
implementation.  It rebuilds the K10-plus-census union from locked CSV inputs,
checks all archive commit fragments and complete source-name coverage, then
re-hashes and decodes every materialized PNG.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from statistics import fmean, median
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageChops, ImageStat


CATALOG_SHA256 = "3CD87825C2F7B604FF1E599A988A68D7D4B7C27AB8DB984B94F3139837572B41"
METADATA_SHA256 = "C69A6DACA3549AF707CA9CACBDF9F9A7B6A9188E8C61157DF653520BB72D8EB1"
K10_SHA256 = "2D749FB7B70823114A69FD55921277B0D0FF2F9AEC8FECD239159C553F3A0C06"
CENSUS_SHA256 = "7896B9FF931040CEC319EF755BADF098962B008EBDD6E531B5EEE33AD7AA74FE"
INVENTORY_SHA256 = "AD34344F962FAC7052114A3486D42D2B48CB5C9F27DDDC70365E07B715F68CC3"
CONTENT_SET_SHA256 = "E983A21B9B8558CE38F1AD5BA7C0F6BC51787CC7CBA739399ACE1976E2FB068E"
SOURCE_NAME_SET_SHA256 = "9E749A0B70E21F2803B792B4E4D4C2D7BF7560E2C4BD75780A5B83904DF79B7A"
EXPECTED_SOURCE = 112_120
EXPECTED_K10 = 38_492
EXPECTED_CENSUS = 11_096
EXPECTED_UNION = 42_423
EXPECTED_ARCHIVES = 12
STATE_SCHEMA = "nih-cxr14-selective-archive-commit/v1"
REPORT_SCHEMA = "nih-cxr14-pa-k10-plus-census-independent-verification/v1"
ARCHIVE_RE = re.compile(r"^images_(\d{3})\.tar\.gz$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f"{path.name}.tmp"
    if temporary.exists():
        require(temporary.is_file(), f"unexpected temporary output: {temporary}")
        temporary.unlink()
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_locked_csv(path: Path, expected_hash: str, expected_rows: int) -> list[dict[str, str]]:
    require(path.is_file(), f"locked CSV missing: {path}")
    require(sha_file(path) == expected_hash, f"locked CSV hash mismatch: {path.name}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == expected_rows, f"locked CSV row count mismatch: {path.name}")
    return rows


def build_source_metadata(path: Path) -> dict[str, dict[str, str]]:
    rows = read_locked_csv(path, METADATA_SHA256, EXPECTED_SOURCE)
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        image_id = row["Image Index"].strip()
        require(image_id not in result, f"duplicate source metadata image: {image_id}")
        result[image_id] = {
            "patient_id": row["Patient ID"].strip(),
            "view": row["View Position"].strip(),
            "finding_labels": row["Finding Labels"].strip(),
        }
    return result


def build_union(k10_path: Path, census_path: Path) -> dict[str, dict[str, str]]:
    k10_rows = read_locked_csv(k10_path, K10_SHA256, EXPECTED_K10)
    census_rows = read_locked_csv(census_path, CENSUS_SHA256, EXPECTED_CENSUS)
    union: dict[str, dict[str, str]] = {}
    for row in k10_rows:
        image_id = row["image_id"].strip()
        require(row["selected_k10"].strip() == "1", f"unselected K10 row: {image_id}")
        require(row["view"].strip() == "PA", f"non-PA K10 row: {image_id}")
        require(row["image_filename"].strip() == image_id, f"K10 filename mismatch: {image_id}")
        require(image_id not in union, f"duplicate K10 image: {image_id}")
        union[image_id] = {
            "patient_id": row["patient_id"].strip(),
            "partition": row["partition"].strip(),
            "target_patient": row["target_patient"].strip(),
            "finding_labels": row["finding_labels"].strip(),
            "primary_groups": row["primary_groups"].strip(),
            "in_k10": "1",
            "in_official_pa_test_census": "0",
        }
    for row in census_rows:
        image_id = row["image_id"].strip()
        require(row["view"].strip() == "PA", f"non-PA census row: {image_id}")
        require(row["image_filename"].strip() == image_id, f"census filename mismatch: {image_id}")
        if image_id in union:
            current = union[image_id]
            for field in ("patient_id", "target_patient", "finding_labels", "primary_groups"):
                require(current[field] == row[field].strip(), f"K10/census mismatch: {image_id}: {field}")
            current["in_official_pa_test_census"] = "1"
        else:
            union[image_id] = {
                "patient_id": row["patient_id"].strip(),
                "partition": "official_pa_test_census_only",
                "target_patient": row["target_patient"].strip(),
                "finding_labels": row["finding_labels"].strip(),
                "primary_groups": row["primary_groups"].strip(),
                "in_k10": "0",
                "in_official_pa_test_census": "1",
            }
    require(len(union) == EXPECTED_UNION, f"independent union count mismatch: {len(union)}")
    return union


def source_name_digest(names: Sequence[str]) -> str:
    require(list(names) == sorted(names), "source names are not sorted")
    payload = ("\n".join(names) + "\n").encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()


def content_set_digest(rows: Iterable[Mapping[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["image_id"]):
        digest.update(row["image_id"].encode("ascii"))
        digest.update(b"\x00")
        digest.update(row["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest().upper()


def distribution(values: Sequence[float | int]) -> dict[str, float | int]:
    ordered = sorted(values)
    require(bool(ordered), "empty distribution")
    return {
        "min": ordered[0],
        "median": round(float(median(ordered)), 6),
        "mean": round(float(fmean(ordered)), 6),
        "max": ordered[-1],
    }


def inspect_png_independent(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        require((image.format or "").upper() == "PNG", f"not PNG: {path.name}")
        image.load()
        require(image.size == (1024, 1024), f"dimension mismatch: {path.name}: {image.size}")
        native_mode = image.mode
        alpha_min = -1
        alpha_max = -1
        if native_mode == "L":
            grayscale = image
        elif native_mode == "RGB":
            red, green, blue = image.split()
            require(
                ImageChops.difference(red, green).getbbox() is None
                and ImageChops.difference(red, blue).getbbox() is None,
                f"actual RGB color information: {path.name}",
            )
            grayscale = red
        elif native_mode == "LA":
            grayscale, alpha = image.split()
            alpha_min, alpha_max = alpha.getextrema()
            require((alpha_min, alpha_max) == (255, 255), f"nonopaque LA alpha: {path.name}")
        elif native_mode == "RGBA":
            red, green, blue, alpha = image.split()
            require(
                ImageChops.difference(red, green).getbbox() is None
                and ImageChops.difference(red, blue).getbbox() is None,
                f"actual RGBA color information: {path.name}",
            )
            alpha_min, alpha_max = alpha.getextrema()
            require((alpha_min, alpha_max) == (255, 255), f"nonopaque RGBA alpha: {path.name}")
            grayscale = red
        else:
            raise ValueError(f"unsupported PNG mode: {path.name}: {native_mode}")
        extrema = grayscale.getextrema()
        statistics = ImageStat.Stat(grayscale)
    return {
        "width": 1024,
        "height": 1024,
        "mode": native_mode,
        "format": "PNG",
        "grayscale_equivalent": 1,
        "alpha_min": int(alpha_min),
        "alpha_max": int(alpha_max),
        "pixel_min": int(extrema[0]),
        "pixel_max": int(extrema[1]),
        "pixel_mean": round(float(statistics.mean[0]), 6),
    }


def load_catalog_and_archive_map(
    catalog_path: Path,
    state_dir: Path,
    source_ids: set[str],
) -> tuple[dict[str, Any], dict[str, str], dict[str, dict[str, str]]]:
    require(sha_file(catalog_path) == CATALOG_SHA256, "catalog hash mismatch")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    archives = catalog.get("archives")
    require(isinstance(archives, list) and len(archives) == EXPECTED_ARCHIVES, "catalog archive count mismatch")
    require(sum(int(item["bytes"]) for item in archives) == 45_079_862_784, "catalog byte sum mismatch")
    image_to_archive: dict[str, str] = {}
    fragment_rows: dict[str, dict[str, str]] = {}
    all_names: list[str] = []
    for archive in archives:
        name = str(archive["name"])
        match = ARCHIVE_RE.fullmatch(name)
        require(match is not None, f"unexpected archive name: {name}")
        prefix = f"archive_{match.group(1)}"
        state_path = state_dir / f"{prefix}_commit.json"
        names_path = state_dir / f"{prefix}_all_png_names_private.txt"
        fragment_path = state_dir / f"{prefix}_selected_inventory_private.csv"
        require(state_path.is_file() and names_path.is_file() and fragment_path.is_file(), f"commit files missing: {name}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        require(state.get("schema") == STATE_SCHEMA, f"state schema mismatch: {name}")
        for field in ("box_file_id", "box_file_version_id", "bytes"):
            require(str(state["archive"][field]) == str(archive[field]), f"state/catalog mismatch: {name}: {field}")
        provider = str(archive["sha1"]).upper()
        require(state["archive"]["provider_sha1"] == provider, f"provider SHA mismatch in state: {name}")
        require(state["archive"]["verified_sha1"] == provider, f"unverified provider SHA in state: {name}")
        require(len(state["archive"]["local_sha256"]) == 64, f"local archive SHA absent: {name}")
        names_bytes = names_path.read_bytes()
        require(sha_file(names_path) == state["all_png_names"]["sha256"], f"name-list hash mismatch: {name}")
        names = names_bytes.decode("ascii").splitlines()
        require(names == sorted(names) and len(names) == len(set(names)), f"noncanonical name list: {name}")
        require(len(names) == int(state["all_png_names"]["count"]), f"name-list count mismatch: {name}")
        for image_id in names:
            require(image_id not in image_to_archive, f"image occurs in multiple archives: {image_id}")
            image_to_archive[image_id] = name
        all_names.extend(names)
        require(sha_file(fragment_path) == state["selected"]["inventory_sha256"], f"fragment hash mismatch: {name}")
        with fragment_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        require(len(rows) == int(state["selected"]["images"]), f"fragment row count mismatch: {name}")
        for row in rows:
            image_id = row["image_id"]
            require(row["archive_name"] == name, f"fragment archive mismatch: {image_id}")
            require(image_id not in fragment_rows, f"duplicate fragment image: {image_id}")
            fragment_rows[image_id] = row
    require(len(all_names) == EXPECTED_SOURCE, f"source-name total mismatch: {len(all_names)}")
    ordered_names = sorted(all_names)
    require(set(ordered_names) == source_ids, "source-name lists differ from exact metadata")
    require(source_name_digest(ordered_names) == SOURCE_NAME_SET_SHA256, "source-name-set digest mismatch")
    return catalog, image_to_archive, fragment_rows


def default_paths() -> dict[str, Path]:
    working = Path(__file__).resolve().parents[1]
    intake = working / "_data" / "intake" / "nih_chestxray14_metadata"
    derived = working / "_data" / "derived" / "nih_cxr14_pa_target_enriched_v1"
    raw = working / "_data" / "raw" / "nih_cxr14_pa_k10_plus_census_v1"
    return {
        "catalog": intake / "official_image_archives_v1.json",
        "metadata": intake / "Data_Entry_2017_v2020.csv",
        "k10": derived / "k10_private.csv",
        "census": derived / "official_pa_test_census_private.csv",
        "inventory": raw / "content_inventory_private.csv",
        "images": raw / "images",
        "state": raw / "_acquisition_state_private",
        "report": working / "_reports" / "nih_cxr14_union_verification_v1_001" / "report.json",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=paths["catalog"])
    parser.add_argument("--metadata", type=Path, default=paths["metadata"])
    parser.add_argument("--k10-manifest", type=Path, default=paths["k10"])
    parser.add_argument("--census-manifest", type=Path, default=paths["census"])
    parser.add_argument("--inventory", type=Path, default=paths["inventory"])
    parser.add_argument("--image-dir", type=Path, default=paths["images"])
    parser.add_argument("--state-dir", type=Path, default=paths["state"])
    parser.add_argument("--report", type=Path, default=paths["report"])
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        source_metadata = build_source_metadata(args.metadata)
        union = build_union(args.k10_manifest, args.census_manifest)
        for image_id, expected in union.items():
            require(image_id in source_metadata, f"union image absent from source metadata: {image_id}")
            source = source_metadata[image_id]
            require(source["view"] == "PA", f"union source is non-PA: {image_id}")
            require(source["patient_id"] == expected["patient_id"], f"source patient mismatch: {image_id}")
            source_tokens = source["finding_labels"].split("|")
            require(len(source_tokens) == len(set(source_tokens)), f"duplicate source label: {image_id}")
            canonical_labels = "|".join(sorted(source_tokens))
            require(
                canonical_labels == expected["finding_labels"],
                f"source label-set/canonicalization mismatch: {image_id}",
            )

        catalog, image_to_archive, fragment_rows = load_catalog_and_archive_map(
            args.catalog, args.state_dir, set(source_metadata)
        )
        require(args.inventory.is_file(), f"full inventory missing: {args.inventory}")
        require(sha_file(args.inventory) == INVENTORY_SHA256, "full inventory hash mismatch")
        with args.inventory.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        require(len(rows) == EXPECTED_UNION, f"full inventory row count mismatch: {len(rows)}")
        require([row["image_id"] for row in rows] == sorted(union), "full inventory order/coverage mismatch")
        require(set(fragment_rows) == set(union), "archive fragment coverage mismatch")

        disk_names = {path.name for path in args.image_dir.glob("*.png") if path.is_file()}
        require(disk_names == set(union), "image directory missing/extra set mismatch")
        require(not list(args.image_dir.glob("*.part")), "partial image file remains")

        modes: Counter[str] = Counter()
        formats: Counter[str] = Counter()
        dimensions: Counter[str] = Counter()
        byte_values: list[int] = []
        means: list[float] = []
        minima: list[int] = []
        maxima: list[int] = []
        total_bytes = 0
        patient_ids: set[str] = set()

        for index, row in enumerate(rows, start=1):
            image_id = row["image_id"]
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
                require(row[field] == expected[field], f"inventory/union mismatch: {image_id}: {field}")
            require(fragment_rows[image_id] == row, f"full/fragment inventory mismatch: {image_id}")
            expected_archive = image_to_archive[image_id]
            require(row["archive_name"] == expected_archive, f"archive mapping mismatch: {image_id}")
            member_path = PurePosixPath(row["archive_member"])
            require(not member_path.is_absolute() and ".." not in member_path.parts, f"unsafe committed member: {image_id}")
            require(member_path.name == image_id, f"member basename mismatch: {image_id}")

            path = args.image_dir / image_id
            require(path.is_file(), f"materialized file missing: {image_id}")
            size = path.stat().st_size
            require(size == int(row["bytes"]), f"file byte mismatch: {image_id}")
            actual_sha = sha_file(path)
            require(actual_sha == row["sha256"], f"file SHA-256 mismatch: {image_id}")
            observed = inspect_png_independent(path)
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
                require(str(observed[field]) == row[field], f"decoded {field} mismatch: {image_id}")
            require(float(observed["pixel_mean"]) == float(row["pixel_mean"]), f"pixel mean mismatch: {image_id}")
            modes[observed["mode"]] += 1
            formats[observed["format"]] += 1
            dimensions[f"{observed['width']}x{observed['height']}"] += 1
            byte_values.append(size)
            means.append(float(observed["pixel_mean"]))
            minima.append(int(observed["pixel_min"]))
            maxima.append(int(observed["pixel_max"]))
            total_bytes += size
            patient_ids.add(row["patient_id"])
            if index % 1000 == 0 or index == len(rows):
                print(f"independent verification: {index}/{len(rows)} PNGs", flush=True)

        require(content_set_digest(rows) == CONTENT_SET_SHA256, "ordered content-set digest mismatch")
        report = {
            "schema": REPORT_SCHEMA,
            "status": "PASS",
            "decision_date": "2026-09-02",
            "verifier_sha256": sha_file(Path(__file__).resolve()),
            "input_locks": {
                "catalog_sha256": CATALOG_SHA256,
                "metadata_sha256": METADATA_SHA256,
                "k10_manifest_sha256": K10_SHA256,
                "census_manifest_sha256": CENSUS_SHA256,
                "content_inventory_sha256": INVENTORY_SHA256,
            },
            "checks": {
                "archives": len(catalog["archives"]),
                "source_metadata_images": len(source_metadata),
                "complete_source_name_coverage": len(image_to_archive),
                "union_images": len(rows),
                "unique_patients": len(patient_ids),
                "missing_images": 0,
                "extra_images": 0,
                "duplicate_images": 0,
                "mapping_mismatches": 0,
                "hash_mismatches": 0,
                "decode_failures": 0,
                "actual_color_images": 0,
                "nonopaque_alpha_images": 0,
                "total_bytes": total_bytes,
                "total_gib": round(total_bytes / 1024**3, 6),
                "ordered_content_set_sha256": content_set_digest(rows),
                "source_name_set_sha256": source_name_digest(sorted(image_to_archive)),
                "native_modes": dict(modes),
                "formats": dict(formats),
                "dimensions": dict(dimensions),
                "bytes": distribution(byte_values),
                "pixel_mean": distribution(means),
                "pixel_min": distribution(minima),
                "pixel_max": distribution(maxima),
            },
            "claim_limit": (
                "Independent source/manifest/file/decode verification only; not radiologist "
                "adjudication, model utility, differential privacy, attack resistance, provenance "
                "robustness or release authorization."
            ),
            "next_gate": (
                "Freeze exact raw-to-model preprocessing and run the SD 2.1 VAE/LoRA/PP-Mark "
                "image-interface gate before any generator or DP training."
            ),
        }
        report_bytes = canonical_json(report)
        if args.verify_only:
            require(args.report.is_file(), f"verification report missing: {args.report}")
            require(args.report.read_bytes() == report_bytes, "verification report regeneration mismatch")
            action = "verified"
        else:
            atomic_write(args.report, report_bytes)
            action = "materialized"
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "action": action,
                    "images": len(rows),
                    "patients": len(patient_ids),
                    "native_modes": dict(modes),
                    "content_set_sha256": CONTENT_SET_SHA256,
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
