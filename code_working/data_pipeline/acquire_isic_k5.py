#!/usr/bin/env python3
"""Acquire and verify the frozen K5 SIIM-ISIC 2020 JPEG population.

The detailed inventory stays under ``_data``.  The report directory receives
only aggregate statistics and cryptographic commitments.  Downloads are
restartable: completed JPEGs are verified and reused, while ``.part`` files
resume with HTTP range requests when the server permits it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


SOURCE_URL_TEMPLATE = "https://isic-archive.s3.amazonaws.com/images/{image_id}.jpg"
EXPECTED_MANIFEST_SHA256 = (
    "721CBA21612D23A368086F4A0E8E5DE3173333DB3896A417102739A1DB63DC23"
)
EXPECTED_IMAGES = 9_495
EXPECTED_PARTITION_IMAGES = {
    "private_train": 6_512,
    "public_development": 919,
    "privacy_attack_holdout": 1_093,
    "final_test": 971,
}
EXPECTED_PARTITION_PATIENTS = {
    "private_train": 1_415,
    "public_development": 200,
    "privacy_attack_holdout": 233,
    "final_test": 208,
}
EXPECTED_PATIENTS = 2_056
INVENTORY_FIELDS = (
    "image_id",
    "partition",
    "target",
    "image_filename",
    "bytes",
    "sha256",
    "width",
    "height",
    "format",
)


@dataclass(frozen=True)
class AcquisitionResult:
    image_id: str
    partition: str
    target: int
    image_filename: str
    bytes: int
    sha256: str
    width: int
    height: int
    format: str
    disposition: str
    elapsed_sec: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def percentile_nearest(values: list[int], fraction: float) -> int:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def distribution(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values),
        "median": float(statistics.median(values)),
        "mean": round(float(statistics.fmean(values)), 3),
        "p95": percentile_nearest(values, 0.95),
        "max": max(values),
    }


def validate_manifest(path: Path) -> list[dict[str, str]]:
    if sha256_file(path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("K5 manifest SHA-256 does not match the frozen commitment")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_IMAGES:
        raise ValueError(f"K5 row count changed: {len(rows)}")

    ids = [row["image_id"].strip() for row in rows]
    if len(set(ids)) != EXPECTED_IMAGES:
        raise ValueError("K5 image IDs are not unique")
    if any(row["selected_k5"].strip() != "1" for row in rows):
        raise ValueError("K5 manifest contains an unselected row")
    if any(row["image_filename"].strip() != f"{row['image_id'].strip()}.jpg" for row in rows):
        raise ValueError("K5 filename mapping changed")

    partition_images = Counter(row["partition"].strip() for row in rows)
    if dict(partition_images) != EXPECTED_PARTITION_IMAGES:
        raise ValueError(f"K5 partition image counts changed: {dict(partition_images)}")

    patient_partitions: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        patient_partitions[row["patient_id"].strip()].add(row["partition"].strip())
    if len(patient_partitions) != EXPECTED_PATIENTS:
        raise ValueError(f"K5 patient count changed: {len(patient_partitions)}")
    crossing = [patient for patient, parts in patient_partitions.items() if len(parts) != 1]
    if crossing:
        raise ValueError("patient partition crossing detected")
    partition_patients = Counter(next(iter(parts)) for parts in patient_partitions.values())
    if dict(partition_patients) != EXPECTED_PARTITION_PATIENTS:
        raise ValueError(f"K5 partition patient counts changed: {dict(partition_patients)}")
    return rows


def inspect_jpeg(path: Path) -> tuple[int, str, int, int, str]:
    size = path.stat().st_size
    if size <= 0:
        raise ValueError("empty file")
    digest = sha256_file(path)
    with Image.open(path) as image:
        width, height = image.size
        image_format = image.format or ""
        image.verify()
    if image_format.upper() != "JPEG":
        raise ValueError(f"unexpected image format: {image_format}")
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid dimensions: {width}x{height}")
    return size, digest, width, height, image_format.upper()


def quarantine(path: Path, quarantine_dir: Path) -> None:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = quarantine_dir / f"{path.name}.{timestamp}.corrupt"
    os.replace(path, target)


def download_once(url: str, partial_path: Path, timeout: float) -> None:
    resume_from = partial_path.stat().st_size if partial_path.exists() else 0
    headers = {
        "User-Agent": "unified-private-image-dissertation-acquisition/1.0",
        "Accept": "image/jpeg",
    }
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", response.getcode())
        content_type = (response.headers.get("Content-Type") or "").lower()
        if status not in {200, 206}:
            raise ValueError(f"unexpected HTTP status {status}")
        # Historical ISIC S3 objects are inconsistent: some valid JPEGs are
        # served as binary/application octet-stream.  Accept those generic
        # types but let Pillow's full JPEG verification remain authoritative.
        allowed_content_types = {
            "image/jpeg",
            "image/jpg",
            "application/octet-stream",
            "binary/octet-stream",
        }
        if content_type.split(";", 1)[0].strip() not in allowed_content_types:
            raise ValueError(f"unexpected content type {content_type!r}")
        append = bool(resume_from and status == 206)
        mode = "ab" if append else "wb"
        with partial_path.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
            handle.flush()
            os.fsync(handle.fileno())


def acquire_one(
    row: dict[str, str],
    image_dir: Path,
    quarantine_dir: Path,
    retries: int,
    timeout: float,
    verify_only: bool,
) -> AcquisitionResult:
    started = time.perf_counter()
    image_id = row["image_id"].strip()
    partition = row["partition"].strip()
    target_value = int(row["target"].strip())
    filename = row["image_filename"].strip()
    destination = image_dir / filename
    partial = image_dir / f"{filename}.part"

    if destination.exists():
        try:
            size, digest, width, height, image_format = inspect_jpeg(destination)
            return AcquisitionResult(
                image_id,
                partition,
                target_value,
                filename,
                size,
                digest,
                width,
                height,
                image_format,
                "VERIFIED_EXISTING",
                round(time.perf_counter() - started, 4),
            )
        except Exception:
            if verify_only:
                raise
            quarantine(destination, quarantine_dir)

    if verify_only:
        raise FileNotFoundError(f"missing image: {filename}")

    url = SOURCE_URL_TEMPLATE.format(image_id=image_id)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            download_once(url, partial, timeout)
            size, digest, width, height, image_format = inspect_jpeg(partial)
            os.replace(partial, destination)
            return AcquisitionResult(
                image_id,
                partition,
                target_value,
                filename,
                size,
                digest,
                width,
                height,
                image_format,
                "DOWNLOADED",
                round(time.perf_counter() - started, 4),
            )
        except Exception as exc:
            last_error = exc
            if partial.exists():
                try:
                    inspect_jpeg(partial)
                except Exception:
                    pass
                else:
                    os.replace(partial, destination)
                    size, digest, width, height, image_format = inspect_jpeg(destination)
                    return AcquisitionResult(
                        image_id,
                        partition,
                        target_value,
                        filename,
                        size,
                        digest,
                        width,
                        height,
                        image_format,
                        "DOWNLOADED_AFTER_RESPONSE_ERROR",
                        round(time.perf_counter() - started, 4),
                    )
            if attempt < retries:
                time.sleep(min(16.0, 2.0**attempt))
    assert last_error is not None
    raise RuntimeError(f"{image_id}: {type(last_error).__name__}: {last_error}")


def write_private_inventory(path: Path, rows: Iterable[AcquisitionResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_FIELDS, lineterminator="\n")
        writer.writeheader()
        for result in rows:
            record = asdict(result)
            writer.writerow({field: record[field] for field in INVENTORY_FIELDS})


def content_set_digest(results: list[AcquisitionResult]) -> str:
    digest = hashlib.sha256()
    for result in sorted(results, key=lambda item: item.image_id):
        line = f"{result.image_id}\x1f{result.sha256}\x1f{result.bytes}\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest().upper()


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "_data" / "derived" / "isic2020_v2_split_v1" / "k5_private.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "_data" / "raw" / "isic2020_k5_v1",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=root / "_reports" / "isic2020_k5_acquisition_v1_001",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.workers > 32:
        raise ValueError("workers must be in [1, 32]")
    if args.retries < 0:
        raise ValueError("retries must be non-negative")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")

    manifest = args.manifest.resolve()
    rows = validate_manifest(manifest)
    if args.limit is not None:
        rows = rows[: args.limit]

    output_root = args.output_root.resolve()
    image_dir = output_root / "images"
    quarantine_dir = output_root / "quarantine"
    report_dir = args.report_dir.resolve()
    image_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"ACQUISITION_START images={len(rows)} workers={args.workers} "
        f"verify_only={args.verify_only}",
        flush=True,
    )
    started = time.perf_counter()
    results: list[AcquisitionResult] = []
    failures: list[dict[str, str]] = []
    futures: dict[Future[AcquisitionResult], dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in rows:
            future = pool.submit(
                acquire_one,
                row,
                image_dir,
                quarantine_dir,
                args.retries,
                args.timeout,
                args.verify_only,
            )
            futures[future] = row

        last_progress = time.perf_counter()
        for completed, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append(
                    {
                        "image_id": row["image_id"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            now = time.perf_counter()
            if completed == len(rows) or completed % 100 == 0 or now - last_progress >= 15:
                downloaded = sum(item.disposition.startswith("DOWNLOADED") for item in results)
                existing = sum(item.disposition == "VERIFIED_EXISTING" for item in results)
                elapsed = max(0.001, now - started)
                print(
                    f"ACQUISITION_PROGRESS completed={completed}/{len(rows)} "
                    f"downloaded={downloaded} existing={existing} failures={len(failures)} "
                    f"rate={completed / elapsed:.2f}_images_per_sec",
                    flush=True,
                )
                progress = {
                    "schema": "isic-k5-acquisition-progress/v1",
                    "status": "RUNNING" if completed < len(rows) else "CHECKING",
                    "completed": completed,
                    "expected": len(rows),
                    "downloaded": downloaded,
                    "verified_existing": existing,
                    "failures": len(failures),
                    "elapsed_sec": round(elapsed, 3),
                }
                (report_dir / "acquisition_progress.json").write_text(
                    canonical_json(progress), encoding="utf-8"
                )
                last_progress = now

    results.sort(key=lambda item: item.image_id)
    failures.sort(key=lambda item: item["image_id"])
    failure_path = output_root / "failures_private.json"
    if failures:
        failure_path.write_text(canonical_json(failures), encoding="utf-8")
        print(f"ACQUISITION_INCOMPLETE failures={len(failures)}", file=sys.stderr)
        return 1
    if failure_path.exists():
        failure_path.unlink()

    if len(results) != len(rows):
        raise RuntimeError("result count mismatch")
    inventory_path = output_root / "acquisition_inventory_private.csv"
    write_private_inventory(inventory_path, results)

    selected_ids = {row["image_id"].strip() for row in rows}
    present_jpegs = {path.stem for path in image_dir.glob("*.jpg")}
    missing_ids = sorted(selected_ids - present_jpegs)
    extra_ids = sorted(present_jpegs - selected_ids)
    if missing_ids or (args.limit is None and extra_ids):
        raise RuntimeError(
            f"file-set mismatch: missing={len(missing_ids)}, extra={len(extra_ids)}"
        )

    widths = [item.width for item in results]
    heights = [item.height for item in results]
    sizes = [item.bytes for item in results]
    by_partition: dict[str, dict[str, int]] = {}
    for partition in EXPECTED_PARTITION_IMAGES:
        subset = [item for item in results if item.partition == partition]
        by_partition[partition] = {
            "images": len(subset),
            "bytes": sum(item.bytes for item in subset),
            "malignant_images": sum(item.target for item in subset),
        }

    elapsed = time.perf_counter() - started
    report = {
        "schema": "isic-k5-acquisition-report/v1",
        "status": "PASS" if args.limit is None else "DIAGNOSTIC_LIMITED_PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "release": "SIIM-ISIC 2020 Challenge Training / Collection 70",
            "doi": "10.34970/2020-ds01",
            "url_template": SOURCE_URL_TEMPLATE,
            "license_boundary": "CC BY-NC 4.0 aggregate-release rule",
        },
        "frozen_input": {
            "manifest": manifest.name,
            "manifest_sha256": sha256_file(manifest),
            "manifest_rows": EXPECTED_IMAGES,
            "selected_rows_this_run": len(rows),
        },
        "result": {
            "images": len(results),
            "bytes": sum(sizes),
            "gib": round(sum(sizes) / 1024**3, 6),
            "malignant_images": sum(item.target for item in results),
            "downloaded_this_run": sum(
                item.disposition.startswith("DOWNLOADED") for item in results
            ),
            "verified_existing": sum(
                item.disposition == "VERIFIED_EXISTING" for item in results
            ),
            "failures": 0,
            "elapsed_sec": round(elapsed, 3),
            "content_set_sha256": content_set_digest(results),
            "private_inventory_sha256": sha256_file(inventory_path),
            "private_inventory_disclosure": "LOCAL_ONLY",
            "by_partition": by_partition,
            "file_bytes": distribution(sizes),
            "width_pixels": distribution(widths),
            "height_pixels": distribution(heights),
        },
        "claim_limit": (
            "Acquisition and JPEG integrity only; not preprocessing equivalence, model utility, "
            "privacy, attack resistance, or release authorization."
        ),
    }
    report_path = report_dir / "acquisition_report.json"
    report_path.write_text(canonical_json(report), encoding="utf-8")
    report_hash = sha256_file(report_path)
    (report_dir / "acquisition_progress.json").write_text(
        canonical_json(
            {
                "schema": "isic-k5-acquisition-progress/v1",
                "status": "PASS",
                "completed": len(results),
                "expected": len(rows),
                "failures": 0,
                "report_sha256": report_hash,
            }
        ),
        encoding="utf-8",
    )
    print(f"ACQUISITION_PASS images={len(results)} bytes={sum(sizes)}", flush=True)
    print(f"CONTENT_SET_SHA256={report['result']['content_set_sha256']}", flush=True)
    print(f"REPORT_SHA256={report_hash}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
