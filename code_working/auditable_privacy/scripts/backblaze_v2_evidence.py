"""Regenerate exact, diagnostic-only Backblaze mapping evidence.

The full Q1 2025 archive contains tens of millions of drive-day rows.  This
program scans the bound archive once, builds an exact anonymous histogram of
each drive's contiguous-run profile, and computes multiplicities from that
sufficient statistic without materializing tens of millions of window rows.

Two window semantics are reported separately:

* ``ordered_snapshot``: windows span a fixed number of available observations;
  gaps in calendar dates are compressed.  This reproduces the legacy probe's
  semantics and must not be described simply as "N consecutive days".
* ``calendar_contiguous``: windows are restarted at every missing calendar day
  and therefore contain genuinely consecutive daily snapshots.

This artifact has no executed DP mechanism or accountant report and is never a
privacy certificate.  It reports observed incidence and candidate stability
bounds only under the fixed drive/date-slot replacement adjacency stated in
the evidence file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import urllib.request
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


SCHEMA_VERSION = "backblaze_mapping_evidence_v2.0"
BASE_URL = "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data"
DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")


@dataclass(frozen=True)
class WindowPolicy:
    scenario: str
    semantics: str
    window_length: int
    stride: int
    policy: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def quarter_url(year: int, quarter: int) -> str:
    return f"{BASE_URL}/data_Q{quarter}_{year}.zip"


def download_archive(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "unit-aware-dp-audit-v2/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with path.open("wb") as handle:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                handle.write(chunk)


def ensure_archive(
    year: int,
    quarter: int,
    cache_dir: Path,
    download: bool,
) -> Path:
    path = cache_dir / f"data_Q{quarter}_{year}.zip"
    if path.is_file():
        return path
    if not download:
        raise FileNotFoundError(
            f"Missing {path}; use --download to fetch {quarter_url(year, quarter)}"
        )
    download_archive(quarter_url(year, quarter), path)
    return path


def dated_members(zf: zipfile.ZipFile, max_days: int | None) -> list[zipfile.ZipInfo]:
    selected: list[tuple[str, zipfile.ZipInfo]] = []
    for info in zf.infolist():
        if info.is_dir() or not info.filename.lower().endswith(".csv"):
            continue
        match = DATE_PATTERN.search(info.filename)
        if match is not None:
            selected.append((match.group(1), info))
    selected.sort(key=lambda item: (item[0], item[1].filename))
    dates = [date for date, _ in selected]
    duplicates = [date for date, count in Counter(dates).items() if count != 1]
    if duplicates:
        raise ValueError(f"Archive has duplicate daily CSV dates: {duplicates[:5]}")
    infos = [info for _, info in selected]
    if max_days is not None:
        infos = infos[:max_days]
    if not infos:
        raise ValueError("No dated CSV members found in archive")
    return infos


def scan_archive(
    archive: Path,
    max_days: int | None,
) -> tuple[dict[str, int], list[dict[str, object]], dict[str, int]]:
    owner_masks: dict[str, int] = {}
    member_rows: list[dict[str, object]] = []
    raw_rows = 0
    blank_serial_rows = 0
    duplicate_serial_day_rows = 0
    with zipfile.ZipFile(archive) as zf:
        infos = dated_members(zf, max_days)
        for day_index, info in enumerate(infos):
            serial_rows = 0
            member_blank = 0
            member_duplicates = 0
            bit = 1 << day_index
            with zf.open(info) as raw:
                wrapper = io.TextIOWrapper(raw, encoding="utf-8", newline="")
                reader = csv.DictReader(wrapper)
                if reader.fieldnames is None or "serial_number" not in reader.fieldnames:
                    raise ValueError(f"{info.filename} has no serial_number column")
                for row in reader:
                    serial = (row.get("serial_number") or "").strip()
                    if not serial:
                        blank_serial_rows += 1
                        member_blank += 1
                        continue
                    serial_rows += 1
                    raw_rows += 1
                    prior = owner_masks.get(serial, 0)
                    if prior & bit:
                        duplicate_serial_day_rows += 1
                        member_duplicates += 1
                    owner_masks[serial] = prior | bit
            match = DATE_PATTERN.search(info.filename)
            assert match is not None
            member_rows.append(
                {
                    "day_index": day_index,
                    "date": match.group(1),
                    "member": info.filename,
                    "uncompressed_bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "crc32": f"{info.CRC:08x}",
                    "serial_rows": serial_rows,
                    "blank_serial_rows": member_blank,
                    "duplicate_serial_day_rows": member_duplicates,
                }
            )
    stats = {
        "daily_csv_members": len(member_rows),
        "raw_rows_with_serial": raw_rows,
        "blank_serial_rows": blank_serial_rows,
        "duplicate_serial_day_rows": duplicate_serial_day_rows,
        "unique_drive_day_events": sum(mask.bit_count() for mask in owner_masks.values()),
        "unique_drives": len(owner_masks),
    }
    return owner_masks, member_rows, stats


def contiguous_run_lengths(mask: int, number_of_days: int) -> tuple[int, ...]:
    runs: list[int] = []
    current = 0
    for day_index in range(number_of_days):
        if mask & (1 << day_index):
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return tuple(runs)


def profile_histogram(
    owner_masks: Iterable[int],
    number_of_days: int,
) -> Counter[tuple[int, ...]]:
    profiles: Counter[tuple[int, ...]] = Counter()
    for mask in owner_masks:
        runs = contiguous_run_lengths(mask, number_of_days)
        if not runs:
            raise ValueError("Encountered an owner with no selected day")
        profiles[runs] += 1
    return profiles


def window_count(length: int, window: int, stride: int) -> int:
    if length < window:
        return 0
    return 1 + (length - window) // stride


def event_kappa_for_length(length: int, window: int, stride: int) -> int:
    if length < window:
        return 0
    deltas: Counter[int] = Counter()
    for start in range(0, length - window + 1, stride):
        deltas[start] += 1
        deltas[start + window] -= 1
    current = 0
    maximum = 0
    for position in sorted(deltas):
        current += deltas[position]
        maximum = max(maximum, current)
    return maximum


def nearest_rank_quantile(values: Sequence[int], probability: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = math.ceil(probability * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def policy_owner_count(
    runs: tuple[int, ...],
    policy: WindowPolicy,
) -> tuple[int, int]:
    if policy.semantics == "ordered_snapshot":
        length = sum(runs)
        return (
            window_count(length, policy.window_length, policy.stride),
            event_kappa_for_length(length, policy.window_length, policy.stride),
        )
    if policy.semantics == "calendar_contiguous":
        return (
            sum(
                window_count(length, policy.window_length, policy.stride)
                for length in runs
            ),
            max(
                (
                    event_kappa_for_length(
                        length,
                        policy.window_length,
                        policy.stride,
                    )
                    for length in runs
                ),
                default=0,
            ),
        )
    raise ValueError(f"Unsupported semantics: {policy.semantics}")


def summarize_policy(
    profile_counts: Counter[tuple[int, ...]],
    policy: WindowPolicy,
) -> dict[str, object]:
    per_owner_counts: list[int] = []
    event_kappa = 0
    total_windows = 0
    for runs, owner_frequency in profile_counts.items():
        count, profile_event_kappa = policy_owner_count(runs, policy)
        event_kappa = max(event_kappa, profile_event_kappa)
        if count:
            total_windows += count * owner_frequency
            per_owner_counts.extend([count] * owner_frequency)
    owner_kappa = max(per_owner_counts, default=0)
    return {
        **asdict(policy),
        "num_windows": total_windows,
        "owners_with_windows": len(per_owner_counts),
        "observed_event_kappa": event_kappa,
        "observed_owner_kappa": owner_kappa,
        "owner_kappa_p95": nearest_rank_quantile(per_owner_counts, 0.95),
        "owner_kappa_p99": nearest_rank_quantile(per_owner_counts, 0.99),
        "event_raw_adjacency": "replace_one_fixed_drive_date_slot",
        "accountant_metric_for_candidate_bound": "generated_add_remove",
        "candidate_event_stability_k": 2 * event_kappa,
        "event_stability_status": "candidate_fixed_slot_bound",
        "owner_raw_adjacency": "owner_add_remove",
        "candidate_owner_stability_k": owner_kappa,
        "owner_stability_status": "candidate_fixed_owner_partition_bound",
        "privacy_authority": "diagnostic_only",
        "privacy_certificate_status": "not_evaluated_no_mapping_or_bound_mechanism",
        "same_number_event_dp_reuse_allowed": False,
    }


def summarize_owner_cap(
    profile_counts: Counter[tuple[int, ...]],
    cap: int,
) -> dict[str, object]:
    per_owner_counts: list[int] = []
    for runs, owner_frequency in profile_counts.items():
        count = min(sum(runs), cap)
        if count:
            per_owner_counts.extend([count] * owner_frequency)
    owner_kappa = max(per_owner_counts, default=0)
    event_kappa = 1 if per_owner_counts else 0
    return {
        "scenario": f"backblaze_v2_owner_snapshot_prefix_cap{cap}",
        "semantics": "owner_snapshot_prefix_cap",
        "window_length": 1,
        "stride": None,
        "policy": "owner_prefix_cap",
        "cap": cap,
        "num_windows": sum(per_owner_counts),
        "owners_with_windows": len(per_owner_counts),
        "observed_event_kappa": event_kappa,
        "observed_owner_kappa": owner_kappa,
        "owner_kappa_p95": nearest_rank_quantile(per_owner_counts, 0.95),
        "owner_kappa_p99": nearest_rank_quantile(per_owner_counts, 0.99),
        "event_raw_adjacency": "replace_one_fixed_drive_date_slot",
        "accountant_metric_for_candidate_bound": "generated_add_remove",
        "candidate_event_stability_k": 2 * event_kappa,
        "event_stability_status": "candidate_fixed_slot_bound",
        "owner_raw_adjacency": "owner_add_remove",
        "candidate_owner_stability_k": owner_kappa,
        "owner_stability_status": "candidate_fixed_owner_partition_bound",
        "privacy_authority": "diagnostic_only",
        "privacy_certificate_status": "not_evaluated_no_mapping_or_bound_mechanism",
        "same_number_event_dp_reuse_allowed": False,
    }


def policies() -> list[WindowPolicy]:
    result: list[WindowPolicy] = []
    for semantics in ("ordered_snapshot", "calendar_contiguous"):
        unit = "snapshot" if semantics == "ordered_snapshot" else "day"
        for window in (7, 30, 60):
            stride_policies = [("dense", 1)]
            half = max(1, window // 2)
            if half not in {1, window}:
                stride_policies.append(("half", half))
            stride_policies.append(("non_overlap", window))
            for label, stride in stride_policies:
                result.append(
                    WindowPolicy(
                        scenario=(
                            f"backblaze_v2_{semantics}_{window}{unit}_{label}"
                        ),
                        semantics=semantics,
                        window_length=window,
                        stride=stride,
                        policy=label,
                    )
                )
    return result


def write_member_manifest(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fields = (
        "day_index",
        "date",
        "member",
        "uncompressed_bytes",
        "compressed_bytes",
        "crc32",
        "serial_rows",
        "blank_serial_rows",
        "duplicate_serial_day_rows",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_profile_histogram(
    path: Path,
    profile_counts: Counter[tuple[int, ...]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("run_lengths", "snapshot_count", "run_count", "owners"),
        )
        writer.writeheader()
        for runs, owner_count in sorted(
            profile_counts.items(),
            key=lambda item: (sum(item[0]), item[0]),
        ):
            writer.writerow(
                {
                    "run_lengths": ";".join(str(value) for value in runs),
                    "snapshot_count": sum(runs),
                    "run_count": len(runs),
                    "owners": owner_count,
                }
            )


def write_policy_summary(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fields = (
        "scenario",
        "semantics",
        "window_length",
        "stride",
        "policy",
        "cap",
        "num_windows",
        "owners_with_windows",
        "observed_event_kappa",
        "observed_owner_kappa",
        "owner_kappa_p95",
        "owner_kappa_p99",
        "candidate_event_stability_k",
        "candidate_owner_stability_k",
        "privacy_authority",
        "privacy_certificate_status",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    evidence: dict[str, object],
    policy_rows: Sequence[dict[str, object]],
) -> None:
    stats = evidence["scan_statistics"]
    assert isinstance(stats, dict)
    lines = [
        "# Backblaze Q1 2025 v2 Mapping Evidence",
        "",
        "Status: **DIAGNOSTIC ONLY — NOT A DP CERTIFICATE**",
        "",
        f"Archive SHA-256: `{evidence['archive_sha256']}`",
        f"Generator SHA-256: `{evidence['generator_sha256']}`",
        (
            f"Scanned {stats['raw_rows_with_serial']} rows, "
            f"{stats['unique_drive_day_events']} unique drive/date events, and "
            f"{stats['unique_drives']} drives."
        ),
        (
            f"Duplicate serial/date rows: "
            f"{stats['duplicate_serial_day_rows']}"
        ),
        "",
        "`ordered_snapshot` counts available observations and compresses date gaps.",
        "`calendar_contiguous` restarts windows after every missing calendar day.",
        "The two are intentionally reported separately.",
        "",
        "For event claims the candidate raw adjacency replaces values in one fixed",
        "drive/date slot.  Against a generated add/remove accountant the candidate",
        "distance is twice observed event κ, so observed κ=1 is not same-number",
        "DIRECT evidence.  No epsilon/delta statement is evaluated here.",
        "",
        "| policy | semantics | windows | owners | observed event κ | owner κ | candidate event K |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in policy_rows:
        lines.append(
            "| {scenario} | {semantics} | {num_windows} | "
            "{owners_with_windows} | {observed_event_kappa} | "
            "{observed_owner_kappa} | {candidate_event_stability_k} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--quarter", type=int, default=1)
    parser.add_argument("--cache-dir", default="data/backblaze")
    parser.add_argument(
        "--output-dir",
        default="reports/backblaze_v2_mapping_evidence_q1_2025_001",
    )
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    if not cache_dir.is_absolute():
        cache_dir = project_root / cache_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    archive = ensure_archive(args.year, args.quarter, cache_dir, args.download)
    archive_sha256 = sha256_file(archive)
    print(f"Scanning {archive}")
    owner_masks, member_rows, scan_stats = scan_archive(archive, args.max_days)
    profile_counts = profile_histogram(
        owner_masks.values(),
        scan_stats["daily_csv_members"],
    )
    del owner_masks

    member_manifest_path = output_dir / "input_members.csv"
    profile_path = output_dir / "owner_profile_histogram.csv"
    write_member_manifest(member_manifest_path, member_rows)
    write_profile_histogram(profile_path, profile_counts)

    policy_rows = [
        summarize_policy(profile_counts, policy)
        for policy in policies()
    ]
    policy_rows.extend(
        summarize_owner_cap(profile_counts, cap)
        for cap in (1, 8, 30, 90)
    )
    policy_summary_path = output_dir / "policy_summary.csv"
    write_policy_summary(policy_summary_path, policy_rows)

    evidence = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": f"Backblaze_Drive_Stats_Q{args.quarter}_{args.year}",
        "source_type": "public_benchmark_local_archive",
        "source_url": quarter_url(args.year, args.quarter),
        "archive_file": archive.relative_to(project_root).as_posix(),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_sha256,
        "input_members_file": "input_members.csv",
        "input_members_sha256": sha256_file(member_manifest_path),
        "input_members_semantic_sha256": stable_json_sha256(member_rows),
        "owner_profile_histogram_file": "owner_profile_histogram.csv",
        "owner_profile_histogram_sha256": sha256_file(profile_path),
        "generator_file": "scripts/backblaze_v2_evidence.py",
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "scan_statistics": scan_stats,
        "record_identity_policy": "fixed_drive_date_slots",
        "event_adjacency_scope": (
            "replace SMART values/label within one fixed drive/date slot; "
            "drive and date identity do not change"
        ),
        "unsupported_event_adjacencies": [
            "drive-day insertion",
            "drive-day deletion",
            "serial-number reassignment",
            "date reassignment",
        ],
        "owner_adjacency_scope": "add or remove one complete drive partition",
        "analytic_evidence_scope": (
            "exact for the bound archive and declared window constructors; "
            "not a complete per-window v2 mapping export"
        ),
        "privacy_authority": "diagnostic_only",
        "privacy_certificate_status": "not_evaluated_no_mapping_or_bound_mechanism",
        "policies": policy_rows,
    }
    evidence_path = output_dir / "evidence.json"
    write_json(evidence_path, evidence)
    summary_path = output_dir / "summary.md"
    write_summary(summary_path, evidence, policy_rows)

    indexed_paths = [
        Path(__file__).resolve(),
        member_manifest_path,
        profile_path,
        policy_summary_path,
        evidence_path,
        summary_path,
    ]
    artifact_index = {
        "schema_version": SCHEMA_VERSION,
        "files": [
            {
                "path": (
                    path.relative_to(project_root).as_posix()
                    if path.is_relative_to(project_root)
                    else str(path)
                ),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in indexed_paths
        ],
    }
    write_json(output_dir / "artifact_index.json", artifact_index)
    print(f"Wrote {evidence_path}")
    print(f"Wrote {policy_summary_path}")


if __name__ == "__main__":
    main()
