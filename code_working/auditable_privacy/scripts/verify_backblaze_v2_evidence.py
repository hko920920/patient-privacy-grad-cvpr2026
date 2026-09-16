"""Independent verifier for the Backblaze v2 mapping-evidence bundle.

By default this verifier reopens and rescans every daily CSV in the bound
archive. ``--skip-archive-rescan`` is the portable review mode: it validates
the archive declaration and independently recomputes every reported policy
from the bundled owner-profile histogram without requiring the 1.02 GB archive.
Neither mode treats the analytic summary as a DP certificate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence


DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def forbidden_privacy_keys(value: object, prefix: str = "") -> list[str]:
    forbidden = {"epsilon", "delta", "supported_statement"}
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in forbidden:
                found.append(path)
            found.extend(forbidden_privacy_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(forbidden_privacy_keys(child, f"{prefix}[{index}]"))
    return found


def contiguous_runs(mask: int, days: int) -> tuple[int, ...]:
    output: list[int] = []
    length = 0
    for bit_index in range(days):
        if mask & (1 << bit_index):
            length += 1
        elif length:
            output.append(length)
            length = 0
    if length:
        output.append(length)
    return tuple(output)


def count_windows(length: int, width: int, stride: int) -> int:
    return 0 if length < width else 1 + (length - width) // stride


def max_overlap(length: int, width: int, stride: int) -> int:
    if length < width:
        return 0
    differences: Counter[int] = Counter()
    for start in range(0, length - width + 1, stride):
        differences[start] += 1
        differences[start + width] -= 1
    running = 0
    maximum = 0
    for point in sorted(differences):
        running += differences[point]
        maximum = max(maximum, running)
    return maximum


def nearest_rank(values: Sequence[int], probability: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = math.ceil(probability * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def policy_specs() -> dict[str, tuple[str, int, int, str]]:
    result: dict[str, tuple[str, int, int, str]] = {}
    for semantics in ("ordered_snapshot", "calendar_contiguous"):
        unit = "snapshot" if semantics == "ordered_snapshot" else "day"
        for width in (7, 30, 60):
            candidates = [("dense", 1), ("half", width // 2), ("non_overlap", width)]
            for label, stride in candidates:
                scenario = f"backblaze_v2_{semantics}_{width}{unit}_{label}"
                result[scenario] = (semantics, width, stride, label)
    return result


def load_profile_histogram(path: Path) -> Counter[tuple[int, ...]]:
    profiles: Counter[tuple[int, ...]] = Counter()
    for row in csv.DictReader(path.open(encoding="utf-8")):
        runs = tuple(int(value) for value in row["run_lengths"].split(";") if value)
        if (
            not runs
            or sum(runs) != int(row["snapshot_count"])
            or len(runs) != int(row["run_count"])
        ):
            raise ValueError(f"Invalid profile row: {row}")
        profiles[runs] += int(row["owners"])
    return profiles


def summarize_from_profiles(
    profiles: Counter[tuple[int, ...]],
    semantics: str,
    width: int,
    stride: int,
) -> dict[str, int]:
    counts: list[int] = []
    event_kappa = 0
    total = 0
    for runs, frequency in profiles.items():
        if semantics == "ordered_snapshot":
            length = sum(runs)
            count = count_windows(length, width, stride)
            event = max_overlap(length, width, stride)
        elif semantics == "calendar_contiguous":
            count = sum(count_windows(length, width, stride) for length in runs)
            event = max(
                (max_overlap(length, width, stride) for length in runs),
                default=0,
            )
        else:
            raise ValueError(semantics)
        event_kappa = max(event_kappa, event)
        if count:
            total += count * frequency
            counts.extend([count] * frequency)
    return {
        "num_windows": total,
        "owners_with_windows": len(counts),
        "observed_event_kappa": event_kappa,
        "observed_owner_kappa": max(counts, default=0),
        "owner_kappa_p95": nearest_rank(counts, 0.95),
        "owner_kappa_p99": nearest_rank(counts, 0.99),
    }


def summarize_cap(
    profiles: Counter[tuple[int, ...]],
    cap: int,
) -> dict[str, int]:
    counts: list[int] = []
    for runs, frequency in profiles.items():
        count = min(sum(runs), cap)
        if count:
            counts.extend([count] * frequency)
    event = 1 if counts else 0
    return {
        "num_windows": sum(counts),
        "owners_with_windows": len(counts),
        "observed_event_kappa": event,
        "observed_owner_kappa": max(counts, default=0),
        "owner_kappa_p95": nearest_rank(counts, 0.95),
        "owner_kappa_p99": nearest_rank(counts, 0.99),
    }


def re_scan_archive(
    archive: Path,
) -> tuple[Counter[tuple[int, ...]], list[dict[str, object]], dict[str, int]]:
    masks: dict[str, int] = {}
    member_rows: list[dict[str, object]] = []
    raw_rows = 0
    blanks = 0
    duplicates = 0
    with zipfile.ZipFile(archive) as zf:
        selected: list[tuple[str, zipfile.ZipInfo]] = []
        for info in zf.infolist():
            match = DATE_PATTERN.search(info.filename)
            if (
                not info.is_dir()
                and info.filename.lower().endswith(".csv")
                and match is not None
            ):
                selected.append((match.group(1), info))
        selected.sort(key=lambda item: (item[0], item[1].filename))
        dates = [date for date, _ in selected]
        if any(count != 1 for count in Counter(dates).values()):
            raise ValueError("Archive has duplicate daily CSV dates")

        for day_index, (date, info) in enumerate(selected):
            serial_rows = 0
            member_blanks = 0
            member_duplicates = 0
            bit = 1 << day_index
            with zf.open(info) as raw:
                reader = csv.DictReader(
                    io.TextIOWrapper(raw, encoding="utf-8", newline="")
                )
                if reader.fieldnames is None or "serial_number" not in reader.fieldnames:
                    raise ValueError(f"{info.filename}: missing serial_number")
                for row in reader:
                    serial = (row.get("serial_number") or "").strip()
                    if not serial:
                        blanks += 1
                        member_blanks += 1
                        continue
                    raw_rows += 1
                    serial_rows += 1
                    previous = masks.get(serial, 0)
                    if previous & bit:
                        duplicates += 1
                        member_duplicates += 1
                    masks[serial] = previous | bit
            member_rows.append(
                {
                    "day_index": day_index,
                    "date": date,
                    "member": info.filename,
                    "uncompressed_bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "crc32": f"{info.CRC:08x}",
                    "serial_rows": serial_rows,
                    "blank_serial_rows": member_blanks,
                    "duplicate_serial_day_rows": member_duplicates,
                }
            )
    profiles: Counter[tuple[int, ...]] = Counter(
        contiguous_runs(mask, len(member_rows)) for mask in masks.values()
    )
    stats = {
        "daily_csv_members": len(member_rows),
        "raw_rows_with_serial": raw_rows,
        "blank_serial_rows": blanks,
        "duplicate_serial_day_rows": duplicates,
        "unique_drive_day_events": sum(mask.bit_count() for mask in masks.values()),
        "unique_drives": len(masks),
    }
    return profiles, member_rows, stats


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def require(self, identifier: str, condition: bool, detail: str) -> None:
        self.rows.append(
            {"id": identifier, "passed": bool(condition), "detail": detail}
        )
        if not condition:
            raise RuntimeError(f"{identifier}: {detail}")


def verify_artifact_index(
    project_root: Path,
    bundle_dir: Path,
    checks: Checks,
) -> None:
    index = json.loads((bundle_dir / "artifact_index.json").read_text(encoding="utf-8"))
    entries = index.get("files")
    checks.require(
        "artifact_index_nonempty",
        isinstance(entries, list) and bool(entries),
        "artifact index",
    )
    assert isinstance(entries, list)
    for entry in entries:
        target = project_root / entry["path"]
        checks.require(
            f"artifact_hash:{target.name}",
            target.is_file()
            and target.stat().st_size == int(entry["bytes"])
            and sha256_file(target) == entry["sha256"],
            str(entry["path"]),
        )


def compare_policy(
    policy: Mapping[str, object],
    recomputed: Mapping[str, int],
    checks: Checks,
) -> None:
    scenario = str(policy["scenario"])
    numeric_fields = (
        "num_windows",
        "owners_with_windows",
        "observed_event_kappa",
        "observed_owner_kappa",
        "owner_kappa_p95",
        "owner_kappa_p99",
    )
    checks.require(
        f"{scenario}:exact_policy_values",
        all(int(policy[key]) == recomputed[key] for key in numeric_fields),
        str(dict(recomputed)),
    )
    event_kappa = recomputed["observed_event_kappa"]
    owner_kappa = recomputed["observed_owner_kappa"]
    checks.require(
        f"{scenario}:stability_and_authority_boundary",
        int(policy["candidate_event_stability_k"]) == 2 * event_kappa
        and int(policy["candidate_owner_stability_k"]) == owner_kappa
        and policy["same_number_event_dp_reuse_allowed"] is False
        and policy["privacy_authority"] == "diagnostic_only"
        and policy["privacy_certificate_status"]
        == "not_evaluated_no_mapping_or_bound_mechanism",
        f"event_k={event_kappa}, owner_k={owner_kappa}",
    )


def verify_policy_csv(
    bundle_dir: Path,
    policies: Sequence[Mapping[str, object]],
    checks: Checks,
) -> None:
    rows = list(
        csv.DictReader((bundle_dir / "policy_summary.csv").open(encoding="utf-8"))
    )
    by_name = {row["scenario"]: row for row in rows}
    checks.require(
        "policy_csv_scenario_set",
        set(by_name) == {str(row["scenario"]) for row in policies},
        f"rows={len(rows)}",
    )
    keys = (
        "semantics",
        "window_length",
        "stride",
        "policy",
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
    for policy in policies:
        row = by_name[str(policy["scenario"])]
        for key in keys:
            expected = "" if policy.get(key) is None else str(policy.get(key))
            if row[key] != expected:
                raise RuntimeError(
                    f"{policy['scenario']}: policy CSV drift at {key}"
                )
    checks.require("policy_csv_matches_evidence", True, f"rows={len(rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle-dir",
        default="reports/backblaze_v2_mapping_evidence_q1_2025_001",
    )
    parser.add_argument("--skip-archive-rescan", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    bundle_dir = Path(args.bundle_dir)
    if not bundle_dir.is_absolute():
        bundle_dir = project_root / bundle_dir
    output_path = bundle_dir / "independent_verification.json"
    checks = Checks()
    failure = ""
    try:
        evidence = json.loads(
            (bundle_dir / "evidence.json").read_text(encoding="utf-8")
        )
        checks.require(
            "schema_and_authority",
            evidence.get("schema_version") == "backblaze_mapping_evidence_v2.0"
            and evidence.get("privacy_authority") == "diagnostic_only"
            and evidence.get("privacy_certificate_status")
            == "not_evaluated_no_mapping_or_bound_mechanism",
            "v2 diagnostic-only contract",
        )
        checks.require(
            "no_dp_parameters_or_wording",
            not forbidden_privacy_keys(evidence),
            str(forbidden_privacy_keys(evidence)),
        )
        generator = project_root / str(evidence["generator_file"])
        checks.require(
            "generator_source_hash",
            sha256_file(generator) == evidence["generator_sha256"],
            str(evidence["generator_file"]),
        )
        verify_artifact_index(project_root, bundle_dir, checks)

        archive = project_root / str(evidence["archive_file"])
        if args.skip_archive_rescan:
            checks.require(
                "archive_declaration",
                int(evidence["archive_bytes"]) > 0
                and bool(
                    re.fullmatch(
                        r"[0-9a-f]{64}",
                        str(evidence["archive_sha256"]),
                    )
                )
                and not Path(str(evidence["archive_file"])).is_absolute(),
                (
                    f"{evidence['archive_file']}; "
                    f"bytes={evidence['archive_bytes']}; "
                    f"sha256={evidence['archive_sha256']}"
                ),
            )
        else:
            checks.require(
                "archive_hash_and_size",
                archive.stat().st_size == int(evidence["archive_bytes"])
                and sha256_file(archive) == evidence["archive_sha256"],
                str(evidence["archive_file"]),
            )
        profile_path = bundle_dir / str(evidence["owner_profile_histogram_file"])
        checks.require(
            "profile_file_hash",
            sha256_file(profile_path)
            == evidence["owner_profile_histogram_sha256"],
            str(evidence["owner_profile_histogram_file"]),
        )
        profiles = load_profile_histogram(profile_path)
        stats = evidence["scan_statistics"]
        checks.require(
            "profile_internal_totals",
            sum(profiles.values()) == int(stats["unique_drives"])
            and sum(sum(runs) * count for runs, count in profiles.items())
            == int(stats["unique_drive_day_events"]),
            f"profiles={len(profiles)}",
        )

        if not args.skip_archive_rescan:
            rescanned_profiles, member_rows, rescanned_stats = re_scan_archive(archive)
            checks.require(
                "full_archive_rescan_statistics",
                rescanned_stats == stats,
                str(rescanned_stats),
            )
            checks.require(
                "full_archive_rescan_profiles",
                rescanned_profiles == profiles,
                f"profiles={len(rescanned_profiles)}",
            )
            exported_member_rows = [
                {
                    "day_index": int(row["day_index"]),
                    "date": row["date"],
                    "member": row["member"],
                    "uncompressed_bytes": int(row["uncompressed_bytes"]),
                    "compressed_bytes": int(row["compressed_bytes"]),
                    "crc32": row["crc32"],
                    "serial_rows": int(row["serial_rows"]),
                    "blank_serial_rows": int(row["blank_serial_rows"]),
                    "duplicate_serial_day_rows": int(
                        row["duplicate_serial_day_rows"]
                    ),
                }
                for row in csv.DictReader(
                    (bundle_dir / str(evidence["input_members_file"])).open(
                        encoding="utf-8"
                    )
                )
            ]
            checks.require(
                "full_archive_member_manifest",
                member_rows == exported_member_rows,
                f"members={len(member_rows)}",
            )

        policies_value = evidence.get("policies")
        checks.require(
            "policy_list_nonempty",
            isinstance(policies_value, list) and bool(policies_value),
            "policy list",
        )
        assert isinstance(policies_value, list)
        specs = policy_specs()
        expected_scenarios = set(specs) | {
            f"backblaze_v2_owner_snapshot_prefix_cap{cap}"
            for cap in (1, 8, 30, 90)
        }
        checks.require(
            "expected_policy_set",
            {str(row["scenario"]) for row in policies_value}
            == expected_scenarios,
            f"expected={len(expected_scenarios)}",
        )
        for policy in policies_value:
            scenario = str(policy["scenario"])
            if scenario in specs:
                semantics, width, stride, label = specs[scenario]
                checks.require(
                    f"{scenario}:declared_parameters",
                    policy["semantics"] == semantics
                    and int(policy["window_length"]) == width
                    and int(policy["stride"]) == stride
                    and policy["policy"] == label,
                    str(specs[scenario]),
                )
                recomputed = summarize_from_profiles(
                    profiles,
                    semantics,
                    width,
                    stride,
                )
            else:
                cap = int(scenario.rsplit("cap", 1)[1])
                checks.require(
                    f"{scenario}:declared_parameters",
                    policy["semantics"] == "owner_snapshot_prefix_cap"
                    and int(policy["cap"]) == cap
                    and policy["policy"] == "owner_prefix_cap",
                    f"cap={cap}",
                )
                recomputed = summarize_cap(profiles, cap)
            compare_policy(policy, recomputed, checks)
        verify_policy_csv(bundle_dir, policies_value, checks)
    except Exception as exc:  # preserve a machine-readable failure record
        failure = f"{type(exc).__name__}: {exc}"

    passed = not failure and all(bool(row["passed"]) for row in checks.rows)
    record = {
        "schema_version": "backblaze_mapping_evidence_independent_verification_v1",
        "verification_passed": passed,
        "archive_rescanned": not args.skip_archive_rescan,
        "checks_total": len(checks.rows),
        "checks_passed": sum(bool(row["passed"]) for row in checks.rows),
        "failure": failure,
        "verifier_file": "scripts/verify_backblaze_v2_evidence.py",
        "verifier_sha256": sha256_file(Path(__file__).resolve()),
        "checks": checks.rows,
    }
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"verification_passed={passed}")
    print(f"checks={record['checks_passed']}/{record['checks_total']}")
    if failure:
        print(f"failure={failure}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
