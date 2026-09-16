import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from mapping_evidence import (
    canonical_records_sha256,
    file_sha256,
    parse_owner_ids,
)

def canonical_selected_sha256(rows):
    records = []
    for row in rows:
        owners = parse_owner_ids(row)
        records.append(
            {
                "scenario": str(row.get("scenario", "")),
                "window_id": str(row.get("window_id") or ""),
                "generated_unit": str(row.get("generated_unit") or ""),
                "owner_ids": sorted(str(owner) for owner in owners),
                "start": int(row["start"]),
                "end": int(row["end"]),
            }
        )
    return canonical_records_sha256(records)


def read_rows(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return fieldnames, rows


def as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def add_check(checks, name, status, detail):
    checks.append({"check": name, "status": status, "detail": str(detail)})


def main():
    parser = argparse.ArgumentParser(
        description="Check that an exported unit-audit mapping is complete enough to certify."
    )
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--expected-scenarios", default="")
    parser.add_argument("--privacy-report-json", type=Path, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames, all_rows = read_rows(args.mapping)
    selected_rows = [
        row for row in all_rows if args.scenario is None or row.get("scenario") == args.scenario
    ]

    checks = []
    required = {"scenario", "window_id", "generated_unit", "start", "end"}
    missing_required = sorted(required - set(fieldnames))
    add_check(
        checks,
        "required_v2_mapping_columns",
        "fail" if missing_required else "pass",
        f"missing={missing_required}",
    )

    has_owner = bool({"owner_id", "owner_ids", "owners"} & set(fieldnames))
    add_check(
        checks,
        "owner_attribution_column",
        "pass" if has_owner else "fail",
        "owner_id/owner_ids/owners present" if has_owner else "missing owner attribution",
    )

    if args.scenario is not None:
        add_check(
            checks,
            "scenario_rows_present",
            "pass" if selected_rows else "fail",
            f"scenario={args.scenario}, rows={len(selected_rows)}",
        )

    if args.expected_rows is not None:
        add_check(
            checks,
            "expected_row_count",
            "pass" if len(selected_rows) == args.expected_rows else "fail",
            f"observed={len(selected_rows)}, expected={args.expected_rows}",
        )

    expected_scenarios = [s.strip() for s in args.expected_scenarios.split(",") if s.strip()]
    if expected_scenarios:
        observed_scenarios = {row.get("scenario", "") for row in all_rows}
        missing_scenarios = sorted(set(expected_scenarios) - observed_scenarios)
        add_check(
            checks,
            "expected_scenarios_present",
            "pass" if not missing_scenarios else "fail",
            f"missing={missing_scenarios}",
        )

    invalid_intervals = 0
    missing_owners = 0
    invalid_generated_units = 0
    duplicate_keys = 0
    key_counts = Counter()
    for idx, row in enumerate(selected_rows):
        start = as_int(row.get("start"))
        end = as_int(row.get("end"))
        if start is None or end is None or end <= start:
            invalid_intervals += 1
        try:
            parse_owner_ids(row)
        except ValueError:
            missing_owners += 1
        if row.get("generated_unit") not in {"window", "event", "owner"}:
            invalid_generated_units += 1
        key = (row.get("scenario", ""), row.get("window_id") or str(idx))
        key_counts[key] += 1
    duplicate_keys = sum(count - 1 for count in key_counts.values() if count > 1)

    add_check(
        checks,
        "valid_intervals",
        "pass" if invalid_intervals == 0 else "fail",
        f"invalid_intervals={invalid_intervals}",
    )
    add_check(
        checks,
        "nonempty_owner_attribution",
        "pass" if missing_owners == 0 else "fail",
        f"missing_owner_rows={missing_owners}",
    )
    add_check(
        checks,
        "valid_generated_unit",
        "pass" if invalid_generated_units == 0 else "fail",
        f"invalid_generated_unit_rows={invalid_generated_units}",
    )
    add_check(
        checks,
        "unique_window_ids_within_scenario",
        "pass" if duplicate_keys == 0 else "fail",
        f"duplicate_window_keys={duplicate_keys}",
    )

    mapping_sha = file_sha256(args.mapping)
    selected_sha = ""
    if (
        selected_rows
        and not missing_required
        and has_owner
        and invalid_intervals == 0
        and missing_owners == 0
        and invalid_generated_units == 0
        and duplicate_keys == 0
    ):
        selected_sha = canonical_selected_sha256(selected_rows)
    add_check(checks, "mapping_file_sha256", "info", mapping_sha)
    add_check(checks, "selected_mapping_sha256", "info", selected_sha)

    if args.privacy_report_json:
        report = json.loads(args.privacy_report_json.read_text(encoding="utf-8"))
        pipeline = report.get("pipeline")
        if not isinstance(pipeline, dict):
            pipeline = {}
        report_hash = (
            pipeline.get("selected_mapping_sha256")
            or report.get("selected_mapping_sha256")
            or report.get("mapping_sha256")
            or ""
        )
        if not report_hash:
            status = "fail"
            detail = "privacy report has no selected_mapping_sha256 or mapping_sha256"
        elif report_hash == selected_sha or report_hash == mapping_sha:
            status = "pass"
            detail = "privacy report hash matches exported mapping"
        else:
            status = "fail"
            detail = f"privacy_report_hash={report_hash}"
        add_check(checks, "privacy_report_mapping_hash", status, detail)

    out_csv = args.output_dir / "mapping_crosscheck.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "status", "detail"])
        writer.writeheader()
        writer.writerows(checks)

    fail_count = sum(1 for row in checks if row["status"] == "fail")
    warning_count = sum(1 for row in checks if row["status"] == "warning")
    summary = args.output_dir / "summary.md"
    summary.write_text(
        "\n".join(
            [
                "# Mapping Cross-Check",
                "",
                "This diagnostic report checks whether the exported mapping has the basic metadata needed by the v2.0 validator. It cannot authorize privacy wording.",
                "",
                "```text",
                f"mapping = {args.mapping}",
                f"scenario = {args.scenario or '<all>'}",
                f"rows_all = {len(all_rows)}",
                f"rows_selected = {len(selected_rows)}",
                f"fail_count = {fail_count}",
                f"warning_count = {warning_count}",
                "```",
                "",
                "The cross-check is a guardrail, not a proof of complete provenance. It can catch row-count mismatches, missing owner metadata, invalid intervals, duplicate window ids, and privacy-report hash mismatches.",
                "",
                "| Check | Status | Detail |",
                "| --- | --- | --- |",
                *[
                    f"| {row['check']} | {row['status']} | {row['detail']} |"
                    for row in checks
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote {out_csv}")
    print(f"Wrote {summary}")
    if fail_count:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
