#!/usr/bin/env python3
"""Separately rebuild and verify the Compiler V3.9 format gate."""

from __future__ import annotations

import json

try:
    from scripts.build_v39_format_gate import DEFAULT_OUTPUT, build_report
except ModuleNotFoundError:
    from build_v39_format_gate import DEFAULT_OUTPUT, build_report


DEFAULT_VERIFICATION = DEFAULT_OUTPUT.with_name("format_gate_verification.json")


def main() -> None:
    if DEFAULT_VERIFICATION.exists():
        raise FileExistsError("refusing to overwrite V3.9 format verification")
    observed = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    rebuilt = build_report()
    checks = {
        "exact_rebuild": observed == rebuilt,
        "status_and_all_fourteen_checks": (
            observed.get("status") == "PASS"
            and len(observed.get("checks", {})) == 14
            and all(observed.get("checks", {}).values())
        ),
        "scientific_text_exact": observed.get("checks", {}).get(
            "rendered_scientific_text_byte_exact"
        ) is True,
        "single_source_and_table_font_gate": (
            observed.get("checks", {}).get("single_source_and_minimum_nine_point_tables")
            is True
            and observed.get("checks", {}).get("only_permitted_tabcolsep_setlength")
            is True
        ),
        "page_partition": observed.get("checks", {}).get(
            "content_and_disclosure_end_by_page_six"
        ) is True,
        "scope_bounded": (
            "not a new scientific result" in observed.get("scope", "")
            and "not a new scientific result" in rebuilt.get("scope", "")
        ),
    }
    result = {
        "schema_version": "unitdp.compiler_v39_format_gate_verification.v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "passed": sum(checks.values()),
        "required": len(checks),
        "report_payload_sha256": observed.get("payload_sha256"),
    }
    DEFAULT_VERIFICATION.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
