#!/usr/bin/env python3
"""Rebuild and verify the V3.5 random-allocation accountant gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_v35_random_allocation_accountant_gate import (
    DEFAULT_OUTPUT,
    build_report,
)


DEFAULT_VERIFICATION = (
    DEFAULT_OUTPUT.parent
    / "random_allocation_accountant_verification_v35.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_VERIFICATION,
    )
    args = parser.parse_args()
    observed = json.loads(args.report.read_text(encoding="utf-8"))
    rebuilt = build_report()
    cases = observed.get("cases", [])
    checks = {
        "exact_rebuild": observed == rebuilt,
        "status_pass": observed.get("status") == "PASS",
        "three_profiles": (
            len(cases) == 3
            and all(case.get("pass") for case in cases)
        ),
        "two_directions": (
            observed.get("accountant", {}).get("directions")
            == ["remove", "add"]
            and observed.get("accountant", {}).get(
                "reported_epsilon"
            )
            == "maximum_of_both_directions"
        ),
        "oracle_tolerance": all(
            case.get("maximum_absolute_rdp_difference", 1.0)
            <= observed.get("numeric_tolerance", 0.0)
            for case in cases
        ),
        "query_identity": (
            observed.get("query_identity_checks", {}).get("pass")
            is True
            and observed.get("query_identity_checks", {}).get(
                "passed"
            )
            == 6
        ),
        "invalid_queries": (
            observed.get("invalid_query_checks", {}).get("pass")
            is True
            and observed.get("invalid_query_checks", {}).get(
                "rejected"
            )
            == 13
        ),
        "baseline_preserved": all(
            row.get("pass")
            for row in observed.get("frozen_baseline", [])
        ),
        "claim_firewall": (
            observed.get("verdict", {}).get("accountant_gate")
            == "PASS"
            and len(
                observed.get("verdict", {}).get(
                    "does_not_authorize",
                    [],
                )
            )
            == 6
            and observed.get("external_probe", {}).get("role")
            == "comparison_only_not_route_authorization"
        ),
    }
    passed = all(checks.values())
    result = {
        "schema_version": (
            "unitdp.v35_random_allocation_accountant_verification.v1"
        ),
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "report_payload_sha256": observed.get("payload_sha256"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
