#!/usr/bin/env python3
"""Verify the frozen V3.4 route-obligation feasibility report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_v34_route_obligation_gate import (
    DEFAULT_OUTPUT,
    build_report,
)

DEFAULT_VERIFICATION = (
    DEFAULT_OUTPUT.parent / "route_obligation_verification_v34.json"
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
    checks = {
        "exact_rebuild": observed == rebuilt,
        "status_pass": observed.get("status") == "PASS",
        "six_baselines": (
            observed.get("baseline_contracts", {}).get("accepted") == 6
            and observed.get("baseline_contracts", {}).get("required") == 6
        ),
        "two_unique_entries": (
            observed.get("registry", {}).get("entry_count") == 2
            and observed.get("registry", {}).get(
                "unique_nominal_tuple_count"
            )
            == 2
        ),
        "eight_obligations_bound": (
            len(observed.get("obligations", [])) == 8
            and all(
                row.get("both_routes_bound")
                for row in observed.get("obligations", [])
            )
        ),
        "bounded_verdict_only": (
            observed.get("verdict", {}).get(
                "L0_fixed_pf_noninterchangeability"
            )
            == "PASS"
            and observed.get("verdict", {}).get(
                "L1_bounded_finite_registry_obligation_model"
            )
            == "PASS"
            and observed.get("verdict", {}).get(
                "L2_generic_route_privacy_or_correctness"
            )
            == "FORBIDDEN"
            and observed.get("registry", {}).get(
                "generic_plugin_loader_present"
            )
            is False
        ),
    }
    passed = all(checks.values())
    result = {
        "schema_version": "unitdp.v34_route_obligation_verification.v1",
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
