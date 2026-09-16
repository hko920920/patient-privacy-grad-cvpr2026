#!/usr/bin/env python3
"""Independently rebuild the V3.5 allocation contract/compiler gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_v35_allocation_contract_gate import (
    DEFAULT_OUTPUT,
    build_report,
    canonical_sha256,
)


DEFAULT_VERIFICATION = (
    DEFAULT_OUTPUT.parent / "allocation_contract_verification_v35.json"
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
    payload = dict(observed)
    reported_payload_sha256 = payload.pop("payload_sha256", "")
    cases = observed.get("cases", [])
    checks = {
        "exact_rebuild": observed == rebuilt,
        "payload_sha256": (
            canonical_sha256(payload) == reported_payload_sha256
        ),
        "status_pass": observed.get("status") == "PASS",
        "three_profiles": (
            len(cases) == 3
            and all(case.get("pass") is True for case in cases)
        ),
        "semantic_mutations": (
            observed.get("semantic_mutation_gate", {}).get("rejected")
            == 26
        ),
        "strict_boundaries": (
            observed.get("strict_boundary_gate", {}).get("rejected")
            == 5
        ),
        "frozen_v3": all(
            row.get("pass") is True
            for row in observed.get("frozen_v3_baseline", [])
        ),
        "claim_firewall": (
            observed.get("verdict", {}).get("contract_compiler_gate")
            == "PASS"
            and len(
                observed.get("verdict", {}).get(
                    "does_not_authorize",
                    [],
                )
            )
            == 5
        ),
    }
    passed = all(checks.values())
    result = {
        "schema_version": (
            "unitdp.v35_allocation_contract_verification.v1"
        ),
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "report_payload_sha256": reported_payload_sha256,
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
