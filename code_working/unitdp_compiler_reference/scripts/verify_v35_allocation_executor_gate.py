#!/usr/bin/env python3
"""Rebuild and verify the V3.5 Route-A executor gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_v35_allocation_executor_gate import (
    DEFAULT_OUTPUT,
    build_report,
    canonical_sha256,
)


DEFAULT_VERIFICATION = (
    DEFAULT_OUTPUT.parent / "allocation_executor_verification_v35.json"
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
    checks = {
        "exact_rebuild": observed == rebuilt,
        "payload_sha256": (
            canonical_sha256(payload) == reported_payload_sha256
        ),
        "status_pass": observed.get("status") == "PASS",
        "correspondence_7_of_7": (
            sum(observed.get("correspondence", {}).values()) == 7
        ),
        "execution_invariants_10_of_10": (
            sum(observed.get("execution_invariants", {}).values())
            == 10
        ),
        "empty_step": (
            observed.get("fixture", {}).get("empty_steps") == [12]
        ),
        "model_hash_match": (
            observed.get("fixture", {}).get("production_model_sha256")
            == observed.get("fixture", {}).get("reference_model_sha256")
        ),
        "joint_support": (
            observed.get("joint_support_probe", {}).get("cells_observed")
            == 16
            and observed.get("joint_support_probe", {}).get("pass")
            is True
        ),
        "frozen_v3": all(
            row.get("pass") is True
            for row in observed.get("frozen_v3_baseline", [])
        ),
        "claim_firewall": (
            len(
                observed.get("verdict", {}).get(
                    "does_not_authorize",
                    [],
                )
            )
            == 6
        ),
    }
    passed = all(checks.values())
    result = {
        "schema_version": (
            "unitdp.v35_allocation_executor_verification.v1"
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
