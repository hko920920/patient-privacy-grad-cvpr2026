#!/usr/bin/env python3
"""Rebuild and verify the V3.5 handler-registry gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_v35_handler_registry_gate import (
    DEFAULT_OUTPUT,
    build_report,
    canonical_sha256,
)


DEFAULT_VERIFICATION = (
    DEFAULT_OUTPUT.parent / "handler_registry_verification_v35.json"
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
        "three_handlers": (
            len(observed.get("registry_rows", [])) == 3
            and all(
                row.get("pass") is True
                for row in observed.get("registry_rows", [])
            )
        ),
        "seven_obligations": (
            len(
                observed.get(
                    "required_registration_obligations",
                    [],
                )
            )
            == 7
        ),
        "nine_compilations": (
            len(observed.get("compilation_rows", [])) == 9
            and all(
                row.get("pass") is True
                for row in observed.get("compilation_rows", [])
            )
        ),
        "six_conservative_extensions": (
            len(
                observed.get("conservative_extension_rows", [])
            )
            == 6
            and all(
                row.get("pass") is True
                for row in observed.get(
                    "conservative_extension_rows",
                    [],
                )
            )
        ),
        "eight_negative_registrations": (
            len(observed.get("negative_registration_rows", [])) == 8
            and all(
                row.get("rejected") is True
                for row in observed.get(
                    "negative_registration_rows",
                    [],
                )
            )
        ),
        "branch_free": (
            observed.get("generic_branch_check", {}).get("pass")
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
            "unitdp.v35_handler_registry_verification.v1"
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
