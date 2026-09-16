#!/usr/bin/env python3
"""Rebuild and verify the V3.5 pairwise hybrid lattice gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_v35_pairwise_hybrid_lattice_gate import (
    DEFAULT_OUTPUT,
    build_report,
    canonical_sha256,
)


DEFAULT_VERIFICATION = (
    DEFAULT_OUTPUT.parent
    / "pairwise_hybrid_lattice_verification_v35.json"
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
    rows = observed.get("pair_dataset_rows", [])
    summary = observed.get("summary", {})
    checks = {
        "exact_rebuild": observed == rebuilt,
        "payload_sha256": (
            canonical_sha256(payload) == reported_payload_sha256
        ),
        "status_pass": observed.get("status") == "PASS",
        "partition": (
            observed.get("partition", {}).get("pass") is True
            and observed.get("partition", {}).get("surface_count") == 8
        ),
        "nine_lattices": (
            len(rows) == 9
            and all(row.get("pass") is True for row in rows)
        ),
        "all_surfaces_different": all(
            all(
                surface.get("different") is True
                for surface in row.get("surface_rows", [])
            )
            for row in rows
        ),
        "all_2286_rejected": (
            summary.get("required_nonendpoint_hybrids")
            == summary.get("rejected_nonendpoint_hybrids")
            == 2286
            and summary.get("accepted_nonendpoint_hybrids") == 0
        ),
        "unique_payloads": all(
            row.get("unique_hybrid_payloads") == 254
            for row in rows
        ),
        "complete_transcripts": all(
            isinstance(
                row.get("rejection_transcript_sha256"),
                str,
            )
            and len(row["rejection_transcript_sha256"]) == 64
            for row in rows
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
            "unitdp.v35_pairwise_hybrid_lattice_verification.v1"
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
