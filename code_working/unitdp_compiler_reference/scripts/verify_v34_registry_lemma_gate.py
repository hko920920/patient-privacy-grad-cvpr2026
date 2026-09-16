#!/usr/bin/env python3
"""Rebuild and verify the bounded V3.4 registry-separation lemma gate."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from build_v34_registry_lemma_gate import DEFAULT_OUTPUT, build_report


DEFAULT_VERIFICATION = (
    DEFAULT_OUTPUT.parent / "registry_lemma_verification_v34.json"
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
    registry = observed.get("registry", {})
    compilations = observed.get("successful_compilations", {})
    tamper = observed.get("postcompile_tamper_checks", {})
    obligations = observed.get("proof_obligations", [])
    anchors = observed.get("source_anchors", [])
    tamper_rows = tamper.get("rows", [])

    checks = {
        "exact_rebuild": observed == rebuilt,
        "status_pass": observed.get("status") == "PASS",
        "prerequisite_reports_pass": (
            observed.get("prerequisites", {}).get("pass") is True
        ),
        "frozen_unique_registry": (
            registry.get("pass") is True
            and registry.get("dispatch_entry_count") == 2
            and registry.get("unique_nominal_tuple_count") == 2
            and registry.get("dispatch_mapping_immutable") is True
            and registry.get("p_registry_immutable") is True
            and registry.get("f_registry_immutable") is True
        ),
        "six_source_anchors": (
            len(anchors) == 6 and all(row.get("pass") for row in anchors)
        ),
        "six_exact_executable_compilations": (
            compilations.get("passed") == 6
            and compilations.get("required") == 6
            and len(compilations.get("rows", [])) == 6
            and all(
                row.get("all_bindings_pass")
                for row in compilations.get("rows", [])
            )
        ),
        "eighteen_postcompile_tamper_rejections": (
            tamper.get("rejected") == 18
            and tamper.get("required") == 18
            and len(tamper_rows) == 18
            and all(row.get("passed") for row in tamper_rows)
            and Counter(row.get("family") for row in tamper_rows)
            == {
                "compiled_registry_swap": 6,
                "contract_executor_id_swap": 6,
                "contract_accountant_id_swap": 6,
            }
        ),
        "seven_proof_obligations": (
            len(obligations) == 7
            and all(row.get("pass") for row in obligations)
        ),
        "claim_firewall": (
            observed.get("verdict", {}).get(
                "bounded_registry_separation_lemma"
            )
            == "PASS"
            and observed.get("verdict", {}).get("allowed_lemma_wording")
            and len(
                observed.get("verdict", {}).get("claims_forbidden", [])
            )
            == 6
            and len(observed.get("closest_system_boundary", [])) == 4
        ),
    }
    passed = all(checks.values())
    result = {
        "schema_version": "unitdp.v34_registry_lemma_verification.v1",
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
