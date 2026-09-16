#!/usr/bin/env python3
"""Independently rebuild and verify the V3.4 matched-shell report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from build_v34_matched_shell_gate import (
    DEFAULT_OUTPUT,
    EXPECTED_CORRECT_REPLACE_ONE_EPSILON,
    EXPECTED_DIFFERENT_PATHS,
    EXPECTED_OMITTED_2C_EPSILON,
    EXPECTED_ONLY_F_PATHS,
    EXPECTED_ONLY_P_PATHS,
    EXPECTED_SHARED_PATHS,
    MUTATION_SPECS,
    build_report,
)


DEFAULT_VERIFICATION = (
    DEFAULT_OUTPUT.parent / "matched_shell_verification_v34.json"
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
    decomposition = observed.get("paired_contract_decomposition", {})
    substitutions = observed.get("cross_route_substitutions", {})
    counterfactual = observed.get("sensitivity_counterfactual", {})
    cases = substitutions.get("cases", [])
    expected_families = Counter(
        {spec["family"]: 6 for spec in MUTATION_SPECS}
    )

    checks = {
        "exact_rebuild": observed == rebuilt,
        "status_pass": observed.get("status") == "PASS",
        "six_exact_baselines": (
            observed.get("baseline_contracts", {}).get("accepted") == 6
            and observed.get("baseline_contracts", {}).get("required") == 6
        ),
        "three_exact_pair_decompositions": (
            decomposition.get("pairs_passed") == 3
            and decomposition.get("pairs_required") == 3
            and all(
                row.get("shared_leaf_paths")
                == list(EXPECTED_SHARED_PATHS)
                and row.get("route_different_leaf_paths")
                == list(EXPECTED_DIFFERENT_PATHS)
                and row.get("only_p_leaf_paths")
                == list(EXPECTED_ONLY_P_PATHS)
                and row.get("only_f_leaf_paths")
                == list(EXPECTED_ONLY_F_PATHS)
                for row in decomposition.get("rows", [])
            )
        ),
        "forty_eight_nonvacuous_shell_preserving_rejections": (
            substitutions.get("required") == 48
            and substitutions.get("rejected") == 48
            and substitutions.get("passed") == 48
            and substitutions.get("all_shared_shells_preserved") is True
            and substitutions.get("all_changed_nonvacuously") is True
            and len(cases) == 48
            and all(case.get("passed") for case in cases)
        ),
        "eight_prespecified_families_bidirectional": (
            Counter(substitutions.get("family_counts", {}))
            == expected_families
            and substitutions.get("direction_counts")
            == {"F_from_P": 24, "P_from_F": 24}
            and substitutions.get("dataset_counts")
            == {"sepsis": 16, "uci": 16, "wisdm": 16}
        ),
        "named_rejection_stages": (
            substitutions.get("rejection_stage_counts")
            == {
                "route_dispatch": 12,
                "route_specific_contract": 36,
            }
            and all(
                case.get("observed_rejection_stage")
                == case.get("expected_rejection_stage")
                for case in cases
            )
        ),
        "sensitivity_counterfactual_exact": (
            counterfactual.get("correct_replace_one_epsilon")
            == EXPECTED_CORRECT_REPLACE_ONE_EPSILON
            and counterfactual.get("epsilon_if_2C_factor_were_omitted")
            == EXPECTED_OMITTED_2C_EPSILON
            and counterfactual.get("exact_binding_pass") is True
            and counterfactual.get(
                "sensitivity_factor_drop_case", {}
            ).get("observed_outcome")
            == "reject"
        ),
        "bounded_claim_firewall": (
            observed.get("verdict", {}).get(
                "finite_registered_noninterchangeability"
            )
            == "PASS"
            and len(
                observed.get("verdict", {}).get("claims_forbidden", [])
            )
            >= 5
        ),
    }
    passed = all(checks.values())
    result = {
        "schema_version": "unitdp.v34_matched_shell_verification.v1",
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
