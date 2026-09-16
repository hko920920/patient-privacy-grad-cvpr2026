from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from scripts.build_v34_matched_shell_gate import (
    DEFAULT_OUTPUT,
    EXPECTED_DIFFERENT_PATHS,
    EXPECTED_ONLY_F_PATHS,
    EXPECTED_ONLY_P_PATHS,
    EXPECTED_SHARED_PATHS,
    MUTATION_SPECS,
    build_report,
)


def test_v34_matched_shell_report_rebuilds_exactly() -> None:
    frozen = json.loads(Path(DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    assert frozen == build_report()


def test_v34_matched_shell_decomposition_is_exact() -> None:
    report = build_report()
    assert report["status"] == "PASS"
    assert report["baseline_contracts"]["accepted"] == 6
    decomposition = report["paired_contract_decomposition"]
    assert decomposition["pairs_passed"] == 3
    for row in decomposition["rows"]:
        assert row["shared_leaf_paths"] == list(EXPECTED_SHARED_PATHS)
        assert row["route_different_leaf_paths"] == list(
            EXPECTED_DIFFERENT_PATHS
        )
        assert row["only_p_leaf_paths"] == list(EXPECTED_ONLY_P_PATHS)
        assert row["only_f_leaf_paths"] == list(EXPECTED_ONLY_F_PATHS)


def test_v34_all_prespecified_splices_fail_before_execution() -> None:
    report = build_report()
    substitutions = report["cross_route_substitutions"]
    assert substitutions["required"] == 48
    assert substitutions["rejected"] == 48
    assert substitutions["passed"] == 48
    assert substitutions["all_shared_shells_preserved"] is True
    assert substitutions["all_changed_nonvacuously"] is True
    assert Counter(substitutions["family_counts"]) == Counter(
        {spec["family"]: 6 for spec in MUTATION_SPECS}
    )
    assert all(
        case["observed_rejection_stage"]
        == case["expected_rejection_stage"]
        for case in substitutions["cases"]
    )


def test_v34_matched_shell_claim_is_bounded() -> None:
    report = build_report()
    assert (
        report["verdict"]["finite_registered_noninterchangeability"]
        == "PASS"
    )
    assert len(report["verdict"]["claims_forbidden"]) >= 5
    counterfactual = report["sensitivity_counterfactual"]
    assert counterfactual["exact_binding_pass"] is True
    assert (
        counterfactual["sensitivity_factor_drop_case"]["observed_outcome"]
        == "reject"
    )
