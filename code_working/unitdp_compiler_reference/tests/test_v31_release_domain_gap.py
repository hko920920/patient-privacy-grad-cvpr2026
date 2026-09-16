from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_v31_release_domain_gap import (  # noqa: E402
    DEFAULT_OUTPUT,
    _assert_public_report,
    build_report,
    file_sha256,
    load_json,
    payload_sha256,
)


def _without_payload(report: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in report.items()
        if key != "payload_sha256"
    }


def test_frozen_release_domain_report_regenerates_exactly():
    frozen = load_json(DEFAULT_OUTPUT)
    assert build_report() == frozen
    assert frozen["payload_sha256"] == payload_sha256(
        _without_payload(frozen)
    )
    for relative, expected in frozen["source_files_sha256"].items():
        assert file_sha256(ROOT / relative) == expected


def test_release_domain_boundary_remains_fail_closed_and_nonrelease():
    report = load_json(DEFAULT_OUTPUT)
    _assert_public_report(report)
    assert report["contract_audit"] == {
        "active_contracts": 6,
        "registered_profiles": {"P": 3, "F": 3},
        "research_compiles": {"P": 3, "F": 3},
        "exact_public_preprocessing_only": True,
        "protected_data_preprocessing_admitted": False,
    }
    assert report["release_request_audit"] == {
        "route_p": {
            "schema_parse_accepts": 3,
            "executable_requests_denied": 3,
            "denial_stage": "executable_backend_approval",
            "release_executor_available": False,
        },
        "route_f": {
            "schema_parse_accepts": 0,
            "contract_requests_denied": 3,
            "denial_stage": "exact_contract_binding",
            "release_executor_available": False,
        },
        "all_release_requests_fail_closed": True,
    }
    executions = report["public_execution_audit"]
    assert executions["executions"] == 30
    assert executions["by_route"] == {"P": 15, "F": 15}
    for field in (
        "current_source_bundle_matches",
        "research_non_release",
        "accountant_only_claim_status",
        "secure_rng_false",
        "random_coins_not_disclosed",
        "public_fixed_preprocessing",
        "public_metric_records",
    ):
        assert executions[field] == 30
    assert executions["public_metric_values"] == 120
    assert executions["blocked_field_violations"] == 0
    assert report["release_artifact_boundary"][
        "approved_release_implementations"
    ] == 0
    assert report["source_boundary"][
        "strict_executors_integrating_secure_factory"
    ] == 0
    assert report["source_boundary"][
        "secure_adapter_security_mode"
    ] == "system_csprng_prototype"
    assert report["source_boundary"][
        "registered_multiple_release_ledger_components"
    ] == 0
    assert report["decision"] == {
        "risk_disposition": (
            "contained_by_fail_closed_denial_not_closed_as_deployment"
        ),
        "generic_release_claim_authorized": False,
        "current_release_requests": "deny",
        "current_research_artifacts": (
            "30_of_30_explicitly_research_non_release"
        ),
        "exact_profile_binding_action": "retain",
        "prototype_secure_adapter_action": (
            "do_not_register_with_strict_executors"
        ),
        "manuscript_action": (
            "retain_V3.1_scope_boundary_without_new_release_claim"
        ),
        "new_manuscript_candidate_required": False,
        "full_solution_requirement": (
            "new_joint_release_route_covering_profile_authority_"
            "preprocessing_secure_randomness_outputs_and_composition"
        ),
    }
    encoded = json.dumps(report, sort_keys=True)
    for forbidden in (
        '"seed"',
        "SO" + "GANG",
        "C:" + chr(92),
        "selected_owners",
        "selected_windows",
        "selection_diagnostics",
    ):
        assert forbidden not in encoded
