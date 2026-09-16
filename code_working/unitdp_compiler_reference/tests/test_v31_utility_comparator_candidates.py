from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_v31_utility_comparator_candidates import (  # noqa: E402
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


def test_frozen_comparator_feasibility_report_regenerates_exactly():
    frozen = load_json(DEFAULT_OUTPUT)
    assert build_report() == frozen
    assert frozen["payload_sha256"] == payload_sha256(
        _without_payload(frozen)
    )
    for relative, expected in frozen["source_files_sha256"].items():
        assert file_sha256(ROOT / relative) == expected


def test_comparator_verdicts_preserve_scientific_boundary():
    report = load_json(DEFAULT_OUTPUT)
    _assert_public_report(report)
    verdicts = {
        row["candidate_id"]: row["verdict"]
        for row in report["candidates"]
    }
    assert verdicts == {
        "same_route_clean_room_replay": (
            "eligible_execution_correspondence_only"
        ),
        "route_p_versus_route_f": "ineligible_superiority_comparison",
        "legacy_poisson_mog_pld_els": (
            "ineligible_current_evidence_requires_new_route"
        ),
        "legacy_fixed_size_mog_pld_els": (
            "ineligible_current_evidence_requires_new_route"
        ),
        "legacy_group_privacy_window_fallback": (
            "valid_accounting_idea_but_ineligible_legacy_experiment"
        ),
        "same_route_single_window_policy_ablation": (
            "feasible_new_ablation_not_prior_method_baseline"
        ),
        "new_strict_els_route": (
            "scientifically_possible_only_after_full_new_route"
        ),
    }
    assert report["decision"] == {
        "reuse_legacy_utility_rows": False,
        "rank_route_p_against_route_f": False,
        "current_mechanism_matched_comparator_role": (
            "execution correspondence only"
        ),
        "new_utility_experiment_authorized_by_this_audit": False,
        "manuscript_action": (
            "retain V3.1 results and explicit non-ranking limitation"
        ),
    }
    encoded = json.dumps(report, sort_keys=True)
    for forbidden in (
        "research_seed",
        "seed_13",
        "seed_23",
        "seed_31",
        "seed_37",
        "seed_41",
        "SO" + "GANG",
        "C:" + chr(92),
    ):
        assert forbidden not in encoded
