from __future__ import annotations

import json
import hashlib
from pathlib import Path

from scripts.build_v35_random_allocation_accountant_gate import (
    DEFAULT_OUTPUT,
    FROZEN_BASELINE,
    ROOT,
    build_report,
)


FROZEN = json.loads(Path(DEFAULT_OUTPUT).read_text(encoding="utf-8"))
BASELINE_FILES_AVAILABLE = all(
    (ROOT / relative).is_file() for relative in FROZEN_BASELINE
)
REPORT = build_report() if BASELINE_FILES_AVAILABLE else FROZEN


def test_v35_accountant_report_is_bound_or_rebuilds_exactly() -> None:
    if BASELINE_FILES_AVAILABLE:
        assert FROZEN == REPORT
        return

    payload = dict(FROZEN)
    reported = payload.pop("payload_sha256")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == reported
    for relative, expected in FROZEN["source_sha256"].items():
        source = ROOT / relative
        assert source.is_file()
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected


def test_v35_accountant_has_three_bidirectional_oracle_matches() -> None:
    assert REPORT["status"] == "PASS"
    assert len(REPORT["cases"]) == 3
    assert all(case["pass"] for case in REPORT["cases"])
    assert REPORT["accountant"]["directions"] == ["remove", "add"]
    assert REPORT["accountant"]["reported_epsilon"] == "maximum_of_both_directions"


def test_v35_accountant_keeps_external_pld_non_authorizing() -> None:
    assert REPORT["external_probe"]["role"] == "comparison_only_not_route_authorization"
    for case in REPORT["cases"]:
        assert case["external_pld_probe"]["authorizes_route"] is False
        assert (
            case["external_pld_probe"]["epsilon_upper"] < case["production"]["epsilon"]
        )


def test_v35_accountant_query_identity_and_invalid_inputs() -> None:
    identity = REPORT["query_identity_checks"]
    invalid = REPORT["invalid_query_checks"]
    assert identity["pass"] is True
    assert identity["passed"] == identity["required"] == 6
    assert invalid["pass"] is True
    assert invalid["rejected"] == invalid["required"] == 13
    floor_row = next(
        row for row in identity["rows"] if row["name"] == "num_steps_floor_equivalent"
    )
    assert floor_row["floor_reduction_same_numeric_bound"] is True


def test_v35_accountant_gate_has_bounded_authorization() -> None:
    assert REPORT["verdict"]["accountant_gate"] == "PASS"
    assert (
        REPORT["verdict"]["authorizes_next_step"]
        == "A contract/compiler implementation"
    )
    assert len(REPORT["verdict"]["does_not_authorize"]) == 6
