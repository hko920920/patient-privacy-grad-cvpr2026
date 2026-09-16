from __future__ import annotations

import json
from pathlib import Path

from scripts.build_v34_route_obligation_gate import (
    DEFAULT_OUTPUT,
    build_report,
)


def test_v34_route_obligation_report_rebuilds_exactly() -> None:
    frozen = json.loads(Path(DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    assert frozen == build_report()


def test_v34_route_obligation_claim_boundary() -> None:
    report = build_report()
    assert report["status"] == "PASS"
    assert report["baseline_contracts"]["accepted"] == 6
    assert report["baseline_contracts"]["required"] == 6
    assert report["registry"]["entry_count"] == 2
    assert report["registry"]["unique_nominal_tuple_count"] == 2
    assert report["registry"]["generic_plugin_loader_present"] is False
    assert len(report["obligations"]) == 8
    assert all(row["both_routes_bound"] for row in report["obligations"])
    assert (
        report["verdict"]["L1_bounded_finite_registry_obligation_model"]
        == "PASS"
    )
    assert (
        report["verdict"]["L2_generic_route_privacy_or_correctness"]
        == "FORBIDDEN"
    )
