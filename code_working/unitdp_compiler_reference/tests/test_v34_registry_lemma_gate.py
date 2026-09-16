from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from scripts.build_v34_registry_lemma_gate import (
    DEFAULT_OUTPUT,
    build_report,
)


REPORT = build_report()


def test_v34_registry_lemma_report_rebuilds_exactly() -> None:
    frozen = json.loads(Path(DEFAULT_OUTPUT).read_text(encoding="utf-8"))
    assert frozen == REPORT


def test_v34_registry_lemma_binds_all_six_compilations() -> None:
    assert REPORT["status"] == "PASS"
    registry = REPORT["registry"]
    assert registry["pass"] is True
    assert registry["dispatch_entry_count"] == 2
    assert registry["unique_nominal_tuple_count"] == 2
    compiled = REPORT["successful_compilations"]
    assert compiled["passed"] == compiled["required"] == 6
    assert all(row["all_bindings_pass"] for row in compiled["rows"])


def test_v34_registry_lemma_rejects_postcompile_route_drift() -> None:
    tamper = REPORT["postcompile_tamper_checks"]
    assert tamper["rejected"] == tamper["required"] == 18
    assert all(row["passed"] for row in tamper["rows"])
    assert Counter(row["family"] for row in tamper["rows"]) == {
        "compiled_registry_swap": 6,
        "contract_executor_id_swap": 6,
        "contract_accountant_id_swap": 6,
    }


def test_v34_registry_lemma_is_program_property_only() -> None:
    assert len(REPORT["proof_obligations"]) == 7
    assert all(row["pass"] for row in REPORT["proof_obligations"])
    assert (
        REPORT["verdict"]["bounded_registry_separation_lemma"] == "PASS"
    )
    assert len(REPORT["verdict"]["claims_forbidden"]) == 6
    assert len(REPORT["closest_system_boundary"]) == 4
