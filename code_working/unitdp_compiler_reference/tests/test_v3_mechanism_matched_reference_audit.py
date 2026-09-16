from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_v3_mechanism_matched_reference_audit import (  # noqa: E402
    DEFAULT_OUTPUT,
    MechanismMatchedReferenceError,
    _assert_public_report,
    load_json,
    payload_sha256,
)
from verify_v3_mechanism_matched_reference_audit import (  # noqa: E402
    verify_integrity,
)


def _production_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "unitdp" or node.module.startswith("unitdp."):
                imports.append(node.module)
        if isinstance(node, ast.Import):
            imports.extend(
                alias.name
                for alias in node.names
                if alias.name == "unitdp"
                or alias.name.startswith("unitdp.")
            )
    return imports


def test_reference_executors_import_no_production_package():
    assert _production_imports(
        ROOT / "tests" / "owner_poisson_trace_reference.py"
    ) == []
    assert _production_imports(
        ROOT / "tests" / "owner_srswor_trace_reference.py"
    ) == []


def test_frozen_mechanism_matched_report_integrity_and_counts():
    report = verify_integrity(DEFAULT_OUTPUT)
    assert report["route_count"] == 2
    assert report["totals"] == {
        "datasets": 6,
        "runs": 30,
        "diagnostic_steps_exact": 850,
        "final_models_bitwise": 30,
        "final_tensors_bitwise": 60,
        "public_metrics_exact": 120,
        "maximum_public_metric_gap": 0.0,
    }
    assert all(
        route["reference_imports_production_executor"] is False
        for route in report["routes"]
    )


def test_public_report_redaction_and_forgery_rejection(tmp_path: Path):
    report = load_json(DEFAULT_OUTPUT)
    _assert_public_report(report)
    encoded = json.dumps(report, sort_keys=True)
    for forbidden in (
        "research_seed",
        "sampled_owner_positions",
        "selected_rows",
        "step_diagnostics",
        "SO" + "GANG",
        "C:" + chr(92),
    ):
        assert forbidden not in encoded

    forged = copy.deepcopy(report)
    forged["totals"]["final_models_bitwise"] = 29
    forged_path = tmp_path / "forged.json"
    forged_path.write_text(
        json.dumps(forged, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(MechanismMatchedReferenceError):
        verify_integrity(forged_path)

    repaired_digest = copy.deepcopy(forged)
    repaired_digest["payload_sha256"] = payload_sha256(
        {
            key: value
            for key, value in repaired_digest.items()
            if key != "payload_sha256"
        }
    )
    repaired_path = tmp_path / "repaired_forgery.json"
    repaired_path.write_text(
        json.dumps(repaired_digest, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(MechanismMatchedReferenceError):
        verify_integrity(repaired_path)
