#!/usr/bin/env python3
"""Verify the Audit V14 package manifest and designed-evidence closure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.json"
REPORT = ROOT / "reports" / "validator_ablation_v3_1_200_001"
GENERATED = ROOT / "paper" / "supplement_v14_generated"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    require(MANIFEST.is_file(), "MANIFEST.json is missing")
    manifest = load_json(MANIFEST)
    require(
        manifest.get("schema_version") == "audit_anonymous_artifact_manifest_v14",
        "Unexpected manifest schema",
    )
    rows = manifest.get("files", [])
    indexed = {row["path"]: row for row in rows}
    require(len(indexed) == len(rows), "Duplicate manifest paths")
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path != MANIFEST
    }
    require(actual == set(indexed), "Manifest membership mismatch")
    for relative, row in indexed.items():
        path = ROOT / relative
        require(path.stat().st_size == int(row["bytes"]), f"Size mismatch: {relative}")
        require(sha256_file(path) == row["sha256"], f"Hash mismatch: {relative}")

    required = {
        "paper_notes/AAAI27_AUDIT_V19_CONFORMANCE_200_PROTOCOL_V1_2026-07-28.md",
        "scripts/run_validator_ablation_v3.py",
        "scripts/run_validator_ablation_v3_1.py",
        "scripts/verify_validator_ablation_v3.py",
        "scripts/verify_validator_ablation_v3_2.py",
        "scripts/build_validator_ablation_v3_1_supplement_tables.py",
        "reports/validator_ablation_v3_200_001/cases.json",
        "reports/validator_ablation_v3_1_200_001/cases.json",
        "reports/validator_ablation_v3_1_200_001/aggregate_metrics.csv",
        "reports/validator_ablation_v3_1_200_001/artifact_index.json",
        "reports/validator_ablation_v3_1_200_001/independent_verification.json",
        "paper/supplement_v14_generated/provenance.json",
        "paper/supplement_v14_generated/ablation_metrics.tex",
        "paper/supplement_v14_generated/allowed_grid.tex",
        "paper/supplement_v14_generated/operator_matrix.tex",
        "paper/supplement_v14_generated/artifact_inputs.tex",
    }
    require(required <= actual, f"Required files absent: {sorted(required - actual)}")

    raw_inputs = [
        path
        for path in (REPORT / "case_inputs").rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".csv"}
    ]
    require(len(raw_inputs) == 600, f"Expected 600 raw case inputs, got {len(raw_inputs)}")

    verification = load_json(REPORT / "independent_verification.json")
    require(verification.get("passed") is True, "Independent verifier did not pass")
    require(verification.get("check_count") == 20909, "Verifier count changed")
    require(verification.get("failure_count") == 0, "Verifier failures are nonzero")
    require(
        verification.get("standard_library_only") is True,
        "Verifier is not standard-library-only",
    )

    provenance = load_json(GENERATED / "provenance.json")
    require(
        provenance.get("counts")
        == {
            "allowed_cases": 100,
            "artifact_index_files": 627,
            "full_mapping_hashes": 100,
            "independent_checks": 20909,
            "non_allow_cases": 100,
            "operators": 25,
            "raw_case_input_files": 600,
            "selected_semantic_mapping_hashes": 50,
        },
        "Generated-table counts changed",
    )
    for row in provenance["inputs"].values():
        path = ROOT / row["path"]
        require(path.is_file(), f"Generated-table input absent: {row['path']}")
        require(path.stat().st_size == row["bytes"], f"Input size mismatch: {row['path']}")
        require(sha256_file(path) == row["sha256"], f"Input hash mismatch: {row['path']}")
    for name, row in provenance["outputs"].items():
        path = GENERATED / name
        require(path.is_file(), f"Generated table absent: {name}")
        require(path.stat().st_size == row["bytes"], f"Output size mismatch: {name}")
        require(sha256_file(path) == row["sha256"], f"Output hash mismatch: {name}")

    print(
        json.dumps(
            {
                "status": "PASS",
                "manifest_files": len(rows),
                "raw_case_inputs": len(raw_inputs),
                "independent_checks": verification["check_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
