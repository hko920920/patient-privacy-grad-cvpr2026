#!/usr/bin/env python3
"""Independently verify the sealed G8 route, gate, and cost reports."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
CANDIDATE_MODULES = ROOT / "v36_candidate" / "src" / "unitdp"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import unitdp  # noqa: E402

if str(CANDIDATE_MODULES) not in unitdp.__path__:
    unitdp.__path__.append(str(CANDIDATE_MODULES))

import build_g8_external_handler_cost as cost_builder  # noqa: E402
import build_g8_external_handler_gate as gate_builder  # noqa: E402
from build_v35_handler_hybrid_ablation_gate import (  # noqa: E402
    SURFACES,
    hybrid,
)
from unitdp.compiler_v36 import (  # noqa: E402
    EXTERNAL_PLD_HANDLER_V36,
    ROUTE_DISPATCH_REGISTRY_V36,
    compile_registered_contract_v36,
    load_registered_contract_v36,
)
from unitdp.compiler_v4 import (  # noqa: E402
    ALLOCATION_HANDLER_V4,
    POISSON_HANDLER_V4,
    REQUIRED_REGISTRATION_OBLIGATIONS_V4,
    SRSWOR_HANDLER_V4,
    CompilerDispatchV4Error,
    validate_route_handler_v4,
)
from unitdp.external_pld_backend_v36 import (  # noqa: E402
    BACKEND_SITE_ENV_V36,
    verify_external_pld_environment_v36,
)


EVIDENCE_PATH = (
    ROOT / "reports/g8_external_handler_evidence_v2_20260725/"
    "external_pld_handler_evidence_v36.json"
)
GATE_PATH = (
    ROOT / "reports/g8_external_handler_gate_v2_20260725/"
    "g8_external_handler_gate_v2.json"
)
COST_PATH = (
    ROOT / "reports/g8_external_handler_cost_v2_20260725/"
    "g8_external_handler_cost_v2.json"
)
OUTPUT = GATE_PATH.parent / "g8_external_handler_verification_v2.json"
HANDLERS = (
    ("P", POISSON_HANDLER_V4),
    ("F", SRSWOR_HANDLER_V4),
    ("A", ALLOCATION_HANDLER_V4),
    ("H", EXTERNAL_PLD_HANDLER_V36),
)


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_hashed_report(path: Path) -> tuple[dict[str, Any], bool]:
    value = json.loads(path.read_text(encoding="utf-8"))
    without = dict(value)
    reported = without.pop("payload_sha256", None)
    return value, canonical_sha256(without) == reported


def verify_preservation(gate: dict[str, Any]) -> bool:
    expected_rows = {
        (row["route"], row["dataset"]): (
            row["public_contract_sha256"],
            row["public_plan_sha256"],
        )
        for row in gate["preservation"]["rows"]
    }
    observed: dict[tuple[str, str], tuple[str, str]] = {}
    for route in gate_builder.ROUTES:
        for dataset in gate_builder.DATASETS:
            contract = load_registered_contract_v36(
                gate_builder.config_path(route, dataset)
            )
            compiled = compile_registered_contract_v36(
                contract,
                mapping_path=gate_builder.ARTIFACTS[dataset]["mapping"],
                preprocessing_artifact_path=(
                    gate_builder.ARTIFACTS[dataset]["preprocessor"]
                ),
                require_executable=True,
            )
            observed[(route, dataset)] = (
                contract.public_contract_sha256,
                compiled.public_plan_sha256,
            )
    return observed == expected_rows and observed == gate_builder.FROZEN_OUTPUTS


def verify_hybrid_lattice(gate: dict[str, Any]) -> bool:
    reported_rows = gate["handler_hybrids"]["rows"]
    reported = {
        (
            row["pair"],
            row["mask"],
            row["registration_sha256"],
        )
        for row in reported_rows
        if (row["weak_plugin_accept"] and not row["sealed_validator_accept"])
    }
    observed: set[tuple[str, int, str]] = set()
    for (left_label, left), (right_label, right) in combinations(
        HANDLERS,
        2,
    ):
        pair = f"{left_label}/{right_label}"
        for mask in range(1, 2 ** len(SURFACES) - 1):
            candidate = hybrid(left, right, mask)
            rejected = False
            try:
                validate_route_handler_v4(candidate)
            except CompilerDispatchV4Error:
                rejected = True
            if not rejected:
                return False
            observed.add((pair, mask, candidate.registration_sha256))
    return len(observed) == 756 and observed == reported


def verify_witness_deletion(gate: dict[str, Any]) -> bool:
    observed: set[str] = set()
    witnesses = EXTERNAL_PLD_HANDLER_V36.obligation_witnesses
    for removed in witnesses:
        candidate = replace(
            EXTERNAL_PLD_HANDLER_V36,
            obligation_witnesses=tuple(
                witness
                for witness in witnesses
                if witness.obligation_id != removed.obligation_id
            ),
        )
        try:
            validate_route_handler_v4(candidate)
        except CompilerDispatchV4Error:
            observed.add(removed.obligation_id)
    reported = {
        row["removed"] for row in gate["witness_deletion"]["rows"] if row["rejected"]
    }
    return observed == reported == REQUIRED_REGISTRATION_OBLIGATIONS_V4


def verify_execution_files(evidence: dict[str, Any]) -> bool:
    manifest = evidence["execution"]["manifest"]
    directory = EVIDENCE_PATH.parent
    for key in ("public", "private"):
        row = manifest["files"][key]
        if file_sha256(directory / row["path"]) != row["file_sha256"]:
            return False
    manifest_path = directory / "uci_external_pld_execution_manifest_v36.json"
    return (
        file_sha256(manifest_path) == evidence["execution"]["manifest_file_sha256"]
        and canonical_sha256(
            {key: value for key, value in manifest.items() if key != "payload_sha256"}
        )
        == manifest["payload_sha256"]
    )


def verify_cost_sources(cost: dict[str, Any]) -> bool:
    rows = (
        cost["source_cost"]["production_rows"]
        + cost["source_cost"]["test_and_gate_rows"]
        + cost["source_cost"]["config_rows"]
    )
    return all(
        file_sha256(ROOT / row["path"]) == row["sha256"]
        and cost_builder.disclosed_logical_lines(ROOT / row["path"])
        == row["logical_lines"]
        for row in rows
    )


def verify_claim_sources(report: dict[str, Any]) -> bool:
    rows = report.get("claim_source_sha256", {})
    return bool(rows) and all(
        file_sha256(ROOT / relative) == expected for relative, expected in rows.items()
    )


def verify_evidence_source_bundles(evidence: dict[str, Any]) -> bool:
    bundles = evidence.get("source_bundles", {})
    for name in ("compiler", "execution"):
        rows = bundles.get(name, {})
        if not rows or not all(
            file_sha256(ROOT / relative) == expected
            for relative, expected in rows.items()
        ):
            return False
        if canonical_sha256(rows) != bundles.get(f"{name}_sha256"):
            return False
    return True


def build_verification() -> dict[str, Any]:
    evidence, evidence_payload = load_hashed_report(EVIDENCE_PATH)
    gate, gate_payload = load_hashed_report(GATE_PATH)
    cost, cost_payload = load_hashed_report(COST_PATH)
    endpoint_valid = True
    for _, handler in HANDLERS:
        try:
            validate_route_handler_v4(handler)
        except CompilerDispatchV4Error:
            endpoint_valid = False
    frozen_files_valid = all(
        file_sha256(ROOT / row["path"])
        == row["expected_sha256"]
        == row["observed_sha256"]
        for row in gate["frozen_files"]
    )
    hybrid_summary_valid = (
        gate["handler_hybrids"]["required"] == 756
        and gate["handler_hybrids"]["unique"] == 756
        and gate["handler_hybrids"]["weak_accepted"] == 756
        and gate["handler_hybrids"]["sealed_rejected"] == 756
        and gate["handler_hybrids"]["pair_count"] == 6
    )
    tamper_valid = gate["tamper_rejections"]["total"] >= 10 and all(
        row["rejected"]
        for group in (
            gate["tamper_rejections"]["environment"],
            gate["tamper_rejections"]["response"],
            gate["tamper_rejections"]["registration"],
        )
        for row in group
    )
    latency_hashes = {
        row["dataset"]: row["response_sha256"] for row in cost["backend_latency"]
    }
    evidence_hashes = {
        row["dataset"]: row["external_response_sha256"]
        for row in evidence["compilations"]
    }
    checks = {
        "route_evidence_status": evidence["status"] == "PASS",
        "route_evidence_payload": evidence_payload,
        "route_evidence_claim_sources_exact": verify_claim_sources(evidence),
        "route_evidence_source_bundles_exact": (
            verify_evidence_source_bundles(evidence)
        ),
        "G8_gate_status": gate["status"] == "PASS",
        "G8_gate_payload": gate_payload,
        "G8_gate_claim_sources_exact": verify_claim_sources(gate),
        "cost_status": cost["status"] == "PASS",
        "cost_payload": cost_payload,
        "four_endpoints_validate": endpoint_valid,
        "four_registry_entries": (len(ROUTE_DISPATCH_REGISTRY_V36) == 4),
        "H_core_matches_evidence": (
            EXTERNAL_PLD_HANDLER_V36.registration_core_sha256
            == evidence["registration_core_sha256"]
        ),
        "H_exact_obligations": (
            {
                witness.obligation_id
                for witness in EXTERNAL_PLD_HANDLER_V36.obligation_witnesses
            }
            == REQUIRED_REGISTRATION_OBLIGATIONS_V4
        ),
        "frozen_files_exact": frozen_files_valid,
        "nine_old_outputs_recompiled_exact": verify_preservation(gate),
        "seven_witness_deletions_rejected": verify_witness_deletion(gate),
        "hybrid_summary_exact": hybrid_summary_valid,
        "hybrid_lattice_recomputed": verify_hybrid_lattice(gate),
        "tamper_rows_rejected": tamper_valid,
        "three_H_compilations_and_repeats": (
            len(evidence["compilations"]) == 3
            and all(
                row["pass"] and row["two_clean_responses_byte_identical"]
                for row in evidence["compilations"]
            )
        ),
        "UCI_execution_files_exact": verify_execution_files(evidence),
        "cost_source_rows_exact": verify_cost_sources(cost),
        "cost_response_hashes_match_evidence": (latency_hashes == evidence_hashes),
        "cost_reports_no_existing_production_changes": (
            cost["source_cost"]["existing_production_files_changed"] == 0
        ),
        "cost_reports_no_formula_copy_or_torch": (
            not cost["source_cost"]["formula_checks"]["production_imports_torch"]
            and cost["source_cost"]["formula_checks"][
                "forbidden_accountant_formula_names_absent"
            ]
        ),
        "external_environment_still_exact": bool(
            os.environ.get(BACKEND_SITE_ENV_V36)
            and verify_external_pld_environment_v36()
        ),
        "cost_cache_backup_absent": not (resolve_cost_backup()).exists(),
    }
    return {
        "schema_version": ("unitdp.g8_external_handler_verification.v2"),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "passed": sum(checks.values()),
        "required": len(checks),
        "input_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): (file_sha256(path))
            for path in (EVIDENCE_PATH, GATE_PATH, COST_PATH)
        },
    }


def resolve_cost_backup() -> Path:
    site = Path(os.environ[BACKEND_SITE_ENV_V36]).resolve()
    return site.parent / "numba_cache_g8_cost_backup"


def main() -> None:
    verification = build_verification()
    verification["payload_sha256"] = canonical_sha256(verification)
    OUTPUT.write_text(
        json.dumps(
            verification,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": verification["status"],
                "passed": verification["passed"],
                "required": verification["required"],
                "payload_sha256": verification["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    if verification["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
