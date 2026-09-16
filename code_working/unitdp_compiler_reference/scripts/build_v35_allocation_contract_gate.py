#!/usr/bin/env python3
"""Build the V3.5 strict random-allocation contract/compiler gate."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unitdp.compiler_allocation_v4 import (  # noqa: E402
    AllocationContractV4Error,
    AllocationExecutorNotReadyV4Error,
    compile_owner_random_allocation_contract_v4,
    load_owner_random_allocation_contract_v4,
    parse_owner_random_allocation_contract_v4,
)
from unitdp.source_bundle_allocation_v4 import (  # noqa: E402
    compiler_source_bundle_allocation_v4,
    compiler_source_bundle_sha256_allocation_v4,
)


DEFAULT_OUTPUT = (
    ROOT
    / "reports"
    / "v35_allocation_contract_gate_20260725"
    / "allocation_contract_gate_v35.json"
)
DEFAULT_MARKDOWN = DEFAULT_OUTPUT.with_suffix(".md")
SCHEMA_VERSION = "unitdp.v35_allocation_contract_gate.v1"
REPORT_DATE = "2026-07-25"

CASES = (
    {
        "profile": "uci",
        "config": "configs/v4/uci_owner_random_allocation_v4.yaml",
        "mapping": (
            "reports/uci_public_preprocessing_smoke_20260724/"
            "uci_train_mapping.csv"
        ),
        "preprocessor": (
            "configs/preprocessing/"
            "uci_har_published_train_standard_scaler_v1.json"
        ),
        "population": 21,
        "selected_records": 336,
        "expected_order": 4,
    },
    {
        "profile": "wisdm",
        "config": "configs/v4/wisdm_owner_random_allocation_v4.yaml",
        "mapping": (
            "reports/wisdm_public_protocol_smoke_20260724/"
            "wisdm_train_mapping.csv"
        ),
        "preprocessor": (
            "configs/preprocessing/"
            "wisdm_v1_1_train_owners_1_25_stats24_standard_scaler_v1.json"
        ),
        "population": 25,
        "selected_records": 400,
        "expected_order": 4,
    },
    {
        "profile": "sepsis",
        "config": "configs/v4/sepsis_owner_random_allocation_v4.yaml",
        "mapping": (
            "reports/sepsis_public_protocol_smoke_20260724/"
            "sepsis_train_mapping.csv"
        ),
        "preprocessor": (
            "configs/preprocessing/"
            "physionet2019_setA_first400_basic12x6_public_scaler_v1.json"
        ),
        "population": 312,
        "selected_records": 1507,
        "expected_order": 3,
    },
)

FROZEN_BASELINE = {
    "src/unitdp/compiler_v3.py": (
        "9a083ecd83943503ceb0d7b061296084c754e6fb6218b2786af85a2896aaa3d0"
    ),
    "dist/compiler_submission_upload_manifest_v3.json": (
        "d5fa835d92cdbb69c8dc4832c032cbc46f53e9a5f53eae6a5906b27bd5bdecbe"
    ),
    "dist/compiler_code_and_data_supplement_aaai27_v3.zip": (
        "7214060b29c91530c67c8c202647be495c5011e6c131839d9e5c271382d4c3b3"
    ),
}

CLAIM_SOURCES = (
    "configs/v4/sepsis_owner_random_allocation_v4.yaml",
    "configs/v4/uci_owner_random_allocation_v4.yaml",
    "configs/v4/wisdm_owner_random_allocation_v4.yaml",
    "scripts/build_v35_allocation_contract_gate.py",
    "scripts/verify_v35_allocation_contract_gate.py",
    "src/unitdp/benchmark_registry_allocation_v4.py",
    "src/unitdp/compiler_allocation_v4.py",
    "src/unitdp/random_allocation_accountant_v4.py",
    "src/unitdp/source_bundle_allocation_v4.py",
    "src/unitdp_spec_oracle/random_allocation_oracle_v4.py",
    "tests/test_compiler_allocation_v4.py",
    "tests/test_random_allocation_accountant_v4.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _set_nested(
    raw: dict[str, Any],
    path: tuple[str, ...],
    value: object,
) -> dict[str, Any]:
    result = copy.deepcopy(raw)
    cursor: dict[str, Any] = result
    for key in path[:-1]:
        child = cursor[key]
        if not isinstance(child, dict):
            raise TypeError(path)
        cursor = child
    cursor[path[-1]] = value
    return result


def _compile_case(case: dict[str, Any]) -> dict[str, Any]:
    contract = load_owner_random_allocation_contract_v4(
        ROOT / case["config"]
    )
    route = compile_owner_random_allocation_contract_v4(
        contract,
        mapping_path=ROOT / case["mapping"],
        preprocessing_artifact_path=ROOT / case["preprocessor"],
    )
    checks = {
        "population_bound": (
            len(route.source_mapping.by_owner()) == case["population"]
        ),
        "selected_mapping_bound": (
            len(route.selected_mapping.records)
            == case["selected_records"]
        ),
        "contract_not_execution_ready": not route.execution_ready,
        "target_met": (
            route.accountant_epsilon
            <= contract.target_epsilon + 1e-10
        ),
        "production_oracle_agree": (
            max(
                abs(route.accountant_epsilon - route.oracle_epsilon),
                abs(
                    route.accountant_epsilon_remove
                    - route.oracle_epsilon_remove
                ),
                abs(
                    route.accountant_epsilon_add
                    - route.oracle_epsilon_add
                ),
            )
            <= 2e-12
        ),
        "optimal_order": (
            route.accountant_optimal_remove_order
            == case["expected_order"]
        ),
        "public_plan_self_authenticates": (
            route.public_plan()["public_plan_sha256"]
            == route.public_plan_sha256
        ),
    }
    return {
        "profile": case["profile"],
        "config": case["config"],
        "benchmark_profile_id": contract.benchmark_profile_id,
        "benchmark_profile_sha256": (
            contract.benchmark_profile_sha256
        ),
        "public_contract_sha256": contract.public_contract_sha256,
        "public_plan_sha256": route.public_plan_sha256,
        "compiler_source_bundle_sha256": (
            route.compiler_source_bundle_sha256
        ),
        "private_compilation_binding_sha256": (
            route.private_compilation_binding_sha256
        ),
        "query": {
            "num_steps": contract.num_steps,
            "num_selected_steps_per_owner": (
                contract.num_selected_steps_per_owner
            ),
            "num_epochs": contract.num_epochs,
            "noise_multiplier": contract.noise_multiplier,
            "delta": contract.delta,
        },
        "accounting": {
            "epsilon": route.accountant_epsilon,
            "epsilon_remove": route.accountant_epsilon_remove,
            "epsilon_add": route.accountant_epsilon_add,
            "oracle_epsilon": route.oracle_epsilon,
            "oracle_epsilon_remove": route.oracle_epsilon_remove,
            "oracle_epsilon_add": route.oracle_epsilon_add,
            "optimal_remove_order": (
                route.accountant_optimal_remove_order
            ),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def _mutation_gate() -> dict[str, Any]:
    config = ROOT / CASES[0]["config"]
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("UCI contract must load as a mapping")
    mutations = (
        ("adjacency", ("privacy", "adjacency"), "replace_one_owner"),
        ("target_epsilon", ("privacy", "target_epsilon"), 9.0),
        ("delta", ("privacy", "delta"), 1e-6),
        ("sampler", ("mechanism", "sampler"), "independent_bernoulli_owner"),
        ("population", ("mechanism", "source_dataset_size"), 20),
        ("num_steps", ("mechanism", "num_steps"), 16),
        ("num_selected", ("mechanism", "num_selected_steps_per_owner"), 5),
        ("num_epochs", ("mechanism", "num_epochs"), 2),
        ("total_steps", ("mechanism", "total_steps"), 16),
        ("schedule", ("mechanism", "assignment_schedule"), "per_step"),
        ("noise", ("mechanism", "noise_multiplier"), 1.25),
        ("denominator", ("mechanism", "update_denominator"), 8.4),
        ("empty_step", ("mechanism", "empty_step_behavior"), "skip_update"),
        ("contribution", ("mechanism", "per_owner_contribution"), "other"),
        ("accountant", ("accountant", "id"), "poisson"),
        ("sensitivity", ("accountant", "sensitivity_multiplier"), 2.0),
        ("theorem", ("accountant", "theorem_id"), "other"),
        ("oracle", ("accountant", "oracle_implementation_id"), "other"),
        ("reduction", ("accountant", "reduction_id"), "other"),
        ("direction", ("accountant", "direction_policy"), "remove_only"),
        ("executor", ("execution", "implementation_id"), "other"),
        ("public_coins", ("execution", "random_coins_public"), True),
        ("profile_hash", ("data", "benchmark_profile_sha256"), "0" * 64),
        ("base_profile", ("data", "base_data_profile_id"), "wisdm_aaai27_v1"),
        (
            "policy",
            ("contribution_policy", "windows_per_owner_per_step"),
            7,
        ),
        ("optimizer", ("optimization", "learning_rate"), 0.1),
    )
    rows: list[dict[str, Any]] = []
    for name, path, value in mutations:
        rejected = False
        error_type = ""
        try:
            parse_owner_random_allocation_contract_v4(
                _set_nested(raw, path, value)
            )
        except AllocationContractV4Error as exc:
            rejected = True
            error_type = type(exc).__name__
        rows.append(
            {
                "name": name,
                "path": list(path),
                "rejected": rejected,
                "error_type": error_type,
            }
        )
    return {
        "required": len(rows),
        "rejected": sum(row["rejected"] for row in rows),
        "rows": rows,
        "pass": all(row["rejected"] for row in rows),
    }


def _boundary_gate() -> dict[str, Any]:
    case = CASES[0]
    config = ROOT / case["config"]
    contract = load_owner_random_allocation_contract_v4(config)
    duplicate_text = config.read_text(encoding="utf-8").replace(
        "  num_steps: 15\n",
        "  num_steps: 15\n  num_steps: 15\n",
        1,
    )
    results: dict[str, bool] = {}
    with tempfile.TemporaryDirectory() as tmp:
        temporary = Path(tmp)
        duplicate = temporary / "duplicate.yaml"
        duplicate.write_text(duplicate_text, encoding="utf-8")
        try:
            load_owner_random_allocation_contract_v4(duplicate)
        except AllocationContractV4Error:
            results["duplicate_key"] = True
        else:
            results["duplicate_key"] = False

        mapping = ROOT / case["mapping"]
        mutated_mapping = temporary / "mapping.csv"
        with mapping.open("r", newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            rows = list(reader)
            columns = list(reader.fieldnames or [])
        rows[-1]["scenario"] += "_mutated"
        with mutated_mapping.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        try:
            compile_owner_random_allocation_contract_v4(
                contract,
                mapping_path=mutated_mapping,
                preprocessing_artifact_path=ROOT / case["preprocessor"],
            )
        except AllocationContractV4Error:
            results["mapping_one_field"] = True
        else:
            results["mapping_one_field"] = False

    unknown = copy.deepcopy(contract.public_payload())
    unknown["unexpected"] = True
    try:
        parse_owner_random_allocation_contract_v4(unknown)
    except AllocationContractV4Error:
        results["unknown_top_level_field"] = True
    else:
        results["unknown_top_level_field"] = False

    spliced = copy.deepcopy(contract.public_payload())
    spliced["route_id"] = "owa_owner_srswor_rdp_replace_one_v3"
    try:
        parse_owner_random_allocation_contract_v4(spliced)
    except AllocationContractV4Error:
        results["cross_route_splice"] = True
    else:
        results["cross_route_splice"] = False

    try:
        compile_owner_random_allocation_contract_v4(
            contract,
            mapping_path=ROOT / case["mapping"],
            preprocessing_artifact_path=ROOT / case["preprocessor"],
            require_executable=True,
        )
    except AllocationExecutorNotReadyV4Error:
        results["premature_execution"] = True
    else:
        results["premature_execution"] = False

    return {
        "required": len(results),
        "rejected": sum(results.values()),
        "checks": results,
        "pass": all(results.values()),
    }


def build_report() -> dict[str, Any]:
    cases = [_compile_case(case) for case in CASES]
    mutation_gate = _mutation_gate()
    boundary_gate = _boundary_gate()
    source_bundle = dict(compiler_source_bundle_allocation_v4())
    frozen = [
        {
            "path": path,
            "expected_sha256": expected,
            "observed_sha256": file_sha256(ROOT / path),
            "pass": file_sha256(ROOT / path) == expected,
        }
        for path, expected in FROZEN_BASELINE.items()
    ]
    claim_sources = {
        path: file_sha256(ROOT / path) for path in CLAIM_SOURCES
    }
    checks = {
        "three_registered_profiles": (
            len(cases) == 3 and all(case["pass"] for case in cases)
        ),
        "semantic_mutations": mutation_gate["pass"],
        "strict_boundaries": boundary_gate["pass"],
        "one_compiler_bundle": (
            len(
                {
                    case["compiler_source_bundle_sha256"]
                    for case in cases
                }
            )
            == 1
            and compiler_source_bundle_sha256_allocation_v4(
                source_bundle
            )
            == cases[0]["compiler_source_bundle_sha256"]
        ),
        "frozen_v3_preserved": all(row["pass"] for row in frozen),
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "report_date": REPORT_DATE,
        "candidate": "AAAI27 Algorithm V3.5",
        "object": (
            "strict owner random-allocation registration and compilation"
        ),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "cases": cases,
        "semantic_mutation_gate": mutation_gate,
        "strict_boundary_gate": boundary_gate,
        "compiler_source_bundle": source_bundle,
        "compiler_source_bundle_sha256": (
            compiler_source_bundle_sha256_allocation_v4(source_bundle)
        ),
        "frozen_v3_baseline": frozen,
        "claim_source_sha256": claim_sources,
        "verdict": {
            "contract_compiler_gate": (
                "PASS" if all(checks.values()) else "FAIL"
            ),
            "authorizes": (
                "implementation of the dedicated Route-A executor"
            ),
            "does_not_authorize": [
                "an execution-correspondence claim",
                "a utility result",
                "a privacy proof beyond the cited finite-order bound",
                "an extensibility claim for the generic dispatcher",
                "a manuscript novelty claim",
            ],
        },
    }
    report["payload_sha256"] = canonical_sha256(report)
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V3.5 Random-Allocation Contract/Compiler Gate",
        "",
        f"- Status: `{report['status']}`",
        f"- Payload SHA-256: `{report['payload_sha256']}`",
        f"- Profiles: `{len(report['cases'])}/3`",
        (
            "- Semantic mutations rejected: "
            f"`{report['semantic_mutation_gate']['rejected']}/"
            f"{report['semantic_mutation_gate']['required']}`"
        ),
        (
            "- Strict boundary checks: "
            f"`{report['strict_boundary_gate']['rejected']}/"
            f"{report['strict_boundary_gate']['required']}`"
        ),
        (
            "- Frozen V3 artifacts preserved: "
            f"`{sum(row['pass'] for row in report['frozen_v3_baseline'])}/"
            f"{len(report['frozen_v3_baseline'])}`"
        ),
        "",
        "This gate authorizes only the dedicated Route-A executor stage.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--markdown",
        type=Path,
        default=DEFAULT_MARKDOWN,
    )
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["checks"], indent=2, sort_keys=True))
    print(report["payload_sha256"])
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
