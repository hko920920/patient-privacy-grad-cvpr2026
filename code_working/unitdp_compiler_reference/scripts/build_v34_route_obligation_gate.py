#!/usr/bin/env python3
"""Build the bounded V3.4 P/F route-obligation feasibility gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unitdp.compiler_srswor_v3 import (  # noqa: E402
    CORE_ROUTE_ID_SRSWOR_V3,
    CompiledOwnerSrsworRouteV3,
    OwnerSrsworContractV3,
    ROUTE_REGISTRY_SRSWOR_V3,
)
from unitdp.compiler_v2 import (  # noqa: E402
    CORE_ROUTE_ID_V2,
    CompiledOwnerPoissonRouteV2,
    OwnerPoissonContractV2,
    ROUTE_REGISTRY_V2,
)
from unitdp.compiler_v3 import (  # noqa: E402
    ROUTE_DISPATCH_REGISTRY_V3,
    parse_registered_contract_v3,
)


DEFAULT_OUTPUT = (
    ROOT
    / "reports"
    / "v34_route_obligation_gate_20260725"
    / "route_obligation_gate_v34.json"
)
DEFAULT_MARKDOWN = DEFAULT_OUTPUT.with_suffix(".md")

P_MUTATION_REPORT = (
    ROOT
    / "reports"
    / "v2_mutation_audit_v32_20260724"
    / "public_mutation_audit_v2.json"
)
F_MUTATION_REPORT = (
    ROOT
    / "reports"
    / "v3_route_mutation_audit_v32_20260724"
    / "public_route_mutation_audit_v3.json"
)
OPACUS_REPORT = (
    ROOT
    / "reports"
    / "opacus_boundary_probe_v1_20260724"
    / "opacus_boundary_probe_v1.json"
)
MECHANISM_REPORT = (
    ROOT
    / "reports"
    / "v3_mechanism_matched_reference_v32_20260724"
    / "public_mechanism_matched_reference_v32.json"
)

DATASETS = ("uci", "wisdm", "sepsis")
CONFIGS = {
    "P": {
        dataset: ROOT / "configs" / "v2" / f"{dataset}_owner_poisson_v2.yaml"
        for dataset in DATASETS
    },
    "F": {
        dataset: ROOT / "configs" / "v3" / f"{dataset}_owner_srswor_v3.yaml"
        for dataset in DATASETS
    },
}


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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected YAML mapping: {path}")
    return value


def case_index(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["case_id"]): row
        for row in report["cases"]
        if isinstance(row, dict) and "case_id" in row
    }


def all_rejected(
    index: dict[str, dict[str, Any]],
    case_ids: list[str],
) -> bool:
    return all(
        case_id in index
        and (
            index[case_id].get("observed_outcome") == "reject"
            or index[case_id].get("observed") == "rejected"
        )
        for case_id in case_ids
    )


def contract_row(
    route: str,
    dataset: str,
    path: Path,
) -> dict[str, Any]:
    raw = load_yaml(path)
    contract = parse_registered_contract_v3(raw)
    dispatch = ROUTE_DISPATCH_REGISTRY_V3[contract.route_id]
    if route == "P":
        specific = ROUTE_REGISTRY_V2[CORE_ROUTE_ID_V2]
        expected_contract_type = OwnerPoissonContractV2
        expected_compiled_type = CompiledOwnerPoissonRouteV2
    else:
        specific = ROUTE_REGISTRY_SRSWOR_V3[
            CORE_ROUTE_ID_SRSWOR_V3
        ]
        expected_contract_type = OwnerSrsworContractV3
        expected_compiled_type = CompiledOwnerSrsworRouteV3

    bindings = {
        "route_id": contract.route_id == dispatch.route_id,
        "schema_version": contract.schema_version == dispatch.schema_version,
        "adjacency": contract.adjacency == dispatch.adjacency,
        "sampler": contract.sampler == dispatch.sampler,
        "contract_type": (
            type(contract).__name__
            == dispatch.contract_type
            == expected_contract_type.__name__
        ),
        "compiled_type": (
            dispatch.compiled_type == expected_compiled_type.__name__
        ),
        "specific_registry_route": contract.route_id == specific.route_id,
        "specific_registry_adjacency": (
            contract.adjacency == specific.adjacency
        ),
        "specific_registry_sampler": contract.sampler == specific.sampler,
        "specific_registry_accountant": (
            contract.accountant_id == specific.accountant_id
        ),
        "specific_registry_executor": (
            contract.implementation_id == specific.implementation_id
        ),
    }
    return {
        "route": route,
        "dataset": dataset,
        "config": str(path.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": file_sha256(path),
        "contract_type": type(contract).__name__,
        "compiled_type": dispatch.compiled_type,
        "route_id": contract.route_id,
        "schema_version": contract.schema_version,
        "privacy_unit": contract.privacy_unit,
        "adjacency": contract.adjacency,
        "sampler": contract.sampler,
        "accountant_id": contract.accountant_id,
        "implementation_id": contract.implementation_id,
        "target_epsilon": contract.target_epsilon,
        "delta": contract.delta,
        "total_steps": contract.total_steps,
        "clip_norm": contract.clip_norm,
        "bindings": bindings,
        "all_bindings_pass": all(bindings.values()),
    }


def obligation_rows(
    p_cases: dict[str, dict[str, Any]],
    f_cases: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    definitions = [
        {
            "id": "O1",
            "name": "strict serialization schema",
            "shared_surface": ["schema_version", "closed nested sections"],
            "p_code": [
                "parse_owner_poisson_contract_v2",
                "_UniqueKeySafeLoader",
            ],
            "f_code": ["parse_owner_srswor_contract_v3"],
            "p_cases": [
                "unknown_top_level_field",
                "unknown_nested_field",
                "duplicate_yaml_key",
                "duplicate_json_key",
            ],
            "f_cases": [
                "unknown_top_level_field",
                "missing_accountant_section",
                "schema_splice",
            ],
            "limit": "Closed schemas cover only the two implemented contract versions.",
        },
        {
            "id": "O2",
            "name": "unique route identity",
            "shared_surface": [
                "route_id",
                "schema_version",
                "contract_type",
                "compiled_type",
            ],
            "p_code": [
                "ROUTE_DISPATCH_REGISTRY_V3",
                "parse_registered_contract_v3",
            ],
            "f_code": [
                "ROUTE_DISPATCH_REGISTRY_V3",
                "parse_registered_contract_v3",
            ],
            "p_cases": [
                "replace_one_adjacency",
                "fixed_size_sampler",
            ],
            "f_cases": [
                "route_id_splice",
                "schema_splice",
                "unknown_route",
            ],
            "limit": "The dispatcher has explicit P/F branches and is not a generic plugin loader.",
        },
        {
            "id": "O3",
            "name": "exact public profile",
            "shared_surface": [
                "benchmark_profile_id",
                "benchmark_profile_sha256",
                "dataset_id",
                "public_protocol",
            ],
            "p_code": [
                "get_benchmark_profile_v2",
                "parse_owner_poisson_contract_v2",
            ],
            "f_code": [
                "get_srswor_benchmark_profile_v3",
                "parse_owner_srswor_contract_v3",
            ],
            "p_cases": [
                "unregistered_benchmark_profile",
                "benchmark_profile_digest_drift",
                "protocol_owner_count_drift",
            ],
            "f_cases": [
                "profile_hash_change",
                "base_profile_swap",
            ],
            "limit": "Exact public fixtures improve reproducibility but are not release-domain totality.",
        },
        {
            "id": "O4",
            "name": "mapping and attribution",
            "shared_surface": [
                "generated-record mapping",
                "single owner per generated record",
                "mapping digest",
            ],
            "p_code": [
                "WindowMapping",
                "compile_owner_poisson_contract_v2",
            ],
            "f_code": [
                "WindowMapping",
                "compile_owner_srswor_contract_v3",
            ],
            "p_cases": [
                "duplicate_row_index",
                "gapped_row_index",
                "multi_owner_row",
                "compiled_selected_mapping_mutation",
            ],
            "f_cases": ["full_mapping_byte_change"],
            "limit": "The current implementation assumes single attribution, not multi-owner records.",
        },
        {
            "id": "O5",
            "name": "owner-local contribution policy",
            "shared_surface": [
                "policy type",
                "owner cap",
                "windows per owner per step",
                "policy scope",
            ],
            "p_code": [
                "ContributionPolicy",
                "compile_owner_poisson_contract_v2",
            ],
            "f_code": [
                "ContributionPolicy",
                "compile_owner_srswor_contract_v3",
            ],
            "p_cases": [
                "per_step_budget_above_cap",
                "cross_owner_policy",
                "compiled_policy_mutation",
            ],
            "f_cases": ["within_owner_budget_change"],
            "limit": "Only the registered support-balanced cap is evaluated.",
        },
        {
            "id": "O6",
            "name": "fixed preprocessing",
            "shared_surface": [
                "preprocessor schema",
                "artifact digest",
                "source-reference digest",
                "in-memory state digest",
            ],
            "p_code": [
                "FixedAffinePreprocessor",
                "compile_owner_poisson_contract_v2",
            ],
            "f_code": [
                "FixedAffinePreprocessor",
                "compile_owner_srswor_contract_v3",
            ],
            "p_cases": [
                "private_fitted_scaler",
                "wrong_preprocessing_artifact",
                "loaded_preprocessor_array_write",
            ],
            "f_cases": ["preprocessor_state_change"],
            "limit": "Protected-data adaptive preprocessing requires a separate privacy analysis.",
        },
        {
            "id": "O7",
            "name": "route-specific mechanism and accountant",
            "shared_surface": [
                "adjacency",
                "sampler",
                "sensitivity convention",
                "noise",
                "steps",
                "accountant",
            ],
            "p_code": [
                "ROUTE_REGISTRY_V2",
                "dual Poisson-Gaussian RDP recomputation",
            ],
            "f_code": [
                "ROUTE_REGISTRY_SRSWOR_V3",
                "library/direct SRSWOR RDP recomputation",
            ],
            "p_cases": [
                "replace_one_adjacency",
                "fixed_size_sampler",
                "unregistered_accountant",
                "profile_noise_drift",
            ],
            "f_cases": [
                "adjacency_swap",
                "sampler_swap",
                "sensitivity_factor_drop",
                "poisson_accountant_reuse",
                "theorem_swap",
                "sensitivity_one_direct_oracle",
            ],
            "limit": "A new route still needs its own valid privacy theorem and implementation.",
        },
        {
            "id": "O8",
            "name": "executor, source, and artifact binding",
            "shared_surface": [
                "implementation id",
                "source-bundle digest",
                "typed public plan",
                "private execution binding",
                "model/run/summary/collection chain",
            ],
            "p_code": [
                "CompiledOwnerPoissonRouteV2.assert_private_execution_integrity",
                "execution_source_bundle_sha256_v2",
            ],
            "f_code": [
                "CompiledOwnerSrsworRouteV3.assert_private_execution_integrity",
                "execution_source_bundle_sha256_srswor_v3",
            ],
            "p_cases": [
                "stale_executor_id",
                "compiled_contract_mutation",
                "compiled_source_bundle_mutation",
            ],
            "f_cases": [
                "executor_swap",
                "poisson_compiled_object_to_srswor_executor",
            ],
            "limit": "Hashes and typed objects assume honest execution and are not attestation.",
        },
    ]
    rows: list[dict[str, Any]] = []
    for definition in definitions:
        p_case_ids = list(definition.pop("p_cases"))
        f_case_ids = list(definition.pop("f_cases"))
        p_pass = all_rejected(p_cases, p_case_ids)
        f_pass = all_rejected(f_cases, f_case_ids)
        rows.append(
            {
                **definition,
                "p_evidence_cases": p_case_ids,
                "f_evidence_cases": f_case_ids,
                "p_evidence_pass": p_pass,
                "f_evidence_pass": f_pass,
                "both_routes_bound": p_pass and f_pass,
            }
        )
    return rows


def build_report() -> dict[str, Any]:
    p_report = load_json(P_MUTATION_REPORT)
    f_report = load_json(F_MUTATION_REPORT)
    opacus = load_json(OPACUS_REPORT)
    mechanism = load_json(MECHANISM_REPORT)
    p_cases = case_index(p_report)
    f_cases = case_index(f_report)

    contracts = [
        contract_row(route, dataset, CONFIGS[route][dataset])
        for route in ("P", "F")
        for dataset in DATASETS
    ]
    dispatch_entries = {
        route_id: asdict(entry)
        for route_id, entry in ROUTE_DISPATCH_REGISTRY_V3.items()
    }
    nominal_tuples = {
        (
            entry.route_id,
            entry.schema_version,
            entry.adjacency,
            entry.sampler,
            entry.accountant_family,
            entry.contract_type,
            entry.compiled_type,
        )
        for entry in ROUTE_DISPATCH_REGISTRY_V3.values()
    }
    obligations = obligation_rows(p_cases, f_cases)

    dispatch_case_ids = (
        "route_id_splice",
        "schema_splice",
        "unknown_route",
    )
    dispatch_rejections = all_rejected(
        f_cases,
        list(dispatch_case_ids),
    )
    executor_cross_type_rejection = all_rejected(
        f_cases,
        ["poisson_compiled_object_to_srswor_executor"],
    )
    opacus_external_fields = opacus["opacus_observations"][
        "route_fields_absent_from_make_private"
    ]
    mechanism_totals = mechanism["totals"]

    l0_pass = (
        len(dispatch_entries) == 2
        and len(nominal_tuples) == 2
        and all(row["all_bindings_pass"] for row in contracts)
        and dispatch_rejections
        and executor_cross_type_rejection
    )
    l1_pass = (
        l0_pass
        and len(obligations) == 8
        and all(row["both_routes_bound"] for row in obligations)
    )

    result: dict[str, Any] = {
        "schema_version": "unitdp.v34_route_obligation_gate.v1",
        "scope": (
            "bounded nominal obligation and finite-registry separation "
            "for the two implemented P/F routes; not generic privacy "
            "soundness, completeness, formal refinement, or attestation"
        ),
        "status": "PASS" if l0_pass and l1_pass else "FAIL",
        "registry": {
            "entry_count": len(dispatch_entries),
            "entries": dispatch_entries,
            "unique_nominal_tuple_count": len(nominal_tuples),
            "mapping_proxy_immutable": (
                type(ROUTE_DISPATCH_REGISTRY_V3).__name__
                == "mappingproxy"
            ),
            "explicit_route_branches": 2,
            "generic_plugin_loader_present": False,
        },
        "baseline_contracts": {
            "accepted": sum(
                row["all_bindings_pass"] for row in contracts
            ),
            "required": len(contracts),
            "rows": contracts,
        },
        "obligations": obligations,
        "separation_evidence": {
            "route_dispatch_case_ids": list(dispatch_case_ids),
            "route_dispatch_rejections": (
                len(dispatch_case_ids) if dispatch_rejections else 0
            ),
            "executor_cross_type_rejection": (
                executor_cross_type_rejection
            ),
            "p_mutation_rejections": (
                p_report["mutation_case_count"]
                if p_report["all_mutations_rejected"]
                else 0
            ),
            "p_mutation_required": p_report["mutation_case_count"],
            "f_mutation_rejections": f_report["observed_rejections"],
            "f_mutation_required": f_report["required_rejections"],
            "mechanism_matched_runs": mechanism_totals["runs"],
            "mechanism_matched_steps": mechanism_totals[
                "diagnostic_steps_exact"
            ],
            "opacus_1_6_external_route_fields": opacus_external_fields,
        },
        "verdict": {
            "L0_fixed_pf_noninterchangeability": (
                "PASS" if l0_pass else "FAIL"
            ),
            "L1_bounded_finite_registry_obligation_model": (
                "PASS" if l1_pass else "FAIL"
            ),
            "L2_generic_route_privacy_or_correctness": "FORBIDDEN",
            "manuscript_action": (
                "A bounded registry-separation program lemma is allowed "
                "for the frozen P/F registry. The manuscript must not "
                "claim that the schema proves a new route private or "
                "correct, and must retain the two-route limitation."
            ),
            "code_refactor_required": False,
            "reason_no_refactor": (
                "The supported statement concerns exact finite-registry "
                "identity. Replacing explicit branches with callables "
                "would not add privacy evidence or scientific generality."
            ),
        },
        "source_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(
                path
            )
            for path in (
                Path(__file__).resolve(),
                ROOT / "src" / "unitdp" / "compiler_v2.py",
                ROOT / "src" / "unitdp" / "compiler_srswor_v3.py",
                ROOT / "src" / "unitdp" / "compiler_v3.py",
                P_MUTATION_REPORT,
                F_MUTATION_REPORT,
                OPACUS_REPORT,
                MECHANISM_REPORT,
                *(
                    CONFIGS[route][dataset]
                    for route in ("P", "F")
                    for dataset in DATASETS
                ),
            )
        },
    }
    result["payload_sha256"] = canonical_sha256(result)
    return result


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# V3.4 Route-Obligation Feasibility Gate",
        "",
        f"Status: **{report['status']}**",
        "",
        "## Verdict",
        "",
        f"- L0 fixed P/F noninterchangeability: "
        f"**{report['verdict']['L0_fixed_pf_noninterchangeability']}**",
        f"- L1 bounded finite-registry obligation model: "
        f"**{report['verdict']['L1_bounded_finite_registry_obligation_model']}**",
        "- L2 generic route privacy/correctness: **FORBIDDEN**",
        "",
        report["verdict"]["manuscript_action"],
        "",
        "## Baselines and registry",
        "",
        f"- Accepted exact contracts: "
        f"{report['baseline_contracts']['accepted']}/"
        f"{report['baseline_contracts']['required']}",
        f"- Frozen registry entries / unique nominal tuples: "
        f"{report['registry']['entry_count']}/"
        f"{report['registry']['unique_nominal_tuple_count']}",
        f"- Generic plugin loader present: "
        f"{report['registry']['generic_plugin_loader_present']}",
        "",
        "## Obligation matrix",
        "",
        "| ID | Obligation | P evidence | F evidence | Both |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in report["obligations"]:
        lines.append(
            f"| {row['id']} | {row['name']} | "
            f"{len(row['p_evidence_cases'])} | "
            f"{len(row['f_evidence_cases'])} | "
            f"{'PASS' if row['both_routes_bound'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Limit",
            "",
            "This gate supports a program-property statement only for the "
            "frozen P/F registry. A new route still requires its own privacy "
            "argument, parser, accountant, executor, and validation.",
            "",
            f"Payload SHA-256: `{report['payload_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.markdown.write_text(
        markdown_report(report),
        encoding="utf-8",
    )
    print(
        f"status={report['status']} "
        f"payload_sha256={report['payload_sha256']}"
    )
    print(f"result={args.output}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
