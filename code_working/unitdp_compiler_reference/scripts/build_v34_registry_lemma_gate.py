#!/usr/bin/env python3
"""Build the bounded V3.4 finite-registry separation-lemma gate."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unitdp.benchmark_registry_srswor_v3 import (  # noqa: E402
    ACCOUNTANT_ID_SRSWOR_V3,
    CONTRACT_SCHEMA_SRSWOR_V3,
    CORE_ROUTE_ID_SRSWOR_V3,
    EXECUTOR_IMPLEMENTATION_ID_SRSWOR_V3,
)
from unitdp.compiler_srswor_v3 import (  # noqa: E402
    CompiledOwnerSrsworRouteV3,
    OwnerSrsworContractV3,
    ROUTE_REGISTRY_SRSWOR_V3,
    compile_owner_srswor_contract_v3,
)
from unitdp.compiler_v2 import (  # noqa: E402
    ACCOUNTANT_ID_V2,
    CONTRACT_SCHEMA_V2,
    CORE_ROUTE_ID_V2,
    EXECUTOR_IMPLEMENTATION_ID_V2,
    CompiledOwnerPoissonRouteV2,
    OwnerPoissonContractV2,
    ROUTE_REGISTRY_V2,
    compile_owner_poisson_contract_v2,
)
from unitdp.compiler_v3 import (  # noqa: E402
    ROUTE_DISPATCH_REGISTRY_V3,
    compile_registered_contract_v3,
    load_registered_contract_v3,
    parse_registered_contract_v3,
)


DEFAULT_OUTPUT = (
    ROOT
    / "reports"
    / "v34_registry_lemma_gate_20260725"
    / "registry_lemma_gate_v34.json"
)
DEFAULT_MARKDOWN = DEFAULT_OUTPUT.with_suffix(".md")

OBLIGATION_REPORT = (
    ROOT
    / "reports"
    / "v34_route_obligation_gate_20260725"
    / "route_obligation_gate_v34.json"
)
MATCHED_SHELL_REPORT = (
    ROOT
    / "reports"
    / "v34_matched_shell_gate_20260725"
    / "matched_shell_gate_v34.json"
)
DATASETS = ("uci", "wisdm", "sepsis")
ARTIFACTS = {
    "uci": {
        "mapping": (
            ROOT
            / "reports"
            / "uci_public_preprocessing_smoke_20260724"
            / "uci_train_mapping.csv"
        ),
        "preprocessor": (
            ROOT
            / "configs"
            / "preprocessing"
            / "uci_har_published_train_standard_scaler_v1.json"
        ),
    },
    "wisdm": {
        "mapping": (
            ROOT
            / "reports"
            / "wisdm_public_protocol_smoke_20260724"
            / "wisdm_train_mapping.csv"
        ),
        "preprocessor": (
            ROOT
            / "configs"
            / "preprocessing"
            / "wisdm_v1_1_train_owners_1_25_stats24_standard_scaler_v1.json"
        ),
    },
    "sepsis": {
        "mapping": (
            ROOT
            / "reports"
            / "sepsis_public_protocol_smoke_20260724"
            / "sepsis_train_mapping.csv"
        ),
        "preprocessor": (
            ROOT
            / "configs"
            / "preprocessing"
            / "physionet2019_setA_first400_basic12x6_public_scaler_v1.json"
        ),
    },
}

ROUTE_SPECS = {
    "P": {
        "config_dir": "v2",
        "config_suffix": "owner_poisson_v2",
        "route_id": CORE_ROUTE_ID_V2,
        "schema_version": CONTRACT_SCHEMA_V2,
        "accountant_family": "poisson_gaussian_rdp",
        "accountant_id": ACCOUNTANT_ID_V2,
        "implementation_id": EXECUTOR_IMPLEMENTATION_ID_V2,
        "contract_type": OwnerPoissonContractV2,
        "compiled_type": CompiledOwnerPoissonRouteV2,
        "specific_registry": ROUTE_REGISTRY_V2,
    },
    "F": {
        "config_dir": "v3",
        "config_suffix": "owner_srswor_v3",
        "route_id": CORE_ROUTE_ID_SRSWOR_V3,
        "schema_version": CONTRACT_SCHEMA_SRSWOR_V3,
        "accountant_family": "srswor_gaussian_rdp",
        "accountant_id": ACCOUNTANT_ID_SRSWOR_V3,
        "implementation_id": EXECUTOR_IMPLEMENTATION_ID_SRSWOR_V3,
        "contract_type": OwnerSrsworContractV3,
        "compiled_type": CompiledOwnerSrsworRouteV3,
        "specific_registry": ROUTE_REGISTRY_SRSWOR_V3,
    },
}

ANCHOR_SPECS: tuple[
    tuple[str, Callable[..., Any], tuple[str, ...]], ...
] = (
    (
        "exact_dispatch_and_strict_parse",
        parse_registered_contract_v3,
        (
            "entry = ROUTE_DISPATCH_REGISTRY_V3[route_id]",
            "if schema_version != entry.schema_version:",
            "return parse_owner_poisson_contract_v2(raw_value)",
            "return parse_owner_srswor_contract_v3(raw_value)",
        ),
    ),
    (
        "typed_compile_dispatch",
        compile_registered_contract_v3,
        (
            "if isinstance(contract, OwnerPoissonContractV2):",
            "return compile_owner_poisson_contract_v2(",
            "if isinstance(contract, OwnerSrsworContractV3):",
            "return compile_owner_srswor_contract_v3(",
        ),
    ),
    (
        "p_canonical_compile_and_integrity",
        compile_owner_poisson_contract_v2,
        (
            "contract = parse_owner_poisson_contract_v2(",
            "registry = get_route_registry_entry_v2(contract.route_id)",
            "compiled = CompiledOwnerPoissonRouteV2(",
            "compiled.assert_private_execution_integrity()",
            "if require_executable and not compiled.execution_ready:",
        ),
    ),
    (
        "p_postcompile_revalidation",
        CompiledOwnerPoissonRouteV2.assert_private_execution_integrity,
        (
            "canonical_contract = parse_owner_poisson_contract_v2(",
            "get_route_registry_entry_v2(self.contract.route_id)",
            "self.registry_entry",
            "self.public_plan()",
        ),
    ),
    (
        "f_canonical_compile_and_integrity",
        compile_owner_srswor_contract_v3,
        (
            "contract = parse_owner_srswor_contract_v3(",
            "registry = get_route_registry_entry_srswor_v3(",
            "compiled = CompiledOwnerSrsworRouteV3(",
            "compiled.assert_private_execution_integrity()",
            "if require_executable and not compiled.execution_ready:",
        ),
    ),
    (
        "f_postcompile_revalidation",
        CompiledOwnerSrsworRouteV3.assert_private_execution_integrity,
        (
            "canonical = parse_owner_srswor_contract_v3(",
            "get_route_registry_entry_srswor_v3(",
            "self.registry_entry",
            "self.public_plan()",
        ),
    ),
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


def path_key(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def jsonable(value: object) -> object:
    return json.loads(
        json.dumps(value, sort_keys=True, ensure_ascii=False)
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def config_path(route: str, dataset: str) -> Path:
    spec = ROUTE_SPECS[route]
    return (
        ROOT
        / "configs"
        / str(spec["config_dir"])
        / f"{dataset}_{spec['config_suffix']}.yaml"
    )


def function_anchor(
    name: str,
    function: Callable[..., Any],
    required_fragments: tuple[str, ...],
) -> dict[str, Any]:
    source = inspect.getsource(function)
    source_lines, start_line = inspect.getsourcelines(function)
    source_path_value = inspect.getsourcefile(function)
    if source_path_value is None:
        raise RuntimeError(f"source path unavailable for {name}")
    source_path = Path(source_path_value).resolve()
    missing = [
        fragment for fragment in required_fragments if fragment not in source
    ]
    return {
        "name": name,
        "source_path": path_key(source_path),
        "start_line": start_line,
        "end_line": start_line + len(source_lines) - 1,
        "function_source_sha256": hashlib.sha256(
            source.encode("utf-8")
        ).hexdigest(),
        "required_fragments": list(required_fragments),
        "missing_fragments": missing,
        "pass": not missing,
    }


def registry_snapshot() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    nominal_tuples: set[tuple[str, ...]] = set()
    for route in ("P", "F"):
        spec = ROUTE_SPECS[route]
        route_id = str(spec["route_id"])
        dispatch = ROUTE_DISPATCH_REGISTRY_V3[route_id]
        specific = spec["specific_registry"][route_id]
        nominal = (
            dispatch.route_id,
            dispatch.schema_version,
            dispatch.adjacency,
            dispatch.sampler,
            dispatch.accountant_family,
            dispatch.contract_type,
            dispatch.compiled_type,
        )
        nominal_tuples.add(nominal)
        bindings = {
            "dispatch_key_equals_route_id": dispatch.route_id == route_id,
            "schema_version": (
                dispatch.schema_version == spec["schema_version"]
            ),
            "adjacency_matches_specific_registry": (
                dispatch.adjacency == specific.adjacency
            ),
            "sampler_matches_specific_registry": (
                dispatch.sampler == specific.sampler
            ),
            "accountant_family": (
                dispatch.accountant_family == spec["accountant_family"]
            ),
            "accountant_id": (
                specific.accountant_id == spec["accountant_id"]
            ),
            "implementation_id": (
                specific.implementation_id == spec["implementation_id"]
            ),
            "contract_type": (
                dispatch.contract_type
                == spec["contract_type"].__name__
            ),
            "compiled_type": (
                dispatch.compiled_type
                == spec["compiled_type"].__name__
            ),
        }
        rows.append(
            {
                "route": route,
                "dispatch_entry": asdict(dispatch),
                "specific_registry_entry": jsonable(asdict(specific)),
                "bindings": bindings,
                "all_bindings_pass": all(bindings.values()),
            }
        )
    pass_status = (
        isinstance(ROUTE_DISPATCH_REGISTRY_V3, MappingProxyType)
        and isinstance(ROUTE_REGISTRY_V2, MappingProxyType)
        and isinstance(ROUTE_REGISTRY_SRSWOR_V3, MappingProxyType)
        and len(ROUTE_DISPATCH_REGISTRY_V3) == 2
        and len(ROUTE_REGISTRY_V2) == 1
        and len(ROUTE_REGISTRY_SRSWOR_V3) == 1
        and len(nominal_tuples) == 2
        and all(row["all_bindings_pass"] for row in rows)
    )
    return {
        "dispatch_mapping_immutable": isinstance(
            ROUTE_DISPATCH_REGISTRY_V3, MappingProxyType
        ),
        "p_registry_immutable": isinstance(
            ROUTE_REGISTRY_V2, MappingProxyType
        ),
        "f_registry_immutable": isinstance(
            ROUTE_REGISTRY_SRSWOR_V3, MappingProxyType
        ),
        "dispatch_entry_count": len(ROUTE_DISPATCH_REGISTRY_V3),
        "p_specific_entry_count": len(ROUTE_REGISTRY_V2),
        "f_specific_entry_count": len(ROUTE_REGISTRY_SRSWOR_V3),
        "unique_nominal_tuple_count": len(nominal_tuples),
        "rows": rows,
        "pass": pass_status,
    }


def compiled_row(route: str, dataset: str) -> tuple[dict[str, Any], Any]:
    spec = ROUTE_SPECS[route]
    source_config = config_path(route, dataset)
    contract = load_registered_contract_v3(source_config)
    compiled = compile_registered_contract_v3(
        contract,
        mapping_path=ARTIFACTS[dataset]["mapping"],
        preprocessing_artifact_path=ARTIFACTS[dataset]["preprocessor"],
        require_executable=True,
    )
    dispatch = ROUTE_DISPATCH_REGISTRY_V3[contract.route_id]
    specific = spec["specific_registry"][contract.route_id]
    compiled.assert_private_execution_integrity()
    public_plan = compiled.public_plan()
    bindings = {
        "contract_route_id": contract.route_id == dispatch.route_id,
        "contract_schema_version": (
            contract.schema_version == dispatch.schema_version
        ),
        "contract_adjacency": contract.adjacency == dispatch.adjacency,
        "contract_sampler": contract.sampler == dispatch.sampler,
        "contract_type": type(contract).__name__ == dispatch.contract_type,
        "compiled_type": type(compiled).__name__ == dispatch.compiled_type,
        "compiled_contract": compiled.contract == contract,
        "compiled_registry": compiled.registry_entry == specific,
        "accountant_id": contract.accountant_id == specific.accountant_id,
        "executor_id": (
            contract.implementation_id == specific.implementation_id
        ),
        "execution_ready": compiled.execution_ready is True,
        "public_plan_contract": (
            public_plan["contract"] == contract.public_payload()
        ),
        "public_plan_registry": (
            jsonable(public_plan["route_registry"])
            == jsonable(asdict(specific))
        ),
        "integrity_check": True,
    }
    return (
        {
            "route": route,
            "dataset": dataset,
            "config": path_key(source_config),
            "mapping": path_key(ARTIFACTS[dataset]["mapping"]),
            "preprocessor": path_key(
                ARTIFACTS[dataset]["preprocessor"]
            ),
            "contract_type": type(contract).__name__,
            "compiled_type": type(compiled).__name__,
            "route_id": contract.route_id,
            "schema_version": contract.schema_version,
            "adjacency": contract.adjacency,
            "sampler": contract.sampler,
            "accountant_family": dispatch.accountant_family,
            "accountant_id": contract.accountant_id,
            "implementation_id": contract.implementation_id,
            "public_plan_sha256": compiled.public_plan_sha256,
            "private_execution_binding_sha256": (
                compiled.private_execution_binding_sha256
            ),
            "bindings": bindings,
            "all_bindings_pass": all(bindings.values()),
        },
        compiled,
    )


def tamper_cases(route: str, dataset: str, compiled: Any) -> list[dict[str, Any]]:
    other_route = "F" if route == "P" else "P"
    spec = ROUTE_SPECS[route]
    other = ROUTE_SPECS[other_route]
    other_registry = other["specific_registry"][other["route_id"]]
    mutations = (
        (
            "compiled_registry_swap",
            replace(compiled, registry_entry=other_registry),
        ),
        (
            "contract_executor_id_swap",
            replace(
                compiled,
                contract=replace(
                    compiled.contract,
                    implementation_id=other["implementation_id"],
                ),
            ),
        ),
        (
            "contract_accountant_id_swap",
            replace(
                compiled,
                contract=replace(
                    compiled.contract,
                    accountant_id=other["accountant_id"],
                ),
            ),
        ),
    )
    rows: list[dict[str, Any]] = []
    for family, mutated in mutations:
        observed = "accept"
        error_type = None
        error_message = None
        try:
            mutated.assert_private_execution_integrity()
        except Exception as error:
            observed = "reject"
            error_type = type(error).__name__
            error_message = str(error)
        rows.append(
            {
                "case_id": f"{route.lower()}_{dataset}_{family}",
                "route": route,
                "dataset": dataset,
                "family": family,
                "required_outcome": "reject",
                "observed_outcome": observed,
                "stage": "compiled_integrity",
                "exception_type": error_type,
                "exception_message": error_message,
                "passed": observed == "reject",
                "target_specific_registry_type": type(
                    spec["specific_registry"][spec["route_id"]]
                ).__name__,
            }
        )
    return rows


def build_report() -> dict[str, Any]:
    obligation_report = load_json(OBLIGATION_REPORT)
    matched_shell_report = load_json(MATCHED_SHELL_REPORT)
    anchors = [
        function_anchor(name, function, fragments)
        for name, function, fragments in ANCHOR_SPECS
    ]
    registry = registry_snapshot()

    compiled_rows: list[dict[str, Any]] = []
    tamper_rows: list[dict[str, Any]] = []
    for route in ("P", "F"):
        for dataset in DATASETS:
            row, compiled = compiled_row(route, dataset)
            compiled_rows.append(row)
            tamper_rows.extend(tamper_cases(route, dataset, compiled))

    prerequisite_pass = (
        obligation_report.get("status") == "PASS"
        and matched_shell_report.get("status") == "PASS"
    )
    anchors_pass = len(anchors) == 6 and all(
        row["pass"] for row in anchors
    )
    compiled_pass = (
        len(compiled_rows) == 6
        and all(row["all_bindings_pass"] for row in compiled_rows)
    )
    tamper_pass = (
        len(tamper_rows) == 18
        and all(row["passed"] for row in tamper_rows)
    )

    obligations = [
        {
            "id": "R1",
            "statement": (
                "The dispatch registry is immutable, has exactly two "
                "entries, and their nominal tuples are unique."
            ),
            "pass": registry["pass"],
        },
        {
            "id": "R2",
            "statement": (
                "Exact route lookup and schema equality precede either "
                "route-specific strict parser."
            ),
            "pass": anchors[0]["pass"],
        },
        {
            "id": "R3",
            "statement": (
                "The parsed contract class selects exactly one of the P/F "
                "compile branches."
            ),
            "pass": anchors[1]["pass"],
        },
        {
            "id": "R4",
            "statement": (
                "Each route compiler canonicalizes the contract, resolves "
                "its route-specific registry, builds the typed plan, and "
                "runs its integrity check before return."
            ),
            "pass": (
                anchors[2]["pass"]
                and anchors[4]["pass"]
                and compiled_pass
            ),
        },
        {
            "id": "R5",
            "statement": (
                "Successful executable compilation binds the nominal tuple, "
                "contract/plan types, accountant id, and executor id."
            ),
            "pass": compiled_pass,
        },
        {
            "id": "R6",
            "statement": (
                "Postcompile integrity revalidates canonical contracts and "
                "rejects stale registries or swapped executor/accountant ids."
            ),
            "pass": (
                anchors[3]["pass"]
                and anchors[5]["pass"]
                and tamper_pass
            ),
        },
        {
            "id": "R7",
            "statement": (
                "The obligation and matched-shell prerequisite reports are "
                "both current PASS inputs."
            ),
            "pass": prerequisite_pass,
        },
    ]
    all_obligations_pass = all(row["pass"] for row in obligations)
    status_pass = (
        prerequisite_pass
        and anchors_pass
        and registry["pass"]
        and compiled_pass
        and tamper_pass
        and all_obligations_pass
    )

    allowed_lemma = (
        "Fix the current P/F dispatch registry and its two route-specific "
        "registries. On every terminating call, exact parse followed by "
        "executable compilation either raises or returns a plan. Whenever it "
        "returns, there is a unique selected entry whose route id, schema, "
        "adjacency, sampler, contract type, and compiled type match that "
        "plan, and whose route-specific accountant and executor identities "
        "match the compiled contract."
    )
    report: dict[str, Any] = {
        "schema_version": "unitdp.v34_registry_lemma_gate.v1",
        "scope": (
            "source-bound program property of the frozen two-entry P/F "
            "registry; not a privacy or semantic-refinement theorem"
        ),
        "status": "PASS" if status_pass else "FAIL",
        "prerequisites": {
            "route_obligation_report_status": obligation_report.get(
                "status"
            ),
            "route_obligation_payload_sha256": obligation_report.get(
                "payload_sha256"
            ),
            "matched_shell_report_status": matched_shell_report.get(
                "status"
            ),
            "matched_shell_payload_sha256": matched_shell_report.get(
                "payload_sha256"
            ),
            "pass": prerequisite_pass,
        },
        "source_anchors": anchors,
        "registry": registry,
        "successful_compilations": {
            "passed": sum(
                row["all_bindings_pass"] for row in compiled_rows
            ),
            "required": 6,
            "rows": compiled_rows,
        },
        "postcompile_tamper_checks": {
            "rejected": sum(row["passed"] for row in tamper_rows),
            "required": 18,
            "rows": tamper_rows,
        },
        "proof_obligations": obligations,
        "verdict": {
            "bounded_registry_separation_lemma": (
                "PASS" if status_pass else "FAIL"
            ),
            "allowed_lemma_wording": allowed_lemma,
            "proof_style": (
                "finite source-structure case split plus six exact "
                "compilations and 18 postcompile tamper checks"
            ),
            "claims_forbidden": [
                "the common obligations prove an unregistered route private",
                "the compiler is complete for arbitrary DP programs",
                "successful compilation formally refines arbitrary execution",
                "the finite mutation set detects every semantic mismatch",
                "hash binding attests malicious execution",
                "this program lemma is a new privacy theorem",
            ],
        },
        "closest_system_boundary": [
            {
                "system": "Opacus 1.6.0",
                "prior_scope": (
                    "core P-route model, optimizer, loader, clipping/noising, "
                    "Poisson sampling, and accounting stack"
                ),
                "residual_scope": (
                    "external owner/window attribution, fixed data and "
                    "preprocessor profile, P/F route identity, executor/source "
                    "identity, and artifact chain"
                ),
                "required_limit": (
                    "scope distinction only; no Opacus bug or general "
                    "executor-superiority claim"
                ),
            },
            {
                "system": "OpenDP and typed DP languages",
                "prior_scope": (
                    "broader semantic domains, metrics, compatibility, "
                    "stability/privacy maps, and composition"
                ),
                "residual_scope": (
                    "a nominal two-entry ML execution registry with concrete "
                    "generated-record and artifact bindings"
                ),
                "required_limit": (
                    "the present registry is narrower and does not improve "
                    "their semantic foundations"
                ),
            },
            {
                "system": "LightDP, CheckDP, DPCheck, and Re:cord-play",
                "prior_scope": (
                    "proof, counterexample generation, statistical testing, "
                    "or grey-box execution analysis"
                ),
                "residual_scope": (
                    "pre-execution registered-premise and lifecycle binding"
                ),
                "required_limit": (
                    "no DP-definition test, quantified false-acceptance "
                    "guarantee, or arbitrary-program proof"
                ),
            },
            {
                "system": "TrainVerify and VeriDP",
                "prior_scope": (
                    "stronger execution-equivalence or attestation settings"
                ),
                "residual_scope": (
                    "honest-execution source, plan, model, run, and artifact "
                    "hash correspondence"
                ),
                "required_limit": (
                    "hashes are not cryptographic execution attestation"
                ),
            },
        ],
        "source_sha256": {
            path_key(path): file_sha256(path)
            for path in (
                Path(__file__).resolve(),
                ROOT / "src" / "unitdp" / "compiler_v2.py",
                ROOT / "src" / "unitdp" / "compiler_srswor_v3.py",
                ROOT / "src" / "unitdp" / "compiler_v3.py",
                OBLIGATION_REPORT,
                MATCHED_SHELL_REPORT,
                *(
                    config_path(route, dataset)
                    for route in ("P", "F")
                    for dataset in DATASETS
                ),
                *(
                    ARTIFACTS[dataset][kind]
                    for dataset in DATASETS
                    for kind in ("mapping", "preprocessor")
                ),
            )
        },
    }
    report["payload_sha256"] = canonical_sha256(report)
    return report


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# V3.4 Finite-Registry Separation-Lemma Gate",
        "",
        f"Status: **{report['status']}**",
        "",
        "## Allowed lemma",
        "",
        f"> {report['verdict']['allowed_lemma_wording']}",
        "",
        "## Machine and source checks",
        "",
        f"- Proof obligations: "
        f"{sum(row['pass'] for row in report['proof_obligations'])}/"
        f"{len(report['proof_obligations'])}",
        f"- Exact executable compilations: "
        f"{report['successful_compilations']['passed']}/"
        f"{report['successful_compilations']['required']}",
        f"- Postcompile tamper rejections: "
        f"{report['postcompile_tamper_checks']['rejected']}/"
        f"{report['postcompile_tamper_checks']['required']}",
        f"- Source anchors: "
        f"{sum(row['pass'] for row in report['source_anchors'])}/"
        f"{len(report['source_anchors'])}",
        "",
        "| ID | Program-property premise | Result |",
        "| --- | --- | ---: |",
    ]
    for row in report["proof_obligations"]:
        lines.append(
            f"| {row['id']} | {row['statement']} | "
            f"{'PASS' if row['pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This is a source-bound program property of the frozen P/F "
            "registry. It is not a new privacy theorem, a generic DP type "
            "system, a completeness result, formal executor refinement, or "
            "attestation.",
            "",
            "Opacus retains the core P training-stack baseline; OpenDP and "
            "typed DP languages retain the broader semantic-composition "
            "baseline. The residual here is the concrete owner/window, "
            "route, executor/source, and artifact binding for two registered "
            "ML routes.",
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
        f"obligations={sum(row['pass'] for row in report['proof_obligations'])}/7 "
        f"compilations={report['successful_compilations']['passed']}/6 "
        f"tamper={report['postcompile_tamper_checks']['rejected']}/18 "
        f"payload_sha256={report['payload_sha256']}"
    )
    print(f"result={args.output}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
