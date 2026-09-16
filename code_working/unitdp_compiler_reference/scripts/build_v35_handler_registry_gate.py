#!/usr/bin/env python3
"""Build the V3.5 handler-registry and conservative-extension gate."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unitdp.compiler_v3 import (  # noqa: E402
    compile_registered_contract_v3,
    load_registered_contract_v3,
)
from unitdp.compiler_v4 import (  # noqa: E402
    ALLOCATION_HANDLER_V4,
    POISSON_HANDLER_V4,
    REQUIRED_REGISTRATION_OBLIGATIONS_V4,
    ROUTE_DISPATCH_REGISTRY_V4,
    SRSWOR_HANDLER_V4,
    CompilerDispatchV4Error,
    build_route_dispatch_registry_v4,
    compile_registered_contract_v4,
    load_registered_contract_v4,
    parse_registered_contract_v4,
    validate_route_handler_v4,
)


DEFAULT_OUTPUT = (
    ROOT
    / "reports"
    / "v35_handler_registry_gate_20260725"
    / "handler_registry_gate_v35.json"
)
DEFAULT_MARKDOWN = DEFAULT_OUTPUT.with_suffix(".md")
SCHEMA_VERSION = "unitdp.v35_handler_registry_gate.v1"
REPORT_DATE = "2026-07-25"

DATASETS = ("uci", "wisdm", "sepsis")
ROUTES = {
    "P": ("v2", "owner_poisson_v2"),
    "F": ("v3", "owner_srswor_v3"),
    "A": ("v4", "owner_random_allocation_v4"),
}
ARTIFACTS = {
    "uci": {
        "mapping": (
            ROOT
            / "reports/uci_public_preprocessing_smoke_20260724/"
            "uci_train_mapping.csv"
        ),
        "preprocessor": (
            ROOT
            / "configs/preprocessing/"
            "uci_har_published_train_standard_scaler_v1.json"
        ),
    },
    "wisdm": {
        "mapping": (
            ROOT
            / "reports/wisdm_public_protocol_smoke_20260724/"
            "wisdm_train_mapping.csv"
        ),
        "preprocessor": (
            ROOT
            / "configs/preprocessing/"
            "wisdm_v1_1_train_owners_1_25_stats24_standard_scaler_v1.json"
        ),
    },
    "sepsis": {
        "mapping": (
            ROOT
            / "reports/sepsis_public_protocol_smoke_20260724/"
            "sepsis_train_mapping.csv"
        ),
        "preprocessor": (
            ROOT
            / "configs/preprocessing/"
            "physionet2019_setA_first400_basic12x6_public_scaler_v1.json"
        ),
    },
}

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
    "scripts/build_v35_handler_registry_gate.py",
    "scripts/verify_v35_handler_registry_gate.py",
    "src/unitdp/compiler_v4.py",
    "tests/test_compiler_v4.py",
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


def config_path(route: str, dataset: str) -> Path:
    directory, suffix = ROUTES[route]
    return (
        ROOT
        / "configs"
        / directory
        / f"{dataset}_{suffix}.yaml"
    )


def _registry_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for route, handler in zip(
        ("P", "F", "A"),
        (
            POISSON_HANDLER_V4,
            SRSWOR_HANDLER_V4,
            ALLOCATION_HANDLER_V4,
        ),
        strict=True,
    ):
        validate_route_handler_v4(handler)
        ids = {
            witness.obligation_id
            for witness in handler.obligation_witnesses
        }
        checks = {
            "dispatch_key": (
                ROUTE_DISPATCH_REGISTRY_V4[handler.route_id]
                == handler
            ),
            "seven_exact_obligations": (
                ids == REQUIRED_REGISTRATION_OBLIGATIONS_V4
                and len(handler.obligation_witnesses) == 7
            ),
            "evidence_report_pass": (
                json.loads(
                    (ROOT / handler.evidence_report_path).read_text(
                        encoding="utf-8"
                    )
                ).get("status")
                == "PASS"
            ),
            "executor_ready": (
                getattr(
                    handler.specific_registry_entry,
                    "executor_status",
                    "",
                )
                == "implemented_research_validated"
            ),
        }
        rows.append(
            {
                "route": route,
                "handler": json.loads(
                    json.dumps(
                        handler.public_payload(),
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                ),
                "registration_sha256": handler.registration_sha256,
                "checks": checks,
                "pass": all(checks.values()),
            }
        )
    return rows


def _compilation_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for route in ROUTES:
        for dataset in DATASETS:
            contract = load_registered_contract_v4(
                config_path(route, dataset)
            )
            compiled = compile_registered_contract_v4(
                contract,
                mapping_path=ARTIFACTS[dataset]["mapping"],
                preprocessing_artifact_path=(
                    ARTIFACTS[dataset]["preprocessor"]
                ),
                require_executable=True,
            )
            compiled.assert_private_execution_integrity()
            rows.append(
                {
                    "route": route,
                    "dataset": dataset,
                    "contract_type": type(contract).__name__,
                    "compiled_type": type(compiled).__name__,
                    "public_contract_sha256": (
                        contract.public_contract_sha256
                    ),
                    "public_plan_sha256": compiled.public_plan_sha256,
                    "execution_ready": compiled.execution_ready,
                    "pass": compiled.execution_ready is True,
                }
            )
    return rows


def _extension_rows() -> list[dict[str, Any]]:
    legacy_registry = build_route_dispatch_registry_v4(
        (POISSON_HANDLER_V4, SRSWOR_HANDLER_V4)
    )
    rows: list[dict[str, Any]] = []
    for route in ("P", "F"):
        for dataset in DATASETS:
            config = config_path(route, dataset)
            legacy_v3_contract = load_registered_contract_v3(config)
            before = load_registered_contract_v4(
                config,
                registry=legacy_registry,
            )
            after = load_registered_contract_v4(config)
            legacy_v3_compiled = compile_registered_contract_v3(
                legacy_v3_contract,
                mapping_path=ARTIFACTS[dataset]["mapping"],
                preprocessing_artifact_path=(
                    ARTIFACTS[dataset]["preprocessor"]
                ),
                require_executable=True,
            )
            before_compiled = compile_registered_contract_v4(
                before,
                mapping_path=ARTIFACTS[dataset]["mapping"],
                preprocessing_artifact_path=(
                    ARTIFACTS[dataset]["preprocessor"]
                ),
                require_executable=True,
                registry=legacy_registry,
            )
            after_compiled = compile_registered_contract_v4(
                after,
                mapping_path=ARTIFACTS[dataset]["mapping"],
                preprocessing_artifact_path=(
                    ARTIFACTS[dataset]["preprocessor"]
                ),
                require_executable=True,
            )
            checks = {
                "contract_v3_equals_before": (
                    legacy_v3_contract == before
                ),
                "contract_before_equals_after": before == after,
                "public_contract_hash_exact": (
                    legacy_v3_contract.public_contract_sha256
                    == before.public_contract_sha256
                    == after.public_contract_sha256
                ),
                "public_plan_v3_equals_before": (
                    legacy_v3_compiled.public_plan()
                    == before_compiled.public_plan()
                ),
                "public_plan_before_equals_after": (
                    before_compiled.public_plan()
                    == after_compiled.public_plan()
                ),
                "public_plan_hash_exact": (
                    legacy_v3_compiled.public_plan_sha256
                    == before_compiled.public_plan_sha256
                    == after_compiled.public_plan_sha256
                ),
            }
            rows.append(
                {
                    "route": route,
                    "dataset": dataset,
                    "public_contract_sha256": (
                        after.public_contract_sha256
                    ),
                    "public_plan_sha256": (
                        after_compiled.public_plan_sha256
                    ),
                    "checks": checks,
                    "pass": all(checks.values()),
                }
            )
    return rows


def _negative_registration_rows() -> list[dict[str, Any]]:
    bad_hash_witnesses = list(
        ALLOCATION_HANDLER_V4.obligation_witnesses
    )
    bad_hash_witnesses[0] = replace(
        bad_hash_witnesses[0],
        source_sha256="0" * 64,
    )
    downgraded_registry = replace(
        ALLOCATION_HANDLER_V4.specific_registry_entry,
        executor_status="planned_not_implemented",
    )
    cases = (
        (
            "missing_obligation",
            (
                replace(
                    ALLOCATION_HANDLER_V4,
                    obligation_witnesses=(
                        ALLOCATION_HANDLER_V4.obligation_witnesses[:-1]
                    ),
                ),
            ),
        ),
        (
            "duplicate_obligation",
            (
                replace(
                    ALLOCATION_HANDLER_V4,
                    obligation_witnesses=(
                        *ALLOCATION_HANDLER_V4.obligation_witnesses,
                        ALLOCATION_HANDLER_V4.obligation_witnesses[0],
                    ),
                ),
            ),
        ),
        (
            "wrong_source_hash",
            (
                replace(
                    ALLOCATION_HANDLER_V4,
                    obligation_witnesses=tuple(bad_hash_witnesses),
                ),
            ),
        ),
        (
            "parser_splice",
            (
                replace(
                    ALLOCATION_HANDLER_V4,
                    parser=SRSWOR_HANDLER_V4.parser,
                ),
            ),
        ),
        (
            "compiled_type_splice",
            (
                replace(
                    ALLOCATION_HANDLER_V4,
                    compiled_type=SRSWOR_HANDLER_V4.compiled_type,
                ),
            ),
        ),
        (
            "evidence_hash_splice",
            (
                replace(
                    ALLOCATION_HANDLER_V4,
                    evidence_report_sha256="0" * 64,
                ),
            ),
        ),
        (
            "executor_not_validated",
            (
                replace(
                    ALLOCATION_HANDLER_V4,
                    specific_registry_entry=downgraded_registry,
                ),
            ),
        ),
        (
            "duplicate_route",
            (POISSON_HANDLER_V4, POISSON_HANDLER_V4),
        ),
    )
    rows: list[dict[str, Any]] = []
    for name, handlers in cases:
        rejected = False
        error_type = ""
        try:
            build_route_dispatch_registry_v4(handlers)
        except CompilerDispatchV4Error as exc:
            rejected = True
            error_type = type(exc).__name__
        rows.append(
            {
                "name": name,
                "rejected": rejected,
                "error_type": error_type,
            }
        )
    return rows


def _generic_branch_check() -> dict[str, Any]:
    parse_source = inspect.getsource(parse_registered_contract_v4)
    compile_source = inspect.getsource(compile_registered_contract_v4)
    forbidden = (
        "CORE_ROUTE_ID_V2",
        "CORE_ROUTE_ID_SRSWOR_V3",
        "CORE_ROUTE_ID_ALLOCATION_V4",
        "OwnerPoissonContractV2",
        "OwnerSrsworContractV3",
        "OwnerRandomAllocationContractV4",
        "if route_id ==",
    )
    occurrences = {
        token: parse_source.count(token) + compile_source.count(token)
        for token in forbidden
    }
    checks = {
        "no_forbidden_branches": all(
            count == 0 for count in occurrences.values()
        ),
        "parser_called_from_handler": (
            "handler.parser(raw_value)" in parse_source
        ),
        "compiler_called_from_handler": (
            "handler.compiler(" in compile_source
        ),
        "exact_contract_type_check": (
            "type(contract) is not handler.contract_type"
            in compile_source
        ),
        "exact_compiled_type_check": (
            "type(compiled) is not handler.compiled_type"
            in compile_source
        ),
    }
    return {
        "forbidden_occurrences": occurrences,
        "checks": checks,
        "pass": all(checks.values()),
    }


def build_report() -> dict[str, Any]:
    registry_rows = _registry_rows()
    compilation_rows = _compilation_rows()
    extension_rows = _extension_rows()
    negative_rows = _negative_registration_rows()
    branch_check = _generic_branch_check()
    frozen = [
        {
            "path": path,
            "expected_sha256": expected,
            "observed_sha256": file_sha256(ROOT / path),
            "pass": file_sha256(ROOT / path) == expected,
        }
        for path, expected in FROZEN_BASELINE.items()
    ]
    checks = {
        "immutable_three_route_registry": (
            isinstance(ROUTE_DISPATCH_REGISTRY_V4, MappingProxyType)
            and len(ROUTE_DISPATCH_REGISTRY_V4) == 3
        ),
        "registration_judgments": all(
            row["pass"] for row in registry_rows
        ),
        "nine_compilations": (
            len(compilation_rows) == 9
            and all(row["pass"] for row in compilation_rows)
        ),
        "six_exact_conservative_extensions": (
            len(extension_rows) == 6
            and all(row["pass"] for row in extension_rows)
        ),
        "eight_invalid_registrations_rejected": (
            len(negative_rows) == 8
            and all(row["rejected"] for row in negative_rows)
        ),
        "generic_branch_free_dispatch": branch_check["pass"],
        "frozen_v3": all(row["pass"] for row in frozen),
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "report_date": REPORT_DATE,
        "candidate": "AAAI27 Algorithm V3.5",
        "object": (
            "evidence-carrying handler registration and bounded "
            "conservative extension"
        ),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "required_registration_obligations": sorted(
            REQUIRED_REGISTRATION_OBLIGATIONS_V4
        ),
        "registry_rows": registry_rows,
        "compilation_rows": compilation_rows,
        "conservative_extension_rows": extension_rows,
        "negative_registration_rows": negative_rows,
        "generic_branch_check": branch_check,
        "frozen_v3_baseline": frozen,
        "claim_source_sha256": {
            path: file_sha256(ROOT / path) for path in CLAIM_SOURCES
        },
        "verdict": {
            "handler_registry_gate": (
                "PASS" if all(checks.values()) else "FAIL"
            ),
            "bounded_statement": (
                "For the three registered handlers and nine registered "
                "profile-route pairs, dispatch is handler-bound; adding A "
                "preserves the six exact P/F parsed contracts and public plans."
            ),
            "does_not_authorize": [
                "open-world plugin safety",
                "arbitrary third-party route correctness",
                "formal semantic refinement",
                "arbitrary-input completeness",
                "a privacy proof",
                "a manuscript novelty claim",
            ],
        },
    }
    report["payload_sha256"] = canonical_sha256(report)
    return report


def _markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# V3.5 Handler Registry Gate",
            "",
            f"- Status: `{report['status']}`",
            f"- Payload SHA-256: `{report['payload_sha256']}`",
            f"- Registration judgments: `{len(report['registry_rows'])}/3`",
            (
                "- Generic compilations: "
                f"`{len(report['compilation_rows'])}/9`"
            ),
            (
                "- Exact P/F conservative-extension rows: "
                f"`{len(report['conservative_extension_rows'])}/6`"
            ),
            (
                "- Invalid registrations rejected: "
                f"`{sum(row['rejected'] for row in report['negative_registration_rows'])}/8`"
            ),
            "",
            "The statement is bounded to the frozen finite registry.",
            "",
        ]
    )


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
