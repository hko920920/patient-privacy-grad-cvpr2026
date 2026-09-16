"""Evidence-independent construction of the provisional V3.6 H handler."""

from __future__ import annotations

import hashlib
from pathlib import Path

from unitdp.compiler_external_pld_v36 import (
    CONTRACT_SCHEMA_EXTERNAL_PLD_V36,
    CORE_ROUTE_ID_EXTERNAL_PLD_V36,
    ROUTE_REGISTRY_EXTERNAL_PLD_V36,
    CompiledOwnerExternalPldAllocationRouteV36,
    ExternalPldContractV36Error,
    ExternalPldExecutorNotReadyV36Error,
    OwnerExternalPldAllocationContractV36,
    compile_owner_external_pld_contract_v36,
    parse_owner_external_pld_contract_v36,
)
from unitdp.compiler_v4 import (
    RouteHandlerV4,
    RouteObligationWitnessV4,
)


DEFAULT_EXTERNAL_PLD_EVIDENCE_PATH_V36 = (
    "reports/g8_external_handler_evidence_v2_20260725/"
    "external_pld_handler_evidence_v36.json"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _file_sha256(relative: str) -> str:
    root = _repo_root().resolve()
    path = (root / relative).resolve()
    path.relative_to(root)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _witness(
    obligation_id: str,
    implementation_id: str,
    source_path: str,
) -> RouteObligationWitnessV4:
    return RouteObligationWitnessV4(
        obligation_id=obligation_id,
        implementation_id=implementation_id,
        source_path=source_path,
        source_sha256=_file_sha256(source_path),
    )


def build_external_pld_handler_v36(
    *,
    evidence_report_sha256: str,
    evidence_report_path: str = DEFAULT_EXTERNAL_PLD_EVIDENCE_PATH_V36,
) -> RouteHandlerV4:
    """Construct H; its complete core is independent of evidence-file hash."""

    return RouteHandlerV4(
        route_id=CORE_ROUTE_ID_EXTERNAL_PLD_V36,
        schema_version=CONTRACT_SCHEMA_EXTERNAL_PLD_V36,
        adjacency="add_remove_one_owner",
        sampler="independent_uniform_k_subset_of_steps_per_owner",
        accountant_family="external_random_allocation_pld",
        contract_type=OwnerExternalPldAllocationContractV36,
        compiled_type=CompiledOwnerExternalPldAllocationRouteV36,
        parser=parse_owner_external_pld_contract_v36,
        compiler=compile_owner_external_pld_contract_v36,
        validation_error_types=(
            ExternalPldContractV36Error,
            ExternalPldExecutorNotReadyV36Error,
        ),
        integrity_method_name="assert_private_execution_integrity",
        specific_registry_entry=ROUTE_REGISTRY_EXTERNAL_PLD_V36[
            CORE_ROUTE_ID_EXTERNAL_PLD_V36
        ],
        evidence_report_path=evidence_report_path,
        evidence_report_sha256=evidence_report_sha256,
        obligation_witnesses=(
            _witness(
                "strict_parser",
                (
                    "unitdp.compiler_external_pld_v36."
                    "parse_owner_external_pld_contract_v36"
                ),
                "v36_candidate/src/unitdp/compiler_external_pld_v36.py",
            ),
            _witness(
                "strict_compiler",
                (
                    "unitdp.compiler_external_pld_v36."
                    "compile_owner_external_pld_contract_v36"
                ),
                "v36_candidate/src/unitdp/compiler_external_pld_v36.py",
            ),
            _witness(
                "accountant",
                "unitdp.external_pld_backend_v36.account_external_pld_v36",
                "v36_candidate/src/unitdp/external_pld_backend_v36.py",
            ),
            _witness(
                "numeric_crosscheck",
                (
                    "unitdp.external_pld_backend_v36."
                    "validate_external_pld_numeric_crosscheck_v36"
                ),
                "v36_candidate/src/unitdp/external_pld_backend_v36.py",
            ),
            _witness(
                "executor",
                "unitdp.owner_external_pld_v36.train_owner_external_pld_fixed_v36",
                "v36_candidate/src/unitdp/owner_external_pld_v36.py",
            ),
            _witness(
                "postcompile_integrity",
                (
                    "unitdp.compiler_external_pld_v36."
                    "CompiledOwnerExternalPldAllocationRouteV36."
                    "assert_private_execution_integrity"
                ),
                "v36_candidate/src/unitdp/compiler_external_pld_v36.py",
            ),
            _witness(
                "source_binding",
                (
                    "unitdp.source_bundle_external_pld_v36."
                    "compiler_source_bundle_sha256_external_pld_v36"
                ),
                "v36_candidate/src/unitdp/source_bundle_external_pld_v36.py",
            ),
        ),
    )
