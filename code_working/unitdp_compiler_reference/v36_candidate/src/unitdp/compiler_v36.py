"""Separate four-handler registry candidate over the frozen V4 dispatcher."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

from unitdp.compiler_v4 import (
    ALLOCATION_HANDLER_V4,
    POISSON_HANDLER_V4,
    SRSWOR_HANDLER_V4,
    RouteHandlerV4,
    build_route_dispatch_registry_v4,
    compile_registered_contract_v4,
    load_registered_contract_v4,
    parse_registered_contract_v4,
)
from unitdp.handler_external_pld_v36 import (
    DEFAULT_EXTERNAL_PLD_EVIDENCE_PATH_V36,
    build_external_pld_handler_v36,
)

EXTERNAL_PLD_EVIDENCE_PATH_V36 = DEFAULT_EXTERNAL_PLD_EVIDENCE_PATH_V36
EXTERNAL_PLD_EVIDENCE_SHA256_V36 = (
    "478b29b7df26aa99236a99818e0766e6553df3ed9165e67b8410db45f4df1b28"
)


def build_route_dispatch_registry_v36(
    *,
    evidence_report_sha256: str,
    evidence_report_path: str = EXTERNAL_PLD_EVIDENCE_PATH_V36,
) -> MappingProxyType[str, RouteHandlerV4]:
    """Build P/F/A/H after the separately supplied H evidence validates."""

    handler = build_external_pld_handler_v36(
        evidence_report_sha256=evidence_report_sha256,
        evidence_report_path=evidence_report_path,
    )
    return build_route_dispatch_registry_v4(
        (
            POISSON_HANDLER_V4,
            SRSWOR_HANDLER_V4,
            ALLOCATION_HANDLER_V4,
            handler,
        )
    )


EXTERNAL_PLD_HANDLER_V36 = build_external_pld_handler_v36(
    evidence_report_sha256=EXTERNAL_PLD_EVIDENCE_SHA256_V36
)
ROUTE_DISPATCH_REGISTRY_V36 = build_route_dispatch_registry_v4(
    (
        POISSON_HANDLER_V4,
        SRSWOR_HANDLER_V4,
        ALLOCATION_HANDLER_V4,
        EXTERNAL_PLD_HANDLER_V36,
    )
)


def parse_registered_contract_v36(raw_value: object) -> object:
    """Use the frozen branch-free parser with the four-handler registry."""

    return parse_registered_contract_v4(
        raw_value,
        registry=ROUTE_DISPATCH_REGISTRY_V36,
    )


def load_registered_contract_v36(path: str | Path) -> object:
    """Load one strict P/F/A/H contract through the frozen loader."""

    return load_registered_contract_v4(
        path,
        registry=ROUTE_DISPATCH_REGISTRY_V36,
    )


def compile_registered_contract_v36(
    contract: object,
    *,
    mapping_path: str | Path,
    preprocessing_artifact_path: str | Path,
    require_executable: bool = False,
) -> object:
    """Compile one P/F/A/H contract through the frozen compiler."""

    return compile_registered_contract_v4(
        contract,
        mapping_path=mapping_path,
        preprocessing_artifact_path=preprocessing_artifact_path,
        require_executable=require_executable,
        registry=ROUTE_DISPATCH_REGISTRY_V36,
    )
