"""Two-route strict compiler dispatcher.

The dispatcher does not normalize one route into another.  It selects a
route-specific strict parser and compiler from an immutable registry, so
sampler, adjacency, sensitivity, accountant, and executor remain inseparable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias

import yaml

from unitdp.benchmark_registry_srswor_v3 import (
    CONTRACT_SCHEMA_SRSWOR_V3,
    CORE_ROUTE_ID_SRSWOR_V3,
)
from unitdp.compiler_srswor_v3 import (
    CompiledOwnerSrsworRouteV3,
    OwnerSrsworContractV3,
    SrsworContractV3Error,
    compile_owner_srswor_contract_v3,
    parse_owner_srswor_contract_v3,
)
from unitdp.compiler_v2 import (
    CONTRACT_SCHEMA_V2,
    CORE_ROUTE_ID_V2,
    CompiledOwnerPoissonRouteV2,
    ContractV2Error,
    OwnerPoissonContractV2,
    _UniqueKeySafeLoader,
    _construct_unique_json_object,
    compile_owner_poisson_contract_v2,
    parse_owner_poisson_contract_v2,
)


RegisteredContractV3: TypeAlias = (
    OwnerPoissonContractV2 | OwnerSrsworContractV3
)
CompiledRegisteredRouteV3: TypeAlias = (
    CompiledOwnerPoissonRouteV2 | CompiledOwnerSrsworRouteV3
)


class CompilerDispatchV3Error(ValueError):
    """Raised before a route-specific compiler can be selected safely."""


@dataclass(frozen=True)
class RouteDispatchEntryV3:
    route_id: str
    schema_version: str
    adjacency: str
    sampler: str
    accountant_family: str
    contract_type: str
    compiled_type: str


ROUTE_DISPATCH_REGISTRY_V3 = MappingProxyType(
    {
        CORE_ROUTE_ID_V2: RouteDispatchEntryV3(
            route_id=CORE_ROUTE_ID_V2,
            schema_version=CONTRACT_SCHEMA_V2,
            adjacency="add_remove_one_owner",
            sampler="independent_bernoulli_owner",
            accountant_family="poisson_gaussian_rdp",
            contract_type="OwnerPoissonContractV2",
            compiled_type="CompiledOwnerPoissonRouteV2",
        ),
        CORE_ROUTE_ID_SRSWOR_V3: RouteDispatchEntryV3(
            route_id=CORE_ROUTE_ID_SRSWOR_V3,
            schema_version=CONTRACT_SCHEMA_SRSWOR_V3,
            adjacency="replace_one_owner",
            sampler="uniform_fixed_size_without_replacement_owner",
            accountant_family="srswor_gaussian_rdp",
            contract_type="OwnerSrsworContractV3",
            compiled_type="CompiledOwnerSrsworRouteV3",
        ),
    }
)


def parse_registered_contract_v3(
    raw_value: object,
) -> RegisteredContractV3:
    if not isinstance(raw_value, dict):
        raise CompilerDispatchV3Error("contract must be a mapping")
    route_id = raw_value.get("route_id")
    schema_version = raw_value.get("schema_version")
    if not isinstance(route_id, str):
        raise CompilerDispatchV3Error("route_id must be a string")
    try:
        entry = ROUTE_DISPATCH_REGISTRY_V3[route_id]
    except KeyError as exc:
        raise CompilerDispatchV3Error(
            f"Unsupported registered route_id: {route_id!r}"
        ) from exc
    if schema_version != entry.schema_version:
        raise CompilerDispatchV3Error(
            f"route {route_id!r} requires schema {entry.schema_version!r}"
        )
    try:
        if route_id == CORE_ROUTE_ID_V2:
            return parse_owner_poisson_contract_v2(raw_value)
        if route_id == CORE_ROUTE_ID_SRSWOR_V3:
            return parse_owner_srswor_contract_v3(raw_value)
    except (ContractV2Error, SrsworContractV3Error) as exc:
        raise CompilerDispatchV3Error(
            f"Route-specific contract validation failed: {exc}"
        ) from exc
    raise CompilerDispatchV3Error(
        "Registered route has no strict parser"
    )


def load_registered_contract_v3(
    path: str | Path,
) -> RegisteredContractV3:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    try:
        if source.suffix.lower() == ".json":
            raw = json.loads(
                text,
                object_pairs_hook=_construct_unique_json_object,
            )
        else:
            raw = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except (ContractV2Error, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise CompilerDispatchV3Error(
            f"Could not load strict registered contract: {exc}"
        ) from exc
    return parse_registered_contract_v3(raw)


def compile_registered_contract_v3(
    contract: RegisteredContractV3,
    *,
    mapping_path: str | Path,
    preprocessing_artifact_path: str | Path,
    require_executable: bool = False,
) -> CompiledRegisteredRouteV3:
    if isinstance(contract, OwnerPoissonContractV2):
        return compile_owner_poisson_contract_v2(
            contract,
            mapping_path=mapping_path,
            preprocessing_artifact_path=preprocessing_artifact_path,
            require_executable=require_executable,
        )
    if isinstance(contract, OwnerSrsworContractV3):
        return compile_owner_srswor_contract_v3(
            contract,
            mapping_path=mapping_path,
            preprocessing_artifact_path=preprocessing_artifact_path,
            require_executable=require_executable,
        )
    raise CompilerDispatchV3Error(
        "contract type is not present in the immutable route registry"
    )
