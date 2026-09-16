"""Evidence-carrying, handler-bound dispatcher for registered routes.

Route behavior is selected only through validated handler callables.  The
generic parse and compile paths contain no route-name or contract-type branch.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, TypeAlias

import yaml

from unitdp.benchmark_registry_allocation_v4 import (
    CONTRACT_SCHEMA_ALLOCATION_V4,
    CORE_ROUTE_ID_ALLOCATION_V4,
)
from unitdp.benchmark_registry_srswor_v3 import (
    CONTRACT_SCHEMA_SRSWOR_V3,
    CORE_ROUTE_ID_SRSWOR_V3,
)
from unitdp.compiler_allocation_v4 import (
    AllocationContractV4Error,
    CompiledOwnerRandomAllocationRouteV4,
    OwnerRandomAllocationContractV4,
    ROUTE_REGISTRY_ALLOCATION_V4,
    compile_owner_random_allocation_contract_v4,
    parse_owner_random_allocation_contract_v4,
)
from unitdp.compiler_srswor_v3 import (
    CompiledOwnerSrsworRouteV3,
    OwnerSrsworContractV3,
    ROUTE_REGISTRY_SRSWOR_V3,
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
    ROUTE_REGISTRY_V2,
    _UniqueKeySafeLoader,
    _construct_unique_json_object,
    compile_owner_poisson_contract_v2,
    parse_owner_poisson_contract_v2,
)


RegisteredContractV4: TypeAlias = (
    OwnerPoissonContractV2
    | OwnerSrsworContractV3
    | OwnerRandomAllocationContractV4
)
CompiledRegisteredRouteV4: TypeAlias = (
    CompiledOwnerPoissonRouteV2
    | CompiledOwnerSrsworRouteV3
    | CompiledOwnerRandomAllocationRouteV4
)
ParserV4: TypeAlias = Callable[[object], RegisteredContractV4]
CompilerV4: TypeAlias = Callable[..., CompiledRegisteredRouteV4]


class CompilerDispatchV4Error(ValueError):
    """Raised when registration or generic dispatch is not justified."""


REQUIRED_REGISTRATION_OBLIGATIONS_V4 = frozenset(
    {
        "strict_parser",
        "strict_compiler",
        "accountant",
        "numeric_crosscheck",
        "executor",
        "postcompile_integrity",
        "source_binding",
    }
)


@dataclass(frozen=True)
class RouteObligationWitnessV4:
    obligation_id: str
    implementation_id: str
    source_path: str
    source_sha256: str


@dataclass(frozen=True)
class RouteHandlerV4:
    route_id: str
    schema_version: str
    adjacency: str
    sampler: str
    accountant_family: str
    contract_type: type
    compiled_type: type
    parser: ParserV4
    compiler: CompilerV4
    validation_error_types: tuple[type[Exception], ...]
    integrity_method_name: str
    specific_registry_entry: object
    evidence_report_path: str
    evidence_report_sha256: str
    obligation_witnesses: tuple[RouteObligationWitnessV4, ...]

    def public_payload(self) -> dict[str, Any]:
        registry = self.specific_registry_entry
        if not hasattr(registry, "__dataclass_fields__"):
            raise CompilerDispatchV4Error(
                "specific registry entry must be a frozen dataclass"
            )
        return {
            "route_id": self.route_id,
            "schema_version": self.schema_version,
            "adjacency": self.adjacency,
            "sampler": self.sampler,
            "accountant_family": self.accountant_family,
            "contract_type": _implementation_id(self.contract_type),
            "compiled_type": _implementation_id(self.compiled_type),
            "parser": _implementation_id(self.parser),
            "compiler": _implementation_id(self.compiler),
            "validation_error_types": [
                _implementation_id(value)
                for value in self.validation_error_types
            ],
            "integrity_method_name": self.integrity_method_name,
            "specific_registry_entry": asdict(registry),
            "evidence_report_path": self.evidence_report_path,
            "evidence_report_sha256": self.evidence_report_sha256,
            "obligation_witnesses": [
                asdict(value) for value in self.obligation_witnesses
            ],
        }

    def registration_core_payload(self) -> dict[str, Any]:
        """Return the complete handler tuple committed by route evidence."""

        value = self.public_payload()
        value.pop("evidence_report_path")
        value.pop("evidence_report_sha256")
        return value

    @property
    def registration_core_sha256(self) -> str:
        return _payload_sha256(self.registration_core_payload())

    @property
    def registration_sha256(self) -> str:
        return _payload_sha256(self.public_payload())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _payload_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_id(value: object) -> str:
    module = getattr(value, "__module__", "")
    name = getattr(value, "__qualname__", "")
    if not module or not name:
        raise CompilerDispatchV4Error(
            "registered implementation has no stable qualified name"
        )
    return f"{module}.{name}"


def _resolve_implementation_id(implementation_id: str) -> object:
    if not isinstance(implementation_id, str) or not implementation_id:
        raise CompilerDispatchV4Error(
            "obligation implementation id must be nonempty"
        )
    parts = implementation_id.split(".")
    for split in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split])
        try:
            value: object = importlib.import_module(module_name)
        except ImportError:
            continue
        try:
            for component in parts[split:]:
                value = getattr(value, component)
        except AttributeError:
            continue
        return value
    raise CompilerDispatchV4Error(
        f"Could not resolve registered implementation {implementation_id!r}"
    )


def _validated_source_path(relative: str) -> Path:
    root = _repo_root().resolve()
    candidate = (root / Path(relative)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise CompilerDispatchV4Error(
            f"registration source escapes the repository: {relative}"
        ) from exc
    if not candidate.is_file():
        raise CompilerDispatchV4Error(
            f"registration source is missing: {relative}"
        )
    return candidate


def validate_route_handler_v4(handler: RouteHandlerV4) -> None:
    """Validate one complete registration judgment or reject it."""

    if not isinstance(handler, RouteHandlerV4):
        raise CompilerDispatchV4Error(
            "route registration must be RouteHandlerV4"
        )
    for name, value in (
        ("route_id", handler.route_id),
        ("schema_version", handler.schema_version),
        ("adjacency", handler.adjacency),
        ("sampler", handler.sampler),
        ("accountant_family", handler.accountant_family),
        ("integrity_method_name", handler.integrity_method_name),
    ):
        if not isinstance(value, str) or not value:
            raise CompilerDispatchV4Error(
                f"route registration {name} must be nonempty"
            )
    if not inspect.isclass(handler.contract_type):
        raise CompilerDispatchV4Error(
            "contract_type must be a class"
        )
    if not inspect.isclass(handler.compiled_type):
        raise CompilerDispatchV4Error(
            "compiled_type must be a class"
        )
    if not callable(handler.parser) or not callable(handler.compiler):
        raise CompilerDispatchV4Error(
            "parser and compiler witnesses must be callable"
        )
    if (
        not handler.validation_error_types
        or any(
            not inspect.isclass(value)
            or not issubclass(value, Exception)
            for value in handler.validation_error_types
        )
    ):
        raise CompilerDispatchV4Error(
            "validation_error_types must contain exception classes"
        )
    integrity = getattr(
        handler.compiled_type,
        handler.integrity_method_name,
        None,
    )
    if not callable(integrity):
        raise CompilerDispatchV4Error(
            "compiled type lacks the registered integrity method"
        )

    registry = handler.specific_registry_entry
    for field_name, expected in (
        ("route_id", handler.route_id),
        ("adjacency", handler.adjacency),
        ("sampler", handler.sampler),
    ):
        if getattr(registry, field_name, None) != expected:
            raise CompilerDispatchV4Error(
                f"specific registry does not bind {field_name}"
            )
    if getattr(registry, "executor_status", "") not in {
        "implemented_research_validated",
        "implemented_release_validated",
    }:
        raise CompilerDispatchV4Error(
            "registered executor has not passed its route-specific gate"
        )

    obligation_ids = {
        witness.obligation_id
        for witness in handler.obligation_witnesses
    }
    if len(obligation_ids) != len(handler.obligation_witnesses):
        raise CompilerDispatchV4Error(
            "registration contains duplicate obligation witnesses"
        )
    if obligation_ids != REQUIRED_REGISTRATION_OBLIGATIONS_V4:
        missing = sorted(
            REQUIRED_REGISTRATION_OBLIGATIONS_V4 - obligation_ids
        )
        extra = sorted(
            obligation_ids - REQUIRED_REGISTRATION_OBLIGATIONS_V4
        )
        raise CompilerDispatchV4Error(
            f"registration obligation mismatch; missing={missing}, extra={extra}"
        )

    witnesses = {
        witness.obligation_id: witness
        for witness in handler.obligation_witnesses
    }
    expected_callable_ids = {
        "strict_parser": _implementation_id(handler.parser),
        "strict_compiler": _implementation_id(handler.compiler),
        "executor": str(getattr(registry, "implementation_id", "")),
        "postcompile_integrity": _implementation_id(integrity),
    }
    for obligation_id, expected in expected_callable_ids.items():
        if witnesses[obligation_id].implementation_id != expected:
            raise CompilerDispatchV4Error(
                f"{obligation_id} witness is spliced"
            )

    root = _repo_root().resolve()
    for witness in handler.obligation_witnesses:
        source_path = _validated_source_path(witness.source_path)
        if _file_sha256(source_path) != witness.source_sha256:
            raise CompilerDispatchV4Error(
                f"{witness.obligation_id} source digest changed"
            )
        implementation = _resolve_implementation_id(
            witness.implementation_id
        )
        if not callable(implementation):
            raise CompilerDispatchV4Error(
                f"{witness.obligation_id} implementation is not callable"
            )
        source_file = inspect.getsourcefile(implementation)
        if source_file is None:
            raise CompilerDispatchV4Error(
                f"{witness.obligation_id} source file is unavailable"
            )
        observed_source = Path(source_file).resolve()
        try:
            observed_relative = str(
                observed_source.relative_to(root)
            ).replace("\\", "/")
        except ValueError as exc:
            raise CompilerDispatchV4Error(
                f"{witness.obligation_id} implementation is external"
            ) from exc
        if observed_relative != witness.source_path:
            raise CompilerDispatchV4Error(
                f"{witness.obligation_id} source path is spliced"
            )

    evidence_path = _validated_source_path(
        handler.evidence_report_path
    )
    if _file_sha256(evidence_path) != handler.evidence_report_sha256:
        raise CompilerDispatchV4Error(
            "registration evidence report digest changed"
        )
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CompilerDispatchV4Error(
            "registration evidence report is unreadable"
        ) from exc
    if (
        not isinstance(evidence, dict)
        or evidence.get("status") != "PASS"
    ):
        raise CompilerDispatchV4Error(
            "registration evidence report is not a passing object"
        )
    evidence_without_hash = dict(evidence)
    reported_payload_sha256 = evidence_without_hash.pop(
        "payload_sha256",
        None,
    )
    if (
        not isinstance(reported_payload_sha256, str)
        or _payload_sha256(evidence_without_hash)
        != reported_payload_sha256
    ):
        raise CompilerDispatchV4Error(
            "registration evidence report payload digest is invalid"
        )
    evidence_core = evidence.get("registration_core")
    if (
        evidence.get("route_id") != handler.route_id
        or evidence.get("schema_id") != handler.schema_version
        or evidence.get("registration_core_sha256")
        != handler.registration_core_sha256
        or not isinstance(evidence_core, dict)
        or _payload_sha256(evidence_core)
        != handler.registration_core_sha256
    ):
        raise CompilerDispatchV4Error(
            "registration evidence does not bind the complete handler core"
        )


def build_route_dispatch_registry_v4(
    handlers: tuple[RouteHandlerV4, ...],
) -> MappingProxyType[str, RouteHandlerV4]:
    """Return an immutable registry only after every judgment validates."""

    result: dict[str, RouteHandlerV4] = {}
    for handler in handlers:
        validate_route_handler_v4(handler)
        if handler.route_id in result:
            raise CompilerDispatchV4Error(
                f"duplicate route registration: {handler.route_id!r}"
            )
        result[handler.route_id] = handler
    if not result:
        raise CompilerDispatchV4Error(
            "route dispatch registry must not be empty"
        )
    return MappingProxyType(result)


def _witness(
    obligation_id: str,
    implementation_id: str,
    source_path: str,
    source_sha256: str,
) -> RouteObligationWitnessV4:
    return RouteObligationWitnessV4(
        obligation_id=obligation_id,
        implementation_id=implementation_id,
        source_path=source_path,
        source_sha256=source_sha256,
    )


POISSON_HANDLER_V4 = RouteHandlerV4(
    route_id=CORE_ROUTE_ID_V2,
    schema_version=CONTRACT_SCHEMA_V2,
    adjacency="add_remove_one_owner",
    sampler="independent_bernoulli_owner",
    accountant_family="poisson_gaussian_rdp",
    contract_type=OwnerPoissonContractV2,
    compiled_type=CompiledOwnerPoissonRouteV2,
    parser=parse_owner_poisson_contract_v2,
    compiler=compile_owner_poisson_contract_v2,
    validation_error_types=(ContractV2Error,),
    integrity_method_name="assert_private_execution_integrity",
    specific_registry_entry=ROUTE_REGISTRY_V2[CORE_ROUTE_ID_V2],
    evidence_report_path=(
        "reports/v35_route_handler_evidence_20260725/"
        "poisson_handler_evidence_v35.json"
    ),
    evidence_report_sha256=(
        "b03df3b2571c4aa712724a6eecc9060d987815739fda08b298c0d97c73058c74"
    ),
    obligation_witnesses=(
        _witness(
            "strict_parser",
            "unitdp.compiler_v2.parse_owner_poisson_contract_v2",
            "src/unitdp/compiler_v2.py",
            "6664f38f95ed9d6ab8d77c1fbade62d5464ac1c073f2d8a17ce87f04bcc69745",
        ),
        _witness(
            "strict_compiler",
            "unitdp.compiler_v2.compile_owner_poisson_contract_v2",
            "src/unitdp/compiler_v2.py",
            "6664f38f95ed9d6ab8d77c1fbade62d5464ac1c073f2d8a17ce87f04bcc69745",
        ),
        _witness(
            "accountant",
            "unitdp.accountant.epsilon_for_noise",
            "src/unitdp/accountant.py",
            "73094bb9c6af57c0164a6e7b68be422f0fcf94c83a647f9e79350df064a8e58f",
        ),
        _witness(
            "numeric_crosscheck",
            "unitdp.accountant.epsilon_for_noise_dp_accounting_add_remove",
            "src/unitdp/accountant.py",
            "73094bb9c6af57c0164a6e7b68be422f0fcf94c83a647f9e79350df064a8e58f",
        ),
        _witness(
            "executor",
            "unitdp.owa_dpsgd.train_owner_poisson_fixed_v2",
            "src/unitdp/owa_dpsgd.py",
            "8fc7f512af47be7acf6bb3103484cd9b7fd4bab2dacbcf820065b3eaaee958a8",
        ),
        _witness(
            "postcompile_integrity",
            (
                "unitdp.compiler_v2.CompiledOwnerPoissonRouteV2."
                "assert_private_execution_integrity"
            ),
            "src/unitdp/compiler_v2.py",
            "6664f38f95ed9d6ab8d77c1fbade62d5464ac1c073f2d8a17ce87f04bcc69745",
        ),
        _witness(
            "source_binding",
            "unitdp.source_bundle_v2.execution_source_bundle_sha256_v2",
            "src/unitdp/source_bundle_v2.py",
            "4efe9d5b66cccc33b4a9fc5782219ba4a03fa62cd6700048eadddc444a619970",
        ),
    ),
)


SRSWOR_HANDLER_V4 = RouteHandlerV4(
    route_id=CORE_ROUTE_ID_SRSWOR_V3,
    schema_version=CONTRACT_SCHEMA_SRSWOR_V3,
    adjacency="replace_one_owner",
    sampler="uniform_fixed_size_without_replacement_owner",
    accountant_family="srswor_gaussian_rdp",
    contract_type=OwnerSrsworContractV3,
    compiled_type=CompiledOwnerSrsworRouteV3,
    parser=parse_owner_srswor_contract_v3,
    compiler=compile_owner_srswor_contract_v3,
    validation_error_types=(SrsworContractV3Error,),
    integrity_method_name="assert_private_execution_integrity",
    specific_registry_entry=(
        ROUTE_REGISTRY_SRSWOR_V3[CORE_ROUTE_ID_SRSWOR_V3]
    ),
    evidence_report_path=(
        "reports/v35_route_handler_evidence_20260725/"
        "srswor_handler_evidence_v35.json"
    ),
    evidence_report_sha256=(
        "7efb77f324eb53f8f712479a53fb55d7c949320dfa403ca9390dc9bfa1f488e1"
    ),
    obligation_witnesses=(
        _witness(
            "strict_parser",
            "unitdp.compiler_srswor_v3.parse_owner_srswor_contract_v3",
            "src/unitdp/compiler_srswor_v3.py",
            "ea4b929ed23a2bb6a7f25cdc462b24fdd77535f70f3ce86d62201a69a3cdb6fc",
        ),
        _witness(
            "strict_compiler",
            "unitdp.compiler_srswor_v3.compile_owner_srswor_contract_v3",
            "src/unitdp/compiler_srswor_v3.py",
            "ea4b929ed23a2bb6a7f25cdc462b24fdd77535f70f3ce86d62201a69a3cdb6fc",
        ),
        _witness(
            "accountant",
            "unitdp.accountant.epsilon_for_srswor_replace_one_noise",
            "src/unitdp/accountant.py",
            "73094bb9c6af57c0164a6e7b68be422f0fcf94c83a647f9e79350df064a8e58f",
        ),
        _witness(
            "numeric_crosscheck",
            (
                "unitdp_spec_oracle.srswor_rdp_oracle_v3."
                "fixed_size_srswor_epsilon_v3"
            ),
            "src/unitdp_spec_oracle/srswor_rdp_oracle_v3.py",
            "82d810b8d464994833dcfc107c929949f225cabca231451e9302fd3d8b5f7f32",
        ),
        _witness(
            "executor",
            "unitdp.owner_srswor_v3.train_owner_srswor_fixed_v3",
            "src/unitdp/owner_srswor_v3.py",
            "d1ee843a8057745725bdb232de7dd4a8f4ac280a6601d27fecaca49fa4fe9817",
        ),
        _witness(
            "postcompile_integrity",
            (
                "unitdp.compiler_srswor_v3.CompiledOwnerSrsworRouteV3."
                "assert_private_execution_integrity"
            ),
            "src/unitdp/compiler_srswor_v3.py",
            "ea4b929ed23a2bb6a7f25cdc462b24fdd77535f70f3ce86d62201a69a3cdb6fc",
        ),
        _witness(
            "source_binding",
            (
                "unitdp.source_bundle_srswor_v3."
                "execution_source_bundle_sha256_srswor_v3"
            ),
            "src/unitdp/source_bundle_srswor_v3.py",
            "6bc4044666cd51400c16bd68bb3403dc2305eb380a8ad0f5c85642592395774b",
        ),
    ),
)


ALLOCATION_HANDLER_V4 = RouteHandlerV4(
    route_id=CORE_ROUTE_ID_ALLOCATION_V4,
    schema_version=CONTRACT_SCHEMA_ALLOCATION_V4,
    adjacency="add_remove_one_owner",
    sampler="independent_uniform_k_subset_of_steps_per_owner",
    accountant_family="random_allocation_gaussian_direct",
    contract_type=OwnerRandomAllocationContractV4,
    compiled_type=CompiledOwnerRandomAllocationRouteV4,
    parser=parse_owner_random_allocation_contract_v4,
    compiler=compile_owner_random_allocation_contract_v4,
    validation_error_types=(AllocationContractV4Error,),
    integrity_method_name="assert_private_execution_integrity",
    specific_registry_entry=(
        ROUTE_REGISTRY_ALLOCATION_V4[CORE_ROUTE_ID_ALLOCATION_V4]
    ),
    evidence_report_path=(
        "reports/v35_route_handler_evidence_20260725/"
        "allocation_handler_evidence_v35.json"
    ),
    evidence_report_sha256=(
        "5f167e7a319fd68f5d5f4f67956d685fa15550bc8611b08c6e009500f02eadca"
    ),
    obligation_witnesses=(
        _witness(
            "strict_parser",
            (
                "unitdp.compiler_allocation_v4."
                "parse_owner_random_allocation_contract_v4"
            ),
            "src/unitdp/compiler_allocation_v4.py",
            "fd39201a76f4c806a6f0a8903e05fb6987f48bc01a715f1d79e21e4a67556564",
        ),
        _witness(
            "strict_compiler",
            (
                "unitdp.compiler_allocation_v4."
                "compile_owner_random_allocation_contract_v4"
            ),
            "src/unitdp/compiler_allocation_v4.py",
            "fd39201a76f4c806a6f0a8903e05fb6987f48bc01a715f1d79e21e4a67556564",
        ),
        _witness(
            "accountant",
            (
                "unitdp.random_allocation_accountant_v4."
                "account_random_allocation_v4"
            ),
            "src/unitdp/random_allocation_accountant_v4.py",
            "529622d58cbd48cc19da1592bd6d97398510b5a04d08b31169f6b6ffd1dc9e8c",
        ),
        _witness(
            "numeric_crosscheck",
            (
                "unitdp_spec_oracle.random_allocation_oracle_v4."
                "account_random_allocation_oracle_v4"
            ),
            "src/unitdp_spec_oracle/random_allocation_oracle_v4.py",
            "bfbaf62ba6e1e801531e47cca97d8284dfc58e7841d9f1a9eceec942497fa61d",
        ),
        _witness(
            "executor",
            (
                "unitdp.owner_random_allocation_v4."
                "train_owner_random_allocation_fixed_v4"
            ),
            "src/unitdp/owner_random_allocation_v4.py",
            "869dffb013952c926dec3d7117c308217dbc504b724a8383b10290726743bc6b",
        ),
        _witness(
            "postcompile_integrity",
            (
                "unitdp.compiler_allocation_v4."
                "CompiledOwnerRandomAllocationRouteV4."
                "assert_private_execution_integrity"
            ),
            "src/unitdp/compiler_allocation_v4.py",
            "fd39201a76f4c806a6f0a8903e05fb6987f48bc01a715f1d79e21e4a67556564",
        ),
        _witness(
            "source_binding",
            (
                "unitdp.source_bundle_allocation_v4."
                "execution_source_bundle_sha256_allocation_v4"
            ),
            "src/unitdp/source_bundle_allocation_v4.py",
            "145bbcca7fff23bd0a5634bc1958a0d288c79bb423391502128e86c8dded079c",
        ),
    ),
)


ROUTE_DISPATCH_REGISTRY_V4 = build_route_dispatch_registry_v4(
    (
        POISSON_HANDLER_V4,
        SRSWOR_HANDLER_V4,
        ALLOCATION_HANDLER_V4,
    )
)


def parse_registered_contract_v4(
    raw_value: object,
    *,
    registry: MappingProxyType[str, RouteHandlerV4] = (
        ROUTE_DISPATCH_REGISTRY_V4
    ),
) -> RegisteredContractV4:
    if not isinstance(raw_value, dict):
        raise CompilerDispatchV4Error("contract must be a mapping")
    route_id = raw_value.get("route_id")
    schema_version = raw_value.get("schema_version")
    if not isinstance(route_id, str):
        raise CompilerDispatchV4Error("route_id must be a string")
    try:
        handler = registry[route_id]
    except KeyError as exc:
        raise CompilerDispatchV4Error(
            f"Unsupported registered route_id: {route_id!r}"
        ) from exc
    validate_route_handler_v4(handler)
    if schema_version != handler.schema_version:
        raise CompilerDispatchV4Error(
            f"route {route_id!r} requires schema "
            f"{handler.schema_version!r}"
        )
    try:
        contract = handler.parser(raw_value)
    except handler.validation_error_types as exc:
        raise CompilerDispatchV4Error(
            f"Route-specific contract validation failed: {exc}"
        ) from exc
    if type(contract) is not handler.contract_type:
        raise CompilerDispatchV4Error(
            "registered parser returned the wrong contract type"
        )
    if (
        getattr(contract, "route_id", None) != handler.route_id
        or getattr(contract, "schema_version", None)
        != handler.schema_version
    ):
        raise CompilerDispatchV4Error(
            "registered parser returned a spliced contract"
        )
    return contract


def load_registered_contract_v4(
    path: str | Path,
    *,
    registry: MappingProxyType[str, RouteHandlerV4] = (
        ROUTE_DISPATCH_REGISTRY_V4
    ),
) -> RegisteredContractV4:
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
    except (
        ContractV2Error,
        json.JSONDecodeError,
        yaml.YAMLError,
    ) as exc:
        raise CompilerDispatchV4Error(
            f"Could not load strict registered contract: {exc}"
        ) from exc
    return parse_registered_contract_v4(raw, registry=registry)


def compile_registered_contract_v4(
    contract: RegisteredContractV4,
    *,
    mapping_path: str | Path,
    preprocessing_artifact_path: str | Path,
    require_executable: bool = False,
    registry: MappingProxyType[str, RouteHandlerV4] = (
        ROUTE_DISPATCH_REGISTRY_V4
    ),
) -> CompiledRegisteredRouteV4:
    route_id = getattr(contract, "route_id", None)
    schema_version = getattr(contract, "schema_version", None)
    if not isinstance(route_id, str):
        raise CompilerDispatchV4Error(
            "contract has no registered route_id"
        )
    try:
        handler = registry[route_id]
    except KeyError as exc:
        raise CompilerDispatchV4Error(
            "contract route is not present in the immutable registry"
        ) from exc
    validate_route_handler_v4(handler)
    if (
        schema_version != handler.schema_version
        or type(contract) is not handler.contract_type
    ):
        raise CompilerDispatchV4Error(
            "contract type/schema does not match the registered handler"
        )
    try:
        compiled = handler.compiler(
            contract,
            mapping_path=mapping_path,
            preprocessing_artifact_path=preprocessing_artifact_path,
            require_executable=require_executable,
        )
    except handler.validation_error_types as exc:
        raise CompilerDispatchV4Error(
            f"Route-specific compilation failed: {exc}"
        ) from exc
    if type(compiled) is not handler.compiled_type:
        raise CompilerDispatchV4Error(
            "registered compiler returned the wrong compiled type"
        )
    if getattr(compiled, "contract", None) != contract:
        raise CompilerDispatchV4Error(
            "registered compiler changed the contract"
        )
    integrity = getattr(compiled, handler.integrity_method_name, None)
    if not callable(integrity):
        raise CompilerDispatchV4Error(
            "compiled object lost its integrity witness"
        )
    try:
        integrity()
    except handler.validation_error_types as exc:
        raise CompilerDispatchV4Error(
            f"Postcompile integrity failed: {exc}"
        ) from exc
    return compiled
