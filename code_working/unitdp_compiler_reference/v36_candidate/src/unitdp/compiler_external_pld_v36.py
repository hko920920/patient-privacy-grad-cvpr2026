"""Strict external-PLD handler layered over the frozen allocation route."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from unitdp.benchmark_registry_allocation_v4 import (
    CONTRACT_SCHEMA_ALLOCATION_V4,
    CORE_ROUTE_ID_ALLOCATION_V4,
    EXECUTOR_IMPLEMENTATION_ID_ALLOCATION_V4,
    ORACLE_IMPLEMENTATION_ID_ALLOCATION_V4,
    RDP_ORDERS_ID_ALLOCATION_V4,
    REDUCTION_ID_ALLOCATION_V4,
    SAMPLER_ID_ALLOCATION_V4,
)
from unitdp.compiler_allocation_v4 import (
    AllocationContractV4Error,
    AllocationExecutorNotReadyV4Error,
    CompiledOwnerRandomAllocationRouteV4,
    OwnerRandomAllocationContractV4,
    compile_owner_random_allocation_contract_v4,
    parse_owner_random_allocation_contract_v4,
)
from unitdp.compiler_v2 import _sha256_payload
from unitdp.external_pld_backend_v36 import (
    CONVOLUTION_METHOD_V36,
    LOSS_DISCRETIZATION_V36,
    LOWER_BOUND_TYPE_V36,
    NUMERICAL_LIBRARY_VERSIONS_V36,
    PLD_DISTRIBUTION_V36,
    PLD_LICENSE_V36,
    PLD_PAPER_URL_V36,
    PLD_RECORD_SHA256_V36,
    PLD_REPOSITORY_URL_V36,
    PLD_TREE_SHA256_V36,
    PLD_VERSION_V36,
    PLD_WHEEL_SHA256_V36,
    TAIL_TRUNCATION_V36,
    TRANSITIVE_DISTRIBUTION_V36,
    TRANSITIVE_LICENSE_V36,
    TRANSITIVE_RECORD_SHA256_V36,
    TRANSITIVE_REPOSITORY_URL_V36,
    TRANSITIVE_TREE_SHA256_V36,
    TRANSITIVE_VERSION_V36,
    TRANSITIVE_WHEEL_SHA256_V36,
    UPPER_BOUND_TYPE_V36,
    ExternalPldBackendV36Error,
    ExternalPldRequestV36,
    ExternalPldResponseV36,
    account_external_pld_v36,
    build_external_pld_request_v36,
    parse_external_pld_response_v36,
    validate_external_pld_numeric_crosscheck_v36,
    verify_external_pld_environment_v36,
)
from unitdp.random_allocation_accountant_v4 import (
    RANDOM_ALLOCATION_ACCOUNTANT_ID_V4,
    RANDOM_ALLOCATION_THEOREM_ID_V4,
)
from unitdp.source_bundle_external_pld_v36 import (
    SourceBundleExternalPldV36Error,
    compiler_source_bundle_sha256_external_pld_v36,
    execution_source_bundle_sha256_external_pld_v36,
)


CONTRACT_SCHEMA_EXTERNAL_PLD_V36 = (
    "unitdp.owner_random_allocation_external_pld_contract.v36"
)
CORE_ROUTE_ID_EXTERNAL_PLD_V36 = (
    "owa_owner_random_allocation_external_pld_add_remove_v36"
)
EXECUTOR_IMPLEMENTATION_ID_EXTERNAL_PLD_V36 = (
    "unitdp.owner_external_pld_v36.train_owner_external_pld_fixed_v36"
)
EXTERNAL_PLD_ACCOUNTANT_ID_V36 = "random_allocation_gaussian_external_pld_v36"
EXTERNAL_PLD_THEOREM_ID_V36 = "shenfeld_2026_efficient_pld_random_allocation"
EXTERNAL_PLD_DIRECTION_POLICY_V36 = "joint_add_remove_dominating_pld"
EXTERNAL_PLD_BACKEND_MODE_V36 = "isolated_subprocess_site"


class ExternalPldContractV36Error(AllocationContractV4Error):
    """Raised when the V3.6 handler cannot preserve its exact contract."""


class ExternalPldExecutorNotReadyV36Error(RuntimeError):
    """Raised when executable H compilation is requested before validation."""


@dataclass(frozen=True)
class RouteRegistryEntryExternalPldV36:
    route_id: str
    contract_schema: str
    contract_status: str
    executor_status: str
    implementation_id: str
    adjacency: str
    sampler: str
    accountant_id: str
    accountant_family: str
    theorem_id: str
    direction_policy: str
    backend_mode: str
    primary_distribution: str
    primary_version: str
    primary_license: str
    primary_repository_url: str
    primary_paper_url: str
    primary_wheel_sha256: str
    primary_record_sha256: str
    primary_tree_sha256: str
    declared_transitive_distribution: str
    declared_transitive_version: str
    declared_transitive_license: str
    declared_transitive_repository_url: str
    declared_transitive_wheel_sha256: str
    declared_transitive_record_sha256: str
    declared_transitive_tree_sha256: str
    numerical_library_versions: tuple[tuple[str, str], ...]
    loss_discretization: float
    tail_truncation: float
    convolution_method: str
    accepted_bound_type: str
    crosscheck_bound_type: str
    nested_route_id: str
    required_invariants: tuple[str, ...]


ROUTE_REGISTRY_EXTERNAL_PLD_V36 = MappingProxyType(
    {
        CORE_ROUTE_ID_EXTERNAL_PLD_V36: (
            RouteRegistryEntryExternalPldV36(
                route_id=CORE_ROUTE_ID_EXTERNAL_PLD_V36,
                contract_schema=CONTRACT_SCHEMA_EXTERNAL_PLD_V36,
                contract_status=("strict_external_accountant_overlay_implemented"),
                executor_status="implemented_research_validated",
                implementation_id=(EXECUTOR_IMPLEMENTATION_ID_EXTERNAL_PLD_V36),
                adjacency="add_remove_one_owner",
                sampler=SAMPLER_ID_ALLOCATION_V4,
                accountant_id=EXTERNAL_PLD_ACCOUNTANT_ID_V36,
                accountant_family="external_random_allocation_pld",
                theorem_id=EXTERNAL_PLD_THEOREM_ID_V36,
                direction_policy=EXTERNAL_PLD_DIRECTION_POLICY_V36,
                backend_mode=EXTERNAL_PLD_BACKEND_MODE_V36,
                primary_distribution=PLD_DISTRIBUTION_V36,
                primary_version=PLD_VERSION_V36,
                primary_license=PLD_LICENSE_V36,
                primary_repository_url=PLD_REPOSITORY_URL_V36,
                primary_paper_url=PLD_PAPER_URL_V36,
                primary_wheel_sha256=PLD_WHEEL_SHA256_V36,
                primary_record_sha256=PLD_RECORD_SHA256_V36,
                primary_tree_sha256=PLD_TREE_SHA256_V36,
                declared_transitive_distribution=(TRANSITIVE_DISTRIBUTION_V36),
                declared_transitive_version=TRANSITIVE_VERSION_V36,
                declared_transitive_license=TRANSITIVE_LICENSE_V36,
                declared_transitive_repository_url=(TRANSITIVE_REPOSITORY_URL_V36),
                declared_transitive_wheel_sha256=(TRANSITIVE_WHEEL_SHA256_V36),
                declared_transitive_record_sha256=(TRANSITIVE_RECORD_SHA256_V36),
                declared_transitive_tree_sha256=(TRANSITIVE_TREE_SHA256_V36),
                numerical_library_versions=(NUMERICAL_LIBRARY_VERSIONS_V36),
                loss_discretization=LOSS_DISCRETIZATION_V36,
                tail_truncation=TAIL_TRUNCATION_V36,
                convolution_method=CONVOLUTION_METHOD_V36,
                accepted_bound_type=UPPER_BOUND_TYPE_V36,
                crosscheck_bound_type=LOWER_BOUND_TYPE_V36,
                nested_route_id=CORE_ROUTE_ID_ALLOCATION_V4,
                required_invariants=(
                    "same registered owner-allocation mechanism as route A",
                    "strict mechanical H-to-A contract projection",
                    "frozen nested A compiler and research executor",
                    "isolated external backend location supplied explicitly",
                    "exact external distribution version and source tree",
                    "exact declared transitive version and source tree",
                    "exact numerical-library versions",
                    "dominating PLD bound used as the accepted epsilon",
                    "dominated PLD bound used only as a bracket cross-check",
                    "registered numerical grid and tail truncation",
                    "external response bound to the exact request digest",
                    "nested A and H source bundles unchanged",
                    "fixed-case evidence is not open-world plugin safety",
                ),
            )
        )
    }
)


def get_route_registry_entry_external_pld_v36(
    route_id: str,
) -> RouteRegistryEntryExternalPldV36:
    try:
        return ROUTE_REGISTRY_EXTERNAL_PLD_V36[route_id]
    except KeyError as exc:
        raise ExternalPldContractV36Error(
            f"Unsupported external-PLD V3.6 route_id: {route_id!r}"
        ) from exc


def _external_accountant_payload() -> dict[str, Any]:
    return {
        "id": EXTERNAL_PLD_ACCOUNTANT_ID_V36,
        "package": PLD_DISTRIBUTION_V36,
        "package_version": PLD_VERSION_V36,
        "declared_transitive_package": (TRANSITIVE_DISTRIBUTION_V36),
        "declared_transitive_version": TRANSITIVE_VERSION_V36,
        "theorem_id": EXTERNAL_PLD_THEOREM_ID_V36,
        "direction_policy": EXTERNAL_PLD_DIRECTION_POLICY_V36,
        "backend_mode": EXTERNAL_PLD_BACKEND_MODE_V36,
        "l2_sensitivity_multiplier": 1.0,
        "loss_discretization": LOSS_DISCRETIZATION_V36,
        "tail_truncation": TAIL_TRUNCATION_V36,
        "convolution_method": CONVOLUTION_METHOD_V36,
        "accepted_bound_type": UPPER_BOUND_TYPE_V36,
        "crosscheck_bound_type": LOWER_BOUND_TYPE_V36,
    }


def _base_accountant_payload() -> dict[str, Any]:
    return {
        "id": RANDOM_ALLOCATION_ACCOUNTANT_ID_V4,
        "rdp_orders_id": RDP_ORDERS_ID_ALLOCATION_V4,
        "sensitivity_multiplier": 1.0,
        "theorem_id": RANDOM_ALLOCATION_THEOREM_ID_V4,
        "oracle_implementation_id": (ORACLE_IMPLEMENTATION_ID_ALLOCATION_V4),
        "reduction_id": REDUCTION_ID_ALLOCATION_V4,
        "direction_policy": "max_add_remove",
    }


def _expect_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExternalPldContractV36Error(f"{label} must be a mapping")
    return value


def _expect_keys(
    value: dict[str, Any],
    label: str,
    expected: set[str],
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ExternalPldContractV36Error(
            f"{label} fields mismatch; missing={missing}, extra={extra}"
        )


def _validate_external_accountant_payload(value: object) -> None:
    observed = _expect_mapping(value, "accountant")
    expected = _external_accountant_payload()
    _expect_keys(observed, "accountant", set(expected))
    for key, expected_value in expected.items():
        actual = observed[key]
        if isinstance(expected_value, float):
            if (
                isinstance(actual, bool)
                or not isinstance(actual, (int, float))
                or float(actual) != expected_value
            ):
                raise ExternalPldContractV36Error(
                    f"accountant.{key} must equal {expected_value!r}"
                )
        elif type(actual) is not type(expected_value) or actual != expected_value:
            raise ExternalPldContractV36Error(
                f"accountant.{key} must equal {expected_value!r}"
            )


def _project_external_raw_to_base_v4(
    raw_value: object,
) -> dict[str, Any]:
    raw = _expect_mapping(raw_value, "contract")
    _expect_keys(
        raw,
        "contract",
        {
            "schema_version",
            "route_id",
            "execution",
            "privacy",
            "mechanism",
            "accountant",
            "contribution_policy",
            "optimization",
            "data",
        },
    )
    if raw["schema_version"] != CONTRACT_SCHEMA_EXTERNAL_PLD_V36:
        raise ExternalPldContractV36Error(
            "schema_version is not the registered V3.6 schema"
        )
    if raw["route_id"] != CORE_ROUTE_ID_EXTERNAL_PLD_V36:
        raise ExternalPldContractV36Error(
            "route_id is not the registered external-PLD route"
        )
    execution = _expect_mapping(raw["execution"], "execution")
    _expect_keys(
        execution,
        "execution",
        {
            "profile",
            "rng_backend",
            "implementation_id",
            "random_coins_public",
        },
    )
    if execution["implementation_id"] != EXECUTOR_IMPLEMENTATION_ID_EXTERNAL_PLD_V36:
        raise ExternalPldContractV36Error(
            "execution.implementation_id is not the H executor wrapper"
        )
    _validate_external_accountant_payload(raw["accountant"])

    projected = copy.deepcopy(raw)
    projected["schema_version"] = CONTRACT_SCHEMA_ALLOCATION_V4
    projected["route_id"] = CORE_ROUTE_ID_ALLOCATION_V4
    projected["execution"]["implementation_id"] = (
        EXECUTOR_IMPLEMENTATION_ID_ALLOCATION_V4
    )
    projected["accountant"] = _base_accountant_payload()
    return projected


@dataclass(frozen=True)
class OwnerExternalPldAllocationContractV36:
    base_contract: OwnerRandomAllocationContractV4

    @property
    def schema_version(self) -> str:
        return CONTRACT_SCHEMA_EXTERNAL_PLD_V36

    @property
    def route_id(self) -> str:
        return CORE_ROUTE_ID_EXTERNAL_PLD_V36

    @property
    def implementation_id(self) -> str:
        return EXECUTOR_IMPLEMENTATION_ID_EXTERNAL_PLD_V36

    @property
    def public_contract_sha256(self) -> str:
        return _sha256_payload(self.public_payload())

    def public_payload(self) -> dict[str, Any]:
        payload = copy.deepcopy(self.base_contract.public_payload())
        payload["schema_version"] = self.schema_version
        payload["route_id"] = self.route_id
        payload["execution"]["implementation_id"] = self.implementation_id
        payload["accountant"] = _external_accountant_payload()
        return payload


def parse_owner_external_pld_contract_v36(
    raw_value: object,
) -> OwnerExternalPldAllocationContractV36:
    try:
        projected = _project_external_raw_to_base_v4(raw_value)
        base = parse_owner_random_allocation_contract_v4(projected)
        contract = OwnerExternalPldAllocationContractV36(base_contract=base)
        if contract.public_payload() != raw_value:
            raise ExternalPldContractV36Error(
                "External contract is not in canonical registered form"
            )
        return contract
    except ExternalPldContractV36Error:
        raise
    except AllocationContractV4Error as exc:
        raise ExternalPldContractV36Error(str(exc)) from exc


def load_owner_external_pld_contract_v36(
    path: str | Path,
) -> OwnerExternalPldAllocationContractV36:
    from unitdp.compiler_v4 import load_registered_contract_v4
    from unitdp.compiler_v4 import build_route_dispatch_registry_v4
    from unitdp.compiler_v36 import EXTERNAL_PLD_HANDLER_V36

    registry = build_route_dispatch_registry_v4((EXTERNAL_PLD_HANDLER_V36,))
    value = load_registered_contract_v4(path, registry=registry)
    if not isinstance(value, OwnerExternalPldAllocationContractV36):
        raise ExternalPldContractV36Error("Loaded external contract has the wrong type")
    return value


@dataclass(frozen=True)
class CompiledOwnerExternalPldAllocationRouteV36:
    contract: OwnerExternalPldAllocationContractV36
    base_route: CompiledOwnerRandomAllocationRouteV4
    registry_entry: RouteRegistryEntryExternalPldV36
    external_request: ExternalPldRequestV36
    external_response: ExternalPldResponseV36
    compiler_source_bundle_sha256: str
    execution_source_bundle_sha256: str
    public_plan_sha256: str
    private_compilation_binding_sha256: str
    private_execution_binding_sha256: str
    status: str
    notes: tuple[str, ...]

    @property
    def execution_ready(self) -> bool:
        return (
            self.base_route.execution_ready
            and self.contract.implementation_id == self.registry_entry.implementation_id
            and self.registry_entry.executor_status
            in {
                "implemented_research_validated",
                "implemented_release_validated",
            }
        )

    def _public_plan_without_hash(self) -> dict[str, Any]:
        return {
            "schema_version": ("unitdp.compiled_external_pld_public_plan.v36"),
            "status": self.status,
            "execution_ready": self.execution_ready,
            "route_registry": asdict(self.registry_entry),
            "contract": self.contract.public_payload(),
            "nested_allocation_route": {
                "route_id": self.base_route.contract.route_id,
                "public_contract_sha256": (
                    self.base_route.contract.public_contract_sha256
                ),
                "public_plan_sha256": (self.base_route.public_plan_sha256),
                "direct_epsilon": self.base_route.accountant_epsilon,
            },
            "external_accounting": {
                "request": self.external_request.payload(),
                "request_sha256": self.external_request.sha256,
                "response": self.external_response.payload(),
                "accepted_epsilon": (self.external_response.epsilon_upper),
                "crosscheck_epsilon": (self.external_response.epsilon_lower),
                "target_epsilon": (self.contract.base_contract.target_epsilon),
            },
            "compiler_source_bundle_sha256": (self.compiler_source_bundle_sha256),
            "execution_source_bundle_sha256": (self.execution_source_bundle_sha256),
            "notes": list(self.notes),
        }

    def public_plan(self) -> dict[str, Any]:
        payload = self._public_plan_without_hash()
        if _sha256_payload(payload) != self.public_plan_sha256:
            raise ExternalPldContractV36Error(
                "Compiled H public-plan hash is inconsistent"
            )
        return {**payload, "public_plan_sha256": self.public_plan_sha256}

    def assert_private_execution_integrity(self) -> None:
        canonical = parse_owner_external_pld_contract_v36(
            self.contract.public_payload()
        )
        if canonical != self.contract:
            raise ExternalPldContractV36Error(
                "Compiled H contract differs from its canonical form"
            )
        if self.base_route.contract != canonical.base_contract:
            raise ExternalPldContractV36Error(
                "Compiled H route contains a spliced nested A contract"
            )
        try:
            self.base_route.assert_private_execution_integrity()
        except (AllocationContractV4Error, RuntimeError) as exc:
            raise ExternalPldContractV36Error(
                f"Nested A route failed integrity: {exc}"
            ) from exc
        registry = get_route_registry_entry_external_pld_v36(canonical.route_id)
        if registry != self.registry_entry:
            raise ExternalPldContractV36Error("Compiled H registry entry is stale")
        expected_request = build_external_pld_request_v36(
            sigma=canonical.base_contract.noise_multiplier,
            num_steps=canonical.base_contract.num_steps,
            num_selected=(canonical.base_contract.num_selected_steps_per_owner),
            num_epochs=canonical.base_contract.num_epochs,
            delta=canonical.base_contract.delta,
        )
        if expected_request != self.external_request:
            raise ExternalPldContractV36Error("Compiled H external request is stale")
        canonical_response = parse_external_pld_response_v36(
            self.external_response.payload()
        )
        if canonical_response != self.external_response:
            raise ExternalPldContractV36Error(
                "Compiled H external response is not canonical"
            )
        try:
            environment = verify_external_pld_environment_v36()
            validate_external_pld_numeric_crosscheck_v36(
                canonical_response,
                expected_request,
                target_epsilon=canonical.base_contract.target_epsilon,
                direct_epsilon=self.base_route.accountant_epsilon,
            )
        except ExternalPldBackendV36Error as exc:
            raise ExternalPldContractV36Error(
                f"External PLD integrity failed: {exc}"
            ) from exc
        if environment != canonical_response.environment:
            raise ExternalPldContractV36Error("Compiled H environment binding is stale")
        try:
            compiler_bundle = compiler_source_bundle_sha256_external_pld_v36()
            execution_bundle = execution_source_bundle_sha256_external_pld_v36()
        except SourceBundleExternalPldV36Error as exc:
            raise ExternalPldContractV36Error(
                f"Registered H source bundle is unavailable: {exc}"
            ) from exc
        if compiler_bundle != self.compiler_source_bundle_sha256:
            raise ExternalPldContractV36Error(
                "Registered H compiler source bundle changed"
            )
        if execution_bundle != self.execution_source_bundle_sha256:
            raise ExternalPldContractV36Error(
                "Registered H execution source bundle changed"
            )
        expected_binding = _sha256_payload(
            {
                "public_contract_sha256": (canonical.public_contract_sha256),
                "registry_entry": asdict(registry),
                "nested_public_plan_sha256": (self.base_route.public_plan_sha256),
                "nested_private_execution_binding_sha256": (
                    self.base_route.private_execution_binding_sha256
                ),
                "external_request_sha256": expected_request.sha256,
                "external_response_sha256": (canonical_response.response_sha256),
                "compiler_source_bundle_sha256": compiler_bundle,
                "execution_source_bundle_sha256": execution_bundle,
            }
        )
        if expected_binding != self.private_compilation_binding_sha256:
            raise ExternalPldContractV36Error(
                "Compiled H private binding is inconsistent"
            )
        if expected_binding != self.private_execution_binding_sha256:
            raise ExternalPldContractV36Error(
                "Compiled H execution binding is inconsistent"
            )
        self.public_plan()


def compile_owner_external_pld_contract_v36(
    contract: OwnerExternalPldAllocationContractV36,
    *,
    mapping_path: str | Path,
    preprocessing_artifact_path: str | Path,
    require_executable: bool = False,
) -> CompiledOwnerExternalPldAllocationRouteV36:
    """Compile one strict H overlay and seal its external PLD response."""

    if not isinstance(contract, OwnerExternalPldAllocationContractV36):
        raise ExternalPldContractV36Error(
            "H compiler requires OwnerExternalPldAllocationContractV36"
        )
    contract = parse_owner_external_pld_contract_v36(contract.public_payload())
    registry = get_route_registry_entry_external_pld_v36(contract.route_id)
    try:
        base_route = compile_owner_random_allocation_contract_v4(
            contract.base_contract,
            mapping_path=mapping_path,
            preprocessing_artifact_path=preprocessing_artifact_path,
            require_executable=require_executable,
        )
    except (
        AllocationContractV4Error,
        AllocationExecutorNotReadyV4Error,
    ) as exc:
        raise ExternalPldContractV36Error(
            f"Nested A compilation failed: {exc}"
        ) from exc
    external_request = build_external_pld_request_v36(
        sigma=contract.base_contract.noise_multiplier,
        num_steps=contract.base_contract.num_steps,
        num_selected=(contract.base_contract.num_selected_steps_per_owner),
        num_epochs=contract.base_contract.num_epochs,
        delta=contract.base_contract.delta,
    )
    try:
        response = account_external_pld_v36(
            sigma=contract.base_contract.noise_multiplier,
            num_steps=contract.base_contract.num_steps,
            num_selected=(contract.base_contract.num_selected_steps_per_owner),
            num_epochs=contract.base_contract.num_epochs,
            delta=contract.base_contract.delta,
            target_epsilon=contract.base_contract.target_epsilon,
            direct_epsilon=base_route.accountant_epsilon,
        )
    except ExternalPldBackendV36Error as exc:
        raise ExternalPldContractV36Error(
            f"External PLD backend rejected the contract: {exc}"
        ) from exc
    try:
        compiler_bundle = compiler_source_bundle_sha256_external_pld_v36()
        execution_bundle = execution_source_bundle_sha256_external_pld_v36()
    except SourceBundleExternalPldV36Error as exc:
        raise ExternalPldContractV36Error(
            f"Registered H source bundle is unavailable: {exc}"
        ) from exc
    private_binding = _sha256_payload(
        {
            "public_contract_sha256": contract.public_contract_sha256,
            "registry_entry": asdict(registry),
            "nested_public_plan_sha256": (base_route.public_plan_sha256),
            "nested_private_execution_binding_sha256": (
                base_route.private_execution_binding_sha256
            ),
            "external_request_sha256": external_request.sha256,
            "external_response_sha256": response.response_sha256,
            "compiler_source_bundle_sha256": compiler_bundle,
            "execution_source_bundle_sha256": execution_bundle,
        }
    )
    compiled = CompiledOwnerExternalPldAllocationRouteV36(
        contract=contract,
        base_route=base_route,
        registry_entry=registry,
        external_request=external_request,
        external_response=response,
        compiler_source_bundle_sha256=compiler_bundle,
        execution_source_bundle_sha256=execution_bundle,
        public_plan_sha256="",
        private_compilation_binding_sha256=private_binding,
        private_execution_binding_sha256=private_binding,
        status=(
            "validated_external_accountant_research_executor_ready"
            if base_route.execution_ready
            else "validated_external_accountant_executor_not_ready"
        ),
        notes=(
            "H reuses the exact registered A mechanism and executor",
            "the external dominating PLD value is the accepted epsilon",
            "the external dominated value is only a numerical bracket",
            "the backend is externally maintained but not an independent "
            "privacy certification",
            "fixed-case registration is not open-world plugin safety",
        ),
    )
    compiled = replace(
        compiled,
        public_plan_sha256=_sha256_payload(compiled._public_plan_without_hash()),
    )
    compiled.assert_private_execution_integrity()
    if require_executable and not compiled.execution_ready:
        raise ExternalPldExecutorNotReadyV36Error(
            "The H contract is valid, but its executor is not ready"
        )
    return compiled
