"""Strict compiler for the owner random-allocation V4 route."""

from __future__ import annotations

import importlib
import json
from dataclasses import asdict, dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from unitdp.benchmark_registry_allocation_v4 import (
    CONTRACT_SCHEMA_ALLOCATION_V4,
    CORE_ROUTE_ID_ALLOCATION_V4,
    EXECUTOR_IMPLEMENTATION_ID_ALLOCATION_V4,
    ORACLE_IMPLEMENTATION_ID_ALLOCATION_V4,
    RDP_ORDERS_ID_ALLOCATION_V4,
    REDUCTION_ID_ALLOCATION_V4,
    SAMPLER_ID_ALLOCATION_V4,
    AllocationBenchmarkProfileV4,
    get_allocation_benchmark_profile_v4,
)
from unitdp.compiler_v2 import (
    ContractV2Error,
    _UniqueKeySafeLoader,
    _construct_unique_json_object,
    _expect_keys,
    _expect_mapping,
    _load_strict_single_owner_mapping_v2,
    _positive_float,
    _positive_int,
    _preprocessor_state_sha256,
    _require_exact_profile_binding,
    _sha256_payload,
    _sha256_string,
    _verify_registered_data_preparer_callable,
)
from unitdp.mapping import WindowMapping
from unitdp.policies import ContributionPolicy
from unitdp.preprocessing import (
    FixedAffinePreprocessor,
    PreprocessingArtifactError,
    load_fixed_affine_preprocessor,
)
from unitdp.random_allocation_accountant_v4 import (
    RANDOM_ALLOCATION_ACCOUNTANT_ID_V4,
    RANDOM_ALLOCATION_RDP_ORDERS_V4,
    RANDOM_ALLOCATION_THEOREM_ID_V4,
    RandomAllocationAccountingV4Error,
    account_random_allocation_v4,
)
from unitdp.source_bundle_allocation_v4 import (
    SourceBundleAllocationV4Error,
    compiler_source_bundle_sha256_allocation_v4,
    execution_source_bundle_sha256_allocation_v4,
)
from unitdp_spec_oracle.random_allocation_oracle_v4 import (
    account_random_allocation_oracle_v4,
)


class AllocationContractV4Error(ContractV2Error):
    """Raised when a V4 random-allocation invariant is violated."""


class AllocationExecutorNotReadyV4Error(RuntimeError):
    """Raised when executable compilation is requested before validation."""


@dataclass(frozen=True)
class RouteRegistryEntryAllocationV4:
    route_id: str
    contract_schema: str
    contract_status: str
    executor_status: str
    implementation_id: str
    adjacency: str
    sampler: str
    accountant_id: str
    rdp_orders: tuple[int, ...]
    accountant_library_versions: tuple[tuple[str, str], ...]
    theorem_id: str
    oracle_implementation_id: str
    reduction_id: str
    required_invariants: tuple[str, ...]


ROUTE_REGISTRY_ALLOCATION_V4 = MappingProxyType(
    {
        CORE_ROUTE_ID_ALLOCATION_V4: RouteRegistryEntryAllocationV4(
            route_id=CORE_ROUTE_ID_ALLOCATION_V4,
            contract_schema=CONTRACT_SCHEMA_ALLOCATION_V4,
            contract_status="strict_contract_validator_implemented",
            executor_status="implemented_research_validated",
            implementation_id=EXECUTOR_IMPLEMENTATION_ID_ALLOCATION_V4,
            adjacency="add_remove_one_owner",
            sampler=SAMPLER_ID_ALLOCATION_V4,
            accountant_id=RANDOM_ALLOCATION_ACCOUNTANT_ID_V4,
            rdp_orders=RANDOM_ALLOCATION_RDP_ORDERS_V4,
            accountant_library_versions=(
                ("mpmath", "1.3.0"),
                ("scipy", "1.16.1"),
            ),
            theorem_id=RANDOM_ALLOCATION_THEOREM_ID_V4,
            oracle_implementation_id=(
                ORACLE_IMPLEMENTATION_ID_ALLOCATION_V4
            ),
            reduction_id=REDUCTION_ID_ALLOCATION_V4,
            required_invariants=(
                "public fixed owner population",
                "public fixed positive t, k, and epoch count",
                "fresh independent uniform k-subset per owner per epoch",
                "exactly k distinct assigned steps per owner per epoch",
                "allocation random coins remain private",
                "one clipped owner vector per assigned step",
                "Gaussian update on every step including empty steps",
                "public fixed update denominator",
                "add/remove owner adjacency with sensitivity C",
                "conservative maximum of add and remove bounds",
                "direct accountant and high-precision oracle agreement",
                "exact registered full-data mapping",
                "public-fixed preprocessing artifact",
                "single-attribution owner mapping",
            ),
        )
    }
)


def get_route_registry_entry_allocation_v4(
    route_id: str,
) -> RouteRegistryEntryAllocationV4:
    try:
        return ROUTE_REGISTRY_ALLOCATION_V4[route_id]
    except KeyError as exc:
        raise AllocationContractV4Error(
            f"Unsupported random-allocation V4 route_id: {route_id!r}"
        ) from exc


def _verify_accountant_library_versions(
    registry: RouteRegistryEntryAllocationV4,
) -> dict[str, str]:
    observed: dict[str, str] = {}
    for distribution, expected in registry.accountant_library_versions:
        try:
            installed = version(distribution)
        except PackageNotFoundError as exc:
            raise AllocationContractV4Error(
                f"Required accountant dependency is missing: {distribution}"
            ) from exc
        if installed != expected:
            raise AllocationContractV4Error(
                f"Accountant dependency {distribution} must be {expected}, "
                f"got {installed}; register a new accountant version before use"
            )
        observed[distribution] = installed
    return observed


def _verify_registered_callable(implementation_id: str, label: str) -> None:
    module_name, separator, function_name = implementation_id.rpartition(".")
    if not separator or not module_name or not function_name:
        raise AllocationContractV4Error(
            f"Registered {label} implementation id is malformed"
        )
    try:
        module = importlib.import_module(module_name)
        implementation = getattr(module, function_name)
    except (ImportError, AttributeError) as exc:
        raise AllocationContractV4Error(
            f"Registered {label} implementation is not importable: "
            f"{implementation_id}"
        ) from exc
    if not callable(implementation):
        raise AllocationContractV4Error(
            f"Registered {label} implementation is not callable"
        )


def _executor_ready(
    contract: "OwnerRandomAllocationContractV4",
    registry: RouteRegistryEntryAllocationV4,
) -> bool:
    return (
        contract.execution_profile == "research_benchmark"
        and contract.rng_backend == "research_default"
        and registry.executor_status
        in {"implemented_research_validated", "implemented_release_validated"}
    )


@dataclass(frozen=True)
class OwnerRandomAllocationContractV4:
    schema_version: str
    route_id: str
    execution_profile: str
    rng_backend: str
    implementation_id: str
    privacy_unit: str
    adjacency: str
    target_epsilon: float
    delta: float
    sampler: str
    sampling_unit: str
    accounting_unit: str
    parameter_source: str
    source_dataset_size: int
    num_steps: int
    num_selected_steps_per_owner: int
    num_epochs: int
    total_steps: int
    assignment_schedule: str
    clipping_unit: str
    noising_unit: str
    clip_norm: float
    noise_multiplier: float
    update_denominator: float
    empty_step_behavior: str
    per_owner_contribution: str
    owner_aggregation: str
    accountant_id: str
    rdp_orders_id: str
    sensitivity_multiplier: float
    theorem_id: str
    oracle_implementation_id: str
    reduction_id: str
    direction_policy: str
    policy_type: str
    max_windows_per_owner: int
    windows_per_owner_per_step: int
    policy_scope: str
    optimizer: str
    learning_rate: float
    loss: str
    model_id: str
    class_weights: tuple[float, ...] | None
    benchmark_profile_id: str
    benchmark_profile_sha256: str
    base_data_profile_id: str
    base_data_profile_sha256: str
    dataset_id: str
    input_dim: int
    num_classes: int
    preprocessing_binding: dict[str, Any]
    public_protocol: dict[str, Any]

    @property
    def marginal_participation_rate(self) -> float:
        return self.num_selected_steps_per_owner / self.num_steps

    @property
    def expected_owners_per_step(self) -> float:
        return self.source_dataset_size * self.marginal_participation_rate

    @property
    def public_contract_sha256(self) -> str:
        return _sha256_payload(self.public_payload())

    def public_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "route_id": self.route_id,
            "execution": {
                "profile": self.execution_profile,
                "rng_backend": self.rng_backend,
                "implementation_id": self.implementation_id,
                "random_coins_public": False,
            },
            "privacy": {
                "unit": self.privacy_unit,
                "adjacency": self.adjacency,
                "target_epsilon": self.target_epsilon,
                "delta": self.delta,
            },
            "mechanism": {
                "sampler": self.sampler,
                "sampling_unit": self.sampling_unit,
                "accounting_unit": self.accounting_unit,
                "parameter_source": self.parameter_source,
                "source_dataset_size": self.source_dataset_size,
                "num_steps": self.num_steps,
                "num_selected_steps_per_owner": (
                    self.num_selected_steps_per_owner
                ),
                "num_epochs": self.num_epochs,
                "total_steps": self.total_steps,
                "assignment_schedule": self.assignment_schedule,
                "clipping_unit": self.clipping_unit,
                "noising_unit": self.noising_unit,
                "clip_norm": self.clip_norm,
                "noise_multiplier": self.noise_multiplier,
                "update_denominator": self.update_denominator,
                "empty_step_behavior": self.empty_step_behavior,
                "per_owner_contribution": self.per_owner_contribution,
                "owner_aggregation": self.owner_aggregation,
            },
            "accountant": {
                "id": self.accountant_id,
                "rdp_orders_id": self.rdp_orders_id,
                "sensitivity_multiplier": self.sensitivity_multiplier,
                "theorem_id": self.theorem_id,
                "oracle_implementation_id": self.oracle_implementation_id,
                "reduction_id": self.reduction_id,
                "direction_policy": self.direction_policy,
            },
            "contribution_policy": {
                "type": self.policy_type,
                "max_windows_per_owner": self.max_windows_per_owner,
                "windows_per_owner_per_step": (
                    self.windows_per_owner_per_step
                ),
                "scope": self.policy_scope,
            },
            "optimization": {
                "optimizer": self.optimizer,
                "learning_rate": self.learning_rate,
                "loss": self.loss,
                "model_id": self.model_id,
                "class_weights": (
                    list(self.class_weights)
                    if self.class_weights is not None
                    else None
                ),
            },
            "data": {
                "benchmark_profile_id": self.benchmark_profile_id,
                "benchmark_profile_sha256": (
                    self.benchmark_profile_sha256
                ),
                "base_data_profile_id": self.base_data_profile_id,
                "base_data_profile_sha256": (
                    self.base_data_profile_sha256
                ),
                "dataset_id": self.dataset_id,
                "input_dim": self.input_dim,
                "num_classes": self.num_classes,
                "preprocessing": dict(self.preprocessing_binding),
                "public_protocol": dict(self.public_protocol),
            },
        }


def _parse_owner_random_allocation_contract_v4_impl(
    raw_value: object,
) -> OwnerRandomAllocationContractV4:
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
    if raw["schema_version"] != CONTRACT_SCHEMA_ALLOCATION_V4:
        raise AllocationContractV4Error(
            f"schema_version must be {CONTRACT_SCHEMA_ALLOCATION_V4!r}"
        )
    if raw["route_id"] != CORE_ROUTE_ID_ALLOCATION_V4:
        raise AllocationContractV4Error(
            f"route_id must be {CORE_ROUTE_ID_ALLOCATION_V4!r}"
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
    _require_exact_profile_binding(
        execution,
        {
            "profile": "research_benchmark",
            "rng_backend": "research_default",
            "implementation_id": (
                EXECUTOR_IMPLEMENTATION_ID_ALLOCATION_V4
            ),
            "random_coins_public": False,
        },
        "execution",
    )
    if execution["random_coins_public"] is not False:
        raise AllocationContractV4Error(
            "execution.random_coins_public must be the boolean false"
        )

    sections = {
        name: _expect_mapping(raw[name], name)
        for name in (
            "privacy",
            "mechanism",
            "accountant",
            "contribution_policy",
            "optimization",
            "data",
        )
    }
    profile_id = sections["data"].get("benchmark_profile_id")
    if not isinstance(profile_id, str):
        raise AllocationContractV4Error(
            "data.benchmark_profile_id must be a string"
        )
    try:
        profile = get_allocation_benchmark_profile_v4(profile_id)
    except KeyError as exc:
        raise AllocationContractV4Error(str(exc)) from exc
    expected = profile.expected_contract_binding()
    for name, value in sections.items():
        _require_exact_profile_binding(value, expected[name], name)

    privacy = sections["privacy"]
    mechanism = sections["mechanism"]
    accountant = sections["accountant"]
    policy = sections["contribution_policy"]
    optimization = sections["optimization"]
    data = sections["data"]

    target_epsilon = _positive_float(
        privacy["target_epsilon"],
        "privacy.target_epsilon",
    )
    delta = _positive_float(privacy["delta"], "privacy.delta")
    if delta >= 1:
        raise AllocationContractV4Error(
            "privacy.delta must be smaller than one"
        )
    source_dataset_size = _positive_int(
        mechanism["source_dataset_size"],
        "mechanism.source_dataset_size",
    )
    num_steps = _positive_int(
        mechanism["num_steps"],
        "mechanism.num_steps",
    )
    num_selected = _positive_int(
        mechanism["num_selected_steps_per_owner"],
        "mechanism.num_selected_steps_per_owner",
    )
    if num_selected > num_steps:
        raise AllocationContractV4Error(
            "num_selected_steps_per_owner must not exceed num_steps"
        )
    num_epochs = _positive_int(
        mechanism["num_epochs"],
        "mechanism.num_epochs",
    )
    total_steps = _positive_int(
        mechanism["total_steps"],
        "mechanism.total_steps",
    )
    if total_steps != num_steps * num_epochs:
        raise AllocationContractV4Error(
            "mechanism.total_steps must equal num_steps times num_epochs"
        )
    clip_norm = _positive_float(
        mechanism["clip_norm"],
        "mechanism.clip_norm",
    )
    noise_multiplier = _positive_float(
        mechanism["noise_multiplier"],
        "mechanism.noise_multiplier",
    )
    update_denominator = _positive_float(
        mechanism["update_denominator"],
        "mechanism.update_denominator",
    )
    sensitivity_multiplier = _positive_float(
        accountant["sensitivity_multiplier"],
        "accountant.sensitivity_multiplier",
    )
    if sensitivity_multiplier != 1.0:
        raise AllocationContractV4Error(
            "accountant.sensitivity_multiplier must equal 1.0"
        )
    max_windows = _positive_int(
        policy["max_windows_per_owner"],
        "contribution_policy.max_windows_per_owner",
    )
    step_windows = _positive_int(
        policy["windows_per_owner_per_step"],
        "contribution_policy.windows_per_owner_per_step",
    )
    if step_windows > max_windows:
        raise AllocationContractV4Error(
            "windows_per_owner_per_step must not exceed max_windows_per_owner"
        )
    learning_rate = _positive_float(
        optimization["learning_rate"],
        "optimization.learning_rate",
    )
    input_dim = _positive_int(data["input_dim"], "data.input_dim")
    num_classes = _positive_int(data["num_classes"], "data.num_classes")
    benchmark_profile_sha256 = _sha256_string(
        data["benchmark_profile_sha256"],
        "data.benchmark_profile_sha256",
    )
    base_data_profile_sha256 = _sha256_string(
        data["base_data_profile_sha256"],
        "data.base_data_profile_sha256",
    )
    preprocessing = _expect_mapping(
        data["preprocessing"],
        "data.preprocessing",
    )
    public_protocol = _expect_mapping(
        data["public_protocol"],
        "data.public_protocol",
    )
    for path, value in (
        (
            "data.preprocessing.artifact_sha256",
            preprocessing["artifact_sha256"],
        ),
        (
            "data.preprocessing.source_reference_sha256",
            preprocessing["source_reference_sha256"],
        ),
        (
            "data.public_protocol.source_reference_sha256",
            public_protocol["source_reference_sha256"],
        ),
        (
            "data.public_protocol.full_data_conformance_sha256",
            public_protocol["full_data_conformance_sha256"],
        ),
    ):
        _sha256_string(value, path)
    _verify_registered_data_preparer_callable(
        str(public_protocol["data_preparation_implementation_id"])
    )

    raw_weights = optimization["class_weights"]
    if raw_weights is None:
        class_weights = None
    else:
        if not isinstance(raw_weights, list) or len(raw_weights) != num_classes:
            raise AllocationContractV4Error(
                "optimization.class_weights must be null or one value per class"
            )
        class_weights = tuple(
            _positive_float(
                value,
                f"optimization.class_weights[{index}]",
            )
            for index, value in enumerate(raw_weights)
        )

    return OwnerRandomAllocationContractV4(
        schema_version=str(raw["schema_version"]),
        route_id=str(raw["route_id"]),
        execution_profile=str(execution["profile"]),
        rng_backend=str(execution["rng_backend"]),
        implementation_id=str(execution["implementation_id"]),
        privacy_unit=str(privacy["unit"]),
        adjacency=str(privacy["adjacency"]),
        target_epsilon=target_epsilon,
        delta=delta,
        sampler=str(mechanism["sampler"]),
        sampling_unit=str(mechanism["sampling_unit"]),
        accounting_unit=str(mechanism["accounting_unit"]),
        parameter_source=str(mechanism["parameter_source"]),
        source_dataset_size=source_dataset_size,
        num_steps=num_steps,
        num_selected_steps_per_owner=num_selected,
        num_epochs=num_epochs,
        total_steps=total_steps,
        assignment_schedule=str(mechanism["assignment_schedule"]),
        clipping_unit=str(mechanism["clipping_unit"]),
        noising_unit=str(mechanism["noising_unit"]),
        clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
        update_denominator=update_denominator,
        empty_step_behavior=str(mechanism["empty_step_behavior"]),
        per_owner_contribution=str(mechanism["per_owner_contribution"]),
        owner_aggregation=str(mechanism["owner_aggregation"]),
        accountant_id=str(accountant["id"]),
        rdp_orders_id=str(accountant["rdp_orders_id"]),
        sensitivity_multiplier=sensitivity_multiplier,
        theorem_id=str(accountant["theorem_id"]),
        oracle_implementation_id=str(
            accountant["oracle_implementation_id"]
        ),
        reduction_id=str(accountant["reduction_id"]),
        direction_policy=str(accountant["direction_policy"]),
        policy_type=str(policy["type"]),
        max_windows_per_owner=max_windows,
        windows_per_owner_per_step=step_windows,
        policy_scope=str(policy["scope"]),
        optimizer=str(optimization["optimizer"]),
        learning_rate=learning_rate,
        loss=str(optimization["loss"]),
        model_id=str(optimization["model_id"]),
        class_weights=class_weights,
        benchmark_profile_id=profile.profile_id,
        benchmark_profile_sha256=benchmark_profile_sha256,
        base_data_profile_id=str(data["base_data_profile_id"]),
        base_data_profile_sha256=base_data_profile_sha256,
        dataset_id=str(data["dataset_id"]),
        input_dim=input_dim,
        num_classes=num_classes,
        preprocessing_binding=preprocessing,
        public_protocol=public_protocol,
    )


def parse_owner_random_allocation_contract_v4(
    raw_value: object,
) -> OwnerRandomAllocationContractV4:
    try:
        return _parse_owner_random_allocation_contract_v4_impl(raw_value)
    except AllocationContractV4Error:
        raise
    except ContractV2Error as exc:
        raise AllocationContractV4Error(str(exc)) from exc


def load_owner_random_allocation_contract_v4(
    path: str | Path,
) -> OwnerRandomAllocationContractV4:
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
        raise AllocationContractV4Error(
            f"Could not load strict random-allocation V4 contract: {exc}"
        ) from exc
    return parse_owner_random_allocation_contract_v4(raw)


@dataclass(frozen=True)
class CompiledOwnerRandomAllocationRouteV4:
    contract: OwnerRandomAllocationContractV4
    source_mapping: WindowMapping
    selected_mapping: WindowMapping
    policy: ContributionPolicy
    preprocessor: FixedAffinePreprocessor
    profile: AllocationBenchmarkProfileV4
    registry_entry: RouteRegistryEntryAllocationV4
    accountant_epsilon: float
    accountant_epsilon_remove: float
    accountant_epsilon_add: float
    accountant_optimal_remove_order: int
    oracle_epsilon: float
    oracle_epsilon_remove: float
    oracle_epsilon_add: float
    compiler_source_bundle_sha256: str
    execution_source_bundle_sha256: str
    public_plan_sha256: str
    private_source_mapping_sha256: str
    private_source_mapping_canonical_sha256: str
    private_selected_mapping_sha256: str
    private_preprocessor_state_sha256: str
    private_compilation_binding_sha256: str
    private_execution_binding_sha256: str
    status: str
    notes: tuple[str, ...]

    @property
    def execution_ready(self) -> bool:
        return (
            _executor_ready(self.contract, self.registry_entry)
            and self.contract.implementation_id
            == self.registry_entry.implementation_id
        )

    def _public_plan_without_hash(self) -> dict[str, Any]:
        return {
            "schema_version": (
                "unitdp.compiled_random_allocation_public_plan.v4"
            ),
            "status": self.status,
            "execution_ready": self.execution_ready,
            "route_registry": asdict(self.registry_entry),
            "contract": self.contract.public_payload(),
            "derived_public_quantities": {
                "marginal_participation_rate": (
                    self.contract.marginal_participation_rate
                ),
                "expected_owners_per_step": (
                    self.contract.expected_owners_per_step
                ),
            },
            "compiler_source_bundle_sha256": (
                self.compiler_source_bundle_sha256
            ),
            "execution_source_bundle_sha256": (
                self.execution_source_bundle_sha256
            ),
            "accountant_verification": {
                "rdp_orders": list(self.registry_entry.rdp_orders),
                "library_versions": dict(
                    self.registry_entry.accountant_library_versions
                ),
                "production_epsilon": self.accountant_epsilon,
                "production_epsilon_remove": (
                    self.accountant_epsilon_remove
                ),
                "production_epsilon_add": self.accountant_epsilon_add,
                "production_optimal_remove_order": (
                    self.accountant_optimal_remove_order
                ),
                "high_precision_oracle_epsilon": self.oracle_epsilon,
                "high_precision_oracle_epsilon_remove": (
                    self.oracle_epsilon_remove
                ),
                "high_precision_oracle_epsilon_add": (
                    self.oracle_epsilon_add
                ),
                "maximum_absolute_difference": max(
                    abs(self.accountant_epsilon - self.oracle_epsilon),
                    abs(
                        self.accountant_epsilon_remove
                        - self.oracle_epsilon_remove
                    ),
                    abs(
                        self.accountant_epsilon_add
                        - self.oracle_epsilon_add
                    ),
                ),
                "target_epsilon": self.contract.target_epsilon,
                "actual_noise_multiplier": (
                    self.contract.noise_multiplier
                ),
            },
            "notes": list(self.notes),
        }

    def public_plan(self) -> dict[str, Any]:
        payload = self._public_plan_without_hash()
        if _sha256_payload(payload) != self.public_plan_sha256:
            raise RuntimeError(
                "Compiled random-allocation public-plan hash is inconsistent"
            )
        return {**payload, "public_plan_sha256": self.public_plan_sha256}

    def assert_private_execution_integrity(self) -> None:
        canonical = parse_owner_random_allocation_contract_v4(
            self.contract.public_payload()
        )
        if canonical != self.contract:
            raise AllocationContractV4Error(
                "Compiled contract differs from its canonical validated form"
            )
        if get_allocation_benchmark_profile_v4(
            canonical.benchmark_profile_id
        ) != self.profile:
            raise AllocationContractV4Error(
                "Compiled random-allocation benchmark profile is stale"
            )
        if get_route_registry_entry_allocation_v4(
            canonical.route_id
        ) != self.registry_entry:
            raise AllocationContractV4Error(
                "Compiled random-allocation route registry entry is stale"
            )
        _verify_accountant_library_versions(self.registry_entry)
        _verify_registered_callable(
            self.registry_entry.oracle_implementation_id,
            "oracle",
        )
        if _executor_ready(canonical, self.registry_entry):
            _verify_registered_callable(
                self.registry_entry.implementation_id,
                "executor",
            )
        expected_policy = ContributionPolicy(
            policy_type=canonical.policy_type,
            max_windows_per_owner=canonical.max_windows_per_owner,
            windows_per_owner_per_step=(
                canonical.windows_per_owner_per_step
            ),
            seed=0,
        )
        if self.policy != expected_policy:
            raise AllocationContractV4Error(
                "Compiled contribution policy was mutated"
            )
        source_canonical = self.source_mapping.canonical_hash()
        selected_canonical = self.selected_mapping.canonical_hash()
        if source_canonical != self.private_source_mapping_canonical_sha256:
            raise AllocationContractV4Error(
                "Compiled source mapping was mutated"
            )
        if (
            source_canonical
            != self.profile.base_profile.train_mapping_canonical_sha256
        ):
            raise AllocationContractV4Error(
                "Compiled source mapping no longer matches registered data"
            )
        if selected_canonical != self.private_selected_mapping_sha256:
            raise AllocationContractV4Error(
                "Compiled selected mapping was mutated"
            )
        expected_selected = self.policy.select_mapping(self.source_mapping)
        if expected_selected.canonical_hash() != selected_canonical:
            raise AllocationContractV4Error(
                "Compiled selected mapping no longer matches the policy"
            )
        preprocessor_state = _preprocessor_state_sha256(self.preprocessor)
        if preprocessor_state != self.private_preprocessor_state_sha256:
            raise AllocationContractV4Error(
                "Compiled preprocessor state was mutated"
            )
        try:
            source_bundle = compiler_source_bundle_sha256_allocation_v4()
        except SourceBundleAllocationV4Error as exc:
            raise AllocationContractV4Error(
                f"Registered V4 source bundle is unavailable: {exc}"
            ) from exc
        if source_bundle != self.compiler_source_bundle_sha256:
            raise AllocationContractV4Error(
                "Registered random-allocation compiler source bundle changed"
            )
        try:
            execution_source_bundle = (
                execution_source_bundle_sha256_allocation_v4()
            )
        except SourceBundleAllocationV4Error as exc:
            raise AllocationContractV4Error(
                f"Registered V4 execution source bundle is unavailable: {exc}"
            ) from exc
        if execution_source_bundle != self.execution_source_bundle_sha256:
            raise AllocationContractV4Error(
                "Registered random-allocation execution source bundle changed"
            )
        expected_binding = _sha256_payload(
            {
                "public_contract_sha256": (
                    canonical.public_contract_sha256
                ),
                "implementation_id": self.registry_entry.implementation_id,
                "compiler_source_bundle_sha256": source_bundle,
                "execution_source_bundle_sha256": execution_source_bundle,
                "source_mapping_file_sha256": (
                    self.private_source_mapping_sha256
                ),
                "source_mapping_canonical_sha256": source_canonical,
                "selected_mapping_canonical_sha256": selected_canonical,
                "preprocessor_state_sha256": preprocessor_state,
            }
        )
        if expected_binding != self.private_compilation_binding_sha256:
            raise AllocationContractV4Error(
                "Compiled random-allocation private binding is inconsistent"
            )
        if expected_binding != self.private_execution_binding_sha256:
            raise AllocationContractV4Error(
                "Compiled random-allocation execution binding is inconsistent"
            )
        self.public_plan()


def compile_owner_random_allocation_contract_v4(
    contract: OwnerRandomAllocationContractV4,
    *,
    mapping_path: str | Path,
    preprocessing_artifact_path: str | Path,
    require_executable: bool = False,
) -> CompiledOwnerRandomAllocationRouteV4:
    """Compile one exact owner random-allocation contract."""

    if not isinstance(contract, OwnerRandomAllocationContractV4):
        raise AllocationContractV4Error(
            "compile_owner_random_allocation_contract_v4 requires "
            "OwnerRandomAllocationContractV4"
        )
    contract = parse_owner_random_allocation_contract_v4(
        contract.public_payload()
    )
    registry = get_route_registry_entry_allocation_v4(contract.route_id)
    _verify_accountant_library_versions(registry)
    _verify_registered_callable(
        registry.oracle_implementation_id,
        "oracle",
    )
    if _executor_ready(contract, registry):
        _verify_registered_callable(registry.implementation_id, "executor")
    profile = get_allocation_benchmark_profile_v4(
        contract.benchmark_profile_id
    )
    base = profile.base_profile

    try:
        preprocessor = load_fixed_affine_preprocessor(
            preprocessing_artifact_path,
            expected_payload_sha256=str(
                contract.preprocessing_binding["artifact_sha256"]
            ),
        )
    except PreprocessingArtifactError as exc:
        raise AllocationContractV4Error(
            f"Preprocessing artifact validation failed: {exc}"
        ) from exc
    if preprocessor.contract_binding() != contract.preprocessing_binding:
        raise AllocationContractV4Error(
            "Loaded preprocessor does not match the public contract"
        )
    if preprocessor.input_dim != contract.input_dim:
        raise AllocationContractV4Error(
            "Preprocessor dimension does not match the contract"
        )

    try:
        source_mapping = _load_strict_single_owner_mapping_v2(
            mapping_path,
            profile=base,
        )
    except ContractV2Error as exc:
        raise AllocationContractV4Error(str(exc)) from exc
    source_canonical = source_mapping.canonical_hash()
    if source_canonical != base.train_mapping_canonical_sha256:
        raise AllocationContractV4Error(
            "Source mapping must exactly match the registered full-data mapping"
        )
    if len(source_mapping.by_owner()) != contract.source_dataset_size:
        raise AllocationContractV4Error(
            "Observed owner population does not equal the public fixed N"
        )

    policy = ContributionPolicy(
        policy_type=contract.policy_type,
        max_windows_per_owner=contract.max_windows_per_owner,
        windows_per_owner_per_step=contract.windows_per_owner_per_step,
        seed=0,
    )
    selected_mapping = policy.select_mapping(source_mapping)
    if not selected_mapping.records:
        raise AllocationContractV4Error(
            "Contribution policy selected no windows"
        )
    selected_by_owner = selected_mapping.by_owner()
    if len(selected_by_owner) != contract.source_dataset_size:
        raise AllocationContractV4Error(
            "Contribution policy removed an owner from the fixed population"
        )
    if any(
        len(records) > contract.max_windows_per_owner
        for records in selected_by_owner.values()
    ):
        raise AllocationContractV4Error(
            "Contribution policy exceeded max_windows_per_owner"
        )

    try:
        accountant = account_random_allocation_v4(
            num_steps=contract.num_steps,
            num_selected=contract.num_selected_steps_per_owner,
            num_epochs=contract.num_epochs,
            sigma=contract.noise_multiplier,
            delta=contract.delta,
            orders=registry.rdp_orders,
        )
    except RandomAllocationAccountingV4Error as exc:
        raise AllocationContractV4Error(
            f"Direct random-allocation accountant rejected the contract: {exc}"
        ) from exc
    try:
        oracle = account_random_allocation_oracle_v4(
            num_steps=contract.num_steps,
            num_selected=contract.num_selected_steps_per_owner,
            num_epochs=contract.num_epochs,
            sigma=contract.noise_multiplier,
            delta=contract.delta,
            orders=registry.rdp_orders,
            decimal_digits=80,
        )
    except (ValueError, ArithmeticError) as exc:
        raise AllocationContractV4Error(
            f"High-precision allocation oracle rejected the contract: {exc}"
        ) from exc
    max_difference = max(
        abs(accountant.epsilon - oracle.epsilon),
        abs(accountant.epsilon_remove - oracle.epsilon_remove),
        abs(accountant.epsilon_add - oracle.epsilon_add),
        *(
            abs(left_value - right_value)
            for (left_order, left_value), (right_order, right_value)
            in zip(
                accountant.remove_rdp_by_order,
                oracle.remove_rdp_by_order,
                strict=True,
            )
            if left_order == right_order
        ),
    )
    if max_difference > 2e-12:
        raise AllocationContractV4Error(
            "Direct accountant and high-precision oracle disagree"
        )
    if accountant.epsilon > contract.target_epsilon + 1e-10:
        raise AllocationContractV4Error(
            "Declared random-allocation noise is insufficient for target epsilon"
        )

    try:
        source_bundle_sha256 = (
            compiler_source_bundle_sha256_allocation_v4()
        )
    except SourceBundleAllocationV4Error as exc:
        raise AllocationContractV4Error(
            f"Registered V4 source bundle is unavailable: {exc}"
        ) from exc
    try:
        execution_source_bundle_sha256 = (
            execution_source_bundle_sha256_allocation_v4()
        )
    except SourceBundleAllocationV4Error as exc:
        raise AllocationContractV4Error(
            f"Registered V4 execution source bundle is unavailable: {exc}"
        ) from exc
    source_file_sha256 = source_mapping.file_hash()
    selected_canonical = selected_mapping.canonical_hash()
    preprocessor_state = _preprocessor_state_sha256(preprocessor)
    private_binding = _sha256_payload(
        {
            "public_contract_sha256": contract.public_contract_sha256,
            "implementation_id": registry.implementation_id,
            "compiler_source_bundle_sha256": source_bundle_sha256,
            "execution_source_bundle_sha256": (
                execution_source_bundle_sha256
            ),
            "source_mapping_file_sha256": source_file_sha256,
            "source_mapping_canonical_sha256": source_canonical,
            "selected_mapping_canonical_sha256": selected_canonical,
            "preprocessor_state_sha256": preprocessor_state,
        }
    )
    ready = _executor_ready(contract, registry)
    compiled = CompiledOwnerRandomAllocationRouteV4(
        contract=contract,
        source_mapping=source_mapping,
        selected_mapping=selected_mapping,
        policy=policy,
        preprocessor=preprocessor,
        profile=profile,
        registry_entry=registry,
        accountant_epsilon=accountant.epsilon,
        accountant_epsilon_remove=accountant.epsilon_remove,
        accountant_epsilon_add=accountant.epsilon_add,
        accountant_optimal_remove_order=(
            accountant.optimal_remove_order
        ),
        oracle_epsilon=oracle.epsilon,
        oracle_epsilon_remove=oracle.epsilon_remove,
        oracle_epsilon_add=oracle.epsilon_add,
        compiler_source_bundle_sha256=source_bundle_sha256,
        execution_source_bundle_sha256=execution_source_bundle_sha256,
        public_plan_sha256="",
        private_source_mapping_sha256=source_file_sha256,
        private_source_mapping_canonical_sha256=source_canonical,
        private_selected_mapping_sha256=selected_canonical,
        private_preprocessor_state_sha256=preprocessor_state,
        private_compilation_binding_sha256=private_binding,
        private_execution_binding_sha256=private_binding,
        status=(
            "validated_contract_research_executor_ready"
            if ready
            else "validated_contract_executor_not_ready"
        ),
        notes=(
            "t, k, epochs, and sigma are registry-bound public constants",
            "each owner must receive exactly k distinct steps per epoch",
            "assignments are independent across owners and private",
            "the add/remove guarantee uses the larger direct bound",
            "the full source mapping is bound to the public data profile",
            "fixed-case compilation is evidence, not a completeness proof",
        ),
    )
    compiled = replace(
        compiled,
        public_plan_sha256=_sha256_payload(
            compiled._public_plan_without_hash()
        ),
    )
    compiled.assert_private_execution_integrity()
    if require_executable and not compiled.execution_ready:
        raise AllocationExecutorNotReadyV4Error(
            "The random-allocation contract is valid, but its executor "
            "has not passed the implementation gate"
        )
    return compiled
