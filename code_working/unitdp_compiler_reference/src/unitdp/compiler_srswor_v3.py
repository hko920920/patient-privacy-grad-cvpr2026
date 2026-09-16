"""Strict compiler for the fixed-size owner-SRSWOR V3 route."""

from __future__ import annotations

import importlib
import json
from dataclasses import asdict, dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from unitdp.accountant import epsilon_for_srswor_replace_one_noise
from unitdp.benchmark_registry_srswor_v3 import (
    ACCOUNTANT_ID_SRSWOR_V3,
    CONTRACT_SCHEMA_SRSWOR_V3,
    CORE_ROUTE_ID_SRSWOR_V3,
    EXECUTOR_IMPLEMENTATION_ID_SRSWOR_V3,
    RDP_ORDERS_ID_SRSWOR_V3,
    RDP_ORDERS_SRSWOR_V3,
    SrsworBenchmarkProfileV3,
    get_srswor_benchmark_profile_v3,
)
from unitdp.compiler_v2 import (
    ContractV2Error,
    _UniqueKeySafeLoader,
    _construct_unique_json_object,
    _expect_keys,
    _expect_mapping,
    _finite_float,
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
from unitdp.source_bundle_srswor_v3 import (
    SourceBundleSrsworV3Error,
    execution_source_bundle_sha256_srswor_v3,
)
from unitdp_spec_oracle.srswor_rdp_oracle_v3 import (
    SrsworRdpOracleV3Error,
    fixed_size_srswor_epsilon_v3,
)


class SrsworContractV3Error(ContractV2Error):
    """Raised when a V3 SRSWOR contract violates a registered invariant."""


class SrsworExecutorNotReadyV3Error(RuntimeError):
    """Raised when executable compilation is requested without an executor."""


@dataclass(frozen=True)
class RouteRegistryEntrySrsworV3:
    route_id: str
    contract_status: str
    executor_status: str
    implementation_id: str
    adjacency: str
    sampler: str
    accountant_id: str
    rdp_orders: tuple[int, ...]
    accountant_library_versions: tuple[tuple[str, str], ...]
    required_invariants: tuple[str, ...]


ROUTE_REGISTRY_SRSWOR_V3 = MappingProxyType(
    {
        CORE_ROUTE_ID_SRSWOR_V3: RouteRegistryEntrySrsworV3(
            route_id=CORE_ROUTE_ID_SRSWOR_V3,
            contract_status="strict_contract_validator_implemented",
            executor_status="implemented_research_validated",
            implementation_id=EXECUTOR_IMPLEMENTATION_ID_SRSWOR_V3,
            adjacency="replace_one_owner",
            sampler="uniform_fixed_size_without_replacement_owner",
            accountant_id=ACCOUNTANT_ID_SRSWOR_V3,
            rdp_orders=RDP_ORDERS_SRSWOR_V3,
            accountant_library_versions=(("dp-accounting", "0.6.0"),),
            required_invariants=(
                "public fixed source-dataset size",
                "public fixed positive sample size",
                "uniform independent subset sampling without replacement",
                "public fixed total step count",
                "exact registered full-data mapping",
                "exact registered executor implementation id",
                "one clipped vector per sampled owner",
                "replace-one sensitivity bounded by two clip norms",
                "Gaussian update on every step",
                "fixed update denominator equal to sample size",
                "public-fixed preprocessing artifact",
                "single-attribution owner mapping",
                "library and direct-theorem RDP agreement",
                "random coins are never public",
            ),
        )
    }
)


def get_route_registry_entry_srswor_v3(
    route_id: str,
) -> RouteRegistryEntrySrsworV3:
    try:
        return ROUTE_REGISTRY_SRSWOR_V3[route_id]
    except KeyError as exc:
        raise SrsworContractV3Error(
            f"Unsupported SRSWOR V3 route_id: {route_id!r}"
        ) from exc


def _verify_accountant_library_versions(
    registry: RouteRegistryEntrySrsworV3,
) -> dict[str, str]:
    observed: dict[str, str] = {}
    for distribution, expected in registry.accountant_library_versions:
        try:
            installed = version(distribution)
        except PackageNotFoundError as exc:
            raise SrsworContractV3Error(
                f"Required accountant dependency is missing: {distribution}"
            ) from exc
        if installed != expected:
            raise SrsworContractV3Error(
                f"Accountant dependency {distribution} must be {expected}, "
                f"got {installed}; register a new accountant version before use"
            )
        observed[distribution] = installed
    return observed


def _verify_registered_executor_callable(
    registry: RouteRegistryEntrySrsworV3,
) -> None:
    module_name, separator, function_name = (
        registry.implementation_id.rpartition(".")
    )
    if not separator or not module_name or not function_name:
        raise SrsworContractV3Error(
            "Registered executor implementation id is malformed"
        )
    try:
        module = importlib.import_module(module_name)
        implementation = getattr(module, function_name)
    except (ImportError, AttributeError) as exc:
        raise SrsworContractV3Error(
            "Registered executor implementation is not importable: "
            f"{registry.implementation_id}"
        ) from exc
    if not callable(implementation):
        raise SrsworContractV3Error(
            "Registered executor implementation is not callable"
        )


def _executor_ready(
    contract: "OwnerSrsworContractV3",
    registry: RouteRegistryEntrySrsworV3,
) -> bool:
    return (
        contract.execution_profile == "research_benchmark"
        and contract.rng_backend == "research_default"
        and registry.executor_status
        in {"implemented_research_validated", "implemented_release_validated"}
    )


@dataclass(frozen=True)
class OwnerSrsworContractV3:
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
    sample_size: int
    total_steps: int
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
    def sample_rate(self) -> float:
        return self.sample_size / self.source_dataset_size

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
                "sample_size": self.sample_size,
                "total_steps": self.total_steps,
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


def _parse_owner_srswor_contract_v3_impl(
    raw_value: object,
) -> OwnerSrsworContractV3:
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
    if raw["schema_version"] != CONTRACT_SCHEMA_SRSWOR_V3:
        raise SrsworContractV3Error(
            f"schema_version must be {CONTRACT_SCHEMA_SRSWOR_V3!r}"
        )
    if raw["route_id"] != CORE_ROUTE_ID_SRSWOR_V3:
        raise SrsworContractV3Error(
            f"route_id must be {CORE_ROUTE_ID_SRSWOR_V3!r}"
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
    expected_execution = {
        "profile": "research_benchmark",
        "rng_backend": "research_default",
        "implementation_id": EXECUTOR_IMPLEMENTATION_ID_SRSWOR_V3,
        "random_coins_public": False,
    }
    _require_exact_profile_binding(
        execution,
        expected_execution,
        "execution",
    )
    if execution["random_coins_public"] is not False:
        raise SrsworContractV3Error(
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
        raise SrsworContractV3Error(
            "data.benchmark_profile_id must be a string"
        )
    try:
        profile = get_srswor_benchmark_profile_v3(profile_id)
    except KeyError as exc:
        raise SrsworContractV3Error(str(exc)) from exc
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
        raise SrsworContractV3Error("privacy.delta must be smaller than one")
    source_dataset_size = _positive_int(
        mechanism["source_dataset_size"],
        "mechanism.source_dataset_size",
    )
    sample_size = _positive_int(
        mechanism["sample_size"],
        "mechanism.sample_size",
    )
    if sample_size > source_dataset_size:
        raise SrsworContractV3Error(
            "mechanism.sample_size must not exceed source_dataset_size"
        )
    total_steps = _positive_int(
        mechanism["total_steps"],
        "mechanism.total_steps",
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
    if update_denominator != float(sample_size):
        raise SrsworContractV3Error(
            "mechanism.update_denominator must equal sample_size"
        )
    sensitivity_multiplier = _positive_float(
        accountant["sensitivity_multiplier"],
        "accountant.sensitivity_multiplier",
    )
    if sensitivity_multiplier != 2.0:
        raise SrsworContractV3Error(
            "accountant.sensitivity_multiplier must equal 2.0"
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
        raise SrsworContractV3Error(
            "windows_per_owner_per_step must not exceed max_windows_per_owner"
        )
    learning_rate = _positive_float(
        optimization["learning_rate"],
        "optimization.learning_rate",
    )
    input_dim = _positive_int(data["input_dim"], "data.input_dim")
    num_classes = _positive_int(
        data["num_classes"],
        "data.num_classes",
    )
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
    _sha256_string(
        preprocessing["artifact_sha256"],
        "data.preprocessing.artifact_sha256",
    )
    _sha256_string(
        preprocessing["source_reference_sha256"],
        "data.preprocessing.source_reference_sha256",
    )
    _sha256_string(
        public_protocol["source_reference_sha256"],
        "data.public_protocol.source_reference_sha256",
    )
    _sha256_string(
        public_protocol["full_data_conformance_sha256"],
        "data.public_protocol.full_data_conformance_sha256",
    )
    _verify_registered_data_preparer_callable(
        str(public_protocol["data_preparation_implementation_id"])
    )

    raw_weights = optimization["class_weights"]
    if raw_weights is None:
        class_weights = None
    else:
        if not isinstance(raw_weights, list) or len(raw_weights) != num_classes:
            raise SrsworContractV3Error(
                "optimization.class_weights must be null or one value per class"
            )
        class_weights = tuple(
            _positive_float(
                value,
                f"optimization.class_weights[{index}]",
            )
            for index, value in enumerate(raw_weights)
        )

    return OwnerSrsworContractV3(
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
        sample_size=sample_size,
        total_steps=total_steps,
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


def parse_owner_srswor_contract_v3(
    raw_value: object,
) -> OwnerSrsworContractV3:
    try:
        return _parse_owner_srswor_contract_v3_impl(raw_value)
    except SrsworContractV3Error:
        raise
    except ContractV2Error as exc:
        raise SrsworContractV3Error(str(exc)) from exc


def load_owner_srswor_contract_v3(
    path: str | Path,
) -> OwnerSrsworContractV3:
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
        raise SrsworContractV3Error(
            f"Could not load strict SRSWOR V3 contract: {exc}"
        ) from exc
    return parse_owner_srswor_contract_v3(raw)


@dataclass(frozen=True)
class CompiledOwnerSrsworRouteV3:
    contract: OwnerSrsworContractV3
    source_mapping: WindowMapping
    selected_mapping: WindowMapping
    policy: ContributionPolicy
    preprocessor: FixedAffinePreprocessor
    profile: SrsworBenchmarkProfileV3
    registry_entry: RouteRegistryEntrySrsworV3
    accountant_epsilon_dp_accounting: float
    accountant_epsilon_direct_oracle: float
    accountant_optimal_order_direct_oracle: int
    execution_source_bundle_sha256: str
    public_plan_sha256: str
    private_source_mapping_sha256: str
    private_source_mapping_canonical_sha256: str
    private_selected_mapping_sha256: str
    private_preprocessor_state_sha256: str
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
            "schema_version": "unitdp.compiled_srswor_public_plan.v3",
            "status": self.status,
            "execution_ready": self.execution_ready,
            "route_registry": asdict(self.registry_entry),
            "contract": self.contract.public_payload(),
            "execution_source_bundle_sha256": (
                self.execution_source_bundle_sha256
            ),
            "accountant_verification": {
                "rdp_orders": list(self.registry_entry.rdp_orders),
                "library_versions": dict(
                    self.registry_entry.accountant_library_versions
                ),
                "dp_accounting_rdp_epsilon": (
                    self.accountant_epsilon_dp_accounting
                ),
                "direct_theorem_oracle_epsilon": (
                    self.accountant_epsilon_direct_oracle
                ),
                "direct_theorem_oracle_optimal_order": (
                    self.accountant_optimal_order_direct_oracle
                ),
                "absolute_difference": abs(
                    self.accountant_epsilon_dp_accounting
                    - self.accountant_epsilon_direct_oracle
                ),
                "target_epsilon": self.contract.target_epsilon,
                "actual_noise_multiplier": (
                    self.contract.noise_multiplier
                ),
                "normalized_noise_multiplier": (
                    self.contract.noise_multiplier
                    / self.contract.sensitivity_multiplier
                ),
            },
            "notes": list(self.notes),
        }

    def public_plan(self) -> dict[str, Any]:
        payload = self._public_plan_without_hash()
        if _sha256_payload(payload) != self.public_plan_sha256:
            raise RuntimeError(
                "Compiled SRSWOR public-plan hash is inconsistent"
            )
        return {**payload, "public_plan_sha256": self.public_plan_sha256}

    def assert_private_execution_integrity(self) -> None:
        canonical = parse_owner_srswor_contract_v3(
            self.contract.public_payload()
        )
        if canonical != self.contract:
            raise SrsworContractV3Error(
                "Compiled contract differs from its canonical validated form"
            )
        if get_srswor_benchmark_profile_v3(
            canonical.benchmark_profile_id
        ) != self.profile:
            raise SrsworContractV3Error(
                "Compiled SRSWOR benchmark profile is stale"
            )
        if get_route_registry_entry_srswor_v3(
            canonical.route_id
        ) != self.registry_entry:
            raise SrsworContractV3Error(
                "Compiled SRSWOR route registry entry is stale"
            )
        _verify_accountant_library_versions(self.registry_entry)
        if _executor_ready(canonical, self.registry_entry):
            _verify_registered_executor_callable(self.registry_entry)
        expected_policy = ContributionPolicy(
            policy_type=canonical.policy_type,
            max_windows_per_owner=canonical.max_windows_per_owner,
            windows_per_owner_per_step=(
                canonical.windows_per_owner_per_step
            ),
            seed=0,
        )
        if self.policy != expected_policy:
            raise SrsworContractV3Error(
                "Compiled contribution policy was mutated"
            )
        source_canonical = self.source_mapping.canonical_hash()
        selected_canonical = self.selected_mapping.canonical_hash()
        if source_canonical != self.private_source_mapping_canonical_sha256:
            raise SrsworContractV3Error(
                "Compiled source mapping was mutated"
            )
        if (
            source_canonical
            != self.profile.base_profile.train_mapping_canonical_sha256
        ):
            raise SrsworContractV3Error(
                "Compiled source mapping no longer matches the registered data"
            )
        if selected_canonical != self.private_selected_mapping_sha256:
            raise SrsworContractV3Error(
                "Compiled selected mapping was mutated"
            )
        expected_selected = self.policy.select_mapping(self.source_mapping)
        if expected_selected.canonical_hash() != selected_canonical:
            raise SrsworContractV3Error(
                "Compiled selected mapping no longer matches the policy"
            )
        preprocessor_state = _preprocessor_state_sha256(self.preprocessor)
        if preprocessor_state != self.private_preprocessor_state_sha256:
            raise SrsworContractV3Error(
                "Compiled preprocessor state was mutated"
            )
        try:
            source_bundle = execution_source_bundle_sha256_srswor_v3()
        except SourceBundleSrsworV3Error as exc:
            raise SrsworContractV3Error(
                f"Registered SRSWOR source bundle is unavailable: {exc}"
            ) from exc
        if source_bundle != self.execution_source_bundle_sha256:
            raise SrsworContractV3Error(
                "Registered SRSWOR execution source bundle changed"
            )
        expected_binding = _sha256_payload(
            {
                "public_contract_sha256": (
                    canonical.public_contract_sha256
                ),
                "implementation_id": self.registry_entry.implementation_id,
                "execution_source_bundle_sha256": source_bundle,
                "source_mapping_file_sha256": (
                    self.private_source_mapping_sha256
                ),
                "source_mapping_canonical_sha256": source_canonical,
                "selected_mapping_canonical_sha256": selected_canonical,
                "preprocessor_state_sha256": preprocessor_state,
            }
        )
        if expected_binding != self.private_execution_binding_sha256:
            raise SrsworContractV3Error(
                "Compiled SRSWOR private execution binding is inconsistent"
            )
        self.public_plan()


def compile_owner_srswor_contract_v3(
    contract: OwnerSrsworContractV3,
    *,
    mapping_path: str | Path,
    preprocessing_artifact_path: str | Path,
    require_executable: bool = False,
) -> CompiledOwnerSrsworRouteV3:
    """Compile one exact fixed-size route without data-derived parameters."""

    if not isinstance(contract, OwnerSrsworContractV3):
        raise SrsworContractV3Error(
            "compile_owner_srswor_contract_v3 requires "
            "OwnerSrsworContractV3"
        )
    contract = parse_owner_srswor_contract_v3(contract.public_payload())
    registry = get_route_registry_entry_srswor_v3(contract.route_id)
    _verify_accountant_library_versions(registry)
    if _executor_ready(contract, registry):
        _verify_registered_executor_callable(registry)
    profile = get_srswor_benchmark_profile_v3(
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
        raise SrsworContractV3Error(
            f"Preprocessing artifact validation failed: {exc}"
        ) from exc
    if preprocessor.contract_binding() != contract.preprocessing_binding:
        raise SrsworContractV3Error(
            "Loaded preprocessor does not match the public contract"
        )
    if preprocessor.input_dim != contract.input_dim:
        raise SrsworContractV3Error(
            "Preprocessor dimension does not match the contract"
        )

    try:
        source_mapping = _load_strict_single_owner_mapping_v2(
            mapping_path,
            profile=base,
        )
    except ContractV2Error as exc:
        raise SrsworContractV3Error(str(exc)) from exc
    source_canonical = source_mapping.canonical_hash()
    if source_canonical != base.train_mapping_canonical_sha256:
        raise SrsworContractV3Error(
            "Source mapping must exactly match the registered full-data mapping"
        )
    owners = source_mapping.by_owner()
    if len(owners) != contract.source_dataset_size:
        raise SrsworContractV3Error(
            "Observed owner population does not equal the public fixed N"
        )
    if contract.sample_size > len(owners):
        raise SrsworContractV3Error(
            "Fixed sample size exceeds the compiled owner population"
        )

    policy = ContributionPolicy(
        policy_type=contract.policy_type,
        max_windows_per_owner=contract.max_windows_per_owner,
        windows_per_owner_per_step=contract.windows_per_owner_per_step,
        seed=0,
    )
    selected_mapping = policy.select_mapping(source_mapping)
    if not selected_mapping.records:
        raise SrsworContractV3Error(
            "Contribution policy selected no windows"
        )
    if len(selected_mapping.by_owner()) != contract.source_dataset_size:
        raise SrsworContractV3Error(
            "Contribution policy removed an owner from the fixed population"
        )
    if any(
        len(records) > contract.max_windows_per_owner
        for records in selected_mapping.by_owner().values()
    ):
        raise SrsworContractV3Error(
            "Contribution policy exceeded max_windows_per_owner"
        )

    epsilon_library = epsilon_for_srswor_replace_one_noise(
        noise_multiplier=contract.noise_multiplier,
        source_dataset_size=contract.source_dataset_size,
        sample_size=contract.sample_size,
        steps=contract.total_steps,
        delta=contract.delta,
        alphas=list(registry.rdp_orders),
    )
    try:
        oracle = fixed_size_srswor_epsilon_v3(
            actual_noise_multiplier=contract.noise_multiplier,
            source_dataset_size=contract.source_dataset_size,
            sample_size=contract.sample_size,
            steps=contract.total_steps,
            delta=contract.delta,
            orders=registry.rdp_orders,
            sensitivity_multiplier=contract.sensitivity_multiplier,
        )
    except SrsworRdpOracleV3Error as exc:
        raise SrsworContractV3Error(
            f"Direct SRSWOR theorem oracle rejected the contract: {exc}"
        ) from exc
    if abs(epsilon_library - oracle.epsilon) > 1e-10:
        raise SrsworContractV3Error(
            "SRSWOR library accountant and direct theorem oracle disagree"
        )
    if max(epsilon_library, oracle.epsilon) > (
        contract.target_epsilon + 1e-10
    ):
        raise SrsworContractV3Error(
            "Declared SRSWOR noise is insufficient for the target epsilon"
        )

    try:
        source_bundle_sha256 = (
            execution_source_bundle_sha256_srswor_v3()
        )
    except SourceBundleSrsworV3Error as exc:
        raise SrsworContractV3Error(
            f"Registered SRSWOR source bundle is unavailable: {exc}"
        ) from exc
    source_file_sha256 = source_mapping.file_hash()
    selected_canonical = selected_mapping.canonical_hash()
    preprocessor_state = _preprocessor_state_sha256(preprocessor)
    private_binding = _sha256_payload(
        {
            "public_contract_sha256": contract.public_contract_sha256,
            "implementation_id": registry.implementation_id,
            "execution_source_bundle_sha256": source_bundle_sha256,
            "source_mapping_file_sha256": source_file_sha256,
            "source_mapping_canonical_sha256": source_canonical,
            "selected_mapping_canonical_sha256": selected_canonical,
            "preprocessor_state_sha256": preprocessor_state,
        }
    )
    ready = _executor_ready(contract, registry)
    compiled = CompiledOwnerSrsworRouteV3(
        contract=contract,
        source_mapping=source_mapping,
        selected_mapping=selected_mapping,
        policy=policy,
        preprocessor=preprocessor,
        profile=profile,
        registry_entry=registry,
        accountant_epsilon_dp_accounting=epsilon_library,
        accountant_epsilon_direct_oracle=oracle.epsilon,
        accountant_optimal_order_direct_oracle=oracle.optimal_order,
        execution_source_bundle_sha256=source_bundle_sha256,
        public_plan_sha256="",
        private_source_mapping_sha256=source_file_sha256,
        private_source_mapping_canonical_sha256=source_canonical,
        private_selected_mapping_sha256=selected_canonical,
        private_preprocessor_state_sha256=preprocessor_state,
        private_execution_binding_sha256=private_binding,
        status=(
            "validated_contract_research_executor_ready"
            if ready
            else "validated_contract_executor_not_ready"
        ),
        notes=(
            "N, m, and total_steps are read only from the public contract",
            "the full source mapping is digest-bound to the public data profile",
            "the executor must sample a fresh uniform size-m subset each step",
            "replace-one owner sensitivity is bounded by 2C",
            "the actual Gaussian standard deviation is noise_multiplier times C",
            "the accountant normalizes that noise by the registered 2C sensitivity",
            "dp_accounting and an import-independent direct theorem oracle agree",
            "the fixed denominator equals m and empty samples are impossible",
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
        raise SrsworExecutorNotReadyV3Error(
            "The validated SRSWOR contract has no approved executor"
        )
    return compiled
