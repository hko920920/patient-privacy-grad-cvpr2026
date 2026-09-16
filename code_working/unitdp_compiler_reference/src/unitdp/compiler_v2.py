"""Strict compiler for the single supported owner-Poisson DP route."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from unitdp.accountant import (
    epsilon_for_noise,
    epsilon_for_noise_dp_accounting_add_remove,
)
from unitdp.benchmark_registry_v2 import (
    BenchmarkProfileV2,
    get_benchmark_profile_v2,
)
from unitdp.mapping import WindowMapping, WindowRecord
from unitdp.policies import ContributionPolicy
from unitdp.preprocessing import (
    FixedAffinePreprocessor,
    PreprocessingArtifactError,
    load_fixed_affine_preprocessor,
)
from unitdp.source_bundle_v2 import (
    SourceBundleV2Error,
    execution_source_bundle_sha256_v2,
)


CONTRACT_SCHEMA_V2 = "unitdp.owner_poisson_contract.v2"
CORE_ROUTE_ID_V2 = "owa_owner_poisson_rdp_add_remove_v2"
ACCOUNTANT_ID_V2 = "poisson_gaussian_rdp_add_remove_dual_v1"
RDP_ORDERS_ID_V2 = "unitdp_integer_rdp_orders_v1"
RDP_ORDERS_V2 = (2, 3, 4, 5, 8, 16, 32, 64, 128, 256, 512)
EXECUTOR_IMPLEMENTATION_ID_V2 = (
    "unitdp.owa_dpsgd.train_owner_poisson_fixed_v2"
)


class ContractV2Error(ValueError):
    """Raised when a V2 contract fails a required invariant."""


class ExecutorNotReadyError(RuntimeError):
    """Raised when a validated contract is requested as an executable route."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_yaml_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ContractV2Error("YAML mapping keys must be hashable") from exc
        if duplicate:
            raise ContractV2Error(f"Duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_yaml_mapping,
)


def _construct_unique_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ContractV2Error(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class RouteRegistryEntryV2:
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


ROUTE_REGISTRY_V2 = MappingProxyType(
    {
        CORE_ROUTE_ID_V2: RouteRegistryEntryV2(
            route_id=CORE_ROUTE_ID_V2,
            contract_status="strict_contract_validator_implemented",
            executor_status="implemented_research_validated",
            implementation_id=EXECUTOR_IMPLEMENTATION_ID_V2,
            adjacency="add_remove_one_owner",
            sampler="independent_bernoulli_owner",
            accountant_id=ACCOUNTANT_ID_V2,
            rdp_orders=RDP_ORDERS_V2,
            accountant_library_versions=(
                ("dp-accounting", "0.6.0"),
                ("opacus", "1.6.0"),
            ),
            required_invariants=(
                "explicit data-independent owner sample rate",
                "explicit data-independent total step count",
                "exact registered benchmark profile",
                "exact registered executor implementation id",
                "one clipped vector per sampled owner",
                "Gaussian update on every step including empty samples",
                "fixed update denominator",
                "public-fixed preprocessing artifact",
                "single-attribution owner mapping",
                "dual RDP accountant agreement",
                "random coins are never public",
            ),
        )
    }
)


def get_route_registry_entry_v2(route_id: str) -> RouteRegistryEntryV2:
    try:
        return ROUTE_REGISTRY_V2[route_id]
    except KeyError as exc:
        raise ContractV2Error(f"Unsupported V2 route_id: {route_id!r}") from exc


def _verify_accountant_library_versions(
    registry: RouteRegistryEntryV2,
) -> dict[str, str]:
    observed: dict[str, str] = {}
    for distribution, expected in registry.accountant_library_versions:
        try:
            installed = version(distribution)
        except PackageNotFoundError as exc:
            raise ContractV2Error(
                f"Required accountant dependency is missing: {distribution}"
            ) from exc
        if installed != expected:
            raise ContractV2Error(
                f"Accountant dependency {distribution} must be {expected}, "
                f"got {installed}; register a new accountant version before use"
            )
        observed[distribution] = installed
    return observed


def _executor_ready_for_contract(
    contract: "OwnerPoissonContractV2",
    registry: RouteRegistryEntryV2,
) -> bool:
    if (
        contract.execution_profile == "research_benchmark"
        and contract.rng_backend == "research_default"
    ):
        return registry.executor_status in {
            "implemented_research_validated",
            "implemented_release_validated",
        }
    if (
        contract.execution_profile == "privacy_release"
        and contract.rng_backend == "system_csprng"
    ):
        return registry.executor_status == "implemented_release_validated"
    return False


def _verify_registered_executor_callable(
    registry: RouteRegistryEntryV2,
) -> None:
    module_name, separator, function_name = (
        registry.implementation_id.rpartition(".")
    )
    if not separator or not module_name or not function_name:
        raise ContractV2Error(
            "Registered executor implementation id is malformed"
        )
    try:
        module = importlib.import_module(module_name)
        implementation = getattr(module, function_name)
    except (ImportError, AttributeError) as exc:
        raise ContractV2Error(
            "Registered executor implementation is not importable: "
            f"{registry.implementation_id}"
        ) from exc
    if not callable(implementation):
        raise ContractV2Error(
            "Registered executor implementation is not callable"
        )


def _verify_registered_data_preparer_callable(
    implementation_id: str,
) -> None:
    module_name, separator, function_name = implementation_id.rpartition(".")
    if not separator or not module_name or not function_name:
        raise ContractV2Error(
            "Registered data-preparation implementation id is malformed"
        )
    try:
        module = importlib.import_module(module_name)
        implementation = getattr(module, function_name)
    except (ImportError, AttributeError) as exc:
        raise ContractV2Error(
            "Registered data-preparation implementation is not importable: "
            f"{implementation_id}"
        ) from exc
    if not callable(implementation):
        raise ContractV2Error(
            "Registered data-preparation implementation is not callable"
        )


def _sha256_payload(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _preprocessor_state_sha256(
    preprocessor: FixedAffinePreprocessor,
) -> str:
    return _sha256_payload(
        {
            "binding": preprocessor.contract_binding(),
            "mean": preprocessor.mean.tolist(),
            "scale": preprocessor.scale.tolist(),
        }
    )


def _expect_mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractV2Error(f"{path} must be a mapping")
    return dict(value)


def _expect_keys(
    value: dict[str, Any],
    path: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = sorted(required.difference(value))
    unknown = sorted(set(value).difference(required | optional))
    if missing:
        raise ContractV2Error(f"{path} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ContractV2Error(f"{path} has unknown fields: {', '.join(unknown)}")


def _finite_float(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractV2Error(f"{path} must be a JSON/YAML number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractV2Error(f"{path} must be finite")
    return result


def _positive_float(value: object, path: str) -> float:
    result = _finite_float(value, path)
    if result <= 0:
        raise ContractV2Error(f"{path} must be positive")
    return result


def _positive_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractV2Error(f"{path} must be an integer")
    if value <= 0:
        raise ContractV2Error(f"{path} must be positive")
    return int(value)


def _sha256_string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ContractV2Error(f"{path} must be a string")
    text = value
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ContractV2Error(f"{path} must be a lowercase SHA-256")
    return text


def _require_exact_profile_binding(
    actual: object,
    expected: object,
    path: str = "benchmark_profile",
) -> None:
    """Report the first field that differs from an immutable profile."""

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ContractV2Error(f"{path} must be a mapping")
        expected_keys = set(expected)
        actual_keys = set(actual)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            detail = []
            if missing:
                detail.append(f"missing={missing}")
            if extra:
                detail.append(f"extra={extra}")
            raise ContractV2Error(
                f"{path} differs from the registered profile ({'; '.join(detail)})"
            )
        for key in sorted(expected):
            _require_exact_profile_binding(
                actual[key],
                expected[key],
                f"{path}.{key}",
            )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ContractV2Error(
                f"{path} differs from the registered profile"
            )
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected)
        ):
            _require_exact_profile_binding(
                actual_item,
                expected_item,
                f"{path}[{index}]",
            )
        return
    if actual != expected:
        raise ContractV2Error(
            f"{path} must equal the registered value {expected!r}, got {actual!r}"
        )


@dataclass(frozen=True)
class OwnerPoissonContractV2:
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
    owner_sample_rate: float
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
    dataset_id: str
    input_dim: int
    num_classes: int
    preprocessing_binding: dict[str, Any]
    public_protocol: dict[str, Any]

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
                "owner_sample_rate": self.owner_sample_rate,
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
            },
            "contribution_policy": {
                "type": self.policy_type,
                "max_windows_per_owner": self.max_windows_per_owner,
                "windows_per_owner_per_step": self.windows_per_owner_per_step,
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
                "dataset_id": self.dataset_id,
                "input_dim": self.input_dim,
                "num_classes": self.num_classes,
                "preprocessing": self.preprocessing_binding,
                "public_protocol": self.public_protocol,
            },
        }

    @property
    def public_contract_sha256(self) -> str:
        return _sha256_payload(self.public_payload())


def parse_owner_poisson_contract_v2(raw_value: object) -> OwnerPoissonContractV2:
    """Parse a strict V2 contract and reject unknown or missing fields."""

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
    if raw["schema_version"] != CONTRACT_SCHEMA_V2:
        raise ContractV2Error(
            f"schema_version must be {CONTRACT_SCHEMA_V2!r}"
        )
    route_id = str(raw["route_id"])
    registry = get_route_registry_entry_v2(route_id)

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
    profile = str(execution["profile"])
    if profile not in {"research_benchmark", "privacy_release"}:
        raise ContractV2Error(
            "execution.profile must be research_benchmark or privacy_release"
        )
    rng_backend = str(execution["rng_backend"])
    if rng_backend not in {"research_default", "system_csprng"}:
        raise ContractV2Error("Unsupported execution.rng_backend")
    implementation_id = str(execution["implementation_id"])
    if implementation_id != registry.implementation_id:
        raise ContractV2Error(
            "execution.implementation_id does not match the registered route"
        )
    if execution["random_coins_public"] is not False:
        raise ContractV2Error("execution.random_coins_public must be false")
    if profile == "privacy_release" and rng_backend != "system_csprng":
        raise ContractV2Error(
            "privacy_release requires execution.rng_backend=system_csprng"
        )

    privacy = _expect_mapping(raw["privacy"], "privacy")
    _expect_keys(
        privacy,
        "privacy",
        {"unit", "adjacency", "target_epsilon", "delta"},
    )
    privacy_unit = str(privacy["unit"])
    adjacency = str(privacy["adjacency"])
    if privacy_unit != "owner":
        raise ContractV2Error("privacy.unit must be owner")
    if adjacency != registry.adjacency:
        raise ContractV2Error(
            f"privacy.adjacency must be {registry.adjacency}"
        )
    target_epsilon = _positive_float(
        privacy["target_epsilon"], "privacy.target_epsilon"
    )
    delta = _positive_float(privacy["delta"], "privacy.delta")
    if delta >= 1:
        raise ContractV2Error("privacy.delta must be less than one")

    mechanism = _expect_mapping(raw["mechanism"], "mechanism")
    _expect_keys(
        mechanism,
        "mechanism",
        {
            "sampler",
            "sampling_unit",
            "accounting_unit",
            "parameter_source",
            "owner_sample_rate",
            "total_steps",
            "clipping_unit",
            "noising_unit",
            "clip_norm",
            "noise_multiplier",
            "update_denominator",
            "empty_step_behavior",
            "per_owner_contribution",
            "owner_aggregation",
        },
    )
    sampler = str(mechanism["sampler"])
    if sampler != registry.sampler:
        raise ContractV2Error(f"mechanism.sampler must be {registry.sampler}")
    sampling_unit = str(mechanism["sampling_unit"])
    if sampling_unit != "owner":
        raise ContractV2Error("mechanism.sampling_unit must be owner")
    accounting_unit = str(mechanism["accounting_unit"])
    if accounting_unit != "owner":
        raise ContractV2Error("mechanism.accounting_unit must be owner")
    parameter_source = str(mechanism["parameter_source"])
    if parameter_source != "registered_benchmark_profile":
        raise ContractV2Error(
            "mechanism.parameter_source must be registered_benchmark_profile"
        )
    owner_sample_rate = _positive_float(
        mechanism["owner_sample_rate"], "mechanism.owner_sample_rate"
    )
    if owner_sample_rate > 1:
        raise ContractV2Error("mechanism.owner_sample_rate must be at most one")
    total_steps = _positive_int(
        mechanism["total_steps"], "mechanism.total_steps"
    )
    clipping_unit = str(mechanism["clipping_unit"])
    noising_unit = str(mechanism["noising_unit"])
    if clipping_unit != "owner":
        raise ContractV2Error("mechanism.clipping_unit must be owner")
    if noising_unit != "owner_sum":
        raise ContractV2Error("mechanism.noising_unit must be owner_sum")
    clip_norm = _positive_float(
        mechanism["clip_norm"], "mechanism.clip_norm"
    )
    noise_multiplier = _positive_float(
        mechanism["noise_multiplier"], "mechanism.noise_multiplier"
    )
    update_denominator = _positive_float(
        mechanism["update_denominator"], "mechanism.update_denominator"
    )
    empty_step_behavior = str(mechanism["empty_step_behavior"])
    if empty_step_behavior != "apply_gaussian_update":
        raise ContractV2Error(
            "mechanism.empty_step_behavior must be apply_gaussian_update"
        )
    per_owner_contribution = str(mechanism["per_owner_contribution"])
    if per_owner_contribution != "one_clipped_vector":
        raise ContractV2Error(
            "mechanism.per_owner_contribution must be one_clipped_vector"
        )
    owner_aggregation = str(mechanism["owner_aggregation"])
    if owner_aggregation != "mean":
        raise ContractV2Error("mechanism.owner_aggregation must be mean")

    accountant = _expect_mapping(raw["accountant"], "accountant")
    _expect_keys(accountant, "accountant", {"id", "rdp_orders_id"})
    accountant_id = str(accountant["id"])
    orders_id = str(accountant["rdp_orders_id"])
    if accountant_id != registry.accountant_id:
        raise ContractV2Error(
            f"accountant.id must be {registry.accountant_id}"
        )
    if orders_id != RDP_ORDERS_ID_V2:
        raise ContractV2Error(
            f"accountant.rdp_orders_id must be {RDP_ORDERS_ID_V2}"
        )

    policy = _expect_mapping(
        raw["contribution_policy"], "contribution_policy"
    )
    _expect_keys(
        policy,
        "contribution_policy",
        {
            "type",
            "max_windows_per_owner",
            "windows_per_owner_per_step",
            "scope",
        },
    )
    policy_type = str(policy["type"])
    if policy_type != "support_balanced_cap":
        raise ContractV2Error(
            "contribution_policy.type must be support_balanced_cap"
        )
    max_windows = _positive_int(
        policy["max_windows_per_owner"],
        "contribution_policy.max_windows_per_owner",
    )
    per_step_windows = _positive_int(
        policy["windows_per_owner_per_step"],
        "contribution_policy.windows_per_owner_per_step",
    )
    if per_step_windows > max_windows:
        raise ContractV2Error(
            "windows_per_owner_per_step cannot exceed max_windows_per_owner"
        )
    policy_scope = str(policy["scope"])
    if policy_scope != "owner_local":
        raise ContractV2Error("contribution_policy.scope must be owner_local")

    optimization = _expect_mapping(raw["optimization"], "optimization")
    _expect_keys(
        optimization,
        "optimization",
        {
            "optimizer",
            "learning_rate",
            "loss",
            "model_id",
            "class_weights",
        },
    )
    optimizer = str(optimization["optimizer"])
    loss = str(optimization["loss"])
    model_id = str(optimization["model_id"])
    if optimizer != "sgd":
        raise ContractV2Error("optimization.optimizer must be sgd")
    if loss != "cross_entropy":
        raise ContractV2Error("optimization.loss must be cross_entropy")
    if not model_id:
        raise ContractV2Error("optimization.model_id must be non-empty")
    learning_rate = _positive_float(
        optimization["learning_rate"], "optimization.learning_rate"
    )

    data = _expect_mapping(raw["data"], "data")
    _expect_keys(
        data,
        "data",
        {
            "benchmark_profile_id",
            "benchmark_profile_sha256",
            "dataset_id",
            "input_dim",
            "num_classes",
            "preprocessing",
            "public_protocol",
        },
    )
    benchmark_profile_id = str(data["benchmark_profile_id"])
    try:
        benchmark_profile: BenchmarkProfileV2 = get_benchmark_profile_v2(
            benchmark_profile_id
        )
    except KeyError as exc:
        raise ContractV2Error(str(exc)) from exc
    benchmark_profile_sha256 = _sha256_string(
        data["benchmark_profile_sha256"],
        "data.benchmark_profile_sha256",
    )
    if benchmark_profile_sha256 != benchmark_profile.registry_sha256:
        raise ContractV2Error(
            "data.benchmark_profile_sha256 does not match the registered profile"
        )
    dataset_id = str(data["dataset_id"])
    if not dataset_id:
        raise ContractV2Error("data.dataset_id must be non-empty")
    input_dim = _positive_int(data["input_dim"], "data.input_dim")
    num_classes = _positive_int(data["num_classes"], "data.num_classes")
    if num_classes < 2:
        raise ContractV2Error("data.num_classes must be at least two")

    class_weights_raw = optimization["class_weights"]
    class_weights: tuple[float, ...] | None
    if class_weights_raw is None:
        class_weights = None
    else:
        if not isinstance(class_weights_raw, list):
            raise ContractV2Error(
                "optimization.class_weights must be null or a list"
            )
        if len(class_weights_raw) != num_classes:
            raise ContractV2Error(
                "optimization.class_weights length must equal data.num_classes"
            )
        class_weights = tuple(
            _positive_float(value, f"optimization.class_weights[{index}]")
            for index, value in enumerate(class_weights_raw)
        )

    preprocessing = _expect_mapping(data["preprocessing"], "data.preprocessing")
    _expect_keys(
        preprocessing,
        "data.preprocessing",
        {
            "scope",
            "schema",
            "artifact_id",
            "artifact_sha256",
            "source_reference_sha256",
            "input_dim",
            "transform",
            "fit_on_protected_data",
        },
    )
    if preprocessing["scope"] != "public_fixed":
        raise ContractV2Error("data.preprocessing.scope must be public_fixed")
    if preprocessing["schema"] != "unitdp.fixed_affine_preprocessor.v1":
        raise ContractV2Error("Unsupported data.preprocessing.schema")
    if not str(preprocessing["artifact_id"]):
        raise ContractV2Error("data.preprocessing.artifact_id is required")
    _sha256_string(
        preprocessing["artifact_sha256"],
        "data.preprocessing.artifact_sha256",
    )
    _sha256_string(
        preprocessing["source_reference_sha256"],
        "data.preprocessing.source_reference_sha256",
    )
    if _positive_int(
        preprocessing["input_dim"], "data.preprocessing.input_dim"
    ) != input_dim:
        raise ContractV2Error(
            "data.preprocessing.input_dim must equal data.input_dim"
        )
    if preprocessing["transform"] != "standard_scaler":
        raise ContractV2Error(
            "data.preprocessing.transform must be standard_scaler"
        )
    if preprocessing["fit_on_protected_data"] is not False:
        raise ContractV2Error(
            "data.preprocessing.fit_on_protected_data must be false"
        )

    public_protocol = _expect_mapping(
        data["public_protocol"], "data.public_protocol"
    )
    _expect_keys(
        public_protocol,
        "data.public_protocol",
        {
            "protocol_id",
            "data_preparation_implementation_id",
            "source_reference_sha256",
            "train_owners",
            "test_owners",
            "evaluation_scope",
            "full_data_conformance_sha256",
        },
    )
    if not str(public_protocol["protocol_id"]):
        raise ContractV2Error("data.public_protocol.protocol_id is required")
    data_preparation_implementation_id = str(
        public_protocol["data_preparation_implementation_id"]
    )
    _verify_registered_data_preparer_callable(
        data_preparation_implementation_id
    )
    protocol_source_sha = _sha256_string(
        public_protocol["source_reference_sha256"],
        "data.public_protocol.source_reference_sha256",
    )
    if protocol_source_sha != str(preprocessing["source_reference_sha256"]):
        raise ContractV2Error(
            "public protocol and preprocessing source digests must match"
        )
    train_owners = _positive_int(
        public_protocol["train_owners"],
        "data.public_protocol.train_owners",
    )
    _positive_int(
        public_protocol["test_owners"],
        "data.public_protocol.test_owners",
    )
    if public_protocol["evaluation_scope"] != "public_benchmark":
        raise ContractV2Error(
            "data.public_protocol.evaluation_scope must be public_benchmark"
        )
    _sha256_string(
        public_protocol["full_data_conformance_sha256"],
        "data.public_protocol.full_data_conformance_sha256",
    )
    if delta >= 1.0 / train_owners:
        raise ContractV2Error(
            "privacy.delta must be smaller than 1 / public train owners"
        )

    actual_profile_binding = {
        "privacy": {
            "target_epsilon": target_epsilon,
            "delta": delta,
        },
        "mechanism": {
            "owner_sample_rate": owner_sample_rate,
            "total_steps": total_steps,
            "clip_norm": clip_norm,
            "noise_multiplier": noise_multiplier,
            "update_denominator": update_denominator,
        },
        "contribution_policy": {
            "max_windows_per_owner": max_windows,
            "windows_per_owner_per_step": per_step_windows,
        },
        "optimization": {
            "learning_rate": learning_rate,
            "model_id": model_id,
            "class_weights": (
                list(class_weights) if class_weights is not None else None
            ),
        },
        "data": {
            "benchmark_profile_id": benchmark_profile_id,
            "benchmark_profile_sha256": benchmark_profile_sha256,
            "dataset_id": dataset_id,
            "input_dim": input_dim,
            "num_classes": num_classes,
            "preprocessing": preprocessing,
            "public_protocol": public_protocol,
        },
    }
    _require_exact_profile_binding(
        actual_profile_binding,
        benchmark_profile.expected_contract_binding(),
    )

    return OwnerPoissonContractV2(
        schema_version=CONTRACT_SCHEMA_V2,
        route_id=route_id,
        execution_profile=profile,
        rng_backend=rng_backend,
        implementation_id=implementation_id,
        privacy_unit=privacy_unit,
        adjacency=adjacency,
        target_epsilon=target_epsilon,
        delta=delta,
        sampler=sampler,
        sampling_unit=sampling_unit,
        accounting_unit=accounting_unit,
        parameter_source=parameter_source,
        owner_sample_rate=owner_sample_rate,
        total_steps=total_steps,
        clipping_unit=clipping_unit,
        noising_unit=noising_unit,
        clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
        update_denominator=update_denominator,
        empty_step_behavior=empty_step_behavior,
        per_owner_contribution=per_owner_contribution,
        owner_aggregation=owner_aggregation,
        accountant_id=accountant_id,
        rdp_orders_id=orders_id,
        policy_type=policy_type,
        max_windows_per_owner=max_windows,
        windows_per_owner_per_step=per_step_windows,
        policy_scope=policy_scope,
        optimizer=optimizer,
        learning_rate=learning_rate,
        loss=loss,
        model_id=model_id,
        class_weights=class_weights,
        benchmark_profile_id=benchmark_profile_id,
        benchmark_profile_sha256=benchmark_profile_sha256,
        dataset_id=dataset_id,
        input_dim=input_dim,
        num_classes=num_classes,
        preprocessing_binding=preprocessing,
        public_protocol=public_protocol,
    )


def load_owner_poisson_contract_v2(
    path: str | Path,
) -> OwnerPoissonContractV2:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        raw = json.loads(
            text,
            object_pairs_hook=_construct_unique_json_object,
        )
    else:
        raw = yaml.load(text, Loader=_UniqueKeySafeLoader)
    return parse_owner_poisson_contract_v2(raw)


_NONNEGATIVE_INTEGER_TOKEN = re.compile(r"(?:0|[1-9][0-9]*)\Z")


def _parse_mapping_integer(value: object, path: str) -> int:
    if not isinstance(value, str) or not _NONNEGATIVE_INTEGER_TOKEN.fullmatch(
        value
    ):
        raise ContractV2Error(
            f"{path} must be a canonical nonnegative integer"
        )
    return int(value)


def _load_strict_single_owner_mapping_v2(
    path: str | Path,
    *,
    profile: BenchmarkProfileV2,
) -> WindowMapping:
    """Load the registered mapping schema without permissive coercions."""

    source = Path(path)
    records: list[WindowRecord] = []
    try:
        with source.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ContractV2Error("Mapping CSV has no header")
            observed_columns = tuple(reader.fieldnames)
            if observed_columns != profile.mapping_columns:
                raise ContractV2Error(
                    "Mapping CSV columns do not exactly match the registered "
                    f"profile: expected {profile.mapping_columns!r}, "
                    f"got {observed_columns!r}"
                )
            if len(observed_columns) != len(set(observed_columns)):
                raise ContractV2Error(
                    "Mapping CSV contains duplicate column names"
                )

            for position, row in enumerate(reader):
                row_path = f"mapping row {position + 2}"
                if None in row or any(value is None for value in row.values()):
                    raise ContractV2Error(
                        f"{row_path} has too many or too few columns"
                    )
                values = {key: str(value) for key, value in row.items()}
                if any(
                    value != value.strip()
                    for value in values.values()
                ):
                    raise ContractV2Error(
                        f"{row_path} contains surrounding whitespace"
                    )
                window_id = values["window_id"]
                owner_id = values["owner_id"]
                owner_ids = values["owner_ids"]
                scenario = values["scenario"]
                label = values["label"]
                if not window_id:
                    raise ContractV2Error(f"{row_path}.window_id is empty")
                if not owner_id or not owner_ids:
                    raise ContractV2Error(
                        f"{row_path} has empty owner attribution"
                    )
                if owner_ids != owner_id or ";" in owner_ids:
                    raise ContractV2Error(
                        f"{row_path} must attribute exactly one matching owner"
                    )
                if not scenario:
                    raise ContractV2Error(f"{row_path}.scenario is empty")
                if not label:
                    raise ContractV2Error(f"{row_path}.label is empty")
                for column in profile.mapping_columns:
                    if column not in {
                        "scenario",
                        "window_id",
                        "owner_id",
                        "owner_ids",
                        "start",
                        "end",
                        "row_index",
                        "label",
                    } and not values[column]:
                        raise ContractV2Error(
                            f"{row_path}.{column} is empty"
                        )

                start = _parse_mapping_integer(
                    values["start"], f"{row_path}.start"
                )
                end = _parse_mapping_integer(
                    values["end"], f"{row_path}.end"
                )
                row_index = _parse_mapping_integer(
                    values["row_index"], f"{row_path}.row_index"
                )
                if end <= start:
                    raise ContractV2Error(
                        f"{row_path} requires end > start"
                    )
                records.append(
                    WindowRecord(
                        window_id=window_id,
                        owner_ids=(owner_id,),
                        start=start,
                        end=end,
                        row_index=row_index,
                        label=label,
                        scenario=scenario,
                    )
                )
    except (OSError, csv.Error) as exc:
        raise ContractV2Error(
            f"Could not load strict mapping CSV {source}: {exc}"
        ) from exc
    if not records:
        raise ContractV2Error("Mapping CSV contains no records")
    return WindowMapping(records, source)


@dataclass(frozen=True)
class CompiledOwnerPoissonRouteV2:
    contract: OwnerPoissonContractV2
    source_mapping: WindowMapping
    selected_mapping: WindowMapping
    policy: ContributionPolicy
    preprocessor: FixedAffinePreprocessor
    registry_entry: RouteRegistryEntryV2
    accountant_epsilon_opacus: float
    accountant_epsilon_dp_accounting: float
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
            _executor_ready_for_contract(
                self.contract,
                self.registry_entry,
            )
            and self.contract.implementation_id
            == self.registry_entry.implementation_id
        )

    def _public_plan_without_hash(self) -> dict[str, Any]:
        return {
            "schema_version": "unitdp.compiled_public_plan.v2",
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
                "opacus_rdp_epsilon": self.accountant_epsilon_opacus,
                "dp_accounting_rdp_epsilon": (
                    self.accountant_epsilon_dp_accounting
                ),
                "absolute_difference": abs(
                    self.accountant_epsilon_opacus
                    - self.accountant_epsilon_dp_accounting
                ),
                "target_epsilon": self.contract.target_epsilon,
            },
            "notes": list(self.notes),
        }

    def public_plan(self) -> dict[str, Any]:
        payload = self._public_plan_without_hash()
        observed_hash = _sha256_payload(payload)
        if observed_hash != self.public_plan_sha256:
            raise RuntimeError("Compiled public-plan hash is internally inconsistent")
        return {
            **payload,
            "public_plan_sha256": self.public_plan_sha256,
        }

    def assert_private_execution_integrity(self) -> None:
        """Reject mutation of a compiled object before an executor can use it."""

        canonical_contract = parse_owner_poisson_contract_v2(
            self.contract.public_payload()
        )
        if canonical_contract != self.contract:
            raise ContractV2Error(
                "Compiled contract differs from its canonical validated form"
            )
        if get_route_registry_entry_v2(self.contract.route_id) != (
            self.registry_entry
        ):
            raise ContractV2Error("Compiled route registry entry is stale")
        _verify_accountant_library_versions(self.registry_entry)
        if _executor_ready_for_contract(
            self.contract,
            self.registry_entry,
        ):
            _verify_registered_executor_callable(self.registry_entry)
        expected_policy = ContributionPolicy(
            policy_type=self.contract.policy_type,
            max_windows_per_owner=self.contract.max_windows_per_owner,
            windows_per_owner_per_step=(
                self.contract.windows_per_owner_per_step
            ),
            seed=0,
        )
        if self.policy != expected_policy:
            raise ContractV2Error(
                "Compiled contribution policy was mutated"
            )
        source_canonical = self.source_mapping.canonical_hash()
        selected_canonical = self.selected_mapping.canonical_hash()
        if source_canonical != self.private_source_mapping_canonical_sha256:
            raise ContractV2Error("Compiled source mapping was mutated")
        if selected_canonical != self.private_selected_mapping_sha256:
            raise ContractV2Error("Compiled selected mapping was mutated")
        expected_selected = self.policy.select_mapping(self.source_mapping)
        if expected_selected.canonical_hash() != selected_canonical:
            raise ContractV2Error(
                "Compiled selected mapping no longer matches the policy"
            )
        preprocessor_state = _preprocessor_state_sha256(self.preprocessor)
        if preprocessor_state != self.private_preprocessor_state_sha256:
            raise ContractV2Error("Compiled preprocessor state was mutated")
        try:
            source_bundle_sha256 = execution_source_bundle_sha256_v2()
        except SourceBundleV2Error as exc:
            raise ContractV2Error(
                f"Registered execution source bundle is unavailable: {exc}"
            ) from exc
        if source_bundle_sha256 != self.execution_source_bundle_sha256:
            raise ContractV2Error(
                "Registered execution source bundle changed after compile"
            )
        expected_binding = _sha256_payload(
            {
                "public_contract_sha256": (
                    canonical_contract.public_contract_sha256
                ),
                "implementation_id": self.registry_entry.implementation_id,
                "execution_source_bundle_sha256": source_bundle_sha256,
                "source_mapping_file_sha256": (
                    self.private_source_mapping_sha256
                ),
                "source_mapping_canonical_sha256": source_canonical,
                "selected_mapping_canonical_sha256": selected_canonical,
                "preprocessor_state_sha256": preprocessor_state,
            }
        )
        if expected_binding != self.private_execution_binding_sha256:
            raise ContractV2Error(
                "Compiled private execution binding is inconsistent"
            )
        self.public_plan()


def compile_owner_poisson_contract_v2(
    contract: OwnerPoissonContractV2,
    *,
    mapping_path: str | Path,
    preprocessing_artifact_path: str | Path,
    require_executable: bool = False,
) -> CompiledOwnerPoissonRouteV2:
    """Validate a V2 plan without deriving any schedule value from the data."""

    if not isinstance(contract, OwnerPoissonContractV2):
        raise ContractV2Error(
            "compile_owner_poisson_contract_v2 requires an "
            "OwnerPoissonContractV2"
        )
    # Never trust that a dataclass instance still reflects a previously parsed
    # contract: nested dictionaries are mutable and callers can construct or
    # replace dataclass fields directly.  Canonicalize and revalidate at the
    # compiler boundary.
    contract = parse_owner_poisson_contract_v2(contract.public_payload())
    registry = get_route_registry_entry_v2(contract.route_id)
    _verify_accountant_library_versions(registry)
    if _executor_ready_for_contract(contract, registry):
        _verify_registered_executor_callable(registry)
    try:
        profile = get_benchmark_profile_v2(contract.benchmark_profile_id)
    except KeyError as exc:
        raise ContractV2Error(str(exc)) from exc
    try:
        preprocessor = load_fixed_affine_preprocessor(
            preprocessing_artifact_path,
            expected_payload_sha256=str(
                contract.preprocessing_binding["artifact_sha256"]
            ),
        )
    except PreprocessingArtifactError as exc:
        raise ContractV2Error(
            f"Preprocessing artifact validation failed: {exc}"
        ) from exc
    if preprocessor.contract_binding() != contract.preprocessing_binding:
        raise ContractV2Error(
            "Loaded preprocessing artifact does not exactly match the contract binding"
        )
    if preprocessor.input_dim != contract.input_dim:
        raise ContractV2Error(
            "Preprocessor input dimension does not match the contract"
        )

    source_mapping = _load_strict_single_owner_mapping_v2(
        mapping_path,
        profile=profile,
    )
    window_ids = [record.window_id for record in source_mapping.records]
    row_indices = [record.row_index for record in source_mapping.records]
    if len(window_ids) != len(set(window_ids)):
        raise ContractV2Error("Source mapping contains duplicate window_id values")
    if any(index < 0 for index in row_indices):
        raise ContractV2Error("Source mapping contains a negative row_index")
    if len(row_indices) != len(set(row_indices)):
        raise ContractV2Error("Source mapping contains duplicate row_index values")
    if sorted(row_indices) != list(range(len(row_indices))):
        raise ContractV2Error(
            "Source mapping row_index values must be contiguous from zero"
        )
    if any(len(record.owner_ids) != 1 for record in source_mapping.records):
        raise ContractV2Error(
            "V2 owner route requires exactly one owner attribution per window"
        )

    policy = ContributionPolicy(
        policy_type=contract.policy_type,
        max_windows_per_owner=contract.max_windows_per_owner,
        windows_per_owner_per_step=contract.windows_per_owner_per_step,
        seed=0,
    )
    selected_mapping = policy.select_mapping(source_mapping)
    if not selected_mapping.records:
        raise ContractV2Error("Contribution policy selected no windows")
    if any(len(record.owner_ids) != 1 for record in selected_mapping.records):
        raise ContractV2Error(
            "Selected mapping contains a multi-owner window"
        )
    if any(
        len(records) > contract.max_windows_per_owner
        for records in selected_mapping.by_owner().values()
    ):
        raise ContractV2Error(
            "Contribution policy exceeded max_windows_per_owner"
        )

    epsilon_opacus = epsilon_for_noise(
        noise_multiplier=contract.noise_multiplier,
        sample_rate=contract.owner_sample_rate,
        steps=contract.total_steps,
        delta=contract.delta,
        alphas=RDP_ORDERS_V2,
    )
    epsilon_dp_accounting = epsilon_for_noise_dp_accounting_add_remove(
        noise_multiplier=contract.noise_multiplier,
        sample_rate=contract.owner_sample_rate,
        steps=contract.total_steps,
        delta=contract.delta,
        alphas=RDP_ORDERS_V2,
    )
    if abs(epsilon_opacus - epsilon_dp_accounting) > 1e-10:
        raise ContractV2Error(
            "Independent RDP accountants disagree beyond tolerance"
        )
    if max(epsilon_opacus, epsilon_dp_accounting) > (
        contract.target_epsilon + 1e-10
    ):
        raise ContractV2Error(
            "Declared noise multiplier is insufficient for the target epsilon"
        )

    executor_note = (
        "fixed-step V2 research executor is registered and ready"
        if _executor_ready_for_contract(contract, registry)
        else "executor is unavailable for the requested execution profile"
    )
    notes = (
        "q and total_steps are read only from the public contract",
        "observed owner count is neither emitted nor used to derive or validate q, T, or the denominator",
        "preprocessing parameters are digest-bound public side information",
        "the registered execution source bundle is fixed at compile time",
        "dual add/remove Poisson-Gaussian RDP accountants agree",
        executor_note,
    )
    status = (
        "validated_contract_research_executor_ready"
        if _executor_ready_for_contract(contract, registry)
        else "validated_contract_executor_not_ready"
    )
    source_mapping_file_sha256 = source_mapping.file_hash()
    source_mapping_canonical_sha256 = source_mapping.canonical_hash()
    selected_mapping_canonical_sha256 = selected_mapping.canonical_hash()
    preprocessor_state_sha256 = _preprocessor_state_sha256(preprocessor)
    try:
        execution_source_bundle_sha256 = (
            execution_source_bundle_sha256_v2()
        )
    except SourceBundleV2Error as exc:
        raise ContractV2Error(
            f"Registered execution source bundle is unavailable: {exc}"
        ) from exc
    private_execution_binding_sha256 = _sha256_payload(
        {
            "public_contract_sha256": contract.public_contract_sha256,
            "implementation_id": registry.implementation_id,
            "execution_source_bundle_sha256": (
                execution_source_bundle_sha256
            ),
            "source_mapping_file_sha256": source_mapping_file_sha256,
            "source_mapping_canonical_sha256": (
                source_mapping_canonical_sha256
            ),
            "selected_mapping_canonical_sha256": (
                selected_mapping_canonical_sha256
            ),
            "preprocessor_state_sha256": preprocessor_state_sha256,
        }
    )
    compiled = CompiledOwnerPoissonRouteV2(
        contract=contract,
        source_mapping=source_mapping,
        selected_mapping=selected_mapping,
        policy=policy,
        preprocessor=preprocessor,
        registry_entry=registry,
        accountant_epsilon_opacus=epsilon_opacus,
        accountant_epsilon_dp_accounting=epsilon_dp_accounting,
        execution_source_bundle_sha256=(
            execution_source_bundle_sha256
        ),
        public_plan_sha256="",
        private_source_mapping_sha256=source_mapping_file_sha256,
        private_source_mapping_canonical_sha256=(
            source_mapping_canonical_sha256
        ),
        private_selected_mapping_sha256=selected_mapping_canonical_sha256,
        private_preprocessor_state_sha256=preprocessor_state_sha256,
        private_execution_binding_sha256=(
            private_execution_binding_sha256
        ),
        status=status,
        notes=notes,
    )
    plan_sha256 = _sha256_payload(compiled._public_plan_without_hash())
    compiled = replace(compiled, public_plan_sha256=plan_sha256)
    compiled.assert_private_execution_integrity()
    if require_executable and not compiled.execution_ready:
        raise ExecutorNotReadyError(
            "V2 contract is valid, but no executor is approved for the "
            "requested execution profile"
        )
    return compiled
