"""Immutable benchmark registrations for owner random allocation V4."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from unitdp.benchmark_registry_v2 import (
    BenchmarkProfileV2,
    get_benchmark_profile_v2,
)
from unitdp.random_allocation_accountant_v4 import (
    RANDOM_ALLOCATION_ACCOUNTANT_ID_V4,
    RANDOM_ALLOCATION_RDP_ORDERS_V4,
    RANDOM_ALLOCATION_THEOREM_ID_V4,
)


CONTRACT_SCHEMA_ALLOCATION_V4 = (
    "unitdp.owner_random_allocation_contract.v4"
)
CORE_ROUTE_ID_ALLOCATION_V4 = (
    "owa_owner_random_allocation_direct_add_remove_v4"
)
RDP_ORDERS_ID_ALLOCATION_V4 = (
    "unitdp_random_allocation_integer_orders_v4"
)
EXECUTOR_IMPLEMENTATION_ID_ALLOCATION_V4 = (
    "unitdp.owner_random_allocation_v4."
    "train_owner_random_allocation_fixed_v4"
)
ORACLE_IMPLEMENTATION_ID_ALLOCATION_V4 = (
    "unitdp_spec_oracle.random_allocation_oracle_v4."
    "account_random_allocation_oracle_v4"
)
SAMPLER_ID_ALLOCATION_V4 = (
    "independent_uniform_k_subset_of_steps_per_owner"
)
REDUCTION_ID_ALLOCATION_V4 = (
    "k_out_of_t_to_k_composed_one_out_of_floor_t_over_k"
)


@dataclass(frozen=True)
class AllocationBenchmarkProfileV4:
    """Route constants bound exactly to one public V2 data profile."""

    profile_id: str
    base_data_profile_id: str
    source_dataset_size: int
    num_steps: int
    num_selected_steps_per_owner: int
    noise_multiplier: float
    num_epochs: int = 1
    target_epsilon: float = 8.0
    delta: float = 1e-5
    sensitivity_multiplier: float = 1.0

    def __post_init__(self) -> None:
        base = self.base_profile
        if self.source_dataset_size != base.train_owners:
            raise ValueError(
                "Random-allocation source_dataset_size must equal the "
                "registered public train-owner count"
            )
        for name, value in (
            ("num_steps", self.num_steps),
            (
                "num_selected_steps_per_owner",
                self.num_selected_steps_per_owner,
            ),
            ("num_epochs", self.num_epochs),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if self.num_steps != base.total_steps:
            raise ValueError(
                "Random-allocation num_steps must equal the registered "
                "benchmark step count"
            )
        if self.num_selected_steps_per_owner > self.num_steps:
            raise ValueError(
                "num_selected_steps_per_owner must not exceed num_steps"
            )
        if self.num_steps // self.num_selected_steps_per_owner < 2:
            raise ValueError(
                "Registered direct reduction requires floor(t/k) >= 2"
            )
        if self.num_epochs != 1:
            raise ValueError(
                "The current benchmark registry fixes exactly one epoch"
            )
        if self.noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be positive")
        if self.target_epsilon <= 0 or not 0 < self.delta < 1:
            raise ValueError("invalid registered privacy target")
        if self.sensitivity_multiplier != 1.0:
            raise ValueError(
                "The registered add/remove Gaussian query uses sensitivity C"
            )

    @property
    def base_profile(self) -> BenchmarkProfileV2:
        return get_benchmark_profile_v2(self.base_data_profile_id)

    @property
    def total_steps(self) -> int:
        return self.num_steps * self.num_epochs

    @property
    def marginal_participation_rate(self) -> float:
        return self.num_selected_steps_per_owner / self.num_steps

    @property
    def expected_owners_per_step(self) -> float:
        return (
            self.source_dataset_size * self.marginal_participation_rate
        )

    def registry_payload(self) -> dict[str, Any]:
        base = self.base_profile
        return {
            "schema_version": (
                "unitdp.random_allocation_benchmark_profile.v4"
            ),
            "profile_id": self.profile_id,
            "base_data_profile_id": self.base_data_profile_id,
            "base_data_profile_sha256": base.registry_sha256,
            "full_data_conformance_sha256": (
                base.full_data_conformance_sha256
            ),
            "route_id": CORE_ROUTE_ID_ALLOCATION_V4,
            "privacy": {
                "unit": "owner",
                "adjacency": "add_remove_one_owner",
                "target_epsilon": self.target_epsilon,
                "delta": self.delta,
            },
            "mechanism": {
                "sampler": SAMPLER_ID_ALLOCATION_V4,
                "source_dataset_size": self.source_dataset_size,
                "num_steps": self.num_steps,
                "num_selected_steps_per_owner": (
                    self.num_selected_steps_per_owner
                ),
                "num_epochs": self.num_epochs,
                "total_steps": self.total_steps,
                "marginal_participation_rate": (
                    self.marginal_participation_rate
                ),
                "expected_owners_per_step": (
                    self.expected_owners_per_step
                ),
                "clip_norm": base.clip_norm,
                "noise_multiplier": self.noise_multiplier,
                "update_denominator": base.update_denominator,
            },
            "accountant": {
                "id": RANDOM_ALLOCATION_ACCOUNTANT_ID_V4,
                "rdp_orders_id": RDP_ORDERS_ID_ALLOCATION_V4,
                "rdp_orders": list(RANDOM_ALLOCATION_RDP_ORDERS_V4),
                "sensitivity_multiplier": self.sensitivity_multiplier,
                "theorem_id": RANDOM_ALLOCATION_THEOREM_ID_V4,
                "oracle_implementation_id": (
                    ORACLE_IMPLEMENTATION_ID_ALLOCATION_V4
                ),
                "reduction_id": REDUCTION_ID_ALLOCATION_V4,
                "direction_policy": "max_add_remove",
            },
            "contribution_policy": {
                "type": "support_balanced_cap",
                "max_windows_per_owner": base.max_windows_per_owner,
                "windows_per_owner_per_step": (
                    base.windows_per_owner_per_step
                ),
                "scope": "compile_time_public_policy",
            },
            "optimization": {
                "optimizer": "sgd",
                "learning_rate": base.learning_rate,
                "loss": "cross_entropy",
                "model_id": base.model_id,
                "class_weights": (
                    list(base.class_weights)
                    if base.class_weights is not None
                    else None
                ),
            },
        }

    @property
    def registry_sha256(self) -> str:
        encoded = json.dumps(
            self.registry_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def expected_contract_binding(self) -> dict[str, Any]:
        base = self.base_profile
        return {
            "privacy": {
                "unit": "owner",
                "adjacency": "add_remove_one_owner",
                "target_epsilon": self.target_epsilon,
                "delta": self.delta,
            },
            "mechanism": {
                "sampler": SAMPLER_ID_ALLOCATION_V4,
                "sampling_unit": "owner",
                "accounting_unit": "owner",
                "parameter_source": "registered_benchmark_profile",
                "source_dataset_size": self.source_dataset_size,
                "num_steps": self.num_steps,
                "num_selected_steps_per_owner": (
                    self.num_selected_steps_per_owner
                ),
                "num_epochs": self.num_epochs,
                "total_steps": self.total_steps,
                "assignment_schedule": (
                    "fresh_independent_uniform_k_subset_per_owner_per_epoch"
                ),
                "clipping_unit": "owner",
                "noising_unit": "owner_sum",
                "clip_norm": base.clip_norm,
                "noise_multiplier": self.noise_multiplier,
                "update_denominator": base.update_denominator,
                "empty_step_behavior": (
                    "apply_gaussian_update_with_zero_owner_sum"
                ),
                "per_owner_contribution": (
                    "one_clipped_vector_per_assigned_step"
                ),
                "owner_aggregation": (
                    "mean_over_selected_windows_then_clip"
                ),
            },
            "accountant": {
                "id": RANDOM_ALLOCATION_ACCOUNTANT_ID_V4,
                "rdp_orders_id": RDP_ORDERS_ID_ALLOCATION_V4,
                "sensitivity_multiplier": self.sensitivity_multiplier,
                "theorem_id": RANDOM_ALLOCATION_THEOREM_ID_V4,
                "oracle_implementation_id": (
                    ORACLE_IMPLEMENTATION_ID_ALLOCATION_V4
                ),
                "reduction_id": REDUCTION_ID_ALLOCATION_V4,
                "direction_policy": "max_add_remove",
            },
            "contribution_policy": {
                "type": "support_balanced_cap",
                "max_windows_per_owner": base.max_windows_per_owner,
                "windows_per_owner_per_step": (
                    base.windows_per_owner_per_step
                ),
                "scope": "compile_time_public_policy",
            },
            "optimization": {
                "optimizer": "sgd",
                "learning_rate": base.learning_rate,
                "loss": "cross_entropy",
                "model_id": base.model_id,
                "class_weights": (
                    list(base.class_weights)
                    if base.class_weights is not None
                    else None
                ),
            },
            "data": {
                "benchmark_profile_id": self.profile_id,
                "benchmark_profile_sha256": self.registry_sha256,
                "base_data_profile_id": base.profile_id,
                "base_data_profile_sha256": base.registry_sha256,
                "dataset_id": base.dataset_id,
                "input_dim": base.input_dim,
                "num_classes": base.num_classes,
                "preprocessing": {
                    "scope": "public_fixed",
                    "schema": "unitdp.fixed_affine_preprocessor.v1",
                    "artifact_id": base.preprocessing_artifact_id,
                    "artifact_sha256": (
                        base.preprocessing_artifact_sha256
                    ),
                    "source_reference_sha256": (
                        base.source_reference_sha256
                    ),
                    "input_dim": base.input_dim,
                    "transform": "standard_scaler",
                    "fit_on_protected_data": False,
                },
                "public_protocol": {
                    "protocol_id": base.protocol_id,
                    "data_preparation_implementation_id": (
                        base.data_preparation_implementation_id
                    ),
                    "source_reference_sha256": (
                        base.source_reference_sha256
                    ),
                    "train_owners": base.train_owners,
                    "test_owners": base.test_owners,
                    "evaluation_scope": "public_benchmark",
                    "full_data_conformance_sha256": (
                        base.full_data_conformance_sha256
                    ),
                },
            },
        }


UCI_HAR_RANDOM_ALLOCATION_AAAI27_V4 = AllocationBenchmarkProfileV4(
    profile_id="uci_har_random_allocation_aaai27_v4",
    base_data_profile_id="uci_har_aaai27_v1",
    source_dataset_size=21,
    num_steps=15,
    num_selected_steps_per_owner=6,
    noise_multiplier=1.2295752282782475,
)

WISDM_RANDOM_ALLOCATION_AAAI27_V4 = AllocationBenchmarkProfileV4(
    profile_id="wisdm_random_allocation_aaai27_v4",
    base_data_profile_id="wisdm_aaai27_v1",
    source_dataset_size=25,
    num_steps=20,
    num_selected_steps_per_owner=6,
    noise_multiplier=1.090645987511382,
)

SEPSIS_RANDOM_ALLOCATION_AAAI27_V4 = AllocationBenchmarkProfileV4(
    profile_id="sepsis_random_allocation_aaai27_v4",
    base_data_profile_id="sepsis_aaai27_v1",
    source_dataset_size=312,
    num_steps=50,
    num_selected_steps_per_owner=5,
    noise_multiplier=0.776172784941656,
)


ALLOCATION_BENCHMARK_PROFILES_V4 = MappingProxyType(
    {
        profile.profile_id: profile
        for profile in (
            UCI_HAR_RANDOM_ALLOCATION_AAAI27_V4,
            WISDM_RANDOM_ALLOCATION_AAAI27_V4,
            SEPSIS_RANDOM_ALLOCATION_AAAI27_V4,
        )
    }
)


def get_allocation_benchmark_profile_v4(
    profile_id: str,
) -> AllocationBenchmarkProfileV4:
    try:
        return ALLOCATION_BENCHMARK_PROFILES_V4[profile_id]
    except KeyError as exc:
        raise KeyError(
            f"Unknown random-allocation V4 benchmark profile: {profile_id!r}"
        ) from exc
