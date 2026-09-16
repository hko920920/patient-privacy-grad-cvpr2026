"""Immutable benchmark registrations for the strict V3 SRSWOR route."""

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
from unitdp_spec_oracle.srswor_rdp_oracle_v3 import (
    ORACLE_IMPLEMENTATION_ID_V3,
    ORACLE_THEOREM_ID_V3,
    SUPPORTED_INTEGER_ORDERS_V3,
)


CONTRACT_SCHEMA_SRSWOR_V3 = "unitdp.owner_srswor_contract.v3"
CORE_ROUTE_ID_SRSWOR_V3 = "owa_owner_srswor_rdp_replace_one_v3"
ACCOUNTANT_ID_SRSWOR_V3 = (
    "sampled_without_replacement_gaussian_rdp_replace_one_dual_v1"
)
RDP_ORDERS_ID_SRSWOR_V3 = "unitdp_srswor_integer_orders_v1"
RDP_ORDERS_SRSWOR_V3 = SUPPORTED_INTEGER_ORDERS_V3
EXECUTOR_IMPLEMENTATION_ID_SRSWOR_V3 = (
    "unitdp.owner_srswor_v3.train_owner_srswor_fixed_v3"
)


@dataclass(frozen=True)
class SrsworBenchmarkProfileV3:
    """Route constants plus an exact binding to one V2 public data profile."""

    profile_id: str
    base_data_profile_id: str
    source_dataset_size: int
    sample_size: int
    noise_multiplier: float
    target_epsilon: float = 8.0
    delta: float = 1e-5
    sensitivity_multiplier: float = 2.0

    def __post_init__(self) -> None:
        base = self.base_profile
        if self.source_dataset_size != base.train_owners:
            raise ValueError(
                "SRSWOR source_dataset_size must equal the public train-owner "
                "count in the bound base-data profile"
            )
        if (
            isinstance(self.sample_size, bool)
            or not isinstance(self.sample_size, int)
            or self.sample_size <= 0
            or self.sample_size > self.source_dataset_size
        ):
            raise ValueError(
                "SRSWOR sample_size must be an integer in [1, N]"
            )
        if self.sample_size != int(base.update_denominator):
            raise ValueError(
                "The registered fixed-size sample must equal the update "
                "denominator"
            )
        if self.noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be positive")
        if self.target_epsilon <= 0 or not 0 < self.delta < 1:
            raise ValueError("invalid registered privacy target")
        if self.sensitivity_multiplier != 2.0:
            raise ValueError(
                "replace-one owner adjacency requires the registered 2C bound"
            )

    @property
    def base_profile(self) -> BenchmarkProfileV2:
        return get_benchmark_profile_v2(self.base_data_profile_id)

    @property
    def sample_rate(self) -> float:
        return self.sample_size / self.source_dataset_size

    def registry_payload(self) -> dict[str, Any]:
        base = self.base_profile
        return {
            "schema_version": "unitdp.srswor_benchmark_profile.v3",
            "profile_id": self.profile_id,
            "base_data_profile_id": self.base_data_profile_id,
            "base_data_profile_sha256": base.registry_sha256,
            "full_data_conformance_sha256": (
                base.full_data_conformance_sha256
            ),
            "route_id": CORE_ROUTE_ID_SRSWOR_V3,
            "privacy": {
                "unit": "owner",
                "adjacency": "replace_one_owner",
                "target_epsilon": self.target_epsilon,
                "delta": self.delta,
            },
            "mechanism": {
                "sampler": (
                    "uniform_fixed_size_without_replacement_owner"
                ),
                "source_dataset_size": self.source_dataset_size,
                "sample_size": self.sample_size,
                "sample_rate": self.sample_rate,
                "total_steps": base.total_steps,
                "clip_norm": base.clip_norm,
                "noise_multiplier": self.noise_multiplier,
                "update_denominator": float(self.sample_size),
            },
            "accountant": {
                "id": ACCOUNTANT_ID_SRSWOR_V3,
                "rdp_orders_id": RDP_ORDERS_ID_SRSWOR_V3,
                "rdp_orders": list(RDP_ORDERS_SRSWOR_V3),
                "sensitivity_multiplier": self.sensitivity_multiplier,
                "theorem_id": ORACLE_THEOREM_ID_V3,
                "oracle_implementation_id": ORACLE_IMPLEMENTATION_ID_V3,
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
                "adjacency": "replace_one_owner",
                "target_epsilon": self.target_epsilon,
                "delta": self.delta,
            },
            "mechanism": {
                "sampler": (
                    "uniform_fixed_size_without_replacement_owner"
                ),
                "sampling_unit": "owner",
                "accounting_unit": "owner",
                "parameter_source": "public_contract",
                "source_dataset_size": self.source_dataset_size,
                "sample_size": self.sample_size,
                "total_steps": base.total_steps,
                "clipping_unit": "owner",
                "noising_unit": "owner",
                "clip_norm": base.clip_norm,
                "noise_multiplier": self.noise_multiplier,
                "update_denominator": float(self.sample_size),
                "empty_step_behavior": (
                    "impossible_for_positive_fixed_sample_size"
                ),
                "per_owner_contribution": (
                    "one_clipped_vector_per_sampled_owner"
                ),
                "owner_aggregation": (
                    "mean_over_selected_windows_then_clip"
                ),
            },
            "accountant": {
                "id": ACCOUNTANT_ID_SRSWOR_V3,
                "rdp_orders_id": RDP_ORDERS_ID_SRSWOR_V3,
                "sensitivity_multiplier": self.sensitivity_multiplier,
                "theorem_id": ORACLE_THEOREM_ID_V3,
                "oracle_implementation_id": ORACLE_IMPLEMENTATION_ID_V3,
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


UCI_HAR_SRSWOR_AAAI27_V3 = SrsworBenchmarkProfileV3(
    profile_id="uci_har_srswor_aaai27_v3",
    base_data_profile_id="uci_har_aaai27_v1",
    source_dataset_size=21,
    sample_size=8,
    noise_multiplier=3.806632095748225,
)

WISDM_SRSWOR_AAAI27_V3 = SrsworBenchmarkProfileV3(
    profile_id="wisdm_srswor_aaai27_v3",
    base_data_profile_id="wisdm_aaai27_v1",
    source_dataset_size=25,
    sample_size=8,
    noise_multiplier=3.8386964649452464,
)

SEPSIS_SRSWOR_AAAI27_V3 = SrsworBenchmarkProfileV3(
    profile_id="sepsis_srswor_aaai27_v3",
    base_data_profile_id="sepsis_aaai27_v1",
    source_dataset_size=312,
    sample_size=32,
    noise_multiplier=2.3803808386399714,
)


SRSWOR_BENCHMARK_PROFILES_V3 = MappingProxyType(
    {
        profile.profile_id: profile
        for profile in (
            UCI_HAR_SRSWOR_AAAI27_V3,
            WISDM_SRSWOR_AAAI27_V3,
            SEPSIS_SRSWOR_AAAI27_V3,
        )
    }
)


def get_srswor_benchmark_profile_v3(
    profile_id: str,
) -> SrsworBenchmarkProfileV3:
    try:
        return SRSWOR_BENCHMARK_PROFILES_V3[profile_id]
    except KeyError as exc:
        raise KeyError(
            f"Unknown SRSWOR V3 benchmark profile: {profile_id!r}"
        ) from exc
