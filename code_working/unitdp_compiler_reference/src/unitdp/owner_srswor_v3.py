"""Reference research executor for the strict owner-SRSWOR V3 route."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from unitdp.benchmark_data_v2 import array_bytes_sha256
from unitdp.compiler_srswor_v3 import (
    CompiledOwnerSrsworRouteV3,
    SrsworContractV3Error,
)
from unitdp.owner_poisson_v2 import (
    _ResearchStream,
    _build_registered_model,
    _clip_owner_gradients,
    _model_state_sha256,
    _owner_gradient,
    _predict,
    _predict_probabilities,
    _rank_metrics,
)
from unitdp.release_artifacts import assert_public_certificate_redacted
from unitdp.source_bundle_srswor_v3 import (
    SourceBundleSrsworV3Error,
    execution_source_bundle_sha256_srswor_v3,
    execution_source_bundle_srswor_v3,
)


PUBLIC_EXECUTION_SCHEMA_SRSWOR_V3 = (
    "unitdp.owner_srswor_execution_public.v3"
)
PRIVATE_EXECUTION_SCHEMA_SRSWOR_V3 = (
    "unitdp.owner_srswor_execution_private.v3"
)


class OwnerSrsworExecutionV3Error(RuntimeError):
    """Raised when execution cannot preserve the compiled V3 invariants."""


def _sha256_payload(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _derive_research_seed(master_seed: int, domain: str) -> int:
    digest = hashlib.sha256(
        b"unitdp-owner-srswor-v3\0"
        + master_seed.to_bytes(8, "big", signed=False)
        + b"\0"
        + domain.encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


@dataclass(frozen=True)
class _SrsworResearchStreamsV3:
    model_initialization: _ResearchStream
    owner_sampling: _ResearchStream
    within_owner_selection: _ResearchStream
    gaussian_noise: _ResearchStream

    @classmethod
    def from_master_seed(
        cls,
        master_seed: int,
    ) -> "_SrsworResearchStreamsV3":
        if (
            isinstance(master_seed, bool)
            or not isinstance(master_seed, int)
            or master_seed < 0
            or master_seed >= 2**63
        ):
            raise OwnerSrsworExecutionV3Error(
                "research_seed must be an integer in [0, 2**63)"
            )
        return cls(
            model_initialization=_ResearchStream(
                _derive_research_seed(
                    master_seed,
                    "model_initialization",
                )
            ),
            owner_sampling=_ResearchStream(
                _derive_research_seed(master_seed, "owner_sampling")
            ),
            within_owner_selection=_ResearchStream(
                _derive_research_seed(
                    master_seed,
                    "within_owner_selection",
                )
            ),
            gaussian_noise=_ResearchStream(
                _derive_research_seed(master_seed, "gaussian_noise")
            ),
        )


def _validate_feature_matrix(
    value: np.ndarray,
    *,
    name: str,
    input_dim: int,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] <= 0:
        raise OwnerSrsworExecutionV3Error(
            f"{name} must be a nonempty two-dimensional matrix"
        )
    if array.shape[1] != input_dim:
        raise OwnerSrsworExecutionV3Error(
            f"{name} must have {input_dim} columns"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise OwnerSrsworExecutionV3Error(f"{name} must be numeric")
    if not np.isfinite(array).all():
        raise OwnerSrsworExecutionV3Error(
            f"{name} contains non-finite values"
        )
    return array


def _validate_labels(
    value: np.ndarray,
    *,
    name: str,
    rows: int,
    num_classes: int,
) -> np.ndarray:
    labels = np.asarray(value)
    if labels.ndim != 1 or labels.shape[0] != rows:
        raise OwnerSrsworExecutionV3Error(
            f"{name} must contain exactly {rows} labels"
        )
    if np.issubdtype(labels.dtype, np.bool_) or not np.issubdtype(
        labels.dtype,
        np.integer,
    ):
        raise OwnerSrsworExecutionV3Error(
            f"{name} must use an integer dtype"
        )
    labels = labels.astype(np.int64, copy=False)
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise OwnerSrsworExecutionV3Error(
            f"{name} labels must lie in [0, {num_classes})"
        )
    return labels


def _library_versions() -> dict[str, str]:
    result = {
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    for distribution in (
        "scikit-learn",
        "dp-accounting",
    ):
        try:
            result[distribution] = version(distribution)
        except PackageNotFoundError:
            result[distribution] = "not-installed"
    return result


class OwnerSrsworExecutionResultV3:
    """Trained model with separate public and private execution records."""

    def __init__(
        self,
        *,
        model: nn.Module,
        public_record: dict[str, Any],
        private_record: dict[str, Any],
    ):
        self.model = model
        self._public_record = copy.deepcopy(public_record)
        self._private_record = copy.deepcopy(private_record)

    def assert_model_integrity(self) -> None:
        observed = _model_state_sha256(self.model)
        expected = str(
            self._public_record["model_output"]["state_sha256"]
        )
        if observed != expected:
            raise OwnerSrsworExecutionV3Error(
                "The trained model changed after record construction"
            )

    def public_payload(self) -> dict[str, Any]:
        self.assert_model_integrity()
        public = copy.deepcopy(self._public_record)
        reported = str(public.pop("public_execution_sha256"))
        if _sha256_payload(public) != reported:
            raise OwnerSrsworExecutionV3Error(
                "Public SRSWOR execution hash is inconsistent"
            )
        public["public_execution_sha256"] = reported
        assert_public_certificate_redacted(public)
        return public

    def private_manifest(self) -> dict[str, Any]:
        self.assert_model_integrity()
        private = copy.deepcopy(self._private_record)
        reported = str(private.pop("private_manifest_sha256"))
        if _sha256_payload(private) != reported:
            raise OwnerSrsworExecutionV3Error(
                "Private SRSWOR execution hash is inconsistent"
            )
        private["private_manifest_sha256"] = reported
        return private


def train_owner_srswor_fixed_v3(
    *,
    route: CompiledOwnerSrsworRouteV3,
    raw_train_x: np.ndarray,
    train_y: np.ndarray,
    raw_test_x: np.ndarray,
    test_y: np.ndarray,
    research_seed: int,
) -> OwnerSrsworExecutionResultV3:
    """Execute exactly the compiled fixed-N, fixed-m, fixed-T mechanism."""

    if not isinstance(route, CompiledOwnerSrsworRouteV3):
        raise OwnerSrsworExecutionV3Error(
            "V3 executor requires CompiledOwnerSrsworRouteV3"
        )
    try:
        route.assert_private_execution_integrity()
    except (SrsworContractV3Error, RuntimeError) as exc:
        raise OwnerSrsworExecutionV3Error(
            f"Compiled route integrity check failed: {exc}"
        ) from exc
    if not route.execution_ready:
        raise OwnerSrsworExecutionV3Error(
            "Compiled SRSWOR route is not execution-ready"
        )
    contract = route.contract
    try:
        source_bundle_before = dict(
            execution_source_bundle_srswor_v3()
        )
        source_bundle_sha256 = (
            execution_source_bundle_sha256_srswor_v3(
                source_bundle_before
            )
        )
    except SourceBundleSrsworV3Error as exc:
        raise OwnerSrsworExecutionV3Error(
            f"Could not bind the SRSWOR source bundle: {exc}"
        ) from exc
    if source_bundle_sha256 != route.execution_source_bundle_sha256:
        raise OwnerSrsworExecutionV3Error(
            "Execution source differs from the compiled route"
        )

    train_raw = _validate_feature_matrix(
        raw_train_x,
        name="raw_train_x",
        input_dim=contract.input_dim,
    )
    test_raw = _validate_feature_matrix(
        raw_test_x,
        name="raw_test_x",
        input_dim=contract.input_dim,
    )
    if train_raw.shape[0] != len(route.source_mapping.records):
        raise OwnerSrsworExecutionV3Error(
            "raw_train_x rows must equal the registered source mapping"
        )
    train_labels = _validate_labels(
        train_y,
        name="train_y",
        rows=train_raw.shape[0],
        num_classes=contract.num_classes,
    )
    test_labels = _validate_labels(
        test_y,
        name="test_y",
        rows=test_raw.shape[0],
        num_classes=contract.num_classes,
    )
    base = route.profile.base_profile
    label_to_index = dict(base.mapping_label_to_index)
    for record in route.source_mapping.records:
        if record.label not in label_to_index:
            raise OwnerSrsworExecutionV3Error(
                f"Mapping label {record.label!r} is not registered"
            )
        if label_to_index[record.label] != int(
            train_labels[record.row_index]
        ):
            raise OwnerSrsworExecutionV3Error(
                "Mapping labels do not match train_y"
            )

    train_features = route.preprocessor.transform(train_raw)
    test_features = route.preprocessor.transform(test_raw)
    observed_data_hashes = {
        "train_transformed_sha256": array_bytes_sha256(
            train_features
        ),
        "test_transformed_sha256": array_bytes_sha256(test_features),
        "train_labels_sha256": array_bytes_sha256(train_labels),
        "test_labels_sha256": array_bytes_sha256(test_labels),
    }
    expected_data_hashes = {
        key: base.full_data_conformance_payload()[key]
        for key in observed_data_hashes
    }
    if observed_data_hashes != expected_data_hashes:
        raise OwnerSrsworExecutionV3Error(
            "Executor arrays differ from the registered full-data profile"
        )

    streams = _SrsworResearchStreamsV3.from_master_seed(research_seed)
    model = _build_registered_model(
        model_id=contract.model_id,
        input_dim=contract.input_dim,
        num_classes=contract.num_classes,
        stream=streams.model_initialization,
    )
    initial_model_sha256 = _model_state_sha256(model)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=contract.learning_rate,
        momentum=0.0,
        dampening=0.0,
        weight_decay=0.0,
        nesterov=False,
        maximize=False,
        foreach=False,
        differentiable=False,
        fused=False,
    )
    class_weights = (
        torch.tensor(contract.class_weights, dtype=torch.float32)
        if contract.class_weights is not None
        else None
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        reduction="mean",
    )
    parameters = list(model.parameters())
    selected_by_owner = route.selected_mapping.by_owner()
    owners = sorted(selected_by_owner)
    owner_to_position = {
        owner: position for position, owner in enumerate(owners)
    }
    if len(owners) != contract.source_dataset_size:
        raise OwnerSrsworExecutionV3Error(
            "Execution owner population differs from fixed public N"
        )

    private_steps: list[dict[str, Any]] = []
    for step in range(contract.total_steps):
        owner_positions = streams.owner_sampling.generator.choice(
            contract.source_dataset_size,
            size=contract.sample_size,
            replace=False,
        )
        if (
            owner_positions.ndim != 1
            or len(owner_positions) != contract.sample_size
            or len(set(int(value) for value in owner_positions))
            != contract.sample_size
        ):
            raise OwnerSrsworExecutionV3Error(
                "Owner sampler did not return exactly m distinct positions"
            )
        sampled_owners = [
            owners[int(position)] for position in owner_positions.tolist()
        ]
        summed_gradients = [
            torch.zeros_like(parameter) for parameter in parameters
        ]
        owner_norms: list[float] = []
        clipped_norms: list[float] = []
        selected_rows: list[dict[str, Any]] = []
        for owner in sampled_owners:
            records = route.policy.sample_for_step(
                selected_by_owner[owner],
                streams.within_owner_selection.generator,
            )
            if not records:
                raise OwnerSrsworExecutionV3Error(
                    "Within-owner selector returned no records"
                )
            indices = [record.row_index for record in records]
            gradients, norm = _owner_gradient(
                model,
                criterion,
                torch.from_numpy(train_features[indices]),
                torch.from_numpy(train_labels[indices]),
            )
            clipped, realized = _clip_owner_gradients(
                gradients,
                norm=norm,
                clip_norm=contract.clip_norm,
            )
            for index, gradient in enumerate(clipped):
                summed_gradients[index].add_(gradient)
            owner_norms.append(norm)
            clipped_norms.append(realized)
            selected_rows.append(
                {
                    "owner_position": owner_to_position[owner],
                    "row_indices": indices,
                }
            )

        optimizer.zero_grad(set_to_none=True)
        noise_tensors = 0
        for parameter, gradient_sum in zip(
            parameters,
            summed_gradients,
        ):
            noise = torch.from_numpy(
                streams.gaussian_noise.normal_array(
                    contract.noise_multiplier * contract.clip_norm,
                    tuple(parameter.shape),
                )
            )
            parameter.grad = (
                gradient_sum + noise
            ) / contract.update_denominator
            noise_tensors += 1
        optimizer.step()
        if any(
            not torch.isfinite(parameter).all()
            for parameter in parameters
        ):
            raise OwnerSrsworExecutionV3Error(
                f"Model became non-finite at step {step}"
            )
        private_steps.append(
            {
                "step": step,
                "sampled_owner_positions": [
                    int(value) for value in owner_positions.tolist()
                ],
                "selected_rows": selected_rows,
                "sampled_owner_count": len(sampled_owners),
                "owner_vectors_computed": len(sampled_owners),
                "noise_parameter_tensors": noise_tensors,
                "optimizer_step_applied": True,
                "max_unclipped_owner_norm": max(owner_norms),
                "max_clipped_owner_norm": max(clipped_norms),
            }
        )

    try:
        route.assert_private_execution_integrity()
    except (SrsworContractV3Error, RuntimeError) as exc:
        raise OwnerSrsworExecutionV3Error(
            f"Compiled route changed during execution: {exc}"
        ) from exc
    try:
        source_bundle_after = dict(
            execution_source_bundle_srswor_v3()
        )
    except SourceBundleSrsworV3Error as exc:
        raise OwnerSrsworExecutionV3Error(
            f"Could not recheck the source bundle: {exc}"
        ) from exc
    if source_bundle_after != source_bundle_before:
        raise OwnerSrsworExecutionV3Error(
            "Execution source bundle changed during execution"
        )

    predictions = _predict(model, test_features)
    probabilities = _predict_probabilities(model, test_features)
    auroc, auprc = _rank_metrics(test_labels, probabilities)
    final_model_sha256 = _model_state_sha256(model)
    library_versions = _library_versions()
    public_record: dict[str, Any] = {
        "schema_version": PUBLIC_EXECUTION_SCHEMA_SRSWOR_V3,
        "visibility": "public",
        "release_status": "research_non_release",
        "privacy_claim": {
            "status": "accountant_output_only_not_a_release_claim",
            "reason_codes": [
                "noncryptographic_research_rng",
                "privacy_release_executor_not_audited",
            ],
        },
        "benchmark_profile_id": contract.benchmark_profile_id,
        "route": {
            "route_id": contract.route_id,
            "implementation_id": contract.implementation_id,
            "public_contract_sha256": (
                contract.public_contract_sha256
            ),
            "public_plan_sha256": route.public_plan_sha256,
        },
        "mechanism": {
            "privacy_unit": contract.privacy_unit,
            "adjacency": contract.adjacency,
            "sampler": contract.sampler,
            "sampling_unit": contract.sampling_unit,
            "clipping_unit": contract.clipping_unit,
            "noising_unit": contract.noising_unit,
            "accounting_unit": contract.accounting_unit,
            "source_dataset_size": contract.source_dataset_size,
            "sample_size": contract.sample_size,
            "sample_rate": contract.sample_rate,
            "total_steps": contract.total_steps,
            "clip_norm": contract.clip_norm,
            "actual_noise_multiplier": contract.noise_multiplier,
            "sensitivity_multiplier": (
                contract.sensitivity_multiplier
            ),
            "normalized_noise_multiplier": (
                contract.noise_multiplier
                / contract.sensitivity_multiplier
            ),
            "update_denominator": contract.update_denominator,
            "per_owner_contribution": contract.per_owner_contribution,
            "owner_aggregation": contract.owner_aggregation,
        },
        "accountant": {
            "id": contract.accountant_id,
            "epsilon": max(
                route.accountant_epsilon_dp_accounting,
                route.accountant_epsilon_direct_oracle,
            ),
            "delta": contract.delta,
            "rdp_orders": list(route.registry_entry.rdp_orders),
            "dp_accounting_epsilon": (
                route.accountant_epsilon_dp_accounting
            ),
            "direct_theorem_oracle_epsilon": (
                route.accountant_epsilon_direct_oracle
            ),
            "direct_theorem_oracle_optimal_order": (
                route.accountant_optimal_order_direct_oracle
            ),
        },
        "execution_checks": {
            "compiled_object_only": True,
            "compiled_integrity_before_and_after": True,
            "fixed_population_size_matches_mapping": True,
            "fixed_sample_size": True,
            "uniform_sampling_without_replacement": True,
            "distinct_owner_positions_every_step": True,
            "one_clipped_vector_per_sampled_owner": True,
            "replace_one_sensitivity_bound_2C": True,
            "Gaussian_noise_every_step": True,
            "optimizer_update_every_step": True,
            "fixed_update_denominator_equals_sample_size": True,
            "registered_full_data_arrays": True,
            "preprocessing_applied_inside_executor": True,
            "source_bundle_unchanged_during_execution": True,
            "random_streams_domain_separated": True,
            "random_coins_disclosed": False,
        },
        "rng": {
            "backend": "numpy_pcg64_domain_separated_research",
            "secure_rng": False,
            "random_coins_disclosed": False,
            "domains": [
                "model_initialization",
                "owner_sampling",
                "within_owner_selection",
                "gaussian_noise",
            ],
        },
        "model": {
            "model_id": contract.model_id,
            "dtype": "float32",
            "device": "cpu",
            "initialization": "domain_separated_uniform_linear_v1",
            "optimizer": "torch_sgd_plain",
            "learning_rate": contract.learning_rate,
            "loss": contract.loss,
        },
        "public_data_bindings": {
            "dataset_id": contract.dataset_id,
            "base_data_profile_id": contract.base_data_profile_id,
            "preprocessing": contract.preprocessing_binding,
            "protocol_id": contract.public_protocol["protocol_id"],
            "full_data_conformance_sha256": (
                contract.public_protocol[
                    "full_data_conformance_sha256"
                ]
            ),
        },
        "public_evaluation": {
            "scope": contract.public_protocol["evaluation_scope"],
            "accuracy": float(
                accuracy_score(test_labels, predictions)
            ),
            "macro_f1": float(
                f1_score(
                    test_labels,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            "auroc": auroc,
            "auprc": auprc,
        },
        "model_output": {
            "format": "state_dict_hash_only",
            "state_sha256": final_model_sha256,
        },
        "execution_source_bundle_sha256": source_bundle_sha256,
        "library_versions": library_versions,
    }
    public_record["public_execution_sha256"] = _sha256_payload(
        public_record
    )
    assert_public_certificate_redacted(public_record)

    private_record: dict[str, Any] = {
        "schema_version": PRIVATE_EXECUTION_SCHEMA_SRSWOR_V3,
        "visibility": "private",
        "handling": (
            "do_not_publish_contains_research_seed_and_sampling_trace"
        ),
        "research_seed": research_seed,
        "initial_model_sha256": initial_model_sha256,
        "final_model_sha256": final_model_sha256,
        "public_execution_sha256": (
            public_record["public_execution_sha256"]
        ),
        "compiled_private_execution_binding_sha256": (
            route.private_execution_binding_sha256
        ),
        "source_mapping_file_sha256": (
            route.private_source_mapping_sha256
        ),
        "source_mapping_canonical_sha256": (
            route.private_source_mapping_canonical_sha256
        ),
        "selected_mapping_canonical_sha256": (
            route.private_selected_mapping_sha256
        ),
        "step_diagnostics": private_steps,
    }
    private_record["private_manifest_sha256"] = _sha256_payload(
        private_record
    )
    return OwnerSrsworExecutionResultV3(
        model=model,
        public_record=public_record,
        private_record=private_record,
    )
