"""Research executor for the strict owner random-allocation V4 route.

The executor accepts only a compiled registered route.  It uses a
domain-separated reproducibility backend and is not an audited release RNG.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from unitdp.benchmark_data_v2 import array_bytes_sha256
from unitdp.compiler_allocation_v4 import (
    AllocationContractV4Error,
    CompiledOwnerRandomAllocationRouteV4,
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
from unitdp.source_bundle_allocation_v4 import (
    SourceBundleAllocationV4Error,
    execution_source_bundle_allocation_v4,
    execution_source_bundle_sha256_allocation_v4,
)


PUBLIC_EXECUTION_SCHEMA_ALLOCATION_V4 = (
    "unitdp.owner_random_allocation_execution_public.v4"
)
PRIVATE_EXECUTION_SCHEMA_ALLOCATION_V4 = (
    "unitdp.owner_random_allocation_execution_private.v4"
)


class OwnerRandomAllocationExecutionV4Error(RuntimeError):
    """Raised when execution cannot preserve the compiled V4 invariants."""


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
        b"unitdp-owner-random-allocation-v4\0"
        + master_seed.to_bytes(8, "big", signed=False)
        + b"\0"
        + domain.encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


@dataclass(frozen=True)
class _AllocationResearchStreamsV4:
    model_initialization: _ResearchStream
    owner_step_allocation: _ResearchStream
    within_owner_selection: _ResearchStream
    gaussian_noise: _ResearchStream

    @classmethod
    def from_master_seed(
        cls,
        master_seed: int,
    ) -> "_AllocationResearchStreamsV4":
        if (
            isinstance(master_seed, bool)
            or not isinstance(master_seed, int)
            or master_seed < 0
            or master_seed >= 2**63
        ):
            raise OwnerRandomAllocationExecutionV4Error(
                "research_seed must be an integer in [0, 2**63)"
            )
        return cls(
            model_initialization=_ResearchStream(
                _derive_research_seed(
                    master_seed,
                    "model_initialization",
                )
            ),
            owner_step_allocation=_ResearchStream(
                _derive_research_seed(
                    master_seed,
                    "owner_step_allocation",
                )
            ),
            within_owner_selection=_ResearchStream(
                _derive_research_seed(
                    master_seed,
                    "within_owner_selection",
                )
            ),
            gaussian_noise=_ResearchStream(
                _derive_research_seed(
                    master_seed,
                    "gaussian_noise",
                )
            ),
        )


@dataclass(frozen=True)
class OwnerAllocationScheduleV4:
    """Private exact-k schedule materialized before adaptive training."""

    owner_positions_by_step: tuple[tuple[int, ...], ...]
    local_steps_by_epoch_and_owner: tuple[
        tuple[tuple[int, ...], ...],
        ...,
    ]

    @property
    def sha256(self) -> str:
        return _sha256_payload(
            {
                "owner_positions_by_step": [
                    list(row) for row in self.owner_positions_by_step
                ],
                "local_steps_by_epoch_and_owner": [
                    [list(row) for row in epoch]
                    for epoch in self.local_steps_by_epoch_and_owner
                ],
            }
        )


def build_owner_allocation_schedule_v4(
    *,
    num_owners: int,
    num_steps: int,
    num_selected_steps_per_owner: int,
    num_epochs: int,
    generator: np.random.Generator,
) -> OwnerAllocationScheduleV4:
    """Sample an independent uniform k-subset for every owner and epoch."""

    for name, value in (
        ("num_owners", num_owners),
        ("num_steps", num_steps),
        (
            "num_selected_steps_per_owner",
            num_selected_steps_per_owner,
        ),
        ("num_epochs", num_epochs),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise OwnerRandomAllocationExecutionV4Error(
                f"{name} must be a positive integer"
            )
    if num_selected_steps_per_owner > num_steps:
        raise OwnerRandomAllocationExecutionV4Error(
            "num_selected_steps_per_owner must not exceed num_steps"
        )
    if not isinstance(generator, np.random.Generator):
        raise OwnerRandomAllocationExecutionV4Error(
            "generator must be numpy.random.Generator"
        )

    assignments: list[list[int]] = [
        [] for _ in range(num_steps * num_epochs)
    ]
    private_owner_rows: list[tuple[tuple[int, ...], ...]] = []
    for epoch in range(num_epochs):
        epoch_rows: list[tuple[int, ...]] = []
        for owner_position in range(num_owners):
            selected = tuple(
                sorted(
                    int(value)
                    for value in generator.choice(
                        num_steps,
                        size=num_selected_steps_per_owner,
                        replace=False,
                    ).tolist()
                )
            )
            if (
                len(selected) != num_selected_steps_per_owner
                or len(set(selected)) != num_selected_steps_per_owner
                or any(step < 0 or step >= num_steps for step in selected)
            ):
                raise OwnerRandomAllocationExecutionV4Error(
                    "Owner allocation did not produce exactly k distinct steps"
                )
            epoch_rows.append(selected)
            for local_step in selected:
                assignments[epoch * num_steps + local_step].append(
                    owner_position
                )
        private_owner_rows.append(tuple(epoch_rows))

    schedule = OwnerAllocationScheduleV4(
        owner_positions_by_step=tuple(
            tuple(row) for row in assignments
        ),
        local_steps_by_epoch_and_owner=tuple(private_owner_rows),
    )
    if len(schedule.owner_positions_by_step) != num_steps * num_epochs:
        raise OwnerRandomAllocationExecutionV4Error(
            "Allocation schedule has the wrong total step count"
        )
    for epoch_rows in schedule.local_steps_by_epoch_and_owner:
        if len(epoch_rows) != num_owners:
            raise OwnerRandomAllocationExecutionV4Error(
                "Allocation schedule has the wrong owner count"
            )
        for selected in epoch_rows:
            if (
                len(selected) != num_selected_steps_per_owner
                or len(set(selected)) != num_selected_steps_per_owner
            ):
                raise OwnerRandomAllocationExecutionV4Error(
                    "Allocation schedule violates exact-k distinctness"
                )
    return schedule


def _validate_feature_matrix(
    value: np.ndarray,
    *,
    name: str,
    input_dim: int,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] <= 0:
        raise OwnerRandomAllocationExecutionV4Error(
            f"{name} must be a nonempty two-dimensional matrix"
        )
    if array.shape[1] != input_dim:
        raise OwnerRandomAllocationExecutionV4Error(
            f"{name} must have {input_dim} columns"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise OwnerRandomAllocationExecutionV4Error(
            f"{name} must be numeric"
        )
    if not np.isfinite(array).all():
        raise OwnerRandomAllocationExecutionV4Error(
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
        raise OwnerRandomAllocationExecutionV4Error(
            f"{name} must contain exactly {rows} labels"
        )
    if np.issubdtype(labels.dtype, np.bool_) or not np.issubdtype(
        labels.dtype,
        np.integer,
    ):
        raise OwnerRandomAllocationExecutionV4Error(
            f"{name} must use an integer dtype"
        )
    labels = labels.astype(np.int64, copy=False)
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise OwnerRandomAllocationExecutionV4Error(
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
        "mpmath",
        "scipy",
    ):
        try:
            result[distribution] = version(distribution)
        except PackageNotFoundError:
            result[distribution] = "not-installed"
    return result


class OwnerRandomAllocationExecutionResultV4:
    """Trained model plus separate public and private execution records."""

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
            raise OwnerRandomAllocationExecutionV4Error(
                "The trained model changed after record construction"
            )

    def public_payload(self) -> dict[str, Any]:
        self.assert_model_integrity()
        public = copy.deepcopy(self._public_record)
        reported = str(public.pop("public_execution_sha256"))
        if _sha256_payload(public) != reported:
            raise OwnerRandomAllocationExecutionV4Error(
                "Public random-allocation execution hash is inconsistent"
            )
        public["public_execution_sha256"] = reported
        assert_public_certificate_redacted(public)
        return public

    def private_manifest(self) -> dict[str, Any]:
        self.assert_model_integrity()
        private = copy.deepcopy(self._private_record)
        reported = str(private.pop("private_manifest_sha256"))
        if _sha256_payload(private) != reported:
            raise OwnerRandomAllocationExecutionV4Error(
                "Private random-allocation execution hash is inconsistent"
            )
        private["private_manifest_sha256"] = reported
        return private


def train_owner_random_allocation_fixed_v4(
    *,
    route: CompiledOwnerRandomAllocationRouteV4,
    raw_train_x: np.ndarray,
    train_y: np.ndarray,
    raw_test_x: np.ndarray,
    test_y: np.ndarray,
    research_seed: int,
) -> OwnerRandomAllocationExecutionResultV4:
    """Execute exactly the compiled fixed-N, exact-k-of-t mechanism."""

    if not isinstance(route, CompiledOwnerRandomAllocationRouteV4):
        raise OwnerRandomAllocationExecutionV4Error(
            "V4 executor requires CompiledOwnerRandomAllocationRouteV4"
        )
    try:
        route.assert_private_execution_integrity()
    except (AllocationContractV4Error, RuntimeError) as exc:
        raise OwnerRandomAllocationExecutionV4Error(
            f"Compiled route integrity check failed: {exc}"
        ) from exc
    if not route.execution_ready:
        raise OwnerRandomAllocationExecutionV4Error(
            "Compiled random-allocation route is not execution-ready"
        )
    contract = route.contract
    try:
        source_bundle_before = dict(
            execution_source_bundle_allocation_v4()
        )
        source_bundle_sha256 = (
            execution_source_bundle_sha256_allocation_v4(
                source_bundle_before
            )
        )
    except SourceBundleAllocationV4Error as exc:
        raise OwnerRandomAllocationExecutionV4Error(
            f"Could not bind the random-allocation source bundle: {exc}"
        ) from exc
    if source_bundle_sha256 != route.execution_source_bundle_sha256:
        raise OwnerRandomAllocationExecutionV4Error(
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
        raise OwnerRandomAllocationExecutionV4Error(
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
            raise OwnerRandomAllocationExecutionV4Error(
                f"Mapping label {record.label!r} is not registered"
            )
        if label_to_index[record.label] != int(
            train_labels[record.row_index]
        ):
            raise OwnerRandomAllocationExecutionV4Error(
                "Mapping labels do not match train_y"
            )

    train_features = route.preprocessor.transform(train_raw)
    test_features = route.preprocessor.transform(test_raw)
    observed_data_hashes = {
        "train_transformed_sha256": array_bytes_sha256(train_features),
        "test_transformed_sha256": array_bytes_sha256(test_features),
        "train_labels_sha256": array_bytes_sha256(train_labels),
        "test_labels_sha256": array_bytes_sha256(test_labels),
    }
    expected_data_hashes = {
        key: base.full_data_conformance_payload()[key]
        for key in observed_data_hashes
    }
    if observed_data_hashes != expected_data_hashes:
        raise OwnerRandomAllocationExecutionV4Error(
            "Executor arrays differ from the registered full-data profile"
        )

    streams = _AllocationResearchStreamsV4.from_master_seed(research_seed)
    selected_by_owner = route.selected_mapping.by_owner()
    owners = sorted(selected_by_owner)
    owner_to_position = {
        owner: position for position, owner in enumerate(owners)
    }
    if len(owners) != contract.source_dataset_size:
        raise OwnerRandomAllocationExecutionV4Error(
            "Execution owner population differs from fixed public N"
        )
    schedule = build_owner_allocation_schedule_v4(
        num_owners=contract.source_dataset_size,
        num_steps=contract.num_steps,
        num_selected_steps_per_owner=(
            contract.num_selected_steps_per_owner
        ),
        num_epochs=contract.num_epochs,
        generator=streams.owner_step_allocation.generator,
    )

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

    private_steps: list[dict[str, Any]] = []
    for step, owner_positions in enumerate(
        schedule.owner_positions_by_step
    ):
        if len(owner_positions) != len(set(owner_positions)):
            raise OwnerRandomAllocationExecutionV4Error(
                "One owner appeared twice in an allocation step"
            )
        assigned_owners = [
            owners[position] for position in owner_positions
        ]
        summed_gradients = [
            torch.zeros_like(parameter) for parameter in parameters
        ]
        owner_norms: list[float] = []
        clipped_norms: list[float] = []
        selected_rows: list[dict[str, Any]] = []
        for owner in assigned_owners:
            records = route.policy.sample_for_step(
                selected_by_owner[owner],
                streams.within_owner_selection.generator,
            )
            if not records:
                raise OwnerRandomAllocationExecutionV4Error(
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
            raise OwnerRandomAllocationExecutionV4Error(
                f"Model became non-finite at step {step}"
            )
        private_steps.append(
            {
                "step": step,
                "assigned_owner_positions": list(owner_positions),
                "selected_rows": selected_rows,
                "assigned_owner_count": len(assigned_owners),
                "owner_vectors_computed": len(assigned_owners),
                "noise_parameter_tensors": noise_tensors,
                "optimizer_step_applied": True,
                "max_unclipped_owner_norm": max(
                    owner_norms,
                    default=0.0,
                ),
                "max_clipped_owner_norm": max(
                    clipped_norms,
                    default=0.0,
                ),
            }
        )

    try:
        route.assert_private_execution_integrity()
    except (AllocationContractV4Error, RuntimeError) as exc:
        raise OwnerRandomAllocationExecutionV4Error(
            f"Compiled route changed during execution: {exc}"
        ) from exc
    try:
        source_bundle_after = dict(
            execution_source_bundle_allocation_v4()
        )
    except SourceBundleAllocationV4Error as exc:
        raise OwnerRandomAllocationExecutionV4Error(
            f"Could not recheck the source bundle: {exc}"
        ) from exc
    if source_bundle_after != source_bundle_before:
        raise OwnerRandomAllocationExecutionV4Error(
            "Execution source bundle changed during execution"
        )

    predictions = _predict(model, test_features)
    probabilities = _predict_probabilities(model, test_features)
    auroc, auprc = _rank_metrics(test_labels, probabilities)
    final_model_sha256 = _model_state_sha256(model)
    public_record: dict[str, Any] = {
        "schema_version": PUBLIC_EXECUTION_SCHEMA_ALLOCATION_V4,
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
            "assignment_schedule": contract.assignment_schedule,
            "sampling_unit": contract.sampling_unit,
            "clipping_unit": contract.clipping_unit,
            "noising_unit": contract.noising_unit,
            "accounting_unit": contract.accounting_unit,
            "source_dataset_size": contract.source_dataset_size,
            "num_steps": contract.num_steps,
            "num_selected_steps_per_owner": (
                contract.num_selected_steps_per_owner
            ),
            "num_epochs": contract.num_epochs,
            "total_steps": contract.total_steps,
            "marginal_participation_rate": (
                contract.marginal_participation_rate
            ),
            "expected_owners_per_step": (
                contract.expected_owners_per_step
            ),
            "clip_norm": contract.clip_norm,
            "actual_noise_multiplier": contract.noise_multiplier,
            "sensitivity_multiplier": contract.sensitivity_multiplier,
            "update_denominator": contract.update_denominator,
            "per_owner_contribution": contract.per_owner_contribution,
            "owner_aggregation": contract.owner_aggregation,
        },
        "accountant": {
            "id": contract.accountant_id,
            "epsilon": route.accountant_epsilon,
            "epsilon_remove": route.accountant_epsilon_remove,
            "epsilon_add": route.accountant_epsilon_add,
            "delta": contract.delta,
            "rdp_orders": list(route.registry_entry.rdp_orders),
            "high_precision_oracle_epsilon": route.oracle_epsilon,
            "optimal_remove_order": (
                route.accountant_optimal_remove_order
            ),
        },
        "execution_checks": {
            "compiled_object_only": True,
            "compiled_integrity_before_and_after": True,
            "exact_k_distinct_steps_per_owner_per_epoch": True,
            "independent_owner_allocation_calls": True,
            "allocation_materialized_before_adaptive_training": True,
            "one_clipped_vector_per_assigned_owner": True,
            "add_remove_sensitivity_bound_C": True,
            "Gaussian_noise_every_step_including_empty": True,
            "optimizer_update_every_step": True,
            "fixed_public_update_denominator": True,
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
                "owner_step_allocation",
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
        "library_versions": _library_versions(),
    }
    public_record["public_execution_sha256"] = _sha256_payload(
        public_record
    )
    assert_public_certificate_redacted(public_record)

    private_record: dict[str, Any] = {
        "schema_version": PRIVATE_EXECUTION_SCHEMA_ALLOCATION_V4,
        "visibility": "private",
        "handling": (
            "do_not_publish_contains_research_seed_and_allocation_trace"
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
        "allocation_schedule_sha256": schedule.sha256,
        "local_steps_by_epoch_and_owner": [
            [list(row) for row in epoch]
            for epoch in schedule.local_steps_by_epoch_and_owner
        ],
        "step_diagnostics": private_steps,
    }
    private_record["private_manifest_sha256"] = _sha256_payload(
        private_record
    )
    return OwnerRandomAllocationExecutionResultV4(
        model=model,
        public_record=public_record,
        private_record=private_record,
    )
