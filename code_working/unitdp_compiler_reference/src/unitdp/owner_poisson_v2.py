"""Reference research executor for the strict V2 owner-Poisson route.

The executor intentionally accepts no free-form optimizer, model, schedule, or
privacy configuration.  Those values come only from a validated compiled V2
route.  This module implements the reproducible research backend; it is not an
audited privacy-release RNG backend.
"""

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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from torch import nn

from unitdp.compiler_v2 import (
    CompiledOwnerPoissonRouteV2,
    ContractV2Error,
    EXECUTOR_IMPLEMENTATION_ID_V2,
)
from unitdp.benchmark_registry_v2 import get_benchmark_profile_v2
from unitdp.release_artifacts import assert_public_certificate_redacted
from unitdp.source_bundle_v2 import (
    SourceBundleV2Error,
    execution_source_bundle_sha256_v2,
    execution_source_bundle_v2,
)


PUBLIC_EXECUTION_SCHEMA_V2 = "unitdp.owner_poisson_execution_public.v2.1"
PRIVATE_EXECUTION_SCHEMA_V2 = "unitdp.owner_poisson_execution_private.v2.1"


class OwnerPoissonExecutionV2Error(RuntimeError):
    """Raised when execution cannot preserve the compiled V2 invariants."""


def _sha256_payload(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_state_sha256(model: nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        hasher.update(name.encode("utf-8"))
        hasher.update(str(value.dtype).encode("utf-8"))
        hasher.update(
            json.dumps(
                list(value.shape),
                separators=(",", ":"),
            ).encode("utf-8")
        )
        hasher.update(value.numpy().tobytes())
    return hasher.hexdigest()


def _derive_research_seed(master_seed: int, domain: str) -> int:
    digest = hashlib.sha256(
        b"unitdp-owner-poisson-v2\0"
        + master_seed.to_bytes(8, "big", signed=False)
        + b"\0"
        + domain.encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


class _ResearchStream:
    def __init__(self, seed: int):
        self.generator = np.random.default_rng(seed)

    def random(self) -> float:
        return float(self.generator.random())

    def uniform_array(
        self,
        low: float,
        high: float,
        shape: tuple[int, ...],
    ) -> np.ndarray:
        return self.generator.uniform(low, high, size=shape).astype(np.float32)

    def normal_array(
        self,
        std: float,
        shape: tuple[int, ...],
    ) -> np.ndarray:
        return self.generator.normal(
            loc=0.0,
            scale=std,
            size=shape,
        ).astype(np.float32)


@dataclass(frozen=True)
class _ResearchStreams:
    model_initialization: _ResearchStream
    owner_sampling: _ResearchStream
    within_owner_selection: _ResearchStream
    gaussian_noise: _ResearchStream

    @classmethod
    def from_master_seed(cls, master_seed: int) -> "_ResearchStreams":
        if (
            isinstance(master_seed, bool)
            or not isinstance(master_seed, int)
            or master_seed < 0
            or master_seed >= 2**63
        ):
            raise OwnerPoissonExecutionV2Error(
                "research_seed must be an integer in [0, 2**63)"
            )
        return cls(
            model_initialization=_ResearchStream(
                _derive_research_seed(master_seed, "model_initialization")
            ),
            owner_sampling=_ResearchStream(
                _derive_research_seed(master_seed, "owner_sampling")
            ),
            within_owner_selection=_ResearchStream(
                _derive_research_seed(master_seed, "within_owner_selection")
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
    if array.ndim != 2:
        raise OwnerPoissonExecutionV2Error(
            f"{name} must be a two-dimensional matrix"
        )
    if array.shape[0] <= 0:
        raise OwnerPoissonExecutionV2Error(f"{name} must contain rows")
    if array.shape[1] != input_dim:
        raise OwnerPoissonExecutionV2Error(
            f"{name} must have {input_dim} columns, got {array.shape[1]}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise OwnerPoissonExecutionV2Error(f"{name} must be numeric")
    if not np.isfinite(array).all():
        raise OwnerPoissonExecutionV2Error(
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
        raise OwnerPoissonExecutionV2Error(
            f"{name} must be one-dimensional with {rows} entries"
        )
    if np.issubdtype(labels.dtype, np.bool_) or not np.issubdtype(
        labels.dtype,
        np.integer,
    ):
        raise OwnerPoissonExecutionV2Error(
            f"{name} must use an integer dtype"
        )
    labels = labels.astype(np.int64, copy=False)
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise OwnerPoissonExecutionV2Error(
            f"{name} values must be in [0, {num_classes})"
        )
    return labels


def _build_registered_model(
    *,
    model_id: str,
    input_dim: int,
    num_classes: int,
    stream: _ResearchStream,
) -> nn.Linear:
    expected_model_id = f"linear_{input_dim}x{num_classes}"
    if model_id != expected_model_id:
        raise OwnerPoissonExecutionV2Error(
            f"V2 reference executor requires model_id={expected_model_id}"
        )
    model = nn.Linear(
        input_dim,
        num_classes,
        bias=True,
        device="cpu",
        dtype=torch.float32,
    )
    bound = 1.0 / math.sqrt(float(input_dim))
    with torch.no_grad():
        model.weight.copy_(
            torch.from_numpy(
                stream.uniform_array(
                    -bound,
                    bound,
                    tuple(model.weight.shape),
                )
            )
        )
        model.bias.copy_(
            torch.from_numpy(
                stream.uniform_array(
                    -bound,
                    bound,
                    tuple(model.bias.shape),
                )
            )
        )
    return model


def _owner_gradient(
    model: nn.Module,
    criterion: nn.Module,
    owner_x: torch.Tensor,
    owner_y: torch.Tensor,
) -> tuple[list[torch.Tensor], float]:
    model.zero_grad(set_to_none=True)
    loss = criterion(model(owner_x), owner_y)
    if not torch.isfinite(loss):
        raise OwnerPoissonExecutionV2Error(
            "Owner loss became non-finite"
        )
    loss.backward()
    gradients: list[torch.Tensor] = []
    norm_squared = 0.0
    for parameter in model.parameters():
        gradient = (
            torch.zeros_like(parameter)
            if parameter.grad is None
            else parameter.grad.detach().clone()
        )
        if not torch.isfinite(gradient).all():
            raise OwnerPoissonExecutionV2Error(
                "Owner gradient became non-finite"
            )
        gradients.append(gradient)
        norm_squared += float(
            torch.sum(gradient.to(torch.float64) ** 2).item()
        )
    norm = math.sqrt(norm_squared)
    if not math.isfinite(norm):
        raise OwnerPoissonExecutionV2Error(
            "Owner gradient norm became non-finite"
        )
    return gradients, norm


def _clip_owner_gradients(
    gradients: list[torch.Tensor],
    *,
    norm: float,
    clip_norm: float,
) -> tuple[list[torch.Tensor], float]:
    """Clip with a one-ULP inward target and verify the realized norm."""

    inward_target = float(
        np.nextafter(
            np.float32(clip_norm),
            np.float32(0.0),
        )
    )
    scale = min(1.0, inward_target / max(norm, 1e-30))
    clipped = [gradient * scale for gradient in gradients]

    def realized_norm(values: list[torch.Tensor]) -> float:
        squared = sum(
            float(torch.sum(value.to(torch.float64) ** 2).item())
            for value in values
        )
        return math.sqrt(squared)

    actual_norm = realized_norm(clipped)
    if actual_norm > clip_norm:
        correction = inward_target / actual_norm
        clipped = [value * correction for value in clipped]
        actual_norm = realized_norm(clipped)
    if actual_norm > clip_norm:
        raise OwnerPoissonExecutionV2Error(
            "Numerical owner clipping exceeded the registered clip norm"
        )
    return clipped, actual_norm


def _predict(model: nn.Module, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(
            np.asarray(features, dtype=np.float32)
        )
        return model(tensor).argmax(dim=1).cpu().numpy()


def _predict_probabilities(
    model: nn.Module,
    features: np.ndarray,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(
            np.asarray(features, dtype=np.float32)
        )
        return torch.softmax(model(tensor), dim=1).cpu().numpy()


def _rank_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float | None, float | None]:
    if len(np.unique(labels)) < 2:
        return None, None
    try:
        if probabilities.shape[1] == 2:
            auroc = roc_auc_score(labels, probabilities[:, 1])
            auprc = average_precision_score(labels, probabilities[:, 1])
        else:
            classes = np.arange(probabilities.shape[1])
            one_hot = np.eye(probabilities.shape[1], dtype=int)[labels]
            auroc = roc_auc_score(
                labels,
                probabilities,
                labels=classes,
                multi_class="ovr",
                average="macro",
            )
            auprc = average_precision_score(
                one_hot,
                probabilities,
                average="macro",
            )
    except ValueError:
        return None, None
    return float(auroc), float(auprc)


def _library_versions() -> dict[str, str]:
    result = {
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    for distribution in ("scikit-learn",):
        try:
            result[distribution] = version(distribution)
        except PackageNotFoundError:
            result[distribution] = "not-installed"
    return result


class OwnerPoissonExecutionResultV2:
    """Trained model plus explicitly separated public and private records."""

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
            raise OwnerPoissonExecutionV2Error(
                "The trained model changed after the execution record was built"
            )

    def public_payload(self) -> dict[str, Any]:
        self.assert_model_integrity()
        public = copy.deepcopy(self._public_record)
        reported = str(public.pop("public_execution_sha256"))
        if _sha256_payload(public) != reported:
            raise OwnerPoissonExecutionV2Error(
                "Public execution record hash is inconsistent"
            )
        public["public_execution_sha256"] = reported
        assert_public_certificate_redacted(public)
        return public

    def private_manifest(self) -> dict[str, Any]:
        self.assert_model_integrity()
        private = copy.deepcopy(self._private_record)
        reported = str(private.pop("private_manifest_sha256"))
        if _sha256_payload(private) != reported:
            raise OwnerPoissonExecutionV2Error(
                "Private execution record hash is inconsistent"
            )
        private["private_manifest_sha256"] = reported
        return private


def train_owner_poisson_fixed_v2(
    *,
    route: CompiledOwnerPoissonRouteV2,
    raw_train_x: np.ndarray,
    train_y: np.ndarray,
    raw_test_x: np.ndarray,
    test_y: np.ndarray,
    research_seed: int,
) -> OwnerPoissonExecutionResultV2:
    """Execute exactly the compiled fixed-q, fixed-T research mechanism."""

    if not isinstance(route, CompiledOwnerPoissonRouteV2):
        raise OwnerPoissonExecutionV2Error(
            "V2 executor requires CompiledOwnerPoissonRouteV2"
        )
    try:
        route.assert_private_execution_integrity()
    except (ContractV2Error, RuntimeError) as exc:
        raise OwnerPoissonExecutionV2Error(
            f"Compiled route integrity check failed: {exc}"
        ) from exc
    if not route.execution_ready:
        raise OwnerPoissonExecutionV2Error(
            "Compiled route is not ready for the research V2 executor"
        )
    contract = route.contract
    if contract.execution_profile != "research_benchmark":
        raise OwnerPoissonExecutionV2Error(
            "The reference V2 executor supports research_benchmark only"
        )
    if contract.rng_backend != "research_default":
        raise OwnerPoissonExecutionV2Error(
            "The reference V2 executor requires research_default RNG"
        )
    if contract.implementation_id != EXECUTOR_IMPLEMENTATION_ID_V2:
        raise OwnerPoissonExecutionV2Error(
            "Compiled implementation id does not match the V2 executor"
        )
    try:
        execution_source_bundle_before = dict(execution_source_bundle_v2())
        execution_source_bundle_sha256 = (
            execution_source_bundle_sha256_v2(
                execution_source_bundle_before
            )
        )
    except SourceBundleV2Error as exc:
        raise OwnerPoissonExecutionV2Error(
            f"Could not bind the registered execution source bundle: {exc}"
        ) from exc
    if (
        execution_source_bundle_sha256
        != route.execution_source_bundle_sha256
    ):
        raise OwnerPoissonExecutionV2Error(
            "Execution source bundle differs from the compiled route"
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
        raise OwnerPoissonExecutionV2Error(
            "raw_train_x row count must equal the compiled source mapping"
        )
    train_labels = _validate_labels(
        train_y,
        name="train_y",
        rows=train_raw.shape[0],
        num_classes=contract.num_classes,
    )
    profile = get_benchmark_profile_v2(contract.benchmark_profile_id)
    label_to_index = dict(profile.mapping_label_to_index)
    for record in route.source_mapping.records:
        if record.label not in label_to_index:
            raise OwnerPoissonExecutionV2Error(
                f"Mapping label {record.label!r} is not registered"
            )
        if label_to_index[record.label] != int(
            train_labels[record.row_index]
        ):
            raise OwnerPoissonExecutionV2Error(
                "Mapping labels do not match train_y at the registered row indices"
            )
    test_labels = _validate_labels(
        test_y,
        name="test_y",
        rows=test_raw.shape[0],
        num_classes=contract.num_classes,
    )
    train_features = route.preprocessor.transform(train_raw)
    test_features = route.preprocessor.transform(test_raw)

    streams = _ResearchStreams.from_master_seed(research_seed)
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
    if not owners:
        raise OwnerPoissonExecutionV2Error("Compiled route has no owners")

    step_diagnostics: list[dict[str, Any]] = []
    for step in range(contract.total_steps):
        sampled_owners = [
            owner
            for owner in owners
            if streams.owner_sampling.random()
            < contract.owner_sample_rate
        ]
        summed_gradients = [
            torch.zeros_like(parameter) for parameter in parameters
        ]
        owner_norms: list[float] = []
        clipped_norms: list[float] = []
        owner_vectors = 0

        for owner in sampled_owners:
            records = route.policy.sample_for_step(
                selected_by_owner[owner],
                streams.within_owner_selection.generator,
            )
            if not records:
                raise OwnerPoissonExecutionV2Error(
                    "Registered within-owner selector returned no records"
                )
            indices = [record.row_index for record in records]
            owner_x = torch.from_numpy(train_features[indices])
            owner_y = torch.from_numpy(train_labels[indices])
            gradients, norm = _owner_gradient(
                model,
                criterion,
                owner_x,
                owner_y,
            )
            clipped_gradients, realized_clipped_norm = (
                _clip_owner_gradients(
                    gradients,
                    norm=norm,
                    clip_norm=contract.clip_norm,
                )
            )
            for index, gradient in enumerate(clipped_gradients):
                summed_gradients[index].add_(gradient)
            owner_vectors += 1
            owner_norms.append(norm)
            clipped_norms.append(realized_clipped_norm)

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
            raise OwnerPoissonExecutionV2Error(
                f"Model became non-finite at step {step}"
            )
        step_diagnostics.append(
            {
                "step": step,
                "sampled_owner_count": len(sampled_owners),
                "owner_vectors_computed": owner_vectors,
                "empty_sample": len(sampled_owners) == 0,
                "noise_applied": noise_tensors == len(parameters),
                "noise_parameter_tensors": noise_tensors,
                "optimizer_step_applied": True,
                "update_denominator": contract.update_denominator,
                "max_unclipped_owner_norm": (
                    max(owner_norms) if owner_norms else 0.0
                ),
                "max_clipped_owner_norm": (
                    max(clipped_norms) if clipped_norms else 0.0
                ),
            }
        )

    try:
        route.assert_private_execution_integrity()
    except (ContractV2Error, RuntimeError) as exc:
        raise OwnerPoissonExecutionV2Error(
            f"Compiled route changed during execution: {exc}"
        ) from exc
    try:
        execution_source_bundle_after = dict(execution_source_bundle_v2())
    except SourceBundleV2Error as exc:
        raise OwnerPoissonExecutionV2Error(
            f"Could not recheck the execution source bundle: {exc}"
        ) from exc
    if execution_source_bundle_after != execution_source_bundle_before:
        raise OwnerPoissonExecutionV2Error(
            "Execution source bundle changed during execution"
        )

    predictions = _predict(model, test_features)
    probabilities = _predict_probabilities(model, test_features)
    auroc, auprc = _rank_metrics(test_labels, probabilities)
    final_model_sha256 = _model_state_sha256(model)
    library_versions = _library_versions()
    public_record: dict[str, Any] = {
        "schema_version": PUBLIC_EXECUTION_SCHEMA_V2,
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
            "public_contract_sha256": contract.public_contract_sha256,
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
            "owner_sample_rate": contract.owner_sample_rate,
            "total_steps": contract.total_steps,
            "clip_norm": contract.clip_norm,
            "noise_multiplier": contract.noise_multiplier,
            "update_denominator": contract.update_denominator,
            "empty_step_behavior": contract.empty_step_behavior,
            "per_owner_contribution": contract.per_owner_contribution,
            "owner_aggregation": contract.owner_aggregation,
        },
        "accountant": {
            "id": contract.accountant_id,
            "epsilon": max(
                route.accountant_epsilon_opacus,
                route.accountant_epsilon_dp_accounting,
            ),
            "delta": contract.delta,
            "rdp_orders": list(route.registry_entry.rdp_orders),
            "opacus_epsilon": route.accountant_epsilon_opacus,
            "dp_accounting_epsilon": (
                route.accountant_epsilon_dp_accounting
            ),
        },
        "execution_checks": {
            "compiled_object_only": True,
            "compiled_integrity_before_and_after": True,
            "fixed_q": True,
            "fixed_total_steps": True,
            "fixed_update_denominator": True,
            "bernoulli_owner_sampling": True,
            "one_clipped_vector_per_sampled_owner": True,
            "Gaussian_noise_every_step": True,
            "optimizer_update_every_step": True,
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
            "optimizer_parameters": {
                "momentum": 0.0,
                "dampening": 0.0,
                "weight_decay": 0.0,
                "nesterov": False,
                "maximize": False,
                "foreach": False,
                "differentiable": False,
                "fused": False,
            },
            "loss": contract.loss,
        },
        "public_data_bindings": {
            "dataset_id": contract.dataset_id,
            "preprocessing": contract.preprocessing_binding,
            "protocol_id": contract.public_protocol["protocol_id"],
            "data_preparation_implementation_id": contract.public_protocol[
                "data_preparation_implementation_id"
            ],
            "evaluation_scope": contract.public_protocol[
                "evaluation_scope"
            ],
        },
        "public_evaluation": {
            "scope": contract.public_protocol["evaluation_scope"],
            "accuracy": float(accuracy_score(test_labels, predictions)),
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
            "state_sha256": final_model_sha256,
            "artifact_status": "not_serialized_by_executor",
        },
        "implementation": {
            "source_bundle_schema": "unitdp.execution_source_bundle.v2",
            "source_bundle": execution_source_bundle_before,
            "source_bundle_sha256": execution_source_bundle_sha256,
            "library_versions": library_versions,
        },
    }
    public_record["public_execution_sha256"] = _sha256_payload(public_record)
    assert_public_certificate_redacted(public_record)

    private_record: dict[str, Any] = {
        "schema_version": PRIVATE_EXECUTION_SCHEMA_V2,
        "visibility": "private",
        "handling": (
            "do_not_publish_contains_seed_private_mapping_commitments_"
            "and_realized_sampling_statistics"
        ),
        "public_execution_sha256": public_record[
            "public_execution_sha256"
        ],
        "private_execution_binding_sha256": (
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
        "research_seed": research_seed,
        "initial_model_sha256": initial_model_sha256,
        "final_model_sha256": final_model_sha256,
        "observed_source_owner_count": len(
            route.source_mapping.owners()
        ),
        "observed_selected_owner_count": len(owners),
        "observed_source_window_count": len(
            route.source_mapping.records
        ),
        "observed_selected_window_count": len(
            route.selected_mapping.records
        ),
        "step_diagnostics": step_diagnostics,
    }
    private_record["private_manifest_sha256"] = _sha256_payload(
        private_record
    )
    return OwnerPoissonExecutionResultV2(
        model=model,
        public_record=public_record,
        private_record=private_record,
    )
