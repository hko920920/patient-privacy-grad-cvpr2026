"""Owner-Window Aligned DP-SGD training loop."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
import numpy as np
import torch
from opacus.accountants import RDPAccountant
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch import nn

from unitdp.accountant import (
    RDPConfig,
    calibrate_noise,
    calibrate_noise_srswor_replace_one,
    epsilon_for_srswor_replace_one_noise,
)
from unitdp.compiler import CompiledRoute
from unitdp.contract import OWNER_UNITS
from unitdp.rng import make_random_source


ROUTE_IMPLEMENTATION_ID = "unitdp.owa_dpsgd.train_owa_dpsgd:v1"
LOADER_BINDING = "internal_owner_sampler_no_external_dataloader"


def _sha256_payload(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _update_trace_hash(hasher: "hashlib._Hash", payload: object) -> None:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    hasher.update(encoded)
    hasher.update(b"\n")


@dataclass(frozen=True)
class OwaDpsgdConfig:
    epochs: int = 20
    owner_batch_size: int = 8
    learning_rate: float = 0.05
    max_grad_norm: float = 1.0
    target_epsilon: float = 8.0
    target_delta: float = 1e-5
    noise_multiplier: float | None = None
    seed: int = 0
    device: str = "cpu"
    class_weights: list[float] | None = None
    owner_aggregation: str = "mean"
    fixed_window_normalizer: int | None = None
    owner_sampling_scheme: str = "bernoulli"
    rng_backend: str = "research_default"

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.owner_batch_size <= 0:
            raise ValueError("owner_batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.target_epsilon <= 0:
            raise ValueError("target_epsilon must be positive")
        if self.target_delta <= 0 or self.target_delta >= 1:
            raise ValueError("target_delta must be in (0, 1)")
        if self.noise_multiplier is not None and self.noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be positive")
        if self.owner_aggregation not in {"mean", "sum", "fixed_normalizer"}:
            raise ValueError("owner_aggregation must be mean, sum, or fixed_normalizer")
        if self.owner_aggregation == "fixed_normalizer":
            if self.fixed_window_normalizer is None or self.fixed_window_normalizer <= 0:
                raise ValueError(
                    "fixed_window_normalizer must be positive for fixed_normalizer aggregation"
                )
        if self.owner_sampling_scheme not in {
            "bernoulli",
            "fixed_without_replacement",
            "shuffled_epoch",
        }:
            raise ValueError(
                "owner_sampling_scheme must be bernoulli, fixed_without_replacement, or shuffled_epoch"
            )
        if self.rng_backend not in {"research_default", "system_csprng"}:
            raise ValueError("rng_backend must be research_default or system_csprng")


@dataclass(frozen=True)
class OwaDpsgdResult:
    epsilon: float
    delta: float
    noise_multiplier: float
    owner_sample_rate: float
    accountant_sample_rate: float
    steps: int
    sampling_scheme: str
    accountant_backend: str
    accountant_unit: str
    clipping_unit: str
    noising_unit: str
    per_step_contribution_bound: str
    route_implementation_id: str
    loader_binding: str
    public_schedule_sha256: str
    sampler_trace_sha256: str
    secure_rng: bool
    rng_backend: str
    rng_security_mode: str
    rng_caveat: str
    model_artifact_sha256: str
    training_code_sha256: str
    library_versions: dict[str, str]
    sampled_owner_count_mean: float
    sampled_owner_count_std: float
    sampled_owner_count_min: int
    sampled_owner_count_max: int
    actual_owner_sample_rate_mean: float
    empty_steps: int
    train_accuracy: float
    test_accuracy: float
    test_macro_f1: float
    test_auroc: float | None
    test_auprc: float | None
    selected_owners: int
    selected_windows: int
    config: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _owner_gradient(
    model: nn.Module,
    mean_criterion: nn.Module,
    sum_criterion: nn.Module,
    owner_x: torch.Tensor,
    owner_y: torch.Tensor,
    aggregation: str,
    fixed_normalizer: int | None,
) -> tuple[list[torch.Tensor], float]:
    model.zero_grad(set_to_none=True)
    logits = model(owner_x)
    if aggregation == "mean":
        loss = mean_criterion(logits, owner_y)
    elif aggregation == "sum":
        loss = sum_criterion(logits, owner_y)
    elif aggregation == "fixed_normalizer":
        normalizer = fixed_normalizer if fixed_normalizer is not None else owner_x.shape[0]
        loss = sum_criterion(logits, owner_y) / max(1, normalizer)
    else:
        raise ValueError(f"Unsupported owner aggregation: {aggregation}")
    loss.backward()
    grads: list[torch.Tensor] = []
    total_norm_sq = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            grad = torch.zeros_like(parameter)
        else:
            grad = parameter.grad.detach().clone()
        grads.append(grad)
        total_norm_sq += float(torch.sum(grad * grad).detach().cpu())
    return grads, math.sqrt(total_norm_sq)


def _model_state_sha256(model: nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        cpu_tensor = tensor.detach().cpu().contiguous()
        hasher.update(name.encode("utf-8"))
        hasher.update(str(cpu_tensor.dtype).encode("utf-8"))
        hasher.update(json.dumps(list(cpu_tensor.shape)).encode("utf-8"))
        hasher.update(cpu_tensor.numpy().tobytes())
    return hasher.hexdigest()


def _predict(model: nn.Module, x: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(x, dtype=torch.float32, device=device)
        return model(tensor).argmax(dim=1).detach().cpu().numpy()


def _predict_proba(model: nn.Module, x: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(x, dtype=torch.float32, device=device)
        logits = model(tensor)
        return torch.softmax(logits, dim=1).detach().cpu().numpy()


def _rank_metrics(y_true: np.ndarray, proba: np.ndarray) -> tuple[float | None, float | None]:
    """Return AUROC/AUPRC when the test labels make the metric defined."""

    classes = np.unique(y_true)
    if len(classes) < 2:
        return None, None
    try:
        if proba.shape[1] == 2:
            auroc = roc_auc_score(y_true, proba[:, 1])
            auprc = average_precision_score(y_true, proba[:, 1])
        else:
            labels = np.arange(proba.shape[1])
            one_hot = np.eye(proba.shape[1], dtype=int)[y_true]
            auroc = roc_auc_score(
                y_true,
                proba,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
            auprc = average_precision_score(one_hot, proba, average="macro")
    except ValueError:
        return None, None
    return float(auroc), float(auprc)


def _validate_owner_route_for_training(route: CompiledRoute) -> None:
    contract = route.contract
    if contract.privacy_unit not in OWNER_UNITS:
        raise ValueError(f"OWA-DPSGD requires owner-level privacy_unit, got {contract.privacy_unit!r}")
    for key in ["sampling_unit", "clipping_unit", "noising_unit"]:
        value = contract.mechanism.get(key)
        if value not in OWNER_UNITS:
            raise ValueError(f"OWA-DPSGD requires mechanism.{key}=owner, got {value!r}")
    accountant_unit = contract.accountant.get("unit")
    if accountant_unit not in OWNER_UNITS:
        raise ValueError(f"OWA-DPSGD requires accountant.unit=owner, got {accountant_unit!r}")
    multi_owner_windows = int(route.selected_mapping.stats()["multi_owner_windows"])
    if multi_owner_windows:
        raise ValueError(
            "OWA-DPSGD requires single-attribution selected windows; "
            f"found {multi_owner_windows} multi-owner windows"
        )


def _count_stats(counts: list[int], denominator: int) -> dict[str, float | int]:
    values = np.asarray(counts, dtype=float)
    if values.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0,
            "max": 0,
            "rate_mean": 0.0,
        }
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": int(values.min()),
        "max": int(values.max()),
        "rate_mean": float(values.mean() / max(1, denominator)),
    }


def train_owa_dpsgd(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    route: CompiledRoute,
    config: OwaDpsgdConfig,
) -> OwaDpsgdResult:
    """Train with owner sampling and owner-window aggregation before clipping."""

    config.validate()
    _validate_owner_route_for_training(route)
    torch.manual_seed(config.seed)
    rng = make_random_source(config.rng_backend, config.seed)
    rng_metadata = rng.metadata()
    device = config.device
    model.to(device)

    selected_by_owner = route.selected_mapping.by_owner()
    owners = sorted(selected_by_owner)
    if not owners:
        raise ValueError("No selected owners")
    owner_batch_size = min(config.owner_batch_size, len(owners))
    owner_sample_rate = min(1.0, owner_batch_size / len(owners))
    steps_per_epoch = math.ceil(len(owners) / owner_batch_size)
    total_steps = config.epochs * steps_per_epoch
    shuffled_batches: list[list[str]] | None = None
    if config.owner_sampling_scheme == "shuffled_epoch":
        shuffled_batches = []
        for _epoch in range(config.epochs):
            shuffled: list[object] = list(owners)
            rng.shuffle(shuffled)
            for start in range(0, len(shuffled), owner_batch_size):
                shuffled_batches.append([str(owner) for owner in shuffled[start : start + owner_batch_size]])
        total_steps = len(shuffled_batches)

    rdp_config = RDPConfig(
        delta=config.target_delta,
        target_epsilon=config.target_epsilon,
        noise_multiplier=config.noise_multiplier,
    )
    rdp_config.validate()
    if config.noise_multiplier is not None:
        noise_multiplier = config.noise_multiplier
    elif config.owner_sampling_scheme in {"fixed_without_replacement", "shuffled_epoch"}:
        noise_multiplier = calibrate_noise_srswor_replace_one(
            target_epsilon=config.target_epsilon,
            source_dataset_size=len(owners),
            sample_size=owner_batch_size,
            steps=total_steps,
            delta=config.target_delta,
            alphas=rdp_config.alphas,
        )
    else:
        noise_multiplier = calibrate_noise(
            target_epsilon=config.target_epsilon,
            sample_rate=owner_sample_rate,
            steps=total_steps,
            delta=config.target_delta,
            alphas=rdp_config.alphas,
        )

    public_schedule = {
        "schedule_mode": route.contract.schedule_mode,
        "owner_sampling_scheme": config.owner_sampling_scheme,
        "owner_sample_rate": float(owner_sample_rate),
        "owner_batch_size": int(owner_batch_size),
        "selected_owners": int(len(owners)),
        "epochs": int(config.epochs),
        "steps": int(total_steps),
        "max_grad_norm": float(config.max_grad_norm),
        "target_epsilon": float(config.target_epsilon),
        "target_delta": float(config.target_delta),
        "noise_multiplier": float(noise_multiplier),
        "owner_aggregation": config.owner_aggregation,
        "fixed_window_normalizer": config.fixed_window_normalizer,
        "max_windows_per_owner": route.policy.max_windows_per_owner,
        "windows_per_owner_per_step": route.policy.windows_per_owner_per_step,
        "policy_type": route.policy.policy_type,
        "rng_backend": config.rng_backend,
    }
    public_schedule_sha256 = _sha256_payload(public_schedule)

    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    class_weight_tensor = None
    if config.class_weights is not None:
        class_weight_tensor = torch.tensor(config.class_weights, dtype=torch.float32, device=device)
    mean_criterion = nn.CrossEntropyLoss(weight=class_weight_tensor, reduction="mean")
    sum_criterion = nn.CrossEntropyLoss(weight=class_weight_tensor, reduction="sum")
    parameters = list(model.parameters())
    accountant = RDPAccountant()
    sampled_owner_counts: list[int] = []
    empty_steps = 0
    sampler_trace_hasher = hashlib.sha256()

    for _step in range(total_steps):
        if config.owner_sampling_scheme == "fixed_without_replacement":
            selected_owners = [str(owner) for owner in rng.choice(owners, size=owner_batch_size, replace=False).tolist()]
        elif config.owner_sampling_scheme == "shuffled_epoch":
            assert shuffled_batches is not None
            selected_owners = list(shuffled_batches[_step])
        else:
            selected_owners = [owner for owner in owners if rng.random() < owner_sample_rate]
        sampled_owner_counts.append(len(selected_owners))
        if config.owner_sampling_scheme == "bernoulli":
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=owner_sample_rate)
        owner_records_by_owner: dict[str, list[object]] = {}
        for owner in selected_owners:
            owner_records_by_owner[owner] = route.policy.sample_for_step(selected_by_owner[owner], rng)
        _update_trace_hash(
            sampler_trace_hasher,
            {
                "step": int(_step),
                "selected_owners": list(selected_owners),
                "selected_windows_by_owner": [
                    {
                        "owner": owner,
                        "windows": [
                            {
                                "window_id": record.window_id,
                                "row_index": int(record.row_index),
                            }
                            for record in owner_records_by_owner[owner]
                        ],
                    }
                    for owner in selected_owners
                ],
            },
        )
        if not selected_owners:
            empty_steps += 1
            continue

        summed = [torch.zeros_like(parameter, device=device) for parameter in parameters]
        for owner in selected_owners:
            owner_records = owner_records_by_owner[owner]
            if not owner_records:
                continue
            indices = [record.row_index for record in owner_records]
            owner_x = torch.tensor(train_x[indices], dtype=torch.float32, device=device)
            owner_y = torch.tensor(train_y[indices], dtype=torch.long, device=device)
            grads, norm = _owner_gradient(
                model,
                mean_criterion,
                sum_criterion,
                owner_x,
                owner_y,
                config.owner_aggregation,
                config.fixed_window_normalizer,
            )
            scale = min(1.0, config.max_grad_norm / max(norm, 1e-12))
            for idx, grad in enumerate(grads):
                summed[idx] += grad.to(device) * scale

        for parameter, grad_sum in zip(parameters, summed):
            noise = rng.normal_tensor(
                grad_sum.shape,
                std=noise_multiplier * config.max_grad_norm,
                device=device,
            )
            parameter.grad = (grad_sum + noise) / owner_batch_size
        optimizer.step()

    if config.owner_sampling_scheme in {"fixed_without_replacement", "shuffled_epoch"}:
        epsilon = epsilon_for_srswor_replace_one_noise(
            noise_multiplier=noise_multiplier,
            source_dataset_size=len(owners),
            sample_size=owner_batch_size,
            steps=total_steps,
            delta=config.target_delta,
            alphas=rdp_config.alphas,
        )
        if config.owner_sampling_scheme == "fixed_without_replacement":
            sampling_scheme = "fixed_without_replacement_owner"
            accountant_backend = "dp_accounting_rdp_srswor_replace_one"
        else:
            sampling_scheme = "shuffled_epoch_owner_conservative_srswor"
            accountant_backend = "dp_accounting_rdp_srswor_replace_one_conservative_epoch"
    else:
        epsilon = float(accountant.get_epsilon(delta=config.target_delta, alphas=rdp_config.alphas))
        sampling_scheme = "independent_bernoulli_owner"
        accountant_backend = "rdp"
    sampler_trace_sha256 = _sha256_payload(
        {
            "sampling_scheme": sampling_scheme,
            "selected_owners": int(len(owners)),
            "owner_batch_size": int(owner_batch_size),
            "steps": int(total_steps),
            "trace_digest": sampler_trace_hasher.hexdigest(),
        }
    )
    train_pred = _predict(model, train_x, device)
    test_pred = _predict(model, test_x, device)
    test_proba = _predict_proba(model, test_x, device)
    test_auroc, test_auprc = _rank_metrics(test_y, test_proba)
    sample_stats = _count_stats(sampled_owner_counts, len(owners))
    library_versions = {
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    training_code_sha256 = _sha256_payload(
        {
            "route_implementation_id": ROUTE_IMPLEMENTATION_ID,
            "library_versions": library_versions,
            "public_schedule_sha256": public_schedule_sha256,
        }
    )

    return OwaDpsgdResult(
        epsilon=epsilon,
        delta=float(config.target_delta),
        noise_multiplier=float(noise_multiplier),
        owner_sample_rate=float(owner_sample_rate),
        accountant_sample_rate=float(owner_sample_rate),
        steps=int(total_steps),
        sampling_scheme=sampling_scheme,
        accountant_backend=accountant_backend,
        accountant_unit="owner",
        clipping_unit="owner",
        noising_unit="owner",
        per_step_contribution_bound="one_clipped_vector_per_sampled_owner",
        route_implementation_id=ROUTE_IMPLEMENTATION_ID,
        loader_binding=LOADER_BINDING,
        public_schedule_sha256=public_schedule_sha256,
        sampler_trace_sha256=sampler_trace_sha256,
        secure_rng=rng_metadata.secure_rng,
        rng_backend=rng_metadata.rng_backend,
        rng_security_mode=rng_metadata.rng_security_mode,
        rng_caveat=rng_metadata.rng_caveat,
        model_artifact_sha256=_model_state_sha256(model),
        training_code_sha256=training_code_sha256,
        library_versions=library_versions,
        sampled_owner_count_mean=float(sample_stats["mean"]),
        sampled_owner_count_std=float(sample_stats["std"]),
        sampled_owner_count_min=int(sample_stats["min"]),
        sampled_owner_count_max=int(sample_stats["max"]),
        actual_owner_sample_rate_mean=float(sample_stats["rate_mean"]),
        empty_steps=int(empty_steps),
        train_accuracy=float(accuracy_score(train_y, train_pred)),
        test_accuracy=float(accuracy_score(test_y, test_pred)),
        test_macro_f1=float(f1_score(test_y, test_pred, average="macro", zero_division=0)),
        test_auroc=test_auroc,
        test_auprc=test_auprc,
        selected_owners=len(owners),
        selected_windows=len(route.selected_mapping.records),
        config=asdict(config),
    )


def train_owner_poisson_fixed_v2(**kwargs):
    """Public implementation-id entry point for the strict V2 executor."""

    from unitdp.owner_poisson_v2 import (
        train_owner_poisson_fixed_v2 as implementation,
    )

    return implementation(**kwargs)
