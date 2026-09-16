"""Independent, test-only replay of registered linear owner routes.

This module intentionally imports no ``unitdp`` implementation code.  It
reconstructs the public preprocessing, domain-separated random streams,
Bernoulli owner sampling, within-owner record selection, per-owner gradient,
clipping, Gaussian noising, and fixed-denominator update from primitive NumPy
and PyTorch operations.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ReferenceRecord:
    """Minimal test-side record needed to replay owner-local selection."""

    owner_id: str
    window_id: str
    start: int
    end: int
    row_index: int


@dataclass(frozen=True)
class ReferenceTrace:
    model: nn.Linear
    sampled_owners: tuple[tuple[str, ...], ...]
    selected_rows_by_owner: tuple[
        tuple[tuple[str, tuple[int, ...]], ...],
        ...,
    ]
    sampled_owner_counts: tuple[int, ...]
    owner_vectors_computed: tuple[int, ...]
    max_unclipped_norms: tuple[float, ...]
    max_clipped_norms: tuple[float, ...]

    @property
    def sampled_steps(self) -> tuple[bool, ...]:
        """Compatibility view for the original one-owner correspondence."""

        return tuple(count > 0 for count in self.sampled_owner_counts)

    @property
    def unclipped_norms(self) -> tuple[float, ...]:
        """Compatibility alias for the original one-owner correspondence."""

        return self.max_unclipped_norms

    @property
    def clipped_norms(self) -> tuple[float, ...]:
        """Compatibility alias for the original one-owner correspondence."""

        return self.max_clipped_norms


def _domain_seed(master_seed: int, domain: str) -> int:
    digest = hashlib.sha256(
        b"unitdp-owner-poisson-v2\0"
        + master_seed.to_bytes(8, "big", signed=False)
        + b"\0"
        + domain.encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _load_and_transform(
    raw_train_x: np.ndarray,
    artifact_path: Path,
) -> np.ndarray:
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    mean = np.asarray(artifact["mean"], dtype=np.float64)
    scale = np.asarray(artifact["scale"], dtype=np.float64)
    transformed = np.asarray(raw_train_x).astype(np.float32, copy=True)
    transformed -= mean
    transformed /= scale
    return transformed


def _realized_norm(gradients: list[torch.Tensor]) -> float:
    squared = sum(
        float(torch.sum(value.to(torch.float64) ** 2).item())
        for value in gradients
    )
    return math.sqrt(squared)


def _initialize_linear_model(
    *,
    master_seed: int,
    input_dim: int,
    num_classes: int,
) -> nn.Linear:
    initialization_rng = np.random.default_rng(
        _domain_seed(master_seed, "model_initialization")
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
                initialization_rng.uniform(
                    -bound,
                    bound,
                    size=tuple(model.weight.shape),
                ).astype(np.float32)
            )
        )
        model.bias.copy_(
            torch.from_numpy(
                initialization_rng.uniform(
                    -bound,
                    bound,
                    size=tuple(model.bias.shape),
                ).astype(np.float32)
            )
        )
    return model


def replay_owner_poisson_linear_trace(
    *,
    raw_train_x: np.ndarray,
    train_y: np.ndarray,
    records: tuple[ReferenceRecord, ...],
    preprocessing_artifact_path: Path,
    master_seed: int,
    input_dim: int,
    num_classes: int,
    owner_sample_rate: float,
    total_steps: int,
    clip_norm: float,
    noise_multiplier: float,
    update_denominator: float,
    learning_rate: float,
    windows_per_owner_per_step: int,
    class_weights: tuple[float, ...] | None,
) -> ReferenceTrace:
    """Replay a registered multi-owner route without production helpers."""

    features = _load_and_transform(
        raw_train_x,
        preprocessing_artifact_path,
    )
    labels = np.asarray(train_y, dtype=np.int64)
    grouped: dict[str, list[ReferenceRecord]] = defaultdict(list)
    for record in records:
        grouped[record.owner_id].append(record)
    records_by_owner = {
        owner: sorted(
            owner_records,
            key=lambda record: (
                record.start,
                record.end,
                record.window_id,
            ),
        )
        for owner, owner_records in grouped.items()
    }
    owners = sorted(records_by_owner)
    if not owners:
        raise ValueError("Reference route requires at least one owner")

    sampling_rng = np.random.default_rng(
        _domain_seed(master_seed, "owner_sampling")
    )
    within_owner_rng = np.random.default_rng(
        _domain_seed(master_seed, "within_owner_selection")
    )
    noise_rng = np.random.default_rng(
        _domain_seed(master_seed, "gaussian_noise")
    )

    model = _initialize_linear_model(
        master_seed=master_seed,
        input_dim=input_dim,
        num_classes=num_classes,
    )
    weight_tensor = (
        torch.tensor(class_weights, dtype=torch.float32)
        if class_weights is not None
        else None
    )
    criterion = nn.CrossEntropyLoss(
        weight=weight_tensor,
        reduction="mean",
    )
    parameters = list(model.parameters())
    sampled_owners_by_step: list[tuple[str, ...]] = []
    selected_rows_by_step: list[
        tuple[tuple[str, tuple[int, ...]], ...]
    ] = []
    sampled_owner_counts: list[int] = []
    owner_vectors_computed: list[int] = []
    max_unclipped_norms: list[float] = []
    max_clipped_norms: list[float] = []
    inward_target = float(
        np.nextafter(
            np.float32(clip_norm),
            np.float32(0.0),
        )
    )

    for _ in range(total_steps):
        sampled_owners = tuple(
            owner
            for owner in owners
            if sampling_rng.random() < owner_sample_rate
        )
        sampled_owners_by_step.append(sampled_owners)
        sampled_owner_counts.append(len(sampled_owners))
        gradient_sums = [
            torch.zeros_like(parameter) for parameter in parameters
        ]
        step_unclipped_norms: list[float] = []
        step_clipped_norms: list[float] = []
        step_selected_rows: list[tuple[str, tuple[int, ...]]] = []

        for owner in sampled_owners:
            owner_records = records_by_owner[owner]
            if windows_per_owner_per_step >= len(owner_records):
                selected_records = list(owner_records)
            else:
                positions = within_owner_rng.choice(
                    len(owner_records),
                    size=windows_per_owner_per_step,
                    replace=False,
                )
                selected_records = [
                    owner_records[int(position)]
                    for position in sorted(positions.tolist())
                ]
            indices = tuple(
                record.row_index for record in selected_records
            )
            step_selected_rows.append((owner, indices))
            owner_x = torch.from_numpy(features[list(indices)])
            owner_y = torch.from_numpy(labels[list(indices)])
            model.zero_grad(set_to_none=True)
            criterion(model(owner_x), owner_y).backward()
            gradients = [
                (
                    torch.zeros_like(parameter)
                    if parameter.grad is None
                    else parameter.grad.detach().clone()
                )
                for parameter in parameters
            ]
            norm = _realized_norm(gradients)
            scale = min(1.0, inward_target / max(norm, 1e-30))
            clipped = [gradient * scale for gradient in gradients]
            clipped_norm = _realized_norm(clipped)
            if clipped_norm > clip_norm:
                correction = inward_target / clipped_norm
                clipped = [
                    gradient * correction for gradient in clipped
                ]
                clipped_norm = _realized_norm(clipped)
            for target, gradient in zip(gradient_sums, clipped):
                target.add_(gradient)
            step_unclipped_norms.append(norm)
            step_clipped_norms.append(clipped_norm)

        selected_rows_by_step.append(tuple(step_selected_rows))
        owner_vectors_computed.append(len(step_selected_rows))
        max_unclipped_norms.append(
            max(step_unclipped_norms)
            if step_unclipped_norms
            else 0.0
        )
        max_clipped_norms.append(
            max(step_clipped_norms)
            if step_clipped_norms
            else 0.0
        )

        with torch.no_grad():
            for parameter, gradient_sum in zip(
                parameters,
                gradient_sums,
            ):
                noise = torch.from_numpy(
                    noise_rng.normal(
                        loc=0.0,
                        scale=noise_multiplier * clip_norm,
                        size=tuple(parameter.shape),
                    ).astype(np.float32)
                )
                gradient = (
                    gradient_sum + noise
                ) / update_denominator
                parameter.add_(gradient, alpha=-learning_rate)

    return ReferenceTrace(
        model=model,
        sampled_owners=tuple(sampled_owners_by_step),
        selected_rows_by_owner=tuple(selected_rows_by_step),
        sampled_owner_counts=tuple(sampled_owner_counts),
        owner_vectors_computed=tuple(owner_vectors_computed),
        max_unclipped_norms=tuple(max_unclipped_norms),
        max_clipped_norms=tuple(max_clipped_norms),
    )


def replay_single_owner_linear_trace(
    *,
    raw_train_x: np.ndarray,
    train_label: int,
    preprocessing_artifact_path: Path,
    master_seed: int,
    input_dim: int,
    num_classes: int,
    owner_sample_rate: float,
    total_steps: int,
    clip_norm: float,
    noise_multiplier: float,
    update_denominator: float,
    learning_rate: float,
) -> ReferenceTrace:
    """Compatibility wrapper for the original one-owner, one-record trace."""

    return replay_owner_poisson_linear_trace(
        raw_train_x=raw_train_x,
        train_y=np.asarray([train_label], dtype=np.int64),
        records=(
            ReferenceRecord(
                owner_id="owner",
                window_id="window",
                start=0,
                end=1,
                row_index=0,
            ),
        ),
        preprocessing_artifact_path=preprocessing_artifact_path,
        master_seed=master_seed,
        input_dim=input_dim,
        num_classes=num_classes,
        owner_sample_rate=owner_sample_rate,
        total_steps=total_steps,
        clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
        update_denominator=update_denominator,
        learning_rate=learning_rate,
        windows_per_owner_per_step=1,
        class_weights=None,
    )
