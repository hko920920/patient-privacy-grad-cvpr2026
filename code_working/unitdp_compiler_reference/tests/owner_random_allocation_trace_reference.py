"""Production-import-free replay of the random-allocation V4 executor."""

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
    owner_id: str
    window_id: str
    start: int
    end: int
    row_index: int


@dataclass(frozen=True)
class ReferenceAllocationTrace:
    model: nn.Linear
    assigned_owner_positions: tuple[tuple[int, ...], ...]
    local_steps_by_epoch_and_owner: tuple[
        tuple[tuple[int, ...], ...],
        ...,
    ]
    selected_rows_by_position: tuple[
        tuple[tuple[int, tuple[int, ...]], ...],
        ...,
    ]
    max_unclipped_norms: tuple[float, ...]
    max_clipped_norms: tuple[float, ...]


def _domain_seed(master_seed: int, domain: str) -> int:
    digest = hashlib.sha256(
        b"unitdp-owner-random-allocation-v4\0"
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
    rng = np.random.default_rng(
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
                rng.uniform(
                    -bound,
                    bound,
                    size=tuple(model.weight.shape),
                ).astype(np.float32)
            )
        )
        model.bias.copy_(
            torch.from_numpy(
                rng.uniform(
                    -bound,
                    bound,
                    size=tuple(model.bias.shape),
                ).astype(np.float32)
            )
        )
    return model


def _schedule(
    *,
    master_seed: int,
    source_dataset_size: int,
    num_steps: int,
    num_selected_steps_per_owner: int,
    num_epochs: int,
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[tuple[tuple[int, ...], ...], ...],
]:
    rng = np.random.default_rng(
        _domain_seed(master_seed, "owner_step_allocation")
    )
    by_step: list[list[int]] = [
        [] for _ in range(num_steps * num_epochs)
    ]
    by_owner: list[tuple[tuple[int, ...], ...]] = []
    for epoch in range(num_epochs):
        rows: list[tuple[int, ...]] = []
        for owner_position in range(source_dataset_size):
            selected = tuple(
                sorted(
                    int(value)
                    for value in rng.choice(
                        num_steps,
                        size=num_selected_steps_per_owner,
                        replace=False,
                    ).tolist()
                )
            )
            rows.append(selected)
            for local_step in selected:
                by_step[epoch * num_steps + local_step].append(
                    owner_position
                )
        by_owner.append(tuple(rows))
    return (
        tuple(tuple(row) for row in by_step),
        tuple(by_owner),
    )


def replay_owner_random_allocation_linear_trace(
    *,
    raw_train_x: np.ndarray,
    train_y: np.ndarray,
    records: tuple[ReferenceRecord, ...],
    preprocessing_artifact_path: Path,
    master_seed: int,
    input_dim: int,
    num_classes: int,
    source_dataset_size: int,
    num_steps: int,
    num_selected_steps_per_owner: int,
    num_epochs: int,
    clip_norm: float,
    noise_multiplier: float,
    update_denominator: float,
    learning_rate: float,
    windows_per_owner_per_step: int,
    class_weights: tuple[float, ...] | None,
) -> ReferenceAllocationTrace:
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
    if len(owners) != source_dataset_size:
        raise ValueError("Reference owner population differs from N")

    assigned_by_step, local_steps_by_owner = _schedule(
        master_seed=master_seed,
        source_dataset_size=source_dataset_size,
        num_steps=num_steps,
        num_selected_steps_per_owner=num_selected_steps_per_owner,
        num_epochs=num_epochs,
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
    inward_target = float(
        np.nextafter(np.float32(clip_norm), np.float32(0.0))
    )
    selected_rows_by_step: list[
        tuple[tuple[int, tuple[int, ...]], ...]
    ] = []
    max_unclipped_norms: list[float] = []
    max_clipped_norms: list[float] = []

    for assigned_positions in assigned_by_step:
        gradient_sums = [
            torch.zeros_like(parameter) for parameter in parameters
        ]
        step_rows: list[tuple[int, tuple[int, ...]]] = []
        step_unclipped: list[float] = []
        step_clipped: list[float] = []
        for position in assigned_positions:
            owner_records = records_by_owner[owners[position]]
            if windows_per_owner_per_step >= len(owner_records):
                selected_records = list(owner_records)
            else:
                positions = within_owner_rng.choice(
                    len(owner_records),
                    size=windows_per_owner_per_step,
                    replace=False,
                )
                selected_records = [
                    owner_records[int(index)]
                    for index in sorted(positions.tolist())
                ]
            indices = tuple(
                record.row_index for record in selected_records
            )
            step_rows.append((position, indices))
            model.zero_grad(set_to_none=True)
            criterion(
                model(torch.from_numpy(features[list(indices)])),
                torch.from_numpy(labels[list(indices)]),
            ).backward()
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
            step_unclipped.append(norm)
            step_clipped.append(clipped_norm)
        selected_rows_by_step.append(tuple(step_rows))
        max_unclipped_norms.append(max(step_unclipped, default=0.0))
        max_clipped_norms.append(max(step_clipped, default=0.0))

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

    return ReferenceAllocationTrace(
        model=model,
        assigned_owner_positions=assigned_by_step,
        local_steps_by_epoch_and_owner=local_steps_by_owner,
        selected_rows_by_position=tuple(selected_rows_by_step),
        max_unclipped_norms=tuple(max_unclipped_norms),
        max_clipped_norms=tuple(max_clipped_norms),
    )
