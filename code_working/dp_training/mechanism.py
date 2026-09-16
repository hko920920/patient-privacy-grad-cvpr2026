#!/usr/bin/env python3
"""Small, testable core for the frozen image-DP and patient-DP mechanisms.

This module deliberately knows nothing about X-ray identifiers, files, model
checkpoints, or losses.  The caller supplies exactly one flat fp32 vector per
privacy unit.  For the patient mechanism that vector MUST already be the
gradient of the mean loss over the patient's selected images.

The sampled Gaussian mechanism implemented here is:

    (sum_i clip_C(g_i) + Normal(0, (sigma*C)^2 I)) / B_expected

The denominator is public and fixed.  In particular, an empty Poisson sample
still produces a noise-only update.  This property is central to the frozen
accounting contract and is tested separately.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import torch


SCHEMA = "nih-cxr14-dp-training-mechanism/v1"
TRACE_SCHEMA = "nih-cxr14-public-dp-event-trace/v1"
TRACE_GENESIS = "0" * 64
PUBLIC_EVENT_KEYS = {
    "schema",
    "mechanism_schema",
    "arm",
    "privacy_unit",
    "adjacency",
    "population",
    "expected_batch",
    "poisson_sample_rate",
    "clip_norm",
    "noise_multiplier",
    "noise_std_before_division",
    "fixed_denominator",
    "first_step",
    "event_count",
    "rng_security_mode",
    "rng_backend",
}
FORBIDDEN_PUBLIC_EVENT_KEYS = {
    "selected_ids",
    "selected_unit_ids",
    "realized_batch",
    "realized_batch_size",
    "realized_unit_count",
    "empty_batch",
    "sampling_seed",
    "noise_seed",
}
RNG_SECURITY_MODES = {
    "TEST_ONLY_DETERMINISTIC",
    "RESEARCH_ONLY_NONCRYPTOGRAPHIC",
    "RELEASE_GRADE_CSPRNG",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest().upper()


@dataclass(frozen=True)
class MechanismConfig:
    arm: str
    privacy_unit: str
    population: int
    expected_batch: int
    poisson_sample_rate: float
    clip_norm: float
    noise_multiplier: float
    fixed_denominator: float
    max_steps: int
    rng_security_mode: str
    rng_backend: str
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        _require(self.schema == SCHEMA, "mechanism schema mismatch")
        _require(bool(self.arm), "arm must be non-empty")
        _require(self.privacy_unit in {"image", "patient"}, "privacy_unit must be image or patient")
        _require(isinstance(self.population, int) and self.population > 0, "population must be positive")
        _require(
            isinstance(self.expected_batch, int)
            and 0 < self.expected_batch <= self.population,
            "expected_batch must be an integer in [1, population]",
        )
        expected_q = self.expected_batch / self.population
        _require(
            math.isclose(self.poisson_sample_rate, expected_q, rel_tol=0.0, abs_tol=1e-15),
            "poisson_sample_rate must equal expected_batch/population",
        )
        _require(math.isfinite(self.clip_norm) and self.clip_norm > 0.0, "clip_norm must be positive")
        _require(
            math.isfinite(self.noise_multiplier) and self.noise_multiplier > 0.0,
            "noise_multiplier must be positive",
        )
        _require(
            math.isclose(float(self.fixed_denominator), float(self.expected_batch), rel_tol=0.0, abs_tol=0.0),
            "fixed_denominator must equal the public expected_batch",
        )
        _require(isinstance(self.max_steps, int) and self.max_steps > 0, "max_steps must be positive")
        _require(self.rng_security_mode in RNG_SECURITY_MODES, "unknown rng_security_mode")
        _require(bool(self.rng_backend), "rng_backend must be non-empty")

    @property
    def adjacency(self) -> str:
        return f"add_or_remove_one_{self.privacy_unit}"

    @property
    def noise_std_before_division(self) -> float:
        return self.noise_multiplier * self.clip_norm

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["adjacency"] = self.adjacency
        value["noise_std_before_division"] = self.noise_std_before_division
        return value


def _validate_vector(vector: torch.Tensor, *, name: str, shape: torch.Size | None = None) -> None:
    _require(isinstance(vector, torch.Tensor), f"{name} must be a torch.Tensor")
    _require(vector.dtype == torch.float32, f"{name} must be fp32")
    _require(vector.ndim == 1, f"{name} must be a flat vector")
    if shape is not None:
        _require(vector.shape == shape, f"{name} shape mismatch")
    _require(bool(torch.isfinite(vector).all()), f"{name} contains a non-finite value")


def clip_unit_vector(vector: torch.Tensor, clip_norm: float) -> tuple[torch.Tensor, dict[str, Any]]:
    """Clip one complete privacy-unit vector to the declared L2 norm."""

    _validate_vector(vector, name="unit_vector")
    _require(math.isfinite(clip_norm) and clip_norm > 0.0, "clip_norm must be positive")
    norm = float(torch.linalg.vector_norm(vector.double()).cpu())
    factor = min(1.0, clip_norm / max(norm, torch.finfo(torch.float64).tiny))
    clipped = vector * factor
    clipped_norm = float(torch.linalg.vector_norm(clipped.double()).cpu())
    _require(clipped_norm <= clip_norm * (1.0 + 2e-6), "clipping postcondition failed")
    return clipped, {
        "unclipped_l2_norm": norm,
        "clip_factor": factor,
        "clipped_l2_norm": clipped_norm,
        "was_clipped": norm > clip_norm,
    }


def mean_unit_vectors(vectors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Average within one patient before patient-level clipping."""

    _require(len(vectors) > 0, "a patient unit must contain at least one image vector")
    shape = vectors[0].shape
    device = vectors[0].device
    for index, vector in enumerate(vectors):
        _validate_vector(vector, name=f"patient_image_vector[{index}]", shape=shape)
        _require(vector.device == device, "patient image vectors must share a device")
    return torch.stack(tuple(vectors), dim=0).mean(dim=0)


def poisson_select(
    sorted_unit_ids: Sequence[str],
    sample_rate: float,
    *,
    generator: torch.Generator,
) -> list[str]:
    """Independent Bernoulli selection over an already canonicalized population."""

    _require(0.0 < sample_rate <= 1.0, "sample_rate must be in (0, 1]")
    _require(list(sorted_unit_ids) == sorted(sorted_unit_ids), "unit ids must be sorted")
    _require(len(set(sorted_unit_ids)) == len(sorted_unit_ids), "unit ids must be unique")
    draws = torch.rand(len(sorted_unit_ids), generator=generator, device="cpu")
    return [unit_id for unit_id, draw in zip(sorted_unit_ids, draws.tolist()) if draw < sample_rate]


def aggregate_noised_update(
    unit_vectors: Sequence[torch.Tensor],
    *,
    template: torch.Tensor,
    config: MechanismConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Clip per unit, add Gaussian noise even if empty, then divide by fixed B."""

    _validate_vector(template, name="template")
    shape = template.shape
    device = template.device
    clipped_sum = torch.zeros_like(template)
    unit_stats: list[dict[str, Any]] = []
    for index, vector in enumerate(unit_vectors):
        _validate_vector(vector, name=f"unit_vector[{index}]", shape=shape)
        _require(vector.device == device, "all unit vectors and template must share a device")
        clipped, stats = clip_unit_vector(vector, config.clip_norm)
        clipped_sum.add_(clipped)
        unit_stats.append(stats)

    # This call is intentionally unconditional.  Empty samples are noise-only
    # sampled-Gaussian events, not skipped optimizer steps.
    noise = torch.randn(
        tuple(template.shape),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ) * config.noise_std_before_division
    update = (clipped_sum + noise) / config.fixed_denominator
    _require(bool(torch.isfinite(update).all()), "noised update is non-finite")
    private_diagnostics = {
        "realized_unit_count": len(unit_vectors),
        "empty_sample": len(unit_vectors) == 0,
        "clipped_unit_count": sum(bool(item["was_clipped"]) for item in unit_stats),
        "maximum_unclipped_l2_norm": max(
            (float(item["unclipped_l2_norm"]) for item in unit_stats), default=0.0
        ),
        "clipped_sum_sha256": tensor_sha256(clipped_sum),
        "noise_sha256": tensor_sha256(noise),
        "update_sha256": tensor_sha256(update),
        "fixed_denominator_used": float(config.fixed_denominator),
        "noise_std_before_division": float(config.noise_std_before_division),
    }
    return update, private_diagnostics


def make_public_scheduled_event(
    config: MechanismConfig,
    *,
    first_step: int,
    event_count: int,
) -> dict[str, Any]:
    _require(isinstance(first_step, int) and first_step >= 1, "first_step must be >= 1")
    _require(isinstance(event_count, int) and event_count >= 1, "event_count must be >= 1")
    _require(first_step + event_count - 1 <= config.max_steps, "event range exceeds max_steps")
    event = {
        "schema": TRACE_SCHEMA,
        "mechanism_schema": config.schema,
        "arm": config.arm,
        "privacy_unit": config.privacy_unit,
        "adjacency": config.adjacency,
        "population": config.population,
        "expected_batch": config.expected_batch,
        "poisson_sample_rate": config.poisson_sample_rate,
        "clip_norm": config.clip_norm,
        "noise_multiplier": config.noise_multiplier,
        "noise_std_before_division": config.noise_std_before_division,
        "fixed_denominator": config.fixed_denominator,
        "first_step": first_step,
        "event_count": event_count,
        "rng_security_mode": config.rng_security_mode,
        "rng_backend": config.rng_backend,
    }
    _require(set(event) == PUBLIC_EVENT_KEYS, "public event schema drift")
    return event


def append_public_trace(
    trace: list[dict[str, Any]],
    event: dict[str, Any],
) -> dict[str, Any]:
    _require(set(event) == PUBLIC_EVENT_KEYS, "public event has missing or extra fields")
    _require(not (set(event) & FORBIDDEN_PUBLIC_EVENT_KEYS), "public event leaks sample realization")
    previous_hash = TRACE_GENESIS if not trace else str(trace[-1]["event_hash"])
    event_hash = sha256_bytes(canonical_json({"previous_hash": previous_hash, "event": event}))
    record = {"previous_hash": previous_hash, "event": dict(event), "event_hash": event_hash}
    trace.append(record)
    return record


def verify_public_trace(trace: Iterable[dict[str, Any]]) -> dict[str, Any]:
    previous = TRACE_GENESIS
    total_events = 0
    records = list(trace)
    for index, record in enumerate(records):
        _require(set(record) == {"previous_hash", "event", "event_hash"}, f"trace record {index} schema drift")
        event = record["event"]
        _require(isinstance(event, dict), f"trace record {index} event is not an object")
        _require(set(event) == PUBLIC_EVENT_KEYS, f"trace record {index} event schema drift")
        _require(not (set(event) & FORBIDDEN_PUBLIC_EVENT_KEYS), f"trace record {index} leaks realization")
        _require(record["previous_hash"] == previous, f"trace record {index} previous hash mismatch")
        expected = sha256_bytes(canonical_json({"previous_hash": previous, "event": event}))
        _require(record["event_hash"] == expected, f"trace record {index} hash mismatch")
        _require(
            math.isclose(
                float(event["poisson_sample_rate"]),
                int(event["expected_batch"]) / int(event["population"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            f"trace record {index} sampling rate mismatch",
        )
        _require(
            float(event["fixed_denominator"]) == float(event["expected_batch"]),
            f"trace record {index} denominator mismatch",
        )
        expected_adjacency = f"add_or_remove_one_{event['privacy_unit']}"
        _require(event["adjacency"] == expected_adjacency, f"trace record {index} adjacency mismatch")
        total_events += int(event["event_count"])
        previous = expected
    return {
        "schema": TRACE_SCHEMA,
        "records": len(records),
        "total_events": total_events,
        "head_sha256": previous,
        "status": "PASS",
    }
