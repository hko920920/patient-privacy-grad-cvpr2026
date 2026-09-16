"""Randomness utilities for research and deployment-mode DP runs."""

from __future__ import annotations

import math
import secrets
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class RngMetadata:
    secure_rng: bool
    rng_backend: str
    rng_security_mode: str
    rng_caveat: str


class ResearchRandomSource:
    """Default deterministic source used by the paper's reproducible runs."""

    def __init__(self, seed: int):
        self._rng = np.random.default_rng(seed)

    def random(self) -> float:
        return float(self._rng.random())

    def choice(self, values: int | Sequence[object], size: int, replace: bool = False) -> np.ndarray:
        return self._rng.choice(values, size=size, replace=replace)

    def shuffle(self, values: list[object]) -> None:
        self._rng.shuffle(values)

    def normal_tensor(self, shape: torch.Size | tuple[int, ...], std: float, device: str) -> torch.Tensor:
        return torch.normal(mean=0.0, std=std, size=tuple(shape), device=device)

    def normal_array(self, size: int, std: float) -> np.ndarray:
        return self._rng.normal(loc=0.0, scale=std, size=size)

    def metadata(self) -> RngMetadata:
        return RngMetadata(
            secure_rng=False,
            rng_backend="numpy_default_rng_and_torch_default_generator",
            rng_security_mode="research_noncryptographic",
            rng_caveat=(
                "Research runs use NumPy/PyTorch default pseudorandom generators; "
                "privacy-critical production runs should use a cryptographically secure noise source."
            ),
        )


class SystemRandomSource:
    """Prototype CSPRNG-backed source using the operating-system entropy pool.

    This backend is intentionally simple and CPU-oriented. It is meant to
    exercise the artifact interface and certificate fields; production systems
    should plug in an audited high-throughput secure noise implementation.
    """

    def __init__(self):
        self._rng = secrets.SystemRandom()

    def random(self) -> float:
        return float(self._rng.random())

    def choice(self, values: int | Sequence[object], size: int, replace: bool = False) -> np.ndarray:
        if isinstance(values, int):
            population: list[object] = list(range(values))
        else:
            population = list(values)
        if replace:
            selected = [self._rng.choice(population) for _ in range(size)]
        else:
            selected = self._rng.sample(population, size)
        return np.asarray(selected, dtype=object if not isinstance(values, int) else int)

    def shuffle(self, values: list[object]) -> None:
        self._rng.shuffle(values)

    def _standard_normals(self, count: int) -> np.ndarray:
        values: list[float] = []
        while len(values) < count:
            u1 = max(self._rng.random(), np.finfo(float).tiny)
            u2 = self._rng.random()
            radius = math.sqrt(-2.0 * math.log(u1))
            angle = 2.0 * math.pi * u2
            values.append(radius * math.cos(angle))
            if len(values) < count:
                values.append(radius * math.sin(angle))
        return np.asarray(values, dtype=np.float32)

    def normal_tensor(self, shape: torch.Size | tuple[int, ...], std: float, device: str) -> torch.Tensor:
        count = int(np.prod(tuple(shape)))
        noise = self._standard_normals(count).reshape(tuple(shape)) * float(std)
        return torch.from_numpy(noise).to(device=device)

    def normal_array(self, size: int, std: float) -> np.ndarray:
        return self._standard_normals(size).astype(float) * float(std)

    def metadata(self) -> RngMetadata:
        return RngMetadata(
            secure_rng=True,
            rng_backend="python_secrets_system_random_box_muller",
            rng_security_mode="system_csprng_prototype",
            rng_caveat=(
                "Noise and sampling are drawn from Python secrets.SystemRandom. "
                "For high-throughput deployment, replace this prototype with an audited CSPRNG backend."
            ),
        )


def make_random_source(backend: str, seed: int) -> ResearchRandomSource | SystemRandomSource:
    if backend == "research_default":
        return ResearchRandomSource(seed)
    if backend == "system_csprng":
        return SystemRandomSource()
    raise ValueError("rng_backend must be research_default or system_csprng")


def gaussian_noise_sanity(
    samples: np.ndarray,
    expected_std: float,
    mean_abs_tolerance: float = 0.08,
    std_relative_tolerance: float = 0.08,
) -> dict[str, object]:
    values = np.asarray(samples, dtype=float)
    if values.size == 0:
        raise ValueError("samples must be non-empty")
    empirical_mean = float(values.mean())
    empirical_std = float(values.std(ddof=0))
    relative_std_error = abs(empirical_std - expected_std) / max(expected_std, 1e-12)
    passed = (
        abs(empirical_mean) <= mean_abs_tolerance * max(expected_std, 1e-12)
        and relative_std_error <= std_relative_tolerance
    )
    return {
        "samples": int(values.size),
        "expected_std": float(expected_std),
        "empirical_mean": empirical_mean,
        "empirical_std": empirical_std,
        "relative_std_error": float(relative_std_error),
        "mean_abs_tolerance": float(mean_abs_tolerance),
        "std_relative_tolerance": float(std_relative_tolerance),
        "passed": bool(passed),
    }
