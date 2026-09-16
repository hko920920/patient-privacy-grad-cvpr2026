"""RDP accounting helpers for owner-aligned DP-SGD."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from dp_accounting import dp_event
from dp_accounting.privacy_accountant import NeighboringRelation
from dp_accounting.rdp import RdpAccountant as DpAccountingRdpAccountant
from opacus.accountants import RDPAccountant


DEFAULT_ALPHAS = [
    1.1,
    1.2,
    1.5,
    2,
    3,
    4,
    5,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
]


@dataclass(frozen=True)
class RDPConfig:
    """Configuration for an RDP accountant binding."""

    delta: float = 1e-5
    target_epsilon: float | None = None
    noise_multiplier: float | None = None
    alphas: list[float] = field(default_factory=lambda: list(DEFAULT_ALPHAS))

    def validate(self) -> None:
        if self.delta <= 0 or self.delta >= 1:
            raise ValueError("delta must be in (0, 1)")
        if self.target_epsilon is None and self.noise_multiplier is None:
            raise ValueError("Provide target_epsilon or noise_multiplier")
        if self.target_epsilon is not None and self.target_epsilon <= 0:
            raise ValueError("target_epsilon must be positive")
        if self.noise_multiplier is not None and self.noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be positive")
        if any(alpha <= 1 for alpha in self.alphas):
            raise ValueError("All RDP orders must exceed one")


def epsilon_for_noise(
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    delta: float,
    alphas: list[float] | None = None,
) -> float:
    """Return epsilon for repeated sampled Gaussian steps."""

    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be positive")
    if sample_rate <= 0 or sample_rate > 1:
        raise ValueError("sample_rate must be in (0, 1]")
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    orders = alphas or DEFAULT_ALPHAS
    accountant = RDPAccountant()
    for _ in range(steps):
        accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    return float(accountant.get_epsilon(delta=delta, alphas=orders))


def epsilon_for_noise_dp_accounting_add_remove(
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    delta: float,
    alphas: list[float] | None = None,
) -> float:
    """Independently compute Poisson-Gaussian RDP under add/remove adjacency."""

    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be positive")
    if sample_rate <= 0 or sample_rate > 1:
        raise ValueError("sample_rate must be in (0, 1]")
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0, 1)")
    if steps == 0:
        return 0.0
    orders = alphas or DEFAULT_ALPHAS
    accountant = DpAccountingRdpAccountant(
        orders=orders,
        neighboring_relation=NeighboringRelation.ADD_OR_REMOVE_ONE,
    )
    event = dp_event.PoissonSampledDpEvent(
        sampling_probability=sample_rate,
        event=dp_event.GaussianDpEvent(noise_multiplier),
    )
    accountant.compose(event, count=steps)
    return float(accountant.get_epsilon(delta))


def calibrate_noise(
    target_epsilon: float,
    sample_rate: float,
    steps: int,
    delta: float,
    alphas: list[float] | None = None,
) -> float:
    """Binary-search a noise multiplier for the target epsilon."""

    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be positive")
    if steps == 0:
        return math.inf

    low = 0.01
    high = 1.0
    while epsilon_for_noise(high, sample_rate, steps, delta, alphas) > target_epsilon:
        high *= 2.0
        if high > 1024:
            raise RuntimeError("Could not bracket the target epsilon")

    for _ in range(50):
        mid = (low + high) / 2.0
        if epsilon_for_noise(mid, sample_rate, steps, delta, alphas) > target_epsilon:
            low = mid
        else:
            high = mid
    return float(high)


def epsilon_for_srswor_replace_one_noise(
    noise_multiplier: float,
    source_dataset_size: int,
    sample_size: int,
    steps: int,
    delta: float,
    alphas: list[float] | None = None,
) -> float:
    """Return epsilon for fixed-size owner sampling without replacement.

    The reported ``noise_multiplier`` is relative to the owner clip norm ``C``.
    Under replace-one owner adjacency, replacing one owner can remove one
    clipped vector and add another, so the Gaussian event sensitivity is
    bounded by ``2C``. The `dp_accounting` Gaussian event expects the noise
    multiplier relative to that sensitivity; hence the factor of two below.
    """

    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be positive")
    if source_dataset_size <= 0:
        raise ValueError("source_dataset_size must be positive")
    if sample_size <= 0 or sample_size > source_dataset_size:
        raise ValueError("sample_size must be in [1, source_dataset_size]")
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0, 1)")
    if steps == 0:
        return 0.0
    orders = alphas or DEFAULT_ALPHAS
    accountant = DpAccountingRdpAccountant(
        orders=orders,
        neighboring_relation=NeighboringRelation.REPLACE_ONE,
    )
    event = dp_event.SampledWithoutReplacementDpEvent(
        source_dataset_size=source_dataset_size,
        sample_size=sample_size,
        event=dp_event.GaussianDpEvent(noise_multiplier / 2.0),
    )
    accountant.compose(event, count=steps)
    return float(accountant.get_epsilon(delta))


def calibrate_noise_srswor_replace_one(
    target_epsilon: float,
    source_dataset_size: int,
    sample_size: int,
    steps: int,
    delta: float,
    alphas: list[float] | None = None,
) -> float:
    """Binary-search actual noise/C for fixed-size SRSWOR owner sampling."""

    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be positive")
    if steps == 0:
        return math.inf

    low = 0.01
    high = 1.0
    while (
        epsilon_for_srswor_replace_one_noise(
            high,
            source_dataset_size,
            sample_size,
            steps,
            delta,
            alphas,
        )
        > target_epsilon
    ):
        high *= 2.0
        if high > 2048:
            raise RuntimeError("Could not bracket the target epsilon")

    for _ in range(50):
        mid = (low + high) / 2.0
        if (
            epsilon_for_srswor_replace_one_noise(
                mid,
                source_dataset_size,
                sample_size,
                steps,
                delta,
                alphas,
            )
            > target_epsilon
        ):
            low = mid
        else:
            high = mid
    return float(high)


def log_geometric_sum(epsilon: float, k: int) -> float:
    if k <= 0:
        return float("-inf")
    if k == 1:
        return 0.0
    if abs(epsilon) < 1e-12:
        return math.log(k)
    max_term = (k - 1) * epsilon
    total = sum(math.exp(i * epsilon - max_term) for i in range(k))
    return max_term + math.log(total)


def group_privacy_conversion(epsilon: float, delta: float, k: int) -> dict[str, object]:
    """Conservative standard DP group-privacy conversion."""

    if k <= 1:
        return {
            "epsilon": float(epsilon),
            "delta": float(delta),
            "k": int(k),
            "vacuous": bool(delta >= 1),
        }
    epsilon_group = k * epsilon
    log_delta_group = math.log(delta) + log_geometric_sum(epsilon, k)
    if log_delta_group >= 0:
        delta_group: float | str = ">=1"
        vacuous = True
    else:
        delta_group = float(math.exp(log_delta_group))
        vacuous = False
    return {
        "epsilon": float(epsilon_group),
        "delta": delta_group,
        "log10_delta": float(log_delta_group / math.log(10)),
        "k": int(k),
        "vacuous": bool(vacuous),
    }


def required_base_delta_for_group(target_delta: float, epsilon: float, k: int) -> float:
    """Return the base-unit delta needed for group delta <= target_delta."""

    if target_delta <= 0 or target_delta >= 1:
        raise ValueError("target_delta must be in (0, 1)")
    if k <= 0:
        raise ValueError("k must be positive")
    log_required = math.log(target_delta) - log_geometric_sum(epsilon, k)
    return float(math.exp(log_required))
