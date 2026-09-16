"""Conservative direct accounting for owner random allocation.

This module implements the integer-order direct bound from Feldman and
Shenfeld (NeurIPS 2025) for the remove direction, together with a conservative
composition of their Gaussian add-direction bound.  It does not implement the
tighter 2026 PLD accountant.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from scipy import special


RANDOM_ALLOCATION_THEOREM_ID_V4 = (
    "feldman_shenfeld_2025_direct_random_allocation"
)
RANDOM_ALLOCATION_ACCOUNTANT_ID_V4 = (
    "random_allocation_gaussian_direct_bidir_v4"
)
RANDOM_ALLOCATION_RDP_ORDERS_V4 = tuple(range(2, 17))


class RandomAllocationAccountingV4Error(ValueError):
    """Raised when a direct random-allocation query is not well formed."""


@dataclass(frozen=True)
class RandomAllocationAccountingResultV4:
    """A complete bidirectional accounting result."""

    num_steps: int
    num_selected: int
    num_epochs: int
    sigma: float
    delta: float
    orders: tuple[int, ...]
    epsilon: float
    epsilon_remove: float
    epsilon_add: float
    optimal_remove_order: int
    remove_rdp_by_order: tuple[tuple[int, float], ...]
    steps_per_single_allocation: int
    composed_single_allocations: int
    add_effective_gaussian_sigma: float
    add_gaussian_epsilon: float
    add_shift_per_allocation: float
    add_total_shift: float


def _require_integer(
    value: object,
    *,
    name: str,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RandomAllocationAccountingV4Error(
            f"{name} must be an integer"
        )
    if value < minimum:
        raise RandomAllocationAccountingV4Error(
            f"{name} must be at least {minimum}"
        )
    return value


def _validate_query(
    *,
    num_steps: int,
    num_selected: int,
    num_epochs: int,
    sigma: float,
    delta: float,
    orders: Iterable[int],
) -> tuple[int, int, int, float, float, tuple[int, ...]]:
    steps = _require_integer(
        num_steps,
        name="num_steps",
        minimum=1,
    )
    selected = _require_integer(
        num_selected,
        name="num_selected",
        minimum=1,
    )
    epochs = _require_integer(
        num_epochs,
        name="num_epochs",
        minimum=1,
    )
    if selected > steps:
        raise RandomAllocationAccountingV4Error(
            "num_selected must not exceed num_steps"
        )
    if (
        isinstance(sigma, bool)
        or not isinstance(sigma, (int, float))
        or not math.isfinite(float(sigma))
        or float(sigma) <= 0.0
    ):
        raise RandomAllocationAccountingV4Error(
            "sigma must be a finite positive number"
        )
    if (
        isinstance(delta, bool)
        or not isinstance(delta, (int, float))
        or not math.isfinite(float(delta))
        or not 0.0 < float(delta) < 1.0
    ):
        raise RandomAllocationAccountingV4Error(
            "delta must lie strictly between zero and one"
        )
    normalized_orders = tuple(orders)
    if not normalized_orders:
        raise RandomAllocationAccountingV4Error(
            "orders must not be empty"
        )
    for order in normalized_orders:
        _require_integer(order, name="RDP order", minimum=2)
    if (
        tuple(sorted(normalized_orders)) != normalized_orders
        or len(set(normalized_orders)) != len(normalized_orders)
    ):
        raise RandomAllocationAccountingV4Error(
            "orders must be strictly increasing and unique"
        )
    return (
        steps,
        selected,
        epochs,
        float(sigma),
        float(delta),
        normalized_orders,
    )


@lru_cache(maxsize=None)
def _integer_partitions_descending(
    total: int,
    max_parts: int,
    max_value: int,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate descending integer partitions with at most ``max_parts``."""

    if total == 0:
        return ((),)
    if total < 0 or max_parts <= 0 or max_value <= 0:
        return ()
    result: list[tuple[int, ...]] = []
    for first in range(min(total, max_value), 0, -1):
        for suffix in _integer_partitions_descending(
            total - first,
            max_parts - 1,
            first,
        ):
            result.append((first, *suffix))
    return tuple(result)


def integer_partitions_v4(
    total: int,
    *,
    max_parts: int,
) -> tuple[tuple[int, ...], ...]:
    """Return the exact partition family used by the direct theorem."""

    normalized_total = _require_integer(
        total,
        name="total",
        minimum=1,
    )
    normalized_max_parts = _require_integer(
        max_parts,
        name="max_parts",
        minimum=1,
    )
    return _integer_partitions_descending(
        normalized_total,
        normalized_max_parts,
        normalized_total,
    )


def _log_falling_factorial(n: int, count: int) -> float:
    if count < 0 or count > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(n - count + 1)


def _partition_log_coefficient(
    *,
    partition: tuple[int, ...],
    alpha: int,
    steps: int,
) -> float:
    multiplicities = Counter(partition)
    allocation_positions = _log_falling_factorial(
        steps,
        len(partition),
    ) - math.fsum(
        math.lgamma(count + 1)
        for count in multiplicities.values()
    )
    alpha_multinomial = math.lgamma(alpha + 1) - math.fsum(
        math.lgamma(part + 1) for part in partition
    )
    return allocation_positions + alpha_multinomial


def random_allocation_remove_rdp_v4(
    *,
    alpha: int,
    sigma: float,
    steps_per_single_allocation: int,
) -> float:
    """Compute one direct integer-order remove-direction RDP value."""

    normalized_alpha = _require_integer(
        alpha,
        name="alpha",
        minimum=2,
    )
    normalized_steps = _require_integer(
        steps_per_single_allocation,
        name="steps_per_single_allocation",
        minimum=1,
    )
    if (
        isinstance(sigma, bool)
        or not isinstance(sigma, (int, float))
        or not math.isfinite(float(sigma))
        or float(sigma) <= 0.0
    ):
        raise RandomAllocationAccountingV4Error(
            "sigma must be a finite positive number"
        )
    normalized_sigma = float(sigma)
    inverse_two_variance = 1.0 / (
        2.0 * normalized_sigma * normalized_sigma
    )
    log_terms: list[float] = []
    for partition in integer_partitions_v4(
        normalized_alpha,
        max_parts=normalized_steps,
    ):
        log_terms.append(
            _partition_log_coefficient(
                partition=partition,
                alpha=normalized_alpha,
                steps=normalized_steps,
            )
            + math.fsum(part * part for part in partition)
            * inverse_two_variance
        )
    maximum = max(log_terms)
    log_sum = maximum + math.log(
        math.fsum(math.exp(value - maximum) for value in log_terms)
    )
    rdp = (
        log_sum
        - normalized_alpha
        * (
            inverse_two_variance
            + math.log(normalized_steps)
        )
    ) / (normalized_alpha - 1)
    if rdp < 0.0 and abs(rdp) <= 1e-14:
        return 0.0
    if not math.isfinite(rdp) or rdp < 0.0:
        raise ArithmeticError(
            "direct random-allocation RDP was invalid"
        )
    return rdp


def _rdp_to_epsilon(
    *,
    alpha: int,
    rdp: float,
    delta: float,
) -> float:
    correction = max(
        math.log1p(-1.0 / alpha)
        - math.log(delta * alpha) / (alpha - 1),
        0.0,
    )
    return rdp + correction


def _gaussian_delta(
    *,
    epsilon: float,
    sigma: float,
) -> float:
    first = float(
        special.ndtr(0.5 / sigma - sigma * epsilon)
    )
    second_log = (
        epsilon
        + float(
            special.log_ndtr(
                -0.5 / sigma - sigma * epsilon
            )
        )
    )
    second = math.exp(second_log) if second_log > -745.0 else 0.0
    value = first - second
    return min(1.0, max(0.0, value))


def analytic_gaussian_epsilon_v4(
    *,
    sigma: float,
    delta: float,
    tolerance: float = 1e-12,
) -> float:
    """Invert the exact Gaussian privacy profile by bisection."""

    if (
        isinstance(sigma, bool)
        or not isinstance(sigma, (int, float))
        or not math.isfinite(float(sigma))
        or float(sigma) <= 0.0
    ):
        raise RandomAllocationAccountingV4Error(
            "sigma must be a finite positive number"
        )
    if (
        isinstance(delta, bool)
        or not isinstance(delta, (int, float))
        or not math.isfinite(float(delta))
        or not 0.0 < float(delta) < 1.0
    ):
        raise RandomAllocationAccountingV4Error(
            "delta must lie strictly between zero and one"
        )
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or float(tolerance) <= 0.0
    ):
        raise RandomAllocationAccountingV4Error(
            "tolerance must be finite and positive"
        )
    normalized_sigma = float(sigma)
    normalized_delta = float(delta)
    if _gaussian_delta(
        epsilon=0.0,
        sigma=normalized_sigma,
    ) <= normalized_delta:
        return 0.0
    lower = 0.0
    upper = max(
        1.0,
        math.sqrt(2.0 * math.log(1.25 / normalized_delta))
        / normalized_sigma,
    )
    while (
        _gaussian_delta(
            epsilon=upper,
            sigma=normalized_sigma,
        )
        > normalized_delta
    ):
        upper *= 2.0
        if not math.isfinite(upper):
            raise ArithmeticError(
                "could not bracket analytic Gaussian epsilon"
            )
    target_tolerance = float(tolerance)
    while upper - lower > target_tolerance:
        midpoint = (lower + upper) / 2.0
        if (
            _gaussian_delta(
                epsilon=midpoint,
                sigma=normalized_sigma,
            )
            > normalized_delta
        ):
            lower = midpoint
        else:
            upper = midpoint
    return upper


def account_random_allocation_v4(
    *,
    num_steps: int,
    num_selected: int,
    num_epochs: int,
    sigma: float,
    delta: float,
    orders: Iterable[int] = RANDOM_ALLOCATION_RDP_ORDERS_V4,
) -> RandomAllocationAccountingResultV4:
    """Return a conservative bidirectional direct accounting result."""

    (
        steps,
        selected,
        epochs,
        normalized_sigma,
        normalized_delta,
        normalized_orders,
    ) = _validate_query(
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        sigma=sigma,
        delta=delta,
        orders=orders,
    )
    steps_per_single_allocation = steps // selected
    composed_single_allocations = selected * epochs
    remove_rows: list[tuple[int, float]] = []
    remove_epsilons: list[tuple[float, int]] = []
    for alpha in normalized_orders:
        composed_rdp = (
            composed_single_allocations
            * random_allocation_remove_rdp_v4(
                alpha=alpha,
                sigma=normalized_sigma,
                steps_per_single_allocation=(
                    steps_per_single_allocation
                ),
            )
        )
        remove_rows.append((alpha, composed_rdp))
        remove_epsilons.append(
            (
                _rdp_to_epsilon(
                    alpha=alpha,
                    rdp=composed_rdp,
                    delta=normalized_delta,
                ),
                alpha,
            )
        )
    epsilon_remove, optimal_remove_order = min(remove_epsilons)

    add_effective_sigma = normalized_sigma * math.sqrt(
        steps_per_single_allocation
        / composed_single_allocations
    )
    add_gaussian_epsilon = analytic_gaussian_epsilon_v4(
        sigma=add_effective_sigma,
        delta=normalized_delta,
    )
    add_shift_per_allocation = (
        1.0 - 1.0 / steps_per_single_allocation
    ) / (2.0 * normalized_sigma * normalized_sigma)
    # Compose the normalizer shift conservatively across the kE
    # single-allocation reductions.
    add_total_shift = (
        composed_single_allocations
        * add_shift_per_allocation
    )
    epsilon_add = add_gaussian_epsilon + add_total_shift
    epsilon = max(epsilon_remove, epsilon_add)
    return RandomAllocationAccountingResultV4(
        num_steps=steps,
        num_selected=selected,
        num_epochs=epochs,
        sigma=normalized_sigma,
        delta=normalized_delta,
        orders=normalized_orders,
        epsilon=epsilon,
        epsilon_remove=epsilon_remove,
        epsilon_add=epsilon_add,
        optimal_remove_order=optimal_remove_order,
        remove_rdp_by_order=tuple(remove_rows),
        steps_per_single_allocation=steps_per_single_allocation,
        composed_single_allocations=composed_single_allocations,
        add_effective_gaussian_sigma=add_effective_sigma,
        add_gaussian_epsilon=add_gaussian_epsilon,
        add_shift_per_allocation=add_shift_per_allocation,
        add_total_shift=add_total_shift,
    )


def calibrate_random_allocation_sigma_v4(
    *,
    target_epsilon: float,
    delta: float,
    num_steps: int,
    num_selected: int,
    num_epochs: int,
    orders: Iterable[int] = RANDOM_ALLOCATION_RDP_ORDERS_V4,
    lower: float = 0.1,
    upper: float = 16.0,
    tolerance: float = 1e-13,
) -> float:
    """Find the smallest bracketed sigma meeting ``target_epsilon``."""

    if (
        isinstance(target_epsilon, bool)
        or not isinstance(target_epsilon, (int, float))
        or not math.isfinite(float(target_epsilon))
        or float(target_epsilon) <= 0.0
    ):
        raise RandomAllocationAccountingV4Error(
            "target_epsilon must be finite and positive"
        )
    if not 0.0 < float(lower) < float(upper):
        raise RandomAllocationAccountingV4Error(
            "calibration bounds must satisfy 0 < lower < upper"
        )
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or float(tolerance) <= 0.0
    ):
        raise RandomAllocationAccountingV4Error(
            "tolerance must be finite and positive"
        )
    normalized_orders = tuple(orders)
    target = float(target_epsilon)
    low = float(lower)
    high = float(upper)
    if (
        account_random_allocation_v4(
            num_steps=num_steps,
            num_selected=num_selected,
            num_epochs=num_epochs,
            sigma=high,
            delta=delta,
            orders=normalized_orders,
        ).epsilon
        > target
    ):
        raise RandomAllocationAccountingV4Error(
            "upper calibration bound does not meet target epsilon"
        )
    if (
        account_random_allocation_v4(
            num_steps=num_steps,
            num_selected=num_selected,
            num_epochs=num_epochs,
            sigma=low,
            delta=delta,
            orders=normalized_orders,
        ).epsilon
        <= target
    ):
        raise RandomAllocationAccountingV4Error(
            "lower calibration bound already meets target epsilon"
        )
    target_tolerance = float(tolerance)
    while high - low > target_tolerance:
        midpoint = (low + high) / 2.0
        result = account_random_allocation_v4(
            num_steps=num_steps,
            num_selected=num_selected,
            num_epochs=num_epochs,
            sigma=midpoint,
            delta=delta,
            orders=normalized_orders,
        )
        if result.epsilon > target:
            low = midpoint
        else:
            high = midpoint
    return high
