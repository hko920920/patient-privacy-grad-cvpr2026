"""Import-independent SRSWOR Gaussian RDP oracle for the strict V3 route.

This module deliberately imports no ``unitdp`` production module and no
third-party privacy accountant.  It directly evaluates the integer-order
forward-difference bound used for fixed-size sampling without replacement
under replace-one adjacency:

    Wang, Balle, and Kasiviswanathan, AISTATS 2019,
    "Subsampled Renyi Differential Privacy and Analytical Moments
    Accountant."

Only the small, registered integer-order set is supported.  Decimal arithmetic
keeps this implementation numerically independent from ``dp_accounting``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Iterable


ORACLE_IMPLEMENTATION_ID_V3 = (
    "unitdp_spec_oracle.srswor_rdp_oracle_v3."
    "fixed_size_srswor_epsilon_v3"
)
ORACLE_THEOREM_ID_V3 = "wang_balle_kasiviswanathan_2019_srswor_rdp"
ORACLE_DECIMAL_PRECISION_V3 = 100
SUPPORTED_INTEGER_ORDERS_V3 = (2, 3, 4, 5, 8, 16)


class SrsworRdpOracleV3Error(ValueError):
    """Raised when an oracle input lies outside the registered domain."""


@dataclass(frozen=True)
class SrsworOrderResultV3:
    order: int
    per_step_rdp: float
    composed_rdp: float
    epsilon: float


@dataclass(frozen=True)
class SrsworOracleResultV3:
    epsilon: float
    optimal_order: int
    source_dataset_size: int
    sample_size: int
    sample_rate: float
    steps: int
    delta: float
    actual_noise_multiplier: float
    sensitivity_multiplier: float
    normalized_noise_multiplier: float
    order_results: tuple[SrsworOrderResultV3, ...]


def _strict_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SrsworRdpOracleV3Error(f"{name} must be a positive integer")
    return value


def _strict_positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SrsworRdpOracleV3Error(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise SrsworRdpOracleV3Error(f"{name} must be finite and positive")
    return result


def _registered_orders(values: Iterable[int]) -> tuple[int, ...]:
    orders = tuple(values)
    if not orders:
        raise SrsworRdpOracleV3Error("orders must not be empty")
    if len(set(orders)) != len(orders):
        raise SrsworRdpOracleV3Error("orders must be distinct")
    if any(
        isinstance(order, bool)
        or not isinstance(order, int)
        or order not in SUPPORTED_INTEGER_ORDERS_V3
        for order in orders
    ):
        raise SrsworRdpOracleV3Error(
            "orders must be a subset of the registered integer orders "
            f"{SUPPORTED_INTEGER_ORDERS_V3}"
        )
    return orders


def _decimal(value: float) -> Decimal:
    return Decimal(repr(value))


def _gaussian_cgf(index: int, sigma: Decimal) -> Decimal:
    value = Decimal(index)
    return value * (value + 1) / (2 * sigma * sigma)


def _forward_difference_exp_cgf(
    *,
    order: int,
    sigma: Decimal,
) -> Decimal:
    """Return Delta^order exp(K)(-1) by its defining finite sum."""

    total = Decimal(0)
    for index in range(order + 1):
        sign = -1 if (order - index) % 2 else 1
        coefficient = Decimal(sign * math.comb(order, index))
        cgf = _gaussian_cgf(-1 + index, sigma)
        total += coefficient * cgf.exp()
    if total <= 0:
        raise ArithmeticError(
            "The registered forward difference was nonpositive"
        )
    return total


def _integer_order_srswor_rdp(
    *,
    sample_rate: Decimal,
    normalized_noise_multiplier: Decimal,
    order: int,
) -> Decimal:
    """Evaluate the registered integer-order SRSWOR Gaussian RDP bound."""

    sigma = normalized_noise_multiplier
    gaussian_rdp_at_two = Decimal(1) / (sigma * sigma)
    exp_at_two = gaussian_rdp_at_two.exp()
    zeta_two = min(
        4 * (exp_at_two - 1),
        2 * exp_at_two,
    )
    moment_bound = Decimal(1)
    for index in range(2, order + 1):
        if index == 2:
            zeta = zeta_two
        else:
            lower_order = 2 * (index // 2)
            upper_order = 2 * ((index + 1) // 2)
            lower = _forward_difference_exp_cgf(
                order=lower_order,
                sigma=sigma,
            )
            upper = _forward_difference_exp_cgf(
                order=upper_order,
                sigma=sigma,
            )
            forward_bound = 4 * (lower * upper).sqrt()
            cgf_bound = 2 * _gaussian_cgf(index - 1, sigma).exp()
            zeta = min(forward_bound, cgf_bound)
        moment_bound += (
            Decimal(math.comb(order, index))
            * (sample_rate**index)
            * zeta
        )
    return moment_bound.ln() / Decimal(order - 1)


def _epsilon_from_rdp(
    *,
    rdp: Decimal,
    order: int,
    delta: Decimal,
) -> Decimal:
    """Use the same published improved RDP-to-DP conversion bound."""

    alpha = Decimal(order)
    epsilon = (
        rdp
        + (1 - 1 / alpha).ln()
        - (delta * alpha).ln() / (alpha - 1)
    )
    return max(Decimal(0), epsilon)


def fixed_size_srswor_epsilon_v3(
    *,
    actual_noise_multiplier: float,
    source_dataset_size: int,
    sample_size: int,
    steps: int,
    delta: float,
    orders: Iterable[int] = SUPPORTED_INTEGER_ORDERS_V3,
    sensitivity_multiplier: float = 2.0,
) -> SrsworOracleResultV3:
    """Return epsilon for the registered fixed-size replace-one route.

    ``actual_noise_multiplier`` is the Gaussian standard deviation divided by
    the owner clip norm ``C``.  The Gaussian event is normalized by the
    replace-one sensitivity ``2C``, hence the registered sensitivity
    multiplier is exactly two.
    """

    population = _strict_positive_int(
        source_dataset_size,
        "source_dataset_size",
    )
    batch = _strict_positive_int(sample_size, "sample_size")
    if batch > population:
        raise SrsworRdpOracleV3Error(
            "sample_size must not exceed source_dataset_size"
        )
    count = _strict_positive_int(steps, "steps")
    noise = _strict_positive_float(
        actual_noise_multiplier,
        "actual_noise_multiplier",
    )
    delta_value = _strict_positive_float(delta, "delta")
    if delta_value >= 1:
        raise SrsworRdpOracleV3Error("delta must be smaller than one")
    sensitivity = _strict_positive_float(
        sensitivity_multiplier,
        "sensitivity_multiplier",
    )
    if sensitivity != 2.0:
        raise SrsworRdpOracleV3Error(
            "the registered replace-one sensitivity multiplier is exactly 2"
        )
    registered_orders = _registered_orders(orders)

    with localcontext() as context:
        context.prec = ORACLE_DECIMAL_PRECISION_V3
        sample_rate_decimal = Decimal(batch) / Decimal(population)
        normalized_noise = _decimal(noise) / _decimal(sensitivity)
        delta_decimal = _decimal(delta_value)
        results: list[SrsworOrderResultV3] = []
        for order in registered_orders:
            per_step = _integer_order_srswor_rdp(
                sample_rate=sample_rate_decimal,
                normalized_noise_multiplier=normalized_noise,
                order=order,
            )
            composed = Decimal(count) * per_step
            epsilon = _epsilon_from_rdp(
                rdp=composed,
                order=order,
                delta=delta_decimal,
            )
            results.append(
                SrsworOrderResultV3(
                    order=order,
                    per_step_rdp=float(per_step),
                    composed_rdp=float(composed),
                    epsilon=float(epsilon),
                )
            )

    best = min(results, key=lambda item: item.epsilon)
    return SrsworOracleResultV3(
        epsilon=best.epsilon,
        optimal_order=best.order,
        source_dataset_size=population,
        sample_size=batch,
        sample_rate=batch / population,
        steps=count,
        delta=delta_value,
        actual_noise_multiplier=noise,
        sensitivity_multiplier=sensitivity,
        normalized_noise_multiplier=noise / sensitivity,
        order_results=tuple(results),
    )
