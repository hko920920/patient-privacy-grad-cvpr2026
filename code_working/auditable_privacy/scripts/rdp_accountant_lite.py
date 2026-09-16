"""Small, dependency-light compatibility checker for Opacus RDP accounting.

The numerical formulas and default Renyi orders follow the Apache-2.0
licensed Opacus 1.6.0 implementation in
``opacus/accountants/analysis/rdp.py`` and ``opacus/accountants/rdp.py``.
This local checker avoids importing PyTorch or SciPy, so certificate
validation remains usable on constrained reviewer machines.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple


DEFAULT_ALPHAS = tuple(
    [1 + value / 10.0 for value in range(1, 100)]
    + list(range(12, 64))
)
NEG_INF = float("-inf")


def _log_add(log_x: float, log_y: float) -> float:
    lower, upper = min(log_x, log_y), max(log_x, log_y)
    if lower == NEG_INF:
        return upper
    return math.log1p(math.exp(lower - upper)) + upper


def _log_sub(log_x: float, log_y: float) -> float:
    if log_x < log_y:
        raise ValueError("Negative result in log-space subtraction.")
    if log_y == NEG_INF:
        return log_x
    if log_x == log_y:
        return NEG_INF
    try:
        return math.log(math.expm1(log_x - log_y)) + log_y
    except OverflowError:
        return log_x


def _log_erfc(x: float) -> float:
    """Compute log(erfc(x)), including the positive-tail underflow regime."""

    direct = math.erfc(x)
    if direct > 0.0:
        return math.log(direct)
    if x <= 0.0:
        raise ArithmeticError("Unexpected erfc underflow outside positive tail.")

    inverse_two_x_sq = 1.0 / (2.0 * x * x)
    series = 1.0
    term = 1.0
    previous_magnitude = float("inf")
    for order in range(1, 200):
        term *= -(2 * order - 1) * inverse_two_x_sq
        magnitude = abs(term)
        if magnitude >= previous_magnitude:
            break
        series += term
        if magnitude <= abs(series) * 1e-16:
            break
        previous_magnitude = magnitude
    if series <= 0.0:
        raise ArithmeticError("Unstable asymptotic erfc series.")
    return (
        -(x * x)
        - math.log(x)
        - 0.5 * math.log(math.pi)
        + math.log(series)
    )


def _log_a_integer(q: float, sigma: float, alpha: int) -> float:
    log_a = NEG_INF
    log_q = math.log(q)
    log_one_minus_q = math.log1p(-q)
    for index in range(alpha + 1):
        log_binomial = (
            math.lgamma(alpha + 1)
            - math.lgamma(index + 1)
            - math.lgamma(alpha - index + 1)
        )
        log_coefficient = (
            log_binomial
            + index * log_q
            + (alpha - index) * log_one_minus_q
        )
        exponent = (
            log_coefficient
            + (index * index - index) / (2.0 * sigma * sigma)
        )
        log_a = _log_add(log_a, exponent)
    return log_a


def _log_a_fractional(q: float, sigma: float, alpha: float) -> float:
    log_a0 = NEG_INF
    log_a1 = NEG_INF
    z0 = sigma * sigma * math.log(1.0 / q - 1.0) + 0.5
    log_q = math.log(q)
    log_one_minus_q = math.log1p(-q)
    coefficient = 1.0

    for index in range(10000):
        if index:
            coefficient *= (alpha - index + 1.0) / index
        if coefficient == 0.0:
            break
        log_coefficient = math.log(abs(coefficient))
        complement = alpha - index
        log_t0 = (
            log_coefficient
            + index * log_q
            + complement * log_one_minus_q
        )
        log_t1 = (
            log_coefficient
            + complement * log_q
            + index * log_one_minus_q
        )
        log_e0 = math.log(0.5) + _log_erfc(
            (index - z0) / (math.sqrt(2.0) * sigma)
        )
        log_e1 = math.log(0.5) + _log_erfc(
            (z0 - complement) / (math.sqrt(2.0) * sigma)
        )
        log_s0 = (
            log_t0
            + (index * index - index) / (2.0 * sigma * sigma)
            + log_e0
        )
        log_s1 = (
            log_t1
            + (complement * complement - complement)
            / (2.0 * sigma * sigma)
            + log_e1
        )
        if coefficient > 0.0:
            log_a0 = _log_add(log_a0, log_s0)
            log_a1 = _log_add(log_a1, log_s1)
        else:
            log_a0 = _log_sub(log_a0, log_s0)
            log_a1 = _log_sub(log_a1, log_s1)
        if max(log_s0, log_s1) < -30.0:
            return _log_add(log_a0, log_a1)
    raise ArithmeticError("Fractional-order RDP series did not converge.")


def _rdp_at_order(q: float, sigma: float, alpha: float) -> float:
    if q == 0.0:
        return 0.0
    if sigma == 0.0:
        return float("inf")
    if q == 1.0:
        return alpha / (2.0 * sigma * sigma)
    log_a = (
        _log_a_integer(q, sigma, int(alpha))
        if float(alpha).is_integer()
        else _log_a_fractional(q, sigma, alpha)
    )
    return log_a / (alpha - 1.0)


def epsilon_for_poisson_gaussian(
    sample_rate: float,
    steps: int,
    noise_multiplier: float,
    delta: float,
    *,
    orders: Iterable[float] = DEFAULT_ALPHAS,
) -> Tuple[float, float]:
    """Return ``(epsilon, best_alpha)`` for repeated Poisson SGM steps."""

    if not math.isfinite(sample_rate) or not 0.0 < sample_rate <= 1.0:
        raise ValueError("sample_rate must lie in (0, 1].")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer.")
    if not math.isfinite(noise_multiplier) or noise_multiplier <= 0.0:
        raise ValueError("noise_multiplier must be finite and positive.")
    if not math.isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1).")

    best_epsilon = float("inf")
    best_alpha = float("nan")
    for alpha in orders:
        alpha = float(alpha)
        if not math.isfinite(alpha) or alpha <= 1.0:
            raise ValueError("Every Renyi order must be finite and greater than one.")
        rdp = steps * _rdp_at_order(
            sample_rate,
            noise_multiplier,
            alpha,
        )
        epsilon = (
            rdp
            - (math.log(delta) + math.log(alpha)) / (alpha - 1.0)
            + math.log((alpha - 1.0) / alpha)
        )
        if epsilon < best_epsilon:
            best_epsilon = epsilon
            best_alpha = alpha
    return best_epsilon, best_alpha
