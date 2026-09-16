"""High-precision oracle for the V4 direct random-allocation accountant.

The oracle intentionally uses exact integer coefficients and direct
high-precision summation rather than the production log-sum-exp path.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Iterator

import mpmath as mp


@dataclass(frozen=True)
class RandomAllocationOracleResultV4:
    epsilon: float
    epsilon_remove: float
    epsilon_add: float
    optimal_remove_order: int
    remove_rdp_by_order: tuple[tuple[int, float], ...]


def _ascending_partitions(
    total: int,
    *,
    minimum: int = 1,
) -> Iterator[tuple[int, ...]]:
    if total == 0:
        yield ()
        return
    for first in range(minimum, total + 1):
        for suffix in _ascending_partitions(
            total - first,
            minimum=first,
        ):
            yield (first, *suffix)


def _coefficient(
    *,
    partition: tuple[int, ...],
    alpha: int,
    steps: int,
) -> int:
    counts = Counter(partition)
    choose_positions = math.factorial(steps) // math.factorial(
        steps - len(partition)
    )
    for count in counts.values():
        choose_positions //= math.factorial(count)
    alpha_multinomial = math.factorial(alpha)
    for part in partition:
        alpha_multinomial //= math.factorial(part)
    return choose_positions * alpha_multinomial


def remove_rdp_oracle_v4(
    *,
    alpha: int,
    sigma: float,
    steps_per_single_allocation: int,
    decimal_digits: int = 80,
) -> float:
    if alpha < 2 or steps_per_single_allocation < 1 or sigma <= 0:
        raise ValueError("invalid high-precision RDP query")
    with mp.workdps(decimal_digits):
        mp_sigma = mp.mpf(str(sigma))
        total = mp.mpf("0")
        for partition in _ascending_partitions(alpha):
            if len(partition) > steps_per_single_allocation:
                continue
            coefficient = _coefficient(
                partition=partition,
                alpha=alpha,
                steps=steps_per_single_allocation,
            )
            squared_sum = sum(part * part for part in partition)
            total += mp.mpf(coefficient) * mp.exp(
                mp.mpf(squared_sum)
                / (2 * mp_sigma * mp_sigma)
            )
        value = (
            mp.log(total)
            - mp.mpf(alpha)
            * (
                1 / (2 * mp_sigma * mp_sigma)
                + mp.log(steps_per_single_allocation)
            )
        ) / (alpha - 1)
        return float(value)


def _normal_cdf(value: mp.mpf) -> mp.mpf:
    return mp.erfc(-value / mp.sqrt(2)) / 2


def _gaussian_delta(
    *,
    epsilon: mp.mpf,
    sigma: mp.mpf,
) -> mp.mpf:
    return _normal_cdf(
        mp.mpf("0.5") / sigma - sigma * epsilon
    ) - mp.exp(epsilon) * _normal_cdf(
        -mp.mpf("0.5") / sigma - sigma * epsilon
    )


def _gaussian_epsilon(
    *,
    sigma: mp.mpf,
    delta: mp.mpf,
) -> mp.mpf:
    if _gaussian_delta(epsilon=mp.mpf("0"), sigma=sigma) <= delta:
        return mp.mpf("0")
    lower = mp.mpf("0")
    upper = max(
        mp.mpf("1"),
        mp.sqrt(2 * mp.log(mp.mpf("1.25") / delta)) / sigma,
    )
    while _gaussian_delta(epsilon=upper, sigma=sigma) > delta:
        upper *= 2
    for _ in range(280):
        midpoint = (lower + upper) / 2
        if _gaussian_delta(epsilon=midpoint, sigma=sigma) > delta:
            lower = midpoint
        else:
            upper = midpoint
    return upper


def account_random_allocation_oracle_v4(
    *,
    num_steps: int,
    num_selected: int,
    num_epochs: int,
    sigma: float,
    delta: float,
    orders: Iterable[int],
    decimal_digits: int = 80,
) -> RandomAllocationOracleResultV4:
    if (
        num_steps < 1
        or num_selected < 1
        or num_selected > num_steps
        or num_epochs < 1
        or sigma <= 0
        or not 0 < delta < 1
    ):
        raise ValueError("invalid high-precision accounting query")
    normalized_orders = tuple(orders)
    if (
        not normalized_orders
        or tuple(sorted(set(normalized_orders))) != normalized_orders
        or any(order < 2 for order in normalized_orders)
    ):
        raise ValueError("invalid high-precision order grid")
    single_steps = num_steps // num_selected
    compositions = num_selected * num_epochs
    remove_rows: list[tuple[int, float]] = []
    remove_candidates: list[tuple[mp.mpf, int]] = []
    with mp.workdps(decimal_digits):
        mp_delta = mp.mpf(str(delta))
        for alpha in normalized_orders:
            composed = mp.mpf(compositions) * mp.mpf(
                str(
                    remove_rdp_oracle_v4(
                        alpha=alpha,
                        sigma=sigma,
                        steps_per_single_allocation=single_steps,
                        decimal_digits=decimal_digits,
                    )
                )
            )
            remove_rows.append((alpha, float(composed)))
            correction = max(
                mp.log(1 - mp.mpf(1) / alpha)
                - mp.log(mp_delta * alpha) / (alpha - 1),
                mp.mpf("0"),
            )
            remove_candidates.append((composed + correction, alpha))
        epsilon_remove_mp, optimal_order = min(remove_candidates)

        mp_sigma = mp.mpf(str(sigma))
        effective_sigma = mp_sigma * mp.sqrt(
            mp.mpf(single_steps) / compositions
        )
        gaussian_epsilon = _gaussian_epsilon(
            sigma=effective_sigma,
            delta=mp_delta,
        )
        shift = (
            mp.mpf(compositions)
            * (1 - mp.mpf(1) / single_steps)
            / (2 * mp_sigma * mp_sigma)
        )
        epsilon_add_mp = gaussian_epsilon + shift
        epsilon_mp = max(epsilon_remove_mp, epsilon_add_mp)
        return RandomAllocationOracleResultV4(
            epsilon=float(epsilon_mp),
            epsilon_remove=float(epsilon_remove_mp),
            epsilon_add=float(epsilon_add_mp),
            optimal_remove_order=optimal_order,
            remove_rdp_by_order=tuple(remove_rows),
        )
