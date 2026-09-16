"""Exploratory owner-level accountant for example/window-level sampling.

This module is a diagnostic bridge toward tight ELS-style owner accounting. It
models one owner with at most ``k`` selected windows under independent
window-level Poisson sampling. In one step, the owner's changed contribution is a
binomial mixture of Gaussian shifts. We compute RDP numerically with
Gauss-Hermite quadrature in the worst aligned one-dimensional direction.

The routines are intentionally separate from the production certificate path:
they are useful for pressure analysis, but a paper claim should audit the
assumptions against the exact training sampler and accountant definition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from dp_accounting.pld import privacy_loss_distribution as pld_privacy_loss_distribution

from unitdp.accountant import DEFAULT_ALPHAS


@dataclass(frozen=True)
class ElsMixtureConfig:
    """Configuration for the exploratory ELS owner accountant."""

    owner_kappa: int
    sample_rate: float
    noise_multiplier: float
    steps: int
    delta: float = 1e-5
    alphas: list[float] = field(default_factory=lambda: list(DEFAULT_ALPHAS))
    quadrature_nodes: int = 96

    def validate(self) -> None:
        if self.owner_kappa <= 0:
            raise ValueError("owner_kappa must be positive")
        if self.sample_rate <= 0 or self.sample_rate > 1:
            raise ValueError("sample_rate must be in (0, 1]")
        if self.noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be positive")
        if self.steps < 0:
            raise ValueError("steps must be nonnegative")
        if self.delta <= 0 or self.delta >= 1:
            raise ValueError("delta must be in (0, 1)")
        if any(alpha <= 1 for alpha in self.alphas):
            raise ValueError("All RDP orders must exceed one")
        if self.quadrature_nodes < 16:
            raise ValueError("quadrature_nodes must be at least 16")


@dataclass(frozen=True)
class ElsMogPldConfig:
    """Configuration for a PLD Mixture-of-Gaussians owner accountant.

    This uses the `dp_accounting` implementation of Mixture-of-Gaussians privacy
    loss distributions. It models example/window-level Poisson sampling for one
    owner with at most `owner_kappa` selected examples, so the owner's sampled
    sensitivity follows a binomial distribution over {0, ..., owner_kappa}.
    """

    owner_kappa: int
    sample_rate: float
    noise_multiplier: float
    steps: int
    delta: float = 1e-5
    value_discretization_interval: float = 1e-3
    log_mass_truncation_bound: float = -50.0
    tail_mass_truncation: float = 1e-15
    use_connect_dots: bool = True
    sampling_model: str = "poisson"
    population_size: int | None = None
    sample_size: int | None = None

    def validate(self) -> None:
        if self.owner_kappa <= 0:
            raise ValueError("owner_kappa must be positive")
        if self.sampling_model not in {"poisson", "fixed_size"}:
            raise ValueError("sampling_model must be poisson or fixed_size")
        if self.sampling_model == "poisson":
            if self.sample_rate <= 0 or self.sample_rate > 1:
                raise ValueError("sample_rate must be in (0, 1]")
        else:
            if self.population_size is None or self.population_size <= 0:
                raise ValueError("population_size must be positive for fixed_size")
            if self.sample_size is None or self.sample_size <= 0:
                raise ValueError("sample_size must be positive for fixed_size")
            if self.sample_size > self.population_size:
                raise ValueError("sample_size cannot exceed population_size")
            if self.owner_kappa > self.population_size:
                raise ValueError("owner_kappa cannot exceed population_size")
        if self.noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be positive")
        if self.steps < 0:
            raise ValueError("steps must be nonnegative")
        if self.delta <= 0 or self.delta >= 1:
            raise ValueError("delta must be in (0, 1)")
        if self.value_discretization_interval <= 0:
            raise ValueError("value_discretization_interval must be positive")
        if self.log_mass_truncation_bound > 0:
            raise ValueError("log_mass_truncation_bound must be nonpositive")
        if self.tail_mass_truncation <= 0 or self.tail_mass_truncation >= 1:
            raise ValueError("tail_mass_truncation must be in (0, 1)")


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    max_value = np.max(values, axis=axis, keepdims=True)
    shifted = np.exp(values - max_value)
    out = max_value + np.log(np.sum(shifted, axis=axis, keepdims=True))
    if axis is None:
        return np.asarray(out.squeeze())
    return np.squeeze(out, axis=axis)


def _binomial_log_probs(k: int, q: float) -> np.ndarray:
    if q <= 0 or q > 1:
        raise ValueError("q must be in (0, 1]")
    logs: list[float] = []
    for j in range(k + 1):
        if q == 1.0 and j != k:
            logs.append(float("-inf"))
            continue
        if q == 1.0 and j == k:
            logs.append(0.0)
            continue
        log_comb = math.lgamma(k + 1) - math.lgamma(j + 1) - math.lgamma(k - j + 1)
        logs.append(log_comb + j * math.log(q) + (k - j) * math.log1p(-q))
    return np.asarray(logs, dtype=np.float64)


def binomial_sensitivity_distribution(owner_kappa: int, sample_rate: float) -> tuple[list[float], list[float]]:
    """Return sensitivity support/probabilities for ELS owner MoG accounting."""

    if owner_kappa <= 0:
        raise ValueError("owner_kappa must be positive")
    if sample_rate <= 0 or sample_rate > 1:
        raise ValueError("sample_rate must be in (0, 1]")
    sensitivities = [float(value) for value in range(owner_kappa + 1)]
    probabilities = np.exp(_binomial_log_probs(owner_kappa, sample_rate))
    probabilities = probabilities / probabilities.sum()
    return sensitivities, probabilities.astype(float).tolist()


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def hypergeometric_sensitivity_distribution(
    population_size: int,
    owner_kappa: int,
    sample_size: int,
) -> tuple[list[float], list[float]]:
    """Return SRSWOR owner-change sensitivities/probabilities."""

    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if owner_kappa <= 0:
        raise ValueError("owner_kappa must be positive")
    if sample_size <= 0 or sample_size > population_size:
        raise ValueError("sample_size must be in [1, population_size]")
    if owner_kappa > population_size:
        raise ValueError("owner_kappa cannot exceed population_size")
    lower = max(0, sample_size - (population_size - owner_kappa))
    upper = min(owner_kappa, sample_size)
    denominator = _log_comb(population_size, sample_size)
    sensitivities = [float(value) for value in range(upper + 1)]
    probabilities: list[float] = []
    for value in range(upper + 1):
        if value < lower:
            probabilities.append(0.0)
            continue
        log_probability = (
            _log_comb(owner_kappa, value)
            + _log_comb(population_size - owner_kappa, sample_size - value)
            - denominator
        )
        probabilities.append(math.exp(log_probability))
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("invalid hypergeometric distribution")
    probabilities = [probability / total for probability in probabilities]
    return sensitivities, probabilities


def mog_pld_sensitivity_distribution(config: ElsMogPldConfig) -> tuple[list[float], list[float]]:
    """Return the MoG-PLD sensitivity law for the configured loader."""

    config.validate()
    if config.sampling_model == "fixed_size":
        assert config.population_size is not None
        assert config.sample_size is not None
        return hypergeometric_sensitivity_distribution(
            population_size=config.population_size,
            owner_kappa=config.owner_kappa,
            sample_size=config.sample_size,
        )
    return binomial_sensitivity_distribution(config.owner_kappa, config.sample_rate)


def _log_density_ratio(
    xs: np.ndarray,
    owner_kappa: int,
    sample_rate: float,
    noise_multiplier: float,
) -> np.ndarray:
    """Return log(P/Q) at points xs.

    Q is the absent-owner Gaussian N(0, sigma^2). P is the present-owner mixture
    sum_j Bin(k, q)[j] N(j, sigma^2), after normalizing by the clipping norm.
    """

    js = np.arange(owner_kappa + 1, dtype=np.float64)
    log_probs = _binomial_log_probs(owner_kappa, sample_rate)
    sigma_sq = noise_multiplier * noise_multiplier
    terms = (
        log_probs[None, :]
        + (xs[:, None] * js[None, :] / sigma_sq)
        - ((js[None, :] * js[None, :]) / (2.0 * sigma_sq))
    )
    return _logsumexp(terms, axis=1)


def _log_expectation_under_absent(
    power: float,
    owner_kappa: int,
    sample_rate: float,
    noise_multiplier: float,
    quadrature_nodes: int,
) -> float:
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_nodes)
    xs = math.sqrt(2.0) * noise_multiplier * nodes
    log_ratio = _log_density_ratio(xs, owner_kappa, sample_rate, noise_multiplier)
    log_terms = np.log(weights) - 0.5 * math.log(math.pi) + power * log_ratio
    return float(_logsumexp(log_terms))


def per_step_els_mixture_rdp(config: ElsMixtureConfig) -> dict[float, float]:
    """Return per-step owner-level RDP for each alpha.

    The value is the max of D_alpha(P||Q) and D_alpha(Q||P), where P is the
    present-owner binomial mixture and Q is the absent-owner Gaussian.
    """

    config.validate()
    values: dict[float, float] = {}
    for alpha in config.alphas:
        log_pq = _log_expectation_under_absent(
            alpha,
            config.owner_kappa,
            config.sample_rate,
            config.noise_multiplier,
            config.quadrature_nodes,
        )
        log_qp = _log_expectation_under_absent(
            1.0 - alpha,
            config.owner_kappa,
            config.sample_rate,
            config.noise_multiplier,
            config.quadrature_nodes,
        )
        values[float(alpha)] = max(float(log_pq), float(log_qp)) / (alpha - 1.0)
    return values


def epsilon_for_els_mixture(config: ElsMixtureConfig) -> float:
    """Convert composed RDP to epsilon with the simple RDP-to-DP bound."""

    config.validate()
    if config.steps == 0:
        return 0.0
    per_step = per_step_els_mixture_rdp(config)
    epsilons = [
        config.steps * rdp + math.log(1.0 / config.delta) / (alpha - 1.0)
        for alpha, rdp in per_step.items()
    ]
    return float(min(epsilons))


def calibrate_noise_els_mixture(
    target_epsilon: float,
    owner_kappa: int,
    sample_rate: float,
    steps: int,
    delta: float,
    alphas: list[float] | None = None,
    quadrature_nodes: int = 96,
) -> float:
    """Binary-search a noise multiplier for the exploratory ELS accountant."""

    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be positive")
    if steps == 0:
        return math.inf
    orders = alphas or DEFAULT_ALPHAS

    def epsilon_at(noise: float) -> float:
        return epsilon_for_els_mixture(
            ElsMixtureConfig(
                owner_kappa=owner_kappa,
                sample_rate=sample_rate,
                noise_multiplier=noise,
                steps=steps,
                delta=delta,
                alphas=orders,
                quadrature_nodes=quadrature_nodes,
            )
        )

    low = 0.01
    high = 1.0
    while epsilon_at(high) > target_epsilon:
        high *= 2.0
        if high > 2048:
            raise RuntimeError("Could not bracket the target epsilon")

    for _ in range(50):
        mid = (low + high) / 2.0
        if epsilon_at(mid) > target_epsilon:
            low = mid
        else:
            high = mid
    return float(high)


def epsilon_for_els_mog_pld(config: ElsMogPldConfig) -> float:
    """Compute owner epsilon with dp_accounting's MoG PLD implementation."""

    config.validate()
    if config.steps == 0:
        return 0.0
    sensitivities, probabilities = mog_pld_sensitivity_distribution(config)
    pld = pld_privacy_loss_distribution.from_mixture_gaussian_mechanism(
        standard_deviation=config.noise_multiplier,
        sensitivities=sensitivities,
        sampling_probs=probabilities,
        pessimistic_estimate=True,
        value_discretization_interval=config.value_discretization_interval,
        log_mass_truncation_bound=config.log_mass_truncation_bound,
        use_connect_dots=config.use_connect_dots,
    )
    composed = pld.self_compose(
        config.steps,
        tail_mass_truncation=config.tail_mass_truncation,
    )
    return float(composed.get_epsilon_for_delta(config.delta))


def calibrate_noise_els_mog_pld(
    target_epsilon: float,
    owner_kappa: int,
    sample_rate: float,
    steps: int,
    delta: float,
    value_discretization_interval: float = 1e-3,
    log_mass_truncation_bound: float = -50.0,
    tail_mass_truncation: float = 1e-15,
    use_connect_dots: bool = True,
    sampling_model: str = "poisson",
    population_size: int | None = None,
    sample_size: int | None = None,
) -> float:
    """Binary-search noise for the MoG PLD owner accountant."""

    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be positive")
    if steps == 0:
        return math.inf

    def epsilon_at(noise: float) -> float:
        return epsilon_for_els_mog_pld(
            ElsMogPldConfig(
                owner_kappa=owner_kappa,
                sample_rate=sample_rate,
                noise_multiplier=noise,
                steps=steps,
                delta=delta,
                value_discretization_interval=value_discretization_interval,
                log_mass_truncation_bound=log_mass_truncation_bound,
                tail_mass_truncation=tail_mass_truncation,
                use_connect_dots=use_connect_dots,
                sampling_model=sampling_model,
                population_size=population_size,
                sample_size=sample_size,
            )
        )

    low = 0.01
    high = 1.0
    while epsilon_at(high) > target_epsilon:
        high *= 2.0
        if high > 2048:
            raise RuntimeError("Could not bracket the target epsilon")

    for _ in range(40):
        mid = (low + high) / 2.0
        if epsilon_at(mid) > target_epsilon:
            low = mid
        else:
            high = mid
    return float(high)
