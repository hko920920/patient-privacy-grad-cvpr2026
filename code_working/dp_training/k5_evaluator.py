"""Frozen metric and preprocessing primitives for the K5 chest-X-ray evaluator."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_real_p256_grayscale(path: Path) -> Image.Image:
    """Apply the frozen NIH native-mode and full-field P256 information boundary."""
    with Image.open(path) as source:
        require(source.size == (1024, 1024), f"unexpected NIH source geometry: {path.name}")
        if source.mode == "L":
            grayscale = source.copy()
        elif source.mode == "RGBA":
            array = np.asarray(source, dtype=np.uint8)
            require(array.shape == (1024, 1024, 4), f"unexpected RGBA shape: {path.name}")
            require(bool(np.all(array[..., 0] == array[..., 1])), f"non-grayscale RGBA R/G: {path.name}")
            require(bool(np.all(array[..., 0] == array[..., 2])), f"non-grayscale RGBA R/B: {path.name}")
            require(bool(np.all(array[..., 3] == 255)), f"nonopaque RGBA: {path.name}")
            grayscale = Image.fromarray(array[..., 0])
        else:
            raise RuntimeError(f"unsupported NIH source mode {source.mode}: {path.name}")
    resized = grayscale.resize(
        (256, 256),
        resample=Image.Resampling.LANCZOS,
        reducing_gap=None,
    )
    require(resized.mode == "L" and resized.size == (256, 256), "P256 preprocessing drift")
    return resized


def polynomial_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    require(x64.ndim == 2 and y64.ndim == 2, "kernel inputs must be matrices")
    require(x64.shape[1] == y64.shape[1], "kernel feature dimensions differ")
    dimension = x64.shape[1]
    return (x64 @ y64.T / float(dimension) + 1.0) ** 3


def kid_unbiased(x: np.ndarray, y: np.ndarray) -> float:
    """Unbiased degree-three polynomial MMD used for one KID subset."""
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    require(x64.ndim == 2 and y64.ndim == 2, "KID inputs must be matrices")
    require(x64.shape[0] >= 2 and y64.shape[0] >= 2, "KID needs at least two samples")
    k_xx = polynomial_kernel(x64, x64)
    k_yy = polynomial_kernel(y64, y64)
    k_xy = polynomial_kernel(x64, y64)
    m = x64.shape[0]
    n = y64.shape[0]
    within_x = (float(k_xx.sum()) - float(np.trace(k_xx))) / (m * (m - 1))
    within_y = (float(k_yy.sum()) - float(np.trace(k_yy))) / (n * (n - 1))
    cross = float(k_xy.mean())
    return within_x + within_y - 2.0 * cross


def deterministic_kid(
    real: np.ndarray,
    generated: np.ndarray,
    *,
    subset_size: int,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    real64 = np.asarray(real, dtype=np.float64)
    generated64 = np.asarray(generated, dtype=np.float64)
    require(subset_size <= len(real64) and subset_size <= len(generated64), "KID subset too large")
    rng = np.random.Generator(np.random.PCG64(seed))
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        real_index = rng.choice(len(real64), size=subset_size, replace=False)
        generated_index = rng.choice(len(generated64), size=subset_size, replace=False)
        values[index] = kid_unbiased(real64[real_index], generated64[generated_index])
    require(bool(np.isfinite(values).all()), "non-finite KID")
    return {
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=1)),
        "replicates": replicates,
        "subset_size": subset_size,
    }


def pairwise_euclidean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    require(x64.ndim == 2 and y64.ndim == 2 and x64.shape[1] == y64.shape[1], "distance shapes")
    squared = (
        np.sum(x64 * x64, axis=1, keepdims=True)
        + np.sum(y64 * y64, axis=1, keepdims=True).T
        - 2.0 * (x64 @ y64.T)
    )
    return np.sqrt(np.maximum(squared, 0.0))


def prdc(real: np.ndarray, generated: np.ndarray, *, nearest_k: int = 5) -> dict[str, float]:
    real64 = np.asarray(real, dtype=np.float64)
    generated64 = np.asarray(generated, dtype=np.float64)
    require(len(real64) > nearest_k and len(generated64) > nearest_k, "PRDC sample too small")
    real_pairwise = pairwise_euclidean(real64, real64)
    generated_pairwise = pairwise_euclidean(generated64, generated64)
    real_radius = np.partition(real_pairwise, nearest_k, axis=1)[:, nearest_k]
    generated_radius = np.partition(generated_pairwise, nearest_k, axis=1)[:, nearest_k]
    cross = pairwise_euclidean(real64, generated64)
    precision = np.mean(np.any(cross < real_radius[:, None], axis=0))
    recall = np.mean(np.any(cross < generated_radius[None, :], axis=1))
    density = np.mean(np.sum(cross < real_radius[:, None], axis=0) / float(nearest_k))
    coverage = np.mean(np.min(cross, axis=1) < real_radius)
    values = {
        "precision": float(precision),
        "recall": float(recall),
        "density": float(density),
        "coverage": float(coverage),
    }
    require(all(math.isfinite(value) for value in values.values()), "non-finite PRDC")
    return values


def effective_rank(features: np.ndarray) -> float:
    matrix = np.asarray(features, dtype=np.float64)
    require(matrix.ndim == 2 and matrix.shape[0] >= 2, "effective-rank matrix")
    covariance = np.cov(matrix, rowvar=False, ddof=1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    positive_sum = float(eigenvalues.sum())
    require(positive_sum > 0.0 and math.isfinite(positive_sum), "degenerate covariance")
    probabilities = eigenvalues[eigenvalues > 0.0] / positive_sum
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def frechet_distance(real: np.ndarray, generated: np.ndarray) -> float:
    from scipy.linalg import sqrtm

    real64 = np.asarray(real, dtype=np.float64)
    generated64 = np.asarray(generated, dtype=np.float64)
    require(real64.ndim == 2 and generated64.ndim == 2, "Frechet inputs")
    require(real64.shape[1] == generated64.shape[1], "Frechet dimensions")
    mean_real = real64.mean(axis=0)
    mean_generated = generated64.mean(axis=0)
    covariance_real = np.cov(real64, rowvar=False, ddof=1)
    covariance_generated = np.cov(generated64, rowvar=False, ddof=1)
    covariance_mean = sqrtm(covariance_real @ covariance_generated)
    if np.iscomplexobj(covariance_mean):
        require(float(np.max(np.abs(covariance_mean.imag))) <= 1e-6, "Frechet sqrtm imaginary drift")
        covariance_mean = covariance_mean.real
    value = (
        float(np.sum((mean_real - mean_generated) ** 2))
        + float(np.trace(covariance_real))
        + float(np.trace(covariance_generated))
        - 2.0 * float(np.trace(covariance_mean))
    )
    require(math.isfinite(value), "non-finite Frechet distance")
    return max(0.0, value)


def stratified_bootstrap_mean(
    differences: Sequence[float],
    strata: Sequence[str],
    *,
    seed: int,
    replicates: int = 2_000,
) -> dict[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    labels = np.asarray(strata, dtype=object)
    require(values.ndim == 1 and labels.shape == values.shape, "bootstrap shape")
    require(bool(np.isfinite(values).all()), "bootstrap values are not finite")
    groups = [np.flatnonzero(labels == label) for label in sorted(set(labels.tolist()))]
    require(all(len(group) > 0 for group in groups), "empty bootstrap stratum")
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        total = 0.0
        count = 0
        for group in groups:
            sampled = rng.choice(group, size=len(group), replace=True)
            total += float(values[sampled].sum())
            count += len(group)
        draws[replicate] = total / count
    lower, upper = np.quantile(draws, [0.025, 0.975], method="linear")
    return {
        "mean_difference": float(values.mean()),
        "standard_deviation": float(values.std(ddof=1)),
        "lower_95_percentile": float(lower),
        "upper_95_percentile": float(upper),
        "replicates": replicates,
        "strata": len(groups),
    }


def aggregate_by_label(values: Sequence[float], labels: Sequence[str]) -> dict[str, dict[str, float | int]]:
    array = np.asarray(values, dtype=np.float64)
    label_array = np.asarray(labels, dtype=object)
    require(array.shape == label_array.shape, "aggregate shape")
    output: dict[str, dict[str, float | int]] = {}
    for label in sorted(set(label_array.tolist())):
        selected = array[label_array == label]
        output[str(label)] = {
            "count": int(len(selected)),
            "mean": float(selected.mean()),
            "standard_deviation": float(selected.std(ddof=1)) if len(selected) > 1 else 0.0,
        }
    return output


def tree_sha256(files: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest().upper()
