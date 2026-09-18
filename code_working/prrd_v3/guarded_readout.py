"""Differentiable guarded quadratic readout; no data loader or DP mechanism.

This is the section19 learner core. The supplied point matrix must already be
SPD. A future noisy-summary adapter must apply its declared repair BEFORE this
module; no unaccounted private access or implicit data-dependent repair occurs.
"""
from dataclasses import dataclass
import math
import torch


@dataclass
class GuardedReadout:
    weight: torch.Tensor
    point_weight: torch.Tensor
    relation_weight: torch.Tensor
    point_matrix: torch.Tensor
    alpha: torch.Tensor
    radius_squared: torch.Tensor
    proposed_shift_squared: torch.Tensor
    guard_enabled: bool


@dataclass(frozen=True)
class FunctionalTarget:
    weight: torch.Tensor
    matrix: torch.Tensor


def _system(matrix, linear):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected a square point/readout matrix")
    if linear.shape != (matrix.shape[0],):
        raise ValueError("Linear term has incompatible dimensions")
    if matrix.dtype not in (torch.float32, torch.float64):
        raise ValueError("Use FP32 or FP64 readout arithmetic")
    if matrix.device != linear.device or matrix.dtype != linear.dtype:
        raise ValueError("Matrix and linear term must share dtype and device")
    if not bool(torch.isfinite(matrix).all() and torch.isfinite(linear).all()):
        raise ValueError("Nonfinite quadratic system")
    if not torch.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-6):
        raise ValueError("Quadratic matrix is not symmetric")
    matrix = (matrix + matrix.T) / 2
    _, info = torch.linalg.cholesky_ex(matrix)
    if int(info.detach()) != 0:
        raise ValueError("Quadratic matrix must be SPD; apply declared repair upstream")
    return matrix, linear


def guarded_readout(matrix, linear, relation_second=None, relation_mean=None,
                    *, beta=1.0, rho=None, kappa=None, guard=True):
    """Compute point and relational solutions, then limit the relational shift.

    Supply exactly one radius policy:
      rho: explicit nonnegative radius;
      kappa: rho^2 = kappa * w0^T M w0.
    The relative policy bounds point-objective degradation by kappa times
    F0(0)-F0(w0). Synthetic and recipient learners use their OWN point matrices.
    It is a radial shrinkage rule, not a constrained quadratic minimizer.
    """
    if (rho is None) == (kappa is None):
        raise ValueError("Specify exactly one of rho and kappa")
    if not math.isfinite(float(beta)) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    value = rho if rho is not None else kappa
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError("Radius policy must be finite and nonnegative")
    matrix, linear = _system(matrix, linear)
    w0 = torch.linalg.solve(matrix, linear)
    if (relation_second is None) != (relation_mean is None):
        raise ValueError("Supply both relation moments or neither")
    if relation_second is None or beta == 0:
        w1 = w0
    else:
        if relation_second.shape != matrix.shape or relation_mean.shape != linear.shape:
            raise ValueError("Relation moments have incompatible dimensions")
        if (relation_second.dtype != matrix.dtype or relation_mean.dtype != linear.dtype
                or relation_second.device != matrix.device or relation_mean.device != linear.device):
            raise ValueError("Relation moments must share dtype and device")
        km, kv = _system(matrix + beta * relation_second, linear + beta * relation_mean)
        w1 = torch.linalg.solve(km, kv)
    d = w1 - w0
    energy = (d @ matrix @ d).clamp_min(0)
    radius2 = (matrix.new_tensor(float(rho) ** 2) if rho is not None
               else float(kappa) * (w0 @ matrix @ w0).clamp_min(0))
    # Avoid evaluating sqrt(0)/0 even in an unused torch.where branch.
    # The policy is piecewise differentiable; ties use the identity branch.
    if not guard or bool((energy <= radius2).detach()):
        alpha = energy.new_ones(())
    elif bool((radius2 == 0).detach()):
        alpha = energy.new_zeros(())
    else:
        alpha = torch.sqrt(radius2 / energy)
    return GuardedReadout(w0 + alpha * d, w0, w1, matrix,
                          alpha, radius2, energy, bool(guard))


def readout_from_moments(moments, *, beta=1.0, ridge=0.1,
                         rho=None, kappa=None, guard=True):
    """Use existing class/patient-weighted point and relation moment semantics.

    Upstream defines empty populations, point and raw-relation transformations,
    and all global denominators. This function does not redefine those weights.
    """
    if not math.isfinite(float(ridge)) or ridge <= 0:
        raise ValueError("ridge must be finite and strictly positive")
    means, seconds = moments["m"], moments["A"]
    if means.ndim != 2 or means.shape[0] != 2:
        raise ValueError("Expected two class means")
    d = means.shape[1]
    if seconds.shape != (2, d, d):
        raise ValueError("Expected two class second moments")
    matrix = seconds.mean(0) + ridge * torch.eye(d, device=seconds.device, dtype=seconds.dtype)
    linear = (means[1] - means[0]) / 2
    return guarded_readout(matrix, linear, moments.get("C"), moments.get("delta"),
                           beta=beta, rho=rho, kappa=kappa, guard=guard)


def freeze_functional_target(readout):
    """Copy fixed target quantities; the synthetic path must not optimize them."""
    return FunctionalTarget(readout.weight.detach().clone(),
                            readout.point_matrix.detach().clone())


def functional_loss(synthetic_weight, target):
    """Fixed target-M distance, not the synthetic learner's moving metric."""
    matrix, weight = _system(target.matrix, target.weight)
    if matrix.requires_grad or weight.requires_grad:
        raise ValueError("Functional target must be frozen")
    if synthetic_weight.shape != weight.shape:
        raise ValueError("Source and synthetic readouts have different dimensions")
    if synthetic_weight.dtype != weight.dtype or synthetic_weight.device != weight.device:
        raise ValueError("Functional readouts must share dtype and device")
    delta = synthetic_weight - weight
    return delta @ matrix @ delta
