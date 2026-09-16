"""NumPy FP64 numerical mechanisms for fixed patient quadratic objectives.

Each patient's A and B already average that patient's images/noise/positions.
The data loss SUMS output channels: gradient = 2*(A_u @ W - B_u).
Divide the unregularized loss by output_channels for conventional MSE; do not
silently divide this gradient while keeping the same ridge/learning rate.

This module does not account for epsilon, certify public calibration, choose
hyperparameters, or run models. Caller fixes public scales, denominator, ridge,
learning rates and initial state before private evaluation. Add/remove user
sensitivity of a clipped SUM is C; replacement needs its own 2C accounting.
Poisson masks are independent per user/step, including empty sampled batches.

Explicit seeds and supplied standard normals support INTERNAL reproducible
experiments. Publishing the noise seed, realized noise, masks or unprotected
diagnostics is not a DP release. NumPy PRNG is not a cryptographic generator.
Default results exclude unprotected patient counts and clipping diagnostics.
No thread/environment settings or file writes are performed here.
"""

import math
import numpy as np


def _finite(x, name):
    a = np.asarray(x, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError(name + " must contain finite values")
    return a


def _scalar(x, name, *, zero=False):
    if isinstance(x, (bool, np.bool_)):
        raise ValueError(name + " must be numeric, not boolean")
    v = float(x)
    if not math.isfinite(v) or (v < 0 if zero else v <= 0):
        raise ValueError(name + (" must be nonnegative" if zero else " must be positive"))
    return v


def _integer(x, name, *, zero=False):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    n = int(x)
    if n < (0 if zero else 1):
        raise ValueError(name + " is outside its allowed range")
    return n


def _patients(A, B):
    a, b = _finite(A, "A"), _finite(B, "B")
    if a.ndim != 3 or b.ndim != 3 or a.shape[1] == 0 or b.shape[2] == 0:
        raise ValueError("A must be (users,d,d), B must be (users,d,outputs)")
    if a.shape[1] != a.shape[2] or a.shape[:2] != b.shape[:2]:
        raise ValueError("patient A/B dimensions do not match")
    if not np.allclose(a, a.swapaxes(1, 2), rtol=1e-12, atol=1e-12):
        raise ValueError("each A must be symmetric up to FP64 accumulation roundoff")
    return (a + a.swapaxes(1, 2)) * 0.5, b


def _weights(W, shape):
    w = _finite(W, "W")
    if w.shape != shape:
        raise ValueError("W has incorrect feature/output dimensions")
    return w.copy()


def _rng(seed, name):
    if seed is None:
        raise ValueError(name + " is required when values are not supplied")
    return np.random.default_rng(_integer(seed, name, zero=True))


def _norm_and_factors(vectors, clip_norm):
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(norms).all():
        raise FloatingPointError("norm overflow before clipping")
    factors = np.ones_like(norms)
    np.divide(clip_norm, norms, out=factors, where=norms > clip_norm)
    return norms, factors


def patient_gradients(A, B, W):
    """Return [users,d,outputs] gradients of channel-SUM data losses.

    This does NOT include ridge, clipping, DP noise or patient aggregation.
    Empty (0,d,d)/(0,d,o) inputs return an empty gradient tensor.
    """
    a, b = _patients(A, B)
    w = _weights(W, b.shape[1:])
    return _finite(2.0 * (np.einsum("nij,jk->nik", a, w, optimize=True) - b), "gradients")


def ssp_release(A, B, *, a0, b0, clip_norm, sigma, public_denominator,
                ridge, eigenvalue_floor, seed=None, standard_normal=None,
                include_diagnostics=False):
    """Joint patient SSP, followed by explicit PSD floor and ridge solve.

    s_u = concat(svec(A_u)/a0, vec_C(B_u)/b0)
    v = (sum_u clip_C(s_u) + sigma*C*Z) / public_denominator
    W = (PSD_floor(unpack_A(v)) + ridge*I)^(-1) unpack_B(v).

    svec uses np.triu_indices order, sqrt(2) on off-diagonal entries. Supply
    EITHER seed OR an explicit standard_normal vector, shape d*(d+1)/2+d*o.
    Noise is generated even for no patients or sigma=0 (nonprivate control).
    A_released is the decoded noisy symmetric matrix BEFORE PSD processing.
    A_projected records the possibly different surrogate actually solved.
    The PSD/ridge rules are public, mandatory parameters; zero is permitted,
    but a singular/nonpositive final system fails instead of adding jitter.
    """
    a, b = _patients(A, B)
    n, d, o = b.shape
    sa, sb = _scalar(a0, "a0"), _scalar(b0, "b0")
    c = _scalar(clip_norm, "clip_norm")
    sig = _scalar(sigma, "sigma", zero=True)
    n0 = _integer(public_denominator, "public_denominator")
    lam = _scalar(ridge, "ridge", zero=True)
    floor = _scalar(eigenvalue_floor, "eigenvalue_floor", zero=True)
    ii, jj = np.triu_indices(d)
    triangular_scale = np.where(ii == jj, 1.0, math.sqrt(2.0))
    na = len(ii)
    packed = np.concatenate((a[:, ii, jj] * triangular_scale / sa,
                             b.reshape(n, d*o) / sb), axis=1)
    norms, factors = _norm_and_factors(packed, c)
    total = (packed * factors[:, None]).sum(axis=0)
    size = na + d*o
    if standard_normal is None:
        z = _rng(seed, "seed").standard_normal(size)
    else:
        if seed is not None:
            raise ValueError("provide standard_normal or seed, not both")
        z = _finite(standard_normal, "standard_normal")
        if z.shape != (size,):
            raise ValueError("standard_normal has incorrect joint statistic shape")
    released = _finite((total + sig*c*z) / n0, "released statistics")
    aa = np.zeros((d, d), dtype=np.float64)
    upper = released[:na] * sa / triangular_scale
    aa[ii, jj] = upper
    aa[jj, ii] = upper
    bb = released[na:].reshape(d, o) * sb
    values, vectors = np.linalg.eigh(aa)
    projected = (vectors * np.maximum(values, floor)) @ vectors.T
    projected = (projected + projected.T) * 0.5
    system = projected + lam*np.eye(d)
    if np.linalg.eigvalsh(system)[0] <= 0:
        raise ValueError("PSD floor plus ridge does not produce a positive definite system")
    w = _finite(np.linalg.solve(system, bb), "W")
    result = dict(W=w, released_vector=released, A_released=aa,
                  A_projected=projected, B_released=bb,
                  public_denominator=n0, a0=sa, b0=sb, clip_norm=c,
                  sigma=sig, ridge=lam, eigenvalue_floor=floor,
                  noise_std_on_scaled_mean=sig*c/n0)
    if include_diagnostics:
        result["diagnostics"] = dict(
            scope="INTERNAL_NONPRIVATE_DIAGNOSTICS_DO_NOT_RELEASE",
            patient_norms=norms, clipping_factors=factors,
            clipped_sum=total, standard_normal=np.asarray(z).copy(),
            patient_count=n, clipped_patients=int((factors < 1).sum()),
            noisy_gram_eigenvalues=values,
            stationarity_norm=float(np.linalg.norm(system @ w - bb)))
    return result


def poisson_sgd(A, B, *, steps, q, public_denominator, clip_norm, sigma,
                ridge, learning_rates, initial_weights=None,
                sampling_seed=None, noise_seed=None, sampling_masks=None,
                standard_normals=None, include_diagnostics=False):
    """Same-head user-Poisson DP-SGD with fixed expected batch denominator.

    g_u=2*(A_u@W-B_u); G=(sum_selected clip_C(g_u)+sigma*C*Z)/(q*N0)
    W_next=W-learning_rate[step]*(G+2*ridge*W).

    Ridge is public postprocessing OUTSIDE patient clipping. A deterministic
    scalar or length-steps learning_rates schedule is required. initial_weights
    defaults to zero; any supplied initialization must be public or previously
    privatized under caller's accounting. No momentum/Adam/early stop is used.

    sampling_masks, when supplied, must be bool[steps,users]; otherwise an
    explicit sampling_seed creates independent Bernoulli(q) per user per step.
    standard_normals must be [steps,d,outputs], or explicit noise_seed generates
    it. Seeds and supplied arrays are mutually exclusive within each stream.
    A supplied mask is for faithful replay/testing, NOT proof of Poisson sampling.
    Noise and ridge are applied even if a batch or the entire dataset is empty.
    No division by the realized batch size, redraw or empty-batch skip occurs.
    clip_norm=+inf is permitted ONLY with sigma=0 for a nonprivate convergence
    control. Its noise scale is explicitly zero, never the undefined 0*inf.
    Returned clip_norm is None with clipping_disabled=True in this control,
    so scalar metadata can be serialized as strict JSON without Infinity.
    """
    a, b = _patients(A, B)
    n, d, o = b.shape
    tmax = _integer(steps, "steps", zero=True)
    rate = _scalar(q, "q")
    if rate > 1:
        raise ValueError("q must be at most one")
    n0 = _integer(public_denominator, "public_denominator")
    sig = _scalar(sigma, "sigma", zero=True)
    c = float(clip_norm)
    if c == math.inf:
        if sig != 0:
            raise ValueError("infinite clip_norm is only a sigma=0 nonprivate control")
    else:
        c = _scalar(clip_norm, "clip_norm")
    noise_std = 0.0 if sig == 0 else sig*c
    lam = _scalar(ridge, "ridge", zero=True)
    lr = _finite(learning_rates, "learning_rates")
    if lr.ndim == 0:
        lr = np.full(tmax, float(lr), dtype=np.float64)
    if lr.shape != (tmax,) or np.any(lr < 0):
        raise ValueError("learning_rates must be nonnegative scalar or length-steps vector")
    w = np.zeros((d, o), dtype=np.float64) if initial_weights is None else _weights(initial_weights, (d,o))
    if sampling_seed is not None and noise_seed is not None and sampling_seed == noise_seed:
        raise ValueError("sampling_seed and noise_seed must select distinct PRNG streams")
    if sampling_masks is None:
        mask_rng = _rng(sampling_seed, "sampling_seed")
        masks = None
    else:
        if sampling_seed is not None:
            raise ValueError("provide sampling_masks or sampling_seed, not both")
        masks = np.asarray(sampling_masks)
        if masks.dtype != np.bool_ or masks.shape != (tmax, n):
            raise ValueError("sampling_masks must have bool dtype and shape (steps,users)")
        mask_rng = None
    if standard_normals is None:
        gaussian_rng = _rng(noise_seed, "noise_seed")
        normals = None
    else:
        if noise_seed is not None:
            raise ValueError("provide standard_normals or noise_seed, not both")
        normals = _finite(standard_normals, "standard_normals")
        if normals.shape != (tmax, d, o):
            raise ValueError("standard_normals must have shape (steps,d,outputs)")
        gaussian_rng = None
    denom = rate*n0
    if not math.isfinite(denom) or denom <= 0:
        raise ValueError("q*public_denominator must be finite and positive")
    if include_diagnostics:
        saved_masks = np.zeros((tmax,n), dtype=bool)
        saved_normals = np.zeros((tmax,d,o), dtype=np.float64)
        history = np.empty((tmax+1,d,o), dtype=np.float64)
        history[0] = w
        counts = np.zeros(tmax, dtype=np.int64)
        clips = np.zeros(tmax, dtype=np.int64)
        max_norm = np.zeros(tmax, dtype=np.float64)
    for t in range(tmax):
        mask = mask_rng.random(n) < rate if masks is None else masks[t]
        z = gaussian_rng.standard_normal((d,o)) if normals is None else normals[t]
        count = int(np.count_nonzero(mask))
        if count and c == math.inf and not include_diagnostics:
            # No clipping: sum the quadratics before multiplication. In exact
            # arithmetic this is the sum of all patient gradients; FP64
            # accumulation order may differ from explicit gradient summation.
            total = 2.0*(a[mask].sum(axis=0) @ w - b[mask].sum(axis=0))
            norms = factors = np.empty(0, dtype=np.float64)
        elif count:
            gradients = _finite(2.0*(np.einsum("nij,jk->nik", a[mask], w,
                                              optimize=True)-b[mask]), "patient gradients")
            norms, factors = _norm_and_factors(gradients.reshape(count,d*o), c)
            total = np.einsum("n,nij->ij", factors, gradients, optimize=True)
        else:
            total = np.zeros((d,o), dtype=np.float64)
            norms = factors = np.empty(0, dtype=np.float64)
        update = (total+noise_std*z)/denom + 2.0*lam*w
        w = _finite(w-lr[t]*update, "SGD iterate")
        if include_diagnostics:
            saved_masks[t], saved_normals[t] = mask, z
            history[t+1] = w
            counts[t], clips[t] = count, int((factors < 1).sum())
            max_norm[t] = float(norms.max()) if count else 0.0
    result = dict(W=w, steps=tmax, q=rate, public_denominator=n0,
                  expected_batch_denominator=denom,
                  clip_norm=None if c == math.inf else c,
                  clipping_disabled=(c == math.inf), sigma=sig,
                  ridge=lam, learning_rates=lr.copy())
    if include_diagnostics:
        result["diagnostics"] = dict(
            scope="INTERNAL_NONPRIVATE_DIAGNOSTICS_DO_NOT_RELEASE",
            sampling_masks=saved_masks, standard_normals=saved_normals,
            weight_history=history, sampled_patients=counts,
            clipped_patients=clips, maximum_sampled_gradient_norm=max_norm,
            empty_batches=int((counts == 0).sum()),
            patient_gradient_evaluations=int(counts.sum()))
    return result
