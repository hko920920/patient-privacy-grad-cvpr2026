"""CPU float64 algebra for a frozen, spatially shared denoising residual head.

Layout: channels last; projection P has shape (C, 15); prepend a constant one
to obtain h16. Features have shape (...,64) in [P0*h16,...,P3*h16] order. Residual is
epsilon - frozen_prediction, so corrected prediction is frozen_prediction+phi@W.
Loss means over observations and SUMS the four output channels; divide by four
for conventional elementwise MSE. Images, noise draws and patients are not
interchangeable independent units: explicitly average within each patient first.

DP helpers implement algebra only, not an accountant or a privacy guarantee.
They use reproducible NumPy RNG; a public seed does NOT implement private noise.
Never publish a DP release's noise seed, seed-derived noise, or unprotected Q.
"""

from collections.abc import Mapping
import math
import numpy as np


FEATURE_DIM = 64
OUTPUT_DIM = 4
PROJECTED_CHANNELS = 15
TIME_BASIS_DIM = 4


def _array(value, name, ndim=None):
    out = np.asarray(value, dtype=np.float64)
    if ndim is not None and out.ndim != ndim:
        raise ValueError("%s must have %d dimensions" % (name, ndim))
    if not np.isfinite(out).all():
        raise ValueError("%s must be finite" % name)
    return out


def _positive(value, name, allow_zero=False):
    out = float(value)
    if not math.isfinite(out) or (out < 0 if allow_zero else out <= 0):
        raise ValueError("%s must be finite and %s" %
                         (name, "nonnegative" if allow_zero else "positive"))
    return out


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("%s must be a positive integer" % name)
    if value <= 0:
        raise ValueError("%s must be a positive integer" % name)
    return int(value)


def _symmetric(value, name="A"):
    out = _array(value, name, 2)
    if out.shape[0] != out.shape[1] or out.shape[0] == 0:
        raise ValueError("%s must be nonempty and square" % name)
    scale = max(1.0, float(np.max(np.abs(out))))
    if np.max(np.abs(out - out.T)) > 1e-12 * scale:
        raise ValueError("%s must be symmetric" % name)
    return (out + out.T) * 0.5


def make_channel_projection(in_channels, projected_channels=PROJECTED_CHANNELS, *, seed):
    """Return C x k Gaussian-QR projection with orthonormal columns.

    QR signs are canonicalized by the diagonal of R. No private feature values
    are used. Persist P itself: floating point QR is not promised bitwise equal
    across BLAS implementations or NumPy versions.
    """
    c = _integer(in_channels, "in_channels")
    k = _integer(projected_channels, "projected_channels")
    if k > c:
        raise ValueError("projected_channels cannot exceed in_channels")
    q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((c, k)), mode="reduced")
    signs = np.where(np.diag(r) < 0, -1.0, 1.0)
    return np.asarray(q * signs, dtype=np.float64)


def legendre_time_basis(timesteps, alphas_cumprod, *, logsnr_min, logsnr_max):
    """Return bounded P0..P3 on u=2*(logSNR-min)/(max-min)-1.

    Bounds must be fixed externally (e.g. from the public scheduler) and cover
    queried timesteps. No data-derived normalization, endpoint epsilon, or
    out-of-range clipping is silently introduced. Only roundoff within 1e-12
    of [-1,1] is clipped. Scalar t returns (4,), vector t returns (...,4).
    """
    schedule = _array(alphas_cumprod, "alphas_cumprod", 1)
    ts = np.asarray(timesteps)
    if ts.dtype.kind not in "iu" or ts.dtype.kind == "b":
        raise ValueError("timesteps must be integers")
    if np.any(ts < 0) or np.any(ts >= len(schedule)):
        raise ValueError("timestep outside scheduler")
    alpha = schedule[ts]
    if np.any(alpha <= 0) or np.any(alpha >= 1):
        raise ValueError("queried alpha must be strictly between zero and one")
    lo, hi = float(logsnr_min), float(logsnr_max)
    if not math.isfinite(lo) or not math.isfinite(hi) or not lo < hi:
        raise ValueError("finite ordered public logSNR bounds required")
    logsnr = np.log(alpha) - np.log1p(-alpha)
    u = 2.0 * (logsnr - lo) / (hi - lo) - 1.0
    if np.any(u < -1.0 - 1e-12) or np.any(u > 1.0 + 1e-12):
        raise ValueError("queried logSNR outside declared bounds")
    u = np.clip(u, -1.0, 1.0)
    return np.stack((np.ones_like(u), u, (3*u*u-1)/2,
                     (5*u*u*u-3*u)/2), axis=-1)


def features_from_projected(projected, basis):
    """Build (..., 4*k) basis-major features from (...,k) and (...,4).

    A single basis (4,) broadcasts everywhere. For batched maps (N,H,W,k),
    explicitly supply basis (N,1,1,4); ambiguous automatic axis inference is
    deliberately avoided.
    """
    h = _array(projected, "projected")
    b = _array(basis, "basis")
    if h.ndim < 1 or h.shape[-1] == 0 or b.ndim < 1 or b.shape[-1] != 4:
        raise ValueError("projected needs nonempty channel axis; basis needs four terms")
    if np.any(np.abs(b) > 1.0 + 1e-12):
        raise ValueError("time basis terms must be bounded by one")
    product = b[..., :, None] * h[..., None, :]
    return _array(product.reshape(product.shape[:-2] + (4*h.shape[-1],)), "phi")


def add_constant_channel(projected):
    """Prepend a constant one, e.g. (...,15) -> (...,16); no data normalization."""
    h = _array(projected, "projected")
    if h.ndim < 1 or h.shape[-1] == 0:
        raise ValueError("projected features need a nonempty channel dimension")
    return np.concatenate((np.ones(h.shape[:-1]+(1,), dtype=np.float64), h), axis=-1)


def make_features(hidden, projection, basis, *, prepend_constant=True):
    """Project frozen features, optionally prepend one, then form time features."""
    h = _array(hidden, "hidden")
    p = _array(projection, "projection", 2)
    if h.ndim < 1 or h.shape[-1] != p.shape[0] or p.shape[1] == 0:
        raise ValueError("hidden channel count must match projection rows")
    if not np.allclose(p.T @ p, np.eye(p.shape[1]), rtol=1e-10, atol=1e-10):
        raise ValueError("projection columns must be orthonormal")
    projected = h @ p
    if prepend_constant:
        projected = add_constant_channel(projected)
    return features_from_projected(projected, basis)


def sufficient_statistics(phi, residual):
    """Spatial/observation mean A=phi.T@phi, B=phi.T@r, Q=mean sum(r^2).

    Leading dimensions must match. For patient-equal training first average
    equal-weight image/draw statistics within a patient, then average patients.
    Calling this once on pooled images instead defines an image-weighted loss.
    """
    x = _array(phi, "phi")
    y = _array(residual, "residual")
    if x.ndim < 2 or y.ndim != x.ndim or x.shape[:-1] != y.shape[:-1]:
        raise ValueError("phi and residual leading observation dimensions must match")
    if x.shape[-1] == 0 or y.shape[-1] == 0:
        raise ValueError("feature and output dimensions must be nonempty")
    x, y = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])
    n = _integer(len(x), "observation_count")
    a, b, q = x.T @ x / n, x.T @ y / n, float(np.sum(y*y) / n)
    return {"A": _array(a, "A"), "B": _array(b, "B"),
            "Q": _positive(q, "Q", True), "observation_count": n}


def sufficient_statistics_separable(spatial_features, residual, basis):
    """One draw's time-expanded stats without materializing all 64 features.

    spatial_features already includes the prepended constant and is (...,16).
    basis is one (4,) vector shared by all spatial observations in this draw.
    A64=kron(b*b.T,A16), B64=kron(b[:,None],B16); Q is unchanged. This equals
    sufficient_statistics(features_from_projected(h16,b),r) in real arithmetic;
    different FP64 accumulation orders need not be bitwise identical.
    """
    b = _array(basis, "basis", 1)
    if b.shape != (4,) or np.any(np.abs(b) > 1.0 + 1e-12):
        raise ValueError("basis must have four bounded terms")
    small = sufficient_statistics(spatial_features, residual)
    return {"A": _array(np.kron(np.outer(b, b), small["A"]), "A"),
            "B": _array(np.kron(b[:, None], small["B"]), "B"),
            "Q": small["Q"], "observation_count": small["observation_count"]}


def mean_statistics(statistics):
    """Equal-weight mean of supplied A/B/Q units; NEVER weights by row count.

    Use once per image over its draws, then once per patient over its images
    when counts differ. Metadata unit_count records this call's units only.
    """
    rows = list(statistics)
    if not rows:
        raise ValueError("at least one statistic unit is required")
    aa, bb, qq = [], [], []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("each statistic unit must map A, B and Q")
        a, b = _symmetric(row["A"]), _array(row["B"], "B", 2)
        if a.shape[0] != b.shape[0] or b.shape[1] == 0:
            raise ValueError("incompatible A/B shapes")
        aa.append(a); bb.append(b); qq.append(_positive(row["Q"], "Q", True))
    return {"A": np.mean(np.stack(aa), axis=0),
            "B": np.mean(np.stack(bb), axis=0),
            "Q": float(np.mean(qq)), "unit_count": len(rows)}


def mean_patient_statistics(patient_statistics):
    """Equal-patient A/B/Q objective; each supplied row must already be a patient."""
    result = mean_statistics(patient_statistics)
    result["patient_count"] = result.pop("unit_count")
    return result


def project_psd(a, *, eigenvalue_floor=0.0):
    """Symmetric eigenvalue projection; a data-independent postprocessing rule."""
    a = _symmetric(a)
    floor = _positive(eigenvalue_floor, "eigenvalue_floor", True)
    values, vectors = np.linalg.eigh(a)
    out = (vectors * np.maximum(values, floor)) @ vectors.T
    return (out + out.T) * 0.5


def solve_ridge(a, b, ridge, *, psd_floor=None):
    """Return W=(A+ridge*I)^(-1)B for the channel-sum objective.

    Optional explicit PSD projection changes A and the optimized surrogate.
    No ridge choice, noise accounting, or private hyperparameter fitting occurs.
    """
    a, b = _symmetric(a), _array(b, "B", 2)
    if a.shape[0] != b.shape[0] or b.shape[1] == 0:
        raise ValueError("incompatible A/B shapes")
    lam = _positive(ridge, "ridge", True)
    if psd_floor is not None:
        a = project_psd(a, eigenvalue_floor=psd_floor)
    h = a + lam * np.eye(a.shape[0])
    if np.linalg.eigvalsh(h)[0] <= 0:
        raise ValueError("ridge system must be positive definite; choose explicit PSD handling")
    return _array(np.linalg.solve(h, b), "W")


def loss_from_statistics(a, b, q, w, *, ridge=0.0):
    """Return Q-2<W,B>+<W,AW>+ridge*||W||F^2, without clipping negatives.

    Set ridge=0 for held-out denoising loss. Divide the returned unregularized
    loss by number of output channels (four) for elementwise MSE. Statistical
    cancellation can give tiny negative values; callers should report/audit it.
    """
    a, b, w = _symmetric(a), _array(b, "B", 2), _array(w, "W", 2)
    if w.shape != b.shape or a.shape[0] != w.shape[0]:
        raise ValueError("incompatible A, B and W shapes")
    q, lam = _positive(q, "Q", True), _positive(ridge, "ridge", True)
    value = q - 2*float(np.sum(w*b)) + float(np.sum(w*(a@w))) + lam*float(np.sum(w*w))
    if not math.isfinite(value):
        raise ValueError("loss calculation overflowed")
    return value


def svec(a):
    """Upper triangle in np.triu_indices order; off-diagonals scaled sqrt(2)."""
    a = _symmetric(a)
    i, j = np.triu_indices(len(a))
    return a[i, j] * np.where(i == j, 1.0, math.sqrt(2.0))


def unsvec(vector, dimension):
    """Inverse of svec, preserving Euclidean/Frobenius norm equivalence."""
    d = _integer(dimension, "dimension")
    v = _array(vector, "svec", 1)
    if len(v) != d*(d+1)//2:
        raise ValueError("invalid svec length")
    i, j = np.triu_indices(d)
    values = v / np.where(i == j, 1.0, math.sqrt(2.0))
    a = np.zeros((d, d), dtype=np.float64)
    a[i, j] = values; a[j, i] = values
    return a


def pack_statistics(a, b, *, a0, b0):
    """Join svec(A)/a0 and C-order vec(B)/b0; Q is deliberately excluded."""
    a, b = _symmetric(a), _array(b, "B", 2)
    if a.shape[0] != b.shape[0] or b.shape[1] == 0:
        raise ValueError("incompatible A/B shapes")
    return _array(np.concatenate((svec(a)/_positive(a0, "a0"),
                                 b.ravel(order="C")/_positive(b0, "b0"))), "packed")


def unpack_statistics(vector, dimension=FEATURE_DIM, outputs=OUTPUT_DIM, *, a0, b0):
    """Recover A/B from the declared public scaling and dimensions."""
    d, o = _integer(dimension, "dimension"), _integer(outputs, "outputs")
    v = _array(vector, "vector", 1)
    n = d*(d+1)//2
    if len(v) != n+d*o:
        raise ValueError("invalid joint statistic length")
    return {"A": unsvec(v[:n], d)*_positive(a0, "a0"),
            "B": v[n:].reshape(d, o)*_positive(b0, "b0")}


def clip_patient_statistics(a, b, *, a0, b0, clip_norm):
    """Return an INTERNAL clipped vector and factor; factors are not DP outputs."""
    v = pack_statistics(a, b, a0=a0, b0=b0)
    c = _positive(clip_norm, "clip_norm")
    norm = float(np.linalg.norm(v))
    if not math.isfinite(norm):
        raise ValueError("joint statistic norm overflowed")
    factor = 1.0 if norm <= c else c/norm
    return v*factor, factor


def release_patient_statistics(patient_statistics, *, public_denominator,
                               denominator_is_public, a0, b0, clip_norm, sigma, seed,
                               dimension=FEATURE_DIM, outputs=OUTPUT_DIM):
    """Jointly clip patient A/B, add N(0,(sigma*C)^2 I) to SUM, divide by N0.

    N0 must be a declared public positive integer, independent of actual private
    count; the explicit flag is a caller assertion, not certification. No DP
    epsilon is inferred; replacement/add-remove accounting belongs to caller.
    sigma=0 is allowed only as a nonprivate numerical control. This reproducible
    implementation must not use a public/released noise seed for a DP claim.
    Returned values exclude private Q, observed patient count and clip factors.
    Dimensions are public and explicit, so empty datasets are supported with
    the same noise distribution and denominator as nonempty neighbors.
    """
    n0 = _integer(public_denominator, "public_denominator")
    if denominator_is_public is not True:
        raise ValueError("explicit denominator_is_public=True assertion required")
    c, noise_multiplier = _positive(clip_norm, "clip_norm"), _positive(sigma, "sigma", True)
    _positive(a0, "a0"); _positive(b0, "b0")
    dimension, outputs = _integer(dimension, "dimension"), _integer(outputs, "outputs")
    total = np.zeros(dimension*(dimension+1)//2+dimension*outputs, dtype=np.float64)
    for row in patient_statistics:
        a, b = _symmetric(row["A"]), _array(row["B"], "B", 2)
        if a.shape != (dimension, dimension) or b.shape != (dimension, outputs):
            raise ValueError("all patients must have identical A/B dimensions")
        clipped, _ = clip_patient_statistics(a, b, a0=a0, b0=b0, clip_norm=c)
        total += clipped
    std = noise_multiplier*c
    released = (total + np.random.default_rng(seed).normal(0.0, std, size=total.shape)) / n0
    out = unpack_statistics(released, dimension, outputs, a0=a0, b0=b0)
    out.update({"released_vector": _array(released, "released_vector"),
                "public_denominator": n0, "noise_std_on_scaled_mean": std/n0})
    return out
