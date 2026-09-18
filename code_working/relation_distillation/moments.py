"""Feature-only patient contributions and quadratic readouts.

No file access, model loading, image decoding or random privacy release occurs
in this module. Raw contributions are private internal state, not DP outputs.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import brentq
from scipy.special import log_ndtr, ndtr


def checked_features(z):
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or not len(z) or not z.shape[1] or not np.isfinite(z).all():
        raise ValueError('Expected a nonempty finite feature matrix')
    if np.linalg.norm(z, axis=1).max() > 1.0 + 1e-7:
        raise ValueError('Feature norm exceeds the public unit bound')
    return z


def svec(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError('Expected square matrix')
    if not np.isfinite(a).all() or not np.allclose(a, a.T, atol=1e-12, rtol=0):
        raise ValueError('Expected finite symmetric matrix')
    i, j = np.triu_indices(len(a))
    return a[i, j] * np.where(i == j, 1.0, math.sqrt(2.0))


def smat(v, d):
    v = np.asarray(v, dtype=np.float64)
    if v.shape != (d * (d + 1) // 2,) or not np.isfinite(v).all():
        raise ValueError('Invalid symmetric vector')
    i, j = np.triu_indices(d)
    a = np.zeros((d, d), dtype=np.float64)
    a[i, j] = v / np.where(i == j, 1.0, math.sqrt(2.0))
    a[j, i] = a[i, j]
    return a


@dataclass(frozen=True)
class PatientContribution:
    patient_id: str
    present: np.ndarray
    means: np.ndarray
    seconds: np.ndarray
    delta: np.ndarray

    @property
    def mixed(self):
        return int(self.present.prod())

    def vector(self, relation=True):
        parts = [self.present.astype(float)]
        if relation:
            parts.append(np.array([self.mixed], dtype=float))
        for c in (0, 1):
            parts.extend([svec(self.seconds[c]), self.means[c]])
        if relation:
            parts.extend([svec(np.outer(self.delta, self.delta)), self.delta])
        q = np.concatenate(parts)
        bound = 3.0 if relation else math.sqrt(6.0)
        if np.linalg.norm(q) > bound + 1e-6:
            raise ValueError('Patient contribution exceeds analytical bound')
        return q


def patient_contributions(features, labels, patient_ids):
    z = checked_features(features)
    y = np.asarray(labels)
    ids = np.asarray(patient_ids, dtype=str)
    if y.shape != (len(z),) or ids.shape != y.shape or not np.isin(y, [0, 1]).all():
        raise ValueError('Invalid labels or patient IDs')
    if np.any(ids == ''):
        raise ValueError('Empty patient identifier')
    result = []
    for pid in sorted(set(ids)):
        means = np.zeros((2, z.shape[1]))
        seconds = np.zeros((2, z.shape[1], z.shape[1]))
        present = np.zeros(2, dtype=int)
        for c in (0, 1):
            x = z[(ids == pid) & (y == c)]
            if len(x):
                present[c] = 1
                means[c] = x.mean(axis=0)
                seconds[c] = x.T @ x / len(x)
        delta = (means[1] - means[0]) / 2 if present.prod() else np.zeros(z.shape[1])
        result.append(PatientContribution(pid, present, means, seconds, delta))
    return result


def sum_query(patients, relation=True):
    if not patients:
        raise ValueError('No patients supplied')
    if len({p.patient_id for p in patients}) != len(patients):
        raise ValueError('Repeated patient contribution')
    return np.stack([p.vector(relation) for p in patients]).sum(axis=0)


def repair_mean_second(m, a):
    """Deterministic moment repair, not nearest projection or image feasibility."""
    m = np.asarray(m, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    if a.shape != (len(m), len(m)) or not np.isfinite(m).all() or not np.isfinite(a).all():
        raise ValueError('Invalid moments')
    m = m / max(1.0, float(np.linalg.norm(m)))
    covariance = (a + a.T) / 2 - np.outer(m, m)
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, 0)
    room = max(0.0, 1.0 - float(m @ m))
    if values.sum() > room:
        values *= room / values.sum()
    return m, np.outer(m, m) + (vectors * values) @ vectors.T


def decode_sums(q, d, relation=True, noisy=False, repair=False):
    """Normalize *combined* exact-public + exact/noisy-private sums once.

    Noisy counts use the predeclared floor of one; mixed count is then capped
    by both class counts. This is postprocessing, not a true-count estimate.
    """
    q = np.asarray(q, dtype=np.float64)
    expected = (3 if relation else 2) + (3 if relation else 2) * (d * (d + 1) // 2 + d)
    if q.shape != (expected,) or not np.isfinite(q).all():
        raise ValueError('Query layout or finite-value mismatch')
    ncount = 3 if relation else 2
    counts = q[:ncount].copy()
    if noisy:
        counts[:2] = np.maximum(counts[:2], 1.0)
        if relation:
            counts[2] = np.clip(counts[2], 1.0, counts[:2].min())
    elif np.any(counts[:2] <= 0) or (relation and not 0 < counts[2] <= counts[:2].min()):
        raise ValueError('Exact target requires populated classes and relation population')
    k = ncount
    blocks = []
    for n in counts:
        size = d * (d + 1) // 2
        a = smat(q[k:k+size], d) / n
        k += size
        m = q[k:k+d] / n
        k += d
        if repair:
            m, a = repair_mean_second(m, a)
        blocks.append((m, a))
    return {'counts': counts, 'm': np.stack([b[0] for b in blocks[:2]]),
            'A': np.stack([b[1] for b in blocks[:2]]),
            'delta': blocks[2][0] if relation else np.zeros(d),
            'C': blocks[2][1] if relation else np.zeros((d, d))}


def shuffled_relation(private_patients, public_patients, permutation):
    """Non-DP control. Shuffle private negative centroids; public pairs fixed."""
    ids = [p.patient_id for p in private_patients + public_patients]
    if len(ids) != len(set(ids)):
        raise ValueError('Public/private patient contributions must be disjoint and unique')
    private = [p for p in private_patients if p.mixed]
    public = [p for p in public_patients if p.mixed]
    perm = np.asarray(permutation)
    if perm.shape != (len(private),) or sorted(perm.tolist()) != list(range(len(private))):
        raise ValueError('Permutation must preserve exactly the private mixed population')
    if np.any(perm == np.arange(len(private))):
        raise ValueError('Protocol uses a derangement, without self pairs')
    delta = np.stack([(p.means[1] - private[j].means[0]) / 2
                      for p, j in zip(private, perm)] + [p.delta for p in public])
    return delta.mean(axis=0), delta.T @ delta / len(delta)


def risk_system(target, beta=1.0):
    if beta < 0 or not math.isfinite(beta):
        raise ValueError('Invalid relation weight')
    h = (target['A'][0] + target['A'][1]) / 2 + beta * target['C']
    v = (target['m'][1] - target['m'][0]) / 2 + beta * target['delta']
    return h, v


def solve_readout(target, beta=1.0, ridge=0.01):
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError('Positive finite ridge required')
    h, v = risk_system(target, beta)
    if not np.allclose(h, h.T, atol=1e-10, rtol=0):
        raise ValueError('Nonsymmetric Hessian')
    regularized = (h + h.T) / 2 + ridge * np.eye(len(v))
    if np.linalg.eigvalsh(regularized).min() <= 0:
        raise ValueError('No positive-definite readout; repair or frozen stabilization required')
    return np.linalg.solve(regularized, v)


def analytic_gaussian_std(epsilon, delta, sensitivity):
    """Calibrate the scalar Gaussian std for add/remove sensitivity.

    This calculates a scale only. It does not draw or publish DP noise.
    """
    if not (math.isfinite(epsilon) and epsilon > 0 and 0 < delta < 1
            and math.isfinite(sensitivity) and sensitivity > 0):
        raise ValueError('Invalid privacy parameters')
    def profile(ratio):
        a, b = 0.5 / ratio, epsilon * ratio
        return float(ndtr(a-b) - np.exp(epsilon + log_ndtr(-a-b)))
    lo, hi = 1e-8, 1.0
    while profile(hi) > delta:
        hi *= 2
    return sensitivity * brentq(lambda r: profile(r) - delta, lo, hi, xtol=1e-13)
