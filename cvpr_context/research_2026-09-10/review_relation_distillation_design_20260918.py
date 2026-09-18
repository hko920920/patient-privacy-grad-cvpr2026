"""Independent algebra/design checks. No image, checkpoint, or GPU access."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parent
CODE = ROOT.parents[1] / "code_working"


def outer_mean(x):
    return x.T @ x / len(x)


def svec(a):
    i, j = np.triu_indices(a.shape[0])
    return a[i, j] * np.where(i == j, 1.0, math.sqrt(2.0))


def patient_stats(images_by_class, d):
    ms, As, present = [], [], []
    for c in (0, 1):
        x = images_by_class.get(c)
        yes = x is not None and len(x) > 0
        present.append(int(yes))
        ms.append(x.mean(0) if yes else np.zeros(d))
        As.append(outer_mean(x) if yes else np.zeros((d, d)))
    h = present[0] * present[1]
    delta = (ms[1] - ms[0]) / 2 if h else np.zeros(d)
    C = np.outer(delta, delta)
    q = np.concatenate([present, [h], svec(As[0]), ms[0],
                        svec(As[1]), ms[1], svec(C), delta])
    return present, ms, As, h, delta, C, q


def moments(patient_list, d):
    stats = [patient_stats(p, d) for p in patient_list]
    m = [np.mean([s[1][c] for s in stats if s[0][c]], 0) for c in (0, 1)]
    A = [np.mean([s[2][c] for s in stats if s[0][c]], 0) for c in (0, 1)]
    delta = np.mean([s[4] for s in stats if s[3]], 0)
    C = np.mean([s[5] for s in stats if s[3]], 0)
    return m, A, delta, C, stats


def mean_second_moment_postprocess(m_raw, A_raw):
    # One valid deterministic construction, not the nearest projection.
    m = m_raw / max(1.0, np.linalg.norm(m_raw))
    S = (A_raw + A_raw.T) / 2 - np.outer(m, m)
    eig, V = np.linalg.eigh(S)
    eig = np.maximum(eig, 0)
    room = max(0.0, 1.0 - float(m @ m))
    if eig.sum() > room:
        eig *= room / eig.sum()
    return m, np.outer(m, m) + (V * eig) @ V.T


def main():
    rng = np.random.default_rng(20260918)
    d, beta, lam = 3, .7, .2
    pats = []
    for counts in [(2, 1), (1, 3), (0, 1), (3, 0), (2, 2)]:
        p = {}
        for c, n in enumerate(counts):
            if n:
                x = rng.normal(size=(n, d))
                x /= np.maximum(1.0, np.linalg.norm(x, axis=1))[:, None]
                p[c] = x
        pats.append(p)
    m, A, delta, C, stats = moments(pats, d)
    H = (A[0] + A[1]) / 2 + beta * C
    v = (m[1] - m[0]) / 2 + beta * delta
    errors = []
    for _ in range(10):
        w = rng.normal(size=d)
        direct_point = sum(
            np.mean([np.mean((p[c] @ w - (2*c-1))**2)
                     for p in pats if c in p]) / 4 for c in (0, 1)
        )
        direct_relation = beta/2 * np.mean([(w @ s[4] - 1)**2 for s in stats if s[3]])
        direct = direct_point + direct_relation + lam/2 * float(w @ w)
        quadratic = .5*w @ H @ w - v @ w + lam/2*(w @ w) + .5 + beta/2
        errors.append(abs(direct-quadratic))
    assert max(errors) < 1e-12

    # Sensitivity witness: both classes, opposite unit vectors.
    e = np.eye(16)[0]
    witness = patient_stats({0: -e[None], 1: e[None]}, 16)[-1]
    assert len(witness) == 459 and abs(np.linalg.norm(witness)-3) < 1e-12

    # Fixed class marginals and fixed mixed-patient population; only pairing changes.
    pos = np.array([[.8, .1], [.1, .8]])
    neg = np.array([[.4, .1], [.1, .4]])
    dt = (pos-neg)/2
    ds = (pos-neg[::-1])/2
    Ct, Cs = outer_mean(dt), outer_mean(ds)
    assert np.allclose(dt.mean(0), ds.mean(0))
    w = np.array([1., -1.])
    rt = .5 * np.mean((dt @ w - 1)**2)
    rs = .5 * np.mean((ds @ w - 1)**2)

    # Per-patient centroid loss excludes within-visit variation.
    xp = np.array([[.8, 0], [.2, 0]])
    xn = np.array([[0, .8], [0, .2]])
    md = (xp.mean(0)-xn.mean(0))/2
    vp = outer_mean(xp-xp.mean(0))
    vn = outer_mean(xn-xn.mean(0))
    centroid_loss = float((md @ w - 1)**2)
    visit_loss = float(np.mean([((p-n)/2 @ w - 1)**2 for p in xp for n in xn]))
    variance_term = float(w @ (vp+vn) @ w / 4)
    assert abs(visit_loss-centroid_loss-variance_term) < 1e-12

    # Entire-class and mixed-only means need not match.
    one_dim = [{0: np.array([[0.]]), 1: np.array([[.4]])},
               {1: np.array([[1.]])}, {0: np.array([[-1.]])}]
    mm, _, dd, _, _ = moments(one_dim, 1)
    all_half_gap = float((mm[1]-mm[0])[0]/2)
    mixed_gap = float(dd[0])
    assert abs(all_half_gap-mixed_gap) > .1

    # The quoted optimizer perturbation bound needs PSD Hs, positive ridge.
    T = rng.normal(size=(d,d))
    Hs = T.T @ T / 10
    vs = rng.normal(size=d)
    wd = np.linalg.solve(H+lam*np.eye(d), v)
    ws = np.linalg.solve(Hs+lam*np.eye(d), vs)
    lhs = float(np.linalg.norm(ws-wd))
    bound = float((np.linalg.norm(vs-v)+np.linalg.norm(Hs-H,2)*np.linalg.norm(wd))/lam)
    assert lhs <= bound + 1e-12

    raw_m = np.array([.4, .2])
    raw_A = np.array([[-.1, .4], [.4, .1]])
    pp_m, pp_A = mean_second_moment_postprocess(raw_m, raw_A)
    assert np.linalg.eigvalsh(pp_A-np.outer(pp_m,pp_m)).min() > -1e-12
    assert np.trace(pp_A) <= 1+1e-12

    eps, privacy_delta, sensitivity = 8., 1e-5, 3.
    def gaussian_profile(sig):
        a, b = sensitivity/(2*sig), eps*sig/sensitivity
        return ndtr(a-b)-math.exp(eps)*ndtr(-a-b)
    sigma = brentq(lambda sig: gaussian_profile(sig)-privacy_delta, .03, 30.)
    noise = {
        str(k): {
            "joint_dimension": 3+3*(k*(k+1)//2+k),
            "relation_noise_rms_l2_over_102_excluding_count": sigma*math.sqrt(k*(k+1)//2+k)/102,
            "relation_noise_rms_l2_over_105_excluding_count": sigma*math.sqrt(k*(k+1)//2+k)/105,
            "per_coordinate_std_over_102_excluding_count": sigma/102,
            "interpretation": "sqrt(E ||noise||_2^2), not per-coordinate RMS, signal-to-noise ratio, or noisy-count normalized error"
        } for k in (16,32)
    }

    roster = CODE / "_reports/real_support_plan_20260917_v1/expanded_training_manifest_private.csv"
    grouped, nimages = defaultdict(lambda: defaultdict(set)), defaultdict(int)
    with roster.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            src = r["planned_namespace"]
            grouped[src][r["patient_id"]].add(int(r["label"]))
            nimages[src] += 1
    counts = {
        src: dict(patients=len(ids), images=nimages[src],
                  positive_patients=sum(1 in y for y in ids.values()),
                  mixed_patients=sum(y == {0,1} for y in ids.values()))
        for src, ids in grouped.items()
    }
    assert counts["diagnostic_real/former_classifier_selection"]["mixed_patients"] == 102
    result = {
        "scope": "Independent construction from pasted proposal; random/toy arrays plus existing CSV metadata only. No patient pixels, encoder, checkpoint, GPU, classifier, synthesis or actual DP release.",
        "risk_expansion_max_error": max(errors),
        "sensitivity_witness_norm": float(np.linalg.norm(witness)),
        "sensitivity_witness_dimension": len(witness),
        "permuted_pairing": {
            "delta_mean_max_change": float(np.max(np.abs(dt.mean(0)-ds.mean(0)))),
            "C_operator_change": float(np.linalg.norm(Ct-Cs,2)),
            "true_pair_risk": float(rt), "permuted_pair_risk": float(rs),
            "interpretation": "With fixed mixed class lists, v is invariant; pair information changes H through C."
        },
        "centroid_vs_visit_pair_risk": {
            "centroid_square_loss": centroid_loss,
            "all_visit_pairs_square_loss": visit_loss,
            "missing_within_visit_variance_term": variance_term
        },
        "two_population_means": {"all_class_half_gap":all_half_gap, "mixed_half_gap":mixed_gap},
        "parameter_perturbation_bound": {"actual_distance":lhs, "bound":bound},
        "moment_postprocess": {
            "raw_min_eigenvalue": float(np.linalg.eigvalsh(raw_A).min()),
            "processed_covariance_min_eigenvalue": float(np.linalg.eigvalsh(pp_A-np.outer(pp_m,pp_m)).min()),
            "processed_trace":float(np.trace(pp_A)),
            "claim": "PSD/mean/trace feasibility only; encoder-image realizability is not proved."
        },
        "gaussian": {"epsilon":eps,"delta":privacy_delta,"sensitivity":sensitivity,
                     "sigma_sum":sigma,"delta_recomputed":float(gaussian_profile(sigma)),
                     "point_only_sensitivity":math.sqrt(6),
                     "point_only_sigma_sum":sigma*math.sqrt(6)/3,"dimensions":noise},
        "roster_counts":counts,
        "rank_limit": {"32_uniform_samples_max_centered_covariance_rank":31,
                       "d32_full_rank_covariance_not_exactly_representable_by_32_samples":True}
    }
    out = ROOT / "spec_sources/relation_distillation_design_review_20260918.json"
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
