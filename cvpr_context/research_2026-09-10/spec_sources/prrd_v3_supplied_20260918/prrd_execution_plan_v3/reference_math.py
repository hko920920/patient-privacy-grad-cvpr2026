#!/usr/bin/env python3
"""PRRD planning reference: CPU algebra, not a medical/production DP implementation.

Run: python reference_math.py --output reference_checks.json
Requires numpy and scipy. A small optional torch test checks two-pass derivatives.
No patient files, model weights, network access, or synthetic medical images are used.
No privacy noise seed is released or production DP noise sampled by this script.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Literal
import numpy as np
from scipy.optimize import brentq
from scipy.special import log_ndtr

Mode = Literal['point', 'relation', 'joint_pair']


def svec(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.allclose(a, a.T, atol=1e-12):
        raise ValueError('svec requires a symmetric square matrix')
    i, j = np.triu_indices(a.shape[0])
    return a[i, j] * np.where(i == j, 1.0, math.sqrt(2))


def unsvec(v: np.ndarray, d: int) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if d < 1 or v.shape != (d * (d + 1) // 2,):
        raise ValueError('invalid symmetric-vector dimension')
    i, j = np.triu_indices(d)
    a = np.zeros((d, d), dtype=np.float64)
    vals = v / np.where(i == j, 1.0, math.sqrt(2))
    a[i, j] = vals
    a[j, i] = vals
    return a


def patient_terms(z: np.ndarray, labels: np.ndarray) -> dict:
    z = np.asarray(z, dtype=np.float64)
    labels = np.asarray(labels)
    if z.ndim != 2 or len(z) < 1 or labels.shape != (len(z),):
        raise ValueError('one nonempty patient with aligned image labels is required')
    if not np.isfinite(z).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError('features must be finite; labels must be binary')
    if np.max(np.linalg.norm(z, axis=1)) > 1 + 1e-10:
        raise ValueError('feature norm bound violated')
    d = z.shape[1]
    out = {}
    for c in (0, 1):
        u = z[labels == c]
        out[f'n{c}'] = float(len(u) > 0)
        out[f'm{c}'] = u.mean(0) if len(u) else np.zeros(d)
        out[f'A{c}'] = u.T @ u / len(u) if len(u) else np.zeros((d, d))
    out['b'] = out['n0'] * out['n1']
    out['delta'] = out['b'] * (out['m1'] - out['m0']) / 2
    out['C'] = np.outer(out['delta'], out['delta'])
    out['u'] = out['b'] * np.concatenate([out['m1'], out['m0']]) / math.sqrt(2)
    out['U'] = np.outer(out['u'], out['u'])
    return out


def pack(t: dict, mode: Mode = 'relation') -> np.ndarray:
    if mode not in ('point', 'relation', 'joint_pair'):
        raise ValueError('unknown summary mode')
    counts = [t['n0'], t['n1']] + ([] if mode == 'point' else [t['b']])
    blocks = [np.array(counts), svec(t['A0']), t['m0'], svec(t['A1']), t['m1']]
    if mode == 'relation':
        blocks += [svec(t['C']), t['delta']]
    elif mode == 'joint_pair':
        blocks += [svec(t['U']), t['u']]
    return np.concatenate(blocks)


def moment_dimension(d: int, mode: Mode = 'relation') -> int:
    p = d * (d + 1) // 2 + d
    if mode == 'point':
        return 2 + 2 * p
    if mode == 'relation':
        return 3 + 3 * p
    if mode == 'joint_pair':
        return 3 + 2 * p + (2*d)*(2*d+1)//2 + 2*d
    raise ValueError('unknown mode')


def aggregate(terms: list[dict]) -> dict:
    if not terms:
        raise ValueError('need at least one patient')
    d = terms[0]['m0'].shape[0]
    out = {}
    for c in (0, 1):
        n = sum(t[f'n{c}'] for t in terms)
        out[f'n{c}'] = n
        out[f'm{c}'] = sum((t[f'm{c}'] for t in terms), np.zeros(d)) / max(n, 1)
        out[f'A{c}'] = sum((t[f'A{c}'] for t in terms), np.zeros((d,d))) / max(n, 1)
    m = sum(t['b'] for t in terms)
    out['b'] = m
    out['delta'] = sum((t['delta'] for t in terms), np.zeros(d)) / max(m, 1)
    out['C'] = sum((t['C'] for t in terms), np.zeros((d,d))) / max(m, 1)
    return out


def quadratic(agg: dict, beta: float, ridge: float) -> tuple[np.ndarray, np.ndarray, float]:
    if beta < 0 or ridge <= 0:
        raise ValueError('beta must be >=0 and ridge >0')
    h = (agg['A0'] + agg['A1']) / 2 + beta * agg['C']
    v = (agg['m1'] - agg['m0']) / 2 + beta * agg['delta']
    # Empty-class convention: absent empirical risks contribute zero.
    const = 0.25 * (int(agg['n0'] > 0) + int(agg['n1'] > 0)) + 0.5 * beta * int(agg['b'] > 0)
    return h + ridge * np.eye(h.shape[0]), v, const


def direct_risk(patients: list[tuple[np.ndarray, np.ndarray]], w: np.ndarray,
                beta: float, ridge: float) -> float:
    terms = [patient_terms(z,y) for z,y in patients]
    value = ridge * (w @ w) / 2
    for c in (0, 1):
        vals = [np.mean((z[y == c] @ w - (2*c-1))**2) for z,y in patients if np.any(y == c)]
        if vals:
            value += np.mean(vals) / 4
    pairs = [(w @ t['delta'] - 1)**2 for t in terms if t['b']]
    if pairs:
        value += beta * np.mean(pairs) / 2
    return float(value)


def delta_gaussian(epsilon: float, sensitivity: float, sigma: float) -> float:
    if epsilon < 0 or sensitivity <= 0 or sigma <= 0:
        raise ValueError('invalid Gaussian calibration inputs')
    a = sensitivity/(2*sigma) - epsilon*sigma/sensitivity
    b = -sensitivity/(2*sigma) - epsilon*sigma/sensitivity
    la = float(log_ndtr(a))
    lb = epsilon + float(log_ndtr(b))
    # Stable evaluation of Phi(a)-exp(epsilon)*Phi(b).
    return float(math.exp(la) * (-math.expm1(min(0.0, lb-la))))


def analytic_sigma(epsilon: float, delta: float, sensitivity: float) -> float:
    """Mathematical calibration only; not a floating-point DP/security certificate."""
    if epsilon <= 0 or not 0 < delta < 1 or sensitivity <= 0:
        raise ValueError('epsilon>0, 0<delta<1, sensitivity>0 required')
    lo, hi = sensitivity * 1e-5, sensitivity
    while delta_gaussian(epsilon, sensitivity, lo) <= delta:
        lo /= 2
    while delta_gaussian(epsilon, sensitivity, hi) > delta:
        hi *= 2
    return float(brentq(lambda s: delta_gaussian(epsilon, sensitivity, s)-delta,
                        lo, hi, xtol=1e-13, rtol=1e-13))


def stable_direct_readout(h: np.ndarray, v: np.ndarray, ridge: float) -> np.ndarray:
    """Explicit PSD repair for a DIRECT noisy-moment comparator, not synthesis targets."""
    h = (np.asarray(h) + np.asarray(h).T) / 2
    eig, q = np.linalg.eigh(h)
    hpsd = (q * np.maximum(eig, 0.0)) @ q.T
    return np.linalg.solve(hpsd + ridge * np.eye(len(v)), v)


def check_two_pass() -> dict:
    import torch
    torch.manual_seed(17)
    dtype = torch.float64
    x = torch.randn(12, 5, dtype=dtype, requires_grad=True)
    a = torch.randn(5, 4, dtype=dtype)
    target = torch.randn(20, dtype=dtype)
    def response(y):
        h = torch.tanh(y @ a)
        return torch.cat([h.mean(0), (h.T @ h / len(y)).flatten()])
    r = response(x) - target
    direct = torch.autograd.grad(r.square().sum(), x)[0]
    with torch.no_grad():
        residual = response(x) - target
    gx = torch.zeros_like(x)
    for start in range(0,len(x),3):
        g = response(x[start:start+3]) * (3 / len(x))
        v = (2 * residual.detach() * g).sum()
        gx += torch.autograd.grad(v,x)[0]
    err = float((direct-gx).abs().max())
    if err > 1e-10:
        raise AssertionError('two-pass derivative mismatch')
    return {'max_abs_error':err,'scope':'toy differentiable features; no image model'}


def run_checks() -> dict:
    rng = np.random.default_rng(20260918)
    d = 16
    patients = []
    for _ in range(35):
        n = int(rng.integers(1,8))
        z = rng.normal(size=(n,d)); z /= np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1)
        patients.append((z,rng.integers(0,2,size=n)))
    terms = [patient_terms(z,y) for z,y in patients]
    agg = aggregate(terms)
    k,v,const = quadratic(agg,1.0,.1)
    errors = []
    for _ in range(100):
        w = rng.normal(size=d)
        errors.append(abs(direct_risk(patients,w,1,.1) - (.5*w@k@w-w@v+const)))
    assert max(errors) < 1e-10
    norms = {}
    for mode,bound in [('point',math.sqrt(6)),('relation',3.0),('joint_pair',3.0)]:
        rows=np.array([pack(t,mode) for t in terms])
        assert rows.shape[1] == moment_dimension(d,mode)
        norms[mode] = {'dimension':rows.shape[1], 'observed_max':float(np.linalg.norm(rows,axis=1).max()),
                       'analytic_add_remove_bound':bound}
        assert norms[mode]['observed_max'] <= bound+1e-10
        full=rows.sum(0)
        for j in range(len(rows)):
            assert np.linalg.norm(full-np.delete(rows,j,axis=0).sum(0)) <= bound+1e-10
    # Empty class and no mixed patient cases retain the explicitly defined risks.
    z=patients[0][0]
    for ys in [np.zeros(len(z),dtype=int), np.ones(len(z),dtype=int)]:
        data=[(z,ys)]; w=rng.normal(size=d); a0=aggregate([patient_terms(z,ys)])
        kk,vv,c0=quadratic(a0,1,.1)
        assert abs(direct_risk(data,w,1,.1)-(.5*w@kk@w-w@vv+c0)) < 1e-10
    # Private-pair shuffle only; marginal endpoint centroid lists unchanged.
    mixed=[t for t in terms if t['b']]
    p=np.array([t['m1'] for t in mixed]); n=np.array([t['m0'] for t in mixed])
    perm=np.roll(np.arange(len(mixed)),1)
    dt=(p-n)/2; ds=(p-n[perm])/2
    assert np.allclose(dt.mean(0),ds.mean(0),atol=1e-12)
    ct=dt.T@dt/len(dt); cs=ds.T@ds/len(ds)
    assert np.linalg.norm(ct-cs)>1e-8
    hbase=(agg['A0']+agg['A1'])/2
    vv=(agg['m1']-agg['m0'])/2+dt.mean(0)
    kt=hbase+ct+.1*np.eye(d); ks=hbase+cs+.1*np.eye(d)
    wt=np.linalg.solve(kt,vv); ws=np.linalg.solve(ks,vv)
    ident=np.linalg.solve(kt,(cs-ct)@ws)
    assert np.max(np.abs(wt-ws-ident))<1e-10
    # Ordinary joint endpoint moment projects EXACTLY to difference moments.
    u=np.concatenate([p,n],axis=1)/math.sqrt(2)
    transform=np.concatenate([np.eye(d),-np.eye(d)],axis=1)/math.sqrt(2)
    assert np.allclose(u@transform.T,dt)
    assert np.allclose(transform@(u.T@u/len(u))@transform.T,ct)
    # Mandatory sqrt(2) in symmetric vectorization.
    a=rng.normal(size=(d,d)); a=(a+a.T)/2
    assert np.allclose(unsvec(svec(a),d),a)
    assert abs(np.linalg.norm(svec(a))-np.linalg.norm(a,'fro'))<1e-10
    # Noise target can be infeasible while squared matching remains defined.
    bad_h=np.diag([-2.,1.]); sol=stable_direct_readout(bad_h,np.array([1.,-1.]),.1)
    assert np.isfinite(sol).all()
    # Published pixels are discrete; quantization must be measured, not assumed equal.
    gray=rng.random((4,16,16)); q8=np.round(gray*255).astype(np.uint8).astype(float)/255
    assert np.max(abs(gray-q8)) <= .5/255+1e-12
    noise={}
    for mode,sens in [('point',math.sqrt(6)),('relation',3.),('joint_pair',3.)]:
        noise[mode]={}
        for eps in [4.,8.]:
            sigma=analytic_sigma(eps,1e-5,sens)
            assert abs(delta_gaussian(eps,sens,sigma)-1e-5)<1e-12
            noise[mode][str(int(eps))]={'sigma_abs':sigma,'sensitivity':sens,
                'delta':1e-5,'calibration_delta':delta_gaussian(eps,sens,sigma)}
    two=check_two_pass()
    return {
      'status':'PASS_TOY_ALGEBRA_ONLY',
      'scope':'Constructed arrays only. No patient data, encoder weights, medical utility, or production privacy certification.',
      'patient_count_in_toy':len(patients),
      'risk_expansion_max_abs_error':float(max(errors)),
      'norm_and_dimension_checks':norms,
      'shuffle_delta_max_abs_error':float(np.max(np.abs(dt.mean(0)-ds.mean(0)))),
      'shuffle_relation_frobenius_difference':float(np.linalg.norm(ct-cs)),
      'solution_difference_identity_max_abs_error':float(np.max(np.abs(wt-ws-ident))),
      'joint_pair_to_difference_moments':'EXACT_TO_NUMERICAL_TOLERANCE; not a distinct invention',
      'two_pass_gradient':two,
      'noise_calibration':noise,
      'production_noise_sampled':False,
      'new_patient_training':0,
      'expert_or_reserved_access':0,
      'warnings':[
        'Source-encoder moment preservation does not guarantee cross-encoder transfer.',
        'Patient bootstrap does not account automatically for optimization/DP noise variation.',
        'Publishing several DP releases requires joint composition; public reproducible DP noise is prohibited.',
        'This code is a mathematical reference, not a secure implementation of a Gaussian sampler.'
      ]
    }

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='reference_checks.json')
    args=parser.parse_args()
    result=run_checks()
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(result['status'])
    print(path.resolve())
