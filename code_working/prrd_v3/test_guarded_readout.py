"""Independent CPU checks of the guarded learner, not medical/runtime efficacy."""
import argparse
import hashlib
import json
import time
import unittest
from pathlib import Path

import numpy as np
import torch
from .guarded_readout import (guarded_readout, readout_from_moments,
                             freeze_functional_target, functional_loss)

MEASUREMENTS = {}


def moments(features):
    # Two distinct banks; full-bank counts are used, never microbatch counts.
    z, raw = features[:, :4], features[:, 4:]
    q = len(z) // 4
    neg, pos = z[:q], z[2*q:3*q]
    d = (raw[3*q:] - raw[q:2*q]) / 2
    r = d / torch.maximum(d.norm(dim=1, keepdim=True), d.new_tensor(0.7))
    return {"m": torch.stack([neg.mean(0), pos.mean(0)]),
            "A": torch.stack([neg.T @ neg / len(neg), pos.T @ pos / len(pos)]),
            "delta": r.mean(0), "C": r.T @ r / len(r)}


def feature_map(x):
    raw = torch.stack([x[:, 0] + .3*x[:, 1], x[:, 1]**2 + .1*x[:, 2],
                       torch.sin(x[:, 2]) + .2*x[:, 3], x[:, 3] - .4*x[:, 0]], 1)
    return torch.cat([raw.tanh()/2, raw], 1)


def joint_loss(features, target_moments, target, guard=True, eta=1.0):
    m = moments(features)
    loss = .25 * sum((m[k]-target_moments[k]).square().sum() for k in ("m", "A"))
    loss = loss + .5 * sum((m[k]-target_moments[k]).square().sum() for k in ("delta", "C"))
    learner = readout_from_moments(m, kappa=.1, guard=guard)
    return loss + eta*functional_loss(learner.weight, target)


class GuardedCoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(918)
        self.rng = np.random.default_rng(918)
        a = self.rng.normal(size=(4, 4)); c = self.rng.normal(size=(6, 4))
        self.M = torch.tensor(a.T @ a + .3*np.eye(4), dtype=torch.float64)
        self.v = torch.tensor(self.rng.normal(size=4), dtype=torch.float64)
        self.C = torch.tensor(c.T @ c/6, dtype=torch.float64)
        self.mu = torch.tensor(self.rng.normal(size=4), dtype=torch.float64)

    def test_independent_numpy_solution_and_degradation(self):
        M,v,C,mu = (t.numpy() for t in (self.M,self.v,self.C,self.mu))
        w0 = np.linalg.solve(M,v); w1 = np.linalg.solve(M+C,v+mu)
        d = w1-w0; e = float(d @ M @ d)
        errors = []
        for rho in (0., .1*np.sqrt(e), 2*np.sqrt(e)):
            out = guarded_readout(self.M,self.v,self.C,self.mu,rho=rho)
            alpha = 1. if e == 0 else min(1.,rho/np.sqrt(e))
            expected = w0 + alpha*d
            np.testing.assert_allclose(out.weight.numpy(),expected,atol=1e-12,rtol=1e-12)
            F = lambda w: .5*w @ M @ w-v @ w
            gap = F(expected)-F(w0)
            self.assertLessEqual(gap,.5*rho*rho+1e-12)
            errors.append(abs(gap-.5*alpha*alpha*e))
        rel = guarded_readout(self.M,self.v,self.C,self.mu,kappa=.1)
        self.assertLessEqual(float(.5*(rel.weight-rel.point_weight) @ self.M @
                                  (rel.weight-rel.point_weight)),
                             .1*.5*float(rel.point_weight @ self.M @ rel.point_weight)+1e-12)
        MEASUREMENTS["direct_quadratic_identity_max_abs"] = max(errors)

    def test_zero_point_zero_delta_and_ablation(self):
        zero = torch.zeros_like(self.v)
        out = guarded_readout(self.M,zero,self.C,self.mu,kappa=.1)
        torch.testing.assert_close(out.weight,zero,atol=0,rtol=0)
        for beta in (0.,1.):
            out = guarded_readout(self.M,self.v,None,None,beta=beta,kappa=.1)
            self.assertEqual(float(out.alpha),1.)
            torch.testing.assert_close(out.weight,out.point_weight,atol=0,rtol=0)
        ung = guarded_readout(self.M,self.v,self.C,self.mu,rho=0.,guard=False)
        torch.testing.assert_close(ung.weight,ung.relation_weight)
        x = torch.zeros(4,dtype=torch.float64,requires_grad=True)
        # Zero radius and a zero point optimum must not produce NaN gradients.
        z = guarded_readout(self.M,x,self.C,self.mu,kappa=.1)
        z.weight.square().sum().backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))

    def test_fixed_target_scores_and_combined_bound(self):
        f = torch.tensor(self.rng.normal(size=(12,4)),dtype=torch.float64)
        M = f.T @ f/len(f)+.1*torch.eye(4,dtype=torch.float64)
        out = guarded_readout(M,self.v,self.C,self.mu,kappa=.1)
        target = freeze_functional_target(out)
        saved = target.matrix.clone()
        ws = out.weight.detach()+torch.tensor([.03,-.02,.01,.04],dtype=torch.float64)
        diff = ws-out.weight
        loss = functional_loss(ws,target)
        score_loss = (f @ diff).square().mean()+.1*diff.square().sum()
        torch.testing.assert_close(loss,score_loss,atol=1e-13,rtol=1e-12)
        excess = .5*(ws-out.point_weight) @ M @ (ws-out.point_weight)
        bound = .5*(out.radius_squared.sqrt()+loss.sqrt()).square()
        self.assertLessEqual(float(excess),float(bound)+1e-12)
        # Copy isolation: mutating a caller tensor cannot move the target metric.
        M.add_(torch.eye(4,dtype=torch.float64))
        torch.testing.assert_close(target.matrix,saved,atol=0,rtol=0)
        self.assertFalse(target.matrix.requires_grad or target.weight.requires_grad)
        MEASUREMENTS["functional_vs_direct_scores_abs"] = float(abs(loss-score_loss))
        MEASUREMENTS["combined_bound_slack"] = float(bound-excess)

    def test_finite_difference_through_active_cap_and_solve(self):
        C = self.C; mu = self.mu
        x = self.v.clone().requires_grad_(True)
        self.assertLess(float(guarded_readout(self.M,x,C,mu,kappa=.01).alpha),1.)
        def fn(v):
            out = guarded_readout(self.M,v,C,mu,kappa=.01)
            return out.weight
        self.assertTrue(torch.autograd.gradcheck(fn,(x,),eps=1e-6,atol=2e-6,rtol=2e-4))
        MEASUREMENTS["active_cap_finite_difference"] = "PASS"

    def test_joint_objective_full_vs_replayed_gradient(self):
        x = torch.randn(16,4,dtype=torch.float64)*.3
        tx = torch.randn(16,4,dtype=torch.float64)*.4+.1
        tm = {k:v.detach() for k,v in moments(feature_map(tx)).items()}
        target = freeze_functional_target(readout_from_moments(tm,kappa=.1))
        direct = x.clone().requires_grad_(True)
        loss = joint_loss(feature_map(direct),tm,target)+.01*direct.square().mean()
        g, = torch.autograd.grad(loss,direct)
        self.assertGreater(float(g.norm()),0.)
        errors = []
        for micro in (1,3,5,16):
            replay = x.clone().requires_grad_(True)
            with torch.no_grad():
                gathered = torch.cat([feature_map(replay[a:a+micro])
                                      for a in range(0,len(x),micro)])
            leaf = gathered.detach().requires_grad_(True)
            feature_loss = joint_loss(leaf,tm,target)
            response, = torch.autograd.grad(feature_loss,leaf)
            for a in range(0,len(x),micro):
                chunk = replay[a:a+micro]
                value = (feature_map(chunk)*response[a:a+micro]).sum()
                value = value + .01*chunk.square().sum()/replay.numel()
                value.backward()
            torch.testing.assert_close(replay.grad,g,atol=2e-12,rtol=2e-10)
            errors.append(float((replay.grad-g).abs().max()))
        MEASUREMENTS["joint_full_vs_replay_gradient_max_abs"] = max(errors)
        MEASUREMENTS["microbatches_checked"] = [1,3,5,16]

    def test_invalid_contracts_rejected(self):
        with self.assertRaises(ValueError):
            guarded_readout(-self.M,self.v,rho=1.)
        with self.assertRaises(ValueError):
            guarded_readout(self.M,self.v,rho=1.,kappa=.1)
        with self.assertRaises(ValueError):
            guarded_readout(self.M,self.v,kappa=-.1)
        with self.assertRaises(ValueError):
            guarded_readout(self.M,self.v,self.C,None,kappa=.1)
        out = guarded_readout(self.M,self.v,rho=1.)
        target = freeze_functional_target(out)
        target.matrix.requires_grad_(True)
        with self.assertRaises(ValueError):
            functional_loss(out.weight,target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report",type=Path,required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Preserve earlier verification output; use a new report path")
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(GuardedCoreTests))
    files = [Path(__file__),Path(__file__).with_name("guarded_readout.py")]
    payload = {
        "status":"PASS_CPU_GUARDED_LEARNER_ONLY" if result.wasSuccessful() else "FAIL_CPU_GUARDED_LEARNER",
        "tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
        "seconds":time.perf_counter()-started,"measurements":MEASUREMENTS,
        "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "scope":"Constructed arrays and differentiable toy features only; no medical encoder, patient pixels, DP sampler or synthesis efficacy",
        "actual_model_W1_passed":False,"new_GPU_runs":0,"new_patient_pixels":0,
        "new_DP_releases":0
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(payload,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
