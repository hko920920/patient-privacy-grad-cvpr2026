"""Independent mathematics and replay checks; no patient pixels or GPU."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import unittest
import numpy as np
import torch
from torch.nn import functional as F
from prrd_v3.contracts import save
from prrd_v3.render import Renderer
from .signals import (conditions, heads, per_image_gradient, patient_class_sums,
                      pool_patient_sums, synthetic_signal, bind_objective, objective_digest)
from .objectives import one_pass, two_pass

METRICS = {}


class SignalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        self.g = torch.Generator().manual_seed(812)
        self.z = torch.randn(7, 3, dtype=torch.float64, generator=self.g) * 0.2
        self.ys = [0, 0, 1, 0, 1, 1, 0]
        self.pids = ['p0', 'p0', 'p0', 'p1', 'q0', 'q0', 'q1']
        self.hs = heads('test', 3)
        self.gs = torch.stack([per_image_gradient(self.z, self.ys, h) for h in self.hs])

    def test_analytic_head_gradient_and_feature_derivative_against_autograd(self):
        x = self.z.clone().requires_grad_()
        h = self.hs[0]
        w, b = h.weight.clone().requires_grad_(), h.bias.clone().requires_grad_()
        independent = []
        for i, y in enumerate(self.ys):
            loss = F.cross_entropy(x[i:i+1] @ w.T + b, torch.tensor([y]))
            dw, db = torch.autograd.grad(loss, (w, b), create_graph=True)
            independent.append(torch.cat((dw.flatten(), db)))
        independent = torch.stack(independent)
        analytic = per_image_gradient(x, self.ys, h)
        err = (independent - analytic).abs().max().item()
        weights = torch.randn(analytic.shape, dtype=torch.float64, generator=self.g)
        a = torch.autograd.grad((analytic * weights).sum(), x, retain_graph=True)[0]
        b = torch.autograd.grad((independent * weights).sum(), x)[0]
        derivative_error = (a-b).abs().max().item()
        self.assertLess(err, 1e-12)
        self.assertLess(derivative_error, 1e-12)
        METRICS.update(analytic_gradient_max_abs=err, image_feature_derivative_max_abs=derivative_error)

    def test_patient_class_weighted_loss_independent_autograd(self):
        total = patient_class_sums(self.gs, self.ys, self.pids)
        expected = []
        for head in self.hs:
            w, b = head.weight.clone().requires_grad_(), head.bias.clone().requires_grad_()
            value = 0
            for c in (0, 1):
                patients = sorted({p for p, y in zip(self.pids, self.ys) if y == c})
                for p in patients:
                    ids = [i for i in range(7) if self.pids[i] == p and self.ys[i] == c]
                    value = value + 0.5/len(patients) * F.cross_entropy(
                        self.z[ids] @ w.T+b, torch.tensor([self.ys[i] for i in ids]))
            gw, gb = torch.autograd.grad(value, (w, b))
            expected.append(torch.cat((gw.flatten(), gb)))
        error = (total.balanced()-torch.stack(expected)).abs().max().item()
        self.assertLess(error, 1e-12)
        self.assertEqual(total.counts.tolist(), [3, 2])
        METRICS['patient_weighted_autograd_max_abs'] = error

    def test_repeat_visits_and_conditions_do_not_change_patient_weight(self):
        ids = [0, 1, 0, 1, 2, 2, 3, 4, 5, 6]
        original = patient_class_sums(self.gs, self.ys, self.pids)
        repeated = patient_class_sums(self.gs[:, ids], [self.ys[i] for i in ids],
                                      [self.pids[i] for i in ids])
        torch.testing.assert_close(repeated.balanced(), original.balanced(), atol=1e-15, rtol=1e-14)
        twice = patient_class_sums(self.gs.repeat(2, 1, 1), self.ys, self.pids)
        torch.testing.assert_close(twice.balanced()[:4], original.balanced(), atol=0, rtol=0)
        self.assertTrue(torch.equal(repeated.counts, original.counts))

    def test_population_pooling_uses_class_sums_and_counts(self):
        p = patient_class_sums(self.gs[:, :4], self.ys[:4], self.pids[:4])
        q = patient_class_sums(self.gs[:, 4:], self.ys[4:], self.pids[4:])
        actual = pool_patient_sums(p, q)
        expected = patient_class_sums(self.gs, self.ys, self.pids)
        torch.testing.assert_close(actual.balanced(), expected.balanced(), atol=1e-15, rtol=1e-14)
        self.assertFalse(torch.allclose(actual.balanced(), 0.5*(p.balanced()+q.balanced())))
        with self.assertRaisesRegex(RuntimeError, 'share a patient'):
            pool_patient_sums(p, p)

    def test_empty_class_is_zero_without_reweighting_present_class(self):
        gs = self.gs[:, [0, 1, 3]]
        result = patient_class_sums(gs, [0, 0, 0], ['a', 'a', 'b'])
        expected = 0.25*(gs[:, :2].mean(1)+gs[:, 2])
        torch.testing.assert_close(result.balanced(), expected, atol=0, rtol=0)
        self.assertEqual(result.counts.tolist(), [2, 0])
        empty = patient_class_sums(gs[:, :0], [], [])
        self.assertTrue(torch.equal(empty.balanced(), torch.zeros_like(expected)))

    def test_feature_gradient_finite_difference(self):
        target = synthetic_signal(self.z*.7, self.ys, self.hs[0]).detach()
        def loss(z):
            actual = synthetic_signal(z, self.ys, self.hs[0])
            return 1-F.cosine_similarity(actual[None], target[None], eps=1e-12).sum()
        self.assertTrue(torch.autograd.gradcheck(loss, (self.z.clone().requires_grad_(),),
                                                eps=1e-6, atol=1e-6, rtol=1e-4))

    def test_conditions_and_heads_replay_without_global_rng(self):
        torch.manual_seed(192)
        before = torch.random.get_rng_state().clone()
        a, b = conditions(), conditions()
        x = torch.rand(3, 1, 16, 16, generator=self.g)
        for first, second in zip(a, b):
            torch.testing.assert_close(first.apply(x), second.apply(x), atol=0, rtol=0)
            torch.testing.assert_close(first.apply(x), torch.cat([first.apply(x[i:i+1]) for i in range(3)]),
                                       atol=2e-6, rtol=0)
        h = heads('source', 3)
        heads('second', 5)
        other = heads('source', 3)
        for first, second in zip(h, other):
            self.assertTrue(torch.equal(first.weight, second.weight))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertFalse(torch.equal(a[0].theta, a[1].theta))

    def test_fixed_target_and_a2_common_block_unchanged(self):
        target = patient_class_sums(self.gs, self.ys, self.pids).balanced()
        original = target.clone()
        cs = conditions()
        one = bind_objective(cs, {'source': self.hs}, {'source': target}, self.ys)
        two = bind_objective(cs, {'source': self.hs, 'second': self.hs},
                            {'source': target, 'second': target*.9+.01}, self.ys)
        digest = objective_digest(one)
        target.add_(10)
        self.assertEqual(digest, objective_digest(one))
        features = self.z[None].repeat(4, 1, 1)*.8
        one_loss, one_blocks = one.matching({'source': features})
        two_loss, two_blocks = two.matching({'source': features, 'second': features})
        self.assertTrue(torch.equal(one.targets['source'], two.targets['source']))
        self.assertTrue(torch.equal(one_blocks['source'], two_blocks['source']))
        torch.testing.assert_close(two_loss, .5*(one_loss+two_blocks['second']), atol=0, rtol=0)
        with self.assertRaisesRegex(RuntimeError, 'Zero target'):
            bind_objective(cs, {'source': self.hs}, {'source': original*0}, self.ys)

    def test_two_encoder_global_replay_and_single_prior(self):
        class TinyEncoder:
            def __init__(self, scale):
                self.scale = scale
            def __call__(self, x):
                return torch.stack((x.mean((1, 2, 3)), x.square().mean((1, 2, 3)),
                                    x[:, :, :112].mean((1, 2, 3))), -1)*self.scale
        templates = torch.rand(4, 1, 224, 224, generator=self.g)*.6+.2
        model = Renderer(templates, 'B', 101)
        model.set_step(500)
        names = ('source', 'second')
        enc = {n: TinyEncoder(s) for n, s in zip(names, (.7, 1.2))}
        hs = {n: heads(n, 3, count=2) for n in names}
        labels = (0, 0, 1, 1)
        target = {n: torch.stack([synthetic_signal(self.z[:4], labels, h) for h in hs[n]])
                  for n in names}
        objective = bind_objective(conditions(2), hs, target, labels)
        values = one_pass(model, enc, objective, 3)
        values['loss'].backward()
        reference = [p.grad.clone() for p in model.parameters()]
        reference_loss = float(values['loss'].detach())
        prior = float(values['prior'].detach())
        del values
        result = two_pass(model, enc, objective, 3)
        errors = [(p.grad-r).abs().max().item() for p, r in zip(model.parameters(), reference)]
        self.assertLess(abs(result['loss']-reference_loss), 2e-6)
        self.assertLess(max(errors), 2e-7)
        self.assertEqual(result['prior'], prior)
        self.assertGreater(sum(float(r.norm()) for r in reference), 0)
        METRICS['two_encoder_replay_max_abs'] = max(errors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SignalTests))
    report = {'status': 'PASS' if result.wasSuccessful() else 'FAIL', 'tests': result.testsRun,
              'failures': len(result.failures), 'errors': len(result.errors), 'metrics': METRICS,
              'seconds': time.perf_counter()-started, 'GPU': False, 'patient_pixels': 0}
    save(args.output, report)
    print(json.dumps(report), flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()

