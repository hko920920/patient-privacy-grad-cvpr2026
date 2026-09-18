"""Independent feature-array tests. No medical pixels, weights, CUDA or release."""
import unittest
import math
from dataclasses import replace
import numpy as np
import torch
from scipy.special import ndtr
from . import moments as m
from .objective import matching_loss, feature_gradient


class CoreTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(917)
        self.z = rng.normal(size=(9, 4))
        self.z /= np.maximum(1, np.linalg.norm(self.z, axis=1))[:, None]
        self.y = np.array([0, 0, 1, 0, 1, 1, 1, 0, 0])
        self.ids = np.array(['a', 'a', 'a', 'b', 'b', 'b', 'c', 'd', 'd'])
        self.p = m.patient_contributions(self.z, self.y, self.ids)
        self.target = m.decode_sums(m.sum_query(self.p), 4)

    def test_patient_weighted_risk_against_raw_images(self):
        w = np.array([.3, -.2, .6, .9])
        beta, ridge = .7, .01
        point = 0.0
        for c in (0, 1):
            per_patient = [np.mean((self.z[(self.ids == pid) & (self.y == c)] @ w - (2*c-1))**2)
                           for pid in sorted(set(self.ids)) if np.any((self.ids == pid) & (self.y == c))]
            point += np.mean(per_patient) / 4
        relation = beta / 2 * np.mean([(w @ p.delta - 1)**2 for p in self.p if p.mixed])
        h, v = m.risk_system(self.target, beta)
        quadratic = .5 * w @ h @ w - v @ w + ridge/2 * (w @ w) + .5 + beta/2
        self.assertAlmostEqual(point + relation + ridge/2 * (w @ w), quadratic, places=13)
        solved = m.solve_readout(self.target, beta, ridge)
        np.testing.assert_allclose((h + ridge*np.eye(4)) @ solved, v, atol=1e-13)

    def test_repeat_all_visits_of_one_patient_does_not_change_patient_weight(self):
        a = self.ids == 'a'
        repeated = m.patient_contributions(np.concatenate([self.z, self.z[a], self.z[a]]),
                                           np.concatenate([self.y, self.y[a], self.y[a]]),
                                           np.concatenate([self.ids, self.ids[a], self.ids[a]]))
        np.testing.assert_allclose(m.sum_query(repeated), m.sum_query(self.p), atol=1e-14)

    def test_isometry_layout_and_patient_sensitivity(self):
        d = 16
        e = np.eye(d)[0]
        p = m.patient_contributions(np.stack([-e, e]), [0, 1], ['one', 'one'])[0]
        self.assertEqual(len(p.vector()), 459)
        self.assertEqual(len(p.vector(False)), 306)
        self.assertAlmostEqual(np.linalg.norm(p.vector()), 3.0, places=13)
        self.assertAlmostEqual(np.linalg.norm(p.vector(False)), math.sqrt(6), places=13)
        a = np.arange(16, dtype=float).reshape(4, 4)
        a = a + a.T
        self.assertAlmostEqual(np.linalg.norm(m.svec(a)), np.linalg.norm(a), places=12)
        np.testing.assert_allclose(m.smat(m.svec(a), 4), a, atol=1e-13)

    def test_private_shuffle_changes_second_not_mean_and_keeps_public(self):
        public = [self.p[0]]
        private = [self.p[1], replace(self.p[0], patient_id='e')]
        delta, c = m.shuffled_relation(private, public, [1, 0])
        real_delta = np.stack([p.delta for p in private + public])
        np.testing.assert_allclose(delta, real_delta.mean(0), atol=1e-13)
        self.assertGreater(np.linalg.norm(c - real_delta.T @ real_delta / 3), 1e-5)
        with self.assertRaises(ValueError):
            m.shuffled_relation(private, public, [0, 1])

    def test_joint_moment_contraction_is_the_same_relation_statistic(self):
        mixed = [p for p in self.p if p.mixed]
        u = np.stack([np.r_[p.means[1], p.means[0]] for p in mixed])
        l = np.c_[np.eye(4)/2, -np.eye(4)/2]
        np.testing.assert_allclose(l @ u.mean(0), self.target['delta'], atol=1e-13)
        np.testing.assert_allclose(l @ (u.T @ u / len(u)) @ l.T, self.target['C'], atol=1e-13)

    def test_combined_sums_are_normalized_once(self):
        q = m.sum_query(self.p[:2]) + m.sum_query(self.p[2:])
        np.testing.assert_allclose(q, m.sum_query(self.p), atol=1e-13)
        combined = m.decode_sums(q, 4)
        for key in ('m', 'A', 'delta', 'C'):
            np.testing.assert_allclose(combined[key], self.target[key], atol=1e-13)

    def test_noisy_target_optimization_and_readout_are_separate(self):
        q = m.sum_query(self.p)
        q[:3] = [-4, 0, 20]
        q[3:] *= -3
        noisy = m.decode_sums(q, 4, noisy=True)
        self.assertTrue(np.isfinite(noisy['A']).all())
        repaired = m.decode_sums(q, 4, noisy=True, repair=True)
        for mu, a in zip(list(repaired['m'])+[repaired['delta']], list(repaired['A'])+[repaired['C']]):
            self.assertGreaterEqual(np.linalg.eigvalsh(a - np.outer(mu, mu)).min(), -1e-12)
            self.assertLessEqual(np.trace(a), 1+1e-12)
        self.assertTrue(np.isfinite(m.solve_readout(repaired)).all())
        z = torch.zeros((128, 4), dtype=torch.float64, requires_grad=True)
        loss = matching_loss(z, noisy, True)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(z.grad).all())
        bad = dict(self.target)
        bad['A'] = np.stack([-np.eye(4), -np.eye(4)])
        bad['C'] = np.zeros((4, 4))
        with self.assertRaises(ValueError):
            m.solve_readout(bad)

    def test_gaussian_scale_matches_analytical_privacy_profile(self):
        eps, delta = 8., 1e-5
        sigma = m.analytic_gaussian_std(eps, delta, 3.)
        profile = ndtr(3/(2*sigma)-eps*sigma/3) - math.exp(eps)*ndtr(-3/(2*sigma)-eps*sigma/3)
        self.assertAlmostEqual(profile, delta, places=12)
        self.assertAlmostEqual(sigma, 1.800687, places=6)
        self.assertAlmostEqual(m.analytic_gaussian_std(eps,delta,math.sqrt(6))/sigma, math.sqrt(6)/3, places=12)

    def test_whole_bank_two_pass_gradient_matches_full_autograd(self):
        generator = torch.Generator().manual_seed(23)
        x = torch.randn((128, 7), generator=generator, dtype=torch.float64, requires_grad=True)
        fixed = torch.randn((7, 4), generator=generator, dtype=torch.float64)
        def encode(a):
            return torch.tanh(a @ fixed) / 2
        expected_loss = matching_loss(encode(x), self.target, True)
        expected, = torch.autograd.grad(expected_loss, x)
        with torch.no_grad():
            all_z = torch.cat([encode(x[k:k+11]) for k in range(0,128,11)])
        loss, gz = feature_gradient(all_z, self.target, True)
        actual = []
        for k in range(0,128,11):
            chunk = x[k:k+11].detach().clone().requires_grad_(True)
            gradient, = torch.autograd.grad(encode(chunk), chunk, grad_outputs=gz[k:k+11])
            actual.append(gradient)
        torch.testing.assert_close(loss, expected_loss.detach(), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(torch.cat(actual), expected, rtol=1e-12, atol=1e-12)

    def test_invalid_inputs_fail(self):
        with self.assertRaises(ValueError):
            m.patient_contributions(self.z*10, self.y, self.ids)
        with self.assertRaises(ValueError):
            m.sum_query(self.p+self.p)
        with self.assertRaises(ValueError):
            m.decode_sums(np.zeros(4), 4)
        with self.assertRaises(ValueError):
            m.analytic_gaussian_std(0,1e-5,3)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
