"""Changed-path tests; no medical pixels, model loading or GPU work."""
import argparse
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock
import numpy as np
import torch
from .augmented_targets import RULE, view_parameters, patient_augmented_moments
from .contracts import save, sha, seed
from .guarded_objectives import LearnerPolicy, bank_moments, bind_objective, feature_objective
from .bank_runtime import (check_augmented_target_binding, tree_digest, digest,
                           save_checkpoint, load_checkpoint, objective_target_state)
from .fit_banks import load_augmented_target

METRICS = {}


class AugmentedTargetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        self.rng = np.random.default_rng(932)
        self.policy = LearnerPolicy(0., .1, kappa=.1)
        self.paths = lambda x: {'z': x, 'a': x*.7}
        z = torch.tensor(self.rng.normal(size=(8, 3))*.12, dtype=torch.float64)
        self.target = bank_moments(self.paths(z), 'B', {})
        z2 = torch.tensor(self.rng.normal(size=(8, 3))*.2, dtype=torch.float64)
        self.aug = bank_moments(self.paths(z2), 'B', {})

    def objective(self, aug=None):
        return bind_objective(self.target, 'B', {}, self.policy, eta=1., aug_target=aug)

    def test_legacy_identity_and_clean_function_are_unchanged(self):
        x = torch.tensor(self.rng.normal(size=(8, 3))*.2, dtype=torch.float64, requires_grad=True)
        y = (x.detach()*.8+.01).requires_grad_(True)
        base = self.objective()
        explicit = self.objective(self.target)
        different = self.objective(self.aug)
        a = feature_objective(self.paths(x), self.paths(y), base)
        b = feature_objective(self.paths(x), self.paths(y), explicit)
        c = feature_objective(self.paths(x), self.paths(y), different)
        for name in ('loss', 'statistics', 'functional', 'augmentation_matching'):
            self.assertTrue(torch.equal(a[name], b[name]))
        ga = torch.autograd.grad(a['loss'], (x, y), retain_graph=True)
        gb = torch.autograd.grad(b['loss'], (x, y), retain_graph=True)
        gc = torch.autograd.grad(c['loss'], (x, y), retain_graph=True)
        self.assertTrue(all(torch.equal(i, j) for i, j in zip(ga, gb)))
        self.assertTrue(torch.equal(ga[0], gc[0]))
        self.assertFalse(torch.equal(ga[1], gc[1]))
        for name in ('statistics', 'functional'):
            self.assertTrue(torch.equal(a[name], c[name]))
        for name in ('matrix', 'weight'):
            self.assertTrue(torch.equal(getattr(base.functional_target, name),
                                        getattr(different.functional_target, name)))
        METRICS['legacy_loss_gradient_exact'] = True
        METRICS['changed_aug_target_clean_gradient_exact'] = True

    def test_augmented_target_is_detached_copy_and_validated(self):
        source = {k: v.clone().requires_grad_() for k, v in self.aug.items()}
        objective = self.objective(source)
        bound = tree_digest(objective_target_state(objective))
        with torch.no_grad():
            source['m'].add_(7.)
        self.assertEqual(bound, tree_digest(objective_target_state(objective)))
        self.assertTrue(all(not t.requires_grad for t in objective.aug_moments.values()))
        bad = dict(self.aug, m=torch.zeros(2, 4))
        with self.assertRaisesRegex(RuntimeError, 'shape'):
            self.objective(bad)
        bad = dict(self.aug, m=torch.full((2, 3), float('nan')))
        with self.assertRaisesRegex(RuntimeError, 'values'):
            self.objective(bad)

    def test_patient_and_class_weighting_independent_direct_sum(self):
        z = self.rng.normal(size=(6, 4, 3))*.1
        ids = np.array(['p1', 'p1', 'p1', 'p2', 'q1', 'q2'])
        labels = np.array([0, 0, 1, 0, 1, 0])
        roles = np.array(['P', 'P', 'P', 'P', 'Q', 'Q'])
        result = patient_augmented_moments(z, labels, ids, roles)
        means, seconds, counts = [], [], []
        for c in (0, 1):
            patients = sorted(set(ids[labels == c]))
            m, A = np.zeros(3), np.zeros((3, 3))
            for pid in patients:
                indexes = [i for i in range(len(z)) if ids[i] == pid and labels[i] == c]
                for i in indexes:
                    for k in range(4):
                        m += z[i, k]/(len(patients)*len(indexes)*4)
                        A += np.outer(z[i, k], z[i, k])/(len(patients)*len(indexes)*4)
            means.append(m); seconds.append(A); counts.append(len(patients))
        expected = {'m': np.array(means), 'A': np.array(seconds), 'counts': np.array(counts)}
        errors = []
        for k in expected:
            np.testing.assert_allclose(result['target'][k], expected[k], atol=1e-14, rtol=1e-14)
            errors.append(float(np.max(abs(result['target'][k]-expected[k]))))
        self.assertFalse(np.allclose(result['target']['m'][0], z[labels == 0].mean(axis=(0, 1))))
        p, q = result['populations']['P'], result['populations']['Q']
        equal_population_mix = .5*(p['m_sum'][0]/p['counts'][0] + q['m_sum'][0]/q['counts'][0])
        self.assertFalse(np.allclose(result['target']['m'][0], equal_population_mix))
        # Raw second moments retain within-patient/view variation.
        means_only = np.array([np.outer(v, v) for v in result['target']['m']])
        self.assertFalse(np.allclose(result['target']['A'], means_only))
        repeated = np.r_[np.arange(len(z)), [0, 1, 2], [0, 1, 2]]
        again = patient_augmented_moments(np.repeat(z[repeated], 2, axis=1),
                                          labels[repeated], ids[repeated], roles[repeated])
        for k in expected:
            np.testing.assert_allclose(again['target'][k], expected[k], atol=1e-14, rtol=1e-14)
        METRICS['independent_patient_moment_max_abs'] = max(errors)

    def test_absent_class_and_wrong_role_rejected(self):
        z = self.rng.normal(size=(2, 4, 3))*.1
        result = patient_augmented_moments(z, [0, 0], ['p1', 'p2'], ['P', 'P'])
        self.assertEqual(int(result['target']['counts'][1]), 0)
        self.assertTrue(np.array_equal(result['target']['m'][1], np.zeros(3)))
        self.assertTrue(np.array_equal(result['target']['A'][1], np.zeros((3, 3))))
        with self.assertRaisesRegex(RuntimeError, 'role'):
            patient_augmented_moments(z, [0, 1], ['p1', 'p2'], ['P', 'V'])
        with self.assertRaisesRegex(RuntimeError, 'multiple roles'):
            patient_augmented_moments(z, [0, 1], ['p1', 'p1'], ['P', 'Q'])

    def test_view_randomness_is_reproducible_and_does_not_change_global_rng(self):
        torch.manual_seed(442)
        before = torch.get_rng_state().clone()
        a = view_parameters('constructed-image', 'P')
        b = view_parameters('constructed-image', 'P')
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(torch.equal(a['theta'], b['theta']))
        self.assertTrue(torch.equal(a['brightness'], b['brightness']))
        self.assertEqual(len(set(a['seeds'])), 4)
        self.assertEqual(a['seeds'][0], seed(RULE['seed_domain'], 101, 'P', 'constructed-image', 0))
        self.assertFalse(torch.equal(a['theta'], view_parameters('constructed-image', 'Q')['theta']))
        self.assertLessEqual(float(a['theta'][:, :, 2].abs().max()), .020001)
        self.assertTrue(bool(((a['brightness'] >= .98) & (a['brightness'] <= 1.02)).all()))

    def test_job_file_and_rule_hashes_are_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'target.npz'
            np.savez(path, aug_m=self.aug['m'].numpy(), aug_A=self.aug['A'].numpy(),
                     rule_sha256=np.array(digest(RULE)))
            job = {'arm': 'B', 'aug_target_file': str(path), 'aug_target_sha256': sha(path),
                   'aug_target_prefix': 'aug_', 'aug_target_rule_sha256': digest(RULE)}
            got = load_augmented_target(job)
            np.testing.assert_array_equal(got['m'], self.aug['m'].numpy())
            self.assertIsNone(load_augmented_target({'arm': 'B'}))
            with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
                load_augmented_target({'arm': 'B', 'aug_target_file': str(path)})
            with self.assertRaisesRegex(RuntimeError, 'rule changed'):
                load_augmented_target(dict(job, aug_target_rule_sha256='f'*64))
            with path.open('ab') as f:
                f.write(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'file changed'):
                load_augmented_target(job)

    def test_augmented_content_changes_checkpoint_signature(self):
        objective = self.objective(self.aug)
        spec = {'arm': 'B', 'aug_target_sha256': 'a'*64,
                'aug_target_rule_sha256': digest(RULE),
                'aug_target_digest': tree_digest(dict(objective.aug_moments))}
        check_augmented_target_binding(objective, spec)
        with self.assertRaisesRegex(RuntimeError, 'binding'):
            check_augmented_target_binding(objective, {'arm': 'B'})
        changed = dict(spec, aug_target_sha256='b'*64)
        self.assertNotEqual(digest(spec), digest(changed))
        # A few-byte parameter fixture checks refusal before any replay; no image/model run.
        with tempfile.TemporaryDirectory() as folder, mock.patch('torch.cuda.is_available', return_value=False):
            model = torch.nn.Linear(2, 1); model.active = 1
            optimizer = torch.optim.AdamW(model.parameters())
            save_checkpoint(Path(folder), model, optimizer, 0, digest(spec))
            with self.assertRaisesRegex(RuntimeError, 'another run'):
                load_checkpoint(Path(folder), model, optimizer, digest(changed))
        objective.aug_moments['m'][0, 0] += .01
        with self.assertRaisesRegex(RuntimeError, 'content differs'):
            check_augmented_target_binding(objective, spec)
        with self.assertRaisesRegex(RuntimeError, 'without target'):
            check_augmented_target_binding(self.objective(), spec)
        METRICS['changed_aug_target_replay_refused'] = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AugmentedTargetTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'status': 'PASS_CHANGED_AUGMENTATION_PATH_CPU' if result.wasSuccessful() else 'FAIL',
              'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'metrics': METRICS, 'seconds': time.perf_counter()-started,
              'medical_pixels': 0, 'model_execution': 0, 'GPU': False}
    save(args.report, report)
    print(json.dumps(report), flush=True)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
