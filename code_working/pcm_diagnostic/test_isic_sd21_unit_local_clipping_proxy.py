#!/usr/bin/env python3
"""CPU-only tests for the patient-unit-local clipping proxy."""

from __future__ import annotations

import unittest

import numpy as np

from run_isic_sd21_multipatient_diagnostic import exact_estimator_metrics
from run_isic_sd21_timestep_bin_ablation import (
    enumerate_distinct_perturbation_weights,
    enumerate_group_balanced_weights,
)
from run_isic_sd21_unit_local_clipping_proxy import (
    b2_global_allocations,
    b2_unit_local_allocations,
    hierarchical_bootstrap,
)


GROUPS = ((0, 1), (2, 3), (4, 5), (6, 7))


class UnitLocalClippingProxyTests(unittest.TestCase):
    def test_b2_global_allocations_are_complete_and_complementary(self) -> None:
        allocations = b2_global_allocations()
        self.assertEqual(len(allocations), 70)
        self.assertEqual(len(set(allocations)), 70)
        first_counts = np.zeros(8, dtype=np.int64)
        for first, second in allocations:
            self.assertEqual(len(first), 4)
            self.assertEqual(len(second), 4)
            self.assertEqual(sorted(first + second), list(range(8)))
            self.assertFalse(set(first) & set(second))
            first_counts[list(first)] += 1
        np.testing.assert_array_equal(first_counts, np.full(8, 35))

    def test_b2_unit_local_allocations_balance_every_patient(self) -> None:
        allocations = b2_unit_local_allocations(GROUPS)
        self.assertEqual(len(allocations), 16)
        self.assertEqual(len(set(allocations)), 16)
        first_counts = np.zeros(8, dtype=np.int64)
        for first, second in allocations:
            self.assertEqual(sorted(first + second), list(range(8)))
            for group in GROUPS:
                self.assertEqual(len(set(first) & set(group)), 1)
                self.assertEqual(len(set(second) & set(group)), 1)
            first_counts[list(first)] += 1
        np.testing.assert_array_equal(first_counts, np.full(8, 8))

    def test_patient_marginal_state_counts_and_unbiasedness(self) -> None:
        global_weights = enumerate_distinct_perturbation_weights(5, 8)
        local_weights = enumerate_group_balanced_weights(5, 8, GROUPS)
        self.assertEqual(global_weights.shape, (8400, 40))
        self.assertEqual(local_weights.shape, (1920, 40))
        for weights in (global_weights, local_weights):
            np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-14)
            np.testing.assert_allclose(
                weights.mean(axis=0), np.full(40, 0.025), atol=1e-14
            )

    def test_exact_postclip_metric_matches_explicit_gradient_vectors(self) -> None:
        rng = np.random.default_rng(260903)
        gradients = rng.normal(scale=0.2, size=(40, 11))
        gram = gradients @ gradients.T
        weights = enumerate_group_balanced_weights(5, 8, GROUPS)
        primary, clipping = exact_estimator_metrics(weights, gram, 11, (0.20,))
        estimates = weights @ gradients
        reference = gradients.mean(axis=0)
        estimate_norms = np.linalg.norm(estimates, axis=1)
        estimate_scales = np.minimum(1.0, 0.20 / estimate_norms)
        reference_scale = min(1.0, 0.20 / np.linalg.norm(reference))
        clipped = estimate_scales[:, None] * estimates
        clipped_reference = reference_scale * reference
        expected_postclip = float(
            np.mean(np.mean((clipped - clipped_reference) ** 2, axis=1))
        )
        expected_preclip = float(
            np.mean(np.mean((estimates - reference) ** 2, axis=1))
        )
        self.assertAlmostEqual(primary["preclip_mse"], expected_preclip, places=13)
        self.assertAlmostEqual(
            clipping[0]["postclip_mse"], expected_postclip, places=13
        )

    def test_no_clipping_limit_reproduces_preclip_mse(self) -> None:
        rng = np.random.default_rng(77)
        gradients = rng.normal(scale=0.01, size=(40, 5))
        gram = gradients @ gradients.T
        weights = enumerate_distinct_perturbation_weights(5, 8)
        primary, clipping = exact_estimator_metrics(weights, gram, 5, (100.0,))
        self.assertAlmostEqual(primary["preclip_mse"], clipping[0]["postclip_mse"], places=14)
        self.assertEqual(clipping[0]["clip_rate"], 0.0)
        self.assertEqual(clipping[0]["reference_clipped"], 0)

    def test_hierarchical_bootstrap_is_deterministic(self) -> None:
        ratios = np.linspace(0.80, 1.05, 80).reshape(5, 16)
        first = hierarchical_bootstrap(ratios, replicates=200, seed=19)
        second = hierarchical_bootstrap(ratios, replicates=200, seed=19)
        self.assertEqual(first, second)
        self.assertLess(first["geometric_mean_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
