#!/usr/bin/env python3
"""CPU-only tests for exact B=2 joint-batch locality analysis."""

from __future__ import annotations

import unittest

import numpy as np

from run_isic_sd21_joint_batch_locality import (
    allocation_statistics,
    all_subsets,
    bank_bootstrap,
    complement,
    exact_joint_pair_metric,
    fixed_subset_weights,
    local_subsets,
)


GROUPS = ((0, 1), (2, 3), (4, 5), (6, 7))


def clip_rows(vectors: np.ndarray, clip_norm: float) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    scales = np.minimum(
        1.0,
        np.divide(clip_norm, norms, out=np.ones_like(norms), where=norms > 0),
    )
    return scales[:, None] * vectors


class JointBatchLocalityTests(unittest.TestCase):
    def test_global_and_local_subset_structures(self) -> None:
        subsets = all_subsets()
        local = local_subsets(GROUPS)
        self.assertEqual(len(subsets), 70)
        self.assertEqual(len(set(subsets)), 70)
        self.assertEqual(len(local), 16)
        self.assertEqual(len(set(local)), 16)
        for subset in subsets:
            self.assertEqual(sorted(subset + complement(subset)), list(range(8)))
        for subset in local:
            for group in GROUPS:
                self.assertEqual(len(set(subset) & set(group)), 1)
            self.assertIn(complement(subset), local)

    def test_fixed_subset_record_states(self) -> None:
        weights = fixed_subset_weights((0, 2, 4, 6))
        self.assertEqual(weights.shape, (120, 40))
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-14)
        used_records = np.sum(weights.reshape(120, 5, 8), axis=2) > 0
        np.testing.assert_array_equal(np.sum(used_records, axis=1), np.full(120, 4))

    def test_complete_patient_marginals_are_unbiased(self) -> None:
        subsets = all_subsets()
        global_weights = np.concatenate(
            [fixed_subset_weights(subset) for subset in subsets], axis=0
        )
        local_weights = np.concatenate(
            [fixed_subset_weights(subset) for subset in local_subsets(GROUPS)], axis=0
        )
        self.assertEqual(global_weights.shape, (8400, 40))
        self.assertEqual(local_weights.shape, (1920, 40))
        np.testing.assert_allclose(global_weights.mean(axis=0), 0.025, atol=1e-14)
        np.testing.assert_allclose(local_weights.mean(axis=0), 0.025, atol=1e-14)

    def test_conditional_moment_formula_matches_explicit_joint_states(self) -> None:
        rng = np.random.default_rng(26090305)
        left_gradients = rng.normal(scale=0.08, size=(40, 9))
        right_gradients = rng.normal(scale=0.08, size=(40, 9))
        left_gram = left_gradients @ left_gradients.T
        right_gram = right_gradients @ right_gradients.T
        cross_gram = left_gradients @ right_gradients.T
        left_subset = (0, 2, 4, 6)
        right_subset = complement(left_subset)
        left_weights = fixed_subset_weights(left_subset)
        right_weights = fixed_subset_weights(right_subset)
        left_stats = allocation_statistics(left_gram, [left_weights], (0.20,))
        right_stats = allocation_statistics(right_gram, [right_weights], (0.20,))
        exact = exact_joint_pair_metric(
            left_stats,
            right_stats,
            cross_gram,
            clip_index=0,
            first_subset_indices=np.asarray([0]),
            second_subset_indices=np.asarray([0]),
            dimension=9,
        )

        left_estimates = clip_rows(left_weights @ left_gradients, 0.20)
        right_estimates = clip_rows(right_weights @ right_gradients, 0.20)
        left_reference = clip_rows(left_gradients.mean(axis=0, keepdims=True), 0.20)[0]
        right_reference = clip_rows(right_gradients.mean(axis=0, keepdims=True), 0.20)[0]
        target = 0.5 * (left_reference + right_reference)
        joint = 0.5 * (
            left_estimates[:, None, :] + right_estimates[None, :, :]
        )
        explicit = float(np.mean(np.mean((joint - target) ** 2, axis=2)))
        self.assertAlmostEqual(exact["joint_mse"], explicit, places=13)

    def test_cross_component_vanishes_for_zero_cross_gram(self) -> None:
        rng = np.random.default_rng(91)
        gradients = rng.normal(size=(40, 5))
        gram = gradients @ gradients.T
        weights = fixed_subset_weights((0, 1, 2, 3))
        stats = allocation_statistics(gram, [weights], (0.20,))
        metric = exact_joint_pair_metric(
            stats,
            stats,
            np.zeros((40, 40)),
            0,
            np.asarray([0]),
            np.asarray([0]),
            5,
        )
        self.assertEqual(metric["cross_component_mse"], 0.0)
        self.assertAlmostEqual(
            metric["joint_mse"], metric["within_component_mse"], places=14
        )

    def test_bank_bootstrap_is_deterministic(self) -> None:
        ratios = np.asarray([0.88, 0.91, 0.93, 0.89, 0.95])
        first = bank_bootstrap(ratios, replicates=500, seed=12)
        second = bank_bootstrap(ratios, replicates=500, seed=12)
        self.assertEqual(first, second)
        self.assertLess(first["bootstrap_ci95_upper"], 1.0)


if __name__ == "__main__":
    unittest.main()
