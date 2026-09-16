#!/usr/bin/env python3
"""CPU-only tests for timestep-bin mechanism ablation."""

from __future__ import annotations

import unittest

import numpy as np

from run_isic_sd21_timestep_bin_ablation import (
    enumerate_distinct_perturbation_weights,
    enumerate_group_balanced_weights,
    mse_from_covariance,
    pair_partitions,
    weight_covariance,
)


class TimestepBinAblationTests(unittest.TestCase):
    def test_all_pair_partitions_are_unique_and_complete(self) -> None:
        partitions = list(pair_partitions(tuple(range(8))))
        self.assertEqual(len(partitions), 105)
        self.assertEqual(len(set(partitions)), 105)
        for partition in partitions:
            self.assertEqual(sorted(value for pair in partition for value in pair), list(range(8)))

    def test_state_counts_and_unbiasedness(self) -> None:
        distinct = enumerate_distinct_perturbation_weights(5, 8)
        grouped = enumerate_group_balanced_weights(
            5, 8, ((0, 1), (2, 3), (4, 5), (6, 7))
        )
        self.assertEqual(distinct.shape, (8400, 40))
        self.assertEqual(grouped.shape, (1920, 40))
        for weights in (distinct, grouped):
            np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-14)
            np.testing.assert_allclose(weights.mean(axis=0), np.full(40, 0.025), atol=1e-14)

    def test_covariance_mse_matches_explicit_vectors(self) -> None:
        rng = np.random.default_rng(23)
        gradients = rng.normal(size=(40, 9))
        weights = enumerate_group_balanced_weights(
            5, 8, ((0, 1), (2, 3), (4, 5), (6, 7))
        )
        covariance = weight_covariance(weights)
        gram = gradients @ gradients.T
        covariance_mse = mse_from_covariance(covariance, gram, 9)
        estimates = weights @ gradients
        reference = gradients.mean(axis=0)
        explicit_mse = float(np.mean(np.mean((estimates - reference) ** 2, axis=1)))
        self.assertAlmostEqual(covariance_mse, explicit_mse, places=12)


if __name__ == "__main__":
    unittest.main()
