#!/usr/bin/env python3
"""CPU-only tests for multi-bank replication aggregation."""

from __future__ import annotations

import unittest

import numpy as np

from run_isic_sd21_multibank_aggregate import geometric_mean, hierarchical_bootstrap


class MultiBankAggregateTests(unittest.TestCase):
    def test_geometric_mean(self) -> None:
        self.assertAlmostEqual(geometric_mean(np.asarray([0.5, 2.0])), 1.0)

    def test_hierarchical_bootstrap_is_deterministic(self) -> None:
        ratios = np.asarray(
            [[0.80 + 0.01 * bank + 0.001 * patient for patient in range(16)] for bank in range(5)]
        )
        first = hierarchical_bootstrap(ratios, replicates=200, seed=7)
        second = hierarchical_bootstrap(ratios, replicates=200, seed=7)
        self.assertEqual(first, second)
        self.assertLess(first["bootstrap_ci95_upper"], 1.0)
        self.assertAlmostEqual(first["geometric_mean_ratio"], geometric_mean(ratios))

    def test_nonpositive_ratio_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            hierarchical_bootstrap(np.zeros((5, 16)), replicates=10)


if __name__ == "__main__":
    unittest.main()
