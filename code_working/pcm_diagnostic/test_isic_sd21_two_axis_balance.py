#!/usr/bin/env python3
"""CPU-only tests for exact two-axis balanced enumeration."""

from __future__ import annotations

import unittest

import numpy as np

from run_isic_sd21_two_axis_balance import enumerate_two_axis_weights


class TwoAxisBalanceTests(unittest.TestCase):
    def test_enumeration_sizes_and_unbiasedness(self) -> None:
        expected_sizes = {(1, 4): 5, (2, 2): 60, (4, 1): 120}
        reference = np.full(20, 1.0 / 20.0)
        for (m, r), states in expected_sizes.items():
            weights = enumerate_two_axis_weights(5, 4, m, r)
            with self.subTest(m=m, r=r):
                self.assertEqual(weights.shape, (states, 20))
                np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-14)
                np.testing.assert_allclose(weights.mean(axis=0), reference, atol=1e-14)

    def test_each_state_uses_every_perturbation_once(self) -> None:
        for m, r in ((1, 4), (2, 2), (4, 1)):
            weights = enumerate_two_axis_weights(5, 4, m, r)
            for row in weights:
                matrix = row.reshape(5, 4)
                np.testing.assert_allclose(matrix.sum(axis=0), 0.25, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
