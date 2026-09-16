#!/usr/bin/env python3
"""CPU-only unit tests for paired-strata oracle enumeration helpers."""

from __future__ import annotations

import itertools
import unittest

import numpy as np

from run_isic_sd21_multipatient_diagnostic import enumerate_stratified_weights
from run_isic_sd21_oracle_headroom import labels_for_pair


class OracleHeadroomTests(unittest.TestCase):
    def test_all_ten_pairs_form_four_nonempty_strata(self) -> None:
        seen: set[tuple[int, int]] = set()
        for pair in itertools.combinations(range(5), 2):
            labels = labels_for_pair(pair)
            groups = [tuple(np.flatnonzero(labels == value)) for value in np.unique(labels)]
            paired = [group for group in groups if len(group) == 2]
            self.assertEqual(len(groups), 4)
            self.assertEqual(paired, [pair])
            seen.add(pair)
        self.assertEqual(len(seen), 10)

    def test_every_pair_estimator_is_unbiased(self) -> None:
        reference = np.full(20, 1.0 / 20.0)
        for pair in itertools.combinations(range(5), 2):
            weights = enumerate_stratified_weights(labels_for_pair(pair), 4, 1)
            self.assertEqual(weights.shape, (512, 20))
            np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-14)
            np.testing.assert_allclose(weights.mean(axis=0), reference, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
