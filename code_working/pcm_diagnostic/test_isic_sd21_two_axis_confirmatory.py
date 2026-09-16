#!/usr/bin/env python3
"""CPU-only tests for the independent two-axis confirmatory design."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from run_isic_sd21_two_axis_confirmatory import (
    confirmatory_specs,
    enumerate_bin_balanced_weights,
)


class TwoAxisConfirmatoryTests(unittest.TestCase):
    def test_specs_are_deterministic_and_two_per_bin(self) -> None:
        scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
        first = confirmatory_specs(scheduler)
        second = confirmatory_specs(scheduler)
        self.assertEqual(first, second)
        self.assertEqual([row["perturbation_index"] for row in first], list(range(8)))
        self.assertEqual(
            [sum(row["bin_index"] == bin_index for row in first) for bin_index in range(4)],
            [2, 2, 2, 2],
        )
        self.assertTrue(
            all(
                row["timestep_bin_start"]
                <= row["timestep"]
                < row["timestep_bin_end_exclusive"]
                for row in first
            )
        )

    def test_replication_banks_are_deterministic_and_distinct(self) -> None:
        scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
        banks = [confirmatory_specs(scheduler, f"repl_{index:02d}") for index in range(1, 6)]
        self.assertEqual(banks[0], confirmatory_specs(scheduler, "repl_01"))
        seed_sets = [tuple(row["noise_seed"] for row in bank) for bank in banks]
        self.assertEqual(len(set(seed_sets)), 5)
        self.assertEqual(len({seed for seeds in seed_sets for seed in seeds}), 40)
        original_seeds = {
            row["noise_seed"] for row in confirmatory_specs(scheduler)
        }
        self.assertFalse(original_seeds & {seed for seeds in seed_sets for seed in seeds})

    def test_balanced_enumeration_is_exact_and_unbiased(self) -> None:
        scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
        weights = enumerate_bin_balanced_weights(5, confirmatory_specs(scheduler))
        self.assertEqual(weights.shape, (1920, 40))
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-14)
        np.testing.assert_allclose(weights.mean(axis=0), np.full(40, 1.0 / 40.0), atol=1e-14)
        matrices = weights.reshape(-1, 5, 8)
        for matrix in matrices:
            bin_totals = [matrix[:, start : start + 2].sum() for start in (0, 2, 4, 6)]
            np.testing.assert_allclose(bin_totals, 0.25, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
