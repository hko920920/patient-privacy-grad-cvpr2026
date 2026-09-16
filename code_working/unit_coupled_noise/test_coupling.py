from __future__ import annotations

import unittest

import numpy as np

from coupling import METHODS, paired_plan


class CouplingTests(unittest.TestCase):
    def test_all_methods_are_deterministic(self) -> None:
        for method in METHODS:
            first = paired_plan(
                salt="test", bank="b1", patient_id="p1", method=method,
                record_count=4, num_train_timesteps=1000,
            )
            second = paired_plan(
                salt="test", bank="b1", patient_id="p1", method=method,
                record_count=4, num_train_timesteps=1000,
            )
            self.assertEqual(first, second)

    def test_antithetic_pairs_reuse_seed_and_flip_sign(self) -> None:
        plan = paired_plan(
            salt="test", bank="b1", patient_id="p1",
            method="antithetic_pair_shared_t", record_count=4,
            num_train_timesteps=1000,
        )
        for left, right in ((0, 1), (2, 3)):
            self.assertEqual(plan[left]["noise_seed"], plan[right]["noise_seed"])
            self.assertEqual(plan[left]["noise_sign"], 1)
            self.assertEqual(plan[right]["noise_sign"], -1)
            self.assertEqual(plan[left]["timestep"], plan[right]["timestep"])

    def test_independent_t_antithetic_does_not_force_shared_t(self) -> None:
        unequal = 0
        for bank_index in range(64):
            plan = paired_plan(
                salt="test", bank=f"b{bank_index}", patient_id="p1",
                method="antithetic_independent_t", record_count=4,
                num_train_timesteps=1000,
            )
            unequal += int(plan[0]["timestep"] != plan[1]["timestep"])
            self.assertEqual(plan[0]["noise_seed"], plan[1]["noise_seed"])
        self.assertGreater(unequal, 50)

    def test_each_timestep_marginal_is_uniform(self) -> None:
        counts = np.zeros((4, 10), dtype=np.int64)
        trials = 20_000
        for bank_index in range(trials):
            plan = paired_plan(
                salt="uniformity", bank=f"b{bank_index}", patient_id="p1",
                method="antithetic_pair_shared_t", record_count=4,
                num_train_timesteps=10,
            )
            for item in plan:
                counts[item["record_index"], item["timestep"]] += 1
        expected = trials / 10
        self.assertLess(float(np.max(np.abs(counts - expected) / expected)), 0.07)

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            paired_plan(
                salt="x", bank="b", patient_id="p", method="unknown",
                record_count=4, num_train_timesteps=1000,
            )
        with self.assertRaises(ValueError):
            paired_plan(
                salt="x", bank="b", patient_id="p", method=METHODS[0],
                record_count=0, num_train_timesteps=1000,
            )


if __name__ == "__main__":
    unittest.main()

