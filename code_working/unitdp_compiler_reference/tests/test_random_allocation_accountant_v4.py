from __future__ import annotations

import math
import unittest

from unitdp.random_allocation_accountant_v4 import (
    RANDOM_ALLOCATION_RDP_ORDERS_V4,
    RandomAllocationAccountingV4Error,
    account_random_allocation_v4,
    calibrate_random_allocation_sigma_v4,
    integer_partitions_v4,
    random_allocation_remove_rdp_v4,
)
from unitdp_spec_oracle.random_allocation_oracle_v4 import (
    account_random_allocation_oracle_v4,
)


CASES = (
    (
        "uci",
        15,
        6,
        1.2295752282782475,
        4,
        6.7189308002345465,
    ),
    (
        "wisdm",
        20,
        6,
        1.090645987511382,
        4,
        6.952359145280982,
    ),
    (
        "sepsis",
        50,
        5,
        0.776172784941656,
        3,
        7.0705960633291065,
    ),
)


class RandomAllocationAccountantV4Tests(unittest.TestCase):
    def test_partition_family_is_exact_and_unique(self):
        self.assertEqual(
            integer_partitions_v4(5, max_parts=2),
            ((5,), (4, 1), (3, 2)),
        )
        partitions = integer_partitions_v4(16, max_parts=10)
        self.assertEqual(len(partitions), len(set(partitions)))
        self.assertTrue(all(sum(row) == 16 for row in partitions))
        self.assertTrue(all(len(row) <= 10 for row in partitions))
        self.assertTrue(
            all(
                all(left >= right for left, right in zip(row, row[1:]))
                for row in partitions
            )
        )

    def test_production_matches_independent_high_precision_oracle(self):
        for name, steps, selected, sigma, order, _ in CASES:
            with self.subTest(name=name):
                result = account_random_allocation_v4(
                    num_steps=steps,
                    num_selected=selected,
                    num_epochs=1,
                    sigma=sigma,
                    delta=1e-5,
                )
                oracle = account_random_allocation_oracle_v4(
                    num_steps=steps,
                    num_selected=selected,
                    num_epochs=1,
                    sigma=sigma,
                    delta=1e-5,
                    orders=RANDOM_ALLOCATION_RDP_ORDERS_V4,
                )
                self.assertLessEqual(
                    abs(result.epsilon - oracle.epsilon),
                    1e-10,
                )
                self.assertLessEqual(
                    abs(
                        result.epsilon_remove
                        - oracle.epsilon_remove
                    ),
                    1e-10,
                )
                self.assertLessEqual(
                    abs(result.epsilon_add - oracle.epsilon_add),
                    1e-10,
                )
                self.assertEqual(
                    result.optimal_remove_order,
                    oracle.optimal_remove_order,
                )
                self.assertEqual(result.optimal_remove_order, order)
                self.assertLessEqual(result.epsilon, 8.0 + 1e-12)
                self.assertEqual(result.epsilon, result.epsilon_remove)
                for (_, observed), (_, reference) in zip(
                    result.remove_rdp_by_order,
                    oracle.remove_rdp_by_order,
                ):
                    self.assertLessEqual(
                        abs(observed - reference),
                        1e-10,
                    )

    def test_external_pld_probe_is_non_authorizing_and_tighter(self):
        for (
            name,
            steps,
            selected,
            sigma,
            _,
            external_pld_upper,
        ) in CASES:
            with self.subTest(name=name):
                result = account_random_allocation_v4(
                    num_steps=steps,
                    num_selected=selected,
                    num_epochs=1,
                    sigma=sigma,
                    delta=1e-5,
                )
                self.assertLess(external_pld_upper, result.epsilon)

    def test_calibration_reproduces_prespecified_sigmas(self):
        for name, steps, selected, sigma, _, _ in CASES:
            with self.subTest(name=name):
                calibrated = calibrate_random_allocation_sigma_v4(
                    target_epsilon=8.0,
                    delta=1e-5,
                    num_steps=steps,
                    num_selected=selected,
                    num_epochs=1,
                )
                self.assertLessEqual(abs(calibrated - sigma), 2e-13)

    def test_more_noise_strictly_reduces_the_bound(self):
        for name, steps, selected, sigma, _, _ in CASES:
            with self.subTest(name=name):
                lower = account_random_allocation_v4(
                    num_steps=steps,
                    num_selected=selected,
                    num_epochs=1,
                    sigma=sigma,
                    delta=1e-5,
                )
                higher = account_random_allocation_v4(
                    num_steps=steps,
                    num_selected=selected,
                    num_epochs=1,
                    sigma=sigma * 1.01,
                    delta=1e-5,
                )
                self.assertLess(higher.epsilon, lower.epsilon)
                self.assertLess(
                    higher.epsilon_remove,
                    lower.epsilon_remove,
                )

    def test_each_prespecified_parameter_changes_the_result(self):
        baseline = account_random_allocation_v4(
            num_steps=20,
            num_selected=6,
            num_epochs=1,
            sigma=1.090645987511382,
            delta=1e-5,
        )
        mutations = (
            {"num_steps": 21},
            {"num_selected": 5},
            {"num_epochs": 2},
            {"sigma": 1.090645987511383},
            {"delta": 1.1e-5},
            {"orders": tuple(range(2, 16))},
        )
        common = {
            "num_steps": 20,
            "num_selected": 6,
            "num_epochs": 1,
            "sigma": 1.090645987511382,
            "delta": 1e-5,
        }
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                changed = account_random_allocation_v4(
                    **{**common, **mutation}
                )
                self.assertNotEqual(changed, baseline)

    def test_floor_reduction_does_not_erase_query_identity(self):
        first = account_random_allocation_v4(
            num_steps=20,
            num_selected=6,
            num_epochs=1,
            sigma=1.090645987511382,
            delta=1e-5,
        )
        second = account_random_allocation_v4(
            num_steps=21,
            num_selected=6,
            num_epochs=1,
            sigma=1.090645987511382,
            delta=1e-5,
        )
        self.assertEqual(first.epsilon, second.epsilon)
        self.assertEqual(
            first.remove_rdp_by_order,
            second.remove_rdp_by_order,
        )
        self.assertNotEqual(first.num_steps, second.num_steps)
        self.assertNotEqual(first, second)

    def test_invalid_queries_fail_closed(self):
        invalid = (
            {"num_steps": True},
            {"num_steps": 0},
            {"num_selected": 0},
            {"num_selected": 21},
            {"num_epochs": 0},
            {"sigma": 0.0},
            {"sigma": math.inf},
            {"delta": 0.0},
            {"delta": 1.0},
            {"orders": ()},
            {"orders": (3, 2)},
            {"orders": (2, 2)},
            {"orders": (1, 2)},
        )
        common = {
            "num_steps": 20,
            "num_selected": 6,
            "num_epochs": 1,
            "sigma": 1.1,
            "delta": 1e-5,
        }
        for mutation in invalid:
            with self.subTest(mutation=mutation):
                with self.assertRaises(
                    RandomAllocationAccountingV4Error
                ):
                    account_random_allocation_v4(
                        **{**common, **mutation}
                    )

    def test_single_step_reduces_to_gaussian_rdp(self):
        for alpha in (2, 3, 4, 8, 16):
            with self.subTest(alpha=alpha):
                sigma = 1.7
                observed = random_allocation_remove_rdp_v4(
                    alpha=alpha,
                    sigma=sigma,
                    steps_per_single_allocation=1,
                )
                expected = alpha / (2.0 * sigma * sigma)
                self.assertLessEqual(abs(observed - expected), 1e-12)


if __name__ == "__main__":
    unittest.main()
