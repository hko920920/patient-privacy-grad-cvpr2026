import unittest

from dp_accounting import dp_event
from dp_accounting.privacy_accountant import NeighboringRelation
from dp_accounting.rdp import RdpAccountant

from unitdp.benchmark_registry_srswor_v3 import (
    SRSWOR_BENCHMARK_PROFILES_V3,
)
from unitdp_spec_oracle.srswor_rdp_oracle_v3 import (
    SUPPORTED_INTEGER_ORDERS_V3,
    SrsworRdpOracleV3Error,
    fixed_size_srswor_epsilon_v3,
)


class SrsworRdpOracleV3Test(unittest.TestCase):
    def test_direct_decimal_oracle_matches_dp_accounting_per_order(self):
        for profile in SRSWOR_BENCHMARK_PROFILES_V3.values():
            with self.subTest(profile=profile.profile_id):
                direct = fixed_size_srswor_epsilon_v3(
                    actual_noise_multiplier=profile.noise_multiplier,
                    source_dataset_size=profile.source_dataset_size,
                    sample_size=profile.sample_size,
                    steps=profile.base_profile.total_steps,
                    delta=profile.delta,
                    orders=SUPPORTED_INTEGER_ORDERS_V3,
                    sensitivity_multiplier=2.0,
                )
                library = RdpAccountant(
                    orders=SUPPORTED_INTEGER_ORDERS_V3,
                    neighboring_relation=NeighboringRelation.REPLACE_ONE,
                )
                library.compose(
                    dp_event.SampledWithoutReplacementDpEvent(
                        source_dataset_size=profile.source_dataset_size,
                        sample_size=profile.sample_size,
                        event=dp_event.GaussianDpEvent(
                            profile.noise_multiplier / 2.0
                        ),
                    ),
                    count=profile.base_profile.total_steps,
                )
                epsilon, order = (
                    library.get_epsilon_and_optimal_order(profile.delta)
                )
                self.assertLessEqual(
                    abs(direct.epsilon - float(epsilon)),
                    1e-12,
                )
                self.assertEqual(direct.optimal_order, int(order))
                for direct_order, library_rdp in zip(
                    direct.order_results,
                    library._rdp,
                ):
                    self.assertLessEqual(
                        abs(
                            direct_order.composed_rdp
                            - float(library_rdp)
                        ),
                        1e-12,
                    )

    def test_registered_profiles_hit_target_with_expected_orders(self):
        expected = {
            "uci_har_srswor_aaai27_v3": 5,
            "wisdm_srswor_aaai27_v3": 4,
            "sepsis_srswor_aaai27_v3": 4,
        }
        for profile_id, profile in (
            SRSWOR_BENCHMARK_PROFILES_V3.items()
        ):
            result = fixed_size_srswor_epsilon_v3(
                actual_noise_multiplier=profile.noise_multiplier,
                source_dataset_size=profile.source_dataset_size,
                sample_size=profile.sample_size,
                steps=profile.base_profile.total_steps,
                delta=profile.delta,
            )
            self.assertLessEqual(result.epsilon, 8.0 + 1e-10)
            self.assertEqual(result.optimal_order, expected[profile_id])
            self.assertEqual(
                result.normalized_noise_multiplier,
                profile.noise_multiplier / 2.0,
            )

    def test_oracle_rejects_nonregistered_sensitivity_and_bad_domains(self):
        valid = {
            "actual_noise_multiplier": 3.0,
            "source_dataset_size": 21,
            "sample_size": 8,
            "steps": 15,
            "delta": 1e-5,
        }
        mutations = (
            {"sensitivity_multiplier": 1.0},
            {"source_dataset_size": 0},
            {"source_dataset_size": 7},
            {"sample_size": 0},
            {"sample_size": 22},
            {"steps": 0},
            {"delta": 0.0},
            {"delta": 1.0},
            {"actual_noise_multiplier": 0.0},
            {"orders": (2, 6)},
            {"orders": (2, 2)},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(SrsworRdpOracleV3Error):
                    fixed_size_srswor_epsilon_v3(
                        **{**valid, **mutation}
                    )
