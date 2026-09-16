import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unitdp.accountant import (
    calibrate_noise_srswor_replace_one,
    epsilon_for_srswor_replace_one_noise,
    group_privacy_conversion,
    required_base_delta_for_group,
)
from unitdp.accountant_registry import get_accountant_route
from unitdp.certificate import build_certificate
from unitdp.compiler import compile_contract
from unitdp.contract import UnitContract
from unitdp.els_accountant import (
    ElsMixtureConfig,
    ElsMogPldConfig,
    binomial_sensitivity_distribution,
    calibrate_noise_els_mixture,
    calibrate_noise_els_mog_pld,
    epsilon_for_els_mixture,
    epsilon_for_els_mog_pld,
    hypergeometric_sensitivity_distribution,
)
from unitdp.models import build_classifier, build_sequence_classifier
from unitdp.owa_dpsgd import OwaDpsgdConfig, train_owa_dpsgd
from unitdp.rng import gaussian_noise_sanity, make_random_source
from unitdp.synthetic import make_synthetic_window_data


class UnitDpCoreTest(unittest.TestCase):
    def test_accountant_registry_scopes_external_mog(self):
        owa = get_accountant_route("owa_owner_rdp")
        implemented_mog = get_accountant_route("mog_pld_els_owner")
        fixed_mog = get_accountant_route("fixed_size_mog_pld_els_owner")
        mog = get_accountant_route("external_tight_els_mog")
        srswor = get_accountant_route("external_owner_fixed_size_rdp")
        fixed_els = get_accountant_route("external_fixed_size_els_owner")
        shuffled_els = get_accountant_route("external_shuffled_epoch_els_owner")

        self.assertEqual(owa.status, "legacy_not_release_validated")
        self.assertEqual(owa.sampling_unit, "owner")
        self.assertEqual(implemented_mog.status, "legacy_not_execution_bound")
        self.assertEqual(implemented_mog.sampling_unit, "window")
        self.assertEqual(fixed_mog.status, "legacy_not_execution_bound")
        self.assertEqual(fixed_mog.sampling_unit, "window")
        self.assertEqual(fixed_mog.accounting_unit, "owner")
        self.assertEqual(mog.status, "external_hook")
        self.assertEqual(mog.accounting_unit, "owner")
        self.assertEqual(srswor.status, "implemented_probe")
        self.assertEqual(srswor.sampling_unit, "owner")
        self.assertEqual(fixed_els.status, "external_hook")
        self.assertEqual(fixed_els.sampling_unit, "window")
        self.assertEqual(fixed_els.accounting_unit, "owner")
        self.assertEqual(shuffled_els.status, "external_hook")
        self.assertEqual(shuffled_els.sampling_unit, "window")

    def test_rng_backends_and_gaussian_sanity(self):
        research = make_random_source("research_default", seed=11)
        secure = make_random_source("system_csprng", seed=11)

        self.assertFalse(research.metadata().secure_rng)
        self.assertTrue(secure.metadata().secure_rng)

        samples = research.normal_array(4096, std=2.0)
        sanity = gaussian_noise_sanity(
            samples,
            expected_std=2.0,
            mean_abs_tolerance=0.12,
            std_relative_tolerance=0.12,
        )
        self.assertTrue(sanity["passed"])

    def test_model_factory_builds_linear_and_mlp_classifiers(self):
        batch = torch.zeros(3, 5)
        linear = build_classifier(5, 2, "linear")
        mlp = build_classifier(5, 2, "mlp", hidden_dim=7)

        self.assertEqual(tuple(linear(batch).shape), (3, 2))
        self.assertEqual(tuple(mlp(batch).shape), (3, 2))

    def test_model_factory_builds_sequence_cnn(self):
        batch = torch.zeros(3, 4, 16)
        model = build_sequence_classifier(4, 2, "cnn1d", hidden_channels=5, sequence_length=16)

        self.assertEqual(tuple(model(batch).shape), (3, 2))

    def test_group_delta_requirement_matches_conversion(self):
        target_delta = 1e-5
        epsilon = 0.5
        k = 4
        base_delta = required_base_delta_for_group(target_delta, epsilon, k)
        converted = group_privacy_conversion(epsilon, base_delta, k)

        self.assertFalse(converted["vacuous"])
        self.assertAlmostEqual(float(converted["delta"]), target_delta)

    def test_els_mixture_accountant_is_monotone(self):
        base = {
            "sample_rate": 0.05,
            "steps": 5,
            "delta": 1e-5,
            "alphas": [2, 4, 8],
            "quadrature_nodes": 32,
        }
        eps_k1 = epsilon_for_els_mixture(
            ElsMixtureConfig(owner_kappa=1, noise_multiplier=3.0, **base)
        )
        eps_k4 = epsilon_for_els_mixture(
            ElsMixtureConfig(owner_kappa=4, noise_multiplier=3.0, **base)
        )
        eps_more_noise = epsilon_for_els_mixture(
            ElsMixtureConfig(owner_kappa=4, noise_multiplier=5.0, **base)
        )

        self.assertGreater(eps_k4, eps_k1)
        self.assertLess(eps_more_noise, eps_k4)

    def test_els_mixture_noise_calibration(self):
        target = 5.0
        noise = calibrate_noise_els_mixture(
            target_epsilon=target,
            owner_kappa=3,
            sample_rate=0.05,
            steps=5,
            delta=1e-5,
            alphas=[2, 4, 8],
            quadrature_nodes=32,
        )
        epsilon = epsilon_for_els_mixture(
            ElsMixtureConfig(
                owner_kappa=3,
                sample_rate=0.05,
                noise_multiplier=noise,
                steps=5,
                delta=1e-5,
                alphas=[2, 4, 8],
                quadrature_nodes=32,
            )
        )

        self.assertLessEqual(epsilon, target + 1e-6)

    def test_srswor_replace_one_accountant_calibrates(self):
        target = 6.0
        noise = calibrate_noise_srswor_replace_one(
            target_epsilon=target,
            source_dataset_size=20,
            sample_size=5,
            steps=4,
            delta=1e-5,
            alphas=[2, 4, 8, 16],
        )
        epsilon = epsilon_for_srswor_replace_one_noise(
            noise_multiplier=noise,
            source_dataset_size=20,
            sample_size=5,
            steps=4,
            delta=1e-5,
            alphas=[2, 4, 8, 16],
        )

        self.assertLessEqual(epsilon, target + 1e-6)

    def test_mog_pld_accountant_is_monotone_and_calibrates(self):
        base = {
            "owner_kappa": 3,
            "sample_rate": 0.05,
            "steps": 3,
            "delta": 1e-5,
            "value_discretization_interval": 5e-3,
        }
        eps_less_noise = epsilon_for_els_mog_pld(
            ElsMogPldConfig(noise_multiplier=4.0, **base)
        )
        eps_more_noise = epsilon_for_els_mog_pld(
            ElsMogPldConfig(noise_multiplier=6.0, **base)
        )
        self.assertLess(eps_more_noise, eps_less_noise)

        target = 5.0
        noise = calibrate_noise_els_mog_pld(
            target_epsilon=target,
            **base,
        )
        epsilon = epsilon_for_els_mog_pld(
            ElsMogPldConfig(noise_multiplier=noise, **base)
        )
        self.assertLessEqual(epsilon, target + 1e-2)

    def test_binomial_sensitivity_distribution(self):
        sensitivities, probabilities = binomial_sensitivity_distribution(2, 0.25)

        self.assertEqual(sensitivities, [0.0, 1.0, 2.0])
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertAlmostEqual(probabilities[0], 0.75 * 0.75)
        self.assertAlmostEqual(probabilities[1], 2 * 0.25 * 0.75)
        self.assertAlmostEqual(probabilities[2], 0.25 * 0.25)

    def test_fixed_size_mog_pld_accountant_calibrates(self):
        sensitivities, probabilities = hypergeometric_sensitivity_distribution(
            population_size=100,
            owner_kappa=10,
            sample_size=20,
        )

        self.assertEqual(sensitivities[0], 0.0)
        self.assertAlmostEqual(sum(probabilities), 1.0, places=12)
        self.assertGreater(probabilities[2], 0.0)

        base = {
            "owner_kappa": 10,
            "sample_rate": 0.2,
            "steps": 5,
            "delta": 1e-5,
            "value_discretization_interval": 0.01,
            "sampling_model": "fixed_size",
            "population_size": 100,
            "sample_size": 20,
        }
        eps_less_noise = epsilon_for_els_mog_pld(
            ElsMogPldConfig(noise_multiplier=4.0, **base)
        )
        eps_more_noise = epsilon_for_els_mog_pld(
            ElsMogPldConfig(noise_multiplier=6.0, **base)
        )
        self.assertLess(eps_more_noise, eps_less_noise)

        target = 5.0
        noise = calibrate_noise_els_mog_pld(
            target_epsilon=target,
            log_mass_truncation_bound=-40.0,
            tail_mass_truncation=1e-12,
            **base,
        )
        epsilon = epsilon_for_els_mog_pld(
            ElsMogPldConfig(
                noise_multiplier=noise,
                log_mass_truncation_bound=-40.0,
                tail_mass_truncation=1e-12,
                **base,
            )
        )
        self.assertLessEqual(epsilon, target + 1e-2)

    def test_owner_contract_rejects_window_accountant(self):
        contract = UnitContract(
            privacy_unit="owner",
            mapping="missing.csv",
            mechanism={
                "sampling_unit": "window",
                "clipping_unit": "window",
                "noising_unit": "window",
            },
            accountant={"unit": "window"},
        )
        with self.assertRaises(ValueError):
            contract.validate()

    def test_owner_route_rejects_multi_owner_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping_path = Path(tmp) / "multi_owner.csv"
            mapping_path.write_text(
                "\n".join(
                    [
                        "scenario,window_id,owner_id,owner_ids,start,end,row_index,label",
                        "synthetic,w0,u0,u0;u1,0,4,0,0",
                        "synthetic,w1,u1,u1,4,8,1,1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            contract = UnitContract(
                privacy_unit="owner",
                mapping=str(mapping_path),
                owner_policy={"type": "all"},
                mechanism={
                    "sampling_unit": "owner",
                    "clipping_unit": "owner",
                    "noising_unit": "owner",
                },
                accountant={"backend": "rdp", "unit": "owner"},
            )

            with self.assertRaisesRegex(ValueError, "single-attribution"):
                compile_contract(contract)

    def test_event_route_fails_closed_for_overlap_and_compiles_after_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping_path = Path(tmp) / "event_overlap.csv"
            mapping_path.write_text(
                "\n".join(
                    [
                        "scenario,window_id,owner_id,owner_ids,start,end,row_index,label",
                        "synthetic,w0,u0,u0,0,4,0,0",
                        "synthetic,w1,u0,u0,2,6,1,1",
                        "synthetic,w2,u1,u1,0,4,2,0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            overlapping = UnitContract(
                privacy_unit="event",
                mapping=str(mapping_path),
                owner_policy={"type": "all"},
                mechanism={
                    "sampling_unit": "window",
                    "clipping_unit": "window",
                    "noising_unit": "window",
                },
                accountant={"backend": "rdp", "unit": "event"},
            )
            route = compile_contract(overlapping)
            self.assertEqual(route.selected_mapping.event_kappa(), 2)
            self.assertEqual(route.route_status, "requires_external_event_accountant")
            self.assertTrue(any("external structured accountant" in note for note in route.route_notes))

            capped = UnitContract(
                privacy_unit="event",
                mapping=str(mapping_path),
                owner_policy={"type": "first_k", "max_windows_per_owner": 1},
                mechanism={
                    "sampling_unit": "window",
                    "clipping_unit": "window",
                    "noising_unit": "window",
                },
                accountant={"backend": "rdp", "unit": "event"},
            )
            capped_route = compile_contract(capped)
            self.assertEqual(capped_route.selected_mapping.event_kappa(), 1)
            self.assertEqual(capped_route.route_status, "compiled")
            self.assertTrue(any("event-direct" in note for note in capped_route.route_notes))

    def test_compile_and_train_synthetic_owner_route(self):
        data = make_synthetic_window_data(
            train_owners=8,
            test_owners=4,
            sequence_length=32,
            window_length=8,
            stride=4,
            channels=3,
            seed=7,
        )
        with tempfile.TemporaryDirectory() as tmp:
            mapping_path = Path(tmp) / "mapping.csv"
            data.train_mapping.write_csv(mapping_path)
            contract = UnitContract(
                privacy_unit="owner",
                mapping=str(mapping_path),
                owner_policy={
                    "type": "support_balanced_cap",
                    "max_windows_per_owner": 4,
                    "windows_per_owner_per_step": 2,
                },
                mechanism={
                    "sampling_unit": "owner",
                    "clipping_unit": "owner",
                    "noising_unit": "owner",
                },
                accountant={"backend": "rdp", "unit": "owner"},
                training={"owner_batch_size": 4},
            )
            route = compile_contract(contract)
            self.assertEqual(route.selected_mapping.stats()["owner_kappa"], 4)
            model = nn.Linear(data.train_x.shape[1], int(data.train_y.max() + 1))
            result = train_owa_dpsgd(
                model=model,
                train_x=data.train_x.astype(np.float32),
                train_y=data.train_y,
                test_x=data.test_x.astype(np.float32),
                test_y=data.test_y,
                route=route,
                config=OwaDpsgdConfig(
                    epochs=1,
                    owner_batch_size=4,
                    learning_rate=0.01,
                    target_epsilon=20.0,
                    target_delta=1e-5,
                    seed=3,
                ),
            )
            certificate = build_certificate(route, result.to_dict())
            self.assertEqual(certificate.mechanism_units["accounting_unit"], "owner")
            self.assertTrue(certificate.implementation_checks["owner_mechanism_units_aligned"])
            self.assertTrue(certificate.implementation_checks["selected_windows_single_attribution"])
            self.assertTrue(
                certificate.implementation_checks["accountant_sample_rate_matches_owner_sampling"]
            )
            self.assertTrue(certificate.implementation_checks["bernoulli_owner_sampling_recorded"])
            self.assertTrue(
                certificate.implementation_checks["one_clipped_vector_per_sampled_owner_recorded"]
            )
            self.assertTrue(certificate.implementation_checks["internal_owner_sampler_recorded"])
            self.assertTrue(certificate.implementation_checks["public_schedule_digest_recorded"])
            self.assertTrue(certificate.implementation_checks["sampler_trace_digest_recorded"])
            self.assertTrue(certificate.implementation_checks["execution_binding_digest_recorded"])
            self.assertTrue(certificate.implementation_checks["rng_backend_recorded"])
            self.assertTrue(certificate.implementation_checks["research_rng_caveat_recorded"])
            self.assertFalse(certificate.implementation_checks["production_secure_rng_used"])
            self.assertEqual(
                certificate.privacy_assumptions["loader_binding"],
                "internal_owner_sampler_no_external_dataloader",
            )
            self.assertEqual(
                certificate.training_result["route_implementation_id"],
                "unitdp.owa_dpsgd.train_owa_dpsgd:v1",
            )
            self.assertEqual(len(certificate.accountant["public_schedule_sha256"]), 64)
            self.assertEqual(len(certificate.privacy_assumptions["sampler_trace_sha256"]), 64)
            self.assertEqual(len(certificate.privacy_assumptions["execution_binding_sha256"]), 64)
            self.assertEqual(certificate.release_binding["binding_status"], "recorded")
            self.assertEqual(len(certificate.release_binding["model_artifact_sha256"]), 64)
            self.assertEqual(len(certificate.release_binding["training_code_sha256"]), 64)
            self.assertEqual(len(certificate.release_binding["release_binding_sha256"]), 64)
            self.assertEqual(result.sampling_scheme, "independent_bernoulli_owner")
            self.assertEqual(result.accountant_sample_rate, result.owner_sample_rate)
            self.assertGreater(result.epsilon, 0.0)

    def test_compile_and_train_synthetic_srswor_owner_route(self):
        data = make_synthetic_window_data(
            train_owners=8,
            test_owners=4,
            sequence_length=32,
            window_length=8,
            stride=4,
            channels=3,
            seed=23,
        )
        with tempfile.TemporaryDirectory() as tmp:
            mapping_path = Path(tmp) / "mapping.csv"
            data.train_mapping.write_csv(mapping_path)
            contract = UnitContract(
                privacy_unit="owner",
                mapping=str(mapping_path),
                owner_policy={
                    "type": "support_balanced_cap",
                    "max_windows_per_owner": 4,
                    "windows_per_owner_per_step": 2,
                },
                mechanism={
                    "sampling_unit": "owner",
                    "clipping_unit": "owner",
                    "noising_unit": "owner",
                },
                accountant={"backend": "rdp", "unit": "owner"},
                training={"owner_batch_size": 4},
            )
            route = compile_contract(contract)
            model = nn.Linear(data.train_x.shape[1], int(data.train_y.max() + 1))
            result = train_owa_dpsgd(
                model=model,
                train_x=data.train_x.astype(np.float32),
                train_y=data.train_y,
                test_x=data.test_x.astype(np.float32),
                test_y=data.test_y,
                route=route,
                config=OwaDpsgdConfig(
                    epochs=1,
                    owner_batch_size=4,
                    learning_rate=0.01,
                    target_epsilon=20.0,
                    target_delta=1e-5,
                    seed=3,
                    owner_sampling_scheme="fixed_without_replacement",
                ),
            )

            self.assertEqual(result.sampling_scheme, "fixed_without_replacement_owner")
            self.assertEqual(result.accountant_backend, "dp_accounting_rdp_srswor_replace_one")
            self.assertEqual(result.sampled_owner_count_min, 4)
            self.assertEqual(result.sampled_owner_count_max, 4)
            self.assertGreater(result.epsilon, 0.0)

    def test_compile_and_train_synthetic_shuffled_epoch_owner_route(self):
        data = make_synthetic_window_data(
            train_owners=8,
            test_owners=4,
            sequence_length=32,
            window_length=8,
            stride=4,
            channels=3,
            seed=29,
        )
        with tempfile.TemporaryDirectory() as tmp:
            mapping_path = Path(tmp) / "mapping.csv"
            data.train_mapping.write_csv(mapping_path)
            contract = UnitContract(
                privacy_unit="owner",
                mapping=str(mapping_path),
                owner_policy={
                    "type": "support_balanced_cap",
                    "max_windows_per_owner": 4,
                    "windows_per_owner_per_step": 2,
                },
                mechanism={
                    "sampling_unit": "owner",
                    "clipping_unit": "owner",
                    "noising_unit": "owner",
                },
                accountant={"backend": "rdp", "unit": "owner"},
                training={"owner_batch_size": 4},
            )
            route = compile_contract(contract)
            model = nn.Linear(data.train_x.shape[1], int(data.train_y.max() + 1))
            result = train_owa_dpsgd(
                model=model,
                train_x=data.train_x.astype(np.float32),
                train_y=data.train_y,
                test_x=data.test_x.astype(np.float32),
                test_y=data.test_y,
                route=route,
                config=OwaDpsgdConfig(
                    epochs=1,
                    owner_batch_size=4,
                    learning_rate=0.01,
                    target_epsilon=20.0,
                    target_delta=1e-5,
                    seed=3,
                    owner_sampling_scheme="shuffled_epoch",
                ),
            )

            self.assertEqual(result.sampling_scheme, "shuffled_epoch_owner_conservative_srswor")
            self.assertEqual(
                result.accountant_backend,
                "dp_accounting_rdp_srswor_replace_one_conservative_epoch",
            )
            self.assertEqual(result.sampled_owner_count_min, 4)
            self.assertEqual(result.sampled_owner_count_max, 4)
            self.assertGreater(result.epsilon, 0.0)

    def test_compile_and_train_synthetic_secure_rng_route(self):
        data = make_synthetic_window_data(
            train_owners=4,
            test_owners=2,
            sequence_length=24,
            window_length=8,
            stride=4,
            channels=2,
            seed=31,
        )
        with tempfile.TemporaryDirectory() as tmp:
            mapping_path = Path(tmp) / "mapping.csv"
            data.train_mapping.write_csv(mapping_path)
            contract = UnitContract(
                privacy_unit="owner",
                mapping=str(mapping_path),
                owner_policy={
                    "type": "support_balanced_cap",
                    "max_windows_per_owner": 2,
                    "windows_per_owner_per_step": 1,
                },
                mechanism={
                    "sampling_unit": "owner",
                    "clipping_unit": "owner",
                    "noising_unit": "owner",
                },
                accountant={"backend": "rdp", "unit": "owner"},
                training={"owner_batch_size": 2},
            )
            route = compile_contract(contract)
            model = nn.Linear(data.train_x.shape[1], int(data.train_y.max() + 1))
            result = train_owa_dpsgd(
                model=model,
                train_x=data.train_x.astype(np.float32),
                train_y=data.train_y,
                test_x=data.test_x.astype(np.float32),
                test_y=data.test_y,
                route=route,
                config=OwaDpsgdConfig(
                    epochs=1,
                    owner_batch_size=2,
                    learning_rate=0.01,
                    target_epsilon=20.0,
                    target_delta=1e-5,
                    seed=3,
                    rng_backend="system_csprng",
                ),
            )
            certificate = build_certificate(route, result.to_dict())

            self.assertTrue(result.secure_rng)
            self.assertEqual(result.rng_security_mode, "system_csprng_prototype")
            self.assertTrue(certificate.implementation_checks["production_secure_rng_used"])
            self.assertIn("SystemRandom", certificate.privacy_assumptions["rng_caveat"])

    def test_owa_training_supports_public_aggregation_modes(self):
        data = make_synthetic_window_data(
            train_owners=6,
            test_owners=3,
            sequence_length=24,
            window_length=8,
            stride=4,
            channels=2,
            seed=17,
        )
        with tempfile.TemporaryDirectory() as tmp:
            mapping_path = Path(tmp) / "mapping.csv"
            data.train_mapping.write_csv(mapping_path)
            contract = UnitContract(
                privacy_unit="owner",
                mapping=str(mapping_path),
                owner_policy={
                    "type": "support_balanced_cap",
                    "max_windows_per_owner": 3,
                    "windows_per_owner_per_step": 2,
                },
                mechanism={
                    "sampling_unit": "owner",
                    "clipping_unit": "owner",
                    "noising_unit": "owner",
                },
                accountant={"backend": "rdp", "unit": "owner"},
                training={"owner_batch_size": 3},
            )
            route = compile_contract(contract)
            for mode in ["mean", "sum", "fixed_normalizer"]:
                model = nn.Linear(data.train_x.shape[1], int(data.train_y.max() + 1))
                result = train_owa_dpsgd(
                    model=model,
                    train_x=data.train_x.astype(np.float32),
                    train_y=data.train_y,
                    test_x=data.test_x.astype(np.float32),
                    test_y=data.test_y,
                    route=route,
                    config=OwaDpsgdConfig(
                        epochs=1,
                        owner_batch_size=3,
                        learning_rate=0.01,
                        target_epsilon=20.0,
                        target_delta=1e-5,
                        seed=3,
                        owner_aggregation=mode,
                        fixed_window_normalizer=2,
                    ),
                )
                self.assertEqual(result.config["owner_aggregation"], mode)
                self.assertGreater(result.epsilon, 0.0)

    def test_owa_training_rejects_bad_aggregation_config(self):
        with self.assertRaisesRegex(ValueError, "owner_aggregation"):
            OwaDpsgdConfig(owner_aggregation="median").validate()
        with self.assertRaisesRegex(ValueError, "fixed_window_normalizer"):
            OwaDpsgdConfig(owner_aggregation="fixed_normalizer").validate()

    def test_owa_training_rejects_non_owner_route(self):
        data = make_synthetic_window_data(
            train_owners=4,
            test_owners=2,
            sequence_length=24,
            window_length=8,
            stride=4,
            channels=2,
            seed=9,
        )
        with tempfile.TemporaryDirectory() as tmp:
            mapping_path = Path(tmp) / "mapping.csv"
            data.train_mapping.write_csv(mapping_path)
            contract = UnitContract(
                privacy_unit="window",
                mapping=str(mapping_path),
                owner_policy={"type": "support_balanced_cap", "max_windows_per_owner": 2},
                mechanism={
                    "sampling_unit": "window",
                    "clipping_unit": "window",
                    "noising_unit": "window",
                },
                accountant={"backend": "rdp", "unit": "window"},
                training={"owner_batch_size": 2},
            )
            route = compile_contract(contract)
            model = nn.Linear(data.train_x.shape[1], int(data.train_y.max() + 1))

            with self.assertRaisesRegex(ValueError, "owner-level privacy_unit"):
                train_owa_dpsgd(
                    model=model,
                    train_x=data.train_x.astype(np.float32),
                    train_y=data.train_y,
                    test_x=data.test_x.astype(np.float32),
                    test_y=data.test_y,
                    route=route,
                    config=OwaDpsgdConfig(
                        epochs=1,
                        owner_batch_size=2,
                        learning_rate=0.01,
                        target_epsilon=20.0,
                        target_delta=1e-5,
                        seed=3,
                    ),
                )


if __name__ == "__main__":
    unittest.main()
