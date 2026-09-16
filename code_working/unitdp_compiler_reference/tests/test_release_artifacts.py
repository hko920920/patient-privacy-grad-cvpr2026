import copy
import json
import tempfile
import unittest
from pathlib import Path

from unitdp.certificate import write_certificate
from unitdp.release_artifacts import (
    PUBLIC_CERTIFICATE_SCHEMA,
    build_private_execution_manifest,
    build_public_release_certificate,
    write_private_execution_manifest,
)


def fake_internal_certificate() -> dict[str, object]:
    checks = {
        "owner_mechanism_units_aligned": True,
        "selected_windows_single_attribution": True,
        "accountant_sample_rate_matches_owner_sampling": True,
        "bernoulli_owner_sampling_recorded": True,
        "one_clipped_vector_per_sampled_owner_recorded": True,
        "internal_owner_sampler_recorded": True,
        "preprocessing_contract_bound": True,
    }
    return {
        "privacy_unit": "owner",
        "route_status": "compiled",
        "contract_sha256": "c" * 64,
        "source_mapping_sha256": "a" * 64,
        "selected_mapping_sha256": "b" * 64,
        "source_mapping_stats": {"num_owners": 20},
        "selected_mapping_stats": {"num_owners": 20},
        "selection_diagnostics": {"source_label_counts": {"0": 10}},
        "mechanism_units": {
            "sampling_unit": "owner",
            "clipping_unit": "owner",
            "noising_unit": "owner",
            "accounting_unit": "owner",
        },
        "accountant": {
            "backend": "rdp",
            "epsilon": 8.0,
            "delta": 1e-5,
            "noise_multiplier": 1.25,
            "accountant_sample_rate": 0.1,
            "steps": 20,
        },
        "privacy_assumptions": {
            "adjacency": "replace_all_owner_contributions",
            "sampling_scheme": "independent_bernoulli_owner",
            "route_implementation_id": "unitdp.owa_dpsgd.train_owa_dpsgd:v1",
            "loader_binding": "internal_owner_sampler_no_external_dataloader",
            "sampler_trace_sha256": "d" * 64,
            "secure_rng": False,
            "rng_backend": "numpy_default_rng_and_torch_default_generator",
            "rng_security_mode": "research_noncryptographic",
            "rng_caveat": "research only",
            "preprocessing_binding": {
                "scope": "public_fixed",
                "artifact_id": "public_data_v1",
                "artifact_sha256": "e" * 64,
                "source_reference_sha256": "f" * 64,
                "schema": "unitdp.fixed_affine_preprocessor.v1",
                "input_dim": 24,
                "transform": "standard_scaler",
                "fit_on_protected_data": False,
            },
        },
        "implementation_checks": checks,
        "training_result": {
            "per_step_contribution_bound": "one_clipped_vector_per_sampled_owner",
            "sampler_trace_sha256": "d" * 64,
            "sampled_owner_count_mean": 2.3,
            "empty_steps": 1,
            "train_accuracy": 0.9,
            "selected_owners": 20,
            "selected_windows": 100,
            "config": {
                "seed": 13,
                "max_grad_norm": 1.0,
                "owner_batch_size": 2,
            },
        },
        "release_binding": {
            "model_artifact_sha256": "1" * 64,
            "training_code_sha256": "2" * 64,
            "config_sha256": "3" * 64,
            "release_binding_sha256": "4" * 64,
            "library_versions": {"numpy": "1.26.4"},
        },
    }


class ReleaseArtifactBoundaryTest(unittest.TestCase):
    def test_research_certificate_is_redacted_and_not_release_eligible(self):
        internal = fake_internal_certificate()
        public = build_public_release_certificate(internal)
        encoded = json.dumps(public, sort_keys=True)

        self.assertEqual(public["schema_version"], PUBLIC_CERTIFICATE_SCHEMA)
        self.assertEqual(public["release_status"], "research_non_release")
        self.assertIn(
            "noncryptographic_research_rng",
            public["privacy_claim"]["reason_codes"],
        )
        self.assertNotIn('"seed"', encoded)
        self.assertNotIn("d" * 64, encoded)
        self.assertNotIn("source_mapping", encoded)
        self.assertNotIn("selection_diagnostics", encoded)
        self.assertNotIn("train_accuracy", encoded)
        self.assertFalse(public["rng"]["random_coins_disclosed"])

    def test_secure_fixed_v2_id_is_still_closed_until_release_audit(self):
        internal = fake_internal_certificate()
        assumptions = internal["privacy_assumptions"]
        assert isinstance(assumptions, dict)
        assumptions["adjacency"] = "add_remove_one_owner"
        assumptions[
            "route_implementation_id"
        ] = "unitdp.owa_dpsgd.train_owner_poisson_fixed_v2"
        assumptions["secure_rng"] = True
        assumptions["rng_backend"] = "system_csprng"
        assumptions["rng_security_mode"] = "system_csprng"

        public = build_public_release_certificate(internal)

        self.assertEqual(public["release_status"], "research_non_release")
        self.assertIn(
            "implementation_not_release_approved",
            public["privacy_claim"]["reason_codes"],
        )

    def test_private_manifest_retains_execution_secrets(self):
        internal = fake_internal_certificate()
        private = build_private_execution_manifest(internal)
        full = private["full_execution_record"]

        self.assertEqual(private["visibility"], "private")
        self.assertEqual(full["training_result"]["config"]["seed"], 13)
        self.assertEqual(
            full["privacy_assumptions"]["sampler_trace_sha256"],
            "d" * 64,
        )

    def test_default_certificate_writer_is_public_and_private_writer_is_explicit(self):
        internal = fake_internal_certificate()
        with tempfile.TemporaryDirectory() as tmp:
            public_path = Path(tmp) / "certificate.json"
            private_path = Path(tmp) / "private_execution_manifest.json"
            write_certificate(internal, public_path)
            write_private_execution_manifest(internal, private_path)
            public = json.loads(public_path.read_text(encoding="utf-8"))
            private = json.loads(private_path.read_text(encoding="utf-8"))

        self.assertEqual(public["visibility"], "public")
        self.assertNotIn("training_result", public)
        self.assertEqual(private["visibility"], "private")
        self.assertIn("training_result", private["full_execution_record"])


if __name__ == "__main__":
    unittest.main()
