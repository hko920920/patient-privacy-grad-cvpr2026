"""Regression tests for the fail-closed privacy-claim certificate path."""

from __future__ import annotations

import copy
import csv
import importlib.metadata
import json
import subprocess
import sys
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from mapping_evidence import inspect_mapping_evidence  # noqa: E402
from privacy_claim_validator import (  # noqa: E402
    _opacus_rdp_epsilon,
    canonical_json_sha256,
    file_sha256,
    group_privacy_conversion,
    validate_privacy_claim,
)


RUNTIME_EXACT_FIELDS = (
    "mechanism_id",
    "mechanism_version",
    "accountant_id",
    "accountant_version",
    "accounting_unit",
    "accountant_adjacency",
    "sampling_unit",
    "sampler_law",
    "accountant_sampler_law",
    "sample_rate_numerator",
    "sample_rate_denominator",
    "sampling_implementation",
    "steps",
    "secure_mode",
    "secure_rng_backend",
    "noise_generation",
    "noise_hardening",
    "clipping_unit",
    "noising_unit",
    "privacy_convention",
    "gradient_aggregation",
    "update_normalization",
    "adapter_registry_entry_sha256",
    "code_artifact_id",
    "code_sha256",
    "accountant_checker_artifact_id",
    "accountant_checker_sha256",
    "selected_mapping_sha256",
    "pipeline_sha256",
    "delta_convention",
    "runtime_evidence_sha256",
)
RUNTIME_NUMERIC_FIELDS = (
    "sample_rate",
    "noise_multiplier",
    "clipping_norm",
    "optimizer_step_size",
    "epsilon",
    "delta",
)

REGISTRY_PATH = PROJECT_ROOT / "docs" / "mechanism_registry_v2_0.json"
REGISTERED_CODE_PATH = PROJECT_ROOT / "scripts" / "wisdm_v2_gold_pipeline.py"
MECHANISM_ID = "secure_systemrandom_dpsgd"
MECHANISM_VERSION = "2.0.0"
SAMPLING_IMPLEMENTATION = "systemrandom_randrange_bernoulli_v1"
SECURE_RNG_BACKEND = "python.secrets.SystemRandom"
NOISE_GENERATION = "normalvariate_discard1_sum4_div2_v1"
NOISE_HARDENING = "known_fp_reconstruction_mitigation_four_draw_v1"
GRADIENT_AGGREGATION = "clipped_sum_plus_gaussian_noise"
UPDATE_NORMALIZATION = "public_constant_step_no_dataset_denominator"
RAW_DOMAIN_ID = "fixed_owner_slot_payload_domain_v1"
RAW_EVENT_ADJACENCY = "fixed_owner_slot_payload_replace_one_v1"
OWNER_CAP_POLICY_ID = "public_owner_generated_record_cap_v1"
OWNER_CAP_ENFORCEMENT = "deterministic_owner_window_prefix_v1"


def registered_entry() -> dict:
    document = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return document["mechanisms"][f"{MECHANISM_ID}@{MECHANISM_VERSION}"]


@lru_cache(maxsize=32)
def recompute_opacus_epsilon(
    sample_rate: float = 0.1,
    steps: int = 10,
    noise_multiplier: float = 2.0,
    delta: float = 1e-6,
) -> float:
    _version, epsilon = _opacus_rdp_epsilon(
        sample_rate,
        steps,
        noise_multiplier,
        delta,
    )
    return epsilon


def write_mapping(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "window_id",
                "generated_unit",
                "owner_id",
                "owner_ids",
                "start",
                "end",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def default_rows() -> list[dict[str, str]]:
    return [
        {
            "scenario": "gold",
            "window_id": "w0",
            "generated_unit": "window",
            "owner_id": "u0",
            "owner_ids": "u0",
            "start": "0",
            "end": "2",
        },
        {
            "scenario": "gold",
            "window_id": "w1",
            "generated_unit": "window",
            "owner_id": "u0",
            "owner_ids": "u0",
            "start": "2",
            "end": "4",
        },
        {
            "scenario": "gold",
            "window_id": "w2",
            "generated_unit": "window",
            "owner_id": "u1",
            "owner_ids": "u1",
            "start": "0",
            "end": "2",
        },
    ]


def make_audit_row(mapping_path: Path, claim_unit: str) -> dict[str, str]:
    evidence = inspect_mapping_evidence(mapping_path, "gold")
    if not evidence.valid:
        raise AssertionError(evidence.issues)
    return {
        "scenario": "gold",
        "claimed_privacy_unit": claim_unit,
        "verdict": "diagnostic_only",
        "observed_alignment": "diagnostic_only",
        "num_windows_used": str(evidence.selected_record_count),
        "event_kappa_used": str(evidence.event_kappa),
        "owner_windows_max_used": str(evidence.owner_kappa),
        "mapping_file_sha256": evidence.mapping_file_sha256,
        "selected_mapping_sha256": evidence.selected_mapping_sha256,
    }


def rebind_report(report: dict, mapping_path: Path) -> dict:
    evidence = inspect_mapping_evidence(mapping_path, report["scenario"])
    if not evidence.valid:
        raise AssertionError(evidence.issues)
    mechanism = report["mechanism"]
    privacy = report["privacy"]
    manifest = report["pipeline"]["manifest"]
    manifest["mapping_file_sha256"] = evidence.mapping_file_sha256
    manifest["selected_mapping_sha256"] = evidence.selected_mapping_sha256
    manifest["selected_record_count"] = evidence.selected_record_count
    manifest["mechanism_sha256"] = canonical_json_sha256(mechanism)
    report["pipeline"]["mapping_file_sha256"] = evidence.mapping_file_sha256
    report["pipeline"]["selected_mapping_sha256"] = evidence.selected_mapping_sha256
    report["pipeline"]["pipeline_sha256"] = canonical_json_sha256(manifest)

    runtime = report["runtime_trace"]
    if runtime.get("status") == "matched":
        evidence_payload = runtime.get("evidence")
        if not isinstance(evidence_payload, dict):
            evidence_payload = {}
        evidence_payload.update(
            {
                "executed_updates": mechanism["steps"],
                "released_model_sha256": manifest["released_model_sha256"],
                "batch_trace_sha256": manifest["batch_trace_sha256"],
                "secure_sampling_rng": mechanism["sampling_implementation"],
                "secure_noise_rng": mechanism["noise_generation"],
            }
        )
        evidence_hash = canonical_json_sha256(evidence_payload)
        expected = {
            **{key: mechanism.get(key) for key in RUNTIME_EXACT_FIELDS},
            **{key: mechanism.get(key) for key in RUNTIME_NUMERIC_FIELDS},
            "code_artifact_id": manifest["code_artifact_id"],
            "code_sha256": manifest["code_sha256"],
            "accountant_checker_artifact_id": manifest[
                "accountant_checker_artifact_id"
            ],
            "accountant_checker_sha256": manifest[
                "accountant_checker_sha256"
            ],
            "selected_mapping_sha256": evidence.selected_mapping_sha256,
            "pipeline_sha256": report["pipeline"]["pipeline_sha256"],
            "epsilon": privacy["epsilon"],
            "delta": privacy["delta"],
            "delta_convention": privacy["delta_convention"],
            "runtime_evidence_sha256": evidence_hash,
        }
        payload = {
            key: expected.get(key)
            for key in (*RUNTIME_EXACT_FIELDS, *RUNTIME_NUMERIC_FIELDS)
        }
        runtime.clear()
        runtime.update(
            {
                "status": "matched",
                **payload,
                "evidence": evidence_payload,
            }
        )
        runtime["trace_hash"] = canonical_json_sha256(payload)
    return report


def make_report(
    mapping_path: Path,
    claim_unit: str,
    *,
    runtime_status: str = "matched",
    record_identity_policy: str = "public_fixed_ids",
) -> dict:
    evidence = inspect_mapping_evidence(mapping_path, "gold")
    if not evidence.valid:
        raise AssertionError(evidence.issues)
    accounting_unit = evidence.generated_unit
    accountant_adjacency = (
        "owner_add_remove" if accounting_unit == "owner" else "add_remove"
    )
    registry = registered_entry()
    mechanism = {
        "mechanism_id": MECHANISM_ID,
        "mechanism_version": MECHANISM_VERSION,
        "accountant_id": "opacus_rdp",
        "accountant_version": importlib.metadata.version("opacus"),
        "accounting_unit": accounting_unit,
        "accountant_adjacency": accountant_adjacency,
        "sampling_unit": accounting_unit,
        "sampler_law": "poisson",
        "accountant_sampler_law": "poisson",
        "sample_rate": 0.1,
        "sample_rate_numerator": 1,
        "sample_rate_denominator": 10,
        "sampling_implementation": SAMPLING_IMPLEMENTATION,
        "steps": 10,
        "noise_multiplier": 2.0,
        "secure_mode": True,
        "secure_rng_backend": SECURE_RNG_BACKEND,
        "noise_generation": NOISE_GENERATION,
        "noise_hardening": NOISE_HARDENING,
        "clipping_unit": accounting_unit,
        "clipping_norm": 1.0,
        "noising_unit": accounting_unit,
        "privacy_convention": "rdp_converted",
        "gradient_aggregation": GRADIENT_AGGREGATION,
        "update_normalization": UPDATE_NORMALIZATION,
        "optimizer_step_size": 0.01,
        "adapter_registry_entry_sha256": canonical_json_sha256(registry),
    }
    privacy = {
        "epsilon": recompute_opacus_epsilon(),
        "delta": 1e-6,
        "delta_convention": "accountant_delta_only",
    }
    raw_domain_parameters = {
        "canonicalization": "valid_rows_only_before_private_domain_v1",
        "fixed_presence": True,
        "immutable_fields": ["owner_id", "event_slot"],
        "mutable_fields": ["payload"],
    }
    raw_domain = {
        "id": RAW_DOMAIN_ID,
        "classification": "public_fixed",
        "parameters": raw_domain_parameters,
        "parameters_sha256": canonical_json_sha256(raw_domain_parameters),
    }
    owner_cap = int(evidence.owner_kappa or 1)
    contribution_parameters = {
        "cap": owner_cap,
        "enforcement": OWNER_CAP_ENFORCEMENT,
    }
    manifest = {
        "dataset_id": "public-gold",
        "split_id": "public-fixed-split",
        "code_version": "test-code-1",
        "code_artifact_id": registry["code_artifact_id"],
        "code_sha256": registry["code_sha256"],
        "accountant_checker_artifact_id": registry[
            "accountant_checker_artifact_id"
        ],
        "accountant_checker_sha256": registry[
            "accountant_checker_sha256"
        ],
        "support_convention": "half_open_integer_intervals",
        "record_identity_policy": record_identity_policy,
        "generated_record_unit": accounting_unit,
        "cross_owner_dependency": "none",
        "mapping_file_sha256": evidence.mapping_file_sha256,
        "selected_mapping_sha256": evidence.selected_mapping_sha256,
        "selected_record_count": evidence.selected_record_count,
        "mechanism_sha256": canonical_json_sha256(mechanism),
        "influence_support": {
            "status": "complete",
            "components": [
                "features",
                "labels",
                "weights",
                "selection",
                "preprocessing",
            ],
        },
        "owner_attribution_status": "complete",
        "preprocessing": [
            {
                "id": "identity",
                "classification": "record_local",
                "parameters": {"operation": "identity"},
                "parameters_sha256": canonical_json_sha256(
                    {"operation": "identity"}
                ),
            }
        ],
        "raw_domain": raw_domain,
        "contribution_policy": {
            "id": OWNER_CAP_POLICY_ID,
            "classification": "public_fixed",
            "parameters": contribution_parameters,
            "parameters_sha256": canonical_json_sha256(
                contribution_parameters
            ),
        },
        "schedule": {
            "id": "selected_mapping_once",
            "mode": "fixed_mapping",
            "evidence_status": "validated",
            "parameters": {
                "mode": "fixed_mapping",
                "selected_record_count": evidence.selected_record_count,
            },
            "parameters_sha256": canonical_json_sha256(
                {
                    "mode": "fixed_mapping",
                    "selected_record_count": evidence.selected_record_count,
                }
            ),
        },
        "population_definition": "three public generated windows",
        "population_size_policy": "public_fixed",
        "model_selection_status": "public",
        "released_model_sha256": "1" * 64,
        "batch_trace_sha256": "2" * 64,
    }
    if claim_unit == accounting_unit:
        contract = {
            "claimed_unit": claim_unit,
            "raw_adjacency": accountant_adjacency,
            "stability_status": "validated",
            "stability_method": "identity_generated_unit",
            "stability_bound": 1,
            "support_status": "complete",
            "schedule_mode": "fixed_mapping",
            "schedule_evidence_status": "validated",
            "conversion_method": "identity",
        }
    elif claim_unit == "event" and accounting_unit == "window":
        contract = {
            "claimed_unit": "event",
            "raw_adjacency": RAW_EVENT_ADJACENCY,
            "raw_domain_id": RAW_DOMAIN_ID,
            "raw_domain_sha256": canonical_json_sha256(raw_domain),
            "stability_status": "validated",
            "stability_method": "fixed_influence_replacements",
            "stability_bound": 2 * int(evidence.event_kappa or 0),
            "support_status": "complete",
            "schedule_mode": "fixed_mapping",
            "schedule_evidence_status": "validated",
            "conversion_method": "builtin_group",
        }
    elif claim_unit == "owner" and accounting_unit == "window":
        contract = {
            "claimed_unit": "owner",
            "raw_adjacency": "owner_add_remove",
            "stability_status": "validated",
            "stability_method": "owner_partition_add_remove",
            "stability_bound": owner_cap,
            "support_status": "complete",
            "schedule_mode": "fixed_mapping",
            "schedule_evidence_status": "validated",
            "conversion_method": (
                "identity" if evidence.owner_kappa == 1 else "builtin_group"
            ),
        }
    else:
        raise ValueError(claim_unit)

    report = {
        "schema_version": "2.0",
        "scenario": "gold",
        "mechanism": mechanism,
        "privacy": privacy,
        "pipeline": {
            "mapping_file_sha256": evidence.mapping_file_sha256,
            "selected_mapping_sha256": evidence.selected_mapping_sha256,
            "pipeline_sha256": canonical_json_sha256(manifest),
            "manifest": manifest,
        },
        "runtime_trace": {"status": runtime_status},
        "rng_assurance": "known_attack_hardened_secure",
        "noise_seed_policy": "secure_private_internal",
        "public_certificate": {"metadata_classification": "public_benchmark"},
        "claim_contracts": {claim_unit: contract},
    }
    return rebind_report(report, mapping_path)


class PrivacyClaimValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.mapping = self.root / "mapping.csv"
        write_mapping(self.mapping, default_rows())

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def validate(self, report: dict, claim_unit: str, audit: dict | None = None):
        return validate_privacy_claim(
            audit or make_audit_row(self.mapping, claim_unit),
            report,
            claim_unit,
            mapping_path=self.mapping,
        )

    def assert_blocked_without_wording(self, result) -> None:
        self.assertNotEqual(result.release_status, "ALLOWED")
        self.assertEqual(result.unit_path, "UNDERSPECIFIED")
        self.assertEqual(result.supported_statement, "")
        self.assertEqual(result.as_public_record()["supported_statement"], "")

    def test_valid_window_direct(self) -> None:
        result = self.validate(make_report(self.mapping, "window"), "window")
        self.assertEqual(result.release_status, "ALLOWED")
        self.assertEqual(result.unit_path, "DIRECT")
        self.assertEqual(result.assurance_status, "EVIDENCE_VALIDATED")
        self.assertTrue(result.supported_statement)

    def test_secure_systemrandom_registered_mechanism_direct(self) -> None:
        report = make_report(self.mapping, "window")
        result = self.validate(report, "window")
        self.assertEqual(result.release_status, "ALLOWED")
        self.assertEqual(result.unit_path, "DIRECT")

    def test_accountant_covered_population_size_is_not_accepted(self) -> None:
        report = make_report(self.mapping, "window")
        report["pipeline"]["manifest"][
            "population_size_policy"
        ] = "accountant_covered"
        rebind_report(report, self.mapping)
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn("POPULATION_POLICY_UNVERIFIED", result.issue_codes())

    def test_valid_event_conversion_recomputed(self) -> None:
        result = self.validate(make_report(self.mapping, "event"), "event")
        base_epsilon = make_report(self.mapping, "event")["privacy"]["epsilon"]
        self.assertEqual(result.release_status, "ALLOWED")
        self.assertEqual(result.unit_path, "CONVERT")
        self.assertEqual(result.stability_bound, 2)
        self.assertAlmostEqual(result.reported_epsilon, 2 * base_epsilon)
        self.assertAlmostEqual(
            result.reported_delta,
            1e-6 * (1 + pow(2.718281828459045, base_epsilon)),
        )

    def test_generic_replace_one_event_domain_is_insufficient(self) -> None:
        report = make_report(self.mapping, "event")
        report["claim_contracts"]["event"]["raw_adjacency"] = "replace_one"
        result = self.validate(report, "event")
        self.assert_blocked_without_wording(result)
        self.assertIn("RAW_ADJACENCY_NOT_SUPPORTED", result.issue_codes())

    def test_native_event_direct_exact_metric(self) -> None:
        rows = default_rows()
        for index, row in enumerate(rows):
            row["generated_unit"] = "event"
            row["start"] = str(index)
            row["end"] = str(index + 1)
        write_mapping(self.mapping, rows)
        result = self.validate(make_report(self.mapping, "event"), "event")
        self.assertEqual(result.release_status, "ALLOWED")
        self.assertEqual(result.unit_path, "DIRECT")
        self.assertEqual(result.stability_bound, 1)

    def test_native_owner_direct(self) -> None:
        rows = [
            {
                "scenario": "gold",
                "window_id": "owner-u0",
                "generated_unit": "owner",
                "owner_id": "u0",
                "owner_ids": "u0",
                "start": "0",
                "end": "4",
            },
            {
                "scenario": "gold",
                "window_id": "owner-u1",
                "generated_unit": "owner",
                "owner_id": "u1",
                "owner_ids": "u1",
                "start": "0",
                "end": "2",
            },
        ]
        write_mapping(self.mapping, rows)
        report = make_report(
            self.mapping,
            "owner",
            record_identity_policy="owner_partition_ids",
        )
        result = self.validate(report, "owner")
        self.assertEqual(result.release_status, "ALLOWED")
        self.assertEqual(result.unit_path, "DIRECT")
        self.assertEqual(result.accounting_unit, "owner")

    def test_owner_to_window_k_one_direct(self) -> None:
        rows = default_rows()[1:]
        rows[0]["owner_id"] = "u0"
        rows[0]["owner_ids"] = "u0"
        rows[1]["owner_id"] = "u1"
        rows[1]["owner_ids"] = "u1"
        write_mapping(self.mapping, rows)
        report = make_report(
            self.mapping,
            "owner",
            record_identity_policy="owner_partition_ids",
        )
        result = self.validate(report, "owner")
        self.assertEqual(result.release_status, "ALLOWED")
        self.assertEqual(result.unit_path, "DIRECT")
        self.assertEqual(result.stability_bound, 1)

    def test_owner_group_conversion_nonvacuous(self) -> None:
        report = make_report(
            self.mapping,
            "owner",
            record_identity_policy="owner_partition_ids",
        )
        result = self.validate(report, "owner")
        self.assertEqual(result.release_status, "ALLOWED")
        self.assertEqual(result.unit_path, "GROUP")
        self.assertEqual(result.stability_bound, 2)
        self.assertTrue(result.supported_statement)

    def test_owner_observed_kappa_without_public_cap_blocks(self) -> None:
        report = make_report(
            self.mapping,
            "owner",
            record_identity_policy="owner_partition_ids",
        )
        policy = report["pipeline"]["manifest"]["contribution_policy"]
        policy["id"] = "no_additional_cap"
        policy["classification"] = "not_applicable"
        policy["parameters"] = {"cap": None}
        policy["parameters_sha256"] = canonical_json_sha256(
            policy["parameters"]
        )
        rebind_report(report, self.mapping)
        result = self.validate(report, "owner")
        self.assert_blocked_without_wording(result)
        self.assertIn("OWNER_PUBLIC_CAP_UNVERIFIED", result.issue_codes())

    def test_owner_k_comes_from_public_cap_not_observed_maximum(self) -> None:
        report = make_report(
            self.mapping,
            "owner",
            record_identity_policy="owner_partition_ids",
        )
        policy = report["pipeline"]["manifest"]["contribution_policy"]
        policy["parameters"]["cap"] = 3
        policy["parameters_sha256"] = canonical_json_sha256(
            policy["parameters"]
        )
        rebind_report(report, self.mapping)
        result = self.validate(report, "owner")
        self.assert_blocked_without_wording(result)
        self.assertIn("STABILITY_BOUND_MISMATCH", result.issue_codes())

    def test_missing_actual_mapping_blocks(self) -> None:
        report = make_report(self.mapping, "window")
        result = validate_privacy_claim(
            make_audit_row(self.mapping, "window"),
            report,
            "window",
            mapping_path=None,
        )
        self.assert_blocked_without_wording(result)
        self.assertIn("MAPPING_EVIDENCE_FILE_REQUIRED", result.issue_codes())

    def test_mapping_bytes_mutation_blocks(self) -> None:
        report = make_report(self.mapping, "window")
        audit = make_audit_row(self.mapping, "window")
        rows = default_rows()
        rows[0]["end"] = "3"
        write_mapping(self.mapping, rows)
        result = self.validate(report, "window", audit)
        self.assert_blocked_without_wording(result)
        self.assertIn("ACTUAL_MAPPING_DIGEST_MISMATCH", result.issue_codes())

    def test_audit_kappa_tamper_blocks(self) -> None:
        report = make_report(self.mapping, "event")
        audit = make_audit_row(self.mapping, "event")
        audit["event_kappa_used"] = "9"
        result = self.validate(report, "event", audit)
        self.assert_blocked_without_wording(result)
        self.assertIn("AUDIT_EVENT_KAPPA_MISMATCH", result.issue_codes())

    def test_sampler_accountant_mismatch_blocks_when_rebound(self) -> None:
        report = make_report(self.mapping, "window")
        report["mechanism"]["sampler_law"] = "shuffle"
        rebind_report(report, self.mapping)
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn("SAMPLER_ACCOUNTANT_MISMATCH", result.issue_codes())

    def test_runtime_noise_mismatch_blocks(self) -> None:
        report = make_report(self.mapping, "window")
        report["runtime_trace"]["noise_multiplier"] = 9.0
        payload = {
            key: report["runtime_trace"].get(key)
            for key in (*RUNTIME_EXACT_FIELDS, *RUNTIME_NUMERIC_FIELDS)
        }
        report["runtime_trace"]["trace_hash"] = canonical_json_sha256(payload)
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn("RUNTIME_NUMERIC_FIELD_MISMATCH", result.issue_codes())

    def test_runtime_trace_digest_mismatch_blocks(self) -> None:
        report = make_report(self.mapping, "window")
        report["runtime_trace"]["trace_hash"] = "0" * 64
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn("RUNTIME_TRACE_DIGEST_MISMATCH", result.issue_codes())

    def test_self_consistent_forged_pipeline_code_hash_blocks(self) -> None:
        report = make_report(self.mapping, "window")
        report["pipeline"]["manifest"]["code_sha256"] = "0" * 64
        rebind_report(report, self.mapping)
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn(
            "PIPELINE_CODE_REGISTRY_DIGEST_MISMATCH",
            result.issue_codes(),
        )

    def test_fake_rng_semantics_block_even_when_rebound(self) -> None:
        report = make_report(self.mapping, "window")
        report["mechanism"]["secure_rng_backend"] = "attacker.fake_rng"
        report["mechanism"]["noise_generation"] = "one_draw_unhardened"
        rebind_report(report, self.mapping)
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn(
            "REGISTERED_MECHANISM_SEMANTICS_MISMATCH",
            result.issue_codes(),
        )

    def test_data_dependent_update_normalization_blocks(self) -> None:
        report = make_report(self.mapping, "window")
        report["mechanism"][
            "update_normalization"
        ] = "divide_by_sample_rate_times_private_dataset_size"
        rebind_report(report, self.mapping)
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn(
            "REGISTERED_MECHANISM_SEMANTICS_MISMATCH",
            result.issue_codes(),
        )

    def test_exact_sample_rate_mismatch_blocks(self) -> None:
        report = make_report(self.mapping, "window")
        report["mechanism"]["sample_rate_numerator"] = 2
        rebind_report(report, self.mapping)
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn("EXACT_SAMPLE_RATE_MISMATCH", result.issue_codes())

    def test_self_consistent_runtime_artifact_forgery_blocks(self) -> None:
        report = make_report(self.mapping, "window")
        runtime = report["runtime_trace"]
        runtime["evidence"]["released_model_sha256"] = "3" * 64
        runtime["runtime_evidence_sha256"] = canonical_json_sha256(
            runtime["evidence"]
        )
        payload = {
            key: runtime.get(key)
            for key in (*RUNTIME_EXACT_FIELDS, *RUNTIME_NUMERIC_FIELDS)
        }
        runtime["trace_hash"] = canonical_json_sha256(payload)
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn(
            "RUNTIME_ARTIFACT_DIGEST_MISMATCH",
            result.issue_codes(),
        )

    def test_zero_noise_blocks(self) -> None:
        report = make_report(self.mapping, "window")
        report["mechanism"]["noise_multiplier"] = 0.0
        rebind_report(report, self.mapping)
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn("NOISE_MULTIPLIER_INVALID", result.issue_codes())

    def test_invalid_delta_blocks(self) -> None:
        report = make_report(self.mapping, "window")
        report["privacy"]["delta"] = 1.0
        rebind_report(report, self.mapping)
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn("DELTA_INVALID", result.issue_codes())

    def test_nonfinite_epsilon_blocks(self) -> None:
        report = make_report(self.mapping, "window")
        report["privacy"]["epsilon"] = float("nan")
        rebind_report(report, self.mapping)
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn("EPSILON_INVALID", result.issue_codes())

    def test_schema_type_attack_blocks(self) -> None:
        report = make_report(self.mapping, "window")
        report["mechanism"]["steps"] = True
        rebind_report(report, self.mapping)
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn("REPORT_SCHEMA_INVALID", result.issue_codes())

    def test_private_global_preprocessing_blocks(self) -> None:
        report = make_report(self.mapping, "window")
        report["pipeline"]["manifest"]["preprocessing"][0][
            "classification"
        ] = "private_global"
        rebind_report(report, self.mapping)
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn("PRIVATE_PREPROCESSING_UNACCOUNTED", result.issue_codes())

    def test_preprocessing_parameter_digest_tamper_blocks(self) -> None:
        report = make_report(self.mapping, "window")
        report["pipeline"]["manifest"]["preprocessing"][0]["parameters"][
            "operation"
        ] = "private-fit"
        rebind_report(report, self.mapping)
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn(
            "PREPROCESSING_PARAMETERS_DIGEST_MISMATCH", result.issue_codes()
        )

    def test_separately_dp_without_registered_composition_blocks(self) -> None:
        report = make_report(self.mapping, "window")
        report["pipeline"]["manifest"]["preprocessing"][0][
            "classification"
        ] = "separately_dp"
        rebind_report(report, self.mapping)
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertEqual(result.assurance_status, "EXTERNAL_UNVERIFIED")
        self.assertIn(
            "SEPARATELY_DP_PREPROCESSING_UNVERIFIED", result.issue_codes()
        )

    def test_incomplete_influence_support_blocks(self) -> None:
        report = make_report(self.mapping, "event")
        report["pipeline"]["manifest"]["influence_support"]["components"].remove(
            "labels"
        )
        rebind_report(report, self.mapping)
        result = self.validate(report, "event")
        self.assert_blocked_without_wording(result)
        self.assertIn(
            "INFLUENCE_SUPPORT_MANIFEST_INCOMPLETE", result.issue_codes()
        )

    def test_unregistered_schedule_blocks(self) -> None:
        report = make_report(self.mapping, "event")
        report["claim_contracts"]["event"]["schedule_mode"] = "union"
        result = self.validate(report, "event")
        self.assert_blocked_without_wording(result)
        self.assertIn("SCHEDULE_MODE_UNREGISTERED", result.issue_codes())

    def test_private_adaptive_schedule_blocks(self) -> None:
        report = make_report(self.mapping, "event")
        report["claim_contracts"]["event"]["schedule_mode"] = "adaptive_private"
        report["claim_contracts"]["event"][
            "schedule_evidence_status"
        ] = "unverified"
        result = self.validate(report, "event")
        self.assert_blocked_without_wording(result)
        self.assertIn("SCHEDULE_MODE_UNREGISTERED", result.issue_codes())

    def test_user_stochastic_bound_label_blocks(self) -> None:
        report = make_report(self.mapping, "event")
        report["claim_contracts"]["event"]["schedule_mode"] = "stochastic_bound"
        result = self.validate(report, "event")
        self.assert_blocked_without_wording(result)
        self.assertIn("SCHEDULE_MODE_UNREGISTERED", result.issue_codes())

    def test_observed_kappa_one_without_validated_stability_blocks(self) -> None:
        report = make_report(self.mapping, "event")
        report["claim_contracts"]["event"]["stability_status"] = "unverified"
        result = self.validate(report, "event")
        self.assert_blocked_without_wording(result)
        self.assertIn("STABILITY_STATUS_UNVERIFIED", result.issue_codes())

    def test_external_conversion_attachment_cannot_authorize(self) -> None:
        report = make_report(self.mapping, "event")
        report["claim_contracts"]["event"][
            "conversion_method"
        ] = "external_json_claim"
        result = self.validate(report, "event")
        self.assert_blocked_without_wording(result)
        self.assertIn("EXTERNAL_CONVERSION_UNVERIFIED", result.issue_codes())

    def test_public_deterministic_noise_blocks(self) -> None:
        report = make_report(self.mapping, "window")
        report["noise_seed_policy"] = "public_deterministic"
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn("PUBLIC_DETERMINISTIC_NOISE", result.issue_codes())

    def test_unknown_accountant_checker_blocks(self) -> None:
        report = make_report(self.mapping, "window")
        report["mechanism"]["accountant_id"] = "plausible_external_rdp"
        rebind_report(report, self.mapping)
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn("ACCOUNTANT_CHECKER_UNREGISTERED", result.issue_codes())

    def test_accountant_version_drift_blocks(self) -> None:
        report = make_report(self.mapping, "window")
        report["mechanism"]["accountant_version"] = "0.0-unverified"
        rebind_report(report, self.mapping)
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn("ACCOUNTANT_VERSION_MISMATCH", result.issue_codes())

    def test_registered_mechanism_version_drift_blocks(self) -> None:
        report = make_report(self.mapping, "window")
        report["mechanism"]["mechanism_version"] = "wrong"
        rebind_report(report, self.mapping)
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn("MECHANISM_CHECKER_UNREGISTERED", result.issue_codes())

    def test_accountant_epsilon_tamper_blocks(self) -> None:
        report = make_report(self.mapping, "window")
        report["privacy"]["epsilon"] += 0.01
        rebind_report(report, self.mapping)
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn("ACCOUNTANT_EPSILON_MISMATCH", result.issue_codes())

    def test_research_prng_is_not_release_allowed(self) -> None:
        report = make_report(self.mapping, "window")
        report["mechanism"]["secure_mode"] = False
        report["rng_assurance"] = "research_prng"
        rebind_report(report, self.mapping)
        result = self.validate(report, "window")
        self.assertEqual(result.release_status, "RESEARCH_ONLY")
        self.assertEqual(result.unit_path, "DIRECT")
        self.assertEqual(result.supported_statement, "")

    def test_secure_assurance_with_secure_mode_false_blocks(self) -> None:
        report = make_report(self.mapping, "window")
        report["mechanism"]["secure_mode"] = False
        rebind_report(report, self.mapping)
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn("RELEASE_RNG_SECURE_MODE_MISMATCH", result.issue_codes())

    def test_universal_release_grade_rng_wording_is_rejected(self) -> None:
        report = make_report(self.mapping, "window")
        report["rng_assurance"] = "release_grade_secure"
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn("RNG_ASSURANCE_OVERCLAIM", result.issue_codes())

    def test_private_population_policy_blocks(self) -> None:
        report = make_report(self.mapping, "window")
        report["pipeline"]["manifest"][
            "population_size_policy"
        ] = "private_dynamic"
        rebind_report(report, self.mapping)
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn("POPULATION_POLICY_UNVERIFIED", result.issue_codes())

    def test_uncovered_private_model_selection_blocks(self) -> None:
        report = make_report(self.mapping, "window")
        report["pipeline"]["manifest"][
            "model_selection_status"
        ] = "not_covered"
        rebind_report(report, self.mapping)
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn("MODEL_SELECTION_NOT_COVERED", result.issue_codes())

    def test_private_metadata_is_redacted_not_leaked(self) -> None:
        report = make_report(self.mapping, "window")
        report["public_certificate"][
            "metadata_classification"
        ] = "private_internal"
        result = self.validate(report, "window")
        self.assertEqual(result.release_status, "ALLOWED")
        public = result.as_public_record()
        self.assertTrue(public["metadata_redacted"])
        self.assertNotIn("scenario", public)
        self.assertNotIn("selected_mapping_sha256", public)
        self.assertNotIn("observed_event_kappa", public)

    def test_missing_runtime_blocks_post_execution_wording(self) -> None:
        report = make_report(
            self.mapping,
            "window",
            runtime_status="missing",
        )
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn("RUNTIME_TRACE_NOT_BOUND", result.issue_codes())

    def test_vacuous_group_conversion_blocks(self) -> None:
        rows = []
        for index in range(10):
            rows.append(
                {
                    "scenario": "gold",
                    "window_id": f"overlap-{index}",
                    "generated_unit": "window",
                    "owner_id": "u0",
                    "owner_ids": "u0",
                    "start": "0",
                    "end": "2",
                }
            )
        write_mapping(self.mapping, rows)
        report = make_report(self.mapping, "event")
        report["privacy"]["delta"] = 0.1
        report["privacy"]["epsilon"] = recompute_opacus_epsilon(delta=0.1)
        rebind_report(report, self.mapping)
        result = self.validate(report, "event")
        self.assertEqual(result.release_status, "BLOCKED_VACUOUS")
        self.assertEqual(result.supported_statement, "")

    def test_owner_group_conversion_vacuous(self) -> None:
        rows = []
        for index in range(10):
            rows.append(
                {
                    "scenario": "gold",
                    "window_id": f"owner-window-{index}",
                    "generated_unit": "window",
                    "owner_id": "u0",
                    "owner_ids": "u0",
                    "start": str(2 * index),
                    "end": str(2 * index + 2),
                }
            )
        write_mapping(self.mapping, rows)
        report = make_report(
            self.mapping,
            "owner",
            record_identity_policy="owner_partition_ids",
        )
        report["privacy"]["delta"] = 0.1
        report["privacy"]["epsilon"] = recompute_opacus_epsilon(delta=0.1)
        rebind_report(report, self.mapping)
        result = self.validate(report, "owner")
        self.assertEqual(result.release_status, "BLOCKED_VACUOUS")
        self.assertEqual(result.unit_path, "GROUP")
        self.assertEqual(result.supported_statement, "")

    def test_owner_partition_rejects_multiowner_records(self) -> None:
        rows = default_rows()
        rows[0]["owner_ids"] = "u0;u2"
        write_mapping(self.mapping, rows)
        report = make_report(
            self.mapping,
            "owner",
            record_identity_policy="owner_partition_ids",
        )
        result = self.validate(report, "owner")
        self.assert_blocked_without_wording(result)
        self.assertIn(
            "OWNER_PARTITION_ATTRIBUTION_ARITY_INVALID", result.issue_codes()
        )

    def test_generated_unit_manifest_mismatch_blocks(self) -> None:
        report = make_report(self.mapping, "window")
        report["pipeline"]["manifest"]["generated_record_unit"] = "owner"
        rebind_report(report, self.mapping)
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn("ACTUAL_GENERATED_UNIT_MISMATCH", result.issue_codes())

    def test_mapping_without_generated_unit_column_blocks(self) -> None:
        report = make_report(self.mapping, "window")
        audit = make_audit_row(self.mapping, "window")
        self.mapping.write_text(
            "\n".join(
                [
                    "scenario,window_id,owner_id,owner_ids,start,end",
                    "gold,w0,u0,u0,0,2",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        result = validate_privacy_claim(
            audit,
            report,
            "window",
            mapping_path=self.mapping,
        )
        self.assert_blocked_without_wording(result)
        self.assertIn(
            "MAPPING_REQUIRED_COLUMNS_MISSING", result.issue_codes()
        )

    def test_legacy_report_blocks_early(self) -> None:
        result = self.validate({"schema_version": "1.2"}, "window")
        self.assert_blocked_without_wording(result)
        self.assertEqual(result.issue_codes(), ["UNSUPPORTED_SCHEMA_VERSION"])

    def test_group_conversion_known_values(self) -> None:
        epsilon_k, delta_k, log10_delta, vacuous = group_privacy_conversion(
            0.1, 1e-6, 2
        )
        self.assertAlmostEqual(epsilon_k, 0.2)
        self.assertAlmostEqual(delta_k, 1e-6 * (1 + pow(2.718281828459045, 0.1)))
        self.assertAlmostEqual(log10_delta, -5.676713, places=5)
        self.assertFalse(vacuous)

    def test_builder_cli_valid_and_blocked_outputs(self) -> None:
        report = make_report(self.mapping, "window")
        report_path = self.root / "report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        audit_path = self.root / "audit.csv"
        audit = make_audit_row(self.mapping, "window")
        with audit_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(audit))
            writer.writeheader()
            writer.writerow(audit)
        output_dir = self.root / "valid-output"
        command = [
            sys.executable,
            str(SCRIPTS / "build_unit_certificate.py"),
            "--audit-csv",
            str(audit_path),
            "--mapping-csv",
            str(self.mapping),
            "--privacy-report-json",
            str(report_path),
            "--scenario",
            "gold",
            "--claim-unit",
            "window",
            "--output-dir",
            str(output_dir),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        public = json.loads(
            (output_dir / "certificate_gold_window.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(public["release_status"], "ALLOWED")
        self.assertTrue(public["supported_statement"])

        report["mechanism"]["accountant_sampler_law"] = "shuffle"
        rebind_report(report, self.mapping)
        report_path.write_text(json.dumps(report), encoding="utf-8")
        blocked_dir = self.root / "blocked-output"
        command[-1] = str(blocked_dir)
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 2, completed.stdout)
        public = json.loads(
            (blocked_dir / "certificate_gold_window.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertNotEqual(public["release_status"], "ALLOWED")
        self.assertEqual(public["supported_statement"], "")


if __name__ == "__main__":
    unittest.main()
