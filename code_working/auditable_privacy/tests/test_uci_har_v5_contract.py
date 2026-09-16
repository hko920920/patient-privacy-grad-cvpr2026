"""Route-specific regression tests for the frozen Audit v5 UCI HAR evidence."""

from __future__ import annotations

import copy
import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from privacy_claim_validator import (  # noqa: E402
    canonical_json_sha256,
    validate_privacy_claim,
)


RUN_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "uci_har_v5_multirun_001"
    / "confirmatory_run01"
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


def rebind_self_consistent_report(report: dict) -> None:
    """Rebind internal digests without repairing a registry/premise mutation."""

    mechanism = report["mechanism"]
    privacy = report["privacy"]
    pipeline = report["pipeline"]
    manifest = pipeline["manifest"]
    manifest["mechanism_sha256"] = canonical_json_sha256(mechanism)
    pipeline["pipeline_sha256"] = canonical_json_sha256(manifest)

    runtime = report["runtime_trace"]
    evidence = runtime["evidence"]
    evidence_hash = canonical_json_sha256(evidence)
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
        "selected_mapping_sha256": pipeline["selected_mapping_sha256"],
        "pipeline_sha256": pipeline["pipeline_sha256"],
        "epsilon": privacy["epsilon"],
        "delta": privacy["delta"],
        "delta_convention": privacy["delta_convention"],
        "runtime_evidence_sha256": evidence_hash,
    }
    for key in (*RUNTIME_EXACT_FIELDS, *RUNTIME_NUMERIC_FIELDS):
        runtime[key] = expected.get(key)
    payload = {
        key: runtime.get(key)
        for key in (*RUNTIME_EXACT_FIELDS, *RUNTIME_NUMERIC_FIELDS)
    }
    runtime["trace_hash"] = canonical_json_sha256(payload)


class UciHarV5ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base_report = json.loads(
            (RUN_ROOT / "privacy_report.json").read_text(encoding="utf-8")
        )
        with (RUN_ROOT / "audit" / "unit_audit.csv").open(
            newline="",
            encoding="utf-8",
        ) as handle:
            rows = list(csv.DictReader(handle))
        cls.audit_rows = {
            "window": rows[0],
            "event": rows[1],
            "owner": rows[2],
        }
        cls.mapping = RUN_ROOT / "mapping.csv"

    def validate(
        self,
        report: dict,
        claim: str,
        *,
        mapping_path: Path | None = None,
    ):
        return validate_privacy_claim(
            self.audit_rows[claim],
            report,
            claim,
            mapping_path=mapping_path or self.mapping,
        )

    def assert_blocked_without_wording(self, result) -> None:
        self.assertNotEqual(result.release_status, "ALLOWED")
        self.assertEqual(result.unit_path, "UNDERSPECIFIED")
        self.assertEqual(result.supported_statement, "")
        self.assertEqual(result.as_public_record()["supported_statement"], "")

    def test_frozen_route_authorizes_only_generated_window(self) -> None:
        window = self.validate(copy.deepcopy(self.base_report), "window")
        event = self.validate(copy.deepcopy(self.base_report), "event")
        owner = self.validate(copy.deepcopy(self.base_report), "owner")

        self.assertEqual(
            (
                window.release_status,
                window.unit_path,
                window.stability_bound,
                bool(window.supported_statement),
            ),
            ("ALLOWED", "DIRECT", 1, True),
        )
        self.assert_blocked_without_wording(event)
        self.assert_blocked_without_wording(owner)
        self.assertEqual(event.release_status, "BLOCKED_UNVERIFIED")
        self.assertEqual(owner.release_status, "BLOCKED_UNVERIFIED")

    def test_self_consistent_pipeline_source_forgery_blocks(self) -> None:
        report = copy.deepcopy(self.base_report)
        report["pipeline"]["manifest"]["code_sha256"] = "0" * 64
        rebind_self_consistent_report(report)
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn(
            "PIPELINE_CODE_REGISTRY_DIGEST_MISMATCH",
            result.issue_codes(),
        )

    def test_self_consistent_mechanism_semantics_forgery_blocks(self) -> None:
        report = copy.deepcopy(self.base_report)
        report["mechanism"][
            "update_normalization"
        ] = "divide_by_private_dataset_size"
        rebind_self_consistent_report(report)
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn(
            "REGISTERED_MECHANISM_SEMANTICS_MISMATCH",
            result.issue_codes(),
        )

    def test_self_consistent_exact_rate_forgery_blocks(self) -> None:
        report = copy.deepcopy(self.base_report)
        report["mechanism"]["sample_rate_numerator"] = 2
        rebind_self_consistent_report(report)
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn("EXACT_SAMPLE_RATE_MISMATCH", result.issue_codes())

    def test_missing_runtime_blocks_concrete_run_wording(self) -> None:
        report = copy.deepcopy(self.base_report)
        report["runtime_trace"] = {"status": "missing"}
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn("RUNTIME_TRACE_NOT_BOUND", result.issue_codes())

    def test_self_consistent_runtime_model_forgery_blocks(self) -> None:
        report = copy.deepcopy(self.base_report)
        report["runtime_trace"]["evidence"][
            "released_model_sha256"
        ] = "3" * 64
        rebind_self_consistent_report(report)
        result = self.validate(report, "window")
        self.assert_blocked_without_wording(result)
        self.assertIn(
            "RUNTIME_ARTIFACT_DIGEST_MISMATCH",
            result.issue_codes(),
        )

    def test_actual_mapping_mutation_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            changed_mapping = Path(temporary) / "mapping.csv"
            shutil.copyfile(self.mapping, changed_mapping)
            text = changed_mapping.read_text(encoding="utf-8")
            changed_mapping.write_text(
                text.replace(
                    "train_row_00000",
                    "train_row_99999",
                    1,
                ),
                encoding="utf-8",
            )
            result = self.validate(
                copy.deepcopy(self.base_report),
                "window",
                mapping_path=changed_mapping,
            )
        self.assert_blocked_without_wording(result)
        self.assertIn(
            "ACTUAL_MAPPING_DIGEST_MISMATCH",
            result.issue_codes(),
        )

    def test_observed_event_or_owner_k_cannot_be_forged_positive(self) -> None:
        event_report = copy.deepcopy(self.base_report)
        event_contract = event_report["claim_contracts"]["event"]
        event_contract["stability_status"] = "validated"
        event_contract["support_status"] = "complete"
        event_contract["stability_method"] = "fixed_influence_replacements"
        event = self.validate(event_report, "event")

        owner_report = copy.deepcopy(self.base_report)
        owner_contract = owner_report["claim_contracts"]["owner"]
        owner_contract["stability_status"] = "validated"
        owner = self.validate(owner_report, "owner")

        self.assert_blocked_without_wording(event)
        self.assert_blocked_without_wording(owner)


if __name__ == "__main__":
    unittest.main()
