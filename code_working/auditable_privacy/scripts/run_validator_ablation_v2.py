"""Run the frozen v2 validator baseline/ablation comparison.

The experiment uses manually labeled positive and adversarial case families.
It compares field presence, JSON Schema, multiplicity, hash, accountant, and
two partial-validator ablations with the authoritative full validator.
Ground-truth tuples are declared by the case definitions below rather than
copied from the full validator's output.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from jsonschema import Draft202012Validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from mapping_evidence import inspect_mapping_evidence  # noqa: E402
from privacy_claim_validator import (  # noqa: E402
    canonical_json_sha256,
    validate_privacy_claim,
)
from tests.test_privacy_claim_validator import (  # noqa: E402
    RUNTIME_EXACT_FIELDS,
    RUNTIME_NUMERIC_FIELDS,
    default_rows,
    make_audit_row,
    make_report,
    rebind_report,
    recompute_opacus_epsilon,
    write_mapping,
)


METHODS = (
    "B0_FIELD_PRESENCE",
    "B1_JSON_SCHEMA",
    "B2_MULTIPLICITY_ONLY",
    "B3_HASH_ONLY",
    "B4_ACCOUNTANT_ONLY",
    "A1_FULL_MINUS_RUNTIME",
    "A2_FULL_MINUS_VACUITY",
    "FULL",
)
SIMPLE_BASELINES = METHODS[:5]


@dataclass
class Prediction:
    release_status: str
    unit_path: str
    epsilon: float | None = None
    delta: float | None = None
    assurance_status: str = "NOT_EVALUATED"
    issue_codes: tuple[str, ...] = ()


@dataclass
class Case:
    case_id: str
    description: str
    truth_row: str
    claim_unit: str
    failure_family: str
    expected_release_status: str
    expected_unit_path: str
    expected_assurance_status: str
    expected_issue_code: str
    expected_numeric_rule: str
    mapping_path: Path
    report_path: Path
    audit_path: Path
    expected_epsilon: float | None
    expected_delta: float | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_conversion(epsilon: float, delta: float, k: int) -> tuple[float, float]:
    return (
        k * epsilon,
        delta * math.fsum(math.exp(index * epsilon) for index in range(k)),
    )


def expected_numbers(
    report: Mapping[str, object],
    rule: str,
    mapping_path: Path,
    claim_unit: str,
) -> tuple[float | None, float | None]:
    if rule == "none":
        return None, None
    privacy = report["privacy"]
    assert isinstance(privacy, dict)
    epsilon = float(privacy["epsilon"])
    delta = float(privacy["delta"])
    if rule == "base":
        return epsilon, delta
    evidence = inspect_mapping_evidence(mapping_path, "gold")
    if not evidence.valid:
        raise RuntimeError(evidence.issues)
    if claim_unit == "event":
        k = 2 * int(evidence.event_kappa or 0)
    elif claim_unit == "owner":
        k = int(evidence.owner_kappa or 0)
    else:
        raise ValueError(claim_unit)
    return group_conversion(epsilon, delta, k)


def runtime_payload(report: Mapping[str, object]) -> dict[str, object]:
    runtime = report["runtime_trace"]
    assert isinstance(runtime, dict)
    return {
        key: runtime.get(key)
        for key in (*RUNTIME_EXACT_FIELDS, *RUNTIME_NUMERIC_FIELDS)
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def make_case(
    root: Path,
    *,
    case_id: str,
    description: str,
    truth_row: str,
    claim_unit: str,
    failure_family: str,
    expected_release_status: str,
    expected_unit_path: str,
    expected_assurance_status: str,
    expected_issue_code: str = "",
    expected_numeric_rule: str = "none",
    rows: list[dict[str, str]] | None = None,
    record_identity_policy: str = "public_fixed_ids",
    mutate: Callable[
        [dict[str, object], dict[str, str], Path],
        None,
    ]
    | None = None,
) -> Case:
    case_dir = root / "case_inputs" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = case_dir / "mapping.csv"
    write_mapping(mapping_path, copy.deepcopy(rows or default_rows()))
    report = make_report(
        mapping_path,
        claim_unit,
        record_identity_policy=record_identity_policy,
    )
    audit = make_audit_row(mapping_path, claim_unit)
    if mutate is not None:
        mutate(report, audit, mapping_path)
    report_path = case_dir / "privacy_report.json"
    audit_path = case_dir / "audit_row.json"
    write_json(report_path, report)
    write_json(audit_path, audit)
    expected_epsilon, expected_delta = expected_numbers(
        report,
        expected_numeric_rule,
        mapping_path,
        claim_unit,
    )
    return Case(
        case_id=case_id,
        description=description,
        truth_row=truth_row,
        claim_unit=claim_unit,
        failure_family=failure_family,
        expected_release_status=expected_release_status,
        expected_unit_path=expected_unit_path,
        expected_assurance_status=expected_assurance_status,
        expected_issue_code=expected_issue_code,
        expected_numeric_rule=expected_numeric_rule,
        mapping_path=mapping_path,
        report_path=report_path,
        audit_path=audit_path,
        expected_epsilon=expected_epsilon,
        expected_delta=expected_delta,
    )


def build_cases(root: Path) -> list[Case]:
    def mapping_mutation(
        report: dict[str, object],
        audit: dict[str, str],
        mapping: Path,
    ) -> None:
        rows = default_rows()
        rows[0]["end"] = "3"
        write_mapping(mapping, rows)

    def schema_boolean_steps(
        report: dict[str, object],
        _audit: dict[str, str],
        mapping: Path,
    ) -> None:
        mechanism = report["mechanism"]
        assert isinstance(mechanism, dict)
        mechanism["steps"] = True
        rebind_report(report, mapping)

    def sampler_mismatch(
        report: dict[str, object],
        _audit: dict[str, str],
        mapping: Path,
    ) -> None:
        mechanism = report["mechanism"]
        assert isinstance(mechanism, dict)
        mechanism["sampler_law"] = "shuffle"
        rebind_report(report, mapping)

    def runtime_noise_mismatch(
        report: dict[str, object],
        _audit: dict[str, str],
        _mapping: Path,
    ) -> None:
        runtime = report["runtime_trace"]
        assert isinstance(runtime, dict)
        runtime["noise_multiplier"] = 9.0
        runtime["trace_hash"] = canonical_json_sha256(runtime_payload(report))

    def epsilon_tamper(
        report: dict[str, object],
        _audit: dict[str, str],
        mapping: Path,
    ) -> None:
        privacy = report["privacy"]
        assert isinstance(privacy, dict)
        privacy["epsilon"] = float(privacy["epsilon"]) + 0.01
        rebind_report(report, mapping)

    def secure_mode_false(
        report: dict[str, object],
        _audit: dict[str, str],
        mapping: Path,
    ) -> None:
        mechanism = report["mechanism"]
        assert isinstance(mechanism, dict)
        mechanism["secure_mode"] = False
        rebind_report(report, mapping)

    def research_rng(
        report: dict[str, object],
        _audit: dict[str, str],
        mapping: Path,
    ) -> None:
        mechanism = report["mechanism"]
        assert isinstance(mechanism, dict)
        mechanism["secure_mode"] = False
        report["rng_assurance"] = "research_prng"
        rebind_report(report, mapping)

    def secure_systemrandom_adapter(
        report: dict[str, object],
        _audit: dict[str, str],
        mapping: Path,
    ) -> None:
        mechanism = report["mechanism"]
        assert isinstance(mechanism, dict)
        mechanism["mechanism_id"] = "secure_systemrandom_dpsgd"
        mechanism["mechanism_version"] = "2.0.0"
        rebind_report(report, mapping)

    def missing_runtime(
        report: dict[str, object],
        _audit: dict[str, str],
        _mapping: Path,
    ) -> None:
        report["runtime_trace"] = {"status": "missing"}

    def private_preprocessing(
        report: dict[str, object],
        _audit: dict[str, str],
        mapping: Path,
    ) -> None:
        report["pipeline"]["manifest"]["preprocessing"][0][
            "classification"
        ] = "private_global"
        rebind_report(report, mapping)

    def incomplete_influence(
        report: dict[str, object],
        _audit: dict[str, str],
        mapping: Path,
    ) -> None:
        report["pipeline"]["manifest"]["influence_support"]["components"].remove(
            "labels"
        )
        rebind_report(report, mapping)

    def adaptive_schedule(
        report: dict[str, object],
        _audit: dict[str, str],
        _mapping: Path,
    ) -> None:
        contract = report["claim_contracts"]["event"]
        contract["schedule_mode"] = "adaptive_private"
        contract["schedule_evidence_status"] = "unverified"

    def external_conversion(
        report: dict[str, object],
        _audit: dict[str, str],
        _mapping: Path,
    ) -> None:
        report["claim_contracts"]["event"][
            "conversion_method"
        ] = "external_json_claim"

    def private_model_selection(
        report: dict[str, object],
        _audit: dict[str, str],
        mapping: Path,
    ) -> None:
        report["pipeline"]["manifest"][
            "model_selection_status"
        ] = "not_covered"
        rebind_report(report, mapping)

    def unknown_accountant(
        report: dict[str, object],
        _audit: dict[str, str],
        mapping: Path,
    ) -> None:
        report["mechanism"]["accountant_id"] = "plausible_external_rdp"
        rebind_report(report, mapping)

    def generated_unit_mismatch(
        report: dict[str, object],
        _audit: dict[str, str],
        mapping: Path,
    ) -> None:
        report["pipeline"]["manifest"]["generated_record_unit"] = "event"
        rebind_report(report, mapping)

    def unverified_stability(
        report: dict[str, object],
        _audit: dict[str, str],
        _mapping: Path,
    ) -> None:
        report["claim_contracts"]["event"]["stability_status"] = "unverified"

    def forged_code_hash(
        report: dict[str, object],
        _audit: dict[str, str],
        mapping: Path,
    ) -> None:
        report["pipeline"]["manifest"]["code_sha256"] = "0" * 64
        rebind_report(report, mapping)

    def fake_rng_backend(
        report: dict[str, object],
        _audit: dict[str, str],
        mapping: Path,
    ) -> None:
        report["mechanism"]["secure_rng_backend"] = "attacker.fake_rng"
        rebind_report(report, mapping)

    def one_draw_noise(
        report: dict[str, object],
        _audit: dict[str, str],
        mapping: Path,
    ) -> None:
        report["mechanism"]["noise_generation"] = "one_draw_unhardened"
        report["mechanism"]["noise_hardening"] = "none"
        rebind_report(report, mapping)

    def data_dependent_normalization(
        report: dict[str, object],
        _audit: dict[str, str],
        mapping: Path,
    ) -> None:
        report["mechanism"][
            "update_normalization"
        ] = "divide_by_sample_rate_times_private_dataset_size"
        rebind_report(report, mapping)

    def no_public_owner_cap(
        report: dict[str, object],
        _audit: dict[str, str],
        mapping: Path,
    ) -> None:
        policy = report["pipeline"]["manifest"]["contribution_policy"]
        policy["id"] = "no_additional_cap"
        policy["classification"] = "not_applicable"
        policy["parameters"] = {"cap": None}
        policy["parameters_sha256"] = canonical_json_sha256(
            policy["parameters"]
        )
        rebind_report(report, mapping)

    def generic_event_adjacency(
        report: dict[str, object],
        _audit: dict[str, str],
        _mapping: Path,
    ) -> None:
        report["claim_contracts"]["event"]["raw_adjacency"] = "replace_one"

    def exact_sample_rate_mismatch(
        report: dict[str, object],
        _audit: dict[str, str],
        mapping: Path,
    ) -> None:
        report["mechanism"]["sample_rate_numerator"] = 2
        rebind_report(report, mapping)

    def runtime_artifact_forgery(
        report: dict[str, object],
        _audit: dict[str, str],
        _mapping: Path,
    ) -> None:
        runtime = report["runtime_trace"]
        runtime["evidence"]["released_model_sha256"] = "3" * 64
        runtime["runtime_evidence_sha256"] = canonical_json_sha256(
            runtime["evidence"]
        )
        runtime["trace_hash"] = canonical_json_sha256(runtime_payload(report))

    def vacuous_owner(
        report: dict[str, object],
        _audit: dict[str, str],
        mapping: Path,
    ) -> None:
        privacy = report["privacy"]
        assert isinstance(privacy, dict)
        privacy["delta"] = 0.1
        privacy["epsilon"] = recompute_opacus_epsilon(delta=0.1)
        rebind_report(report, mapping)

    owner_rows = [
        {
            "scenario": "gold",
            "window_id": f"owner-window-{index}",
            "generated_unit": "window",
            "owner_id": "u0",
            "owner_ids": "u0",
            "start": str(2 * index),
            "end": str(2 * index + 2),
        }
        for index in range(10)
    ]
    native_event_rows = copy.deepcopy(default_rows())
    for index, row in enumerate(native_event_rows):
        row["generated_unit"] = "event"
        row["start"] = str(index)
        row["end"] = str(index + 1)
    native_owner_rows = [
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

    specs = [
        dict(
            case_id="P01_valid_window",
            description="Valid generated-window DIRECT claim.",
            claim_unit="window",
            failure_family="positive",
            expected_release_status="ALLOWED",
            expected_unit_path="DIRECT",
            expected_assurance_status="EVIDENCE_VALIDATED",
            expected_numeric_rule="base",
        ),
        dict(
            case_id="P02_valid_event_factor2",
            description=(
                "Fixed-owner/fixed-slot event payload replacement maps to "
                "generated add/remove distance K=2 and CONVERT."
            ),
            claim_unit="event",
            failure_family="positive",
            expected_release_status="ALLOWED",
            expected_unit_path="CONVERT",
            expected_assurance_status="EVIDENCE_VALIDATED",
            expected_numeric_rule="group",
        ),
        dict(
            case_id="P03_valid_owner_group",
            description="Valid nonvacuous owner group conversion with K=2.",
            claim_unit="owner",
            failure_family="positive",
            expected_release_status="ALLOWED",
            expected_unit_path="GROUP",
            expected_assurance_status="EVIDENCE_VALIDATED",
            expected_numeric_rule="group",
            record_identity_policy="owner_partition_ids",
        ),
        dict(
            case_id="P04_native_event_direct",
            description="Native event sampling/accounting uses the exact event metric.",
            claim_unit="event",
            failure_family="positive",
            expected_release_status="ALLOWED",
            expected_unit_path="DIRECT",
            expected_assurance_status="EVIDENCE_VALIDATED",
            expected_numeric_rule="base",
            rows=native_event_rows,
        ),
        dict(
            case_id="P05_native_owner_direct",
            description="Native owner sampling, clipping, noising, and accounting.",
            claim_unit="owner",
            failure_family="positive",
            expected_release_status="ALLOWED",
            expected_unit_path="DIRECT",
            expected_assurance_status="EVIDENCE_VALIDATED",
            expected_numeric_rule="base",
            rows=native_owner_rows,
            record_identity_policy="owner_partition_ids",
        ),
        dict(
            case_id="P06_secure_systemrandom_adapter",
            description="Registered known-attack-hardened SystemRandom mechanism adapter.",
            claim_unit="window",
            failure_family="positive",
            expected_release_status="ALLOWED",
            expected_unit_path="DIRECT",
            expected_assurance_status="EVIDENCE_VALIDATED",
            expected_numeric_rule="base",
            mutate=secure_systemrandom_adapter,
        ),
        dict(
            case_id="F00_missing_runtime_unverified",
            description="No runtime observation blocks post-execution wording.",
            claim_unit="window",
            failure_family="runtime",
            expected_release_status="BLOCKED_UNVERIFIED",
            expected_unit_path="UNDERSPECIFIED",
            expected_assurance_status="DIAGNOSTIC_ONLY",
            expected_issue_code="RUNTIME_TRACE_NOT_BOUND",
            mutate=missing_runtime,
        ),
        dict(
            case_id="F01_vacuous_owner_group",
            description="Owner K=10 conversion has delta at least one.",
            claim_unit="owner",
            failure_family="conversion_vacuity",
            expected_release_status="BLOCKED_VACUOUS",
            expected_unit_path="GROUP",
            expected_assurance_status="DIAGNOSTIC_ONLY",
            rows=owner_rows,
            record_identity_policy="owner_partition_ids",
            mutate=vacuous_owner,
        ),
        dict(
            case_id="F02_mapping_bytes_changed",
            description="Actual mapping bytes changed after report binding.",
            claim_unit="window",
            failure_family="mapping_binding",
            expected_release_status="BLOCKED_INVALID",
            expected_unit_path="UNDERSPECIFIED",
            expected_assurance_status="DIAGNOSTIC_ONLY",
            expected_issue_code="ACTUAL_MAPPING_DIGEST_MISMATCH",
            mutate=mapping_mutation,
        ),
        dict(
            case_id="F03_schema_boolean_steps",
            description="Boolean injected into integer steps field.",
            claim_unit="window",
            failure_family="schema",
            expected_release_status="BLOCKED_INVALID",
            expected_unit_path="UNDERSPECIFIED",
            expected_assurance_status="DIAGNOSTIC_ONLY",
            expected_issue_code="REPORT_SCHEMA_INVALID",
            mutate=schema_boolean_steps,
        ),
        dict(
            case_id="F04_sampler_accountant_mismatch",
            description="Shuffle execution paired with Poisson accountant.",
            claim_unit="window",
            failure_family="mechanism",
            expected_release_status="BLOCKED_INVALID",
            expected_unit_path="UNDERSPECIFIED",
            expected_assurance_status="DIAGNOSTIC_ONLY",
            expected_issue_code="SAMPLER_ACCOUNTANT_MISMATCH",
            mutate=sampler_mismatch,
        ),
        dict(
            case_id="F05_runtime_noise_mismatch",
            description="Runtime noise differs from mechanism report.",
            claim_unit="window",
            failure_family="runtime",
            expected_release_status="BLOCKED_INVALID",
            expected_unit_path="UNDERSPECIFIED",
            expected_assurance_status="DIAGNOSTIC_ONLY",
            expected_issue_code="RUNTIME_NUMERIC_FIELD_MISMATCH",
            mutate=runtime_noise_mismatch,
        ),
        dict(
            case_id="F06_accountant_epsilon_tamper",
            description="Reported epsilon differs from registered recomputation.",
            claim_unit="window",
            failure_family="accountant",
            expected_release_status="BLOCKED_INVALID",
            expected_unit_path="UNDERSPECIFIED",
            expected_assurance_status="DIAGNOSTIC_ONLY",
            expected_issue_code="ACCOUNTANT_EPSILON_MISMATCH",
            mutate=epsilon_tamper,
        ),
        dict(
            case_id="F07_secure_assurance_mode_false",
            description="Known-attack-hardened assurance declares secure mode false.",
            claim_unit="window",
            failure_family="rng",
            expected_release_status="BLOCKED_INVALID",
            expected_unit_path="UNDERSPECIFIED",
            expected_assurance_status="DIAGNOSTIC_ONLY",
            expected_issue_code="RELEASE_RNG_SECURE_MODE_MISMATCH",
            mutate=secure_mode_false,
        ),
        dict(
            case_id="F08_research_prng",
            description="Research PRNG cannot authorize concrete-run public wording.",
            claim_unit="window",
            failure_family="rng",
            expected_release_status="RESEARCH_ONLY",
            expected_unit_path="DIRECT",
            expected_assurance_status="DIAGNOSTIC_ONLY",
            expected_issue_code="RNG_RESEARCH_ONLY",
            mutate=research_rng,
        ),
        dict(
            case_id="F09_private_global_preprocessing",
            description="Private-global preprocessing is unaccounted.",
            claim_unit="window",
            failure_family="private_dependency",
            expected_release_status="BLOCKED_INVALID",
            expected_unit_path="UNDERSPECIFIED",
            expected_assurance_status="DIAGNOSTIC_ONLY",
            expected_issue_code="PRIVATE_PREPROCESSING_UNACCOUNTED",
            mutate=private_preprocessing,
        ),
        dict(
            case_id="F10_incomplete_influence_support",
            description="Label influence omitted from complete support.",
            claim_unit="event",
            failure_family="stability",
            expected_release_status="BLOCKED_UNVERIFIED",
            expected_unit_path="UNDERSPECIFIED",
            expected_assurance_status="DIAGNOSTIC_ONLY",
            expected_issue_code="INFLUENCE_SUPPORT_MANIFEST_INCOMPLETE",
            mutate=incomplete_influence,
        ),
        dict(
            case_id="F11_private_adaptive_schedule",
            description="Observed private-adaptive schedule has no registered bound.",
            claim_unit="event",
            failure_family="schedule",
            expected_release_status="BLOCKED_UNVERIFIED",
            expected_unit_path="UNDERSPECIFIED",
            expected_assurance_status="DIAGNOSTIC_ONLY",
            expected_issue_code="SCHEDULE_MODE_UNREGISTERED",
            mutate=adaptive_schedule,
        ),
        dict(
            case_id="F12_fabricated_external_conversion",
            description="Unregistered external JSON conversion attachment.",
            claim_unit="event",
            failure_family="external_attachment",
            expected_release_status="BLOCKED_UNVERIFIED",
            expected_unit_path="UNDERSPECIFIED",
            expected_assurance_status="EXTERNAL_UNVERIFIED",
            expected_issue_code="EXTERNAL_CONVERSION_UNVERIFIED",
            mutate=external_conversion,
        ),
        dict(
            case_id="F13_private_model_selection",
            description="Private model selection is not covered.",
            claim_unit="window",
            failure_family="private_dependency",
            expected_release_status="BLOCKED_INVALID",
            expected_unit_path="UNDERSPECIFIED",
            expected_assurance_status="DIAGNOSTIC_ONLY",
            expected_issue_code="MODEL_SELECTION_NOT_COVERED",
            mutate=private_model_selection,
        ),
        dict(
            case_id="F14_unknown_accountant",
            description="Plausible but unregistered accountant identifier.",
            claim_unit="window",
            failure_family="accountant",
            expected_release_status="BLOCKED_INVALID",
            expected_unit_path="UNDERSPECIFIED",
            expected_assurance_status="DIAGNOSTIC_ONLY",
            expected_issue_code="ACCOUNTANT_CHECKER_UNREGISTERED",
            mutate=unknown_accountant,
        ),
        dict(
            case_id="F15_generated_unit_mismatch",
            description="Manifest generated unit conflicts with actual mapping.",
            claim_unit="window",
            failure_family="unit_binding",
            expected_release_status="BLOCKED_INVALID",
            expected_unit_path="UNDERSPECIFIED",
            expected_assurance_status="DIAGNOSTIC_ONLY",
            expected_issue_code="ACTUAL_GENERATED_UNIT_MISMATCH",
            mutate=generated_unit_mismatch,
        ),
        dict(
            case_id="F16_unverified_stability",
            description="Observed kappa one lacks validated stability status.",
            claim_unit="event",
            failure_family="stability",
            expected_release_status="BLOCKED_UNVERIFIED",
            expected_unit_path="UNDERSPECIFIED",
            expected_assurance_status="DIAGNOSTIC_ONLY",
            expected_issue_code="STABILITY_STATUS_UNVERIFIED",
            mutate=unverified_stability,
        ),
        dict(
            case_id="F17_forged_code_hash",
            description="Self-consistent report rebinding cannot forge registered source code.",
            claim_unit="window",
            failure_family="executable_binding",
            expected_release_status="BLOCKED_INVALID",
            expected_unit_path="UNDERSPECIFIED",
            expected_assurance_status="DIAGNOSTIC_ONLY",
            expected_issue_code="PIPELINE_CODE_REGISTRY_DIGEST_MISMATCH",
            mutate=forged_code_hash,
        ),
        dict(
            case_id="F18_fake_rng_backend",
            description="Attacker-controlled RNG backend violates registered semantics.",
            claim_unit="window",
            failure_family="executable_binding",
            expected_release_status="BLOCKED_INVALID",
            expected_unit_path="UNDERSPECIFIED",
            expected_assurance_status="DIAGNOSTIC_ONLY",
            expected_issue_code="REGISTERED_MECHANISM_SEMANTICS_MISMATCH",
            mutate=fake_rng_backend,
        ),
        dict(
            case_id="F19_one_draw_noise",
            description="One-draw finite-precision Gaussian lacks registered hardening.",
            claim_unit="window",
            failure_family="rng",
            expected_release_status="BLOCKED_INVALID",
            expected_unit_path="UNDERSPECIFIED",
            expected_assurance_status="DIAGNOSTIC_ONLY",
            expected_issue_code="REGISTERED_MECHANISM_SEMANTICS_MISMATCH",
            mutate=one_draw_noise,
        ),
        dict(
            case_id="F20_data_dependent_normalization",
            description="Dataset-size normalization changes the registered release mechanism.",
            claim_unit="window",
            failure_family="mechanism",
            expected_release_status="BLOCKED_INVALID",
            expected_unit_path="UNDERSPECIFIED",
            expected_assurance_status="DIAGNOSTIC_ONLY",
            expected_issue_code="REGISTERED_MECHANISM_SEMANTICS_MISMATCH",
            mutate=data_dependent_normalization,
        ),
        dict(
            case_id="F21_owner_observed_k_no_cap",
            description="Observed owner incidence is not a global public contribution cap.",
            claim_unit="owner",
            failure_family="stability",
            expected_release_status="BLOCKED_UNVERIFIED",
            expected_unit_path="UNDERSPECIFIED",
            expected_assurance_status="DIAGNOSTIC_ONLY",
            expected_issue_code="OWNER_PUBLIC_CAP_UNVERIFIED",
            record_identity_policy="owner_partition_ids",
            mutate=no_public_owner_cap,
        ),
        dict(
            case_id="F22_generic_raw_event_adjacency",
            description="Generic replace-one omits fixed owner, slot, presence, and payload domain.",
            claim_unit="event",
            failure_family="stability",
            expected_release_status="BLOCKED_INVALID",
            expected_unit_path="UNDERSPECIFIED",
            expected_assurance_status="DIAGNOSTIC_ONLY",
            expected_issue_code="RAW_ADJACENCY_NOT_SUPPORTED",
            mutate=generic_event_adjacency,
        ),
        dict(
            case_id="F23_exact_sample_rate_mismatch",
            description="Floating accountant q differs from the executed exact rational q.",
            claim_unit="window",
            failure_family="mechanism",
            expected_release_status="BLOCKED_INVALID",
            expected_unit_path="UNDERSPECIFIED",
            expected_assurance_status="DIAGNOSTIC_ONLY",
            expected_issue_code="EXACT_SAMPLE_RATE_MISMATCH",
            mutate=exact_sample_rate_mismatch,
        ),
        dict(
            case_id="F24_runtime_artifact_forgery",
            description="Self-consistent runtime trace cannot swap the manifest-bound model.",
            claim_unit="window",
            failure_family="runtime",
            expected_release_status="BLOCKED_INVALID",
            expected_unit_path="UNDERSPECIFIED",
            expected_assurance_status="DIAGNOSTIC_ONLY",
            expected_issue_code="RUNTIME_ARTIFACT_DIGEST_MISMATCH",
            mutate=runtime_artifact_forgery,
        ),
    ]
    truth_rows = {
        "P01_valid_window": "T01",
        "P02_valid_event_factor2": "T07,T08",
        "P03_valid_owner_group": "T12",
        "P04_native_event_direct": "T06,T27",
        "P05_native_owner_direct": "T10,T27",
        "P06_secure_systemrandom_adapter": "T32",
        "F00_missing_runtime_unverified": "T02",
        "F01_vacuous_owner_group": "T13",
        "F02_mapping_bytes_changed": "T28",
        "F03_schema_boolean_steps": "T04",
        "F04_sampler_accountant_mismatch": "T03",
        "F05_runtime_noise_mismatch": "T30",
        "F06_accountant_epsilon_tamper": "T36",
        "F07_secure_assurance_mode_false": "T32",
        "F08_research_prng": "T32",
        "F09_private_global_preprocessing": "T16",
        "F10_incomplete_influence_support": "T29",
        "F11_private_adaptive_schedule": "T20,T22",
        "F12_fabricated_external_conversion": "T25,T26",
        "F13_private_model_selection": "T31",
        "F14_unknown_accountant": "T35",
        "F15_generated_unit_mismatch": "T37",
        "F16_unverified_stability": "T05",
        "F17_forged_code_hash": "R01",
        "F18_fake_rng_backend": "R02",
        "F19_one_draw_noise": "R03",
        "F20_data_dependent_normalization": "R04",
        "F21_owner_observed_k_no_cap": "R05",
        "F22_generic_raw_event_adjacency": "R06",
        "F23_exact_sample_rate_mismatch": "R07",
        "F24_runtime_artifact_forgery": "R08",
    }
    if set(truth_rows) != {str(spec["case_id"]) for spec in specs}:
        raise RuntimeError("Truth-table linkage is incomplete")
    return [
        make_case(root, truth_row=truth_rows[str(spec["case_id"])], **spec)
        for spec in specs
    ]


def base_numbers(report: Mapping[str, object]) -> tuple[float | None, float | None]:
    privacy = report.get("privacy")
    if not isinstance(privacy, dict):
        return None, None
    try:
        return float(privacy["epsilon"]), float(privacy["delta"])
    except (KeyError, TypeError, ValueError):
        return None, None


def block(issue: str) -> Prediction:
    return Prediction(
        release_status="BLOCKED",
        unit_path="UNDERSPECIFIED",
        issue_codes=(issue,),
    )


def field_presence_baseline(report: Mapping[str, object]) -> Prediction:
    required = {
        "schema_version",
        "scenario",
        "mechanism",
        "privacy",
        "pipeline",
        "claim_contracts",
    }
    if not required.issubset(report):
        return block("MISSING_TOP_LEVEL_FIELD")
    epsilon, delta = base_numbers(report)
    return Prediction("ALLOWED", "DIRECT", epsilon, delta)


def schema_baseline(
    report: Mapping[str, object],
    schema_validator: Draft202012Validator,
) -> Prediction:
    if list(schema_validator.iter_errors(report)):
        return block("SCHEMA_REJECTED")
    epsilon, delta = base_numbers(report)
    return Prediction("ALLOWED", "DIRECT", epsilon, delta)


def multiplicity_baseline(
    report: Mapping[str, object],
    mapping_path: Path,
    claim_unit: str,
) -> Prediction:
    evidence = inspect_mapping_evidence(mapping_path, str(report.get("scenario", "")))
    if not evidence.valid:
        return block("MAPPING_INVALID")
    epsilon, delta = base_numbers(report)
    if epsilon is None or delta is None:
        return block("PRIVACY_NUMBERS_MISSING")
    if claim_unit == "window":
        return Prediction("ALLOWED", "DIRECT", epsilon, delta)
    if claim_unit == "event":
        k = int(evidence.event_kappa or 0)
        path = "DIRECT" if k <= 1 else "CONVERT"
    else:
        k = int(evidence.owner_kappa or 0)
        path = "DIRECT" if k <= 1 else "GROUP"
    if k <= 1:
        return Prediction("ALLOWED", path, epsilon, delta)
    converted_epsilon, converted_delta = group_conversion(epsilon, delta, k)
    return Prediction(
        "ALLOWED",
        path,
        converted_epsilon,
        converted_delta,
    )


def hash_baseline(
    report: Mapping[str, object],
    mapping_path: Path,
) -> Prediction:
    evidence = inspect_mapping_evidence(mapping_path, str(report.get("scenario", "")))
    pipeline = report.get("pipeline")
    if not evidence.valid or not isinstance(pipeline, dict):
        return block("HASH_INPUT_INVALID")
    if (
        pipeline.get("mapping_file_sha256") != evidence.mapping_file_sha256
        or pipeline.get("selected_mapping_sha256")
        != evidence.selected_mapping_sha256
    ):
        return block("HASH_MISMATCH")
    epsilon, delta = base_numbers(report)
    return Prediction("ALLOWED", "DIRECT", epsilon, delta)


def accountant_baseline(report: Mapping[str, object]) -> Prediction:
    mechanism = report.get("mechanism")
    privacy = report.get("privacy")
    if not isinstance(mechanism, dict) or not isinstance(privacy, dict):
        return block("ACCOUNTANT_INPUT_MISSING")
    if (
        mechanism.get("accountant_id") != "opacus_rdp"
        or str(mechanism.get("accountant_version")) != "1.6.0"
    ):
        return block("ACCOUNTANT_UNREGISTERED")
    try:
        sample_rate = float(mechanism["sample_rate"])
        steps = int(mechanism["steps"])
        noise = float(mechanism["noise_multiplier"])
        delta = float(privacy["delta"])
        reported = float(privacy["epsilon"])
        if (
            isinstance(mechanism["steps"], bool)
            or steps <= 0
            or noise <= 0
            or not 0 < sample_rate <= 1
            or not 0 < delta < 1
        ):
            return block("ACCOUNTANT_DOMAIN_INVALID")
        recomputed = recompute_opacus_epsilon(
            sample_rate=sample_rate,
            steps=steps,
            noise_multiplier=noise,
            delta=delta,
        )
    except (KeyError, TypeError, ValueError):
        return block("ACCOUNTANT_INPUT_INVALID")
    if not math.isclose(recomputed, reported, rel_tol=1e-10, abs_tol=1e-12):
        return block("ACCOUNTANT_EPSILON_MISMATCH")
    return Prediction("ALLOWED", "DIRECT", reported, delta)


def full_prediction(
    report: Mapping[str, object],
    audit: Mapping[str, str],
    mapping_path: Path,
    claim_unit: str,
) -> Prediction:
    result = validate_privacy_claim(
        audit,
        report,
        claim_unit,
        mapping_path=mapping_path,
    )
    return Prediction(
        release_status=result.release_status,
        unit_path=result.unit_path,
        epsilon=result.reported_epsilon,
        delta=result.reported_delta,
        assurance_status=result.assurance_status,
        issue_codes=tuple(result.issue_codes()),
    )


def predict(
    method: str,
    report: Mapping[str, object],
    audit: Mapping[str, str],
    mapping_path: Path,
    claim_unit: str,
    schema_validator: Draft202012Validator,
) -> Prediction:
    if method == "B0_FIELD_PRESENCE":
        return field_presence_baseline(report)
    if method == "B1_JSON_SCHEMA":
        return schema_baseline(report, schema_validator)
    if method == "B2_MULTIPLICITY_ONLY":
        return multiplicity_baseline(report, mapping_path, claim_unit)
    if method == "B3_HASH_ONLY":
        return hash_baseline(report, mapping_path)
    if method == "B4_ACCOUNTANT_ONLY":
        return accountant_baseline(report)
    if method == "A1_FULL_MINUS_RUNTIME":
        without_runtime = copy.deepcopy(report)
        manifest = report["pipeline"]["manifest"]
        reference = make_report(
            mapping_path,
            claim_unit,
            record_identity_policy=str(
                manifest.get("record_identity_policy")
                or "public_fixed_ids"
            ),
        )
        without_runtime["runtime_trace"] = reference["runtime_trace"]
        return full_prediction(without_runtime, audit, mapping_path, claim_unit)
    full = full_prediction(report, audit, mapping_path, claim_unit)
    if method == "A2_FULL_MINUS_VACUITY":
        if full.release_status == "BLOCKED_VACUOUS":
            full.release_status = "ALLOWED"
            full.assurance_status = "EVIDENCE_VALIDATED"
        return full
    if method == "FULL":
        return full
    raise ValueError(method)


def numbers_match(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)


def exact_tuple(case: Case, prediction: Prediction) -> bool:
    if (
        prediction.release_status != case.expected_release_status
        or prediction.unit_path != case.expected_unit_path
        or prediction.assurance_status != case.expected_assurance_status
    ):
        return False
    if case.expected_release_status == "ALLOWED":
        return numbers_match(prediction.epsilon, case.expected_epsilon) and numbers_match(
            prediction.delta,
            case.expected_delta,
        )
    return True


def build_prediction_rows(
    cases: Sequence[Case],
    schema_validator: Draft202012Validator,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in cases:
        report = json.loads(case.report_path.read_text(encoding="utf-8"))
        audit = json.loads(case.audit_path.read_text(encoding="utf-8"))
        base_epsilon, base_delta = base_numbers(report)
        for method in METHODS:
            started = time.perf_counter()
            prediction = predict(
                method,
                report,
                audit,
                case.mapping_path,
                case.claim_unit,
                schema_validator,
            )
            wall_ms = (time.perf_counter() - started) * 1000
            unsafe = (
                case.expected_release_status != "ALLOWED"
                and prediction.release_status == "ALLOWED"
            )
            binary_safe_correct = (
                (case.expected_release_status == "ALLOWED")
                == (prediction.release_status == "ALLOWED")
            )
            valid_decision_exact = (
                case.expected_release_status == "ALLOWED"
                and prediction.release_status == "ALLOWED"
                and prediction.unit_path == case.expected_unit_path
                and numbers_match(prediction.epsilon, case.expected_epsilon)
                and numbers_match(prediction.delta, case.expected_delta)
            )
            wrong_same_number = (
                case.expected_release_status == "ALLOWED"
                and case.expected_unit_path in {"CONVERT", "GROUP"}
                and prediction.release_status == "ALLOWED"
                and numbers_match(prediction.epsilon, base_epsilon)
                and numbers_match(prediction.delta, base_delta)
            )
            rows.append(
                {
                    "case_id": case.case_id,
                    "failure_family": case.failure_family,
                    "expected_release_status": case.expected_release_status,
                    "expected_unit_path": case.expected_unit_path,
                    "expected_assurance_status": case.expected_assurance_status,
                    "method": method,
                    "predicted_release_status": prediction.release_status,
                    "predicted_unit_path": prediction.unit_path,
                    "predicted_assurance_status": prediction.assurance_status,
                    "predicted_epsilon": (
                        "" if prediction.epsilon is None else repr(prediction.epsilon)
                    ),
                    "predicted_delta": (
                        "" if prediction.delta is None else repr(prediction.delta)
                    ),
                    "issue_codes": ";".join(prediction.issue_codes),
                    "supported_positive": prediction.release_status == "ALLOWED",
                    "unsafe_positive": unsafe,
                    "binary_safe_correct": binary_safe_correct,
                    "valid_decision_exact": valid_decision_exact,
                    "exact_tuple": exact_tuple(case, prediction),
                    "wrong_same_number": wrong_same_number,
                    "valid_retained": (
                        case.expected_release_status == "ALLOWED"
                        and prediction.release_status == "ALLOWED"
                    ),
                    "wall_ms": f"{wall_ms:.6f}",
                }
            )
    return rows


def aggregate(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        blocked_truth = [
            row for row in selected if row["expected_release_status"] != "ALLOWED"
        ]
        valid_truth = [
            row for row in selected if row["expected_release_status"] == "ALLOWED"
        ]
        unsafe = sum(bool(row["unsafe_positive"]) for row in blocked_truth)
        binary_safe = sum(bool(row["binary_safe_correct"]) for row in selected)
        valid_exact = sum(bool(row["valid_decision_exact"]) for row in valid_truth)
        exact = sum(bool(row["exact_tuple"]) for row in selected)
        retained = sum(bool(row["valid_retained"]) for row in valid_truth)
        same_number = sum(bool(row["wrong_same_number"]) for row in selected)
        wall_values = sorted(float(row["wall_ms"]) for row in selected)
        median = wall_values[len(wall_values) // 2]
        output.append(
            {
                "method": method,
                "cases": len(selected),
                "blocked_truth_cases": len(blocked_truth),
                "unsafe_positives": unsafe,
                "unsafe_positive_rate": unsafe / len(blocked_truth),
                "binary_safe_correct": binary_safe,
                "binary_safe_accuracy": binary_safe / len(selected),
                "valid_exact_decisions": valid_exact,
                "valid_exact_decision_rate": valid_exact / len(valid_truth),
                "exact_tuples": exact,
                "exact_tuple_accuracy": exact / len(selected),
                "valid_truth_cases": len(valid_truth),
                "valid_retained": retained,
                "valid_retention_rate": retained / len(valid_truth),
                "wrong_same_number": same_number,
                "median_wall_ms": median,
                "max_wall_ms": max(wall_values),
            }
        )
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="reports/validator_ablation_v2_001",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases(output_dir)
    schema_path = PROJECT_ROOT / "docs" / "privacy_report_schema_v2_0.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    schema_validator = Draft202012Validator(schema)
    prediction_rows = build_prediction_rows(cases, schema_validator)
    aggregate_rows = aggregate(prediction_rows)

    full_rows = [row for row in prediction_rows if row["method"] == "FULL"]
    if not all(bool(row["exact_tuple"]) for row in full_rows):
        failures = [
            str(row["case_id"]) for row in full_rows if not bool(row["exact_tuple"])
        ]
        raise RuntimeError(f"FULL failed frozen expected tuples: {failures}")
    if any(bool(row["unsafe_positive"]) for row in full_rows):
        raise RuntimeError("FULL emitted an unsafe positive")
    for case, row in zip(cases, full_rows, strict=True):
        if case.expected_issue_code and case.expected_issue_code not in str(
            row["issue_codes"]
        ).split(";"):
            raise RuntimeError(
                f"FULL did not localize {case.case_id} as "
                f"{case.expected_issue_code}"
            )
    for method in SIMPLE_BASELINES:
        row = next(item for item in aggregate_rows if item["method"] == method)
        if int(row["unsafe_positives"]) == 0:
            raise RuntimeError(f"{method} has no demonstrated unsafe positive")

    cases_json: list[dict[str, object]] = []
    for case in cases:
        record = asdict(case)
        for key in ("mapping_path", "report_path", "audit_path"):
            path = Path(record[key])
            record[key] = path.relative_to(PROJECT_ROOT).as_posix()
            record[f"{key}_sha256"] = sha256_file(path)
        cases_json.append(record)
    cases_path = output_dir / "cases.json"
    predictions_path = output_dir / "predictions.csv"
    aggregate_path = output_dir / "aggregate_metrics.csv"
    write_json(
        cases_path,
        {
            "schema_version": "validator_ablation_cases_v2.0",
            "ground_truth_source": (
                "manually frozen contract/truth-table expectations in "
                "scripts/run_validator_ablation_v2.py"
            ),
            "cases": cases_json,
        },
    )
    write_csv(predictions_path, prediction_rows)
    write_csv(aggregate_path, aggregate_rows)

    family_rows: list[dict[str, object]] = []
    families = sorted({str(row["failure_family"]) for row in prediction_rows})
    for method in METHODS:
        for family in families:
            selected = [
                row
                for row in prediction_rows
                if row["method"] == method and row["failure_family"] == family
            ]
            family_rows.append(
                {
                    "method": method,
                    "failure_family": family,
                    "cases": len(selected),
                    "unsafe_positives": sum(
                        bool(row["unsafe_positive"]) for row in selected
                    ),
                    "exact_tuples": sum(bool(row["exact_tuple"]) for row in selected),
                }
            )
    family_path = output_dir / "confusion_by_failure_family.csv"
    write_csv(family_path, family_rows)

    valid_case_count = sum(
        case.expected_release_status == "ALLOWED" for case in cases
    )
    lines = [
        "# Validator Baseline/Ablation v2",
        "",
        (
            f"Cases: {len(cases)} ({valid_case_count} valid, "
            f"{len(cases) - valid_case_count} blocked/research-only)"
        ),
        "",
        "| method | unsafe positives | binary safe accuracy | valid exact decision | full tuple accuracy | wrong same-number |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {method} | {unsafe_positives}/{blocked_truth_cases} | "
            "{binary_safe_accuracy:.3f} | {valid_exact_decision_rate:.3f} | "
            "{exact_tuple_accuracy:.3f} | {wrong_same_number} |".format(**row)
        )
    lines.extend(
        [
            "",
            "An unsafe positive means the method emitted ALLOWED where the frozen",
            "contract requires blocked, vacuous, or research-only output. Binary",
            "safe accuracy checks only allow versus non-allow. Valid exact decision",
            "checks path and epsilon/delta on valid cases while ignoring",
            "assurance metadata. Full tuple accuracy additionally checks assurance",
            "status and blocked/research subtypes. FULL passed every tuple with zero",
            "unsafe positives.",
        ]
    )
    summary_path = output_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    source_paths = [
        Path(__file__).resolve(),
        PROJECT_ROOT / "tests" / "test_privacy_claim_validator.py",
        PROJECT_ROOT / "scripts" / "privacy_claim_validator.py",
        PROJECT_ROOT / "scripts" / "mapping_evidence.py",
        PROJECT_ROOT / "docs" / "privacy_claim_contract_v2_0.md",
        PROJECT_ROOT / "docs" / "privacy_claim_truth_table_v2_0.md",
        schema_path,
        cases_path,
        predictions_path,
        aggregate_path,
        family_path,
        summary_path,
    ]
    artifact_index = {
        "schema_version": "validator_ablation_artifact_v2.0",
        "files": [
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in source_paths
        ],
    }
    write_json(output_dir / "artifact_index.json", artifact_index)
    print(f"Wrote {summary_path}")
    print(f"FULL exact tuples: {len(full_rows)}/{len(full_rows)}")


if __name__ == "__main__":
    main()
