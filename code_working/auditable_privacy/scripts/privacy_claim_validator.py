"""Fail-closed privacy-claim validator for the v2.0 audit contract.

This module is the only authority for copy-ready privacy wording. Mapping
audits may compute observed incidence statistics, but those statistics cannot
authorize a claim without a validated transformation/accountant contract.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from mapping_evidence import MappingEvidence, inspect_mapping_evidence


SCHEMA_VERSION = "2.0"

OWNER_ALIASES = {"owner", "user", "patient", "series", "subject", "client"}
EVENT_ALIASES = {"event", "timestamp", "raw_event"}
WINDOW_ALIASES = {"window", "example", "generated_window", "generated_example"}

VALID_ADJACENCIES = {
    "add_remove",
    "replace_one",
    "fixed_owner_slot_payload_replace_one_v1",
    "owner_add_remove",
    "owner_replace_one",
}
VALID_SAMPLERS = {
    "poisson",
    "shuffle",
    "fixed_without_replacement",
    "full_batch",
    "custom_registered",
}
VALID_PRIVACY_CONVENTIONS = {"approx_dp", "rdp_converted", "pld_converted"}
VALID_DELTA_CONVENTIONS = {
    "accountant_delta_only",
    "delta_augmented_with_bound_failure",
    "conditional_on_bound_event",
}
VALID_PREPROCESSING = {
    "public_fixed",
    "record_local",
    "owner_local",
    "separately_dp",
    "private_global",
    "private_adaptive",
}
VALID_RNG_ASSURANCE = {
    # Retained as a syntactically recognized negative/legacy input so the
    # validator can return a precise overclaim issue rather than a parse error.
    "release_grade_secure",
    "known_attack_hardened_secure",
    "research_prng",
    "unknown",
}
VALID_NOISE_SEED_POLICIES = {
    "secure_private_unrecorded",
    "secure_private_internal",
    "public_deterministic",
    "unknown",
}
VALID_METADATA_CLASSIFICATION = {
    "public_benchmark",
    "controller_approved",
    "dp_released",
    "private_internal",
}
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
MECHANISM_REGISTRY_FILE = "mechanism_registry_v2_0.json"
REGISTERED_EVENT_RAW_DOMAIN_ID = "fixed_owner_slot_payload_domain_v1"
REGISTERED_EVENT_RAW_ADJACENCY = "fixed_owner_slot_payload_replace_one_v1"
REGISTERED_OWNER_CAP_POLICY_ID = "public_owner_generated_record_cap_v1"
REGISTERED_OWNER_CAP_ENFORCEMENT = "deterministic_owner_window_prefix_v1"


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    category: str
    field: str
    message: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "code": self.code,
            "category": self.category,
            "field": self.field,
            "message": self.message,
        }


@dataclass
class ClaimValidationResult:
    scenario: str
    claimed_privacy_unit: str
    normalized_claimed_unit: str
    accounting_unit: str = ""
    raw_adjacency: str = ""
    accountant_adjacency: str = ""
    unit_path: str = "UNDERSPECIFIED"
    release_status: str = "BLOCKED_UNVERIFIED"
    assurance_status: str = "DIAGNOSTIC_ONLY"
    stability_method: str = ""
    stability_bound: Optional[int] = None
    base_epsilon: Optional[float] = None
    base_delta: Optional[float] = None
    reported_epsilon: Optional[float] = None
    reported_delta: Optional[float] = None
    reported_delta_log10: Optional[float] = None
    conversion_vacuous: Optional[bool] = None
    mapping_file_sha256: str = ""
    selected_mapping_sha256: str = ""
    pipeline_sha256: str = ""
    runtime_status: str = ""
    rng_assurance: str = ""
    noise_seed_policy: str = ""
    metadata_classification: str = ""
    observed_event_kappa: Optional[int] = None
    observed_owner_kappa: Optional[int] = None
    selected_record_count: Optional[int] = None
    attribution_arity_max: Optional[int] = None
    generated_record_unit: str = ""
    supported_statement: str = ""
    assurance_boundary: str = (
        "Conditional on the mapping export being complete and faithful to the "
        "bound pipeline; the validator checks registered semantics and "
        "cross-artifact consistency, not malicious or omitted dependencies."
    )
    do_not_say: str = ""
    next_step: str = ""
    issues: List[ValidationIssue] = field(default_factory=list)

    def issue_codes(self) -> List[str]:
        return [issue.code for issue in self.issues]

    def as_internal_record(self) -> Dict[str, str]:
        return {
            "schema_version": SCHEMA_VERSION,
            "scenario": self.scenario,
            "claimed_privacy_unit": self.claimed_privacy_unit,
            "normalized_claimed_unit": self.normalized_claimed_unit,
            "accounting_unit": self.accounting_unit,
            "raw_adjacency": self.raw_adjacency,
            "accountant_adjacency": self.accountant_adjacency,
            "unit_path": self.unit_path,
            "release_status": self.release_status,
            "assurance_status": self.assurance_status,
            "stability_method": self.stability_method,
            "stability_bound": _display(self.stability_bound),
            "base_epsilon": _display(self.base_epsilon),
            "base_delta": _display(self.base_delta),
            "reported_epsilon": _display(self.reported_epsilon),
            "reported_delta": _display(self.reported_delta),
            "reported_delta_log10": _display(self.reported_delta_log10),
            "conversion_vacuous": _display(self.conversion_vacuous),
            "mapping_file_sha256": self.mapping_file_sha256,
            "selected_mapping_sha256": self.selected_mapping_sha256,
            "pipeline_sha256": self.pipeline_sha256,
            "runtime_status": self.runtime_status,
            "rng_assurance": self.rng_assurance,
            "noise_seed_policy": self.noise_seed_policy,
            "metadata_classification": self.metadata_classification,
            "observed_event_kappa": _display(self.observed_event_kappa),
            "observed_owner_kappa": _display(self.observed_owner_kappa),
            "selected_record_count": _display(self.selected_record_count),
            "attribution_arity_max": _display(self.attribution_arity_max),
            "generated_record_unit": self.generated_record_unit,
            "supported_statement": self.supported_statement,
            "assurance_boundary": self.assurance_boundary,
            "do_not_say": self.do_not_say,
            "next_step": self.next_step,
            "issue_codes": ";".join(self.issue_codes()),
            "issues_json": json.dumps(
                [issue.as_dict() for issue in self.issues],
                sort_keys=True,
                separators=(",", ":"),
            ),
        }

    def as_public_record(self) -> Dict[str, Any]:
        allowed = self.release_status == "ALLOWED"
        public_benchmark = self.metadata_classification in {
            "public_benchmark",
            "controller_approved",
            "dp_released",
        }
        record: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "claimed_privacy_unit": self.claimed_privacy_unit,
            "accounting_unit": self.accounting_unit,
            "raw_adjacency": self.raw_adjacency,
            "accountant_adjacency": self.accountant_adjacency,
            "unit_path": self.unit_path,
            "release_status": self.release_status,
            "assurance_status": self.assurance_status,
            "supported_statement": self.supported_statement if allowed else "",
            "assurance_boundary": self.assurance_boundary,
            "do_not_say": self.do_not_say,
            "issue_codes": self.issue_codes(),
            "metadata_redacted": not public_benchmark,
        }
        if allowed:
            record.update(
                {
                    "epsilon": self.reported_epsilon,
                    "delta": self.reported_delta,
                    "stability_bound": self.stability_bound,
                }
            )
        if public_benchmark:
            record.update(
                {
                    "scenario": self.scenario,
                    "selected_mapping_sha256": self.selected_mapping_sha256,
                    "pipeline_sha256": self.pipeline_sha256,
                    "observed_event_kappa": self.observed_event_kappa,
                    "observed_owner_kappa": self.observed_owner_kappa,
                    "selected_record_count": self.selected_record_count,
                    "attribution_arity_max": self.attribution_arity_max,
                    "generated_record_unit": self.generated_record_unit,
                }
            )
        return record


def _display(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def normalize_unit(value: Any) -> str:
    unit = str(value or "").strip().lower()
    if unit in OWNER_ALIASES:
        return "owner"
    if unit in EVENT_ALIASES:
        return "event"
    if unit in WINDOW_ALIASES:
        return "window"
    return unit


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return bool(SHA256_RE.fullmatch(str(value or "").strip()))


def _mechanism_registry_path() -> Path:
    return Path(__file__).resolve().parents[1] / "docs" / MECHANISM_REGISTRY_FILE


def _load_registered_mechanism(
    mechanism: Mapping[str, Any],
    collector: "_Collector",
) -> Mapping[str, Any]:
    """Load and authenticate the exact executable-semantics registry entry."""

    mechanism_id = str(mechanism.get("mechanism_id") or "")
    mechanism_version = str(mechanism.get("mechanism_version") or "")
    registry_key = f"{mechanism_id}@{mechanism_version}"
    try:
        document = json.loads(
            _mechanism_registry_path().read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        collector.add(
            "MECHANISM_REGISTRY_UNAVAILABLE",
            "unverified",
            "mechanism",
            f"Cannot load the trusted mechanism registry: {exc}",
        )
        return {}
    entries = document.get("mechanisms")
    entry = entries.get(registry_key) if isinstance(entries, Mapping) else None
    if not isinstance(entry, Mapping):
        collector.add(
            "MECHANISM_CHECKER_UNREGISTERED",
            "unverified",
            "mechanism.mechanism_id",
            f"No exact executable-semantics adapter is registered for {registry_key!r}.",
        )
        return {}

    declared_entry_hash = str(
        mechanism.get("adapter_registry_entry_sha256") or ""
    )
    expected_entry_hash = canonical_json_sha256(entry)
    if not is_sha256(declared_entry_hash):
        collector.add(
            "MECHANISM_REGISTRY_ENTRY_DIGEST_INVALID",
            "invalid",
            "mechanism.adapter_registry_entry_sha256",
            "The mechanism must bind its trusted registry entry.",
        )
    elif declared_entry_hash.lower() != expected_entry_hash.lower():
        collector.add(
            "MECHANISM_REGISTRY_ENTRY_DIGEST_MISMATCH",
            "invalid",
            "mechanism.adapter_registry_entry_sha256",
            "The declared registry-entry digest does not match the trusted entry.",
        )

    for key in (
        "mechanism_id",
        "mechanism_version",
        "accountant_id",
        "sampler_law",
        "accountant_sampler_law",
        "sampling_implementation",
        "secure_rng_backend",
        "noise_generation",
        "noise_hardening",
        "gradient_aggregation",
        "update_normalization",
    ):
        expected = entry.get(key)
        if str(mechanism.get(key)) != str(expected):
            collector.add(
                "REGISTERED_MECHANISM_SEMANTICS_MISMATCH",
                "invalid",
                f"mechanism.{key}",
                f"Declared {key} does not match the trusted executable-semantics registry.",
            )
    return entry


def _as_finite_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _as_positive_int(value: Any) -> Optional[int]:
    try:
        parsed_float = float(value)
        parsed = int(parsed_float)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed_float != parsed or parsed < 1:
        return None
    return parsed


def _observed_kappa(audit_row: Mapping[str, Any], unit: str) -> Optional[int]:
    if unit == "event":
        return _as_positive_int(
            audit_row.get("event_kappa_used")
            or audit_row.get("event_kappa_max")
            or audit_row.get("event_kappa")
        )
    if unit == "owner":
        return _as_positive_int(
            audit_row.get("owner_windows_max_used")
            or audit_row.get("owner_kappa_max")
            or audit_row.get("owner_kappa")
        )
    if unit == "window":
        return 1
    return None


def _log_geometric_sum(epsilon: float, k: int) -> float:
    if k == 1:
        return 0.0
    if epsilon == 0.0:
        return math.log(k)

    def log_expm1(x: float) -> float:
        if not math.isfinite(x):
            return x
        if x < math.log(2.0):
            return math.log(math.expm1(x))
        return x + math.log1p(-math.exp(-x))

    k_epsilon = k * epsilon
    if not math.isfinite(k_epsilon):
        return float("inf")
    return log_expm1(k_epsilon) - log_expm1(epsilon)


def group_privacy_conversion(
    epsilon: float, delta: float, k: int
) -> Tuple[float, float, float, bool]:
    """Return (epsilon_k, delta_k, log10_delta_k, vacuous).

    Inputs are strict: epsilon must be finite/nonnegative, delta must lie in
    (0, 1), and k must be a positive integer.
    """

    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    if not math.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta must be finite and lie strictly between 0 and 1")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    epsilon_k = k * epsilon
    if not math.isfinite(epsilon_k):
        raise ValueError("converted epsilon is not finite")
    log_delta_k = math.log(delta) + _log_geometric_sum(epsilon, k)
    log10_delta_k = log_delta_k / math.log(10)
    vacuous = log_delta_k >= 0
    delta_k = 1.0 if vacuous else math.exp(log_delta_k)
    return epsilon_k, delta_k, log10_delta_k, vacuous


@lru_cache(maxsize=256)
def _opacus_rdp_epsilon(
    sample_rate: float,
    steps: int,
    noise_multiplier: float,
    delta: float,
) -> Tuple[str, float]:
    from rdp_accountant_lite import epsilon_for_poisson_gaussian

    distribution = importlib.metadata.distribution("opacus")
    epsilon, _best_alpha = epsilon_for_poisson_gaussian(
        sample_rate,
        steps,
        noise_multiplier,
        delta,
    )
    return distribution.version, epsilon


class _Collector:
    def __init__(self, result: ClaimValidationResult):
        self.result = result

    def add(self, code: str, category: str, field: str, message: str) -> None:
        self.result.issues.append(
            ValidationIssue(
                code=code,
                category=category,
                field=field,
                message=message,
            )
        )

    def require(
        self,
        mapping: Mapping[str, Any],
        key: str,
        path: str,
        *,
        category: str = "invalid",
    ) -> Any:
        value = mapping.get(key)
        if value is None or value == "":
            self.add(
                "MISSING_REQUIRED_FIELD",
                category,
                f"{path}.{key}",
                f"Required field {path}.{key} is missing.",
            )
        return value

    def has(self, category: str) -> bool:
        return any(issue.category == category for issue in self.result.issues)

    def has_any(self, categories: Iterable[str]) -> bool:
        selected = set(categories)
        return any(issue.category in selected for issue in self.result.issues)


def _mapping_block(
    report: Mapping[str, Any], key: str, collector: _Collector
) -> Mapping[str, Any]:
    value = report.get(key)
    if not isinstance(value, Mapping):
        collector.add(
            "INVALID_REQUIRED_OBJECT",
            "invalid",
            key,
            f"{key} must be an object.",
        )
        return {}
    return value


def _validate_report_schema(
    privacy_report: Mapping[str, Any], collector: _Collector
) -> None:
    """Apply the published JSON Schema before semantic validation.

    Schema validation is deliberately fail closed.  Manual coercion must not
    turn arrays, booleans, or other malformed values into plausible strings or
    numbers.
    """

    schema_path = Path(__file__).resolve().parents[1] / "docs" / (
        "privacy_report_schema_v2_0.json"
    )
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        collector.add(
            "SCHEMA_VALIDATOR_UNAVAILABLE",
            "unverified",
            "schema",
            "The required jsonschema dependency is unavailable.",
        )
        return
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validator = Draft202012Validator(schema)
        errors = sorted(
            validator.iter_errors(privacy_report),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        collector.add(
            "SCHEMA_LOAD_FAILED",
            "unverified",
            "schema",
            str(exc),
        )
        return

    for error in errors[:50]:
        path = ".".join(str(part) for part in error.absolute_path) or "<root>"
        collector.add(
            "REPORT_SCHEMA_INVALID",
            "invalid",
            path,
            error.message,
        )
    if len(errors) > 50:
        collector.add(
            "REPORT_SCHEMA_ERROR_LIMIT",
            "invalid",
            "schema",
            f"{len(errors) - 50} additional schema errors were suppressed.",
        )


def _validate_registered_accountant(
    mechanism: Mapping[str, Any],
    privacy: Mapping[str, Any],
    collector: _Collector,
) -> Mapping[str, Any]:
    """Recompute the base accountant output through an allowlisted checker."""

    accountant_id = str(mechanism.get("accountant_id") or "")
    accountant_version = str(mechanism.get("accountant_version") or "")
    registry_entry = _load_registered_mechanism(mechanism, collector)
    if accountant_id != "opacus_rdp":
        collector.add(
            "ACCOUNTANT_CHECKER_UNREGISTERED",
            "unverified",
            "mechanism.accountant_id",
            f"No base-accountant checker is registered for {accountant_id!r}.",
        )
        return registry_entry

    accounting_unit = normalize_unit(mechanism.get("accounting_unit"))
    expected_adjacency = (
        "owner_add_remove" if accounting_unit == "owner" else "add_remove"
    )
    if str(mechanism.get("accountant_adjacency") or "") != expected_adjacency:
        collector.add(
            "REGISTERED_ACCOUNTANT_ADJACENCY_MISMATCH",
            "invalid",
            "mechanism.accountant_adjacency",
            f"opacus_rdp expects {expected_adjacency} for {accounting_unit} records.",
        )
    if (
        str(mechanism.get("sampler_law") or "") != "poisson"
        or str(mechanism.get("accountant_sampler_law") or "") != "poisson"
    ):
        collector.add(
            "REGISTERED_ACCOUNTANT_SAMPLER_MISMATCH",
            "invalid",
            "mechanism.sampler_law",
            "The registered opacus_rdp checker requires Poisson sampling.",
        )
    if str(mechanism.get("privacy_convention") or "") != "rdp_converted":
        collector.add(
            "REGISTERED_ACCOUNTANT_CONVENTION_MISMATCH",
            "invalid",
            "mechanism.privacy_convention",
            "The registered opacus_rdp checker requires rdp_converted.",
        )
    if str(privacy.get("delta_convention") or "") != "accountant_delta_only":
        collector.add(
            "REGISTERED_ACCOUNTANT_DELTA_CONVENTION_MISMATCH",
            "invalid",
            "privacy.delta_convention",
            "The initial opacus_rdp path accepts accountant_delta_only.",
        )

    sample_rate = _as_finite_float(mechanism.get("sample_rate"))
    sample_rate_numerator = _as_positive_int(
        mechanism.get("sample_rate_numerator")
    )
    sample_rate_denominator = _as_positive_int(
        mechanism.get("sample_rate_denominator")
    )
    if (
        sample_rate_numerator is None
        or sample_rate_denominator is None
        or sample_rate_numerator > sample_rate_denominator
    ):
        collector.add(
            "EXACT_SAMPLE_RATE_INVALID",
            "invalid",
            "mechanism.sample_rate_numerator",
            "Poisson sampling requires a positive exact rational numerator/denominator with numerator <= denominator.",
        )
    elif (
        sample_rate is not None
        and sample_rate != sample_rate_numerator / sample_rate_denominator
    ):
        collector.add(
            "EXACT_SAMPLE_RATE_MISMATCH",
            "invalid",
            "mechanism.sample_rate",
            "The accountant sample rate must equal the bound exact rational sampling probability.",
        )
    steps = _as_positive_int(mechanism.get("steps"))
    noise = _as_finite_float(mechanism.get("noise_multiplier"))
    epsilon = _as_finite_float(privacy.get("epsilon"))
    delta = _as_finite_float(privacy.get("delta"))
    if (
        sample_rate is None
        or not 0 < sample_rate <= 1
        or steps is None
        or noise is None
        or noise <= 0
        or epsilon is None
        or epsilon < 0
        or delta is None
        or not 0 < delta < 1
    ):
        return registry_entry

    try:
        installed_version, recomputed_epsilon = _opacus_rdp_epsilon(
            sample_rate,
            steps,
            noise,
            delta,
        )
    except (ImportError, RuntimeError, ValueError, OverflowError) as exc:
        collector.add(
            "ACCOUNTANT_RECOMPUTATION_UNAVAILABLE",
            "unverified",
            "privacy.epsilon",
            str(exc),
        )
        return registry_entry
    if accountant_version != installed_version:
        collector.add(
            "ACCOUNTANT_VERSION_MISMATCH",
            "invalid",
            "mechanism.accountant_version",
            f"Report version={accountant_version!r}; checker version={installed_version!r}.",
        )
    if not math.isclose(
        epsilon,
        recomputed_epsilon,
        rel_tol=1e-7,
        abs_tol=1e-10,
    ):
        collector.add(
            "ACCOUNTANT_EPSILON_MISMATCH",
            "invalid",
            "privacy.epsilon",
            f"Reported epsilon={epsilon:.12g}; recomputed epsilon={recomputed_epsilon:.12g}.",
        )
    return registry_entry


def _validate_manifest(
    pipeline: Mapping[str, Any],
    audit_row: Mapping[str, Any],
    mechanism: Mapping[str, Any],
    registry_entry: Mapping[str, Any],
    collector: _Collector,
) -> Tuple[Mapping[str, Any], str, str, str]:
    mapping_file_hash = str(
        collector.require(pipeline, "mapping_file_sha256", "pipeline") or ""
    )
    selected_hash = str(
        collector.require(pipeline, "selected_mapping_sha256", "pipeline") or ""
    )
    pipeline_hash = str(
        collector.require(pipeline, "pipeline_sha256", "pipeline") or ""
    )
    for field_name, value in (
        ("pipeline.mapping_file_sha256", mapping_file_hash),
        ("pipeline.selected_mapping_sha256", selected_hash),
        ("pipeline.pipeline_sha256", pipeline_hash),
    ):
        if value and not is_sha256(value):
            collector.add(
                "INVALID_SHA256",
                "invalid",
                field_name,
                f"{field_name} must be a 64-character SHA-256 digest.",
            )

    audit_mapping_hash = str(audit_row.get("mapping_file_sha256") or "")
    audit_selected_hash = str(audit_row.get("selected_mapping_sha256") or "")
    if not is_sha256(audit_mapping_hash):
        collector.add(
            "AUDIT_MAPPING_HASH_MISSING",
            "invalid",
            "audit_row.mapping_file_sha256",
            "The audit row does not contain a valid mapping-file digest.",
        )
    elif mapping_file_hash and audit_mapping_hash.lower() != mapping_file_hash.lower():
        collector.add(
            "MAPPING_FILE_HASH_MISMATCH",
            "invalid",
            "pipeline.mapping_file_sha256",
            "The privacy report and audit row use different mapping files.",
        )
    if not is_sha256(audit_selected_hash):
        collector.add(
            "AUDIT_SELECTED_HASH_MISSING",
            "invalid",
            "audit_row.selected_mapping_sha256",
            "The audit row does not contain a valid selected-mapping digest.",
        )
    elif selected_hash and audit_selected_hash.lower() != selected_hash.lower():
        collector.add(
            "SELECTED_MAPPING_HASH_MISMATCH",
            "invalid",
            "pipeline.selected_mapping_sha256",
            "The privacy report and audit row use different selected mappings.",
        )

    manifest = pipeline.get("manifest")
    if not isinstance(manifest, Mapping):
        collector.add(
            "PIPELINE_MANIFEST_MISSING",
            "unverified",
            "pipeline.manifest",
            "A CSV digest alone is not a pipeline digest; the canonical manifest is required.",
        )
        manifest = {}
    else:
        computed = canonical_json_sha256(manifest)
        if pipeline_hash and computed.lower() != pipeline_hash.lower():
            collector.add(
                "PIPELINE_HASH_MISMATCH",
                "invalid",
                "pipeline.pipeline_sha256",
                "The supplied pipeline digest does not match the canonical manifest.",
            )

    required_manifest_fields = (
        "dataset_id",
        "split_id",
        "code_version",
        "code_artifact_id",
        "code_sha256",
        "accountant_checker_artifact_id",
        "accountant_checker_sha256",
        "support_convention",
        "record_identity_policy",
        "generated_record_unit",
        "mapping_file_sha256",
        "selected_mapping_sha256",
        "selected_record_count",
        "mechanism_sha256",
        "influence_support",
        "owner_attribution_status",
        "preprocessing",
        "raw_domain",
        "contribution_policy",
        "schedule",
        "population_definition",
        "population_size_policy",
        "model_selection_status",
        "released_model_sha256",
        "batch_trace_sha256",
    )
    for key in required_manifest_fields:
        collector.require(manifest, key, "pipeline.manifest", category="unverified")

    code_artifact_id = str(manifest.get("code_artifact_id") or "")
    code_hash = str(manifest.get("code_sha256") or "")
    registered_artifact_id = str(registry_entry.get("code_artifact_id") or "")
    registered_code_hash = str(registry_entry.get("code_sha256") or "")
    if not is_sha256(code_hash):
        collector.add(
            "PIPELINE_CODE_DIGEST_INVALID",
            "invalid",
            "pipeline.manifest.code_sha256",
            "The executable pipeline source must be bound by SHA-256.",
        )
    if registry_entry:
        if code_artifact_id != registered_artifact_id:
            collector.add(
                "PIPELINE_CODE_ARTIFACT_MISMATCH",
                "invalid",
                "pipeline.manifest.code_artifact_id",
                "The declared executable artifact differs from the trusted registry entry.",
            )
        if (
            is_sha256(code_hash)
            and is_sha256(registered_code_hash)
            and code_hash.lower() != registered_code_hash.lower()
        ):
            collector.add(
                "PIPELINE_CODE_REGISTRY_DIGEST_MISMATCH",
                "invalid",
                "pipeline.manifest.code_sha256",
                "The declared source digest differs from the trusted registry entry.",
            )
        project_root = Path(__file__).resolve().parents[1]
        source_path = (project_root / registered_artifact_id).resolve()
        try:
            source_path.relative_to(project_root.resolve())
        except ValueError:
            collector.add(
                "REGISTERED_CODE_PATH_INVALID",
                "invalid",
                "pipeline.manifest.code_artifact_id",
                "The trusted source path escapes the verifier root.",
            )
        else:
            try:
                actual_source_hash = file_sha256(source_path)
            except OSError as exc:
                collector.add(
                    "REGISTERED_CODE_UNAVAILABLE",
                    "unverified",
                    "pipeline.manifest.code_artifact_id",
                    f"Cannot read the registered executable source: {exc}",
                )
            else:
                if (
                    not is_sha256(registered_code_hash)
                    or actual_source_hash.lower()
                    != registered_code_hash.lower()
                ):
                    collector.add(
                        "REGISTERED_CODE_DIGEST_MISMATCH",
                        "invalid",
                        "pipeline.manifest.code_sha256",
                        "The executable source in the verifier bundle does not match the trusted registry.",
                    )

    checker_artifact_id = str(
        manifest.get("accountant_checker_artifact_id") or ""
    )
    checker_hash = str(manifest.get("accountant_checker_sha256") or "")
    registered_checker_id = str(
        registry_entry.get("accountant_checker_artifact_id") or ""
    )
    registered_checker_hash = str(
        registry_entry.get("accountant_checker_sha256") or ""
    )
    if not is_sha256(checker_hash):
        collector.add(
            "ACCOUNTANT_CHECKER_DIGEST_INVALID",
            "invalid",
            "pipeline.manifest.accountant_checker_sha256",
            "The executable accountant checker must be bound by SHA-256.",
        )
    if registry_entry:
        if checker_artifact_id != registered_checker_id:
            collector.add(
                "ACCOUNTANT_CHECKER_ARTIFACT_MISMATCH",
                "invalid",
                "pipeline.manifest.accountant_checker_artifact_id",
                "The declared accountant checker differs from the trusted registry.",
            )
        if (
            is_sha256(checker_hash)
            and is_sha256(registered_checker_hash)
            and checker_hash.lower() != registered_checker_hash.lower()
        ):
            collector.add(
                "ACCOUNTANT_CHECKER_REGISTRY_DIGEST_MISMATCH",
                "invalid",
                "pipeline.manifest.accountant_checker_sha256",
                "The declared accountant checker digest differs from the trusted registry.",
            )
        project_root = Path(__file__).resolve().parents[1]
        checker_path = (project_root / registered_checker_id).resolve()
        try:
            checker_path.relative_to(project_root.resolve())
            actual_checker_hash = file_sha256(checker_path)
        except ValueError:
            collector.add(
                "REGISTERED_ACCOUNTANT_CHECKER_PATH_INVALID",
                "invalid",
                "pipeline.manifest.accountant_checker_artifact_id",
                "The trusted accountant-checker path escapes the verifier root.",
            )
        except OSError as exc:
            collector.add(
                "REGISTERED_ACCOUNTANT_CHECKER_UNAVAILABLE",
                "unverified",
                "pipeline.manifest.accountant_checker_artifact_id",
                f"Cannot read the registered accountant checker: {exc}",
            )
        else:
            if (
                not is_sha256(registered_checker_hash)
                or actual_checker_hash.lower()
                != registered_checker_hash.lower()
            ):
                collector.add(
                    "REGISTERED_ACCOUNTANT_CHECKER_DIGEST_MISMATCH",
                    "invalid",
                    "pipeline.manifest.accountant_checker_sha256",
                    "The accountant checker in the verifier bundle does not match the trusted registry.",
                )

    support_convention = str(manifest.get("support_convention") or "")
    if support_convention != "half_open_integer_intervals":
        collector.add(
            "SUPPORT_CONVENTION_UNREGISTERED",
            "unverified",
            "pipeline.manifest.support_convention",
            "The built-in mapping checker requires half_open_integer_intervals.",
        )

    raw_generated_record_unit = str(manifest.get("generated_record_unit") or "")
    generated_record_unit = normalize_unit(raw_generated_record_unit)
    accounting_unit = normalize_unit(mechanism.get("accounting_unit"))
    if raw_generated_record_unit not in {"window", "event", "owner"}:
        collector.add(
            "GENERATED_RECORD_UNIT_INVALID",
            "invalid",
            "pipeline.manifest.generated_record_unit",
            "Generated record unit must be window, event, or owner.",
        )
    elif generated_record_unit != accounting_unit:
        collector.add(
            "GENERATED_ACCOUNTING_UNIT_MISMATCH",
            "invalid",
            "pipeline.manifest.generated_record_unit",
            "The mapping records and accountant must use the same generated unit.",
        )

    for key, expected in (
        ("mapping_file_sha256", mapping_file_hash),
        ("selected_mapping_sha256", selected_hash),
    ):
        value = str(manifest.get(key) or "")
        if not is_sha256(value):
            collector.add(
                "MANIFEST_DIGEST_INVALID",
                "invalid",
                f"pipeline.manifest.{key}",
                f"Manifest {key} must be a SHA-256 digest.",
            )
        elif expected and value.lower() != expected.lower():
            collector.add(
                "MANIFEST_DIGEST_MISMATCH",
                "invalid",
                f"pipeline.manifest.{key}",
                f"Manifest {key} does not match the pipeline field.",
            )

    selected_record_count = _as_positive_int(manifest.get("selected_record_count"))
    if selected_record_count is None:
        collector.add(
            "MANIFEST_RECORD_COUNT_INVALID",
            "invalid",
            "pipeline.manifest.selected_record_count",
            "Selected record count must be a positive integer.",
        )

    mechanism_hash = str(manifest.get("mechanism_sha256") or "")
    expected_mechanism_hash = canonical_json_sha256(mechanism)
    if not is_sha256(mechanism_hash):
        collector.add(
            "MECHANISM_DIGEST_INVALID",
            "invalid",
            "pipeline.manifest.mechanism_sha256",
            "The manifest must bind the canonical mechanism object.",
        )
    elif mechanism_hash.lower() != expected_mechanism_hash.lower():
        collector.add(
            "MECHANISM_DIGEST_MISMATCH",
            "invalid",
            "pipeline.manifest.mechanism_sha256",
            "The manifest mechanism digest does not match the declared mechanism.",
        )

    influence_support = manifest.get("influence_support")
    required_components = {
        "features",
        "labels",
        "weights",
        "selection",
        "preprocessing",
    }
    if not isinstance(influence_support, Mapping):
        collector.add(
            "INFLUENCE_SUPPORT_MANIFEST_INVALID",
            "unverified",
            "pipeline.manifest.influence_support",
            "Influence support must explicitly cover every required component.",
        )
    else:
        status = str(influence_support.get("status") or "")
        components = influence_support.get("components")
        component_set = (
            {str(component) for component in components}
            if isinstance(components, list)
            else set()
        )
        if status != "complete" or not required_components.issubset(component_set):
            collector.add(
                "INFLUENCE_SUPPORT_MANIFEST_INCOMPLETE",
                "unverified",
                "pipeline.manifest.influence_support",
                "Complete feature, label, weight, selection, and preprocessing support is required.",
            )

    if str(manifest.get("owner_attribution_status") or "") != "complete":
        collector.add(
            "OWNER_ATTRIBUTION_INCOMPLETE",
            "unverified",
            "pipeline.manifest.owner_attribution_status",
            "Owner attribution must be explicitly complete.",
        )

    preprocessing = manifest.get("preprocessing")
    if not isinstance(preprocessing, list) or not preprocessing:
        collector.add(
            "PREPROCESSING_MANIFEST_INVALID",
            "unverified",
            "pipeline.manifest.preprocessing",
            "Preprocessing must be a nonempty list, including an explicit identity step when no transform is used.",
        )
    else:
        for index, step in enumerate(preprocessing):
            field_name = f"pipeline.manifest.preprocessing[{index}]"
            if not isinstance(step, Mapping):
                collector.add(
                    "PREPROCESSING_STEP_INVALID",
                    "invalid",
                    field_name,
                    "Each preprocessing entry must be an object.",
                )
                continue
            classification = str(step.get("classification") or "")
            parameters = step.get("parameters")
            parameters_hash = str(step.get("parameters_sha256") or "")
            if not isinstance(parameters, Mapping):
                collector.add(
                    "PREPROCESSING_PARAMETERS_MISSING",
                    "unverified",
                    f"{field_name}.parameters",
                    "Every preprocessing step must include its canonical parameter object.",
                )
            if not is_sha256(parameters_hash):
                collector.add(
                    "PREPROCESSING_PARAMETERS_DIGEST_INVALID",
                    "unverified",
                    f"{field_name}.parameters_sha256",
                    "Every preprocessing step must bind its parameters.",
                )
            elif (
                isinstance(parameters, Mapping)
                and parameters_hash.lower()
                != canonical_json_sha256(parameters).lower()
            ):
                collector.add(
                    "PREPROCESSING_PARAMETERS_DIGEST_MISMATCH",
                    "invalid",
                    f"{field_name}.parameters_sha256",
                    "Preprocessing parameter digest does not match its canonical object.",
                )
            if classification not in VALID_PREPROCESSING:
                collector.add(
                    "PREPROCESSING_CLASSIFICATION_INVALID",
                    "invalid",
                    f"{field_name}.classification",
                    f"Unknown preprocessing classification: {classification!r}.",
                )
            elif classification in {"private_global", "private_adaptive"}:
                collector.add(
                    "PRIVATE_PREPROCESSING_UNACCOUNTED",
                    "invalid",
                    f"{field_name}.classification",
                    "Unaccounted private global/adaptive preprocessing blocks a positive claim.",
                )
            elif classification == "separately_dp":
                collector.add(
                    "SEPARATELY_DP_PREPROCESSING_UNVERIFIED",
                    "unverified",
                    field_name,
                    "Separately DP preprocessing requires a registered release/composition checker.",
                )

    contribution_policy = manifest.get("contribution_policy")
    if isinstance(contribution_policy, Mapping):
        classification = str(contribution_policy.get("classification") or "")
        policy_id = str(contribution_policy.get("id") or "")
        parameters = contribution_policy.get("parameters")
        parameters_hash = str(contribution_policy.get("parameters_sha256") or "")
        if not policy_id:
            collector.add(
                "CONTRIBUTION_POLICY_ID_MISSING",
                "unverified",
                "pipeline.manifest.contribution_policy.id",
                "Contribution policy identifier is required.",
            )
        if not is_sha256(parameters_hash):
            collector.add(
                "CONTRIBUTION_POLICY_DIGEST_INVALID",
                "unverified",
                "pipeline.manifest.contribution_policy.parameters_sha256",
                "Contribution policy parameters must be content-bound.",
            )
        elif not isinstance(parameters, Mapping):
            collector.add(
                "CONTRIBUTION_POLICY_PARAMETERS_MISSING",
                "unverified",
                "pipeline.manifest.contribution_policy.parameters",
                "Contribution policy parameters must be an object.",
            )
        elif parameters_hash.lower() != canonical_json_sha256(parameters).lower():
            collector.add(
                "CONTRIBUTION_POLICY_DIGEST_MISMATCH",
                "invalid",
                "pipeline.manifest.contribution_policy.parameters_sha256",
                "Contribution policy digest does not match its canonical object.",
            )
        if classification not in {"public_fixed", "not_applicable"}:
            collector.add(
                "CONTRIBUTION_POLICY_NOT_PUBLIC",
                "invalid",
                "pipeline.manifest.contribution_policy.classification",
                "The built-in path accepts only public-fixed or not-applicable contribution policies.",
            )
    elif contribution_policy not in {None, ""}:
        collector.add(
            "CONTRIBUTION_POLICY_INVALID",
            "invalid",
            "pipeline.manifest.contribution_policy",
            "Contribution policy must be an object.",
        )

    raw_domain = manifest.get("raw_domain")
    if not isinstance(raw_domain, Mapping):
        collector.add(
            "RAW_DOMAIN_MANIFEST_INVALID",
            "unverified",
            "pipeline.manifest.raw_domain",
            "A content-bound raw-data domain is required.",
        )
    else:
        raw_domain_id = str(raw_domain.get("id") or "")
        classification = str(raw_domain.get("classification") or "")
        parameters = raw_domain.get("parameters")
        parameters_hash = str(raw_domain.get("parameters_sha256") or "")
        if not raw_domain_id:
            collector.add(
                "RAW_DOMAIN_ID_MISSING",
                "unverified",
                "pipeline.manifest.raw_domain.id",
                "Raw-data domain identifier is required.",
            )
        if classification != "public_fixed":
            collector.add(
                "RAW_DOMAIN_NOT_PUBLIC_FIXED",
                "invalid",
                "pipeline.manifest.raw_domain.classification",
                "The registered event-stability path requires a public-fixed raw domain.",
            )
        if not isinstance(parameters, Mapping):
            collector.add(
                "RAW_DOMAIN_PARAMETERS_MISSING",
                "unverified",
                "pipeline.manifest.raw_domain.parameters",
                "Raw-domain parameters must be explicit.",
            )
        if not is_sha256(parameters_hash):
            collector.add(
                "RAW_DOMAIN_PARAMETERS_DIGEST_INVALID",
                "unverified",
                "pipeline.manifest.raw_domain.parameters_sha256",
                "Raw-domain parameters must be content-bound.",
            )
        elif (
            isinstance(parameters, Mapping)
            and parameters_hash.lower()
            != canonical_json_sha256(parameters).lower()
        ):
            collector.add(
                "RAW_DOMAIN_PARAMETERS_DIGEST_MISMATCH",
                "invalid",
                "pipeline.manifest.raw_domain.parameters_sha256",
                "Raw-domain parameter digest does not match its canonical object.",
            )

    schedule = manifest.get("schedule")
    if not isinstance(schedule, Mapping):
        collector.add(
            "SCHEDULE_MANIFEST_INVALID",
            "unverified",
            "pipeline.manifest.schedule",
            "A content-bound schedule manifest is required.",
        )
    else:
        schedule_parameters = schedule.get("parameters")
        if str(schedule.get("mode") or "") != "fixed_mapping":
            collector.add(
                "SCHEDULE_MANIFEST_MODE_UNREGISTERED",
                "unverified",
                "pipeline.manifest.schedule.mode",
                "The built-in path accepts only a fixed_mapping schedule.",
            )
        if str(schedule.get("evidence_status") or "") != "validated":
            collector.add(
                "SCHEDULE_MANIFEST_UNVERIFIED",
                "unverified",
                "pipeline.manifest.schedule.evidence_status",
                "Schedule evidence must be validated.",
            )
        if not is_sha256(schedule.get("parameters_sha256")):
            collector.add(
                "SCHEDULE_PARAMETERS_DIGEST_INVALID",
                "unverified",
                "pipeline.manifest.schedule.parameters_sha256",
                "Schedule parameters must be content-bound.",
            )
        elif not isinstance(schedule_parameters, Mapping):
            collector.add(
                "SCHEDULE_PARAMETERS_MISSING",
                "unverified",
                "pipeline.manifest.schedule.parameters",
                "Schedule parameters must be an object.",
            )
        elif str(schedule.get("parameters_sha256")).lower() != canonical_json_sha256(
            schedule_parameters
        ).lower():
            collector.add(
                "SCHEDULE_PARAMETERS_DIGEST_MISMATCH",
                "invalid",
                "pipeline.manifest.schedule.parameters_sha256",
                "Schedule parameter digest does not match its canonical object.",
            )

    population_policy = str(manifest.get("population_size_policy") or "")
    if population_policy not in {
        "public_fixed",
        "not_used_in_release_mechanism",
    }:
        collector.add(
            "POPULATION_POLICY_UNVERIFIED",
            "unverified",
            "pipeline.manifest.population_size_policy",
            "Population size must be public-fixed or absent from the release mechanism and update normalization.",
        )

    model_selection = str(manifest.get("model_selection_status") or "")
    if model_selection not in {"public", "public_validation", "not_applicable"}:
        collector.add(
            "MODEL_SELECTION_NOT_COVERED",
            "invalid",
            "pipeline.manifest.model_selection_status",
            "Private or unknown model selection is not covered by the built-in certificate.",
        )

    for key in ("released_model_sha256", "batch_trace_sha256"):
        if not is_sha256(manifest.get(key)):
            collector.add(
                "EXECUTION_ARTIFACT_DIGEST_INVALID",
                "invalid",
                f"pipeline.manifest.{key}",
                "Matched post-execution evidence requires a valid artifact digest.",
            )

    return manifest, mapping_file_hash, selected_hash, pipeline_hash


def _validate_actual_mapping(
    mapping_path: Optional[Path],
    scenario: str,
    pipeline: Mapping[str, Any],
    manifest: Mapping[str, Any],
    audit_row: Mapping[str, Any],
    result: ClaimValidationResult,
    collector: _Collector,
) -> Optional[MappingEvidence]:
    if mapping_path is None:
        collector.add(
            "MAPPING_EVIDENCE_FILE_REQUIRED",
            "unverified",
            "mapping_file",
            "Positive certification requires the actual mapping CSV, not only copied digest strings.",
        )
        return None

    evidence = inspect_mapping_evidence(Path(mapping_path), scenario)
    for issue in evidence.issues:
        category = (
            "unverified"
            if issue.code
            in {"MAPPING_EVIDENCE_FILE_MISSING", "MAPPING_SCENARIO_EMPTY"}
            else "invalid"
        )
        collector.add(issue.code, category, issue.field, issue.message)
    if not evidence.valid:
        return evidence

    result.mapping_file_sha256 = evidence.mapping_file_sha256
    result.selected_mapping_sha256 = evidence.selected_mapping_sha256
    result.observed_event_kappa = evidence.event_kappa
    result.observed_owner_kappa = evidence.owner_kappa
    result.selected_record_count = evidence.selected_record_count
    result.attribution_arity_max = evidence.attribution_arity_max
    result.generated_record_unit = evidence.generated_unit

    declared_generated_unit = normalize_unit(manifest.get("generated_record_unit"))
    if evidence.generated_unit != declared_generated_unit:
        collector.add(
            "ACTUAL_GENERATED_UNIT_MISMATCH",
            "invalid",
            "pipeline.manifest.generated_record_unit",
            "The actual mapping generated_unit does not match the pipeline manifest.",
        )

    declared_mapping_hash = str(pipeline.get("mapping_file_sha256") or "")
    declared_selected_hash = str(pipeline.get("selected_mapping_sha256") or "")
    audit_mapping_hash = str(audit_row.get("mapping_file_sha256") or "")
    audit_selected_hash = str(audit_row.get("selected_mapping_sha256") or "")
    manifest_mapping_hash = str(manifest.get("mapping_file_sha256") or "")
    manifest_selected_hash = str(manifest.get("selected_mapping_sha256") or "")
    for field_name, supplied, observed in (
        (
            "pipeline.mapping_file_sha256",
            declared_mapping_hash,
            evidence.mapping_file_sha256,
        ),
        (
            "pipeline.selected_mapping_sha256",
            declared_selected_hash,
            evidence.selected_mapping_sha256,
        ),
        (
            "audit_row.mapping_file_sha256",
            audit_mapping_hash,
            evidence.mapping_file_sha256,
        ),
        (
            "audit_row.selected_mapping_sha256",
            audit_selected_hash,
            evidence.selected_mapping_sha256,
        ),
        (
            "pipeline.manifest.mapping_file_sha256",
            manifest_mapping_hash,
            evidence.mapping_file_sha256,
        ),
        (
            "pipeline.manifest.selected_mapping_sha256",
            manifest_selected_hash,
            evidence.selected_mapping_sha256,
        ),
    ):
        if supplied.lower() != observed.lower():
            collector.add(
                "ACTUAL_MAPPING_DIGEST_MISMATCH",
                "invalid",
                field_name,
                "The supplied digest does not match the mapping CSV opened by the validator.",
            )

    manifest_count = _as_positive_int(manifest.get("selected_record_count"))
    audit_count = _as_positive_int(audit_row.get("num_windows_used"))
    if manifest_count != evidence.selected_record_count:
        collector.add(
            "MAPPING_RECORD_COUNT_MISMATCH",
            "invalid",
            "pipeline.manifest.selected_record_count",
            f"Manifest count={manifest_count}; actual selected rows={evidence.selected_record_count}.",
        )
    if audit_count is None:
        collector.add(
            "AUDIT_RECORD_COUNT_MISSING",
            "unverified",
            "audit_row.num_windows_used",
            "The mapping diagnostic must record the selected row count.",
        )
    elif audit_count != evidence.selected_record_count:
        collector.add(
            "AUDIT_RECORD_COUNT_MISMATCH",
            "invalid",
            "audit_row.num_windows_used",
            f"Audit count={audit_count}; actual selected rows={evidence.selected_record_count}.",
        )

    audit_event_kappa = _observed_kappa(audit_row, "event")
    audit_owner_kappa = _observed_kappa(audit_row, "owner")
    if audit_event_kappa != evidence.event_kappa:
        collector.add(
            "AUDIT_EVENT_KAPPA_MISMATCH",
            "invalid",
            "audit_row.event_kappa_used",
            f"Audit kappa={audit_event_kappa}; recomputed kappa={evidence.event_kappa}.",
        )
    if audit_owner_kappa != evidence.owner_kappa:
        collector.add(
            "AUDIT_OWNER_KAPPA_MISMATCH",
            "invalid",
            "audit_row.owner_windows_max_used",
            f"Audit kappa={audit_owner_kappa}; recomputed kappa={evidence.owner_kappa}.",
        )
    return evidence


def _validate_runtime(
    runtime: Mapping[str, Any],
    mechanism: Mapping[str, Any],
    privacy: Mapping[str, Any],
    manifest: Mapping[str, Any],
    selected_hash: str,
    pipeline_hash: str,
    collector: _Collector,
) -> str:
    status = str(collector.require(runtime, "status", "runtime_trace") or "")
    if status == "mismatched":
        collector.add(
            "RUNTIME_TRACE_MISMATCH",
            "invalid",
            "runtime_trace.status",
            "The runtime trace is explicitly mismatched.",
        )
    elif status in {"missing", "not_applicable"}:
        collector.add(
            "RUNTIME_TRACE_NOT_BOUND",
            "unverified",
            "runtime_trace.status",
            "Post-execution wording is blocked without a matched runtime trace.",
        )
    elif status != "matched":
        collector.add(
            "RUNTIME_TRACE_STATUS_INVALID",
            "invalid",
            "runtime_trace.status",
            f"Unknown runtime trace status: {status!r}.",
        )
    else:
        exact_fields = (
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
        numeric_fields = (
            "sample_rate",
            "noise_multiplier",
            "clipping_norm",
            "optimizer_step_size",
            "epsilon",
            "delta",
        )
        evidence = runtime.get("evidence")
        if not isinstance(evidence, Mapping):
            collector.add(
                "RUNTIME_EVIDENCE_MISSING",
                "unverified",
                "runtime_trace.evidence",
                "Matched runtime status requires content-bound execution evidence.",
            )
            evidence = {}
        evidence_hash = canonical_json_sha256(evidence)
        expected: Dict[str, Any] = {
            **{key: mechanism.get(key) for key in exact_fields},
            **{key: mechanism.get(key) for key in numeric_fields},
            "code_artifact_id": manifest.get("code_artifact_id"),
            "code_sha256": manifest.get("code_sha256"),
            "accountant_checker_artifact_id": manifest.get(
                "accountant_checker_artifact_id"
            ),
            "accountant_checker_sha256": manifest.get(
                "accountant_checker_sha256"
            ),
            "selected_mapping_sha256": selected_hash,
            "pipeline_sha256": pipeline_hash,
            "epsilon": privacy.get("epsilon"),
            "delta": privacy.get("delta"),
            "delta_convention": privacy.get("delta_convention"),
            "runtime_evidence_sha256": evidence_hash,
        }
        trace_payload = {
            key: runtime.get(key) for key in (*exact_fields, *numeric_fields)
        }
        trace_hash = str(runtime.get("trace_hash") or "")
        if not is_sha256(trace_hash):
            collector.add(
                "RUNTIME_TRACE_HASH_INVALID",
                "invalid",
                "runtime_trace.trace_hash",
                "A matched runtime trace requires a valid trace digest.",
            )
        elif trace_hash.lower() != canonical_json_sha256(trace_payload).lower():
            collector.add(
                "RUNTIME_TRACE_DIGEST_MISMATCH",
                "invalid",
                "runtime_trace.trace_hash",
                "The trace digest does not match the canonical runtime fields.",
            )

        for key in exact_fields:
            if str(runtime.get(key)) != str(expected.get(key)):
                collector.add(
                    "RUNTIME_FIELD_MISMATCH",
                    "invalid",
                    f"runtime_trace.{key}",
                    f"Runtime {key} does not match the privacy contract.",
                )
        for key in numeric_fields:
            actual_value = _as_finite_float(runtime.get(key))
            expected_value = _as_finite_float(expected.get(key))
            if (
                actual_value is None
                or expected_value is None
                or not math.isclose(
                    actual_value,
                    expected_value,
                    rel_tol=1e-12,
                    abs_tol=0.0,
                )
            ):
                collector.add(
                    "RUNTIME_NUMERIC_FIELD_MISMATCH",
                    "invalid",
                    f"runtime_trace.{key}",
                    f"Runtime {key} does not match the privacy contract.",
                )
        executed_updates = _as_positive_int(evidence.get("executed_updates"))
        if executed_updates != _as_positive_int(mechanism.get("steps")):
            collector.add(
                "RUNTIME_EXECUTED_STEPS_MISMATCH",
                "invalid",
                "runtime_trace.evidence.executed_updates",
                "Execution evidence must record exactly the declared number of updates.",
            )
        for key in ("released_model_sha256", "batch_trace_sha256"):
            evidence_value = str(evidence.get(key) or "")
            manifest_value = str(manifest.get(key) or "")
            if not is_sha256(evidence_value):
                collector.add(
                    "RUNTIME_ARTIFACT_DIGEST_INVALID",
                    "invalid",
                    f"runtime_trace.evidence.{key}",
                    "Execution evidence requires a valid artifact digest.",
                )
            elif evidence_value.lower() != manifest_value.lower():
                collector.add(
                    "RUNTIME_ARTIFACT_DIGEST_MISMATCH",
                    "invalid",
                    f"runtime_trace.evidence.{key}",
                    "Runtime and pipeline manifest bind different execution artifacts.",
                )
        if str(evidence.get("secure_sampling_rng") or "") != str(
            mechanism.get("sampling_implementation") or ""
        ):
            collector.add(
                "RUNTIME_SAMPLING_IMPLEMENTATION_MISMATCH",
                "invalid",
                "runtime_trace.evidence.secure_sampling_rng",
                "Runtime sampling implementation differs from the registered mechanism.",
            )
        if str(evidence.get("secure_noise_rng") or "") != str(
            mechanism.get("noise_generation") or ""
        ):
            collector.add(
                "RUNTIME_NOISE_IMPLEMENTATION_MISMATCH",
                "invalid",
                "runtime_trace.evidence.secure_noise_rng",
                "Runtime noise implementation differs from the registered mechanism.",
            )
    return status


def _compute_stability_bound(
    claim: str,
    raw_adjacency: str,
    accountant_adjacency: str,
    claim_contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
    mapping_evidence: Optional[MappingEvidence],
    accounting_unit: str,
    collector: _Collector,
) -> Tuple[str, Optional[int]]:
    method = str(
        collector.require(
            claim_contract,
            "stability_method",
            f"claim_contracts.{claim}",
            category="unverified",
        )
        or ""
    )
    supplied_k = _as_positive_int(claim_contract.get("stability_bound"))
    if supplied_k is None:
        collector.add(
            "STABILITY_BOUND_INVALID",
            "unverified",
            f"claim_contracts.{claim}.stability_bound",
            "A positive integer stability bound is required.",
        )

    expected_k: Optional[int] = None
    if method == "identity_generated_unit":
        if claim != accounting_unit:
            collector.add(
                "IDENTITY_UNIT_MISMATCH",
                "invalid",
                f"claim_contracts.{claim}.stability_method",
                "Identity stability requires claimed and accounting units to match.",
            )
        if raw_adjacency != accountant_adjacency:
            collector.add(
                "IDENTITY_ADJACENCY_MISMATCH",
                "invalid",
                f"claim_contracts.{claim}.raw_adjacency",
                "Identity stability requires the exact accountant adjacency.",
            )
        expected_k = 1
    elif method == "fixed_influence_replacements":
        if claim != "event":
            collector.add(
                "STABILITY_METHOD_UNIT_MISMATCH",
                "invalid",
                f"claim_contracts.{claim}.stability_method",
                "fixed_influence_replacements is registered only for event claims.",
            )
        if raw_adjacency != REGISTERED_EVENT_RAW_ADJACENCY:
            collector.add(
                "RAW_ADJACENCY_NOT_SUPPORTED",
                "invalid",
                f"claim_contracts.{claim}.raw_adjacency",
                "The fixed-influence proof requires the registered fixed-owner/fixed-slot payload-replacement adjacency.",
            )
        raw_domain = manifest.get("raw_domain")
        raw_domain_id = (
            str(raw_domain.get("id") or "")
            if isinstance(raw_domain, Mapping)
            else ""
        )
        declared_domain_id = str(claim_contract.get("raw_domain_id") or "")
        declared_domain_hash = str(
            claim_contract.get("raw_domain_sha256") or ""
        )
        expected_domain_hash = (
            canonical_json_sha256(raw_domain)
            if isinstance(raw_domain, Mapping)
            else ""
        )
        if (
            raw_domain_id != REGISTERED_EVENT_RAW_DOMAIN_ID
            or declared_domain_id != REGISTERED_EVENT_RAW_DOMAIN_ID
        ):
            collector.add(
                "RAW_EVENT_DOMAIN_UNREGISTERED",
                "unverified",
                f"claim_contracts.{claim}.raw_domain_id",
                "Event replacement stability requires the registered fixed-owner/fixed-slot payload domain.",
            )
        if (
            not is_sha256(declared_domain_hash)
            or declared_domain_hash.lower() != expected_domain_hash.lower()
        ):
            collector.add(
                "RAW_EVENT_DOMAIN_DIGEST_MISMATCH",
                "invalid",
                f"claim_contracts.{claim}.raw_domain_sha256",
                "The event claim does not bind the pipeline raw-domain declaration.",
            )
        raw_parameters = (
            raw_domain.get("parameters")
            if isinstance(raw_domain, Mapping)
            else None
        )
        if not isinstance(raw_parameters, Mapping) or not (
            raw_parameters.get("fixed_presence") is True
            and set(raw_parameters.get("immutable_fields") or ())
            >= {"owner_id", "event_slot"}
            and set(raw_parameters.get("mutable_fields") or ()).isdisjoint(
                {"owner_id", "event_slot"}
            )
        ):
            collector.add(
                "RAW_EVENT_DOMAIN_SEMANTICS_INVALID",
                "invalid",
                "pipeline.manifest.raw_domain.parameters",
                "The registered domain must fix event presence, owner identity, and slot identity while varying payload only.",
            )
        if str(manifest.get("record_identity_policy") or "") not in {
            "public_fixed_ids",
            "owner_partition_ids",
        }:
            collector.add(
                "RECORD_IDENTITY_NOT_FIXED",
                "unverified",
                "pipeline.manifest.record_identity_policy",
                "Fixed-influence replacement stability requires public-fixed generated-record identities.",
            )
        observed = (
            mapping_evidence.event_kappa
            if mapping_evidence is not None and mapping_evidence.valid
            else None
        )
        if observed is None:
            collector.add(
                "OBSERVED_EVENT_KAPPA_MISSING",
                "unverified",
                "mapping_file",
                "The event influence count was not recomputed from valid mapping evidence.",
            )
        elif accountant_adjacency == "replace_one":
            expected_k = observed
        elif accountant_adjacency == "add_remove":
            expected_k = 2 * observed
        else:
            collector.add(
                "ACCOUNTANT_ADJACENCY_NOT_SUPPORTED",
                "invalid",
                "mechanism.accountant_adjacency",
                "The built-in event replacement proof supports replace-one or add/remove accountants.",
            )
    elif method == "owner_partition_add_remove":
        if claim != "owner":
            collector.add(
                "STABILITY_METHOD_UNIT_MISMATCH",
                "invalid",
                f"claim_contracts.{claim}.stability_method",
                "owner_partition_add_remove is registered only for owner claims.",
            )
        if raw_adjacency != "owner_add_remove" or accountant_adjacency != "add_remove":
            collector.add(
                "OWNER_ADJACENCY_MISMATCH",
                "invalid",
                f"claim_contracts.{claim}.raw_adjacency",
                "Owner partition removal requires owner add/remove mapped to generated add/remove distance.",
            )
        if str(manifest.get("record_identity_policy") or "") != "owner_partition_ids":
            collector.add(
                "OWNER_PARTITION_NOT_FIXED",
                "unverified",
                "pipeline.manifest.record_identity_policy",
                "Owner removal stability requires a fixed owner partition.",
            )
        if str(manifest.get("cross_owner_dependency") or "") != "none":
            collector.add(
                "CROSS_OWNER_DEPENDENCY_UNVERIFIED",
                "unverified",
                "pipeline.manifest.cross_owner_dependency",
                "Owner removal stability requires an explicit no-cross-owner-dependency declaration.",
            )
        if (
            mapping_evidence is not None
            and mapping_evidence.valid
            and mapping_evidence.attribution_arity_max != 1
        ):
            collector.add(
                "OWNER_PARTITION_ATTRIBUTION_ARITY_INVALID",
                "invalid",
                "mapping_file.owner_ids",
                "owner_partition_add_remove requires exactly one attributed owner per generated record.",
            )
        observed = (
            mapping_evidence.owner_kappa
            if mapping_evidence is not None and mapping_evidence.valid
            else None
        )
        if observed is None:
            collector.add(
                "OBSERVED_OWNER_KAPPA_MISSING",
                "unverified",
                "mapping_file",
                "The owner contribution count was not recomputed from valid mapping evidence.",
            )
        contribution_policy = manifest.get("contribution_policy")
        policy_id = (
            str(contribution_policy.get("id") or "")
            if isinstance(contribution_policy, Mapping)
            else ""
        )
        classification = (
            str(contribution_policy.get("classification") or "")
            if isinstance(contribution_policy, Mapping)
            else ""
        )
        parameters = (
            contribution_policy.get("parameters")
            if isinstance(contribution_policy, Mapping)
            else None
        )
        public_cap = (
            _as_positive_int(parameters.get("cap"))
            if isinstance(parameters, Mapping)
            else None
        )
        enforcement = (
            str(parameters.get("enforcement") or "")
            if isinstance(parameters, Mapping)
            else ""
        )
        if (
            policy_id != REGISTERED_OWNER_CAP_POLICY_ID
            or classification != "public_fixed"
            or public_cap is None
            or enforcement != REGISTERED_OWNER_CAP_ENFORCEMENT
        ):
            collector.add(
                "OWNER_PUBLIC_CAP_UNVERIFIED",
                "unverified",
                "pipeline.manifest.contribution_policy",
                "Owner conversion requires a registered public generated-record cap and deterministic enforcement.",
            )
        else:
            expected_k = public_cap
            if observed is not None and observed > public_cap:
                collector.add(
                    "OWNER_PUBLIC_CAP_VIOLATED",
                    "invalid",
                    "mapping_file.owner_ids",
                    f"Observed owner contribution {observed} exceeds public cap {public_cap}.",
                )
    elif method:
        collector.add(
            "STABILITY_METHOD_UNREGISTERED",
            "unverified",
            f"claim_contracts.{claim}.stability_method",
            f"Stability method {method!r} has no registered checker.",
        )

    if expected_k is not None and supplied_k is not None and supplied_k != expected_k:
        collector.add(
            "STABILITY_BOUND_MISMATCH",
            "invalid",
            f"claim_contracts.{claim}.stability_bound",
            f"Supplied K={supplied_k} but the registered checker computes K={expected_k}.",
        )
    return method, expected_k


def validate_privacy_claim(
    audit_row: Mapping[str, Any],
    privacy_report: Mapping[str, Any],
    claim_unit: str,
    mapping_path: Optional[Path] = None,
) -> ClaimValidationResult:
    scenario = str(audit_row.get("scenario") or privacy_report.get("scenario") or "")
    normalized_claim = normalize_unit(claim_unit)
    result = ClaimValidationResult(
        scenario=scenario,
        claimed_privacy_unit=str(claim_unit),
        normalized_claimed_unit=normalized_claim,
        observed_event_kappa=_observed_kappa(audit_row, "event"),
        observed_owner_kappa=_observed_kappa(audit_row, "owner"),
    )
    collector = _Collector(result)

    if normalized_claim not in {"window", "event", "owner"}:
        collector.add(
            "CLAIM_UNIT_UNSUPPORTED",
            "invalid",
            "claim_unit",
            f"Unsupported claimed unit: {claim_unit!r}.",
        )

    schema_version = str(privacy_report.get("schema_version") or "")
    if schema_version != SCHEMA_VERSION:
        collector.add(
            "UNSUPPORTED_SCHEMA_VERSION",
            "unverified",
            "schema_version",
            f"Expected schema version {SCHEMA_VERSION}; received {schema_version or 'legacy/missing'}.",
        )
        result.release_status = (
            "BLOCKED_INVALID"
            if collector.has("invalid")
            else "BLOCKED_UNVERIFIED"
        )
        result.assurance_status = "DIAGNOSTIC_ONLY"
        result.do_not_say = (
            "Do not emit any positive privacy sentence from a legacy or unknown report schema."
        )
        result.next_step = (
            "Regenerate the privacy report with schema v2.0 and rerun the validator."
        )
        return result

    _validate_report_schema(privacy_report, collector)

    audit_scenario = str(audit_row.get("scenario") or "")
    if not audit_scenario:
        collector.add(
            "AUDIT_SCENARIO_MISSING",
            "invalid",
            "audit_row.scenario",
            "The mapping diagnostic row must name its scenario.",
        )
    audit_claim_unit = normalize_unit(audit_row.get("claimed_privacy_unit"))
    if audit_claim_unit != normalized_claim:
        collector.add(
            "AUDIT_CLAIM_UNIT_MISMATCH",
            "invalid",
            "audit_row.claimed_privacy_unit",
            "The selected mapping diagnostic row names a different claim unit.",
        )

    report_scenario = str(privacy_report.get("scenario") or "")
    if not report_scenario:
        collector.add(
            "REPORT_SCENARIO_MISSING",
            "invalid",
            "scenario",
            "Privacy report scenario is required.",
        )
    elif scenario and report_scenario != scenario:
        collector.add(
            "SCENARIO_MISMATCH",
            "invalid",
            "scenario",
            "Privacy report and audit row name different scenarios.",
        )

    mechanism = _mapping_block(privacy_report, "mechanism", collector)
    privacy = _mapping_block(privacy_report, "privacy", collector)
    pipeline = _mapping_block(privacy_report, "pipeline", collector)
    runtime = _mapping_block(privacy_report, "runtime_trace", collector)

    for key in (
        "mechanism_id",
        "mechanism_version",
        "accountant_id",
        "accountant_version",
        "accounting_unit",
        "accountant_adjacency",
        "sampling_unit",
        "sampler_law",
        "accountant_sampler_law",
        "sample_rate",
        "sample_rate_numerator",
        "sample_rate_denominator",
        "sampling_implementation",
        "steps",
        "noise_multiplier",
        "secure_mode",
        "secure_rng_backend",
        "noise_generation",
        "noise_hardening",
        "clipping_unit",
        "clipping_norm",
        "noising_unit",
        "privacy_convention",
        "gradient_aggregation",
        "update_normalization",
        "optimizer_step_size",
        "adapter_registry_entry_sha256",
    ):
        collector.require(mechanism, key, "mechanism")

    accounting_unit = normalize_unit(mechanism.get("accounting_unit"))
    sampling_unit = normalize_unit(mechanism.get("sampling_unit"))
    clipping_unit = normalize_unit(mechanism.get("clipping_unit"))
    noising_unit = normalize_unit(mechanism.get("noising_unit"))
    accountant_adjacency = str(mechanism.get("accountant_adjacency") or "")
    result.accounting_unit = accounting_unit
    result.accountant_adjacency = accountant_adjacency

    if accounting_unit not in {"window", "event", "owner"}:
        collector.add(
            "ACCOUNTING_UNIT_INVALID",
            "invalid",
            "mechanism.accounting_unit",
            f"Unsupported accounting unit: {accounting_unit!r}.",
        )
    if accountant_adjacency not in VALID_ADJACENCIES:
        collector.add(
            "ACCOUNTANT_ADJACENCY_INVALID",
            "invalid",
            "mechanism.accountant_adjacency",
            f"Unsupported accountant adjacency: {accountant_adjacency!r}.",
        )
    if any(
        unit and unit != accounting_unit
        for unit in (sampling_unit, clipping_unit, noising_unit)
    ):
        collector.add(
            "MECHANISM_UNIT_MISMATCH",
            "invalid",
            "mechanism",
            "The built-in path requires sampling, clipping, noising, and accounting units to match.",
        )

    sampler_law = str(mechanism.get("sampler_law") or "")
    accountant_sampler = str(mechanism.get("accountant_sampler_law") or "")
    if sampler_law not in VALID_SAMPLERS or accountant_sampler not in VALID_SAMPLERS:
        collector.add(
            "SAMPLER_LAW_INVALID",
            "invalid",
            "mechanism.sampler_law",
            "Sampler laws must use registered exact identifiers.",
        )
    elif sampler_law != accountant_sampler:
        collector.add(
            "SAMPLER_ACCOUNTANT_MISMATCH",
            "invalid",
            "mechanism.accountant_sampler_law",
            "Executed and accountant sampler laws differ.",
        )

    sample_rate = _as_finite_float(mechanism.get("sample_rate"))
    if sample_rate is None or not 0 < sample_rate <= 1:
        collector.add(
            "SAMPLE_RATE_INVALID",
            "invalid",
            "mechanism.sample_rate",
            "Sample rate must be finite and lie in (0, 1].",
        )
    optimizer_step_size = _as_finite_float(
        mechanism.get("optimizer_step_size")
    )
    if optimizer_step_size is None or optimizer_step_size <= 0:
        collector.add(
            "OPTIMIZER_STEP_SIZE_INVALID",
            "invalid",
            "mechanism.optimizer_step_size",
            "The update step size must be a finite public positive constant.",
        )
    if _as_positive_int(mechanism.get("steps")) is None:
        collector.add(
            "STEPS_INVALID",
            "invalid",
            "mechanism.steps",
            "Steps must be a positive integer.",
        )
    noise = _as_finite_float(mechanism.get("noise_multiplier"))
    if noise is None or noise <= 0:
        collector.add(
            "NOISE_MULTIPLIER_INVALID",
            "invalid",
            "mechanism.noise_multiplier",
            "The registered Gaussian-mechanism path requires a finite, strictly positive noise multiplier.",
        )
    secure_mode = mechanism.get("secure_mode")
    if not isinstance(secure_mode, bool):
        collector.add(
            "SECURE_MODE_INVALID",
            "invalid",
            "mechanism.secure_mode",
            "secure_mode must be a JSON boolean.",
        )
    clip = _as_finite_float(mechanism.get("clipping_norm"))
    if clip is None or clip <= 0:
        collector.add(
            "CLIPPING_NORM_INVALID",
            "invalid",
            "mechanism.clipping_norm",
            "Clipping norm must be finite and strictly positive.",
        )
    privacy_convention = str(mechanism.get("privacy_convention") or "")
    if privacy_convention not in VALID_PRIVACY_CONVENTIONS:
        collector.add(
            "PRIVACY_CONVENTION_INVALID",
            "invalid",
            "mechanism.privacy_convention",
            "Privacy convention must identify approximate-DP or a registered RDP/PLD conversion.",
        )

    epsilon = _as_finite_float(collector.require(privacy, "epsilon", "privacy"))
    delta = _as_finite_float(collector.require(privacy, "delta", "privacy"))
    delta_convention = str(
        collector.require(privacy, "delta_convention", "privacy") or ""
    )
    if delta_convention not in VALID_DELTA_CONVENTIONS:
        collector.add(
            "DELTA_CONVENTION_INVALID",
            "invalid",
            "privacy.delta_convention",
            "Delta convention is missing or unsupported.",
        )
    if epsilon is None or epsilon < 0:
        collector.add(
            "EPSILON_INVALID",
            "invalid",
            "privacy.epsilon",
            "Epsilon must be finite and nonnegative.",
        )
    if delta is None or not 0 < delta < 1:
        collector.add(
            "DELTA_INVALID",
            "invalid",
            "privacy.delta",
            "Delta must be finite and lie strictly between zero and one.",
        )
    result.base_epsilon = epsilon
    result.base_delta = delta

    registry_entry = _validate_registered_accountant(
        mechanism, privacy, collector
    )

    manifest, mapping_hash, selected_hash, pipeline_hash = _validate_manifest(
        pipeline, audit_row, mechanism, registry_entry, collector
    )
    result.mapping_file_sha256 = mapping_hash
    result.selected_mapping_sha256 = selected_hash
    result.pipeline_sha256 = pipeline_hash

    mapping_evidence = _validate_actual_mapping(
        mapping_path,
        scenario,
        pipeline,
        manifest,
        audit_row,
        result,
        collector,
    )

    runtime_status = _validate_runtime(
        runtime,
        mechanism,
        privacy,
        manifest,
        selected_hash,
        pipeline_hash,
        collector,
    )
    result.runtime_status = runtime_status

    rng_assurance = str(privacy_report.get("rng_assurance") or "")
    result.rng_assurance = rng_assurance
    if rng_assurance not in VALID_RNG_ASSURANCE:
        collector.add(
            "RNG_ASSURANCE_INVALID",
            "unverified",
            "rng_assurance",
            "RNG assurance must use a registered value.",
        )
    elif rng_assurance == "unknown":
        collector.add(
            "RNG_ASSURANCE_UNKNOWN",
            "unverified",
            "rng_assurance",
            "Unknown DP-noise randomness does not support a release claim.",
        )
    elif rng_assurance == "research_prng":
        collector.add(
            "RNG_RESEARCH_ONLY",
            "research",
            "rng_assurance",
            "The accountant may be inspected, but this randomness evidence does not support copy-ready privacy wording for the concrete run.",
        )
    if rng_assurance == "release_grade_secure":
        collector.add(
            "RNG_ASSURANCE_OVERCLAIM",
            "invalid",
            "rng_assurance",
            "Universal release-grade security is not established; use the registered known-attack-hardened assurance class.",
        )
    if (
        rng_assurance
        in {"release_grade_secure", "known_attack_hardened_secure"}
        and secure_mode is not True
    ):
        collector.add(
            "RELEASE_RNG_SECURE_MODE_MISMATCH",
            "invalid",
            "mechanism.secure_mode",
            "The registered known-attack-hardened secure path requires secure_mode=true.",
        )
    if rng_assurance == "research_prng" and secure_mode is not False:
        collector.add(
            "RESEARCH_RNG_SECURE_MODE_MISMATCH",
            "invalid",
            "mechanism.secure_mode",
            "research_prng evidence must record secure_mode=false.",
        )

    noise_seed_policy = str(privacy_report.get("noise_seed_policy") or "")
    result.noise_seed_policy = noise_seed_policy
    if noise_seed_policy not in VALID_NOISE_SEED_POLICIES:
        collector.add(
            "NOISE_SEED_POLICY_INVALID",
            "unverified",
            "noise_seed_policy",
            "DP-noise seed policy must use a registered value.",
        )
    elif noise_seed_policy == "public_deterministic":
        collector.add(
            "PUBLIC_DETERMINISTIC_NOISE",
            "invalid",
            "noise_seed_policy",
            "A public deterministic DP-noise seed blocks a released-model DP claim.",
        )
    elif noise_seed_policy == "unknown":
        collector.add(
            "NOISE_SEED_POLICY_UNKNOWN",
            "unverified",
            "noise_seed_policy",
            "Unknown DP-noise seed handling does not support a release claim.",
        )
    if (
        rng_assurance
        in {"release_grade_secure", "known_attack_hardened_secure"}
        and noise_seed_policy
        not in {"secure_private_unrecorded", "secure_private_internal"}
    ):
        collector.add(
            "SECURE_RNG_SEED_POLICY_MISMATCH",
            "invalid",
            "noise_seed_policy",
            "The registered known-attack-hardened secure path requires a private noise-seed policy.",
        )

    public_certificate = privacy_report.get("public_certificate")
    if not isinstance(public_certificate, Mapping):
        collector.add(
            "PUBLIC_CERTIFICATE_POLICY_MISSING",
            "unverified",
            "public_certificate",
            "A public-metadata classification is required.",
        )
        public_certificate = {}
    metadata_classification = str(
        public_certificate.get("metadata_classification") or ""
    )
    result.metadata_classification = metadata_classification
    if metadata_classification not in VALID_METADATA_CLASSIFICATION:
        collector.add(
            "METADATA_CLASSIFICATION_INVALID",
            "unverified",
            "public_certificate.metadata_classification",
            "Public certificate metadata must be explicitly classified.",
        )

    contracts = privacy_report.get("claim_contracts")
    if not isinstance(contracts, Mapping):
        collector.add(
            "CLAIM_CONTRACTS_MISSING",
            "unverified",
            "claim_contracts",
            "Per-unit stability contracts are required.",
        )
        contracts = {}
    claim_contract = contracts.get(str(claim_unit))
    if not isinstance(claim_contract, Mapping):
        claim_contract = contracts.get(normalized_claim)
    if not isinstance(claim_contract, Mapping):
        collector.add(
            "CLAIM_CONTRACT_MISSING",
            "unverified",
            f"claim_contracts.{normalized_claim}",
            "No contract exists for the requested privacy unit.",
        )
        claim_contract = {}

    contract_unit = normalize_unit(claim_contract.get("claimed_unit"))
    if contract_unit != normalized_claim:
        collector.add(
            "CLAIM_CONTRACT_UNIT_MISMATCH",
            "invalid",
            f"claim_contracts.{normalized_claim}.claimed_unit",
            "The selected claim contract names a different unit.",
        )
    raw_adjacency = str(claim_contract.get("raw_adjacency") or "")
    result.raw_adjacency = raw_adjacency
    if raw_adjacency not in VALID_ADJACENCIES:
        collector.add(
            "RAW_ADJACENCY_INVALID",
            "invalid",
            f"claim_contracts.{normalized_claim}.raw_adjacency",
            f"Unsupported raw adjacency: {raw_adjacency!r}.",
        )
    if str(claim_contract.get("stability_status") or "") != "validated":
        collector.add(
            "STABILITY_STATUS_UNVERIFIED",
            "unverified",
            f"claim_contracts.{normalized_claim}.stability_status",
            "Transformation stability is not validated.",
        )
    if str(claim_contract.get("support_status") or "") != "complete":
        collector.add(
            "INFLUENCE_SUPPORT_INCOMPLETE",
            "unverified",
            f"claim_contracts.{normalized_claim}.support_status",
            "Complete influence support is required.",
        )
    schedule_mode = str(claim_contract.get("schedule_mode") or "")
    schedule_status = str(claim_contract.get("schedule_evidence_status") or "")
    if schedule_mode != "fixed_mapping":
        collector.add(
            "SCHEDULE_MODE_UNREGISTERED",
            "unverified",
            f"claim_contracts.{normalized_claim}.schedule_mode",
            "The v2.0 built-in path initially validates only fixed_mapping.",
        )
    if schedule_status != "validated":
        collector.add(
            "SCHEDULE_EVIDENCE_UNVERIFIED",
            "unverified",
            f"claim_contracts.{normalized_claim}.schedule_evidence_status",
            "Schedule evidence is not validated.",
        )

    conversion_method = str(
        collector.require(
            claim_contract,
            "conversion_method",
            f"claim_contracts.{normalized_claim}",
            category="unverified",
        )
        or ""
    )
    if conversion_method not in {"identity", "builtin_group"}:
        collector.add(
            "EXTERNAL_CONVERSION_UNVERIFIED",
            "unverified",
            f"claim_contracts.{normalized_claim}.conversion_method",
            "Only identity and the built-in black-box group conversion are registered.",
        )

    stability_method, stability_bound = _compute_stability_bound(
        normalized_claim,
        raw_adjacency,
        accountant_adjacency,
        claim_contract,
        manifest,
        mapping_evidence,
        accounting_unit,
        collector,
    )
    result.stability_method = stability_method
    result.stability_bound = stability_bound
    if stability_bound is not None:
        if stability_bound > 1 and conversion_method == "identity":
            collector.add(
                "IDENTITY_CONVERSION_WITH_K_GT_ONE",
                "invalid",
                f"claim_contracts.{normalized_claim}.conversion_method",
                "Identity conversion cannot authorize a stability bound greater than one.",
            )
        if stability_bound == 1 and conversion_method == "builtin_group":
            collector.add(
                "UNNECESSARY_GROUP_CONVERSION",
                "warning",
                f"claim_contracts.{normalized_claim}.conversion_method",
                "K=1 is reported as DIRECT even when builtin_group was requested.",
            )

    if collector.has("invalid"):
        result.unit_path = "UNDERSPECIFIED"
        result.release_status = "BLOCKED_INVALID"
        result.assurance_status = "DIAGNOSTIC_ONLY"
    elif collector.has("unverified"):
        result.unit_path = "UNDERSPECIFIED"
        result.release_status = "BLOCKED_UNVERIFIED"
        result.assurance_status = (
            "EXTERNAL_UNVERIFIED"
            if any(
                issue.code.startswith("EXTERNAL_")
                or issue.code.startswith("SEPARATELY_DP_")
                for issue in result.issues
            )
            else "DIAGNOSTIC_ONLY"
        )
    else:
        assert epsilon is not None
        assert delta is not None
        assert stability_bound is not None
        if stability_bound == 1:
            result.unit_path = "DIRECT"
            result.reported_epsilon = epsilon
            result.reported_delta = delta
            result.reported_delta_log10 = math.log10(delta)
            result.conversion_vacuous = False
        else:
            result.unit_path = (
                "GROUP" if normalized_claim == "owner" else "CONVERT"
            )
            try:
                (
                    result.reported_epsilon,
                    result.reported_delta,
                    result.reported_delta_log10,
                    result.conversion_vacuous,
                ) = group_privacy_conversion(epsilon, delta, stability_bound)
            except ValueError as exc:
                collector.add(
                    "GROUP_CONVERSION_FAILED",
                    "invalid",
                    "privacy",
                    str(exc),
                )
                result.unit_path = "UNDERSPECIFIED"
                result.release_status = "BLOCKED_INVALID"
                result.assurance_status = "DIAGNOSTIC_ONLY"

        if result.unit_path != "UNDERSPECIFIED":
            if result.conversion_vacuous:
                result.release_status = "BLOCKED_VACUOUS"
                result.assurance_status = "DIAGNOSTIC_ONLY"
            elif collector.has("research"):
                result.release_status = "RESEARCH_ONLY"
                result.assurance_status = "DIAGNOSTIC_ONLY"
            else:
                result.release_status = "ALLOWED"
                result.assurance_status = "EVIDENCE_VALIDATED"

    if result.release_status == "ALLOWED":
        qualifier = "Conditional on a complete and faithful mapping export for the bound pipeline, "
        result.supported_statement = (
            f"{qualifier}the released mechanism is "
            f"({result.reported_epsilon:.6g}, {result.reported_delta:.6g})-DP "
            f"for one {result.claimed_privacy_unit} under "
            f"{result.raw_adjacency} adjacency."
        )
        if result.unit_path == "DIRECT":
            result.do_not_say = (
                "Do not generalize this number to a different unit, adjacency, "
                "sampler, schedule, or pipeline."
            )
        else:
            result.do_not_say = (
                "Do not report the base accountant number for the stronger raw "
                "unit; report the validated converted number and K."
            )
        result.next_step = "Publish only the redacted public certificate."
    else:
        result.supported_statement = ""
        if result.release_status == "BLOCKED_VACUOUS":
            result.do_not_say = (
                "Do not publish a vacuous converted DP guarantee as meaningful privacy."
            )
            result.next_step = (
                "Use a tighter registered accountant, a public contribution bound, "
                "a native raw-unit mechanism, or narrower wording."
            )
        elif result.release_status == "RESEARCH_ONLY":
            result.do_not_say = (
                "Do not describe the research-PRNG run as an evidence-validated private model."
            )
            result.next_step = (
                "Rerun from scratch with a registered private CSPRNG mechanism "
                "whose exact implementation and runtime evidence are bound."
            )
        else:
            result.do_not_say = (
                "Do not emit any positive privacy sentence for the requested unit."
            )
            result.next_step = (
                "Resolve every blocking issue and regenerate the report and certificate."
            )

    return result
