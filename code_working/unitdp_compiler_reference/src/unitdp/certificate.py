"""Training certificate emitter for unit-aligned runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from unitdp.accountant import group_privacy_conversion
from unitdp.compiler import CompiledRoute
from unitdp.contract import OWNER_UNITS
from unitdp.release_artifacts import write_public_release_certificate


@dataclass(frozen=True)
class TrainingCertificate:
    privacy_unit: str
    route_status: str
    source_mapping_sha256: str
    selected_mapping_sha256: str
    contract_sha256: str
    source_mapping_stats: dict[str, Any]
    selected_mapping_stats: dict[str, Any]
    selection_diagnostics: dict[str, Any]
    mechanism_units: dict[str, str]
    accountant: dict[str, Any]
    privacy_assumptions: dict[str, Any]
    implementation_checks: dict[str, Any]
    training_result: dict[str, Any]
    group_fallback: dict[str, Any]
    release_binding: dict[str, Any]
    route_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_payload(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_certificate(route: CompiledRoute, training_result: dict[str, Any]) -> TrainingCertificate:
    contract = route.contract
    epsilon = float(training_result["epsilon"])
    delta = float(training_result["delta"])
    source_mapping_sha256 = route.source_mapping.file_hash()
    selected_mapping_sha256 = route.selected_mapping.canonical_hash()
    source_stats = route.source_mapping.stats()
    selected_stats = route.selected_mapping.stats()
    owner_kappa = max(1, int(selected_stats["owner_kappa"]))
    group_fallback = group_privacy_conversion(epsilon, delta, owner_kappa)
    mechanism_units = {
        "sampling_unit": str(contract.mechanism.get("sampling_unit", "unspecified")),
        "clipping_unit": str(contract.mechanism.get("clipping_unit", "unspecified")),
        "noising_unit": str(contract.mechanism.get("noising_unit", "unspecified")),
        "accounting_unit": str(contract.accountant.get("unit", "unspecified")),
    }
    selected_owners = int(training_result.get("selected_owners") or selected_stats["num_owners"])
    owner_batch_size = int(training_result.get("config", {}).get("owner_batch_size", selected_owners))
    expected_sample_rate = min(1.0, owner_batch_size / max(1, selected_owners))
    accountant_sample_rate = training_result.get(
        "accountant_sample_rate",
        training_result.get("owner_sample_rate"),
    )
    public_schedule_sha256 = str(training_result.get("public_schedule_sha256", ""))
    sampler_trace_sha256 = str(training_result.get("sampler_trace_sha256", ""))
    route_implementation_id = str(training_result.get("route_implementation_id", "legacy_unspecified"))
    loader_binding = str(training_result.get("loader_binding", "legacy_unspecified"))
    rng_backend = str(training_result.get("rng_backend", "legacy_unspecified"))
    rng_security_mode = str(training_result.get("rng_security_mode", "legacy_unspecified"))
    execution_binding_sha256 = _sha256_payload(
        {
            "source_mapping_sha256": source_mapping_sha256,
            "selected_mapping_sha256": selected_mapping_sha256,
            "contract_sha256": route.contract_hash,
            "route_implementation_id": route_implementation_id,
            "loader_binding": loader_binding,
            "public_schedule_sha256": public_schedule_sha256,
            "sampler_trace_sha256": sampler_trace_sha256,
            "accountant_sample_rate": accountant_sample_rate,
            "steps": training_result.get("steps"),
            "noise_multiplier": training_result.get("noise_multiplier"),
            "epsilon": epsilon,
            "delta": delta,
        }
    )
    model_artifact_sha256 = str(training_result.get("model_artifact_sha256", "not_recorded"))
    training_code_sha256 = str(training_result.get("training_code_sha256", "not_recorded"))
    library_versions = training_result.get("library_versions", {})
    config_sha256 = _sha256_payload(training_result.get("config", {}))
    release_binding_sha256 = _sha256_payload(
        {
            "execution_binding_sha256": execution_binding_sha256,
            "model_artifact_sha256": model_artifact_sha256,
            "training_code_sha256": training_code_sha256,
            "config_sha256": config_sha256,
            "library_versions": library_versions,
        }
    )
    sample_rate_matches = (
        accountant_sample_rate is not None
        and abs(float(accountant_sample_rate) - expected_sample_rate) <= 1e-12
    )
    owner_claim = contract.privacy_unit in OWNER_UNITS
    owner_units_aligned = all(
        mechanism_units[key] in OWNER_UNITS
        for key in ["sampling_unit", "clipping_unit", "noising_unit", "accounting_unit"]
    )
    preprocessing_binding = contract.data.get("preprocessing")
    privacy_assumptions = {
        "adjacency": contract.adjacency,
        "schedule_mode": contract.schedule_mode,
        "selected_windows_single_attribution": int(selected_stats["multi_owner_windows"]) == 0,
        "contribution_policy_type": contract.owner_policy.get("type", "unspecified"),
        "contribution_policy_label_source": contract.owner_policy.get("label_source", "unused"),
        "sampling_scheme": training_result.get("sampling_scheme", "legacy_unspecified"),
        "accountant_backend": training_result.get(
            "accountant_backend",
            contract.accountant.get("backend", "rdp"),
        ),
        "route_implementation_id": route_implementation_id,
        "loader_binding": loader_binding,
        "public_schedule_sha256": public_schedule_sha256,
        "sampler_trace_sha256": sampler_trace_sha256,
        "execution_binding_sha256": execution_binding_sha256,
        "secure_rng": bool(training_result.get("secure_rng", False)),
        "rng_backend": rng_backend,
        "rng_security_mode": rng_security_mode,
        "rng_caveat": training_result.get(
            "rng_caveat",
            "No secure-RNG field in this legacy certificate; rerun with current code for full trace.",
        ),
        "preprocessing_binding": preprocessing_binding or "not_recorded",
    }
    implementation_checks = {
        "owner_claim": owner_claim,
        "owner_mechanism_units_aligned": (not owner_claim) or owner_units_aligned,
        "selected_windows_single_attribution": int(selected_stats["multi_owner_windows"]) == 0,
        "accountant_sample_rate_matches_owner_sampling": sample_rate_matches,
        "bernoulli_owner_sampling_recorded": (
            training_result.get("sampling_scheme") == "independent_bernoulli_owner"
        ),
        "one_clipped_vector_per_sampled_owner_recorded": (
            training_result.get("per_step_contribution_bound")
            == "one_clipped_vector_per_sampled_owner"
        ),
        "internal_owner_sampler_recorded": (
            loader_binding == "internal_owner_sampler_no_external_dataloader"
        ),
        "public_schedule_digest_recorded": bool(public_schedule_sha256),
        "sampler_trace_digest_recorded": bool(sampler_trace_sha256),
        "execution_binding_digest_recorded": bool(execution_binding_sha256),
        "rng_backend_recorded": bool(rng_backend and rng_backend != "legacy_unspecified"),
        "research_rng_caveat_recorded": bool(training_result.get("rng_caveat")),
        "production_secure_rng_used": bool(training_result.get("secure_rng", False)),
        "preprocessing_contract_bound": preprocessing_binding is not None,
    }
    return TrainingCertificate(
        privacy_unit=contract.privacy_unit,
        route_status=route.route_status,
        source_mapping_sha256=source_mapping_sha256,
        selected_mapping_sha256=selected_mapping_sha256,
        contract_sha256=route.contract_hash,
        source_mapping_stats=source_stats,
        selected_mapping_stats=selected_stats,
        selection_diagnostics=route.source_mapping.selection_diagnostics(route.selected_mapping),
        mechanism_units=mechanism_units,
        accountant={
            "backend": contract.accountant.get("backend", "rdp"),
            "unit": contract.accountant.get("unit", "owner"),
            "epsilon": epsilon,
            "delta": delta,
            "noise_multiplier": training_result.get("noise_multiplier"),
            "sample_rate": training_result.get("owner_sample_rate"),
            "accountant_sample_rate": accountant_sample_rate,
            "expected_sample_rate_from_owner_batch": expected_sample_rate,
            "steps": training_result.get("steps"),
            "public_schedule_sha256": public_schedule_sha256,
        },
        privacy_assumptions=privacy_assumptions,
        implementation_checks=implementation_checks,
        training_result=training_result,
        group_fallback=group_fallback,
        release_binding={
            "binding_status": (
                "recorded"
                if model_artifact_sha256 != "not_recorded"
                and training_code_sha256 != "not_recorded"
                else "not_recorded"
            ),
            "model_artifact_sha256": model_artifact_sha256,
            "training_code_sha256": training_code_sha256,
            "config_sha256": config_sha256,
            "library_versions": library_versions,
            "release_binding_sha256": release_binding_sha256,
        },
        route_notes=list(route.route_notes),
    )


def write_certificate(certificate: TrainingCertificate, path: str | Path) -> None:
    """Write the redacted public view.

    The complete TrainingCertificate object is an internal execution record.
    Use release_artifacts.write_private_execution_manifest only for a
    restricted internal destination.
    """

    write_public_release_certificate(certificate, path)
