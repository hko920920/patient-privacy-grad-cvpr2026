"""Public/private artifact boundary for DP training runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PUBLIC_CERTIFICATE_SCHEMA = "unitdp.public_release_certificate.v2"
PRIVATE_MANIFEST_SCHEMA = "unitdp.private_execution_manifest.v2"
# Approval is intentionally empty until a secure-RNG V2 executor receives a
# separate release audit.  A matching implementation id alone is insufficient.
RELEASE_ELIGIBLE_IMPLEMENTATIONS: set[str] = set()

FORBIDDEN_PUBLIC_KEYS = {
    "seed",
    "sampler_trace_sha256",
    "source_mapping_sha256",
    "selected_mapping_sha256",
    "source_mapping_stats",
    "selected_mapping_stats",
    "selection_diagnostics",
    "training_result",
    "contract_sha256",
    "execution_binding_sha256",
    "config_sha256",
    "release_binding_sha256",
    "selected_owners",
    "selected_windows",
    "sampled_owner_count_mean",
    "sampled_owner_count_std",
    "sampled_owner_count_min",
    "sampled_owner_count_max",
    "actual_owner_sample_rate_mean",
    "empty_steps",
    "train_accuracy",
}


def _sha256_payload(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _as_dict(certificate: Any) -> dict[str, Any]:
    if isinstance(certificate, dict):
        return certificate
    if hasattr(certificate, "to_dict"):
        return certificate.to_dict()
    raise TypeError("certificate must be a mapping or expose to_dict()")


def _implementation_checks(raw: dict[str, Any]) -> dict[str, bool]:
    checks = raw.get("implementation_checks", {})
    public_names = [
        "owner_mechanism_units_aligned",
        "selected_windows_single_attribution",
        "accountant_sample_rate_matches_owner_sampling",
        "bernoulli_owner_sampling_recorded",
        "one_clipped_vector_per_sampled_owner_recorded",
        "internal_owner_sampler_recorded",
        "preprocessing_contract_bound",
    ]
    return {name: bool(checks.get(name, False)) for name in public_names}


def build_public_release_certificate(certificate: Any) -> dict[str, Any]:
    """Build a redacted public certificate that contains no execution secrets."""

    raw = _as_dict(certificate)
    result = raw.get("training_result", {})
    config = result.get("config", {})
    assumptions = raw.get("privacy_assumptions", {})
    release = raw.get("release_binding", {})
    route_implementation_id = str(
        assumptions.get(
            "route_implementation_id",
            result.get("route_implementation_id", "unspecified"),
        )
    )
    secure_rng = bool(assumptions.get("secure_rng", result.get("secure_rng", False)))
    adjacency = str(assumptions.get("adjacency", "unspecified"))
    sampling_scheme = str(assumptions.get("sampling_scheme", "unspecified"))
    checks = _implementation_checks(raw)

    reason_codes: list[str] = []
    if route_implementation_id not in RELEASE_ELIGIBLE_IMPLEMENTATIONS:
        reason_codes.append("implementation_not_release_approved")
    if not secure_rng:
        reason_codes.append("noncryptographic_research_rng")
    if adjacency != "add_remove_one_owner":
        reason_codes.append("unsupported_release_adjacency")
    if sampling_scheme != "independent_bernoulli_owner":
        reason_codes.append("unsupported_release_sampler")
    if raw.get("route_status") != "compiled":
        reason_codes.append("route_not_compiled")
    if not all(checks.values()):
        reason_codes.append("required_implementation_check_failed")

    release_eligible = not reason_codes
    accountant = raw.get("accountant", {})
    preprocessing = assumptions.get("preprocessing_binding")
    if not isinstance(preprocessing, dict):
        preprocessing = None

    public: dict[str, Any] = {
        "schema_version": PUBLIC_CERTIFICATE_SCHEMA,
        "visibility": "public",
        "release_status": (
            "privacy_release_eligible"
            if release_eligible
            else "research_non_release"
        ),
        "privacy_claim": {
            "status": (
                "release_eligible"
                if release_eligible
                else "accountant_output_only_not_a_release_claim"
            ),
            "reason_codes": reason_codes,
        },
        "privacy_unit": str(raw.get("privacy_unit", "unspecified")),
        "mechanism": {
            "adjacency": adjacency,
            "sampling_unit": raw.get("mechanism_units", {}).get(
                "sampling_unit", "unspecified"
            ),
            "clipping_unit": raw.get("mechanism_units", {}).get(
                "clipping_unit", "unspecified"
            ),
            "noising_unit": raw.get("mechanism_units", {}).get(
                "noising_unit", "unspecified"
            ),
            "accounting_unit": raw.get("mechanism_units", {}).get(
                "accounting_unit", "unspecified"
            ),
            "sampling_scheme": sampling_scheme,
            "contribution_bound": result.get(
                "per_step_contribution_bound", "unspecified"
            ),
        },
        "accountant_output": {
            "backend": accountant.get("backend", "unspecified"),
            "epsilon": accountant.get("epsilon"),
            "delta": accountant.get("delta"),
            "noise_multiplier": accountant.get("noise_multiplier"),
            "sample_rate": accountant.get("accountant_sample_rate"),
            "steps": accountant.get("steps"),
            "max_grad_norm": config.get("max_grad_norm"),
        },
        "public_data_bindings": {
            "preprocessing": preprocessing or "not_recorded",
        },
        "implementation": {
            "route_implementation_id": route_implementation_id,
            "loader_binding": assumptions.get("loader_binding", "unspecified"),
            "checks": checks,
        },
        "rng": {
            "secure_rng": secure_rng,
            "backend": assumptions.get("rng_backend", "unspecified"),
            "security_mode": assumptions.get(
                "rng_security_mode", "unspecified"
            ),
            "caveat": assumptions.get("rng_caveat", "unspecified"),
            "random_coins_disclosed": False,
        },
        "released_artifact_bindings": {
            "model_artifact_sha256": release.get(
                "model_artifact_sha256", "not_recorded"
            ),
            "training_code_sha256": release.get(
                "training_code_sha256", "not_recorded"
            ),
            "library_versions": release.get("library_versions", {}),
        },
    }
    public["public_certificate_sha256"] = _sha256_payload(public)
    assert_public_certificate_redacted(public)
    return public


def assert_public_certificate_redacted(public: dict[str, Any]) -> None:
    """Reject a public artifact containing a known private execution field."""

    violations: list[str] = []

    def walk(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else key
                if key in FORBIDDEN_PUBLIC_KEYS:
                    violations.append(path)
                walk(child, path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{prefix}[{index}]")

    walk(public)
    if violations:
        raise ValueError(
            "Public certificate contains private execution fields: "
            + ", ".join(sorted(violations))
        )


def build_private_execution_manifest(certificate: Any) -> dict[str, Any]:
    """Wrap the complete execution record with an explicit private label."""

    full_record = _as_dict(certificate)
    private = {
        "schema_version": PRIVATE_MANIFEST_SCHEMA,
        "visibility": "private",
        "handling": "do_not_publish_contains_execution_secrets_and_private_statistics",
        "full_execution_record": full_record,
    }
    private["private_manifest_sha256"] = _sha256_payload(private)
    return private


def write_public_release_certificate(certificate: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    public = build_public_release_certificate(certificate)
    if target.suffix.lower() == ".json":
        target.write_text(
            json.dumps(public, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return

    lines = [
        "# UnitDP Public Release Certificate",
        "",
        f"- schema: `{public['schema_version']}`",
        f"- release status: `{public['release_status']}`",
        f"- privacy claim status: `{public['privacy_claim']['status']}`",
        f"- privacy unit: `{public['privacy_unit']}`",
        f"- epsilon: `{public['accountant_output']['epsilon']}`",
        f"- delta: `{public['accountant_output']['delta']}`",
        f"- sample rate: `{public['accountant_output']['sample_rate']}`",
        f"- steps: `{public['accountant_output']['steps']}`",
        f"- secure RNG: `{public['rng']['secure_rng']}`",
        f"- random coins disclosed: `{public['rng']['random_coins_disclosed']}`",
        "",
        "## Reason Codes",
        "",
    ]
    for reason in public["privacy_claim"]["reason_codes"]:
        lines.append(f"- `{reason}`")
    lines.extend(["", "## Implementation Checks", ""])
    for key, value in public["implementation"]["checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Public Bindings",
            "",
            f"- model artifact sha256: `{public['released_artifact_bindings']['model_artifact_sha256']}`",
            f"- training code sha256: `{public['released_artifact_bindings']['training_code_sha256']}`",
            f"- public certificate sha256: `{public['public_certificate_sha256']}`",
        ]
    )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_private_execution_manifest(certificate: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    private = build_private_execution_manifest(certificate)
    target.write_text(
        json.dumps(private, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
