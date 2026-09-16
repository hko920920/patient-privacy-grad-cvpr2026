"""Audit the boundary between registered research fixtures and release use.

The two strict routes are intentionally execution-bound to exact public
benchmark profiles.  This audit proves that the current research paths still
compile, that release-shaped requests fail closed, and that every current
public execution remains explicitly non-release.  It does not authorize a
privacy release or treat the prototype secure-randomness adapter as audited.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unitdp.benchmark_registry_srswor_v3 import (  # noqa: E402
    SRSWOR_BENCHMARK_PROFILES_V3,
)
from unitdp.benchmark_registry_v2 import (  # noqa: E402
    BENCHMARK_PROFILES_V2,
)
from unitdp.compiler_srswor_v3 import (  # noqa: E402
    SrsworContractV3Error,
    compile_owner_srswor_contract_v3,
    parse_owner_srswor_contract_v3,
)
from unitdp.compiler_v2 import (  # noqa: E402
    ExecutorNotReadyError,
    compile_owner_poisson_contract_v2,
    parse_owner_poisson_contract_v2,
)
from unitdp.release_artifacts import (  # noqa: E402
    FORBIDDEN_PUBLIC_KEYS,
    RELEASE_ELIGIBLE_IMPLEMENTATIONS,
    assert_public_certificate_redacted,
    build_public_release_certificate,
)
from unitdp.rng import make_random_source  # noqa: E402
from unitdp.source_bundle_srswor_v3 import (  # noqa: E402
    execution_source_bundle_sha256_srswor_v3,
)
from unitdp.source_bundle_v2 import (  # noqa: E402
    execution_source_bundle_sha256_v2,
)


DEFAULT_OUTPUT = (
    ROOT
    / "reports"
    / "v32_release_domain_gap_20260724"
    / "public_release_domain_gap_v32.json"
)
DATASETS = ("uci", "wisdm", "sepsis")
ROUTE_P_EXECUTIONS = (
    ROOT / "reports" / "v2_registered_5seed_v32_20260724"
)
ROUTE_F_EXECUTIONS = (
    ROOT / "reports" / "v3_srswor_registered_5seed_v32_20260724"
)
COMPILER_P = ROOT / "src" / "unitdp" / "compiler_v2.py"
COMPILER_F = ROOT / "src" / "unitdp" / "compiler_srswor_v3.py"
EXECUTOR_P = ROOT / "src" / "unitdp" / "owner_poisson_v2.py"
EXECUTOR_F = ROOT / "src" / "unitdp" / "owner_srswor_v3.py"
RELEASE_ARTIFACTS = ROOT / "src" / "unitdp" / "release_artifacts.py"
RNG_SOURCE = ROOT / "src" / "unitdp" / "rng.py"
LOCAL_IDENTITY_TOKEN = "SO" + "GANG"
WINDOWS_DRIVE_PREFIX = "C:" + chr(92)

PUBLIC_FIXTURES = {
    "uci": {
        "mapping": (
            ROOT
            / "reports"
            / "uci_public_preprocessing_smoke_20260724"
            / "uci_train_mapping.csv"
        ),
        "preprocessor": (
            ROOT
            / "configs"
            / "preprocessing"
            / "uci_har_published_train_standard_scaler_v1.json"
        ),
    },
    "wisdm": {
        "mapping": (
            ROOT
            / "reports"
            / "wisdm_public_protocol_smoke_20260724"
            / "wisdm_train_mapping.csv"
        ),
        "preprocessor": (
            ROOT
            / "configs"
            / "preprocessing"
            / "wisdm_v1_1_train_owners_1_25_stats24_standard_scaler_v1.json"
        ),
    },
    "sepsis": {
        "mapping": (
            ROOT
            / "reports"
            / "sepsis_public_protocol_smoke_20260724"
            / "sepsis_train_mapping.csv"
        ),
        "preprocessor": (
            ROOT
            / "configs"
            / "preprocessing"
            / "physionet2019_setA_first400_basic12x6_public_scaler_v1.json"
        ),
    },
}


class ReleaseDomainAuditError(RuntimeError):
    """Raised when the frozen release-domain boundary changes."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseDomainAuditError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_pairs,
    )


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ReleaseDomainAuditError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.load(
        path.read_text(encoding="utf-8"),
        Loader=_UniqueKeyLoader,
    )
    if not isinstance(value, dict):
        raise ReleaseDomainAuditError(f"Expected YAML object: {path}")
    return value


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _config_path(route: str, dataset: str) -> Path:
    if route == "P":
        return (
            ROOT
            / "configs"
            / "v2"
            / f"{dataset}_owner_poisson_v2.yaml"
        )
    return (
        ROOT
        / "configs"
        / "v3"
        / f"{dataset}_owner_srswor_v3.yaml"
    )


def _compile_research_contracts() -> dict[str, Any]:
    compiled_counts = {"P": 0, "F": 0}
    profile_ids: dict[str, set[str]] = {"P": set(), "F": set()}
    for dataset in DATASETS:
        fixture = PUBLIC_FIXTURES[dataset]

        raw_p = load_yaml(_config_path("P", dataset))
        contract_p = parse_owner_poisson_contract_v2(raw_p)
        compiled_p = compile_owner_poisson_contract_v2(
            contract_p,
            mapping_path=fixture["mapping"],
            preprocessing_artifact_path=fixture["preprocessor"],
            require_executable=True,
        )
        if not compiled_p.execution_ready:
            raise ReleaseDomainAuditError(
                f"Registered Route P research contract is not ready: {dataset}"
            )
        if (
            contract_p.preprocessing_binding["scope"] != "public_fixed"
            or contract_p.preprocessing_binding["fit_on_protected_data"]
            is not False
        ):
            raise ReleaseDomainAuditError(
                f"Route P preprocessing boundary changed: {dataset}"
            )
        compiled_counts["P"] += 1
        profile_ids["P"].add(contract_p.benchmark_profile_id)

        raw_f = load_yaml(_config_path("F", dataset))
        contract_f = parse_owner_srswor_contract_v3(raw_f)
        compiled_f = compile_owner_srswor_contract_v3(
            contract_f,
            mapping_path=fixture["mapping"],
            preprocessing_artifact_path=fixture["preprocessor"],
            require_executable=True,
        )
        if not compiled_f.execution_ready:
            raise ReleaseDomainAuditError(
                f"Registered Route F research contract is not ready: {dataset}"
            )
        if (
            contract_f.preprocessing_binding["scope"] != "public_fixed"
            or contract_f.preprocessing_binding["fit_on_protected_data"]
            is not False
        ):
            raise ReleaseDomainAuditError(
                f"Route F preprocessing boundary changed: {dataset}"
            )
        compiled_counts["F"] += 1
        profile_ids["F"].add(contract_f.benchmark_profile_id)

    if len(BENCHMARK_PROFILES_V2) != 3:
        raise ReleaseDomainAuditError("Route P profile count changed")
    if len(SRSWOR_BENCHMARK_PROFILES_V3) != 3:
        raise ReleaseDomainAuditError("Route F profile count changed")
    if len(profile_ids["P"]) != 3 or len(profile_ids["F"]) != 3:
        raise ReleaseDomainAuditError(
            "Active contracts do not bind six distinct registered profiles"
        )
    return {
        "active_contracts": 6,
        "registered_profiles": {"P": 3, "F": 3},
        "research_compiles": compiled_counts,
        "exact_public_preprocessing_only": True,
        "protected_data_preprocessing_admitted": False,
    }


def _audit_release_requests() -> dict[str, Any]:
    p_parse_accepts = 0
    p_execution_denials = 0
    f_contract_denials = 0
    for dataset in DATASETS:
        fixture = PUBLIC_FIXTURES[dataset]
        raw_p = copy.deepcopy(load_yaml(_config_path("P", dataset)))
        raw_p["execution"]["profile"] = "privacy_release"
        raw_p["execution"]["rng_backend"] = "system_csprng"
        contract_p = parse_owner_poisson_contract_v2(raw_p)
        p_parse_accepts += 1
        try:
            compile_owner_poisson_contract_v2(
                contract_p,
                mapping_path=fixture["mapping"],
                preprocessing_artifact_path=fixture["preprocessor"],
                require_executable=True,
            )
        except ExecutorNotReadyError as exc:
            if str(exc) != (
                "V2 contract is valid, but no executor is approved for the "
                "requested execution profile"
            ):
                raise ReleaseDomainAuditError(
                    f"Route P denial message changed: {exc}"
                ) from exc
            p_execution_denials += 1
        else:
            raise ReleaseDomainAuditError(
                f"Route P release request unexpectedly compiled: {dataset}"
            )

        raw_f = copy.deepcopy(load_yaml(_config_path("F", dataset)))
        raw_f["execution"]["profile"] = "privacy_release"
        raw_f["execution"]["rng_backend"] = "system_csprng"
        try:
            parse_owner_srswor_contract_v3(raw_f)
        except SrsworContractV3Error as exc:
            if "execution.profile must equal the registered value" not in str(
                exc
            ):
                raise ReleaseDomainAuditError(
                    f"Route F denial reason changed: {exc}"
                ) from exc
            f_contract_denials += 1
        else:
            raise ReleaseDomainAuditError(
                f"Route F release request unexpectedly parsed: {dataset}"
            )

    return {
        "route_p": {
            "schema_parse_accepts": p_parse_accepts,
            "executable_requests_denied": p_execution_denials,
            "denial_stage": "executable_backend_approval",
            "release_executor_available": False,
        },
        "route_f": {
            "schema_parse_accepts": 0,
            "contract_requests_denied": f_contract_denials,
            "denial_stage": "exact_contract_binding",
            "release_executor_available": False,
        },
        "all_release_requests_fail_closed": (
            p_parse_accepts == 3
            and p_execution_denials == 3
            and f_contract_denials == 3
        ),
    }


def _release_artifact_boundary() -> dict[str, Any]:
    if RELEASE_ELIGIBLE_IMPLEMENTATIONS:
        raise ReleaseDomainAuditError(
            "Release implementation approval is no longer empty"
        )
    checks = {
        "owner_mechanism_units_aligned": True,
        "selected_windows_single_attribution": True,
        "accountant_sample_rate_matches_owner_sampling": True,
        "bernoulli_owner_sampling_recorded": True,
        "one_clipped_vector_per_sampled_owner_recorded": True,
        "internal_owner_sampler_recorded": True,
        "preprocessing_contract_bound": True,
    }
    synthetic = {
        "route_status": "compiled",
        "privacy_unit": "owner",
        "mechanism_units": {
            "sampling_unit": "owner",
            "clipping_unit": "owner",
            "noising_unit": "owner_sum",
            "accounting_unit": "owner",
        },
        "implementation_checks": checks,
        "privacy_assumptions": {
            "route_implementation_id": (
                "unitdp.owa_dpsgd.train_owner_poisson_fixed_v2"
            ),
            "secure_rng": True,
            "rng_backend": "system_csprng",
            "rng_security_mode": "system_csprng",
            "rng_caveat": "synthetic metadata-only gate exercise",
            "adjacency": "add_remove_one_owner",
            "sampling_scheme": "independent_bernoulli_owner",
            "loader_binding": "registered_route",
            "preprocessing_binding": {
                "scope": "public_fixed",
                "fit_on_protected_data": False,
            },
        },
        "training_result": {
            "route_implementation_id": (
                "unitdp.owa_dpsgd.train_owner_poisson_fixed_v2"
            ),
            "per_step_contribution_bound": "one_clipped_owner_vector",
            "config": {"max_grad_norm": 1.0},
        },
        "accountant": {
            "backend": "dual_rdp",
            "epsilon": 8.0,
            "delta": 1e-5,
            "noise_multiplier": 1.0,
            "accountant_sample_rate": 0.25,
            "steps": 1,
        },
        "release_binding": {
            "model_artifact_sha256": "1" * 64,
            "training_code_sha256": "2" * 64,
            "library_versions": {},
        },
    }
    public = build_public_release_certificate(synthetic)
    assert_public_certificate_redacted(public)
    expected_reasons = ["implementation_not_release_approved"]
    if public["release_status"] != "research_non_release":
        raise ReleaseDomainAuditError(
            "Secure metadata unexpectedly opened release status"
        )
    if public["privacy_claim"]["reason_codes"] != expected_reasons:
        raise ReleaseDomainAuditError(
            "Secure metadata denial reasons changed"
        )
    return {
        "approved_release_implementations": 0,
        "secure_metadata_only_status": "research_non_release",
        "secure_metadata_only_reason_codes": expected_reasons,
        "public_redaction_gate_passed": True,
        "blocked_sensitive_field_classes": len(FORBIDDEN_PUBLIC_KEYS),
    }


def _walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_walk_keys(child))
    return keys


def _audit_public_executions() -> dict[str, Any]:
    current_bundles = {
        "P": execution_source_bundle_sha256_v2(),
        "F": execution_source_bundle_sha256_srswor_v3(),
    }
    roots = {
        "P": (ROUTE_P_EXECUTIONS, "public_execution_v2.json"),
        "F": (
            ROUTE_F_EXECUTIONS,
            "public_execution_srswor_v3.json",
        ),
    }
    counts = {"P": 0, "F": 0}
    dataset_counts = {dataset: 0 for dataset in DATASETS}
    manifest: list[dict[str, Any]] = []
    metric_values = 0
    expected_reasons = [
        "noncryptographic_research_rng",
        "privacy_release_executor_not_audited",
    ]
    expected_metric_keys = {
        "scope",
        "accuracy",
        "macro_f1",
        "auroc",
        "auprc",
    }
    for route, (root, filename) in roots.items():
        for dataset in DATASETS:
            for run_index in range(1, 6):
                path = (
                    root
                    / dataset
                    / f"run_{run_index:03d}"
                    / filename
                )
                public = load_json(path)
                if public.get("visibility") != "public":
                    raise ReleaseDomainAuditError(
                        f"Execution visibility changed: {route}/{dataset}"
                    )
                if public.get("release_status") != "research_non_release":
                    raise ReleaseDomainAuditError(
                        f"Execution release status changed: {route}/{dataset}"
                    )
                claim = public.get("privacy_claim", {})
                if claim.get("status") != (
                    "accountant_output_only_not_a_release_claim"
                ):
                    raise ReleaseDomainAuditError(
                        f"Execution claim status changed: {route}/{dataset}"
                    )
                if claim.get("reason_codes") != expected_reasons:
                    raise ReleaseDomainAuditError(
                        f"Execution reason codes changed: {route}/{dataset}"
                    )
                rng = public.get("rng", {})
                if (
                    rng.get("secure_rng") is not False
                    or rng.get("random_coins_disclosed") is not False
                ):
                    raise ReleaseDomainAuditError(
                        f"Execution randomness boundary changed: {route}/{dataset}"
                    )
                checks = public.get("execution_checks", {})
                if checks.get("random_coins_disclosed") is not False:
                    raise ReleaseDomainAuditError(
                        f"Execution coin check changed: {route}/{dataset}"
                    )
                preprocessing = public.get(
                    "public_data_bindings", {}
                ).get("preprocessing", {})
                if (
                    preprocessing.get("scope") != "public_fixed"
                    or preprocessing.get("fit_on_protected_data") is not False
                ):
                    raise ReleaseDomainAuditError(
                        f"Execution preprocessing changed: {route}/{dataset}"
                    )
                metrics = public.get("public_evaluation", {})
                if set(metrics) != expected_metric_keys:
                    raise ReleaseDomainAuditError(
                        f"Public metric surface changed: {route}/{dataset}"
                    )
                metric_values += 4
                observed_bundle = (
                    public["implementation"]["source_bundle_sha256"]
                    if route == "P"
                    else public["execution_source_bundle_sha256"]
                )
                if observed_bundle != current_bundles[route]:
                    raise ReleaseDomainAuditError(
                        f"Execution source bundle is stale: {route}/{dataset}"
                    )
                forbidden = _walk_keys(public).intersection(
                    FORBIDDEN_PUBLIC_KEYS
                )
                if forbidden:
                    raise ReleaseDomainAuditError(
                        "Public execution contains blocked fields: "
                        + ", ".join(sorted(forbidden))
                    )
                encoded = json.dumps(
                    public,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                if (
                    '"seed"' in encoded
                    or LOCAL_IDENTITY_TOKEN in encoded
                    or WINDOWS_DRIVE_PREFIX in encoded
                ):
                    raise ReleaseDomainAuditError(
                        f"Public execution exposes local data: {route}/{dataset}"
                    )
                manifest.append(
                    {
                        "route": route,
                        "dataset": dataset,
                        "run_index": run_index,
                        "sha256": file_sha256(path),
                    }
                )
                counts[route] += 1
                dataset_counts[dataset] += 1

    if counts != {"P": 15, "F": 15}:
        raise ReleaseDomainAuditError(
            f"Execution count changed: {counts}"
        )
    if dataset_counts != {"uci": 10, "wisdm": 10, "sepsis": 10}:
        raise ReleaseDomainAuditError(
            f"Dataset execution count changed: {dataset_counts}"
        )
    return {
        "executions": 30,
        "by_route": counts,
        "by_dataset": dataset_counts,
        "current_source_bundle_sha256": current_bundles,
        "current_source_bundle_matches": 30,
        "research_non_release": 30,
        "accountant_only_claim_status": 30,
        "secure_rng_false": 30,
        "random_coins_not_disclosed": 30,
        "public_fixed_preprocessing": 30,
        "public_metric_records": 30,
        "public_metric_values": metric_values,
        "blocked_field_violations": 0,
        "execution_artifact_manifest_sha256": payload_sha256(manifest),
    }


def _source_boundary() -> dict[str, Any]:
    compiler_p = COMPILER_P.read_text(encoding="utf-8")
    compiler_f = COMPILER_F.read_text(encoding="utf-8")
    executor_p = EXECUTOR_P.read_text(encoding="utf-8")
    executor_f = EXECUTOR_F.read_text(encoding="utf-8")
    rng_source = RNG_SOURCE.read_text(encoding="utf-8")
    release_source = RELEASE_ARTIFACTS.read_text(encoding="utf-8")

    expected = {
        "route_p_schema_has_release_shape": (
            '"privacy_release"' in compiler_p
            and '"system_csprng"' in compiler_p
        ),
        "route_p_executor_has_research_profile_guard": (
            'contract.execution_profile != "research_benchmark"'
            in executor_p
        ),
        "route_p_executor_has_research_rng_guard": (
            'contract.rng_backend != "research_default"' in executor_p
        ),
        "route_f_contract_is_exact_research_only": (
            '"profile": "research_benchmark"' in compiler_f
            and '"rng_backend": "research_default"' in compiler_f
        ),
        "route_f_executor_requires_execution_ready": (
            "if not route.execution_ready:" in executor_f
        ),
        "route_p_uses_research_generator": (
            "np.random.default_rng" in executor_p
        ),
        "route_f_reuses_research_stream": (
            "from unitdp.owner_poisson_v2 import" in executor_f
            and "_ResearchStream" in executor_f
        ),
        "route_p_integrates_secure_factory": (
            "make_random_source" in executor_p
            or "SystemRandomSource" in executor_p
        ),
        "route_f_integrates_secure_factory": (
            "make_random_source" in executor_f
            or "SystemRandomSource" in executor_f
        ),
        "route_p_emits_benchmark_metrics": (
            '"public_evaluation": {' in executor_p
        ),
        "route_f_emits_benchmark_metrics": (
            '"public_evaluation": {' in executor_f
        ),
        "prototype_secure_adapter_present": (
            "class SystemRandomSource" in rng_source
            and "system_csprng_prototype" in rng_source
        ),
        "prototype_requires_audited_replacement": (
            "audited high-throughput secure noise implementation"
            in rng_source
        ),
        "release_approval_is_explicit_allowlist": (
            "RELEASE_ELIGIBLE_IMPLEMENTATIONS" in release_source
            and "implementation_not_release_approved" in release_source
        ),
    }
    required_true = {
        key
        for key in expected
        if key
        not in {
            "route_p_integrates_secure_factory",
            "route_f_integrates_secure_factory",
        }
    }
    if any(not expected[key] for key in required_true):
        raise ReleaseDomainAuditError(
            f"Expected source boundary changed: {expected}"
        )
    if (
        expected["route_p_integrates_secure_factory"]
        or expected["route_f_integrates_secure_factory"]
    ):
        raise ReleaseDomainAuditError(
            "A strict executor now imports the prototype secure adapter"
        )

    secure_metadata = make_random_source(
        "system_csprng",
        0,
    ).metadata()
    if (
        secure_metadata.rng_security_mode != "system_csprng_prototype"
        or "audited CSPRNG backend" not in secure_metadata.rng_caveat
    ):
        raise ReleaseDomainAuditError(
            "Prototype secure-adapter caveat changed"
        )

    ledger_markers = (
        "release_budget_ledger",
        "privacy_budget_ledger",
        "composition_ledger",
        "reserve_release_budget",
        "commit_release_budget",
    )
    unitdp_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "src" / "unitdp").glob("*.py"))
    ).lower()
    ledger_hits = [
        marker for marker in ledger_markers if marker in unitdp_sources
    ]
    if ledger_hits:
        raise ReleaseDomainAuditError(
            f"A release-ledger marker appeared: {ledger_hits}"
        )

    return {
        **expected,
        "strict_executors_integrating_secure_factory": 0,
        "secure_adapter_security_mode": (
            secure_metadata.rng_security_mode
        ),
        "secure_adapter_has_deployment_caveat": True,
        "registered_multiple_release_ledger_components": 0,
    }


def _readiness_matrix() -> list[dict[str, str]]:
    return [
        {
            "requirement": "public_q_T_denominator_authority",
            "route_p": "established_registered_research",
            "route_f": "established_registered_research",
            "release_domain": "not_generalized",
        },
        {
            "requirement": "exact_mapping_label_cap_validation",
            "route_p": "established_registered_research",
            "route_f": "established_registered_research",
            "release_domain": "not_generic",
        },
        {
            "requirement": "public_private_artifact_redaction",
            "route_p": "established",
            "route_f": "established",
            "release_domain": "certificate_gate_established",
        },
        {
            "requirement": "generic_protected_profile_admission",
            "route_p": "not_supported",
            "route_f": "not_supported",
            "release_domain": "not_established",
        },
        {
            "requirement": "protected_or_dp_preprocessing_authority",
            "route_p": "not_supported",
            "route_f": "not_supported",
            "release_domain": "not_established",
        },
        {
            "requirement": "release_contract_surface",
            "route_p": "schema_only",
            "route_f": "absent",
            "release_domain": "not_executable",
        },
        {
            "requirement": "audited_secure_gaussian_randomness",
            "route_p": "research_generator_only",
            "route_f": "research_generator_only",
            "release_domain": "prototype_not_approved",
        },
        {
            "requirement": "release_executor_approval",
            "route_p": "denied",
            "route_f": "denied",
            "release_domain": "empty_allowlist",
        },
        {
            "requirement": "release_safe_metric_policy",
            "route_p": "benchmark_metrics_only",
            "route_f": "benchmark_metrics_only",
            "release_domain": "not_established",
        },
        {
            "requirement": "safe_abort_and_public_silence",
            "route_p": "execution_denied",
            "route_f": "contract_denied",
            "release_domain": "full_protocol_not_established",
        },
        {
            "requirement": "multiple_release_composition_ledger",
            "route_p": "absent",
            "route_f": "absent",
            "release_domain": "not_established",
        },
        {
            "requirement": "total_release_domain_algorithm",
            "route_p": "not_established",
            "route_f": "not_established",
            "release_domain": "not_established",
        },
    ]


def _assert_public_report(report: dict[str, Any]) -> None:
    if report.get("visibility") != "public":
        raise ReleaseDomainAuditError("Release-domain report must be public")
    encoded = json.dumps(report, sort_keys=True, ensure_ascii=False)
    for token in (
        '"seed"',
        LOCAL_IDENTITY_TOKEN,
        WINDOWS_DRIVE_PREFIX,
        "selected_owners",
        "selected_windows",
        "selection_diagnostics",
        "source_mapping_stats",
    ):
        if token in encoded:
            raise ReleaseDomainAuditError(
                f"Public release-domain report exposes: {token}"
            )


def build_report() -> dict[str, Any]:
    contract_audit = _compile_research_contracts()
    release_requests = _audit_release_requests()
    artifact_boundary = _release_artifact_boundary()
    public_executions = _audit_public_executions()
    source_boundary = _source_boundary()
    source_files = (
        Path(__file__).resolve(),
        COMPILER_P,
        COMPILER_F,
        EXECUTOR_P,
        EXECUTOR_F,
        RELEASE_ARTIFACTS,
        RNG_SOURCE,
        ROOT / "src" / "unitdp" / "benchmark_registry_v2.py",
        ROOT / "src" / "unitdp" / "benchmark_registry_srswor_v3.py",
        ROOT / "src" / "unitdp" / "source_bundle_v2.py",
        ROOT / "src" / "unitdp" / "source_bundle_srswor_v3.py",
        *(_config_path(route, dataset) for route in ("P", "F")
          for dataset in DATASETS),
        *(
            PUBLIC_FIXTURES[dataset][kind]
            for dataset in DATASETS
            for kind in ("mapping", "preprocessor")
        ),
    )
    report: dict[str, Any] = {
        "schema_version": "unitdp.release_domain_gap_public.v31",
        "visibility": "public",
        "scope": (
            "registered research-to-release boundary audit; "
            "not a privacy-release authorization"
        ),
        "contract_audit": contract_audit,
        "release_request_audit": release_requests,
        "release_artifact_boundary": artifact_boundary,
        "public_execution_audit": public_executions,
        "source_boundary": source_boundary,
        "readiness_matrix": _readiness_matrix(),
        "decision": {
            "risk_disposition": (
                "contained_by_fail_closed_denial_not_closed_as_deployment"
            ),
            "generic_release_claim_authorized": False,
            "current_release_requests": "deny",
            "current_research_artifacts": (
                "30_of_30_explicitly_research_non_release"
            ),
            "exact_profile_binding_action": "retain",
            "prototype_secure_adapter_action": (
                "do_not_register_with_strict_executors"
            ),
            "manuscript_action": (
                "retain_V3.1_scope_boundary_without_new_release_claim"
            ),
            "new_manuscript_candidate_required": False,
            "full_solution_requirement": (
                "new_joint_release_route_covering_profile_authority_"
                "preprocessing_secure_randomness_outputs_and_composition"
            ),
        },
        "source_files_sha256": {
            _relative(path): file_sha256(path) for path in source_files
        },
    }
    _assert_public_report(report)
    report["payload_sha256"] = payload_sha256(report)
    _assert_public_report(report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {output}")
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            report,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        "release-domain boundary audit: "
        f"{report['public_execution_audit']['executions']} executions"
    )
    print(f"payload_sha256={report['payload_sha256']}")


if __name__ == "__main__":
    main()
