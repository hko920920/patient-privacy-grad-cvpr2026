#!/usr/bin/env python3
"""Independently verify one Audit v5 UCI HAR evidence directory.

The verifier does not import the production pipeline, mapping utilities,
privacy-claim validator, certificate builder, or local accountant checker.  In
full mode it reopens all eight published UCI HAR files and reconstructs the
train mapping.  Portable mode omits only those raw-source rescans.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO = "uci_har_v5_generated_window"
REGISTRY_KEY = "secure_systemrandom_uci_har_dpsgd@1.0.0"
EXPECTED_FILES = {
    "train/X_train.txt": (
        66006256,
        "9c1246099c8d5463779eec5a3845cc3898df9d8c715441ab2a1232d4421fc42f",
    ),
    "train/y_train.txt": (
        14704,
        "415b3357e2e2d5e70a45c781947c6bdd80e0410a579d862eedcd6bbce3c694ef",
    ),
    "train/subject_train.txt": (
        20152,
        "66e1e2fbe9b396fa5155f531dc3348c803f503be663e2318772d49947db9153a",
    ),
    "test/X_test.txt": (
        26458166,
        "99035209add50d17650e847d8a4658d784a5707466ebb7c2deb97c28e5250a78",
    ),
    "test/y_test.txt": (
        5894,
        "2a94cb53ca46b956c1b2aebd4a09a72badd8a8e9ea9cdfe9f31b3306a08666bb",
    ),
    "test/subject_test.txt": (
        7934,
        "63b8c577f4a85431c06bd87f6384608c90a9f72f05c7956dd21255dc63515c07",
    ),
    "activity_labels.txt": (
        80,
        "f6e6b292704438261f0283b2a82659ab0661447dc16520938bdf379ed1bf8de0",
    ),
    "features.txt": (
        15785,
        "b384889885b1c8680ecf250e0e605b4c79536cdf09ce9f2fa7225a3c6773b5c7",
    ),
}
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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )


def canonical_records_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    canonical = []
    for record in records:
        canonical.append(
            {
                "scenario": str(record["scenario"]),
                "window_id": str(record["window_id"]),
                "generated_unit": str(record["generated_unit"]),
                "owner_ids": sorted(str(owner) for owner in record["owner_ids"]),
                "start": int(record["start"]),
                "end": int(record["end"]),
            }
        )
    canonical.sort(
        key=lambda row: (
            row["scenario"],
            row["window_id"],
            row["generated_unit"],
            ";".join(row["owner_ids"]),
            row["start"],
            row["end"],
        )
    )
    return canonical_json_sha256(canonical)


def add_check(
    checks: Dict[str, Dict[str, Any]],
    name: str,
    passed: bool,
    detail: Any,
) -> None:
    if name in checks:
        raise ValueError(f"Duplicate check: {name}")
    checks[name] = {"passed": bool(passed), "detail": detail}


def read_mapping(path: Path) -> list[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    records = []
    for row in rows:
        owners = [
            value.strip()
            for value in str(row["owner_ids"]).replace("|", ";").split(";")
            if value.strip()
        ]
        records.append(
            {
                **row,
                "owner_ids": owners,
                "start": int(row["start"]),
                "end": int(row["end"]),
                "activity_index": int(row["activity_index"]),
                "source_row_index": int(row["source_row_index"]),
            }
        )
    return records


def mapping_facts(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    deltas: Dict[str, Counter[int]] = defaultdict(Counter)
    owner_counts: Counter[str] = Counter()
    for record in records:
        for owner in record["owner_ids"]:
            owner_counts[str(owner)] += 1
            deltas[str(owner)][int(record["start"])] += 1
            deltas[str(owner)][int(record["end"])] -= 1
    event_kappa = 0
    for owner_deltas in deltas.values():
        active = 0
        for point in sorted(owner_deltas):
            active += owner_deltas[point]
            event_kappa = max(event_kappa, active)
    return {
        "rows": len(records),
        "event_kappa": event_kappa,
        "owner_kappa": max(owner_counts.values(), default=0),
        "owners": len(owner_counts),
        "attribution_arity_max": max(
            (len(record["owner_ids"]) for record in records),
            default=0,
        ),
    }


def expected_mapping(subjects: np.ndarray, labels: np.ndarray) -> list[Dict[str, Any]]:
    owner_ordinals: Dict[int, int] = {}
    rows = []
    for row_index, (subject, label) in enumerate(zip(subjects, labels)):
        subject_id = int(subject)
        ordinal = owner_ordinals.get(subject_id, 0)
        owner_ordinals[subject_id] = ordinal + 1
        rows.append(
            {
                "scenario": SCENARIO,
                "window_id": f"train_row_{row_index:05d}",
                "generated_unit": "window",
                "owner_id": str(subject_id),
                "owner_ids": [str(subject_id)],
                "start": ordinal,
                "end": ordinal + 1,
                "split": "train",
                "activity_index": int(label),
                "source_row_index": row_index,
            }
        )
    return rows


def independent_epsilon(
    sample_rate: float,
    noise_multiplier: float,
    steps: int,
    delta: float,
) -> float:
    from dp_accounting import dp_event, rdp

    accountant = rdp.RdpAccountant()
    accountant.compose(
        dp_event.PoissonSampledDpEvent(
            sample_rate,
            dp_event.GaussianDpEvent(noise_multiplier),
        ),
        steps,
    )
    return float(accountant.get_epsilon(delta))


def verify(
    evidence_dir: Path,
    data_root: Path,
    *,
    skip_input_rescan: bool,
) -> Dict[str, Any]:
    checks: Dict[str, Dict[str, Any]] = {}
    required = {
        "input_manifest.json",
        "mapping.csv",
        "released_model.json",
        "batch_trace.csv",
        "metrics.json",
        "privacy_report.json",
        "route_assertions.json",
        "certificates/certificate_uci_har_v5_generated_window_window.json",
        "certificates/certificate_uci_har_v5_generated_window_window_internal.json",
        "certificates/certificate_uci_har_v5_generated_window_event.json",
        "certificates/certificate_uci_har_v5_generated_window_event_internal.json",
        "certificates/certificate_uci_har_v5_generated_window_owner.json",
        "certificates/certificate_uci_har_v5_generated_window_owner_internal.json",
    }
    missing = sorted(
        relative
        for relative in required
        if not (evidence_dir / relative).is_file()
    )
    add_check(checks, "required_artifacts", not missing, missing)

    report = json.loads(
        (evidence_dir / "privacy_report.json").read_text(encoding="utf-8")
    )
    mechanism = report["mechanism"]
    privacy = report["privacy"]
    pipeline = report["pipeline"]
    manifest = pipeline["manifest"]
    runtime = report["runtime_trace"]
    input_manifest = json.loads(
        (evidence_dir / "input_manifest.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (evidence_dir / "metrics.json").read_text(encoding="utf-8")
    )
    model = json.loads(
        (evidence_dir / "released_model.json").read_text(encoding="utf-8")
    )
    assertions = json.loads(
        (evidence_dir / "route_assertions.json").read_text(encoding="utf-8")
    )
    certificates = {
        claim: json.loads(
            (
                evidence_dir
                / "certificates"
                / f"certificate_{SCENARIO}_{claim}.json"
            ).read_text(encoding="utf-8")
        )
        for claim in ("window", "event", "owner")
    }
    internal_certificates = {
        claim: json.loads(
            (
                evidence_dir
                / "certificates"
                / f"certificate_{SCENARIO}_{claim}_internal.json"
            ).read_text(encoding="utf-8")
        )
        for claim in ("window", "event", "owner")
    }
    records = read_mapping(evidence_dir / "mapping.csv")
    facts = mapping_facts(records)

    add_check(
        checks,
        "report_identity",
        report["schema_version"] == "2.0"
        and report["scenario"] == SCENARIO,
        {"schema": report["schema_version"], "scenario": report["scenario"]},
    )
    add_check(
        checks,
        "mapping_file_digest",
        file_sha256(evidence_dir / "mapping.csv")
        == pipeline["mapping_file_sha256"]
        == manifest["mapping_file_sha256"],
        pipeline["mapping_file_sha256"],
    )
    selected_digest = canonical_records_sha256(records)
    add_check(
        checks,
        "selected_mapping_digest",
        selected_digest
        == pipeline["selected_mapping_sha256"]
        == manifest["selected_mapping_sha256"],
        selected_digest,
    )
    add_check(
        checks,
        "mapping_facts",
        facts
        == {
            "rows": 7352,
            "event_kappa": 1,
            "owner_kappa": 409,
            "owners": 21,
            "attribution_arity_max": 1,
        },
        facts,
    )

    manifest_core = {
        "dataset_id": input_manifest["dataset_id"],
        "files": input_manifest["files"],
        "total_bytes": input_manifest["total_bytes"],
    }
    add_check(
        checks,
        "input_manifest_content_digest",
        canonical_json_sha256(manifest_core)
        == input_manifest["dataset_bundle_sha256"]
        == manifest["dataset_bundle_sha256"],
        input_manifest["dataset_bundle_sha256"],
    )
    add_check(
        checks,
        "input_manifest_file_digest",
        file_sha256(evidence_dir / "input_manifest.json")
        == manifest["input_manifest_sha256"]
        == runtime["evidence"]["input_manifest_sha256"],
        manifest["input_manifest_sha256"],
    )
    declared_inputs = {
        item["path"]: (item["bytes"], item["sha256"])
        for item in input_manifest["files"]
    }
    add_check(
        checks,
        "declared_input_set",
        declared_inputs == EXPECTED_FILES,
        declared_inputs,
    )

    if not skip_input_rescan:
        actual_inputs = {}
        for relative, expected in EXPECTED_FILES.items():
            path = data_root / relative
            actual_inputs[relative] = (path.stat().st_size, file_sha256(path))
            add_check(
                checks,
                f"raw_input_{relative.replace('/', '_')}",
                actual_inputs[relative] == expected,
                actual_inputs[relative],
            )
        train_labels = np.loadtxt(
            data_root / "train" / "y_train.txt",
            dtype=np.int64,
        )
        train_subjects = np.loadtxt(
            data_root / "train" / "subject_train.txt",
            dtype=np.int64,
        )
        regenerated = expected_mapping(train_subjects, train_labels)
        add_check(
            checks,
            "raw_to_mapping_regeneration",
            regenerated == records,
            {"expected": len(regenerated), "actual": len(records)},
        )
        train_features = np.loadtxt(
            data_root / "train" / "X_train.txt",
            dtype=np.float64,
        )
        test_features = np.loadtxt(
            data_root / "test" / "X_test.txt",
            dtype=np.float64,
        )
        add_check(
            checks,
            "raw_feature_shapes",
            train_features.shape == (7352, 561)
            and test_features.shape == (2947, 561),
            {
                "train": train_features.shape,
                "test": test_features.shape,
            },
        )
        add_check(
            checks,
            "raw_feature_public_bounds",
            np.all(np.isfinite(train_features))
            and np.all(np.isfinite(test_features))
            and float(train_features.min()) == -1.0
            and float(train_features.max()) == 1.0
            and float(test_features.min()) == -1.0
            and float(test_features.max()) == 1.0,
            {
                "train_min": float(train_features.min()),
                "train_max": float(train_features.max()),
                "test_min": float(test_features.min()),
                "test_max": float(test_features.max()),
            },
        )

    add_check(
        checks,
        "mechanism_digest",
        canonical_json_sha256(mechanism) == manifest["mechanism_sha256"],
        manifest["mechanism_sha256"],
    )
    add_check(
        checks,
        "pipeline_digest",
        canonical_json_sha256(manifest) == pipeline["pipeline_sha256"],
        pipeline["pipeline_sha256"],
    )
    for index, step in enumerate(manifest["preprocessing"]):
        add_check(
            checks,
            f"preprocessing_parameters_{index}",
            canonical_json_sha256(step["parameters"])
            == step["parameters_sha256"],
            step["id"],
        )
    for name in ("raw_domain", "contribution_policy"):
        block = manifest[name]
        add_check(
            checks,
            f"{name}_parameters",
            canonical_json_sha256(block["parameters"])
            == block["parameters_sha256"],
            block["id"],
        )
    schedule = manifest["schedule"]
    add_check(
        checks,
        "schedule_parameters",
        canonical_json_sha256(schedule["parameters"])
        == schedule["parameters_sha256"],
        schedule["id"],
    )

    registry_path = PROJECT_ROOT / "docs" / "mechanism_registry_v2_0.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entry = registry["mechanisms"].get(REGISTRY_KEY)
    add_check(
        checks,
        "registry_entry",
        isinstance(entry, Mapping)
        and canonical_json_sha256(entry)
        == mechanism["adapter_registry_entry_sha256"],
        REGISTRY_KEY,
    )
    if not isinstance(entry, Mapping):
        entry = {}
    semantics = (
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
    )
    add_check(
        checks,
        "registered_semantics",
        all(
            str(entry.get(key)) == str(mechanism.get(key))
            for key in semantics
        ),
        {key: mechanism.get(key) for key in semantics},
    )
    for prefix, id_key, hash_key in (
        ("pipeline_source", "code_artifact_id", "code_sha256"),
        (
            "accountant_source",
            "accountant_checker_artifact_id",
            "accountant_checker_sha256",
        ),
    ):
        artifact_id = str(entry.get(id_key) or "")
        artifact_path = (PROJECT_ROOT / artifact_id).resolve()
        try:
            artifact_path.relative_to(PROJECT_ROOT.resolve())
            in_root = True
        except ValueError:
            in_root = False
        actual_hash = file_sha256(artifact_path) if in_root else ""
        add_check(
            checks,
            f"{prefix}_digest",
            in_root
            and actual_hash
            == entry.get(hash_key)
            == manifest.get(hash_key),
            {
                "artifact": artifact_id,
                "actual": actual_hash,
                "registered": entry.get(hash_key),
            },
        )

    pipeline_text = (
        PROJECT_ROOT / str(entry["code_artifact_id"])
    ).read_text(encoding="utf-8")
    add_check(
        checks,
        "pipeline_semantic_anchors",
        "rng.randrange(sample_rate_denominator) < sample_rate_numerator"
        in pipeline_text
        and "weights -= optimizer_step_size * (summed_gradient + noise)"
        in pipeline_text
        and "return (draw() + draw() + draw() + draw()) / 2.0"
        in pipeline_text
        and "expected_batch = sample_rate * len(train_x)" not in pipeline_text,
        entry["code_artifact_id"],
    )
    add_check(
        checks,
        "exact_sampling_contract",
        mechanism["sample_rate_numerator"] == 1
        and mechanism["sample_rate_denominator"] == 50
        and mechanism["sample_rate"] == 1 / 50
        and mechanism["sampling_implementation"]
        == "systemrandom_randrange_bernoulli_v1",
        {
            "numerator": mechanism["sample_rate_numerator"],
            "denominator": mechanism["sample_rate_denominator"],
            "rate": mechanism["sample_rate"],
        },
    )
    add_check(
        checks,
        "public_constant_update",
        mechanism["update_normalization"]
        == "public_constant_step_no_dataset_denominator"
        and mechanism["optimizer_step_size"] == 1 / 200,
        {
            "normalization": mechanism["update_normalization"],
            "step": mechanism["optimizer_step_size"],
        },
    )
    add_check(
        checks,
        "four_draw_boundary",
        mechanism["noise_generation"]
        == "normalvariate_discard1_sum4_div2_v1"
        and mechanism["noise_hardening"]
        == "known_fp_reconstruction_mitigation_four_draw_v1"
        and report["rng_assurance"] == "known_attack_hardened_secure",
        {
            "noise_generation": mechanism["noise_generation"],
            "noise_hardening": mechanism["noise_hardening"],
        },
    )
    independent = independent_epsilon(
        mechanism["sample_rate"],
        mechanism["noise_multiplier"],
        mechanism["steps"],
        privacy["delta"],
    )
    add_check(
        checks,
        "independent_accountant",
        math.isclose(
            independent,
            privacy["epsilon"],
            rel_tol=1e-7,
            abs_tol=1e-10,
        ),
        {
            "independent": independent,
            "reported": privacy["epsilon"],
            "dp_accounting": importlib.metadata.version("dp-accounting"),
        },
    )

    runtime_payload = {
        key: runtime.get(key)
        for key in (*RUNTIME_EXACT_FIELDS, *RUNTIME_NUMERIC_FIELDS)
    }
    add_check(
        checks,
        "runtime_trace_digest",
        canonical_json_sha256(runtime_payload) == runtime["trace_hash"],
        runtime["trace_hash"],
    )
    add_check(
        checks,
        "runtime_evidence_digest",
        canonical_json_sha256(runtime["evidence"])
        == runtime["runtime_evidence_sha256"],
        runtime["runtime_evidence_sha256"],
    )
    expected_runtime = {
        **{key: mechanism.get(key) for key in RUNTIME_EXACT_FIELDS},
        **{key: mechanism.get(key) for key in RUNTIME_NUMERIC_FIELDS},
        "code_artifact_id": manifest["code_artifact_id"],
        "code_sha256": manifest["code_sha256"],
        "accountant_checker_artifact_id": manifest[
            "accountant_checker_artifact_id"
        ],
        "accountant_checker_sha256": manifest["accountant_checker_sha256"],
        "selected_mapping_sha256": manifest["selected_mapping_sha256"],
        "pipeline_sha256": pipeline["pipeline_sha256"],
        "epsilon": privacy["epsilon"],
        "delta": privacy["delta"],
        "delta_convention": privacy["delta_convention"],
        "runtime_evidence_sha256": canonical_json_sha256(runtime["evidence"]),
    }
    add_check(
        checks,
        "runtime_exact_fields",
        all(runtime.get(key) == expected_runtime.get(key) for key in RUNTIME_EXACT_FIELDS),
        {
            key: {
                "actual": runtime.get(key),
                "expected": expected_runtime.get(key),
            }
            for key in RUNTIME_EXACT_FIELDS
            if runtime.get(key) != expected_runtime.get(key)
        },
    )
    add_check(
        checks,
        "runtime_numeric_fields",
        all(runtime.get(key) == expected_runtime.get(key) for key in RUNTIME_NUMERIC_FIELDS),
        {
            key: {
                "actual": runtime.get(key),
                "expected": expected_runtime.get(key),
            }
            for key in RUNTIME_NUMERIC_FIELDS
            if runtime.get(key) != expected_runtime.get(key)
        },
    )
    add_check(
        checks,
        "execution_artifact_digests",
        file_sha256(evidence_dir / "released_model.json")
        == manifest["released_model_sha256"]
        == runtime["evidence"]["released_model_sha256"]
        and file_sha256(evidence_dir / "batch_trace.csv")
        == manifest["batch_trace_sha256"]
        == runtime["evidence"]["batch_trace_sha256"],
        {
            "model": manifest["released_model_sha256"],
            "trace": manifest["batch_trace_sha256"],
        },
    )

    weights = np.asarray(model["weights_including_bias"], dtype=np.float64)
    add_check(
        checks,
        "model_shape_and_finiteness",
        weights.shape == (562, 6) and np.all(np.isfinite(weights)),
        weights.shape,
    )
    with (evidence_dir / "batch_trace.csv").open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        trace_rows = list(csv.DictReader(handle))
    add_check(
        checks,
        "trace_steps",
        len(trace_rows) == mechanism["steps"] == 100
        and [int(row["step"]) for row in trace_rows] == list(range(1, 101)),
        len(trace_rows),
    )
    add_check(
        checks,
        "trace_values",
        all(
            0 <= int(row["poisson_batch_count"]) <= 7352
            and 0 <= int(row["clipped_record_count"])
            <= int(row["poisson_batch_count"])
            and math.isfinite(float(row["max_unclipped_gradient_norm"]))
            and float(row["max_unclipped_gradient_norm"]) >= 0
            for row in trace_rows
        ),
        {
            "min_batch": min(int(row["poisson_batch_count"]) for row in trace_rows),
            "max_batch": max(int(row["poisson_batch_count"]) for row in trace_rows),
        },
    )
    add_check(
        checks,
        "metrics_domain",
        metrics["train_records"] == 7352
        and metrics["test_records"] == 2947
        and metrics["train_owners"] == 21
        and metrics["test_owners"] == 9
        and metrics["feature_count"] == 561
        and all(
            math.isfinite(float(metrics[key])) and 0 <= float(metrics[key]) <= 1
            for key in ("train_accuracy", "test_accuracy", "test_macro_f1")
        ),
        metrics,
    )

    add_check(
        checks,
        "window_authorization",
        certificates["window"]["release_status"] == "ALLOWED"
        and certificates["window"]["unit_path"] == "DIRECT"
        and certificates["window"]["stability_bound"] == 1
        and bool(certificates["window"]["supported_statement"])
        and not internal_certificates["window"]["issue_codes"],
        certificates["window"],
    )
    add_check(
        checks,
        "event_fail_closed",
        certificates["event"]["release_status"] == "BLOCKED_UNVERIFIED"
        and certificates["event"]["unit_path"] == "UNDERSPECIFIED"
        and not certificates["event"]["supported_statement"]
        and {
            "STABILITY_STATUS_UNVERIFIED",
            "INFLUENCE_SUPPORT_INCOMPLETE",
            "STABILITY_METHOD_UNREGISTERED",
        }.issubset(
            set(
                str(internal_certificates["event"]["issue_codes"]).split(";")
            )
        ),
        internal_certificates["event"]["issue_codes"],
    )
    add_check(
        checks,
        "owner_fail_closed",
        certificates["owner"]["release_status"] == "BLOCKED_UNVERIFIED"
        and certificates["owner"]["unit_path"] == "UNDERSPECIFIED"
        and not certificates["owner"]["supported_statement"]
        and {
            "STABILITY_STATUS_UNVERIFIED",
            "OWNER_PARTITION_NOT_FIXED",
            "OWNER_PUBLIC_CAP_UNVERIFIED",
        }.issubset(
            set(
                str(internal_certificates["owner"]["issue_codes"]).split(";")
            )
        ),
        internal_certificates["owner"]["issue_codes"],
    )
    add_check(
        checks,
        "route_assertions",
        assertions
        == {
            "event_blocked_without_wording": True,
            "owner_blocked_without_wording": True,
            "window_allowed_direct": True,
        },
        assertions,
    )
    add_check(
        checks,
        "claim_boundary_manifest",
        manifest["raw_domain"]["id"]
        == "uci_har_published_window_row_domain_v1"
        and manifest["raw_domain"]["parameters"][
            "exact_raw_session_support_available"
        ]
        is False
        and manifest["contribution_policy"]["parameters"]["cap"] is None
        and report["claim_contracts"]["event"]["stability_status"]
        == "unverified"
        and report["claim_contracts"]["owner"]["stability_status"]
        == "unverified",
        {
            "raw_domain": manifest["raw_domain"]["id"],
            "event": report["claim_contracts"]["event"]["stability_status"],
            "owner": report["claim_contracts"]["owner"]["stability_status"],
        },
    )

    paths_text = "\n".join(
        str(path.relative_to(evidence_dir))
        for path in evidence_dir.rglob("*")
        if path.is_file()
    )
    serialized = json.dumps(report, ensure_ascii=False) + paths_text
    add_check(
        checks,
        "portable_path_boundary",
        "C:\\Users\\" not in serialized
        and "/home/" not in serialized
        and "SOGANG" not in serialized,
        "no local identity/path token",
    )

    passed = sum(item["passed"] for item in checks.values())
    result = {
        "schema_version": "uci_har_v5_independent_verification_v1",
        "evidence_dir": evidence_dir.name,
        "raw_inputs_rescanned": not skip_input_rescan,
        "checks_passed": passed,
        "checks_total": len(checks),
        "all_passed": passed == len(checks),
        "report_sha256": file_sha256(evidence_dir / "privacy_report.json"),
        "model_sha256": file_sha256(evidence_dir / "released_model.json"),
        "verifier_sha256": file_sha256(Path(__file__).resolve()),
        "checks": checks,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-dir",
        default="reports/uci_har_v5_pilot_001",
    )
    parser.add_argument(
        "--data-root",
        default="data/uci_har/extracted/UCI HAR Dataset",
    )
    parser.add_argument("--skip-input-rescan", action="store_true")
    args = parser.parse_args()
    evidence_dir = Path(args.evidence_dir)
    data_root = Path(args.data_root)
    if not evidence_dir.is_absolute():
        evidence_dir = PROJECT_ROOT / evidence_dir
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    result = verify(
        evidence_dir,
        data_root,
        skip_input_rescan=args.skip_input_rescan,
    )
    output_name = (
        "independent_verification_portable.json"
        if args.skip_input_rescan
        else "independent_verification.json"
    )
    (evidence_dir / output_name).write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"UCI HAR verification: {result['checks_passed']}/{result['checks_total']}")
    if not result["all_passed"]:
        failed = [
            name for name, item in result["checks"].items() if not item["passed"]
        ]
        raise SystemExit(f"Failed checks: {failed}")


if __name__ == "__main__":
    main()
