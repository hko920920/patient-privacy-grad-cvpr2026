#!/usr/bin/env python3
"""Separately verify the frozen Audit retrospective failure lineage.

This verifier uses only the Python standard library.  It does not import the
builder, production validator, mapping code, certificate builder, or any
historical executable.  The anonymous-package route checks the construction
protocol by its frozen digest without distributing the internal source text.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = PROJECT_ROOT / "reports" / "audit_retrospective_failures_v5_001"

EXPECTED_PROTOCOL_SHA256 = (
    "9bcef8d1dae5fa0ee5f42d40345bef853bbfc7f4cef0c7c96206fce09bfea984"
)
EXPECTED_ORIGINAL_VERIFIER_SHA256 = (
    "7f514fd060d7468ede4b7e4825846a0ee1ea6ef07df448ba4a695c9674a3e836"
)
EXPECTED_ARCHIVES = {
    "unit_aware_privacy_reporting_supplementary_materials.zip": (
        "854d84d688660bcbc92d15a622bf6f168e90eb464873c7c273ef93e94ea24730"
    ),
    "dist/audit_code_and_data_supplement_aaai27_v2.zip": (
        "706509e226c900911bdb472a08dfb3dea200c34590eb09d3f0ca54ddf128d567"
    ),
    "dist/audit_code_and_data_supplement_aaai27_v4.zip": (
        "646d44f70effc6d1b69f6d8dab2aa4840fcb478881314fcf1f02e65c2b3c1ee9"
    ),
}
EXPECTED_DISCOVERY = {
    "paper_notes/AAAI27_AUDIT_PAPER_SOURCE_AUDIT_2026-07-24.md": (
        "76383eeb06daf0ee3f3f46dabdbadeacfe2a474f9e254ac15960359be9a95803"
    ),
    "paper_notes/AAAI27_AUDIT_FULL_REAUDIT_2026-07-24.md": (
        "6f4df3973c5947e0af171263fe8c39c131c1a78072130725b67e9f2925d4daf7"
    ),
}
EXPECTED_FILE_SOURCES = {
    "scripts/opacus_uci_har_sequence_dpsgd_demo.py": (
        "e4ac2c83e601474270652da64e3efe176864491b0883126a5586560d03b5b677"
    ),
}
EXPECTED_IDS = tuple(f"R{index:02d}" for index in range(1, 11))
EXPECTED_ROOTS = {
    "R01": "legacy_verdict_ignored_mapping_sampler_schedule_validity",
    "R02": "unvalidated_external_conversion_attachment",
    "R03": "private_global_fitted_preprocessing_omitted_from_support",
    "R04": "private_data_dependent_update_denominator",
    "R05": "one_draw_gaussian_overstated_as_release_grade",
    "R06": "observed_owner_incidence_used_as_global_stability",
    "R07": "generic_replace_one_raw_event_domain",
    "R08": "mechanism_registry_did_not_bind_executable_semantics",
    "R09": "missing_runtime_still_authorized_conditional_positive",
    "R10": "float_threshold_sampler_not_exact_accountant_rational",
}
INDEPENDENT_REQUIRED_ANCHORS = {
    "R01": (
        "sampler_check(sampler, accountant_assumption)",
        '"mapping_hash_status": mapping_hash_status(audit_row, privacy_report)',
        "def test_mapping_bytes_mutation_blocks",
    ),
    "R02": (
        'verdict = "external_conversion_attached"',
        "def test_external_conversion_attachment_cannot_authorize",
    ),
    "R03": (
        "mean = train_x.mean(axis=(0, 2), keepdims=True)",
        "def test_private_global_preprocessing_blocks",
    ),
    "R04": (
        "expected_batch = sample_rate * len(train_x)",
        "weights -= optimizer_step_size * (summed_gradient + noise)",
    ),
    "R05": (
        '"rng_assurance": "release_grade_secure"',
        "return (draw() + draw() + draw() + draw()) / 2.0",
    ),
    "R06": (
        '"stability_bound": int(mapping_evidence.owner_kappa or 0)',
        "def test_owner_observed_kappa_without_public_cap_blocks",
    ),
    "R07": (
        '"raw_adjacency": "replace_one"',
        "def test_generic_replace_one_event_domain_is_insufficient",
    ),
    "R08": (
        "registered_mechanisms = {",
        "def test_self_consistent_forged_pipeline_code_hash_blocks",
    ),
    "R09": (
        'elif status in {"missing", "not_applicable"}:',
        "def test_missing_runtime_blocks_post_execution_wording",
    ),
    "R10": (
        "rng.random() < sample_rate",
        "rng.randrange(sample_rate_denominator)",
    ),
}


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


def read_identity(identity: Mapping[str, Any]) -> tuple[bytes, str]:
    kind = identity["kind"]
    if kind == "file":
        path = PROJECT_ROOT / identity["path"]
        data = path.read_bytes()
        return data, identity["path"]
    if kind == "zip_member":
        archive_path = PROJECT_ROOT / identity["archive"]
        with ZipFile(archive_path) as archive:
            data = archive.read(identity["member"])
        return data, f"{identity['archive']}::{identity['member']}"
    raise ValueError(f"Unknown source kind: {kind!r}")


def add_check(
    checks: Dict[str, Dict[str, Any]],
    name: str,
    passed: bool,
    detail: Any,
) -> None:
    if name in checks:
        raise ValueError(f"Duplicate check name: {name}")
    checks[name] = {"passed": bool(passed), "detail": detail}


def verify_identity(
    checks: Dict[str, Dict[str, Any]],
    prefix: str,
    identity: Mapping[str, Any],
) -> str:
    data, label = read_identity(identity)
    actual_member_sha = sha256_bytes(data)
    expected_member_sha = (
        identity["sha256"]
        if identity["kind"] == "file"
        else identity["member_sha256"]
    )
    add_check(
        checks,
        f"{prefix}.bytes_sha256",
        actual_member_sha == expected_member_sha,
        {"label": label, "actual": actual_member_sha, "expected": expected_member_sha},
    )
    if identity["kind"] == "file":
        expected_file = EXPECTED_FILE_SOURCES.get(identity["path"])
        add_check(
            checks,
            f"{prefix}.frozen_file_identity",
            expected_file == expected_member_sha,
            {"path": identity["path"], "sha256": expected_member_sha},
        )
    else:
        actual_archive = file_sha256(PROJECT_ROOT / identity["archive"])
        expected_archive = EXPECTED_ARCHIVES.get(identity["archive"])
        add_check(
            checks,
            f"{prefix}.archive_sha256",
            actual_archive == identity["archive_sha256"] == expected_archive,
            {
                "archive": identity["archive"],
                "actual": actual_archive,
                "reported": identity["archive_sha256"],
                "expected": expected_archive,
            },
        )
    text = data.decode("utf-8")
    for index, anchor in enumerate(identity["verified_present_anchors"], start=1):
        add_check(
            checks,
            f"{prefix}.reported_anchor_{index:02d}",
            anchor in text,
            anchor,
        )
    for index, anchor in enumerate(identity["verified_absent_anchors"], start=1):
        add_check(
            checks,
            f"{prefix}.reported_absence_{index:02d}",
            anchor not in text,
            anchor,
        )
    return text


def verify(
    bundle: Path,
    *,
    skip_internal_discovery_sources: bool = False,
) -> Dict[str, Any]:
    checks: Dict[str, Dict[str, Any]] = {}
    report_path = bundle / "retrospective_failure_lineage.json"
    csv_path = bundle / "incidents.csv"
    summary_path = bundle / "summary.md"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    add_check(
        checks,
        "schema",
        report.get("schema_version") == "audit_retrospective_failure_lineage_v1",
        report.get("schema_version"),
    )
    protocol_path = PROJECT_ROOT / report["protocol"]["path"]
    protocol_present = protocol_path.is_file()
    add_check(
        checks,
        "protocol_hash",
        report["protocol"]["sha256"] == EXPECTED_PROTOCOL_SHA256
        and (
            file_sha256(protocol_path) == EXPECTED_PROTOCOL_SHA256
            if protocol_present
            else skip_internal_discovery_sources
        ),
        {
            **report["protocol"],
            "material_present": protocol_present,
            "verification_mode": (
                "direct_bytes"
                if protocol_present
                else "frozen_digest_anchor_only"
            ),
        },
    )
    verifier_sha = file_sha256(Path(__file__).resolve())
    verifier_text = Path(__file__).read_text(encoding="utf-8")
    verifier_tree = ast.parse(verifier_text)
    imported_modules = {
        alias.name
        for node in ast.walk(verifier_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module or ""
        for node in ast.walk(verifier_tree)
        if isinstance(node, ast.ImportFrom)
    )
    add_check(
        checks,
        "verifier_binding",
        report["independent_verifier"]["sha256"]
        == EXPECTED_ORIGINAL_VERIFIER_SHA256
        and report["independent_verifier"]["imports_production_validator"] is False
        and "privacy_claim_validator" not in imported_modules,
        {
            "package_safe_verifier_sha256": verifier_sha,
            "frozen_original": report["independent_verifier"],
        },
    )
    builder_sha = file_sha256(PROJECT_ROOT / report["builder"]["path"])
    add_check(
        checks,
        "builder_binding",
        report["builder"]["sha256"] == builder_sha,
        {"actual": builder_sha, "reported": report["builder"]["sha256"]},
    )

    reported_content_hash = report["content_sha256"]
    content = dict(report)
    del content["content_sha256"]
    add_check(
        checks,
        "content_digest",
        canonical_json_sha256(content) == reported_content_hash,
        {
            "actual": canonical_json_sha256(content),
            "reported": reported_content_hash,
        },
    )

    incidents = report["incidents"]
    ids = tuple(incident["id"] for incident in incidents)
    add_check(checks, "fixed_inclusion_ids", ids == EXPECTED_IDS, ids)
    add_check(
        checks,
        "aggregate_count",
        report["aggregate"]["incident_count"] == len(incidents) == 10,
        report["aggregate"],
    )
    add_check(
        checks,
        "design_boundary",
        report["design"]["controlled_probe_cases_mixed"] is False
        and report["design"]["independent_submission_sample"] is False
        and report["design"]["prevalence_estimate"] is False,
        report["design"],
    )

    independent_anchor_hits = {}
    for incident in incidents:
        incident_id = incident["id"]
        add_check(
            checks,
            f"{incident_id}.root_cause",
            incident["root_cause"] == EXPECTED_ROOTS[incident_id],
            incident["root_cause"],
        )
        discovery = incident["discovery"]
        if not skip_internal_discovery_sources:
            discovery_path = PROJECT_ROOT / discovery["path"]
            actual_discovery_sha = file_sha256(discovery_path)
            discovery_text = discovery_path.read_text(encoding="utf-8")
            add_check(
                checks,
                f"{incident_id}.discovery_sha256",
                actual_discovery_sha
                == discovery["sha256"]
                == EXPECTED_DISCOVERY.get(discovery["path"]),
                {
                    "path": discovery["path"],
                    "actual": actual_discovery_sha,
                    "reported": discovery["sha256"],
                },
            )
            add_check(
                checks,
                f"{incident_id}.discovery_anchor",
                discovery["verified_anchor"] in discovery_text,
                discovery["verified_anchor"],
            )

        historical_text = verify_identity(
            checks,
            f"{incident_id}.historical",
            incident["historical_source"],
        )
        control_texts = []
        for index, control in enumerate(incident["repaired_controls"], start=1):
            control_texts.append(
                verify_identity(
                    checks,
                    f"{incident_id}.control_{index:02d}",
                    control,
                )
            )
        joined = historical_text + "\n" + "\n".join(control_texts)
        required = INDEPENDENT_REQUIRED_ANCHORS[incident_id]
        hits = [anchor for anchor in required if anchor in joined]
        independent_anchor_hits[incident_id] = hits
        add_check(
            checks,
            f"{incident_id}.independent_required_anchors",
            len(hits) == len(required),
            {"required": required, "hits": hits},
        )
        add_check(
            checks,
            f"{incident_id}.status_boundary",
            incident["evidence_class"] == "retrospective_system_lineage"
            and bool(incident["current_status"]),
            {
                "evidence_class": incident["evidence_class"],
                "current_status": incident["current_status"],
            },
        )

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    add_check(
        checks,
        "csv_ids",
        tuple(row["id"] for row in rows) == EXPECTED_IDS,
        [row["id"] for row in rows],
    )
    add_check(
        checks,
        "csv_roots",
        all(row["root_cause"] == EXPECTED_ROOTS[row["id"]] for row in rows),
        len(rows),
    )
    summary = summary_path.read_text(encoding="utf-8")
    add_check(
        checks,
        "summary_boundary",
        "not an independent" in summary
        and "prevalence estimate" in summary
        and "completeness result" in summary,
        summary.splitlines()[:8],
    )
    add_check(
        checks,
        "summary_all_ids",
        all(f"| {incident_id} |" in summary for incident_id in EXPECTED_IDS),
        list(EXPECTED_IDS),
    )

    passed = sum(item["passed"] for item in checks.values())
    try:
        bundle_label = str(bundle.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        bundle_label = str(bundle)
    result: Dict[str, Any] = {
        "schema_version": "audit_retrospective_failure_verification_v1",
        "bundle": bundle_label,
        "report_sha256": file_sha256(report_path),
        "verifier_sha256": verifier_sha,
        "checks_passed": passed,
        "checks_total": len(checks),
        "all_passed": passed == len(checks),
        "internal_discovery_sources_reopened": (
            not skip_internal_discovery_sources
        ),
        "independent_required_anchor_hits": independent_anchor_hits,
        "checks": checks,
    }
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE))
    parser.add_argument(
        "--portable-package",
        action="store_true",
        help=(
            "Omit the two non-distributed discovery-source anchors from the "
            "rerun; their hashes remain in the frozen 176-check full record."
        ),
    )
    args = parser.parse_args()
    bundle = Path(args.bundle_dir)
    if not bundle.is_absolute():
        bundle = PROJECT_ROOT / bundle
    result = verify(
        bundle,
        skip_internal_discovery_sources=args.portable_package,
    )
    output = bundle / (
        "independent_verification_portable.json"
        if args.portable_package
        else "independent_verification.json"
    )
    output.write_text(
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
    print(
        f"Retrospective verification: "
        f"{result['checks_passed']}/{result['checks_total']}"
    )
    if not result["all_passed"]:
        failed = [
            name for name, item in result["checks"].items() if not item["passed"]
        ]
        raise SystemExit(f"Failed checks: {failed}")


if __name__ == "__main__":
    main()
