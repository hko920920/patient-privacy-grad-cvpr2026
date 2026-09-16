#!/usr/bin/env python3
"""Build the frozen Audit v5 retrospective development-failure lineage.

This builder does not execute or import a historical certificate builder.  It
authenticates preserved source bytes, discovery records, repaired controls,
and exact semantic anchors fixed by the pre-run protocol.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "audit_retrospective_failures_v5_001"

PROTOCOL = (
    "paper_notes/AAAI27_AUDIT_V5_STRENGTHENING_PROTOCOL_2026-07-24.md"
)
PROTOCOL_SHA256 = (
    "9bcef8d1dae5fa0ee5f42d40345bef853bbfc7f4cef0c7c96206fce09bfea984"
)
SOURCE_AUDIT = "paper_notes/AAAI27_AUDIT_PAPER_SOURCE_AUDIT_2026-07-24.md"
SOURCE_AUDIT_SHA256 = (
    "76383eeb06daf0ee3f3f46dabdbadeacfe2a474f9e254ac15960359be9a95803"
)
FULL_REAUDIT = "paper_notes/AAAI27_AUDIT_FULL_REAUDIT_2026-07-24.md"
FULL_REAUDIT_SHA256 = (
    "6f4df3973c5947e0af171263fe8c39c131c1a78072130725b67e9f2925d4daf7"
)

LEGACY_ZIP = "unit_aware_privacy_reporting_supplementary_materials.zip"
LEGACY_ZIP_SHA256 = (
    "854d84d688660bcbc92d15a622bf6f168e90eb464873c7c273ef93e94ea24730"
)
V2_ZIP = "dist/audit_code_and_data_supplement_aaai27_v2.zip"
V2_ZIP_SHA256 = (
    "706509e226c900911bdb472a08dfb3dea200c34590eb09d3f0ca54ddf128d567"
)
V4_ZIP = "dist/audit_code_and_data_supplement_aaai27_v4.zip"
V4_ZIP_SHA256 = (
    "646d44f70effc6d1b69f6d8dab2aa4840fcb478881314fcf1f02e65c2b3c1ee9"
)

LEGACY_BUILDER_MEMBER = "scripts/build_unit_certificate.py"
LEGACY_BUILDER_SHA256 = (
    "0866782f12ac000f93fec178cc89c108b98c1e330838a044b667c336a8d603cb"
)
V2_PIPELINE_MEMBER = "scripts/wisdm_v2_gold_pipeline.py"
V2_PIPELINE_SHA256 = (
    "2a36859e03d2a59bb26489458eb7f867663c800902999ec8c9b09ac45087d435"
)
V2_VALIDATOR_MEMBER = "scripts/privacy_claim_validator.py"
V2_VALIDATOR_SHA256 = (
    "a2f745e834ad79f2342a251acc0281b3b9e729c0cf26d0a40bb45e5600addb27"
)
V4_PIPELINE_MEMBER = "scripts/wisdm_v2_gold_pipeline.py"
V4_PIPELINE_SHA256 = (
    "be21ff288cf1e9f24966a48352cf289737301b904d3883e1edf764d44cb6526b"
)
V4_VALIDATOR_MEMBER = "scripts/privacy_claim_validator.py"
V4_VALIDATOR_SHA256 = (
    "24d75408f656fcf1d586884ae5229ab8422186450d53cf68fc86aefb6d309f1a"
)
V4_TESTS_MEMBER = "tests/test_privacy_claim_validator.py"
V4_TESTS_SHA256 = (
    "36acb525dbbe026cb45374f324e2ca2edf4bf002f18c922b33147876db7e9049"
)
V4_REGISTRY_MEMBER = "docs/mechanism_registry_v2_0.json"
V4_REGISTRY_SHA256 = (
    "49a56d8f60f5373eb766a46b2bb4cc596086bfd4f8ecdb764e5a16beab5a2d05"
)
LEGACY_UCI_SEQUENCE = "scripts/opacus_uci_har_sequence_dpsgd_demo.py"
LEGACY_UCI_SEQUENCE_SHA256 = (
    "e4ac2c83e601474270652da64e3efe176864491b0883126a5586560d03b5b677"
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
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256_bytes(payload)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty retrospective corpus.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def file_source(
    path: str,
    sha256: str,
    anchors: Sequence[str],
    absent: Sequence[str] = (),
) -> Dict[str, Any]:
    return {
        "kind": "file",
        "path": path,
        "sha256": sha256,
        "anchors": list(anchors),
        "absent_anchors": list(absent),
    }


def zip_source(
    archive: str,
    archive_sha256: str,
    member: str,
    member_sha256: str,
    anchors: Sequence[str],
    absent: Sequence[str] = (),
) -> Dict[str, Any]:
    return {
        "kind": "zip_member",
        "archive": archive,
        "archive_sha256": archive_sha256,
        "member": member,
        "member_sha256": member_sha256,
        "anchors": list(anchors),
        "absent_anchors": list(absent),
    }


V4_TEST_CONTROL = lambda *anchors: zip_source(  # noqa: E731
    V4_ZIP,
    V4_ZIP_SHA256,
    V4_TESTS_MEMBER,
    V4_TESTS_SHA256,
    anchors,
)
V4_VALIDATOR_CONTROL = lambda *anchors: zip_source(  # noqa: E731
    V4_ZIP,
    V4_ZIP_SHA256,
    V4_VALIDATOR_MEMBER,
    V4_VALIDATOR_SHA256,
    anchors,
)
V4_PIPELINE_CONTROL = lambda *anchors: zip_source(  # noqa: E731
    V4_ZIP,
    V4_ZIP_SHA256,
    V4_PIPELINE_MEMBER,
    V4_PIPELINE_SHA256,
    anchors,
)


CASES: Sequence[Dict[str, Any]] = (
    {
        "id": "R01",
        "root_cause": "legacy_verdict_ignored_mapping_sampler_schedule_validity",
        "pre_repair_effect": "unsafe_positive_possible",
        "discovery": SOURCE_AUDIT,
        "discovery_sha256": SOURCE_AUDIT_SHA256,
        "discovery_anchor": "The certificate builder is not fail-closed",
        "historical": zip_source(
            LEGACY_ZIP,
            LEGACY_ZIP_SHA256,
            LEGACY_BUILDER_MEMBER,
            LEGACY_BUILDER_SHA256,
            (
                "sampler_check(sampler, accountant_assumption)",
                '"mapping_hash_status": mapping_hash_status(audit_row, privacy_report)',
                "elif verdict in DIRECT_VERDICTS:",
                "supported_statement = (",
            ),
        ),
        "controls": (
            V4_TEST_CONTROL(
                "def test_mapping_bytes_mutation_blocks",
                "def test_sampler_accountant_mismatch_blocks_when_rebound",
                "def test_private_adaptive_schedule_blocks",
            ),
        ),
        "current_status": "blocked_by_hardened_validator",
    },
    {
        "id": "R02",
        "root_cause": "unvalidated_external_conversion_attachment",
        "pre_repair_effect": "fabricated_numeric_claim_possible",
        "discovery": SOURCE_AUDIT,
        "discovery_sha256": SOURCE_AUDIT_SHA256,
        "discovery_anchor": "fabricated external conversion with epsilon=-999 and delta=2",
        "historical": zip_source(
            LEGACY_ZIP,
            LEGACY_ZIP_SHA256,
            LEGACY_BUILDER_MEMBER,
            LEGACY_BUILDER_SHA256,
            (
                "elif external_conversion:",
                'verdict = "external_conversion_attached"',
                "external_epsilon = external_conversion.get",
            ),
        ),
        "controls": (
            V4_TEST_CONTROL(
                "def test_external_conversion_attachment_cannot_authorize",
                "EXTERNAL_CONVERSION_UNVERIFIED",
            ),
        ),
        "current_status": "blocked_by_hardened_validator",
    },
    {
        "id": "R03",
        "root_cause": "private_global_fitted_preprocessing_omitted_from_support",
        "pre_repair_effect": "local_support_claim_invalid",
        "discovery": SOURCE_AUDIT,
        "discovery_sha256": SOURCE_AUDIT_SHA256,
        "discovery_anchor": "The concrete DIRECT DP-SGD demos omit global preprocessing support",
        "historical": file_source(
            LEGACY_UCI_SEQUENCE,
            LEGACY_UCI_SEQUENCE_SHA256,
            (
                "mean = train_x.mean(axis=(0, 2), keepdims=True)",
                "std = train_x.std(axis=(0, 2), keepdims=True)",
            ),
        ),
        "controls": (
            V4_TEST_CONTROL(
                "def test_private_global_preprocessing_blocks",
                "PRIVATE_PREPROCESSING_UNACCOUNTED",
            ),
        ),
        "current_status": "blocked_by_hardened_validator",
    },
    {
        "id": "R04",
        "root_cause": "private_data_dependent_update_denominator",
        "pre_repair_effect": "registered_accountant_did_not_cover_executed_update",
        "discovery": FULL_REAUDIT,
        "discovery_sha256": FULL_REAUDIT_SHA256,
        "discovery_anchor": "The WISDM update uses a private-data-dependent denominator",
        "historical": zip_source(
            V2_ZIP,
            V2_ZIP_SHA256,
            V2_PIPELINE_MEMBER,
            V2_PIPELINE_SHA256,
            (
                "expected_batch = sample_rate * len(train_x)",
                "weights -= learning_rate * (summed_gradient + noise) / expected_batch",
                '"population_size_policy": "accountant_covered"',
            ),
        ),
        "controls": (
            V4_PIPELINE_CONTROL(
                "a public constant optimizer step with no dataset-size normalization",
                "weights -= optimizer_step_size * (summed_gradient + noise)",
            ),
            V4_TEST_CONTROL(
                "def test_data_dependent_update_normalization_blocks",
                "REGISTERED_MECHANISM_SEMANTICS_MISMATCH",
            ),
        ),
        "current_status": "repaired_and_mutation_blocked",
    },
    {
        "id": "R05",
        "root_cause": "one_draw_gaussian_overstated_as_release_grade",
        "pre_repair_effect": "assurance_overclaim",
        "discovery": FULL_REAUDIT,
        "discovery_sha256": FULL_REAUDIT_SHA256,
        "discovery_anchor": "A single floating-point Gaussian draw is called release-grade",
        "historical": zip_source(
            V2_ZIP,
            V2_ZIP_SHA256,
            V2_PIPELINE_MEMBER,
            V2_PIPELINE_SHA256,
            (
                "rng.normalvariate(0.0, standard_deviation) for _ in range(count)",
                '"rng_assurance": "release_grade_secure"',
            ),
        ),
        "controls": (
            V4_PIPELINE_CONTROL(
                "discard one full draw, then sum four and divide by two",
                "return (draw() + draw() + draw() + draw()) / 2.0",
            ),
            V4_TEST_CONTROL(
                "def test_universal_release_grade_rng_wording_is_rejected",
                "RNG_ASSURANCE_OVERCLAIM",
            ),
        ),
        "current_status": "repaired_and_overclaim_blocked",
    },
    {
        "id": "R06",
        "root_cause": "observed_owner_incidence_used_as_global_stability",
        "pre_repair_effect": "owner_K_unproved",
        "discovery": FULL_REAUDIT,
        "discovery_sha256": FULL_REAUDIT_SHA256,
        "discovery_anchor": "Observed owner incidence is incorrectly used as global stability K",
        "historical": zip_source(
            V2_ZIP,
            V2_ZIP_SHA256,
            V2_PIPELINE_MEMBER,
            V2_PIPELINE_SHA256,
            (
                '"stability_bound": int(mapping_evidence.owner_kappa or 0)',
                '"raw_adjacency": "owner_add_remove"',
            ),
        ),
        "controls": (
            V4_TEST_CONTROL(
                "def test_owner_observed_kappa_without_public_cap_blocks",
                "OWNER_PUBLIC_CAP_UNVERIFIED",
                "def test_owner_k_comes_from_public_cap_not_observed_maximum",
            ),
        ),
        "current_status": "public_cap_required_and_vacuous_claim_blocked",
    },
    {
        "id": "R07",
        "root_cause": "generic_replace_one_raw_event_domain",
        "pre_repair_effect": "event_K_not_licensed",
        "discovery": FULL_REAUDIT,
        "discovery_sha256": FULL_REAUDIT_SHA256,
        "discovery_anchor": "The raw-event adjacency domain is not defined tightly enough",
        "historical": zip_source(
            V2_ZIP,
            V2_ZIP_SHA256,
            V2_PIPELINE_MEMBER,
            V2_PIPELINE_SHA256,
            ('"raw_adjacency": "replace_one"',),
        ),
        "controls": (
            V4_TEST_CONTROL(
                "def test_generic_replace_one_event_domain_is_insufficient",
                "RAW_ADJACENCY_NOT_SUPPORTED",
            ),
            V4_PIPELINE_CONTROL(
                'RAW_EVENT_ADJACENCY = "fixed_owner_slot_payload_replace_one_v1"',
                'RAW_DOMAIN_ID = "fixed_owner_slot_payload_domain_v1"',
            ),
        ),
        "current_status": "registered_domain_required",
    },
    {
        "id": "R08",
        "root_cause": "mechanism_registry_did_not_bind_executable_semantics",
        "pre_repair_effect": "self_consistent_forgery_possible",
        "discovery": FULL_REAUDIT,
        "discovery_sha256": FULL_REAUDIT_SHA256,
        "discovery_anchor": "The mechanism registry does not bind executable semantics",
        "historical": zip_source(
            V2_ZIP,
            V2_ZIP_SHA256,
            V2_VALIDATOR_MEMBER,
            V2_VALIDATOR_SHA256,
            (
                'registered_mechanisms = {',
                '"secure_systemrandom_dpsgd"',
                "The opacus_rdp checker has no registered mechanism adapter",
            ),
            (
                "MECHANISM_REGISTRY_ENTRY_DIGEST_MISMATCH",
                "PIPELINE_CODE_REGISTRY_DIGEST_MISMATCH",
            ),
        ),
        "controls": (
            V4_TEST_CONTROL(
                "def test_self_consistent_forged_pipeline_code_hash_blocks",
                "PIPELINE_CODE_REGISTRY_DIGEST_MISMATCH",
                "def test_fake_rng_semantics_block_even_when_rebound",
            ),
            zip_source(
                V4_ZIP,
                V4_ZIP_SHA256,
                V4_REGISTRY_MEMBER,
                V4_REGISTRY_SHA256,
                (
                    '"code_artifact_id": "scripts/wisdm_v2_gold_pipeline.py"',
                    '"noise_generation": "normalvariate_discard1_sum4_div2_v1"',
                    '"update_normalization": "public_constant_step_no_dataset_denominator"',
                ),
            ),
        ),
        "current_status": "exact_source_and_semantics_bound",
    },
    {
        "id": "R09",
        "root_cause": "missing_runtime_still_authorized_conditional_positive",
        "pre_repair_effect": "post_execution_claim_without_execution_evidence",
        "discovery": FULL_REAUDIT,
        "discovery_sha256": FULL_REAUDIT_SHA256,
        "discovery_anchor": "Missing runtime is currently a positive case",
        "historical": zip_source(
            V2_ZIP,
            V2_ZIP_SHA256,
            V2_VALIDATOR_MEMBER,
            V2_VALIDATOR_SHA256,
            (
                'elif status in {"missing", "not_applicable"}:',
                "conditional = True",
                "Only evidence-conditional wording is allowed without a matched runtime trace.",
            ),
        ),
        "controls": (
            V4_TEST_CONTROL(
                "def test_missing_runtime_blocks_post_execution_wording",
                "RUNTIME_TRACE_NOT_BOUND",
            ),
            V4_VALIDATOR_CONTROL(
                'result.release_status = "BLOCKED_UNVERIFIED"',
                'result.assurance_status = "DIAGNOSTIC_ONLY"',
            ),
        ),
        "current_status": "blocked_without_matched_runtime",
    },
    {
        "id": "R10",
        "root_cause": "float_threshold_sampler_not_exact_accountant_rational",
        "pre_repair_effect": "exact_fail_closed_semantics_violated",
        "discovery": FULL_REAUDIT,
        "discovery_sha256": FULL_REAUDIT_SHA256,
        "discovery_anchor": "The Poisson sampler probability is not exactly the accountant q",
        "historical": zip_source(
            V2_ZIP,
            V2_ZIP_SHA256,
            V2_PIPELINE_MEMBER,
            V2_PIPELINE_SHA256,
            ("rng.random() < sample_rate",),
        ),
        "controls": (
            V4_PIPELINE_CONTROL(
                "rng.randrange(sample_rate_denominator)",
                "< sample_rate_numerator",
            ),
            V4_TEST_CONTROL(
                "def test_exact_sample_rate_mismatch_blocks",
                "EXACT_SAMPLE_RATE_MISMATCH",
            ),
        ),
        "current_status": "exact_rational_sampler_bound",
    },
)


def read_source(descriptor: Mapping[str, Any]) -> tuple[bytes, Dict[str, Any]]:
    kind = descriptor["kind"]
    if kind == "file":
        path = PROJECT_ROOT / descriptor["path"]
        data = path.read_bytes()
        actual_sha256 = sha256_bytes(data)
        if actual_sha256 != descriptor["sha256"]:
            raise RuntimeError(
                f"Source drift: {descriptor['path']} "
                f"{actual_sha256} != {descriptor['sha256']}"
            )
        identity = {
            "kind": kind,
            "path": descriptor["path"],
            "sha256": actual_sha256,
            "bytes": len(data),
        }
        return data, identity
    if kind == "zip_member":
        archive_path = PROJECT_ROOT / descriptor["archive"]
        actual_archive_sha256 = file_sha256(archive_path)
        if actual_archive_sha256 != descriptor["archive_sha256"]:
            raise RuntimeError(
                f"Archive drift: {descriptor['archive']} "
                f"{actual_archive_sha256} != {descriptor['archive_sha256']}"
            )
        with ZipFile(archive_path) as archive:
            data = archive.read(descriptor["member"])
        actual_member_sha256 = sha256_bytes(data)
        if actual_member_sha256 != descriptor["member_sha256"]:
            raise RuntimeError(
                f"Member drift: {descriptor['archive']}::{descriptor['member']} "
                f"{actual_member_sha256} != {descriptor['member_sha256']}"
            )
        identity = {
            "kind": kind,
            "archive": descriptor["archive"],
            "archive_sha256": actual_archive_sha256,
            "member": descriptor["member"],
            "member_sha256": actual_member_sha256,
            "bytes": len(data),
        }
        return data, identity
    raise ValueError(f"Unknown source kind: {kind!r}")


def authenticate_source(descriptor: Mapping[str, Any]) -> Dict[str, Any]:
    data, identity = read_source(descriptor)
    text = data.decode("utf-8")
    present = []
    for anchor in descriptor["anchors"]:
        if anchor not in text:
            raise RuntimeError(f"Missing source anchor {anchor!r} in {identity}")
        present.append(anchor)
    absent = []
    for anchor in descriptor.get("absent_anchors", ()):
        if anchor in text:
            raise RuntimeError(
                f"Forbidden historical anchor {anchor!r} unexpectedly present in {identity}"
            )
        absent.append(anchor)
    return {
        **identity,
        "verified_present_anchors": present,
        "verified_absent_anchors": absent,
    }


def authenticate_discovery(case: Mapping[str, Any]) -> Dict[str, Any]:
    path = PROJECT_ROOT / case["discovery"]
    actual_sha256 = file_sha256(path)
    if actual_sha256 != case["discovery_sha256"]:
        raise RuntimeError(
            f"Discovery record drift: {case['discovery']} "
            f"{actual_sha256} != {case['discovery_sha256']}"
        )
    text = path.read_text(encoding="utf-8")
    if case["discovery_anchor"] not in text:
        raise RuntimeError(
            f"Discovery anchor missing for {case['id']}: "
            f"{case['discovery_anchor']!r}"
        )
    return {
        "path": case["discovery"],
        "sha256": actual_sha256,
        "verified_anchor": case["discovery_anchor"],
    }


def build(output_dir: Path) -> Dict[str, Any]:
    protocol_path = PROJECT_ROOT / PROTOCOL
    if file_sha256(protocol_path) != PROTOCOL_SHA256:
        raise RuntimeError("The frozen v5 protocol changed.")

    verifier_path = PROJECT_ROOT / "scripts" / "verify_audit_retrospective_failures_v5.py"
    if not verifier_path.exists():
        raise RuntimeError("Independent retrospective verifier is missing.")

    incidents = []
    for case in CASES:
        discovery = authenticate_discovery(case)
        historical = authenticate_source(case["historical"])
        controls = [
            authenticate_source(descriptor) for descriptor in case["controls"]
        ]
        incidents.append(
            {
                "id": case["id"],
                "root_cause": case["root_cause"],
                "pre_repair_effect": case["pre_repair_effect"],
                "discovery": discovery,
                "historical_source": historical,
                "repaired_controls": controls,
                "current_status": case["current_status"],
                "evidence_class": "retrospective_system_lineage",
            }
        )

    ids = [incident["id"] for incident in incidents]
    if ids != [f"R{index:02d}" for index in range(1, 11)]:
        raise RuntimeError(f"Protocol inclusion set is incomplete: {ids}")

    report: Dict[str, Any] = {
        "schema_version": "audit_retrospective_failure_lineage_v1",
        "protocol": {
            "path": PROTOCOL,
            "sha256": PROTOCOL_SHA256,
            "fixed_before_construction": True,
        },
        "builder": {
            "path": "scripts/build_audit_retrospective_failures_v5.py",
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "independent_verifier": {
            "path": "scripts/verify_audit_retrospective_failures_v5.py",
            "sha256": file_sha256(verifier_path),
            "imports_production_validator": False,
        },
        "design": {
            "case_source": "pre_protocol_source_first_audits",
            "selection": "all_ten_fixed_root_causes",
            "controlled_probe_cases_mixed": False,
            "independent_submission_sample": False,
            "prevalence_estimate": False,
            "claim_boundary": (
                "Retrospective system-lineage failures test whether repaired "
                "controls localize known historical defects; they do not "
                "estimate external prevalence or verifier completeness."
            ),
        },
        "incidents": incidents,
        "aggregate": {
            "incident_count": len(incidents),
            "authenticated_historical_sources": len(incidents),
            "authenticated_repaired_control_sources": sum(
                len(incident["repaired_controls"]) for incident in incidents
            ),
            "all_discovered_before_protocol": True,
            "all_currently_repaired_or_blocked": True,
        },
    }
    report["content_sha256"] = canonical_json_sha256(report)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "retrospective_failure_lineage.json"
    write_json(report_path, report)
    rows = [
        {
            "id": incident["id"],
            "root_cause": incident["root_cause"],
            "pre_repair_effect": incident["pre_repair_effect"],
            "evidence_class": incident["evidence_class"],
            "discovery_record": incident["discovery"]["path"],
            "historical_source_kind": incident["historical_source"]["kind"],
            "historical_source": (
                incident["historical_source"].get("path")
                or (
                    f"{incident['historical_source']['archive']}::"
                    f"{incident['historical_source']['member']}"
                )
            ),
            "current_status": incident["current_status"],
            "repaired_control_sources": len(incident["repaired_controls"]),
        }
        for incident in incidents
    ]
    write_csv(output_dir / "incidents.csv", rows)

    summary = [
        "# Audit v5 retrospective development-failure lineage",
        "",
        "This is a protocol-fixed retrospective system-lineage corpus.",
        "It is separate from the 31 controlled probes and is not an independent",
        "sample of papers, a prevalence estimate, or a completeness result.",
        "",
        f"- Incidents: {report['aggregate']['incident_count']}",
        (
            "- Authenticated repaired-control sources: "
            f"{report['aggregate']['authenticated_repaired_control_sources']}"
        ),
        f"- Protocol SHA-256: `{PROTOCOL_SHA256}`",
        f"- Content SHA-256: `{report['content_sha256']}`",
        "",
        "| ID | Historical root cause | Current status |",
        "| --- | --- | --- |",
    ]
    for incident in incidents:
        summary.append(
            f"| {incident['id']} | {incident['root_cause']} | "
            f"{incident['current_status']} |"
        )
    summary.extend(
        [
            "",
            "All cases were discovered and byte-addressable before the v5",
            "protocol. No external prevalence claim is licensed.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text(
        "\n".join(summary),
        encoding="utf-8",
    )
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    report = build(output_dir)
    print(
        "Built retrospective lineage: "
        f"{report['aggregate']['incident_count']} incidents, "
        f"content={report['content_sha256']}"
    )


if __name__ == "__main__":
    main()
