"""Materialize and run the frozen 200-case validator conformance suite.

This V3 runner expands the retained V2 experiment into the predeclared
5 x 5 x 4 positive grid and 25 x 4 paired semantic mutations.  Expected
labels are declared here from the frozen protocol; they are never copied from
the production validator's output.  V2 artifacts remain untouched.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from jsonschema import Draft202012Validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_validator_ablation_v2 as v2  # noqa: E402


PROTOCOL_PATH = (
    PROJECT_ROOT
    / "paper_notes"
    / "AAAI27_AUDIT_V19_CONFORMANCE_200_PROTOCOL_V1_2026-07-28.md"
)
PROTOCOL_SHA256 = (
    "57b747f147f16569a43d8a4d361c0090cf99d1fab8f48a614dce7a7618abe811"
)
FROZEN_INPUT_HASHES = {
    "scripts/run_validator_ablation_v2.py": (
        "adf44933ee7cfda60830d5aaa2c79f588ee3f0f6e78a106540deb7732d0b44a8"
    ),
    "scripts/verify_validator_ablation_v2.py": (
        "83adf054faf556818e4e4d0c3157ac5a829810c56b95c29bff980a51155be5c7"
    ),
    "scripts/privacy_claim_validator.py": (
        "24d75408f656fcf1d586884ae5229ab8422186450d53cf68fc86aefb6d309f1a"
    ),
    "scripts/mapping_evidence.py": (
        "8acefe2d4b7c170f132ac6cbebcdf91f542d33493c88ca5bdaa4c55680ac07ca"
    ),
    "tests/test_privacy_claim_validator.py": (
        "36acb525dbbe026cb45374f324e2ca2edf4bf002f18c922b33147876db7e9049"
    ),
}

CONTEXT_NAMES = {
    1: "CANONICAL",
    2: "IRREGULAR",
    3: "DECOY_SCENARIO",
    4: "PERMUTED",
}
STRATA = {
    "S1": {
        "name": "DIRECT_WINDOW",
        "generated_unit": "window",
        "requested_unit": "window",
        "path": "DIRECT",
        "levels": (1, 2, 4, 8, 12),
    },
    "S2": {
        "name": "DIRECT_EVENT",
        "generated_unit": "event",
        "requested_unit": "event",
        "path": "DIRECT",
        "levels": (1, 2, 4, 8, 12),
    },
    "S3": {
        "name": "DIRECT_OWNER",
        "generated_unit": "owner",
        "requested_unit": "owner",
        "path": "DIRECT",
        "levels": (1, 2, 4, 8, 12),
    },
    "S4": {
        "name": "CONVERT_EVENT",
        "generated_unit": "window",
        "requested_unit": "event",
        "path": "CONVERT",
        "levels": (1, 2, 3, 4, 6),
    },
    "S5": {
        "name": "GROUP_OWNER",
        "generated_unit": "window",
        "requested_unit": "owner",
        "path": "GROUP",
        "levels": (2, 3, 5, 8, 13),
    },
}

BASE_EPSILON = 1.1053169171072421
BASE_DELTA = 1e-6


@dataclass(frozen=True)
class CaseMetadata:
    case_kind: str
    coverage_cell: str
    stratum_id: str
    stratum_name: str
    level: int
    context_id: str
    context_name: str
    generated_unit: str
    requested_unit: str
    accountant_metric: str
    selected_record_count: int
    observed_event_kappa: int
    observed_owner_kappa: int
    expected_k: int
    parent_provenance: str
    parent_case_id: str = ""
    operator_id: str = ""
    direct_mutation_paths: tuple[str, ...] = ()
    derived_rebind_paths: tuple[str, ...] = ()
    required_issue_codes: tuple[str, ...] = ()
    normative_anchors: tuple[str, ...] = ()


Mutation = Callable[[dict[str, object], dict[str, str], Path], None]


@dataclass(frozen=True)
class Operator:
    operator_id: str
    description: str
    failure_family: str
    expected_release_status: str
    required_issue_codes: tuple[str, ...]
    normative_anchors: tuple[str, ...]
    direct_mutation_paths: tuple[str, ...]
    derived_rebind_paths: tuple[str, ...]
    mutate: Mutation


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_frozen_inputs() -> None:
    if sha256_file(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise RuntimeError("The frozen V1 conformance protocol hash changed")
    for relative, expected in FROZEN_INPUT_HASHES.items():
        actual = sha256_file(PROJECT_ROOT / relative)
        if actual != expected:
            raise RuntimeError(
                f"Frozen Stage-1 input changed: {relative}: {actual} != {expected}"
            )


def selected_base_rows(
    stratum_id: str,
    level: int,
    target: int,
    *,
    irregular: bool,
) -> list[dict[str, str]]:
    spec = STRATA[stratum_id]
    unit = str(spec["generated_unit"])
    rows: list[dict[str, str]] = []
    if stratum_id == "S4":
        # All intervals contain a shared point; exact event incidence is target.
        for index in range(target):
            if irregular:
                start = 3 * index + 1
                end = 200 - 2 * index
            else:
                start = index
                end = 20 + target - index
            rows.append(
                {
                    "scenario": "gold",
                    "window_id": f"s4-l{level}-r{index:02d}",
                    "generated_unit": "window",
                    "owner_id": "u0",
                    "owner_ids": "u0",
                    "start": str(start),
                    "end": str(end),
                }
            )
        return rows

    if stratum_id == "S5":
        # Exactly target selected generated records belong to the same owner.
        for index in range(target):
            if irregular:
                start = index * index + 3 * index + 1
                width = 1 + index % 3
            else:
                start = 2 * index
                width = 2
            rows.append(
                {
                    "scenario": "gold",
                    "window_id": f"s5-l{level}-r{index:02d}",
                    "generated_unit": "window",
                    "owner_id": "u0",
                    "owner_ids": "u0",
                    "start": str(start),
                    "end": str(start + width),
                }
            )
        return rows

    # Native/direct strata vary selected-record count without changing K=1.
    for index in range(target):
        if irregular:
            start = index * index + 4 * index + 1
            width = 1 + index % 3
        else:
            start = 2 * index
            width = 2
        owner = f"u{index}" if unit == "owner" else f"u{index % 3}"
        rows.append(
            {
                "scenario": "gold",
                "window_id": f"{stratum_id.lower()}-l{level}-r{index:02d}",
                "generated_unit": unit,
                "owner_id": owner,
                "owner_ids": owner,
                "start": str(start),
                "end": str(start + width),
            }
        )
    return rows


def contextualize_rows(
    selected: list[dict[str, str]], context: int
) -> list[dict[str, str]]:
    rows = copy.deepcopy(selected)
    if context == 1 or context == 2:
        return rows
    if context == 3:
        decoys = [
            {
                "scenario": "decoy",
                "window_id": f"decoy-{index:02d}",
                "generated_unit": "window",
                "owner_id": f"z{index}",
                "owner_ids": f"z{index}",
                "start": str(500 + 5 * index),
                "end": str(502 + 5 * index),
            }
            for index in range(3)
        ]
        return rows + decoys
    if context == 4:
        output = list(reversed(rows))
        for row in output:
            row["start"] = str(int(row["start"])).zfill(5)
            row["end"] = str(int(row["end"])).zfill(5)
        return output
    raise ValueError(context)


def positive_rows(
    stratum_id: str, level: int, context: int
) -> tuple[list[dict[str, str]], int]:
    target = int(STRATA[stratum_id]["levels"][level - 1])
    selected = selected_base_rows(
        stratum_id,
        level,
        target,
        irregular=context == 2,
    )
    return contextualize_rows(selected, context), target


def positive_truth_anchor(stratum_id: str) -> str:
    return {
        "S1": "T01",
        "S2": "T06,T27",
        "S3": "T10,T27",
        "S4": "T07,T08",
        "S5": "T12",
    }[stratum_id]


def build_positive_cases(
    root: Path,
) -> tuple[list[v2.Case], dict[str, CaseMetadata]]:
    cases: list[v2.Case] = []
    metadata: dict[str, CaseMetadata] = {}
    for stratum_id, spec in STRATA.items():
        for level in range(1, 6):
            for context in range(1, 5):
                case_id = f"P3_{stratum_id}_L{level}_C{context}"
                rows, target = positive_rows(stratum_id, level, context)
                claim_unit = str(spec["requested_unit"])
                expected_k = 1
                numeric_rule = "base"
                if stratum_id == "S4":
                    expected_k = 2 * target
                    numeric_rule = "group"
                elif stratum_id == "S5":
                    expected_k = target
                    numeric_rule = "group"
                record_policy = (
                    "owner_partition_ids"
                    if stratum_id in {"S3", "S5"}
                    else "public_fixed_ids"
                )
                case = v2.make_case(
                    root,
                    case_id=case_id,
                    description=(
                        f"Frozen {spec['name']} positive cell L{level}/C{context}."
                    ),
                    truth_row=positive_truth_anchor(stratum_id),
                    claim_unit=claim_unit,
                    failure_family="positive",
                    expected_release_status="ALLOWED",
                    expected_unit_path=str(spec["path"]),
                    expected_assurance_status="EVIDENCE_VALIDATED",
                    expected_numeric_rule=numeric_rule,
                    rows=rows,
                    record_identity_policy=record_policy,
                )
                evidence = v2.inspect_mapping_evidence(case.mapping_path, "gold")
                if not evidence.valid:
                    raise RuntimeError(f"Invalid generated mapping {case_id}")
                observed_event = int(evidence.event_kappa or 0)
                observed_owner = int(evidence.owner_kappa or 0)
                if evidence.generated_unit != spec["generated_unit"]:
                    raise RuntimeError(f"Generated unit mismatch in {case_id}")
                if stratum_id == "S4" and observed_event != target:
                    raise RuntimeError(
                        f"Event kappa mismatch in {case_id}: {observed_event} != {target}"
                    )
                if stratum_id == "S5" and observed_owner != target:
                    raise RuntimeError(
                        f"Owner kappa mismatch in {case_id}: {observed_owner} != {target}"
                    )
                if case.expected_delta is not None and case.expected_delta >= 1:
                    raise RuntimeError(f"Positive conversion is vacuous: {case_id}")
                cell = f"{stratum_id}:L{level}:C{context}"
                metadata[case_id] = CaseMetadata(
                    case_kind="allowed",
                    coverage_cell=cell,
                    stratum_id=stratum_id,
                    stratum_name=str(spec["name"]),
                    level=level,
                    context_id=f"C{context}",
                    context_name=CONTEXT_NAMES[context],
                    generated_unit=str(spec["generated_unit"]),
                    requested_unit=claim_unit,
                    accountant_metric=(
                        "owner_add_remove"
                        if spec["generated_unit"] == "owner"
                        else "add_remove"
                    ),
                    selected_record_count=int(evidence.selected_record_count),
                    observed_event_kappa=observed_event,
                    observed_owner_kappa=observed_owner,
                    expected_k=expected_k,
                    parent_provenance="parent_free_frozen_grid",
                    normative_anchors=tuple(
                        positive_truth_anchor(stratum_id).split(",")
                    ),
                )
                cases.append(case)
    return cases, metadata


REBIND_DERIVED = (
    "pipeline.manifest.mechanism_sha256",
    "pipeline.mapping_file_sha256",
    "pipeline.selected_mapping_sha256",
    "pipeline.pipeline_sha256",
    "runtime_trace",
)


def read_mapping_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def mutation_operators() -> dict[str, Operator]:
    def rebind(report: dict[str, object], mapping: Path) -> None:
        v2.rebind_report(report, mapping)

    def o01(report: dict[str, object], _audit: dict[str, str], _mapping: Path) -> None:
        report["runtime_trace"] = {"status": "missing"}

    def o02(report: dict[str, object], _audit: dict[str, str], mapping: Path) -> None:
        privacy = report["privacy"]
        assert isinstance(privacy, dict)
        privacy["delta"] = 0.1
        privacy["epsilon"] = v2.recompute_opacus_epsilon(delta=0.1)
        rebind(report, mapping)

    def o03(_report: dict[str, object], _audit: dict[str, str], mapping: Path) -> None:
        rows = read_mapping_rows(mapping)
        selected = next(row for row in rows if row["scenario"] == "gold")
        selected["end"] = str(int(selected["end"]) + 1)
        v2.write_mapping(mapping, rows)

    def o04(report: dict[str, object], _audit: dict[str, str], mapping: Path) -> None:
        report["mechanism"]["steps"] = True
        rebind(report, mapping)

    def o05(report: dict[str, object], _audit: dict[str, str], mapping: Path) -> None:
        report["mechanism"]["sampler_law"] = "shuffle"
        rebind(report, mapping)

    def o06(report: dict[str, object], _audit: dict[str, str], _mapping: Path) -> None:
        runtime = report["runtime_trace"]
        runtime["noise_multiplier"] = 9.0
        runtime["trace_hash"] = v2.canonical_json_sha256(v2.runtime_payload(report))

    def o07(report: dict[str, object], _audit: dict[str, str], mapping: Path) -> None:
        report["privacy"]["epsilon"] = float(report["privacy"]["epsilon"]) + 0.01
        rebind(report, mapping)

    def o08(report: dict[str, object], _audit: dict[str, str], mapping: Path) -> None:
        report["mechanism"]["secure_mode"] = False
        rebind(report, mapping)

    def o09(report: dict[str, object], _audit: dict[str, str], mapping: Path) -> None:
        report["mechanism"]["secure_mode"] = False
        report["rng_assurance"] = "research_prng"
        rebind(report, mapping)

    def o10(report: dict[str, object], _audit: dict[str, str], mapping: Path) -> None:
        report["pipeline"]["manifest"]["preprocessing"][0][
            "classification"
        ] = "private_global"
        rebind(report, mapping)

    def o11(report: dict[str, object], _audit: dict[str, str], mapping: Path) -> None:
        report["pipeline"]["manifest"]["influence_support"][
            "components"
        ].remove("labels")
        rebind(report, mapping)

    def o12(report: dict[str, object], _audit: dict[str, str], _mapping: Path) -> None:
        contract = report["claim_contracts"]["event"]
        contract["schedule_mode"] = "adaptive_private"
        contract["schedule_evidence_status"] = "unverified"

    def o13(report: dict[str, object], _audit: dict[str, str], _mapping: Path) -> None:
        report["claim_contracts"]["event"][
            "conversion_method"
        ] = "external_json_claim"

    def o14(report: dict[str, object], _audit: dict[str, str], mapping: Path) -> None:
        report["pipeline"]["manifest"]["model_selection_status"] = "not_covered"
        rebind(report, mapping)

    def o15(report: dict[str, object], _audit: dict[str, str], mapping: Path) -> None:
        report["mechanism"]["accountant_id"] = "plausible_external_rdp"
        rebind(report, mapping)

    def o16(report: dict[str, object], _audit: dict[str, str], mapping: Path) -> None:
        current = report["pipeline"]["manifest"]["generated_record_unit"]
        replacement = {"window": "event", "event": "owner", "owner": "window"}[
            str(current)
        ]
        report["pipeline"]["manifest"]["generated_record_unit"] = replacement
        rebind(report, mapping)

    def o17(report: dict[str, object], _audit: dict[str, str], _mapping: Path) -> None:
        report["claim_contracts"]["event"]["stability_status"] = "unverified"

    def o18(report: dict[str, object], _audit: dict[str, str], mapping: Path) -> None:
        report["pipeline"]["manifest"]["code_sha256"] = "0" * 64
        rebind(report, mapping)

    def o19(report: dict[str, object], _audit: dict[str, str], mapping: Path) -> None:
        report["mechanism"]["secure_rng_backend"] = "attacker.fake_rng"
        rebind(report, mapping)

    def o20(report: dict[str, object], _audit: dict[str, str], mapping: Path) -> None:
        report["mechanism"]["noise_generation"] = "one_draw_unhardened"
        report["mechanism"]["noise_hardening"] = "none"
        rebind(report, mapping)

    def o21(report: dict[str, object], _audit: dict[str, str], mapping: Path) -> None:
        report["mechanism"][
            "update_normalization"
        ] = "divide_by_sample_rate_times_private_dataset_size"
        rebind(report, mapping)

    def o22(report: dict[str, object], _audit: dict[str, str], mapping: Path) -> None:
        policy = report["pipeline"]["manifest"]["contribution_policy"]
        policy["id"] = "no_additional_cap"
        policy["classification"] = "not_applicable"
        policy["parameters"] = {"cap": None}
        policy["parameters_sha256"] = v2.canonical_json_sha256(policy["parameters"])
        rebind(report, mapping)

    def o23(report: dict[str, object], _audit: dict[str, str], _mapping: Path) -> None:
        report["claim_contracts"]["event"]["raw_adjacency"] = "replace_one"

    def o24(report: dict[str, object], _audit: dict[str, str], mapping: Path) -> None:
        report["mechanism"]["sample_rate_numerator"] = 2
        rebind(report, mapping)

    def o25(report: dict[str, object], _audit: dict[str, str], _mapping: Path) -> None:
        runtime = report["runtime_trace"]
        runtime["evidence"]["released_model_sha256"] = "3" * 64
        runtime["runtime_evidence_sha256"] = v2.canonical_json_sha256(
            runtime["evidence"]
        )
        runtime["trace_hash"] = v2.canonical_json_sha256(v2.runtime_payload(report))

    rows = [
        ("O01", "runtime trace missing", "runtime", "BLOCKED_UNVERIFIED", ("RUNTIME_TRACE_NOT_BOUND",), ("T02",), ("runtime_trace",), (), o01),
        ("O02", "owner conversion made vacuous", "conversion_vacuity", "BLOCKED_VACUOUS", (), ("T13",), ("privacy.epsilon", "privacy.delta"), REBIND_DERIVED, o02),
        ("O03", "mapping bytes changed after binding", "mapping_binding", "BLOCKED_INVALID", ("ACTUAL_MAPPING_DIGEST_MISMATCH",), ("T28",), ("mapping.rows[gold][0].end",), (), o03),
        ("O04", "Boolean in integer steps", "schema", "BLOCKED_INVALID", ("REPORT_SCHEMA_INVALID",), ("T04",), ("mechanism.steps",), REBIND_DERIVED, o04),
        ("O05", "sampler/accountant mismatch", "mechanism", "BLOCKED_INVALID", ("SAMPLER_ACCOUNTANT_MISMATCH",), ("T03",), ("mechanism.sampler_law",), REBIND_DERIVED, o05),
        ("O06", "runtime noise mismatch", "runtime", "BLOCKED_INVALID", ("RUNTIME_NUMERIC_FIELD_MISMATCH",), ("T30",), ("runtime_trace.noise_multiplier",), ("runtime_trace.trace_hash",), o06),
        ("O07", "epsilon differs from recomputation", "accountant", "BLOCKED_INVALID", ("ACCOUNTANT_EPSILON_MISMATCH",), ("T36",), ("privacy.epsilon",), REBIND_DERIVED, o07),
        ("O08", "secure mode false under release assurance", "rng", "BLOCKED_INVALID", ("RELEASE_RNG_SECURE_MODE_MISMATCH",), ("T32",), ("mechanism.secure_mode",), REBIND_DERIVED, o08),
        ("O09", "research PRNG", "rng", "RESEARCH_ONLY", ("RNG_RESEARCH_ONLY",), ("T32",), ("mechanism.secure_mode", "rng_assurance"), REBIND_DERIVED, o09),
        ("O10", "private-global preprocessing", "private_dependency", "BLOCKED_INVALID", ("PRIVATE_PREPROCESSING_UNACCOUNTED",), ("T16",), ("pipeline.manifest.preprocessing[0].classification",), REBIND_DERIVED, o10),
        ("O11", "incomplete influence support", "stability", "BLOCKED_UNVERIFIED", ("INFLUENCE_SUPPORT_MANIFEST_INCOMPLETE",), ("T29",), ("pipeline.manifest.influence_support.components",), REBIND_DERIVED, o11),
        ("O12", "private adaptive schedule", "schedule", "BLOCKED_UNVERIFIED", ("SCHEDULE_MODE_UNREGISTERED",), ("T20", "T22"), ("claim_contracts.event.schedule_mode", "claim_contracts.event.schedule_evidence_status"), (), o12),
        ("O13", "external conversion attachment", "external_attachment", "BLOCKED_UNVERIFIED", ("EXTERNAL_CONVERSION_UNVERIFIED",), ("T25", "T26"), ("claim_contracts.event.conversion_method",), (), o13),
        ("O14", "private model selection not covered", "private_dependency", "BLOCKED_INVALID", ("MODEL_SELECTION_NOT_COVERED",), ("T31",), ("pipeline.manifest.model_selection_status",), REBIND_DERIVED, o14),
        ("O15", "unregistered accountant and registry mismatch", "accountant", "BLOCKED_INVALID", ("REGISTERED_MECHANISM_SEMANTICS_MISMATCH", "ACCOUNTANT_CHECKER_UNREGISTERED"), ("T35",), ("mechanism.accountant_id",), REBIND_DERIVED, o15),
        ("O16", "manifest generated-unit mismatch", "unit_binding", "BLOCKED_INVALID", ("ACTUAL_GENERATED_UNIT_MISMATCH",), ("T37",), ("pipeline.manifest.generated_record_unit",), REBIND_DERIVED, o16),
        ("O17", "event stability unverified", "stability", "BLOCKED_UNVERIFIED", ("STABILITY_STATUS_UNVERIFIED",), ("T05",), ("claim_contracts.event.stability_status",), (), o17),
        ("O18", "forged code digest", "executable_binding", "BLOCKED_INVALID", ("PIPELINE_CODE_REGISTRY_DIGEST_MISMATCH",), ("R01",), ("pipeline.manifest.code_sha256",), REBIND_DERIVED, o18),
        ("O19", "unregistered RNG backend", "executable_binding", "BLOCKED_INVALID", ("REGISTERED_MECHANISM_SEMANTICS_MISMATCH",), ("R02",), ("mechanism.secure_rng_backend",), REBIND_DERIVED, o19),
        ("O20", "unhardened one-draw Gaussian", "rng", "BLOCKED_INVALID", ("REGISTERED_MECHANISM_SEMANTICS_MISMATCH",), ("R03",), ("mechanism.noise_generation", "mechanism.noise_hardening"), REBIND_DERIVED, o20),
        ("O21", "data-dependent normalization", "mechanism", "BLOCKED_INVALID", ("REGISTERED_MECHANISM_SEMANTICS_MISMATCH",), ("R04",), ("mechanism.update_normalization",), REBIND_DERIVED, o21),
        ("O22", "absent public owner cap", "stability", "BLOCKED_UNVERIFIED", ("OWNER_PUBLIC_CAP_UNVERIFIED",), ("R05",), ("pipeline.manifest.contribution_policy.id", "pipeline.manifest.contribution_policy.classification", "pipeline.manifest.contribution_policy.parameters", "pipeline.manifest.contribution_policy.parameters_sha256"), REBIND_DERIVED, o22),
        ("O23", "generic raw event adjacency", "stability", "BLOCKED_INVALID", ("RAW_ADJACENCY_NOT_SUPPORTED",), ("R06",), ("claim_contracts.event.raw_adjacency",), (), o23),
        ("O24", "exact sample-rate mismatch", "mechanism", "BLOCKED_INVALID", ("EXACT_SAMPLE_RATE_MISMATCH",), ("R07",), ("mechanism.sample_rate_numerator",), REBIND_DERIVED, o24),
        ("O25", "runtime artifact digest mismatch", "runtime", "BLOCKED_INVALID", ("RUNTIME_ARTIFACT_DIGEST_MISMATCH",), ("R08",), ("runtime_trace.evidence.released_model_sha256",), ("runtime_trace.runtime_evidence_sha256", "runtime_trace.trace_hash"), o25),
    ]
    return {row[0]: Operator(*row) for row in rows}


def cells(stratum_id: str) -> list[str]:
    return [
        f"{stratum_id}:L{level}:C{context}"
        for level in range(1, 6)
        for context in range(1, 5)
    ]


def frozen_parent_schedule() -> dict[str, list[str]]:
    assignments: dict[str, list[str]] = defaultdict(list)
    for operator_id, level in (
        ("O11", 1),
        ("O12", 2),
        ("O13", 3),
        ("O17", 4),
        ("O23", 5),
    ):
        assignments[operator_id].extend(
            f"S4:L{level}:C{context}" for context in range(1, 5)
        )
    assignments["O22"].extend(f"S5:L1:C{context}" for context in range(1, 5))
    assignments["O02"].extend(f"S5:L5:C{context}" for context in range(1, 5))

    cross_path = (
        "O01",
        "O03",
        "O04",
        "O05",
        "O06",
        "O07",
        "O08",
        "O09",
        "O10",
        "O14",
        "O15",
        "O25",
    )
    s5_remaining = [
        f"S5:L{level}:C{context}"
        for level in range(2, 5)
        for context in range(1, 5)
    ]
    for index, operator_id in enumerate(cross_path):
        for stratum_id in ("S1", "S2", "S3"):
            assignments[operator_id].append(cells(stratum_id)[index])
        assignments[operator_id].append(s5_remaining[index])

    remaining = ("O16", "O18", "O19", "O20", "O21", "O24")
    extras = (
        "S1:L5:C3",
        "S1:L5:C4",
        "S2:L5:C3",
        "S2:L5:C4",
        "S3:L5:C3",
        "S3:L5:C4",
    )
    for offset, (operator_id, extra) in enumerate(zip(remaining, extras, strict=True)):
        index = 12 + offset
        for stratum_id in ("S1", "S2", "S3"):
            assignments[operator_id].append(cells(stratum_id)[index])
        assignments[operator_id].append(extra)

    operators = mutation_operators()
    if set(assignments) != set(operators):
        raise RuntimeError("Parent schedule does not cover all operators")
    if any(len(values) != 4 for values in assignments.values()):
        raise RuntimeError("Every operator must have exactly four parents")
    used = [cell for values in assignments.values() for cell in values]
    expected = [cell for stratum_id in STRATA for cell in cells(stratum_id)]
    if Counter(used) != Counter(expected):
        raise RuntimeError("Frozen schedule must consume every positive cell once")
    return dict(assignments)


def build_negative_cases(
    root: Path,
    positives: Sequence[v2.Case],
    positive_metadata: Mapping[str, CaseMetadata],
) -> tuple[list[v2.Case], dict[str, CaseMetadata]]:
    operators = mutation_operators()
    schedule = frozen_parent_schedule()
    by_cell = {
        positive_metadata[case.case_id].coverage_cell: case for case in positives
    }
    cases: list[v2.Case] = []
    metadata: dict[str, CaseMetadata] = {}
    for operator_id in sorted(schedule):
        operator = operators[operator_id]
        for parent_cell in schedule[operator_id]:
            parent = by_cell[parent_cell]
            parent_meta = positive_metadata[parent.case_id]
            parent_report = json.loads(parent.report_path.read_text(encoding="utf-8"))
            record_policy = str(
                parent_report["pipeline"]["manifest"]["record_identity_policy"]
            )
            negative_id = f"N3_{operator_id}_{parent.case_id.removeprefix('P3_')}"
            expected_path = "UNDERSPECIFIED"
            if operator_id == "O02":
                expected_path = "GROUP"
            elif operator_id == "O09":
                expected_path = parent.expected_unit_path
            expected_assurance = (
                "EXTERNAL_UNVERIFIED"
                if operator_id == "O13"
                else "DIAGNOSTIC_ONLY"
            )
            case = v2.make_case(
                root,
                case_id=negative_id,
                description=(
                    f"{operator.operator_id} paired mutation of {parent.case_id}: "
                    f"{operator.description}."
                ),
                truth_row=",".join(operator.normative_anchors),
                claim_unit=parent.claim_unit,
                failure_family=operator.failure_family,
                expected_release_status=operator.expected_release_status,
                expected_unit_path=expected_path,
                expected_assurance_status=expected_assurance,
                expected_issue_code=(
                    operator.required_issue_codes[0]
                    if operator.required_issue_codes
                    else ""
                ),
                expected_numeric_rule="none",
                rows=read_mapping_rows(parent.mapping_path),
                record_identity_policy=record_policy,
                mutate=operator.mutate,
            )
            metadata[negative_id] = CaseMetadata(
                case_kind="non_allow",
                coverage_cell=parent_meta.coverage_cell,
                stratum_id=parent_meta.stratum_id,
                stratum_name=parent_meta.stratum_name,
                level=parent_meta.level,
                context_id=parent_meta.context_id,
                context_name=parent_meta.context_name,
                generated_unit=parent_meta.generated_unit,
                requested_unit=parent_meta.requested_unit,
                accountant_metric=parent_meta.accountant_metric,
                selected_record_count=parent_meta.selected_record_count,
                observed_event_kappa=parent_meta.observed_event_kappa,
                observed_owner_kappa=parent_meta.observed_owner_kappa,
                expected_k=parent_meta.expected_k,
                parent_provenance="paired_frozen_allowed_parent",
                parent_case_id=parent.case_id,
                operator_id=operator_id,
                direct_mutation_paths=operator.direct_mutation_paths,
                derived_rebind_paths=operator.derived_rebind_paths,
                required_issue_codes=operator.required_issue_codes,
                normative_anchors=operator.normative_anchors,
            )
            cases.append(case)
    return cases, metadata


def structural_gate_errors(
    cases: Sequence[v2.Case], metadata: Mapping[str, CaseMetadata]
) -> list[str]:
    errors: list[str] = []
    allowed = [case for case in cases if metadata[case.case_id].case_kind == "allowed"]
    nonallow = [case for case in cases if metadata[case.case_id].case_kind == "non_allow"]
    if (len(allowed), len(nonallow)) != (100, 100):
        errors.append(f"case counts are {len(allowed)} allowed/{len(nonallow)} non-allow")
    if len({case.case_id for case in cases}) != len(cases):
        errors.append("case IDs are not unique")
    strata = Counter(metadata[case.case_id].stratum_id for case in allowed)
    if strata != Counter({key: 20 for key in STRATA}):
        errors.append(f"positive stratum counts differ: {dict(strata)}")
    operators = Counter(metadata[case.case_id].operator_id for case in nonallow)
    if operators != Counter({f"O{index:02d}": 4 for index in range(1, 26)}):
        errors.append(f"operator counts differ: {dict(operators)}")
    parents = [metadata[case.case_id].parent_case_id for case in nonallow]
    if Counter(parents) != Counter(case.case_id for case in allowed):
        errors.append("allowed parents are not consumed exactly once")
    expected_status = Counter(
        {
            "BLOCKED_INVALID": 68,
            "BLOCKED_UNVERIFIED": 24,
            "BLOCKED_VACUOUS": 4,
            "RESEARCH_ONLY": 4,
        }
    )
    observed_status = Counter(case.expected_release_status for case in nonallow)
    if observed_status != expected_status:
        errors.append(f"non-allow status arithmetic differs: {dict(observed_status)}")
    triple_hashes: list[str] = []
    for case in cases:
        triple = [
            sha256_file(case.mapping_path),
            sha256_file(case.report_path),
            sha256_file(case.audit_path),
        ]
        triple_hashes.append(canonical_json_sha256(triple))
    if len(set(triple_hashes)) != len(triple_hashes):
        errors.append("case input triples are not unique")
    return errors


def prediction_gate_errors(
    cases: Sequence[v2.Case],
    metadata: Mapping[str, CaseMetadata],
    rows: Sequence[Mapping[str, object]],
    aggregate_rows: Sequence[Mapping[str, object]],
) -> list[str]:
    errors: list[str] = []
    full_rows = [row for row in rows if row["method"] == "FULL"]
    if len(full_rows) != len(cases):
        errors.append("FULL row count differs from case count")
        return errors
    if not all(bool(row["exact_tuple"]) for row in full_rows):
        failed = [str(row["case_id"]) for row in full_rows if not row["exact_tuple"]]
        errors.append(f"FULL expected-tuple failures: {failed}")
    if any(bool(row["unsafe_positive"]) for row in full_rows):
        errors.append("FULL emitted an unsafe positive")
    by_case = {case.case_id: case for case in cases}
    for row in full_rows:
        meta = metadata[str(row["case_id"])]
        issues = set(str(row["issue_codes"]).split(";"))
        missing = set(meta.required_issue_codes) - issues
        if missing:
            errors.append(f"{row['case_id']} missing required diagnostics {sorted(missing)}")
        if meta.case_kind == "non_allow":
            case = by_case[str(row["case_id"])]
            report = json.loads(case.report_path.read_text(encoding="utf-8"))
            audit = json.loads(case.audit_path.read_text(encoding="utf-8"))
            raw = v2.validate_privacy_claim(
                audit, report, case.claim_unit, mapping_path=case.mapping_path
            )
            if raw.supported_statement:
                errors.append(f"{case.case_id} retained forbidden positive wording")
    for method in v2.SIMPLE_BASELINES:
        row = next(item for item in aggregate_rows if item["method"] == method)
        if int(row["unsafe_positives"]) == 0:
            errors.append(f"{method} has no demonstrated unsafe positive")
    case_to_operator = {
        case_id: meta.operator_id for case_id, meta in metadata.items()
    }
    a1_unsafe = [
        row
        for row in rows
        if row["method"] == "A1_FULL_MINUS_RUNTIME"
        and bool(row["unsafe_positive"])
        and case_to_operator[str(row["case_id"])] in {"O01", "O25"}
    ]
    if not a1_unsafe:
        errors.append("A1 has no runtime/artifact-dependent unsafe positive")
    a2_unsafe = [
        row
        for row in rows
        if row["method"] == "A2_FULL_MINUS_VACUITY"
        and bool(row["unsafe_positive"])
        and case_to_operator[str(row["case_id"])] == "O02"
    ]
    if not a2_unsafe:
        errors.append("A2 has no vacuity-dependent unsafe positive")
    return errors


def case_records(
    cases: Sequence[v2.Case], metadata: Mapping[str, CaseMetadata]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for case in cases:
        record = asdict(case)
        record.update(asdict(metadata[case.case_id]))
        hashes: list[str] = []
        for key in ("mapping_path", "report_path", "audit_path"):
            path = Path(record[key])
            digest = sha256_file(path)
            hashes.append(digest)
            record[key] = path.relative_to(PROJECT_ROOT).as_posix()
            record[f"{key}_sha256"] = digest
        record["input_triple_sha256"] = canonical_json_sha256(hashes)
        records.append(record)
    return records


def breakdown_rows(
    prediction_rows: Sequence[Mapping[str, object]],
    metadata: Mapping[str, CaseMetadata],
    dimension: str,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    values = sorted(
        {
            str(getattr(metadata[str(row["case_id"])], dimension))
            for row in prediction_rows
        }
    )
    for method in v2.METHODS:
        for value in values:
            selected = [
                row
                for row in prediction_rows
                if row["method"] == method
                and str(getattr(metadata[str(row["case_id"])], dimension)) == value
            ]
            output.append(
                {
                    "method": method,
                    dimension: value,
                    "cases": len(selected),
                    "unsafe_positives": sum(bool(row["unsafe_positive"]) for row in selected),
                    "valid_exact_decisions": sum(bool(row["valid_decision_exact"]) for row in selected),
                    "exact_tuples": sum(bool(row["exact_tuple"]) for row in selected),
                }
            )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="reports/validator_ablation_v3_200_001"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite retained run directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    verify_frozen_inputs()

    positives, positive_metadata = build_positive_cases(output_dir)
    negatives, negative_metadata = build_negative_cases(
        output_dir, positives, positive_metadata
    )
    cases = positives + negatives
    metadata = {**positive_metadata, **negative_metadata}
    structural_errors = structural_gate_errors(cases, metadata)

    schema_path = PROJECT_ROOT / "docs" / "privacy_report_schema_v2_0.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    schema_validator = Draft202012Validator(schema)
    predictions = v2.build_prediction_rows(cases, schema_validator)
    aggregates = v2.aggregate(predictions)
    prediction_errors = prediction_gate_errors(cases, metadata, predictions, aggregates)
    gate_errors = structural_errors + prediction_errors

    cases_path = output_dir / "cases.json"
    predictions_path = output_dir / "predictions.csv"
    aggregate_path = output_dir / "aggregate_metrics.csv"
    coverage_path = output_dir / "coverage_matrix.csv"
    operator_path = output_dir / "operator_parent_matrix.csv"
    family_path = output_dir / "confusion_by_failure_family.csv"
    context_path = output_dir / "confusion_by_context.csv"
    stratum_path = output_dir / "confusion_by_stratum.csv"
    status_path = output_dir / "execution_status.json"
    summary_path = output_dir / "summary.md"

    v2.write_json(
        cases_path,
        {
            "schema_version": "validator_ablation_cases_v3.0",
            "protocol_path": PROTOCOL_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "protocol_sha256": PROTOCOL_SHA256,
            "ground_truth_source": (
                "Frozen 5x5x4 grid, operator catalog, and one-to-one parent "
                "schedule; labels are not copied from FULL."
            ),
            "cases": case_records(cases, metadata),
        },
    )
    v2.write_csv(predictions_path, predictions)
    v2.write_csv(aggregate_path, aggregates)
    coverage_records = [
        {
            "case_id": case.case_id,
            **asdict(metadata[case.case_id]),
        }
        for case in positives
    ]
    v2.write_csv(coverage_path, coverage_records)
    operator_records = [
        {
            "case_id": case.case_id,
            "parent_case_id": metadata[case.case_id].parent_case_id,
            "operator_id": metadata[case.case_id].operator_id,
            "parent_coverage_cell": metadata[case.case_id].coverage_cell,
            "parent_stratum": metadata[case.case_id].stratum_id,
            "parent_context": metadata[case.case_id].context_id,
            "expected_release_status": case.expected_release_status,
            "expected_unit_path": case.expected_unit_path,
            "required_issue_codes": ";".join(
                metadata[case.case_id].required_issue_codes
            ),
            "direct_mutation_paths": ";".join(
                metadata[case.case_id].direct_mutation_paths
            ),
            "derived_rebind_paths": ";".join(
                metadata[case.case_id].derived_rebind_paths
            ),
        }
        for case in negatives
    ]
    v2.write_csv(operator_path, operator_records)
    v2.write_csv(
        family_path,
        [
            {
                "method": method,
                "failure_family": family,
                "cases": len(selected),
                "unsafe_positives": sum(bool(row["unsafe_positive"]) for row in selected),
                "valid_exact_decisions": sum(bool(row["valid_decision_exact"]) for row in selected),
                "exact_tuples": sum(bool(row["exact_tuple"]) for row in selected),
            }
            for method in v2.METHODS
            for family in sorted({case.failure_family for case in cases})
            for selected in [[
                row for row in predictions
                if row["method"] == method and row["failure_family"] == family
            ]]
        ],
    )
    v2.write_csv(context_path, breakdown_rows(predictions, metadata, "context_id"))
    v2.write_csv(stratum_path, breakdown_rows(predictions, metadata, "stratum_id"))
    v2.write_json(
        status_path,
        {
            "schema_version": "validator_ablation_execution_v3.0",
            "gates_passed": not gate_errors,
            "gate_errors": gate_errors,
            "case_count": len(cases),
            "allowed_count": len(positives),
            "non_allow_count": len(negatives),
        },
    )

    lines = [
        "# Validator Conformance/Ablation V3 (Frozen 200 Cases)",
        "",
        "Designed coverage suite: 100 allowed and 100 paired non-allow cases.",
        "Fractions are deterministic case outcomes, not statistical estimates.",
        "",
        "| method | unsafe-positive fraction | valid-exact fraction | exact-tuple fraction |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in aggregates:
        lines.append(
            "| {method} | {unsafe_positives}/{blocked_truth_cases} ({unsafe_positive_rate:.2f}) "
            "| {valid_exact_decisions}/{valid_truth_cases} ({valid_exact_decision_rate:.2f}) "
            "| {exact_tuples}/{cases} ({exact_tuple_accuracy:.2f}) |".format(**row)
        )
    lines.extend(["", f"Execution gates passed: **{not gate_errors}**."])
    if gate_errors:
        lines.extend(["", "Gate failures:"] + [f"- {error}" for error in gate_errors])
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    source_paths = [
        Path(__file__).resolve(),
        PROJECT_ROOT / "scripts" / "verify_validator_ablation_v3.py",
        PROTOCOL_PATH,
        *[PROJECT_ROOT / relative for relative in FROZEN_INPUT_HASHES],
        PROJECT_ROOT / "docs" / "privacy_claim_contract_v2_0.md",
        PROJECT_ROOT / "docs" / "privacy_claim_truth_table_v2_0.md",
        schema_path,
        cases_path,
        predictions_path,
        aggregate_path,
        coverage_path,
        operator_path,
        family_path,
        context_path,
        stratum_path,
        status_path,
        summary_path,
    ]
    artifact_index = {
        "schema_version": "validator_ablation_artifact_v3.0",
        "files": [
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in source_paths
        ],
    }
    v2.write_json(output_dir / "artifact_index.json", artifact_index)
    if gate_errors:
        raise RuntimeError(
            "Frozen 200-case run retained with gate failures; see execution_status.json"
        )
    print(f"Wrote {summary_path}")
    print("FULL exact tuples: 200/200")


if __name__ == "__main__":
    main()
