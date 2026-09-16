"""Independent standard-library verifier for the strengthened V3.1 run.

This file imports no production, mapping, test, or runner module.  It owns a
separate frozen grid/operator oracle and recomputes mapping evidence, privacy
conversion, mutation closure, internal hash closure, registry/source binding,
prediction flags, aggregates, artifact membership, and V3 input lineage.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "validator_ablation_v3_1_200_001"
REFERENCE_DIR = PROJECT_ROOT / "reports" / "validator_ablation_v3_200_001"
REFERENCE_CASES_SHA256 = (
    "6ece4b3a5cf2a2fa4ac322939b59400e9cfd88396b8357516b80b7c4c610fce9"
)
PROTOCOL_PATH = "paper_notes/AAAI27_AUDIT_V19_CONFORMANCE_200_PROTOCOL_V1_2026-07-28.md"
PROTOCOL_SHA256 = "57b747f147f16569a43d8a4d361c0090cf99d1fab8f48a614dce7a7618abe811"
METHODS = (
    "B0_FIELD_PRESENCE",
    "B1_JSON_SCHEMA",
    "B2_MULTIPLICITY_ONLY",
    "B3_HASH_ONLY",
    "B4_ACCOUNTANT_ONLY",
    "A1_RUNTIME_ASSUMED_MATCHED",
    "A2_FULL_MINUS_VACUITY",
    "FULL",
)
BASE_EPSILON = 1.1053169171072421
BASE_DELTA = 1e-6
REGISTRY_KEY = "secure_systemrandom_dpsgd@2.0.0"
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
REBIND_DERIVED = (
    "pipeline.manifest.mechanism_sha256",
    "pipeline.mapping_file_sha256",
    "pipeline.selected_mapping_sha256",
    "pipeline.pipeline_sha256",
    "runtime_trace",
)
STRATA = {
    "S1": {
        "name": "DIRECT_WINDOW",
        "generated": "window",
        "requested": "window",
        "path": "DIRECT",
        "levels": (1, 2, 4, 8, 12),
        "anchors": ("T01",),
    },
    "S2": {
        "name": "DIRECT_EVENT",
        "generated": "event",
        "requested": "event",
        "path": "DIRECT",
        "levels": (1, 2, 4, 8, 12),
        "anchors": ("T06", "T27"),
    },
    "S3": {
        "name": "DIRECT_OWNER",
        "generated": "owner",
        "requested": "owner",
        "path": "DIRECT",
        "levels": (1, 2, 4, 8, 12),
        "anchors": ("T10", "T27"),
    },
    "S4": {
        "name": "CONVERT_EVENT",
        "generated": "window",
        "requested": "event",
        "path": "CONVERT",
        "levels": (1, 2, 3, 4, 6),
        "anchors": ("T07", "T08"),
    },
    "S5": {
        "name": "GROUP_OWNER",
        "generated": "window",
        "requested": "owner",
        "path": "GROUP",
        "levels": (2, 3, 5, 8, 13),
        "anchors": ("T12",),
    },
}
CONTEXT_NAMES = {
    "C1": "CANONICAL",
    "C2": "IRREGULAR",
    "C3": "DECOY_SCENARIO",
    "C4": "PERMUTED",
}


def op(
    status: str,
    family: str,
    required: tuple[str, ...],
    allowed: tuple[str, ...],
    anchors: tuple[str, ...],
    direct: tuple[str, ...],
    derived: tuple[str, ...] = (),
    path: str = "UNDERSPECIFIED",
    assurance: str = "DIAGNOSTIC_ONLY",
) -> dict[str, object]:
    return {
        "status": status,
        "family": family,
        "required": required,
        "allowed": allowed,
        "anchors": anchors,
        "direct": direct,
        "derived": derived,
        "path": path,
        "assurance": assurance,
    }


OPERATORS = {
    "O01": op(
        "BLOCKED_UNVERIFIED",
        "runtime",
        ("RUNTIME_TRACE_NOT_BOUND",),
        ("RUNTIME_TRACE_NOT_BOUND",),
        ("T02",),
        ("runtime_trace",),
    ),
    "O02": op(
        "BLOCKED_VACUOUS",
        "conversion_vacuity",
        (),
        (),
        ("T13",),
        ("privacy.epsilon", "privacy.delta"),
        REBIND_DERIVED,
        path="GROUP",
    ),
    "O03": op(
        "BLOCKED_INVALID",
        "mapping_binding",
        ("ACTUAL_MAPPING_DIGEST_MISMATCH",),
        ("ACTUAL_MAPPING_DIGEST_MISMATCH",),
        ("T28",),
        ("mapping.rows[gold][0].end",),
    ),
    "O04": op(
        "BLOCKED_INVALID",
        "schema",
        ("REPORT_SCHEMA_INVALID",),
        ("REPORT_SCHEMA_INVALID", "ACCOUNTANT_EPSILON_MISMATCH"),
        ("T04",),
        ("mechanism.steps",),
        REBIND_DERIVED,
    ),
    "O05": op(
        "BLOCKED_INVALID",
        "mechanism",
        ("SAMPLER_ACCOUNTANT_MISMATCH",),
        (
            "SAMPLER_ACCOUNTANT_MISMATCH",
            "REGISTERED_MECHANISM_SEMANTICS_MISMATCH",
            "REGISTERED_ACCOUNTANT_SAMPLER_MISMATCH",
        ),
        ("T03",),
        ("mechanism.sampler_law",),
        REBIND_DERIVED,
    ),
    "O06": op(
        "BLOCKED_INVALID",
        "runtime",
        ("RUNTIME_NUMERIC_FIELD_MISMATCH",),
        ("RUNTIME_NUMERIC_FIELD_MISMATCH",),
        ("T30",),
        ("runtime_trace.noise_multiplier",),
        ("runtime_trace.trace_hash",),
    ),
    "O07": op(
        "BLOCKED_INVALID",
        "accountant",
        ("ACCOUNTANT_EPSILON_MISMATCH",),
        ("ACCOUNTANT_EPSILON_MISMATCH",),
        ("T36",),
        ("privacy.epsilon",),
        REBIND_DERIVED,
    ),
    "O08": op(
        "BLOCKED_INVALID",
        "rng",
        ("RELEASE_RNG_SECURE_MODE_MISMATCH",),
        ("RELEASE_RNG_SECURE_MODE_MISMATCH",),
        ("T32",),
        ("mechanism.secure_mode",),
        REBIND_DERIVED,
    ),
    "O09": op(
        "RESEARCH_ONLY",
        "rng",
        ("RNG_RESEARCH_ONLY",),
        ("RNG_RESEARCH_ONLY",),
        ("T32",),
        ("mechanism.secure_mode", "rng_assurance"),
        REBIND_DERIVED,
        path="PARENT",
    ),
    "O10": op(
        "BLOCKED_INVALID",
        "private_dependency",
        ("PRIVATE_PREPROCESSING_UNACCOUNTED",),
        ("PRIVATE_PREPROCESSING_UNACCOUNTED",),
        ("T16",),
        ("pipeline.manifest.preprocessing[0].classification",),
        REBIND_DERIVED,
    ),
    "O11": op(
        "BLOCKED_UNVERIFIED",
        "stability",
        ("INFLUENCE_SUPPORT_MANIFEST_INCOMPLETE",),
        ("INFLUENCE_SUPPORT_MANIFEST_INCOMPLETE",),
        ("T29",),
        ("pipeline.manifest.influence_support.components",),
        REBIND_DERIVED,
    ),
    "O12": op(
        "BLOCKED_UNVERIFIED",
        "schedule",
        ("SCHEDULE_MODE_UNREGISTERED",),
        ("SCHEDULE_MODE_UNREGISTERED", "SCHEDULE_EVIDENCE_UNVERIFIED"),
        ("T20", "T22"),
        (
            "claim_contracts.event.schedule_mode",
            "claim_contracts.event.schedule_evidence_status",
        ),
    ),
    "O13": op(
        "BLOCKED_UNVERIFIED",
        "external_attachment",
        ("EXTERNAL_CONVERSION_UNVERIFIED",),
        ("EXTERNAL_CONVERSION_UNVERIFIED",),
        ("T25", "T26"),
        ("claim_contracts.event.conversion_method",),
        assurance="EXTERNAL_UNVERIFIED",
    ),
    "O14": op(
        "BLOCKED_INVALID",
        "private_dependency",
        ("MODEL_SELECTION_NOT_COVERED",),
        ("MODEL_SELECTION_NOT_COVERED",),
        ("T31",),
        ("pipeline.manifest.model_selection_status",),
        REBIND_DERIVED,
    ),
    "O15": op(
        "BLOCKED_INVALID",
        "accountant",
        ("REGISTERED_MECHANISM_SEMANTICS_MISMATCH", "ACCOUNTANT_CHECKER_UNREGISTERED"),
        ("REGISTERED_MECHANISM_SEMANTICS_MISMATCH", "ACCOUNTANT_CHECKER_UNREGISTERED"),
        ("T35",),
        ("mechanism.accountant_id",),
        REBIND_DERIVED,
    ),
    "O16": op(
        "BLOCKED_INVALID",
        "unit_binding",
        ("ACTUAL_GENERATED_UNIT_MISMATCH",),
        ("GENERATED_ACCOUNTING_UNIT_MISMATCH", "ACTUAL_GENERATED_UNIT_MISMATCH"),
        ("T37",),
        ("pipeline.manifest.generated_record_unit",),
        REBIND_DERIVED,
    ),
    "O17": op(
        "BLOCKED_UNVERIFIED",
        "stability",
        ("STABILITY_STATUS_UNVERIFIED",),
        ("STABILITY_STATUS_UNVERIFIED",),
        ("T05",),
        ("claim_contracts.event.stability_status",),
    ),
    "O18": op(
        "BLOCKED_INVALID",
        "executable_binding",
        ("PIPELINE_CODE_REGISTRY_DIGEST_MISMATCH",),
        ("PIPELINE_CODE_REGISTRY_DIGEST_MISMATCH",),
        ("R01",),
        ("pipeline.manifest.code_sha256",),
        REBIND_DERIVED,
    ),
    "O19": op(
        "BLOCKED_INVALID",
        "executable_binding",
        ("REGISTERED_MECHANISM_SEMANTICS_MISMATCH",),
        ("REGISTERED_MECHANISM_SEMANTICS_MISMATCH",),
        ("R02",),
        ("mechanism.secure_rng_backend",),
        REBIND_DERIVED,
    ),
    "O20": op(
        "BLOCKED_INVALID",
        "rng",
        ("REGISTERED_MECHANISM_SEMANTICS_MISMATCH",),
        ("REGISTERED_MECHANISM_SEMANTICS_MISMATCH",),
        ("R03",),
        ("mechanism.noise_generation", "mechanism.noise_hardening"),
        REBIND_DERIVED,
    ),
    "O21": op(
        "BLOCKED_INVALID",
        "mechanism",
        ("REGISTERED_MECHANISM_SEMANTICS_MISMATCH",),
        ("REGISTERED_MECHANISM_SEMANTICS_MISMATCH",),
        ("R04",),
        ("mechanism.update_normalization",),
        REBIND_DERIVED,
    ),
    "O22": op(
        "BLOCKED_UNVERIFIED",
        "stability",
        ("OWNER_PUBLIC_CAP_UNVERIFIED",),
        ("OWNER_PUBLIC_CAP_UNVERIFIED",),
        ("R05",),
        (
            "pipeline.manifest.contribution_policy.id",
            "pipeline.manifest.contribution_policy.classification",
            "pipeline.manifest.contribution_policy.parameters",
            "pipeline.manifest.contribution_policy.parameters_sha256",
        ),
        REBIND_DERIVED,
    ),
    "O23": op(
        "BLOCKED_INVALID",
        "stability",
        ("RAW_ADJACENCY_NOT_SUPPORTED",),
        ("RAW_ADJACENCY_NOT_SUPPORTED",),
        ("R06",),
        ("claim_contracts.event.raw_adjacency",),
    ),
    "O24": op(
        "BLOCKED_INVALID",
        "mechanism",
        ("EXACT_SAMPLE_RATE_MISMATCH",),
        ("EXACT_SAMPLE_RATE_MISMATCH",),
        ("R07",),
        ("mechanism.sample_rate_numerator",),
        REBIND_DERIVED,
    ),
    "O25": op(
        "BLOCKED_INVALID",
        "runtime",
        ("RUNTIME_ARTIFACT_DIGEST_MISMATCH",),
        ("RUNTIME_ARTIFACT_DIGEST_MISMATCH",),
        ("R08",),
        ("runtime_trace.evidence.released_model_sha256",),
        ("runtime_trace.runtime_evidence_sha256", "runtime_trace.trace_hash"),
    ),
}


@dataclass
class Checks:
    records: list[dict[str, object]] = field(default_factory=list)

    def require(self, name: str, condition: bool, detail: str = "") -> None:
        self.records.append({"name": name, "passed": bool(condition), "detail": detail})

    @property
    def failures(self) -> list[dict[str, object]]:
        return [record for record in self.records if not record["passed"]]


@dataclass(frozen=True)
class Evidence:
    file_sha256: str
    selected_sha256: str
    count: int
    event_kappa: int
    owner_kappa: int
    generated_unit: str
    raw_rows: tuple[dict[str, str], ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(relative: str) -> Path:
    path = (PROJECT_ROOT / relative).resolve()
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Path escapes project root: {relative}") from exc
    return path


def owners(row: Mapping[str, str]) -> list[str]:
    raw = row.get("owner_ids") or row.get("owners") or row.get("owner_id")
    if raw is None or not str(raw).strip():
        raise ValueError("missing owner attribution")
    output = [
        part.strip() for part in str(raw).replace("|", ";").split(";") if part.strip()
    ]
    if not output or len(output) != len(set(output)):
        raise ValueError("empty or repeated owner attribution")
    return output


def selected_sha(records: Iterable[Mapping[str, object]]) -> str:
    canonical = [
        {
            "scenario": str(record["scenario"]),
            "window_id": str(record["window_id"]),
            "generated_unit": str(record["generated_unit"]),
            "owner_ids": sorted(str(owner) for owner in record["owner_ids"]),
            "start": int(record["start"]),
            "end": int(record["end"]),
        }
        for record in records
    ]
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
    return canonical_sha(canonical)


def inspect_mapping(path: Path) -> Evidence:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = {"scenario", "window_id", "generated_unit", "start", "end"}
        if len(fields) != len(set(fields)) or not required <= set(fields):
            raise ValueError(f"invalid mapping header: {path}")
        if not ({"owner_id", "owner_ids", "owners"} & set(fields)):
            raise ValueError(f"missing owner column: {path}")
        raw_rows = [dict(row) for row in reader]
    selected = [row for row in raw_rows if row.get("scenario") == "gold"]
    if not selected:
        raise ValueError(f"empty gold scenario: {path}")
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    units: set[str] = set()
    for row in selected:
        window_id = str(row.get("window_id") or "").strip()
        if not window_id or window_id in seen:
            raise ValueError(f"invalid/repeated window id: {path}")
        seen.add(window_id)
        start, end = int(str(row["start"])), int(str(row["end"]))
        if end <= start:
            raise ValueError(f"invalid half-open interval: {path}")
        unit = str(row.get("generated_unit") or "")
        if unit not in {"window", "event", "owner"}:
            raise ValueError(f"invalid generated unit: {path}")
        units.add(unit)
        records.append(
            {
                "scenario": "gold",
                "window_id": window_id,
                "generated_unit": unit,
                "owner_ids": owners(row),
                "start": start,
                "end": end,
            }
        )
    if len(units) != 1:
        raise ValueError(f"mixed generated units: {path}")
    deltas: dict[str, Counter[int]] = defaultdict(Counter)
    counts: Counter[str] = Counter()
    for record in records:
        for owner in record["owner_ids"]:
            owner_text = str(owner)
            deltas[owner_text][int(record["start"])] += 1
            deltas[owner_text][int(record["end"])] -= 1
            counts[owner_text] += 1
    event_kappa = 0
    for owner_deltas in deltas.values():
        active = 0
        for position in sorted(owner_deltas):
            active += owner_deltas[position]
            event_kappa = max(event_kappa, active)
    return Evidence(
        file_sha256=sha256_file(path),
        selected_sha256=selected_sha(records),
        count=len(records),
        event_kappa=event_kappa,
        owner_kappa=max(counts.values(), default=0),
        generated_unit=next(iter(units)),
        raw_rows=tuple(raw_rows),
    )


def group_conversion(epsilon: float, delta: float, k: int) -> tuple[float, float]:
    return (
        k * epsilon,
        delta * math.fsum(math.exp(index * epsilon) for index in range(k)),
    )


def numbers_match(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)


def optional_float(value: object) -> float | None:
    return None if value is None or str(value) == "" else float(str(value))


def boolean(value: object) -> bool:
    text = str(value).lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"invalid serialized Boolean: {value!r}")


def path_matches(path: str, prefix: str) -> bool:
    return (
        path == prefix or path.startswith(prefix + ".") or path.startswith(prefix + "[")
    )


def deep_diff(left: object, right: object, prefix: str = "") -> set[str]:
    if type(left) is not type(right):
        return {prefix or "$"}
    if isinstance(left, dict):
        output: set[str] = set()
        for key in set(left) | set(right):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                output.add(path)
            else:
                output.update(deep_diff(left[key], right[key], path))
        return output
    if isinstance(left, list):
        output = set()
        for index in range(max(len(left), len(right))):
            path = f"{prefix}[{index}]"
            if index >= len(left) or index >= len(right):
                output.add(path)
            else:
                output.update(deep_diff(left[index], right[index], path))
        return output
    return set() if left == right else {prefix or "$"}


def cells(stratum: str) -> list[str]:
    return [
        f"{stratum}:L{level}:C{context}"
        for level in range(1, 6)
        for context in range(1, 5)
    ]


def parent_schedule() -> dict[str, set[str]]:
    assignments: dict[str, set[str]] = defaultdict(set)
    for operator, level in (("O11", 1), ("O12", 2), ("O13", 3), ("O17", 4), ("O23", 5)):
        assignments[operator].update(
            f"S4:L{level}:C{context}" for context in range(1, 5)
        )
    assignments["O22"].update(f"S5:L1:C{context}" for context in range(1, 5))
    assignments["O02"].update(f"S5:L5:C{context}" for context in range(1, 5))
    cross = (
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
    s5_middle = [
        f"S5:L{level}:C{context}" for level in range(2, 5) for context in range(1, 5)
    ]
    for index, operator in enumerate(cross):
        assignments[operator].update(cells(s)[index] for s in ("S1", "S2", "S3"))
        assignments[operator].add(s5_middle[index])
    remaining = ("O16", "O18", "O19", "O20", "O21", "O24")
    extras = ("S1:L5:C3", "S1:L5:C4", "S2:L5:C3", "S2:L5:C4", "S3:L5:C3", "S3:L5:C4")
    for offset, (operator, extra) in enumerate(zip(remaining, extras, strict=True)):
        assignments[operator].update(cells(s)[12 + offset] for s in ("S1", "S2", "S3"))
        assignments[operator].add(extra)
    return dict(assignments)


def verify_imports(checks: Checks) -> None:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    allowed = {
        "__future__",
        "argparse",
        "ast",
        "csv",
        "hashlib",
        "json",
        "math",
        "sys",
        "collections",
        "dataclasses",
        "pathlib",
        "typing",
    }
    checks.require("standard_library_only", imports <= allowed, str(sorted(imports)))


def verify_artifact_index(
    report_dir: Path, cases: Sequence[Mapping[str, object]], checks: Checks
) -> dict[str, Mapping[str, object]]:
    document = load_json(report_dir / "artifact_index.json")
    checks.require(
        "artifact_schema",
        document.get("schema_version") == "validator_ablation_artifact_v3.1",
    )
    records = document.get("files", [])
    by_path = {str(record["path"]): record for record in records}
    checks.require(
        "artifact_paths_unique", len(by_path) == len(records), f"records={len(records)}"
    )
    for relative, record in by_path.items():
        path = resolve(relative)
        checks.require(f"artifact_exists:{relative}", path.is_file())
        if path.is_file():
            checks.require(
                f"artifact_size:{relative}", path.stat().st_size == int(record["bytes"])
            )
            checks.require(
                f"artifact_hash:{relative}", sha256_file(path) == str(record["sha256"])
            )
    required = {
        "scripts/run_validator_ablation_v3_1.py",
        "scripts/verify_validator_ablation_v3_2.py",
        "scripts/run_validator_ablation_v3.py",
        "scripts/run_validator_ablation_v2.py",
        "scripts/privacy_claim_validator.py",
        "scripts/mapping_evidence.py",
        "tests/test_privacy_claim_validator.py",
        "docs/mechanism_registry_v2_0.json",
        "scripts/wisdm_v2_gold_pipeline.py",
        "scripts/rdp_accountant_lite.py",
        "docs/privacy_report_schema_v2_0.json",
        PROTOCOL_PATH,
        "reports/validator_ablation_v3_200_001/cases.json",
    }
    for case in cases:
        required.update(
            str(case[key]) for key in ("mapping_path", "report_path", "audit_path")
        )
    prefix = report_dir.relative_to(PROJECT_ROOT).as_posix()
    required.update(
        f"{prefix}/{name}"
        for name in (
            "cases.json",
            "predictions.csv",
            "aggregate_metrics.csv",
            "coverage_matrix.csv",
            "operator_parent_matrix.csv",
            "confusion_by_failure_family.csv",
            "confusion_by_context.csv",
            "confusion_by_stratum.csv",
            "v3_input_lineage_equivalence.json",
            "execution_status.json",
            "summary.md",
        )
    )
    checks.require(
        "artifact_required_membership",
        required <= set(by_path),
        str(sorted(required - set(by_path))),
    )
    checks.require(
        "artifact_all_600_case_inputs",
        sum(
            str(case[key]) in by_path
            for case in cases
            for key in ("mapping_path", "report_path", "audit_path")
        )
        == 600,
    )
    self_relative = Path(__file__).relative_to(PROJECT_ROOT).as_posix()
    checks.require(
        "artifact_actual_verifier",
        self_relative in by_path
        and by_path[self_relative]["sha256"] == sha256_file(Path(__file__)),
    )
    checks.require(
        "frozen_protocol", sha256_file(resolve(PROTOCOL_PATH)) == PROTOCOL_SHA256
    )
    return by_path


def verify_registry(checks: Checks) -> Mapping[str, object]:
    registry_path = PROJECT_ROOT / "docs" / "mechanism_registry_v2_0.json"
    registry = load_json(registry_path)
    entry = registry["mechanisms"][REGISTRY_KEY]
    code_path = resolve(str(entry["code_artifact_id"]))
    checker_path = resolve(str(entry["accountant_checker_artifact_id"]))
    checks.require("registry_code_hash", sha256_file(code_path) == entry["code_sha256"])
    checks.require(
        "registry_checker_hash",
        sha256_file(checker_path) == entry["accountant_checker_sha256"],
    )
    return entry


def verify_internal_hashes(
    case_id: str,
    report: Mapping[str, object],
    evidence: Evidence,
    operator: str,
    registry_entry: Mapping[str, object],
    checks: Checks,
) -> None:
    mechanism = report["mechanism"]
    pipeline = report["pipeline"]
    manifest = pipeline["manifest"]
    checks.require(
        f"mechanism_hash:{case_id}",
        manifest["mechanism_sha256"] == canonical_sha(mechanism),
    )
    checks.require(
        f"pipeline_hash:{case_id}",
        pipeline["pipeline_sha256"] == canonical_sha(manifest),
    )
    checks.require(
        f"pipeline_manifest_mapping_crossbind:{case_id}",
        pipeline["mapping_file_sha256"] == manifest["mapping_file_sha256"]
        and pipeline["selected_mapping_sha256"] == manifest["selected_mapping_sha256"],
    )
    for item in manifest.get("preprocessing", []):
        checks.require(
            f"preprocess_parameters_hash:{case_id}:{item['id']}",
            item["parameters_sha256"] == canonical_sha(item["parameters"]),
        )
    for name in ("raw_domain", "contribution_policy", "schedule"):
        item = manifest[name]
        checks.require(
            f"manifest_parameters_hash:{case_id}:{name}",
            item["parameters_sha256"] == canonical_sha(item["parameters"]),
        )
    for claim, contract in report["claim_contracts"].items():
        if "raw_domain_sha256" in contract:
            checks.require(
                f"contract_raw_domain_hash:{case_id}:{claim}",
                contract["raw_domain_sha256"] == canonical_sha(manifest["raw_domain"]),
            )
    checks.require(
        f"adapter_entry_hash:{case_id}",
        mechanism["adapter_registry_entry_sha256"] == canonical_sha(registry_entry),
    )
    runtime = report["runtime_trace"]
    if runtime.get("status") == "matched":
        checks.require(
            f"runtime_evidence_hash:{case_id}",
            runtime["runtime_evidence_sha256"] == canonical_sha(runtime["evidence"]),
        )
        payload = {
            key: runtime.get(key)
            for key in (*RUNTIME_EXACT_FIELDS, *RUNTIME_NUMERIC_FIELDS)
        }
        checks.require(
            f"runtime_trace_hash:{case_id}",
            runtime["trace_hash"] == canonical_sha(payload),
        )
    else:
        checks.require(
            f"runtime_missing_only_o01:{case_id}",
            operator == "O01" and runtime == {"status": "missing"},
        )
    if not operator:
        semantic_keys = (
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
        checks.require(
            f"positive_registry_semantics:{case_id}",
            all(mechanism[key] == registry_entry[key] for key in semantic_keys),
        )
        checks.require(
            f"positive_registry_sources:{case_id}",
            manifest["code_artifact_id"] == registry_entry["code_artifact_id"]
            and manifest["code_sha256"] == registry_entry["code_sha256"]
            and manifest["accountant_checker_artifact_id"]
            == registry_entry["accountant_checker_artifact_id"]
            and manifest["accountant_checker_sha256"]
            == registry_entry["accountant_checker_sha256"],
        )
    is_o03 = operator == "O03"
    bound = (
        pipeline["mapping_file_sha256"] == evidence.file_sha256
        and pipeline["selected_mapping_sha256"] == evidence.selected_sha256
        and int(manifest["selected_record_count"]) == evidence.count
    )
    checks.require(
        f"intentional_mapping_binding:{case_id}", (not bound) if is_o03 else bound
    )


def positive_truth(case: Mapping[str, object], evidence: Evidence) -> dict[str, object]:
    stratum = str(case["stratum_id"])
    spec = STRATA[stratum]
    level = int(case["level"])
    target = int(spec["levels"][level - 1])
    k = 1
    if stratum == "S4":
        k = 2 * evidence.event_kappa
    elif stratum == "S5":
        k = evidence.owner_kappa
    epsilon, delta = (
        (BASE_EPSILON, BASE_DELTA)
        if k == 1
        else group_conversion(BASE_EPSILON, BASE_DELTA, k)
    )
    return {
        "status": "ALLOWED",
        "path": spec["path"],
        "assurance": "EVIDENCE_VALIDATED",
        "epsilon": epsilon,
        "delta": delta,
        "failure_family": "positive",
        "required": (),
        "allowed": (),
        "k": k,
        "target": target,
    }


def verify_positive_case(
    case: Mapping[str, object],
    evidence: Evidence,
    report: Mapping[str, object],
    checks: Checks,
) -> dict[str, object]:
    case_id = str(case["case_id"])
    stratum = str(case["stratum_id"])
    spec = STRATA[stratum]
    level = int(case["level"])
    context = str(case["context_id"])
    truth = positive_truth(case, evidence)
    expected_meta = (
        case_id == f"P3_{stratum}_L{level}_{context}"
        and case["coverage_cell"] == f"{stratum}:L{level}:{context}"
        and case["stratum_name"] == spec["name"]
        and case["context_name"] == CONTEXT_NAMES[context]
        and case["generated_unit"] == spec["generated"]
        and case["requested_unit"] == spec["requested"]
        and case["claim_unit"] == spec["requested"]
        and tuple(case["normative_anchors"]) == tuple(spec["anchors"])
        and case["truth_row"] == ",".join(spec["anchors"])
        and case["parent_provenance"] == "parent_free_frozen_grid"
        and not case["parent_case_id"]
        and not case["operator_id"]
        and not case["required_issue_codes"]
    )
    checks.require(f"positive_independent_grid:{case_id}", expected_meta)
    checks.require(
        f"positive_evidence_grid:{case_id}",
        evidence.generated_unit == spec["generated"]
        and evidence.count == truth["target"]
        and (stratum != "S4" or evidence.event_kappa == truth["target"])
        and (stratum != "S5" or evidence.owner_kappa == truth["target"]),
    )
    checks.require(
        f"positive_oracle_fields:{case_id}",
        case["expected_release_status"] == truth["status"]
        and case["expected_unit_path"] == truth["path"]
        and case["expected_assurance_status"] == truth["assurance"]
        and int(case["expected_k"]) == truth["k"]
        and numbers_match(float(case["expected_epsilon"]), float(truth["epsilon"]))
        and numbers_match(float(case["expected_delta"]), float(truth["delta"])),
    )
    privacy = report["privacy"]
    contract = report["claim_contracts"][str(spec["requested"])]
    checks.require(
        f"positive_base_pair:{case_id}",
        numbers_match(float(privacy["epsilon"]), BASE_EPSILON)
        and numbers_match(float(privacy["delta"]), BASE_DELTA),
    )
    checks.require(
        f"positive_contract_k:{case_id}", int(contract["stability_bound"]) == truth["k"]
    )
    checks.require(f"positive_nonvacuous:{case_id}", float(truth["delta"]) < 1)
    return truth


def negative_truth(
    case: Mapping[str, object], parent_truth: Mapping[str, object]
) -> dict[str, object]:
    spec = OPERATORS[str(case["operator_id"])]
    path = parent_truth["path"] if spec["path"] == "PARENT" else spec["path"]
    return {
        "status": spec["status"],
        "path": path,
        "assurance": spec["assurance"],
        "epsilon": None,
        "delta": None,
        "failure_family": spec["family"],
        "required": spec["required"],
        "allowed": spec["allowed"],
    }


def direct_semantics(
    operator: str,
    parent: Mapping[str, object],
    child: Mapping[str, object],
    parent_evidence: Evidence,
    child_evidence: Evidence,
) -> bool:
    cm = child["mechanism"]
    pp, cp = parent["privacy"], child["privacy"]
    pman, cman = parent["pipeline"]["manifest"], child["pipeline"]["manifest"]
    cr = child["runtime_trace"]
    if operator == "O01":
        return cr == {"status": "missing"}
    if operator == "O02":
        return (
            float(cp["delta"]) == 0.1
            and float(cp["epsilon"]) != float(pp["epsilon"])
            and group_conversion(float(cp["epsilon"]), float(cp["delta"]), 13)[1] >= 1
        )
    if operator == "O03":
        return child_evidence.file_sha256 != parent_evidence.file_sha256
    if operator == "O04":
        return cm["steps"] is True
    if operator == "O05":
        return cm["sampler_law"] == "shuffle"
    if operator == "O06":
        return float(cr["noise_multiplier"]) == 9.0
    if operator == "O07":
        return numbers_match(float(cp["epsilon"]), float(pp["epsilon"]) + 0.01)
    if operator == "O08":
        return cm["secure_mode"] is False
    if operator == "O09":
        return cm["secure_mode"] is False and child["rng_assurance"] == "research_prng"
    if operator == "O10":
        return cman["preprocessing"][0]["classification"] == "private_global"
    if operator == "O11":
        return "labels" not in cman["influence_support"]["components"]
    if operator == "O12":
        return (
            child["claim_contracts"]["event"]["schedule_mode"] == "adaptive_private"
            and child["claim_contracts"]["event"]["schedule_evidence_status"]
            == "unverified"
        )
    if operator == "O13":
        return (
            child["claim_contracts"]["event"]["conversion_method"]
            == "external_json_claim"
        )
    if operator == "O14":
        return cman["model_selection_status"] == "not_covered"
    if operator == "O15":
        return cm["accountant_id"] == "plausible_external_rdp"
    if operator == "O16":
        return cman["generated_record_unit"] != pman["generated_record_unit"]
    if operator == "O17":
        return child["claim_contracts"]["event"]["stability_status"] == "unverified"
    if operator == "O18":
        return cman["code_sha256"] == "0" * 64
    if operator == "O19":
        return cm["secure_rng_backend"] == "attacker.fake_rng"
    if operator == "O20":
        return (
            cm["noise_generation"] == "one_draw_unhardened"
            and cm["noise_hardening"] == "none"
        )
    if operator == "O21":
        return (
            cm["update_normalization"]
            == "divide_by_sample_rate_times_private_dataset_size"
        )
    if operator == "O22":
        return cman["contribution_policy"]["id"] == "no_additional_cap" and cman[
            "contribution_policy"
        ]["parameters"] == {"cap": None}
    if operator == "O23":
        return child["claim_contracts"]["event"]["raw_adjacency"] == "replace_one"
    if operator == "O24":
        return cm["sample_rate_numerator"] == 2
    if operator == "O25":
        return cr["evidence"]["released_model_sha256"] == "3" * 64
    raise ValueError(operator)


def verify_cases(
    cases: Sequence[Mapping[str, object]],
    registry_entry: Mapping[str, object],
    checks: Checks,
) -> tuple[
    dict[str, Evidence],
    dict[str, Mapping[str, object]],
    dict[str, Mapping[str, object]],
    dict[str, dict[str, object]],
]:
    evidence_by_id: dict[str, Evidence] = {}
    reports: dict[str, Mapping[str, object]] = {}
    audits: dict[str, Mapping[str, object]] = {}
    truths: dict[str, dict[str, object]] = {}
    triples: list[str] = []
    for case in cases:
        case_id = str(case["case_id"])
        paths = {
            key: resolve(str(case[key]))
            for key in ("mapping_path", "report_path", "audit_path")
        }
        hashes = []
        for key, path in paths.items():
            digest = sha256_file(path)
            hashes.append(digest)
            checks.require(
                f"case_hash:{case_id}:{key}", digest == case[f"{key}_sha256"]
            )
        triple = canonical_sha(hashes)
        triples.append(triple)
        checks.require(f"case_triple:{case_id}", triple == case["input_triple_sha256"])
        evidence = inspect_mapping(paths["mapping_path"])
        report = load_json(paths["report_path"])
        audit = load_json(paths["audit_path"])
        evidence_by_id[case_id], reports[case_id], audits[case_id] = (
            evidence,
            report,
            audit,
        )
        operator = str(case.get("operator_id") or "")
        verify_internal_hashes(
            case_id, report, evidence, operator, registry_entry, checks
        )
        checks.require(
            f"mapping_case_metadata:{case_id}",
            evidence.count == int(case["selected_record_count"])
            and evidence.event_kappa == int(case["observed_event_kappa"])
            and evidence.owner_kappa == int(case["observed_owner_kappa"])
            and evidence.generated_unit == case["generated_unit"],
        )
        report_bound = (
            report["pipeline"]["mapping_file_sha256"] == evidence.file_sha256
            and report["pipeline"]["selected_mapping_sha256"]
            == evidence.selected_sha256
        )
        audit_bound = (
            audit["mapping_file_sha256"] == evidence.file_sha256
            and audit["selected_mapping_sha256"] == evidence.selected_sha256
            and int(audit["num_windows_used"]) == evidence.count
            and int(audit["event_kappa_used"]) == evidence.event_kappa
            and int(audit["owner_windows_max_used"]) == evidence.owner_kappa
        )
        checks.require(
            f"report_mapping_binding:{case_id}",
            (not report_bound) if operator == "O03" else report_bound,
        )
        checks.require(
            f"audit_mapping_binding:{case_id}",
            (not audit_bound) if operator == "O03" else audit_bound,
        )
    checks.require("input_triples_unique", len(set(triples)) == len(triples) == 200)

    positives = [case for case in cases if case["case_kind"] == "allowed"]
    for case in positives:
        case_id = str(case["case_id"])
        truths[case_id] = verify_positive_case(
            case, evidence_by_id[case_id], reports[case_id], checks
        )

    positives_by_id = {str(case["case_id"]): case for case in positives}
    negatives = [case for case in cases if case["case_kind"] == "non_allow"]
    observed_schedule: dict[str, set[str]] = defaultdict(set)
    parent_ids: list[str] = []
    for case in negatives:
        case_id, operator = str(case["case_id"]), str(case["operator_id"])
        parent_id = str(case["parent_case_id"])
        parent_ids.append(parent_id)
        parent = positives_by_id[parent_id]
        spec = OPERATORS[operator]
        truth = negative_truth(case, truths[parent_id])
        truths[case_id] = truth
        observed_schedule[operator].add(str(parent["coverage_cell"]))
        expected_path = truth["path"]
        checks.require(
            f"negative_independent_oracle:{case_id}",
            case["expected_release_status"] == truth["status"]
            and case["expected_unit_path"] == expected_path
            and case["expected_assurance_status"] == truth["assurance"]
            and case["failure_family"] == spec["family"]
            and tuple(case["required_issue_codes"]) == tuple(spec["required"])
            and tuple(case["normative_anchors"]) == tuple(spec["anchors"])
            and case["truth_row"] == ",".join(spec["anchors"])
            and tuple(case["direct_mutation_paths"]) == tuple(spec["direct"])
            and tuple(case["derived_rebind_paths"]) == tuple(spec["derived"])
            and case["expected_numeric_rule"] == "none"
            and case["expected_epsilon"] is None
            and case["expected_delta"] is None,
        )
        checks.require(
            f"paired_metadata:{case_id}",
            all(
                str(case[key]) == str(parent[key])
                for key in (
                    "coverage_cell",
                    "stratum_id",
                    "level",
                    "context_id",
                    "generated_unit",
                    "requested_unit",
                    "expected_k",
                )
            ),
        )
        checks.require(f"paired_audit:{case_id}", audits[case_id] == audits[parent_id])
        child_evidence, parent_evidence = (
            evidence_by_id[case_id],
            evidence_by_id[parent_id],
        )
        checks.require(
            f"direct_semantics:{case_id}",
            direct_semantics(
                operator,
                reports[parent_id],
                reports[case_id],
                parent_evidence,
                child_evidence,
            ),
        )
        if operator == "O03":
            diffs = deep_diff(
                list(parent_evidence.raw_rows),
                list(child_evidence.raw_rows),
                "mapping.rows",
            )
            checks.require(
                f"o03_mapping_only:{case_id}",
                len(diffs) == 1
                and next(iter(diffs)).endswith(".end")
                and reports[case_id] == reports[parent_id],
            )
        else:
            checks.require(
                f"paired_mapping:{case_id}",
                child_evidence.file_sha256 == parent_evidence.file_sha256,
            )
            diffs = deep_diff(reports[parent_id], reports[case_id])
            allowed_paths = tuple(spec["direct"]) + tuple(spec["derived"])
            checks.require(
                f"mutation_closure:{case_id}",
                bool(diffs)
                and all(
                    any(path_matches(path, prefix) for prefix in allowed_paths)
                    for path in diffs
                ),
                str(sorted(diffs)),
            )
            for direct_path in spec["direct"]:
                checks.require(
                    f"direct_path_exercised:{case_id}:{direct_path}",
                    any(
                        path_matches(path, str(direct_path))
                        or path_matches(str(direct_path), path)
                        for path in diffs
                    ),
                )
    checks.require(
        "parent_one_to_one", Counter(parent_ids) == Counter(positives_by_id.keys())
    )
    checks.require(
        "operator_four_each",
        Counter(str(case["operator_id"]) for case in negatives)
        == Counter({key: 4 for key in OPERATORS}),
    )
    checks.require(
        "exact_parent_schedule", dict(observed_schedule) == parent_schedule()
    )
    checks.require(
        "status_arithmetic",
        Counter(str(truths[str(case["case_id"])]["status"]) for case in negatives)
        == Counter(
            {
                "BLOCKED_INVALID": 68,
                "BLOCKED_UNVERIFIED": 24,
                "BLOCKED_VACUOUS": 4,
                "RESEARCH_ONLY": 4,
            }
        ),
    )
    return evidence_by_id, reports, audits, truths


def verify_context_metamorphics(
    cases: Sequence[Mapping[str, object]],
    evidence: Mapping[str, Evidence],
    checks: Checks,
) -> None:
    positives = [case for case in cases if case["case_kind"] == "allowed"]
    by_cell = {str(case["coverage_cell"]): case for case in positives}
    checks.require(
        "positive_full_mapping_unique",
        len({evidence[str(case["case_id"])].file_sha256 for case in positives}) == 100,
    )
    checks.require(
        "positive_selected_semantics_50",
        len({evidence[str(case["case_id"])].selected_sha256 for case in positives})
        == 50,
    )
    for stratum in STRATA:
        for level in range(1, 6):
            base = evidence[str(by_cell[f"{stratum}:L{level}:C1"]["case_id"])]
            for context in (3, 4):
                other = evidence[
                    str(by_cell[f"{stratum}:L{level}:C{context}"]["case_id"])
                ]
                checks.require(
                    f"metamorphic_selected:{stratum}:L{level}:C{context}",
                    other.selected_sha256 == base.selected_sha256,
                )
                checks.require(
                    f"metamorphic_file:{stratum}:L{level}:C{context}",
                    other.file_sha256 != base.file_sha256,
                )
    checks.require(
        "vacuity_boundary",
        group_conversion(BASE_EPSILON, BASE_DELTA, 13)[1]
        < 1
        <= group_conversion(BASE_EPSILON, BASE_DELTA, 14)[1],
    )


def verify_v3_lineage(
    report_dir: Path, cases: Sequence[Mapping[str, object]], checks: Checks
) -> None:
    reference_path = REFERENCE_DIR / "cases.json"
    checks.require(
        "reference_cases_hash", sha256_file(reference_path) == REFERENCE_CASES_SHA256
    )
    reference = {
        str(case["case_id"]): case for case in load_json(reference_path)["cases"]
    }
    current = {str(case["case_id"]): case for case in cases}
    checks.require(
        "lineage_case_ids", set(current) == set(reference) and len(current) == 200
    )
    fields = (
        "truth_row",
        "claim_unit",
        "failure_family",
        "expected_release_status",
        "expected_unit_path",
        "expected_assurance_status",
        "expected_issue_code",
        "expected_numeric_rule",
        "expected_epsilon",
        "expected_delta",
        "case_kind",
        "coverage_cell",
        "stratum_id",
        "stratum_name",
        "level",
        "context_id",
        "context_name",
        "generated_unit",
        "requested_unit",
        "accountant_metric",
        "selected_record_count",
        "observed_event_kappa",
        "observed_owner_kappa",
        "expected_k",
        "parent_provenance",
        "parent_case_id",
        "operator_id",
        "direct_mutation_paths",
        "derived_rebind_paths",
        "required_issue_codes",
        "normative_anchors",
    )
    for case_id, case in current.items():
        old = reference[case_id]
        inputs_equal = all(
            case[f"{key}_sha256"] == old[f"{key}_sha256"]
            for key in ("mapping_path", "report_path", "audit_path")
        )
        oracle_equal = all(
            json.dumps(case[field], sort_keys=True)
            == json.dumps(old[field], sort_keys=True)
            for field in fields
        )
        checks.require(f"lineage_inputs:{case_id}", inputs_equal)
        checks.require(f"lineage_oracle:{case_id}", oracle_equal)
    lineage = load_json(report_dir / "v3_input_lineage_equivalence.json")
    checks.require(
        "lineage_report_summary",
        lineage.get("all_200_input_content_equal") is True
        and lineage.get("all_200_oracle_metadata_equal") is True
        and len(lineage.get("cases", [])) == 200,
    )


def prediction_flags(
    truth: Mapping[str, object], report: Mapping[str, object], row: Mapping[str, object]
) -> dict[str, bool]:
    predicted_status = str(row["predicted_release_status"])
    predicted_path = str(row["predicted_unit_path"])
    predicted_assurance = str(row["predicted_assurance_status"])
    predicted_epsilon = optional_float(row["predicted_epsilon"])
    predicted_delta = optional_float(row["predicted_delta"])
    allowed_truth = truth["status"] == "ALLOWED"
    predicted_allowed = predicted_status == "ALLOWED"
    unsafe = not allowed_truth and predicted_allowed
    valid_exact = (
        allowed_truth
        and predicted_allowed
        and predicted_path == truth["path"]
        and numbers_match(predicted_epsilon, truth["epsilon"])
        and numbers_match(predicted_delta, truth["delta"])
    )
    exact = (
        predicted_status == truth["status"]
        and predicted_path == truth["path"]
        and predicted_assurance == truth["assurance"]
        and (
            not allowed_truth
            or (
                numbers_match(predicted_epsilon, truth["epsilon"])
                and numbers_match(predicted_delta, truth["delta"])
            )
        )
    )
    base = report["privacy"]
    wrong_same = (
        allowed_truth
        and truth["path"] in {"CONVERT", "GROUP"}
        and predicted_allowed
        and numbers_match(predicted_epsilon, float(base["epsilon"]))
        and numbers_match(predicted_delta, float(base["delta"]))
    )
    return {
        "supported_positive": predicted_allowed,
        "unsafe_positive": unsafe,
        "binary_safe_correct": allowed_truth == predicted_allowed,
        "valid_decision_exact": valid_exact,
        "exact_tuple": exact,
        "wrong_same_number": wrong_same,
        "valid_retained": allowed_truth and predicted_allowed,
    }


def verify_predictions(
    report_dir: Path,
    cases: Sequence[Mapping[str, object]],
    reports: Mapping[str, Mapping[str, object]],
    truths: Mapping[str, Mapping[str, object]],
    checks: Checks,
) -> None:
    cases_by_id = {str(case["case_id"]): case for case in cases}
    with (report_dir / "predictions.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    keys = [(row["case_id"], row["method"]) for row in rows]
    checks.require("prediction_1600", len(rows) == 1600 and len(set(keys)) == 1600)
    checks.require(
        "prediction_methods",
        Counter(row["method"] for row in rows)
        == Counter({method: 200 for method in METHODS}),
    )
    bool_fields = (
        "supported_positive",
        "unsafe_positive",
        "binary_safe_correct",
        "valid_decision_exact",
        "exact_tuple",
        "wrong_same_number",
        "valid_retained",
    )
    parsed: list[dict[str, object]] = []
    for row in rows:
        case_id = row["case_id"]
        truth = truths[case_id]
        checks.require(
            f"prediction_truth_columns:{case_id}:{row['method']}",
            row["failure_family"] == truth["failure_family"]
            and row["expected_release_status"] == truth["status"]
            and row["expected_unit_path"] == truth["path"]
            and row["expected_assurance_status"] == truth["assurance"],
        )
        flags = prediction_flags(truth, reports[case_id], row)
        for name in bool_fields:
            checks.require(
                f"prediction_flag:{case_id}:{row['method']}:{name}",
                boolean(row[name]) == flags[name],
            )
        item: dict[str, object] = dict(row)
        item.update(flags)
        parsed.append(item)
        if row["method"] == "FULL":
            issues = [code for code in row["issue_codes"].split(";") if code]
            issue_set = set(issues)
            checks.require(
                f"full_required_issues:{case_id}", set(truth["required"]) <= issue_set
            )
            checks.require(
                f"full_allowed_cascades:{case_id}",
                issue_set <= set(truth["allowed"]),
                str(sorted(issue_set - set(truth["allowed"]))),
            )
    full = [row for row in parsed if row["method"] == "FULL"]
    checks.require("full_zero_unsafe", not any(row["unsafe_positive"] for row in full))
    checks.require(
        "full_200_exact", len(full) == 200 and all(row["exact_tuple"] for row in full)
    )

    recomputed: dict[str, dict[str, float | int | str]] = {}
    for method in METHODS:
        selected = [row for row in parsed if row["method"] == method]
        blocked = [
            row
            for row in selected
            if truths[str(row["case_id"])]["status"] != "ALLOWED"
        ]
        valid = [
            row
            for row in selected
            if truths[str(row["case_id"])]["status"] == "ALLOWED"
        ]
        walls = sorted(float(row["wall_ms"]) for row in selected)
        unsafe = sum(bool(row["unsafe_positive"]) for row in blocked)
        binary_count = sum(bool(row["binary_safe_correct"]) for row in selected)
        valid_exact = sum(bool(row["valid_decision_exact"]) for row in valid)
        exact = sum(bool(row["exact_tuple"]) for row in selected)
        retained = sum(bool(row["valid_retained"]) for row in valid)
        same = sum(bool(row["wrong_same_number"]) for row in selected)
        recomputed[method] = {
            "method": method,
            "cases": len(selected),
            "blocked_truth_cases": len(blocked),
            "unsafe_positives": unsafe,
            "unsafe_positive_rate": unsafe / len(blocked),
            "binary_safe_correct": binary_count,
            "binary_safe_accuracy": binary_count / len(selected),
            "valid_exact_decisions": valid_exact,
            "valid_exact_decision_rate": valid_exact / len(valid),
            "exact_tuples": exact,
            "exact_tuple_accuracy": exact / len(selected),
            "valid_truth_cases": len(valid),
            "valid_retained": retained,
            "valid_retention_rate": retained / len(valid),
            "wrong_same_number": same,
            "median_wall_ms": walls[len(walls) // 2],
            "max_wall_ms": max(walls),
        }
    with (report_dir / "aggregate_metrics.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        aggregates = [dict(row) for row in csv.DictReader(handle)]
    checks.require(
        "aggregate_methods",
        Counter(row["method"] for row in aggregates) == Counter(METHODS),
    )
    integer_fields = {
        "cases",
        "blocked_truth_cases",
        "unsafe_positives",
        "binary_safe_correct",
        "valid_exact_decisions",
        "exact_tuples",
        "valid_truth_cases",
        "valid_retained",
        "wrong_same_number",
    }
    for row in aggregates:
        expected = recomputed[row["method"]]
        for key, value in expected.items():
            if key == "method":
                continue
            match = (
                int(row[key]) == int(value)
                if key in integer_fields
                else numbers_match(float(row[key]), float(value))
            )
            checks.require(f"aggregate:{row['method']}:{key}", match)
    operator_by_case = {
        case_id: str(case["operator_id"]) for case_id, case in cases_by_id.items()
    }
    a1 = [row for row in parsed if row["method"] == "A1_RUNTIME_ASSUMED_MATCHED"]
    expected_a1 = Counter({"O01": 4, "O06": 4, "O25": 4})
    checks.require(
        "pure_a1_unsafe_attribution",
        Counter(
            operator_by_case[str(row["case_id"])]
            for row in a1
            if row["unsafe_positive"]
        )
        == expected_a1,
    )
    checks.require(
        "pure_a1_nonexact_attribution",
        Counter(
            operator_by_case[str(row["case_id"])]
            for row in a1
            if not row["exact_tuple"]
        )
        == expected_a1,
    )
    checks.require(
        "pure_a1_aggregate",
        recomputed["A1_RUNTIME_ASSUMED_MATCHED"]["unsafe_positives"] == 12
        and recomputed["A1_RUNTIME_ASSUMED_MATCHED"]["valid_exact_decisions"] == 100
        and recomputed["A1_RUNTIME_ASSUMED_MATCHED"]["exact_tuples"] == 188,
    )
    a2 = [row for row in parsed if row["method"] == "A2_FULL_MINUS_VACUITY"]
    expected_a2 = Counter({"O02": 4})
    checks.require(
        "a2_attribution",
        Counter(
            operator_by_case[str(row["case_id"])]
            for row in a2
            if row["unsafe_positive"]
        )
        == expected_a2
        and Counter(
            operator_by_case[str(row["case_id"])]
            for row in a2
            if not row["exact_tuple"]
        )
        == expected_a2,
    )
    for method in METHODS[:5]:
        checks.require(
            f"baseline_unsafe:{method}", int(recomputed[method]["unsafe_positives"]) > 0
        )


def verify_exports(
    report_dir: Path, cases: Sequence[Mapping[str, object]], checks: Checks
) -> None:
    positive_ids = Counter(
        str(case["case_id"]) for case in cases if case["case_kind"] == "allowed"
    )
    negative_ids = Counter(
        str(case["case_id"]) for case in cases if case["case_kind"] == "non_allow"
    )
    with (report_dir / "coverage_matrix.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        coverage = [dict(row) for row in csv.DictReader(handle)]
    with (report_dir / "operator_parent_matrix.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        operators = [dict(row) for row in csv.DictReader(handle)]
    checks.require(
        "coverage_export", Counter(row["case_id"] for row in coverage) == positive_ids
    )
    checks.require(
        "operator_export", Counter(row["case_id"] for row in operators) == negative_ids
    )
    for filename in (
        "confusion_by_failure_family.csv",
        "confusion_by_context.csv",
        "confusion_by_stratum.csv",
    ):
        with (report_dir / filename).open("r", newline="", encoding="utf-8") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
        checks.require(
            f"breakdown_methods:{filename}",
            set(row["method"] for row in rows) == set(METHODS),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = PROJECT_ROOT / report_dir
    report_dir = report_dir.resolve()
    checks = Checks()
    verify_imports(checks)
    document = load_json(report_dir / "cases.json")
    cases = document["cases"]
    checks.require(
        "cases_schema",
        document.get("schema_version") == "validator_ablation_cases_v3.1",
    )
    checks.require("protocol_hash", document.get("protocol_sha256") == PROTOCOL_SHA256)
    checks.require(
        "case_count",
        len(cases) == 200 and len({case["case_id"] for case in cases}) == 200,
    )
    checks.require(
        "case_split",
        sum(case["case_kind"] == "allowed" for case in cases) == 100
        and sum(case["case_kind"] == "non_allow" for case in cases) == 100,
    )
    verify_artifact_index(report_dir, cases, checks)
    registry_entry = verify_registry(checks)
    evidence, reports, _audits, truths = verify_cases(cases, registry_entry, checks)
    verify_context_metamorphics(cases, evidence, checks)
    verify_v3_lineage(report_dir, cases, checks)
    verify_predictions(report_dir, cases, reports, truths, checks)
    verify_exports(report_dir, cases, checks)
    execution = load_json(report_dir / "execution_status.json")
    checks.require(
        "runner_gates",
        execution.get("gates_passed") is True
        and not execution.get("gate_errors")
        and execution.get("v3_input_content_equal") is True,
        str(execution.get("gate_errors")),
    )

    output = {
        "schema_version": "validator_ablation_independent_verification_v3.2",
        "verifier": Path(__file__).relative_to(PROJECT_ROOT).as_posix(),
        "verifier_sha256": sha256_file(Path(__file__)),
        "standard_library_only": True,
        "independent_grid_operator_oracle": True,
        "internal_registry_hash_closure": True,
        "checks": checks.records,
        "passed": not checks.failures,
        "check_count": len(checks.records),
        "failure_count": len(checks.failures),
    }
    output_path = report_dir / "independent_verification.json"
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    verification_index = {
        "schema_version": "validator_ablation_verification_index_v1.0",
        "files": [
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (
                report_dir / "artifact_index.json",
                Path(__file__),
                output_path,
            )
        ],
    }
    (report_dir / "verification_index.json").write_text(
        json.dumps(verification_index, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    if checks.failures:
        for failure in checks.failures:
            print(f"FAIL {failure['name']}: {failure['detail']}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Independent verification passed: {len(checks.records)} checks")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
