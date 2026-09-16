"""Independent verifier for the frozen V3 200-case conformance run.

The verifier intentionally uses only the Python standard library.  It does
not import the production validator, mapping parser, test helpers, or V3/V2
runner.  It reconstructs evidence, K, group conversion, pairings, prediction
flags, aggregates, and artifact hashes from persisted raw artifacts.
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
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "validator_ablation_v3_200_001"
PROTOCOL_RELATIVE = (
    "paper_notes/AAAI27_AUDIT_V19_CONFORMANCE_200_PROTOCOL_V1_2026-07-28.md"
)
PROTOCOL_SHA256 = (
    "57b747f147f16569a43d8a4d361c0090cf99d1fab8f48a614dce7a7618abe811"
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
BASE_EPSILON = 1.1053169171072421
BASE_DELTA = 1e-6
EXPECTED_STATUS_COUNTS = Counter(
    {
        "BLOCKED_INVALID": 68,
        "BLOCKED_UNVERIFIED": 24,
        "BLOCKED_VACUOUS": 4,
        "RESEARCH_ONLY": 4,
    }
)


@dataclass
class Checks:
    records: list[dict[str, object]] = field(default_factory=list)

    def require(self, name: str, condition: bool, detail: str = "") -> None:
        self.records.append(
            {"name": name, "passed": bool(condition), "detail": detail}
        )

    @property
    def failures(self) -> list[dict[str, object]]:
        return [record for record in self.records if not record["passed"]]


@dataclass(frozen=True)
class Evidence:
    mapping_file_sha256: str
    selected_mapping_sha256: str
    selected_record_count: int
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


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(relative: str) -> Path:
    candidate = (PROJECT_ROOT / relative).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Artifact escapes project root: {relative}") from exc
    return candidate


def parse_owner_ids(row: Mapping[str, str]) -> list[str]:
    raw = row.get("owner_ids") or row.get("owners") or row.get("owner_id")
    if raw is None or not str(raw).strip():
        raise ValueError("missing owner attribution")
    owners = [part.strip() for part in str(raw).replace("|", ";").split(";") if part.strip()]
    if not owners or len(set(owners)) != len(owners):
        raise ValueError("empty or repeated owner attribution")
    return owners


def selected_digest(records: Iterable[Mapping[str, object]]) -> str:
    canonical: list[dict[str, object]] = []
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


def inspect_mapping(path: Path, scenario: str = "gold") -> Evidence:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = {
            "scenario",
            "window_id",
            "generated_unit",
            "start",
            "end",
        }
        if len(fieldnames) != len(set(fieldnames)) or not required.issubset(fieldnames):
            raise ValueError(f"invalid mapping header: {path}")
        if not ({"owner_id", "owner_ids", "owners"} & set(fieldnames)):
            raise ValueError(f"missing owner column: {path}")
        raw_rows = [dict(row) for row in reader]
    chosen = [row for row in raw_rows if row.get("scenario") == scenario]
    if not chosen:
        raise ValueError(f"empty selected scenario: {path}")
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    units: set[str] = set()
    for row in chosen:
        window_id = str(row.get("window_id") or "").strip()
        if not window_id or window_id in seen:
            raise ValueError(f"missing/repeated window id: {path}")
        seen.add(window_id)
        start = int(str(row["start"]))
        end = int(str(row["end"]))
        if end <= start:
            raise ValueError(f"nonpositive half-open support: {path}")
        unit = str(row.get("generated_unit") or "").strip()
        if unit not in {"window", "event", "owner"}:
            raise ValueError(f"unknown generated unit: {path}")
        units.add(unit)
        records.append(
            {
                "scenario": scenario,
                "window_id": window_id,
                "generated_unit": unit,
                "owner_ids": parse_owner_ids(row),
                "start": start,
                "end": end,
            }
        )
    if len(units) != 1:
        raise ValueError(f"mixed generated units: {path}")

    deltas: dict[str, Counter[int]] = defaultdict(Counter)
    owner_counts: Counter[str] = Counter()
    for record in records:
        for owner in record["owner_ids"]:
            owner_text = str(owner)
            deltas[owner_text][int(record["start"])] += 1
            deltas[owner_text][int(record["end"])] -= 1
            owner_counts[owner_text] += 1
    event_kappa = 0
    for owner_deltas in deltas.values():
        active = 0
        for position in sorted(owner_deltas):
            active += owner_deltas[position]
            event_kappa = max(event_kappa, active)
    return Evidence(
        mapping_file_sha256=sha256_file(path),
        selected_mapping_sha256=selected_digest(records),
        selected_record_count=len(records),
        event_kappa=event_kappa,
        owner_kappa=max(owner_counts.values(), default=0),
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


def parse_optional_float(value: object) -> float | None:
    if value is None or str(value) == "":
        return None
    return float(str(value))


def parse_bool(value: object) -> bool:
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"not a serialized Boolean: {value!r}")


def path_matches(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + ".") or path.startswith(prefix + "[")


def deep_diff_paths(left: object, right: object, prefix: str = "") -> set[str]:
    if type(left) is not type(right):
        return {prefix or "$"}
    if isinstance(left, dict):
        output: set[str] = set()
        for key in set(left) | set(right):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                output.add(path)
            else:
                output.update(deep_diff_paths(left[key], right[key], path))
        return output
    if isinstance(left, list):
        output = set()
        maximum = max(len(left), len(right))
        for index in range(maximum):
            path = f"{prefix}[{index}]"
            if index >= len(left) or index >= len(right):
                output.add(path)
            else:
                output.update(deep_diff_paths(left[index], right[index], path))
        return output
    return set() if left == right else {prefix or "$"}


def independent_parent_schedule() -> dict[str, set[str]]:
    assignments: dict[str, set[str]] = defaultdict(set)
    for operator_id, level in (
        ("O11", 1),
        ("O12", 2),
        ("O13", 3),
        ("O17", 4),
        ("O23", 5),
    ):
        assignments[operator_id].update(
            f"S4:L{level}:C{context}" for context in range(1, 5)
        )
    assignments["O22"].update(f"S5:L1:C{context}" for context in range(1, 5))
    assignments["O02"].update(f"S5:L5:C{context}" for context in range(1, 5))
    globals_cross = (
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
    def cells(stratum: str) -> list[str]:
        return [
            f"{stratum}:L{level}:C{context}"
            for level in range(1, 6)
            for context in range(1, 5)
        ]
    s5_middle = [
        f"S5:L{level}:C{context}"
        for level in range(2, 5)
        for context in range(1, 5)
    ]
    for index, operator_id in enumerate(globals_cross):
        assignments[operator_id].update(cells(s)[index] for s in ("S1", "S2", "S3"))
        assignments[operator_id].add(s5_middle[index])
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
        assignments[operator_id].update(
            cells(s)[12 + offset] for s in ("S1", "S2", "S3")
        )
        assignments[operator_id].add(extra)
    return dict(assignments)


def verify_import_independence(checks: Checks) -> None:
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
    checks.require(
        "independent_standard_library_imports_only",
        imports <= allowed,
        f"imports={sorted(imports)}",
    )


def verify_artifact_index(report_dir: Path, checks: Checks) -> None:
    index = load_json(report_dir / "artifact_index.json")
    records = index.get("files", [])
    checks.require("artifact_index_nonempty", bool(records), f"records={len(records)}")
    seen: set[str] = set()
    for record in records:
        relative = str(record["path"])
        path = resolve_project_path(relative)
        unique = relative not in seen
        seen.add(relative)
        checks.require(f"artifact_unique:{relative}", unique)
        checks.require(f"artifact_exists:{relative}", path.is_file())
        if path.is_file():
            checks.require(
                f"artifact_bytes:{relative}",
                path.stat().st_size == int(record["bytes"]),
            )
            checks.require(
                f"artifact_sha256:{relative}",
                sha256_file(path) == str(record["sha256"]),
            )
    protocol = resolve_project_path(PROTOCOL_RELATIVE)
    checks.require(
        "frozen_protocol_sha256",
        protocol.is_file() and sha256_file(protocol) == PROTOCOL_SHA256,
    )


def expected_positive_k(case: Mapping[str, object], evidence: Evidence) -> int:
    generated = evidence.generated_unit
    requested = str(case["requested_unit"])
    if generated == requested:
        return 1
    if generated == "window" and requested == "event":
        return 2 * evidence.event_kappa
    if generated == "window" and requested == "owner":
        return evidence.owner_kappa
    raise ValueError(f"unsupported positive route: {generated}->{requested}")


def verify_case_artifacts(
    cases: Sequence[Mapping[str, object]], checks: Checks
) -> tuple[dict[str, Evidence], dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    evidence_by_case: dict[str, Evidence] = {}
    report_by_case: dict[str, dict[str, object]] = {}
    audit_by_case: dict[str, dict[str, object]] = {}
    triple_hashes: list[str] = []
    for case in cases:
        case_id = str(case["case_id"])
        paths = {
            key: resolve_project_path(str(case[key]))
            for key in ("mapping_path", "report_path", "audit_path")
        }
        actual_hashes: list[str] = []
        for key, path in paths.items():
            digest = sha256_file(path)
            actual_hashes.append(digest)
            checks.require(
                f"case_hash:{case_id}:{key}",
                digest == str(case[f"{key}_sha256"]),
            )
        triple = canonical_json_sha256(actual_hashes)
        triple_hashes.append(triple)
        checks.require(
            f"case_triple_hash:{case_id}",
            triple == str(case["input_triple_sha256"]),
        )
        evidence = inspect_mapping(paths["mapping_path"])
        report = load_json(paths["report_path"])
        audit = load_json(paths["audit_path"])
        evidence_by_case[case_id] = evidence
        report_by_case[case_id] = report
        audit_by_case[case_id] = audit
        checks.require(
            f"mapping_metadata:{case_id}",
            evidence.selected_record_count == int(case["selected_record_count"])
            and evidence.event_kappa == int(case["observed_event_kappa"])
            and evidence.owner_kappa == int(case["observed_owner_kappa"])
            and evidence.generated_unit == str(case["generated_unit"]),
            (
                f"count={evidence.selected_record_count}, event={evidence.event_kappa}, "
                f"owner={evidence.owner_kappa}, unit={evidence.generated_unit}"
            ),
        )
        pipeline = report["pipeline"]
        manifest = pipeline["manifest"]
        report_bound = (
            pipeline["mapping_file_sha256"] == evidence.mapping_file_sha256
            and pipeline["selected_mapping_sha256"] == evidence.selected_mapping_sha256
            and manifest["mapping_file_sha256"] == evidence.mapping_file_sha256
            and manifest["selected_mapping_sha256"] == evidence.selected_mapping_sha256
            and int(manifest["selected_record_count"]) == evidence.selected_record_count
        )
        audit_bound = (
            audit["mapping_file_sha256"] == evidence.mapping_file_sha256
            and audit["selected_mapping_sha256"] == evidence.selected_mapping_sha256
            and int(audit["num_windows_used"]) == evidence.selected_record_count
            and int(audit["event_kappa_used"]) == evidence.event_kappa
            and int(audit["owner_windows_max_used"]) == evidence.owner_kappa
        )
        is_o03 = str(case.get("operator_id", "")) == "O03"
        checks.require(
            f"intentional_report_binding:{case_id}",
            (not report_bound) if is_o03 else report_bound,
        )
        checks.require(
            f"intentional_audit_binding:{case_id}",
            (not audit_bound) if is_o03 else audit_bound,
        )
    checks.require(
        "input_triples_unique",
        len(set(triple_hashes)) == len(triple_hashes),
        f"unique={len(set(triple_hashes))}/{len(triple_hashes)}",
    )
    return evidence_by_case, report_by_case, audit_by_case


def verify_positive_grid(
    positives: Sequence[Mapping[str, object]],
    evidence_by_case: Mapping[str, Evidence],
    report_by_case: Mapping[str, Mapping[str, object]],
    checks: Checks,
) -> None:
    strata = Counter(str(case["stratum_id"]) for case in positives)
    contexts = Counter(str(case["context_id"]) for case in positives)
    cells = Counter(str(case["coverage_cell"]) for case in positives)
    checks.require("positive_strata_20_each", strata == Counter({f"S{i}": 20 for i in range(1, 6)}), str(dict(strata)))
    checks.require("positive_contexts_25_each", contexts == Counter({f"C{i}": 25 for i in range(1, 5)}), str(dict(contexts)))
    checks.require("positive_cells_unique", len(cells) == 100 and max(cells.values()) == 1)
    by_cell = {str(case["coverage_cell"]): case for case in positives}
    for case in positives:
        case_id = str(case["case_id"])
        evidence = evidence_by_case[case_id]
        report = report_by_case[case_id]
        k = expected_positive_k(case, evidence)
        checks.require(f"positive_k:{case_id}", k == int(case["expected_k"]), f"recomputed={k}")
        privacy = report["privacy"]
        epsilon = float(privacy["epsilon"])
        delta = float(privacy["delta"])
        checks.require(
            f"positive_base_pair:{case_id}",
            numbers_match(epsilon, BASE_EPSILON) and numbers_match(delta, BASE_DELTA),
        )
        if k == 1:
            expected_epsilon, expected_delta = epsilon, delta
        else:
            expected_epsilon, expected_delta = group_conversion(epsilon, delta, k)
        checks.require(
            f"positive_numbers:{case_id}",
            numbers_match(expected_epsilon, float(case["expected_epsilon"]))
            and numbers_match(expected_delta, float(case["expected_delta"])),
        )
        checks.require(f"positive_nonvacuous:{case_id}", expected_delta < 1.0, f"delta={expected_delta}")
        contract = report["claim_contracts"][str(case["requested_unit"])]
        checks.require(
            f"positive_contract_k:{case_id}",
            int(contract["stability_bound"]) == k,
        )

    # C3/C4 preserve selected semantics of C1 but change whole-file bytes.
    for stratum in ("S1", "S2", "S3", "S4", "S5"):
        for level in range(1, 6):
            c1 = by_cell[f"{stratum}:L{level}:C1"]
            base = evidence_by_case[str(c1["case_id"])]
            for context in (3, 4):
                other_case = by_cell[f"{stratum}:L{level}:C{context}"]
                other = evidence_by_case[str(other_case["case_id"])]
                checks.require(
                    f"context_selected_invariance:{stratum}:L{level}:C{context}",
                    other.selected_mapping_sha256 == base.selected_mapping_sha256,
                )
                checks.require(
                    f"context_file_distinction:{stratum}:L{level}:C{context}",
                    other.mapping_file_sha256 != base.mapping_file_sha256,
                )
    k13_delta = group_conversion(BASE_EPSILON, BASE_DELTA, 13)[1]
    k14_delta = group_conversion(BASE_EPSILON, BASE_DELTA, 14)[1]
    checks.require("vacuity_boundary_k13_k14", k13_delta < 1 <= k14_delta, f"K13={k13_delta}, K14={k14_delta}")


def verify_pairings_and_mutations(
    positives: Sequence[Mapping[str, object]],
    negatives: Sequence[Mapping[str, object]],
    evidence_by_case: Mapping[str, Evidence],
    report_by_case: Mapping[str, Mapping[str, object]],
    audit_by_case: Mapping[str, Mapping[str, object]],
    checks: Checks,
) -> None:
    positives_by_id = {str(case["case_id"]): case for case in positives}
    parent_ids = [str(case["parent_case_id"]) for case in negatives]
    checks.require(
        "parent_one_to_one",
        Counter(parent_ids) == Counter(positives_by_id),
    )
    operator_counts = Counter(str(case["operator_id"]) for case in negatives)
    checks.require(
        "operators_four_each",
        operator_counts == Counter({f"O{i:02d}": 4 for i in range(1, 26)}),
        str(dict(operator_counts)),
    )
    status_counts = Counter(str(case["expected_release_status"]) for case in negatives)
    checks.require("nonallow_status_arithmetic", status_counts == EXPECTED_STATUS_COUNTS, str(dict(status_counts)))
    expected_schedule = independent_parent_schedule()
    observed_schedule: dict[str, set[str]] = defaultdict(set)

    for case in negatives:
        case_id = str(case["case_id"])
        operator = str(case["operator_id"])
        parent_id = str(case["parent_case_id"])
        parent = positives_by_id[parent_id]
        observed_schedule[operator].add(str(parent["coverage_cell"]))
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
        checks.require(
            f"paired_audit_unchanged:{case_id}",
            audit_by_case[case_id] == audit_by_case[parent_id],
        )
        child_evidence = evidence_by_case[case_id]
        parent_evidence = evidence_by_case[parent_id]
        if operator == "O03":
            checks.require(
                f"o03_mapping_changed:{case_id}",
                child_evidence.mapping_file_sha256 != parent_evidence.mapping_file_sha256,
            )
            checks.require(
                f"o03_report_unchanged:{case_id}",
                report_by_case[case_id] == report_by_case[parent_id],
            )
            child_rows = list(child_evidence.raw_rows)
            parent_rows = list(parent_evidence.raw_rows)
            row_diffs = deep_diff_paths(parent_rows, child_rows, "mapping.rows")
            valid_row_diff = len(row_diffs) == 1 and next(iter(row_diffs)).endswith(".end")
            checks.require(f"o03_single_end_mutation:{case_id}", valid_row_diff, str(sorted(row_diffs)))
        else:
            checks.require(
                f"paired_mapping_unchanged:{case_id}",
                child_evidence.mapping_file_sha256 == parent_evidence.mapping_file_sha256,
            )
            report_diffs = deep_diff_paths(
                report_by_case[parent_id], report_by_case[case_id]
            )
            direct = tuple(str(item) for item in case["direct_mutation_paths"])
            derived = tuple(str(item) for item in case["derived_rebind_paths"])
            allowed = direct + derived
            closed = bool(report_diffs) and all(
                any(path_matches(path, prefix) for prefix in allowed)
                for path in report_diffs
            )
            checks.require(
                f"mutation_closure:{case_id}",
                closed,
                f"diffs={sorted(report_diffs)}; allowed={list(allowed)}",
            )
            for direct_path in direct:
                checks.require(
                    f"direct_mutation_exercised:{case_id}:{direct_path}",
                    any(path_matches(path, direct_path) or path_matches(direct_path, path) for path in report_diffs),
                    str(sorted(report_diffs)),
                )
    checks.require(
        "exact_frozen_parent_schedule",
        dict(observed_schedule) == expected_schedule,
        f"observed={dict(observed_schedule)}",
    )


def recompute_prediction_flags(
    case: Mapping[str, object], report: Mapping[str, object], row: Mapping[str, object]
) -> dict[str, bool]:
    expected_status = str(case["expected_release_status"])
    predicted_status = str(row["predicted_release_status"])
    predicted_path = str(row["predicted_unit_path"])
    predicted_assurance = str(row["predicted_assurance_status"])
    predicted_epsilon = parse_optional_float(row["predicted_epsilon"])
    predicted_delta = parse_optional_float(row["predicted_delta"])
    expected_epsilon = parse_optional_float(case.get("expected_epsilon"))
    expected_delta = parse_optional_float(case.get("expected_delta"))
    allowed_truth = expected_status == "ALLOWED"
    predicted_allowed = predicted_status == "ALLOWED"
    unsafe = not allowed_truth and predicted_allowed
    binary = allowed_truth == predicted_allowed
    valid_exact = (
        allowed_truth
        and predicted_allowed
        and predicted_path == str(case["expected_unit_path"])
        and numbers_match(predicted_epsilon, expected_epsilon)
        and numbers_match(predicted_delta, expected_delta)
    )
    exact = (
        predicted_status == expected_status
        and predicted_path == str(case["expected_unit_path"])
        and predicted_assurance == str(case["expected_assurance_status"])
        and (
            not allowed_truth
            or (
                numbers_match(predicted_epsilon, expected_epsilon)
                and numbers_match(predicted_delta, expected_delta)
            )
        )
    )
    privacy = report.get("privacy", {})
    base_epsilon = parse_optional_float(privacy.get("epsilon")) if isinstance(privacy, dict) else None
    base_delta = parse_optional_float(privacy.get("delta")) if isinstance(privacy, dict) else None
    wrong_same = (
        allowed_truth
        and str(case["expected_unit_path"]) in {"CONVERT", "GROUP"}
        and predicted_allowed
        and numbers_match(predicted_epsilon, base_epsilon)
        and numbers_match(predicted_delta, base_delta)
    )
    return {
        "supported_positive": predicted_allowed,
        "unsafe_positive": unsafe,
        "binary_safe_correct": binary,
        "valid_decision_exact": valid_exact,
        "exact_tuple": exact,
        "wrong_same_number": wrong_same,
        "valid_retained": allowed_truth and predicted_allowed,
    }


def verify_predictions_and_aggregates(
    report_dir: Path,
    cases: Sequence[Mapping[str, object]],
    report_by_case: Mapping[str, Mapping[str, object]],
    checks: Checks,
) -> None:
    cases_by_id = {str(case["case_id"]): case for case in cases}
    with (report_dir / "predictions.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    keys = [(row["case_id"], row["method"]) for row in rows]
    checks.require("prediction_row_count", len(rows) == 1600, f"rows={len(rows)}")
    checks.require("prediction_keys_unique", len(set(keys)) == len(keys))
    checks.require(
        "prediction_cartesian_product",
        Counter(row["case_id"] for row in rows) == Counter({case_id: 8 for case_id in cases_by_id})
        and Counter(row["method"] for row in rows) == Counter({method: 200 for method in METHODS}),
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
    parsed_rows: list[dict[str, object]] = []
    for row in rows:
        case_id = row["case_id"]
        case = cases_by_id[case_id]
        checks.require(
            f"prediction_truth_fields:{case_id}:{row['method']}",
            row["failure_family"] == str(case["failure_family"])
            and row["expected_release_status"] == str(case["expected_release_status"])
            and row["expected_unit_path"] == str(case["expected_unit_path"])
            and row["expected_assurance_status"] == str(case["expected_assurance_status"]),
        )
        recomputed = recompute_prediction_flags(case, report_by_case[case_id], row)
        for field_name in bool_fields:
            checks.require(
                f"prediction_flag:{case_id}:{row['method']}:{field_name}",
                parse_bool(row[field_name]) == recomputed[field_name],
            )
        parsed = dict(row)
        parsed.update(recomputed)
        parsed_rows.append(parsed)
        if row["method"] == "FULL":
            required = set(str(item) for item in case["required_issue_codes"])
            issues = set(row["issue_codes"].split(";"))
            checks.require(
                f"full_required_diagnostics:{case_id}", required <= issues, str(sorted(required - issues))
            )
    full = [row for row in parsed_rows if row["method"] == "FULL"]
    checks.require("full_zero_unsafe", not any(row["unsafe_positive"] for row in full))
    checks.require("full_all_exact", all(row["exact_tuple"] for row in full))

    recomputed_aggregates: dict[str, dict[str, float | int | str]] = {}
    for method in METHODS:
        selected = [row for row in parsed_rows if row["method"] == method]
        blocked = [row for row in selected if row["expected_release_status"] != "ALLOWED"]
        valid = [row for row in selected if row["expected_release_status"] == "ALLOWED"]
        walls = sorted(float(row["wall_ms"]) for row in selected)
        unsafe = sum(bool(row["unsafe_positive"]) for row in blocked)
        binary = sum(bool(row["binary_safe_correct"]) for row in selected)
        valid_exact = sum(bool(row["valid_decision_exact"]) for row in valid)
        exact = sum(bool(row["exact_tuple"]) for row in selected)
        retained = sum(bool(row["valid_retained"]) for row in valid)
        same = sum(bool(row["wrong_same_number"]) for row in selected)
        recomputed_aggregates[method] = {
            "method": method,
            "cases": len(selected),
            "blocked_truth_cases": len(blocked),
            "unsafe_positives": unsafe,
            "unsafe_positive_rate": unsafe / len(blocked),
            "binary_safe_correct": binary,
            "binary_safe_accuracy": binary / len(selected),
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
    with (report_dir / "aggregate_metrics.csv").open("r", newline="", encoding="utf-8") as handle:
        aggregate_rows = [dict(row) for row in csv.DictReader(handle)]
    checks.require("aggregate_methods_once", Counter(row["method"] for row in aggregate_rows) == Counter(METHODS))
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
    for row in aggregate_rows:
        expected = recomputed_aggregates[row["method"]]
        for field_name, expected_value in expected.items():
            if field_name == "method":
                continue
            if field_name in integer_fields:
                matches = int(row[field_name]) == int(expected_value)
            else:
                matches = numbers_match(float(row[field_name]), float(expected_value))
            checks.require(f"aggregate:{row['method']}:{field_name}", matches)
    for method in METHODS[:5]:
        checks.require(
            f"simple_baseline_unsafe:{method}",
            int(recomputed_aggregates[method]["unsafe_positives"]) > 0,
        )
    a1 = [
        row for row in parsed_rows
        if row["method"] == "A1_FULL_MINUS_RUNTIME"
        and row["unsafe_positive"]
        and str(cases_by_id[row["case_id"]]["operator_id"]) in {"O01", "O25"}
    ]
    a2 = [
        row for row in parsed_rows
        if row["method"] == "A2_FULL_MINUS_VACUITY"
        and row["unsafe_positive"]
        and str(cases_by_id[row["case_id"]]["operator_id"]) == "O02"
    ]
    checks.require("a1_runtime_artifact_unsafe", bool(a1))
    checks.require("a2_vacuity_unsafe", bool(a2))


def verify_matrix_exports(
    report_dir: Path,
    positives: Sequence[Mapping[str, object]],
    negatives: Sequence[Mapping[str, object]],
    checks: Checks,
) -> None:
    with (report_dir / "coverage_matrix.csv").open("r", newline="", encoding="utf-8") as handle:
        coverage = [dict(row) for row in csv.DictReader(handle)]
    with (report_dir / "operator_parent_matrix.csv").open("r", newline="", encoding="utf-8") as handle:
        operators = [dict(row) for row in csv.DictReader(handle)]
    checks.require(
        "coverage_export_ids",
        Counter(row["case_id"] for row in coverage)
        == Counter(str(case["case_id"]) for case in positives),
    )
    checks.require(
        "operator_export_ids",
        Counter(row["case_id"] for row in operators)
        == Counter(str(case["case_id"]) for case in negatives),
    )
    for filename, dimension in (
        ("confusion_by_context.csv", "context_id"),
        ("confusion_by_stratum.csv", "stratum_id"),
        ("confusion_by_failure_family.csv", "failure_family"),
    ):
        with (report_dir / filename).open("r", newline="", encoding="utf-8") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
        checks.require(f"breakdown_nonempty:{dimension}", bool(rows))
        checks.require(
            f"breakdown_methods:{dimension}",
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
    verify_import_independence(checks)
    verify_artifact_index(report_dir, checks)

    document = load_json(report_dir / "cases.json")
    cases = document["cases"]
    checks.require("case_schema_version", document.get("schema_version") == "validator_ablation_cases_v3.0")
    checks.require("case_protocol_hash", document.get("protocol_sha256") == PROTOCOL_SHA256)
    checks.require("case_count_200", len(cases) == 200, f"cases={len(cases)}")
    ids = [str(case["case_id"]) for case in cases]
    checks.require("case_ids_unique", len(set(ids)) == len(ids))
    positives = [case for case in cases if case["case_kind"] == "allowed"]
    negatives = [case for case in cases if case["case_kind"] == "non_allow"]
    checks.require("case_split_100_100", len(positives) == len(negatives) == 100)

    evidence, reports, audits = verify_case_artifacts(cases, checks)
    verify_positive_grid(positives, evidence, reports, checks)
    verify_pairings_and_mutations(
        positives, negatives, evidence, reports, audits, checks
    )
    verify_predictions_and_aggregates(report_dir, cases, reports, checks)
    verify_matrix_exports(report_dir, positives, negatives, checks)

    execution = load_json(report_dir / "execution_status.json")
    checks.require(
        "runner_execution_gates_passed",
        execution.get("gates_passed") is True and not execution.get("gate_errors"),
        str(execution.get("gate_errors")),
    )
    output = {
        "schema_version": "validator_ablation_independent_verification_v3.0",
        "verifier": Path(__file__).relative_to(PROJECT_ROOT).as_posix(),
        "standard_library_only": True,
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
    if checks.failures:
        for failure in checks.failures:
            print(f"FAIL {failure['name']}: {failure['detail']}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Independent verification passed: {len(checks.records)} checks")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
