"""Run the strengthened, unchanged-case V3.1 conformance experiment.

V3.1 preserves every frozen V3 case and expected tuple.  Its only scientific
method change is a pure runtime-obligation counterfactual: A1 supplies a trace
that is internally matched to the case's current static report, rather than a
default trace that can introduce unrelated runtime conflicts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

from jsonschema import Draft202012Validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for candidate in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import run_validator_ablation_v2 as v2  # noqa: E402
import run_validator_ablation_v3 as v3  # noqa: E402


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
SIMPLE_BASELINES = METHODS[:5]
V3_RUNNER_SHA256 = "0106b39e2f6a13a8199b0386a02cc521112694f738573f88bfb24b794669df4b"
V3_REFERENCE_DIR = PROJECT_ROOT / "reports" / "validator_ablation_v3_200_001"
V3_REFERENCE_HASHES = {
    "cases.json": "6ece4b3a5cf2a2fa4ac322939b59400e9cfd88396b8357516b80b7c4c610fce9",
    "predictions.csv": "b8baaa0efbf7495a3b12cb53695fe50713b0441f6fa5f8f9c6e700b9191cca6b",
    "aggregate_metrics.csv": "bc935af8966b5ebfb11c158843d014dbbb8f6bf05aeffd7bff3d7f7059675270",
    "execution_status.json": "fe74ee2b159958a70cedcf01487a91878ca614ed12c3271a817fc23ac2e83048",
}
COMPLETE_DEPENDENCIES = (
    "docs/mechanism_registry_v2_0.json",
    "scripts/wisdm_v2_gold_pipeline.py",
    "scripts/rdp_accountant_lite.py",
)
REFERENCE_CASE_FIELDS = (
    "case_id",
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_versioned_inputs() -> None:
    v3.verify_frozen_inputs()
    if (
        sha256_file(PROJECT_ROOT / "scripts" / "run_validator_ablation_v3.py")
        != V3_RUNNER_SHA256
    ):
        raise RuntimeError("The preserved V3 generator dependency changed")
    for filename, expected in V3_REFERENCE_HASHES.items():
        path = V3_REFERENCE_DIR / filename
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"The retained V3 reference changed: {filename}")
    registry = json.loads(
        (PROJECT_ROOT / COMPLETE_DEPENDENCIES[0]).read_text(encoding="utf-8")
    )
    entry = registry["mechanisms"]["secure_systemrandom_dpsgd@2.0.0"]
    if sha256_file(PROJECT_ROOT / entry["code_artifact_id"]) != entry["code_sha256"]:
        raise RuntimeError("Registered pipeline source hash does not match registry")
    if (
        sha256_file(PROJECT_ROOT / entry["accountant_checker_artifact_id"])
        != entry["accountant_checker_sha256"]
    ):
        raise RuntimeError("Registered accountant checker hash does not match registry")


def runtime_assumed_matched(
    report: Mapping[str, object], mapping_path: Path, claim_unit: str
) -> dict[str, object]:
    """Construct an ideal trace matched to the existing static declaration.

    This changes only runtime evidence.  In particular, it does not rebind a
    stale mapping, repair schema/mechanism/accountant premises, or rewrite the
    pipeline manifest.
    """

    mechanism = report["mechanism"]
    privacy = report["privacy"]
    pipeline = report["pipeline"]
    assert isinstance(mechanism, dict)
    assert isinstance(privacy, dict)
    assert isinstance(pipeline, dict)
    manifest = pipeline["manifest"]
    assert isinstance(manifest, dict)
    policy = str(manifest.get("record_identity_policy") or "public_fixed_ids")
    reference = v2.make_report(
        mapping_path,
        claim_unit,
        record_identity_policy=policy,
    )
    reference_runtime = reference["runtime_trace"]
    assert isinstance(reference_runtime, dict)
    evidence = copy.deepcopy(reference_runtime["evidence"])
    assert isinstance(evidence, dict)
    evidence.update(
        {
            "executed_updates": mechanism["steps"],
            "released_model_sha256": manifest["released_model_sha256"],
            "batch_trace_sha256": manifest["batch_trace_sha256"],
            "secure_sampling_rng": mechanism["sampling_implementation"],
            "secure_noise_rng": mechanism["noise_generation"],
        }
    )
    evidence_hash = v2.canonical_json_sha256(evidence)
    expected = {
        **{key: mechanism.get(key) for key in v2.RUNTIME_EXACT_FIELDS},
        **{key: mechanism.get(key) for key in v2.RUNTIME_NUMERIC_FIELDS},
        "code_artifact_id": manifest["code_artifact_id"],
        "code_sha256": manifest["code_sha256"],
        "accountant_checker_artifact_id": manifest["accountant_checker_artifact_id"],
        "accountant_checker_sha256": manifest["accountant_checker_sha256"],
        "selected_mapping_sha256": pipeline["selected_mapping_sha256"],
        "pipeline_sha256": pipeline["pipeline_sha256"],
        "epsilon": privacy["epsilon"],
        "delta": privacy["delta"],
        "delta_convention": privacy["delta_convention"],
        "runtime_evidence_sha256": evidence_hash,
    }
    payload = {
        key: expected.get(key)
        for key in (*v2.RUNTIME_EXACT_FIELDS, *v2.RUNTIME_NUMERIC_FIELDS)
    }
    runtime: dict[str, object] = {
        "status": "matched",
        **payload,
        "evidence": evidence,
    }
    runtime["trace_hash"] = v2.canonical_json_sha256(payload)
    return runtime


def predict(
    method: str,
    report: Mapping[str, object],
    audit: Mapping[str, str],
    mapping_path: Path,
    claim_unit: str,
    schema_validator: Draft202012Validator,
) -> v2.Prediction:
    if method == "A1_RUNTIME_ASSUMED_MATCHED":
        counterfactual = copy.deepcopy(report)
        counterfactual["runtime_trace"] = runtime_assumed_matched(
            counterfactual, mapping_path, claim_unit
        )
        return v2.full_prediction(counterfactual, audit, mapping_path, claim_unit)
    return v2.predict(
        method,
        report,
        audit,
        mapping_path,
        claim_unit,
        schema_validator,
    )


def build_prediction_rows(
    cases: Sequence[v2.Case], schema_validator: Draft202012Validator
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in cases:
        report = json.loads(case.report_path.read_text(encoding="utf-8"))
        audit = json.loads(case.audit_path.read_text(encoding="utf-8"))
        base_epsilon, base_delta = v2.base_numbers(report)
        for method in METHODS:
            started = time.perf_counter()
            prediction = predict(
                method,
                report,
                audit,
                case.mapping_path,
                case.claim_unit,
                schema_validator,
            )
            wall_ms = (time.perf_counter() - started) * 1000
            unsafe = (
                case.expected_release_status != "ALLOWED"
                and prediction.release_status == "ALLOWED"
            )
            binary_safe = (case.expected_release_status == "ALLOWED") == (
                prediction.release_status == "ALLOWED"
            )
            valid_exact = (
                case.expected_release_status == "ALLOWED"
                and prediction.release_status == "ALLOWED"
                and prediction.unit_path == case.expected_unit_path
                and v2.numbers_match(prediction.epsilon, case.expected_epsilon)
                and v2.numbers_match(prediction.delta, case.expected_delta)
            )
            wrong_same_number = (
                case.expected_release_status == "ALLOWED"
                and case.expected_unit_path in {"CONVERT", "GROUP"}
                and prediction.release_status == "ALLOWED"
                and v2.numbers_match(prediction.epsilon, base_epsilon)
                and v2.numbers_match(prediction.delta, base_delta)
            )
            rows.append(
                {
                    "case_id": case.case_id,
                    "failure_family": case.failure_family,
                    "expected_release_status": case.expected_release_status,
                    "expected_unit_path": case.expected_unit_path,
                    "expected_assurance_status": case.expected_assurance_status,
                    "method": method,
                    "predicted_release_status": prediction.release_status,
                    "predicted_unit_path": prediction.unit_path,
                    "predicted_assurance_status": prediction.assurance_status,
                    "predicted_epsilon": (
                        "" if prediction.epsilon is None else repr(prediction.epsilon)
                    ),
                    "predicted_delta": (
                        "" if prediction.delta is None else repr(prediction.delta)
                    ),
                    "issue_codes": ";".join(prediction.issue_codes),
                    "supported_positive": prediction.release_status == "ALLOWED",
                    "unsafe_positive": unsafe,
                    "binary_safe_correct": binary_safe,
                    "valid_decision_exact": valid_exact,
                    "exact_tuple": v2.exact_tuple(case, prediction),
                    "wrong_same_number": wrong_same_number,
                    "valid_retained": (
                        case.expected_release_status == "ALLOWED"
                        and prediction.release_status == "ALLOWED"
                    ),
                    "wall_ms": f"{wall_ms:.6f}",
                }
            )
    return rows


def aggregate(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        blocked = [
            row for row in selected if row["expected_release_status"] != "ALLOWED"
        ]
        valid = [row for row in selected if row["expected_release_status"] == "ALLOWED"]
        wall_values = sorted(float(row["wall_ms"]) for row in selected)
        unsafe = sum(bool(row["unsafe_positive"]) for row in blocked)
        binary = sum(bool(row["binary_safe_correct"]) for row in selected)
        valid_exact = sum(bool(row["valid_decision_exact"]) for row in valid)
        exact = sum(bool(row["exact_tuple"]) for row in selected)
        retained = sum(bool(row["valid_retained"]) for row in valid)
        same_number = sum(bool(row["wrong_same_number"]) for row in selected)
        output.append(
            {
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
                "wrong_same_number": same_number,
                "median_wall_ms": wall_values[len(wall_values) // 2],
                "max_wall_ms": max(wall_values),
            }
        )
    return output


def compare_with_v3(
    records: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[str]]:
    reference_document = json.loads(
        (V3_REFERENCE_DIR / "cases.json").read_text(encoding="utf-8")
    )
    reference = {str(case["case_id"]): case for case in reference_document["cases"]}
    comparison: list[dict[str, object]] = []
    errors: list[str] = []
    for record in records:
        case_id = str(record["case_id"])
        old = reference.get(case_id)
        if old is None:
            errors.append(f"V3 reference is missing {case_id}")
            continue
        input_equal = all(
            record[f"{key}_sha256"] == old[f"{key}_sha256"]
            for key in ("mapping_path", "report_path", "audit_path")
        )
        oracle_equal = all(
            json.dumps(record[field], sort_keys=True, allow_nan=False)
            == json.dumps(old[field], sort_keys=True, allow_nan=False)
            for field in REFERENCE_CASE_FIELDS
        )
        comparison.append(
            {
                "case_id": case_id,
                "mapping_sha256": record["mapping_path_sha256"],
                "report_sha256": record["report_path_sha256"],
                "audit_sha256": record["audit_path_sha256"],
                "input_content_equal": input_equal,
                "oracle_metadata_equal": oracle_equal,
            }
        )
        if not input_equal or not oracle_equal:
            errors.append(
                f"V3 lineage mismatch for {case_id}: "
                f"input={input_equal}, oracle={oracle_equal}"
            )
    if len(comparison) != 200 or len(reference) != 200:
        errors.append(
            f"V3 lineage count mismatch: current={len(comparison)}, reference={len(reference)}"
        )
    return comparison, errors


def prediction_gate_errors(
    cases: Sequence[v2.Case],
    metadata: Mapping[str, v3.CaseMetadata],
    rows: Sequence[Mapping[str, object]],
    aggregates: Sequence[Mapping[str, object]],
) -> list[str]:
    errors: list[str] = []
    by_case = {case.case_id: case for case in cases}
    full = [row for row in rows if row["method"] == "FULL"]
    if len(full) != 200 or not all(bool(row["exact_tuple"]) for row in full):
        errors.append("FULL did not match all 200 frozen tuples")
    if any(bool(row["unsafe_positive"]) for row in full):
        errors.append("FULL emitted an unsafe positive")
    for row in full:
        case_id = str(row["case_id"])
        required = set(metadata[case_id].required_issue_codes)
        issues = set(str(row["issue_codes"]).split(";"))
        if not required <= issues:
            errors.append(
                f"{case_id} lacks required diagnostics {sorted(required - issues)}"
            )
        if metadata[case_id].case_kind == "non_allow":
            case = by_case[case_id]
            report = json.loads(case.report_path.read_text(encoding="utf-8"))
            audit = json.loads(case.audit_path.read_text(encoding="utf-8"))
            raw = v2.validate_privacy_claim(
                audit, report, case.claim_unit, mapping_path=case.mapping_path
            )
            if raw.supported_statement:
                errors.append(f"{case_id} retained forbidden positive wording")
    for method in SIMPLE_BASELINES:
        row = next(item for item in aggregates if item["method"] == method)
        if int(row["unsafe_positives"]) == 0:
            errors.append(f"{method} exposes no unsafe positive")

    operator = {case_id: meta.operator_id for case_id, meta in metadata.items()}
    a1 = [row for row in rows if row["method"] == "A1_RUNTIME_ASSUMED_MATCHED"]
    a1_unsafe = Counter(
        operator[str(row["case_id"])] for row in a1 if bool(row["unsafe_positive"])
    )
    a1_nonexact = Counter(
        operator[str(row["case_id"])] for row in a1 if not bool(row["exact_tuple"])
    )
    expected_a1 = Counter({"O01": 4, "O06": 4, "O25": 4})
    if a1_unsafe != expected_a1 or a1_nonexact != expected_a1:
        errors.append(
            f"Pure A1 operator attribution differs: unsafe={a1_unsafe}, "
            f"nonexact={a1_nonexact}"
        )
    a1_aggregate = next(
        item for item in aggregates if item["method"] == "A1_RUNTIME_ASSUMED_MATCHED"
    )
    if (
        int(a1_aggregate["unsafe_positives"]) != 12
        or int(a1_aggregate["valid_exact_decisions"]) != 100
        or int(a1_aggregate["exact_tuples"]) != 188
    ):
        errors.append(f"Pure A1 aggregate differs: {a1_aggregate}")

    a2 = [row for row in rows if row["method"] == "A2_FULL_MINUS_VACUITY"]
    a2_unsafe = Counter(
        operator[str(row["case_id"])] for row in a2 if bool(row["unsafe_positive"])
    )
    a2_nonexact = Counter(
        operator[str(row["case_id"])] for row in a2 if not bool(row["exact_tuple"])
    )
    expected_a2 = Counter({"O02": 4})
    if a2_unsafe != expected_a2 or a2_nonexact != expected_a2:
        errors.append(
            f"A2 operator attribution differs: unsafe={a2_unsafe}, "
            f"nonexact={a2_nonexact}"
        )
    return errors


def breakdown_rows(
    rows: Sequence[Mapping[str, object]],
    metadata: Mapping[str, v3.CaseMetadata],
    dimension: str,
) -> list[dict[str, object]]:
    values = sorted(
        {str(getattr(metadata[str(row["case_id"])], dimension)) for row in rows}
    )
    return [
        {
            "method": method,
            dimension: value,
            "cases": len(selected),
            "unsafe_positives": sum(bool(row["unsafe_positive"]) for row in selected),
            "valid_exact_decisions": sum(
                bool(row["valid_decision_exact"]) for row in selected
            ),
            "exact_tuples": sum(bool(row["exact_tuple"]) for row in selected),
        }
        for method in METHODS
        for value in values
        for selected in [
            [
                row
                for row in rows
                if row["method"] == method
                and str(getattr(metadata[str(row["case_id"])], dimension)) == value
            ]
        ]
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="reports/validator_ablation_v3_1_200_001"
    )
    return parser.parse_args()


def unique_paths(paths: Sequence[Path]) -> list[Path]:
    output: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            output.append(resolved)
    return output


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite retained run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    verify_versioned_inputs()

    positives, positive_metadata = v3.build_positive_cases(output_dir)
    negatives, negative_metadata = v3.build_negative_cases(
        output_dir, positives, positive_metadata
    )
    cases = positives + negatives
    metadata = {**positive_metadata, **negative_metadata}
    gate_errors = v3.structural_gate_errors(cases, metadata)
    records = v3.case_records(cases, metadata)
    lineage_rows, lineage_errors = compare_with_v3(records)
    gate_errors.extend(lineage_errors)

    schema_path = PROJECT_ROOT / "docs" / "privacy_report_schema_v2_0.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    predictions = build_prediction_rows(cases, validator)
    aggregates = aggregate(predictions)
    gate_errors.extend(prediction_gate_errors(cases, metadata, predictions, aggregates))

    cases_path = output_dir / "cases.json"
    predictions_path = output_dir / "predictions.csv"
    aggregate_path = output_dir / "aggregate_metrics.csv"
    coverage_path = output_dir / "coverage_matrix.csv"
    operator_path = output_dir / "operator_parent_matrix.csv"
    family_path = output_dir / "confusion_by_failure_family.csv"
    context_path = output_dir / "confusion_by_context.csv"
    stratum_path = output_dir / "confusion_by_stratum.csv"
    lineage_path = output_dir / "v3_input_lineage_equivalence.json"
    status_path = output_dir / "execution_status.json"
    summary_path = output_dir / "summary.md"

    v2.write_json(
        cases_path,
        {
            "schema_version": "validator_ablation_cases_v3.1",
            "protocol_path": v3.PROTOCOL_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "protocol_sha256": v3.PROTOCOL_SHA256,
            "ground_truth_source": (
                "Unchanged frozen V3 grid/operator oracle; content hashes and "
                "oracle metadata are compared case-by-case with retained V3."
            ),
            "method_change": (
                "A1 supplies runtime evidence matched to each current static "
                "report without repairing any non-runtime premise."
            ),
            "cases": records,
        },
    )
    v2.write_csv(predictions_path, predictions)
    v2.write_csv(aggregate_path, aggregates)
    v2.write_json(
        lineage_path,
        {
            "schema_version": "validator_ablation_v3_lineage_equivalence_v1.0",
            "reference_cases_path": (V3_REFERENCE_DIR / "cases.json")
            .relative_to(PROJECT_ROOT)
            .as_posix(),
            "reference_cases_sha256": V3_REFERENCE_HASHES["cases.json"],
            "all_200_input_content_equal": all(
                bool(row["input_content_equal"]) for row in lineage_rows
            ),
            "all_200_oracle_metadata_equal": all(
                bool(row["oracle_metadata_equal"]) for row in lineage_rows
            ),
            "cases": lineage_rows,
        },
    )
    v2.write_csv(
        coverage_path,
        [
            {"case_id": case.case_id, **asdict(metadata[case.case_id])}
            for case in positives
        ],
    )
    v2.write_csv(
        operator_path,
        [
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
        ],
    )
    family_rows = [
        {
            "method": method,
            "failure_family": family,
            "cases": len(selected),
            "unsafe_positives": sum(bool(row["unsafe_positive"]) for row in selected),
            "valid_exact_decisions": sum(
                bool(row["valid_decision_exact"]) for row in selected
            ),
            "exact_tuples": sum(bool(row["exact_tuple"]) for row in selected),
        }
        for method in METHODS
        for family in sorted({case.failure_family for case in cases})
        for selected in [
            [
                row
                for row in predictions
                if row["method"] == method and row["failure_family"] == family
            ]
        ]
    ]
    v2.write_csv(family_path, family_rows)
    v2.write_csv(context_path, breakdown_rows(predictions, metadata, "context_id"))
    v2.write_csv(stratum_path, breakdown_rows(predictions, metadata, "stratum_id"))
    v2.write_json(
        status_path,
        {
            "schema_version": "validator_ablation_execution_v3.1",
            "gates_passed": not gate_errors,
            "gate_errors": gate_errors,
            "case_count": len(cases),
            "allowed_count": len(positives),
            "non_allow_count": len(negatives),
            "v3_input_content_equal": not lineage_errors,
        },
    )

    lines = [
        "# Validator Conformance/Ablation V3.1 (Unchanged 200 Cases)",
        "",
        "Designed coverage suite: 100 allowed and 100 paired non-allow cases.",
        "Fractions are deterministic case outcomes, not statistical estimates.",
        "A1 assumes runtime evidence is matched to the current static report;",
        "it does not repair any mapping, schema, mechanism, or accountant premise.",
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
    lines.extend(
        [
            "",
            "All 200 case input triples and oracle metadata match retained V3 by content hash.",
            f"Execution gates passed: **{not gate_errors}**.",
        ]
    )
    if gate_errors:
        lines.extend(["", "Gate failures:"] + [f"- {error}" for error in gate_errors])
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    dependency_paths = [
        Path(__file__).resolve(),
        PROJECT_ROOT / "scripts" / "verify_validator_ablation_v3_2.py",
        PROJECT_ROOT / "scripts" / "run_validator_ablation_v3.py",
        v3.PROTOCOL_PATH,
        *[PROJECT_ROOT / relative for relative in v3.FROZEN_INPUT_HASHES],
        *[PROJECT_ROOT / relative for relative in COMPLETE_DEPENDENCIES],
        PROJECT_ROOT / "docs" / "privacy_claim_contract_v2_0.md",
        PROJECT_ROOT / "docs" / "privacy_claim_truth_table_v2_0.md",
        schema_path,
        V3_REFERENCE_DIR / "cases.json",
        *[
            path
            for case in cases
            for path in (case.mapping_path, case.report_path, case.audit_path)
        ],
        cases_path,
        predictions_path,
        aggregate_path,
        coverage_path,
        operator_path,
        family_path,
        context_path,
        stratum_path,
        lineage_path,
        status_path,
        summary_path,
    ]
    artifact_index = {
        "schema_version": "validator_ablation_artifact_v3.1",
        "completeness_rule": (
            "Includes the actual runner/verifier, registry, registered source, "
            "accountant checker, all 600 raw case inputs, and result tables."
        ),
        "files": [
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in unique_paths(dependency_paths)
        ],
    }
    v2.write_json(output_dir / "artifact_index.json", artifact_index)
    if gate_errors:
        raise RuntimeError(
            "Strengthened run retained with gate failures; see execution_status.json"
        )
    print(f"Wrote {summary_path}")
    print("FULL exact tuples: 200/200")
    print("Pure A1 exact tuples: 188/200")


if __name__ == "__main__":
    main()
