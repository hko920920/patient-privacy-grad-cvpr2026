#!/usr/bin/env python3
"""Independently verify the TensorFlow Privacy statement baseline bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_MODULE_SHA256 = (
    "8e713fc815c9005b9933df08acb9efc9df32dd4ea17bfb54348c6b40386cba25"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def as_bool(value: object) -> bool:
    if value is True or value == "True":
        return True
    if value is False or value == "False":
        return False
    raise ValueError(f"Not a serialized Boolean: {value!r}")


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def require(self, identifier: str, condition: bool, detail: str) -> None:
        row = {"id": identifier, "passed": bool(condition), "detail": detail}
        self.rows.append(row)
        if not condition:
            raise RuntimeError(f"{identifier}: {detail}")


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def display_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def verify_artifact_index(bundle: Path, checks: Checks) -> dict[str, Any]:
    index_path = bundle / "artifact_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    checks.require(
        "artifact_schema",
        index.get("schema_version")
        == "tfprivacy_statement_baseline_artifact_v2.0"
        and index.get("official_module_sha256") == OFFICIAL_MODULE_SHA256,
        display_path(index_path),
    )
    entries = index.get("files")
    checks.require(
        "artifact_index_nonempty",
        isinstance(entries, list) and len(entries) >= 10,
        f"entries={len(entries) if isinstance(entries, list) else 'invalid'}",
    )
    assert isinstance(entries, list)
    for entry in entries:
        path = resolve_project_path(str(entry["path"]))
        checks.require(
            f"artifact_hash:{entry['path']}",
            path.is_file()
            and path.stat().st_size == int(entry["bytes"])
            and sha256_file(path) == entry["sha256"],
            display_path(path),
        )
    return index


def rerun_official_worker(
    bundle: Path,
    interpreter: Path,
    checks: Checks,
) -> None:
    worker = PROJECT_ROOT / "scripts" / "tfprivacy_statement_worker.py"
    with tempfile.TemporaryDirectory(prefix="tfprivacy_verify_") as temp:
        rerun_path = Path(temp) / "official_statements_rerun.json"
        completed = subprocess.run(
            [
                str(interpreter),
                str(worker),
                "--input",
                str(bundle / "requests.json"),
                "--output",
                str(rerun_path),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        checks.require(
            "official_worker_rerun_exit",
            completed.returncode == 0 and rerun_path.is_file(),
            f"returncode={completed.returncode}; stderr={completed.stderr}",
        )
        expected = json.loads(
            (bundle / "official_statements.json").read_text(encoding="utf-8")
        )
        actual = json.loads(rerun_path.read_text(encoding="utf-8"))
        checks.require(
            "official_worker_rerun_exact",
            actual == expected,
            "official statements reproduce byte-semantically as JSON",
        )


def verify_requests_and_statements(bundle: Path, checks: Checks) -> None:
    requests = json.loads((bundle / "requests.json").read_text(encoding="utf-8"))
    statements = json.loads(
        (bundle / "official_statements.json").read_text(encoding="utf-8")
    )
    request_rows = requests.get("requests")
    result_rows = statements.get("results")
    checks.require(
        "request_schema_and_count",
        requests.get("schema_version") == "tfprivacy_statement_requests_v2.0"
        and requests.get("expected_module_sha256") == OFFICIAL_MODULE_SHA256
        and isinstance(request_rows, list)
        and len(request_rows) == 22,
        "22 scalar-valid requests",
    )
    checks.require(
        "worker_schema_versions_and_count",
        statements.get("schema_version")
        == "tfprivacy_statement_worker_output_v2.0"
        and statements.get("tensorflow_privacy_version") == "0.9.0"
        and statements.get("dp_accounting_version") == "0.4.3"
        and statements.get("module_sha256") == OFFICIAL_MODULE_SHA256
        and isinstance(result_rows, list)
        and len(result_rows) == 22,
        "official module/version binding",
    )
    assert isinstance(request_rows, list)
    assert isinstance(result_rows, list)
    request_by_id = {str(row["case_id"]): row for row in request_rows}
    result_by_id = {str(row["case_id"]): row for row in result_rows}
    checks.require(
        "request_result_coverage",
        len(request_by_id) == 22
        and len(result_by_id) == 22
        and set(request_by_id) == set(result_by_id),
        "unique matching case IDs",
    )
    for case_id, result in result_by_id.items():
        statement = str(result["statement"])
        checks.require(
            f"statement_binding:{case_id}",
            result["request"] == request_by_id[case_id]
            and result["statement_sha256"] == sha256_text(statement)
            and result["statement_mentions_add_remove"] is True
            and result["statement_mentions_poisson_caveat"] is True,
            case_id,
        )


def verify_predictions_and_metrics(bundle: Path, checks: Checks) -> None:
    case_document = json.loads(
        (
            PROJECT_ROOT
            / "reports"
            / "validator_ablation_v2_001"
            / "cases.json"
        ).read_text(encoding="utf-8")
    )
    cases = case_document.get("cases")
    checks.require(
        "frozen_case_count",
        isinstance(cases, list) and len(cases) == 23,
        "23 frozen cases",
    )
    assert isinstance(cases, list)
    case_by_id = {str(case["case_id"]): case for case in cases}
    rows = list(
        csv.DictReader((bundle / "predictions.csv").open(encoding="utf-8"))
    )
    row_by_id = {str(row["case_id"]): row for row in rows}
    checks.require(
        "prediction_coverage",
        len(rows) == 23
        and len(row_by_id) == 23
        and set(row_by_id) == set(case_by_id),
        "one row per frozen case",
    )
    for case_id, row in row_by_id.items():
        case = case_by_id[case_id]
        reference_allowed = case["expected_release_status"] == "ALLOWED"
        finite = as_bool(row["tf_finite_claim_covered"])
        checks.require(
            f"prediction_flags:{case_id}",
            row["reference_release_status"] == case["expected_release_status"]
            and row["reference_unit_path"] == case["expected_unit_path"]
            and as_bool(row["valid_claim_retained"])
            == (reference_allowed and finite)
            and as_bool(row["unvalidated_positive_on_reference_nonallow"])
            == (not reference_allowed and finite)
            and as_bool(row["reference_nonallow_detected"])
            == (not reference_allowed and not finite)
            and bool(row["comparison_interpretation"]),
            case_id,
        )
    checks.require(
        "expected_scalar_domain_rejection",
        row_by_id["F03_schema_boolean_steps"]["tf_status"] == "INPUT_REJECTED"
        and all(
            row["tf_status"] == "FINITE_STATEMENT"
            for case_id, row in row_by_id.items()
            if case_id != "F03_schema_boolean_steps"
        ),
        "Boolean steps rejected; remaining 22 statements finite",
    )

    valid_rows = [
        row
        for row in rows
        if row["reference_release_status"] == "ALLOWED"
    ]
    nonallow_rows = [
        row
        for row in rows
        if row["reference_release_status"] != "ALLOWED"
    ]
    recomputed = {
        "cases": len(rows),
        "valid_reference_cases": len(valid_rows),
        "nonallow_reference_cases": len(nonallow_rows),
        "statements_emitted": sum(
            as_bool(row["tf_statement_emitted"]) for row in rows
        ),
        "finite_claims_covered": sum(
            as_bool(row["tf_finite_claim_covered"]) for row in rows
        ),
        "valid_claims_retained": sum(
            as_bool(row["valid_claim_retained"]) for row in valid_rows
        ),
        "unvalidated_positives": sum(
            as_bool(row["unvalidated_positive_on_reference_nonallow"])
            for row in nonallow_rows
        ),
        "reference_nonallow_detected": sum(
            as_bool(row["reference_nonallow_detected"])
            for row in nonallow_rows
        ),
        "status_counts": dict(Counter(row["tf_status"] for row in rows)),
    }
    metrics = json.loads(
        (bundle / "aggregate_metrics.json").read_text(encoding="utf-8")
    )
    for key, value in recomputed.items():
        checks.require(
            f"aggregate:{key}",
            metrics.get(key) == value,
            f"expected={value!r}, actual={metrics.get(key)!r}",
        )
    checks.require(
        "headline_scope_result",
        recomputed["valid_claims_retained"] == 7
        and recomputed["unvalidated_positives"] == 15
        and recomputed["reference_nonallow_detected"] == 1,
        "7/7 retained, 15/16 unvalidated, 1/16 detected",
    )

    by_family: dict[str, dict[str, int]] = {}
    for family in sorted({row["failure_family"] for row in rows}):
        selected = [row for row in rows if row["failure_family"] == family]
        by_family[family] = {
            "cases": len(selected),
            "finite_statements": sum(
                as_bool(row["tf_finite_claim_covered"]) for row in selected
            ),
            "unvalidated_positives": sum(
                as_bool(row["unvalidated_positive_on_reference_nonallow"])
                for row in selected
            ),
        }
    checks.require(
        "aggregate:by_failure_family",
        metrics.get("by_failure_family") == by_family,
        "family metrics independently recomputed",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle-dir",
        default="reports/tfprivacy_statement_baseline_v2_001",
    )
    parser.add_argument(
        "--tf-python",
        default=".venv-tfprivacy-baseline/Scripts/python.exe",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Write the verification record here instead of modifying the bundle.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = resolve_project_path(args.bundle_dir)
    interpreter = resolve_project_path(args.tf_python)
    output_path = (
        resolve_project_path(args.output_json)
        if args.output_json
        else bundle / "independent_verification.json"
    )
    checks = Checks()
    failure = ""
    try:
        checks.require(
            "tf_interpreter_exists",
            interpreter.is_file(),
            display_path(interpreter),
        )
        verify_artifact_index(bundle, checks)
        verify_requests_and_statements(bundle, checks)
        rerun_official_worker(bundle, interpreter, checks)
        verify_predictions_and_metrics(bundle, checks)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"

    passed = not failure and all(bool(row["passed"]) for row in checks.rows)
    record = {
        "schema_version": "tfprivacy_statement_baseline_verification_v2.0",
        "verification_passed": passed,
        "checks_total": len(checks.rows),
        "checks_passed": sum(bool(row["passed"]) for row in checks.rows),
        "failure": failure,
        "official_module_sha256": OFFICIAL_MODULE_SHA256,
        "verifier_file": "scripts/verify_tfprivacy_statement_baseline_v2.py",
        "verifier_sha256": sha256_file(Path(__file__).resolve()),
        "checks": checks.rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"verification_passed={passed}")
    print(f"checks={record['checks_passed']}/{record['checks_total']}")
    if failure:
        print(f"failure={failure}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
