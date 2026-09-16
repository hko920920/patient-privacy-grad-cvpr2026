#!/usr/bin/env python3
"""Evaluate TensorFlow Privacy's statement generator on frozen v2 cases.

This is a scope-comparison baseline, not a claim that TensorFlow Privacy is
incorrect. The official function is designed to produce a conditional privacy
statement from caller-supplied scalar parameters. It is not designed to reopen
an arbitrary run's mapping, runtime trace, preprocessing, or schedule.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence


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


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_frozen_cases(cases_path: Path) -> list[dict[str, Any]]:
    document = json.loads(cases_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != "validator_ablation_cases_v2.0":
        raise ValueError("Unsupported frozen case schema")
    cases = document.get("cases")
    if not isinstance(cases, list) or len(cases) != 23:
        raise ValueError("Expected the frozen 23-case validator matrix")
    case_ids = [str(case.get("case_id", "")) for case in cases]
    if len(case_ids) != len(set(case_ids)) or any(not value for value in case_ids):
        raise ValueError("Frozen case IDs must be non-empty and unique")
    for case in cases:
        for field in ("mapping_path", "report_path", "audit_path"):
            path = resolve_project_path(str(case[field]))
            if (
                not path.is_file()
                or sha256_file(path) != case[f"{field}_sha256"]
            ):
                raise RuntimeError(
                    f"Frozen input hash mismatch for {case['case_id']}:{field}"
                )
    return cases


def positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer, found {value!r}")
    return value


def finite_float(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric, found Boolean")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, found {value!r}")
    return result


def scalar_equivalent_request(
    case: Mapping[str, Any],
    report: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    mechanism = report.get("mechanism")
    privacy = report.get("privacy")
    contracts = report.get("claim_contracts")
    if (
        not isinstance(mechanism, dict)
        or not isinstance(privacy, dict)
        or not isinstance(contracts, dict)
    ):
        raise ValueError("mechanism/privacy/claim_contracts must be objects")

    sample_rate = finite_float(mechanism.get("sample_rate"), "sample_rate")
    if not 0 < sample_rate <= 1:
        raise ValueError(f"sample_rate outside (0,1]: {sample_rate}")
    fraction = Fraction(str(sample_rate)).limit_denominator(1_000_000)
    if not math.isclose(
        float(fraction),
        sample_rate,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("sample_rate has no exact bounded rational adapter")
    batch_size = fraction.numerator
    number_of_examples = fraction.denominator
    steps = positive_int(mechanism.get("steps"), "steps")
    noise = finite_float(mechanism.get("noise_multiplier"), "noise_multiplier")
    delta = finite_float(privacy.get("delta"), "delta")
    if noise < 0 or not 0 < delta < 1:
        raise ValueError("noise_multiplier/delta outside official function domain")

    claim_unit = str(case["claim_unit"])
    accounting_unit = str(mechanism.get("accounting_unit", ""))
    maximum: int | None = None
    selected_output = "example"
    expected_path = "DIRECT"
    if claim_unit != accounting_unit:
        contract = contracts.get(claim_unit)
        if not isinstance(contract, dict):
            raise ValueError(f"Missing claim contract for {claim_unit}")
        maximum = positive_int(
            contract.get("stability_bound"),
            f"{claim_unit}.stability_bound",
        )
        selected_output = "user"
        expected_path = "CONVERT" if claim_unit == "event" else "GROUP"

    request = {
        "case_id": str(case["case_id"]),
        "number_of_examples": number_of_examples,
        "batch_size": batch_size,
        "num_epochs": steps * batch_size / number_of_examples,
        "noise_multiplier": noise,
        "delta": delta,
        # The v2 mechanism uses per-example clipping, not microbatch clipping.
        "used_microbatching": False,
        "max_examples_per_user": maximum,
    }
    interpretation = {
        "claim_unit": claim_unit,
        "accounting_unit_declared": accounting_unit,
        "selected_output": selected_output,
        "selected_path": expected_path,
        "selected_sampling": (
            "poisson"
            if mechanism.get("sampler_law") == "poisson"
            and mechanism.get("accountant_sampler_law") == "poisson"
            else "ordered_conservative"
        ),
        "group_bound_source": (
            "none_identity"
            if maximum is None
            else "self_reported_claim_contract_stability_bound"
        ),
    }
    return request, interpretation


def run_worker(
    interpreter: Path,
    worker: Path,
    request_path: Path,
    result_path: Path,
) -> None:
    completed = subprocess.run(
        [
            str(interpreter),
            str(worker),
            "--input",
            str(request_path),
            "--output",
            str(result_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "TensorFlow Privacy worker failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def selected_epsilon(
    result: Mapping[str, Any],
    interpretation: Mapping[str, Any],
) -> float | str:
    prefix = str(interpretation["selected_output"])
    sampling = str(interpretation["selected_sampling"])
    suffix = "poisson" if sampling == "poisson" else "ordered"
    value = result[f"{prefix}_{suffix}_epsilon_printed"]
    if value is None:
        raise RuntimeError("Official statement omitted the selected epsilon")
    return value


def prediction_rows(
    cases: Sequence[Mapping[str, Any]],
    interpretations: Mapping[str, Mapping[str, Any]],
    worker_results: Mapping[str, Mapping[str, Any]],
    rejected: Mapping[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in cases:
        case_id = str(case["case_id"])
        reference_allowed = case["expected_release_status"] == "ALLOWED"
        if case_id in rejected:
            status = "INPUT_REJECTED"
            statement_emitted = False
            finite_claim = False
            epsilon: float | str = ""
            statement_sha = ""
            selected_path = "UNDERSPECIFIED"
            selected_sampling = ""
            group_bound_source = ""
            issue = rejected[case_id]
            comparison_interpretation = (
                "SCALAR_DOMAIN_REJECTION_DETECTS_THIS_CASE"
            )
        else:
            interpretation = interpretations[case_id]
            result = worker_results[case_id]
            epsilon = selected_epsilon(result, interpretation)
            statement_emitted = True
            finite_claim = epsilon != "inf"
            status = "FINITE_STATEMENT" if finite_claim else "INFINITE_STATEMENT"
            statement_sha = str(result["statement_sha256"])
            selected_path = str(interpretation["selected_path"])
            selected_sampling = str(interpretation["selected_sampling"])
            group_bound_source = str(interpretation["group_bound_source"])
            issue = ""
            family = str(case["failure_family"])
            if family == "positive":
                comparison_interpretation = "VALID_CONDITIONAL_STATEMENT"
            elif family == "conversion_vacuity":
                comparison_interpretation = (
                    "FRESH_TARGET_DELTA_REACCOUNTING_NOT_FIXED_BASE_CONVERSION"
                )
            elif case_id == "F06_accountant_epsilon_tamper":
                comparison_interpretation = (
                    "FRESH_RECOMPUTATION_CAN_REPLACE_TAMPERED_REPORT_IF_BOUND"
                )
            elif family == "mechanism":
                comparison_interpretation = (
                    "ORDERED_CONSERVATIVE_ALTERNATIVE_NOT_BUNDLE_VALIDATION"
                )
            else:
                comparison_interpretation = (
                    "SCALAR_STATEMENT_DOES_NOT_INSPECT_"
                    + family.upper()
                )
        unvalidated_positive = not reference_allowed and finite_claim
        rows.append(
            {
                "case_id": case_id,
                "failure_family": case["failure_family"],
                "reference_release_status": case["expected_release_status"],
                "reference_unit_path": case["expected_unit_path"],
                "tf_status": status,
                "tf_statement_emitted": statement_emitted,
                "tf_finite_claim_covered": finite_claim,
                "tf_selected_path": selected_path,
                "tf_selected_sampling": selected_sampling,
                "tf_group_bound_source": group_bound_source,
                "tf_selected_epsilon_printed": epsilon,
                "tf_delta": (
                    ""
                    if case_id in rejected
                    else worker_results[case_id]["request"]["delta"]
                ),
                "statement_sha256": statement_sha,
                "valid_claim_retained": reference_allowed and finite_claim,
                "unvalidated_positive_on_reference_nonallow": unvalidated_positive,
                "reference_nonallow_detected": (
                    not reference_allowed and not finite_claim
                ),
                "adapter_issue": issue,
                "comparison_interpretation": comparison_interpretation,
            }
        )
    return rows


def aggregate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    valid = [row for row in rows if row["reference_release_status"] == "ALLOWED"]
    nonallow = [
        row for row in rows if row["reference_release_status"] != "ALLOWED"
    ]
    family_counts: dict[str, dict[str, int]] = {}
    for family in sorted({str(row["failure_family"]) for row in rows}):
        selected = [row for row in rows if row["failure_family"] == family]
        family_counts[family] = {
            "cases": len(selected),
            "finite_statements": sum(
                bool(row["tf_finite_claim_covered"]) for row in selected
            ),
            "unvalidated_positives": sum(
                bool(row["unvalidated_positive_on_reference_nonallow"])
                for row in selected
            ),
        }
    return {
        "cases": len(rows),
        "valid_reference_cases": len(valid),
        "nonallow_reference_cases": len(nonallow),
        "statements_emitted": sum(
            bool(row["tf_statement_emitted"]) for row in rows
        ),
        "finite_claims_covered": sum(
            bool(row["tf_finite_claim_covered"]) for row in rows
        ),
        "valid_claims_retained": sum(
            bool(row["valid_claim_retained"]) for row in valid
        ),
        "valid_claim_retention_rate": sum(
            bool(row["valid_claim_retained"]) for row in valid
        )
        / len(valid),
        "unvalidated_positives": sum(
            bool(row["unvalidated_positive_on_reference_nonallow"])
            for row in nonallow
        ),
        "unvalidated_positive_rate": sum(
            bool(row["unvalidated_positive_on_reference_nonallow"])
            for row in nonallow
        )
        / len(nonallow),
        "reference_nonallow_detected": sum(
            bool(row["reference_nonallow_detected"]) for row in nonallow
        ),
        "status_counts": dict(Counter(str(row["tf_status"]) for row in rows)),
        "by_failure_family": family_counts,
    }


def summary_markdown(metrics: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# TensorFlow Privacy Statement Baseline v2",
            "",
            "This experiment executes the official TensorFlow Privacy 0.9.0",
            "`compute_dp_sgd_privacy_statement` implementation from its wheel.",
            "It compares scope, not software correctness: the official function",
            "generates a conditional statement from caller-supplied scalars, while",
            "the v2 validator decides whether an external run-evidence bundle",
            "licenses a requested privacy-unit claim.",
            "",
            f"- Cases: {metrics['cases']}",
            f"- Valid reference cases retained with a finite statement: "
            f"{metrics['valid_claims_retained']}/{metrics['valid_reference_cases']}",
            f"- Non-allow reference cases that still receive a finite conditional "
            f"statement: {metrics['unvalidated_positives']}/"
            f"{metrics['nonallow_reference_cases']}",
            f"- Non-allow cases detected by scalar input rejection or an infinite "
            f"bound: {metrics['reference_nonallow_detected']}/"
            f"{metrics['nonallow_reference_cases']}",
            "",
            "An `unvalidated positive` is not labeled a TensorFlow Privacy bug.",
            "It means that substituting a scalar statement generator for the",
            "evidence validator would leave the bundle inconsistency undetected.",
            "The statement remains conditional on its supplied parameters and",
            "group bound. Group outputs are not compared numerically to the v2",
            "fixed-base conversion because the official function solves for a",
            "smaller example-level delta to target the requested user-level delta.",
            "In particular, the vacuity case is a fresh target-delta accounting",
            "alternative, not the same fixed-base conversion rejected by v2; it",
            "would require its own registered and execution-bound adapter before",
            "the audit validator could accept it.",
            "",
            "Official module SHA-256:",
            f"`{OFFICIAL_MODULE_SHA256}`.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default="reports/validator_ablation_v2_001/cases.json",
    )
    parser.add_argument(
        "--tf-python",
        default=".venv-tfprivacy-baseline/Scripts/python.exe",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/tfprivacy_statement_baseline_v2_001",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases_path = resolve_project_path(args.cases)
    interpreter = resolve_project_path(args.tf_python)
    output_dir = resolve_project_path(args.output_dir)
    worker = PROJECT_ROOT / "scripts" / "tfprivacy_statement_worker.py"
    if not interpreter.is_file():
        raise FileNotFoundError(
            f"Baseline interpreter missing: {interpreter}; run "
            "scripts/setup_tfprivacy_baseline.ps1"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_frozen_cases(cases_path)

    requests: list[dict[str, Any]] = []
    interpretations: dict[str, dict[str, Any]] = {}
    rejected: dict[str, str] = {}
    for case in cases:
        case_id = str(case["case_id"])
        report_path = resolve_project_path(str(case["report_path"]))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        try:
            request, interpretation = scalar_equivalent_request(case, report)
        except (KeyError, TypeError, ValueError) as exc:
            rejected[case_id] = f"{type(exc).__name__}: {exc}"
            continue
        requests.append(request)
        interpretations[case_id] = interpretation

    request_path = output_dir / "requests.json"
    worker_output_path = output_dir / "official_statements.json"
    write_json(
        request_path,
        {
            "schema_version": "tfprivacy_statement_requests_v2.0",
            "expected_module_sha256": OFFICIAL_MODULE_SHA256,
            "cases_file": cases_path.relative_to(PROJECT_ROOT).as_posix(),
            "cases_file_sha256": sha256_file(cases_path),
            "requests": requests,
        },
    )
    run_worker(interpreter, worker, request_path, worker_output_path)
    worker_document = json.loads(worker_output_path.read_text(encoding="utf-8"))
    if (
        worker_document.get("schema_version")
        != "tfprivacy_statement_worker_output_v2.0"
        or worker_document.get("module_sha256") != OFFICIAL_MODULE_SHA256
    ):
        raise RuntimeError("Official worker output failed version/hash binding")
    worker_results = {
        str(result["case_id"]): result
        for result in worker_document["results"]
    }
    if set(worker_results) != {str(request["case_id"]) for request in requests}:
        raise RuntimeError("Worker result/request case coverage mismatch")

    rows = prediction_rows(
        cases,
        interpretations,
        worker_results,
        rejected,
    )
    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "aggregate_metrics.json"
    summary_path = output_dir / "summary.md"
    write_csv(predictions_path, rows)
    metrics = aggregate(rows)
    write_json(metrics_path, metrics)
    summary_path.write_text(summary_markdown(metrics), encoding="utf-8")

    source_paths = [
        Path(__file__).resolve(),
        worker,
        PROJECT_ROOT / "scripts" / "setup_tfprivacy_baseline.ps1",
        PROJECT_ROOT / "requirements-tfprivacy-baseline.txt",
        PROJECT_ROOT / "scripts" / "verify_tfprivacy_statement_baseline_v2.py",
        cases_path,
        request_path,
        worker_output_path,
        predictions_path,
        metrics_path,
        summary_path,
    ]
    write_json(
        output_dir / "artifact_index.json",
        {
            "schema_version": "tfprivacy_statement_baseline_artifact_v2.0",
            "official_module_sha256": OFFICIAL_MODULE_SHA256,
            "files": [
                {
                    "path": path.relative_to(PROJECT_ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in source_paths
            ],
        },
    )
    print(summary_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
