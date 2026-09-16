"""Independently verify validator-ablation artifacts and aggregate metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def as_bool(value: object) -> bool:
    if value is True or value == "True":
        return True
    if value is False or value == "False":
        return False
    raise ValueError(f"Not a serialized Boolean: {value!r}")


def as_optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def numbers_match(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def require(self, identifier: str, condition: bool, detail: str) -> None:
        self.rows.append(
            {"id": identifier, "passed": bool(condition), "detail": detail}
        )
        if not condition:
            raise RuntimeError(f"{identifier}: {detail}")


def verify_artifact_index(
    project_root: Path,
    bundle_dir: Path,
    checks: Checks,
) -> None:
    index = json.loads((bundle_dir / "artifact_index.json").read_text(encoding="utf-8"))
    entries = index.get("files")
    checks.require(
        "artifact_index_nonempty",
        isinstance(entries, list) and bool(entries),
        "artifact index",
    )
    assert isinstance(entries, list)
    for entry in entries:
        path = project_root / entry["path"]
        checks.require(
            f"artifact_hash:{path.name}",
            path.is_file()
            and path.stat().st_size == int(entry["bytes"])
            and sha256_file(path) == entry["sha256"],
            display_path(path, project_root),
        )


def recompute_row_flags(
    row: Mapping[str, str],
    case: Mapping[str, object],
    project_root: Path,
) -> dict[str, bool]:
    expected_allowed = row["expected_release_status"] == "ALLOWED"
    predicted_allowed = row["predicted_release_status"] == "ALLOWED"
    unsafe = not expected_allowed and predicted_allowed
    binary_safe = expected_allowed == predicted_allowed
    predicted_epsilon = as_optional_float(row["predicted_epsilon"])
    predicted_delta = as_optional_float(row["predicted_delta"])
    expected_epsilon = as_optional_float(case.get("expected_epsilon"))
    expected_delta = as_optional_float(case.get("expected_delta"))
    valid_exact = (
        expected_allowed
        and predicted_allowed
        and row["predicted_unit_path"] == row["expected_unit_path"]
        and numbers_match(predicted_epsilon, expected_epsilon)
        and numbers_match(predicted_delta, expected_delta)
    )
    exact = (
        row["predicted_release_status"] == row["expected_release_status"]
        and row["predicted_unit_path"] == row["expected_unit_path"]
        and row["predicted_assurance_status"]
        == row["expected_assurance_status"]
        and (
            not expected_allowed
            or (
                numbers_match(predicted_epsilon, expected_epsilon)
                and numbers_match(predicted_delta, expected_delta)
            )
        )
    )
    report = json.loads(
        (project_root / str(case["report_path"])).read_text(encoding="utf-8")
    )
    base_epsilon = float(report["privacy"]["epsilon"])
    base_delta = float(report["privacy"]["delta"])
    wrong_same = (
        expected_allowed
        and row["expected_unit_path"] in {"CONVERT", "GROUP"}
        and predicted_allowed
        and numbers_match(predicted_epsilon, base_epsilon)
        and numbers_match(predicted_delta, base_delta)
    )
    retained = expected_allowed and predicted_allowed
    return {
        "unsafe_positive": unsafe,
        "binary_safe_correct": binary_safe,
        "valid_decision_exact": valid_exact,
        "exact_tuple": exact,
        "wrong_same_number": wrong_same,
        "valid_retained": retained,
    }


def recompute_aggregate(
    rows: Sequence[Mapping[str, str]],
) -> list[dict[str, float | int | str]]:
    output: list[dict[str, float | int | str]] = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        blocked = [
            row for row in selected if row["expected_release_status"] != "ALLOWED"
        ]
        valid = [
            row for row in selected if row["expected_release_status"] == "ALLOWED"
        ]
        wall = sorted(float(row["wall_ms"]) for row in selected)
        output.append(
            {
                "method": method,
                "cases": len(selected),
                "blocked_truth_cases": len(blocked),
                "unsafe_positives": sum(
                    as_bool(row["unsafe_positive"]) for row in blocked
                ),
                "unsafe_positive_rate": sum(
                    as_bool(row["unsafe_positive"]) for row in blocked
                )
                / len(blocked),
                "binary_safe_correct": sum(
                    as_bool(row["binary_safe_correct"]) for row in selected
                ),
                "binary_safe_accuracy": sum(
                    as_bool(row["binary_safe_correct"]) for row in selected
                )
                / len(selected),
                "valid_exact_decisions": sum(
                    as_bool(row["valid_decision_exact"]) for row in valid
                ),
                "valid_exact_decision_rate": sum(
                    as_bool(row["valid_decision_exact"]) for row in valid
                )
                / len(valid),
                "exact_tuples": sum(as_bool(row["exact_tuple"]) for row in selected),
                "exact_tuple_accuracy": sum(
                    as_bool(row["exact_tuple"]) for row in selected
                )
                / len(selected),
                "valid_truth_cases": len(valid),
                "valid_retained": sum(
                    as_bool(row["valid_retained"]) for row in valid
                ),
                "valid_retention_rate": sum(
                    as_bool(row["valid_retained"]) for row in valid
                )
                / len(valid),
                "wrong_same_number": sum(
                    as_bool(row["wrong_same_number"]) for row in selected
                ),
                "median_wall_ms": wall[len(wall) // 2],
                "max_wall_ms": max(wall),
            }
        )
    return output


def aggregates_match(
    actual: Mapping[str, str],
    expected: Mapping[str, float | int | str],
) -> bool:
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, float):
            if not math.isclose(
                float(actual_value),
                expected_value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                return False
        elif str(actual_value) != str(expected_value):
            return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle-dir",
        default="reports/validator_ablation_hardened_002",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Write the verification record here instead of modifying the bundle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    bundle_dir = Path(args.bundle_dir)
    if not bundle_dir.is_absolute():
        bundle_dir = project_root / bundle_dir
    output_path = (
        Path(args.output_json)
        if args.output_json
        else bundle_dir / "independent_verification.json"
    )
    if not output_path.is_absolute():
        output_path = project_root / output_path
    checks = Checks()
    failure = ""
    try:
        verify_artifact_index(project_root, bundle_dir, checks)
        case_document = json.loads(
            (bundle_dir / "cases.json").read_text(encoding="utf-8")
        )
        cases = case_document.get("cases")
        checks.require(
            "case_schema_and_count",
            case_document.get("schema_version") == "validator_ablation_cases_v2.0"
            and isinstance(cases, list)
            and len(cases) == 31,
            f"cases={len(cases) if isinstance(cases, list) else 'invalid'}",
        )
        assert isinstance(cases, list)
        case_by_id = {str(case["case_id"]): case for case in cases}
        checks.require(
            "case_ids_unique",
            len(case_by_id) == len(cases),
            f"unique={len(case_by_id)}",
        )
        checks.require(
            "truth_table_links_present",
            all(
                isinstance(case.get("truth_row"), str)
                and str(case["truth_row"]).startswith(("T", "R"))
                for case in cases
            ),
            "normative cases link to T01--T37; repair regressions link to R01--R08",
        )
        for case in cases:
            for field in ("mapping_path", "report_path", "audit_path"):
                path = project_root / str(case[field])
                checks.require(
                    f"{case['case_id']}:{field}_hash",
                    path.is_file()
                    and sha256_file(path) == case[f"{field}_sha256"],
                    display_path(path, project_root),
                )

        rows = list(
            csv.DictReader((bundle_dir / "predictions.csv").open(encoding="utf-8"))
        )
        coverage = Counter((row["case_id"], row["method"]) for row in rows)
        checks.require(
            "prediction_matrix_complete",
            len(rows) == len(cases) * len(METHODS)
            and set(row["method"] for row in rows) == set(METHODS)
            and set(row["case_id"] for row in rows) == set(case_by_id)
            and all(value == 1 for value in coverage.values()),
            f"rows={len(rows)}",
        )
        for row in rows:
            flags = recompute_row_flags(row, case_by_id[row["case_id"]], project_root)
            checks.require(
                f"flags:{row['case_id']}:{row['method']}",
                all(as_bool(row[key]) == value for key, value in flags.items()),
                str(flags),
            )

        full_rows = [row for row in rows if row["method"] == "FULL"]
        checks.require(
            "full_exact_and_safe",
            all(as_bool(row["exact_tuple"]) for row in full_rows)
            and not any(as_bool(row["unsafe_positive"]) for row in full_rows),
            f"rows={len(full_rows)}",
        )
        for row in full_rows:
            expected_issue = str(case_by_id[row["case_id"]]["expected_issue_code"])
            if expected_issue:
                checks.require(
                    f"full_localization:{row['case_id']}",
                    expected_issue in row["issue_codes"].split(";"),
                    row["issue_codes"],
                )

        recomputed = recompute_aggregate(rows)
        actual_rows = list(
            csv.DictReader(
                (bundle_dir / "aggregate_metrics.csv").open(encoding="utf-8")
            )
        )
        actual_by_method = {row["method"]: row for row in actual_rows}
        checks.require(
            "aggregate_method_set",
            set(actual_by_method) == set(METHODS),
            str(sorted(actual_by_method)),
        )
        for expected in recomputed:
            method = str(expected["method"])
            checks.require(
                f"aggregate:{method}",
                aggregates_match(actual_by_method[method], expected),
                str(expected),
            )
        checks.require(
            "simple_baselines_have_unsafe_positives",
            all(
                int(actual_by_method[method]["unsafe_positives"]) > 0
                for method in METHODS[:5]
            ),
            "B0--B4",
        )
        checks.require(
            "targeted_ablations_expose_expected_failures",
            int(
                actual_by_method["A1_FULL_MINUS_RUNTIME"]["unsafe_positives"]
            )
            == 3
            and int(
                actual_by_method["A2_FULL_MINUS_VACUITY"]["unsafe_positives"]
            )
            == 1,
            "runtime=3, vacuity=1",
        )
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"

    passed = not failure and all(bool(row["passed"]) for row in checks.rows)
    record = {
        "schema_version": "validator_ablation_independent_verification_v1",
        "verification_passed": passed,
        "checks_total": len(checks.rows),
        "checks_passed": sum(bool(row["passed"]) for row in checks.rows),
        "failure": failure,
        "verifier_file": "scripts/verify_validator_ablation_v2.py",
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
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
