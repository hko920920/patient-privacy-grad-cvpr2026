"""Independently recompute the fixed Audit V12 composition witnesses.

This verifier intentionally does not import the production claim validator or
the witness builder.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ABLATION_ROOT = PROJECT_ROOT / "reports" / "validator_ablation_hardened_002"
CASES_PATH = ABLATION_ROOT / "cases.json"
PREDICTIONS_PATH = ABLATION_ROOT / "predictions.csv"
WISDM_RUN = (
    PROJECT_ROOT
    / "reports"
    / "wisdm_v4_multirun_001"
    / "confirmatory_run01"
)
WISDM_VERIFIER = PROJECT_ROOT / "scripts" / "verify_wisdm_v2_gold.py"
SCENARIO = "wisdm_v2_hardened_overlap50"
PROTOCOL_NAME = (
    "AAAI27_AUDIT_V12_COMPOSITION_RISK_PROTOCOL_V1_2026-07-25.md"
)
PACKAGED_PROTOCOL = PROJECT_ROOT / "paper_notes" / PROTOCOL_NAME


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_bound_path(relative: str) -> Path:
    """Resolve a bound source in either workspace or anonymous-package layout."""
    candidate = (PROJECT_ROOT / relative).resolve()
    if candidate.is_file():
        return candidate
    if Path(relative).name == PROTOCOL_NAME and PACKAGED_PROTOCOL.is_file():
        return PACKAGED_PROTOCOL.resolve()
    return candidate


def recursive_diff(left: Any, right: Any, prefix: str = "") -> list[str]:
    if type(left) is not type(right):
        return [prefix or "<root>"]
    if isinstance(left, Mapping):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(recursive_diff(left[key], right[key], child))
        return paths
    if isinstance(left, list):
        if len(left) != len(right):
            return [prefix or "<root>"]
        paths: list[str] = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            paths.extend(
                recursive_diff(left_item, right_item, f"{prefix}[{index}]")
            )
        return paths
    return [] if left == right else [prefix or "<root>"]


def case_index() -> Dict[str, Dict[str, Any]]:
    return {
        row["case_id"]: row
        for row in load_json(CASES_PATH)["cases"]
    }


def prediction_index() -> Dict[tuple[str, str], Dict[str, str]]:
    with PREDICTIONS_PATH.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["case_id"], row["method"]): row for row in rows}


def case_report(case: Mapping[str, Any]) -> Dict[str, Any]:
    return load_json(PROJECT_ROOT / case["report_path"])


def run_w5() -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="audit_v12_verify_w5_",
        dir=PROJECT_ROOT / "reports",
    ) as temp_name:
        temp_root = Path(temp_name)
        copied = temp_root / "run"
        shutil.copytree(WISDM_RUN, copied)
        baseline_output = temp_root / "baseline.json"
        baseline = subprocess.run(
            [
                sys.executable,
                str(WISDM_VERIFIER),
                "--evidence-dir",
                str(copied),
                "--skip-input-rescan",
                "--output-json",
                str(baseline_output),
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        cert_dir = copied / "certificates"
        window_path = cert_dir / f"certificate_{SCENARIO}_window.json"
        event_path = cert_dir / f"certificate_{SCENARIO}_event.json"
        window = load_json(window_path)
        event = load_json(event_path)
        source_event = event["supported_statement"]
        injected_window = window["supported_statement"]
        event["supported_statement"] = injected_window
        event_path.write_text(
            json.dumps(event, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tampered_output = temp_root / "tampered.json"
        tampered = subprocess.run(
            [
                sys.executable,
                str(WISDM_VERIFIER),
                "--evidence-dir",
                str(copied),
                "--skip-input-rescan",
                "--output-json",
                str(tampered_output),
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        baseline_record = load_json(baseline_output)
        tampered_record = load_json(tampered_output)
        return {
            "passed": (
                baseline.returncode == 0
                and baseline_record["passed"] is True
                and tampered.returncode != 0
                and tampered_record["passed"] is False
                and tampered_record["checks"]["event_certificate"]["passed"]
                is False
            ),
            "baseline_returncode": baseline.returncode,
            "tampered_returncode": tampered.returncode,
            "tampered_event_check_passed": tampered_record["checks"][
                "event_certificate"
            ]["passed"],
            "source_event_statement_sha256": hashlib.sha256(
                source_event.encode("utf-8")
            ).hexdigest(),
            "injected_window_statement_sha256": hashlib.sha256(
                injected_window.encode("utf-8")
            ).hexdigest(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        default="reports/audit_composition_witnesses_v12_003/witnesses.json",
    )
    parser.add_argument(
        "--output-json",
        default=(
            "reports/audit_composition_witnesses_v12_003/"
            "independent_verification.json"
        ),
    )
    args = parser.parse_args()
    report_path = (PROJECT_ROOT / args.report).resolve()
    output_path = (PROJECT_ROOT / args.output_json).resolve()
    report = load_json(report_path)
    checks: Dict[str, Dict[str, Any]] = {}

    def check(name: str, condition: bool, observed: Any) -> None:
        checks[name] = {"passed": bool(condition), "observed": observed}

    check(
        "schema",
        report.get("schema_version") == "audit_composition_witnesses_v12.1",
        report.get("schema_version"),
    )
    check(
        "protocol_binding",
        sha256_file(resolve_bound_path(report["protocol"]["path"]))
        == report["protocol"]["sha256"],
        report["protocol"],
    )

    source_results = []
    for source in report["sources"]:
        path = resolve_bound_path(source["path"])
        actual = {
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        source_results.append(
            actual["exists"]
            and actual["bytes"] == source["bytes"]
            and actual["sha256"] == source["sha256"]
        )
    check(
        "all_source_bindings",
        bool(source_results) and all(source_results),
        {"passed": sum(source_results), "total": len(source_results)},
    )

    certificate_dir = WISDM_RUN / "certificates"
    certs = {
        claim: load_json(
            certificate_dir / f"certificate_{SCENARIO}_{claim}.json"
        )
        for claim in ("window", "event", "owner")
    }
    selected = {row["selected_mapping_sha256"] for row in certs.values()}
    pipeline = {row["pipeline_sha256"] for row in certs.values()}
    rows = []
    for claim in ("window", "event", "owner"):
        row = certs[claim]
        rows.append(
            {
                "query": claim,
                "unit_path": row["unit_path"],
                "release_status": row["release_status"],
                "stability_bound": row.get("stability_bound", 600),
                "epsilon": row.get("epsilon"),
                "delta": row.get("delta"),
                "positive_wording": bool(row["supported_statement"]),
            }
        )
    tuple_count = len(
        {
            canonical_sha256(
                {
                    key: row[key]
                    for key in (
                        "unit_path",
                        "release_status",
                        "stability_bound",
                        "epsilon",
                        "delta",
                        "positive_wording",
                    )
                }
            )
            for row in rows
        }
    )
    recomputed_query_pass = (
        len(selected) == 1
        and len(pipeline) == 1
        and tuple_count == 3
        and [row["release_status"] for row in rows]
        == ["ALLOWED", "ALLOWED", "BLOCKED_VACUOUS"]
        and [row["unit_path"] for row in rows]
        == ["DIRECT", "CONVERT", "GROUP"]
    )
    check(
        "query_separation",
        report["query_separation"]["passed"] is True
        and recomputed_query_pass
        and report["query_separation"]["queries"] == rows
        and report["query_separation"]["distinct_result_tuples"] == tuple_count
        and report["query_separation"][
            "query_oblivious_exact_and_responsive_possible"
        ]
        is False,
        {
            "selected_count": len(selected),
            "pipeline_count": len(pipeline),
            "tuple_count": tuple_count,
        },
    )

    cases = case_index()
    predictions = prediction_index()
    reported_witnesses = {row["id"]: row for row in report["witnesses"]}

    p01 = cases["P01_valid_window"]
    f02 = cases["F02_mapping_bytes_changed"]
    w1_conditions = {
        "same_report": p01["report_path_sha256"] == f02["report_path_sha256"],
        "same_audit_row": p01["audit_path_sha256"] == f02["audit_path_sha256"],
        "same_query": p01["claim_unit"] == f02["claim_unit"] == "window",
        "different_actual_mapping": (
            p01["mapping_path_sha256"] != f02["mapping_path_sha256"]
        ),
        "required_outcomes_differ": (
            predictions[("P01_valid_window", "FULL")][
                "predicted_release_status"
            ]
            == "ALLOWED"
            and predictions[("F02_mapping_bytes_changed", "FULL")][
                "predicted_release_status"
            ]
            == "BLOCKED_INVALID"
        ),
    }
    check(
        "W1",
        all(w1_conditions.values())
        and reported_witnesses["W1"]["passed"] is True
        and reported_witnesses["W1"]["checks"] == w1_conditions,
        w1_conditions,
    )

    p02 = cases["P02_valid_event_factor2"]
    f16 = cases["F16_unverified_stability"]
    p02_report = case_report(p02)
    f16_report = case_report(f16)
    w2_diff = recursive_diff(p02_report, f16_report)
    p02_projection = copy.deepcopy(p02_report)
    f16_projection = copy.deepcopy(f16_report)
    p02_projection["claim_contracts"]["event"].pop("stability_status")
    f16_projection["claim_contracts"]["event"].pop("stability_status")
    w2_conditions = {
        "same_mapping": p02["mapping_path_sha256"] == f16["mapping_path_sha256"],
        "same_audit_row": p02["audit_path_sha256"] == f16["audit_path_sha256"],
        "same_event_query": p02["claim_unit"] == f16["claim_unit"] == "event",
        "same_base_privacy": p02_report["privacy"] == f16_report["privacy"],
        "only_stability_status_differs": w2_diff
        == ["claim_contracts.event.stability_status"],
        "projection_equal_without_stability_status": (
            canonical_sha256(p02_projection)
            == canonical_sha256(f16_projection)
        ),
        "required_outcomes_differ": (
            predictions[("P02_valid_event_factor2", "FULL")][
                "predicted_release_status"
            ]
            == "ALLOWED"
            and predictions[("F16_unverified_stability", "FULL")][
                "predicted_release_status"
            ]
            == "BLOCKED_UNVERIFIED"
        ),
    }
    check(
        "W2",
        all(w2_conditions.values())
        and reported_witnesses["W2"]["passed"] is True
        and reported_witnesses["W2"]["checks"] == w2_conditions
        and reported_witnesses["W2"]["differing_paths"] == w2_diff,
        w2_conditions,
    )

    f05 = cases["F05_runtime_noise_mismatch"]
    p01_report = case_report(p01)
    f05_report = case_report(f05)
    w3_diff = recursive_diff(p01_report, f05_report)
    p01_projection = copy.deepcopy(p01_report)
    f05_projection = copy.deepcopy(f05_report)
    p01_projection.pop("runtime_trace")
    f05_projection.pop("runtime_trace")
    w3_conditions = {
        "same_mapping": p01["mapping_path_sha256"] == f05["mapping_path_sha256"],
        "same_audit_row": p01["audit_path_sha256"] == f05["audit_path_sha256"],
        "same_window_query": p01["claim_unit"] == f05["claim_unit"] == "window",
        "same_base_privacy": p01_report["privacy"] == f05_report["privacy"],
        "runtime_only_diff": w3_diff
        == ["runtime_trace.noise_multiplier", "runtime_trace.trace_hash"],
        "projection_equal_without_runtime": (
            canonical_sha256(p01_projection)
            == canonical_sha256(f05_projection)
        ),
        "required_outcomes_differ": (
            predictions[("P01_valid_window", "FULL")][
                "predicted_release_status"
            ]
            == "ALLOWED"
            and predictions[("F05_runtime_noise_mismatch", "FULL")][
                "predicted_release_status"
            ]
            == "BLOCKED_INVALID"
        ),
    }
    check(
        "W3",
        all(w3_conditions.values())
        and reported_witnesses["W3"]["passed"] is True
        and reported_witnesses["W3"]["checks"] == w3_conditions
        and reported_witnesses["W3"]["differing_paths"] == w3_diff,
        w3_conditions,
    )

    a2 = predictions[("F01_vacuous_owner_group", "A2_FULL_MINUS_VACUITY")]
    full = predictions[("F01_vacuous_owner_group", "FULL")]
    w4_conditions = {
        "a2_allows": a2["predicted_release_status"] == "ALLOWED",
        "a2_unsafe_positive": a2["unsafe_positive"] == "True",
        "full_blocks_vacuous": (
            full["predicted_release_status"] == "BLOCKED_VACUOUS"
        ),
        "full_not_unsafe_positive": full["unsafe_positive"] == "False",
    }
    check(
        "W4",
        all(w4_conditions.values())
        and reported_witnesses["W4"]["passed"] is True
        and reported_witnesses["W4"]["checks"] == w4_conditions,
        w4_conditions,
    )

    recomputed_w5 = run_w5()
    w5 = reported_witnesses["W5"]
    check(
        "W5",
        recomputed_w5["passed"]
        and w5["passed"] is True
        and w5["baseline_verifier_returncode"]
        == recomputed_w5["baseline_returncode"]
        and w5["tampered_verifier_returncode"]
        == recomputed_w5["tampered_returncode"]
        and w5["tampered_event_check_passed"]
        == recomputed_w5["tampered_event_check_passed"]
        and w5["source_event_statement_sha256"]
        == recomputed_w5["source_event_statement_sha256"]
        and w5["injected_window_statement_sha256"]
        == recomputed_w5["injected_window_statement_sha256"],
        recomputed_w5,
    )

    witness_pass_count = sum(
        1 for name in ("W1", "W2", "W3", "W4", "W5") if checks[name]["passed"]
    )
    summary = report["summary"]
    check(
        "summary",
        summary["query_separation_passed"] is True
        and summary["witnesses_passed"] == witness_pass_count == 5
        and summary["witnesses_total"] == 5
        and summary["all_passed"] is True,
        {
            "query": checks["query_separation"]["passed"],
            "witnesses": witness_pass_count,
        },
    )

    passed = all(item["passed"] for item in checks.values())
    output = {
        "schema_version": "audit_composition_witnesses_v12_verification.1",
        "passed": passed,
        "checks_passed": sum(item["passed"] for item in checks.values()),
        "checks_total": len(checks),
        "report_sha256": sha256_file(report_path),
        "verifier_sha256": sha256_file(Path(__file__).resolve()),
        "imports_production_validator": False,
        "imports_witness_builder": False,
        "checks": checks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"passed={passed}")
    print(f"checks={output['checks_passed']}/{output['checks_total']}")
    print(f"Wrote {output_path}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
