"""Build the prespecified Audit V12 composition-necessity witnesses.

This builder consumes only frozen V10/V11 evidence.  It does not import the
production claim validator and does not alter the 31-case oracle.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_PROTOCOL = (
    PROJECT_ROOT.parent
    / "DP-SGD-algorithm"
    / "paper_draft"
    / "AAAI27_AUDIT_V12_COMPOSITION_RISK_PROTOCOL_V1_2026-07-25.md"
)
PACKAGED_PROTOCOL = (
    PROJECT_ROOT
    / "paper_notes"
    / "AAAI27_AUDIT_V12_COMPOSITION_RISK_PROTOCOL_V1_2026-07-25.md"
)
PROTOCOL = (
    PACKAGED_PROTOCOL if PACKAGED_PROTOCOL.is_file() else WORKSPACE_PROTOCOL
)
ABLATION_ROOT = PROJECT_ROOT / "reports" / "validator_ablation_hardened_002"
CASES_PATH = ABLATION_ROOT / "cases.json"
PREDICTIONS_PATH = ABLATION_ROOT / "predictions.csv"
ABLATION_VERIFICATION_PATH = ABLATION_ROOT / "independent_verification.json"
WISDM_RUN = (
    PROJECT_ROOT
    / "reports"
    / "wisdm_v4_multirun_001"
    / "confirmatory_run01"
)
WISDM_VERIFIER = PROJECT_ROOT / "scripts" / "verify_wisdm_v2_gold.py"
INDEPENDENT_VERIFIER = (
    PROJECT_ROOT / "scripts" / "verify_audit_composition_witnesses_v12.py"
)
SCENARIO = "wisdm_v2_hardened_overlap50"


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


def relative(path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), PROJECT_ROOT.resolve())).as_posix()


def recursive_diff(left: Any, right: Any, prefix: str = "") -> list[str]:
    if type(left) is not type(right):
        return [prefix or "<root>"]
    if isinstance(left, Mapping):
        paths: list[str] = []
        keys = sorted(set(left) | set(right))
        for key in keys:
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
            child = f"{prefix}[{index}]"
            paths.extend(recursive_diff(left_item, right_item, child))
        return paths
    return [] if left == right else [prefix or "<root>"]


def nested_delete(value: Mapping[str, Any], path: Iterable[str]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(value))
    cursor: Dict[str, Any] = result
    parts = list(path)
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor.pop(parts[-1], None)
    return result


def prediction_index() -> Dict[tuple[str, str], Dict[str, str]]:
    with PREDICTIONS_PATH.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["case_id"], row["method"]): row for row in rows}


def case_index() -> Dict[str, Dict[str, Any]]:
    payload = load_json(CASES_PATH)
    return {row["case_id"]: row for row in payload["cases"]}


def load_case_report(case: Mapping[str, Any]) -> Dict[str, Any]:
    return load_json(PROJECT_ROOT / case["report_path"])


def exact_tuple(row: Mapping[str, str]) -> list[str]:
    return [
        row["predicted_unit_path"],
        row["predicted_release_status"],
        row["predicted_assurance_status"],
        row["predicted_epsilon"],
        row["predicted_delta"],
    ]


def run_wording_tamper() -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="audit_v12_w5_",
        dir=PROJECT_ROOT / "reports",
    ) as temp_name:
        temp_root = Path(temp_name)
        copied_run = temp_root / "run"
        shutil.copytree(WISDM_RUN, copied_run)

        baseline_output = temp_root / "baseline.json"
        baseline = subprocess.run(
            [
                sys.executable,
                str(WISDM_VERIFIER),
                "--evidence-dir",
                str(copied_run),
                "--skip-input-rescan",
                "--output-json",
                str(baseline_output),
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        certificate_dir = copied_run / "certificates"
        window_path = certificate_dir / f"certificate_{SCENARIO}_window.json"
        event_path = certificate_dir / f"certificate_{SCENARIO}_event.json"
        window = load_json(window_path)
        event = load_json(event_path)
        source_event_statement = event["supported_statement"]
        injected_window_statement = window["supported_statement"]
        event["supported_statement"] = injected_window_statement
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
                str(copied_run),
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
        event_check = tampered_record["checks"]["event_certificate"]
        passed = (
            baseline.returncode == 0
            and baseline_record["passed"] is True
            and tampered.returncode != 0
            and tampered_record["passed"] is False
            and event_check["passed"] is False
        )
        return {
            "id": "W5",
            "name": "exact_wording_surface",
            "passed": passed,
            "baseline_verifier_returncode": baseline.returncode,
            "baseline_checks": baseline_record["check_count"],
            "tampered_verifier_returncode": tampered.returncode,
            "tampered_event_check_passed": event_check["passed"],
            "mutation": "event supported_statement <- window supported_statement",
            "source_event_statement_sha256": hashlib.sha256(
                source_event_statement.encode("utf-8")
            ).hexdigest(),
            "injected_window_statement_sha256": hashlib.sha256(
                injected_window_statement.encode("utf-8")
            ).hexdigest(),
        }


def load_wisdm_certificates() -> Dict[str, Dict[str, Any]]:
    certificate_dir = WISDM_RUN / "certificates"
    return {
        claim: load_json(
            certificate_dir / f"certificate_{SCENARIO}_{claim}.json"
        )
        for claim in ("window", "event", "owner")
    }


def build_report() -> Dict[str, Any]:
    cases = case_index()
    predictions = prediction_index()
    certificates = load_wisdm_certificates()

    source_paths = [
        PROTOCOL,
        CASES_PATH,
        PREDICTIONS_PATH,
        ABLATION_VERIFICATION_PATH,
        WISDM_VERIFIER,
        WISDM_RUN
        / "certificates"
        / f"certificate_{SCENARIO}_window.json",
        WISDM_RUN
        / "certificates"
        / f"certificate_{SCENARIO}_event.json",
        WISDM_RUN
        / "certificates"
        / f"certificate_{SCENARIO}_owner.json",
        Path(__file__).resolve(),
        INDEPENDENT_VERIFIER,
    ]
    sources = [
        {
            "path": relative(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in source_paths
    ]

    common_selected = {
        certificates[claim]["selected_mapping_sha256"]
        for claim in certificates
    }
    common_pipeline = {
        certificates[claim]["pipeline_sha256"] for claim in certificates
    }
    query_rows = []
    for claim in ("window", "event", "owner"):
        certificate = certificates[claim]
        query_rows.append(
            {
                "query": claim,
                "unit_path": certificate["unit_path"],
                "release_status": certificate["release_status"],
                "stability_bound": certificate.get("stability_bound", 600),
                "epsilon": certificate.get("epsilon"),
                "delta": certificate.get("delta"),
                "positive_wording": bool(certificate["supported_statement"]),
            }
        )
    query_tuples = {
        canonical_sha256(
            {
                "unit_path": row["unit_path"],
                "release_status": row["release_status"],
                "stability_bound": row["stability_bound"],
                "epsilon": row["epsilon"],
                "delta": row["delta"],
                "positive_wording": row["positive_wording"],
            }
        )
        for row in query_rows
    }
    query_separation = {
        "passed": (
            len(common_selected) == 1
            and len(common_pipeline) == 1
            and len(query_tuples) == 3
            and [row["release_status"] for row in query_rows]
            == ["ALLOWED", "ALLOWED", "BLOCKED_VACUOUS"]
            and [row["unit_path"] for row in query_rows]
            == ["DIRECT", "CONVERT", "GROUP"]
        ),
        "selected_mapping_sha256": next(iter(common_selected)),
        "pipeline_sha256": next(iter(common_pipeline)),
        "distinct_result_tuples": len(query_tuples),
        "query_oblivious_exact_and_responsive_possible": False,
        "queries": query_rows,
    }

    p01 = cases["P01_valid_window"]
    f02 = cases["F02_mapping_bytes_changed"]
    p01_full = predictions[("P01_valid_window", "FULL")]
    f02_full = predictions[("F02_mapping_bytes_changed", "FULL")]
    w1_checks = {
        "same_report": p01["report_path_sha256"] == f02["report_path_sha256"],
        "same_audit_row": p01["audit_path_sha256"] == f02["audit_path_sha256"],
        "same_query": p01["claim_unit"] == f02["claim_unit"] == "window",
        "different_actual_mapping": (
            p01["mapping_path_sha256"] != f02["mapping_path_sha256"]
        ),
        "required_outcomes_differ": (
            p01_full["predicted_release_status"] == "ALLOWED"
            and f02_full["predicted_release_status"] == "BLOCKED_INVALID"
        ),
    }
    w1 = {
        "id": "W1",
        "name": "actual_evidence_reopening",
        "passed": all(w1_checks.values()),
        "cases": ["P01_valid_window", "F02_mapping_bytes_changed"],
        "projection": "report + audit row + query; actual mapping hidden",
        "checks": w1_checks,
        "full_outputs": [exact_tuple(p01_full), exact_tuple(f02_full)],
    }

    p02 = cases["P02_valid_event_factor2"]
    f16 = cases["F16_unverified_stability"]
    p02_report = load_case_report(p02)
    f16_report = load_case_report(f16)
    w2_diff = recursive_diff(p02_report, f16_report)
    p02_projection = nested_delete(
        p02_report, ["claim_contracts", "event", "stability_status"]
    )
    f16_projection = nested_delete(
        f16_report, ["claim_contracts", "event", "stability_status"]
    )
    p02_full = predictions[("P02_valid_event_factor2", "FULL")]
    f16_full = predictions[("F16_unverified_stability", "FULL")]
    w2_checks = {
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
            p02_full["predicted_release_status"] == "ALLOWED"
            and f16_full["predicted_release_status"] == "BLOCKED_UNVERIFIED"
        ),
    }
    w2 = {
        "id": "W2",
        "name": "stability_semantics",
        "passed": all(w2_checks.values()),
        "cases": ["P02_valid_event_factor2", "F16_unverified_stability"],
        "projection": "mapping + audit row + base scalar + event query; stability status hidden",
        "differing_paths": w2_diff,
        "checks": w2_checks,
        "full_outputs": [exact_tuple(p02_full), exact_tuple(f16_full)],
    }

    f05 = cases["F05_runtime_noise_mismatch"]
    p01_report = load_case_report(p01)
    f05_report = load_case_report(f05)
    w3_diff = recursive_diff(p01_report, f05_report)
    p01_projection = copy.deepcopy(p01_report)
    f05_projection = copy.deepcopy(f05_report)
    p01_projection.pop("runtime_trace", None)
    f05_projection.pop("runtime_trace", None)
    f05_full = predictions[("F05_runtime_noise_mismatch", "FULL")]
    w3_checks = {
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
            p01_full["predicted_release_status"] == "ALLOWED"
            and f05_full["predicted_release_status"] == "BLOCKED_INVALID"
        ),
    }
    w3 = {
        "id": "W3",
        "name": "executed_mechanism_binding",
        "passed": all(w3_checks.values()),
        "cases": ["P01_valid_window", "F05_runtime_noise_mismatch"],
        "projection": "mapping + audit row + base scalar + window query; runtime hidden",
        "differing_paths": w3_diff,
        "checks": w3_checks,
        "full_outputs": [exact_tuple(p01_full), exact_tuple(f05_full)],
    }

    f01_a2 = predictions[("F01_vacuous_owner_group", "A2_FULL_MINUS_VACUITY")]
    f01_full = predictions[("F01_vacuous_owner_group", "FULL")]
    w4_checks = {
        "a2_allows": f01_a2["predicted_release_status"] == "ALLOWED",
        "a2_unsafe_positive": f01_a2["unsafe_positive"] == "True",
        "full_blocks_vacuous": (
            f01_full["predicted_release_status"] == "BLOCKED_VACUOUS"
        ),
        "full_not_unsafe_positive": f01_full["unsafe_positive"] == "False",
    }
    w4 = {
        "id": "W4",
        "name": "reportability_vacuity_gate",
        "passed": all(w4_checks.values()),
        "cases": ["F01_vacuous_owner_group"],
        "projection": "FULL with the vacuity gate removed",
        "checks": w4_checks,
        "outputs": {
            "A2_FULL_MINUS_VACUITY": exact_tuple(f01_a2),
            "FULL": exact_tuple(f01_full),
        },
    }

    w5 = run_wording_tamper()
    witnesses = [w1, w2, w3, w4, w5]
    passed_count = sum(1 for witness in witnesses if witness["passed"])
    report = {
        "schema_version": "audit_composition_witnesses_v12.1",
        "protocol": {
            "path": relative(PROTOCOL),
            "sha256": sha256_file(PROTOCOL),
        },
        "sources": sources,
        "query_separation": query_separation,
        "witnesses": witnesses,
        "summary": {
            "query_separation_passed": query_separation["passed"],
            "witnesses_passed": passed_count,
            "witnesses_total": len(witnesses),
            "all_passed": (
                query_separation["passed"] and passed_count == len(witnesses)
            ),
        },
        "interpretation": {
            "supported": (
                "On the fixed contract and artifacts, omitting the requested "
                "query or one tested evidence/wording edge loses distinctions "
                "required for exact query-responsive reporting."
            ),
            "not_supported": [
                "universal minimality",
                "real-world error prevalence",
                "predecessor-system error rates",
                "execution attestation",
                "independent case sourcing",
            ],
        },
    }
    return report


def write_outputs(report: Mapping[str, Any], output_dir: Path) -> None:
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite nonempty {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "witnesses.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with (output_dir / "witnesses.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "name", "passed", "projection"],
        )
        writer.writeheader()
        for witness in report["witnesses"]:
            projection = witness.get("projection") or witness["mutation"]
            writer.writerow(
                {
                    "id": witness["id"],
                    "name": witness["name"],
                    "passed": witness["passed"],
                    "projection": projection,
                }
            )

    lines = [
        "# Audit V12 Composition-Necessity Witnesses",
        "",
        f"- Query separation: {'PASS' if report['query_separation']['passed'] else 'FAIL'}",
        (
            f"- Fixed witnesses: {report['summary']['witnesses_passed']}/"
            f"{report['summary']['witnesses_total']} PASS"
        ),
        f"- Overall: {'PASS' if report['summary']['all_passed'] else 'FAIL'}",
        "",
        "| ID | Fixed omission/projection | Result |",
        "|---|---|---:|",
    ]
    for witness in report["witnesses"]:
        projection = witness.get("projection") or witness["mutation"]
        lines.append(
            f"| {witness['id']} | {projection} | "
            f"{'PASS' if witness['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "These are fixed information/obligation witnesses, not replicas or",
            "error-rate estimates for predecessor systems. They establish neither",
            "universal minimality nor real-world prevalence.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="reports/audit_composition_witnesses_v12_003",
    )
    args = parser.parse_args()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    report = build_report()
    write_outputs(report, output_dir)
    print(f"all_passed={report['summary']['all_passed']}")
    print(
        "witnesses="
        f"{report['summary']['witnesses_passed']}/"
        f"{report['summary']['witnesses_total']}"
    )
    print(f"Wrote {output_dir}")
    raise SystemExit(0 if report["summary"]["all_passed"] else 1)


if __name__ == "__main__":
    main()
