#!/usr/bin/env python3
"""Build a deterministic machine-readable summary of the Audit v5 P0 gate."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "reports" / "audit_v5_p0_gate_001"
EXPECTED_V4 = {
    "dist/audit_main_aaai27_v4.pdf": (
        "ca70ed5a912551b75e3e3c3a722591edc5a6125e81937ce5b8f0932fc06a35bc"
    ),
    "dist/audit_supplementary_document_aaai27_v4.pdf": (
        "f1ca0822dfe5d20a9d02d2fe3d3ba4cc72d1785603f3fb0b6dcc58baf44bc775"
    ),
    "dist/audit_reproducibility_checklist_aaai27_v4.pdf": (
        "145e668062936acb7f663b278011b5ec195c352b445a9f4736538e2e05a7742a"
    ),
    "dist/audit_code_and_data_supplement_aaai27_v4.zip": (
        "646d44f70effc6d1b69f6d8dab2aa4840fcb478881314fcf1f02e65c2b3c1ee9"
    ),
    "dist/audit_submission_upload_manifest_v4.json": (
        "4dfa2d24f49273fa6a80340280001c774120368d488dba658d40fca3f0436e5d"
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def load_json(relative: str) -> dict[str, Any]:
    return json.loads((PROJECT_ROOT / relative).read_text(encoding="utf-8"))


def main() -> None:
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, detail: Any) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    unittest_log_path = PROJECT_ROOT / "reports/audit_v5_p0_unittest_stderr.log"
    unittest_log = unittest_log_path.read_text(encoding="utf-8")
    run_match = re.search(
        r"Ran\s+(\d+)\s+tests\s+in\s+([0-9.]+)s",
        unittest_log,
    )
    ok_lines = len(re.findall(r"\.\.\.\s+ok$", unittest_log, re.MULTILINE))
    check(
        "unittest",
        run_match is not None
        and int(run_match.group(1)) == 83
        and ok_lines == 83
        and unittest_log.rstrip().endswith("OK")
        and "FAILED" not in unittest_log,
        {
            "reported_tests": int(run_match.group(1)) if run_match else None,
            "ok_lines": ok_lines,
            "seconds": float(run_match.group(2)) if run_match else None,
            "log_sha256": file_sha256(unittest_log_path),
        },
    )

    ablation = load_json("reports/audit_v5_p0_ablation_verification.json")
    with (
        PROJECT_ROOT
        / "reports/validator_ablation_v5_p0_001/aggregate_metrics.csv"
    ).open(newline="", encoding="utf-8") as handle:
        aggregate_rows = list(csv.DictReader(handle))
    full = next(row for row in aggregate_rows if row["method"] == "FULL")
    check(
        "controlled_probe_verification",
        ablation["verification_passed"] is True
        and ablation["checks_passed"] == ablation["checks_total"] == 394,
        {
            "passed": ablation["checks_passed"],
            "total": ablation["checks_total"],
        },
    )
    check(
        "controlled_probe_full",
        int(full["cases"]) == 31
        and int(full["unsafe_positives"]) == 0
        and int(full["exact_tuples"]) == 31,
        {
            "cases": int(full["cases"]),
            "unsafe_positives": int(full["unsafe_positives"]),
            "exact_tuples": int(full["exact_tuples"]),
        },
    )

    wisdm_full = load_json("reports/audit_v5_p0_wisdm_full.json")
    wisdm_portable = load_json("reports/audit_v5_p0_wisdm_portable.json")
    check(
        "wisdm_full",
        wisdm_full["passed"] is True
        and wisdm_full["check_count"] == 38
        and wisdm_full["raw_inputs_rescanned"] is True,
        {
            "checks": wisdm_full["check_count"],
            "raw_inputs_rescanned": wisdm_full["raw_inputs_rescanned"],
        },
    )
    check(
        "wisdm_portable",
        wisdm_portable["passed"] is True
        and wisdm_portable["check_count"] == 35
        and wisdm_portable["raw_inputs_rescanned"] is False,
        {
            "checks": wisdm_portable["check_count"],
            "raw_inputs_rescanned": wisdm_portable[
                "raw_inputs_rescanned"
            ],
        },
    )

    retrospective = load_json(
        "reports/audit_retrospective_failures_v5_001/"
        "independent_verification.json"
    )
    check(
        "retrospective",
        retrospective["all_passed"] is True
        and retrospective["checks_passed"]
        == retrospective["checks_total"]
        == 176,
        {
            "passed": retrospective["checks_passed"],
            "total": retrospective["checks_total"],
        },
    )

    uci_manifest = load_json(
        "reports/uci_har_v5_multirun_001/execution_manifest.json"
    )
    uci_summary = load_json(
        "reports/uci_har_v5_multirun_001/confirmatory_summary.json"
    )
    uci_runs = uci_manifest["runs"]
    check(
        "uci_runs",
        len(uci_runs) == 5
        and all(run["passed"] for run in uci_runs)
        and len({run["artifact_hashes"]["released_model.json"] for run in uci_runs})
        == 5,
        {
            "runs": len(uci_runs),
            "distinct_models": len(
                {
                    run["artifact_hashes"]["released_model.json"]
                    for run in uci_runs
                }
            ),
        },
    )
    check(
        "uci_full_verifiers",
        all(
            run["independent_verification"]["all_passed"]
            and run["independent_verification"]["checks_passed"]
            == run["independent_verification"]["checks_total"]
            == 49
            for run in uci_runs
        ),
        [run["independent_verification"]["checks_passed"] for run in uci_runs],
    )
    check(
        "uci_portable_verifiers",
        all(
            run["independent_verification_portable"]["all_passed"]
            and run["independent_verification_portable"]["checks_passed"]
            == run["independent_verification_portable"]["checks_total"]
            == 38
            for run in uci_runs
        ),
        [
            run["independent_verification_portable"]["checks_passed"]
            for run in uci_runs
        ],
    )
    check(
        "uci_aggregate",
        uci_summary["verification"]["all_passed"] is True
        and uci_summary["verification"]["checks_passed"]
        == uci_summary["verification"]["checks_total"]
        == 31,
        {
            "passed": uci_summary["verification"]["checks_passed"],
            "total": uci_summary["verification"]["checks_total"],
        },
    )
    check(
        "uci_decision_boundary",
        uci_summary["decision_signature"]
        == [
            ["window", "ALLOWED", "DIRECT", 1, True],
            [
                "event",
                "BLOCKED_UNVERIFIED",
                "UNDERSPECIFIED",
                None,
                False,
            ],
            [
                "owner",
                "BLOCKED_UNVERIFIED",
                "UNDERSPECIFIED",
                None,
                False,
            ],
        ],
        uci_summary["decision_signature"],
    )

    for relative, expected in EXPECTED_V4.items():
        actual = file_sha256(PROJECT_ROOT / relative)
        check(
            f"frozen_v4:{Path(relative).name}",
            actual == expected,
            {"actual": actual, "expected": expected},
        )

    source_hashes = {
        relative: file_sha256(PROJECT_ROOT / relative)
        for relative in (
            "scripts/privacy_claim_validator.py",
            "docs/privacy_report_schema_v2_0.json",
            "docs/mechanism_registry_v2_0.json",
            "tests/test_uci_har_v5_contract.py",
            "reports/audit_v5_p0_ablation_verification.json",
            "reports/audit_v5_p0_wisdm_full.json",
            "reports/audit_v5_p0_wisdm_portable.json",
            "reports/audit_retrospective_failures_v5_001/"
            "independent_verification.json",
            "reports/uci_har_v5_multirun_001/execution_manifest.json",
            "reports/uci_har_v5_multirun_001/confirmatory_summary.json",
        )
    }
    passed = sum(item["passed"] for item in checks.values())
    result: dict[str, Any] = {
        "schema_version": "audit_v5_p0_gate_v1",
        "status": "PASS" if passed == len(checks) else "FAIL",
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
        "source_hashes": source_hashes,
        "claims": {
            "contract_tests": 83,
            "controlled_probe_cases": 31,
            "controlled_probe_unsafe_positives": 0,
            "controlled_probe_verifier_checks": 394,
            "retrospective_incidents": 10,
            "retrospective_verifier_checks": 176,
            "uci_confirmatory_runs": 5,
            "uci_full_checks_per_run": 49,
            "uci_portable_checks_per_run": 38,
            "uci_aggregate_checks": 31,
            "wisdm_full_checks": 38,
            "wisdm_portable_checks": 35,
        },
    }
    result["content_sha256"] = canonical_json_sha256(result)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "gate.json").write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Audit v5 P0 machine gate",
        "",
        f"- Status: {result['status']}",
        f"- Checks: {passed}/{len(checks)}",
        "- Contract tests: 83/83",
        "- FULL: 31/31 tuples; 0/25 unsafe positives",
        "- Retrospective verifier: 176/176",
        "- UCI: five runs, 49/49 full and 38/38 portable per run",
        "- WISDM compatibility: 38/38 full and 35/35 portable",
        "",
    ]
    (OUTPUT_ROOT / "summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(f"Audit v5 P0 gate: {passed}/{len(checks)} {result['status']}")
    if result["status"] != "PASS":
        failed = [name for name, item in checks.items() if not item["passed"]]
        raise SystemExit(f"Failed checks: {failed}")


if __name__ == "__main__":
    main()
