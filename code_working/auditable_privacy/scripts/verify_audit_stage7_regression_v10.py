#!/usr/bin/env python3
"""Consolidate the final Audit v10 document and evidence regressions."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "audit_stage7_full_raw_regression_001"

EXPECTED_V5 = {
    "dist/audit_main_aaai27_v5.pdf": (
        "510ea8f3ac3d25d0185fac57defc42f1a521a004541bc5b9641cb2eec973a66d"
    ),
    "dist/audit_supplementary_document_aaai27_v5.pdf": (
        "2f01b7d8fce6a502d5f4a15857456b64fbc41ef978a3a0faaaf127e9ab0605a7"
    ),
    "dist/audit_reproducibility_checklist_aaai27_v5.pdf": (
        "98975d9889be7ee25d9d677120a99fba3d9008c2b1926289e6eb2c8175e1ebde"
    ),
    "dist/audit_code_and_data_supplement_aaai27_v5.zip": (
        "a455f7fdbc91a3ed1b60b9d1b62b6c414b885dbf9e9fdff4c7b61337739f4063"
    ),
    "dist/audit_submission_upload_manifest_v5.json": (
        "63095d616097781683fa57aa4b1b731d4e036f30f5d4ddf49eaba1cbfc1ba7b4"
    ),
}


def load_json(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def passed_count(
    payload: dict[str, Any],
    expected: int,
    *,
    full_input_key: str | None = None,
    full_input_value: bool | None = None,
) -> bool:
    passed = payload.get(
        "verification_passed",
        payload.get("all_passed", payload.get("passed", False)),
    )
    actual = payload.get("checks_passed", payload.get("check_count"))
    total = payload.get("checks_total", payload.get("check_count"))
    input_ok = True
    if full_input_key is not None:
        input_ok = payload.get(full_input_key) is full_input_value
    return bool(
        passed is True
        and actual == expected
        and total == expected
        and input_ok
    )


def main() -> int:
    rows: list[dict[str, Any]] = []

    def check(identifier: str, condition: bool, detail: Any) -> None:
        rows.append(
            {
                "id": identifier,
                "passed": bool(condition),
                "detail": detail,
            }
        )

    wisdm_names = (
        "reference",
        "excluded_pilot",
        "confirmatory_run01",
        "confirmatory_run02",
        "confirmatory_run03",
        "confirmatory_run04",
        "confirmatory_run05",
    )
    uci_source_dirs = (
        "reports/uci_har_v5_pilot_002",
        *(
            f"reports/uci_har_v5_multirun_001/confirmatory_run{index:02d}"
            for index in range(1, 6)
        ),
    )
    uci_reproduced_names = (
        "uci_pilot",
        *(f"uci_confirmatory_run{index:02d}" for index in range(1, 6)),
    )
    required = [
        "reproduced_verification/validator_ablation.json",
        "reproduced_verification/retrospective.json",
        "reproduced_verification/sepsis.json",
        "reproduced_verification/backblaze.json",
        "reproduced_verification/audit_v4_costs.json",
        "reports/audit_retrospective_failures_v5_001/independent_verification.json",
        "reports/tfprivacy_statement_baseline_v10_001/independent_verification.json",
        "reports/sepsis2019_v2_mapping_evidence_5k_001/independent_verification.json",
        "reports/backblaze_v2_mapping_evidence_q1_2025_001/independent_verification.json",
        "reports/uci_har_v5_multirun_001/confirmatory_summary.json",
        "reproduced_verification/uci_multirun_summary/confirmatory_summary.json",
        "reports/audit_manuscript_v10_gate_001/verification.json",
        "reports/audit_supplement_v10_gate_001/verification.json",
        "reports/audit_reproducibility_checklist_v10_gate_001/verification.json",
        "reports/audit_v10_integration_gate_001/gate.json",
        *(
            f"reports/audit_stage7_full_raw_regression_001/wisdm_{name}.json"
            for name in wisdm_names
        ),
        *(
            f"reproduced_verification/{name}.json"
            for name in (
                "wisdm",
                "wisdm_pilot",
                "confirmatory_run01",
                "confirmatory_run02",
                "confirmatory_run03",
                "confirmatory_run04",
                "confirmatory_run05",
            )
        ),
        *(
            f"{directory}/independent_verification.json"
            for directory in uci_source_dirs
        ),
        *(
            f"reproduced_verification/{name}.json"
            for name in uci_reproduced_names
        ),
    ]
    missing = sorted(relative for relative in required if not (ROOT / relative).is_file())
    check("required_records", not missing, missing or f"{len(required)}/{len(required)}")
    if missing:
        return finish(rows, required)

    unit = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    unit_output = unit.stdout + unit.stderr
    check(
        "contract_regression_83",
        unit.returncode == 0 and "Ran 83 tests" in unit_output and "OK" in unit_output,
        {"returncode": unit.returncode, "tail": unit_output[-500:]},
    )

    check(
        "controlled_probe_verifier_394",
        passed_count(
            load_json("reproduced_verification/validator_ablation.json"),
            394,
        ),
        "394/394",
    )
    check(
        "retrospective_full_176",
        passed_count(
            load_json(
                "reports/audit_retrospective_failures_v5_001/"
                "independent_verification.json"
            ),
            176,
        ),
        "176/176",
    )
    check(
        "retrospective_portable_156",
        passed_count(
            load_json("reproduced_verification/retrospective.json"),
            156,
        ),
        "156/156",
    )

    wisdm_full = [
        load_json(
            f"reports/audit_stage7_full_raw_regression_001/wisdm_{name}.json"
        )
        for name in wisdm_names
    ]
    check(
        "wisdm_full_raw_7x38",
        all(
            passed_count(
                payload,
                38,
                full_input_key="raw_inputs_rescanned",
                full_input_value=True,
            )
            for payload in wisdm_full
        ),
        "reference + excluded pilot + five confirmatory runs",
    )
    wisdm_portable = [
        load_json(f"reproduced_verification/{name}.json")
        for name in (
            "wisdm",
            "wisdm_pilot",
            "confirmatory_run01",
            "confirmatory_run02",
            "confirmatory_run03",
            "confirmatory_run04",
            "confirmatory_run05",
        )
    ]
    check(
        "wisdm_portable_7x35",
        all(
            passed_count(
                payload,
                35,
                full_input_key="raw_inputs_rescanned",
                full_input_value=False,
            )
            for payload in wisdm_portable
        ),
        "reference + excluded pilot + five confirmatory runs",
    )

    uci_full = [
        load_json(f"{directory}/independent_verification.json")
        for directory in uci_source_dirs
    ]
    check(
        "uci_full_raw_6x49",
        all(
            passed_count(
                payload,
                49,
                full_input_key="raw_inputs_rescanned",
                full_input_value=True,
            )
            for payload in uci_full
        ),
        "excluded pilot + five confirmatory runs",
    )
    uci_portable = [
        load_json(f"reproduced_verification/{name}.json")
        for name in uci_reproduced_names
    ]
    check(
        "uci_portable_6x38",
        all(
            passed_count(
                payload,
                38,
                full_input_key="raw_inputs_rescanned",
                full_input_value=False,
            )
            for payload in uci_portable
        ),
        "excluded pilot + five confirmatory runs",
    )
    uci_summary = load_json(
        "reports/uci_har_v5_multirun_001/confirmatory_summary.json"
    )
    rebuilt_uci_summary = load_json(
        "reproduced_verification/uci_multirun_summary/confirmatory_summary.json"
    )
    check(
        "uci_aggregate_31_byte_semantics",
        uci_summary == rebuilt_uci_summary
        and uci_summary.get("verification", {}).get("all_passed") is True
        and uci_summary.get("verification", {}).get("checks_passed")
        == uci_summary.get("verification", {}).get("checks_total")
        == 31,
        "31/31 and rebuilt JSON identical",
    )

    check(
        "sepsis_full_raw_128",
        passed_count(
            load_json(
                "reports/sepsis2019_v2_mapping_evidence_5k_001/"
                "independent_verification.json"
            ),
            128,
            full_input_key="raw_inputs_rescanned",
            full_input_value=True,
        ),
        "5,000 raw patient files",
    )
    check(
        "sepsis_portable_128",
        passed_count(
            load_json("reproduced_verification/sepsis.json"),
            128,
            full_input_key="raw_inputs_rescanned",
            full_input_value=False,
        ),
        "bundled mappings",
    )
    check(
        "backblaze_full_archive_86",
        passed_count(
            load_json(
                "reports/backblaze_v2_mapping_evidence_q1_2025_001/"
                "independent_verification.json"
            ),
            86,
            full_input_key="archive_rescanned",
            full_input_value=True,
        ),
        "1.02 GB archive, 90 members",
    )
    check(
        "backblaze_portable_83",
        passed_count(
            load_json("reproduced_verification/backblaze.json"),
            83,
            full_input_key="archive_rescanned",
            full_input_value=False,
        ),
        "owner-profile aggregate",
    )
    check(
        "tfprivacy_official_wheel_79",
        passed_count(
            load_json(
                "reports/tfprivacy_statement_baseline_v10_001/"
                "independent_verification.json"
            ),
            79,
        ),
        "official module rerun and complete setup surface",
    )
    cost = load_json("reproduced_verification/audit_v4_costs.json")
    check(
        "cost_record_portable_13",
        passed_count(cost, 13)
        and cost.get("portable_package_mode") is True,
        "13/13 with current registered mechanism entry",
    )

    document_specs = (
        ("manuscript", "reports/audit_manuscript_v10_gate_001/verification.json", 40),
        ("supplement", "reports/audit_supplement_v10_gate_001/verification.json", 59),
        (
            "checklist",
            "reports/audit_reproducibility_checklist_v10_gate_001/verification.json",
            13,
        ),
    )
    for name, relative, expected in document_specs:
        check(
            f"{name}_gate",
            passed_count(load_json(relative), expected),
            f"{expected}/{expected}",
        )
    integration = load_json("reports/audit_v10_integration_gate_001/gate.json")
    check(
        "integration_gate_29",
        integration.get("status") == "PASS"
        and integration.get("checks_passed")
        == integration.get("checks_total")
        == 29,
        "29/29",
    )
    v5_failures = [
        relative
        for relative, expected in EXPECTED_V5.items()
        if not (ROOT / relative).is_file()
        or sha256_file(ROOT / relative) != expected
    ]
    check(
        "frozen_v5_preserved",
        not v5_failures,
        v5_failures or "5/5",
    )

    return finish(rows, required)


def finish(rows: list[dict[str, Any]], required: list[str]) -> int:
    passed = all(row["passed"] for row in rows)
    artifact_hashes = {
        relative: sha256_file(ROOT / relative)
        for relative in sorted(required)
        if (ROOT / relative).is_file()
    }
    payload = {
        "schema_version": "audit_stage7_full_raw_regression_gate_v10",
        "status": "PASS" if passed else "FAIL",
        "checks_passed": sum(bool(row["passed"]) for row in rows),
        "checks_total": len(rows),
        "checks": rows,
        "artifact_hashes": artifact_hashes,
        "nonblocking_limits": [
            (
                "The historical cost record's exact whole-registry hash predates "
                "later registry extensions; the 13/13 portable gate verifies the "
                "bound mechanism entry and all measured values."
            ),
            (
                "Full raw rescans establish deterministic artifact consistency, "
                "not execution attestation, population prevalence, or "
                "cross-mechanism generality."
            ),
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    gate_path = OUTPUT / "gate.json"
    gate_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    failed = [row for row in rows if not row["passed"]]
    summary = [
        "# Audit v10 Stage-7 Full Regression Gate",
        "",
        f"Status: **{payload['status']}**",
        f"Checks: {payload['checks_passed']}/{payload['checks_total']}",
        "",
        "This gate consolidates fresh unit, portable, full-raw, document,",
        "integration, and frozen-v5 preservation checks.",
        "",
    ]
    if failed:
        summary.extend(
            [
                "## Failed checks",
                "",
                *[f"- `{row['id']}`: {row['detail']}" for row in failed],
                "",
            ]
        )
    summary.extend(
        [
            "## Explicit limits",
            "",
            *[f"- {item}" for item in payload["nonblocking_limits"]],
            "",
        ]
    )
    (OUTPUT / "summary.md").write_text(
        "\n".join(summary),
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"status={payload['status']} "
        f"checks={payload['checks_passed']}/{payload['checks_total']}"
    )
    print(f"result={gate_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
