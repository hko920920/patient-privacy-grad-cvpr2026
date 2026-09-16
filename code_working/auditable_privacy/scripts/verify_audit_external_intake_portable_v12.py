"""Verify the raw-data-free portable Audit V12 ExtraSensory intake.

This verifier authenticates the minimal MIT source snapshot, normalized
attempt record, frozen full-rescan binding, and non-positive decision.  It
does not claim to rescan the excluded raw feature/label files.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COMMIT = "167e0a899cb73c6e20398e8af886517cc9357c3e"
EXPECTED_TREE = "1106aa01d2f8806e72d4f99f5fc60814b3fdc9c7"
LOCAL_USER_TOKEN = "".join(("SO", "GANG"))
WINDOWS_USERS_PREFIX = "C:" + chr(92) + "Users"
EXPECTED_STATUS_HASHES = {
    1: "a5232c6484050b868a0e357fffacb49646722d5c42ed8f478d8ed7dac0bf2f1d",
    2: "18e46877a9cc481e6b0cfee397950aa7fb41fecf31c5fd1010b4bf60ff2e85e9",
}
EXPECTED_STDERR_HASHES = {
    1: "feca35ff27515a8cf727246ee9209e96d2dda03f3a86249fd8f3a2c7fd49185a",
    2: "23e4bfaa2eb11f1c9dda01bf1ba24962f54c3373014c7a7388733a3db1649fb8",
}
EXPECTED_ISSUES = [
    "ATTEMPT_1_LOCAL_IMPORT_FAILURE",
    "ATTEMPT_2_CRLF_UUID_PORTABILITY_FAILURE",
    "COMPLETED_EXTERNAL_RUN_ABSENT",
    "OUTPUT_ARTIFACTS_ABSENT",
    "EXTERNAL_ADAPTER_UNREGISTERED",
    "REGISTERED_RUNTIME_TRACE_ABSENT",
    "SECURE_RNG_DISABLED_IN_SOURCE",
    "OWNER_STABILITY_CONTRACT_ABSENT",
]


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


def source_root() -> tuple[Path, bool]:
    packaged = PROJECT_ROOT / "third_party" / "extrasensory-dp"
    if packaged.is_dir():
        return packaged, True
    local = PROJECT_ROOT / "external_audits" / "extrasensory-dp"
    if local.is_dir():
        return local, False
    raise FileNotFoundError("ExtraSensory source snapshot is absent")


def rng_values(path: Path) -> list[Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "secure_rng"
                and isinstance(value, ast.Constant)
            ):
                values.append(value.value)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record",
        default=(
            "reports/audit_external_extrasensory_v12_intake_001/"
            "portable_intake.json"
        ),
    )
    parser.add_argument(
        "--output-json",
        default=(
            "reports/audit_external_extrasensory_v12_intake_001/"
            "portable_verification.json"
        ),
    )
    args = parser.parse_args()
    record_path = (PROJECT_ROOT / args.record).resolve()
    output_path = (PROJECT_ROOT / args.output_json).resolve()
    record = load_json(record_path)
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, condition: bool, observed: Any) -> None:
        checks[name] = {"passed": bool(condition), "observed": observed}

    check(
        "schema",
        record.get("schema_version")
        == "audit_external_intake_v12_portable.1",
        record.get("schema_version"),
    )
    authority = record.get("portable_authority", {})
    check(
        "empty_positive_wording",
        authority.get("positive_wording_allowed") is False
        and authority.get("supported_statement") == ""
        and authority.get("release_status")
        == "BLOCKED_EXECUTION_INCOMPLETE"
        and authority.get("assurance_status") == "UNVERIFIED",
        {
            "positive_wording_allowed": authority.get(
                "positive_wording_allowed"
            ),
            "statement_length": len(authority.get("supported_statement", "")),
            "release_status": authority.get("release_status"),
            "assurance_status": authority.get("assurance_status"),
        },
    )
    check(
        "portable_authority_boundary",
        authority.get("raw_data_rescan_performed") is False
        and authority.get("completed_training_run") is False
        and authority.get("independent_operator") is False
        and authority.get("issues") == EXPECTED_ISSUES,
        {
            "raw_data_rescan_performed": authority.get(
                "raw_data_rescan_performed"
            ),
            "completed_training_run": authority.get(
                "completed_training_run"
            ),
            "independent_operator": authority.get("independent_operator"),
            "issues": authority.get("issues"),
        },
    )
    if not all(row["passed"] for row in checks.values()):
        result = {
            "schema_version": "audit_external_intake_v12_portable_verify.1",
            "passed": False,
            "checks_passed": sum(row["passed"] for row in checks.values()),
            "checks_total": len(checks),
            "checks": checks,
            "early_rejection": True,
            "raw_data_rescan_performed": False,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"passed=False checks={result['checks_passed']}/{result['checks_total']}")
        raise SystemExit(1)

    external_root, packaged = source_root()
    source_results = []
    for row in record["source_files"]:
        relative = Path(row["portable_path"]).relative_to(
            "third_party/extrasensory-dp"
        )
        path = external_root / relative
        source_results.append(
            path.is_file()
            and path.stat().st_size == row["bytes"]
            and sha256_file(path) == row["sha256"]
        )
    check(
        "source_snapshot_bindings",
        bool(source_results) and all(source_results),
        {"passed": sum(source_results), "total": len(source_results)},
    )
    check(
        "source_inventory_digest",
        canonical_sha256(record["source_files"])
        == record["source_inventory_sha256"],
        record["source_inventory_sha256"],
    )
    license_text = (external_root / "LICENSE").read_text(encoding="utf-8")
    check(
        "mit_license_retained",
        "MIT License" in license_text
        and "Copyright (c) 2023 ExtraSensory DP Team" in license_text,
        sha256_file(external_root / "LICENSE"),
    )
    packaged_raw_files = (
        list(
            (
                external_root / "ExtraSensory.per_uuid_features_labels"
            ).glob("*.features_labels.csv.gz")
        )
        if packaged
        else []
    )
    check(
        "raw_data_not_redistributed",
        record["external_source"]["raw_data_redistributed"] is False
        and not packaged_raw_files,
        {"packaged_mode": packaged, "raw_files": len(packaged_raw_files)},
    )

    evidence_results = []
    for row in record["evidence_files"]:
        path = (PROJECT_ROOT / row["portable_path"]).resolve()
        evidence_results.append(
            path.is_file()
            and path.stat().st_size == row["bytes"]
            and sha256_file(path) == row["sha256"]
        )
    check(
        "portable_evidence_bindings",
        bool(evidence_results) and all(evidence_results),
        {"passed": sum(evidence_results), "total": len(evidence_results)},
    )
    check(
        "evidence_inventory_digest",
        canonical_sha256(record["evidence_files"])
        == record["evidence_inventory_sha256"],
        record["evidence_inventory_sha256"],
    )

    intake_path = (
        PROJECT_ROOT
        / "reports"
        / "audit_external_extrasensory_v12_intake_001"
        / "intake.json"
    )
    intake = load_json(intake_path)
    full = record["full_rescan_record"]
    check(
        "frozen_full_rescan_binding",
        sha256_file(intake_path) == full["intake_sha256"]
        and full["passed"] is True
        and full["checks_passed"] == full["checks_total"] == 18
        and len(full["check_results"]) == 18
        and all(full["check_results"].values()),
        {
            "intake_sha256": sha256_file(intake_path),
            "checks": f"{full['checks_passed']}/{full['checks_total']}",
        },
    )
    check(
        "frozen_mapping_authority",
        full["raw_data_inventory"] == intake["raw_data_inventory"]
        and full["source_mapping"] == intake["source_mapping"]
        and full["raw_data_inventory"]["file_count"] == 60
        and full["raw_data_inventory"]["total_bytes"] == 225325871
        and full["source_mapping"]["selected_training_examples"] == 250906
        and full["source_mapping"]["nonempty_training_owners"] == 47
        and full["source_mapping"]["max_examples_per_owner"] == 9404,
        {
            "raw_data_inventory": full["raw_data_inventory"],
            "selected_training_examples": full["source_mapping"][
                "selected_training_examples"
            ],
            "raw_data_rescan_performed": False,
        },
    )

    attempts_path = (
        PROJECT_ROOT
        / "reports"
        / "audit_external_extrasensory_v12_intake_001"
        / "portable_attempts.json"
    )
    attempts = load_json(attempts_path)
    attempt_rows = {row["attempt"]: row for row in attempts["attempts"]}
    attempt_checks = []
    for attempt in (1, 2):
        row = attempt_rows[attempt]
        signature = (
            "No module named 'dataset'"
            if attempt == 1
            else "\\r.features_labels.csv.gz"
        )
        expected_exception = "ModuleNotFoundError" if attempt == 1 else "OSError"
        attempt_checks.append(
            row["outcome"] == "failed"
            and row["exception_type"] == expected_exception
            and row["before"]["commit"]
            == row["after"]["commit"]
            == EXPECTED_COMMIT
            and row["before"]["tree"]
            == row["after"]["tree"]
            == EXPECTED_TREE
            and row["before"]["status"] == row["after"]["status"] == ""
            and not row["reported_output_files"]
            and row["original_files"]["status"]["sha256"]
            == EXPECTED_STATUS_HASHES[attempt]
            and row["original_files"]["stderr"]["sha256"]
            == EXPECTED_STDERR_HASHES[attempt]
            and signature in row["normalized_stderr"]
            and LOCAL_USER_TOKEN not in row["normalized_stderr"]
            and WINDOWS_USERS_PREFIX not in row["normalized_stderr"]
        )
    check(
        "normalized_attempt_records",
        attempts["path_normalization_only"] is True
        and sha256_file(attempts_path)
        == record["portable_attempts_sha256"]
        and all(attempt_checks),
        attempt_checks,
    )
    check(
        "attempt_summary_alignment",
        record["portable_attempts_summary"]
        == [
            {
                "attempt": attempt,
                "outcome": attempt_rows[attempt]["outcome"],
                "exception_type": attempt_rows[attempt]["exception_type"],
                "status_sha256": EXPECTED_STATUS_HASHES[attempt],
                "stderr_sha256": EXPECTED_STDERR_HASHES[attempt],
                "output_files": 0,
            }
            for attempt in (1, 2)
        ],
        record["portable_attempts_summary"],
    )

    original_config = yaml.safe_load(
        (external_root / "exp_config_files/simple_nn.yaml").read_text(
            encoding="utf-8"
        )
    )
    config_checks = []
    for attempt in (1, 2):
        derived = yaml.safe_load(
            (
                PROJECT_ROOT
                / "reports"
                / f"audit_external_extrasensory_v12_00{attempt}"
                / "simple_nn_v12.yaml"
            ).read_text(encoding="utf-8")
        )
        expected = copy.deepcopy(original_config)
        expected["exp_data_directory"] = (
            f"../../reports/audit_external_extrasensory_v12_00{attempt}/"
            "run_outputs"
        )
        config_checks.append(derived == expected)
    check(
        "derived_config_science_unchanged",
        all(config_checks),
        config_checks,
    )

    split_paths = [
        external_root / value
        for value in (
            *original_config["train_split_uuid_filepaths"],
            *original_config["test_split_uuid_filepaths"],
        )
    ]
    split_entries = []
    crlf = 0
    for path in split_paths:
        raw = path.read_bytes()
        split_entries.extend(
            line.strip()
            for line in raw.decode("utf-8").splitlines()
            if line.strip()
        )
        crlf += raw.count(b"\r\n")
    check(
        "split_crlf_facts",
        len(split_entries) == len(set(split_entries)) == 60 and crlf == 60,
        {"entries": len(split_entries), "unique": len(set(split_entries)), "crlf": crlf},
    )
    observed_rng = rng_values(external_root / "run_experiment.py")
    check(
        "source_secure_rng_false",
        observed_rng == [False]
        and intake["source_randomness"]["declared_value"] is False,
        observed_rng,
    )
    check(
        "external_identity",
        record["external_source"]["commit"] == EXPECTED_COMMIT
        and record["external_source"]["tree"] == EXPECTED_TREE
        and record["external_source"]["license"] == "MIT",
        record["external_source"],
    )

    passed = all(row["passed"] for row in checks.values())
    result = {
        "schema_version": "audit_external_intake_v12_portable_verify.1",
        "passed": passed,
        "checks_passed": sum(row["passed"] for row in checks.values()),
        "checks_total": len(checks),
        "checks": checks,
        "record_sha256": sha256_file(record_path),
        "packaged_source_mode": packaged,
        "raw_data_rescan_performed": False,
        "external_training_executed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"passed={passed} checks={result['checks_passed']}/{result['checks_total']}")
    print(f"Wrote {output_path}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
