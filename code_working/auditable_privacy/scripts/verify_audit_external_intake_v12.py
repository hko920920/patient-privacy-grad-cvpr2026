"""Independently verify the fixed Audit V12 external-source intake.

The verifier does not import the intake builder or production claim validator
and does not execute or retrain the external model.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "external_audits" / "extrasensory-dp"
ATTEMPT1_ROOT = PROJECT_ROOT / "reports" / "audit_external_extrasensory_v12_001"
ATTEMPT2_ROOT = PROJECT_ROOT / "reports" / "audit_external_extrasensory_v12_002"
EXPECTED_COMMIT = "167e0a899cb73c6e20398e8af886517cc9357c3e"
EXPECTED_TREE = "1106aa01d2f8806e72d4f99f5fc60814b3fdc9c7"
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


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(EXTERNAL_ROOT), *args],
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def read_split(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    decoded = raw.decode("utf-8")
    entries = [line.strip() for line in decoded.splitlines() if line.strip()]
    return {
        "path": path,
        "entries": entries,
        "entry_count": len(entries),
        "crlf_count": raw.count(b"\r\n"),
        "external_reader_retained_cr_count": sum(
            line.endswith("\r\n") for line in decoded.splitlines(keepends=True)
        ),
    }


def split_facts(config: dict[str, Any]) -> dict[str, Any]:
    train = [
        read_split((EXTERNAL_ROOT / value).resolve())
        for value in config["train_split_uuid_filepaths"]
    ]
    test = [
        read_split((EXTERNAL_ROOT / value).resolve())
        for value in config["test_split_uuid_filepaths"]
    ]
    train_uuids = [entry for row in train for entry in row["entries"]]
    test_uuids = [entry for row in test for entry in row["entries"]]
    return {
        "train": train,
        "test": test,
        "train_uuids": train_uuids,
        "test_uuids": test_uuids,
        "train_count": len(train_uuids),
        "test_count": len(test_uuids),
        "unique_count": len(set(train_uuids + test_uuids)),
        "crlf_count": sum(
            row["crlf_count"] for row in [*train, *test]
        ),
        "external_reader_retained_cr_count": sum(
            row["external_reader_retained_cr_count"]
            for row in [*train, *test]
        ),
    }


def data_inventory(data_dir: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(data_dir.glob("*.features_labels.csv.gz")):
        rows.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "canonical_inventory_sha256": canonical_sha256(rows),
    }


def mapping_facts(
    data_dir: Path,
    train_uuids: Iterable[str],
    target_labels: Iterable[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    columns = [f"label:{label}" for label in target_labels]
    digest = hashlib.sha256()
    owner_rows = []
    total = 0
    for uuid in train_uuids:
        path = data_dir / f"{uuid}.features_labels.csv.gz"
        selected = 0
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not set(columns).issubset(set(reader.fieldnames or [])):
                raise RuntimeError(f"target columns absent in {path}")
            for row_index, row in enumerate(reader):
                if not all(
                    row.get(column, "").lower() not in {"", "nan"}
                    for column in columns
                ):
                    continue
                timestamp = int(float(row["timestamp"]))
                digest.update(
                    f"{uuid}\0{row_index}\0{timestamp}\n".encode("utf-8")
                )
                selected += 1
        owner_rows.append({"owner_id": uuid, "selected_examples": selected})
        total += selected
    nonempty = [row["selected_examples"] for row in owner_rows if row["selected_examples"]]
    return (
        {
            "mapping_definition": (
                "ordered SHA-256 over owner_id NUL raw_row_index NUL timestamp LF "
                "for rows with all three target labels observed"
            ),
            "selected_training_examples": total,
            "training_owner_entries": len(owner_rows),
            "nonempty_training_owners": len(nonempty),
            "max_examples_per_owner": max(nonempty) if nonempty else 0,
            "mean_examples_per_nonempty_owner": (
                sum(nonempty) / len(nonempty) if nonempty else 0.0
            ),
            "ordered_mapping_sha256": digest.hexdigest(),
        },
        owner_rows,
    )


def secure_rng_fact(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    rows = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "secure_rng"
                and isinstance(value, ast.Constant)
            ):
                rows.append({"line": value.lineno, "value": value.value})
    return {
        "ast_occurrences": rows,
        "declared_value": rows[0]["value"] if len(rows) == 1 else None,
    }


def output_state(root: Path) -> dict[str, Any]:
    output = root / "run_outputs"
    if not output.is_dir():
        return {"files": [], "directories": []}
    return {
        "files": [
            path.relative_to(output).as_posix()
            for path in sorted(output.rglob("*"))
            if path.is_file()
        ],
        "directories": [
            path.relative_to(output).as_posix()
            for path in sorted(output.rglob("*"))
            if path.is_dir()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        default=(
            "reports/audit_external_extrasensory_v12_intake_001/intake.json"
        ),
    )
    parser.add_argument(
        "--output-json",
        default=(
            "reports/audit_external_extrasensory_v12_intake_001/"
            "independent_verification.json"
        ),
    )
    args = parser.parse_args()
    report_path = (PROJECT_ROOT / args.report).resolve()
    output_path = (PROJECT_ROOT / args.output_json).resolve()
    report = load_json(report_path)
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, condition: bool, observed: Any) -> None:
        checks[name] = {"passed": bool(condition), "observed": observed}

    check(
        "schema",
        report.get("schema_version") == "audit_external_intake_v12.1",
        report.get("schema_version"),
    )
    intake = report.get("intake", {})
    check(
        "positive_wording_empty",
        intake.get("positive_wording_allowed") is False
        and intake.get("supported_statement") == "",
        {
            "allowed": intake.get("positive_wording_allowed"),
            "statement_length": len(intake.get("supported_statement", "")),
        },
    )
    check(
        "typed_non_positive_decision",
        intake.get("release_status") == "BLOCKED_EXECUTION_INCOMPLETE"
        and intake.get("assurance_status") == "UNVERIFIED",
        {
            "release_status": intake.get("release_status"),
            "assurance_status": intake.get("assurance_status"),
        },
    )
    check(
        "issue_set",
        intake.get("issues") == EXPECTED_ISSUES,
        intake.get("issues"),
    )

    # A positive-claim forgery is rejected before expensive raw-data scanning.
    if not all(row["passed"] for row in checks.values()):
        result = {
            "schema_version": "audit_external_intake_v12_verification.1",
            "report_path": str(report_path),
            "report_sha256": sha256_file(report_path),
            "passed": False,
            "checks_passed": sum(row["passed"] for row in checks.values()),
            "checks_total": len(checks),
            "checks": checks,
            "early_rejection": True,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"passed=False checks={result['checks_passed']}/{result['checks_total']}")
        raise SystemExit(1)

    bindings_passed = 0
    binding_results = []
    for binding in report["source_bindings"]:
        path = (PROJECT_ROOT / binding["path"]).resolve()
        observed = {
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        passed = (
            observed["exists"]
            and observed["bytes"] == binding["bytes"]
            and observed["sha256"] == binding["sha256"]
        )
        bindings_passed += int(passed)
        binding_results.append(
            {"path": binding["path"], "passed": passed, **observed}
        )
    check(
        "all_source_bindings",
        bindings_passed == len(report["source_bindings"]),
        {
            "passed": bindings_passed,
            "total": len(report["source_bindings"]),
        },
    )

    observed_git = {
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "status": git("status", "--porcelain"),
    }
    check(
        "fixed_clean_external_tree",
        observed_git
        == {"commit": EXPECTED_COMMIT, "tree": EXPECTED_TREE, "status": ""},
        observed_git,
    )
    check(
        "external_authority_boundary",
        report["external_source"]["completed_training_run"] is False
        and report["external_source"]["independent_operator"] is False
        and report["external_source"]["commit"] == EXPECTED_COMMIT
        and report["external_source"]["tree"] == EXPECTED_TREE,
        report["external_source"],
    )

    original_path = EXTERNAL_ROOT / "exp_config_files" / "simple_nn.yaml"
    original = yaml.safe_load(original_path.read_text(encoding="utf-8"))
    config_checks = []
    for attempt, root in ((1, ATTEMPT1_ROOT), (2, ATTEMPT2_ROOT)):
        derived = yaml.safe_load(
            (root / "simple_nn_v12.yaml").read_text(encoding="utf-8")
        )
        expected = copy.deepcopy(original)
        expected["exp_data_directory"] = (
            f"../../reports/audit_external_extrasensory_v12_00{attempt}/"
            "run_outputs"
        )
        config_checks.append(derived == expected)
    check(
        "derived_configs_only_change_output_directory",
        all(config_checks)
        and report["fixed_config"][
            "attempt_configs_differ_only_in_output_directory"
        ]
        is True,
        config_checks,
    )

    status1 = load_json(ATTEMPT1_ROOT / "launch_result.json")
    status2 = load_json(ATTEMPT2_ROOT / "launch_result.json")
    stderr1 = (ATTEMPT1_ROOT / "external_run.stderr.txt").read_text(
        encoding="utf-8"
    )
    stderr2 = (ATTEMPT2_ROOT / "external_run.stderr.txt").read_text(
        encoding="utf-8"
    )
    state1 = output_state(ATTEMPT1_ROOT)
    state2 = output_state(ATTEMPT2_ROOT)
    attempts_ok = (
        status1["outcome"] == "failed"
        and status1["exception_type"] == "ModuleNotFoundError"
        and status2["outcome"] == "failed"
        and status2["exception_type"] == "OSError"
        and "No module named 'dataset'" in stderr1
        and "\\r.features_labels.csv.gz" in stderr2
        and not status1["output_files"]
        and not status2["output_files"]
        and not state1["files"]
        and not state2["files"]
        and status1["before"]["status"] == status1["after"]["status"] == ""
        and status2["before"]["status"] == status2["after"]["status"] == ""
    )
    check(
        "retained_attempt_outcomes",
        attempts_ok,
        {
            "attempt1": {
                "outcome": status1["outcome"],
                "exception": status1["exception_type"],
                "output_files": len(state1["files"]),
            },
            "attempt2": {
                "outcome": status2["outcome"],
                "exception": status2["exception_type"],
                "output_files": len(state2["files"]),
                "output_directories": state2["directories"],
            },
        },
    )
    check(
        "attempt_report_alignment",
        report["attempts"][0]["outcome"] == status1["outcome"]
        and report["attempts"][0]["exception_type"]
        == status1["exception_type"]
        and report["attempts"][0]["observed_output_file_count"]
        == len(state1["files"])
        and report["attempts"][1]["outcome"] == status2["outcome"]
        and report["attempts"][1]["exception_type"]
        == status2["exception_type"]
        and report["attempts"][1]["observed_output_file_count"]
        == len(state2["files"])
        and report["attempts"][1]["observed_output_directories"]
        == state2["directories"],
        report["attempts"],
    )

    splits = split_facts(original)
    split_summary = {
        "train_uuid_entries": splits["train_count"],
        "test_uuid_entries": splits["test_count"],
        "unique_uuid_entries": splits["unique_count"],
        "crlf_count": splits["crlf_count"],
        "external_reader_retained_cr_count": splits[
            "external_reader_retained_cr_count"
        ],
    }
    reported_split_summary = {
        key: report["split_inventory"][key] for key in split_summary
    }
    check(
        "split_counts_and_line_endings",
        split_summary == reported_split_summary
        and split_summary
        == {
            "train_uuid_entries": 48,
            "test_uuid_entries": 12,
            "unique_uuid_entries": 60,
            "crlf_count": 60,
            "external_reader_retained_cr_count": 60,
        },
        split_summary,
    )
    reported_split_files = [
        *report["split_inventory"]["train_files"],
        *report["split_inventory"]["test_files"],
    ]
    observed_split_files = [*splits["train"], *splits["test"]]
    split_file_checks = []
    for reported, observed in zip(reported_split_files, observed_split_files):
        split_file_checks.append(
            reported["entry_count"] == observed["entry_count"]
            and reported["crlf_count"] == observed["crlf_count"]
            and reported["external_reader_retained_cr_count"]
            == observed["external_reader_retained_cr_count"]
            and reported["file"]["sha256"] == sha256_file(observed["path"])
            and reported["file"]["bytes"] == observed["path"].stat().st_size
        )
    check(
        "split_file_bindings",
        len(split_file_checks) == 4 and all(split_file_checks),
        split_file_checks,
    )

    data_dir = (EXTERNAL_ROOT / original["user_data_files_directory"]).resolve()
    inventory = data_inventory(data_dir)
    check(
        "raw_data_inventory",
        inventory == report["raw_data_inventory"]
        and inventory["file_count"] == 60
        and inventory["total_bytes"] == 225325871,
        inventory,
    )
    mapping, owner_rows = mapping_facts(
        data_dir,
        splits["train_uuids"],
        original["target_labels"],
    )
    check(
        "source_mapping_recomputed",
        mapping == report["source_mapping"]
        and mapping["selected_training_examples"] == 250906
        and mapping["nonempty_training_owners"] == 47
        and mapping["max_examples_per_owner"] == 9404,
        mapping,
    )
    owner_counts_path = report_path.parent / "owner_counts.csv"
    with owner_counts_path.open("r", newline="", encoding="utf-8") as handle:
        reported_owner_rows = [
            {
                "owner_id": row["owner_id"],
                "selected_examples": int(row["selected_examples"]),
            }
            for row in csv.DictReader(handle)
        ]
    check(
        "owner_count_table",
        reported_owner_rows == owner_rows,
        {
            "rows": len(reported_owner_rows),
            "sha256": sha256_file(owner_counts_path),
        },
    )

    rng = secure_rng_fact(EXTERNAL_ROOT / "run_experiment.py")
    check(
        "secure_rng_ast",
        rng == report["source_randomness"]
        and rng["declared_value"] is False,
        rng,
    )
    check(
        "summary_alignment",
        report["summary"]
        == {
            "attempts_retained": 2,
            "attempts_completed": 0,
            "output_files": 0,
            "positive_statements": 0,
            "mapped_training_examples": mapping[
                "selected_training_examples"
            ],
            "all_non_positive_gates_passed": True,
        },
        report["summary"],
    )
    check(
        "authority_exclusions",
        set(intake["excluded_authority"])
        == {
            "completed model execution",
            "utility or privacy output",
            "faithful execution",
            "independent validation",
            "public-project error rate or prevalence",
        },
        intake["excluded_authority"],
    )

    passed = all(row["passed"] for row in checks.values())
    result = {
        "schema_version": "audit_external_intake_v12_verification.1",
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "passed": passed,
        "checks_passed": sum(row["passed"] for row in checks.values()),
        "checks_total": len(checks),
        "checks": checks,
        "source_binding_details": binding_results,
        "production_validator_imported": False,
        "builder_imported": False,
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
