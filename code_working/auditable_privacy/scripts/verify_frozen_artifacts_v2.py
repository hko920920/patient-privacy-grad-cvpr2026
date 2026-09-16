#!/usr/bin/env python3
"""Verify the current Audit package and write only fresh reproduction outputs."""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "reproduced_verification"


def default_tf_python() -> str:
    environment = PROJECT_ROOT / ".venv-tfprivacy-baseline"
    candidate = (
        environment / "Scripts" / "python.exe"
        if os.name == "nt"
        else environment / "bin" / "python"
    )
    return str(candidate)


def run(label: str, *arguments: str) -> None:
    print(f"\n[{label}]")
    completed = subprocess.run(
        [sys.executable, *arguments],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if completed.returncode:
        raise SystemExit(f"{label} failed with exit code {completed.returncode}")


def run_preserving_bundle_record(
    label: str,
    bundle_record: Path,
    reproduced_record: Path,
    *arguments: str,
) -> None:
    original = bundle_record.read_bytes()
    try:
        run(label, *arguments)
        reproduced_record.parent.mkdir(parents=True, exist_ok=True)
        reproduced_record.write_bytes(bundle_record.read_bytes())
    finally:
        bundle_record.write_bytes(original)


def run_preserving_bundle_records(
    label: str,
    bundle_records: tuple[Path, ...],
    reproduced_root: Path,
    *arguments: str,
) -> None:
    originals = {path: path.read_bytes() for path in bundle_records}
    try:
        run(label, *arguments)
        reproduced_root.mkdir(parents=True, exist_ok=True)
        for path in bundle_records:
            (reproduced_root / path.name).write_bytes(path.read_bytes())
    finally:
        for path, original in originals.items():
            path.write_bytes(original)


def compare_generated_tree(
    expected: Path,
    actual: Path,
    label: str,
) -> None:
    expected_files = sorted(
        path.relative_to(expected).as_posix()
        for path in expected.rglob("*")
        if path.is_file()
    )
    actual_files = sorted(
        path.relative_to(actual).as_posix()
        for path in actual.rglob("*")
        if path.is_file()
    )
    if expected_files != actual_files:
        raise SystemExit(
            "Generated-table file set differs: "
            f"expected={expected_files}, actual={actual_files}"
        )
    differing = [
        relative
        for relative in expected_files
        if not filecmp.cmp(
            expected / relative,
            actual / relative,
            shallow=False,
        )
    ]
    if differing:
        raise SystemExit(f"{label} bytes differ: {differing}")
    print(
        f"{label} byte-identical: "
        f"{len(expected_files)}/{len(expected_files)}"
    )


def require_byte_identical(expected: Path, actual: Path, label: str) -> None:
    if not expected.is_file() or not actual.is_file():
        raise SystemExit(f"{label}: missing expected or rebuilt file")
    if not filecmp.cmp(expected, actual, shallow=False):
        raise SystemExit(f"{label}: rebuilt bytes differ")
    print(f"{label}: byte-identical")


def frozen_generated_dir(name: str) -> Path:
    packaged = PROJECT_ROOT / "paper" / name
    if packaged.is_dir():
        return packaged
    workspace = (
        PROJECT_ROOT.parent
        / "AAAI27_7p_compressed"
        / "01_audit_reporting_paper"
        / name
    )
    if not workspace.is_dir():
        raise SystemExit(f"Missing frozen generated directory: {name}")
    return workspace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-tfprivacy",
        action="store_true",
        help="Also rerun the official TensorFlow Privacy wheel baseline.",
    )
    parser.add_argument(
        "--tf-python",
        default=None,
        help=(
            "Interpreter prepared by scripts/setup_tfprivacy_baseline.py "
            "(auto-detected when omitted)."
        ),
    )
    args = parser.parse_args()

    schema = json.loads(
        (PROJECT_ROOT / "docs/privacy_report_schema_v2_0.json").read_text(
            encoding="utf-8"
        )
    )
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise SystemExit("Unexpected privacy-report JSON Schema declaration")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run(
        "91-test package regression",
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_*.py",
        "-v",
    )
    run(
        "validator ablation",
        "scripts/verify_validator_ablation_v2.py",
        "--bundle-dir",
        "reports/validator_ablation_v5_p0_001",
        "--output-json",
        "reproduced_verification/validator_ablation.json",
    )
    designed_root = (
        PROJECT_ROOT / "reports/validator_ablation_v3_1_200_001"
    )
    designed_records = tuple(
        designed_root / name
        for name in (
            "independent_verification.json",
            "verification_index.json",
        )
    )
    run_preserving_bundle_records(
        "designed 200-case conformance verification",
        designed_records,
        OUTPUT_DIR / "validator_ablation_v3_1_200",
        "scripts/verify_validator_ablation_v3_2.py",
        "--report-dir",
        "reports/validator_ablation_v3_1_200_001",
    )
    for path in designed_records:
        require_byte_identical(
            path,
            OUTPUT_DIR / "validator_ablation_v3_1_200" / path.name,
            f"designed conformance {path.name}",
        )
    run(
        "fixed composition-necessity witnesses",
        "scripts/verify_audit_composition_witnesses_v13.py",
        "--report",
        "reports/audit_composition_witnesses_v12_003/witnesses.json",
        "--output-json",
        "reproduced_verification/composition_necessity.json",
    )
    run(
        "raw-data-free external-source intake",
        "scripts/verify_audit_external_intake_portable_v12.py",
        "--record",
        (
            "reports/audit_external_extrasensory_v12_intake_001/"
            "portable_intake.json"
        ),
        "--output-json",
        "reproduced_verification/external_intake_portable.json",
    )
    run(
        "raw-data-free V16 external execution evidence",
        "scripts/verify_external_extrasensory_v16_portable.py",
        "--root",
        "reports/audit_external_extrasensory_v16_portable_001",
        "--output",
        "reproduced_verification/external_execution_v16_portable.json",
    )
    run_preserving_bundle_record(
        "retrospective development-lineage verification",
        PROJECT_ROOT
        / "reports/audit_retrospective_failures_v5_001/"
        "independent_verification_portable.json",
        OUTPUT_DIR / "retrospective.json",
        "scripts/verify_audit_retrospective_failures_v13.py",
        "--bundle-dir",
        "reports/audit_retrospective_failures_v5_001",
        "--portable-package",
    )
    run(
        "WISDM bound run",
        "scripts/verify_wisdm_v2_gold.py",
        "--evidence-dir",
        "reports/wisdm_v2_hardened_secure_002",
        "--skip-input-rescan",
        "--output-json",
        "reproduced_verification/wisdm.json",
    )
    multirun_root = "reports/wisdm_v4_multirun_001"
    run(
        "WISDM excluded pilot portable check",
        "scripts/verify_wisdm_v2_gold.py",
        "--evidence-dir",
        f"{multirun_root}/noise2_run01",
        "--skip-input-rescan",
        "--output-json",
        "reproduced_verification/wisdm_pilot.json",
    )
    for index in range(1, 6):
        run_id = f"confirmatory_run{index:02d}"
        run(
            f"WISDM {run_id} portable check",
            "scripts/verify_wisdm_v2_gold.py",
            "--evidence-dir",
            f"{multirun_root}/{run_id}",
            "--skip-input-rescan",
            "--output-json",
            f"reproduced_verification/{run_id}.json",
        )
    uci_pilot_root = (
        PROJECT_ROOT / "reports/uci_har_v5_pilot_002"
    )
    run_preserving_bundle_record(
        "UCI excluded complete pilot portable check",
        uci_pilot_root / "independent_verification_portable.json",
        OUTPUT_DIR / "uci_pilot.json",
        "scripts/verify_uci_har_v5.py",
        "--evidence-dir",
        "reports/uci_har_v5_pilot_002",
        "--skip-input-rescan",
    )
    uci_multirun_root = (
        PROJECT_ROOT / "reports/uci_har_v5_multirun_001"
    )
    for index in range(1, 6):
        run_id = f"confirmatory_run{index:02d}"
        run_preserving_bundle_record(
            f"UCI {run_id} portable check",
            uci_multirun_root
            / run_id
            / "independent_verification_portable.json",
            OUTPUT_DIR / f"uci_{run_id}.json",
            "scripts/verify_uci_har_v5.py",
            "--evidence-dir",
            f"reports/uci_har_v5_multirun_001/{run_id}",
            "--skip-input-rescan",
        )
    uci_aggregate_records = tuple(
        uci_multirun_root / name
        for name in (
            "confirmatory_metrics.csv",
            "confirmatory_summary.json",
            "confirmatory_summary.md",
        )
    )
    run_preserving_bundle_records(
        "UCI five-run aggregate",
        uci_aggregate_records,
        OUTPUT_DIR / "uci_multirun_summary",
        "scripts/summarize_uci_har_v5_multirun.py",
        "--root",
        "reports/uci_har_v5_multirun_001",
    )
    for path in uci_aggregate_records:
        require_byte_identical(
            path,
            OUTPUT_DIR / "uci_multirun_summary" / path.name,
            f"UCI aggregate {path.name}",
        )
    rebuilt_multirun = OUTPUT_DIR / "multirun_summary"
    run(
        "WISDM five-run aggregate",
        "scripts/summarize_wisdm_v4_multirun.py",
        "--root",
        multirun_root,
        "--output-root",
        str(rebuilt_multirun),
    )
    for name in (
        "confirmatory_summary.json",
        "confirmatory_utility.csv",
        "confirmatory_summary.md",
    ):
        require_byte_identical(
            PROJECT_ROOT / multirun_root / name,
            rebuilt_multirun / name,
            f"WISDM aggregate {name}",
        )
    run(
        "cost and pilot record",
        "scripts/verify_audit_v4_costs.py",
        "--portable-package",
        "--output-json",
        "reproduced_verification/audit_v4_costs.json",
    )
    run_preserving_bundle_record(
        "Sepsis portable mapping rescan",
        PROJECT_ROOT
        / "reports/sepsis2019_v2_mapping_evidence_5k_001"
        / "independent_verification.json",
        OUTPUT_DIR / "sepsis.json",
        "scripts/verify_sepsis2019_v2_evidence.py",
        "--bundle-dir",
        "reports/sepsis2019_v2_mapping_evidence_5k_001",
        "--skip-input-rescan",
    )
    run_preserving_bundle_record(
        "Backblaze portable aggregate rescan",
        PROJECT_ROOT
        / "reports/backblaze_v2_mapping_evidence_q1_2025_001"
        / "independent_verification.json",
        OUTPUT_DIR / "backblaze.json",
        "scripts/verify_backblaze_v2_evidence.py",
        "--bundle-dir",
        "reports/backblaze_v2_mapping_evidence_q1_2025_001",
        "--skip-archive-rescan",
    )

    with tempfile.TemporaryDirectory(prefix="audit_tables_v2_") as temp:
        rebuilt_root = Path(temp)
        run(
            "supplement table regeneration",
            "scripts/build_audit_supplement_tables_v2.py",
            "--paper-root",
            str(rebuilt_root),
        )
        compare_generated_tree(
            frozen_generated_dir("supplement_v2_generated"),
            rebuilt_root / "supplement_v2_generated",
            "v2 generated tables",
        )
    with tempfile.TemporaryDirectory(prefix="audit_tables_v5_") as temp:
        rebuilt_root = Path(temp)
        run(
            "v5 supplement table regeneration",
            "scripts/build_audit_v5_tables.py",
            "--project-root",
            str(PROJECT_ROOT),
            "--paper-root",
            str(rebuilt_root),
        )
        compare_generated_tree(
            frozen_generated_dir("supplement_v5_generated"),
            rebuilt_root / "supplement_v5_generated",
            "v5 generated tables",
        )
    with tempfile.TemporaryDirectory(prefix="audit_tables_v14_") as temp:
        rebuilt_root = Path(temp)
        run(
            "v14 200-case supplement table regeneration",
            "scripts/build_validator_ablation_v3_1_supplement_tables.py",
            "--repo-root",
            str(PROJECT_ROOT),
            "--paper-root",
            str(rebuilt_root),
        )
        compare_generated_tree(
            frozen_generated_dir("supplement_v14_generated"),
            rebuilt_root / "supplement_v14_generated",
            "v14 generated tables",
        )

    if args.include_tfprivacy:
        tf_python = args.tf_python or default_tf_python()
        if not Path(tf_python).is_file():
            raise SystemExit(
                "TensorFlow Privacy interpreter not found. Run "
                "'python scripts/setup_tfprivacy_baseline.py' first or pass "
                "--tf-python explicitly."
            )
        run(
            "TensorFlow Privacy official-wheel baseline",
            "scripts/verify_tfprivacy_statement_baseline_v2.py",
            "--bundle-dir",
            "reports/tfprivacy_statement_baseline_v2_001",
            "--tf-python",
            tf_python,
            "--output-json",
            "reproduced_verification/tfprivacy.json",
        )

    print("\nPortable verification surface: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
