#!/usr/bin/env python3
"""Build the deterministic anonymous AAAI-27 Compiler V3.5 code/data ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from scripts.build_compiler_code_data_package_v3 import (
        SCRIPT_ALLOWLIST as V3_SCRIPT_ALLOWLIST,
        SINGLE_REPORT_FILES as V3_SINGLE_REPORT_FILES,
        PackageBuildError,
        add_file,
        add_tree,
        check_staging,
        ensure_inside_repo,
        file_sha256,
        public_execution_tree,
        write_deterministic_zip,
    )
except ModuleNotFoundError:
    from build_compiler_code_data_package_v3 import (
        SCRIPT_ALLOWLIST as V3_SCRIPT_ALLOWLIST,
        SINGLE_REPORT_FILES as V3_SINGLE_REPORT_FILES,
        PackageBuildError,
        add_file,
        add_tree,
        check_staging,
        ensure_inside_repo,
        file_sha256,
        public_execution_tree,
        write_deterministic_zip,
    )


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ID = "compiler_code_and_data_supplement_aaai27_v4"
DEFAULT_OUTPUT = ROOT / "dist" / PACKAGE_ID
DEFAULT_ZIP = ROOT / "dist" / f"{PACKAGE_ID}.zip"
MANIFEST_NAME = "PUBLIC_PACKAGE_MANIFEST.json"

V35_SCRIPTS = (
    "build_compiler_code_data_package_v3.py",
    "build_compiler_code_data_package_v4.py",
    "build_v35_random_allocation_accountant_gate.py",
    "verify_v35_random_allocation_accountant_gate.py",
    "build_v35_allocation_contract_gate.py",
    "verify_v35_allocation_contract_gate.py",
    "build_v35_allocation_executor_gate.py",
    "verify_v35_allocation_executor_gate.py",
    "build_v35_route_handler_evidence.py",
    "build_v35_handler_registry_gate.py",
    "verify_v35_handler_registry_gate.py",
    "build_v35_pairwise_hybrid_lattice_gate.py",
    "verify_v35_pairwise_hybrid_lattice_gate.py",
    "build_v35_allocation_multirun_gate.py",
    "verify_v35_allocation_multirun_gate.py",
    "build_v35_handler_hybrid_ablation_gate.py",
    "verify_v35_handler_hybrid_ablation_gate.py",
    "verify_public_supplement_v35.py",
)

V35_JSON_REPORT_DIRS = (
    "v35_random_allocation_accountant_gate_20260725",
    "v35_allocation_contract_gate_20260725",
    "v35_allocation_executor_gate_20260725",
    "v35_handler_registry_gate_v2_20260725",
    "v35_pairwise_hybrid_lattice_gate_v2_20260725",
    "v35_route_handler_evidence_20260725",
    "v35_handler_hybrid_ablation_gate_20260725",
    "v35_manuscript_gate_20260725",
)

PAPER_FILES = (
    "aaai2027.bst",
    "aaai2027.sty",
    "main_aaai27_v35_candidate.tex",
    "main_aaai27_supplement_v35.tex",
    "references_v28_candidate.bib",
    "references_v30_additions.bib",
    "references_v33_additions.bib",
    "references_v35_additions.bib",
    "tables_v35_candidate/v35_profiles.tex",
    "tables_v35_candidate/v35_handler_ablation.tex",
    "tables_v35_candidate/v35_conformance.tex",
)

TEST_EXCLUSIONS = {
    "test_v34_manuscript_gate.py",
    "test_v35_manuscript_gate.py",
    "test_v35_allocation_multirun_gate.py",
    "test_v35_public_package_gate.py",
}


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_paper_dir(path: Path) -> Path:
    resolved = path.resolve()
    missing = [
        relative
        for relative in PAPER_FILES
        if not (resolved / relative).is_file()
    ]
    if missing:
        raise PackageBuildError(
            "Paper directory is missing: " + ", ".join(missing)
        )
    return resolved


def build_manifest(output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == MANIFEST_NAME:
            continue
        rows.append(
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    rows.sort(key=lambda row: str(row["path"]))
    manifest: dict[str, Any] = {
        "schema_version": "unitdp.public_code_data_package.v35",
        "package_id": PACKAGE_ID,
        "visibility": "public",
        "release_status": "research_non_release",
        "raw_third_party_data_redistributed": False,
        "private_randomizers_or_step_traces_redistributed": False,
        "routes": ["P", "F", "A"],
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "files": rows,
    }
    manifest["payload_sha256"] = canonical_sha256(manifest)
    return manifest


def build(
    *,
    paper_dir: Path,
    output: Path,
    zip_path: Path,
) -> dict[str, Any]:
    output = ensure_inside_repo(output)
    zip_path = ensure_inside_repo(zip_path)
    if output.exists() or zip_path.exists():
        raise PackageBuildError(
            "Refusing to overwrite an existing package or ZIP"
        )
    paper_dir = require_paper_dir(paper_dir)
    output.mkdir(parents=True)
    seen: set[str] = set()

    add_file(
        ROOT / "submission" / "compiler_v35" / "README.md",
        "README.md",
        output=output,
        seen=seen,
    )
    add_file(
        ROOT
        / "submission"
        / "compiler_v35"
        / "requirements-aaai27-v35.txt",
        "requirements-aaai27-v35.txt",
        output=output,
        seen=seen,
    )
    add_file(
        ROOT / "pyproject.toml",
        "pyproject.toml",
        output=output,
        seen=seen,
    )

    add_tree(
        ROOT / "src",
        "src",
        output=output,
        seen=seen,
        suffixes={".py"},
    )
    add_tree(
        ROOT / "configs",
        "configs",
        output=output,
        seen=seen,
        suffixes={".json", ".yaml", ".yml"},
    )
    add_tree(
        ROOT / "specs",
        "specs",
        output=output,
        seen=seen,
        suffixes={".json"},
    )

    for name in sorted(set(V3_SCRIPT_ALLOWLIST).union(V35_SCRIPTS)):
        add_file(
            ROOT / "scripts" / name,
            f"scripts/{name}",
            output=output,
            seen=seen,
        )
    add_tree(
        ROOT / "tests",
        "tests",
        output=output,
        seen=seen,
        suffixes={".py"},
        exclude_names=TEST_EXCLUSIONS,
    )

    for relative_root in (
        "reports/v2_registered_5seed_v32_20260724",
        "reports/v2_nonprivate_reference_v32_20260724",
        "reports/v34_route_obligation_gate_20260725",
        "reports/v34_matched_shell_gate_20260725",
        "reports/v34_registry_lemma_gate_20260725",
    ):
        public_execution_tree(
            relative_root,
            output=output,
            seen=seen,
        )
    public_execution_tree(
        "reports/v3_srswor_registered_5seed_v32_20260724",
        output=output,
        seen=seen,
        exclude_top_level={
            "public_verification_srswor_v3.json",
            "public_verification_srswor_public_v3.json",
        },
    )
    for relative in V3_SINGLE_REPORT_FILES:
        add_file(
            ROOT / relative,
            relative,
            output=output,
            seen=seen,
        )

    for directory in V35_JSON_REPORT_DIRS:
        source_root = ROOT / "reports" / directory
        add_tree(
            source_root,
            f"reports/{directory}",
            output=output,
            seen=seen,
            suffixes={".json"},
        )
    public_execution_tree(
        "reports/v35_allocation_multirun_gate_20260725",
        output=output,
        seen=seen,
    )

    for relative in (
        "dist/compiler_submission_upload_manifest_v3.json",
        "dist/compiler_code_and_data_supplement_aaai27_v3.zip",
    ):
        add_file(
            ROOT / relative,
            relative,
            output=output,
            seen=seen,
        )

    for relative in PAPER_FILES:
        add_file(
            paper_dir / relative,
            f"paper/{relative}",
            output=output,
            seen=seen,
        )

    check_staging(output)
    manifest = build_manifest(output)
    (output / MANIFEST_NAME).write_text(
        json.dumps(
            manifest,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_deterministic_zip(output, zip_path)
    return {
        "package_id": PACKAGE_ID,
        "directory": str(output),
        "zip": str(zip_path),
        "zip_sha256": file_sha256(zip_path),
        "manifest_payload_sha256": manifest["payload_sha256"],
        "manifest_file_sha256": file_sha256(output / MANIFEST_NAME),
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--zip", dest="zip_path", type=Path, default=DEFAULT_ZIP)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build(
        paper_dir=args.paper_dir,
        output=args.output,
        zip_path=args.zip_path,
    )
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
