#!/usr/bin/env python3
"""Build the clean deterministic AAAI-27 Compiler V3.9 code/data ZIP."""

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
    from scripts.build_compiler_code_data_package_v4 import (
        TEST_EXCLUSIONS,
        V35_JSON_REPORT_DIRS,
        V35_SCRIPTS,
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
    from build_compiler_code_data_package_v4 import (
        TEST_EXCLUSIONS,
        V35_JSON_REPORT_DIRS,
        V35_SCRIPTS,
    )


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ID = "compiler_code_and_data_supplement_aaai27_v8"
DEFAULT_OUTPUT = ROOT / "dist" / PACKAGE_ID
DEFAULT_ZIP = ROOT / "dist" / f"{PACKAGE_ID}.zip"
MANIFEST_NAME = "PUBLIC_PACKAGE_MANIFEST.json"

V39_SCRIPTS = (
    "build_compiler_code_data_package_v8.py",
    "build_v39_format_gate.py",
    "build_g8_external_handler_evidence.py",
    "build_g8_external_handler_gate.py",
    "build_g8_external_handler_cost.py",
    "verify_g8_external_handler_gate.py",
    "verify_public_supplement_v39.py",
    "verify_v39_format_gate.py",
)

PAPER_FILES = (
    "aaai2027.bst",
    "aaai2027.sty",
    "main_aaai27_v39_candidate.tex",
    "main_aaai27_supplement_v38.tex",
    "main_aaai27_reproducibility_checklist_v38.tex",
    "main_aaai27_reproducibility_checklist_answers_v38.tex",
    "references_v28_candidate.bib",
    "references_v30_additions.bib",
    "references_v33_additions.bib",
    "references_v35_additions.bib",
    "references_v37_additions.bib",
    "tables_v38_candidate/v38_profiles.tex",
    "tables_v38_candidate/v38_handler_ablation.tex",
    "tables_v38_candidate/v38_conformance.tex",
)

ACTIVE_PUBLIC_FILES = (
    (
        "reports/g8_external_handler_evidence_v2_20260725/"
        "external_pld_handler_evidence_v36.json"
    ),
    (
        "reports/g8_external_handler_evidence_v2_20260725/"
        "uci_external_pld_execution_manifest_v36.json"
    ),
    (
        "reports/g8_external_handler_evidence_v2_20260725/"
        "uci_external_pld_execution_public_v36.json"
    ),
    (
        "reports/g8_external_handler_gate_v2_20260725/"
        "g8_external_handler_gate_v2.json"
    ),
    (
        "reports/g8_external_handler_gate_v2_20260725/"
        "g8_external_handler_verification_v2.json"
    ),
    (
        "reports/g8_external_handler_cost_v2_20260725/"
        "g8_external_handler_cost_v2.json"
    ),
    ("reports/v38_manuscript_gate_20260726/" "manuscript_gate_v38.json"),
    ("reports/v38_manuscript_gate_20260726/" "manuscript_gate_verification_v38.json"),
    ("reports/compiler_v39_format_gate_20260726/" "format_gate.json"),
    ("reports/compiler_v39_format_gate_20260726/" "format_gate_verification.json"),
)

V38_JSON_REPORT_DIRS = tuple(
    name for name in V35_JSON_REPORT_DIRS if name != "v35_manuscript_gate_20260725"
)


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_paper_dir(path: Path) -> Path:
    resolved = path.resolve()
    missing = [
        relative for relative in PAPER_FILES if not (resolved / relative).is_file()
    ]
    if missing:
        raise PackageBuildError("Paper directory is missing: " + ", ".join(missing))
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
        "schema_version": "unitdp.public_code_data_package.v39",
        "package_id": PACKAGE_ID,
        "visibility": "public",
        "release_status": "research_non_release",
        "raw_third_party_data_redistributed": False,
        "private_randomizers_or_step_traces_redistributed": False,
        "third_party_wheels_or_backend_tree_redistributed": False,
        "routes": ["P", "F", "A"],
        "registered_handler_endpoints": [
            "P",
            "F",
            "A",
            "H_external_pld_overlay_on_A",
        ],
        "external_backend_binding": {
            "primary": {
                "distribution": "PLD_accounting",
                "version": "0.5.0",
                "license": "MIT",
                "wheel_sha256": (
                    "1585d4030e4cda6209b6e9726ad81e46"
                    "c2fd7305ec3dd2631d28125059d59804"
                ),
                "tree_sha256": (
                    "50782379259744c44878da0c0164fa590"
                    "539a80c076c05366c10bb8c77ca90dc"
                ),
            },
            "declared_transitive": {
                "distribution": "random-allocation",
                "version": "1.0.5",
                "license": "MIT",
                "wheel_sha256": (
                    "236660af5ae93ae0659f54580ec3cfdfd"
                    "adb6d8e52a17c7f8a17a71326987c06"
                ),
                "tree_sha256": (
                    "757c29118555cc7046b9e007239d443a9"
                    "3544fb918c7ac6009112a6ecbd05179"
                ),
            },
        },
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
        raise PackageBuildError("Refusing to overwrite an existing package or ZIP")
    paper_dir = require_paper_dir(paper_dir)
    output.mkdir(parents=True)
    seen: set[str] = set()

    for name in (
        "README.md",
        "requirements-aaai27-v38.txt",
        "requirements-external-pld-v38.txt",
    ):
        add_file(
            ROOT / "submission" / "compiler_v38" / name,
            name,
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
    add_tree(
        ROOT / "v36_candidate" / "src",
        "v36_candidate/src",
        output=output,
        seen=seen,
        suffixes={".py"},
    )
    add_tree(
        ROOT / "v36_candidate" / "configs",
        "v36_candidate/configs",
        output=output,
        seen=seen,
        suffixes={".json", ".yaml", ".yml"},
    )
    add_tree(
        ROOT / "v36_candidate" / "tests",
        "v36_candidate/tests",
        output=output,
        seen=seen,
        suffixes={".py"},
    )
    add_file(
        ROOT / "v36_candidate" / "README.md",
        "v36_candidate/README.md",
        output=output,
        seen=seen,
    )

    for name in sorted(set(V3_SCRIPT_ALLOWLIST).union(V35_SCRIPTS).union(V39_SCRIPTS)):
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
        public_execution_tree(relative_root, output=output, seen=seen)
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
        add_file(ROOT / relative, relative, output=output, seen=seen)

    for directory in V38_JSON_REPORT_DIRS:
        add_tree(
            ROOT / "reports" / directory,
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

    for relative in ACTIVE_PUBLIC_FILES:
        add_file(ROOT / relative, relative, output=output, seen=seen)

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
            allow_nan=False,
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
