#!/usr/bin/env python3
"""Build the deterministic V4.3-matched Compiler V10 code/data package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts import build_compiler_code_data_package_v8 as core
except (ImportError, ModuleNotFoundError):
    import build_compiler_code_data_package_v8 as core


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ID = "compiler_code_and_data_supplement_aaai27_v10"
DEFAULT_OUTPUT = ROOT / "dist" / PACKAGE_ID
DEFAULT_ZIP = ROOT / "dist" / f"{PACKAGE_ID}.zip"
REVIEWER_README = ROOT / "submission/compiler_v42/README.md"
PUBLIC_REGRESSION = (
    ROOT
    / "reports/v42_public_regression_summary_20260728/"
    "public_regression_summary_v42.json"
)

PAPER_FILES = (
    "aaai2027.bst",
    "aaai2027.sty",
    "main_aaai27_v43_candidate.tex",
    "main_aaai27_supplement_v41.tex",
    "main_aaai27_reproducibility_checklist_v40.tex",
    "main_aaai27_reproducibility_checklist_answers_v40.tex",
    "references_v28_candidate.bib",
    "references_v30_additions.bib",
    "references_v33_additions.bib",
    "references_v35_additions.bib",
    "references_v37_additions.bib",
    "tables_v38_candidate/v38_profiles.tex",
    "figures/figure7.png",
)

EXPECTED_ACTIVE_HASHES = {
    "main_source_sha256": "8a621473ceeaee9a8b365607b49282d6837aa37605a07be81403541de2764bd9",
    "main_pdf_sha256": "07b6b301b76e7efb4b38a8fff23bf58f579e146dc689519af781c234730feac8",
    "supplement_source_sha256": "b0db21a8cdbf74723a849c04d5cdd4e3f2f05d00764296c33fbb3029fab879ad",
    "supplement_pdf_sha256": "1638ca903bd1b816b41a8b99632c35ac11e57defe7f1c68a0c6eb9a42a94050c",
    "checklist_wrapper_sha256": "5bd1d615400587a272525a295c0667638f93331041e40e6eae692e14ac1962f8",
    "checklist_answers_sha256": "33443c9cab2b71907015292db9690aa6b29bb9cae6818f0dfcc88a2f39ee3beb",
    "checklist_pdf_sha256": "747d0a64203895f06db7b3abc74b00a5490c633f6e4cae063abe8992c2628f94",
    "figure7_sha256": "9c1cf1b573607d2140d7ae2e3788a9d603b48baac279f0cb09fb361accea471c",
    "public_regression_summary_sha256": "59846e98cb370e399b537f48af99573e86137db5cc9ba8f6665e5ab6875bcac3",
}


core.PACKAGE_ID = PACKAGE_ID
core.PAPER_FILES = PAPER_FILES
core.V39_SCRIPTS = tuple(
    sorted(
        set(core.V39_SCRIPTS)
        | {
            "build_compiler_code_data_package_v9.py",
            "build_compiler_code_data_package_v10.py",
            "build_v42_public_regression_summary.py",
            "verify_public_supplement_v40.py",
            "verify_public_supplement_v41.py",
        }
    )
)
core.ACTIVE_PUBLIC_FILES = tuple(
    dict.fromkeys(
        (*core.ACTIVE_PUBLIC_FILES, PUBLIC_REGRESSION.relative_to(ROOT).as_posix())
    )
)

_base_add_file = core.add_file
_base_build_manifest = core.build_manifest


def redirected_add_file(
    source: Path,
    destination: str,
    *,
    output: Path,
    seen: set[str],
) -> None:
    if destination == "README.md":
        source = REVIEWER_README if REVIEWER_README.is_file() else ROOT / "README.md"
    elif destination in {
        "requirements-aaai27-v38.txt",
        "requirements-external-pld-v38.txt",
    }:
        extracted_source = ROOT / destination
        if extracted_source.is_file():
            source = extracted_source
    _base_add_file(source, destination, output=output, seen=seen)


def current_build_manifest(output: Path) -> dict[str, Any]:
    manifest = _base_build_manifest(output)
    observed = {
        "main_source_sha256": core.file_sha256(
            output / "paper/main_aaai27_v43_candidate.tex"
        ),
        "supplement_source_sha256": core.file_sha256(
            output / "paper/main_aaai27_supplement_v41.tex"
        ),
        "checklist_wrapper_sha256": core.file_sha256(
            output / "paper/main_aaai27_reproducibility_checklist_v40.tex"
        ),
        "checklist_answers_sha256": core.file_sha256(
            output
            / "paper/main_aaai27_reproducibility_checklist_answers_v40.tex"
        ),
        "figure7_sha256": core.file_sha256(output / "paper/figures/figure7.png"),
        "public_regression_summary_sha256": core.file_sha256(
            output
            / "reports/v42_public_regression_summary_20260728/"
            "public_regression_summary_v42.json"
        ),
    }
    for key, value in observed.items():
        if value != EXPECTED_ACTIVE_HASHES[key]:
            raise core.PackageBuildError(f"active artifact drifted: {key}")

    manifest["schema_version"] = "unitdp.public_code_data_package.v43"
    manifest["package_id"] = PACKAGE_ID
    manifest["active_artifacts"] = {
        **EXPECTED_ACTIVE_HASHES,
        "main_pdf_redistributed_in_code_archive": False,
        "supplement_pdf_redistributed_in_code_archive": False,
        "checklist_pdf_redistributed_in_code_archive": False,
    }
    manifest["evidence_commands"] = {
        "portable_package_verification": (
            "python scripts/verify_public_supplement_v41.py"
        ),
        "portable_regression": "python -B -m pytest -q -p no:cacheprovider",
        "deterministic_rebuild": (
            "python scripts/build_compiler_code_data_package_v10.py "
            "--paper-dir paper --output "
            "_rebuild/compiler_code_and_data_supplement_aaai27_v10 --zip "
            "_rebuild/compiler_code_and_data_supplement_aaai27_v10.zip"
        ),
    }
    manifest.pop("payload_sha256", None)
    manifest["payload_sha256"] = core.canonical_sha256(manifest)
    return manifest


core.add_file = redirected_add_file
core.build_manifest = current_build_manifest


def build(*, paper_dir: Path, output: Path, zip_path: Path) -> dict[str, Any]:
    return core.build(paper_dir=paper_dir, output=output, zip_path=zip_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--zip", dest="zip_path", type=Path, default=DEFAULT_ZIP)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                paper_dir=args.paper_dir,
                output=args.output,
                zip_path=args.zip_path,
            ),
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
