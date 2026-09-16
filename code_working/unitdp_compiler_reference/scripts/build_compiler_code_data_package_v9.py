#!/usr/bin/env python3
"""Build the V9 package with a V3.9-consistent reviewer README/verifier."""

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
PACKAGE_ID = "compiler_code_and_data_supplement_aaai27_v9"
DEFAULT_OUTPUT = ROOT / "dist" / PACKAGE_ID
DEFAULT_ZIP = ROOT / "dist" / f"{PACKAGE_ID}.zip"
REVIEWER_README = ROOT / "submission/compiler_v39/README.md"


def configure_core() -> None:
    core.PACKAGE_ID = PACKAGE_ID
    core.V39_SCRIPTS = (
        "build_compiler_code_data_package_v8.py",
        "build_compiler_code_data_package_v9.py",
        "build_v39_format_gate.py",
        "build_g8_external_handler_evidence.py",
        "build_g8_external_handler_gate.py",
        "build_g8_external_handler_cost.py",
        "verify_g8_external_handler_gate.py",
        "verify_public_supplement_v39.py",
        "verify_public_supplement_v40.py",
        "verify_v39_format_gate.py",
    )
    original_add_file = core.add_file

    def redirected_add_file(
        source: Path,
        destination: str,
        *,
        output: Path,
        seen: set[str],
    ) -> None:
        if destination == "README.md":
            source = REVIEWER_README
        original_add_file(source, destination, output=output, seen=seen)

    core.add_file = redirected_add_file


def build(*, paper_dir: Path, output: Path, zip_path: Path) -> dict[str, Any]:
    configure_core()
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
