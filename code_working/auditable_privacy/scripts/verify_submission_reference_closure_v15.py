#!/usr/bin/env python3
"""Check that package-local file references resolve in an Audit submission ZIP.

The gate scans the package's top-level reviewer documentation and, when
provided, the main and supplementary TeX sources.  It treats executable
scripts, requirements files, and explicit package-local file paths as required
members.  Public URLs, raw-data acquisition paths, and generated output
directories are outside this membership check.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Callable


DEFAULT_DOCUMENTS = ("README.md", "DATA_AND_AUTHORITY.md", "NOTICE.md")

COMMAND_SCRIPT_RE = re.compile(
    r"\b(?:python(?:3)?|py)\s+((?:scripts|tests)/[A-Za-z0-9_./-]+\.py)\b",
    re.IGNORECASE,
)
REQUIREMENTS_RE = re.compile(
    r"(?:^|\s)-r\s+(requirements-[A-Za-z0-9_.-]+\.txt)\b",
    re.IGNORECASE,
)
LOCAL_FILE_RE = re.compile(
    r"(?<![A-Za-z0-9_./:-])("  # Do not harvest a suffix from a URL.
    r"(?:scripts|tests|docs|paper|reports|third_party)/"
    r"[A-Za-z0-9_./-]+\."
    r"(?:py|json|md|tex|csv|txt|yaml|yml|toml|zip)"
    r"|requirements-[A-Za-z0-9_.-]+\.txt"
    r"|README\.md|DATA_AND_AUTHORITY\.md|NOTICE\.md|MANIFEST\.json"
    r")",
    re.IGNORECASE,
)


def normalize_reference(value: str) -> str:
    """Return a safe package-relative POSIX file path."""

    normalized = value.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe package reference: {value}")
    return path.as_posix()


def extract_references(text: str) -> set[str]:
    """Extract command-critical and explicit package-local file references."""

    values: set[str] = set()
    for pattern in (COMMAND_SCRIPT_RE, REQUIREMENTS_RE, LOCAL_FILE_RE):
        for match in pattern.finditer(text):
            values.add(normalize_reference(match.group(1)))
    return values


def directory_package(
    root: Path,
) -> tuple[set[str], Callable[[str], str], str]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Package root does not exist: {root}")
    members = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }

    def read_text(name: str) -> str:
        return (root / name).read_text(encoding="utf-8")

    return members, read_text, f"directory:{root}"


def zip_package(
    path: Path,
) -> tuple[set[str], Callable[[str], str], str, zipfile.ZipFile]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Package ZIP does not exist: {path}")
    archive = zipfile.ZipFile(path, "r")
    members = {row.filename for row in archive.infolist() if not row.is_dir()}

    def read_text(name: str) -> str:
        return archive.read(name).decode("utf-8")

    return members, read_text, f"zip:{path}", archive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    package = parser.add_mutually_exclusive_group()
    package.add_argument("--package-root", type=Path)
    package.add_argument("--package-zip", type=Path)
    parser.add_argument("--main-tex", type=Path)
    parser.add_argument("--supplement-tex", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive: zipfile.ZipFile | None = None
    try:
        if args.package_zip is not None:
            members, read_member, package_label, archive = zip_package(
                args.package_zip
            )
        else:
            package_root = args.package_root
            if package_root is None:
                package_root = Path(__file__).resolve().parents[1]
            members, read_member, package_label = directory_package(package_root)

        missing_sources: list[str] = []
        sources: dict[str, str] = {}
        for name in DEFAULT_DOCUMENTS:
            if name not in members:
                missing_sources.append(name)
                continue
            sources[f"package:{name}"] = read_member(name)

        for label, path in (
            ("manuscript:main", args.main_tex),
            ("manuscript:supplement", args.supplement_tex),
        ):
            if path is None:
                continue
            resolved = path.resolve()
            if not resolved.is_file():
                missing_sources.append(str(resolved))
                continue
            sources[label] = resolved.read_text(encoding="utf-8")

        reference_sources: dict[str, set[str]] = defaultdict(set)
        unsafe_references: list[dict[str, str]] = []
        for source, text in sources.items():
            try:
                references = extract_references(text)
            except ValueError as exc:
                unsafe_references.append({"source": source, "error": str(exc)})
                continue
            for reference in references:
                reference_sources[reference].add(source)

        missing_references = [
            {
                "path": reference,
                "sources": sorted(reference_sources[reference]),
            }
            for reference in sorted(reference_sources)
            if reference not in members
        ]
        status = (
            "PASS"
            if not missing_sources
            and not missing_references
            and not unsafe_references
            else "FAIL"
        )
        payload = {
            "schema_version": "audit_submission_reference_closure_v15_v1.0",
            "status": status,
            "package": package_label,
            "package_members": len(members),
            "sources_scanned": sorted(sources),
            "references_checked": len(reference_sources),
            "references": [
                {
                    "path": reference,
                    "sources": sorted(reference_sources[reference]),
                }
                for reference in sorted(reference_sources)
            ],
            "missing_sources": sorted(missing_sources),
            "missing_references": missing_references,
            "unsafe_references": unsafe_references,
        }
        rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.output_json is not None:
            output = args.output_json.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered, encoding="utf-8", newline="\n")
        print(rendered, end="")
        return 0 if status == "PASS" else 1
    finally:
        if archive is not None:
            archive.close()


if __name__ == "__main__":
    raise SystemExit(main())
