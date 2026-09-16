#!/usr/bin/env python3
"""Run the V15 reference-closure check with a public-safe package label.

The underlying reference extraction and closure semantics are unchanged.  This
release wrapper removes host-specific absolute paths from the machine-readable
report by recording only the package directory or ZIP basename.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable
import zipfile

import verify_submission_reference_closure_v15 as base


_directory_package = base.directory_package
_zip_package = base.zip_package


def directory_package(
    root: Path,
) -> tuple[set[str], Callable[[str], str], str]:
    members, read_text, _ = _directory_package(root)
    return members, read_text, f"directory:{root.resolve().name}"


def zip_package(
    path: Path,
) -> tuple[set[str], Callable[[str], str], str, zipfile.ZipFile]:
    members, read_text, _, archive = _zip_package(path)
    return members, read_text, f"zip:{path.resolve().name}", archive


def main() -> int:
    base.directory_package = directory_package
    base.zip_package = zip_package
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
