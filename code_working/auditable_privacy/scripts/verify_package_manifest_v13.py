#!/usr/bin/env python3
"""Verify every immutable file in the anonymous Audit V13 manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "MANIFEST.json"
SUPPORTED_SCHEMAS = {
    "audit_anonymous_artifact_manifest_v13",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ignored(relative: str) -> bool:
    parts = Path(relative).parts
    return (
        relative == "MANIFEST.json"
        or "reproduced_verification" in parts
        or "__pycache__" in parts
        or relative.endswith(".pyc")
    )


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = manifest.get("files")
    if (
        manifest.get("schema_version") not in SUPPORTED_SCHEMAS
        or not isinstance(entries, list)
        or not entries
    ):
        raise SystemExit("Invalid MANIFEST.json")

    expected = {str(entry["path"]): entry for entry in entries}
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and not ignored(path.relative_to(ROOT).as_posix())
    }
    if actual != set(expected):
        raise SystemExit(
            "Manifest file set mismatch: "
            f"missing={sorted(set(expected) - actual)}, "
            f"extra={sorted(actual - set(expected))}"
        )

    for relative, entry in expected.items():
        path = ROOT / relative
        if (
            path.stat().st_size != int(entry["bytes"])
            or sha256_file(path) != str(entry["sha256"])
        ):
            raise SystemExit(f"Manifest mismatch: {relative}")

    print(f"manifest verification: PASS ({len(entries)}/{len(entries)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
