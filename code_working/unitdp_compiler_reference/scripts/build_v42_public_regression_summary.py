#!/usr/bin/env python3
"""Build the anonymous public summary of the frozen local regression run."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_STDOUT = (
    ROOT
    / "reports/v35_handler_hybrid_ablation_gate_20260725/"
    "full_pytest_stdout_v35.txt"
)
SOURCE_STDERR = SOURCE_STDOUT.with_name("full_pytest_stderr_v35.txt")
OUTPUT = (
    ROOT
    / "reports/v42_public_regression_summary_20260728/"
    "public_regression_summary_v42.json"
)

EXPECTED_STDOUT_SHA256 = (
    "1eb680358fd34177764b4d7ac9508c4776d58de64f5dc8a6c351bd098aabdc53"
)
EXPECTED_STDERR_SHA256 = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_report() -> dict[str, Any]:
    if file_sha256(SOURCE_STDOUT) != EXPECTED_STDOUT_SHA256:
        raise RuntimeError("frozen regression stdout drifted")
    if file_sha256(SOURCE_STDERR) != EXPECTED_STDERR_SHA256:
        raise RuntimeError("frozen regression stderr drifted")

    stdout = SOURCE_STDOUT.read_bytes().decode("utf-16")
    summary = re.search(
        r"(?P<passed>\d+) passed, (?P<skipped>\d+) skipped, "
        r"(?P<warnings>\d+) warnings in (?P<seconds>[0-9.]+)s",
        stdout,
    )
    if summary is None:
        raise RuntimeError("cannot parse frozen regression summary")
    counts = {key: int(summary.group(key)) for key in ("passed", "skipped", "warnings")}
    if counts != {"passed": 154, "skipped": 7, "warnings": 2}:
        raise RuntimeError(f"unexpected regression counts: {counts}")

    report: dict[str, Any] = {
        "schema_version": "unitdp.compiler_v42_public_regression_summary.v1",
        "status": "PASS",
        "authority": "frozen local base-suite regression transcript",
        "command": "python -m pytest -q",
        "counts": {
            **counts,
            "collected": counts["passed"] + counts["skipped"],
        },
        "warnings": [
            {
                "package": "numexpr",
                "installed": "2.8.4",
                "minimum_requested_by_pandas": "2.10.2",
            },
            {
                "package": "bottleneck",
                "installed": "1.3.5",
                "minimum_requested_by_pandas": "1.4.2",
            },
        ],
        "source_binding": {
            "stdout_sha256": EXPECTED_STDOUT_SHA256,
            "stdout_bytes": SOURCE_STDOUT.stat().st_size,
            "stdout_encoding": "UTF-16",
            "stderr_sha256": EXPECTED_STDERR_SHA256,
            "stderr_bytes": SOURCE_STDERR.stat().st_size,
            "raw_transcripts_redistributed": False,
            "reason": (
                "the frozen stdout contains workstation-specific absolute paths; "
                "this path-neutral summary exposes counts and warning versions while "
                "binding the original transcript by SHA-256"
            ),
        },
        "scope": (
            "local frozen-base regression conformance only; distinct from the "
            "base-plus-H 173/180 surface and the portable clean-extraction suite"
        ),
    }
    report["payload_sha256"] = canonical_sha256(report)
    return report


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("refusing to overwrite public regression summary")
    report = build_report()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(OUTPUT),
                "payload_sha256": report["payload_sha256"],
                "file_sha256": file_sha256(OUTPUT),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
