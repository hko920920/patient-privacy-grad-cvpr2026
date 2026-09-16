"""Tamper tests for the package-safe Audit V13 retrospective verifier."""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from verify_audit_retrospective_failures_v13 import verify  # noqa: E402


FROZEN_BUNDLE = PROJECT_ROOT / "reports" / "audit_retrospective_failures_v5_001"
DISCOVERY_SOURCES_PRESENT = all(
    (PROJECT_ROOT / relative).is_file()
    for relative in (
        "paper_notes/AAAI27_AUDIT_PAPER_SOURCE_AUDIT_2026-07-24.md",
        "paper_notes/AAAI27_AUDIT_FULL_REAUDIT_2026-07-24.md",
    )
)


def verify_frozen_or_package(bundle: Path) -> dict:
    return verify(
        bundle,
        skip_internal_discovery_sources=not DISCOVERY_SOURCES_PRESENT,
    )


class RetrospectiveFailureVerifierTests(unittest.TestCase):
    def test_frozen_bundle_passes(self) -> None:
        result = verify_frozen_or_package(FROZEN_BUNDLE)
        self.assertTrue(result["all_passed"])
        self.assertEqual(result["checks_passed"], result["checks_total"])

    def _temporary_bundle(self) -> tuple[tempfile.TemporaryDirectory, Path]:
        temporary = tempfile.TemporaryDirectory()
        destination = Path(temporary.name) / "bundle"
        shutil.copytree(FROZEN_BUNDLE, destination)
        return temporary, destination

    def test_missing_incident_is_rejected(self) -> None:
        temporary, bundle = self._temporary_bundle()
        self.addCleanup(temporary.cleanup)
        report_path = bundle / "retrospective_failure_lineage.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["incidents"] = report["incidents"][:-1]
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result = verify_frozen_or_package(bundle)
        self.assertFalse(result["all_passed"])
        self.assertFalse(result["checks"]["fixed_inclusion_ids"]["passed"])

    def test_forged_archive_identity_is_rejected(self) -> None:
        temporary, bundle = self._temporary_bundle()
        self.addCleanup(temporary.cleanup)
        report_path = bundle / "retrospective_failure_lineage.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        forged = copy.deepcopy(report)
        forged["incidents"][0]["historical_source"]["archive_sha256"] = "0" * 64
        report_path.write_text(
            json.dumps(forged, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result = verify_frozen_or_package(bundle)
        self.assertFalse(result["all_passed"])
        self.assertFalse(
            result["checks"]["R01.historical.archive_sha256"]["passed"]
        )


if __name__ == "__main__":
    unittest.main()
