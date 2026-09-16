"""Tamper regression for the separate Audit V13 witness verifier."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    PROJECT_ROOT
    / "reports"
    / "audit_composition_witnesses_v12_003"
    / "witnesses.json"
)
VERIFIER = (
    PROJECT_ROOT
    / "scripts"
    / "verify_audit_composition_witnesses_v13.py"
)


class CompositionWitnessVerifierTest(unittest.TestCase):
    def invoke(self, report: Path, output: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(VERIFIER),
                "--report",
                str(report),
                "--output-json",
                str(output),
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_recomputed_summary_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="audit_v13_witness_tamper_",
            dir=PROJECT_ROOT / "reports",
        ) as temp_name:
            temp_root = Path(temp_name)
            baseline = self.invoke(REPORT, temp_root / "baseline.json")
            self.assertEqual(
                baseline.returncode,
                0,
                msg=f"baseline verification failed:\n{baseline.stdout}\n{baseline.stderr}",
            )

            payload = json.loads(REPORT.read_text(encoding="utf-8"))
            payload["summary"]["witnesses_passed"] = 4
            payload["summary"]["all_passed"] = True
            tampered_report = temp_root / "witnesses_tampered.json"
            tampered_report.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            tampered = self.invoke(
                tampered_report,
                temp_root / "tampered_verification.json",
            )
            self.assertNotEqual(
                tampered.returncode,
                0,
                msg="verifier accepted a recomputed-outer summary-count forgery",
            )


if __name__ == "__main__":
    unittest.main()
