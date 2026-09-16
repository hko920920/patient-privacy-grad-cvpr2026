"""Package-safe V12 external-intake compatibility tamper regression."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECORD = (
    PROJECT_ROOT
    / "reports"
    / "audit_external_extrasensory_v12_intake_001"
    / "portable_intake.json"
)
VERIFIER = PROJECT_ROOT / "scripts" / "verify_audit_external_intake_portable_v12.py"


class ExternalIntakeV12PortableCompatibilityTest(unittest.TestCase):
    def invoke(self, record: Path, output: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(VERIFIER),
                "--record",
                str(record),
                "--output-json",
                str(output),
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_positive_claim_forgery_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="audit_v12_intake_portable_", dir=PROJECT_ROOT / "reports"
        ) as temp_name:
            root = Path(temp_name)
            baseline = self.invoke(RECORD, root / "baseline.json")
            self.assertEqual(
                baseline.returncode,
                0,
                msg=f"baseline failed:\n{baseline.stdout}\n{baseline.stderr}",
            )
            payload = json.loads(RECORD.read_text(encoding="utf-8"))
            payload["portable_authority"]["release_status"] = "EVIDENCE_VALIDATED"
            payload["portable_authority"]["positive_wording_allowed"] = True
            payload["portable_authority"]["supported_statement"] = (
                "The external model is privacy validated."
            )
            tampered = root / "positive_forgery.json"
            tampered.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            rejected = self.invoke(tampered, root / "rejected.json")
            self.assertNotEqual(rejected.returncode, 0)
            result = json.loads((root / "rejected.json").read_text(encoding="utf-8"))
            self.assertFalse(result["passed"])
            self.assertTrue(result["early_rejection"])


if __name__ == "__main__":
    unittest.main()
