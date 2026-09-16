"""Regression test for exact supported-wording authentication.

The WISDM verifier is intentionally production-import-free.  This test
changes only the public event sentence to the valid window/base sentence and
requires the independent verifier to reject the otherwise unchanged bundle.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERIFIER = PROJECT_ROOT / "scripts" / "verify_wisdm_v2_gold.py"
SOURCE_RUN = (
    PROJECT_ROOT
    / "reports"
    / "wisdm_v4_multirun_001"
    / "confirmatory_run01"
)
SCENARIO = "wisdm_v2_hardened_overlap50"


class ExactWordingTamperTest(unittest.TestCase):
    def run_verifier(self, evidence_dir: Path, output: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(VERIFIER),
                "--evidence-dir",
                str(evidence_dir),
                "--skip-input-rescan",
                "--output-json",
                str(output),
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_event_sentence_cannot_be_replaced_by_window_sentence(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="audit_v12_wording_",
            dir=PROJECT_ROOT / "reports",
        ) as temp_name:
            temp_root = Path(temp_name)
            copied_run = temp_root / "run"
            shutil.copytree(SOURCE_RUN, copied_run)

            baseline = self.run_verifier(
                copied_run,
                temp_root / "baseline_verification.json",
            )
            self.assertEqual(
                baseline.returncode,
                0,
                msg=f"baseline verifier failed:\n{baseline.stdout}\n{baseline.stderr}",
            )

            certificate_dir = copied_run / "certificates"
            window_path = (
                certificate_dir / f"certificate_{SCENARIO}_window.json"
            )
            event_path = (
                certificate_dir / f"certificate_{SCENARIO}_event.json"
            )
            window = json.loads(window_path.read_text(encoding="utf-8"))
            event = json.loads(event_path.read_text(encoding="utf-8"))
            event["supported_statement"] = window["supported_statement"]
            event_path.write_text(
                json.dumps(event, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            tampered = self.run_verifier(
                copied_run,
                temp_root / "tampered_verification.json",
            )
            self.assertNotEqual(
                tampered.returncode,
                0,
                msg=(
                    "independent verifier accepted an event certificate whose "
                    "sentence was replaced by the valid window/base sentence"
                ),
            )


if __name__ == "__main__":
    unittest.main()
