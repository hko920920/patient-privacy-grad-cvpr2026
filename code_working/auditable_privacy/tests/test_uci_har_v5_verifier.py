"""Tamper tests for the independent Audit v5 UCI HAR verifier."""

from __future__ import annotations

import csv
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

from verify_uci_har_v5 import verify  # noqa: E402


FROZEN_PILOT = PROJECT_ROOT / "reports" / "uci_har_v5_pilot_002"
DATA_ROOT = PROJECT_ROOT / "data" / "uci_har" / "extracted" / "UCI HAR Dataset"


class UciHarV5VerifierTests(unittest.TestCase):
    def test_frozen_pilot_portable_passes(self) -> None:
        result = verify(FROZEN_PILOT, DATA_ROOT, skip_input_rescan=True)
        self.assertTrue(result["all_passed"])
        self.assertEqual(result["checks_passed"], result["checks_total"])

    def _temporary_bundle(self) -> tuple[tempfile.TemporaryDirectory, Path]:
        temporary = tempfile.TemporaryDirectory()
        destination = Path(temporary.name) / "bundle"
        shutil.copytree(FROZEN_PILOT, destination)
        return temporary, destination

    def test_model_tamper_is_rejected(self) -> None:
        temporary, bundle = self._temporary_bundle()
        self.addCleanup(temporary.cleanup)
        model_path = bundle / "released_model.json"
        model = json.loads(model_path.read_text(encoding="utf-8"))
        model["weights_including_bias"][0][0] += 1.0
        model_path.write_text(
            json.dumps(model, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result = verify(bundle, DATA_ROOT, skip_input_rescan=True)
        self.assertFalse(result["all_passed"])
        self.assertFalse(
            result["checks"]["execution_artifact_digests"]["passed"]
        )

    def test_mapping_tamper_is_rejected(self) -> None:
        temporary, bundle = self._temporary_bundle()
        self.addCleanup(temporary.cleanup)
        mapping_path = bundle / "mapping.csv"
        with mapping_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
            fieldnames = list(rows[0])
        rows[0]["end"] = "2"
        with mapping_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        result = verify(bundle, DATA_ROOT, skip_input_rescan=True)
        self.assertFalse(result["all_passed"])
        self.assertFalse(result["checks"]["mapping_file_digest"]["passed"])

    def test_unsupported_event_positive_is_rejected(self) -> None:
        temporary, bundle = self._temporary_bundle()
        self.addCleanup(temporary.cleanup)
        certificate_path = (
            bundle
            / "certificates"
            / "certificate_uci_har_v5_generated_window_event.json"
        )
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        certificate["release_status"] = "ALLOWED"
        certificate["supported_statement"] = "forged"
        certificate_path.write_text(
            json.dumps(certificate, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result = verify(bundle, DATA_ROOT, skip_input_rescan=True)
        self.assertFalse(result["all_passed"])
        self.assertFalse(result["checks"]["event_fail_closed"]["passed"])


if __name__ == "__main__":
    unittest.main()
