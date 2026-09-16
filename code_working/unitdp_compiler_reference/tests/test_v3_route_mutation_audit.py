import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from verify_v3_route_mutation_audit import (  # noqa: E402
    payload_sha256,
    verify,
)


REPORT = (
    ROOT
    / "reports"
    / "v3_route_mutation_audit_v32_20260724"
    / "public_route_mutation_audit_v3.json"
)


class RouteMutationAuditV3Test(unittest.TestCase):
    def test_frozen_report_verifies(self):
        result = verify(REPORT)
        self.assertTrue(result["verified"])
        self.assertEqual(result["case_count"], 27)

    def test_forged_outcome_is_rejected_even_with_rehashed_payload(self):
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        report["cases"][0]["observed_outcome"] = "accept"
        report["observed_rejections"] = 26
        report["all_required_rejections_observed"] = False
        report.pop("payload_sha256")
        report["payload_sha256"] = payload_sha256(report)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "forged.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(AssertionError):
                verify(path)

    def test_forged_source_hash_is_rejected_with_rehashed_payload(self):
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        first = next(iter(report["source_files_sha256"]))
        report["source_files_sha256"][first] = "0" * 64
        report.pop("payload_sha256")
        report["payload_sha256"] = payload_sha256(report)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "forged.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(AssertionError):
                verify(path)
