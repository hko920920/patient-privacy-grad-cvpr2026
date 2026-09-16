import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "reports"
    / "v35_handler_registry_gate_v2_20260725"
    / "handler_registry_gate_v35_v2.json"
)
VERIFICATION = (
    REPORT.parent / "handler_registry_verification_v35_v2.json"
)


class HandlerRegistryGateV35Test(unittest.TestCase):
    def test_frozen_report_and_verifier_pass(self):
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        verification = json.loads(
            VERIFICATION.read_text(encoding="utf-8")
        )
        payload = dict(report)
        reported = payload.pop("payload_sha256")
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        self.assertEqual(hashlib.sha256(encoded).hexdigest(), reported)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(verification["status"], "PASS")
        self.assertEqual(len(report["registry_rows"]), 3)
        self.assertEqual(len(report["compilation_rows"]), 9)
        self.assertEqual(
            len(report["conservative_extension_rows"]),
            6,
        )
        self.assertEqual(
            sum(
                row["rejected"]
                for row in report["negative_registration_rows"]
            ),
            8,
        )


if __name__ == "__main__":
    unittest.main()
