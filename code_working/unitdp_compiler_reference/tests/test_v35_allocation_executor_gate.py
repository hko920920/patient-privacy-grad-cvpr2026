import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "reports"
    / "v35_allocation_executor_gate_20260725"
    / "allocation_executor_gate_v35.json"
)
VERIFICATION = REPORT.parent / "allocation_executor_verification_v35.json"


class AllocationExecutorGateV35Test(unittest.TestCase):
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
        self.assertEqual(
            sum(report["correspondence"].values()),
            7,
        )
        self.assertEqual(
            sum(report["execution_invariants"].values()),
            10,
        )
        self.assertEqual(
            verification["checks_passed"],
            verification["checks_total"],
        )


if __name__ == "__main__":
    unittest.main()
