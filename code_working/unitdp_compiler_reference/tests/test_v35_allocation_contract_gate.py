import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "reports"
    / "v35_allocation_contract_gate_20260725"
    / "allocation_contract_gate_v35.json"
)
VERIFICATION = REPORT.parent / "allocation_contract_verification_v35.json"


class AllocationContractGateV35Test(unittest.TestCase):
    def test_report_and_independent_verifier_pass(self):
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
            report["semantic_mutation_gate"]["rejected"],
            26,
        )
        self.assertEqual(
            report["strict_boundary_gate"]["rejected"],
            5,
        )
        self.assertEqual(verification["checks_passed"], 8)


if __name__ == "__main__":
    unittest.main()
