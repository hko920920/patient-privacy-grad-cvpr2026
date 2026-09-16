import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "reports"
    / "v35_pairwise_hybrid_lattice_gate_v2_20260725"
    / "pairwise_hybrid_lattice_gate_v35_v2.json"
)
VERIFICATION = (
    REPORT.parent / "pairwise_hybrid_lattice_verification_v35_v2.json"
)


class PairwiseHybridLatticeGateV35Test(unittest.TestCase):
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
        self.assertEqual(report["partition"]["surface_count"], 8)
        self.assertEqual(len(report["pair_dataset_rows"]), 9)
        self.assertEqual(
            report["summary"]["rejected_nonendpoint_hybrids"],
            2286,
        )
        self.assertEqual(
            report["summary"]["accepted_nonendpoint_hybrids"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
