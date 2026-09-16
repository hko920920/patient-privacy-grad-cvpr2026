import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for source in (SCRIPTS, SRC):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from build_v35_handler_hybrid_ablation_gate import (  # noqa: E402
    hybrid,
    payload_sha256,
    preseal_validator_accepts,
)
from unitdp.compiler_v4 import (  # noqa: E402
    POISSON_HANDLER_V4,
    SRSWOR_HANDLER_V4,
    CompilerDispatchV4Error,
    validate_route_handler_v4,
)
from verify_v35_handler_hybrid_ablation_gate import (  # noqa: E402
    DEFAULT_REPORT,
    HandlerHybridVerificationError,
    verify_report,
)


class HandlerHybridAblationGateV35Test(unittest.TestCase):
    def test_exact_report_rebuilds_all_hybrids(self):
        result = verify_report(DEFAULT_REPORT)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["hybrids_rebuilt"], 378)
        self.assertEqual(result["sealed_rejections"], 378)

    def test_preseal_survivor_is_rejected_by_complete_core_seal(self):
        survivor = None
        for mask in range(1, 127):
            candidate = hybrid(
                POISSON_HANDLER_V4,
                SRSWOR_HANDLER_V4,
                mask,
            )
            if preseal_validator_accepts(candidate):
                survivor = candidate
                break
        self.assertIsNotNone(survivor)
        with self.assertRaisesRegex(
            CompilerDispatchV4Error,
            "complete handler core",
        ):
            validate_route_handler_v4(survivor)

    def test_rehashed_outer_count_forgery_is_rejected(self):
        original = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
        forged = copy.deepcopy(original)
        forged.pop("payload_sha256")
        forged["totals"]["preseal_accepted"] = 41
        forged["payload_sha256"] = payload_sha256(forged)
        with tempfile.TemporaryDirectory(
            prefix="unitdp_v35_handler_gate_"
        ) as temporary:
            path = Path(temporary) / "forged.json"
            path.write_text(
                json.dumps(forged, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                HandlerHybridVerificationError,
                "exact totals mismatch",
            ):
                verify_report(path)


if __name__ == "__main__":
    unittest.main()
