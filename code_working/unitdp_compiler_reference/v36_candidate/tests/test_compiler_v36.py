import hashlib
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
CANDIDATE_MODULES = ROOT / "v36_candidate" / "src" / "unitdp"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import unitdp  # noqa: E402

if str(CANDIDATE_MODULES) not in unitdp.__path__:
    unitdp.__path__.append(str(CANDIDATE_MODULES))

from unitdp.compiler_v36 import (  # noqa: E402
    EXTERNAL_PLD_HANDLER_V36,
    ROUTE_DISPATCH_REGISTRY_V36,
    load_registered_contract_v36,
)
from unitdp.compiler_v4 import (  # noqa: E402
    REQUIRED_REGISTRATION_OBLIGATIONS_V4,
    CompilerDispatchV4Error,
    validate_route_handler_v4,
)

EVIDENCE = (
    ROOT / "reports/g8_external_handler_evidence_v2_20260725/"
    "external_pld_handler_evidence_v36.json"
)
GATE = (
    ROOT / "reports/g8_external_handler_gate_v2_20260725/"
    "g8_external_handler_gate_v2.json"
)
COST = (
    ROOT / "reports/g8_external_handler_cost_v2_20260725/"
    "g8_external_handler_cost_v2.json"
)
VERIFICATION = (
    ROOT / "reports/g8_external_handler_gate_v2_20260725/"
    "g8_external_handler_verification_v2.json"
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _load_hashed(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    reported = value.pop("payload_sha256")
    if _canonical_sha256(value) != reported:
        raise AssertionError(f"invalid report payload: {path}")
    value["payload_sha256"] = reported
    return value


class CompilerV36Test(unittest.TestCase):
    def test_four_handler_registry_and_H_evidence_core(self):
        self.assertEqual(len(ROUTE_DISPATCH_REGISTRY_V36), 4)
        validate_route_handler_v4(EXTERNAL_PLD_HANDLER_V36)
        evidence = _load_hashed(EVIDENCE)
        self.assertEqual(evidence["status"], "PASS")
        self.assertEqual(
            EXTERNAL_PLD_HANDLER_V36.registration_core_sha256,
            evidence["registration_core_sha256"],
        )
        self.assertEqual(
            {
                witness.obligation_id
                for witness in EXTERNAL_PLD_HANDLER_V36.obligation_witnesses
            },
            REQUIRED_REGISTRATION_OBLIGATIONS_V4,
        )

    def test_every_H_witness_is_mandatory(self):
        witnesses = EXTERNAL_PLD_HANDLER_V36.obligation_witnesses
        for removed in witnesses:
            with self.subTest(removed=removed.obligation_id):
                candidate = replace(
                    EXTERNAL_PLD_HANDLER_V36,
                    obligation_witnesses=tuple(
                        witness
                        for witness in witnesses
                        if witness.obligation_id != removed.obligation_id
                    ),
                )
                with self.assertRaises(CompilerDispatchV4Error):
                    validate_route_handler_v4(candidate)

    def test_generic_loader_accepts_H_without_route_branch(self):
        contract = load_registered_contract_v36(
            ROOT / "v36_candidate/configs/uci_owner_external_pld_v36.yaml"
        )
        self.assertEqual(
            type(contract).__name__,
            "OwnerExternalPldAllocationContractV36",
        )
        self.assertEqual(
            contract.base_contract.public_contract_sha256,
            "038d26e2fcbff104f7227bda65b4adc5b0a6de444a2bf066f59743632b0e12a2",
        )

    def test_G8_reports_and_independent_verification_pass(self):
        gate = _load_hashed(GATE)
        cost = _load_hashed(COST)
        verification = _load_hashed(VERIFICATION)
        self.assertEqual(gate["status"], "PASS")
        self.assertEqual(gate["preservation"]["exact"], 9)
        self.assertEqual(
            gate["handler_hybrids"]["sealed_rejected"],
            756,
        )
        self.assertEqual(cost["status"], "PASS")
        self.assertEqual(
            cost["source_cost"]["existing_production_files_changed"],
            0,
        )
        self.assertEqual(verification["status"], "PASS")
        self.assertEqual(
            verification["passed"],
            verification["required"],
        )


if __name__ == "__main__":
    unittest.main()
