#!/usr/bin/env python3
"""Fast contract tests for the frozen NIH CXR14 DP/attack gate."""

from __future__ import annotations

import csv
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_DIR = ROOT / "_reports" / "nih_cxr14_dp_attack_protocol_v1_001"
CLIP_DIR = ROOT / "_reports" / "nih_cxr14_public_clip_calibration_v1_001"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class TestXrayDpAttackProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_json(PROTOCOL_DIR / "protocol.json")
        cls.report = load_json(PROTOCOL_DIR / "report.json")
        cls.clip = load_json(CLIP_DIR / "report.json")

    def test_gate_is_frozen_but_training_and_release_are_blocked(self):
        self.assertEqual(self.protocol["status"], "FROZEN_PRETRAINING_PROTOCOL")
        self.assertEqual(self.report["status"], "PASS_PROTOCOL_FREEZE_TRAINING_STILL_BLOCKED")
        self.assertEqual(self.protocol["release_gate"]["current_status"], "BLOCKED_BEFORE_TRAINING")
        self.assertTrue(self.protocol["randomness_policy"]["release_runtime"].startswith("BLOCKED_"))

    def test_accounting_targets_and_k10_unit_gap(self):
        entries = self.protocol["accounting"]["entries"]
        self.assertEqual(len(entries), 9)
        for entry in entries:
            self.assertLessEqual(entry["opacus_epsilon"], entry["target_epsilon"] + 1e-9)
            self.assertLessEqual(entry["google_dp_accounting_epsilon"], entry["target_epsilon"] + 1e-9)
        standard = next(item for item in entries if item["arm"] == "M1-I8" and item["cap"] == 10)
        matched = next(item for item in entries if item["arm"] == "M1-G8" and item["cap"] == 10)
        direct = next(item for item in entries if item["arm"] == "M2-P8")
        self.assertTrue(standard["group_conversion"]["vacuous"])
        self.assertFalse(matched["group_conversion"]["vacuous"])
        self.assertAlmostEqual(matched["group_conversion"]["patient_epsilon"], 8.0)
        self.assertAlmostEqual(matched["group_conversion"]["patient_delta"], 1e-5)
        self.assertGreater(matched["noise_multiplier"], direct["noise_multiplier"] * 2.0)

    def test_compute_match_is_within_one_percent(self):
        match = self.protocol["mechanisms"]["compute_match"]
        self.assertEqual(match["m1_expected_images_per_step"], 8)
        self.assertAlmostEqual(match["m2_expected_images_per_step"], 8.023124115148654)
        self.assertLess(abs(match["relative_image_work_m2_over_m1"] - 1.0), 0.01)

    def test_mia_cohorts_are_balanced_and_disjoint(self):
        with (PROTOCOL_DIR / "mia_members_private.csv").open("r", encoding="utf-8", newline="") as handle:
            members = list(csv.DictReader(handle))
        with (PROTOCOL_DIR / "mia_nonmembers_private.csv").open("r", encoding="utf-8", newline="") as handle:
            nonmembers = list(csv.DictReader(handle))
        self.assertEqual(len(members), 1816)
        self.assertEqual(len(nonmembers), 1816)
        self.assertEqual(sum(row["target_patient"] == "1" for row in members), 908)
        self.assertEqual(sum(row["target_patient"] == "1" for row in nonmembers), 908)
        self.assertTrue({row["patient_id"] for row in members}.isdisjoint({row["patient_id"] for row in nonmembers}))
        self.assertEqual(
            sorted((row["target_patient"], row["image_count_k10"]) for row in members),
            sorted((row["target_patient"], row["image_count_k10"]) for row in nonmembers),
        )

    def test_public_clip_is_exact_higher_p80_and_fp32(self):
        with (CLIP_DIR / "gradient_norms_private.csv").open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for unit_type, expected_count, expected_above in (("image", 72, 14), ("patient", 72, 14)):
            values = sorted(float(row["gradient_l2_norm"]) for row in rows if row["unit_type"] == unit_type)
            self.assertEqual(len(values), expected_count)
            selected = values[math.ceil((len(values) - 1) * 0.8)]
            self.assertEqual(selected, self.clip["selected_clip_norms"][unit_type])
            self.assertEqual(sum(value > selected for value in values), expected_above)
        self.assertEqual(self.clip["frozen_inputs"]["trainable_dtypes_observed"], ["torch.float32"])
        self.assertTrue(self.clip["parameters_unchanged"])
        self.assertFalse(self.clip["optimizer_created"])
        self.assertFalse(self.clip["optimizer_step"])


if __name__ == "__main__":
    unittest.main()
