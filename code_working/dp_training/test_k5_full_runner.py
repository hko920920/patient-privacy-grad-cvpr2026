from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from dp_training import k5_full_runner as core


def records() -> list[SimpleNamespace]:
    values = []
    for patient in range(8):
        for image in range(2):
            values.append(
                SimpleNamespace(
                    image_id=f"image-{patient:02d}-{image}",
                    patient_id=f"patient-{patient:02d}",
                    cap_rank=image + 1,
                    partition="public_development",
                )
            )
    return values


class K5FullRunnerScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = core.load_frozen_protocol()
        cls.records = records()
        cls.root = bytes(range(32))

    def test_exact_frozen_dp_configs(self) -> None:
        i8 = core.mechanism_config(self.protocol, "M1-I8")
        g8 = core.mechanism_config(self.protocol, "M1-G8")
        p8 = core.mechanism_config(self.protocol, "M2-P8")
        self.assertEqual(i8.population, 18_393)
        self.assertEqual(i8.expected_batch, 8)
        self.assertEqual(i8.clip_norm, g8.clip_norm)
        self.assertLess(i8.noise_multiplier, g8.noise_multiplier)
        self.assertEqual(p8.privacy_unit, "patient")
        self.assertEqual(p8.population, 8_476)
        self.assertEqual(p8.expected_batch, 4)

    def test_m1_schedule_and_draws_are_identical(self) -> None:
        i8, i8_meta, _ = core.build_arm_schedule(
            self.records, self.protocol, "M1-I8", self.root, 1_000, steps=4, fixture=True
        )
        g8, g8_meta, _ = core.build_arm_schedule(
            self.records, self.protocol, "M1-G8", self.root, 1_000, steps=4, fixture=True
        )
        self.assertEqual(i8_meta["commitment"], g8_meta["commitment"])
        self.assertEqual(core._schedule_secret_view(i8), core._schedule_secret_view(g8))

    def test_m0_has_four_fixed_eight_image_batches(self) -> None:
        schedule, meta, config = core.build_arm_schedule(
            self.records, self.protocol, "M0", self.root, 1_000, steps=4, fixture=True
        )
        self.assertIsNone(config)
        self.assertEqual(meta["scheduled_raw_images"], 32)
        self.assertTrue(all(len(step) == 1 for step in schedule))
        self.assertTrue(all(len(step[0]["records"]) == 8 for step in schedule))

    def test_m2_never_crosses_patient_and_means_two_fixture_images(self) -> None:
        schedule, _, config = core.build_arm_schedule(
            self.records, self.protocol, "M2-P8", self.root, 1_000, steps=4, fixture=True
        )
        self.assertEqual(config.privacy_unit, "patient")
        for step in schedule:
            for unit in step:
                self.assertEqual(len(unit["records"]), 2)
                self.assertEqual({item.patient_id for item in unit["records"]}, {unit["patient_id"]})

    def test_commitment_is_keyed_by_experiment_root(self) -> None:
        first, first_meta, _ = core.build_arm_schedule(
            self.records, self.protocol, "M0", self.root, 1_000, steps=4, fixture=True
        )
        second_root = bytes(reversed(range(32)))
        second, second_meta, _ = core.build_arm_schedule(
            self.records, self.protocol, "M0", second_root, 1_000, steps=4, fixture=True
        )
        self.assertNotEqual(first_meta["commitment"], second_meta["commitment"])
        self.assertNotEqual(core._schedule_secret_view(first), core._schedule_secret_view(second))

    def test_atomic_json_rejects_stale_temporary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            path.with_name("value.json.new").write_text("audit", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "stale"):
                core.write_json_atomic(path, {"ok": True})


if __name__ == "__main__":
    unittest.main()
