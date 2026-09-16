#!/usr/bin/env python3
"""CPU-only tests for the bounded private research schedule."""

from __future__ import annotations

import copy
import unittest
from dataclasses import dataclass

from dp_training.run_xray_k5_private_research_dryrun import (
    build_runtime_schedules,
    derive_seed,
)


@dataclass(frozen=True)
class Record:
    image_id: str
    patient_id: str
    cap_rank: int


def records() -> list[Record]:
    return [
        Record(image_id=f"i{patient:02d}_{rank}", patient_id=f"p{patient:02d}", cap_rank=rank)
        for patient in range(10)
        for rank in (1, 2)
    ]


def protocol() -> dict:
    return {
        "runtime": {"steps_per_arm": 4},
        "mechanisms": {
            "M1-I8": {"poisson_sample_rate": 0.25},
            "M2-P8": {"poisson_sample_rate": 0.20},
        },
        "resource_fail_closed": {
            "maximum_realized_image_units_per_m1_step": 20,
            "maximum_realized_patient_units_per_m2_step": 10,
            "maximum_realized_raw_images_per_m2_step": 20,
        },
    }


def schedule_projection(schedule):
    return [
        [
            (
                unit["unit_type"],
                unit["unit_id"],
                tuple(record.image_id for record in unit["records"]),
                tuple((draw["timestep"], draw["noise_seed"]) for draw in unit["draws"]),
            )
            for unit in step
        ]
        for step in schedule
    ]


class TestPrivateResearchDryrun(unittest.TestCase):
    def test_seed_derivation_is_deterministic_and_domain_separated(self):
        secret = bytes(range(32))
        self.assertEqual(derive_seed(secret, "a"), derive_seed(secret, "a"))
        self.assertNotEqual(derive_seed(secret, "a"), derive_seed(secret, "b"))
        self.assertGreaterEqual(derive_seed(secret, "a"), 0)
        self.assertLess(derive_seed(secret, "a"), 2**63)

    def test_schedule_replays_for_test_secret(self):
        secret = bytes(range(32))
        first = build_runtime_schedules(records(), protocol(), secret, 1000)
        second = build_runtime_schedules(records(), protocol(), secret, 1000)
        self.assertEqual(schedule_projection(first[0]), schedule_projection(second[0]))
        self.assertEqual(schedule_projection(first[1]), schedule_projection(second[1]))
        self.assertEqual(first[2], second[2])

    def test_unit_boundaries_and_no_duplicates(self):
        image, patient, _ = build_runtime_schedules(records(), protocol(), b"x" * 32, 1000)
        for step in image:
            ids = [unit["unit_id"] for unit in step]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertTrue(all(len(unit["records"]) == 1 for unit in step))
        for step in patient:
            ids = [unit["unit_id"] for unit in step]
            self.assertEqual(len(ids), len(set(ids)))
            for unit in step:
                self.assertGreaterEqual(len(unit["records"]), 1)
                self.assertLessEqual(len(unit["records"]), 4)
                self.assertEqual({record.patient_id for record in unit["records"]}, {unit["unit_id"]})

    def test_resource_limit_fails_without_resampling(self):
        constrained = copy.deepcopy(protocol())
        constrained["mechanisms"]["M1-I8"]["poisson_sample_rate"] = 1.0
        constrained["resource_fail_closed"]["maximum_realized_image_units_per_m1_step"] = 19
        with self.assertRaisesRegex(RuntimeError, "do not resample"):
            build_runtime_schedules(records(), constrained, b"z" * 32, 1000)

    def test_invalid_master_secret_fails(self):
        with self.assertRaisesRegex(RuntimeError, "256 bits"):
            derive_seed(b"short", "label")


if __name__ == "__main__":
    unittest.main()
