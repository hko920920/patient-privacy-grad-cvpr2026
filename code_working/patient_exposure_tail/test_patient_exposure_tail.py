#!/usr/bin/env python3
"""Pure tests for the patient-exposure-tail diagnostic."""

from __future__ import annotations

import unittest

import numpy as np

import run_nih_cxr14_patient_exposure_tail_premise as target


def record(patient: str, image: str, followup: int, sex: str = "F", flag: int = 0) -> target.Record:
    return target.Record(
        image_id=image,
        patient_id=patient,
        partition="public_development",
        view="PA",
        followup_no=followup,
        patient_sex=sex,
        target_patient=flag,
        finding_labels="No Finding",
        image_filename=f"{image}.png",
    )


class PatientExposureTailTests(unittest.TestCase):
    def test_record_count_buckets(self) -> None:
        self.assertEqual(target.record_count_bucket(2), "2")
        self.assertEqual(target.record_count_bucket(3), "3")
        self.assertEqual(target.record_count_bucket(4), "4-5")
        self.assertEqual(target.record_count_bucket(5), "4-5")
        self.assertEqual(target.record_count_bucket(10), "6-10")

    def test_pair_uses_first_and_last_ordered_record(self) -> None:
        old_expected = (
            target.EXPECTED_PUBLIC_IMAGES,
            target.EXPECTED_PUBLIC_PATIENTS,
            target.EXPECTED_ELIGIBLE_PATIENTS,
            target.EXPECTED_RECORD_COUNTS,
        )
        target.EXPECTED_PUBLIC_IMAGES = 3
        target.EXPECTED_PUBLIC_PATIENTS = 1
        target.EXPECTED_ELIGIBLE_PATIENTS = 1
        target.EXPECTED_RECORD_COUNTS = {3: 1}
        try:
            units = target.select_units(
                [record("p", "middle", 3), record("p", "last", 7), record("p", "first", 1)]
            )
        finally:
            (
                target.EXPECTED_PUBLIC_IMAGES,
                target.EXPECTED_PUBLIC_PATIENTS,
                target.EXPECTED_ELIGIBLE_PATIENTS,
                target.EXPECTED_RECORD_COUNTS,
            ) = old_expected
        self.assertEqual(units[0]["query"].image_id, "first")
        self.assertEqual(units[0]["gallery"].image_id, "last")

    def test_derangement_preserves_registered_cell(self) -> None:
        units = []
        for index in range(32):
            sex = "F" if (index // 16) == 0 else "M"
            flag = (index // 8) % 2
            bucket = ("2", "3", "4-5", "6-10")[(index // 2) % 4]
            count = {"2": 2, "3": 3, "4-5": 4, "6-10": 6}[bucket]
            units.append(
                {
                    "patient_id": f"p{index}",
                    "patient_sex": sex,
                    "target_patient": flag,
                    "record_count": count,
                    "record_count_bucket": bucket,
                }
            )
        mapping = target.matched_negative_indices(units)
        self.assertEqual(len(set(mapping)), len(units))
        self.assertTrue(all(left != right for left, right in enumerate(mapping)))
        self.assertTrue(
            all(target.subgroup_key(units[left]) == target.subgroup_key(units[right]) for left, right in enumerate(mapping))
        )

    def test_rank_auc_handles_separation_and_ties(self) -> None:
        self.assertEqual(target.rank_auc(np.array([0.8, 0.9]), np.array([0.1, 0.2])), 1.0)
        self.assertEqual(target.rank_auc(np.array([0.5]), np.array([0.5])), 0.5)

    def test_spearman_ordering(self) -> None:
        values = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
        self.assertAlmostEqual(target.spearman(values, values * 2), 1.0)
        self.assertAlmostEqual(target.spearman(values, -values), -1.0)

    def test_worst_tie_rank(self) -> None:
        units = [
            {"patient_id": "a", "patient_sex": "F", "target_patient": 0, "record_count_bucket": "2"},
            {"patient_id": "b", "patient_sex": "F", "target_patient": 0, "record_count_bucket": "2"},
        ]
        query = np.array([[1.0, 0.0], [0.0, 1.0]])
        gallery = np.array([[1.0, 0.0], [1.0, 0.0]])
        old_replicates = target.BOOTSTRAP_REPLICATES
        target.BOOTSTRAP_REPLICATES = 20
        try:
            result, margins = target.encoder_metrics(query, gallery, [1, 0], units, "synthetic")
        finally:
            target.BOOTSTRAP_REPLICATES = old_replicates
        self.assertEqual(result["recall_at_1"], 0.0)
        self.assertEqual(result["median_rank"], 2.0)
        self.assertEqual(margins[0], 0.0)

    def test_pass_decision_is_conjunctive(self) -> None:
        result = {
            "true_vs_matched_negative_auc": 0.9,
            "recall_at_1": 0.3,
            "rank_greater_than_10_fraction": 0.2,
            "margin_quantiles": {"p90": 0.01, "p10": -0.01},
            "bootstrap_95_percentile_intervals": {
                "auc": [0.85, 0.95],
                "recall_at_1": [0.2, 0.4],
            },
        }
        cross = {"margin_spearman": 0.4, "bootstrap_95_percentile_interval": [0.3, 0.5]}
        common, per_encoder = target.decision_from_results(
            {"DINOv2": result, "RAD-DINO": dict(result)}, cross, 1_000
        )
        self.assertTrue(all(common.values()))
        self.assertTrue(all(value for checks in per_encoder.values() for value in checks.values()))
        result["recall_at_1"] = 0.1
        _, failed = target.decision_from_results(
            {"DINOv2": result, "RAD-DINO": dict(result)}, cross, 1_000
        )
        self.assertFalse(failed["DINOv2"]["recall_at_1_at_least_0_15"])


if __name__ == "__main__":
    unittest.main()
