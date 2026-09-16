from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

import run_nih_cxr14_patient_set_premise_confirmation as confirmation


class PatientSetConfirmationTests(unittest.TestCase):
    def test_select_units_is_balanced_exact_count_and_exclusion_safe(self) -> None:
        records = []
        excluded = set()
        for target in confirmation.TARGETS:
            for count in confirmation.RECORD_COUNTS:
                for patient_index in range(42):
                    patient_id = f"t{target}-n{count}-p{patient_index:02d}"
                    if patient_index == 0:
                        excluded.add(patient_id)
                    for record_index in range(count):
                        records.append(
                            SimpleNamespace(
                                patient_id=patient_id,
                                target_patient=target,
                                image_id=f"{patient_id}_{record_index:03d}.png",
                            )
                        )
        units = confirmation.select_units(records, excluded)
        self.assertEqual(len(units), confirmation.TOTAL_PATIENTS)
        self.assertEqual(sum(unit["record_count"] for unit in units), confirmation.TOTAL_IMAGES)
        self.assertTrue({unit["patient_id"] for unit in units}.isdisjoint(excluded))
        for target in confirmation.TARGETS:
            for count in confirmation.RECORD_COUNTS:
                selected = [
                    unit
                    for unit in units
                    if unit["target_patient"] == target and unit["record_count"] == count
                ]
                self.assertEqual(len(selected), confirmation.PATIENTS_PER_CELL)

    def test_select_units_is_deterministic_under_input_reversal(self) -> None:
        records = []
        for target in confirmation.TARGETS:
            for count in confirmation.RECORD_COUNTS:
                for patient_index in range(41):
                    patient_id = f"t{target}-n{count}-p{patient_index:02d}"
                    for record_index in range(count):
                        records.append(
                            SimpleNamespace(
                                patient_id=patient_id,
                                target_patient=target,
                                image_id=f"{patient_id}_{record_index:03d}.png",
                            )
                        )
        forward = confirmation.select_units(records, set())
        backward = confirmation.select_units(list(reversed(records)), set())
        self.assertEqual(
            [unit["patient_id"] for unit in forward],
            [unit["patient_id"] for unit in backward],
        )

    def test_balanced_auc_is_one_for_separated_scores(self) -> None:
        similarity = np.eye(4, dtype=np.float64)
        similarity[0, 1] = similarity[1, 0] = 0.9
        similarity[2, 3] = similarity[3, 2] = 0.8
        for left in (0, 1):
            for right in (2, 3):
                similarity[left, right] = similarity[right, left] = 0.1
        records = [SimpleNamespace(image_id=f"image-{index}") for index in range(4)]
        auc, selected = confirmation.balanced_auc(
            similarity,
            [(0, 1), (2, 3)],
            [(0, 2), (0, 3), (1, 2), (1, 3)],
            records,
            "test",
        )
        self.assertEqual(auc, 1.0)
        self.assertEqual(len(selected), 2)

    def test_retrieval_reports_exact_chance(self) -> None:
        similarity = np.eye(8, dtype=np.float64)
        patient_ids = []
        for patient in range(4):
            patient_ids.extend([str(patient), str(patient)])
            left = 2 * patient
            right = left + 1
            similarity[left, right] = similarity[right, left] = 1.0
        metrics = confirmation.retrieval_metrics(similarity, list(range(8)), patient_ids)
        self.assertEqual(metrics["recall_at_1"], 1.0)
        self.assertAlmostEqual(metrics["random_candidate_recall_at_1"], 1.0 / 7.0)
        self.assertAlmostEqual(metrics["recall_at_1_chance_multiple"], 7.0)


if __name__ == "__main__":
    unittest.main()
