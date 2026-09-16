from __future__ import annotations

import unittest
from collections import Counter
from types import SimpleNamespace

import run_nih_cxr14_setadapter_context_signal as context_signal


def make_records(patients_per_target: int = 50):
    records = []
    for target in (0, 1):
        for patient_index in range(patients_per_target):
            patient_id = f"t{target}-p{patient_index:03d}"
            for record_index in range(2):
                records.append(
                    SimpleNamespace(
                        patient_id=patient_id,
                        target_patient=target,
                        image_id=f"{patient_id}_{record_index:03d}.png",
                        finding_labels=("No Finding",) if target == 0 else ("Mass",),
                    )
                )
    return records


class ContextSignalTests(unittest.TestCase):
    def test_selection_is_balanced_disjoint_and_deterministic(self) -> None:
        records = make_records()
        forward = context_signal.select_cohort(records, set())
        backward = context_signal.select_cohort(list(reversed(records)), set())
        train, validation, eligible = forward
        self.assertEqual(len(train), 64)
        self.assertEqual(len(validation), 32)
        self.assertEqual(eligible, {"0": 50, "1": 50})
        self.assertEqual(
            [unit["patient_id"] for unit in train + validation],
            [unit["patient_id"] for unit in backward[0] + backward[1]],
        )
        self.assertTrue(
            {unit["patient_id"] for unit in train}.isdisjoint(
                {unit["patient_id"] for unit in validation}
            )
        )

    def test_training_schedule_has_frozen_length_balance_and_exposure(self) -> None:
        train, _, _ = context_signal.select_cohort(make_records(), set())
        schedule = context_signal.training_schedule(train)
        self.assertEqual(len(schedule), context_signal.TRAIN_STEPS)
        self.assertTrue(all(pair[0]["target_patient"] == 0 for pair in schedule))
        self.assertTrue(all(pair[1]["target_patient"] == 1 for pair in schedule))
        exposure = Counter(unit["patient_id"] for pair in schedule for unit in pair)
        self.assertEqual(set(exposure.values()), {context_signal.TRAIN_EXPOSURES_PER_PATIENT})

    def test_shuffled_companion_is_other_patient_and_exact_label_when_available(self) -> None:
        _, validation, _ = context_signal.select_cohort(make_records(), set())
        unit = validation[0]
        companion = unit["records"][1]
        shuffled, exact, similarity = context_signal.choose_shuffled_companion(
            unit, companion, validation, bank=0
        )
        self.assertNotEqual(str(shuffled.patient_id), unit["patient_id"])
        self.assertTrue(exact)
        self.assertEqual(similarity, 1.0)
        self.assertEqual(
            context_signal.canonical_labels(shuffled),
            context_signal.canonical_labels(companion),
        )

    def test_label_jaccard_fallback(self) -> None:
        left = SimpleNamespace(finding_labels=("Mass", "Nodule"))
        right = SimpleNamespace(finding_labels=("Mass",))
        self.assertAlmostEqual(context_signal.label_jaccard(left, right), 0.5)


if __name__ == "__main__":
    unittest.main()
