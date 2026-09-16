#!/usr/bin/env python3
"""CPU-only tests for the preregistered multi-patient diagnostic."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np

from run_isic_sd21_multipatient_diagnostic import (
    BOOTSTRAP_REPLICATES,
    PATIENTS_PER_SITE_GROUP,
    bootstrap_ratio,
    enumerate_stratified_weights,
    enumerate_uniform_weights,
    exact_estimator_metrics,
    feature_gradient_alignment,
    select_balanced_patients,
    spearman_correlation,
    variance_decomposition,
)


@dataclass(frozen=True)
class DummyRecord:
    patient_id: str
    lesion_id: str
    anatom_site: str
    cap_rank: int


class MultiPatientDiagnosticTests(unittest.TestCase):
    def test_balanced_selection_is_deterministic_and_eligible(self) -> None:
        records: list[DummyRecord] = []
        for group in ("low", "high"):
            for patient_number in range(PATIENTS_PER_SITE_GROUP + 2):
                patient_id = f"{group}_{patient_number:02d}"
                for record_index in range(5):
                    site = (
                        "torso"
                        if group == "low"
                        else ("head", "torso", "leg")[record_index % 3]
                    )
                    records.append(
                        DummyRecord(
                            patient_id=patient_id,
                            lesion_id=f"{patient_id}_lesion_{record_index}",
                            anatom_site=site,
                            cap_rank=record_index,
                        )
                    )

        first, evidence = select_balanced_patients(records)
        second, _ = select_balanced_patients(reversed(records))
        self.assertEqual(
            [row["patient_id"] for row in first],
            [row["patient_id"] for row in second],
        )
        self.assertEqual(evidence["selected_patients"], 16)
        self.assertEqual(sum(row["site_group"] == "low_site" for row in first), 8)
        self.assertEqual(sum(row["site_group"] == "high_site" for row in first), 8)
        self.assertTrue(all(len(row["records"]) == 5 for row in first))

    def test_exact_weight_enumerations_are_unbiased(self) -> None:
        methods = {
            "uniform_m1_r4": enumerate_uniform_weights(5, 4, 1, 4),
            "uniform_m2_r2": enumerate_uniform_weights(5, 4, 2, 2),
            "uniform_m4_r1": enumerate_uniform_weights(5, 4, 4, 1),
            "stratified_m2_r2": enumerate_stratified_weights(
                np.asarray([0, 0, 0, 1, 1]), 4, 2
            ),
            "stratified_m4_r1": enumerate_stratified_weights(
                np.asarray([0, 1, 2, 3, 3]), 4, 1
            ),
        }
        reference = np.full(20, 1.0 / 20.0)
        for name, weights in methods.items():
            with self.subTest(method=name):
                np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-14)
                np.testing.assert_allclose(weights.mean(axis=0), reference, atol=1e-14)

    def test_exact_metric_matches_explicit_gradient_vectors(self) -> None:
        rng = np.random.default_rng(7)
        gradients = rng.normal(size=(20, 7))
        gram = gradients @ gradients.T
        weights = enumerate_uniform_weights(5, 4, 4, 1)
        primary, _ = exact_estimator_metrics(weights, gram, 7, (0.5,))
        reference = gradients.mean(axis=0)
        estimates = weights @ gradients
        brute_mse = np.mean(np.mean((estimates - reference) ** 2, axis=1))
        self.assertAlmostEqual(primary["preclip_mse"], float(brute_mse), places=12)

    def test_variance_decomposition_identity(self) -> None:
        rng = np.random.default_rng(11)
        gradients = rng.normal(size=(20, 13))
        result = variance_decomposition(gradients @ gradients.T, 5, 4, 13)
        self.assertAlmostEqual(
            result["total_variance"],
            result["between_record_variance"]
            + result["within_perturbation_variance"],
            places=12,
        )
        self.assertLess(result["decomposition_residual"], 1e-12)

    def test_feature_gradient_alignment_and_tie_aware_spearman(self) -> None:
        features = np.arange(5, dtype=np.float64)[:, None]
        record_gradients = np.arange(5, dtype=np.float64)[:, None]
        gradients = np.repeat(record_gradients, 4, axis=0)
        result = feature_gradient_alignment(features, gradients @ gradients.T, 4)
        self.assertAlmostEqual(
            result["feature_gradient_distance_spearman"], 1.0, places=12
        )
        self.assertAlmostEqual(
            spearman_correlation(
                np.asarray([1.0, 1.0, 2.0, 3.0]),
                np.asarray([4.0, 4.0, 8.0, 9.0]),
            ),
            1.0,
            places=12,
        )

    def test_bootstrap_is_deterministic(self) -> None:
        ratios = np.asarray([0.8, 0.9, 1.0, 1.1] * 4)
        first = bootstrap_ratio(ratios)
        second = bootstrap_ratio(ratios)
        self.assertEqual(first, second)
        self.assertEqual(first["bootstrap_replicates"], BOOTSTRAP_REPLICATES)
        self.assertLessEqual(first["bootstrap_ci95_lower"], first["geometric_mean_ratio"])
        self.assertGreaterEqual(first["bootstrap_ci95_upper"], first["geometric_mean_ratio"])


if __name__ == "__main__":
    unittest.main()
