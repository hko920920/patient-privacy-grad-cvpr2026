#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from run_isic_sd21_gradient_smoke import (
    farthest_first_strata,
    gram_metrics,
    weights_coverage_stratified,
    weights_noise_only,
    weights_uniform_distinct,
)


class IsicSd21GradientSmokeTests(unittest.TestCase):
    def test_gram_metrics_matches_explicit_vectors(self) -> None:
        rng = np.random.default_rng(260902)
        gradients = rng.normal(size=(6, 11))
        gram = gradients @ gradients.T
        weights = np.asarray([0.5, 0.0, 0.25, 0.25, 0.0, 0.0])
        reference = np.full(6, 1.0 / 6.0)
        observed = gram_metrics(weights, reference, gram, 11, clip_norm=0.7)

        estimate = weights @ gradients
        target = reference @ gradients
        explicit_preclip = float(np.mean(np.square(estimate - target)))
        estimate_scale = min(1.0, 0.7 / float(np.linalg.norm(estimate)))
        target_scale = min(1.0, 0.7 / float(np.linalg.norm(target)))
        explicit_postclip = float(
            np.mean(np.square(estimate_scale * estimate - target_scale * target))
        )
        self.assertAlmostEqual(observed["preclip_mse"], explicit_preclip, places=12)
        self.assertAlmostEqual(observed["postclip_mse"], explicit_postclip, places=12)

    def test_farthest_first_has_no_empty_strata(self) -> None:
        features = np.asarray(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
        )
        labels, anchors = farthest_first_strata(features, strata=4)
        self.assertEqual(len(set(int(value) for value in labels)), 4)
        self.assertEqual(len(set(int(value) for value in anchors)), 4)

    def test_all_estimator_weights_sum_to_one(self) -> None:
        rng = np.random.default_rng(260902)
        labels = np.asarray([0, 1, 2, 3, 0])
        for _ in range(100):
            candidates = (
                weights_noise_only(rng, records=5, perturbations=4, q=4),
                weights_uniform_distinct(rng, records=5, perturbations=4, q=4),
                weights_coverage_stratified(rng, labels, perturbations=4),
            )
            for weights in candidates:
                self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=15)


if __name__ == "__main__":
    unittest.main()
