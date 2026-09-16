from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from dp_training.k5_evaluator import (
    deterministic_kid,
    effective_rank,
    kid_unbiased,
    load_real_p256_grayscale,
    pairwise_euclidean,
    prdc,
    stratified_bootstrap_mean,
)


class TestK5Evaluator(unittest.TestCase):
    def test_kid_matches_direct_formula(self) -> None:
        x = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([[2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
        value = kid_unbiased(x, y)
        kernel = lambda a, b: (a @ b.T / 2.0 + 1.0) ** 3
        kxx, kyy, kxy = kernel(x, x), kernel(y, y), kernel(x, y)
        expected = (kxx.sum() - np.trace(kxx)) / 6 + (kyy.sum() - np.trace(kyy)) / 6 - 2 * kxy.mean()
        self.assertAlmostEqual(value, float(expected), places=14)

    def test_deterministic_kid_replays(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(size=(30, 8))
        y = rng.normal(size=(30, 8))
        first = deterministic_kid(x, y, subset_size=12, replicates=20, seed=9)
        second = deterministic_kid(x, y, subset_size=12, replicates=20, seed=9)
        self.assertEqual(first, second)

    def test_pairwise_and_prdc_are_finite(self) -> None:
        real = np.arange(48, dtype=np.float64).reshape(12, 4) / 10
        fake = real + 0.03
        distances = pairwise_euclidean(real, fake)
        self.assertEqual(distances.shape, (12, 12))
        values = prdc(real, fake, nearest_k=2)
        self.assertEqual(set(values), {"precision", "recall", "density", "coverage"})
        self.assertTrue(all(np.isfinite(value) for value in values.values()))

    def test_effective_rank_identity_axes(self) -> None:
        features = np.vstack([np.eye(4), -np.eye(4)])
        self.assertAlmostEqual(effective_rank(features), 4.0, places=12)

    def test_stratified_bootstrap_is_deterministic(self) -> None:
        differences = [1.0, 2.0, 3.0, 4.0, 8.0, 9.0]
        strata = ["a", "a", "a", "a", "b", "b"]
        first = stratified_bootstrap_mean(differences, strata, seed=17, replicates=100)
        second = stratified_bootstrap_mean(differences, strata, seed=17, replicates=100)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean_difference"], 4.5)

    def test_real_preprocessing_rejects_rgb_and_resizes_l(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.png"
            invalid = root / "invalid.png"
            Image.fromarray(np.full((1024, 1024), 123, dtype=np.uint8)).save(valid)
            Image.fromarray(np.zeros((1024, 1024, 3), dtype=np.uint8)).save(invalid)
            output = load_real_p256_grayscale(valid)
            self.assertEqual(output.mode, "L")
            self.assertEqual(output.size, (256, 256))
            with self.assertRaises(RuntimeError):
                load_real_p256_grayscale(invalid)


if __name__ == "__main__":
    unittest.main()
