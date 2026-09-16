#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from run_synthetic_allocation_diagnostic import (
    Config,
    clip_vector,
    run,
    select_record_indices,
)


class SyntheticAllocationDiagnosticTests(unittest.TestCase):
    def test_clip_vector_bounds_norm_without_changing_small_vector(self) -> None:
        small = np.asarray([0.3, 0.4], dtype=np.float64)
        large = np.asarray([3.0, 4.0], dtype=np.float64)
        np.testing.assert_allclose(clip_vector(small, 1.0), small)
        self.assertAlmostEqual(float(np.linalg.norm(clip_vector(large, 1.0))), 1.0)

    def test_stratified_selector_returns_distinct_balanced_records(self) -> None:
        labels = np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
        rng = np.random.default_rng(260902)
        chosen = select_record_indices(rng, labels, m=4, strategy="stratified")
        self.assertEqual(len(chosen), 4)
        self.assertEqual(len(set(int(index) for index in chosen)), 4)
        self.assertEqual(set(int(label) for label in labels[chosen]), {0, 1, 2, 3})

    def test_quick_integration_controls_pass(self) -> None:
        config = Config(seeds=(260902, 260903), entities=8, trials_per_entity=64)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report"
            result = run(config, output)
            self.assertEqual(result["status"], "PASS")
            self.assertTrue((output / "config.json").is_file())
            self.assertTrue((output / "metrics.csv").is_file())
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue((output / "summary.md").is_file())


if __name__ == "__main__":
    unittest.main()
