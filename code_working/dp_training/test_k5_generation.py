from __future__ import annotations

import tempfile
import unittest
import importlib.util
from pathlib import Path

import numpy as np

from dp_training import k5_generation as generation


class TestK5Generation(unittest.TestCase):
    def test_frozen_task_csv_is_fully_reconstructed(self) -> None:
        path = generation.default_paths()["generation_tasks"]
        rows = generation.load_and_validate_tasks(path)
        self.assertEqual(len(rows), 448)
        self.assertEqual(rows[0]["task_id"], "no_finding_000")
        self.assertEqual(rows[-1]["task_id"], "nodule_opacity_063")
        self.assertEqual(rows[0]["seed"], 6478382064505572947)
        self.assertEqual([len(batch) for batch in generation.expected_batches(rows)], [4] * 112)

    def test_png_encoding_is_deterministic_and_exact_rgb_p256(self) -> None:
        rng = np.random.default_rng(9)
        value = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
        first = generation.encode_rgb_png(value)
        second = generation.encode_rgb_png(value.copy())
        self.assertEqual(first, second)
        result = generation.analyze_png_bytes(first)
        self.assertEqual(result["mode"], "RGB")
        self.assertEqual((result["width"], result["height"]), (256, 256))
        self.assertFalse(result["low_contrast"])
        self.assertFalse(result["saturated"])

    def test_pixel_sanity_boundaries_are_strictly_frozen(self) -> None:
        black = np.zeros((256, 256, 3), dtype=np.uint8)
        result = generation.analyze_rgb_array(black)
        self.assertTrue(result["low_contrast"])
        self.assertTrue(result["saturated"])
        self.assertEqual(result["grayscale_std"], 0.0)
        self.assertEqual(result["extreme_pixel_fraction"], 1.0)

        # Exactly 98% extreme pixels is not saturated; the definition says more than 98%.
        gray = np.full((256, 256, 3), 128, dtype=np.uint8)
        extreme_count = int(0.98 * 256 * 256)
        gray.reshape(-1, 3)[:extreme_count] = 0
        result = generation.analyze_rgb_array(gray)
        self.assertLessEqual(result["extreme_pixel_fraction"], 0.98)
        self.assertFalse(result["saturated"])

    def test_atomic_png_refuses_overwrite(self) -> None:
        body = generation.encode_rgb_png(np.full((256, 256, 3), 127, dtype=np.uint8))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            generation.write_bytes_atomic(path, body)
            self.assertEqual(path.read_bytes(), body)
            with self.assertRaises(RuntimeError):
                generation.write_bytes_atomic(path, body)

    def test_record_summary_duplicate_fraction_uses_surplus(self) -> None:
        records = []
        for index in range(448):
            records.append(
                {
                    "task_index": index,
                    "condition": generation.CONDITIONS[index // 64][0],
                    "sha256": f"{index:064X}",
                    "low_contrast": False,
                    "saturated": False,
                }
            )
        records[-1]["sha256"] = records[0]["sha256"]
        result = generation.summarize_records(records)
        self.assertEqual(result["exact_duplicate_surplus_count"], 1)
        self.assertAlmostEqual(result["exact_duplicate_fraction"], 1 / 448)
        self.assertEqual(result["status"], "PASS")

    def test_quantization_is_truncation_not_rounding(self) -> None:
        value = np.zeros((256, 256, 3), dtype=np.float32)
        value[0, 0] = [0.5, 1.1, -0.1]
        result = generation.quantize_pipeline_image(value)
        self.assertEqual(result[0, 0].tolist(), [127, 255, 0])

    def test_broken_optional_onnx_is_masked_without_masking_other_modules(self) -> None:
        original = importlib.util.find_spec
        try:
            generation.disable_broken_optional_onnx()
            self.assertIsNone(importlib.util.find_spec("onnxruntime"))
            self.assertIsNotNone(importlib.util.find_spec("json"))
            generation.disable_broken_optional_onnx()
            self.assertIsNone(importlib.util.find_spec("onnxruntime.capi"))
        finally:
            importlib.util.find_spec = original


if __name__ == "__main__":
    unittest.main()
