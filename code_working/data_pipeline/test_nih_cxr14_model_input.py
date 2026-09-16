from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import nih_cxr14_model_input as model_input


def record(labels: tuple[str, ...]) -> model_input.CxrRecord:
    return model_input.CxrRecord(
        image_id="00000001_000.png",
        patient_id="1",
        partition="public_development",
        official_source_split="train_val",
        view="PA",
        finding_labels=labels,
        primary_groups=(),
        target_patient=0,
        cap_rank=1,
        cap_key_sha256="A" * 64,
        image_filename="00000001_000.png",
    )


class CxrModelInputTests(unittest.TestCase):
    def test_prompt_policy_is_fixed_and_non_diagnostic(self) -> None:
        self.assertEqual(
            model_input.prompt_for_record(record(("No Finding",))),
            "a frontal posteroanterior chest radiograph with no labeled finding",
        )
        prompt = model_input.prompt_for_record(
            record(("Pneumothorax", "Mass", "Effusion"))
        )
        self.assertEqual(
            prompt,
            "a frontal posteroanterior chest radiograph with radiographic findings of "
            "pleural effusion, mass opacity, and pneumothorax",
        )
        self.assertNotIn("cancer", prompt)

    def test_native_l_and_rgba_produce_identical_pixels(self) -> None:
        values = np.arange(1024 * 1024, dtype=np.uint32).reshape(1024, 1024)
        gray = (values % 256).astype(np.uint8)
        rgba = np.dstack(
            (gray, gray, gray, np.full_like(gray, fill_value=255, dtype=np.uint8))
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            l_path = root / "l.png"
            rgba_path = root / "rgba.png"
            Image.fromarray(gray).save(l_path)
            Image.fromarray(rgba).save(rgba_path)
            l_image, l_meta = model_input.preprocess_path(l_path, 256)
            rgba_image, rgba_meta = model_input.preprocess_path(rgba_path, 256)
        self.assertEqual(l_meta["native_mode"], "L")
        self.assertEqual(rgba_meta["native_mode"], "RGBA")
        np.testing.assert_array_equal(np.asarray(l_image), np.asarray(rgba_image))
        tensor = model_input.pil_to_normalized_tensor(rgba_image)
        self.assertEqual(tuple(tensor.shape), (3, 256, 256))
        self.assertTrue(bool((tensor[0] == tensor[1]).all()))
        self.assertTrue(bool((tensor[0] == tensor[2]).all()))

    def test_colored_or_transparent_rgba_fails_closed(self) -> None:
        base = np.zeros((1024, 1024, 4), dtype=np.uint8)
        base[:, :, 3] = 255
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            colored = base.copy()
            colored[0, 0, 1] = 1
            colored_path = root / "colored.png"
            Image.fromarray(colored).save(colored_path)
            with self.assertRaisesRegex(ValueError, "not grayscale-equivalent"):
                model_input.load_native_grayscale(colored_path)

            transparent = base.copy()
            transparent[0, 0, 3] = 254
            transparent_path = root / "transparent.png"
            Image.fromarray(transparent).save(transparent_path)
            with self.assertRaisesRegex(ValueError, "not fully opaque"):
                model_input.load_native_grayscale(transparent_path)

    def test_only_frozen_profile_sizes_are_permitted(self) -> None:
        with self.assertRaisesRegex(ValueError, "only 256 or 512"):
            model_input.preprocess_path("unused.png", 384)


if __name__ == "__main__":
    unittest.main()
