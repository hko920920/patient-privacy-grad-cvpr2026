from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from dp_training.secure_resume import (
    load_envelope,
    protect_bytes,
    save_envelope_atomic,
    serialize_payload,
    unprotect_bytes,
)


class TestSecureResume(unittest.TestCase):
    def payload(self, step: int, marker: bytes = b"private-marker") -> dict:
        return {
            "schema": "resume-test/v1",
            "arm": "M1-I8",
            "committed_step": step,
            "experiment_root": marker,
            "tensor": torch.arange(9, dtype=torch.float32),
            "nested": {"state": [torch.tensor([step]), "ok"]},
        }

    def test_dpapi_bytes_round_trip(self) -> None:
        plaintext = serialize_payload(self.payload(2))
        ciphertext = protect_bytes(plaintext)
        self.assertNotEqual(ciphertext, plaintext)
        self.assertEqual(unprotect_bytes(ciphertext), plaintext)

    def test_atomic_envelope_round_trip_has_no_plaintext_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "resume.dpapi"
            metadata = save_envelope_atomic(current, self.payload(2))
            self.assertFalse(metadata["plaintext_file_created"])
            self.assertNotIn(b"private-marker", current.read_bytes())
            loaded = load_envelope(current)
            self.assertEqual(loaded["experiment_root"], b"private-marker")
            torch.testing.assert_close(loaded["tensor"], torch.arange(9, dtype=torch.float32))

    def test_rotation_retains_one_previous(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "resume.dpapi"
            save_envelope_atomic(current, self.payload(1))
            first_ciphertext = current.read_bytes()
            metadata = save_envelope_atomic(current, self.payload(2))
            previous = Path(directory) / "resume.previous.dpapi"
            self.assertTrue(metadata["previous_retained"])
            self.assertEqual(previous.read_bytes(), first_ciphertext)
            self.assertEqual(load_envelope(previous)["committed_step"], 1)
            self.assertEqual(load_envelope(current)["committed_step"], 2)

    def test_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "resume.dpapi"
            save_envelope_atomic(current, self.payload(2))
            value = bytearray(current.read_bytes())
            value[len(value) // 2] ^= 0x01
            current.write_bytes(bytes(value))
            with self.assertRaises(OSError):
                load_envelope(current)

    def test_stale_new_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "resume.dpapi"
            (Path(directory) / "resume.dpapi.new").write_bytes(b"audit")
            with self.assertRaises(RuntimeError):
                save_envelope_atomic(current, self.payload(1))


if __name__ == "__main__":
    unittest.main()
