#!/usr/bin/env python3
"""Fast unit tests for the executable DP mechanism core."""

from __future__ import annotations

import copy
import unittest

import torch

from dp_training.mechanism import (
    MechanismConfig,
    append_public_trace,
    aggregate_noised_update,
    clip_unit_vector,
    make_public_scheduled_event,
    mean_unit_vectors,
    poisson_select,
    tensor_sha256,
    verify_public_trace,
)


def config(**overrides) -> MechanismConfig:
    values = {
        "arm": "TEST-M2",
        "privacy_unit": "patient",
        "population": 100,
        "expected_batch": 4,
        "poisson_sample_rate": 0.04,
        "clip_norm": 2.0,
        "noise_multiplier": 0.5,
        "fixed_denominator": 4,
        "max_steps": 10,
        "rng_security_mode": "TEST_ONLY_DETERMINISTIC",
        "rng_backend": "torch_cpu_seeded_test_generator",
    }
    values.update(overrides)
    return MechanismConfig(**values)


class TestMechanism(unittest.TestCase):
    def test_clip_known_vector(self):
        clipped, stats = clip_unit_vector(torch.tensor([3.0, 4.0]), 2.0)
        torch.testing.assert_close(clipped, torch.tensor([1.2, 1.6]))
        self.assertAlmostEqual(stats["unclipped_l2_norm"], 5.0)
        self.assertAlmostEqual(stats["clipped_l2_norm"], 2.0, places=6)
        self.assertTrue(stats["was_clipped"])

    def test_patient_mean_occurs_before_outer_clip(self):
        per_image = [torch.tensor([4.0, 0.0]), torch.tensor([0.0, 0.0])]
        patient = mean_unit_vectors(per_image)
        correct, _ = clip_unit_vector(patient, 1.0)
        illegal = torch.stack([clip_unit_vector(item, 1.0)[0] for item in per_image]).mean(0)
        torch.testing.assert_close(patient, torch.tensor([2.0, 0.0]))
        torch.testing.assert_close(correct, torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(illegal, torch.tensor([0.5, 0.0]))
        self.assertFalse(torch.equal(correct, illegal))

    def test_empty_sample_is_noise_only_and_replay_exact(self):
        template = torch.zeros(4096, dtype=torch.float32)
        first, stats = aggregate_noised_update(
            [], template=template, config=config(), generator=torch.Generator().manual_seed(17)
        )
        second, _ = aggregate_noised_update(
            [], template=template, config=config(), generator=torch.Generator().manual_seed(17)
        )
        self.assertTrue(stats["empty_sample"])
        self.assertEqual(stats["realized_unit_count"], 0)
        self.assertGreater(float(torch.linalg.vector_norm(first)), 0.0)
        self.assertEqual(tensor_sha256(first), tensor_sha256(second))

    def test_noise_cancels_under_same_seed_and_exposes_fixed_denominator(self):
        template = torch.zeros(2, dtype=torch.float32)
        units = [torch.tensor([3.0, 4.0]), torch.tensor([0.0, 1.0])]
        value, stats = aggregate_noised_update(
            units, template=template, config=config(), generator=torch.Generator().manual_seed(9)
        )
        noise_only, _ = aggregate_noised_update(
            [], template=template, config=config(), generator=torch.Generator().manual_seed(9)
        )
        expected = (torch.tensor([1.2, 1.6]) + torch.tensor([0.0, 1.0])) / 4.0
        torch.testing.assert_close(value - noise_only, expected)
        self.assertEqual(stats["fixed_denominator_used"], 4.0)
        self.assertEqual(stats["realized_unit_count"], 2)

    def test_config_rejects_variable_denominator_and_wrong_q(self):
        with self.assertRaisesRegex(ValueError, "fixed_denominator"):
            config(fixed_denominator=3)
        with self.assertRaisesRegex(ValueError, "poisson_sample_rate"):
            config(poisson_sample_rate=0.05)

    def test_vectors_must_be_flat_finite_fp32(self):
        with self.assertRaisesRegex(ValueError, "fp32"):
            clip_unit_vector(torch.ones(2, dtype=torch.float16), 1.0)
        with self.assertRaisesRegex(ValueError, "flat"):
            clip_unit_vector(torch.ones(1, 2, dtype=torch.float32), 1.0)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            clip_unit_vector(torch.tensor([float("nan")], dtype=torch.float32), 1.0)

    def test_poisson_selection_is_deterministic_and_requires_canonical_population(self):
        ids = [f"u{i:03d}" for i in range(100)]
        one = poisson_select(ids, 0.04, generator=torch.Generator().manual_seed(123))
        two = poisson_select(ids, 0.04, generator=torch.Generator().manual_seed(123))
        self.assertEqual(one, two)
        with self.assertRaisesRegex(ValueError, "sorted"):
            poisson_select(list(reversed(ids)), 0.04, generator=torch.Generator().manual_seed(1))

    def test_public_trace_is_schedule_only_and_hash_chained(self):
        trace = []
        event = make_public_scheduled_event(config(), first_step=1, event_count=10)
        append_public_trace(trace, event)
        result = verify_public_trace(trace)
        self.assertEqual(result["total_events"], 10)
        serialized = str(trace)
        for forbidden in ("selected_ids", "realized_unit_count", "noise_seed"):
            self.assertNotIn(forbidden, serialized)

    def test_public_trace_detects_tampering(self):
        trace = []
        append_public_trace(trace, make_public_scheduled_event(config(), first_step=1, event_count=10))
        tampered = copy.deepcopy(trace)
        tampered[0]["event"]["noise_multiplier"] = 999.0
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_public_trace(tampered)

    def test_public_trace_rejects_realized_batch_field(self):
        trace = []
        event = make_public_scheduled_event(config(), first_step=1, event_count=10)
        event["realized_batch_size"] = 4
        with self.assertRaisesRegex(ValueError, "missing or extra"):
            append_public_trace(trace, event)


if __name__ == "__main__":
    unittest.main()
