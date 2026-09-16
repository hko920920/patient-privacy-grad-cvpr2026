from __future__ import annotations

import unittest

import torch

from cross_record_adapter import CrossRecordOnlyAdapter


def activate(adapter: CrossRecordOnlyAdapter) -> None:
    generator = torch.Generator(device="cpu").manual_seed(39762)
    with torch.no_grad():
        adapter.to_delta.weight.copy_(
            torch.randn(adapter.to_delta.weight.shape, generator=generator) * 1e-2
        )
        adapter.to_delta.bias.zero_()


class CrossRecordOnlyAdapterTests(unittest.TestCase):
    def test_frozen_parameter_count(self) -> None:
        adapter = CrossRecordOnlyAdapter(channels=1280, token_dim=128, num_heads=4)
        self.assertEqual(adapter.trainable_parameter_count, 447_360)

    def test_zero_init_is_identity(self) -> None:
        adapter = CrossRecordOnlyAdapter(channels=8, token_dim=4, num_heads=2)
        hidden = torch.randn(4, 8, 2, 2)
        mask = torch.ones(2, 2, dtype=torch.bool)
        self.assertTrue(torch.equal(adapter(hidden, mask), hidden))

    def test_q2_delta_has_no_self_value_path(self) -> None:
        torch.manual_seed(3)
        adapter = CrossRecordOnlyAdapter(channels=8, token_dim=4, num_heads=2)
        activate(adapter)
        hidden = torch.randn(2, 8, 2, 2)
        mask = torch.ones(1, 2, dtype=torch.bool)
        first_delta = adapter.residual(hidden, mask)

        own_changed = hidden.clone()
        own_changed[0] = 20.0 * torch.randn_like(own_changed[0])
        own_delta = adapter.residual(own_changed, mask)
        self.assertTrue(torch.allclose(first_delta[0], own_delta[0], atol=1e-6, rtol=1e-6))

        companion_changed = hidden.clone()
        companion_changed[1] = 20.0 * torch.randn_like(companion_changed[1])
        companion_delta = adapter.residual(companion_changed, mask)
        self.assertFalse(torch.allclose(first_delta[0], companion_delta[0], atol=1e-6, rtol=1e-6))

    def test_permutation_equivariance(self) -> None:
        torch.manual_seed(4)
        adapter = CrossRecordOnlyAdapter(channels=8, token_dim=4, num_heads=2)
        activate(adapter)
        hidden = torch.randn(6, 8, 2, 2)
        mask = torch.tensor([[True, True, True], [True, True, False]])
        permutation = torch.tensor([2, 0, 1])
        inverse = torch.argsort(permutation)
        expected = adapter(hidden, mask).reshape(2, 3, 8, 2, 2)
        permuted_hidden = hidden.reshape(2, 3, 8, 2, 2)[:, permutation].reshape(6, 8, 2, 2)
        observed = adapter(permuted_hidden, mask[:, permutation]).reshape(2, 3, 8, 2, 2)[:, inverse]
        self.assertTrue(torch.allclose(observed, expected, atol=1e-6, rtol=1e-6))

    def test_padding_and_singleton_are_fail_closed_identity_paths(self) -> None:
        adapter = CrossRecordOnlyAdapter(channels=8, token_dim=4, num_heads=2)
        activate(adapter)
        hidden = torch.randn(3, 8, 2, 2)
        changed = hidden.clone()
        changed[2] = 100.0 * torch.randn_like(changed[2])
        mask = torch.tensor([[True, True, False]])
        first = adapter(hidden, mask)
        second = adapter(changed, mask)
        self.assertTrue(torch.allclose(first[:2], second[:2], atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.equal(first[2], hidden[2]))
        singleton = torch.randn(2, 8, 2, 2)
        self.assertTrue(
            torch.equal(adapter(singleton, torch.ones(2, 1, dtype=torch.bool)), singleton)
        )

    def test_invalid_layout_raises(self) -> None:
        adapter = CrossRecordOnlyAdapter(channels=8, token_dim=4, num_heads=2)
        hidden = torch.randn(2, 8, 2, 2)
        with self.assertRaises(ValueError):
            adapter(hidden, torch.tensor([[False, False]]))
        with self.assertRaises(ValueError):
            adapter(hidden[:1], torch.tensor([[True, True]]))


if __name__ == "__main__":
    unittest.main()
