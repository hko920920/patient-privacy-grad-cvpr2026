from __future__ import annotations

import unittest

import torch

from set_adapter import PatientSetAdapter


def activate_output(adapter: PatientSetAdapter) -> None:
    generator = torch.Generator(device="cpu").manual_seed(260903)
    with torch.no_grad():
        adapter.to_delta.weight.copy_(
            torch.randn(adapter.to_delta.weight.shape, generator=generator) * 1e-2
        )
        adapter.to_delta.bias.zero_()


class PatientSetAdapterTests(unittest.TestCase):
    def test_frozen_parameter_count(self) -> None:
        adapter = PatientSetAdapter(channels=1280, token_dim=128, num_heads=4)
        self.assertEqual(adapter.trainable_parameter_count, 463_872)

    def test_zero_initialization_is_exact_identity(self) -> None:
        adapter = PatientSetAdapter(channels=8, token_dim=4, num_heads=2)
        hidden = torch.randn(6, 8, 3, 3)
        mask = torch.ones(2, 3, dtype=torch.bool)
        output = adapter(hidden, mask)
        self.assertTrue(torch.equal(output, hidden))

    def test_permutation_equivariance_after_activation(self) -> None:
        torch.manual_seed(11)
        adapter = PatientSetAdapter(channels=8, token_dim=4, num_heads=2)
        activate_output(adapter)
        hidden = torch.randn(6, 8, 3, 3)
        mask = torch.tensor([[True, True, True], [True, True, False]])
        permutation = torch.tensor([2, 0, 1])
        inverse = torch.argsort(permutation)
        expected = adapter(hidden, mask).reshape(2, 3, 8, 3, 3)
        permuted_hidden = hidden.reshape(2, 3, 8, 3, 3)[:, permutation].reshape(6, 8, 3, 3)
        permuted_mask = mask[:, permutation]
        observed = adapter(permuted_hidden, permuted_mask).reshape(2, 3, 8, 3, 3)[:, inverse]
        self.assertTrue(torch.allclose(observed, expected, atol=1e-6, rtol=1e-6))

    def test_padded_slot_cannot_influence_valid_slots(self) -> None:
        torch.manual_seed(12)
        adapter = PatientSetAdapter(channels=8, token_dim=4, num_heads=2)
        activate_output(adapter)
        hidden = torch.randn(3, 8, 3, 3)
        changed = hidden.clone()
        changed[2] = 100.0 * torch.randn_like(changed[2])
        mask = torch.tensor([[True, True, False]])
        first = adapter(hidden, mask)
        second = adapter(changed, mask)
        self.assertTrue(torch.allclose(first[:2], second[:2], atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.equal(first[2], hidden[2]))
        self.assertTrue(torch.equal(second[2], changed[2]))

    def test_singleton_is_exact_identity_even_after_activation(self) -> None:
        adapter = PatientSetAdapter(channels=8, token_dim=4, num_heads=2)
        activate_output(adapter)
        hidden = torch.randn(2, 8, 3, 3)
        mask = torch.ones(2, 1, dtype=torch.bool)
        self.assertTrue(torch.equal(adapter(hidden, mask), hidden))

    def test_invalid_layout_fails_closed(self) -> None:
        adapter = PatientSetAdapter(channels=8, token_dim=4, num_heads=2)
        hidden = torch.randn(2, 8, 3, 3)
        with self.assertRaises(ValueError):
            adapter(hidden, torch.ones(1, 2))
        with self.assertRaises(ValueError):
            adapter(hidden, torch.tensor([[False, False]]))
        with self.assertRaises(ValueError):
            adapter(hidden[:1], torch.tensor([[True, True]]))


if __name__ == "__main__":
    unittest.main()
