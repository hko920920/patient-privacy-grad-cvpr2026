from __future__ import annotations

import unittest

import numpy as np

import run_isic_sd21_covariance_allocation_discovery as discovery
import run_isic_sd21_joint_batch_locality as joint


def clip_rows(vectors: np.ndarray, clip_norm: float) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    scales = np.minimum(
        1.0,
        np.divide(
            clip_norm,
            norms,
            out=np.ones_like(norms),
            where=norms > 0,
        ),
    )
    return scales[:, None] * vectors


class CovarianceAllocationDiscoveryTests(unittest.TestCase):
    def test_template_catalog_is_complete_modulo_complement(self) -> None:
        templates = discovery.all_rank_templates()
        self.assertEqual(len(templates), 35)
        self.assertEqual(len(set(templates)), 35)
        self.assertTrue(all(discovery.canonical_template(value) == value for value in templates))
        self.assertEqual(
            {shape: sum(discovery.template_shape(value) == shape for value in templates)
             for shape in ("1111", "2110", "2200")},
            {"1111": 8, "2110": 24, "2200": 3},
        )
        for template in templates:
            complement = joint.complement(template)
            self.assertEqual(discovery.canonical_template(complement), template)

    def test_every_symmetric_template_has_uniform_cell_expectation(self) -> None:
        expected = np.full(40, 1.0 / 40.0)
        for template in discovery.all_rank_templates():
            first = joint.fixed_subset_weights(template)
            second = joint.fixed_subset_weights(joint.complement(template))
            observed = np.mean(np.concatenate((first, second), axis=0), axis=0)
            np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-15)

    def test_rank_token_map_uses_public_timestep_order(self) -> None:
        rows = []
        values = ((17, 4), (11, 7), (28, 1), (29, 6), (44, 5), (31, 0), (80, 3), (79, 2))
        for token_pair_index in range(4):
            for within_index in range(2):
                timestep, perturbation = values[2 * token_pair_index + within_index]
                rows.append(
                    {
                        "bank_tag": "joint_01",
                        "bin_index": str(token_pair_index),
                        "timestep": str(timestep),
                        "perturbation_index": str(perturbation),
                    }
                )
        self.assertEqual(
            discovery.rank_token_map(rows, "joint_01"),
            (7, 4, 1, 6, 0, 5, 2, 3),
        )

    def test_symmetric_pair_formula_matches_explicit_state_vectors(self) -> None:
        rng = np.random.default_rng(26090306)
        dimension = 9
        left_gradients = rng.normal(size=(40, dimension))
        right_gradients = rng.normal(size=(40, dimension))
        left_gram = left_gradients @ left_gradients.T
        right_gram = right_gradients @ right_gradients.T
        cross_gram = left_gradients @ right_gradients.T
        clip_norm = 0.7
        subsets = joint.all_subsets()
        subset_to_index = {subset: index for index, subset in enumerate(subsets)}
        weights = [joint.fixed_subset_weights(subset) for subset in subsets]
        left_stats = joint.allocation_statistics(left_gram, weights, (clip_norm,))
        right_stats = joint.allocation_statistics(right_gram, weights, (clip_norm,))
        template = discovery.all_rank_templates()[13]
        other = joint.complement(template)
        first_indices = np.asarray([subset_to_index[template]], dtype=np.int64)
        second_indices = np.asarray([subset_to_index[other]], dtype=np.int64)
        observed, _, _ = discovery.symmetric_pair_components(
            left_stats,
            right_stats,
            cross_gram,
            0,
            first_indices,
            second_indices,
            dimension,
        )

        left_reference = clip_rows(np.mean(left_gradients, axis=0, keepdims=True), clip_norm)[0]
        right_reference = clip_rows(np.mean(right_gradients, axis=0, keepdims=True), clip_norm)[0]

        def orientation(left_subset: tuple[int, ...], right_subset: tuple[int, ...]) -> float:
            left_vectors = weights[subset_to_index[left_subset]] @ left_gradients
            right_vectors = weights[subset_to_index[right_subset]] @ right_gradients
            left_errors = clip_rows(left_vectors, clip_norm) - left_reference
            right_errors = clip_rows(right_vectors, clip_norm) - right_reference
            combined = 0.5 * (
                left_errors[:, None, :] + right_errors[None, :, :]
            )
            return float(np.mean(np.sum(combined * combined, axis=2)) / dimension)

        expected = 0.5 * (
            orientation(template, other) + orientation(other, template)
        )
        self.assertAlmostEqual(float(observed[0]), expected, places=13)

    def test_patient_template_formula_matches_explicit_states(self) -> None:
        rng = np.random.default_rng(991)
        dimension = 7
        gradients = rng.normal(size=(40, dimension))
        gram = gradients @ gradients.T
        clip_norm = 0.9
        subsets = joint.all_subsets()
        subset_to_index = {subset: index for index, subset in enumerate(subsets)}
        weights = [joint.fixed_subset_weights(subset) for subset in subsets]
        stats = joint.allocation_statistics(gram, weights, (clip_norm,))
        template = discovery.all_rank_templates()[22]
        other = joint.complement(template)
        first_indices = np.asarray([subset_to_index[template]], dtype=np.int64)
        second_indices = np.asarray([subset_to_index[other]], dtype=np.int64)
        observed_mse, observed_rate = discovery.patient_template_components(
            stats, 0, first_indices, second_indices, dimension
        )

        reference = clip_rows(np.mean(gradients, axis=0, keepdims=True), clip_norm)[0]
        explicit_mses = []
        explicit_rates = []
        for subset in (template, other):
            vectors = weights[subset_to_index[subset]] @ gradients
            errors = clip_rows(vectors, clip_norm) - reference
            explicit_mses.append(float(np.mean(np.sum(errors * errors, axis=1)) / dimension))
            explicit_rates.append(float(np.mean(np.linalg.norm(vectors, axis=1) > clip_norm)))
        self.assertAlmostEqual(float(observed_mse[0]), float(np.mean(explicit_mses)), places=13)
        self.assertAlmostEqual(float(observed_rate[0]), float(np.mean(explicit_rates)), places=13)

    def test_lobo_selector_does_not_use_heldout_row(self) -> None:
        ratios = np.asarray(
            [
                [0.01, 3.0, 0.01],
                [1.2, 0.8, 1.1],
                [1.1, 0.7, 1.2],
                [1.3, 0.9, 1.1],
                [1.2, 0.8, 1.3],
            ],
            dtype=np.float64,
        )
        selected, scores = discovery.select_training_template(ratios, [1, 2, 3, 4])
        self.assertEqual(selected, 1)
        self.assertLess(scores[1], scores[0])
        self.assertLess(scores[1], scores[2])

    def test_discovery_bootstrap_is_deterministic(self) -> None:
        ratios = np.asarray([0.91, 0.93, 0.94, 0.96, 0.92])
        first = discovery.bank_bootstrap(ratios)
        second = discovery.bank_bootstrap(ratios)
        self.assertEqual(first, second)
        self.assertEqual(first["bootstrap_seed"], 26090306)


if __name__ == "__main__":
    unittest.main()
