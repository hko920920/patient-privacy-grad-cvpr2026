from __future__ import annotations

import unittest

import numpy as np

import run_nih_cxr14_order_proxy_residual_premise as premise


def record(patient: str, followup: int, labels: tuple[str, ...], sex: str = "M") -> premise.Record:
    image = f"{patient}_{followup:03d}.png"
    return premise.Record(
        image_id=image,
        patient_id=patient,
        partition="public_development",
        view="PA",
        followup_no=followup,
        patient_sex=sex,
        finding_labels=labels,
        image_filename=image,
    )


def unit(patient: str, previous: tuple[str, ...], current: tuple[str, ...]) -> dict:
    return {
        "patient_id": patient,
        "transition_class": "no_change" if previous == current else "change",
        "previous": record(patient, 0, previous),
        "current": record(patient, 1, current),
    }


class OrderProxyResidualPremiseTests(unittest.TestCase):
    def test_transition_vector_includes_no_finding(self) -> None:
        value = premise.transition_vector(unit("p", ("No Finding",), ("Mass",)))
        self.assertEqual(value.shape, (30,))
        mass = premise.LABELS.index("Mass")
        no_finding = premise.LABELS.index("No Finding")
        self.assertEqual(value[mass], 1.0)
        self.assertEqual(value[len(premise.LABELS) + no_finding], 1.0)
        self.assertEqual(float(value.sum()), 2.0)

    def test_selection_is_balanced_unique_and_deterministic(self) -> None:
        rows = []
        for index in range(180):
            patient = f"same-{index:03d}"
            rows.extend(
                [record(patient, 0, ("No Finding",)), record(patient, 1, ("No Finding",))]
            )
        for index in range(180):
            patient = f"change-{index:03d}"
            rows.extend(
                [record(patient, 0, ("No Finding",)), record(patient, 1, ("Mass",))]
            )
        first, _ = premise.select_cohort(rows, set())
        second, _ = premise.select_cohort(list(reversed(rows)), set())
        self.assertEqual(len(first), premise.TOTAL_PATIENTS)
        self.assertEqual(
            [(item["patient_id"], item["split"], item["transition_class"]) for item in first],
            [(item["patient_id"], item["split"], item["transition_class"]) for item in second],
        )
        counts = {
            key: sum((item["split"], item["transition_class"]) == key for item in first)
            for key in (
                ("train", "no_change"),
                ("train", "change"),
                ("validation", "no_change"),
                ("validation", "change"),
            )
        }
        self.assertEqual(set(counts.values()), {48, 96})
        self.assertEqual(len({item["patient_id"] for item in first}), premise.TOTAL_PATIENTS)

    def test_ridge_recovers_transition_signal(self) -> None:
        rng = np.random.default_rng(9)
        x = rng.normal(size=(100, 6))
        beta = rng.normal(size=(6, 8))
        y = x @ beta + 0.001 * rng.normal(size=(100, 8))
        units = [{"patient_id": f"p{i:03d}"} for i in range(100)]
        model, report = premise.choose_ridge(x, y, units)
        prediction = premise.predict_ridge(model, x)
        self.assertLess(float(np.mean((y - prediction) ** 2)), 1e-3)
        self.assertIn(report["selected_alpha"], premise.RIDGE_ALPHAS)

    def test_derangement_has_no_fixed_row_and_is_deterministic(self) -> None:
        units = []
        rows = []
        for name in ("no_change", "change"):
            for index in range(10):
                units.append({"patient_id": f"{name}-{index}", "transition_class": name})
                rows.append(np.asarray([index, int(name == "change")], dtype=np.float64))
        x = np.stack(rows)
        first, report = premise.derange_transition_rows(x, units)
        second, _ = premise.derange_transition_rows(x, units)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(all(item["fixed_points"] == 0 for item in report.values()))
        self.assertTrue(all(not np.array_equal(first[i], x[i]) for i in range(len(x))))

    def test_bootstrap_ratio_is_reproducible(self) -> None:
        numerator = np.asarray([1.0, 2.0, 3.0, 4.0])
        denominator = np.asarray([2.0, 3.0, 4.0, 5.0])
        first = premise.bootstrap_ratio(numerator, denominator, 123)
        second = premise.bootstrap_ratio(numerator, denominator, 123)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], 2.5 / 3.5)

    def test_shuffled_prior_prefers_same_sex_then_labels(self) -> None:
        units = [
            unit("query", ("Mass",), ("Mass", "Nodule")),
            unit("wrong-sex", ("Mass",), ("Mass", "Nodule")),
            unit("same-sex-poor", ("No Finding",), ("Mass",)),
            unit("same-sex-exact", ("Mass",), ("Mass", "Nodule")),
        ]
        units[0]["previous"] = record("query", 0, ("Mass",), "F")
        units[0]["current"] = record("query", 1, ("Mass", "Nodule"), "F")
        units[1]["previous"] = record("wrong-sex", 0, ("Mass",), "M")
        units[1]["current"] = record("wrong-sex", 1, ("Mass", "Nodule"), "M")
        units[2]["previous"] = record("same-sex-poor", 0, ("No Finding",), "F")
        units[2]["current"] = record("same-sex-poor", 1, ("Mass",), "F")
        units[3]["previous"] = record("same-sex-exact", 0, ("Mass",), "F")
        units[3]["current"] = record("same-sex-exact", 1, ("Mass", "Nodule"), "F")
        chosen, match = premise.choose_shuffled_prior(0, [0, 1, 2, 3], units)
        self.assertEqual(chosen, 3)
        self.assertTrue(match["same_sex"])
        self.assertTrue(match["previous_exact_label"])
        self.assertTrue(match["future_exact_label"])


if __name__ == "__main__":
    unittest.main()
