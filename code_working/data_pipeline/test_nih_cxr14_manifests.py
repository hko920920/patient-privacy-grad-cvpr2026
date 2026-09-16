from __future__ import annotations

import unittest

import build_nih_cxr14_manifests as builder


class NihCxr14ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_dir, _, _ = builder.default_paths()
        cls.outputs, cls.report = builder.build_outputs(cls.source_dir)

    def test_locked_source_and_population(self) -> None:
        self.assertEqual(self.report["source"]["source_images"], 112_120)
        self.assertEqual(self.report["source"]["source_patients"], 30_805)
        self.assertEqual(self.report["source"]["pa_images"], 67_310)
        self.assertEqual(self.report["source"]["pa_patients"], 28_868)

    def test_balanced_patient_partitions(self) -> None:
        for partition, expected in builder.EXPECTED_PARTITION_PATIENTS.items():
            stats = self.report["caps"]["5"]["partitions"][partition]
            self.assertEqual(stats["patients"], expected)
            self.assertEqual(stats["target_patients"], expected // 2)
            self.assertEqual(stats["nontarget_patients"], expected // 2)

    def test_nested_cap_counts(self) -> None:
        self.assertEqual(self.report["caps"]["2"]["total_images"], 20_687)
        self.assertEqual(self.report["caps"]["5"]["total_images"], 31_451)
        self.assertEqual(self.report["caps"]["10"]["total_images"], 38_492)
        self.assertTrue(self.report["nesting"]["k2_subset_k5"])
        self.assertTrue(self.report["nesting"]["k5_subset_k10"])

    def test_official_pa_test_census_guard(self) -> None:
        census = self.report["official_pa_test_census"]
        self.assertEqual(census["images"], 11_096)
        self.assertEqual(census["patients"], 2_647)
        self.assertFalse(census["independent_test_sample"])
        self.assertFalse(census["may_select_hyperparameters_or_attack_thresholds"])
        self.assertTrue(
            self.report["nesting"]["k10_final_subset_official_pa_test_census"]
        )
        self.assertEqual(
            self.report["selective_acquisition"][
                "k10_plus_official_pa_test_census_unique_images"
            ],
            42_423,
        )

    def test_first_seed_k5_adequacy(self) -> None:
        self.assertEqual(self.report["adequacy_gate"]["status"], "PASS")
        private = self.report["caps"]["5"]["partitions"]["private_train"]
        self.assertEqual(private["primary_groups"]["pneumothorax"]["images"], 792)
        self.assertEqual(
            private["primary_groups"]["pneumonia_or_consolidation"]["images"],
            894,
        )

    def test_build_is_byte_deterministic(self) -> None:
        outputs_again, report_again = builder.build_outputs(self.source_dir)
        self.assertEqual(self.outputs, outputs_again)
        self.assertEqual(self.report, report_again)


if __name__ == "__main__":
    unittest.main()
