#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path

from isic_manifest_dataset import IsicManifestDataset, PatientGroupedView, read_manifest


ROOT = Path(__file__).resolve().parent.parent
DERIVED = ROOT / "_data" / "derived" / "isic2020_v2_split_v1"
SMOKE_IMAGES = ROOT / "_data" / "intake" / "isic2020_collection70" / "smoke_images"


class ManifestPipelineTests(unittest.TestCase):
    def test_expected_cap_counts_and_nested_ids(self) -> None:
        records = {
            cap: read_manifest(DERIVED / f"k{cap}_private.csv") for cap in (2, 5, 10)
        }
        self.assertEqual(len(records[2]), 4_112)
        self.assertEqual(len(records[5]), 9_495)
        self.assertEqual(len(records[10]), 15_924)
        ids = {cap: {record.image_id for record in rows} for cap, rows in records.items()}
        self.assertLess(ids[2], ids[5])
        self.assertLess(ids[5], ids[10])

    def test_patient_partition_and_cap(self) -> None:
        for cap in (2, 5, 10):
            dataset = IsicManifestDataset(
                DERIVED / f"k{cap}_private.csv",
                image_root=SMOKE_IMAGES,
                require_all_images=False,
            )
            grouped = PatientGroupedView(dataset)
            self.assertEqual(len(grouped), 2_056)
            grouped.validate_cap(cap)

    def test_k10_locked_partition_counts(self) -> None:
        expected = {
            "private_train": (1_415, 10_848),
            "public_development": (200, 1_565),
            "privacy_attack_holdout": (233, 1_870),
            "final_test": (208, 1_641),
        }
        for partition, (patients, images) in expected.items():
            records = read_manifest(
                DERIVED / "k10_private.csv", partitions=[partition]
            )
            self.assertEqual(len(records), images)
            self.assertEqual(len({record.patient_id for record in records}), patients)

    def test_smoke_pixels_and_patient_groups(self) -> None:
        dataset = IsicManifestDataset(
            DERIVED / "smoke_subset_private.csv",
            image_root=SMOKE_IMAGES,
            image_size=256,
        )
        self.assertEqual(len(dataset), 8)
        grouped = PatientGroupedView(dataset)
        self.assertEqual(len(grouped), 4)
        self.assertEqual(sorted(grouped.contribution_counts()), [2, 2, 2, 2])
        first = dataset[0]
        self.assertEqual(tuple(first["pixel_values"].shape), (3, 256, 256))
        self.assertGreaterEqual(float(first["pixel_values"].min()), -1.0)
        self.assertLessEqual(float(first["pixel_values"].max()), 1.0)


if __name__ == "__main__":
    unittest.main()
