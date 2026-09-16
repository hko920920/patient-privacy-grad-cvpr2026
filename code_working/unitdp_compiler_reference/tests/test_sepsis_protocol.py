import argparse
import ast
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get("UNITDP_DATA_ROOT", str(ROOT / "data"))
)
SRC = ROOT / "src"
for path in [ROOT, SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import run_sepsis_owa as sepsis
from unitdp.preprocessing import load_fixed_affine_preprocessor

SEPSIS_CACHE = DATA_ROOT / "sepsis2019_cache"


class SepsisPublicProtocolTest(unittest.TestCase):
    def registered_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            max_owners=400,
            train_frac=0.8,
            window_length=12,
            stride=6,
            label_mode="any",
            feature_mode="basic",
            policy="support_balanced_cap",
            max_windows_per_owner=8,
            step_window_budget=8,
        )

    def test_checked_in_artifact_binds_materialized_protocol(self):
        artifact = load_fixed_affine_preprocessor(
            sepsis.DEFAULT_SEPSIS_PREPROCESSING_ARTIFACT,
            expected_payload_sha256=sepsis.DEFAULT_SEPSIS_PREPROCESSING_SHA256,
        )
        raw = json.loads(
            sepsis.DEFAULT_SEPSIS_PREPROCESSING_ARTIFACT.read_text(
                encoding="utf-8"
            )
        )
        protocol = raw["dataset_protocol"]

        self.assertEqual(artifact.input_dim, 240)
        self.assertEqual(protocol["protocol_id"], sepsis.SEPSIS_PROTOCOL_ID)
        self.assertEqual(len(protocol["patient_manifest"]), 400)
        self.assertEqual(len(protocol["train_owner_ids"]), 312)
        self.assertEqual(len(protocol["test_owner_ids"]), 78)
        self.assertEqual(protocol["train_window_count"], 1601)
        self.assertEqual(protocol["test_window_count"], 398)
        self.assertEqual(protocol["selection_policy"]["selected_window_count"], 1507)
        self.assertEqual(protocol["selected_class_counts"], [1470, 37])

    def test_local_cache_matches_every_materialized_patient_hash(self):
        if not SEPSIS_CACHE.is_dir():
            self.skipTest("Local Sepsis cache is not available")
        protocol = sepsis.load_registered_public_protocol(SEPSIS_CACHE)

        self.assertEqual(len(protocol.train_files), 312)
        self.assertEqual(len(protocol.test_files), 78)
        self.assertEqual(
            protocol.fixed_class_weights,
            (0.5125850340136054, 20.364864864864863),
        )

    def test_missing_public_cohort_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "patient file mismatch"):
                sepsis.load_registered_public_protocol(tmp)

    def test_protocol_arguments_are_fixed(self):
        args = self.registered_args()
        sepsis.validate_registered_protocol_args(args)

        args.feature_mode = "trend"
        with self.assertRaisesRegex(ValueError, "feature_mode='basic'"):
            sepsis.validate_registered_protocol_args(args)

    def test_window_features_are_record_local_and_finite(self):
        first = np.arange(12 * 40, dtype=np.float32).reshape(12, 40)
        first[::2, 3] = np.nan
        second = np.full((12, 40), 999.0, dtype=np.float32)

        isolated = sepsis.window_features(first, "basic")
        repeated = sepsis.window_features(first.copy(), "basic")
        _unrelated = sepsis.window_features(second, "basic")

        self.assertEqual(isolated.shape, (240,))
        self.assertTrue(np.array_equal(isolated, repeated))
        self.assertTrue(np.isfinite(isolated).all())

    def test_runtime_paths_do_not_refit_split_scaler_or_class_weights(self):
        function_targets = [
            (ROOT / "scripts" / "run_sepsis_owa.py", "main"),
            (ROOT / "scripts" / "run_sepsis_comparison.py", "main"),
            (
                ROOT / "scripts" / "run_conservative_els_owner_baseline.py",
                "prepare_sepsis",
            ),
        ]
        forbidden_names = {
            "stratified_owner_split",
            "balanced_class_weights",
            "StandardScaler",
        }
        for path, function_name in function_targets:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            function = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == function_name
            )
            called_names = {
                node.func.id
                for node in ast.walk(function)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            fitted = {
                node.attr
                for node in ast.walk(function)
                if isinstance(node, ast.Attribute)
                and node.attr in {"fit", "fit_transform"}
            }
            self.assertFalse(
                called_names & forbidden_names,
                f"{path.name}.{function_name} recomputes public protocol state",
            )
            self.assertFalse(
                fitted,
                f"{path.name}.{function_name} contains runtime fit calls",
            )


if __name__ == "__main__":
    unittest.main()
