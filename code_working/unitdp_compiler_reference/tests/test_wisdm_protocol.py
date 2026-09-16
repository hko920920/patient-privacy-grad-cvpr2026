import ast
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in [ROOT, SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import run_wisdm_owa as wisdm
from unitdp.preprocessing import load_fixed_affine_preprocessor

WISDM_ARTIFACT = (
    ROOT
    / "configs"
    / "preprocessing"
    / "wisdm_v1_1_train_owners_1_25_stats24_standard_scaler_v1.json"
)


class WisdmPublicProtocolTest(unittest.TestCase):
    def test_parser_uses_semicolon_records_and_optional_trailing_comma(self):
        text = "\n".join(
            [
                "1,Walking,1,1.0,2.0,3.0;",
                "1,Jogging,2,4.0,5.0,6.0,;",
                (
                    "2,Sitting,3,7.0,8.0,9.0;"
                    "2,Standing,4,10.0,11.0,12.0;"
                ),
                "2,Walking,5,1.0,2.0",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.txt"
            path.write_text(text, encoding="utf-8")
            rows = wisdm.parse_raw_wisdm(
                path,
                strict_public_protocol=False,
            )

        self.assertEqual([row["activity"] for row in rows[1]], ["Walking", "Jogging"])
        self.assertEqual(
            [row["activity"] for row in rows[2]],
            ["Sitting", "Standing"],
        )
        self.assertEqual(
            [row["event_idx"] for row in rows[2]],
            [0, 1],
        )

    def test_registered_split_does_not_depend_on_observed_fraction_rounding(self):
        rows = {owner: [] for owner in wisdm.WISDM_USER_IDS}
        train, test = wisdm.fixed_owner_split(rows, 0.7)

        self.assertEqual(train, list(range(1, 26)))
        self.assertEqual(test, list(range(26, 37)))
        with self.assertRaisesRegex(ValueError, "registered WISDM protocol"):
            wisdm.fixed_owner_split(rows, 0.8)

    def test_label_vocabulary_is_fixed_and_unknown_labels_fail(self):
        labels = np.asarray(
            ["Downstairs", "Jogging", "Sitting", "Standing", "Upstairs", "Walking"]
        )
        encoded = wisdm.encode_activity_labels(labels)

        self.assertTrue(np.array_equal(encoded, np.arange(6, dtype=np.int64)))
        with self.assertRaisesRegex(ValueError, "Unknown WISDM"):
            wisdm.encode_activity_labels(np.asarray(["Cycling"]))

    def test_window_features_are_exactly_owner_removal_invariant(self):
        rows: dict[int, list[dict[str, object]]] = {}
        for owner in [1, 2]:
            rows[owner] = [
                {
                    "line_idx": index,
                    "event_idx": index,
                    "activity": "Walking",
                    "timestamp": index,
                    "xyz": (float(owner), float(index % 5), -float(owner)),
                }
                for index in range(300)
            ]
        full_records, full_x, _ = wisdm.generate_windows(
            rows,
            [1, 2],
            "train",
            "synthetic",
        )
        reduced_records, reduced_x, _ = wisdm.generate_windows(
            rows,
            [2],
            "train",
            "synthetic",
        )
        full_owner_two = {
            record["window_id"]: full_x[index]
            for index, record in enumerate(full_records)
            if record["owner_id"] == "2"
        }
        reduced_owner_two = {
            record["window_id"]: reduced_x[index]
            for index, record in enumerate(reduced_records)
        }

        self.assertEqual(full_owner_two.keys(), reduced_owner_two.keys())
        for window_id in full_owner_two:
            self.assertTrue(
                np.array_equal(
                    full_owner_two[window_id],
                    reduced_owner_two[window_id],
                )
            )

    def test_checked_in_artifact_binds_parser_split_and_vocabulary(self):
        artifact = load_fixed_affine_preprocessor(
            WISDM_ARTIFACT,
            expected_payload_sha256=wisdm.DEFAULT_WISDM_PREPROCESSING_SHA256,
        )
        raw = json.loads(WISDM_ARTIFACT.read_text(encoding="utf-8"))
        protocol = raw["dataset_protocol"]

        self.assertEqual(artifact.input_dim, 24)
        self.assertEqual(protocol["train_owner_ids"], list(range(1, 26)))
        self.assertEqual(protocol["test_owner_ids"], list(range(26, 37)))
        self.assertEqual(
            protocol["activity_to_index"],
            wisdm.WISDM_ACTIVITY_TO_INDEX,
        )
        self.assertEqual(protocol["parsed_records"], wisdm.WISDM_PARSED_RECORDS)
        self.assertEqual(
            protocol["malformed_records_skipped"],
            wisdm.WISDM_MALFORMED_RECORDS,
        )

    def test_wisdm_runtime_paths_do_not_fit_scalers_or_label_encoders(self):
        full_entrypoints = [
            ROOT / "scripts" / "run_wisdm_owa.py",
            ROOT / "scripts" / "run_wisdm_comparison.py",
        ]
        mixed_entrypoints = [
            ROOT / "scripts" / "run_aggregation_ablation.py",
            ROOT / "scripts" / "run_conservative_els_owner_baseline.py",
        ]

        for path in full_entrypoints:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            forbidden_calls = {
                node.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Attribute)
                and node.attr in {"fit", "fit_transform"}
            }
            forbidden_names = {
                node.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Name) and node.id == "LabelEncoder"
            }
            self.assertFalse(
                forbidden_calls,
                f"{path.name} contains runtime fit calls",
            )
            self.assertFalse(
                forbidden_names,
                f"{path.name} contains LabelEncoder",
            )

        for path in mixed_entrypoints:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            prepare_wisdm = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "prepare_wisdm"
            )
            forbidden_calls = {
                node.attr
                for node in ast.walk(prepare_wisdm)
                if isinstance(node, ast.Attribute)
                and node.attr in {"fit", "fit_transform"}
            }
            self.assertFalse(
                forbidden_calls,
                f"{path.name}.prepare_wisdm contains runtime fit calls",
            )


if __name__ == "__main__":
    unittest.main()
