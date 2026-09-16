import copy
import ast
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from unitdp.preprocessing import (
    FIXED_AFFINE_SCHEMA,
    PreprocessingArtifactError,
    load_fixed_affine_preprocessor,
    preprocessing_payload_sha256,
)
from unitdp.contract import UnitContract

ROOT = Path(__file__).resolve().parents[1]
UCI_ARTIFACT = (
    ROOT
    / "configs"
    / "preprocessing"
    / "uci_har_published_train_standard_scaler_v1.json"
)
UCI_ARTIFACT_SHA256 = (
    "774713e3bef48c784113f4097642a59741481e7495f858ed41cdb1788ae8fadf"
)


def make_artifact() -> dict[str, object]:
    raw: dict[str, object] = {
        "schema": FIXED_AFFINE_SCHEMA,
        "artifact_id": "test_public_scaler_v1",
        "scope": "public_fixed",
        "dataset": "synthetic",
        "representation": "two_features",
        "input_dim": 2,
        "transform": "standard_scaler",
        "mean": [1.0, -2.0],
        "scale": [2.0, 4.0],
        "fit_on_protected_data": False,
        "source": {
            "reference": "synthetic/public_reference.txt",
            "sha256": "a" * 64,
            "is_public_reference": True,
        },
    }
    raw["payload_sha256"] = preprocessing_payload_sha256(raw)
    return raw


class FixedPreprocessingArtifactTest(unittest.TestCase):
    def write_artifact(self, directory: str, raw: dict[str, object]) -> Path:
        path = Path(directory) / "artifact.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        return path

    def test_fixed_transform_is_exactly_removal_invariant(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = load_fixed_affine_preprocessor(
                self.write_artifact(tmp, make_artifact())
            )
            values = np.asarray(
                [[1.0, -2.0], [3.0, 2.0], [-1.0, -6.0]],
                dtype=np.float32,
            )
            owners = np.asarray(["u0", "u1", "u1"])
            full_then_remove = artifact.transform(values)[owners != "u0"]
            remove_then_transform = artifact.transform(values[owners != "u0"])

            self.assertTrue(np.array_equal(full_then_remove, remove_then_transform))
            self.assertEqual(artifact.contract_binding()["fit_on_protected_data"], False)

    def test_payload_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = make_artifact()
            raw["mean"] = [99.0, -2.0]
            path = self.write_artifact(tmp, raw)

            with self.assertRaisesRegex(
                PreprocessingArtifactError, "payload digest mismatch"
            ):
                load_fixed_affine_preprocessor(path)

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = json.dumps(make_artifact())
            duplicated = raw.replace(
                '"input_dim": 2,',
                '"input_dim": 2, "input_dim": 2,',
                1,
            )
            path = Path(tmp) / "artifact.json"
            path.write_text(duplicated, encoding="utf-8")

            with self.assertRaisesRegex(
                PreprocessingArtifactError,
                "Duplicate preprocessing JSON key",
            ):
                load_fixed_affine_preprocessor(path)

    def test_contract_digest_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_artifact(tmp, make_artifact())

            with self.assertRaisesRegex(
                PreprocessingArtifactError, "contract digest"
            ):
                load_fixed_affine_preprocessor(
                    path,
                    expected_payload_sha256="0" * 64,
                )

    def test_nonpositive_scale_is_rejected_after_valid_rehash(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = copy.deepcopy(make_artifact())
            raw["scale"] = [2.0, 0.0]
            raw["payload_sha256"] = preprocessing_payload_sha256(raw)
            path = self.write_artifact(tmp, raw)

            with self.assertRaisesRegex(PreprocessingArtifactError, "positive"):
                load_fixed_affine_preprocessor(path)

    def test_checked_in_uci_artifact_is_digest_bound(self):
        artifact = load_fixed_affine_preprocessor(
            UCI_ARTIFACT,
            expected_payload_sha256=UCI_ARTIFACT_SHA256,
        )

        self.assertEqual(artifact.input_dim, 561)
        self.assertEqual(
            artifact.artifact_id,
            "uci_har_published_train_standard_scaler_v1",
        )
        self.assertFalse(artifact.contract_binding()["fit_on_protected_data"])

    def test_uci_paths_do_not_fit_preprocessing_on_runtime_input(self):
        full_uci_entrypoints = [
            ROOT / "scripts" / "run_uci_comparison.py",
            ROOT / "scripts" / "run_uci_owa.py",
        ]
        mixed_entrypoints = [
            ROOT / "scripts" / "run_aggregation_ablation.py",
            ROOT / "scripts" / "run_conservative_els_owner_baseline.py",
            ROOT / "scripts" / "run_owner_membership_probe.py",
        ]

        for path in full_uci_entrypoints:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            forbidden = {
                node.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Attribute)
                and node.attr in {"fit", "fit_transform"}
            }
            self.assertFalse(forbidden, f"{path.name} contains runtime fit calls")

        for path in mixed_entrypoints:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            prepare_uci = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "prepare_uci"
            )
            forbidden = {
                node.attr
                for node in ast.walk(prepare_uci)
                if isinstance(node, ast.Attribute)
                and node.attr in {"fit", "fit_transform"}
            }
            self.assertFalse(
                forbidden,
                f"{path.name}.prepare_uci contains runtime fit calls",
            )

    def test_contract_rejects_public_artifact_that_can_fit_protected_data(self):
        binding = load_fixed_affine_preprocessor(UCI_ARTIFACT).contract_binding()
        binding["fit_on_protected_data"] = True
        contract = UnitContract(
            privacy_unit="owner",
            mapping="mapping.csv",
            mechanism={
                "sampling_unit": "owner",
                "clipping_unit": "owner",
                "noising_unit": "owner",
            },
            accountant={"unit": "owner"},
            data={"preprocessing": binding},
        )

        with self.assertRaisesRegex(ValueError, "fit_on_protected_data=false"):
            contract.validate()


if __name__ == "__main__":
    unittest.main()
