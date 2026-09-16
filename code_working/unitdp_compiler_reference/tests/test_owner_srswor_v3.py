import ast
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from unitdp.benchmark_data_v2 import (
    prepare_uci_har_v2,
    write_prepared_mapping_v2,
)
from unitdp.compiler_srswor_v3 import (
    compile_owner_srswor_contract_v3,
    load_owner_srswor_contract_v3,
)
from unitdp.execution_artifacts_srswor_v3 import (
    write_srswor_execution_artifacts_v3,
)
from unitdp.execution_artifacts_v2 import load_model_artifact_v2
from unitdp.owner_poisson_v2 import _model_state_sha256
from unitdp.owner_srswor_v3 import (
    OwnerSrsworExecutionV3Error,
    train_owner_srswor_fixed_v3,
)
from unitdp.release_artifacts import assert_public_certificate_redacted
from owner_srswor_trace_reference import (
    ReferenceRecord,
    replay_owner_srswor_linear_trace,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get("UNITDP_DATA_ROOT", str(ROOT / "data"))
)
UCI_ROOT = DATA_ROOT / "uci_har" / "extracted" / "UCI HAR Dataset"
CONFIG = ROOT / "configs" / "v3" / "uci_owner_srswor_v3.yaml"
PREPROCESSOR = (
    ROOT
    / "configs"
    / "preprocessing"
    / "uci_har_published_train_standard_scaler_v1.json"
)


class OwnerSrsworExecutorV3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (UCI_ROOT / "train" / "X_train.txt").is_file():
            raise unittest.SkipTest(
                "Raw UCI HAR files are not included in the public package"
            )
        cls.prepared = prepare_uci_har_v2(
            dataset_root=UCI_ROOT,
            preprocessing_artifact_path=PREPROCESSOR,
        )
        cls.prepared.assert_registered_full_conformance()
        cls._mapping_tmp = tempfile.TemporaryDirectory()
        mapping = write_prepared_mapping_v2(
            cls.prepared,
            Path(cls._mapping_tmp.name) / "mapping.csv",
        )
        cls.route = compile_owner_srswor_contract_v3(
            load_owner_srswor_contract_v3(CONFIG),
            mapping_path=mapping,
            preprocessing_artifact_path=PREPROCESSOR,
            require_executable=True,
        )
        cls.result = train_owner_srswor_fixed_v3(
            route=cls.route,
            raw_train_x=cls.prepared.raw_train_x,
            train_y=cls.prepared.train_y,
            raw_test_x=cls.prepared.raw_test_x,
            test_y=cls.prepared.test_y,
            research_seed=13,
        )

    @classmethod
    def tearDownClass(cls):
        cls._mapping_tmp.cleanup()

    def test_every_step_uses_exact_distinct_m_and_bounded_vectors(self):
        public = self.result.public_payload()
        private = self.result.private_manifest()
        steps = private["step_diagnostics"]
        self.assertEqual(len(steps), self.route.contract.total_steps)
        self.assertTrue(
            all(
                step["sampled_owner_count"]
                == self.route.contract.sample_size
                for step in steps
            )
        )
        self.assertTrue(
            all(
                len(set(step["sampled_owner_positions"]))
                == self.route.contract.sample_size
                for step in steps
            )
        )
        self.assertTrue(
            all(
                step["owner_vectors_computed"]
                == self.route.contract.sample_size
                for step in steps
            )
        )
        self.assertTrue(
            all(
                step["max_clipped_owner_norm"]
                <= self.route.contract.clip_norm
                for step in steps
            )
        )
        self.assertTrue(
            all(
                step["noise_parameter_tensors"] == 2
                and step["optimizer_step_applied"] is True
                for step in steps
            )
        )
        self.assertEqual(private["research_seed"], 13)
        self.assertAlmostEqual(
            public["accountant"]["epsilon"],
            8.0,
            places=9,
        )
        self.assertEqual(
            public["mechanism"]["sensitivity_multiplier"],
            2.0,
        )

    def test_public_record_redacts_seed_and_sampling_trace(self):
        public = self.result.public_payload()
        assert_public_certificate_redacted(public)
        encoded = json.dumps(public, sort_keys=True)
        for forbidden in (
            "research_seed",
            "sampled_owner_positions",
            "selected_rows",
            "step_diagnostics",
        ):
            self.assertNotIn(forbidden, encoded)
        self.assertFalse(public["rng"]["secure_rng"])
        self.assertFalse(public["rng"]["random_coins_disclosed"])
        self.assertEqual(
            public["privacy_claim"]["status"],
            "accountant_output_only_not_a_release_claim",
        )

    def test_same_seed_replays_bitwise(self):
        repeated = train_owner_srswor_fixed_v3(
            route=self.route,
            raw_train_x=self.prepared.raw_train_x,
            train_y=self.prepared.train_y,
            raw_test_x=self.prepared.raw_test_x,
            test_y=self.prepared.test_y,
            research_seed=13,
        )
        self.assertEqual(
            _model_state_sha256(repeated.model),
            _model_state_sha256(self.result.model),
        )
        self.assertEqual(
            repeated.public_payload()["public_execution_sha256"],
            self.result.public_payload()["public_execution_sha256"],
        )
        self.assertEqual(
            repeated.private_manifest()["step_diagnostics"],
            self.result.private_manifest()["step_diagnostics"],
        )

    def test_production_import_free_reference_replay_is_bitwise(self):
        records = tuple(
            ReferenceRecord(
                owner_id=record.owner_ids[0],
                window_id=record.window_id,
                start=record.start,
                end=record.end,
                row_index=record.row_index,
            )
            for record in self.route.selected_mapping.records
        )
        contract = self.route.contract
        reference = replay_owner_srswor_linear_trace(
            raw_train_x=self.prepared.raw_train_x,
            train_y=self.prepared.train_y,
            records=records,
            preprocessing_artifact_path=PREPROCESSOR,
            master_seed=13,
            input_dim=contract.input_dim,
            num_classes=contract.num_classes,
            source_dataset_size=contract.source_dataset_size,
            sample_size=contract.sample_size,
            total_steps=contract.total_steps,
            clip_norm=contract.clip_norm,
            noise_multiplier=contract.noise_multiplier,
            update_denominator=contract.update_denominator,
            learning_rate=contract.learning_rate,
            windows_per_owner_per_step=(
                contract.windows_per_owner_per_step
            ),
            class_weights=contract.class_weights,
        )
        diagnostics = self.result.private_manifest()[
            "step_diagnostics"
        ]
        self.assertEqual(
            tuple(
                tuple(step["sampled_owner_positions"])
                for step in diagnostics
            ),
            reference.sampled_owner_positions,
        )
        self.assertEqual(
            tuple(
                tuple(
                    (
                        item["owner_position"],
                        tuple(item["row_indices"]),
                    )
                    for item in step["selected_rows"]
                )
                for step in diagnostics
            ),
            reference.selected_rows_by_position,
        )
        self.assertEqual(
            tuple(
                step["max_unclipped_owner_norm"]
                for step in diagnostics
            ),
            reference.max_unclipped_norms,
        )
        self.assertEqual(
            tuple(
                step["max_clipped_owner_norm"]
                for step in diagnostics
            ),
            reference.max_clipped_norms,
        )
        for observed, expected in zip(
            self.result.model.parameters(),
            reference.model.parameters(),
        ):
            self.assertTrue(torch.equal(observed, expected))

        source = Path(__file__).with_name(
            "owner_srswor_trace_reference.py"
        )
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.update(
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        )
        self.assertFalse(
            any(
                name == "unitdp" or name.startswith("unitdp.")
                for name in imported
            )
        )

    def test_executor_rejects_one_value_full_data_mutation(self):
        mutated = np.array(
            self.prepared.raw_train_x,
            copy=True,
        )
        mutated[0, 0] = mutated[0, 0] + np.float32(0.25)
        with self.assertRaisesRegex(
            OwnerSrsworExecutionV3Error,
            "registered full-data profile",
        ):
            train_owner_srswor_fixed_v3(
                route=self.route,
                raw_train_x=mutated,
                train_y=self.prepared.train_y,
                raw_test_x=self.prepared.raw_test_x,
                test_y=self.prepared.test_y,
                research_seed=13,
            )

    def test_public_artifact_roundtrip_preserves_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = write_srswor_execution_artifacts_v3(
                self.result,
                tmp,
            )
            loaded = load_model_artifact_v2(written.model_path)
            self.assertEqual(
                _model_state_sha256(loaded),
                _model_state_sha256(self.result.model),
            )
            bundle = json.loads(
                written.public_bundle_path.read_text(encoding="utf-8")
            )
            assert_public_certificate_redacted(bundle)
            self.assertEqual(
                bundle["public_execution"]["payload_sha256"],
                self.result.public_payload()[
                    "public_execution_sha256"
                ],
            )
