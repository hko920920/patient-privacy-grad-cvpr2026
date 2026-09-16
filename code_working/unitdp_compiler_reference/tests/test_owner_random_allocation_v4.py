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
from unitdp.compiler_allocation_v4 import (
    compile_owner_random_allocation_contract_v4,
    load_owner_random_allocation_contract_v4,
)
from unitdp.owner_poisson_v2 import _model_state_sha256
from unitdp.owner_random_allocation_v4 import (
    OwnerRandomAllocationExecutionV4Error,
    _AllocationResearchStreamsV4,
    build_owner_allocation_schedule_v4,
    train_owner_random_allocation_fixed_v4,
)
from unitdp.release_artifacts import assert_public_certificate_redacted
from owner_random_allocation_trace_reference import (
    ReferenceRecord,
    replay_owner_random_allocation_linear_trace,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT.parent / "DP-SGD" / "data"
DATA_ROOT = Path(
    os.environ.get("UNITDP_DATA_ROOT", str(DEFAULT_DATA_ROOT))
)
UCI_ROOT = DATA_ROOT / "uci_har" / "extracted" / "UCI HAR Dataset"
CONFIG = (
    ROOT
    / "configs"
    / "v4"
    / "uci_owner_random_allocation_v4.yaml"
)
PREPROCESSOR = (
    ROOT
    / "configs"
    / "preprocessing"
    / "uci_har_published_train_standard_scaler_v1.json"
)
EMPTY_STEP_SEED = 6761


class OwnerRandomAllocationExecutorV4Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (UCI_ROOT / "train" / "X_train.txt").is_file():
            raise unittest.SkipTest(
                "Raw UCI HAR files are not available for executor evidence"
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
        cls.route = compile_owner_random_allocation_contract_v4(
            load_owner_random_allocation_contract_v4(CONFIG),
            mapping_path=mapping,
            preprocessing_artifact_path=PREPROCESSOR,
            require_executable=True,
        )
        cls.result = train_owner_random_allocation_fixed_v4(
            route=cls.route,
            raw_train_x=cls.prepared.raw_train_x,
            train_y=cls.prepared.train_y,
            raw_test_x=cls.prepared.raw_test_x,
            test_y=cls.prepared.test_y,
            research_seed=EMPTY_STEP_SEED,
        )

    @classmethod
    def tearDownClass(cls):
        cls._mapping_tmp.cleanup()

    def test_exact_k_distinct_assignments_and_empty_step_noise(self):
        contract = self.route.contract
        private = self.result.private_manifest()
        owner_rows = private["local_steps_by_epoch_and_owner"]
        self.assertEqual(len(owner_rows), contract.num_epochs)
        self.assertTrue(
            all(
                len(epoch) == contract.source_dataset_size
                for epoch in owner_rows
            )
        )
        for epoch in owner_rows:
            for selected in epoch:
                self.assertEqual(
                    len(selected),
                    contract.num_selected_steps_per_owner,
                )
                self.assertEqual(
                    len(set(selected)),
                    contract.num_selected_steps_per_owner,
                )
                self.assertTrue(
                    all(0 <= step < contract.num_steps for step in selected)
                )

        diagnostics = private["step_diagnostics"]
        self.assertEqual(len(diagnostics), contract.total_steps)
        reconstructed = [[] for _ in range(contract.total_steps)]
        for epoch_index, epoch in enumerate(owner_rows):
            for owner_position, selected in enumerate(epoch):
                for local_step in selected:
                    reconstructed[
                        epoch_index * contract.num_steps + local_step
                    ].append(owner_position)
        self.assertEqual(
            [row["assigned_owner_positions"] for row in diagnostics],
            reconstructed,
        )
        empty = [
            row for row in diagnostics if row["assigned_owner_count"] == 0
        ]
        self.assertEqual([row["step"] for row in empty], [12])
        for row in diagnostics:
            self.assertEqual(
                row["owner_vectors_computed"],
                row["assigned_owner_count"],
            )
            self.assertLessEqual(
                row["max_clipped_owner_norm"],
                contract.clip_norm,
            )
            self.assertEqual(row["noise_parameter_tensors"], 2)
            self.assertIs(row["optimizer_step_applied"], True)
        self.assertEqual(empty[0]["owner_vectors_computed"], 0)
        self.assertEqual(empty[0]["max_unclipped_owner_norm"], 0.0)
        self.assertEqual(empty[0]["max_clipped_owner_norm"], 0.0)

    def test_production_import_free_reference_is_bitwise_exact(self):
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
        reference = replay_owner_random_allocation_linear_trace(
            raw_train_x=self.prepared.raw_train_x,
            train_y=self.prepared.train_y,
            records=records,
            preprocessing_artifact_path=PREPROCESSOR,
            master_seed=EMPTY_STEP_SEED,
            input_dim=contract.input_dim,
            num_classes=contract.num_classes,
            source_dataset_size=contract.source_dataset_size,
            num_steps=contract.num_steps,
            num_selected_steps_per_owner=(
                contract.num_selected_steps_per_owner
            ),
            num_epochs=contract.num_epochs,
            clip_norm=contract.clip_norm,
            noise_multiplier=contract.noise_multiplier,
            update_denominator=contract.update_denominator,
            learning_rate=contract.learning_rate,
            windows_per_owner_per_step=(
                contract.windows_per_owner_per_step
            ),
            class_weights=contract.class_weights,
        )
        private = self.result.private_manifest()
        diagnostics = private["step_diagnostics"]
        self.assertEqual(
            tuple(
                tuple(row["assigned_owner_positions"])
                for row in diagnostics
            ),
            reference.assigned_owner_positions,
        )
        self.assertEqual(
            tuple(
                tuple(tuple(value) for value in epoch)
                for epoch in private[
                    "local_steps_by_epoch_and_owner"
                ]
            ),
            reference.local_steps_by_epoch_and_owner,
        )
        self.assertEqual(
            tuple(
                tuple(
                    (
                        item["owner_position"],
                        tuple(item["row_indices"]),
                    )
                    for item in row["selected_rows"]
                )
                for row in diagnostics
            ),
            reference.selected_rows_by_position,
        )
        self.assertEqual(
            tuple(
                row["max_unclipped_owner_norm"]
                for row in diagnostics
            ),
            reference.max_unclipped_norms,
        )
        self.assertEqual(
            tuple(
                row["max_clipped_owner_norm"] for row in diagnostics
            ),
            reference.max_clipped_norms,
        )
        for observed, expected in zip(
            self.result.model.parameters(),
            reference.model.parameters(),
        ):
            self.assertTrue(torch.equal(observed, expected))

        source = Path(__file__).with_name(
            "owner_random_allocation_trace_reference.py"
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

    def test_public_record_redacts_seed_and_allocation_trace(self):
        public = self.result.public_payload()
        assert_public_certificate_redacted(public)
        encoded = json.dumps(public, sort_keys=True)
        for forbidden in (
            "research_seed",
            "assigned_owner_positions",
            "local_steps_by_epoch_and_owner",
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
        self.assertAlmostEqual(
            public["accountant"]["epsilon"],
            8.0,
            places=9,
        )

    def test_schedule_replays_and_has_independent_owner_joint_support(self):
        first = _AllocationResearchStreamsV4.from_master_seed(29)
        second = _AllocationResearchStreamsV4.from_master_seed(29)
        left = build_owner_allocation_schedule_v4(
            num_owners=3,
            num_steps=5,
            num_selected_steps_per_owner=2,
            num_epochs=2,
            generator=first.owner_step_allocation.generator,
        )
        right = build_owner_allocation_schedule_v4(
            num_owners=3,
            num_steps=5,
            num_selected_steps_per_owner=2,
            num_epochs=2,
            generator=second.owner_step_allocation.generator,
        )
        self.assertEqual(left, right)

        joint = np.zeros((4, 4), dtype=np.int64)
        for seed in range(4096):
            streams = _AllocationResearchStreamsV4.from_master_seed(seed)
            schedule = build_owner_allocation_schedule_v4(
                num_owners=2,
                num_steps=4,
                num_selected_steps_per_owner=1,
                num_epochs=1,
                generator=streams.owner_step_allocation.generator,
            )
            rows = schedule.local_steps_by_epoch_and_owner[0]
            joint[rows[0][0], rows[1][0]] += 1
        self.assertTrue(np.all(joint > 0))
        self.assertLess(
            int(np.max(np.abs(joint - 256))),
            80,
        )

    def test_executor_rejects_one_value_full_data_mutation(self):
        mutated = np.array(self.prepared.raw_train_x, copy=True)
        mutated[0, 0] = mutated[0, 0] + np.float32(0.25)
        with self.assertRaisesRegex(
            OwnerRandomAllocationExecutionV4Error,
            "registered full-data profile",
        ):
            train_owner_random_allocation_fixed_v4(
                route=self.route,
                raw_train_x=mutated,
                train_y=self.prepared.train_y,
                raw_test_x=self.prepared.raw_test_x,
                test_y=self.prepared.test_y,
                research_seed=EMPTY_STEP_SEED,
            )

    def test_model_integrity_tamper_is_rejected(self):
        parameter = next(self.result.model.parameters())
        original = parameter.detach().clone()
        with torch.no_grad():
            parameter.add_(1.0)
        try:
            with self.assertRaises(
                OwnerRandomAllocationExecutionV4Error
            ):
                self.result.public_payload()
        finally:
            with torch.no_grad():
                parameter.copy_(original)
        self.assertEqual(
            _model_state_sha256(self.result.model),
            self.result.public_payload()["model_output"]["state_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
