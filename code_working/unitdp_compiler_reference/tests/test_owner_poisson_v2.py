import ast
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from unitdp.compiler_v2 import (
    ContractV2Error,
    compile_owner_poisson_contract_v2,
    load_owner_poisson_contract_v2,
)
from unitdp.execution_artifacts_v2 import (
    load_model_artifact_v2,
    write_execution_artifacts_v2,
)
from unitdp.execution_verifier_v2 import (
    V2ArtifactVerificationError,
    verify_run_artifacts_v2,
)
from unitdp.owa_dpsgd import train_owner_poisson_fixed_v2
from unitdp.owner_poisson_v2 import (
    OwnerPoissonExecutionV2Error,
    _ResearchStreams,
    _build_registered_model,
)
from owner_poisson_trace_reference import (
    ReferenceRecord,
    replay_owner_poisson_linear_trace,
)


ROOT = Path(__file__).resolve().parents[1]
UCI_CONFIG = ROOT / "configs" / "v2" / "uci_owner_poisson_v2.yaml"
UCI_PREPROCESSOR = (
    ROOT
    / "configs"
    / "preprocessing"
    / "uci_har_published_train_standard_scaler_v1.json"
)
WISDM_CONFIG = ROOT / "configs" / "v2" / "wisdm_owner_poisson_v2.yaml"
WISDM_PREPROCESSOR = (
    ROOT
    / "configs"
    / "preprocessing"
    / "wisdm_v1_1_train_owners_1_25_stats24_standard_scaler_v1.json"
)
SEPSIS_CONFIG = ROOT / "configs" / "v2" / "sepsis_owner_poisson_v2.yaml"
SEPSIS_PREPROCESSOR = (
    ROOT
    / "configs"
    / "preprocessing"
    / "physionet2019_setA_first400_basic12x6_public_scaler_v1.json"
)
MAPPING_COLUMNS = (
    "scenario",
    "window_id",
    "owner_id",
    "owner_ids",
    "start",
    "end",
    "row_index",
    "label",
)


def mapping_row(
    *,
    window_id: str,
    owner: str,
    row_index: int,
    label: int,
) -> dict[str, str]:
    return {
        "scenario": "v2_test",
        "window_id": window_id,
        "owner_id": owner,
        "owner_ids": owner,
        "start": "0",
        "end": "4",
        "row_index": str(row_index),
        "label": str(label),
    }


def compile_tiny_route(rows: list[dict[str, str]]):
    contract = load_owner_poisson_contract_v2(UCI_CONFIG)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mapping.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(MAPPING_COLUMNS),
            )
            writer.writeheader()
            writer.writerows(rows)
        return compile_owner_poisson_contract_v2(
            contract,
            mapping_path=path,
            preprocessing_artifact_path=UCI_PREPROCESSOR,
            require_executable=True,
        )


def trace_mapping_row(
    *,
    window_id: str,
    owner: str,
    row_index: int,
    label: str,
    start: int,
    patient_file: str | None = None,
) -> dict[str, str]:
    row = {
        "scenario": "v2_trace_correspondence",
        "window_id": window_id,
        "owner_id": owner,
        "owner_ids": owner,
        "start": str(start),
        "end": str(start + 4),
        "row_index": str(row_index),
        "label": label,
    }
    if patient_file is not None:
        row["patient_file"] = patient_file
    return row


def compile_registered_trace_route(
    *,
    config_path: Path,
    preprocessing_artifact_path: Path,
    rows: list[dict[str, str]],
):
    contract = load_owner_poisson_contract_v2(config_path)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mapping.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(rows[0]),
            )
            writer.writeheader()
            writer.writerows(rows)
        return compile_owner_poisson_contract_v2(
            contract,
            mapping_path=path,
            preprocessing_artifact_path=preprocessing_artifact_path,
            require_executable=True,
        )


def registered_trace_fixture(
    *,
    preprocessing_artifact_path: Path,
    input_dim: int,
    label_tokens: tuple[str, ...],
    owner_record_counts: tuple[int, ...],
    include_patient_file: bool,
):
    rows: list[dict[str, str]] = []
    records: list[ReferenceRecord] = []
    labels: list[int] = []
    row_index = 0
    for owner_position, record_count in enumerate(owner_record_counts):
        owner = f"owner_{owner_position}"
        for owner_row in range(record_count):
            label_index = (row_index + owner_position) % len(label_tokens)
            window_id = f"{owner}_window_{owner_row:02d}"
            start = owner_row * 4
            rows.append(
                trace_mapping_row(
                    window_id=window_id,
                    owner=owner,
                    row_index=row_index,
                    label=label_tokens[label_index],
                    start=start,
                    patient_file=(
                        f"{owner}.psv"
                        if include_patient_file
                        else None
                    ),
                )
            )
            records.append(
                ReferenceRecord(
                    owner_id=owner,
                    window_id=window_id,
                    start=start,
                    end=start + 4,
                    row_index=row_index,
                )
            )
            labels.append(label_index)
            row_index += 1

    artifact = json.loads(
        preprocessing_artifact_path.read_text(encoding="utf-8")
    )
    mean = np.asarray(artifact["mean"], dtype=np.float64)
    scale = np.asarray(artifact["scale"], dtype=np.float64)
    row_axis = np.arange(row_index, dtype=np.float64)[:, None]
    column_axis = np.arange(input_dim, dtype=np.float64)[None, :]
    standardized = (
        2.5
        + 0.13 * row_axis
        + 0.07 * ((column_axis % 17.0) - 8.0)
        + 0.03 * ((row_axis * column_axis) % 5.0)
    )
    train_x = (mean + scale * standardized).astype(np.float32)
    train_y = np.asarray(labels, dtype=np.int64)

    test_rows = len(label_tokens)
    test_axis = np.arange(test_rows, dtype=np.float64)[:, None]
    test_standardized = (
        1.5
        + 0.09 * test_axis
        + 0.05 * ((column_axis % 11.0) - 5.0)
    )
    test_x = (mean + scale * test_standardized).astype(np.float32)
    test_y = np.arange(test_rows, dtype=np.int64)
    return (
        rows,
        tuple(records),
        train_x,
        train_y,
        test_x,
        test_y,
    )


def one_owner_data(value: float = 0.0):
    train_x = np.full((1, 561), value, dtype=np.float32)
    train_y = np.asarray([0], dtype=np.int64)
    test_x = np.stack(
        [
            np.full(561, value, dtype=np.float32),
            np.full(561, value + 0.1, dtype=np.float32),
        ]
    )
    test_y = np.asarray([0, 1], dtype=np.int64)
    return train_x, train_y, test_x, test_y


def run_one_owner(seed: int, value: float = 0.0):
    route = compile_tiny_route(
        [mapping_row(window_id="w0", owner="a", row_index=0, label=0)]
    )
    train_x, train_y, test_x, test_y = one_owner_data(value)
    return train_owner_poisson_fixed_v2(
        route=route,
        raw_train_x=train_x,
        train_y=train_y,
        raw_test_x=test_x,
        test_y=test_y,
        research_seed=seed,
    )


class OwnerPoissonExecutorV2Test(unittest.TestCase):
    @staticmethod
    def all_empty_seed(q: float, steps: int) -> int:
        def produces_only_empty_steps(seed: int) -> bool:
            stream = _ResearchStreams.from_master_seed(seed).owner_sampling
            return all(stream.random() >= q for _ in range(steps))

        return next(
            seed
            for seed in range(20000)
            if produces_only_empty_steps(seed)
        )

    def test_fixed_schedule_noise_and_update_hold_on_empty_steps(self):
        route = compile_tiny_route(
            [mapping_row(window_id="w0", owner="a", row_index=0, label=0)]
        )
        q = route.contract.owner_sample_rate
        steps = route.contract.total_steps

        all_empty_seed = self.all_empty_seed(q, steps)
        train_x, train_y, test_x, test_y = one_owner_data()
        result = train_owner_poisson_fixed_v2(
            route=route,
            raw_train_x=train_x,
            train_y=train_y,
            raw_test_x=test_x,
            test_y=test_y,
            research_seed=all_empty_seed,
        )
        public = result.public_payload()
        private = result.private_manifest()
        diagnostics = private["step_diagnostics"]

        self.assertEqual(len(diagnostics), 15)
        self.assertTrue(all(item["empty_sample"] for item in diagnostics))
        self.assertTrue(all(item["noise_applied"] for item in diagnostics))
        self.assertTrue(
            all(item["optimizer_step_applied"] for item in diagnostics)
        )
        self.assertTrue(
            all(item["update_denominator"] == 8.0 for item in diagnostics)
        )
        self.assertTrue(
            all(item["owner_vectors_computed"] == 0 for item in diagnostics)
        )
        self.assertNotEqual(
            private["initial_model_sha256"],
            private["final_model_sha256"],
        )
        self.assertEqual(public["mechanism"]["owner_sample_rate"], 8.0 / 21.0)
        self.assertEqual(public["mechanism"]["total_steps"], 15)
        self.assertEqual(
            public["mechanism"]["empty_step_behavior"],
            "apply_gaussian_update",
        )

    def test_all_empty_execution_matches_exact_noisy_update_equation(self):
        route = compile_tiny_route(
            [mapping_row(window_id="w0", owner="a", row_index=0, label=0)]
        )
        contract = route.contract
        seed = self.all_empty_seed(
            contract.owner_sample_rate,
            contract.total_steps,
        )
        train_x, train_y, test_x, test_y = one_owner_data()
        result = train_owner_poisson_fixed_v2(
            route=route,
            raw_train_x=train_x,
            train_y=train_y,
            raw_test_x=test_x,
            test_y=test_y,
            research_seed=seed,
        )

        streams = _ResearchStreams.from_master_seed(seed)
        expected = _build_registered_model(
            model_id=contract.model_id,
            input_dim=contract.input_dim,
            num_classes=contract.num_classes,
            stream=streams.model_initialization,
        )
        with torch.no_grad():
            for _ in range(contract.total_steps):
                for parameter in expected.parameters():
                    noise = torch.from_numpy(
                        streams.gaussian_noise.normal_array(
                            contract.noise_multiplier * contract.clip_norm,
                            tuple(parameter.shape),
                        )
                    )
                    parameter.add_(
                        noise / contract.update_denominator,
                        alpha=-contract.learning_rate,
                    )

        for observed, target in zip(
            result.model.parameters(),
            expected.parameters(),
        ):
            self.assertTrue(torch.equal(observed, target))

    def test_research_execution_is_deterministic_and_public_output_is_redacted(self):
        first = run_one_owner(13)
        second = run_one_owner(13)
        different = run_one_owner(14)
        first_public = first.public_payload()
        first_private = first.private_manifest()
        encoded = json.dumps(first_public, sort_keys=True)

        self.assertEqual(first_public, second.public_payload())
        self.assertEqual(first_private, second.private_manifest())
        self.assertNotEqual(
            first_public["model_output"]["state_sha256"],
            different.public_payload()["model_output"]["state_sha256"],
        )
        self.assertEqual(
            first_public["release_status"],
            "research_non_release",
        )
        self.assertIn(
            "noncryptographic_research_rng",
            first_public["privacy_claim"]["reason_codes"],
        )
        for forbidden in (
            '"research_seed"',
            "private_execution_binding_sha256",
            "source_mapping",
            "selected_mapping",
            "sampled_owner_count",
            "empty_steps",
            "step_diagnostics",
        ):
            self.assertNotIn(forbidden, encoded)
        self.assertFalse(first_public["rng"]["random_coins_disclosed"])
        self.assertEqual(first_private["research_seed"], 13)

    def test_rng_domains_do_not_consume_each_others_streams(self):
        first = _ResearchStreams.from_master_seed(31)
        expected_noise = first.gaussian_noise.normal_array(1.5, (32,))

        second = _ResearchStreams.from_master_seed(31)
        for _ in range(1000):
            second.owner_sampling.random()
        second.within_owner_selection.generator.choice(
            100,
            size=80,
            replace=False,
        )
        second.model_initialization.uniform_array(-1.0, 1.0, (100,))
        observed_noise = second.gaussian_noise.normal_array(1.5, (32,))

        self.assertTrue(np.array_equal(expected_noise, observed_noise))

    def test_research_sampling_and_gaussian_streams_have_expected_moments(self):
        streams = _ResearchStreams.from_master_seed(91)
        q = 8.0 / 21.0
        draws = np.asarray(
            [
                streams.owner_sampling.random() < q
                for _ in range(100000)
            ],
            dtype=np.float64,
        )
        noise = streams.gaussian_noise.normal_array(1.7, (100000,))

        self.assertLess(abs(float(draws.mean()) - q), 0.005)
        self.assertLess(abs(float(noise.mean())), 0.02)
        self.assertLess(abs(float(noise.std(ddof=0)) - 1.7), 0.02)

    def test_registered_routes_match_independent_multi_trace_replay(self):
        cases = (
            (
                "uci",
                UCI_CONFIG,
                UCI_PREPROCESSOR,
                tuple(str(index) for index in range(6)),
                (1, 4, 10),
                False,
                (13, 97),
            ),
            (
                "wisdm",
                WISDM_CONFIG,
                WISDM_PREPROCESSOR,
                (
                    "Downstairs",
                    "Jogging",
                    "Sitting",
                    "Standing",
                    "Upstairs",
                    "Walking",
                ),
                (1, 4, 10),
                False,
                (13, 97),
            ),
            (
                "sepsis",
                SEPSIS_CONFIG,
                SEPSIS_PREPROCESSOR,
                ("0", "1"),
                (1, 2, 4, 8),
                True,
                (13, 97),
            ),
        )
        total_steps_checked = 0
        final_tensors_checked = 0
        empty_steps = 0
        multi_owner_steps = 0
        owner_vectors_checked = 0
        unclipped_above_bound = 0
        strict_subselection_events = 0
        strict_subselections: set[
            tuple[str, str, tuple[int, ...]]
        ] = set()

        for (
            profile_name,
            config_path,
            preprocessing_artifact_path,
            label_tokens,
            owner_record_counts,
            include_patient_file,
            seeds,
        ) in cases:
            contract = load_owner_poisson_contract_v2(config_path)
            (
                rows,
                records,
                train_x,
                train_y,
                test_x,
                test_y,
            ) = registered_trace_fixture(
                preprocessing_artifact_path=(
                    preprocessing_artifact_path
                ),
                input_dim=contract.input_dim,
                label_tokens=label_tokens,
                owner_record_counts=owner_record_counts,
                include_patient_file=include_patient_file,
            )
            route = compile_registered_trace_route(
                config_path=config_path,
                preprocessing_artifact_path=(
                    preprocessing_artifact_path
                ),
                rows=rows,
            )
            records_per_owner = {
                f"owner_{position}": count
                for position, count in enumerate(owner_record_counts)
            }

            for seed in seeds:
                with self.subTest(profile=profile_name, seed=seed):
                    result = train_owner_poisson_fixed_v2(
                        route=route,
                        raw_train_x=train_x,
                        train_y=train_y,
                        raw_test_x=test_x,
                        test_y=test_y,
                        research_seed=seed,
                    )
                    diagnostics = result.private_manifest()[
                        "step_diagnostics"
                    ]
                    reference = replay_owner_poisson_linear_trace(
                        raw_train_x=train_x,
                        train_y=train_y,
                        records=records,
                        preprocessing_artifact_path=(
                            preprocessing_artifact_path
                        ),
                        master_seed=seed,
                        input_dim=contract.input_dim,
                        num_classes=contract.num_classes,
                        owner_sample_rate=(
                            contract.owner_sample_rate
                        ),
                        total_steps=contract.total_steps,
                        clip_norm=contract.clip_norm,
                        noise_multiplier=contract.noise_multiplier,
                        update_denominator=(
                            contract.update_denominator
                        ),
                        learning_rate=contract.learning_rate,
                        windows_per_owner_per_step=(
                            contract.windows_per_owner_per_step
                        ),
                        class_weights=contract.class_weights,
                    )

                    self.assertEqual(
                        tuple(
                            item["sampled_owner_count"]
                            for item in diagnostics
                        ),
                        reference.sampled_owner_counts,
                    )
                    self.assertEqual(
                        tuple(
                            item["owner_vectors_computed"]
                            for item in diagnostics
                        ),
                        reference.owner_vectors_computed,
                    )
                    self.assertEqual(
                        tuple(
                            item["max_unclipped_owner_norm"]
                            for item in diagnostics
                        ),
                        reference.max_unclipped_norms,
                    )
                    self.assertEqual(
                        tuple(
                            item["max_clipped_owner_norm"]
                            for item in diagnostics
                        ),
                        reference.max_clipped_norms,
                    )
                    self.assertTrue(
                        all(
                            item["update_denominator"]
                            == contract.update_denominator
                            for item in diagnostics
                        )
                    )
                    self.assertTrue(
                        all(
                            item["max_clipped_owner_norm"]
                            <= contract.clip_norm
                            for item in diagnostics
                        )
                    )
                    for observed, expected in zip(
                        result.model.parameters(),
                        reference.model.parameters(),
                    ):
                        self.assertTrue(
                            torch.equal(observed, expected)
                        )
                        final_tensors_checked += 1

                    total_steps_checked += len(diagnostics)
                    empty_steps += sum(
                        item["empty_sample"]
                        for item in diagnostics
                    )
                    multi_owner_steps += sum(
                        item["sampled_owner_count"] > 1
                        for item in diagnostics
                    )
                    owner_vectors_checked += sum(
                        item["owner_vectors_computed"]
                        for item in diagnostics
                    )
                    unclipped_above_bound += sum(
                        item["max_unclipped_owner_norm"]
                        > contract.clip_norm
                        for item in diagnostics
                    )
                    for step_selection in (
                        reference.selected_rows_by_owner
                    ):
                        for owner, indices in step_selection:
                            if (
                                records_per_owner[owner]
                                > contract.windows_per_owner_per_step
                            ):
                                strict_subselection_events += 1
                                strict_subselections.add(
                                    (
                                        profile_name,
                                        owner,
                                        indices,
                                    )
                                )

        self.assertEqual(total_steps_checked, 170)
        self.assertEqual(final_tensors_checked, 12)
        self.assertEqual(empty_steps, 93)
        self.assertEqual(multi_owner_steps, 20)
        self.assertEqual(owner_vectors_checked, 100)
        self.assertEqual(strict_subselection_events, 20)
        self.assertEqual(len(strict_subselections), 19)
        self.assertGreater(unclipped_above_bound, 0)

        reference_source = (
            Path(__file__).with_name(
                "owner_poisson_trace_reference.py"
            )
        )
        tree = ast.parse(reference_source.read_text(encoding="utf-8"))
        imported_modules = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_modules.update(
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        )
        self.assertFalse(
            any(
                name == "unitdp" or name.startswith("unitdp.")
                for name in imported_modules
            )
        )

    def test_owner_removal_keeps_public_schedule_and_accountant_fixed(self):
        full_route = compile_tiny_route(
            [
                mapping_row(
                    window_id="wa",
                    owner="a",
                    row_index=0,
                    label=0,
                ),
                mapping_row(
                    window_id="wb",
                    owner="b",
                    row_index=1,
                    label=1,
                ),
            ]
        )
        removed_route = compile_tiny_route(
            [
                mapping_row(
                    window_id="wb",
                    owner="b",
                    row_index=0,
                    label=1,
                )
            ]
        )
        test_x = np.zeros((2, 561), dtype=np.float32)
        test_y = np.asarray([0, 1], dtype=np.int64)
        full = train_owner_poisson_fixed_v2(
            route=full_route,
            raw_train_x=np.zeros((2, 561), dtype=np.float32),
            train_y=np.asarray([0, 1], dtype=np.int64),
            raw_test_x=test_x,
            test_y=test_y,
            research_seed=17,
        ).public_payload()
        removed = train_owner_poisson_fixed_v2(
            route=removed_route,
            raw_train_x=np.zeros((1, 561), dtype=np.float32),
            train_y=np.asarray([1], dtype=np.int64),
            raw_test_x=test_x,
            test_y=test_y,
            research_seed=17,
        ).public_payload()

        self.assertEqual(full["route"]["public_plan_sha256"], removed["route"]["public_plan_sha256"])
        self.assertEqual(full["mechanism"], removed["mechanism"])
        self.assertEqual(full["accountant"], removed["accountant"])

    def test_data_and_route_mismatches_fail_closed(self):
        route = compile_tiny_route(
            [mapping_row(window_id="w0", owner="a", row_index=0, label=0)]
        )
        train_x, train_y, test_x, test_y = one_owner_data()

        bad_calls = (
            {
                "raw_train_x": np.zeros((2, 561), dtype=np.float32),
                "train_y": np.asarray([0, 0], dtype=np.int64),
            },
            {
                "raw_train_x": np.zeros((1, 560), dtype=np.float32),
                "train_y": train_y,
            },
            {
                "raw_train_x": train_x,
                "train_y": train_y.astype(np.float32),
            },
            {
                "raw_train_x": np.full(
                    (1, 561),
                    np.nan,
                    dtype=np.float32,
                ),
                "train_y": train_y,
            },
            {
                "raw_train_x": train_x,
                "train_y": np.asarray([6], dtype=np.int64),
            },
            {
                "raw_train_x": train_x,
                "train_y": np.asarray([1], dtype=np.int64),
            },
        )
        for overrides in bad_calls:
            with self.subTest(overrides=tuple(overrides)):
                arguments = {
                    "route": route,
                    "raw_train_x": train_x,
                    "train_y": train_y,
                    "raw_test_x": test_x,
                    "test_y": test_y,
                    "research_seed": 13,
                    **overrides,
                }
                with self.assertRaises(OwnerPoissonExecutionV2Error):
                    train_owner_poisson_fixed_v2(**arguments)

        with self.assertRaises(OwnerPoissonExecutionV2Error):
            train_owner_poisson_fixed_v2(
                route="not-a-compiled-route",
                raw_train_x=train_x,
                train_y=train_y,
                raw_test_x=test_x,
                test_y=test_y,
                research_seed=13,
            )

    def test_mutated_compiled_route_and_model_are_detected(self):
        route = compile_tiny_route(
            [mapping_row(window_id="w0", owner="a", row_index=0, label=0)]
        )
        route.contract.public_protocol["test_owners"] = 8
        train_x, train_y, test_x, test_y = one_owner_data()
        with self.assertRaises(OwnerPoissonExecutionV2Error):
            train_owner_poisson_fixed_v2(
                route=route,
                raw_train_x=train_x,
                train_y=train_y,
                raw_test_x=test_x,
                test_y=test_y,
                research_seed=13,
            )

        result = run_one_owner(13)
        result.model.weight.data.zero_()
        with self.assertRaises(OwnerPoissonExecutionV2Error):
            result.public_payload()

    def test_research_seed_validation_is_strict(self):
        route = compile_tiny_route(
            [mapping_row(window_id="w0", owner="a", row_index=0, label=0)]
        )
        train_x, train_y, test_x, test_y = one_owner_data()
        for bad_seed in (-1, True, 1.5, 2**63):
            with self.subTest(seed=bad_seed):
                with self.assertRaises(OwnerPoissonExecutionV2Error):
                    train_owner_poisson_fixed_v2(
                        route=route,
                        raw_train_x=train_x,
                        train_y=train_y,
                        raw_test_x=test_x,
                        test_y=test_y,
                        research_seed=bad_seed,
                    )

    def test_public_artifact_bundle_is_deterministic_and_private_is_opt_in(self):
        result = run_one_owner(13)
        with tempfile.TemporaryDirectory() as tmp:
            first_dir = Path(tmp) / "first"
            second_dir = Path(tmp) / "second"
            private_dir = Path(tmp) / "private"
            first = write_execution_artifacts_v2(result, first_dir)
            second = write_execution_artifacts_v2(result, second_dir)

            self.assertIsNone(first.private_manifest_path)
            self.assertFalse(
                (first_dir / "private_execution_manifest_v2.json").exists()
            )
            self.assertEqual(
                first.model_file_sha256,
                second.model_file_sha256,
            )
            self.assertEqual(
                first.public_execution_file_sha256,
                second.public_execution_file_sha256,
            )
            self.assertEqual(
                first.public_bundle_payload_sha256,
                second.public_bundle_payload_sha256,
            )
            loaded = load_model_artifact_v2(first.model_path)
            for observed, expected in zip(
                loaded.parameters(),
                result.model.parameters(),
            ):
                self.assertTrue(torch.equal(observed, expected))

            public_text = (
                first.public_execution_path.read_text(encoding="utf-8")
                + first.public_bundle_path.read_text(encoding="utf-8")
            )
            self.assertNotIn("research_seed", public_text)
            self.assertNotIn("step_diagnostics", public_text)
            with self.assertRaises(FileExistsError):
                write_execution_artifacts_v2(result, first_dir)

            private = write_execution_artifacts_v2(
                result,
                private_dir,
                include_private_manifest=True,
            )
            self.assertIsNotNone(private.private_manifest_path)
            assert private.private_manifest_path is not None
            self.assertIn(
                '"research_seed": 13',
                private.private_manifest_path.read_text(encoding="utf-8"),
            )

    def test_tampered_model_artifact_is_rejected(self):
        result = run_one_owner(13)
        with tempfile.TemporaryDirectory() as tmp:
            written = write_execution_artifacts_v2(result, tmp)
            raw = json.loads(
                written.model_path.read_text(encoding="utf-8")
            )
            raw["input_dim"] = 560
            written.model_path.write_text(
                json.dumps(raw),
                encoding="utf-8",
            )
            with self.assertRaises(OwnerPoissonExecutionV2Error):
                load_model_artifact_v2(written.model_path)

    def test_public_run_verifier_checks_all_links_and_fails_on_tampering(self):
        result = run_one_owner(13)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_001"
            written = write_execution_artifacts_v2(result, run_dir)
            verified = verify_run_artifacts_v2(run_dir)
            self.assertEqual(
                verified.public_execution_sha256,
                result.public_payload()["public_execution_sha256"],
            )
            self.assertEqual(
                verified.model_state_sha256,
                result.public_payload()["model_output"]["state_sha256"],
            )
            self.assertEqual(
                verified.execution_source_bundle_sha256,
                result.public_payload()["implementation"][
                    "source_bundle_sha256"
                ],
            )

            bundle = json.loads(
                written.public_bundle_path.read_text(encoding="utf-8")
            )
            bundle["model_artifact"]["file_sha256"] = "0" * 64
            written.public_bundle_path.write_text(
                json.dumps(bundle),
                encoding="utf-8",
            )
            with self.assertRaises(V2ArtifactVerificationError):
                verify_run_artifacts_v2(run_dir)


if __name__ == "__main__":
    unittest.main()
