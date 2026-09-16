import copy
import csv
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import yaml

from unitdp.compiler_v2 import (
    ContractV2Error,
    ExecutorNotReadyError,
    compile_owner_poisson_contract_v2,
    load_owner_poisson_contract_v2,
    parse_owner_poisson_contract_v2,
)
from unitdp.source_bundle_v2 import (
    EXECUTION_SOURCE_FILE_SET_V2,
    execution_source_bundle_sha256_v2,
    execution_source_bundle_v2,
)


ROOT = Path(__file__).resolve().parents[1]
UCI_CONFIG = ROOT / "configs" / "v2" / "uci_owner_poisson_v2.yaml"
UCI_MAPPING = (
    ROOT
    / "reports"
    / "uci_public_preprocessing_smoke_20260724"
    / "uci_train_mapping.csv"
)
UCI_PREPROCESSOR = (
    ROOT
    / "configs"
    / "preprocessing"
    / "uci_har_published_train_standard_scaler_v1.json"
)
WISDM_PREPROCESSOR = (
    ROOT
    / "configs"
    / "preprocessing"
    / "wisdm_v1_1_train_owners_1_25_stats24_standard_scaler_v1.json"
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

CASES = (
    (
        "uci",
        ROOT / "configs" / "v2" / "uci_owner_poisson_v2.yaml",
        UCI_MAPPING,
        UCI_PREPROCESSOR,
        336,
    ),
    (
        "wisdm",
        ROOT / "configs" / "v2" / "wisdm_owner_poisson_v2.yaml",
        ROOT
        / "reports"
        / "wisdm_public_protocol_smoke_20260724"
        / "wisdm_train_mapping.csv",
        WISDM_PREPROCESSOR,
        400,
    ),
    (
        "sepsis",
        ROOT / "configs" / "v2" / "sepsis_owner_poisson_v2.yaml",
        ROOT
        / "reports"
        / "sepsis_public_protocol_smoke_20260724"
        / "sepsis_train_mapping.csv",
        ROOT
        / "configs"
        / "preprocessing"
        / "physionet2019_setA_first400_basic12x6_public_scaler_v1.json",
        1507,
    ),
)


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def uci_raw_contract() -> dict[str, object]:
    raw = yaml.safe_load(UCI_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def write_mapping(
    directory: str,
    rows: list[dict[str, str]],
    *,
    columns: tuple[str, ...] = MAPPING_COLUMNS,
) -> Path:
    path = Path(directory) / "mapping.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)
    return path


def mapping_rows() -> list[dict[str, str]]:
    return [
        {
            "scenario": "test",
            "window_id": "w0",
            "owner_id": "a",
            "owner_ids": "a",
            "start": "0",
            "end": "4",
            "row_index": "0",
            "label": "0",
        },
        {
            "scenario": "test",
            "window_id": "w1",
            "owner_id": "b",
            "owner_ids": "b",
            "start": "0",
            "end": "4",
            "row_index": "1",
            "label": "1",
        },
    ]


class StrictCompilerV2Test(unittest.TestCase):
    def test_registered_benchmark_contracts_compile_with_research_executor(self):
        for name, config_path, mapping_path, preprocessor_path, selected in CASES:
            with self.subTest(dataset=name):
                contract = load_owner_poisson_contract_v2(config_path)
                compiled = compile_owner_poisson_contract_v2(
                    contract,
                    mapping_path=mapping_path,
                    preprocessing_artifact_path=preprocessor_path,
                )
                plan = compiled.public_plan()
                plan_without_hash = dict(plan)
                reported_hash = plan_without_hash.pop("public_plan_sha256")

                self.assertEqual(
                    compiled.status,
                    "validated_contract_research_executor_ready",
                )
                self.assertTrue(compiled.execution_ready)
                self.assertEqual(len(compiled.selected_mapping.records), selected)
                self.assertLessEqual(
                    abs(
                        compiled.accountant_epsilon_opacus
                        - compiled.accountant_epsilon_dp_accounting
                    ),
                    1e-10,
                )
                self.assertLessEqual(
                    compiled.accountant_epsilon_opacus,
                    contract.target_epsilon + 1e-10,
                )
                self.assertEqual(reported_hash, canonical_sha256(plan_without_hash))
                self.assertEqual(
                    plan["execution_source_bundle_sha256"],
                    compiled.execution_source_bundle_sha256,
                )
                source_bundle = dict(execution_source_bundle_v2())
                self.assertEqual(
                    set(source_bundle),
                    EXECUTION_SOURCE_FILE_SET_V2,
                )
                self.assertEqual(
                    execution_source_bundle_sha256_v2(source_bundle),
                    compiled.execution_source_bundle_sha256,
                )

                encoded = json.dumps(plan, sort_keys=True)
                self.assertNotIn(str(mapping_path), encoded)
                self.assertNotIn(compiled.private_source_mapping_sha256, encoded)
                self.assertNotIn(
                    compiled.private_source_mapping_canonical_sha256,
                    encoded,
                )
                self.assertNotIn(compiled.private_selected_mapping_sha256, encoded)
                self.assertNotIn(
                    compiled.private_preprocessor_state_sha256,
                    encoded,
                )
                self.assertNotIn(
                    compiled.private_execution_binding_sha256,
                    encoded,
                )
                self.assertNotIn("source_mapping", encoded)
                self.assertNotIn("selected_mapping", encoded)
                self.assertNotIn('"seed"', encoded)

    def test_unknown_fields_wrong_units_and_profile_drift_are_rejected(self):
        mutations = {
            "top-level epochs": lambda raw: raw.__setitem__("epochs", 5),
            "nested typo": lambda raw: raw["mechanism"].__setitem__(
                "total_step", 15
            ),
            "replace adjacency": lambda raw: raw["privacy"].__setitem__(
                "adjacency", "replace_one_owner"
            ),
            "window sampling unit": lambda raw: raw["mechanism"].__setitem__(
                "sampling_unit", "window"
            ),
            "window accounting unit": lambda raw: raw["mechanism"].__setitem__(
                "accounting_unit", "window"
            ),
            "fixed-size sampler": lambda raw: raw["mechanism"].__setitem__(
                "sampler", "fixed_size_owner"
            ),
            "observed schedule source": lambda raw: raw["mechanism"].__setitem__(
                "parameter_source", "observed_owner_count"
            ),
            "zero q": lambda raw: raw["mechanism"].__setitem__(
                "owner_sample_rate", 0.0
            ),
            "q above one": lambda raw: raw["mechanism"].__setitem__(
                "owner_sample_rate", 1.01
            ),
            "numeric string q": lambda raw: raw["mechanism"].__setitem__(
                "owner_sample_rate", "0.38095238095238093"
            ),
            "zero steps": lambda raw: raw["mechanism"].__setitem__(
                "total_steps", 0
            ),
            "fractional steps": lambda raw: raw["mechanism"].__setitem__(
                "total_steps", 15.0
            ),
            "negative clip": lambda raw: raw["mechanism"].__setitem__(
                "clip_norm", -1.0
            ),
            "zero noise": lambda raw: raw["mechanism"].__setitem__(
                "noise_multiplier", 0.0
            ),
            "zero denominator": lambda raw: raw["mechanism"].__setitem__(
                "update_denominator", 0.0
            ),
            "profile q drift": lambda raw: raw["mechanism"].__setitem__(
                "owner_sample_rate", 0.4
            ),
            "profile step drift": lambda raw: raw["mechanism"].__setitem__(
                "total_steps", 16
            ),
            "profile noise drift": lambda raw: raw["mechanism"].__setitem__(
                "noise_multiplier", 1.3
            ),
            "skip empty step": lambda raw: raw["mechanism"].__setitem__(
                "empty_step_behavior", "skip_update"
            ),
            "realized denominator": lambda raw: raw["mechanism"].__setitem__(
                "update_denominator", 7.0
            ),
            "wrong owner contribution": lambda raw: raw["mechanism"].__setitem__(
                "per_owner_contribution", "per_window_vectors"
            ),
            "wrong aggregation": lambda raw: raw["mechanism"].__setitem__(
                "owner_aggregation", "sum"
            ),
            "wrong accountant": lambda raw: raw["accountant"].__setitem__(
                "id", "another_accountant"
            ),
            "per-step budget above cap": lambda raw: raw[
                "contribution_policy"
            ].__setitem__("windows_per_owner_per_step", 17),
            "cross-owner policy": lambda raw: raw[
                "contribution_policy"
            ].__setitem__("scope", "global"),
            "private scaler": lambda raw: raw["data"]["preprocessing"].__setitem__(
                "fit_on_protected_data", True
            ),
            "stale executor": lambda raw: raw["execution"].__setitem__(
                "implementation_id", "unitdp.owa_dpsgd.train_owa_dpsgd:v1"
            ),
            "public random coins": lambda raw: raw["execution"].__setitem__(
                "random_coins_public", True
            ),
            "research RNG release": lambda raw: raw["execution"].__setitem__(
                "profile", "privacy_release"
            ),
            "protocol owner-count drift": lambda raw: raw["data"][
                "public_protocol"
            ].__setitem__("train_owners", 20),
            "unregistered profile": lambda raw: raw["data"].__setitem__(
                "benchmark_profile_id", "custom"
            ),
            "profile digest drift": lambda raw: raw["data"].__setitem__(
                "benchmark_profile_sha256", "0" * 64
            ),
            "learning-rate drift": lambda raw: raw["optimization"].__setitem__(
                "learning_rate", 0.051
            ),
            "invalid class weights": lambda raw: raw["optimization"].__setitem__(
                "class_weights", [1.0]
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(mutation=name):
                raw = copy.deepcopy(uci_raw_contract())
                mutate(raw)
                with self.assertRaises(ContractV2Error):
                    parse_owner_poisson_contract_v2(raw)

    def test_release_profile_requires_system_csprng_but_is_not_yet_executable(self):
        raw = uci_raw_contract()
        raw["execution"]["profile"] = "privacy_release"
        raw["execution"]["rng_backend"] = "system_csprng"
        contract = parse_owner_poisson_contract_v2(raw)

        with self.assertRaises(ExecutorNotReadyError):
            compile_owner_poisson_contract_v2(
                contract,
                mapping_path=UCI_MAPPING,
                preprocessing_artifact_path=UCI_PREPROCESSOR,
                require_executable=True,
            )

    def test_duplicate_yaml_and_json_keys_are_rejected(self):
        yaml_text = UCI_CONFIG.read_text(encoding="utf-8").replace(
            "  total_steps: 15\n",
            "  total_steps: 15\n  total_steps: 15\n",
            1,
        )
        json_text = json.dumps(uci_raw_contract()).replace(
            '"route_id": "owa_owner_poisson_rdp_add_remove_v2",',
            (
                '"route_id": "owa_owner_poisson_rdp_add_remove_v2", '
                '"route_id": "owa_owner_poisson_rdp_add_remove_v2",'
            ),
            1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "duplicate.yaml"
            json_path = Path(tmp) / "duplicate.json"
            yaml_path.write_text(yaml_text, encoding="utf-8")
            json_path.write_text(json_text, encoding="utf-8")

            with self.assertRaisesRegex(ContractV2Error, "Duplicate YAML key"):
                load_owner_poisson_contract_v2(yaml_path)
            with self.assertRaisesRegex(ContractV2Error, "Duplicate JSON key"):
                load_owner_poisson_contract_v2(json_path)

    def test_mapping_removal_does_not_change_public_mechanism_or_plan(self):
        contract = load_owner_poisson_contract_v2(UCI_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            full_mapping = write_mapping(tmp, mapping_rows())
            full = compile_owner_poisson_contract_v2(
                contract,
                mapping_path=full_mapping,
                preprocessing_artifact_path=UCI_PREPROCESSOR,
            )
            one_owner_dir = Path(tmp) / "removed"
            one_owner_dir.mkdir()
            surviving_row = copy.deepcopy(mapping_rows()[1])
            surviving_row["row_index"] = "0"
            removed_mapping = write_mapping(str(one_owner_dir), [surviving_row])
            removed = compile_owner_poisson_contract_v2(
                contract,
                mapping_path=removed_mapping,
                preprocessing_artifact_path=UCI_PREPROCESSOR,
            )

        self.assertEqual(full.contract.public_payload(), removed.contract.public_payload())
        self.assertEqual(full.public_plan(), removed.public_plan())
        self.assertEqual(full.accountant_epsilon_opacus, removed.accountant_epsilon_opacus)
        self.assertNotEqual(
            full.private_source_mapping_sha256,
            removed.private_source_mapping_sha256,
        )

    def test_direct_dataclass_mutation_cannot_bypass_validation(self):
        contract = load_owner_poisson_contract_v2(UCI_CONFIG)
        replaced = replace(contract, owner_sample_rate=0.25)
        with self.assertRaises(ContractV2Error):
            compile_owner_poisson_contract_v2(
                replaced,
                mapping_path=UCI_MAPPING,
                preprocessing_artifact_path=UCI_PREPROCESSOR,
            )
        contract.preprocessing_binding["fit_on_protected_data"] = True
        with self.assertRaises(ContractV2Error):
            compile_owner_poisson_contract_v2(
                contract,
                mapping_path=UCI_MAPPING,
                preprocessing_artifact_path=UCI_PREPROCESSOR,
            )

    def test_compiled_route_detects_post_compile_mutation(self):
        contract = load_owner_poisson_contract_v2(UCI_CONFIG)

        def compile_fresh():
            return compile_owner_poisson_contract_v2(
                contract,
                mapping_path=UCI_MAPPING,
                preprocessing_artifact_path=UCI_PREPROCESSOR,
            )

        compiled = compile_fresh()
        compiled.assert_private_execution_integrity()
        with self.assertRaises(ValueError):
            compiled.preprocessor.mean[0] = 0.0

        compiled = compile_fresh()
        compiled.selected_mapping.records.pop()
        with self.assertRaisesRegex(ContractV2Error, "selected mapping"):
            compiled.assert_private_execution_integrity()

        compiled = compile_fresh()
        compiled.policy.max_windows_per_owner = 15
        with self.assertRaisesRegex(ContractV2Error, "policy"):
            compiled.assert_private_execution_integrity()

        compiled = compile_fresh()
        compiled.contract.public_protocol["test_owners"] = 8
        with self.assertRaises(ContractV2Error):
            compiled.assert_private_execution_integrity()

        compiled = replace(
            compile_fresh(),
            execution_source_bundle_sha256="0" * 64,
        )
        with self.assertRaisesRegex(ContractV2Error, "source bundle"):
            compiled.assert_private_execution_integrity()

    def test_malformed_mapping_variants_are_rejected(self):
        contract = load_owner_poisson_contract_v2(UCI_CONFIG)
        variants: dict[str, tuple[list[dict[str, str]], tuple[str, ...]]] = {}

        decimal_index = mapping_rows()
        decimal_index[1]["row_index"] = "1.5"
        variants["decimal row index"] = (decimal_index, MAPPING_COLUMNS)

        duplicate_index = mapping_rows()
        duplicate_index[1]["row_index"] = "0"
        variants["duplicate row index"] = (duplicate_index, MAPPING_COLUMNS)

        gapped_index = mapping_rows()
        gapped_index[1]["row_index"] = "2"
        variants["gapped row index"] = (gapped_index, MAPPING_COLUMNS)

        duplicate_id = mapping_rows()
        duplicate_id[1]["window_id"] = "w0"
        variants["duplicate window id"] = (duplicate_id, MAPPING_COLUMNS)

        multi_owner = mapping_rows()
        multi_owner[0]["owner_ids"] = "a;b"
        variants["multi-owner row"] = (multi_owner, MAPPING_COLUMNS)

        blank_label = mapping_rows()
        blank_label[0]["label"] = ""
        variants["blank label"] = (blank_label, MAPPING_COLUMNS)

        unknown_column_rows = mapping_rows()
        for row in unknown_column_rows:
            row["extra"] = "x"
        variants["unknown column"] = (
            unknown_column_rows,
            MAPPING_COLUMNS + ("extra",),
        )

        for name, (rows, columns) in variants.items():
            with self.subTest(variant=name), tempfile.TemporaryDirectory() as tmp:
                path = write_mapping(tmp, rows, columns=columns)
                with self.assertRaises(ContractV2Error):
                    compile_owner_poisson_contract_v2(
                        contract,
                        mapping_path=path,
                        preprocessing_artifact_path=UCI_PREPROCESSOR,
                    )

    def test_wrong_preprocessing_artifact_is_rejected(self):
        contract = load_owner_poisson_contract_v2(UCI_CONFIG)
        with self.assertRaisesRegex(
            ContractV2Error,
            "Preprocessing artifact validation failed",
        ):
            compile_owner_poisson_contract_v2(
                contract,
                mapping_path=UCI_MAPPING,
                preprocessing_artifact_path=WISDM_PREPROCESSOR,
            )


if __name__ == "__main__":
    unittest.main()
