import copy
import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import yaml

from unitdp.compiler_allocation_v4 import (
    AllocationContractV4Error,
    compile_owner_random_allocation_contract_v4,
    load_owner_random_allocation_contract_v4,
    parse_owner_random_allocation_contract_v4,
)
from unitdp.source_bundle_allocation_v4 import (
    COMPILER_SOURCE_FILE_SET_ALLOCATION_V4,
    compiler_source_bundle_allocation_v4,
    compiler_source_bundle_sha256_allocation_v4,
)


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    (
        "uci",
        ROOT / "configs" / "v4" / "uci_owner_random_allocation_v4.yaml",
        ROOT
        / "reports"
        / "uci_public_preprocessing_smoke_20260724"
        / "uci_train_mapping.csv",
        ROOT
        / "configs"
        / "preprocessing"
        / "uci_har_published_train_standard_scaler_v1.json",
        21,
        6,
        336,
        4,
    ),
    (
        "wisdm",
        ROOT / "configs" / "v4" / "wisdm_owner_random_allocation_v4.yaml",
        ROOT
        / "reports"
        / "wisdm_public_protocol_smoke_20260724"
        / "wisdm_train_mapping.csv",
        ROOT
        / "configs"
        / "preprocessing"
        / "wisdm_v1_1_train_owners_1_25_stats24_standard_scaler_v1.json",
        25,
        6,
        400,
        4,
    ),
    (
        "sepsis",
        ROOT / "configs" / "v4" / "sepsis_owner_random_allocation_v4.yaml",
        ROOT
        / "reports"
        / "sepsis_public_protocol_smoke_20260724"
        / "sepsis_train_mapping.csv",
        ROOT
        / "configs"
        / "preprocessing"
        / "physionet2019_setA_first400_basic12x6_public_scaler_v1.json",
        312,
        5,
        1507,
        3,
    ),
)


def raw_uci_contract() -> dict[str, object]:
    raw = yaml.safe_load(CASES[0][1].read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def set_nested(
    raw: dict[str, object],
    path: tuple[str, ...],
    value: object,
) -> dict[str, object]:
    result = copy.deepcopy(raw)
    cursor: dict[str, object] = result
    for key in path[:-1]:
        child = cursor[key]
        assert isinstance(child, dict)
        cursor = child
    cursor[path[-1]] = value
    return result


class AllocationCompilerV4Test(unittest.TestCase):
    def test_three_registered_profiles_compile_with_dual_accounting(self):
        for (
            name,
            config,
            mapping,
            preprocessor,
            population,
            selected_steps,
            selected_records,
            optimal_order,
        ) in CASES:
            with self.subTest(dataset=name):
                contract = load_owner_random_allocation_contract_v4(config)
                route = compile_owner_random_allocation_contract_v4(
                    contract,
                    mapping_path=mapping,
                    preprocessing_artifact_path=preprocessor,
                )
                self.assertTrue(route.execution_ready)
                self.assertEqual(
                    route.status,
                    "validated_contract_research_executor_ready",
                )
                self.assertEqual(
                    len(route.source_mapping.by_owner()),
                    population,
                )
                self.assertEqual(
                    contract.num_selected_steps_per_owner,
                    selected_steps,
                )
                self.assertEqual(
                    len(route.selected_mapping.records),
                    selected_records,
                )
                self.assertLessEqual(
                    abs(route.accountant_epsilon - route.oracle_epsilon),
                    2e-12,
                )
                self.assertLessEqual(
                    abs(
                        route.accountant_epsilon_remove
                        - route.oracle_epsilon_remove
                    ),
                    2e-12,
                )
                self.assertLessEqual(
                    abs(
                        route.accountant_epsilon_add
                        - route.oracle_epsilon_add
                    ),
                    2e-12,
                )
                self.assertEqual(
                    route.accountant_optimal_remove_order,
                    optimal_order,
                )
                self.assertLessEqual(
                    route.accountant_epsilon,
                    contract.target_epsilon + 1e-10,
                )
                plan = route.public_plan()
                without_hash = dict(plan)
                reported = without_hash.pop("public_plan_sha256")
                encoded = json.dumps(
                    without_hash,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
                import hashlib

                self.assertEqual(
                    reported,
                    hashlib.sha256(encoded).hexdigest(),
                )

    def test_executable_gate_uses_the_registered_callable(self):
        contract = load_owner_random_allocation_contract_v4(CASES[0][1])
        route = compile_owner_random_allocation_contract_v4(
            contract,
            mapping_path=CASES[0][2],
            preprocessing_artifact_path=CASES[0][3],
            require_executable=True,
        )
        self.assertTrue(route.execution_ready)
        self.assertEqual(
            route.contract.implementation_id,
            route.registry_entry.implementation_id,
        )

    def test_all_route_semantic_surface_mutations_are_rejected(self):
        raw = raw_uci_contract()
        mutations = (
            (("privacy", "adjacency"), "replace_one_owner"),
            (("privacy", "target_epsilon"), 9.0),
            (("privacy", "delta"), 1e-6),
            (("mechanism", "sampler"), "independent_bernoulli_owner"),
            (("mechanism", "source_dataset_size"), 20),
            (("mechanism", "num_steps"), 16),
            (("mechanism", "num_selected_steps_per_owner"), 5),
            (("mechanism", "num_epochs"), 2),
            (("mechanism", "total_steps"), 16),
            (
                ("mechanism", "assignment_schedule"),
                "independent_per_step",
            ),
            (("mechanism", "noise_multiplier"), 1.25),
            (("mechanism", "update_denominator"), 8.4),
            (("mechanism", "empty_step_behavior"), "skip_update"),
            (
                ("mechanism", "per_owner_contribution"),
                "one_vector_per_owner",
            ),
            (
                ("accountant", "id"),
                "poisson_gaussian_rdp_add_remove_dual_v1",
            ),
            (("accountant", "sensitivity_multiplier"), 2.0),
            (("accountant", "theorem_id"), "unregistered_theorem"),
            (
                ("accountant", "oracle_implementation_id"),
                "unregistered.oracle",
            ),
            (("accountant", "reduction_id"), "unregistered_reduction"),
            (("accountant", "direction_policy"), "remove_only"),
            (
                ("execution", "implementation_id"),
                "unitdp.owner_poisson_v2.train_owner_poisson_fixed_v2",
            ),
            (("execution", "random_coins_public"), True),
            (("data", "benchmark_profile_sha256"), "0" * 64),
            (("data", "base_data_profile_id"), "wisdm_aaai27_v1"),
            (
                ("contribution_policy", "windows_per_owner_per_step"),
                7,
            ),
            (("optimization", "learning_rate"), 0.1),
        )
        for path, value in mutations:
            with self.subTest(path=path):
                with self.assertRaises(AllocationContractV4Error):
                    parse_owner_random_allocation_contract_v4(
                        set_nested(raw, path, value)
                    )

    def test_query_identity_is_not_collapsed_by_equal_floor_t_over_k(self):
        raw = raw_uci_contract()
        mutated = set_nested(raw, ("mechanism", "num_steps"), 16)
        mutated = set_nested(
            mutated,
            ("mechanism", "total_steps"),
            16,
        )
        with self.assertRaises(AllocationContractV4Error):
            parse_owner_random_allocation_contract_v4(mutated)

    def test_duplicate_key_unknown_field_and_splicing_are_rejected(self):
        text = CASES[0][1].read_text(encoding="utf-8")
        duplicate = text.replace(
            "  num_steps: 15\n",
            "  num_steps: 15\n  num_steps: 15\n",
            1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "duplicate.yaml"
            path.write_text(duplicate, encoding="utf-8")
            with self.assertRaises(AllocationContractV4Error):
                load_owner_random_allocation_contract_v4(path)
        raw = raw_uci_contract()
        raw["unexpected"] = True
        with self.assertRaises(AllocationContractV4Error):
            parse_owner_random_allocation_contract_v4(raw)
        with self.assertRaises(AllocationContractV4Error):
            parse_owner_random_allocation_contract_v4(
                {
                    **raw_uci_contract(),
                    "route_id": "owa_owner_srswor_rdp_replace_one_v3",
                }
            )

    def test_compiler_rejects_one_byte_full_mapping_change(self):
        _, config, mapping, preprocessor, *_ = CASES[0]
        contract = load_owner_random_allocation_contract_v4(config)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "mapping.csv"
            with mapping.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as source:
                reader = csv.DictReader(source)
                rows = list(reader)
                columns = list(reader.fieldnames or [])
            rows[-1]["scenario"] = rows[-1]["scenario"] + "_mutated"
            with target.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(
                AllocationContractV4Error,
                "exactly match",
            ):
                compile_owner_random_allocation_contract_v4(
                    contract,
                    mapping_path=target,
                    preprocessing_artifact_path=preprocessor,
                )

    def test_compiled_contract_tamper_is_rejected(self):
        contract = load_owner_random_allocation_contract_v4(CASES[0][1])
        route = compile_owner_random_allocation_contract_v4(
            contract,
            mapping_path=CASES[0][2],
            preprocessing_artifact_path=CASES[0][3],
        )
        tampered = replace(
            route,
            contract=replace(route.contract, num_steps=16),
        )
        with self.assertRaises(AllocationContractV4Error):
            tampered.assert_private_execution_integrity()

    def test_source_bundle_is_complete_and_self_consistent(self):
        bundle = dict(compiler_source_bundle_allocation_v4())
        self.assertEqual(
            set(bundle),
            COMPILER_SOURCE_FILE_SET_ALLOCATION_V4,
        )
        self.assertEqual(
            compiler_source_bundle_sha256_allocation_v4(bundle),
            compiler_source_bundle_sha256_allocation_v4(),
        )


if __name__ == "__main__":
    unittest.main()
