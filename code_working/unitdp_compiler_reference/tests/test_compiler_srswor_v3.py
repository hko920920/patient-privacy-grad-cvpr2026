import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from unitdp.compiler_srswor_v3 import (
    SrsworContractV3Error,
    compile_owner_srswor_contract_v3,
    load_owner_srswor_contract_v3,
    parse_owner_srswor_contract_v3,
)
from unitdp.compiler_v3 import (
    ROUTE_DISPATCH_REGISTRY_V3,
    CompilerDispatchV3Error,
    compile_registered_contract_v3,
    load_registered_contract_v3,
    parse_registered_contract_v3,
)
from unitdp.source_bundle_srswor_v3 import (
    EXECUTION_SOURCE_FILE_SET_SRSWOR_V3,
    execution_source_bundle_sha256_srswor_v3,
    execution_source_bundle_srswor_v3,
)


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    (
        "uci",
        ROOT / "configs" / "v3" / "uci_owner_srswor_v3.yaml",
        ROOT
        / "reports"
        / "uci_public_preprocessing_smoke_20260724"
        / "uci_train_mapping.csv",
        ROOT
        / "configs"
        / "preprocessing"
        / "uci_har_published_train_standard_scaler_v1.json",
        21,
        8,
        336,
        5,
    ),
    (
        "wisdm",
        ROOT / "configs" / "v3" / "wisdm_owner_srswor_v3.yaml",
        ROOT
        / "reports"
        / "wisdm_public_protocol_smoke_20260724"
        / "wisdm_train_mapping.csv",
        ROOT
        / "configs"
        / "preprocessing"
        / "wisdm_v1_1_train_owners_1_25_stats24_standard_scaler_v1.json",
        25,
        8,
        400,
        4,
    ),
    (
        "sepsis",
        ROOT / "configs" / "v3" / "sepsis_owner_srswor_v3.yaml",
        ROOT
        / "reports"
        / "sepsis_public_protocol_smoke_20260724"
        / "sepsis_train_mapping.csv",
        ROOT
        / "configs"
        / "preprocessing"
        / "physionet2019_setA_first400_basic12x6_public_scaler_v1.json",
        312,
        32,
        1507,
        4,
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


class SrsworCompilerV3Test(unittest.TestCase):
    def test_three_registered_profiles_compile_and_bind_full_mapping(self):
        for (
            name,
            config,
            mapping,
            preprocessor,
            population,
            sample_size,
            selected,
            optimal_order,
        ) in CASES:
            with self.subTest(dataset=name):
                contract = load_owner_srswor_contract_v3(config)
                route = compile_owner_srswor_contract_v3(
                    contract,
                    mapping_path=mapping,
                    preprocessing_artifact_path=preprocessor,
                    require_executable=True,
                )
                self.assertTrue(route.execution_ready)
                self.assertEqual(
                    len(route.source_mapping.by_owner()),
                    population,
                )
                self.assertEqual(contract.sample_size, sample_size)
                self.assertEqual(
                    len(route.selected_mapping.records),
                    selected,
                )
                self.assertLessEqual(
                    abs(
                        route.accountant_epsilon_dp_accounting
                        - route.accountant_epsilon_direct_oracle
                    ),
                    1e-10,
                )
                self.assertEqual(
                    route.accountant_optimal_order_direct_oracle,
                    optimal_order,
                )
                self.assertLessEqual(
                    route.accountant_epsilon_dp_accounting,
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

    def test_dispatch_registry_selects_two_semantically_distinct_routes(self):
        self.assertEqual(len(ROUTE_DISPATCH_REGISTRY_V3), 2)
        entries = list(ROUTE_DISPATCH_REGISTRY_V3.values())
        self.assertEqual(
            {entry.adjacency for entry in entries},
            {"add_remove_one_owner", "replace_one_owner"},
        )
        self.assertEqual(
            {entry.accountant_family for entry in entries},
            {"poisson_gaussian_rdp", "srswor_gaussian_rdp"},
        )
        for _, config, mapping, preprocessor, *_ in CASES:
            contract = load_registered_contract_v3(config)
            route = compile_registered_contract_v3(
                contract,
                mapping_path=mapping,
                preprocessing_artifact_path=preprocessor,
                require_executable=True,
            )
            self.assertEqual(
                type(contract).__name__,
                "OwnerSrsworContractV3",
            )
            self.assertEqual(
                type(route).__name__,
                "CompiledOwnerSrsworRouteV3",
            )

        poisson = load_registered_contract_v3(
            ROOT / "configs" / "v2" / "uci_owner_poisson_v2.yaml"
        )
        compiled = compile_registered_contract_v3(
            poisson,
            mapping_path=CASES[0][2],
            preprocessing_artifact_path=CASES[0][3],
            require_executable=True,
        )
        self.assertEqual(type(poisson).__name__, "OwnerPoissonContractV2")
        self.assertEqual(
            type(compiled).__name__,
            "CompiledOwnerPoissonRouteV2",
        )

    def test_route_semantic_mutations_are_rejected(self):
        raw = raw_uci_contract()
        mutations = (
            (("privacy", "adjacency"), "add_remove_one_owner"),
            (("mechanism", "sampler"), "independent_bernoulli_owner"),
            (("mechanism", "source_dataset_size"), 20),
            (("mechanism", "sample_size"), 7),
            (("mechanism", "total_steps"), 14),
            (("mechanism", "noise_multiplier"), 1.26970999503374),
            (("mechanism", "update_denominator"), 7.0),
            (("mechanism", "empty_step_behavior"), "skip_update"),
            (("accountant", "sensitivity_multiplier"), 1.0),
            (
                ("accountant", "id"),
                "poisson_gaussian_rdp_add_remove_dual_v1",
            ),
            (
                ("accountant", "theorem_id"),
                "poisson_sampled_gaussian",
            ),
            (
                ("execution", "implementation_id"),
                "unitdp.owa_dpsgd.train_owner_poisson_fixed_v2",
            ),
            (("execution", "random_coins_public"), True),
            (("privacy", "target_epsilon"), 9.0),
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
                with self.assertRaises(SrsworContractV3Error):
                    parse_owner_srswor_contract_v3(
                        set_nested(raw, path, value)
                    )

    def test_dispatch_rejects_route_schema_and_field_splicing(self):
        raw = raw_uci_contract()
        with self.assertRaises(CompilerDispatchV3Error):
            parse_registered_contract_v3(
                {
                    **raw,
                    "route_id": "owa_owner_poisson_rdp_add_remove_v2",
                }
            )
        with self.assertRaises(CompilerDispatchV3Error):
            parse_registered_contract_v3(
                {
                    **raw,
                    "schema_version": "unitdp.owner_poisson_contract.v2",
                }
            )
        with self.assertRaises(CompilerDispatchV3Error):
            parse_registered_contract_v3(
                {**raw, "route_id": "unregistered"}
            )

    def test_compiler_rejects_one_byte_full_mapping_change(self):
        _, config, mapping, preprocessor, *_ = CASES[0]
        contract = load_owner_srswor_contract_v3(config)
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
                SrsworContractV3Error,
                "exactly match",
            ):
                compile_owner_srswor_contract_v3(
                    contract,
                    mapping_path=target,
                    preprocessing_artifact_path=preprocessor,
                )

    def test_duplicate_yaml_key_and_unknown_field_are_rejected(self):
        text = CASES[0][1].read_text(encoding="utf-8")
        duplicate = text.replace(
            "  sample_size: 8\n",
            "  sample_size: 8\n  sample_size: 8\n",
            1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "duplicate.yaml"
            path.write_text(duplicate, encoding="utf-8")
            with self.assertRaises(SrsworContractV3Error):
                load_owner_srswor_contract_v3(path)
        raw = raw_uci_contract()
        raw["unexpected"] = True
        with self.assertRaises(SrsworContractV3Error):
            parse_owner_srswor_contract_v3(raw)

    def test_source_bundle_is_complete_and_self_consistent(self):
        bundle = dict(execution_source_bundle_srswor_v3())
        self.assertEqual(
            set(bundle),
            EXECUTION_SOURCE_FILE_SET_SRSWOR_V3,
        )
        self.assertEqual(
            execution_source_bundle_sha256_srswor_v3(bundle),
            execution_source_bundle_sha256_srswor_v3(),
        )
