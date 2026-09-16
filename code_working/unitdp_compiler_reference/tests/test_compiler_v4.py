import copy
import inspect
import unittest
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import yaml

from unitdp.compiler_v3 import (
    compile_registered_contract_v3,
    load_registered_contract_v3,
)
from unitdp.compiler_v4 import (
    ALLOCATION_HANDLER_V4,
    POISSON_HANDLER_V4,
    REQUIRED_REGISTRATION_OBLIGATIONS_V4,
    ROUTE_DISPATCH_REGISTRY_V4,
    SRSWOR_HANDLER_V4,
    CompilerDispatchV4Error,
    build_route_dispatch_registry_v4,
    compile_registered_contract_v4,
    load_registered_contract_v4,
    parse_registered_contract_v4,
    validate_route_handler_v4,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = {
    "uci": (
        ROOT
        / "reports"
        / "uci_public_preprocessing_smoke_20260724"
        / "uci_train_mapping.csv",
        ROOT
        / "configs"
        / "preprocessing"
        / "uci_har_published_train_standard_scaler_v1.json",
    ),
    "wisdm": (
        ROOT
        / "reports"
        / "wisdm_public_protocol_smoke_20260724"
        / "wisdm_train_mapping.csv",
        ROOT
        / "configs"
        / "preprocessing"
        / "wisdm_v1_1_train_owners_1_25_stats24_standard_scaler_v1.json",
    ),
    "sepsis": (
        ROOT
        / "reports"
        / "sepsis_public_protocol_smoke_20260724"
        / "sepsis_train_mapping.csv",
        ROOT
        / "configs"
        / "preprocessing"
        / "physionet2019_setA_first400_basic12x6_public_scaler_v1.json",
    ),
}
ROUTES = {
    "P": ("v2", "owner_poisson_v2"),
    "F": ("v3", "owner_srswor_v3"),
    "A": ("v4", "owner_random_allocation_v4"),
}


def config_path(route: str, dataset: str) -> Path:
    directory, suffix = ROUTES[route]
    return ROOT / "configs" / directory / f"{dataset}_{suffix}.yaml"


class CompilerV4Test(unittest.TestCase):
    def test_registry_is_immutable_and_every_handler_has_seven_witnesses(self):
        self.assertIsInstance(
            ROUTE_DISPATCH_REGISTRY_V4,
            MappingProxyType,
        )
        self.assertEqual(len(ROUTE_DISPATCH_REGISTRY_V4), 3)
        self.assertEqual(
            len(
                {
                    handler.registration_sha256
                    for handler in ROUTE_DISPATCH_REGISTRY_V4.values()
                }
            ),
            3,
        )
        for handler in ROUTE_DISPATCH_REGISTRY_V4.values():
            validate_route_handler_v4(handler)
            self.assertEqual(
                {
                    witness.obligation_id
                    for witness in handler.obligation_witnesses
                },
                REQUIRED_REGISTRATION_OBLIGATIONS_V4,
            )
            self.assertEqual(len(handler.obligation_witnesses), 7)

    def test_all_nine_registered_profile_routes_compile_generically(self):
        expected_types = {
            "P": (
                "OwnerPoissonContractV2",
                "CompiledOwnerPoissonRouteV2",
            ),
            "F": (
                "OwnerSrsworContractV3",
                "CompiledOwnerSrsworRouteV3",
            ),
            "A": (
                "OwnerRandomAllocationContractV4",
                "CompiledOwnerRandomAllocationRouteV4",
            ),
        }
        for route in ROUTES:
            for dataset in ARTIFACTS:
                with self.subTest(route=route, dataset=dataset):
                    contract = load_registered_contract_v4(
                        config_path(route, dataset)
                    )
                    compiled = compile_registered_contract_v4(
                        contract,
                        mapping_path=ARTIFACTS[dataset][0],
                        preprocessing_artifact_path=ARTIFACTS[dataset][1],
                        require_executable=True,
                    )
                    self.assertEqual(
                        type(contract).__name__,
                        expected_types[route][0],
                    )
                    self.assertEqual(
                        type(compiled).__name__,
                        expected_types[route][1],
                    )
                    self.assertTrue(compiled.execution_ready)
                    compiled.assert_private_execution_integrity()

    def test_adding_a_preserves_exact_p_f_outputs(self):
        legacy_registry = build_route_dispatch_registry_v4(
            (POISSON_HANDLER_V4, SRSWOR_HANDLER_V4)
        )
        self.assertEqual(len(legacy_registry), 2)
        for route in ("P", "F"):
            for dataset in ARTIFACTS:
                with self.subTest(route=route, dataset=dataset):
                    config = config_path(route, dataset)
                    legacy_v3_contract = load_registered_contract_v3(
                        config
                    )
                    before = load_registered_contract_v4(
                        config,
                        registry=legacy_registry,
                    )
                    after = load_registered_contract_v4(config)
                    self.assertEqual(before, after)
                    self.assertEqual(before, legacy_v3_contract)
                    legacy_v3_compiled = compile_registered_contract_v3(
                        legacy_v3_contract,
                        mapping_path=ARTIFACTS[dataset][0],
                        preprocessing_artifact_path=ARTIFACTS[dataset][1],
                        require_executable=True,
                    )
                    before_compiled = compile_registered_contract_v4(
                        before,
                        mapping_path=ARTIFACTS[dataset][0],
                        preprocessing_artifact_path=ARTIFACTS[dataset][1],
                        require_executable=True,
                        registry=legacy_registry,
                    )
                    after_compiled = compile_registered_contract_v4(
                        after,
                        mapping_path=ARTIFACTS[dataset][0],
                        preprocessing_artifact_path=ARTIFACTS[dataset][1],
                        require_executable=True,
                    )
                    self.assertEqual(
                        before_compiled.public_plan(),
                        after_compiled.public_plan(),
                    )
                    self.assertEqual(
                        legacy_v3_compiled.public_plan(),
                        after_compiled.public_plan(),
                    )

    def test_generic_functions_have_no_route_or_contract_type_branch(self):
        parse_source = inspect.getsource(parse_registered_contract_v4)
        compile_source = inspect.getsource(
            compile_registered_contract_v4
        )
        for forbidden in (
            "CORE_ROUTE_ID_V2",
            "CORE_ROUTE_ID_SRSWOR_V3",
            "CORE_ROUTE_ID_ALLOCATION_V4",
            "OwnerPoissonContractV2",
            "OwnerSrsworContractV3",
            "OwnerRandomAllocationContractV4",
            "if route_id ==",
        ):
            self.assertNotIn(forbidden, parse_source)
            self.assertNotIn(forbidden, compile_source)
        self.assertIn("handler.parser(raw_value)", parse_source)
        self.assertIn("handler.compiler(", compile_source)

    def test_incomplete_or_spliced_registration_is_rejected(self):
        missing = replace(
            ALLOCATION_HANDLER_V4,
            obligation_witnesses=(
                ALLOCATION_HANDLER_V4.obligation_witnesses[:-1]
            ),
        )
        bad_source_witnesses = list(
            ALLOCATION_HANDLER_V4.obligation_witnesses
        )
        bad_source_witnesses[0] = replace(
            bad_source_witnesses[0],
            source_sha256="0" * 64,
        )
        bad_source = replace(
            ALLOCATION_HANDLER_V4,
            obligation_witnesses=tuple(bad_source_witnesses),
        )
        cases = (
            missing,
            bad_source,
            replace(
                ALLOCATION_HANDLER_V4,
                parser=SRSWOR_HANDLER_V4.parser,
            ),
            replace(
                ALLOCATION_HANDLER_V4,
                compiled_type=SRSWOR_HANDLER_V4.compiled_type,
            ),
            replace(
                ALLOCATION_HANDLER_V4,
                evidence_report_sha256="0" * 64,
            ),
        )
        for handler in cases:
            with self.subTest(handler=handler):
                with self.assertRaises(CompilerDispatchV4Error):
                    build_route_dispatch_registry_v4((handler,))
        with self.assertRaises(CompilerDispatchV4Error):
            build_route_dispatch_registry_v4(
                (POISSON_HANDLER_V4, POISSON_HANDLER_V4)
            )

    def test_route_schema_and_field_splicing_are_rejected(self):
        raw = yaml.safe_load(
            config_path("A", "uci").read_text(encoding="utf-8")
        )
        self.assertIsInstance(raw, dict)
        with self.assertRaises(CompilerDispatchV4Error):
            parse_registered_contract_v4(
                {
                    **raw,
                    "route_id": POISSON_HANDLER_V4.route_id,
                }
            )
        with self.assertRaises(CompilerDispatchV4Error):
            parse_registered_contract_v4(
                {
                    **raw,
                    "schema_version": (
                        SRSWOR_HANDLER_V4.schema_version
                    ),
                }
            )
        unknown = copy.deepcopy(raw)
        unknown["route_id"] = "unregistered"
        with self.assertRaises(CompilerDispatchV4Error):
            parse_registered_contract_v4(unknown)


if __name__ == "__main__":
    unittest.main()
