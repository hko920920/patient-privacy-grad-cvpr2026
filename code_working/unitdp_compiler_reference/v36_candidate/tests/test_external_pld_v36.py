import ast
import copy
import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
CANDIDATE_MODULES = ROOT / "v36_candidate" / "src" / "unitdp"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import unitdp  # noqa: E402

if str(CANDIDATE_MODULES) not in unitdp.__path__:
    unitdp.__path__.append(str(CANDIDATE_MODULES))

from unitdp.compiler_external_pld_v36 import (  # noqa: E402
    ExternalPldContractV36Error,
    compile_owner_external_pld_contract_v36,
    parse_owner_external_pld_contract_v36,
)
from unitdp.external_pld_backend_v36 import (  # noqa: E402
    BACKEND_SITE_ENV_V36,
    ExternalPldBackendV36Error,
    resolve_external_pld_site_v36,
    verify_external_pld_environment_v36,
)
from unitdp.source_bundle_external_pld_v36 import (  # noqa: E402
    COMPILER_SOURCE_FILE_SET_EXTERNAL_PLD_V36,
    EXECUTION_SOURCE_FILE_SET_EXTERNAL_PLD_V36,
    compiler_source_bundle_external_pld_v36,
    execution_source_bundle_external_pld_v36,
)

CONFIGS = {
    dataset: (
        ROOT / "v36_candidate" / "configs" / f"{dataset}_owner_external_pld_v36.yaml"
    )
    for dataset in ("uci", "wisdm", "sepsis")
}
ARTIFACTS = {
    "uci": {
        "mapping": (
            ROOT / "reports/uci_public_preprocessing_smoke_20260724/"
            "uci_train_mapping.csv"
        ),
        "preprocessor": (
            ROOT / "configs/preprocessing/"
            "uci_har_published_train_standard_scaler_v1.json"
        ),
    }
}
EXPECTED_NESTED_CONTRACTS = {
    "uci": ("038d26e2fcbff104f7227bda65b4adc5b0a6de444a2bf066f59743632b0e12a2"),
    "wisdm": ("f671c65e0ebd053620333c381fe8077c885d15e447be96fe2d75871f220750cb"),
    "sepsis": ("adb84960eb140858cc135c04363951b7856cb18da1e0bcbef529c1af37a9f7a8"),
}
EXPECTED_NESTED_PLANS = {
    "uci": ("f0c6334acbc242d7271e6a5d6436a7406955b187fced24a3463390a277444b20"),
}


def _load_raw(dataset: str) -> dict[str, object]:
    value = yaml.safe_load(CONFIGS[dataset].read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError("test config is not a mapping")
    return value


class ExternalPldContractV36Test(unittest.TestCase):
    def test_three_contracts_project_to_exact_frozen_A_contracts(self):
        for dataset in CONFIGS:
            with self.subTest(dataset=dataset):
                raw = _load_raw(dataset)
                contract = parse_owner_external_pld_contract_v36(raw)
                self.assertEqual(contract.public_payload(), raw)
                self.assertEqual(
                    contract.base_contract.public_contract_sha256,
                    EXPECTED_NESTED_CONTRACTS[dataset],
                )

    def test_strict_parser_rejects_schema_accountant_and_extra_mutations(self):
        mutations = []
        raw = _load_raw("uci")
        changed = copy.deepcopy(raw)
        changed["schema_version"] = "unitdp.unregistered"
        mutations.append(changed)
        changed = copy.deepcopy(raw)
        changed["extra"] = True
        mutations.append(changed)
        changed = copy.deepcopy(raw)
        changed["accountant"]["accepted_bound_type"] = "IS_DOMINATED"
        mutations.append(changed)
        changed = copy.deepcopy(raw)
        changed["accountant"]["loss_discretization"] = 0.01
        mutations.append(changed)
        changed = copy.deepcopy(raw)
        changed["execution"]["implementation_id"] = (
            "unitdp.owner_random_allocation_v4.train_owner_random_allocation_fixed_v4"
        )
        mutations.append(changed)
        for index, value in enumerate(mutations):
            with self.subTest(index=index):
                with self.assertRaises(ExternalPldContractV36Error):
                    parse_owner_external_pld_contract_v36(value)

    def test_missing_backend_location_fails_closed(self):
        previous = os.environ.pop(BACKEND_SITE_ENV_V36, None)
        try:
            with self.assertRaises(ExternalPldBackendV36Error):
                resolve_external_pld_site_v36()
        finally:
            if previous is not None:
                os.environ[BACKEND_SITE_ENV_V36] = previous

    @unittest.skipUnless(
        os.environ.get(BACKEND_SITE_ENV_V36),
        "isolated PLD backend site is not configured",
    )
    def test_environment_binds_exact_external_trees_and_licenses(self):
        environment = verify_external_pld_environment_v36()
        self.assertEqual(
            environment["primary_distribution"]["license"],
            "MIT",
        )
        self.assertEqual(
            environment["declared_transitive_distribution"]["license"],
            "MIT",
        )
        self.assertEqual(
            environment["primary_distribution"]["tree_sha256"],
            "50782379259744c44878da0c0164fa590539a80c076c05366c10bb8c77ca90dc",
        )

    def test_source_bundle_is_exact_and_contains_no_missing_files(self):
        compiler = dict(compiler_source_bundle_external_pld_v36())
        execution = dict(execution_source_bundle_external_pld_v36())
        self.assertEqual(
            set(compiler),
            COMPILER_SOURCE_FILE_SET_EXTERNAL_PLD_V36,
        )
        self.assertEqual(
            set(execution),
            EXECUTION_SOURCE_FILE_SET_EXTERNAL_PLD_V36,
        )
        self.assertTrue(set(compiler).issubset(execution))
        self.assertTrue(
            all(
                len(value) == 64 and int(value, 16) >= 0 for value in execution.values()
            )
        )

    def test_adapter_does_not_import_torch_or_copy_accountant_formulas(self):
        paths = (
            ROOT / "v36_candidate/src/unitdp/external_pld_backend_v36.py",
            ROOT / "v36_candidate/src/unitdp/compiler_external_pld_v36.py",
            ROOT / "v36_candidate/src/unitdp/owner_external_pld_v36.py",
        )
        forbidden_names = {
            "compute_log_a",
            "random_allocation_remove_rdp",
            "account_random_allocation_oracle_v4",
        }
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imports = {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names
            }
            functions = {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            self.assertNotIn("torch", imports)
            self.assertTrue(functions.isdisjoint(forbidden_names))

    @unittest.skipUnless(
        os.environ.get(BACKEND_SITE_ENV_V36),
        "isolated PLD backend site is not configured",
    )
    def test_uci_external_handler_compiles_over_exact_A_plan(self):
        contract = parse_owner_external_pld_contract_v36(_load_raw("uci"))
        compiled = compile_owner_external_pld_contract_v36(
            contract,
            mapping_path=ARTIFACTS["uci"]["mapping"],
            preprocessing_artifact_path=(ARTIFACTS["uci"]["preprocessor"]),
            require_executable=True,
        )
        self.assertTrue(compiled.execution_ready)
        self.assertEqual(
            compiled.base_route.public_plan_sha256,
            EXPECTED_NESTED_PLANS["uci"],
        )
        self.assertLessEqual(
            compiled.external_response.epsilon_lower,
            compiled.external_response.epsilon_upper,
        )
        self.assertLessEqual(
            compiled.external_response.epsilon_upper,
            contract.base_contract.target_epsilon + 1e-10,
        )
        public = compiled.public_plan()
        encoded = json.dumps(
            {
                key: value
                for key, value in public.items()
                if key != "public_plan_sha256"
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        self.assertEqual(
            hashlib.sha256(encoded).hexdigest(),
            compiled.public_plan_sha256,
        )


if __name__ == "__main__":
    unittest.main()
