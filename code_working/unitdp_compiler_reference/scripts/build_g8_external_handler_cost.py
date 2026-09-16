#!/usr/bin/env python3
"""Measure disclosed adapter, backend, integrity, and execution costs."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CANDIDATE_MODULES = ROOT / "v36_candidate" / "src" / "unitdp"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import unitdp  # noqa: E402

if str(CANDIDATE_MODULES) not in unitdp.__path__:
    unitdp.__path__.append(str(CANDIDATE_MODULES))

from unitdp.benchmark_data_v2 import prepare_uci_har_v2  # noqa: E402
from unitdp.compiler_v36 import (  # noqa: E402
    compile_registered_contract_v36,
    load_registered_contract_v36,
)
from unitdp.external_pld_backend_v36 import (  # noqa: E402
    BACKEND_SITE_ENV_V36,
    account_external_pld_v36,
    resolve_external_pld_site_v36,
)
from unitdp.owner_external_pld_v36 import (  # noqa: E402
    train_owner_external_pld_fixed_v36,
)


OUTPUT = ROOT / "reports" / "g8_external_handler_cost_v2_20260725"
DATASETS = {
    "uci": {
        "sigma": 1.2295752282782475,
        "num_steps": 15,
        "num_selected": 6,
        "num_epochs": 1,
        "delta": 1e-5,
        "target_epsilon": 8.0,
        "direct_epsilon": 7.999999999999999,
    },
    "wisdm": {
        "sigma": 1.090645987511382,
        "num_steps": 20,
        "num_selected": 6,
        "num_epochs": 1,
        "delta": 1e-5,
        "target_epsilon": 8.0,
        "direct_epsilon": 8.0,
    },
    "sepsis": {
        "sigma": 0.776172784941656,
        "num_steps": 50,
        "num_selected": 5,
        "num_epochs": 1,
        "delta": 1e-5,
        "target_epsilon": 8.0,
        "direct_epsilon": 8.000000000000004,
    },
}
PRODUCTION_FILES = (
    "v36_candidate/src/unitdp/compiler_v36.py",
    "v36_candidate/src/unitdp/compiler_external_pld_v36.py",
    "v36_candidate/src/unitdp/external_pld_backend_v36.py",
    "v36_candidate/src/unitdp/handler_external_pld_v36.py",
    "v36_candidate/src/unitdp/owner_external_pld_v36.py",
    "v36_candidate/src/unitdp/source_bundle_external_pld_v36.py",
)
TEST_AND_GATE_FILES = (
    "v36_candidate/tests/test_external_pld_v36.py",
    "v36_candidate/tests/test_compiler_v36.py",
    "scripts/build_g8_external_handler_evidence.py",
    "scripts/build_g8_external_handler_gate.py",
    "scripts/build_g8_external_handler_cost.py",
    "scripts/verify_g8_external_handler_gate.py",
)
CONFIG_FILES = tuple(
    f"v36_candidate/configs/{dataset}_owner_external_pld_v36.yaml"
    for dataset in DATASETS
)
ARTIFACTS = {
    "mapping": (
        ROOT / "reports/uci_public_preprocessing_smoke_20260724/uci_train_mapping.csv"
    ),
    "preprocessor": (
        ROOT / "configs/preprocessing/uci_har_published_train_standard_scaler_v1.json"
    ),
}
DEFAULT_DATA_ROOT = ROOT.parent / "DP-SGD" / "data"
DATA_ROOT = Path(os.environ.get("UNITDP_DATA_ROOT", str(DEFAULT_DATA_ROOT)))
UCI_ROOT = DATA_ROOT / "uci_har" / "extracted" / "UCI HAR Dataset"
UCI_CONFIG = ROOT / "v36_candidate/configs/uci_owner_external_pld_v36.yaml"
UCI_SEED = 6761


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_bytes(path: Path) -> int:
    return sum(
        candidate.stat().st_size for candidate in path.rglob("*") if candidate.is_file()
    )


def disclosed_logical_lines(path: Path) -> int:
    return sum(
        bool(line.strip()) and not line.lstrip().startswith("#")
        for line in path.read_text(encoding="utf-8").splitlines()
    )


def source_rows(relative_paths: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {
            "path": relative,
            "logical_lines": disclosed_logical_lines(ROOT / relative),
            "bytes": (ROOT / relative).stat().st_size,
            "sha256": file_sha256(ROOT / relative),
        }
        for relative in relative_paths
    ]


def no_formula_duplication() -> dict[str, bool]:
    forbidden_functions = {
        "compute_log_a",
        "random_allocation_remove_rdp",
        "account_random_allocation_oracle_v4",
    }
    imports_torch = False
    defined: set[str] = set()
    for relative in PRODUCTION_FILES:
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports_torch |= any(alias.name == "torch" for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports_torch |= node.module == "torch"
            elif isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                defined.add(node.name)
    return {
        "production_imports_torch": imports_torch,
        "forbidden_accountant_formula_names_absent": (
            defined.isdisjoint(forbidden_functions)
        ),
    }


def _validated_cache_paths(site: Path) -> tuple[Path, Path]:
    parent = site.parent.resolve()
    cache = (parent / "numba_cache").resolve()
    backup = (parent / "numba_cache_g8_cost_backup").resolve()
    for candidate in (cache, backup):
        candidate.relative_to(parent)
        if candidate.parent != parent:
            raise RuntimeError("unexpected cache target")
    if backup.exists():
        raise FileExistsError(f"cost backup already exists: {backup}")
    return cache, backup


def _remove_generated_cache(path: Path, parent: Path) -> None:
    resolved = path.resolve()
    resolved.relative_to(parent.resolve())
    if resolved.parent != parent.resolve():
        raise RuntimeError("refusing broad cache removal")
    if resolved.exists():
        shutil.rmtree(resolved)


def backend_latency_rows(site: Path) -> list[dict[str, Any]]:
    cache, backup = _validated_cache_paths(site)
    parent = site.parent.resolve()
    had_cache = cache.exists()
    if had_cache:
        cache.rename(backup)
    rows: list[dict[str, Any]] = []
    try:
        for dataset, params in DATASETS.items():
            _remove_generated_cache(cache, parent)
            started = time.perf_counter()
            fresh = account_external_pld_v36(**params)
            fresh_seconds = time.perf_counter() - started
            started = time.perf_counter()
            repeated = account_external_pld_v36(**params)
            repeat_seconds = time.perf_counter() - started
            if fresh.payload() != repeated.payload():
                raise AssertionError(f"{dataset} cache states changed the response")
            rows.append(
                {
                    "dataset": dataset,
                    "fresh_worker_empty_numba_cache_seconds": (fresh_seconds),
                    "immediate_repeat_worker_seconds": repeat_seconds,
                    "epsilon_upper": fresh.epsilon_upper,
                    "epsilon_lower": fresh.epsilon_lower,
                    "response_sha256": fresh.response_sha256,
                    "responses_byte_identical": True,
                }
            )
    finally:
        _remove_generated_cache(cache, parent)
        if had_cache:
            backup.rename(cache)
    return rows


def H_pipeline_costs() -> dict[str, Any]:
    raw = yaml.safe_load(UCI_CONFIG.read_text(encoding="utf-8"))
    parse_samples: list[float] = []
    for _ in range(100):
        started = time.perf_counter()
        load_registered_contract_v36(UCI_CONFIG)
        parse_samples.append(time.perf_counter() - started)
    contract = load_registered_contract_v36(UCI_CONFIG)
    started = time.perf_counter()
    compiled = compile_registered_contract_v36(
        contract,
        mapping_path=ARTIFACTS["mapping"],
        preprocessing_artifact_path=ARTIFACTS["preprocessor"],
        require_executable=True,
    )
    compile_seconds = time.perf_counter() - started
    integrity_samples: list[float] = []
    for _ in range(5):
        started = time.perf_counter()
        compiled.assert_private_execution_integrity()
        integrity_samples.append(time.perf_counter() - started)

    torch.set_num_threads(1)
    prepared = prepare_uci_har_v2(
        dataset_root=UCI_ROOT,
        preprocessing_artifact_path=ARTIFACTS["preprocessor"],
    )
    prepared.assert_registered_full_conformance()
    started = time.perf_counter()
    result = train_owner_external_pld_fixed_v36(
        route=compiled,
        raw_train_x=prepared.raw_train_x,
        train_y=prepared.train_y,
        raw_test_x=prepared.raw_test_x,
        test_y=prepared.test_y,
        research_seed=UCI_SEED,
    )
    execution_seconds = time.perf_counter() - started
    public = result.public_payload()
    return {
        "dataset": "uci",
        "parse_iterations": len(parse_samples),
        "parse_median_seconds": statistics.median(parse_samples),
        "parse_max_seconds": max(parse_samples),
        "compile_seconds": compile_seconds,
        "integrity_iterations": len(integrity_samples),
        "integrity_median_seconds": statistics.median(integrity_samples),
        "integrity_max_seconds": max(integrity_samples),
        "execution_wrapper_seconds": execution_seconds,
        "H_public_plan_sha256": compiled.public_plan_sha256,
        "H_public_execution_sha256": public["public_execution_sha256"],
        "raw_config_canonical_sha256": canonical_sha256(raw),
    }


def build_report() -> dict[str, Any]:
    if not os.environ.get(BACKEND_SITE_ENV_V36):
        raise RuntimeError(f"{BACKEND_SITE_ENV_V36} must be set for cost measurement")
    if not (UCI_ROOT / "train/X_train.txt").is_file():
        raise FileNotFoundError("registered UCI HAR source is unavailable")
    site = resolve_external_pld_site_v36()
    production = source_rows(PRODUCTION_FILES)
    tests = source_rows(TEST_AND_GATE_FILES)
    configs = source_rows(CONFIG_FILES)
    backend_rows = backend_latency_rows(site)
    pipeline = H_pipeline_costs()
    formula_checks = no_formula_duplication()
    if formula_checks["production_imports_torch"]:
        raise AssertionError("H production adapter imports Torch")
    if not formula_checks["forbidden_accountant_formula_names_absent"]:
        raise AssertionError("H duplicated an accountant formula name")
    report: dict[str, Any] = {
        "schema_version": ("unitdp.g8_external_handler_cost.v1"),
        "status": "PASS",
        "measurement_scope": (
            "single Windows/Python environment; wall-clock observations, "
            "not cross-platform performance claims"
        ),
        "source_cost": {
            "new_production_files": len(PRODUCTION_FILES),
            "existing_production_files_changed": 0,
            "new_production_logical_lines": sum(
                row["logical_lines"] for row in production
            ),
            "production_rows": production,
            "test_and_gate_logical_lines": sum(row["logical_lines"] for row in tests),
            "test_and_gate_rows": tests,
            "config_logical_lines": sum(row["logical_lines"] for row in configs),
            "config_rows": configs,
            "logical_line_definition": (
                "nonempty physical line whose first nonspace character is not #"
            ),
            "formula_checks": formula_checks,
        },
        "environment_cost": {
            "backend_site_bytes": tree_bytes(site),
            "numba_cache_bytes_after_restore": (
                tree_bytes(site.parent / "numba_cache")
                if (site.parent / "numba_cache").is_dir()
                else 0
            ),
            "primary_wheel_bytes": (
                site.parent / "pld_accounting-0.5.0-py3-none-any.whl"
            )
            .stat()
            .st_size,
            "transitive_wheel_bytes": (
                site.parent / "random_allocation-1.0.5-py3-none-any.whl"
            )
            .stat()
            .st_size,
            "installation_commands": [
                (
                    "python -m pip install --no-cache-dir --no-deps "
                    "--target <site> numba==0.60.0 llvmlite==0.43.0"
                ),
                (
                    "python -m pip install --no-cache-dir --no-deps "
                    "--target <site> "
                    "pld_accounting-0.5.0-py3-none-any.whl "
                    "random_allocation-1.0.5-py3-none-any.whl"
                ),
            ],
            "caller_environment_mutated": False,
        },
        "backend_latency": backend_rows,
        "H_pipeline_latency": pipeline,
        "limits": [
            "wall-clock values include subprocess and integrity overhead",
            "fresh measurements clear only the disposable Numba cache",
            "one machine does not establish general performance",
            "PLD precision/runtime is materially costlier for Sepsis",
        ],
    }
    report["payload_sha256"] = canonical_sha256(report)
    return report


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite G8 cost report: {OUTPUT}")
    report = build_report()
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "g8_external_handler_cost_v2.json").write_text(
        json.dumps(
            report,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown = "\n".join(
        [
            "# G8 External-Handler Cost",
            "",
            f"Status: **{report['status']}**",
            "",
            "The measurements cover one Windows/Python environment and "
            "must not be generalized as performance results.",
            "",
            f"Canonical payload: `{report['payload_sha256']}`",
            "",
        ]
    )
    (OUTPUT / "g8_external_handler_cost_v2.md").write_text(
        markdown,
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "production_logical_lines": report["source_cost"][
                    "new_production_logical_lines"
                ],
                "backend_latency": report["backend_latency"],
                "H_pipeline_latency": report["H_pipeline_latency"],
                "payload_sha256": report["payload_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
