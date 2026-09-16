"""Build a deterministic public rejection audit for the strict V2 compiler."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unitdp.compiler_v2 import (  # noqa: E402
    ContractV2Error,
    ExecutorNotReadyError,
    compile_owner_poisson_contract_v2,
    load_owner_poisson_contract_v2,
    parse_owner_poisson_contract_v2,
)
from unitdp.source_bundle_v2 import (  # noqa: E402
    execution_source_bundle_sha256_v2,
)


UCI_CONFIG = ROOT / "configs" / "v2" / "uci_owner_poisson_v2.yaml"
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


CONTRACT_MUTATIONS = (
    ("unknown_top_level_field", "schema", ("epochs",), 5),
    (
        "unknown_nested_field",
        "schema",
        ("mechanism", "total_step"),
        15,
    ),
    (
        "replace_one_adjacency",
        "adjacency",
        ("privacy", "adjacency"),
        "replace_one_owner",
    ),
    (
        "window_sampling_unit",
        "unit_alignment",
        ("mechanism", "sampling_unit"),
        "window",
    ),
    (
        "window_accounting_unit",
        "unit_alignment",
        ("mechanism", "accounting_unit"),
        "window",
    ),
    (
        "fixed_size_sampler",
        "sampler",
        ("mechanism", "sampler"),
        "fixed_size_owner",
    ),
    (
        "observed_schedule_source",
        "schedule",
        ("mechanism", "parameter_source"),
        "observed_owner_count",
    ),
    (
        "zero_owner_sample_rate",
        "numeric_domain",
        ("mechanism", "owner_sample_rate"),
        0.0,
    ),
    (
        "owner_sample_rate_above_one",
        "numeric_domain",
        ("mechanism", "owner_sample_rate"),
        1.01,
    ),
    (
        "numeric_string_owner_sample_rate",
        "strict_types",
        ("mechanism", "owner_sample_rate"),
        "0.38095238095238093",
    ),
    (
        "zero_total_steps",
        "numeric_domain",
        ("mechanism", "total_steps"),
        0,
    ),
    (
        "floating_total_steps",
        "strict_types",
        ("mechanism", "total_steps"),
        15.0,
    ),
    (
        "negative_clip_norm",
        "numeric_domain",
        ("mechanism", "clip_norm"),
        -1.0,
    ),
    (
        "zero_noise_multiplier",
        "numeric_domain",
        ("mechanism", "noise_multiplier"),
        0.0,
    ),
    (
        "zero_update_denominator",
        "numeric_domain",
        ("mechanism", "update_denominator"),
        0.0,
    ),
    (
        "profile_owner_sample_rate_drift",
        "profile_drift",
        ("mechanism", "owner_sample_rate"),
        0.4,
    ),
    (
        "profile_total_steps_drift",
        "profile_drift",
        ("mechanism", "total_steps"),
        16,
    ),
    (
        "profile_noise_drift",
        "profile_drift",
        ("mechanism", "noise_multiplier"),
        1.3,
    ),
    (
        "skip_empty_step",
        "mechanism",
        ("mechanism", "empty_step_behavior"),
        "skip_update",
    ),
    (
        "realized_batch_denominator",
        "mechanism",
        ("mechanism", "update_denominator"),
        7.0,
    ),
    (
        "per_window_contributions",
        "mechanism",
        ("mechanism", "per_owner_contribution"),
        "per_window_vectors",
    ),
    (
        "sum_owner_aggregation",
        "mechanism",
        ("mechanism", "owner_aggregation"),
        "sum",
    ),
    (
        "unregistered_accountant",
        "accountant",
        ("accountant", "id"),
        "another_accountant",
    ),
    (
        "per_step_budget_above_cap",
        "contribution_policy",
        ("contribution_policy", "windows_per_owner_per_step"),
        17,
    ),
    (
        "cross_owner_policy",
        "contribution_policy",
        ("contribution_policy", "scope"),
        "global",
    ),
    (
        "private_fitted_scaler",
        "preprocessing",
        ("data", "preprocessing", "fit_on_protected_data"),
        True,
    ),
    (
        "stale_executor_id",
        "implementation",
        ("execution", "implementation_id"),
        "unitdp.owa_dpsgd.train_owa_dpsgd:v1",
    ),
    (
        "public_random_coins",
        "rng_boundary",
        ("execution", "random_coins_public"),
        True,
    ),
    (
        "research_rng_release_profile",
        "rng_boundary",
        ("execution", "profile"),
        "privacy_release",
    ),
    (
        "protocol_owner_count_drift",
        "profile_drift",
        ("data", "public_protocol", "train_owners"),
        20,
    ),
    (
        "unregistered_benchmark_profile",
        "profile_drift",
        ("data", "benchmark_profile_id"),
        "custom",
    ),
    (
        "benchmark_profile_digest_drift",
        "profile_drift",
        ("data", "benchmark_profile_sha256"),
        "0" * 64,
    ),
    (
        "learning_rate_drift",
        "profile_drift",
        ("optimization", "learning_rate"),
        0.051,
    ),
    (
        "invalid_class_weights",
        "optimization",
        ("optimization", "class_weights"),
        [1.0],
    ),
)


def payload_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def raw_contract() -> dict[str, Any]:
    value = yaml.safe_load(UCI_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("Registered UCI contract must be an object")
    return value


def set_nested(
    value: dict[str, Any],
    path: tuple[str, ...],
    replacement: object,
) -> None:
    current: dict[str, Any] = value
    for key in path[:-1]:
        child = current[key]
        if not isinstance(child, dict):
            raise RuntimeError(f"Mutation path is not an object: {path}")
        current = child
    current[path[-1]] = replacement


def mapping_rows() -> list[dict[str, str]]:
    return [
        {
            "scenario": "public_synthetic_mutation_audit",
            "window_id": "w0",
            "owner_id": "a",
            "owner_ids": "a",
            "start": "0",
            "end": "4",
            "row_index": "0",
            "label": "0",
        },
        {
            "scenario": "public_synthetic_mutation_audit",
            "window_id": "w1",
            "owner_id": "b",
            "owner_ids": "b",
            "start": "0",
            "end": "4",
            "row_index": "1",
            "label": "1",
        },
    ]


def write_mapping(
    directory: Path,
    rows: list[dict[str, str]],
    *,
    columns: tuple[str, ...] = MAPPING_COLUMNS,
) -> Path:
    path = directory / "mapping.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)
    return path


def expect_rejection(
    case_id: str,
    family: str,
    stage: str,
    expected_exception: type[BaseException],
    operation: Callable[[], object],
) -> dict[str, str]:
    try:
        operation()
    except expected_exception:
        return {
            "case_id": case_id,
            "family": family,
            "stage": stage,
            "expected": "reject",
            "observed": "rejected",
            "exception_class": expected_exception.__name__,
        }
    except Exception as exc:
        raise RuntimeError(
            f"{case_id} raised {type(exc).__name__}, expected "
            f"{expected_exception.__name__}"
        ) from exc
    raise RuntimeError(f"{case_id} was unexpectedly accepted")


def contract_mutation_results() -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for case_id, family, path, replacement in CONTRACT_MUTATIONS:
        mutated = copy.deepcopy(raw_contract())
        set_nested(mutated, path, replacement)
        results.append(
            expect_rejection(
                case_id,
                family,
                "contract_parse",
                ContractV2Error,
                lambda value=mutated: parse_owner_poisson_contract_v2(
                    value
                ),
            )
        )
    return results


def duplicate_key_results(directory: Path) -> list[dict[str, str]]:
    yaml_text = UCI_CONFIG.read_text(encoding="utf-8").replace(
        "  total_steps: 15\n",
        "  total_steps: 15\n  total_steps: 15\n",
        1,
    )
    json_text = json.dumps(raw_contract()).replace(
        '"route_id": "owa_owner_poisson_rdp_add_remove_v2",',
        (
            '"route_id": "owa_owner_poisson_rdp_add_remove_v2", '
            '"route_id": "owa_owner_poisson_rdp_add_remove_v2",'
        ),
        1,
    )
    yaml_path = directory / "duplicate.yaml"
    json_path = directory / "duplicate.json"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    json_path.write_text(json_text, encoding="utf-8")
    return [
        expect_rejection(
            "duplicate_yaml_key",
            "schema",
            "contract_load",
            ContractV2Error,
            lambda: load_owner_poisson_contract_v2(yaml_path),
        ),
        expect_rejection(
            "duplicate_json_key",
            "schema",
            "contract_load",
            ContractV2Error,
            lambda: load_owner_poisson_contract_v2(json_path),
        ),
    ]


def mapping_mutation_results(
    directory: Path,
    contract: object,
) -> list[dict[str, str]]:
    variants: list[
        tuple[str, list[dict[str, str]], tuple[str, ...]]
    ] = []
    decimal_index = mapping_rows()
    decimal_index[1]["row_index"] = "1.5"
    variants.append(("decimal_row_index", decimal_index, MAPPING_COLUMNS))
    duplicate_index = mapping_rows()
    duplicate_index[1]["row_index"] = "0"
    variants.append(
        ("duplicate_row_index", duplicate_index, MAPPING_COLUMNS)
    )
    gapped_index = mapping_rows()
    gapped_index[1]["row_index"] = "2"
    variants.append(("gapped_row_index", gapped_index, MAPPING_COLUMNS))
    duplicate_id = mapping_rows()
    duplicate_id[1]["window_id"] = "w0"
    variants.append(("duplicate_window_id", duplicate_id, MAPPING_COLUMNS))
    multi_owner = mapping_rows()
    multi_owner[0]["owner_ids"] = "a;b"
    variants.append(("multi_owner_row", multi_owner, MAPPING_COLUMNS))
    blank_label = mapping_rows()
    blank_label[0]["label"] = ""
    variants.append(("blank_label", blank_label, MAPPING_COLUMNS))
    unknown_column = mapping_rows()
    for row in unknown_column:
        row["extra"] = "x"
    variants.append(
        (
            "unknown_mapping_column",
            unknown_column,
            MAPPING_COLUMNS + ("extra",),
        )
    )

    results: list[dict[str, str]] = []
    for case_id, rows, columns in variants:
        case_dir = directory / case_id
        case_dir.mkdir()
        path = write_mapping(case_dir, rows, columns=columns)
        results.append(
            expect_rejection(
                case_id,
                "mapping",
                "compile",
                ContractV2Error,
                lambda mapping_path=path: (
                    compile_owner_poisson_contract_v2(
                        contract,
                        mapping_path=mapping_path,
                        preprocessing_artifact_path=UCI_PREPROCESSOR,
                    )
                ),
            )
        )
    return results


def compile_synthetic(
    contract: object,
    mapping_path: Path,
):
    return compile_owner_poisson_contract_v2(
        contract,
        mapping_path=mapping_path,
        preprocessing_artifact_path=UCI_PREPROCESSOR,
    )


def integrity_mutation_results(
    contract: object,
    mapping_path: Path,
) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []

    mutated_contract = replace(contract, owner_sample_rate=0.25)
    results.append(
        expect_rejection(
            "direct_dataclass_schedule_bypass",
            "post_parse_bypass",
            "compile",
            ContractV2Error,
            lambda: compile_synthetic(mutated_contract, mapping_path),
        )
    )

    nested_contract = load_owner_poisson_contract_v2(UCI_CONFIG)
    nested_contract.preprocessing_binding["fit_on_protected_data"] = True
    results.append(
        expect_rejection(
            "nested_dataclass_preprocessing_bypass",
            "post_parse_bypass",
            "compile",
            ContractV2Error,
            lambda: compile_synthetic(nested_contract, mapping_path),
        )
    )

    compiled = compile_synthetic(contract, mapping_path)
    compiled.selected_mapping.records.pop()
    results.append(
        expect_rejection(
            "compiled_selected_mapping_mutation",
            "post_compile_integrity",
            "pre_execution_integrity",
            ContractV2Error,
            compiled.assert_private_execution_integrity,
        )
    )

    compiled = compile_synthetic(contract, mapping_path)
    compiled.policy.max_windows_per_owner = 15
    results.append(
        expect_rejection(
            "compiled_policy_mutation",
            "post_compile_integrity",
            "pre_execution_integrity",
            ContractV2Error,
            compiled.assert_private_execution_integrity,
        )
    )

    compiled = compile_synthetic(contract, mapping_path)
    compiled.contract.public_protocol["test_owners"] = 8
    results.append(
        expect_rejection(
            "compiled_contract_mutation",
            "post_compile_integrity",
            "pre_execution_integrity",
            ContractV2Error,
            compiled.assert_private_execution_integrity,
        )
    )

    compiled = replace(
        compile_synthetic(contract, mapping_path),
        execution_source_bundle_sha256="0" * 64,
    )
    results.append(
        expect_rejection(
            "compiled_source_bundle_mutation",
            "post_compile_integrity",
            "pre_execution_integrity",
            ContractV2Error,
            compiled.assert_private_execution_integrity,
        )
    )

    compiled = compile_synthetic(contract, mapping_path)
    results.append(
        expect_rejection(
            "loaded_preprocessor_array_write",
            "post_compile_integrity",
            "mutation_attempt",
            ValueError,
            lambda: compiled.preprocessor.mean.__setitem__(0, 0.0),
        )
    )
    return results


def adjacency_invariance_record(
    directory: Path,
    contract: object,
) -> dict[str, Any]:
    full_dir = directory / "adjacency_full"
    full_dir.mkdir()
    full_mapping = write_mapping(full_dir, mapping_rows())
    full = compile_synthetic(contract, full_mapping)

    removed_dir = directory / "adjacency_removed"
    removed_dir.mkdir()
    surviving = copy.deepcopy(mapping_rows()[1])
    surviving["row_index"] = "0"
    removed_mapping = write_mapping(removed_dir, [surviving])
    removed = compile_synthetic(contract, removed_mapping)

    conditions = {
        "public_contract_equal": (
            full.contract.public_payload()
            == removed.contract.public_payload()
        ),
        "public_plan_equal": full.public_plan() == removed.public_plan(),
        "opacus_epsilon_equal": (
            full.accountant_epsilon_opacus
            == removed.accountant_epsilon_opacus
        ),
        "dp_accounting_epsilon_equal": (
            full.accountant_epsilon_dp_accounting
            == removed.accountant_epsilon_dp_accounting
        ),
        "fixed_q_equal": (
            full.contract.owner_sample_rate
            == removed.contract.owner_sample_rate
        ),
        "fixed_steps_equal": (
            full.contract.total_steps == removed.contract.total_steps
        ),
        "fixed_denominator_equal": (
            full.contract.update_denominator
            == removed.contract.update_denominator
        ),
    }
    if not all(conditions.values()):
        raise RuntimeError("Add/remove public-plan invariance failed")
    return {
        "case_id": "add_remove_owner_public_plan_invariance",
        "family": "adjacency",
        "expected": "all_public_mechanism_fields_equal",
        "observed": "passed",
        "checks": conditions,
        "public_plan_sha256": full.public_plan_sha256,
    }


def aggregate_by_family(
    cases: list[dict[str, str]],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for case in cases:
        family = case["family"]
        result[family] = result.get(family, 0) + 1
    return dict(sorted(result.items()))


def write_json(
    path: Path,
    payload: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(
            ROOT
            / "reports"
            / "v2_mutation_audit_v32_20260724"
            / "public_mutation_audit_v2.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    contract = load_owner_poisson_contract_v2(UCI_CONFIG)
    with tempfile.TemporaryDirectory(prefix="unitdp_v2_mutation_audit_") as tmp:
        temporary = Path(tmp)
        baseline_dir = temporary / "baseline"
        baseline_dir.mkdir()
        baseline_mapping = write_mapping(baseline_dir, mapping_rows())
        baseline = compile_synthetic(contract, baseline_mapping)

        cases = contract_mutation_results()
        cases.extend(duplicate_key_results(temporary))
        cases.extend(mapping_mutation_results(temporary, contract))
        cases.append(
            expect_rejection(
                "wrong_preprocessing_artifact",
                "preprocessing",
                "compile",
                ContractV2Error,
                lambda: compile_owner_poisson_contract_v2(
                    contract,
                    mapping_path=baseline_mapping,
                    preprocessing_artifact_path=WISDM_PREPROCESSOR,
                ),
            )
        )

        release_raw = raw_contract()
        release_raw["execution"]["profile"] = "privacy_release"
        release_raw["execution"]["rng_backend"] = "system_csprng"
        release_contract = parse_owner_poisson_contract_v2(release_raw)
        cases.append(
            expect_rejection(
                "release_executor_not_approved",
                "rng_boundary",
                "require_executable",
                ExecutorNotReadyError,
                lambda: compile_owner_poisson_contract_v2(
                    release_contract,
                    mapping_path=baseline_mapping,
                    preprocessing_artifact_path=UCI_PREPROCESSOR,
                    require_executable=True,
                ),
            )
        )
        cases.extend(
            integrity_mutation_results(contract, baseline_mapping)
        )
        adjacency = adjacency_invariance_record(temporary, contract)

    case_spec = [
        {
            "case_id": case["case_id"],
            "family": case["family"],
            "stage": case["stage"],
            "expected": case["expected"],
        }
        for case in cases
    ]
    report: dict[str, Any] = {
        "schema_version": "unitdp.compiler_mutation_audit_public.v2.1",
        "visibility": "public",
        "status": "passed",
        "scope": "strict_registered_owner_poisson_v2_route",
        "handling": "contains_no_private_mapping_or_research_seed",
        "source_bindings": {
            "execution_source_bundle_sha256": (
                execution_source_bundle_sha256_v2()
            ),
            "audit_script_sha256": file_sha256(Path(__file__)),
            "registered_contract_file_sha256": file_sha256(UCI_CONFIG),
            "registered_preprocessor_file_sha256": (
                file_sha256(UCI_PREPROCESSOR)
            ),
        },
        "baseline": {
            "contract_status": baseline.status,
            "execution_ready": baseline.execution_ready,
            "public_contract_sha256": (
                baseline.contract.public_contract_sha256
            ),
            "public_plan_sha256": baseline.public_plan_sha256,
            "accountant_epsilon_opacus": (
                baseline.accountant_epsilon_opacus
            ),
            "accountant_epsilon_dp_accounting": (
                baseline.accountant_epsilon_dp_accounting
            ),
        },
        "mutation_case_count": len(cases),
        "all_mutations_rejected": all(
            case["observed"] == "rejected" for case in cases
        ),
        "case_count_by_family": aggregate_by_family(cases),
        "mutation_spec_sha256": payload_sha256(case_spec),
        "cases": cases,
        "adjacency_invariance": adjacency,
    }
    report["public_mutation_audit_sha256"] = payload_sha256(report)
    output = Path(args.output)
    write_json(output, report, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "status": report["status"],
                "mutation_case_count": report["mutation_case_count"],
                "all_mutations_rejected": report[
                    "all_mutations_rejected"
                ],
                "adjacency_invariance": adjacency["observed"],
                "public_mutation_audit_sha256": report[
                    "public_mutation_audit_sha256"
                ],
                "output": str(output.resolve()),
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
