from __future__ import annotations

import ast
import copy
import csv
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pytest

import unitdp.benchmark_registry_v2 as benchmark_registry_v2
from unitdp.benchmark_data_v2 import (
    BenchmarkDataV2Error,
    PreparedBenchmarkV2,
)
from unitdp.benchmark_registry_v2 import get_benchmark_profile_v2
from unitdp.compiler_v2 import (
    RDP_ORDERS_V2,
    ContractV2Error,
    compile_owner_poisson_contract_v2,
    parse_owner_poisson_contract_v2,
)
from unitdp.preprocessing import load_fixed_affine_preprocessor
from unitdp.source_bundle_v2 import (
    EXECUTION_SOURCE_FILES_V2,
    execution_source_bundle_sha256_v2,
)
from unitdp_spec_oracle import (
    load_contract_file,
    load_oracle_spec,
    registered_public_projection,
    validate_accountant_environment,
    validate_mapping_array_binding,
    validate_mapping_file,
    validate_preprocessor_artifact,
    validate_registered_accounting,
    validate_registered_research_contract,
    validate_registered_source_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = (
    ROOT / "specs" / "owner_poisson_v2_contract_oracle_v1.json"
)
ORACLE_SOURCE = ROOT / "src" / "unitdp_spec_oracle" / "oracle_v1.py"

CASES = {
    "uci": {
        "config": ROOT / "configs" / "v2" / "uci_owner_poisson_v2.yaml",
        "mapping": (
            ROOT
            / "reports"
            / "uci_public_preprocessing_smoke_20260724"
            / "uci_train_mapping.csv"
        ),
        "preprocessor": (
            ROOT
            / "configs"
            / "preprocessing"
            / "uci_har_published_train_standard_scaler_v1.json"
        ),
    },
    "wisdm": {
        "config": (
            ROOT / "configs" / "v2" / "wisdm_owner_poisson_v2.yaml"
        ),
        "mapping": (
            ROOT
            / "reports"
            / "wisdm_public_protocol_smoke_20260724"
            / "wisdm_train_mapping.csv"
        ),
        "preprocessor": (
            ROOT
            / "configs"
            / "preprocessing"
            / "wisdm_v1_1_train_owners_1_25_stats24_standard_scaler_v1.json"
        ),
    },
    "sepsis": {
        "config": (
            ROOT / "configs" / "v2" / "sepsis_owner_poisson_v2.yaml"
        ),
        "mapping": (
            ROOT
            / "reports"
            / "sepsis_public_protocol_smoke_20260724"
            / "sepsis_train_mapping.csv"
        ),
        "preprocessor": (
            ROOT
            / "configs"
            / "preprocessing"
            / "physionet2019_setA_first400_basic12x6_public_scaler_v1.json"
        ),
    },
}


def _detail(result: object, key: str) -> object:
    return dict(getattr(result, "details"))[key]


def _iter_nodes(
    value: object,
    path: tuple[str | int, ...] = (),
) -> Iterable[tuple[tuple[str | int, ...], object]]:
    yield path, value
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _iter_nodes(value[key], path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_nodes(item, path + (index,))


def _replace_at_path(
    value: object,
    path: tuple[str | int, ...],
    replacement: object,
) -> object:
    if not path:
        return copy.deepcopy(replacement)
    result = copy.deepcopy(value)
    cursor: Any = result
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = copy.deepcopy(replacement)
    return result


def _delete_at_path(
    value: object,
    path: tuple[str | int, ...],
) -> object:
    if not path:
        raise ValueError("Cannot delete the root")
    result = copy.deepcopy(value)
    cursor: Any = result
    for part in path[:-1]:
        cursor = cursor[part]
    del cursor[path[-1]]
    return result


def _replacement_values(value: object) -> list[object]:
    if isinstance(value, bool):
        return [not value, 0, 1, None, "false"]
    if isinstance(value, int):
        return [
            0,
            -1,
            float(value),
            str(value),
            None,
            False,
            value + 1,
        ]
    if isinstance(value, float):
        return [
            0,
            -1.0,
            int(value),
            str(value),
            None,
            False,
            math.nan,
            math.inf,
            value + 0.125,
        ]
    if isinstance(value, str):
        return ["", value + "__mutated", 0, None, False, []]
    if value is None:
        return [[], [1.0], {}, "", 0, False]
    if isinstance(value, list):
        return [None, {}, "", 0, False, [], value + [None]]
    if isinstance(value, dict):
        return [None, [], "", 0, False, {}]
    raise TypeError(f"Unsupported JSON-compatible value: {type(value)!r}")


def _path_text(path: tuple[str | int, ...]) -> str:
    if not path:
        return "$"
    return "$." + ".".join(str(part) for part in path)


def _fingerprint(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=True,
    )


def _contract_mutations(
    baseline: dict[str, Any],
) -> Iterable[tuple[str, object]]:
    seen = {_fingerprint(baseline)}
    for path, value in list(_iter_nodes(baseline)):
        for replacement in _replacement_values(value):
            mutated = _replace_at_path(baseline, path, replacement)
            fingerprint = _fingerprint(mutated)
            if fingerprint not in seen:
                seen.add(fingerprint)
                yield (
                    f"replace:{_path_text(path)}:{type(replacement).__name__}",
                    mutated,
                )
        if path:
            mutated = _delete_at_path(baseline, path)
            fingerprint = _fingerprint(mutated)
            if fingerprint not in seen:
                seen.add(fingerprint)
                yield f"delete:{_path_text(path)}", mutated
        if isinstance(value, dict):
            mutated = copy.deepcopy(baseline)
            cursor: Any = mutated
            for part in path:
                cursor = cursor[part]
            cursor["__unregistered_field__"] = "mutation"
            fingerprint = _fingerprint(mutated)
            if fingerprint not in seen:
                seen.add(fingerprint)
                yield f"unknown:{_path_text(path)}", mutated


def _production_registered_research_accepts(
    raw: object,
    *,
    expected_contract_sha256: str,
) -> bool:
    try:
        contract = parse_owner_poisson_contract_v2(raw)
    except ContractV2Error:
        return False
    return (
        contract.execution_profile == "research_benchmark"
        and contract.rng_backend == "research_default"
        and contract.public_contract_sha256 == expected_contract_sha256
    )


def _write_mapping(
    path: Path,
    columns: tuple[str, ...],
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _mapping_rows(*, patient_file: bool = False) -> list[dict[str, str]]:
    rows = [
        {
            "scenario": "registered",
            "window_id": "w0",
            "owner_id": "owner-a",
            "owner_ids": "owner-a",
            "start": "0",
            "end": "4",
            "row_index": "0",
            "label": "0",
        },
        {
            "scenario": "registered",
            "window_id": "w1",
            "owner_id": "owner-b",
            "owner_ids": "owner-b",
            "start": "4",
            "end": "8",
            "row_index": "1",
            "label": "1",
        },
    ]
    if patient_file:
        rows[0]["patient_file"] = "p000001.psv"
        rows[1]["patient_file"] = "p000002.psv"
    return rows


def _production_compile_accepts(
    raw: dict[str, Any],
    *,
    mapping_path: Path,
    preprocessor_path: Path,
) -> bool:
    try:
        contract = parse_owner_poisson_contract_v2(raw)
        compile_owner_poisson_contract_v2(
            contract,
            mapping_path=mapping_path,
            preprocessing_artifact_path=preprocessor_path,
        )
    except ContractV2Error:
        return False
    return True


def test_oracle_source_has_no_production_import() -> None:
    tree = ast.parse(ORACLE_SOURCE.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    forbidden = [
        name
        for name in imported
        if name == "unitdp" or name.startswith("unitdp.")
    ]
    assert forbidden == []


def test_registered_environment_and_source_bundle_match_independently() -> None:
    spec = load_oracle_spec(SPEC_PATH)
    source_result = validate_registered_source_bundle(ROOT, spec)
    accountant_result = validate_accountant_environment(spec)

    assert source_result.accepted, source_result
    assert accountant_result.accepted, accountant_result
    assert tuple(spec["x-unitdp-oracle"]["rdp_orders"]) == RDP_ORDERS_V2
    assert tuple(
        spec["x-unitdp-oracle"]["execution_source_bundle"]["files"]
    ) == EXECUTION_SOURCE_FILES_V2
    assert (
        _detail(source_result, "source_bundle_sha256")
        == execution_source_bundle_sha256_v2()
    )


@pytest.mark.parametrize("case_name", tuple(CASES))
def test_registered_artifacts_agree_with_production(case_name: str) -> None:
    paths = CASES[case_name]
    spec = load_oracle_spec(SPEC_PATH)
    raw = load_contract_file(paths["config"])

    contract_result = validate_registered_research_contract(raw, spec)
    mapping_result = validate_mapping_file(
        paths["mapping"],
        profile_id=raw["data"]["benchmark_profile_id"],
        spec=spec,
    )
    preprocessor_result = validate_preprocessor_artifact(
        paths["preprocessor"],
        contract=raw,
        spec=spec,
    )
    accounting_result = validate_registered_accounting(raw, spec)

    assert contract_result.accepted, contract_result
    assert mapping_result.accepted, mapping_result
    assert preprocessor_result.accepted, preprocessor_result
    assert accounting_result.accepted, accounting_result

    production_contract = parse_owner_poisson_contract_v2(raw)
    production_preprocessor = load_fixed_affine_preprocessor(
        paths["preprocessor"],
        expected_payload_sha256=(
            production_contract.preprocessing_binding["artifact_sha256"]
        ),
    )
    compiled = compile_owner_poisson_contract_v2(
        production_contract,
        mapping_path=paths["mapping"],
        preprocessing_artifact_path=paths["preprocessor"],
    )
    projection = registered_public_projection(raw, spec)

    assert (
        _detail(contract_result, "public_contract_sha256")
        == production_contract.public_contract_sha256
    )
    assert (
        _detail(preprocessor_result, "payload_sha256")
        == production_preprocessor.payload_sha256
    )
    assert (
        _detail(mapping_result, "record_count")
        == len(compiled.source_mapping.records)
    )
    assert (
        projection["execution_source_bundle_sha256"]
        == compiled.execution_source_bundle_sha256
    )
    assert (
        _detail(accounting_result, "accountant_epsilon_opacus")
        == compiled.accountant_epsilon_opacus
    )
    assert (
        _detail(accounting_result, "accountant_epsilon_dp_accounting")
        == compiled.accountant_epsilon_dp_accounting
    )


def test_generative_contract_judgments_match_production() -> None:
    spec = load_oracle_spec(SPEC_PATH)
    disagreements: list[str] = []
    total = 0
    accepted = 0
    baseline_accepted = 0
    nonbaseline_accepted = 0

    for case_name, paths in CASES.items():
        baseline = load_contract_file(paths["config"])
        profile_id = baseline["data"]["benchmark_profile_id"]
        expected_sha256 = spec["x-unitdp-oracle"]["profiles"][profile_id][
            "public_contract_sha256"
        ]
        candidates = [("baseline", baseline), *_contract_mutations(baseline)]
        for label, candidate in candidates:
            total += 1
            oracle_result = validate_registered_research_contract(
                candidate,
                spec,
            )
            production_accepted = _production_registered_research_accepts(
                candidate,
                expected_contract_sha256=expected_sha256,
            )
            accepted += int(oracle_result.accepted)
            baseline_accepted += int(
                oracle_result.accepted and label == "baseline"
            )
            nonbaseline_accepted += int(
                oracle_result.accepted and label != "baseline"
            )
            if oracle_result.accepted != production_accepted:
                disagreements.append(
                    (
                        f"{case_name}:{label}:"
                        f"oracle={oracle_result.accepted}:"
                        f"{oracle_result.stage}:{oracle_result.reason}:"
                        f"production={production_accepted}"
                    )
                )

    assert total == 1_486
    assert accepted == 12
    assert baseline_accepted == 3
    assert nonbaseline_accepted == 9
    assert disagreements == [], "\n".join(disagreements[:25])


def test_mapping_judgments_match_compiler(tmp_path: Path) -> None:
    spec = load_oracle_spec(SPEC_PATH)
    paths = CASES["uci"]
    raw = load_contract_file(paths["config"])
    profile_id = raw["data"]["benchmark_profile_id"]
    columns = get_benchmark_profile_v2(profile_id).mapping_columns
    baseline_rows = _mapping_rows()

    mutations: list[tuple[str, tuple[str, ...], list[dict[str, str]]]] = [
        ("baseline", columns, baseline_rows),
    ]
    field_mutations = {
        "duplicate_window": ("window_id", "w0"),
        "duplicate_row": ("row_index", "0"),
        "row_gap": ("row_index", "2"),
        "owner_mismatch": ("owner_ids", "other-owner"),
        "multi_owner": ("owner_ids", "owner-b;other-owner"),
        "whitespace": ("scenario", " registered"),
        "empty_label": ("label", ""),
        "leading_zero": ("start", "04"),
        "decimal_integer": ("end", "8.0"),
        "negative_integer": ("start", "-1"),
        "nonpositive_span": ("end", "4"),
    }
    for label, (field, value) in field_mutations.items():
        rows = copy.deepcopy(baseline_rows)
        rows[1][field] = value
        mutations.append((label, columns, rows))
    mutations.extend(
        [
            ("missing_column", columns[:-1], baseline_rows),
            ("reordered_columns", tuple(reversed(columns)), baseline_rows),
            ("extra_column", columns + ("extra",), baseline_rows),
        ]
    )

    disagreements: list[str] = []
    judgment_count = 0
    for label, candidate_columns, rows in mutations:
        judgment_count += 1
        mapping_path = tmp_path / f"{label}.csv"
        _write_mapping(mapping_path, candidate_columns, rows)
        oracle_accepted = validate_mapping_file(
            mapping_path,
            profile_id=profile_id,
            spec=spec,
        ).accepted
        production_accepted = _production_compile_accepts(
            raw,
            mapping_path=mapping_path,
            preprocessor_path=paths["preprocessor"],
        )
        if oracle_accepted != production_accepted:
            disagreements.append(
                f"{label}:oracle={oracle_accepted}:production={production_accepted}"
            )

    sepsis_paths = CASES["sepsis"]
    sepsis_raw = load_contract_file(sepsis_paths["config"])
    sepsis_profile_id = sepsis_raw["data"]["benchmark_profile_id"]
    sepsis_columns = get_benchmark_profile_v2(
        sepsis_profile_id
    ).mapping_columns
    sepsis_rows = _mapping_rows(patient_file=True)
    sepsis_rows[0]["patient_file"] = ""
    sepsis_mapping = tmp_path / "sepsis_empty_patient.csv"
    _write_mapping(sepsis_mapping, sepsis_columns, sepsis_rows)
    judgment_count += 1
    oracle_accepted = validate_mapping_file(
        sepsis_mapping,
        profile_id=sepsis_profile_id,
        spec=spec,
    ).accepted
    production_accepted = _production_compile_accepts(
        sepsis_raw,
        mapping_path=sepsis_mapping,
        preprocessor_path=sepsis_paths["preprocessor"],
    )
    if oracle_accepted != production_accepted:
        disagreements.append(
            "sepsis_empty_patient:"
            f"oracle={oracle_accepted}:production={production_accepted}"
        )

    assert judgment_count == 16
    assert disagreements == [], "\n".join(disagreements)


def test_mapping_array_binding_is_independent_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = load_oracle_spec(SPEC_PATH)
    profile_id = "uci_har_aaai27_v1"
    columns = get_benchmark_profile_v2(profile_id).mapping_columns
    mapping_path = tmp_path / "mapping.csv"
    _write_mapping(mapping_path, columns, _mapping_rows())

    accepted = validate_mapping_array_binding(
        mapping_path,
        profile_id=profile_id,
        feature_shape=(2, 561),
        labels=[0, 1],
        spec=spec,
    )
    assert accepted.accepted, accepted

    rejected = (
        validate_mapping_array_binding(
            mapping_path,
            profile_id=profile_id,
            feature_shape=(3, 561),
            labels=[0, 1, 0],
            spec=spec,
        ),
        validate_mapping_array_binding(
            mapping_path,
            profile_id=profile_id,
            feature_shape=(2, 560),
            labels=[0, 1],
            spec=spec,
        ),
        validate_mapping_array_binding(
            mapping_path,
            profile_id=profile_id,
            feature_shape=(2, 561),
            labels=[1, 0],
            spec=spec,
        ),
        validate_mapping_array_binding(
            mapping_path,
            profile_id=profile_id,
            feature_shape=(2, 561),
            labels=[0.0, 1.0],
            spec=spec,
        ),
        validate_mapping_array_binding(
            mapping_path,
            profile_id=profile_id,
            feature_shape=(2, 561),
            labels=[0, 1],
            spec=spec,
            require_registered_train_rows=True,
        ),
    )
    assert all(not result.accepted for result in rejected), rejected

    preprocessor = load_fixed_affine_preprocessor(CASES["uci"]["preprocessor"])
    baseline_features = np.zeros((2, 561), dtype=np.float32)
    baseline = PreparedBenchmarkV2(
        profile_id="synthetic_exact_data_profile",
        raw_train_x=baseline_features,
        train_y=np.asarray([0, 1], dtype=np.int64),
        raw_test_x=baseline_features.copy(),
        test_y=np.asarray([0, 1], dtype=np.int64),
        train_records=tuple(_mapping_rows()),
        test_records=tuple(_mapping_rows()),
        preprocessor=preprocessor,
        source_files=(),
        data_preparation_implementation_id="synthetic_preparer_v1",
    )

    class SyntheticProfile:
        data_preparation_implementation_id = "synthetic_preparer_v1"
        full_data_conformance_sha256 = (
            baseline.full_data_conformance_sha256()
        )

        @staticmethod
        def expected_contract_binding() -> dict[str, object]:
            return {
                "data": {
                    "preprocessing": preprocessor.contract_binding(),
                }
            }

        @staticmethod
        def full_data_conformance_payload() -> dict[str, object]:
            return baseline.full_data_conformance_payload()

    monkeypatch.setattr(
        benchmark_registry_v2,
        "get_benchmark_profile_v2",
        lambda profile_id: SyntheticProfile(),
    )
    baseline.assert_registered_full_conformance()

    same_shape_mutation = baseline.raw_train_x.copy()
    same_shape_mutation[0, 0] = 1.0
    mutated = replace(baseline, raw_train_x=same_shape_mutation)
    assert mutated.raw_train_x.shape == baseline.raw_train_x.shape
    assert (
        mutated.full_data_conformance_sha256()
        != baseline.full_data_conformance_sha256()
    )
    with pytest.raises(
        BenchmarkDataV2Error,
        match="Prepared full benchmark differs",
    ):
        mutated.assert_registered_full_conformance()
