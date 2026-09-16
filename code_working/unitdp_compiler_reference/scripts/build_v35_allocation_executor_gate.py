#!/usr/bin/env python3
"""Build the V3.5 Route-A executor/correspondence gate."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
for candidate in (SRC, TESTS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from owner_random_allocation_trace_reference import (  # noqa: E402
    ReferenceRecord,
    replay_owner_random_allocation_linear_trace,
)
from unitdp.benchmark_data_v2 import prepare_uci_har_v2  # noqa: E402
from unitdp.compiler_allocation_v4 import (  # noqa: E402
    compile_owner_random_allocation_contract_v4,
    load_owner_random_allocation_contract_v4,
)
from unitdp.owner_poisson_v2 import _model_state_sha256  # noqa: E402
from unitdp.owner_random_allocation_v4 import (  # noqa: E402
    OwnerRandomAllocationExecutionV4Error,
    _AllocationResearchStreamsV4,
    build_owner_allocation_schedule_v4,
    train_owner_random_allocation_fixed_v4,
)
from unitdp.release_artifacts import (  # noqa: E402
    assert_public_certificate_redacted,
)
from unitdp.source_bundle_allocation_v4 import (  # noqa: E402
    execution_source_bundle_allocation_v4,
    execution_source_bundle_sha256_allocation_v4,
)


DEFAULT_OUTPUT = (
    ROOT
    / "reports"
    / "v35_allocation_executor_gate_20260725"
    / "allocation_executor_gate_v35.json"
)
DEFAULT_MARKDOWN = DEFAULT_OUTPUT.with_suffix(".md")
SCHEMA_VERSION = "unitdp.v35_allocation_executor_gate.v1"
REPORT_DATE = "2026-07-25"
EMPTY_STEP_FIXTURE_SEED = 6761

DEFAULT_DATA_ROOT = ROOT.parent / "DP-SGD" / "data"
DATA_ROOT = Path(
    os.environ.get("UNITDP_DATA_ROOT", str(DEFAULT_DATA_ROOT))
)
UCI_ROOT = DATA_ROOT / "uci_har" / "extracted" / "UCI HAR Dataset"
CONFIG = ROOT / "configs/v4/uci_owner_random_allocation_v4.yaml"
MAPPING = (
    ROOT
    / "reports/uci_public_preprocessing_smoke_20260724/"
    "uci_train_mapping.csv"
)
PREPROCESSOR = (
    ROOT
    / "configs/preprocessing/"
    "uci_har_published_train_standard_scaler_v1.json"
)

FROZEN_BASELINE = {
    "src/unitdp/compiler_v3.py": (
        "9a083ecd83943503ceb0d7b061296084c754e6fb6218b2786af85a2896aaa3d0"
    ),
    "dist/compiler_submission_upload_manifest_v3.json": (
        "d5fa835d92cdbb69c8dc4832c032cbc46f53e9a5f53eae6a5906b27bd5bdecbe"
    ),
    "dist/compiler_code_and_data_supplement_aaai27_v3.zip": (
        "7214060b29c91530c67c8c202647be495c5011e6c131839d9e5c271382d4c3b3"
    ),
}

CLAIM_SOURCES = (
    "scripts/build_v35_allocation_executor_gate.py",
    "scripts/verify_v35_allocation_executor_gate.py",
    "src/unitdp/benchmark_registry_allocation_v4.py",
    "src/unitdp/compiler_allocation_v4.py",
    "src/unitdp/owner_random_allocation_v4.py",
    "src/unitdp/source_bundle_allocation_v4.py",
    "tests/owner_random_allocation_trace_reference.py",
    "tests/test_owner_random_allocation_v4.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reference_has_no_production_imports() -> bool:
    source = TESTS / "owner_random_allocation_trace_reference.py"
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
    return not any(
        name == "unitdp" or name.startswith("unitdp.")
        for name in imported
    )


def _joint_support_probe() -> dict[str, Any]:
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
    maximum_deviation = int(np.max(np.abs(joint - 256)))
    return {
        "master_seeds": 4096,
        "joint_cells": joint.tolist(),
        "cells_observed": int(np.sum(joint > 0)),
        "cells_total": 16,
        "expected_per_cell": 256,
        "maximum_absolute_deviation": maximum_deviation,
        "prespecified_tolerance": 80,
        "pass": bool(np.all(joint > 0) and maximum_deviation < 80),
        "interpretation": (
            "finite deterministic implementation probe, not an "
            "independence proof"
        ),
    }


def build_report() -> dict[str, Any]:
    if not (UCI_ROOT / "train/X_train.txt").is_file():
        raise FileNotFoundError(
            "Registered UCI HAR source is required for this gate"
        )
    torch.set_num_threads(1)
    prepared = prepare_uci_har_v2(
        dataset_root=UCI_ROOT,
        preprocessing_artifact_path=PREPROCESSOR,
    )
    prepared.assert_registered_full_conformance()
    contract = load_owner_random_allocation_contract_v4(CONFIG)
    route = compile_owner_random_allocation_contract_v4(
        contract,
        mapping_path=MAPPING,
        preprocessing_artifact_path=PREPROCESSOR,
        require_executable=True,
    )
    result = train_owner_random_allocation_fixed_v4(
        route=route,
        raw_train_x=prepared.raw_train_x,
        train_y=prepared.train_y,
        raw_test_x=prepared.raw_test_x,
        test_y=prepared.test_y,
        research_seed=EMPTY_STEP_FIXTURE_SEED,
    )
    public = result.public_payload()
    private = result.private_manifest()
    assert_public_certificate_redacted(public)

    records = tuple(
        ReferenceRecord(
            owner_id=record.owner_ids[0],
            window_id=record.window_id,
            start=record.start,
            end=record.end,
            row_index=record.row_index,
        )
        for record in route.selected_mapping.records
    )
    reference = replay_owner_random_allocation_linear_trace(
        raw_train_x=prepared.raw_train_x,
        train_y=prepared.train_y,
        records=records,
        preprocessing_artifact_path=PREPROCESSOR,
        master_seed=EMPTY_STEP_FIXTURE_SEED,
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
    diagnostics = private["step_diagnostics"]
    owner_rows = private["local_steps_by_epoch_and_owner"]
    observed_schedule = tuple(
        tuple(row["assigned_owner_positions"]) for row in diagnostics
    )
    observed_selected_rows = tuple(
        tuple(
            (
                item["owner_position"],
                tuple(item["row_indices"]),
            )
            for item in row["selected_rows"]
        )
        for row in diagnostics
    )
    model_tensor_matches = [
        bool(torch.equal(observed, expected))
        for observed, expected in zip(
            result.model.parameters(),
            reference.model.parameters(),
        )
    ]
    exact_k_rows = [
        len(selected) == contract.num_selected_steps_per_owner
        and len(set(selected)) == contract.num_selected_steps_per_owner
        and all(0 <= value < contract.num_steps for value in selected)
        for epoch in owner_rows
        for selected in epoch
    ]
    empty_steps = [
        row["step"]
        for row in diagnostics
        if row["assigned_owner_count"] == 0
    ]

    mutated = np.array(prepared.raw_train_x, copy=True)
    mutated[0, 0] += np.float32(0.25)
    mutation_rejected = False
    try:
        train_owner_random_allocation_fixed_v4(
            route=route,
            raw_train_x=mutated,
            train_y=prepared.train_y,
            raw_test_x=prepared.raw_test_x,
            test_y=prepared.test_y,
            research_seed=EMPTY_STEP_FIXTURE_SEED,
        )
    except OwnerRandomAllocationExecutionV4Error:
        mutation_rejected = True

    correspondence = {
        "reference_has_no_unitdp_imports": (
            _reference_has_no_production_imports()
        ),
        "allocation_schedule_exact": (
            observed_schedule
            == reference.assigned_owner_positions
        ),
        "owner_schedule_rows_exact": (
            tuple(
                tuple(tuple(value) for value in epoch)
                for epoch in owner_rows
            )
            == reference.local_steps_by_epoch_and_owner
        ),
        "within_owner_rows_exact": (
            observed_selected_rows
            == reference.selected_rows_by_position
        ),
        "unclipped_norms_exact": (
            tuple(
                row["max_unclipped_owner_norm"]
                for row in diagnostics
            )
            == reference.max_unclipped_norms
        ),
        "clipped_norms_exact": (
            tuple(
                row["max_clipped_owner_norm"] for row in diagnostics
            )
            == reference.max_clipped_norms
        ),
        "model_tensors_bitwise": all(model_tensor_matches),
    }
    execution_checks = {
        "execution_ready": route.execution_ready,
        "exact_k_distinct_all_owners": all(exact_k_rows),
        "owner_rows_checked": (
            len(exact_k_rows)
            == contract.source_dataset_size * contract.num_epochs
        ),
        "one_prespecified_empty_step": empty_steps == [12],
        "noise_on_every_step": all(
            row["noise_parameter_tensors"] == 2
            for row in diagnostics
        ),
        "optimizer_on_every_step": all(
            row["optimizer_step_applied"] is True
            for row in diagnostics
        ),
        "empty_step_zero_owner_vectors": (
            diagnostics[12]["owner_vectors_computed"] == 0
        ),
        "clip_bound": all(
            row["max_clipped_owner_norm"] <= contract.clip_norm
            for row in diagnostics
        ),
        "data_mutation_rejected": mutation_rejected,
        "public_record_redacted": (
            "research_seed" not in json.dumps(public, sort_keys=True)
            and "assigned_owner_positions"
            not in json.dumps(public, sort_keys=True)
        ),
    }
    joint_probe = _joint_support_probe()
    source_bundle = dict(execution_source_bundle_allocation_v4())
    frozen = [
        {
            "path": path,
            "expected_sha256": expected,
            "observed_sha256": file_sha256(ROOT / path),
            "pass": file_sha256(ROOT / path) == expected,
        }
        for path, expected in FROZEN_BASELINE.items()
    ]
    checks = {
        "correspondence": all(correspondence.values()),
        "execution_invariants": all(execution_checks.values()),
        "joint_support_probe": joint_probe["pass"],
        "source_bundle": (
            execution_source_bundle_sha256_allocation_v4(source_bundle)
            == route.execution_source_bundle_sha256
        ),
        "frozen_v3": all(row["pass"] for row in frozen),
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "report_date": REPORT_DATE,
        "candidate": "AAAI27 Algorithm V3.5",
        "object": "Route-A exact-k random-allocation research executor",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "fixture": {
            "fixture_id": "uci_exact_k_empty_step_v1",
            "research_seed": EMPTY_STEP_FIXTURE_SEED,
            "release_status": "fixture_only_not_a_privacy_release",
            "owners": contract.source_dataset_size,
            "num_steps": contract.num_steps,
            "num_selected_steps_per_owner": (
                contract.num_selected_steps_per_owner
            ),
            "num_epochs": contract.num_epochs,
            "assigned_owner_vectors": sum(
                row["owner_vectors_computed"] for row in diagnostics
            ),
            "empty_steps": empty_steps,
            "allocation_schedule_sha256": (
                private["allocation_schedule_sha256"]
            ),
            "public_execution_sha256": (
                public["public_execution_sha256"]
            ),
            "private_manifest_sha256": (
                private["private_manifest_sha256"]
            ),
            "production_model_sha256": (
                _model_state_sha256(result.model)
            ),
            "reference_model_sha256": (
                _model_state_sha256(reference.model)
            ),
            "public_evaluation": public["public_evaluation"],
        },
        "correspondence": correspondence,
        "model_tensor_matches": model_tensor_matches,
        "execution_invariants": execution_checks,
        "joint_support_probe": joint_probe,
        "execution_source_bundle": source_bundle,
        "execution_source_bundle_sha256": (
            execution_source_bundle_sha256_allocation_v4(source_bundle)
        ),
        "frozen_v3_baseline": frozen,
        "claim_source_sha256": {
            path: file_sha256(ROOT / path) for path in CLAIM_SOURCES
        },
        "verdict": {
            "executor_gate": (
                "PASS" if all(checks.values()) else "FAIL"
            ),
            "authorizes": [
                "Route-A research-executor status",
                "generic-dispatch integration testing",
            ],
            "does_not_authorize": [
                "a release-grade secure-RNG claim",
                "arbitrary-input implementation equivalence",
                "an independence proof",
                "a privacy proof",
                "a multi-seed utility claim",
                "a manuscript novelty claim",
            ],
        },
    }
    report["payload_sha256"] = canonical_sha256(report)
    return report


def _markdown(report: dict[str, Any]) -> str:
    fixture = report["fixture"]
    return "\n".join(
        [
            "# V3.5 Route-A Executor Gate",
            "",
            f"- Status: `{report['status']}`",
            f"- Payload SHA-256: `{report['payload_sha256']}`",
            (
                "- Exact production/reference correspondence: "
                f"`{sum(report['correspondence'].values())}/"
                f"{len(report['correspondence'])}`"
            ),
            (
                "- Execution invariant checks: "
                f"`{sum(report['execution_invariants'].values())}/"
                f"{len(report['execution_invariants'])}`"
            ),
            (
                "- Assigned owner vectors / empty steps: "
                f"`{fixture['assigned_owner_vectors']} / "
                f"{fixture['empty_steps']}`"
            ),
            "",
            "This is deterministic fixture evidence for a research backend.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--markdown",
        type=Path,
        default=DEFAULT_MARKDOWN,
    )
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["checks"], indent=2, sort_keys=True))
    print(report["payload_sha256"])
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
