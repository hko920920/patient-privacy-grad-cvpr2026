"""Build the V3.5 multi-profile random-allocation execution gate.

The public tree contains no research seeds or allocation traces.  A sibling
``_private`` tree is retained only for the production-import-free replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
for source in (SRC, TESTS):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from owner_random_allocation_trace_reference import (  # noqa: E402
    ReferenceRecord,
    replay_owner_random_allocation_linear_trace,
)
from unitdp.benchmark_data_v2 import (  # noqa: E402
    PreparedBenchmarkV2,
    prepare_sepsis_v2,
    prepare_uci_har_v2,
    prepare_wisdm_v2,
    write_prepared_mapping_v2,
)
from unitdp.benchmark_registry_allocation_v4 import (  # noqa: E402
    get_allocation_benchmark_profile_v4,
)
from unitdp.compiler_allocation_v4 import (  # noqa: E402
    compile_owner_random_allocation_contract_v4,
    load_owner_random_allocation_contract_v4,
)
from unitdp.execution_artifacts_allocation_v4 import (  # noqa: E402
    write_allocation_execution_artifacts_v4,
)
from unitdp.owner_poisson_v2 import _model_state_sha256  # noqa: E402
from unitdp.owner_random_allocation_v4 import (  # noqa: E402
    train_owner_random_allocation_fixed_v4,
)
from unitdp.release_artifacts import (  # noqa: E402
    assert_public_certificate_redacted,
)


DEFAULT_DATA_ROOT = ROOT.parent / "DP-SGD" / "data"
DATA_ROOT = Path(
    os.environ.get("UNITDP_DATA_ROOT", str(DEFAULT_DATA_ROOT))
)
DEFAULT_OUTPUT = (
    ROOT / "reports" / "v35_allocation_multirun_gate_20260725"
)
DEFAULT_SEEDS = (13, 23, 31, 37, 41)

DATASET_CONFIG: dict[str, dict[str, Any]] = {
    "uci": {
        "profile_id": "uci_har_random_allocation_aaai27_v4",
        "contract": (
            ROOT / "configs" / "v4"
            / "uci_owner_random_allocation_v4.yaml"
        ),
        "preprocessor": (
            ROOT / "configs" / "preprocessing"
            / "uci_har_published_train_standard_scaler_v1.json"
        ),
        "raw": (
            DATA_ROOT / "uci_har" / "extracted" / "UCI HAR Dataset"
        ),
    },
    "wisdm": {
        "profile_id": "wisdm_random_allocation_aaai27_v4",
        "contract": (
            ROOT / "configs" / "v4"
            / "wisdm_owner_random_allocation_v4.yaml"
        ),
        "preprocessor": (
            ROOT / "configs" / "preprocessing"
            / "wisdm_v1_1_train_owners_1_25_stats24_standard_scaler_v1.json"
        ),
        "raw": (
            DATA_ROOT / "wisdm" / "WISDM_ar_v1.1"
            / "WISDM_ar_v1.1_raw.txt"
        ),
    },
    "sepsis": {
        "profile_id": "sepsis_random_allocation_aaai27_v4",
        "contract": (
            ROOT / "configs" / "v4"
            / "sepsis_owner_random_allocation_v4.yaml"
        ),
        "preprocessor": (
            ROOT / "configs" / "preprocessing"
            / "physionet2019_setA_first400_basic12x6_public_scaler_v1.json"
        ),
        "raw": DATA_ROOT / "sepsis2019_cache",
    },
}


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


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(
        int(token.strip())
        for token in value.split(",")
        if token.strip()
    )
    if (
        not seeds
        or len(seeds) != len(set(seeds))
        or any(seed < 0 or seed >= 2**63 for seed in seeds)
    ):
        raise ValueError(
            "research seeds must be distinct integers in [0, 2**63)"
        )
    return seeds


def prepare_dataset(dataset: str) -> PreparedBenchmarkV2:
    config = DATASET_CONFIG[dataset]
    if dataset == "uci":
        prepared = prepare_uci_har_v2(
            dataset_root=config["raw"],
            preprocessing_artifact_path=config["preprocessor"],
        )
    elif dataset == "wisdm":
        prepared = prepare_wisdm_v2(
            raw_path=config["raw"],
            preprocessing_artifact_path=config["preprocessor"],
        )
    elif dataset == "sepsis":
        prepared = prepare_sepsis_v2(
            cache_dir=config["raw"],
            preprocessing_artifact_path=config["preprocessor"],
        )
    else:
        raise KeyError(dataset)
    prepared.assert_registered_full_conformance()
    return prepared


def metric_summary(
    runs: list[dict[str, Any]],
    key: str,
) -> dict[str, float | int | None]:
    values = [
        float(run["public_evaluation"][key])
        for run in runs
        if run["public_evaluation"][key] is not None
    ]
    if not values:
        return {"count": 0, "mean": None, "std": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
    }


def _reference_records(route: Any) -> tuple[ReferenceRecord, ...]:
    return tuple(
        ReferenceRecord(
            owner_id=record.owner_ids[0],
            window_id=record.window_id,
            start=record.start,
            end=record.end,
            row_index=record.row_index,
        )
        for record in route.selected_mapping.records
    )


def _assert_reference_equal(
    *,
    result: Any,
    reference: Any,
    contract: Any,
) -> dict[str, int]:
    private = result.private_manifest()
    diagnostics = private["step_diagnostics"]
    observed_schedule = tuple(
        tuple(row["assigned_owner_positions"]) for row in diagnostics
    )
    observed_owner_rows = tuple(
        tuple(tuple(value) for value in epoch)
        for epoch in private["local_steps_by_epoch_and_owner"]
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
    observed_unclipped = tuple(
        row["max_unclipped_owner_norm"] for row in diagnostics
    )
    observed_clipped = tuple(
        row["max_clipped_owner_norm"] for row in diagnostics
    )
    if observed_schedule != reference.assigned_owner_positions:
        raise AssertionError("reference allocation schedule mismatch")
    if observed_owner_rows != reference.local_steps_by_epoch_and_owner:
        raise AssertionError("reference owner-wise schedule mismatch")
    if observed_selected_rows != reference.selected_rows_by_position:
        raise AssertionError("reference selected-row trace mismatch")
    if observed_unclipped != reference.max_unclipped_norms:
        raise AssertionError("reference unclipped-norm trace mismatch")
    if observed_clipped != reference.max_clipped_norms:
        raise AssertionError("reference clipped-norm trace mismatch")

    tensor_checks = 0
    for observed, expected in zip(
        result.model.parameters(),
        reference.model.parameters(),
        strict=True,
    ):
        if not torch.equal(observed, expected):
            raise AssertionError("reference final tensor mismatch")
        tensor_checks += 1
    if (
        _model_state_sha256(result.model)
        != _model_state_sha256(reference.model)
    ):
        raise AssertionError("reference final model hash mismatch")

    owner_epoch_rows = 0
    for epoch in observed_owner_rows:
        for selected in epoch:
            if len(selected) != contract.num_selected_steps_per_owner:
                raise AssertionError("owner does not have exact k assignments")
            if len(set(selected)) != len(selected):
                raise AssertionError("owner assignment contains duplicates")
            owner_epoch_rows += 1
    for row in diagnostics:
        if row["owner_vectors_computed"] != row["assigned_owner_count"]:
            raise AssertionError("owner-vector count mismatch")
        if row["noise_parameter_tensors"] != 2:
            raise AssertionError("one step did not noise both parameters")
        if row["optimizer_step_applied"] is not True:
            raise AssertionError("one optimizer step was skipped")
        if row["max_clipped_owner_norm"] > contract.clip_norm:
            raise AssertionError("clipped norm exceeds C")
    return {
        "steps": len(diagnostics),
        "owner_vectors": sum(
            row["owner_vectors_computed"] for row in diagnostics
        ),
        "owner_epoch_rows": owner_epoch_rows,
        "empty_steps": sum(
            row["assigned_owner_count"] == 0 for row in diagnostics
        ),
        "tensor_checks": tensor_checks,
    }


def run_dataset(
    dataset: str,
    *,
    seeds: tuple[int, ...],
    public_root: Path,
    private_root: Path,
) -> tuple[dict[str, Any], dict[str, int]]:
    config = DATASET_CONFIG[dataset]
    profile = get_allocation_benchmark_profile_v4(
        config["profile_id"]
    )
    prepared = prepare_dataset(dataset)
    with tempfile.TemporaryDirectory(
        prefix=f"unitdp_{dataset}_allocation_v4_"
    ) as temporary:
        mapping_path = write_prepared_mapping_v2(
            prepared,
            Path(temporary) / "train_mapping.csv",
        )
        route = compile_owner_random_allocation_contract_v4(
            load_owner_random_allocation_contract_v4(
                config["contract"]
            ),
            mapping_path=mapping_path,
            preprocessing_artifact_path=config["preprocessor"],
            require_executable=True,
        )
    if route.profile != profile:
        raise AssertionError("compiled route selected the wrong profile")

    runs: list[dict[str, Any]] = []
    private_index: list[dict[str, Any]] = []
    totals = {
        "production_runs": 0,
        "reference_runs": 0,
        "steps": 0,
        "owner_vectors": 0,
        "owner_epoch_rows": 0,
        "empty_steps": 0,
        "tensor_checks": 0,
    }
    records = _reference_records(route)
    for index, seed in enumerate(seeds, start=1):
        run_id = f"run_{index:03d}"
        prepared.assert_registered_full_conformance()
        result = train_owner_random_allocation_fixed_v4(
            route=route,
            raw_train_x=prepared.raw_train_x,
            train_y=prepared.train_y,
            raw_test_x=prepared.raw_test_x,
            test_y=prepared.test_y,
            research_seed=seed,
        )
        written = write_allocation_execution_artifacts_v4(
            result,
            public_root / dataset / run_id,
        )
        public = result.public_payload()
        private = result.private_manifest()
        private_path = (
            private_root / dataset / run_id
            / "private_execution_random_allocation_v4.json"
        )
        write_json(private_path, private)
        private_index.append(
            {
                "run_id": run_id,
                "research_seed": seed,
                "private_manifest": str(
                    private_path.relative_to(private_root)
                ).replace("\\", "/"),
            }
        )

        reference = replay_owner_random_allocation_linear_trace(
            raw_train_x=prepared.raw_train_x,
            train_y=prepared.train_y,
            records=records,
            preprocessing_artifact_path=config["preprocessor"],
            master_seed=seed,
            input_dim=route.contract.input_dim,
            num_classes=route.contract.num_classes,
            source_dataset_size=route.contract.source_dataset_size,
            num_steps=route.contract.num_steps,
            num_selected_steps_per_owner=(
                route.contract.num_selected_steps_per_owner
            ),
            num_epochs=route.contract.num_epochs,
            clip_norm=route.contract.clip_norm,
            noise_multiplier=route.contract.noise_multiplier,
            update_denominator=route.contract.update_denominator,
            learning_rate=route.contract.learning_rate,
            windows_per_owner_per_step=(
                route.contract.windows_per_owner_per_step
            ),
            class_weights=route.contract.class_weights,
        )
        counts = _assert_reference_equal(
            result=result,
            reference=reference,
            contract=route.contract,
        )
        totals["production_runs"] += 1
        totals["reference_runs"] += 1
        for key, value in counts.items():
            totals[key] += value
        runs.append(
            {
                "run_id": run_id,
                "public_execution_sha256": public[
                    "public_execution_sha256"
                ],
                "public_bundle_payload_sha256": (
                    written.public_bundle_payload_sha256
                ),
                "model_state_sha256": public["model_output"][
                    "state_sha256"
                ],
                "public_evaluation": public["public_evaluation"],
            }
        )
        print(
            f"{dataset} {run_id}: reference exact; "
            f"steps={counts['steps']}, "
            f"owner_vectors={counts['owner_vectors']}",
            flush=True,
        )

    contract = route.contract
    summary: dict[str, Any] = {
        "schema_version": (
            "unitdp.random_allocation_benchmark_summary_public.v4"
        ),
        "visibility": "public",
        "release_status": "research_non_release",
        "dataset": dataset,
        "benchmark_profile_id": profile.profile_id,
        "benchmark_profile_sha256": profile.registry_sha256,
        "base_data_profile_id": profile.base_data_profile_id,
        "full_data_conformance_sha256": (
            profile.base_profile.full_data_conformance_sha256
        ),
        "public_contract_sha256": contract.public_contract_sha256,
        "public_plan_sha256": route.public_plan_sha256,
        "execution_source_bundle_sha256": (
            route.execution_source_bundle_sha256
        ),
        "mechanism": {
            "adjacency": contract.adjacency,
            "sampler": contract.sampler,
            "source_dataset_size": contract.source_dataset_size,
            "num_steps": contract.num_steps,
            "num_selected_steps_per_owner": (
                contract.num_selected_steps_per_owner
            ),
            "num_epochs": contract.num_epochs,
            "total_steps": contract.total_steps,
            "actual_noise_multiplier": contract.noise_multiplier,
            "sensitivity_multiplier": contract.sensitivity_multiplier,
            "update_denominator": contract.update_denominator,
        },
        "accountant": {
            "epsilon": route.accountant_epsilon,
            "epsilon_remove": route.accountant_epsilon_remove,
            "epsilon_add": route.accountant_epsilon_add,
            "high_precision_oracle_epsilon": route.oracle_epsilon,
            "delta": contract.delta,
        },
        "run_count": len(runs),
        "runs": runs,
        "aggregate": {
            key: metric_summary(runs, key)
            for key in ("accuracy", "macro_f1", "auroc", "auprc")
        },
    }
    summary["public_summary_sha256"] = payload_sha256(summary)
    assert_public_certificate_redacted(summary)
    write_json(
        public_root / dataset
        / "public_summary_random_allocation_v4.json",
        summary,
    )
    private_payload: dict[str, Any] = {
        "schema_version": (
            "unitdp.random_allocation_run_index_private.v4"
        ),
        "visibility": "private",
        "handling": "do_not_publish_contains_research_seeds",
        "runs": private_index,
    }
    private_payload["private_index_sha256"] = payload_sha256(
        private_payload
    )
    write_json(
        private_root / dataset
        / "private_run_index_random_allocation_v4.json",
        private_payload,
    )
    return summary, totals


def deterministic_zip(source: Path, target: Path) -> str:
    with zipfile.ZipFile(
        target,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(
            value for value in source.rglob("*") if value.is_file()
        ):
            relative = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(relative, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    return file_sha256(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--research-seeds",
        default=",".join(str(value) for value in DEFAULT_SEEDS),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    seeds = parse_seeds(args.research_seeds)
    if seeds != DEFAULT_SEEDS:
        raise ValueError(
            "The claim-bearing gate requires the prespecified seed tuple"
        )
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite claim-bearing output: {output}"
        )
    public_root = output / "public_artifacts"
    private_root = output / "_private"
    public_root.mkdir(parents=True)
    private_root.mkdir(parents=True)

    summaries: list[dict[str, Any]] = []
    totals = {
        "production_runs": 0,
        "reference_runs": 0,
        "steps": 0,
        "owner_vectors": 0,
        "owner_epoch_rows": 0,
        "empty_steps": 0,
        "tensor_checks": 0,
    }
    for dataset in ("uci", "wisdm", "sepsis"):
        summary, dataset_totals = run_dataset(
            dataset,
            seeds=seeds,
            public_root=public_root,
            private_root=private_root,
        )
        summaries.append(summary)
        for key, value in dataset_totals.items():
            totals[key] += value

    collection: dict[str, Any] = {
        "schema_version": (
            "unitdp.random_allocation_benchmark_collection_public.v4"
        ),
        "visibility": "public",
        "release_status": "research_non_release",
        "datasets": [
            {
                "dataset": summary["dataset"],
                "public_summary_sha256": summary[
                    "public_summary_sha256"
                ],
                "run_count": summary["run_count"],
            }
            for summary in summaries
        ],
    }
    collection["public_collection_sha256"] = payload_sha256(
        collection
    )
    assert_public_certificate_redacted(collection)
    write_json(
        public_root / "public_collection_random_allocation_v4.json",
        collection,
    )

    archive = output / "random_allocation_public_artifacts_v35.zip"
    archive_sha256 = deterministic_zip(public_root, archive)
    with tempfile.TemporaryDirectory(
        prefix="unitdp_v35_zip_rebuild_"
    ) as temporary:
        rebuilt = Path(temporary) / archive.name
        rebuilt_sha256 = deterministic_zip(public_root, rebuilt)
        if archive.read_bytes() != rebuilt.read_bytes():
            raise AssertionError("deterministic public ZIP rebuild differs")
        if rebuilt_sha256 != archive_sha256:
            raise AssertionError("deterministic public ZIP hash differs")
    with tempfile.TemporaryDirectory(
        prefix="unitdp_v35_zip_extract_"
    ) as temporary:
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(temporary)
        extracted = Path(temporary)
        extracted_files = sorted(
            path.relative_to(extracted).as_posix()
            for path in extracted.rglob("*")
            if path.is_file()
        )
        source_files = sorted(
            path.relative_to(public_root).as_posix()
            for path in public_root.rglob("*")
            if path.is_file()
        )
        if extracted_files != source_files:
            raise AssertionError("extracted public file list differs")
        for relative in source_files:
            if (
                file_sha256(extracted / relative)
                != file_sha256(public_root / relative)
            ):
                raise AssertionError(
                    f"extracted public file differs: {relative}"
                )

    source_hashes = {
        path.relative_to(ROOT).as_posix(): file_sha256(path)
        for path in (
            ROOT / "src" / "unitdp"
            / "benchmark_registry_allocation_v4.py",
            ROOT / "src" / "unitdp"
            / "compiler_allocation_v4.py",
            ROOT / "src" / "unitdp"
            / "owner_random_allocation_v4.py",
            ROOT / "src" / "unitdp"
            / "execution_artifacts_allocation_v4.py",
            ROOT / "tests"
            / "owner_random_allocation_trace_reference.py",
            Path(__file__).resolve(),
        )
    }
    gate: dict[str, Any] = {
        "schema_version": (
            "unitdp.v35_allocation_multirun_gate.v1"
        ),
        "decision": "PASS",
        "scope": {
            "datasets": ["uci", "wisdm", "sepsis"],
            "runs_per_dataset": len(seeds),
            "seed_values_public": False,
            "release_status": "research_non_release",
        },
        "results": {
            **totals,
            "all_reference_comparisons_exact": True,
            "public_zip_deterministic_rebuild": True,
            "public_zip_extraction_match": True,
        },
        "public_collection_sha256": collection[
            "public_collection_sha256"
        ],
        "public_archive": {
            "filename": archive.name,
            "file_count": len(
                [
                    path
                    for path in public_root.rglob("*")
                    if path.is_file()
                ]
            ),
            "sha256": archive_sha256,
        },
        "source_hashes": source_hashes,
        "limitations": [
            "finite deterministic fixtures, not arbitrary-input refinement",
            "reference agreement is implementation evidence, not privacy proof",
            "PCG64 is a research backend, not release-grade randomness",
            "metrics establish execution only, not route superiority",
        ],
    }
    gate["report_payload_sha256"] = payload_sha256(gate)
    write_json(output / "allocation_multirun_gate_v35.json", gate)
    print(json.dumps(gate["results"], sort_keys=True), flush=True)
    print(
        f"report_payload_sha256={gate['report_payload_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
