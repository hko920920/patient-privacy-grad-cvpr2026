"""Run an exact-data V2 non-private linear utility reference.

This is not a privacy mechanism or a claimed upper bound.  It uses the exact
registered V2 public benchmark preparation and compiled selected mapping, then
trains a conventional non-private linear model for utility context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get("UNITDP_DATA_ROOT", str(ROOT / "data"))
)
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unitdp.benchmark_data_v2 import (  # noqa: E402
    PreparedBenchmarkV2,
    array_bytes_sha256,
    prepare_sepsis_v2,
    prepare_uci_har_v2,
    prepare_wisdm_v2,
    write_prepared_mapping_v2,
)
from unitdp.benchmark_registry_v2 import (  # noqa: E402
    get_benchmark_profile_v2,
)
from unitdp.compiler_v2 import (  # noqa: E402
    compile_owner_poisson_contract_v2,
    load_owner_poisson_contract_v2,
)
from unitdp.execution_artifacts_v2 import (  # noqa: E402
    build_model_artifact_v2,
)
from unitdp.nonprivate import (  # noqa: E402
    NonPrivateConfig,
    train_nonprivate,
)


REPORT_SCHEMA = "unitdp.v2_nonprivate_reference_collection.v1"
PRIVATE_INDEX_SCHEMA = "unitdp.v2_nonprivate_reference_seed_index.v1"
DEFAULT_UCI_ROOT = (
    DATA_ROOT / "uci_har" / "extracted" / "UCI HAR Dataset"
)
DEFAULT_WISDM_RAW = (
    DATA_ROOT / "wisdm" / "WISDM_ar_v1.1" / "WISDM_ar_v1.1_raw.txt"
)
DEFAULT_SEPSIS_CACHE = DATA_ROOT / "sepsis2019_cache"
DATASET_CONFIG = {
    "uci": {
        "profile_id": "uci_har_aaai27_v1",
        "contract": ROOT / "configs" / "v2" / "uci_owner_poisson_v2.yaml",
        "preprocessor": (
            ROOT
            / "configs"
            / "preprocessing"
            / "uci_har_published_train_standard_scaler_v1.json"
        ),
    },
    "wisdm": {
        "profile_id": "wisdm_aaai27_v1",
        "contract": ROOT / "configs" / "v2" / "wisdm_owner_poisson_v2.yaml",
        "preprocessor": (
            ROOT
            / "configs"
            / "preprocessing"
            / "wisdm_v1_1_train_owners_1_25_stats24_standard_scaler_v1.json"
        ),
    },
    "sepsis": {
        "profile_id": "sepsis_aaai27_v1",
        "contract": ROOT / "configs" / "v2" / "sepsis_owner_poisson_v2.yaml",
        "preprocessor": (
            ROOT
            / "configs"
            / "preprocessing"
            / "physionet2019_setA_first400_basic12x6_public_scaler_v1.json"
        ),
    },
}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _write_json(
    path: Path,
    value: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}")
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


def _parse_seeds(value: str) -> list[int]:
    seeds = [
        int(token.strip())
        for token in value.split(",")
        if token.strip()
    ]
    if (
        not seeds
        or len(seeds) != len(set(seeds))
        or any(seed < 0 or seed >= 2**63 for seed in seeds)
    ):
        raise ValueError(
            "Provide distinct research seeds in [0, 2**63)"
        )
    return seeds


def _prepare_dataset(
    dataset: str,
    args: argparse.Namespace,
) -> PreparedBenchmarkV2:
    preprocessor = DATASET_CONFIG[dataset]["preprocessor"]
    if dataset == "uci":
        return prepare_uci_har_v2(
            dataset_root=args.uci_root,
            preprocessing_artifact_path=preprocessor,
        )
    if dataset == "wisdm":
        return prepare_wisdm_v2(
            raw_path=args.wisdm_raw,
            preprocessing_artifact_path=preprocessor,
        )
    if dataset == "sepsis":
        return prepare_sepsis_v2(
            cache_dir=args.sepsis_cache,
            preprocessing_artifact_path=preprocessor,
        )
    raise KeyError(dataset)


def _metric_aggregate(
    runs: list[dict[str, Any]],
    metric: str,
) -> dict[str, float | int | None]:
    values = [
        float(run["evaluation"][metric])
        for run in runs
        if run["evaluation"][metric] is not None
    ]
    if not values:
        return {"count": 0, "mean": None, "sample_std": None}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "sample_std": (
            statistics.stdev(values) if len(values) > 1 else 0.0
        ),
    }


def _run_dataset(
    dataset: str,
    *,
    seeds: list[int],
    output_root: Path,
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
]:
    config = DATASET_CONFIG[dataset]
    profile = get_benchmark_profile_v2(str(config["profile_id"]))
    prepared = _prepare_dataset(dataset, args)
    prepared.assert_registered_full_conformance()

    with tempfile.TemporaryDirectory(
        prefix=f"unitdp_nonprivate_{dataset}_"
    ) as temporary:
        mapping_path = write_prepared_mapping_v2(
            prepared,
            Path(temporary) / "train_mapping.csv",
        )
        contract = load_owner_poisson_contract_v2(config["contract"])
        route = compile_owner_poisson_contract_v2(
            contract,
            mapping_path=mapping_path,
            preprocessing_artifact_path=config["preprocessor"],
            require_executable=True,
        )

    if (
        route.private_source_mapping_canonical_sha256
        != profile.train_mapping_canonical_sha256
        or prepared.full_data_conformance_sha256()
        != profile.full_data_conformance_sha256
    ):
        raise RuntimeError(
            f"{dataset} exact V2 data/route binding mismatch"
        )

    transformed_train = prepared.preprocessor.transform(
        prepared.raw_train_x
    )
    transformed_test = prepared.preprocessor.transform(
        prepared.raw_test_x
    )
    selected_indices = [
        record.row_index for record in route.selected_mapping.records
    ]
    selected_x = transformed_train[selected_indices]
    selected_y = prepared.train_y[selected_indices]
    class_weights = (
        list(contract.class_weights)
        if contract.class_weights is not None
        else None
    )

    public_runs: list[dict[str, Any]] = []
    private_runs: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, start=1):
        run_id = f"run_{index:03d}"
        torch.manual_seed(seed)
        model = nn.Linear(
            contract.input_dim,
            contract.num_classes,
            bias=True,
            device="cpu",
            dtype=torch.float32,
        )
        result = train_nonprivate(
            model=model,
            train_x=selected_x,
            train_y=selected_y,
            test_x=transformed_test,
            test_y=prepared.test_y,
            config=NonPrivateConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=contract.learning_rate,
                seed=seed,
                class_weights=class_weights,
            ),
        )
        model_artifact = build_model_artifact_v2(model)
        relative_model_path = (
            Path(dataset) / run_id / "model_v2.json"
        )
        model_path = output_root / relative_model_path
        _write_json(
            model_path,
            model_artifact,
            overwrite=args.overwrite,
        )
        evaluation = {
            "accuracy": result.test_accuracy,
            "macro_f1": result.test_macro_f1,
            "auroc": result.test_auroc,
            "auprc": result.test_auprc,
            "train_accuracy": result.train_accuracy,
        }
        public_runs.append(
            {
                "run_id": run_id,
                "model_artifact": (
                    str(relative_model_path).replace("\\", "/")
                ),
                "model_file_sha256": _file_sha256(model_path),
                "model_payload_sha256": model_artifact[
                    "payload_sha256"
                ],
                "model_state_sha256": model_artifact[
                    "state_sha256"
                ],
                "evaluation": evaluation,
            }
        )
        private_runs.append(
            {
                "dataset": dataset,
                "run_id": run_id,
                "research_seed": seed,
            }
        )
        print(
            f"{dataset} {run_id}: "
            f"macro_f1={result.test_macro_f1:.6f}, "
            f"auroc={result.test_auroc:.6f}"
        )

    dataset_report: dict[str, Any] = {
        "dataset": dataset,
        "data_binding": {
            "benchmark_profile_id": profile.profile_id,
            "benchmark_profile_sha256": profile.registry_sha256,
            "full_data_conformance_sha256": (
                profile.full_data_conformance_sha256
            ),
            "preprocessing_artifact_sha256": (
                prepared.preprocessor.payload_sha256
            ),
            "public_contract_sha256": (
                contract.public_contract_sha256
            ),
            "public_plan_sha256": route.public_plan_sha256,
            "execution_source_bundle_sha256": (
                route.execution_source_bundle_sha256
            ),
        },
        "selection_protocol": {
            "scope": "public_benchmark_evaluation_only",
            "policy": "compiled_v2_selected_mapping",
            "data_dependent_details_disclosed": False,
        },
        "training_configuration": {
            "model_id": contract.model_id,
            "optimizer": "sgd",
            "loss": "cross_entropy",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": contract.learning_rate,
            "class_weights": class_weights,
            "shuffle": True,
        },
        "run_count": len(public_runs),
        "runs": public_runs,
        "aggregate": {
            metric: _metric_aggregate(public_runs, metric)
            for metric in (
                "accuracy",
                "macro_f1",
                "auroc",
                "auprc",
                "train_accuracy",
            )
        },
    }
    private_dataset_binding = {
        "dataset": dataset,
        "selected_record_count": len(selected_indices),
        "selected_owner_count": len(
            route.selected_mapping.by_owner()
        ),
        "selected_mapping_canonical_sha256": (
            route.selected_mapping.canonical_hash()
        ),
        "selected_features_sha256": (
            array_bytes_sha256(selected_x)
        ),
        "selected_labels_sha256": (
            array_bytes_sha256(selected_y)
        ),
    }
    return dataset_report, private_runs, private_dataset_binding


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="uci,wisdm,sepsis",
    )
    parser.add_argument(
        "--research-seeds",
        required=True,
        help="Private comma-separated seeds; omitted from public report.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            ROOT / "reports" / "v2_nonprivate_reference_v32_20260724"
        ),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--uci-root", default=str(DEFAULT_UCI_ROOT))
    parser.add_argument("--wisdm-raw", default=str(DEFAULT_WISDM_RAW))
    parser.add_argument(
        "--sepsis-cache",
        default=str(DEFAULT_SEPSIS_CACHE),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.epochs != 20 or args.batch_size != 64:
        raise ValueError(
            "The registered non-private reference requires "
            "epochs=20 and batch-size=64"
        )

    datasets = [
        token.strip().lower()
        for token in args.datasets.split(",")
        if token.strip()
    ]
    if (
        not datasets
        or len(datasets) != len(set(datasets))
        or set(datasets).difference(DATASET_CONFIG)
    ):
        raise ValueError(
            "Provide distinct dataset ids from uci,wisdm,sepsis"
        )
    seeds = _parse_seeds(args.research_seeds)
    output_root = Path(args.output_dir).resolve()
    report_path = output_root / "public_nonprivate_reference_v1.json"
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to reuse {output_root}")

    reference_source_bundle = {
        "scripts/run_v2_nonprivate_reference.py": _file_sha256(
            Path(__file__).resolve()
        ),
        "src/unitdp/nonprivate.py": _file_sha256(
            ROOT / "src" / "unitdp" / "nonprivate.py"
        ),
    }
    dataset_reports: list[dict[str, Any]] = []
    private_runs: list[dict[str, Any]] = []
    private_dataset_bindings: list[dict[str, Any]] = []
    for dataset in datasets:
        (
            dataset_report,
            dataset_private_runs,
            private_dataset_binding,
        ) = _run_dataset(
            dataset,
            seeds=seeds,
            output_root=output_root,
            args=args,
        )
        dataset_reports.append(dataset_report)
        private_runs.extend(dataset_private_runs)
        private_dataset_bindings.append(private_dataset_binding)

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "visibility": "public",
        "claim_status": (
            "nonprivate_reference_no_privacy_or_upper_bound_claim"
        ),
        "research_seeds_disclosed": False,
        "reference_source_bundle": reference_source_bundle,
        "reference_source_bundle_sha256": _canonical_sha256(
            reference_source_bundle
        ),
        "dataset_count": len(dataset_reports),
        "run_count": sum(
            int(dataset["run_count"]) for dataset in dataset_reports
        ),
        "datasets": dataset_reports,
    }
    report["public_report_sha256"] = _canonical_sha256(report)
    _write_json(report_path, report, overwrite=args.overwrite)

    private_index: dict[str, Any] = {
        "schema_version": PRIVATE_INDEX_SCHEMA,
        "visibility": "private",
        "handling": "do_not_publish_contains_research_seeds",
        "public_report_sha256": report["public_report_sha256"],
        "dataset_bindings": private_dataset_bindings,
        "runs": private_runs,
    }
    private_index["private_index_sha256"] = _canonical_sha256(
        private_index
    )
    _write_json(
        output_root / "_private" / "private_seed_index_v1.json",
        private_index,
        overwrite=args.overwrite,
    )
    print(
        f"Wrote {report_path}; "
        f"public_report_sha256={report['public_report_sha256']}"
    )


if __name__ == "__main__":
    main()
