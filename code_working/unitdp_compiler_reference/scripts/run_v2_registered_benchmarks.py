"""Run the three exact registered AAAI-27 V2 research benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get("UNITDP_DATA_ROOT", str(ROOT / "data"))
)
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unitdp.benchmark_data_v2 import (  # noqa: E402
    PreparedBenchmarkV2,
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
    write_execution_artifacts_v2,
    write_private_execution_manifest_v2,
)
from unitdp.owa_dpsgd import train_owner_poisson_fixed_v2  # noqa: E402
from unitdp.release_artifacts import (  # noqa: E402
    assert_public_certificate_redacted,
)


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
        "preparer_source": ROOT / "scripts" / "run_uci_owa.py",
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
        "preparer_source": ROOT / "scripts" / "run_wisdm_owa.py",
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
        "preparer_source": ROOT / "scripts" / "run_sepsis_owa.py",
    },
}


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_seeds(text: str) -> list[int]:
    values: list[int] = []
    for token in text.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        value = int(stripped)
        if value < 0 or value >= 2**63:
            raise ValueError("Every research seed must lie in [0, 2**63)")
        values.append(value)
    if not values or len(values) != len(set(values)):
        raise ValueError("Provide one or more distinct research seeds")
    return values


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


def prepare_dataset(
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


def run_dataset(
    dataset: str,
    *,
    seeds: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = DATASET_CONFIG[dataset]
    profile = get_benchmark_profile_v2(str(config["profile_id"]))
    prepared = prepare_dataset(dataset, args)
    prepared.assert_registered_full_conformance()
    dataset_dir = Path(args.output_dir) / dataset
    if dataset_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to reuse existing dataset directory: {dataset_dir}"
        )
    dataset_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"unitdp_{dataset}_v2_") as tmp:
        mapping_path = write_prepared_mapping_v2(
            prepared,
            Path(tmp) / "train_mapping.csv",
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
    ):
        raise RuntimeError(
            "Compiled full mapping differs from the benchmark profile"
        )
    if (
        contract.public_protocol["full_data_conformance_sha256"]
        != prepared.full_data_conformance_sha256()
    ):
        raise RuntimeError(
            "Compiled contract and prepared full-data digest disagree"
        )

    data_preparer_source_bundle = {
        "src/unitdp/benchmark_data_v2.py": file_sha256(
            ROOT / "src" / "unitdp" / "benchmark_data_v2.py"
        ),
        str(Path(config["preparer_source"]).relative_to(ROOT)).replace(
            "\\", "/"
        ): file_sha256(Path(config["preparer_source"])),
    }
    public_runs: list[dict[str, Any]] = []
    private_run_index: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, start=1):
        run_id = f"run_{index:03d}"
        prepared.assert_registered_full_conformance()
        result = train_owner_poisson_fixed_v2(
            route=route,
            raw_train_x=prepared.raw_train_x,
            train_y=prepared.train_y,
            raw_test_x=prepared.raw_test_x,
            test_y=prepared.test_y,
            research_seed=seed,
        )
        public = result.public_payload()
        written = write_execution_artifacts_v2(
            result,
            dataset_dir / run_id,
            overwrite=args.overwrite,
        )
        public_runs.append(
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
        if args.write_private_manifests:
            private_path = (
                dataset_dir
                / "_private"
                / run_id
                / "private_execution_manifest_v2.json"
            )
            write_private_execution_manifest_v2(
                result,
                private_path,
                overwrite=args.overwrite,
            )
            private_run_index.append(
                {
                    "run_id": run_id,
                    "research_seed": seed,
                    "private_manifest": str(
                        private_path.relative_to(dataset_dir)
                    ).replace("\\", "/"),
                }
            )
        evaluation = public["public_evaluation"]
        print(
            f"{dataset} {run_id}: "
            f"accuracy={evaluation['accuracy']:.6f}, "
            f"macro_f1={evaluation['macro_f1']:.6f}"
        )

    summary: dict[str, Any] = {
        "schema_version": "unitdp.benchmark_summary_public.v2.1",
        "visibility": "public",
        "release_status": "research_non_release",
        "dataset": dataset,
        "benchmark_profile_id": profile.profile_id,
        "benchmark_profile_sha256": profile.registry_sha256,
        "full_data_conformance_sha256": (
            profile.full_data_conformance_sha256
        ),
        "data_preparation_implementation_id": (
            profile.data_preparation_implementation_id
        ),
        "data_preparer_source_bundle": data_preparer_source_bundle,
        "data_preparer_source_bundle_sha256": payload_sha256(
            data_preparer_source_bundle
        ),
        "public_contract_sha256": contract.public_contract_sha256,
        "public_plan_sha256": route.public_plan_sha256,
        "execution_source_bundle_sha256": (
            route.execution_source_bundle_sha256
        ),
        "run_count": len(public_runs),
        "runs": public_runs,
        "aggregate": {
            key: metric_summary(public_runs, key)
            for key in ("accuracy", "macro_f1", "auroc", "auprc")
        },
    }
    summary["public_summary_sha256"] = payload_sha256(summary)
    assert_public_certificate_redacted(summary)
    write_json(
        dataset_dir / "public_summary_v2.json",
        summary,
        overwrite=args.overwrite,
    )
    if args.write_private_manifests:
        private_index: dict[str, Any] = {
            "schema_version": "unitdp.benchmark_run_index_private.v2.1",
            "visibility": "private",
            "handling": "do_not_publish_contains_research_seeds",
            "runs": private_run_index,
        }
        private_index["private_index_sha256"] = payload_sha256(
            private_index
        )
        write_json(
            dataset_dir / "_private" / "private_run_index_v2.json",
            private_index,
            overwrite=args.overwrite,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="uci,wisdm,sepsis",
        help="Comma-separated subset of uci,wisdm,sepsis.",
    )
    parser.add_argument(
        "--research-seeds",
        required=True,
        help="Private comma-separated research seeds; never copied to public output.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            ROOT / "reports" / "v2_registered_5seed_v32_20260724"
        ),
    )
    parser.add_argument("--uci-root", default=str(DEFAULT_UCI_ROOT))
    parser.add_argument("--wisdm-raw", default=str(DEFAULT_WISDM_RAW))
    parser.add_argument(
        "--sepsis-cache",
        default=str(DEFAULT_SEPSIS_CACHE),
    )
    parser.add_argument(
        "--write-private-manifests",
        action="store_true",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    datasets = [
        token.strip().lower()
        for token in args.datasets.split(",")
        if token.strip()
    ]
    if not datasets or len(datasets) != len(set(datasets)):
        raise ValueError("Provide one or more distinct dataset ids")
    unknown = sorted(set(datasets).difference(DATASET_CONFIG))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    seeds = parse_seeds(args.research_seeds)

    summaries = [
        run_dataset(
            dataset,
            seeds=seeds,
            args=args,
        )
        for dataset in datasets
    ]
    execution_source_bundle_hashes = {
        summary["execution_source_bundle_sha256"]
        for summary in summaries
    }
    if len(execution_source_bundle_hashes) != 1:
        raise RuntimeError(
            "Dataset runs used different execution source bundles"
        )
    top_level: dict[str, Any] = {
        "schema_version": "unitdp.benchmark_collection_public.v2.1",
        "visibility": "public",
        "release_status": "research_non_release",
        "execution_source_bundle_sha256": next(
            iter(execution_source_bundle_hashes)
        ),
        "dataset_summaries": [
            {
                "dataset": summary["dataset"],
                "public_summary_sha256": summary[
                    "public_summary_sha256"
                ],
            }
            for summary in summaries
        ],
    }
    top_level["public_collection_sha256"] = payload_sha256(top_level)
    assert_public_certificate_redacted(top_level)
    write_json(
        Path(args.output_dir) / "public_collection_v2.json",
        top_level,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
