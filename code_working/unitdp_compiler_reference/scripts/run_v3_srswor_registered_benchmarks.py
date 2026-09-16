"""Run the three exact registered owner-SRSWOR V3 research benchmarks."""

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
from unitdp.benchmark_registry_srswor_v3 import (  # noqa: E402
    get_srswor_benchmark_profile_v3,
)
from unitdp.compiler_srswor_v3 import (  # noqa: E402
    compile_owner_srswor_contract_v3,
    load_owner_srswor_contract_v3,
)
from unitdp.execution_artifacts_srswor_v3 import (  # noqa: E402
    write_srswor_execution_artifacts_v3,
)
from unitdp.owner_srswor_v3 import (  # noqa: E402
    train_owner_srswor_fixed_v3,
)
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
        "profile_id": "uci_har_srswor_aaai27_v3",
        "contract": ROOT / "configs" / "v3" / "uci_owner_srswor_v3.yaml",
        "preprocessor": (
            ROOT
            / "configs"
            / "preprocessing"
            / "uci_har_published_train_standard_scaler_v1.json"
        ),
    },
    "wisdm": {
        "profile_id": "wisdm_srswor_aaai27_v3",
        "contract": ROOT / "configs" / "v3" / "wisdm_owner_srswor_v3.yaml",
        "preprocessor": (
            ROOT
            / "configs"
            / "preprocessing"
            / "wisdm_v1_1_train_owners_1_25_stats24_standard_scaler_v1.json"
        ),
    },
    "sepsis": {
        "profile_id": "sepsis_srswor_aaai27_v3",
        "contract": ROOT / "configs" / "v3" / "sepsis_owner_srswor_v3.yaml",
        "preprocessor": (
            ROOT
            / "configs"
            / "preprocessing"
            / "physionet2019_setA_first400_basic12x6_public_scaler_v1.json"
        ),
    },
}


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
            raise ValueError(
                "Every research seed must lie in [0, 2**63)"
            )
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
    artifact = DATASET_CONFIG[dataset]["preprocessor"]
    if dataset == "uci":
        return prepare_uci_har_v2(
            dataset_root=args.uci_root,
            preprocessing_artifact_path=artifact,
        )
    if dataset == "wisdm":
        return prepare_wisdm_v2(
            raw_path=args.wisdm_raw,
            preprocessing_artifact_path=artifact,
        )
    if dataset == "sepsis":
        return prepare_sepsis_v2(
            cache_dir=args.sepsis_cache,
            preprocessing_artifact_path=artifact,
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
    profile = get_srswor_benchmark_profile_v3(
        str(config["profile_id"])
    )
    prepared = prepare_dataset(dataset, args)
    prepared.assert_registered_full_conformance()
    dataset_dir = Path(args.output_dir) / dataset
    if dataset_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to reuse existing dataset directory: {dataset_dir}"
        )
    dataset_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f"unitdp_{dataset}_srswor_v3_"
    ) as tmp:
        mapping_path = write_prepared_mapping_v2(
            prepared,
            Path(tmp) / "train_mapping.csv",
        )
        contract = load_owner_srswor_contract_v3(config["contract"])
        route = compile_owner_srswor_contract_v3(
            contract,
            mapping_path=mapping_path,
            preprocessing_artifact_path=config["preprocessor"],
            require_executable=True,
        )
    if route.profile != profile:
        raise RuntimeError("Compiled route selected the wrong profile")

    public_runs: list[dict[str, Any]] = []
    private_index: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, start=1):
        run_id = f"run_{index:03d}"
        prepared.assert_registered_full_conformance()
        result = train_owner_srswor_fixed_v3(
            route=route,
            raw_train_x=prepared.raw_train_x,
            train_y=prepared.train_y,
            raw_test_x=prepared.raw_test_x,
            test_y=prepared.test_y,
            research_seed=seed,
        )
        public = result.public_payload()
        written = write_srswor_execution_artifacts_v3(
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
                / "private_execution_srswor_v3.json"
            )
            write_json(
                private_path,
                result.private_manifest(),
                overwrite=args.overwrite,
            )
            private_index.append(
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
            f"macro_f1={evaluation['macro_f1']:.6f}",
            flush=True,
        )

    summary: dict[str, Any] = {
        "schema_version": "unitdp.srswor_benchmark_summary_public.v3",
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
            "sample_size": contract.sample_size,
            "sample_rate": contract.sample_rate,
            "total_steps": contract.total_steps,
            "actual_noise_multiplier": contract.noise_multiplier,
            "sensitivity_multiplier": contract.sensitivity_multiplier,
            "update_denominator": contract.update_denominator,
        },
        "accountant": {
            "epsilon_dp_accounting": (
                route.accountant_epsilon_dp_accounting
            ),
            "epsilon_direct_theorem_oracle": (
                route.accountant_epsilon_direct_oracle
            ),
            "absolute_gap": abs(
                route.accountant_epsilon_dp_accounting
                - route.accountant_epsilon_direct_oracle
            ),
            "optimal_order_direct_theorem_oracle": (
                route.accountant_optimal_order_direct_oracle
            ),
            "delta": contract.delta,
        },
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
        dataset_dir / "public_summary_srswor_v3.json",
        summary,
        overwrite=args.overwrite,
    )
    if args.write_private_manifests:
        private_payload: dict[str, Any] = {
            "schema_version": (
                "unitdp.srswor_benchmark_run_index_private.v3"
            ),
            "visibility": "private",
            "handling": "do_not_publish_contains_research_seeds",
            "runs": private_index,
        }
        private_payload["private_index_sha256"] = payload_sha256(
            private_payload
        )
        write_json(
            dataset_dir
            / "_private"
            / "private_run_index_srswor_v3.json",
            private_payload,
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
        help="Private comma-separated seeds; never copied to public output.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            ROOT / "reports" / "v3_srswor_registered_5seed_v32_20260724"
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
        value.strip()
        for value in args.datasets.split(",")
        if value.strip()
    ]
    if not datasets or any(
        value not in DATASET_CONFIG for value in datasets
    ):
        raise ValueError("datasets must be drawn from uci,wisdm,sepsis")
    seeds = parse_seeds(args.research_seeds)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries = [
        run_dataset(dataset, seeds=seeds, args=args)
        for dataset in datasets
    ]
    collection: dict[str, Any] = {
        "schema_version": (
            "unitdp.srswor_benchmark_collection_public.v3"
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
        output / "public_collection_srswor_v3.json",
        collection,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
