"""Strictly verify the exact-data V2 non-private utility reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)


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
    load_model_artifact_v2,
)
from unitdp.source_bundle_v2 import (  # noqa: E402
    execution_source_bundle_sha256_v2,
)


REPORT_SCHEMA = "unitdp.v2_nonprivate_reference_collection.v1"
PRIVATE_INDEX_SCHEMA = "unitdp.v2_nonprivate_reference_seed_index.v1"
VERIFICATION_SCHEMA = (
    "unitdp.v2_nonprivate_reference_verification.v1"
)
DEFAULT_UCI_ROOT = (
    DATA_ROOT / "uci_har" / "extracted" / "UCI HAR Dataset"
)
DEFAULT_WISDM_RAW = (
    DATA_ROOT / "wisdm" / "WISDM_ar_v1.1" / "WISDM_ar_v1.1_raw.txt"
)
DEFAULT_SEPSIS_CACHE = DATA_ROOT / "sepsis2019_cache"
DATASET_ORDER = ("uci", "wisdm", "sepsis")
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
REPORT_KEYS = {
    "schema_version",
    "visibility",
    "claim_status",
    "research_seeds_disclosed",
    "reference_source_bundle",
    "reference_source_bundle_sha256",
    "dataset_count",
    "run_count",
    "datasets",
    "public_report_sha256",
}
DATASET_KEYS = {
    "dataset",
    "data_binding",
    "selection_protocol",
    "training_configuration",
    "run_count",
    "runs",
    "aggregate",
}
DATA_BINDING_KEYS = {
    "benchmark_profile_id",
    "benchmark_profile_sha256",
    "full_data_conformance_sha256",
    "preprocessing_artifact_sha256",
    "public_contract_sha256",
    "public_plan_sha256",
    "execution_source_bundle_sha256",
}
SELECTION_PROTOCOL_KEYS = {
    "scope",
    "policy",
    "data_dependent_details_disclosed",
}
PRIVATE_DATASET_BINDING_KEYS = {
    "dataset",
    "selected_record_count",
    "selected_owner_count",
    "selected_mapping_canonical_sha256",
    "selected_features_sha256",
    "selected_labels_sha256",
}
TRAINING_KEYS = {
    "model_id",
    "optimizer",
    "loss",
    "epochs",
    "batch_size",
    "learning_rate",
    "class_weights",
    "shuffle",
}
RUN_KEYS = {
    "run_id",
    "model_artifact",
    "model_file_sha256",
    "model_payload_sha256",
    "model_state_sha256",
    "evaluation",
}
EVALUATION_KEYS = {
    "accuracy",
    "macro_f1",
    "auroc",
    "auprc",
    "train_accuracy",
}
AGGREGATE_KEYS = EVALUATION_KEYS
METRIC_SUMMARY_KEYS = {"count", "mean", "sample_std"}
PRIVATE_INDEX_KEYS = {
    "schema_version",
    "visibility",
    "handling",
    "public_report_sha256",
    "dataset_bindings",
    "runs",
    "private_index_sha256",
}
PRIVATE_RUN_KEYS = {"dataset", "run_id", "research_seed"}


class NonPrivateReferenceVerificationError(RuntimeError):
    """Raised when the V2 non-private reference fails closed."""


def _unique_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise NonPrivateReferenceVerificationError(
                f"Duplicate JSON key: {key!r}"
            )
        result[key] = value
    return result


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    except NonPrivateReferenceVerificationError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise NonPrivateReferenceVerificationError(
            f"Could not load {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise NonPrivateReferenceVerificationError(
            f"{path} must contain a JSON object"
        )
    return value


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


def _require_exact_keys(
    value: dict[str, Any],
    expected: set[str],
    context: str,
) -> None:
    if set(value) != expected:
        missing = sorted(expected.difference(value))
        extra = sorted(set(value).difference(expected))
        raise NonPrivateReferenceVerificationError(
            f"{context} keys mismatch; missing={missing}, extra={extra}"
        )


def _require_sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise NonPrivateReferenceVerificationError(
            f"{context} is not a lowercase SHA-256"
        )
    return value


def _require_self_hash(
    value: dict[str, Any],
    field: str,
    context: str,
) -> str:
    reported = _require_sha256(value.get(field), f"{context}.{field}")
    payload = dict(value)
    payload.pop(field)
    if _canonical_sha256(payload) != reported:
        raise NonPrivateReferenceVerificationError(
            f"{context}.{field} mismatch"
        )
    return reported


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_metric(value: object, context: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise NonPrivateReferenceVerificationError(
            f"{context} must be null or finite in [0,1]"
        )
    return float(value)


def _same_number(
    observed: float | None,
    expected: float | None,
) -> bool:
    if observed is None or expected is None:
        return observed is expected
    return abs(observed - expected) <= 1e-15


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


def _rank_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float | None, float | None]:
    if len(np.unique(labels)) < 2:
        return None, None
    try:
        if probabilities.shape[1] == 2:
            auroc = roc_auc_score(labels, probabilities[:, 1])
            auprc = average_precision_score(
                labels,
                probabilities[:, 1],
            )
        else:
            classes = np.arange(probabilities.shape[1])
            one_hot = np.eye(
                probabilities.shape[1],
                dtype=int,
            )[labels]
            auroc = roc_auc_score(
                labels,
                probabilities,
                labels=classes,
                multi_class="ovr",
                average="macro",
            )
            auprc = average_precision_score(
                one_hot,
                probabilities,
                average="macro",
            )
    except ValueError:
        return None, None
    return float(auroc), float(auprc)


def _evaluate(
    model: torch.nn.Module,
    *,
    selected_x: np.ndarray,
    selected_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> dict[str, float | None]:
    model.eval()
    with torch.no_grad():
        train_logits = model(torch.from_numpy(selected_x))
        test_logits = model(torch.from_numpy(test_x))
        train_predictions = (
            train_logits.argmax(dim=1).cpu().numpy()
        )
        test_predictions = (
            test_logits.argmax(dim=1).cpu().numpy()
        )
        probabilities = (
            torch.softmax(test_logits, dim=1).cpu().numpy()
        )
    auroc, auprc = _rank_metrics(test_y, probabilities)
    return {
        "accuracy": float(
            accuracy_score(test_y, test_predictions)
        ),
        "macro_f1": float(
            f1_score(
                test_y,
                test_predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "auroc": auroc,
        "auprc": auprc,
        "train_accuracy": float(
            accuracy_score(selected_y, train_predictions)
        ),
    }


def _safe_model_path(
    collection_dir: Path,
    relative: object,
    *,
    dataset: str,
    run_id: str,
) -> Path:
    expected = Path(dataset) / run_id / "model_v2.json"
    if not isinstance(relative, str) or Path(relative) != expected:
        raise NonPrivateReferenceVerificationError(
            f"{dataset}.{run_id} model path is non-canonical"
        )
    candidate = (collection_dir / expected).resolve()
    try:
        candidate.relative_to(collection_dir.resolve())
    except ValueError as exc:
        raise NonPrivateReferenceVerificationError(
            f"{dataset}.{run_id} model path escapes collection"
        ) from exc
    if not candidate.is_file():
        raise NonPrivateReferenceVerificationError(
            f"{dataset}.{run_id} model artifact is missing"
        )
    return candidate


def _verify_aggregate(
    aggregate: object,
    runs: list[dict[str, Any]],
    context: str,
) -> None:
    if not isinstance(aggregate, dict):
        raise NonPrivateReferenceVerificationError(
            f"{context} must be an object"
        )
    _require_exact_keys(aggregate, AGGREGATE_KEYS, context)
    for metric in sorted(AGGREGATE_KEYS):
        summary = aggregate[metric]
        if not isinstance(summary, dict):
            raise NonPrivateReferenceVerificationError(
                f"{context}.{metric} must be an object"
            )
        _require_exact_keys(
            summary,
            METRIC_SUMMARY_KEYS,
            f"{context}.{metric}",
        )
        values = [
            float(run["evaluation"][metric])
            for run in runs
            if run["evaluation"][metric] is not None
        ]
        expected = {
            "count": len(values),
            "mean": (
                statistics.mean(values) if values else None
            ),
            "sample_std": (
                statistics.stdev(values)
                if len(values) > 1
                else (0.0 if values else None)
            ),
        }
        if summary["count"] != expected["count"]:
            raise NonPrivateReferenceVerificationError(
                f"{context}.{metric}.count mismatch"
            )
        for field in ("mean", "sample_std"):
            observed_value = (
                None
                if summary[field] is None
                else float(summary[field])
            )
            expected_value = expected[field]
            if not _same_number(observed_value, expected_value):
                raise NonPrivateReferenceVerificationError(
                    f"{context}.{metric}.{field} mismatch"
                )


def verify(
    report_path: Path,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    collection_dir = report_path.resolve().parent
    report = _load_object(report_path)
    _require_exact_keys(report, REPORT_KEYS, "report")
    if (
        report["schema_version"] != REPORT_SCHEMA
        or report["visibility"] != "public"
        or report["claim_status"]
        != "nonprivate_reference_no_privacy_or_upper_bound_claim"
        or report["research_seeds_disclosed"] is not False
    ):
        raise NonPrivateReferenceVerificationError(
            "Report metadata mismatch"
        )
    public_report_sha256 = _require_self_hash(
        report,
        "public_report_sha256",
        "report",
    )

    expected_reference_sources = {
        "scripts/run_v2_nonprivate_reference.py": (
            ROOT / "scripts" / "run_v2_nonprivate_reference.py"
        ),
        "src/unitdp/nonprivate.py": (
            ROOT / "src" / "unitdp" / "nonprivate.py"
        ),
    }
    source_bundle = report["reference_source_bundle"]
    if not isinstance(source_bundle, dict):
        raise NonPrivateReferenceVerificationError(
            "reference_source_bundle must be an object"
        )
    if set(source_bundle) != set(expected_reference_sources):
        raise NonPrivateReferenceVerificationError(
            "Reference source file set mismatch"
        )
    for relative, source in expected_reference_sources.items():
        if source_bundle[relative] != _file_sha256(source):
            raise NonPrivateReferenceVerificationError(
                f"Current source mismatch: {relative}"
            )
    if (
        _canonical_sha256(source_bundle)
        != report["reference_source_bundle_sha256"]
    ):
        raise NonPrivateReferenceVerificationError(
            "Reference source bundle digest mismatch"
        )

    private_index_path = (
        collection_dir / "_private" / "private_seed_index_v1.json"
    )
    private_index = _load_object(private_index_path)
    _require_exact_keys(
        private_index,
        PRIVATE_INDEX_KEYS,
        "private_index",
    )
    if (
        private_index["schema_version"] != PRIVATE_INDEX_SCHEMA
        or private_index["visibility"] != "private"
        or private_index["public_report_sha256"]
        != public_report_sha256
    ):
        raise NonPrivateReferenceVerificationError(
            "Private seed index metadata/link mismatch"
        )
    _require_self_hash(
        private_index,
        "private_index_sha256",
        "private_index",
    )
    private_dataset_bindings = private_index["dataset_bindings"]
    if (
        not isinstance(private_dataset_bindings, list)
        or len(private_dataset_bindings) != len(DATASET_ORDER)
    ):
        raise NonPrivateReferenceVerificationError(
            "private_index.dataset_bindings must contain three entries"
        )
    private_binding_by_dataset: dict[str, dict[str, Any]] = {}
    for index, binding in enumerate(private_dataset_bindings):
        if not isinstance(binding, dict):
            raise NonPrivateReferenceVerificationError(
                f"private_index.dataset_bindings[{index}] "
                "must be an object"
            )
        _require_exact_keys(
            binding,
            PRIVATE_DATASET_BINDING_KEYS,
            f"private_index.dataset_bindings[{index}]",
        )
        dataset = binding["dataset"]
        if (
            dataset not in DATASET_ORDER
            or dataset in private_binding_by_dataset
        ):
            raise NonPrivateReferenceVerificationError(
                "Private dataset binding ids are invalid"
            )
        private_binding_by_dataset[dataset] = binding
    if list(private_binding_by_dataset) != list(DATASET_ORDER):
        raise NonPrivateReferenceVerificationError(
            "Private dataset bindings are non-canonical"
        )
    private_runs = private_index["runs"]
    if not isinstance(private_runs, list):
        raise NonPrivateReferenceVerificationError(
            "private_index.runs must be a list"
        )
    private_run_keys: list[tuple[str, str]] = []
    private_seeds_by_dataset: dict[str, list[int]] = {}
    for index, item in enumerate(private_runs):
        if not isinstance(item, dict):
            raise NonPrivateReferenceVerificationError(
                f"private_index.runs[{index}] must be an object"
            )
        _require_exact_keys(
            item,
            PRIVATE_RUN_KEYS,
            f"private_index.runs[{index}]",
        )
        dataset = item["dataset"]
        run_id = item["run_id"]
        seed = item["research_seed"]
        if (
            dataset not in DATASET_ORDER
            or not isinstance(run_id, str)
            or not _is_int(seed)
            or seed < 0
            or seed >= 2**63
        ):
            raise NonPrivateReferenceVerificationError(
                f"private_index.runs[{index}] is invalid"
            )
        private_run_keys.append((dataset, run_id))
        private_seeds_by_dataset.setdefault(dataset, []).append(seed)
    if any(
        len(seeds) != len(set(seeds))
        for seeds in private_seeds_by_dataset.values()
    ):
        raise NonPrivateReferenceVerificationError(
            "Private seeds repeat within a dataset"
        )
    if len(
        {tuple(seeds) for seeds in private_seeds_by_dataset.values()}
    ) != 1:
        raise NonPrivateReferenceVerificationError(
            "Datasets use different prespecified seed sequences"
        )

    datasets = report["datasets"]
    if (
        not isinstance(datasets, list)
        or [item.get("dataset") for item in datasets if isinstance(item, dict)]
        != list(DATASET_ORDER)
        or report["dataset_count"] != len(DATASET_ORDER)
    ):
        raise NonPrivateReferenceVerificationError(
            "Report datasets are not the three canonical datasets"
        )

    observed_public_run_keys: list[tuple[str, str]] = []
    total_runs = 0
    current_execution_source_sha256 = (
        execution_source_bundle_sha256_v2()
    )
    for dataset_report in datasets:
        dataset = dataset_report["dataset"]
        context = dataset
        _require_exact_keys(
            dataset_report,
            DATASET_KEYS,
            context,
        )
        config = DATASET_CONFIG[dataset]
        profile = get_benchmark_profile_v2(str(config["profile_id"]))
        prepared = _prepare_dataset(dataset, args)
        prepared.assert_registered_full_conformance()
        with tempfile.TemporaryDirectory(
            prefix=f"unitdp_verify_nonprivate_{dataset}_"
        ) as temporary:
            mapping_path = write_prepared_mapping_v2(
                prepared,
                Path(temporary) / "train_mapping.csv",
            )
            contract = load_owner_poisson_contract_v2(
                config["contract"]
            )
            route = compile_owner_poisson_contract_v2(
                contract,
                mapping_path=mapping_path,
                preprocessing_artifact_path=config["preprocessor"],
                require_executable=True,
            )

        binding = dataset_report["data_binding"]
        if not isinstance(binding, dict):
            raise NonPrivateReferenceVerificationError(
                f"{context}.data_binding must be an object"
            )
        _require_exact_keys(
            binding,
            DATA_BINDING_KEYS,
            f"{context}.data_binding",
        )
        expected_binding = {
            "benchmark_profile_id": profile.profile_id,
            "benchmark_profile_sha256": profile.registry_sha256,
            "full_data_conformance_sha256": (
                prepared.full_data_conformance_sha256()
            ),
            "preprocessing_artifact_sha256": (
                prepared.preprocessor.payload_sha256
            ),
            "public_contract_sha256": (
                contract.public_contract_sha256
            ),
            "public_plan_sha256": route.public_plan_sha256,
            "execution_source_bundle_sha256": (
                current_execution_source_sha256
            ),
        }
        if binding != expected_binding:
            raise NonPrivateReferenceVerificationError(
                f"{context}.data_binding mismatch"
            )

        transformed_train = prepared.preprocessor.transform(
            prepared.raw_train_x
        )
        transformed_test = prepared.preprocessor.transform(
            prepared.raw_test_x
        )
        selected_indices = [
            record.row_index
            for record in route.selected_mapping.records
        ]
        selected_x = transformed_train[selected_indices]
        selected_y = prepared.train_y[selected_indices]
        selection_protocol = dataset_report["selection_protocol"]
        if not isinstance(selection_protocol, dict):
            raise NonPrivateReferenceVerificationError(
                f"{context}.selection_protocol must be an object"
            )
        _require_exact_keys(
            selection_protocol,
            SELECTION_PROTOCOL_KEYS,
            f"{context}.selection_protocol",
        )
        if selection_protocol != {
            "scope": "public_benchmark_evaluation_only",
            "policy": "compiled_v2_selected_mapping",
            "data_dependent_details_disclosed": False,
        }:
            raise NonPrivateReferenceVerificationError(
                f"{context} public selection protocol mismatch"
            )
        expected_private_selection = {
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
        if (
            private_binding_by_dataset[dataset]
            != expected_private_selection
        ):
            raise NonPrivateReferenceVerificationError(
                f"{context} private selection binding mismatch"
            )

        training = dataset_report["training_configuration"]
        if not isinstance(training, dict):
            raise NonPrivateReferenceVerificationError(
                f"{context}.training_configuration must be an object"
            )
        _require_exact_keys(
            training,
            TRAINING_KEYS,
            f"{context}.training_configuration",
        )
        expected_training = {
            "model_id": contract.model_id,
            "optimizer": "sgd",
            "loss": "cross_entropy",
            "epochs": 20,
            "batch_size": 64,
            "learning_rate": contract.learning_rate,
            "class_weights": (
                list(contract.class_weights)
                if contract.class_weights is not None
                else None
            ),
            "shuffle": True,
        }
        if training != expected_training:
            raise NonPrivateReferenceVerificationError(
                f"{context} training configuration mismatch"
            )

        runs = dataset_report["runs"]
        expected_run_ids = [
            f"run_{index:03d}" for index in range(1, 6)
        ]
        if (
            not isinstance(runs, list)
            or dataset_report["run_count"] != 5
            or len(runs) != 5
        ):
            raise NonPrivateReferenceVerificationError(
                f"{context} must contain five runs"
            )
        observed_run_ids: list[str] = []
        verified_runs: list[dict[str, Any]] = []
        for index, run in enumerate(runs):
            run_context = f"{context}.runs[{index}]"
            if not isinstance(run, dict):
                raise NonPrivateReferenceVerificationError(
                    f"{run_context} must be an object"
                )
            _require_exact_keys(run, RUN_KEYS, run_context)
            run_id = run["run_id"]
            if not isinstance(run_id, str):
                raise NonPrivateReferenceVerificationError(
                    f"{run_context}.run_id must be a string"
                )
            observed_run_ids.append(run_id)
            observed_public_run_keys.append((dataset, run_id))
            model_path = _safe_model_path(
                collection_dir,
                run["model_artifact"],
                dataset=dataset,
                run_id=run_id,
            )
            if (
                _file_sha256(model_path)
                != run["model_file_sha256"]
            ):
                raise NonPrivateReferenceVerificationError(
                    f"{context}.{run_id} model file hash mismatch"
                )
            model_payload = _load_object(model_path)
            model = load_model_artifact_v2(model_path)
            if (
                model_payload.get("payload_sha256")
                != run["model_payload_sha256"]
                or model_payload.get("state_sha256")
                != run["model_state_sha256"]
            ):
                raise NonPrivateReferenceVerificationError(
                    f"{context}.{run_id} model link mismatch"
                )
            evaluation = run["evaluation"]
            if not isinstance(evaluation, dict):
                raise NonPrivateReferenceVerificationError(
                    f"{context}.{run_id}.evaluation must be an object"
                )
            _require_exact_keys(
                evaluation,
                EVALUATION_KEYS,
                f"{context}.{run_id}.evaluation",
            )
            normalized = {
                metric: _require_metric(
                    evaluation[metric],
                    f"{context}.{run_id}.evaluation.{metric}",
                )
                for metric in EVALUATION_KEYS
            }
            expected_evaluation = _evaluate(
                model,
                selected_x=selected_x,
                selected_y=selected_y,
                test_x=transformed_test,
                test_y=prepared.test_y,
            )
            if any(
                not _same_number(
                    normalized[metric],
                    expected_evaluation[metric],
                )
                for metric in EVALUATION_KEYS
            ):
                raise NonPrivateReferenceVerificationError(
                    f"{context}.{run_id} evaluation mismatch"
                )
            verified_run = dict(run)
            verified_run["evaluation"] = normalized
            verified_runs.append(verified_run)

        if observed_run_ids != expected_run_ids:
            raise NonPrivateReferenceVerificationError(
                f"{context} run ids are non-canonical"
            )
        public_run_dirs = sorted(
            path.name
            for path in (collection_dir / dataset).iterdir()
            if path.is_dir()
        )
        if public_run_dirs != expected_run_ids:
            raise NonPrivateReferenceVerificationError(
                f"{context} run directories mismatch"
            )
        _verify_aggregate(
            dataset_report["aggregate"],
            verified_runs,
            f"{context}.aggregate",
        )
        total_runs += len(runs)

    if (
        private_run_keys != observed_public_run_keys
        or report["run_count"] != total_runs
        or total_runs != 15
    ):
        raise NonPrivateReferenceVerificationError(
            "Public/private run indexes disagree"
        )
    top_level_public_files = sorted(
        path.name
        for path in collection_dir.iterdir()
        if path.is_file()
    )
    if top_level_public_files != [report_path.name]:
        raise NonPrivateReferenceVerificationError(
            "Collection contains unregistered top-level files"
        )

    verification: dict[str, Any] = {
        "schema_version": VERIFICATION_SCHEMA,
        "status": "verified",
        "public_report_sha256": public_report_sha256,
        "dataset_count": len(DATASET_ORDER),
        "run_count": total_runs,
        "exact_v2_data_and_selection_bindings_verified": True,
        "model_artifacts_and_metrics_recomputed": True,
        "current_registered_execution_source_bundle_sha256": (
            current_execution_source_sha256
        ),
        "verifier_sha256": _file_sha256(Path(__file__).resolve()),
    }
    verification["verification_report_sha256"] = _canonical_sha256(
        verification
    )
    return verification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "report",
        nargs="?",
        default=str(
            ROOT
            / "reports"
            / "v2_nonprivate_reference_v32_20260724"
            / "public_nonprivate_reference_v1.json"
        ),
    )
    parser.add_argument("--uci-root", default=str(DEFAULT_UCI_ROOT))
    parser.add_argument("--wisdm-raw", default=str(DEFAULT_WISDM_RAW))
    parser.add_argument(
        "--sepsis-cache",
        default=str(DEFAULT_SEPSIS_CACHE),
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    verification = verify(Path(args.report), args=args)
    encoded = json.dumps(
        verification,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
    )
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
