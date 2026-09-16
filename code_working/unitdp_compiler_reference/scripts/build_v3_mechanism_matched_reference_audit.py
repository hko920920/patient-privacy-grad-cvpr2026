"""Build an exact same-mechanism reference comparison for both strict routes.

The two reference training loops import no ``unitdp`` production module.  This
builder deliberately shares only the bound public benchmark preparation,
preprocessing artifacts, and contract scalars with production.  It reads local
private run manifests to recover research seeds and traces, but emits only
aggregate public match counts.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get("UNITDP_DATA_ROOT", str(ROOT / "data"))
)
SRC = ROOT / "src"
TESTS = ROOT / "tests"
for search_path in (SRC, TESTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from owner_poisson_trace_reference import (  # noqa: E402
    ReferenceRecord as PoissonReferenceRecord,
    replay_owner_poisson_linear_trace,
)
from owner_srswor_trace_reference import (  # noqa: E402
    ReferenceRecord as SrsworReferenceRecord,
    replay_owner_srswor_linear_trace,
)
from unitdp.benchmark_data_v2 import (  # noqa: E402
    PreparedBenchmarkV2,
    prepare_sepsis_v2,
    prepare_uci_har_v2,
    prepare_wisdm_v2,
)


DEFAULT_UCI_ROOT = (
    DATA_ROOT / "uci_har" / "extracted" / "UCI HAR Dataset"
)
DEFAULT_WISDM_RAW = (
    DATA_ROOT / "wisdm" / "WISDM_ar_v1.1" / "WISDM_ar_v1.1_raw.txt"
)
DEFAULT_SEPSIS_CACHE = DATA_ROOT / "sepsis2019_cache"
DEFAULT_POISSON_EVIDENCE = (
    ROOT / "reports" / "v2_registered_5seed_v32_20260724"
)
DEFAULT_SRSWOR_EVIDENCE = (
    ROOT / "reports" / "v3_srswor_registered_5seed_v32_20260724"
)
DEFAULT_OUTPUT = (
    ROOT
    / "reports"
    / "v3_mechanism_matched_reference_v32_20260724"
    / "public_mechanism_matched_reference_v32.json"
)
LOCAL_IDENTITY_TOKEN = "SO" + "GANG"
WINDOWS_DRIVE_PREFIX = "C:" + chr(92)


DATASET_CONFIG = {
    "uci": {
        "preprocessor": (
            ROOT
            / "configs"
            / "preprocessing"
            / "uci_har_published_train_standard_scaler_v1.json"
        ),
    },
    "wisdm": {
        "preprocessor": (
            ROOT
            / "configs"
            / "preprocessing"
            / "wisdm_v1_1_train_owners_1_25_stats24_standard_scaler_v1.json"
        ),
    },
    "sepsis": {
        "preprocessor": (
            ROOT
            / "configs"
            / "preprocessing"
            / "physionet2019_setA_first400_basic12x6_public_scaler_v1.json"
        ),
    },
}


ROUTE_CONFIG = {
    "poisson": {
        "route_id": "owa_owner_poisson_rdp_add_remove_v2",
        "adjacency": "add_remove_one_owner",
        "sampler": "independent_bernoulli_owner",
        "contract_dir": ROOT / "configs" / "v2",
        "contract_suffix": "_owner_poisson_v2.yaml",
        "collection_name": "public_collection_v2.json",
        "summary_name": "public_summary_v2.json",
        "private_index_name": "private_run_index_v2.json",
        "private_manifest_name": "private_execution_manifest_v2.json",
        "public_execution_name": "public_execution_v2.json",
        "model_name": "model_v2.json",
        "reference_file": TESTS / "owner_poisson_trace_reference.py",
        "diagnostic_surface": (
            "sample counts, computed-vector counts, clipping norms, "
            "fixed updates, metrics, and final model"
        ),
    },
    "srswor": {
        "route_id": "owa_owner_srswor_rdp_replace_one_v3",
        "adjacency": "replace_one_owner",
        "sampler": "uniform_fixed_size_without_replacement_owner",
        "contract_dir": ROOT / "configs" / "v3",
        "contract_suffix": "_owner_srswor_v3.yaml",
        "collection_name": "public_collection_srswor_v3.json",
        "summary_name": "public_summary_srswor_v3.json",
        "private_index_name": "private_run_index_srswor_v3.json",
        "private_manifest_name": "private_execution_srswor_v3.json",
        "public_execution_name": "public_execution_srswor_v3.json",
        "model_name": "model_srswor_v3.json",
        "reference_file": TESTS / "owner_srswor_trace_reference.py",
        "diagnostic_surface": (
            "sampled positions, selected rows, clipping norms, fixed "
            "updates, metrics, and final model"
        ),
    },
}


class MechanismMatchedReferenceError(RuntimeError):
    """Raised when any production/reference correspondence check fails."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MechanismMatchedReferenceError(
                f"Duplicate JSON key: {key}"
            )
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_pairs,
    )
    if not isinstance(value, dict):
        raise MechanismMatchedReferenceError(
            f"Expected JSON object: {path}"
        )
    return value


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise MechanismMatchedReferenceError(
                f"Duplicate YAML key: {key}"
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.load(
        path.read_text(encoding="utf-8"),
        Loader=_UniqueKeyLoader,
    )
    if not isinstance(value, dict):
        raise MechanismMatchedReferenceError(
            f"Expected YAML object: {path}"
        )
    return value


def _assert_reference_import_boundary(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        module: str | None = None
        if isinstance(node, ast.ImportFrom):
            module = node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "unitdp" or alias.name.startswith("unitdp."):
                    raise MechanismMatchedReferenceError(
                        f"Reference imports production module: {path}"
                    )
        if module == "unitdp" or (
            module is not None and module.startswith("unitdp.")
        ):
            raise MechanismMatchedReferenceError(
                f"Reference imports production module: {path}"
            )


def _prepare_dataset(
    dataset: str,
    args: argparse.Namespace,
) -> PreparedBenchmarkV2:
    artifact = Path(DATASET_CONFIG[dataset]["preprocessor"])
    if dataset == "uci":
        prepared = prepare_uci_har_v2(
            dataset_root=Path(args.uci_root),
            preprocessing_artifact_path=artifact,
        )
    elif dataset == "wisdm":
        prepared = prepare_wisdm_v2(
            raw_path=Path(args.wisdm_raw),
            preprocessing_artifact_path=artifact,
        )
    elif dataset == "sepsis":
        prepared = prepare_sepsis_v2(
            cache_dir=Path(args.sepsis_cache),
            preprocessing_artifact_path=artifact,
        )
    else:
        raise MechanismMatchedReferenceError(
            f"Unknown dataset: {dataset}"
        )
    prepared.assert_registered_full_conformance()
    return prepared


def _owner_id(record: dict[str, object]) -> str:
    owner_id = str(record["owner_id"])
    owner_ids = [
        value.strip()
        for value in str(record["owner_ids"]).split(";")
        if value.strip()
    ]
    if owner_ids != [owner_id]:
        raise MechanismMatchedReferenceError(
            "Reference requires exactly one matching owner per record"
        )
    return owner_id


def select_support_balanced_records(
    records: tuple[dict[str, object], ...],
    *,
    cap: int,
) -> tuple[dict[str, object], ...]:
    """Independent implementation of the registered public owner-local cap."""

    if cap <= 0:
        raise MechanismMatchedReferenceError("Cap must be positive")
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[_owner_id(record)].append(record)
    selected: list[dict[str, object]] = []
    for owner in sorted(grouped):
        ordered = sorted(
            grouped[owner],
            key=lambda row: (
                int(row["start"]),
                int(row["end"]),
                str(row["window_id"]),
            ),
        )
        if cap >= len(ordered):
            selected.extend(ordered)
            continue
        positions = (
            np.linspace(0, len(ordered) - 1, cap)
            .round()
            .astype(int)
        )
        selected.extend(ordered[int(position)] for position in positions)
    deduped = {
        str(record["window_id"]): record for record in selected
    }
    if len(deduped) != len(selected):
        raise MechanismMatchedReferenceError(
            "Selected mapping contains duplicate window identifiers"
        )
    return tuple(deduped.values())


def mapping_canonical_sha256(
    records: tuple[dict[str, object], ...],
) -> str:
    payload = []
    for record in sorted(records, key=lambda row: str(row["window_id"])):
        payload.append(
            {
                "window_id": str(record["window_id"]),
                "owner_ids": [_owner_id(record)],
                "start": int(record["start"]),
                "end": int(record["end"]),
                "row_index": int(record["row_index"]),
                "label": (
                    None
                    if record.get("label") in {None, ""}
                    else str(record["label"])
                ),
                "scenario": str(record.get("scenario") or "default"),
            }
        )
    return payload_sha256(payload)


def _reference_records(
    records: tuple[dict[str, object], ...],
    record_type: Callable[..., Any],
) -> tuple[Any, ...]:
    return tuple(
        record_type(
            owner_id=_owner_id(record),
            window_id=str(record["window_id"]),
            start=int(record["start"]),
            end=int(record["end"]),
            row_index=int(record["row_index"]),
        )
        for record in records
    )


def _model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(
            json.dumps(
                list(value.shape),
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _compare_model_artifact(
    model: torch.nn.Module,
    artifact: dict[str, Any],
) -> int:
    if artifact.get("schema_version") != "unitdp.linear_model_artifact.v2":
        raise MechanismMatchedReferenceError("Unexpected model schema")
    tensors = artifact.get("tensors")
    if not isinstance(tensors, list):
        raise MechanismMatchedReferenceError("Model tensors are missing")
    by_name = {
        str(item["name"]): item
        for item in tensors
        if isinstance(item, dict)
    }
    state = model.state_dict()
    if set(by_name) != set(state):
        raise MechanismMatchedReferenceError(
            "Reference and artifact tensor names differ"
        )
    for name, tensor in state.items():
        item = by_name[name]
        if item.get("dtype") != "float32_le":
            raise MechanismMatchedReferenceError(
                "Unexpected serialized tensor dtype"
            )
        value = (
            tensor.detach()
            .cpu()
            .contiguous()
            .numpy()
            .astype("<f4", copy=False)
        )
        if list(value.shape) != item.get("shape"):
            raise MechanismMatchedReferenceError(
                f"Tensor shape differs: {name}"
            )
        if value.tobytes(order="C").hex() != item.get("data_hex"):
            raise MechanismMatchedReferenceError(
                f"Tensor bytes differ: {name}"
            )
    observed_hash = _model_state_sha256(model)
    if observed_hash != artifact.get("state_sha256"):
        raise MechanismMatchedReferenceError(
            "Reference model state digest differs"
        )
    return len(state)


def _transform(
    raw_features: np.ndarray,
    artifact_path: Path,
) -> np.ndarray:
    artifact = load_json(artifact_path)
    mean = np.asarray(artifact["mean"], dtype=np.float64)
    scale = np.asarray(artifact["scale"], dtype=np.float64)
    transformed = np.asarray(raw_features).astype(np.float32, copy=True)
    transformed -= mean
    transformed /= scale
    return transformed


def _rank_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float | None, float | None]:
    if len(np.unique(labels)) < 2:
        return None, None
    try:
        if probabilities.shape[1] == 2:
            auroc = roc_auc_score(labels, probabilities[:, 1])
            auprc = average_precision_score(labels, probabilities[:, 1])
        else:
            classes = np.arange(probabilities.shape[1])
            one_hot = np.eye(probabilities.shape[1], dtype=int)[labels]
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


def _reference_metrics(
    model: torch.nn.Module,
    prepared: PreparedBenchmarkV2,
    artifact_path: Path,
) -> dict[str, float | None]:
    features = _transform(prepared.raw_test_x, artifact_path)
    labels = np.asarray(prepared.test_y, dtype=np.int64)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(features))
        predictions = logits.argmax(dim=1).cpu().numpy()
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    auroc, auprc = _rank_metrics(labels, probabilities)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "auroc": auroc,
        "auprc": auprc,
    }


def _metric_gap(
    reference: dict[str, float | None],
    production: dict[str, Any],
) -> float:
    maximum = 0.0
    for key in ("accuracy", "macro_f1", "auroc", "auprc"):
        left = reference[key]
        right = production.get(key)
        if left is None or right is None:
            if left is not None or right is not None:
                raise MechanismMatchedReferenceError(
                    f"Metric nullability differs: {key}"
                )
            continue
        gap = abs(float(left) - float(right))
        maximum = max(maximum, gap)
        if gap != 0.0:
            raise MechanismMatchedReferenceError(
                f"Metric differs for {key}: {gap}"
            )
    return maximum


def _safe_private_path(dataset_root: Path, relative: str) -> Path:
    root = dataset_root.resolve()
    candidate = (dataset_root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise MechanismMatchedReferenceError(
            "Private manifest path escapes its dataset directory"
        )
    return candidate


def _compare_poisson_diagnostics(
    reference: Any,
    diagnostics: list[dict[str, Any]],
) -> None:
    if tuple(
        int(item["sampled_owner_count"]) for item in diagnostics
    ) != reference.sampled_owner_counts:
        raise MechanismMatchedReferenceError(
            "Poisson sampled-owner counts differ"
        )
    if tuple(
        int(item["owner_vectors_computed"]) for item in diagnostics
    ) != reference.owner_vectors_computed:
        raise MechanismMatchedReferenceError(
            "Poisson owner-vector counts differ"
        )
    if tuple(
        float(item["max_unclipped_owner_norm"]) for item in diagnostics
    ) != reference.max_unclipped_norms:
        raise MechanismMatchedReferenceError(
            "Poisson unclipped norms differ"
        )
    if tuple(
        float(item["max_clipped_owner_norm"]) for item in diagnostics
    ) != reference.max_clipped_norms:
        raise MechanismMatchedReferenceError(
            "Poisson clipped norms differ"
        )
    if not all(
        bool(item["noise_applied"])
        and bool(item["optimizer_step_applied"])
        for item in diagnostics
    ):
        raise MechanismMatchedReferenceError(
            "Poisson fixed noisy-update schedule differs"
        )


def _compare_srswor_diagnostics(
    reference: Any,
    diagnostics: list[dict[str, Any]],
) -> None:
    positions = tuple(
        tuple(int(value) for value in item["sampled_owner_positions"])
        for item in diagnostics
    )
    if positions != reference.sampled_owner_positions:
        raise MechanismMatchedReferenceError(
            "SRSWOR sampled positions differ"
        )
    rows = tuple(
        tuple(
            (
                int(selected["owner_position"]),
                tuple(int(value) for value in selected["row_indices"]),
            )
            for selected in item["selected_rows"]
        )
        for item in diagnostics
    )
    if rows != reference.selected_rows_by_position:
        raise MechanismMatchedReferenceError(
            "SRSWOR selected rows differ"
        )
    if tuple(
        float(item["max_unclipped_owner_norm"]) for item in diagnostics
    ) != reference.max_unclipped_norms:
        raise MechanismMatchedReferenceError(
            "SRSWOR unclipped norms differ"
        )
    if tuple(
        float(item["max_clipped_owner_norm"]) for item in diagnostics
    ) != reference.max_clipped_norms:
        raise MechanismMatchedReferenceError(
            "SRSWOR clipped norms differ"
        )
    if not all(
        bool(item["optimizer_step_applied"])
        and int(item["noise_parameter_tensors"]) == 2
        for item in diagnostics
    ):
        raise MechanismMatchedReferenceError(
            "SRSWOR fixed noisy-update schedule differs"
        )


def _contract_path(route: str, dataset: str) -> Path:
    config = ROUTE_CONFIG[route]
    return (
        Path(config["contract_dir"])
        / f"{dataset}{config['contract_suffix']}"
    )


def compare_route_dataset(
    *,
    route: str,
    dataset: str,
    prepared: PreparedBenchmarkV2,
    evidence_root: Path,
) -> dict[str, Any]:
    route_config = ROUTE_CONFIG[route]
    contract_path = _contract_path(route, dataset)
    contract = load_yaml(contract_path)
    mechanism = contract["mechanism"]
    policy = contract["contribution_policy"]
    optimization = contract["optimization"]
    data = contract["data"]
    if policy.get("type") != "support_balanced_cap":
        raise MechanismMatchedReferenceError(
            "Reference only supports the registered support-balanced cap"
        )
    cap = int(policy["max_windows_per_owner"])
    selected = select_support_balanced_records(
        prepared.train_records,
        cap=cap,
    )
    selected_hash = mapping_canonical_sha256(selected)
    source_hash = mapping_canonical_sha256(prepared.train_records)
    if prepared.full_data_conformance_sha256() != (
        data["public_protocol"]["full_data_conformance_sha256"]
    ):
        raise MechanismMatchedReferenceError(
            "Prepared data differs from contract full-data binding"
        )

    dataset_root = evidence_root / dataset
    summary = load_json(
        dataset_root / str(route_config["summary_name"])
    )
    if summary.get("full_data_conformance_sha256") != (
        prepared.full_data_conformance_sha256()
    ):
        raise MechanismMatchedReferenceError(
            "Production summary uses a different prepared dataset"
        )
    private_index = load_json(
        dataset_root
        / "_private"
        / str(route_config["private_index_name"])
    )
    runs = private_index.get("runs")
    if not isinstance(runs, list) or not runs:
        raise MechanismMatchedReferenceError(
            "Private run index is empty"
        )

    class_weights_raw = optimization.get("class_weights")
    class_weights = (
        None
        if class_weights_raw is None
        else tuple(float(value) for value in class_weights_raw)
    )
    artifact_path = Path(DATASET_CONFIG[dataset]["preprocessor"])
    model_matches = 0
    tensor_matches = 0
    metric_matches = 0
    diagnostic_step_matches = 0
    maximum_metric_gap = 0.0

    for run in runs:
        if not isinstance(run, dict):
            raise MechanismMatchedReferenceError(
                "Private run entry is not an object"
            )
        run_id = str(run["run_id"])
        private_path = _safe_private_path(
            dataset_root,
            str(run["private_manifest"]),
        )
        private = load_json(private_path)
        seed = int(run["research_seed"])
        if int(private["research_seed"]) != seed:
            raise MechanismMatchedReferenceError(
                "Run index and private manifest seed differ"
            )
        if private.get("source_mapping_canonical_sha256") != source_hash:
            raise MechanismMatchedReferenceError(
                "Reference source mapping hash differs"
            )
        if private.get("selected_mapping_canonical_sha256") != selected_hash:
            raise MechanismMatchedReferenceError(
                "Independent contribution selection differs"
            )
        diagnostics = private.get("step_diagnostics")
        if not isinstance(diagnostics, list):
            raise MechanismMatchedReferenceError(
                "Private step diagnostics are missing"
            )
        if len(diagnostics) != int(mechanism["total_steps"]):
            raise MechanismMatchedReferenceError(
                "Production step count differs from the contract"
            )

        if route == "poisson":
            records = _reference_records(
                selected,
                PoissonReferenceRecord,
            )
            reference = replay_owner_poisson_linear_trace(
                raw_train_x=prepared.raw_train_x,
                train_y=prepared.train_y,
                records=records,
                preprocessing_artifact_path=artifact_path,
                master_seed=seed,
                input_dim=int(data["input_dim"]),
                num_classes=int(data["num_classes"]),
                owner_sample_rate=float(
                    mechanism["owner_sample_rate"]
                ),
                total_steps=int(mechanism["total_steps"]),
                clip_norm=float(mechanism["clip_norm"]),
                noise_multiplier=float(
                    mechanism["noise_multiplier"]
                ),
                update_denominator=float(
                    mechanism["update_denominator"]
                ),
                learning_rate=float(
                    optimization["learning_rate"]
                ),
                windows_per_owner_per_step=int(
                    policy["windows_per_owner_per_step"]
                ),
                class_weights=class_weights,
            )
            _compare_poisson_diagnostics(reference, diagnostics)
        else:
            records = _reference_records(
                selected,
                SrsworReferenceRecord,
            )
            reference = replay_owner_srswor_linear_trace(
                raw_train_x=prepared.raw_train_x,
                train_y=prepared.train_y,
                records=records,
                preprocessing_artifact_path=artifact_path,
                master_seed=seed,
                input_dim=int(data["input_dim"]),
                num_classes=int(data["num_classes"]),
                source_dataset_size=int(
                    mechanism["source_dataset_size"]
                ),
                sample_size=int(mechanism["sample_size"]),
                total_steps=int(mechanism["total_steps"]),
                clip_norm=float(mechanism["clip_norm"]),
                noise_multiplier=float(
                    mechanism["noise_multiplier"]
                ),
                update_denominator=float(
                    mechanism["update_denominator"]
                ),
                learning_rate=float(
                    optimization["learning_rate"]
                ),
                windows_per_owner_per_step=int(
                    policy["windows_per_owner_per_step"]
                ),
                class_weights=class_weights,
            )
            _compare_srswor_diagnostics(reference, diagnostics)

        model_path = (
            dataset_root
            / run_id
            / str(route_config["model_name"])
        )
        tensor_matches += _compare_model_artifact(
            reference.model,
            load_json(model_path),
        )
        model_matches += 1

        public_execution = load_json(
            dataset_root
            / run_id
            / str(route_config["public_execution_name"])
        )
        reference_metrics = _reference_metrics(
            reference.model,
            prepared,
            artifact_path,
        )
        maximum_metric_gap = max(
            maximum_metric_gap,
            _metric_gap(
                reference_metrics,
                public_execution["public_evaluation"],
            ),
        )
        metric_matches += 4
        diagnostic_step_matches += len(diagnostics)

    return {
        "dataset": dataset,
        "run_count": len(runs),
        "fixed_steps_per_run": int(mechanism["total_steps"]),
        "full_data_binding_match": True,
        "independent_policy_selection_match": True,
        "diagnostic_steps_exact": diagnostic_step_matches,
        "final_models_bitwise": model_matches,
        "final_tensors_bitwise": tensor_matches,
        "public_metrics_exact": metric_matches,
        "maximum_public_metric_gap": maximum_metric_gap,
        "contract_sha256": file_sha256(contract_path),
        "preprocessor_sha256": file_sha256(artifact_path),
    }


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _assert_public_report(report: dict[str, Any]) -> None:
    encoded = json.dumps(
        report,
        sort_keys=True,
        ensure_ascii=False,
    )
    forbidden = (
        "research_seed",
        "sampled_owner_positions",
        "selected_rows",
        "step_diagnostics",
        "max_unclipped_owner_norm",
        "max_clipped_owner_norm",
        LOCAL_IDENTITY_TOKEN,
        WINDOWS_DRIVE_PREFIX,
    )
    for token in forbidden:
        if token in encoded:
            raise MechanismMatchedReferenceError(
                f"Public report exposes forbidden token: {token}"
            )
    if report.get("visibility") != "public":
        raise MechanismMatchedReferenceError(
            "Reference report must be public"
        )
    if report["randomness"]["random_coins_disclosed"]:
        raise MechanismMatchedReferenceError(
            "Reference report must not disclose random coins"
        )


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    for config in ROUTE_CONFIG.values():
        _assert_reference_import_boundary(
            Path(config["reference_file"])
        )

    prepared = {
        dataset: _prepare_dataset(dataset, args)
        for dataset in ("uci", "wisdm", "sepsis")
    }
    evidence_roots = {
        "poisson": Path(args.poisson_evidence),
        "srswor": Path(args.srswor_evidence),
    }
    route_rows = []
    total_runs = 0
    total_steps = 0
    total_models = 0
    total_tensors = 0
    total_metrics = 0
    maximum_metric_gap = 0.0

    for route in ("poisson", "srswor"):
        config = ROUTE_CONFIG[route]
        evidence_root = evidence_roots[route]
        collection_path = (
            evidence_root / str(config["collection_name"])
        )
        if not collection_path.is_file():
            raise MechanismMatchedReferenceError(
                f"Missing production collection: {collection_path}"
            )
        datasets = [
            compare_route_dataset(
                route=route,
                dataset=dataset,
                prepared=prepared[dataset],
                evidence_root=evidence_root,
            )
            for dataset in ("uci", "wisdm", "sepsis")
        ]
        route_runs = sum(int(row["run_count"]) for row in datasets)
        route_steps = sum(
            int(row["diagnostic_steps_exact"]) for row in datasets
        )
        route_models = sum(
            int(row["final_models_bitwise"]) for row in datasets
        )
        route_tensors = sum(
            int(row["final_tensors_bitwise"]) for row in datasets
        )
        route_metrics = sum(
            int(row["public_metrics_exact"]) for row in datasets
        )
        route_gap = max(
            float(row["maximum_public_metric_gap"])
            for row in datasets
        )
        route_rows.append(
            {
                "route_id": config["route_id"],
                "adjacency": config["adjacency"],
                "sampler": config["sampler"],
                "reference_executor": _relative(
                    Path(config["reference_file"])
                ),
                "reference_imports_production_executor": False,
                "diagnostic_surface": config["diagnostic_surface"],
                "production_collection_sha256": file_sha256(
                    collection_path
                ),
                "datasets": datasets,
                "totals": {
                    "runs": route_runs,
                    "diagnostic_steps_exact": route_steps,
                    "final_models_bitwise": route_models,
                    "final_tensors_bitwise": route_tensors,
                    "public_metrics_exact": route_metrics,
                    "maximum_public_metric_gap": route_gap,
                },
            }
        )
        total_runs += route_runs
        total_steps += route_steps
        total_models += route_models
        total_tensors += route_tensors
        total_metrics += route_metrics
        maximum_metric_gap = max(maximum_metric_gap, route_gap)

    source_files = (
        Path(__file__).resolve(),
        TESTS / "owner_poisson_trace_reference.py",
        TESTS / "owner_srswor_trace_reference.py",
        SRC / "unitdp" / "benchmark_data_v2.py",
    )
    report: dict[str, Any] = {
        "schema_version": (
            "unitdp.mechanism_matched_reference_audit_public.v3"
        ),
        "visibility": "public",
        "release_status": "research_non_release",
        "route_count": 2,
        "scope": (
            "finite exact correspondence to frozen production runs; "
            "not a formal refinement proof"
        ),
        "reference_boundary": {
            "shared": [
                "bound public benchmark preparation",
                "fixed preprocessing artifact",
                "registered contract scalars",
                "PyTorch and scikit-learn numerical libraries",
            ],
            "independently_reimplemented": [
                "owner-local support-balanced selection",
                "domain-separated random streams",
                "model initialization",
                "owner sampling",
                "within-owner selection",
                "owner gradients and clipping",
                "Gaussian update",
                "metric recomputation",
            ],
            "production_executor_imported_by_references": False,
        },
        "randomness": {
            "private_inputs_used_locally": True,
            "random_coins_disclosed": False,
            "portable_regeneration_without_private_inputs": False,
        },
        "routes": route_rows,
        "totals": {
            "datasets": 6,
            "runs": total_runs,
            "diagnostic_steps_exact": total_steps,
            "final_models_bitwise": total_models,
            "final_tensors_bitwise": total_tensors,
            "public_metrics_exact": total_metrics,
            "maximum_public_metric_gap": maximum_metric_gap,
        },
        "source_files_sha256": {
            _relative(path): file_sha256(path) for path in source_files
        },
    }
    _assert_public_report(report)
    report["payload_sha256"] = payload_sha256(report)
    _assert_public_report(report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uci-root", default=str(DEFAULT_UCI_ROOT))
    parser.add_argument("--wisdm-raw", default=str(DEFAULT_WISDM_RAW))
    parser.add_argument(
        "--sepsis-cache",
        default=str(DEFAULT_SEPSIS_CACHE),
    )
    parser.add_argument(
        "--poisson-evidence",
        default=str(DEFAULT_POISSON_EVIDENCE),
    )
    parser.add_argument(
        "--srswor-evidence",
        default=str(DEFAULT_SRSWOR_EVIDENCE),
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {output}")
    report = build_report(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            report,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        "mechanism-matched reference audit: "
        f"{report['totals']['runs']} runs, "
        f"{report['totals']['diagnostic_steps_exact']} steps, "
        f"{report['totals']['final_tensors_bitwise']} tensors"
    )
    print(f"payload_sha256={report['payload_sha256']}")


if __name__ == "__main__":
    main()
