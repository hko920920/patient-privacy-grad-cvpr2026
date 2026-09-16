#!/usr/bin/env python3
"""Build the source-bound Audit v5 UCI HAR DP-SGD evidence route.

The published UCI HAR rows are treated as generated windows.  The route
authorizes only a window add/remove statement.  It deliberately blocks raw
event and owner wording because the release does not expose exact raw-session
support and this route registers no owner contribution cap.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from mapping_evidence import file_sha256, inspect_mapping_evidence
from privacy_claim_validator import canonical_json_sha256


SCENARIO = "uci_har_v5_generated_window"
PIPELINE_VERSION = "1.0.0"
CODE_ARTIFACT_ID = "scripts/uci_har_v5_pipeline.py"
ACCOUNTANT_CHECKER_ARTIFACT_ID = "scripts/rdp_accountant_lite.py"
REGISTRY_FILE = "mechanism_registry_v2_0.json"
MECHANISM_ID = "secure_systemrandom_uci_har_dpsgd"
SAMPLING_IMPLEMENTATION = "systemrandom_randrange_bernoulli_v1"
SECURE_RNG_BACKEND = "python.secrets.SystemRandom"
NOISE_GENERATION = "normalvariate_discard1_sum4_div2_v1"
NOISE_HARDENING = "known_fp_reconstruction_mitigation_four_draw_v1"
GRADIENT_AGGREGATION = "clipped_sum_plus_gaussian_noise"
UPDATE_NORMALIZATION = "public_constant_step_no_dataset_denominator"
RAW_DOMAIN_ID = "uci_har_published_window_row_domain_v1"
DATASET_ID = "UCI_HAR_Dataset_v1_0_published_windows"
REQUIRED_FILES = (
    "train/X_train.txt",
    "train/y_train.txt",
    "train/subject_train.txt",
    "test/X_test.txt",
    "test/y_test.txt",
    "test/subject_test.txt",
    "activity_labels.txt",
    "features.txt",
)
EXPECTED_INPUTS = {
    "train/X_train.txt": (
        66006256,
        "9c1246099c8d5463779eec5a3845cc3898df9d8c715441ab2a1232d4421fc42f",
    ),
    "train/y_train.txt": (
        14704,
        "415b3357e2e2d5e70a45c781947c6bdd80e0410a579d862eedcd6bbce3c694ef",
    ),
    "train/subject_train.txt": (
        20152,
        "66e1e2fbe9b396fa5155f531dc3348c803f503be663e2318772d49947db9153a",
    ),
    "test/X_test.txt": (
        26458166,
        "99035209add50d17650e847d8a4658d784a5707466ebb7c2deb97c28e5250a78",
    ),
    "test/y_test.txt": (
        5894,
        "2a94cb53ca46b956c1b2aebd4a09a72badd8a8e9ea9cdfe9f31b3306a08666bb",
    ),
    "test/subject_test.txt": (
        7934,
        "63b8c577f4a85431c06bd87f6384608c90a9f72f05c7956dd21255dc63515c07",
    ),
    "activity_labels.txt": (
        80,
        "f6e6b292704438261f0283b2a82659ab0661447dc16520938bdf379ed1bf8de0",
    ),
    "features.txt": (
        15785,
        "b384889885b1c8680ecf250e0e605b4c79536cdf09ce9f2fa7225a3c6773b5c7",
    ),
}

RUNTIME_EXACT_FIELDS = (
    "mechanism_id",
    "mechanism_version",
    "accountant_id",
    "accountant_version",
    "accounting_unit",
    "accountant_adjacency",
    "sampling_unit",
    "sampler_law",
    "accountant_sampler_law",
    "sample_rate_numerator",
    "sample_rate_denominator",
    "sampling_implementation",
    "steps",
    "secure_mode",
    "secure_rng_backend",
    "noise_generation",
    "noise_hardening",
    "clipping_unit",
    "noising_unit",
    "privacy_convention",
    "gradient_aggregation",
    "update_normalization",
    "adapter_registry_entry_sha256",
    "code_artifact_id",
    "code_sha256",
    "accountant_checker_artifact_id",
    "accountant_checker_sha256",
    "selected_mapping_sha256",
    "pipeline_sha256",
    "delta_convention",
    "runtime_evidence_sha256",
)
RUNTIME_NUMERIC_FIELDS = (
    "sample_rate",
    "noise_multiplier",
    "clipping_norm",
    "optimizer_step_size",
    "epsilon",
    "delta",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def input_manifest(data_root: Path) -> Dict[str, Any]:
    files = []
    for relative in REQUIRED_FILES:
        path = data_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_bytes, expected_sha256 = EXPECTED_INPUTS[relative]
        actual_bytes = path.stat().st_size
        actual_sha256 = file_sha256(path)
        if (actual_bytes, actual_sha256) != (expected_bytes, expected_sha256):
            raise RuntimeError(
                f"UCI input drift for {relative}: "
                f"{actual_bytes}/{actual_sha256} != "
                f"{expected_bytes}/{expected_sha256}"
            )
        files.append(
            {
                "path": relative,
                "bytes": actual_bytes,
                "sha256": actual_sha256,
            }
        )
    manifest = {
        "dataset_id": DATASET_ID,
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files),
    }
    manifest["dataset_bundle_sha256"] = canonical_json_sha256(manifest)
    return manifest


def load_split(
    data_root: Path,
    split: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.loadtxt(
        data_root / split / f"X_{split}.txt",
        dtype=np.float64,
    )
    labels = np.loadtxt(
        data_root / split / f"y_{split}.txt",
        dtype=np.int64,
    )
    subjects = np.loadtxt(
        data_root / split / f"subject_{split}.txt",
        dtype=np.int64,
    )
    if features.ndim != 2 or features.shape[1] != 561:
        raise ValueError(f"Unexpected {split} feature shape: {features.shape}")
    if labels.shape != (len(features),) or subjects.shape != (len(features),):
        raise ValueError(f"Misaligned {split} arrays.")
    if not np.all(np.isfinite(features)):
        raise ValueError(f"Non-finite {split} feature.")
    if float(features.min()) < -1.0000001 or float(features.max()) > 1.0000001:
        raise ValueError(f"Published UCI feature bound violated in {split}.")
    if set(np.unique(labels).tolist()) != {1, 2, 3, 4, 5, 6}:
        raise ValueError(f"Unexpected {split} labels.")
    return features, labels - 1, subjects


def mapping_rows(subjects: np.ndarray, labels: np.ndarray) -> List[Dict[str, Any]]:
    owner_ordinals: Dict[int, int] = {}
    rows: List[Dict[str, Any]] = []
    for row_index, (subject, label) in enumerate(zip(subjects, labels)):
        subject_id = int(subject)
        ordinal = owner_ordinals.get(subject_id, 0)
        owner_ordinals[subject_id] = ordinal + 1
        rows.append(
            {
                "scenario": SCENARIO,
                "window_id": f"train_row_{row_index:05d}",
                "generated_unit": "window",
                "owner_id": str(subject_id),
                "owner_ids": str(subject_id),
                "start": ordinal,
                "end": ordinal + 1,
                "split": "train",
                "activity_index": int(label) + 1,
                "source_row_index": row_index,
            }
        )
    return rows


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def secure_normal_array(
    rng: secrets.SystemRandom,
    shape: Tuple[int, ...],
    standard_deviation: float,
) -> np.ndarray:
    count = int(np.prod(shape))

    def draw() -> np.ndarray:
        values = [
            rng.normalvariate(0.0, standard_deviation)
            for _ in range(count)
        ]
        return np.asarray(values, dtype=np.float64).reshape(shape)

    draw()
    return (draw() + draw() + draw() + draw()) / 2.0


def train_secure_dpsgd(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    sample_rate_numerator: int,
    sample_rate_denominator: int,
    steps: int,
    noise_multiplier: float,
    clipping_norm: float,
    optimizer_step_size: float,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if (
        sample_rate_numerator < 1
        or sample_rate_denominator < 1
        or sample_rate_numerator > sample_rate_denominator
    ):
        raise ValueError("Invalid exact rational sample rate.")
    if steps < 1 or noise_multiplier <= 0 or clipping_norm <= 0:
        raise ValueError("Invalid DP-SGD parameter.")
    if not math.isfinite(optimizer_step_size) or optimizer_step_size <= 0:
        raise ValueError("The public optimizer step must be finite and positive.")

    augmented = np.concatenate(
        [train_x, np.ones((len(train_x), 1), dtype=np.float64)],
        axis=1,
    )
    weights = np.zeros((augmented.shape[1], 6), dtype=np.float64)
    rng = secrets.SystemRandom()
    trace: List[Dict[str, Any]] = []
    for step in range(steps):
        mask = np.fromiter(
            (
                rng.randrange(sample_rate_denominator) < sample_rate_numerator
                for _ in range(len(train_x))
            ),
            dtype=np.bool_,
            count=len(train_x),
        )
        selected_x = augmented[mask]
        selected_y = train_y[mask]
        summed_gradient = np.zeros_like(weights)
        clipped_count = 0
        max_norm = 0.0
        if len(selected_x):
            probabilities = stable_softmax(selected_x @ weights)
            errors = probabilities
            errors[np.arange(len(selected_y)), selected_y] -= 1.0
            per_record = selected_x[:, :, None] * errors[:, None, :]
            norms = np.sqrt(np.sum(per_record * per_record, axis=(1, 2)))
            max_norm = float(norms.max(initial=0.0))
            factors = np.minimum(
                1.0,
                clipping_norm / np.maximum(norms, 1e-300),
            )
            clipped_count = int(np.sum(factors < 1.0))
            summed_gradient = np.sum(
                per_record * factors[:, None, None],
                axis=0,
            )
        noise = secure_normal_array(
            rng,
            weights.shape,
            noise_multiplier * clipping_norm,
        )
        weights -= optimizer_step_size * (summed_gradient + noise)
        trace.append(
            {
                "step": step + 1,
                "poisson_batch_count": int(mask.sum()),
                "clipped_record_count": clipped_count,
                "max_unclipped_gradient_norm": f"{max_norm:.12g}",
            }
        )
    return weights, trace


def predict(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    augmented = np.concatenate(
        [features, np.ones((len(features), 1), dtype=np.float64)],
        axis=1,
    )
    return np.argmax(augmented @ weights, axis=1)


def macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    scores = []
    for class_index in range(6):
        true_positive = int(
            np.sum((labels == class_index) & (predictions == class_index))
        )
        false_positive = int(
            np.sum((labels != class_index) & (predictions == class_index))
        )
        false_negative = int(
            np.sum((labels == class_index) & (predictions != class_index))
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(
            0.0 if denominator == 0 else 2 * true_positive / denominator
        )
    return float(np.mean(scores))


def accountant_epsilon(
    sample_rate: float,
    steps: int,
    noise_multiplier: float,
    delta: float,
) -> float:
    from rdp_accountant_lite import epsilon_for_poisson_gaussian

    epsilon, _ = epsilon_for_poisson_gaussian(
        sample_rate,
        steps,
        noise_multiplier,
        delta,
    )
    return epsilon


def parameter_block(
    identifier: str,
    classification: str,
    parameters: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "id": identifier,
        "classification": classification,
        "parameters": dict(parameters),
        "parameters_sha256": canonical_json_sha256(parameters),
    }


def build_runtime_trace(
    mechanism: Mapping[str, Any],
    privacy: Mapping[str, Any],
    manifest: Mapping[str, Any],
    selected_mapping_sha256: str,
    pipeline_sha256: str,
    evidence: Mapping[str, Any],
) -> Dict[str, Any]:
    evidence_hash = canonical_json_sha256(evidence)
    expected = {
        **{key: mechanism.get(key) for key in RUNTIME_EXACT_FIELDS},
        **{key: mechanism.get(key) for key in RUNTIME_NUMERIC_FIELDS},
        "code_artifact_id": manifest["code_artifact_id"],
        "code_sha256": manifest["code_sha256"],
        "accountant_checker_artifact_id": manifest[
            "accountant_checker_artifact_id"
        ],
        "accountant_checker_sha256": manifest["accountant_checker_sha256"],
        "selected_mapping_sha256": selected_mapping_sha256,
        "pipeline_sha256": pipeline_sha256,
        "epsilon": privacy["epsilon"],
        "delta": privacy["delta"],
        "delta_convention": privacy["delta_convention"],
        "runtime_evidence_sha256": evidence_hash,
    }
    payload = {
        key: expected.get(key)
        for key in (*RUNTIME_EXACT_FIELDS, *RUNTIME_NUMERIC_FIELDS)
    }
    return {
        "status": "matched",
        **payload,
        "trace_hash": canonical_json_sha256(payload),
        "evidence": dict(evidence),
    }


def run_checked(command: Sequence[str], cwd: Path) -> None:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        raise RuntimeError(
            f"Command failed with exit {completed.returncode}: {' '.join(command)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="data/uci_har/extracted/UCI HAR Dataset",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/uci_har_v5_pilot_001",
    )
    parser.add_argument("--sample-rate-numerator", type=int, default=1)
    parser.add_argument("--sample-rate-denominator", type=int, default=50)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--noise-multiplier", type=float, default=2.0)
    parser.add_argument("--clipping-norm", type=float, default=1.0)
    parser.add_argument("--optimizer-step-size", type=float, default=1.0 / 200.0)
    parser.add_argument("--delta", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    if not data_root.is_absolute():
        data_root = project_root / data_root
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    code_hash = file_sha256(Path(__file__).resolve())
    accountant_path = project_root / ACCOUNTANT_CHECKER_ARTIFACT_ID
    accountant_hash = file_sha256(accountant_path)
    registry_path = project_root / "docs" / REGISTRY_FILE
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_key = f"{MECHANISM_ID}@{PIPELINE_VERSION}"
    registry_entry = registry.get("mechanisms", {}).get(registry_key)
    if not isinstance(registry_entry, Mapping):
        raise RuntimeError(f"Missing registry entry {registry_key}.")
    if registry_entry.get("code_artifact_id") != CODE_ARTIFACT_ID:
        raise RuntimeError("Registry code artifact mismatch.")
    if registry_entry.get("code_sha256") != code_hash:
        raise RuntimeError("Pipeline source does not match registry.")
    if (
        registry_entry.get("accountant_checker_artifact_id")
        != ACCOUNTANT_CHECKER_ARTIFACT_ID
        or registry_entry.get("accountant_checker_sha256") != accountant_hash
    ):
        raise RuntimeError("Accountant checker does not match registry.")
    registry_entry_hash = canonical_json_sha256(registry_entry)

    source_manifest = input_manifest(data_root)
    write_json(output_dir / "input_manifest.json", source_manifest)
    train_x, train_y, train_subjects = load_split(data_root, "train")
    test_x, test_y, test_subjects = load_split(data_root, "test")
    if set(train_subjects.tolist()) & set(test_subjects.tolist()):
        raise RuntimeError("Published train/test subject split overlaps.")

    rows = mapping_rows(train_subjects, train_y)
    mapping_path = output_dir / "mapping.csv"
    write_csv(mapping_path, rows)
    mapping = inspect_mapping_evidence(mapping_path, SCENARIO)
    if not mapping.valid:
        raise RuntimeError(mapping.issues)
    if mapping.selected_record_count != 7352 or mapping.event_kappa != 1:
        raise RuntimeError(
            f"Unexpected mapping facts: {mapping.selected_record_count}, "
            f"{mapping.event_kappa}"
        )

    weights, trace = train_secure_dpsgd(
        train_x,
        train_y,
        sample_rate_numerator=args.sample_rate_numerator,
        sample_rate_denominator=args.sample_rate_denominator,
        steps=args.steps,
        noise_multiplier=args.noise_multiplier,
        clipping_norm=args.clipping_norm,
        optimizer_step_size=args.optimizer_step_size,
    )
    train_predictions = predict(train_x, weights)
    test_predictions = predict(test_x, weights)
    metrics = {
        "train_accuracy": float(np.mean(train_predictions == train_y)),
        "test_accuracy": float(np.mean(test_predictions == test_y)),
        "test_macro_f1": macro_f1(test_y, test_predictions),
        "train_records": len(train_x),
        "test_records": len(test_x),
        "train_owners": len(set(train_subjects.tolist())),
        "test_owners": len(set(test_subjects.tolist())),
        "feature_count": train_x.shape[1],
    }
    model_path = output_dir / "released_model.json"
    write_json(
        model_path,
        {
            "mechanism_id": MECHANISM_ID,
            "mechanism_version": PIPELINE_VERSION,
            "feature_count": train_x.shape[1],
            "class_indices": [1, 2, 3, 4, 5, 6],
            "weights_including_bias": weights.tolist(),
        },
    )
    trace_path = output_dir / "batch_trace.csv"
    write_csv(trace_path, trace)
    write_json(output_dir / "metrics.json", metrics)

    sample_rate = args.sample_rate_numerator / args.sample_rate_denominator
    epsilon = accountant_epsilon(
        sample_rate,
        args.steps,
        args.noise_multiplier,
        args.delta,
    )
    mechanism = {
        "mechanism_id": MECHANISM_ID,
        "mechanism_version": PIPELINE_VERSION,
        "accountant_id": "opacus_rdp",
        "accountant_version": importlib.metadata.version("opacus"),
        "accounting_unit": "window",
        "accountant_adjacency": "add_remove",
        "sampling_unit": "window",
        "sampler_law": "poisson",
        "accountant_sampler_law": "poisson",
        "sample_rate": sample_rate,
        "sample_rate_numerator": args.sample_rate_numerator,
        "sample_rate_denominator": args.sample_rate_denominator,
        "sampling_implementation": SAMPLING_IMPLEMENTATION,
        "steps": args.steps,
        "noise_multiplier": args.noise_multiplier,
        "secure_mode": True,
        "clipping_unit": "window",
        "clipping_norm": args.clipping_norm,
        "noising_unit": "window",
        "privacy_convention": "rdp_converted",
        "secure_rng_backend": SECURE_RNG_BACKEND,
        "noise_generation": NOISE_GENERATION,
        "noise_hardening": NOISE_HARDENING,
        "gradient_aggregation": GRADIENT_AGGREGATION,
        "update_normalization": UPDATE_NORMALIZATION,
        "optimizer_step_size": args.optimizer_step_size,
        "adapter_registry_entry_sha256": registry_entry_hash,
    }
    privacy = {
        "epsilon": epsilon,
        "delta": args.delta,
        "delta_convention": "accountant_delta_only",
    }
    raw_domain = parameter_block(
        RAW_DOMAIN_ID,
        "public_fixed",
        {
            "released_unit": "precomputed_window_row",
            "exact_raw_session_support_available": False,
            "feature_count": 561,
            "feature_value_domain": "finite_real_in_closed_interval_minus1_plus1",
            "row_identity": "published_split_and_row_index",
        },
    )
    contribution_policy = parameter_block(
        "no_registered_owner_conversion_cap",
        "not_applicable",
        {
            "cap": None,
            "enforcement": "none",
            "reason": "owner wording intentionally unsupported",
        },
    )
    schedule_parameters = {
        "scenario": SCENARIO,
        "published_split": "train",
        "sample_rate_numerator": args.sample_rate_numerator,
        "sample_rate_denominator": args.sample_rate_denominator,
        "sampling_implementation": SAMPLING_IMPLEMENTATION,
        "steps": args.steps,
        "selected_record_count": mapping.selected_record_count,
    }
    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_bundle_sha256": source_manifest["dataset_bundle_sha256"],
        "input_manifest_sha256": file_sha256(output_dir / "input_manifest.json"),
        "split_id": canonical_json_sha256(
            {
                "policy": "published_train_test_subject_split",
                "train_subjects": sorted(set(train_subjects.tolist())),
                "test_subjects": sorted(set(test_subjects.tolist())),
            }
        ),
        "code_version": "uci_har_v5_pipeline_v1",
        "code_artifact_id": CODE_ARTIFACT_ID,
        "code_sha256": code_hash,
        "accountant_checker_artifact_id": ACCOUNTANT_CHECKER_ARTIFACT_ID,
        "accountant_checker_sha256": accountant_hash,
        "support_convention": "half_open_integer_intervals",
        "record_identity_policy": "public_fixed_ids",
        "generated_record_unit": "window",
        "cross_owner_dependency": "none",
        "mapping_file_sha256": mapping.mapping_file_sha256,
        "selected_mapping_sha256": mapping.selected_mapping_sha256,
        "selected_record_count": mapping.selected_record_count,
        "mechanism_sha256": canonical_json_sha256(mechanism),
        "influence_support": {
            "status": "complete",
            "components": [
                "features",
                "labels",
                "weights",
                "selection",
                "preprocessing",
            ],
        },
        "owner_attribution_status": "complete",
        "preprocessing": [
            parameter_block(
                "uci_har_published_feature_vector",
                "public_fixed",
                {
                    "feature_count": 561,
                    "fitted_on_private_execution_data": False,
                    "additional_pipeline_normalization": "none",
                    "observed_public_file_min": -1.0,
                    "observed_public_file_max": 1.0,
                },
            )
        ],
        "raw_domain": raw_domain,
        "contribution_policy": contribution_policy,
        "schedule": {
            "id": "uci_har_public_train_rows_poisson",
            "mode": "fixed_mapping",
            "evidence_status": "validated",
            "parameters": schedule_parameters,
            "parameters_sha256": canonical_json_sha256(schedule_parameters),
        },
        "population_definition": "published UCI HAR training-window rows",
        "population_size_policy": "not_used_in_release_mechanism",
        "model_selection_status": "public",
        "released_model_sha256": file_sha256(model_path),
        "batch_trace_sha256": file_sha256(trace_path),
    }
    pipeline_sha256 = canonical_json_sha256(manifest)
    runtime = build_runtime_trace(
        mechanism,
        privacy,
        manifest,
        mapping.selected_mapping_sha256,
        pipeline_sha256,
        {
            "input_manifest_sha256": file_sha256(
                output_dir / "input_manifest.json"
            ),
            "batch_trace_sha256": file_sha256(trace_path),
            "released_model_sha256": file_sha256(model_path),
            "executed_updates": len(trace),
            "secure_sampling_rng": SAMPLING_IMPLEMENTATION,
            "secure_noise_rng": NOISE_GENERATION,
        },
    )
    report = {
        "schema_version": "2.0",
        "scenario": SCENARIO,
        "mechanism": mechanism,
        "privacy": privacy,
        "pipeline": {
            "mapping_file_sha256": mapping.mapping_file_sha256,
            "selected_mapping_sha256": mapping.selected_mapping_sha256,
            "pipeline_sha256": pipeline_sha256,
            "manifest": manifest,
        },
        "runtime_trace": runtime,
        "rng_assurance": "known_attack_hardened_secure",
        "noise_seed_policy": "secure_private_unrecorded",
        "public_certificate": {
            "metadata_classification": "public_benchmark",
        },
        "claim_contracts": {
            "window": {
                "claimed_unit": "window",
                "raw_adjacency": "add_remove",
                "stability_status": "validated",
                "stability_method": "identity_generated_unit",
                "stability_bound": 1,
                "support_status": "complete",
                "schedule_mode": "fixed_mapping",
                "schedule_evidence_status": "validated",
                "conversion_method": "identity",
            },
            "event": {
                "claimed_unit": "event",
                "raw_adjacency": "replace_one",
                "stability_status": "unverified",
                "stability_method": "raw_session_support_unavailable",
                "stability_bound": 1,
                "support_status": "incomplete",
                "schedule_mode": "fixed_mapping",
                "schedule_evidence_status": "validated",
                "conversion_method": "identity",
            },
            "owner": {
                "claimed_unit": "owner",
                "raw_adjacency": "owner_add_remove",
                "stability_status": "unverified",
                "stability_method": "owner_partition_add_remove",
                "stability_bound": int(mapping.owner_kappa or 1),
                "support_status": "complete",
                "schedule_mode": "fixed_mapping",
                "schedule_evidence_status": "validated",
                "conversion_method": "builtin_group",
            },
        },
    }
    report_path = output_dir / "privacy_report.json"
    write_json(report_path, report)

    del train_x, train_y, test_x, test_y, weights
    del train_predictions, test_predictions
    gc.collect()

    expected_batch = max(
        1,
        int(round(sample_rate * mapping.selected_record_count)),
    )
    audit_dir = output_dir / "audit"
    run_checked(
        [
            sys.executable,
            str(project_root / "scripts" / "unit_audit.py"),
            "--mapping",
            str(mapping_path),
            "--batch-size",
            str(expected_batch),
            "--epochs",
            "1",
            "--claim-units",
            "window,event,user",
            "--schedule-mode",
            "fixed_mapping",
            "--output-dir",
            str(audit_dir),
        ],
        project_root,
    )
    certificate_dir = output_dir / "certificates"
    for claim in ("window", "event", "owner"):
        command = [
            sys.executable,
            str(project_root / "scripts" / "build_unit_certificate.py"),
            "--audit-csv",
            str(audit_dir / "unit_audit.csv"),
            "--mapping-csv",
            str(mapping_path),
            "--privacy-report-json",
            str(report_path),
            "--scenario",
            SCENARIO,
            "--claim-unit",
            claim,
            "--output-dir",
            str(certificate_dir),
        ]
        if claim in {"event", "owner"}:
            command.append("--diagnostic-exit-zero")
        run_checked(command, project_root)

    certificates = {
        claim: json.loads(
            (
                certificate_dir
                / f"certificate_{SCENARIO}_{claim}.json"
            ).read_text(encoding="utf-8")
        )
        for claim in ("window", "event", "owner")
    }
    assertions = {
        "window_allowed_direct": (
            certificates["window"]["release_status"] == "ALLOWED"
            and certificates["window"]["unit_path"] == "DIRECT"
            and bool(certificates["window"]["supported_statement"])
        ),
        "event_blocked_without_wording": (
            certificates["event"]["release_status"] != "ALLOWED"
            and not certificates["event"]["supported_statement"]
        ),
        "owner_blocked_without_wording": (
            certificates["owner"]["release_status"] != "ALLOWED"
            and not certificates["owner"]["supported_statement"]
        ),
    }
    write_json(output_dir / "route_assertions.json", assertions)
    if not all(assertions.values()):
        raise RuntimeError(f"UCI route assertions failed: {assertions}")

    elapsed = time.perf_counter() - started
    metrics["pipeline_seconds"] = elapsed
    write_json(output_dir / "metrics.json", metrics)
    summary = [
        "# Audit v5 UCI HAR generated-window route",
        "",
        f"- Dataset bundle SHA-256: `{source_manifest['dataset_bundle_sha256']}`",
        f"- Pipeline source SHA-256: `{code_hash}`",
        f"- Mapping rows: {mapping.selected_record_count}",
        f"- Train/test owners: {metrics['train_owners']} / {metrics['test_owners']}",
        f"- Features: {metrics['feature_count']}",
        f"- Privacy: epsilon={epsilon:.12g}, delta={args.delta:.12g}",
        (
            "- Test accuracy / macro-F1: "
            f"{metrics['test_accuracy']:.6f} / {metrics['test_macro_f1']:.6f}"
        ),
        f"- Pipeline seconds: {elapsed:.3f}",
        "",
        "| Query | Path | Release | Positive wording |",
        "| --- | --- | --- | --- |",
    ]
    for claim in ("window", "event", "owner"):
        certificate = certificates[claim]
        summary.append(
            f"| {claim} | {certificate['unit_path']} | "
            f"{certificate['release_status']} | "
            f"{bool(certificate['supported_statement'])} |"
        )
    summary.extend(
        [
            "",
            "This route treats published UCI HAR rows as generated windows.",
            "It does not authorize raw-event or owner wording and is not a",
            "state-of-the-art utility comparison.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text(
        "\n".join(summary),
        encoding="utf-8",
    )
    print(f"Wrote UCI HAR v5 route: {output_dir}")


if __name__ == "__main__":
    main()
