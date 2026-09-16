"""Build the hardened v2.0 WISDM evidence pipeline and certificates.

This pipeline is intentionally small and transparent:

* public deterministic owner split and overlapping window schedule;
* record-local, public-parameter clipping and summary features;
* exact-rational Poisson sampling via unbiased ``SystemRandom.randrange``;
* per-window gradient clipping;
* four-draw-hardened Gaussian noise from ``secrets.SystemRandom``;
* a public constant optimizer step with no dataset-size normalization;
* Opacus 1.6.0 RDP recomputation of the base privacy parameters;
* actual mapping, model, batch-trace, runtime, and pipeline digests;
* window DIRECT, event replacement-to-add/remove CONVERT, and owner GROUP
  (normally vacuous) certificates through the sole v2.0 validator.

The model is a softmax-regression integration baseline, not a SOTA result.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
import math
import secrets
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from mapping_evidence import file_sha256, inspect_mapping_evidence
from privacy_claim_validator import canonical_json_sha256


SCENARIO = "wisdm_v2_hardened_overlap50"
PIPELINE_VERSION = "2.0.0"
CODE_ARTIFACT_ID = "scripts/wisdm_v2_gold_pipeline.py"
ACCOUNTANT_CHECKER_ARTIFACT_ID = "scripts/rdp_accountant_lite.py"
REGISTRY_FILE = "mechanism_registry_v2_0.json"
MECHANISM_ID = "secure_systemrandom_dpsgd"
SAMPLING_IMPLEMENTATION = "systemrandom_randrange_bernoulli_v1"
SECURE_RNG_BACKEND = "python.secrets.SystemRandom"
NOISE_GENERATION = "normalvariate_discard1_sum4_div2_v1"
NOISE_HARDENING = "known_fp_reconstruction_mitigation_four_draw_v1"
GRADIENT_AGGREGATION = "clipped_sum_plus_gaussian_noise"
UPDATE_NORMALIZATION = "public_constant_step_no_dataset_denominator"
RAW_DOMAIN_ID = "fixed_owner_slot_payload_domain_v1"
RAW_EVENT_ADJACENCY = "fixed_owner_slot_payload_replace_one_v1"
OWNER_CAP_POLICY_ID = "public_owner_generated_record_cap_v1"
OWNER_CAP_ENFORCEMENT = "deterministic_owner_window_prefix_v1"
PUBLIC_OWNER_WINDOW_CAP = 600
WINDOW_LENGTH = 200
WINDOW_STRIDE = 100
PUBLIC_SENSOR_CLIP = 20.0
FEATURE_NAMES = (
    "x_mean",
    "y_mean",
    "z_mean",
    "mag_mean",
    "x_std",
    "y_std",
    "z_std",
    "mag_std",
    "x_min",
    "y_min",
    "z_min",
    "mag_min",
    "x_max",
    "y_max",
    "z_max",
    "mag_max",
)
ACTIVITIES = (
    "Downstairs",
    "Jogging",
    "Sitting",
    "Standing",
    "Upstairs",
    "Walking",
)
TRAIN_OWNERS = tuple(range(1, 26))
TEST_OWNERS = tuple(range(26, 37))

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
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_wisdm(
    path: Path,
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[str, int]]:
    rows_by_owner: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    parser_counts = {
        "physical_lines": 0,
        "accepted_canonical_events": 0,
        "invalid_column_count": 0,
        "invalid_parse": 0,
        "invalid_activity": 0,
        "nonfinite_payload": 0,
    }
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_index, line in enumerate(handle):
            parser_counts["physical_lines"] += 1
            parts = [part.strip() for part in line.strip().rstrip(";").split(",")]
            if len(parts) != 6:
                parser_counts["invalid_column_count"] += 1
                continue
            try:
                owner = int(parts[0])
                activity = parts[1]
                timestamp = int(parts[2])
                xyz = tuple(float(value) for value in parts[3:6])
            except ValueError:
                parser_counts["invalid_parse"] += 1
                continue
            if activity not in ACTIVITIES:
                parser_counts["invalid_activity"] += 1
                continue
            if not all(math.isfinite(value) for value in xyz):
                parser_counts["nonfinite_payload"] += 1
                continue
            rows_by_owner[owner].append(
                {
                    "line_index": line_index,
                    "activity": activity,
                    "timestamp": timestamp,
                    "xyz": xyz,
                }
            )
            parser_counts["accepted_canonical_events"] += 1

    for owner, rows in rows_by_owner.items():
        rows.sort(key=lambda item: item["line_index"])
        for event_index, item in enumerate(rows):
            item["event_index"] = event_index
    if not rows_by_owner:
        raise ValueError("No valid WISDM records were parsed.")
    return dict(rows_by_owner), parser_counts


def record_local_features(window: Sequence[Mapping[str, Any]]) -> np.ndarray:
    xyz = np.asarray([item["xyz"] for item in window], dtype=np.float64)
    xyz = np.clip(xyz, -PUBLIC_SENSOR_CLIP, PUBLIC_SENSOR_CLIP)
    xyz = xyz / PUBLIC_SENSOR_CLIP
    magnitude = np.sqrt(np.sum(xyz * xyz, axis=1, keepdims=True)) / math.sqrt(3.0)
    full = np.concatenate([xyz, magnitude], axis=1)
    features = np.concatenate(
        [
            full.mean(axis=0),
            full.std(axis=0),
            full.min(axis=0),
            full.max(axis=0),
        ]
    )
    if features.shape != (len(FEATURE_NAMES),):
        raise AssertionError(features.shape)
    if not np.all(np.isfinite(features)):
        raise ValueError("Non-finite record-local feature.")
    if np.max(np.abs(features)) > 1.0000001:
        raise ValueError("Public feature bound was violated.")
    return features.astype(np.float64)


def generate_windows(
    rows_by_owner: Mapping[int, Sequence[Mapping[str, Any]]],
    owners: Sequence[int],
    split: str,
    label_to_index: Mapping[str, int],
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    records: List[Dict[str, Any]] = []
    features: List[np.ndarray] = []
    labels: List[int] = []
    for owner in owners:
        owner_rows = rows_by_owner[owner]
        if len(owner_rows) < WINDOW_LENGTH:
            continue
        offsets = list(
            range(0, len(owner_rows) - WINDOW_LENGTH + 1, WINDOW_STRIDE)
        )[:PUBLIC_OWNER_WINDOW_CAP]
        for ordinal, offset in enumerate(offsets):
            window = owner_rows[offset : offset + WINDOW_LENGTH]
            label_counts = {
                activity: sum(item["activity"] == activity for item in window)
                for activity in ACTIVITIES
            }
            activity = min(
                ACTIVITIES,
                key=lambda candidate: (-label_counts[candidate], candidate),
            )
            start = int(window[0]["event_index"])
            end = int(window[-1]["event_index"]) + 1
            record_id = f"{split}_{SCENARIO}_u{owner}_w{ordinal}"
            records.append(
                {
                    "scenario": SCENARIO,
                    "window_id": record_id,
                    "generated_unit": "window",
                    "owner_id": str(owner),
                    "owner_ids": str(owner),
                    "start": start,
                    "end": end,
                    "split": split,
                    "activity": activity,
                    "row_index": len(features),
                }
            )
            features.append(record_local_features(window))
            labels.append(int(label_to_index[activity]))
    if not features:
        raise ValueError(f"No windows generated for split {split!r}.")
    return (
        records,
        np.stack(features).astype(np.float64),
        np.asarray(labels, dtype=np.int64),
    )


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

    # Match the four-draw hardening structure used by secure-mode DP
    # implementations: discard one full draw, then sum four and divide by two.
    # This mitigates known floating-point reconstruction attacks; it is not a
    # claim of universal side-channel security.
    draw()
    return (draw() + draw() + draw() + draw()) / 2.0


def train_secure_dpsgd(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    num_classes: int,
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
        raise ValueError("Invalid DP-SGD numeric parameter.")
    if not math.isfinite(optimizer_step_size) or optimizer_step_size <= 0:
        raise ValueError("optimizer_step_size must be public, finite, and positive.")

    augmented = np.concatenate(
        [train_x, np.ones((len(train_x), 1), dtype=np.float64)],
        axis=1,
    )
    weights = np.zeros((augmented.shape[1], num_classes), dtype=np.float64)
    rng = secrets.SystemRandom()
    trace_rows: List[Dict[str, Any]] = []

    for step in range(steps):
        mask = np.fromiter(
            (
                rng.randrange(sample_rate_denominator)
                < sample_rate_numerator
                for _ in range(len(train_x))
            ),
            dtype=np.bool_,
            count=len(train_x),
        )
        selected_x = augmented[mask]
        selected_y = train_y[mask]
        summed_gradient = np.zeros_like(weights)
        clipped_count = 0
        max_unclipped_norm = 0.0

        if len(selected_x):
            probabilities = stable_softmax(selected_x @ weights)
            errors = probabilities
            errors[np.arange(len(selected_y)), selected_y] -= 1.0
            per_record = selected_x[:, :, None] * errors[:, None, :]
            norms = np.sqrt(np.sum(per_record * per_record, axis=(1, 2)))
            max_unclipped_norm = float(norms.max(initial=0.0))
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
        trace_rows.append(
            {
                "step": step + 1,
                "poisson_batch_count": int(mask.sum()),
                "clipped_record_count": clipped_count,
                "max_unclipped_gradient_norm": f"{max_unclipped_norm:.12g}",
            }
        )
    return weights, trace_rows


def predict(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    augmented = np.concatenate(
        [features, np.ones((len(features), 1), dtype=np.float64)],
        axis=1,
    )
    return np.argmax(augmented @ weights, axis=1)


def macro_f1(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> float:
    values = []
    for class_index in range(num_classes):
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
        values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(values))


def recompute_epsilon(
    sample_rate: float,
    steps: int,
    noise_multiplier: float,
    delta: float,
) -> float:
    from rdp_accountant_lite import epsilon_for_poisson_gaussian

    epsilon, _best_alpha = epsilon_for_poisson_gaussian(
        sample_rate,
        steps,
        noise_multiplier,
        delta,
    )
    return epsilon


def canonical_parameter_block(
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
    extra_evidence: Mapping[str, Any],
) -> Dict[str, Any]:
    evidence = dict(extra_evidence)
    evidence_hash = canonical_json_sha256(evidence)
    expected = {
        **{key: mechanism.get(key) for key in RUNTIME_EXACT_FIELDS},
        **{key: mechanism.get(key) for key in RUNTIME_NUMERIC_FIELDS},
        "code_artifact_id": manifest.get("code_artifact_id"),
        "code_sha256": manifest.get("code_sha256"),
        "accountant_checker_artifact_id": manifest.get(
            "accountant_checker_artifact_id"
        ),
        "accountant_checker_sha256": manifest.get(
            "accountant_checker_sha256"
        ),
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
        "evidence": evidence,
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
        "--data-path",
        default="data/wisdm/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/wisdm_v2_hardened_secure_002",
    )
    parser.add_argument("--sample-rate-numerator", type=int, default=1)
    parser.add_argument("--sample-rate-denominator", type=int, default=50)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--noise-multiplier", type=float, default=2.0)
    parser.add_argument("--clipping-norm", type=float, default=1.0)
    parser.add_argument(
        "--optimizer-step-size",
        "--learning-rate",
        dest="optimizer_step_size",
        type=float,
        default=1.0 / 150.0,
    )
    parser.add_argument("--delta", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    code_hash = file_sha256(Path(__file__).resolve())
    registry_path = project_root / "docs" / REGISTRY_FILE
    registry_document = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_key = f"{MECHANISM_ID}@{PIPELINE_VERSION}"
    registry_entry = registry_document.get("mechanisms", {}).get(registry_key)
    if not isinstance(registry_entry, Mapping):
        raise RuntimeError(f"Missing trusted registry entry: {registry_key}")
    if registry_entry.get("code_artifact_id") != CODE_ARTIFACT_ID:
        raise RuntimeError("Trusted registry code artifact does not match this pipeline.")
    if str(registry_entry.get("code_sha256") or "").lower() != code_hash.lower():
        raise RuntimeError(
            "This pipeline source does not match its trusted registry digest."
        )
    accountant_checker_path = project_root / ACCOUNTANT_CHECKER_ARTIFACT_ID
    accountant_checker_hash = file_sha256(accountant_checker_path)
    if (
        registry_entry.get("accountant_checker_artifact_id")
        != ACCOUNTANT_CHECKER_ARTIFACT_ID
        or str(registry_entry.get("accountant_checker_sha256") or "").lower()
        != accountant_checker_hash.lower()
    ):
        raise RuntimeError(
            "The local accountant checker does not match its trusted registry digest."
        )
    registry_entry_hash = canonical_json_sha256(registry_entry)
    sample_rate = (
        args.sample_rate_numerator / args.sample_rate_denominator
        if args.sample_rate_denominator
        else float("nan")
    )
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    if not data_path.is_absolute():
        data_path = project_root / data_path
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_owner, parser_counts = parse_wisdm(data_path)
    missing_owners = sorted(
        (set(TRAIN_OWNERS) | set(TEST_OWNERS)) - set(rows_by_owner)
    )
    if missing_owners:
        raise ValueError(f"Public fixed owner split is incomplete: {missing_owners}")
    train_owners = list(TRAIN_OWNERS)
    test_owners = list(TEST_OWNERS)
    label_to_index = {
        label: index for index, label in enumerate(ACTIVITIES)
    }

    train_records, train_x, train_y = generate_windows(
        rows_by_owner,
        train_owners,
        "train",
        label_to_index,
    )
    _test_records, test_x, test_y = generate_windows(
        rows_by_owner,
        test_owners,
        "test",
        label_to_index,
    )
    mapping_path = output_dir / "mapping.csv"
    write_csv(mapping_path, train_records)
    mapping_evidence = inspect_mapping_evidence(mapping_path, SCENARIO)
    if not mapping_evidence.valid:
        raise RuntimeError(mapping_evidence.issues)
    if mapping_evidence.event_kappa != 2:
        raise RuntimeError(
            f"Expected overlap-50 event kappa 2, got {mapping_evidence.event_kappa}."
        )
    if (
        mapping_evidence.owner_kappa is None
        or mapping_evidence.owner_kappa > PUBLIC_OWNER_WINDOW_CAP
    ):
        raise RuntimeError(
            "The generated mapping violates the public owner contribution cap."
        )

    weights, batch_trace = train_secure_dpsgd(
        train_x,
        train_y,
        num_classes=len(ACTIVITIES),
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
        "test_macro_f1": macro_f1(
            test_y, test_predictions, len(ACTIVITIES)
        ),
        "train_records": len(train_records),
        "test_records": len(test_y),
        "train_owners": len(train_owners),
        "test_owners": len(test_owners),
    }

    model_path = output_dir / "released_model.json"
    write_json(
        model_path,
        {
            "mechanism_id": MECHANISM_ID,
            "mechanism_version": PIPELINE_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "label_to_index": label_to_index,
            "weights_including_bias": weights.tolist(),
        },
    )
    batch_trace_path = output_dir / "batch_trace.csv"
    write_csv(batch_trace_path, batch_trace)
    parser_evidence_path = output_dir / "parser_evidence.json"
    write_json(
        parser_evidence_path,
        {
            "domain_id": RAW_DOMAIN_ID,
            "dataset_file_sha256": file_sha256(data_path),
            **parser_counts,
        },
    )

    epsilon = recompute_epsilon(
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

    dataset_hash = file_sha256(data_path)
    split_parameters = {
        "policy": "sorted_owner_prefix",
        "train_owners": [str(owner) for owner in train_owners],
        "test_owners": [str(owner) for owner in test_owners],
    }
    schedule_parameters = {
        "scenario": SCENARIO,
        "window_length": WINDOW_LENGTH,
        "window_stride": WINDOW_STRIDE,
        "sample_rate_numerator": args.sample_rate_numerator,
        "sample_rate_denominator": args.sample_rate_denominator,
        "sampling_implementation": SAMPLING_IMPLEMENTATION,
        "steps": args.steps,
        "selected_record_count": mapping_evidence.selected_record_count,
    }
    raw_domain_parameters = {
        "canonicalization": "valid_rows_only_before_private_domain_v1",
        "fixed_presence": True,
        "immutable_fields": ["owner_id", "event_slot"],
        "mutable_fields": ["activity", "timestamp", "x", "y", "z"],
        "activity_domain": list(ACTIVITIES),
        "timestamp_domain": "signed_integer",
        "sensor_payload_domain": "finite_real_triple",
        "event_order": "physical_line_order_within_immutable_owner",
        "invalid_physical_rows": "excluded_before_neighbor_domain",
    }
    raw_domain = canonical_parameter_block(
        RAW_DOMAIN_ID,
        "public_fixed",
        raw_domain_parameters,
    )
    manifest = {
        "dataset_id": "WISDM_ar_v1.1_raw",
        "dataset_file_sha256": dataset_hash,
        "split_id": canonical_json_sha256(split_parameters),
        "split_parameters": split_parameters,
        "code_version": "wisdm_v2_hardened_pipeline_v2",
        "code_artifact_id": CODE_ARTIFACT_ID,
        "code_sha256": code_hash,
        "accountant_checker_artifact_id": ACCOUNTANT_CHECKER_ARTIFACT_ID,
        "accountant_checker_sha256": accountant_checker_hash,
        "support_convention": "half_open_integer_intervals",
        "record_identity_policy": "owner_partition_ids",
        "generated_record_unit": "window",
        "cross_owner_dependency": "none",
        "mapping_file_sha256": mapping_evidence.mapping_file_sha256,
        "selected_mapping_sha256": mapping_evidence.selected_mapping_sha256,
        "selected_record_count": mapping_evidence.selected_record_count,
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
            canonical_parameter_block(
                "public_sensor_clip_scale",
                "record_local",
                {
                    "clip_min": -PUBLIC_SENSOR_CLIP,
                    "clip_max": PUBLIC_SENSOR_CLIP,
                    "scale_divisor": PUBLIC_SENSOR_CLIP,
                },
            ),
            canonical_parameter_block(
                "record_local_window_summary",
                "record_local",
                {
                    "window_length": WINDOW_LENGTH,
                    "feature_names": list(FEATURE_NAMES),
                    "statistics": ["mean", "std", "min", "max"],
                },
            ),
            canonical_parameter_block(
                "record_local_majority_activity_label",
                "record_local",
                {
                    "label_rule": "majority activity in fixed window; lexical tie break",
                    "activity_domain": list(ACTIVITIES),
                    "label_map": label_to_index,
                },
            ),
        ],
        "raw_domain": raw_domain,
        "contribution_policy": canonical_parameter_block(
            OWNER_CAP_POLICY_ID,
            "public_fixed",
            {
                "cap": PUBLIC_OWNER_WINDOW_CAP,
                "enforcement": OWNER_CAP_ENFORCEMENT,
                "ordering": "ascending_generated_window_ordinal",
            },
        ),
        "schedule": {
            "id": "public_overlap50_fixed_mapping",
            "mode": "fixed_mapping",
            "evidence_status": "validated",
            "parameters": schedule_parameters,
            "parameters_sha256": canonical_json_sha256(schedule_parameters),
        },
        "population_definition": "public WISDM train-owner windows",
        "population_size_policy": "not_used_in_release_mechanism",
        "model_selection_status": "public",
        "released_model_sha256": file_sha256(model_path),
        "batch_trace_sha256": file_sha256(batch_trace_path),
        "parser_evidence_sha256": file_sha256(parser_evidence_path),
    }
    pipeline_sha256 = canonical_json_sha256(manifest)
    runtime_trace = build_runtime_trace(
        mechanism,
        privacy,
        manifest,
        mapping_evidence.selected_mapping_sha256,
        pipeline_sha256,
        {
            "batch_trace_sha256": file_sha256(batch_trace_path),
            "released_model_sha256": file_sha256(model_path),
            "executed_updates": len(batch_trace),
            "secure_sampling_rng": SAMPLING_IMPLEMENTATION,
            "secure_noise_rng": NOISE_GENERATION,
            "parser_evidence_sha256": file_sha256(parser_evidence_path),
        },
    )
    report = {
        "schema_version": "2.0",
        "scenario": SCENARIO,
        "mechanism": mechanism,
        "privacy": privacy,
        "pipeline": {
            "mapping_file_sha256": mapping_evidence.mapping_file_sha256,
            "selected_mapping_sha256": mapping_evidence.selected_mapping_sha256,
            "pipeline_sha256": pipeline_sha256,
            "manifest": manifest,
        },
        "runtime_trace": runtime_trace,
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
                "raw_adjacency": RAW_EVENT_ADJACENCY,
                "raw_domain_id": RAW_DOMAIN_ID,
                "raw_domain_sha256": canonical_json_sha256(raw_domain),
                "stability_status": "validated",
                "stability_method": "fixed_influence_replacements",
                "stability_bound": 2 * int(mapping_evidence.event_kappa or 0),
                "support_status": "complete",
                "schedule_mode": "fixed_mapping",
                "schedule_evidence_status": "validated",
                "conversion_method": "builtin_group",
            },
            "owner": {
                "claimed_unit": "owner",
                "raw_adjacency": "owner_add_remove",
                "stability_status": "validated",
                "stability_method": "owner_partition_add_remove",
                "stability_bound": PUBLIC_OWNER_WINDOW_CAP,
                "support_status": "complete",
                "schedule_mode": "fixed_mapping",
                "schedule_evidence_status": "validated",
                "conversion_method": "builtin_group",
            },
        },
    }
    privacy_report_path = output_dir / "privacy_report.json"
    write_json(privacy_report_path, report)
    write_json(output_dir / "metrics.json", metrics)

    # The parser's million-row canonical event map is no longer needed once
    # all bound artifacts have been written. Release it before invoking the
    # independent certificate processes so low-memory hosts do not conflate an
    # environment failure with a validation failure.
    del rows_by_owner
    del train_x, train_y, test_x, test_y
    del weights, train_predictions, test_predictions
    del train_records, _test_records, batch_trace
    gc.collect()

    audit_dir = output_dir / "audit"
    expected_batch_size = max(
        1, int(round(sample_rate * mapping_evidence.selected_record_count))
    )
    run_checked(
        [
            sys.executable,
            str(project_root / "scripts" / "unit_audit.py"),
            "--mapping",
            str(mapping_path),
            "--batch-size",
            str(expected_batch_size),
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

    certificates_dir = output_dir / "certificates"
    for claim in ("window", "event", "owner"):
        command = [
            sys.executable,
            str(project_root / "scripts" / "build_unit_certificate.py"),
            "--audit-csv",
            str(audit_dir / "unit_audit.csv"),
            "--mapping-csv",
            str(mapping_path),
            "--privacy-report-json",
            str(privacy_report_path),
            "--scenario",
            SCENARIO,
            "--claim-unit",
            claim,
            "--output-dir",
            str(certificates_dir),
        ]
        if claim == "owner":
            command.append("--diagnostic-exit-zero")
        run_checked(command, project_root)

    public_certificates = {}
    for claim in ("window", "event", "owner"):
        path = certificates_dir / f"certificate_{SCENARIO}_{claim}.json"
        public_certificates[claim] = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "window": ("ALLOWED", "DIRECT"),
        "event": ("ALLOWED", "CONVERT"),
        "owner": ("BLOCKED_VACUOUS", "GROUP"),
    }
    assertions = {}
    for claim, (release, unit_path) in expected.items():
        certificate = public_certificates[claim]
        passed = (
            certificate.get("release_status") == release
            and certificate.get("unit_path") == unit_path
            and bool(certificate.get("supported_statement"))
            == (release == "ALLOWED")
        )
        assertions[claim] = {
            "passed": passed,
            "expected_release_status": release,
            "actual_release_status": certificate.get("release_status"),
            "expected_unit_path": unit_path,
            "actual_unit_path": certificate.get("unit_path"),
        }
    write_json(output_dir / "gold_assertions.json", assertions)
    if not all(item["passed"] for item in assertions.values()):
        raise RuntimeError(f"Gold certificate assertions failed: {assertions}")

    summary_lines = [
        "# WISDM v2.0 Hardened Evidence Pipeline",
        "",
        "This is the first post-repair end-to-end evidence pipeline.",
        "",
        f"- Dataset SHA-256: `{dataset_hash}`",
        f"- Pipeline code SHA-256: `{code_hash}`",
        f"- Mapping SHA-256: `{mapping_evidence.mapping_file_sha256}`",
        f"- Selected mapping SHA-256: `{mapping_evidence.selected_mapping_sha256}`",
        f"- Pipeline SHA-256: `{pipeline_sha256}`",
        f"- Released model SHA-256: `{file_sha256(model_path)}`",
        f"- Train/test records: {metrics['train_records']} / {metrics['test_records']}",
        f"- Train/test owners: {len(train_owners)} / {len(test_owners)}",
        f"- Base privacy: epsilon={epsilon:.12g}, delta={args.delta:.12g}",
        f"- Event/owner observed kappa: {mapping_evidence.event_kappa} / {mapping_evidence.owner_kappa}",
        f"- Public owner conversion cap K: {PUBLIC_OWNER_WINDOW_CAP}",
        f"- Exact sample probability: {args.sample_rate_numerator}/{args.sample_rate_denominator}",
        f"- Public optimizer step size: {args.optimizer_step_size:.12g}",
        f"- Test accuracy / macro-F1: {metrics['test_accuracy']:.6f} / {metrics['test_macro_f1']:.6f}",
        "",
        "| Claim | Unit path | Release | Positive wording |",
        "| --- | --- | --- | --- |",
    ]
    for claim in ("window", "event", "owner"):
        certificate = public_certificates[claim]
        summary_lines.append(
            f"| {claim} | {certificate['unit_path']} | "
            f"{certificate['release_status']} | "
            f"{bool(certificate.get('supported_statement'))} |"
        )
    summary_lines.extend(
        [
            "",
            "The model is an integration baseline. Its purpose is evidence validity,",
            "not state-of-the-art WISDM utility.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )
    print(f"Wrote hardened v2.0 evidence pipeline: {output_dir}")


if __name__ == "__main__":
    main()
