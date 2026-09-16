#!/usr/bin/env python3
"""Independent-patient, eight-perturbation confirmation of two-axis balance."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import os
import platform
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from run_isic_sd21_gradient_smoke import (
    ALLOW_PATTERNS,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_VARIANT,
    cuda_memory,
    disable_broken_optional_onnx,
    encode_patient_inputs,
    package_version,
    sha256_file,
    stable_seed,
    verify_snapshot,
)
from run_isic_sd21_multipatient_diagnostic import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CLIP_NORMS,
    IMAGE_SIZE,
    LORA_RANK,
    bootstrap_ratio,
    compute_patient_gram,
    enumerate_uniform_weights,
    exact_estimator_metrics,
    load_gradient_model,
    write_csv,
)


SCHEMA = "pcm-isic-sd21-two-axis-confirmatory/v1"
PROTOCOL = "ISIC_SD21_TWO_AXIS_CONFIRMATORY_PROTOCOL_V1.md"
SELECTION_SALT = "pcm-isic-sd21-two-axis-confirm-v1"
PATIENTS_PER_SITE_GROUP = 8
PERTURBATION_BINS = 4
PERTURBATIONS_PER_BIN = 2
PERTURBATIONS = PERTURBATION_BINS * PERTURBATIONS_PER_BIN
EXCLUDED_PATIENTS = {
    "IP_0687884",
    "IP_9765850",
    "IP_6292815",
    "IP_3395413",
    "IP_7375528",
    "IP_8111876",
    "IP_5438943",
    "IP_4898383",
    "IP_8575171",
    "IP_2962626",
    "IP_2752722",
    "IP_2661172",
    "IP_2160700",
    "IP_4157535",
    "IP_9557980",
    "IP_0301948",
    "IP_5440659",
}


def selection_hash(site_group: str, patient_id: str) -> str:
    return hashlib.sha256(
        f"{SELECTION_SALT}|{site_group}|{patient_id}".encode("utf-8")
    ).hexdigest().upper()


def select_confirmatory_patients(
    records: Iterable[Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        grouped[record.patient_id].append(record)

    candidates: dict[str, list[dict[str, Any]]] = {"low_site": [], "high_site": []}
    for patient_id, patient_records in grouped.items():
        if patient_id in EXCLUDED_PATIENTS:
            continue
        lesions = {record.lesion_id for record in patient_records}
        sites = {record.anatom_site for record in patient_records if record.anatom_site}
        if len(patient_records) != 5 or len(lesions) != 5:
            continue
        site_group = "high_site" if len(sites) >= 3 else "low_site"
        candidates[site_group].append(
            {
                "patient_id": patient_id,
                "site_group": site_group,
                "site_count": len(sites),
                "selection_hash": selection_hash(site_group, patient_id),
                "records": sorted(patient_records, key=lambda record: record.cap_rank),
            }
        )

    selected: list[dict[str, Any]] = []
    candidate_counts: dict[str, int] = {}
    for site_group in ("low_site", "high_site"):
        candidates[site_group].sort(
            key=lambda row: (row["selection_hash"], row["patient_id"])
        )
        candidate_counts[site_group] = len(candidates[site_group])
        if len(candidates[site_group]) < PATIENTS_PER_SITE_GROUP:
            raise RuntimeError(f"insufficient confirmatory candidates in {site_group}")
        selected.extend(candidates[site_group][:PATIENTS_PER_SITE_GROUP])
    evidence = {
        "selection_salt": SELECTION_SALT,
        "excluded_patients": sorted(EXCLUDED_PATIENTS),
        "eligibility": "exactly_5_images_and_5_distinct_lesions",
        "candidate_counts_after_exclusion": candidate_counts,
        "patients_per_site_group": PATIENTS_PER_SITE_GROUP,
        "selected_patients": len(selected),
    }
    return selected, evidence


def confirmatory_specs(
    scheduler: Any, bank_tag: str = "original_confirmatory"
) -> list[dict[str, int]]:
    timesteps = int(scheduler.config.num_train_timesteps)
    edges = np.linspace(0, timesteps, PERTURBATION_BINS + 1, dtype=int)
    rows: list[dict[str, int]] = []
    for bin_index in range(PERTURBATION_BINS):
        for replicate in range(PERTURBATIONS_PER_BIN):
            if bank_tag == "original_confirmatory":
                timestep_seed = stable_seed(
                    "two_axis_confirm_timestep", bin_index, replicate
                )
                noise_seed = stable_seed(
                    "two_axis_confirm_noise", bin_index, replicate
                )
            else:
                timestep_seed = stable_seed(
                    "two_axis_replication_timestep", bank_tag, bin_index, replicate
                )
                noise_seed = stable_seed(
                    "two_axis_replication_noise", bank_tag, bin_index, replicate
                )
            rng = np.random.default_rng(timestep_seed)
            timestep = int(rng.integers(edges[bin_index], edges[bin_index + 1]))
            rows.append(
                {
                    "perturbation_index": len(rows),
                    "bin_index": bin_index,
                    "bin_replicate": replicate,
                    "timestep": timestep,
                    "timestep_bin_start": int(edges[bin_index]),
                    "timestep_bin_end_exclusive": int(edges[bin_index + 1]),
                    "noise_seed": noise_seed,
                }
            )
    return rows


def enumerate_bin_balanced_weights(
    records: int, specs: list[dict[str, int]]
) -> np.ndarray:
    groups: dict[int, list[int]] = defaultdict(list)
    for spec in specs:
        groups[int(spec["bin_index"])].append(int(spec["perturbation_index"]))
    bins = sorted(groups)
    if len(bins) != 4 or any(len(groups[bin_index]) != 2 for bin_index in bins):
        raise ValueError("expected four bins with two perturbations each")

    rows: list[np.ndarray] = []
    for record_choice in itertools.combinations(range(records), 4):
        for bin_assignment in itertools.permutations(bins):
            choices = [groups[bin_index] for bin_index in bin_assignment]
            for perturbation_assignment in itertools.product(*choices):
                weights = np.zeros(records * len(specs), dtype=np.float64)
                for record, perturbation in zip(
                    record_choice, perturbation_assignment
                ):
                    weights[record * len(specs) + perturbation] = 0.25
                rows.append(weights)
    return np.stack(rows)


def geometric_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.exp(np.mean(np.log(array))))


def build_markdown(result: dict[str, Any], patients: list[dict[str, Any]]) -> str:
    primary = result["primary"]
    lines = [
        "# ISIC SD 2.1 two-axis balance confirmatory diagnostic",
        "",
        f"- Execution status: **{result['status']}**",
        f"- Confirmatory decision: **{result['decision']}**",
        "- Cohort: 16 new patients, disjoint from smoke and first multi-patient cohort",
        "- Bank: 8 perturbations (2 per each of 4 timestep bins), 640 gradients",
        "- Scope: gradient diagnostic only; no optimizer update or DP noise",
        "",
        "## Frozen primary gate",
        "",
        f"- Bin-balanced/replacement geometric MSE ratio: `{primary['geometric_mean_ratio']:.4f}`",
        f"- Patient bootstrap 95% interval: `[{primary['bootstrap_ci95_lower']:.4f}, "
        f"{primary['bootstrap_ci95_upper']:.4f}]`",
        f"- Bin-balanced wins: `{primary['balanced_wins']}/16`",
        "",
    ]
    for condition in result["conditions"]:
        lines.append(
            f"- `{condition['name']}`: **{condition['status']}** "
            f"(observed={condition['observed']}, required={condition['required']})"
        )
    lines.extend(
        [
            "",
            "## Patient-paired result",
            "",
            "| Patient | Site group | Balanced/replacement MSE ratio |",
            "|---|---|---:|",
        ]
    )
    for row in patients:
        lines.append(
            f"| {row['patient_id']} | {row['site_group']} | "
            f"{row['balanced_to_replacement_ratio']:.4f} |"
        )
    lines.extend(
        [
            "",
            "A PASS only advances the exact joint record/timestep-bin sampling rule to prior-work "
            "review and X-ray validation. It is not a DP, training-utility, clinical, generation, "
            "novelty, or venue-acceptance result.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    disable_broken_optional_onnx()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    import torch
    from huggingface_hub import snapshot_download

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = args.protocol_file.resolve()
    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=ALLOW_PATTERNS,
            local_files_only=True,
        )
    ).resolve()
    critical_hash_checks = verify_snapshot(snapshot)

    data_pipeline_dir = Path(__file__).resolve().parents[1] / "data_pipeline"
    sys.path.insert(0, str(data_pipeline_dir))
    from isic_manifest_dataset import read_manifest  # pylint: disable=import-error,import-outside-toplevel

    all_records = read_manifest(args.manifest, partitions={"public_development"})
    selected, selection_evidence = select_confirmatory_patients(all_records)
    flattened_records = [record for patient in selected for record in patient["records"]]
    latents, encoder_states, _, record_rows = encode_patient_inputs(
        snapshot, flattened_records, args.image_root.resolve(), IMAGE_SIZE
    )
    patient_slices: dict[str, slice] = {}
    cursor = 0
    for patient in selected:
        patient_slices[patient["patient_id"]] = slice(cursor, cursor + 5)
        for local_index in range(5):
            row = record_rows[cursor + local_index]
            row["patient_id"] = patient["patient_id"]
            row["site_group"] = patient["site_group"]
            row["patient_site_count"] = patient["site_count"]
            row["patient_selection_hash"] = patient["selection_hash"]
        cursor += 5

    unet, scheduler, trainable, model_metadata = load_gradient_model(snapshot)
    specs = confirmatory_specs(scheduler, args.bank_tag)
    gradient_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    grams: dict[str, np.ndarray] = {}
    overall_started = time.perf_counter()
    for patient_index, patient in enumerate(selected, start=1):
        patient_id = patient["patient_id"]
        patient_slice = patient_slices[patient_id]
        gram, evaluations, runtime = compute_patient_gram(
            patient_id=patient_id,
            latents=latents[patient_slice],
            encoder_states=encoder_states[patient_slice],
            unet=unet,
            scheduler=scheduler,
            trainable=trainable,
            parameter_dimension=model_metadata["parameter_dimension"],
            specs=specs,
        )
        grams[patient_id] = gram
        gradient_rows.extend(evaluations)
        runtime_rows.append(runtime)
        print(
            json.dumps(
                {
                    "progress": "confirmatory_patient_gradient_complete",
                    "patient_index": patient_index,
                    "patients_total": len(selected),
                    "patient_id": patient_id,
                    "elapsed_sec": runtime["elapsed_sec"],
                }
            ),
            flush=True,
        )
    torch.cuda.synchronize()
    gradient_elapsed = round(time.perf_counter() - overall_started, 4)
    gradient_cuda = cuda_memory(torch)
    del unet
    del scheduler
    del trainable
    gc.collect()
    torch.cuda.empty_cache()

    np.savez_compressed(
        output_dir / "patient_gradient_grams_local_only.npz",
        **{patient_id: gram for patient_id, gram in grams.items()},
    )
    write_csv(output_dir / "selected_patient_records.csv", record_rows)
    write_csv(output_dir / "gradient_evaluations.csv", gradient_rows)
    write_csv(output_dir / "patient_runtime.csv", runtime_rows)
    write_csv(output_dir / "perturbation_specs.csv", specs)

    replacement_weights = enumerate_uniform_weights(5, PERTURBATIONS, 4, 1)
    balanced_weights = enumerate_bin_balanced_weights(5, specs)
    reference = np.full(5 * PERTURBATIONS, 1.0 / (5 * PERTURBATIONS))
    maximum_expected_weight_error = max(
        float(np.max(np.abs(np.mean(replacement_weights, axis=0) - reference))),
        float(np.max(np.abs(np.mean(balanced_weights, axis=0) - reference))),
    )
    maximum_row_sum_error = max(
        float(np.max(np.abs(np.sum(replacement_weights, axis=1) - 1.0))),
        float(np.max(np.abs(np.sum(balanced_weights, axis=1) - 1.0))),
    )

    metric_rows: list[dict[str, Any]] = []
    clipping_rows: list[dict[str, Any]] = []
    gram_psd_pass = True
    for patient in selected:
        patient_id = patient["patient_id"]
        gram = grams[patient_id]
        for method, weights in (
            ("replacement_uniform_m4", replacement_weights),
            ("bin_balanced_m4", balanced_weights),
        ):
            metrics, clipping = exact_estimator_metrics(
                weights, gram, model_metadata["parameter_dimension"], CLIP_NORMS
            )
            metric_rows.append(
                {
                    "patient_id": patient_id,
                    "site_group": patient["site_group"],
                    "method": method,
                    **metrics,
                }
            )
            for row in clipping:
                clipping_rows.append(
                    {
                        "patient_id": patient_id,
                        "site_group": patient["site_group"],
                        "method": method,
                        **row,
                    }
                )
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(gram)))
        tolerance = max(1e-6, float(np.max(np.diag(gram))) * 1e-5)
        gram_psd_pass = gram_psd_pass and minimum_eigenvalue >= -tolerance
    write_csv(output_dir / "patient_preclip_metrics.csv", metric_rows)
    write_csv(output_dir / "patient_clipping_metrics.csv", clipping_rows)

    preclip = {
        (row["patient_id"], row["method"]): float(row["preclip_mse"])
        for row in metric_rows
    }
    patient_rows: list[dict[str, Any]] = []
    ratios: list[float] = []
    for patient in selected:
        patient_id = patient["patient_id"]
        replacement = preclip[(patient_id, "replacement_uniform_m4")]
        balanced = preclip[(patient_id, "bin_balanced_m4")]
        ratio = balanced / replacement
        ratios.append(ratio)
        patient_rows.append(
            {
                "patient_id": patient_id,
                "site_group": patient["site_group"],
                "site_count": patient["site_count"],
                "replacement_preclip_mse": replacement,
                "balanced_preclip_mse": balanced,
                "balanced_to_replacement_ratio": ratio,
            }
        )
    write_csv(output_dir / "patient_primary_comparison.csv", patient_rows)
    bootstrap = bootstrap_ratio(np.asarray(ratios, dtype=np.float64))
    balanced_wins = int(sum(value < 1.0 for value in ratios))
    conditions = [
        {
            "name": "geometric_mean_ratio_le_0.90",
            "status": "PASS"
            if bootstrap["geometric_mean_ratio"] <= 0.90
            else "FAIL",
            "observed": bootstrap["geometric_mean_ratio"],
            "required": "<=0.90",
        },
        {
            "name": "bootstrap_upper_lt_1.00",
            "status": "PASS"
            if bootstrap["bootstrap_ci95_upper"] < 1.0
            else "FAIL",
            "observed": bootstrap["bootstrap_ci95_upper"],
            "required": "<1.00",
        },
        {
            "name": "balanced_wins_ge_12_of_16",
            "status": "PASS" if balanced_wins >= 12 else "FAIL",
            "observed": balanced_wins,
            "required": ">=12",
        },
    ]
    decision = (
        "TWO_AXIS_BALANCE_CONFIRMED_FOR_METHOD_REVIEW"
        if all(row["status"] == "PASS" for row in conditions)
        else "RETIRE_CURRENT_TWO_AXIS_BALANCE"
    )

    site_group_summary: list[dict[str, Any]] = []
    for site_group in ("low_site", "high_site"):
        values = [
            row["balanced_to_replacement_ratio"]
            for row in patient_rows
            if row["site_group"] == site_group
        ]
        site_group_summary.append(
            {
                "site_group": site_group,
                "patients": len(values),
                "geometric_mean_ratio": geometric_mean(values),
                "balanced_wins": int(sum(value < 1.0 for value in values)),
            }
        )

    clipping_summary: list[dict[str, Any]] = []
    for clip_norm in CLIP_NORMS:
        postclip = {
            (row["patient_id"], row["method"]): float(row["postclip_mse"])
            for row in clipping_rows
            if float(row["clip_norm"]) == clip_norm
        }
        values = [
            postclip[(patient["patient_id"], "bin_balanced_m4")]
            / postclip[(patient["patient_id"], "replacement_uniform_m4")]
            for patient in selected
        ]
        balanced_source_rows = [
            row
            for row in clipping_rows
            if row["method"] == "bin_balanced_m4"
            and float(row["clip_norm"]) == clip_norm
        ]
        clipping_summary.append(
            {
                "clip_norm": clip_norm,
                "geometric_mean_ratio": geometric_mean(values),
                "balanced_wins": int(sum(value < 1.0 for value in values)),
                "mean_balanced_clip_rate": float(
                    np.mean([float(row["clip_rate"]) for row in balanced_source_rows])
                ),
            }
        )

    bin_counts = {
        bin_index: sum(int(spec["bin_index"]) == bin_index for spec in specs)
        for bin_index in range(PERTURBATION_BINS)
    }
    execution_checks = [
        {
            "name": "critical_model_hashes",
            "status": "PASS"
            if all(row["status"] == "PASS" for row in critical_hash_checks.values())
            else "FAIL",
        },
        {
            "name": "new_balanced_16_patient_selection",
            "status": "PASS"
            if len(selected) == 16
            and not ({row["patient_id"] for row in selected} & EXCLUDED_PATIENTS)
            and sum(row["site_group"] == "low_site" for row in selected) == 8
            and sum(row["site_group"] == "high_site" for row in selected) == 8
            else "FAIL",
        },
        {
            "name": "eight_bank_two_per_bin",
            "status": "PASS"
            if len(specs) == 8 and set(bin_counts.values()) == {2}
            else "FAIL",
            "bin_counts": bin_counts,
        },
        {
            "name": "all_640_gradients_finite",
            "status": "PASS"
            if len(gradient_rows) == 640 and all(row["finite"] for row in gradient_rows)
            else "FAIL",
        },
        {
            "name": "exact_enumeration_sizes",
            "status": "PASS"
            if replacement_weights.shape == (20480, 40)
            and balanced_weights.shape == (1920, 40)
            else "FAIL",
        },
        {
            "name": "both_estimators_unbiased",
            "status": "PASS"
            if maximum_expected_weight_error <= 1e-12
            and maximum_row_sum_error <= 1e-12
            else "FAIL",
            "maximum_expected_weight_error": maximum_expected_weight_error,
            "maximum_row_sum_error": maximum_row_sum_error,
        },
        {
            "name": "all_gram_psd_tolerance",
            "status": "PASS" if gram_psd_pass else "FAIL",
        },
    ]
    status = (
        "PASS"
        if all(row["status"] == "PASS" for row in execution_checks)
        else "FAIL"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "privacy_status": "DIAGNOSTIC_NOT_DP",
        "decision": decision,
        "protocol": {"file": protocol_path.name, "sha256": sha256_file(protocol_path)},
        "input_manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": sha256_file(args.manifest.resolve()),
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "variant": MODEL_VARIANT,
            "critical_hash_checks": critical_hash_checks,
            **model_metadata,
        },
        "selection": {
            **selection_evidence,
            "selected": [
                {
                    key: patient[key]
                    for key in ("patient_id", "site_group", "site_count", "selection_hash")
                }
                for patient in selected
            ],
        },
        "configuration": {
            "image_size": IMAGE_SIZE,
            "rank": LORA_RANK,
            "q": 4,
            "perturbations": PERTURBATIONS,
            "perturbation_bins": PERTURBATION_BINS,
            "perturbations_per_bin": PERTURBATIONS_PER_BIN,
            "replacement_states": int(replacement_weights.shape[0]),
            "balanced_states": int(balanced_weights.shape[0]),
            "clip_norms": list(CLIP_NORMS),
            "common_random_number_across_records": True,
            "bank_tag": args.bank_tag,
        },
        "gradient_execution": {
            "patients": len(selected),
            "total_gradients": len(gradient_rows),
            "elapsed_sec": gradient_elapsed,
            "patient_elapsed_sec_sum": float(sum(row["elapsed_sec"] for row in runtime_rows)),
            "cuda": gradient_cuda,
            "raw_gradient_saved": False,
            "gram_release_status": "LOCAL_ONLY",
        },
        "execution_checks": execution_checks,
        "primary": {
            **bootstrap,
            "balanced_wins": balanced_wins,
            "patients": len(ratios),
        },
        "conditions": conditions,
        "site_group_summary": site_group_summary,
        "clipping_summary": clipping_summary,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "packages": {
                name: package_version(name)
                for name in ("diffusers", "transformers", "peft", "safetensors")
            },
        },
        "artifacts": {
            "selected_patient_records": "selected_patient_records.csv",
            "gradient_evaluations": "gradient_evaluations.csv",
            "patient_runtime": "patient_runtime.csv",
            "perturbation_specs": "perturbation_specs.csv",
            "patient_gradient_grams": "patient_gradient_grams_local_only.npz",
            "patient_preclip_metrics": "patient_preclip_metrics.csv",
            "patient_clipping_metrics": "patient_clipping_metrics.csv",
            "patient_primary_comparison": "patient_primary_comparison.csv",
            "summary": "summary.md",
        },
        "interpretation_limit": (
            "Independent-patient finite-bank gradient diagnostic only; no optimizer update, "
            "DP noise, generation, utility, clinical, novelty, or venue-acceptance claim."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        build_markdown(result, patient_rows), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bank-tag", default="original_confirmatory")
    parser.add_argument(
        "--protocol-file", type=Path, default=Path(__file__).with_name(PROTOCOL)
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = run(args)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        failure = {
            "schema": SCHEMA,
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        args.output_dir.resolve().mkdir(parents=True, exist_ok=True)
        (args.output_dir.resolve() / "failure.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        print(json.dumps(failure, indent=2))
        raise


if __name__ == "__main__":
    main()
