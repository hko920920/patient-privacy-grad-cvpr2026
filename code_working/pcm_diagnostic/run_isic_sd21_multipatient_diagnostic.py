#!/usr/bin/env python3
"""Preregistered 16-patient ISIC SD 2.1 allocation diagnostic.

No optimizer update and no DP noise are used.  This is a public-development
gradient diagnostic, not a private training or medical-utility experiment.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import math
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
    farthest_first_strata,
    package_version,
    sha256_file,
    stable_seed,
    verify_snapshot,
)


SCHEMA = "pcm-isic-sd21-multipatient-diagnostic/v1"
SELECTION_SALT = "pcm-isic-sd21-multipatient-v1"
SMOKE_PATIENT_EXCLUSION = "IP_0687884"
PATIENTS_PER_SITE_GROUP = 8
IMAGE_SIZE = 256
LORA_RANK = 8
REFERENCE_PERTURBATIONS = 4
Q = 4
CLIP_NORMS = (0.05, 0.10, 0.20, 0.50, 1.00)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 260902


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def selection_hash(site_group: str, patient_id: str) -> str:
    return hashlib.sha256(
        f"{SELECTION_SALT}|{site_group}|{patient_id}".encode("utf-8")
    ).hexdigest().upper()


def select_balanced_patients(
    records: Iterable[Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        grouped[record.patient_id].append(record)

    candidates: dict[str, list[dict[str, Any]]] = {"low_site": [], "high_site": []}
    for patient_id, patient_records in grouped.items():
        if patient_id == SMOKE_PATIENT_EXCLUSION:
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
                "selection_hash": selection_hash(site_group, patient_id),
                "site_count": len(sites),
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
            raise RuntimeError(f"insufficient candidates in {site_group}")
        selected.extend(candidates[site_group][:PATIENTS_PER_SITE_GROUP])

    evidence = {
        "selection_salt": SELECTION_SALT,
        "smoke_patient_exclusion": SMOKE_PATIENT_EXCLUSION,
        "eligibility": "exactly_5_images_and_5_distinct_lesions",
        "candidate_counts": candidate_counts,
        "patients_per_site_group": PATIENTS_PER_SITE_GROUP,
        "selected_patients": len(selected),
    }
    return selected, evidence


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        average = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average
        start = end
    return ranks


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Spearman inputs must be same-shape vectors")
    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    left_centered = left_rank - np.mean(left_rank)
    right_centered = right_rank - np.mean(right_rank)
    denominator = float(
        np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    )
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def enumerate_uniform_weights(
    records: int, perturbations: int, m: int, r: int
) -> np.ndarray:
    if m * r != Q:
        raise ValueError("allocation must satisfy m*r=Q")
    rows: list[np.ndarray] = []
    for record_choice in itertools.combinations(range(records), m):
        for perturbation_choice in itertools.product(
            range(perturbations), repeat=m * r
        ):
            weights = np.zeros(records * perturbations, dtype=np.float64)
            cursor = 0
            for record in record_choice:
                for _ in range(r):
                    perturbation = perturbation_choice[cursor]
                    cursor += 1
                    weights[record * perturbations + perturbation] += 1.0 / (m * r)
            rows.append(weights)
    return np.stack(rows)


def enumerate_stratified_weights(
    labels: np.ndarray, perturbations: int, r: int
) -> np.ndarray:
    strata = np.unique(labels)
    m = len(strata)
    if m * r != Q:
        raise ValueError("stratified allocation must satisfy m*r=Q")
    groups = [np.flatnonzero(labels == stratum) for stratum in strata]
    rows: list[np.ndarray] = []
    for record_choice in itertools.product(*groups):
        for perturbation_choice in itertools.product(
            range(perturbations), repeat=m * r
        ):
            weights = np.zeros(len(labels) * perturbations, dtype=np.float64)
            cursor = 0
            for group, record in zip(groups, record_choice):
                record_weight = len(group) / len(labels)
                for _ in range(r):
                    perturbation = perturbation_choice[cursor]
                    cursor += 1
                    weights[int(record) * perturbations + perturbation] += (
                        record_weight / r
                    )
            rows.append(weights)
    return np.stack(rows)


def quadratic_rows(weights: np.ndarray, gram: np.ndarray) -> np.ndarray:
    return np.maximum(
        0.0, np.einsum("bi,ij,bj->b", weights, gram, weights, optimize=True)
    )


def exact_estimator_metrics(
    weights: np.ndarray,
    gram: np.ndarray,
    parameter_dimension: int,
    clip_norms: tuple[float, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reference = np.full(gram.shape[0], 1.0 / gram.shape[0], dtype=np.float64)
    expected_weight_error = float(np.max(np.abs(np.mean(weights, axis=0) - reference)))
    row_sum_error = float(np.max(np.abs(np.sum(weights, axis=1) - 1.0)))
    differences = weights - reference
    preclip = quadratic_rows(differences, gram) / parameter_dimension
    estimate_norm_sq = quadratic_rows(weights, gram)
    estimate_norm = np.sqrt(estimate_norm_sq)
    reference_norm_sq = max(0.0, float(reference @ gram @ reference))
    reference_norm = math.sqrt(reference_norm_sq)
    dots = weights @ gram @ reference
    denominators = estimate_norm * reference_norm
    cosines = np.divide(
        dots,
        denominators,
        out=np.ones_like(dots),
        where=denominators > 0,
    )
    cosines = np.clip(cosines, -1.0, 1.0)
    primary = {
        "enumerated_states": int(weights.shape[0]),
        "max_row_weight_sum_error": row_sum_error,
        "max_expected_weight_error": expected_weight_error,
        "preclip_mse": float(np.mean(preclip)),
        "cosine_error": float(np.mean(1.0 - cosines)),
        "mean_estimate_norm": float(np.mean(estimate_norm)),
        "reference_norm": reference_norm,
    }

    clipping: list[dict[str, Any]] = []
    for clip_norm in clip_norms:
        estimate_scale = np.minimum(
            1.0,
            np.divide(
                clip_norm,
                estimate_norm,
                out=np.ones_like(estimate_norm),
                where=estimate_norm > 0,
            ),
        )
        reference_scale = (
            min(1.0, clip_norm / reference_norm) if reference_norm > 0 else 1.0
        )
        post_difference = (
            estimate_scale[:, None] * weights - reference_scale * reference[None, :]
        )
        displacement = (estimate_scale - 1.0)[:, None] * weights
        clipping.append(
            {
                "clip_norm": clip_norm,
                "postclip_mse": float(
                    np.mean(quadratic_rows(post_difference, gram) / parameter_dimension)
                ),
                "clip_displacement": float(
                    np.mean(quadratic_rows(displacement, gram) / parameter_dimension)
                ),
                "clip_rate": float(np.mean(estimate_norm > clip_norm)),
                "reference_clipped": int(reference_norm > clip_norm),
            }
        )
    return primary, clipping


def variance_decomposition(
    gram: np.ndarray, records: int, perturbations: int, parameter_dimension: int
) -> dict[str, float]:
    total_gradients = records * perturbations
    overall = np.full(total_gradients, 1.0 / total_gradients)
    record_means = np.zeros((records, total_gradients), dtype=np.float64)
    for record in range(records):
        start = record * perturbations
        record_means[record, start : start + perturbations] = 1.0 / perturbations
    basis = np.eye(total_gradients, dtype=np.float64)

    between = float(
        np.mean(quadratic_rows(record_means - overall, gram)) / parameter_dimension
    )
    within_terms = []
    total_terms = []
    for record in range(records):
        for perturbation in range(perturbations):
            index = record * perturbations + perturbation
            within_terms.append(basis[index] - record_means[record])
            total_terms.append(basis[index] - overall)
    within = float(
        np.mean(quadratic_rows(np.stack(within_terms), gram)) / parameter_dimension
    )
    total = float(
        np.mean(quadratic_rows(np.stack(total_terms), gram)) / parameter_dimension
    )
    residual = abs(total - between - within)
    return {
        "between_record_variance": between,
        "within_perturbation_variance": within,
        "total_variance": total,
        "decomposition_residual": residual,
        "between_fraction": between / total if total > 0 else 0.0,
    }


def feature_gradient_alignment(
    features: np.ndarray, gram: np.ndarray, perturbations: int
) -> dict[str, float]:
    records = features.shape[0]
    total = records * perturbations
    record_means = np.zeros((records, total), dtype=np.float64)
    for record in range(records):
        start = record * perturbations
        record_means[record, start : start + perturbations] = 1.0 / perturbations

    feature_distances = np.sum(
        np.square(features[:, None, :] - features[None, :, :]), axis=2
    )
    gradient_distances = np.zeros((records, records), dtype=np.float64)
    for left in range(records):
        for right in range(left + 1, records):
            difference = record_means[left] - record_means[right]
            distance = max(0.0, float(difference @ gram @ difference))
            gradient_distances[left, right] = distance
            gradient_distances[right, left] = distance
    upper = np.triu_indices(records, k=1)
    return {
        "feature_gradient_distance_spearman": spearman_correlation(
            feature_distances[upper], gradient_distances[upper]
        ),
        "mean_feature_distance": float(np.mean(feature_distances[upper])),
        "mean_gradient_distance_sq": float(np.mean(gradient_distances[upper])),
    }


def load_gradient_model(snapshot: Path) -> tuple[Any, Any, list[tuple[str, Any]], dict[str, Any]]:
    import torch
    from diffusers import DDPMScheduler, UNet2DConditionModel
    from peft import LoraConfig

    torch.manual_seed(260902)
    torch.cuda.manual_seed_all(260902)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    scheduler = DDPMScheduler.from_pretrained(snapshot / "scheduler", local_files_only=True)
    unet = UNet2DConditionModel.from_pretrained(
        snapshot / "unet",
        torch_dtype=torch.float16,
        variant=MODEL_VARIANT,
        use_safetensors=True,
        local_files_only=True,
    )
    unet.requires_grad_(False)
    unet.add_adapter(
        LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_RANK,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
    )
    unet.enable_gradient_checkpointing()
    unet.train().to("cuda")
    trainable = [
        (name, parameter)
        for name, parameter in unet.named_parameters()
        if parameter.requires_grad
    ]
    metadata = {
        "parameter_dimension": sum(parameter.numel() for _, parameter in trainable),
        "trainable_tensors": len(trainable),
        "prediction_type": str(scheduler.config.prediction_type),
    }
    return unet, scheduler, trainable, metadata


def perturbation_specs(scheduler: Any) -> list[dict[str, int]]:
    timesteps = int(scheduler.config.num_train_timesteps)
    edges = np.linspace(0, timesteps, REFERENCE_PERTURBATIONS + 1, dtype=int)
    rows: list[dict[str, int]] = []
    for perturbation in range(REFERENCE_PERTURBATIONS):
        timestep_seed = stable_seed("multipatient_timestep_bin", perturbation)
        rng = np.random.default_rng(timestep_seed)
        timestep = int(rng.integers(edges[perturbation], edges[perturbation + 1]))
        rows.append(
            {
                "perturbation_index": perturbation,
                "timestep": timestep,
                "timestep_bin_start": int(edges[perturbation]),
                "timestep_bin_end_exclusive": int(edges[perturbation + 1]),
                "noise_seed": stable_seed("multipatient_common_noise", perturbation),
            }
        )
    return rows


def compute_patient_gram(
    patient_id: str,
    latents: list[Any],
    encoder_states: list[Any],
    unet: Any,
    scheduler: Any,
    trainable: list[tuple[str, Any]],
    parameter_dimension: int,
    specs: list[dict[str, int]],
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    import torch
    import torch.nn.functional as functional

    perturbations = len(specs)
    expected_indices = list(range(perturbations))
    actual_indices = [int(spec["perturbation_index"]) for spec in specs]
    if actual_indices != expected_indices:
        raise ValueError(
            "perturbation indices must be contiguous and ordered from zero: "
            f"expected {expected_indices}, got {actual_indices}"
        )
    bank = np.empty(
        (len(latents) * perturbations, parameter_dimension),
        dtype=np.float32,
    )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for record_index, (latent_cpu, hidden_cpu) in enumerate(
        zip(latents, encoder_states)
    ):
        latent = latent_cpu.to(device="cuda", dtype=torch.float16)
        hidden = hidden_cpu.to(device="cuda", dtype=torch.float16)
        for spec in specs:
            perturbation = spec["perturbation_index"]
            timestep = torch.tensor(
                [spec["timestep"]], device="cuda", dtype=torch.long
            )
            generator = torch.Generator(device="cuda").manual_seed(spec["noise_seed"])
            noise = torch.randn(
                latent.shape,
                generator=generator,
                device="cuda",
                dtype=torch.float16,
            )
            noisy_latent = scheduler.add_noise(latent, noise, timestep)
            prediction_type = str(scheduler.config.prediction_type)
            if prediction_type == "epsilon":
                target = noise
            elif prediction_type == "v_prediction":
                target = scheduler.get_velocity(latent, noise, timestep)
            else:
                raise RuntimeError(f"unsupported prediction_type: {prediction_type}")

            for _, parameter in trainable:
                parameter.grad = None
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = unet(noisy_latent, timestep, hidden).sample
                loss = functional.mse_loss(prediction.float(), target.float())
            loss.backward()

            bank_index = record_index * perturbations + perturbation
            offset = 0
            finite = True
            for _, parameter in trainable:
                count = parameter.numel()
                if parameter.grad is None:
                    bank[bank_index, offset : offset + count] = 0.0
                else:
                    gradient = parameter.grad.detach().float().reshape(-1).cpu().numpy()
                    bank[bank_index, offset : offset + count] = gradient
                    finite = finite and bool(np.isfinite(gradient).all())
                offset += count
            if not finite:
                raise RuntimeError(f"non-finite gradient for {patient_id}")
            rows.append(
                {
                    "patient_id": patient_id,
                    "record_index": record_index,
                    "perturbation_index": perturbation,
                    "timestep": spec["timestep"],
                    "noise_seed": spec["noise_seed"],
                    "loss": float(loss.detach().cpu()),
                    "gradient_norm": float(np.linalg.norm(bank[bank_index])),
                    "finite": finite,
                }
            )
    torch.cuda.synchronize()
    gram = np.asarray(bank @ bank.T, dtype=np.float64)
    gram = 0.5 * (gram + gram.T)
    del bank
    metadata = {
        "patient_id": patient_id,
        "gradients": len(rows),
        "elapsed_sec": round(time.perf_counter() - started, 4),
        "minimum_eigenvalue": float(np.min(np.linalg.eigvalsh(gram))),
        "maximum_diagonal": float(np.max(np.diag(gram))),
    }
    return gram, rows, metadata


def bootstrap_ratio(ratios: np.ndarray) -> dict[str, float]:
    if np.any(ratios <= 0):
        raise ValueError("MSE ratios must be positive")
    logs = np.log(ratios)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0, len(ratios), size=(BOOTSTRAP_REPLICATES, len(ratios))
    )
    bootstrap_means = np.mean(logs[indices], axis=1)
    return {
        "geometric_mean_ratio": float(np.exp(np.mean(logs))),
        "bootstrap_ci95_lower": float(np.exp(np.quantile(bootstrap_means, 0.025))),
        "bootstrap_ci95_upper": float(np.exp(np.quantile(bootstrap_means, 0.975))),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def build_summary_markdown(
    result: dict[str, Any],
    method_summary: list[dict[str, Any]],
    patient_comparisons: list[dict[str, Any]],
) -> str:
    decision = result["method_gate"]["decision"]
    lines = [
        "# ISIC SD 2.1 multi-patient allocation diagnostic v1",
        "",
        f"- Execution status: **{result['status']}**",
        f"- Current VAE coverage decision: **{decision}**",
        f"- Patients: {result['selection']['selected_patients']} (8 low-site + 8 high-site)",
        f"- Gradients: {result['gradient_execution']['total_gradients']}",
        "- Scope: public-development gradient diagnostic; no optimizer update or DP noise",
        "",
        "## Method-level pre-clipping result",
        "",
        "| Method | Mean MSE | Median MSE | Geometric ratio to uniform m4 |",
        "|---|---:|---:|---:|",
    ]
    for row in method_summary:
        lines.append(
            f"| {row['method']} | {row['mean_preclip_mse']:.10e} | "
            f"{row['median_preclip_mse']:.10e} | "
            f"{row['geometric_ratio_to_uniform_m4']:.4f} |"
        )
    gate = result["method_gate"]
    lines.extend(
        [
            "",
            "## Frozen primary gate",
            "",
            f"- Coverage/uniform geometric-mean ratio: `{gate['geometric_mean_ratio']:.4f}`",
            f"- Patient-bootstrap 95% interval: "
            f"`[{gate['bootstrap_ci95_lower']:.4f}, {gate['bootstrap_ci95_upper']:.4f}]`",
            f"- Coverage wins: `{gate['coverage_wins']}/{gate['patients']}`",
            f"- Median feature-gradient Spearman: `{gate['median_feature_gradient_spearman']:.4f}`",
            "",
        ]
    )
    for condition in gate["conditions"]:
        lines.append(
            f"- `{condition['name']}`: **{condition['status']}** "
            f"(observed={condition['observed']}, required={condition['required']})"
        )
    lines.extend(
        [
            "",
            "## Patient-paired primary comparison",
            "",
            "| Patient | Site group | Coverage/uniform MSE ratio | Feature-gradient Spearman |",
            "|---|---|---:|---:|",
        ]
    )
    for row in patient_comparisons:
        lines.append(
            f"| {row['patient_id']} | {row['site_group']} | "
            f"{row['coverage_to_uniform_m4_ratio']:.4f} | "
            f"{row['feature_gradient_distance_spearman']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The frozen decision applies only to the current SD 2.1 VAE-feature coverage rule.",
            "It does not establish DP utility, generation quality, clinical utility, or novelty, and it",
            "does not reject the broader multi-record patient-gradient research question.",
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
    protocol_path = Path(__file__).with_name(
        "ISIC_SD21_MULTIPATIENT_DIAGNOSTIC_PROTOCOL_V1.md"
    )
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
    selected, selection_evidence = select_balanced_patients(all_records)
    flattened_records = [record for patient in selected for record in patient["records"]]
    latents, encoder_states, features, record_rows = encode_patient_inputs(
        snapshot,
        flattened_records,
        args.image_root.resolve(),
        IMAGE_SIZE,
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
    specs = perturbation_specs(scheduler)
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
                    "progress": "patient_gradient_complete",
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
    write_csv(output_dir / "gradient_evaluations.csv", gradient_rows)
    write_csv(output_dir / "patient_runtime.csv", runtime_rows)
    write_csv(output_dir / "perturbation_specs.csv", specs)

    uniform_weights = {
        "uniform_m1_r4": enumerate_uniform_weights(5, REFERENCE_PERTURBATIONS, 1, 4),
        "uniform_m2_r2": enumerate_uniform_weights(5, REFERENCE_PERTURBATIONS, 2, 2),
        "uniform_m4_r1": enumerate_uniform_weights(5, REFERENCE_PERTURBATIONS, 4, 1),
    }
    primary_rows: list[dict[str, Any]] = []
    clipping_rows: list[dict[str, Any]] = []
    variance_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    strata_rows: list[dict[str, Any]] = []
    max_unbiased_error = 0.0
    max_row_sum_error = 0.0
    max_decomposition_residual_ratio = 0.0
    all_strata_nonempty = True
    gram_psd_pass = True

    for patient in selected:
        patient_id = patient["patient_id"]
        patient_slice = patient_slices[patient_id]
        patient_features = features[patient_slice]
        gram = grams[patient_id]
        labels_m2, anchors_m2 = farthest_first_strata(patient_features, 2)
        labels_m4, anchors_m4 = farthest_first_strata(patient_features, 4)
        all_strata_nonempty = all_strata_nonempty and (
            set(int(value) for value in labels_m2) == {0, 1}
            and set(int(value) for value in labels_m4) == {0, 1, 2, 3}
        )
        for local_index in range(5):
            record_rows[patient_slice.start + local_index]["stratum_m2"] = int(
                labels_m2[local_index]
            )
            record_rows[patient_slice.start + local_index]["stratum_m4"] = int(
                labels_m4[local_index]
            )
            strata_rows.append(
                {
                    "patient_id": patient_id,
                    "record_index": local_index,
                    "stratum_m2": int(labels_m2[local_index]),
                    "anchor_m2": int(local_index in set(int(v) for v in anchors_m2)),
                    "stratum_m4": int(labels_m4[local_index]),
                    "anchor_m4": int(local_index in set(int(v) for v in anchors_m4)),
                }
            )

        method_weights = {
            **uniform_weights,
            "vae_stratified_m2_r2": enumerate_stratified_weights(
                labels_m2, REFERENCE_PERTURBATIONS, 2
            ),
            "vae_stratified_m4_r1": enumerate_stratified_weights(
                labels_m4, REFERENCE_PERTURBATIONS, 1
            ),
        }
        for method, weights in method_weights.items():
            primary, clipping = exact_estimator_metrics(
                weights,
                gram,
                model_metadata["parameter_dimension"],
                CLIP_NORMS,
            )
            max_unbiased_error = max(
                max_unbiased_error, float(primary["max_expected_weight_error"])
            )
            max_row_sum_error = max(
                max_row_sum_error, float(primary["max_row_weight_sum_error"])
            )
            primary_rows.append(
                {
                    "patient_id": patient_id,
                    "site_group": patient["site_group"],
                    "site_count": patient["site_count"],
                    "method": method,
                    **primary,
                }
            )
            for clip_row in clipping:
                clipping_rows.append(
                    {
                        "patient_id": patient_id,
                        "site_group": patient["site_group"],
                        "method": method,
                        **clip_row,
                    }
                )

        variance = variance_decomposition(
            gram, 5, REFERENCE_PERTURBATIONS, model_metadata["parameter_dimension"]
        )
        residual_ratio = variance["decomposition_residual"] / max(
            variance["total_variance"], 1e-30
        )
        max_decomposition_residual_ratio = max(
            max_decomposition_residual_ratio, residual_ratio
        )
        variance_rows.append(
            {
                "patient_id": patient_id,
                "site_group": patient["site_group"],
                **variance,
                "decomposition_residual_ratio": residual_ratio,
            }
        )
        alignment = feature_gradient_alignment(
            patient_features, gram, REFERENCE_PERTURBATIONS
        )
        alignment_rows.append(
            {
                "patient_id": patient_id,
                "site_group": patient["site_group"],
                **alignment,
            }
        )
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(gram)))
        tolerance = max(1e-6, float(np.max(np.diag(gram))) * 1e-5)
        gram_psd_pass = gram_psd_pass and minimum_eigenvalue >= -tolerance

    write_csv(output_dir / "selected_patient_records.csv", record_rows)
    write_csv(output_dir / "coverage_strata.csv", strata_rows)
    write_csv(output_dir / "patient_preclip_metrics.csv", primary_rows)
    write_csv(output_dir / "patient_clipping_metrics.csv", clipping_rows)
    write_csv(output_dir / "patient_variance_decomposition.csv", variance_rows)
    write_csv(output_dir / "patient_feature_gradient_alignment.csv", alignment_rows)

    methods = sorted({row["method"] for row in primary_rows})
    uniform_m4_by_patient = {
        row["patient_id"]: row["preclip_mse"]
        for row in primary_rows
        if row["method"] == "uniform_m4_r1"
    }
    method_summary: list[dict[str, Any]] = []
    for method in methods:
        selected_rows = [row for row in primary_rows if row["method"] == method]
        values = np.asarray([row["preclip_mse"] for row in selected_rows])
        ratios = np.asarray(
            [
                row["preclip_mse"] / uniform_m4_by_patient[row["patient_id"]]
                for row in selected_rows
            ]
        )
        method_summary.append(
            {
                "method": method,
                "patients": len(selected_rows),
                "mean_preclip_mse": float(np.mean(values)),
                "median_preclip_mse": float(np.median(values)),
                "geometric_ratio_to_uniform_m4": float(
                    np.exp(np.mean(np.log(ratios)))
                ),
                "wins_vs_uniform_m4": int(np.sum(ratios < 1.0)),
            }
        )
    write_csv(output_dir / "method_preclip_summary.csv", method_summary)

    alignment_by_patient = {
        row["patient_id"]: row["feature_gradient_distance_spearman"]
        for row in alignment_rows
    }
    coverage_by_patient = {
        row["patient_id"]: row["preclip_mse"]
        for row in primary_rows
        if row["method"] == "vae_stratified_m4_r1"
    }
    patient_comparisons: list[dict[str, Any]] = []
    for patient in selected:
        patient_id = patient["patient_id"]
        ratio = coverage_by_patient[patient_id] / uniform_m4_by_patient[patient_id]
        patient_comparisons.append(
            {
                "patient_id": patient_id,
                "site_group": patient["site_group"],
                "site_count": patient["site_count"],
                "uniform_m4_preclip_mse": uniform_m4_by_patient[patient_id],
                "coverage_m4_preclip_mse": coverage_by_patient[patient_id],
                "coverage_to_uniform_m4_ratio": ratio,
                "feature_gradient_distance_spearman": alignment_by_patient[patient_id],
            }
        )
    write_csv(output_dir / "patient_primary_comparison.csv", patient_comparisons)

    ratios = np.asarray(
        [row["coverage_to_uniform_m4_ratio"] for row in patient_comparisons]
    )
    bootstrap = bootstrap_ratio(ratios)
    coverage_wins = int(np.sum(ratios < 1.0))
    median_alignment = float(
        np.median(
            [row["feature_gradient_distance_spearman"] for row in patient_comparisons]
        )
    )
    conditions = [
        {
            "name": "geometric_mean_ratio_le_0.95",
            "status": "PASS"
            if bootstrap["geometric_mean_ratio"] <= 0.95
            else "FAIL",
            "observed": bootstrap["geometric_mean_ratio"],
            "required": "<=0.95",
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
            "name": "coverage_wins_ge_10_of_16",
            "status": "PASS" if coverage_wins >= 10 else "FAIL",
            "observed": coverage_wins,
            "required": ">=10",
        },
        {
            "name": "median_alignment_gt_0.10",
            "status": "PASS" if median_alignment > 0.10 else "FAIL",
            "observed": median_alignment,
            "required": ">0.10",
        },
    ]
    method_decision = (
        "CONTINUE_VAE_COVERAGE"
        if all(condition["status"] == "PASS" for condition in conditions)
        else "RETIRE_CURRENT_VAE_COVERAGE"
    )

    clip_summary: list[dict[str, Any]] = []
    for method in methods:
        for clip_norm in CLIP_NORMS:
            rows = [
                row
                for row in clipping_rows
                if row["method"] == method and row["clip_norm"] == clip_norm
            ]
            clip_summary.append(
                {
                    "method": method,
                    "clip_norm": clip_norm,
                    "mean_postclip_mse": float(
                        np.mean([row["postclip_mse"] for row in rows])
                    ),
                    "mean_clip_rate": float(np.mean([row["clip_rate"] for row in rows])),
                    "mean_clip_displacement": float(
                        np.mean([row["clip_displacement"] for row in rows])
                    ),
                }
            )
    write_csv(output_dir / "method_clipping_summary.csv", clip_summary)

    execution_checks = [
        {
            "name": "critical_model_hashes",
            "status": "PASS"
            if all(row["status"] == "PASS" for row in critical_hash_checks.values())
            else "FAIL",
        },
        {
            "name": "balanced_16_patient_selection",
            "status": "PASS"
            if len(selected) == 16
            and sum(row["site_group"] == "low_site" for row in selected) == 8
            and sum(row["site_group"] == "high_site" for row in selected) == 8
            else "FAIL",
        },
        {
            "name": "all_320_gradients_finite",
            "status": "PASS"
            if len(gradient_rows) == 320 and all(row["finite"] for row in gradient_rows)
            else "FAIL",
        },
        {
            "name": "all_coverage_strata_nonempty",
            "status": "PASS" if all_strata_nonempty else "FAIL",
        },
        {
            "name": "exact_estimator_unbiasedness",
            "status": "PASS"
            if max_unbiased_error <= 1e-12 and max_row_sum_error <= 1e-12
            else "FAIL",
            "max_expected_weight_error": max_unbiased_error,
            "max_row_weight_sum_error": max_row_sum_error,
        },
        {
            "name": "all_gram_psd_tolerance",
            "status": "PASS" if gram_psd_pass else "FAIL",
        },
        {
            "name": "variance_decomposition_identity",
            "status": "PASS"
            if max_decomposition_residual_ratio <= 1e-5
            else "FAIL",
            "max_residual_ratio": max_decomposition_residual_ratio,
        },
    ]
    status = (
        "PASS"
        if all(check["status"] == "PASS" for check in execution_checks)
        else "FAIL"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "privacy_status": "DIAGNOSTIC_NOT_DP",
        "protocol": {
            "file": protocol_path.name,
            "sha256": sha256_file(protocol_path),
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "variant": MODEL_VARIANT,
            "repository_role": "third_party_hash_pinned_mirror",
            "critical_hash_checks": critical_hash_checks,
            **model_metadata,
        },
        "selection": {
            **selection_evidence,
            "selected": [
                {
                    key: patient[key]
                    for key in (
                        "patient_id",
                        "site_group",
                        "site_count",
                        "selection_hash",
                    )
                }
                for patient in selected
            ],
        },
        "configuration": {
            "image_size": IMAGE_SIZE,
            "rank": LORA_RANK,
            "reference_perturbations_per_record": REFERENCE_PERTURBATIONS,
            "q": Q,
            "clip_norms": list(CLIP_NORMS),
            "feature": "sd21_vae_mode_scaled_adaptive_avg_pool_8x8_l2",
            "exact_estimator_enumeration": True,
            "common_random_number_perturbations": True,
        },
        "gradient_execution": {
            "patients": len(selected),
            "total_gradients": len(gradient_rows),
            "elapsed_sec": gradient_elapsed,
            "patient_elapsed_sec_sum": float(
                sum(row["elapsed_sec"] for row in runtime_rows)
            ),
            "cuda": gradient_cuda,
            "raw_gradient_saved": False,
            "gram_release_status": "LOCAL_ONLY",
        },
        "execution_checks": execution_checks,
        "method_gate": {
            "decision": method_decision,
            "patients": len(patient_comparisons),
            "coverage_wins": coverage_wins,
            "median_feature_gradient_spearman": median_alignment,
            **bootstrap,
            "conditions": conditions,
        },
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
            "coverage_strata": "coverage_strata.csv",
            "gradient_evaluations": "gradient_evaluations.csv",
            "patient_runtime": "patient_runtime.csv",
            "perturbation_specs": "perturbation_specs.csv",
            "patient_gradient_grams": "patient_gradient_grams_local_only.npz",
            "patient_preclip_metrics": "patient_preclip_metrics.csv",
            "patient_clipping_metrics": "patient_clipping_metrics.csv",
            "patient_variance_decomposition": "patient_variance_decomposition.csv",
            "patient_feature_gradient_alignment": "patient_feature_gradient_alignment.csv",
            "patient_primary_comparison": "patient_primary_comparison.csv",
            "method_preclip_summary": "method_preclip_summary.csv",
            "method_clipping_summary": "method_clipping_summary.csv",
            "summary": "summary.md",
        },
        "interpretation_limit": (
            "Public-development gradient diagnostic only; no optimizer update, DP noise, "
            "generation result, clinical utility, privacy guarantee, or novelty claim."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        build_summary_markdown(result, method_summary, patient_comparisons),
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
