#!/usr/bin/env python3
"""One-patient real-gradient smoke for fixed-compute PCM diagnostics.

This diagnostic performs no optimizer update and adds no DP noise.  Its output
cannot support a privacy, utility, medical, or novelty claim.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
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


SCHEMA = "pcm-isic-sd21-gradient-smoke/v1"
MODEL_ID = "Manojb/stable-diffusion-2-1-base"
MODEL_REVISION = "0094d483a120f3f33dafbd187ea4aa60d10de75c"
MODEL_VARIANT = "fp16"
SELECTION_SALT = "pcm-isic-sd21-gradient-smoke-v1"
EXPECTED_CRITICAL_SHA256 = {
    "text_encoder/model.fp16.safetensors": (
        "681C555376658C81DC273F2D737A2AEB23DDB6D1D8E5B3A7064636D359A22668"
    ),
    "unet/diffusion_pytorch_model.fp16.safetensors": (
        "28EC9CF3B239C0751C201B1F6FB46B551DF5862731B30A37AA1360101CB3FBAB"
    ),
    "vae/diffusion_pytorch_model.fp16.safetensors": (
        "3E4C08995484EE61270175E9E7A072B66A6E4EEB5F0C266667FE1F45B90DAF9A"
    ),
}
ALLOW_PATTERNS = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]


def disable_broken_optional_onnx() -> None:
    import importlib.util

    original_find_spec = importlib.util.find_spec

    def guarded_find_spec(name: str, package: str | None = None):
        if name == "onnxruntime" or name.startswith("onnxruntime."):
            return None
        return original_find_spec(name, package)

    importlib.util.find_spec = guarded_find_spec


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) & ((1 << 63) - 1)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def verify_snapshot(snapshot: Path) -> dict[str, dict[str, str | None]]:
    checks: dict[str, dict[str, str | None]] = {}
    for relative, expected in EXPECTED_CRITICAL_SHA256.items():
        path = snapshot / relative
        actual = sha256_file(path) if path.is_file() else None
        checks[relative] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "status": "PASS" if actual == expected else "FAIL",
        }
    failures = [name for name, row in checks.items() if row["status"] != "PASS"]
    if failures:
        raise RuntimeError(f"critical model artifact mismatch: {failures}")
    return checks


def select_patient(records: Iterable[Any]) -> tuple[str, list[Any], dict[str, Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        grouped[record.patient_id].append(record)

    candidates: list[tuple[str, str, list[Any]]] = []
    for patient_id, patient_records in grouped.items():
        lesions = {record.lesion_id for record in patient_records}
        sites = {record.anatom_site for record in patient_records if record.anatom_site}
        if len(patient_records) == 5 and len(lesions) == 5 and len(sites) >= 3:
            selection_hash = hashlib.sha256(
                f"{SELECTION_SALT}|{patient_id}".encode("utf-8")
            ).hexdigest().upper()
            candidates.append((selection_hash, patient_id, patient_records))
    if not candidates:
        raise RuntimeError("no patient satisfies the frozen smoke eligibility rule")
    candidates.sort(key=lambda item: (item[0], item[1]))
    selection_hash, patient_id, patient_records = candidates[0]
    patient_records = sorted(patient_records, key=lambda record: record.cap_rank)
    evidence = {
        "selection_salt": SELECTION_SALT,
        "eligible_patients": len(candidates),
        "selected_patient_id": patient_id,
        "selected_patient_hash": selection_hash,
        "images": len(patient_records),
        "distinct_lesions": len({record.lesion_id for record in patient_records}),
        "distinct_nonempty_sites": len(
            {record.anatom_site for record in patient_records if record.anatom_site}
        ),
    }
    return patient_id, patient_records, evidence


def farthest_first_strata(
    features: np.ndarray, strata: int
) -> tuple[np.ndarray, np.ndarray]:
    if features.ndim != 2:
        raise ValueError("features must be a 2-D matrix")
    n = features.shape[0]
    if not 1 <= strata <= n:
        raise ValueError("strata must be between one and number of records")
    squared = np.sum(np.square(features[:, None, :] - features[None, :, :]), axis=2)
    centroid = np.mean(features, axis=0)
    first = int(np.argmax(np.sum(np.square(features - centroid), axis=1)))
    anchors = [first]
    while len(anchors) < strata:
        min_distance = np.min(squared[:, anchors], axis=1)
        min_distance[anchors] = -1.0
        anchors.append(int(np.argmax(min_distance)))
    anchor_array = np.asarray(anchors, dtype=np.int64)
    labels = np.argmin(squared[:, anchor_array], axis=1).astype(np.int64)
    for stratum, anchor in enumerate(anchor_array):
        labels[anchor] = stratum
    if set(int(value) for value in labels) != set(range(strata)):
        raise AssertionError("farthest-first produced an empty stratum")
    return labels, anchor_array


def weights_noise_only(
    rng: np.random.Generator, records: int, perturbations: int, q: int
) -> np.ndarray:
    weights = np.zeros(records * perturbations, dtype=np.float64)
    record = int(rng.integers(records))
    draws = rng.integers(perturbations, size=q)
    for perturbation in draws:
        weights[record * perturbations + int(perturbation)] += 1.0 / q
    return weights


def weights_uniform_distinct(
    rng: np.random.Generator, records: int, perturbations: int, q: int
) -> np.ndarray:
    if q > records:
        raise ValueError("uniform-distinct v1 requires Q <= records")
    weights = np.zeros(records * perturbations, dtype=np.float64)
    selected = rng.choice(records, size=q, replace=False)
    draws = rng.integers(perturbations, size=q)
    for record, perturbation in zip(selected, draws):
        weights[int(record) * perturbations + int(perturbation)] += 1.0 / q
    return weights


def weights_coverage_stratified(
    rng: np.random.Generator,
    labels: np.ndarray,
    perturbations: int,
) -> np.ndarray:
    records = len(labels)
    strata = np.unique(labels)
    weights = np.zeros(records * perturbations, dtype=np.float64)
    for stratum in strata:
        members = np.flatnonzero(labels == stratum)
        record = int(rng.choice(members))
        perturbation = int(rng.integers(perturbations))
        weights[record * perturbations + perturbation] += len(members) / records
    return weights


def gram_metrics(
    weights: np.ndarray,
    reference_weights: np.ndarray,
    gram: np.ndarray,
    dimension: int,
    clip_norm: float,
) -> dict[str, float]:
    def quadratic(vector: np.ndarray) -> float:
        return max(0.0, float(vector @ gram @ vector))

    estimate_norm_sq = quadratic(weights)
    reference_norm_sq = quadratic(reference_weights)
    estimate_norm = math.sqrt(estimate_norm_sq)
    reference_norm = math.sqrt(reference_norm_sq)
    estimate_scale = min(1.0, clip_norm / estimate_norm) if estimate_norm > 0 else 1.0
    reference_scale = min(1.0, clip_norm / reference_norm) if reference_norm > 0 else 1.0

    pre_difference = weights - reference_weights
    post_difference = estimate_scale * weights - reference_scale * reference_weights
    displacement = (estimate_scale - 1.0) * weights
    dot = float(weights @ gram @ reference_weights)
    denominator = estimate_norm * reference_norm
    cosine = dot / denominator if denominator > 0 else 1.0
    return {
        "preclip_mse": quadratic(pre_difference) / dimension,
        "postclip_mse": quadratic(post_difference) / dimension,
        "clip_displacement": quadratic(displacement) / dimension,
        "cosine_error": 1.0 - max(-1.0, min(1.0, cosine)),
        "clip_rate": float(estimate_norm > clip_norm),
        "estimate_norm": estimate_norm,
    }


def evaluate_estimators(
    gram: np.ndarray,
    parameter_dimension: int,
    labels: np.ndarray,
    perturbations: int,
    q: int,
    clip_norm: float,
    seeds: tuple[int, ...],
    trials: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = len(labels)
    reference_weights = np.full(records * perturbations, 1.0 / (records * perturbations))
    methods = ("noise_only", "uniform_distinct", "coverage_stratified")
    by_seed: list[dict[str, Any]] = []

    for seed in seeds:
        for method in methods:
            rng = np.random.default_rng(stable_seed("estimator", seed, method))
            values: dict[str, list[float]] = defaultdict(list)
            weight_sum_error = 0.0
            for _ in range(trials):
                if method == "noise_only":
                    weights = weights_noise_only(rng, records, perturbations, q)
                elif method == "uniform_distinct":
                    weights = weights_uniform_distinct(rng, records, perturbations, q)
                else:
                    weights = weights_coverage_stratified(rng, labels, perturbations)
                weight_sum_error = max(weight_sum_error, abs(float(np.sum(weights)) - 1.0))
                metrics = gram_metrics(
                    weights,
                    reference_weights,
                    gram,
                    parameter_dimension,
                    clip_norm,
                )
                for name, value in metrics.items():
                    values[name].append(value)
            row: dict[str, Any] = {
                "seed": seed,
                "method": method,
                "trials": trials,
                "max_weight_sum_error": weight_sum_error,
            }
            for name, metric_values in values.items():
                row[name] = float(np.mean(metric_values))
            by_seed.append(row)

    aggregate: list[dict[str, Any]] = []
    for method in methods:
        selected = [row for row in by_seed if row["method"] == method]
        row = {"method": method, "seeds": len(selected), "trials_per_seed": trials}
        row["max_weight_sum_error"] = max(
            float(item["max_weight_sum_error"]) for item in selected
        )
        for name in (
            "preclip_mse",
            "postclip_mse",
            "clip_displacement",
            "cosine_error",
            "clip_rate",
            "estimate_norm",
        ):
            metric_values = np.asarray([item[name] for item in selected], dtype=np.float64)
            row[name] = float(np.mean(metric_values))
            row[f"{name}_seed_sd"] = float(np.std(metric_values, ddof=1))
        aggregate.append(row)

    noise_mse = next(row["preclip_mse"] for row in aggregate if row["method"] == "noise_only")
    for row in aggregate:
        row["preclip_mse_ratio_to_noise_only"] = row["preclip_mse"] / noise_mse
    return by_seed, aggregate


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cuda_memory(torch: Any) -> dict[str, float]:
    gib = 1024.0**3
    return {
        "allocated_gib": round(torch.cuda.memory_allocated() / gib, 4),
        "reserved_gib": round(torch.cuda.memory_reserved() / gib, 4),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / gib, 4),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / gib, 4),
    }


def encode_patient_inputs(
    snapshot: Path,
    records: list[Any],
    image_root: Path,
    image_size: int,
) -> tuple[list[Any], list[Any], np.ndarray, list[dict[str, Any]]]:
    import torch
    import torch.nn.functional as functional
    from diffusers import AutoencoderKL
    from PIL import Image
    from transformers import CLIPTextModel, CLIPTokenizer

    data_pipeline_dir = Path(__file__).resolve().parents[1] / "data_pipeline"
    sys.path.insert(0, str(data_pipeline_dir))
    from isic_manifest_dataset import (  # pylint: disable=import-error,import-outside-toplevel
        pil_to_normalized_tensor,
        preprocess_pil,
        prompt_for_record,
    )

    tokenizer = CLIPTokenizer.from_pretrained(
        snapshot / "tokenizer", local_files_only=True
    )
    text_encoder = CLIPTextModel.from_pretrained(
        snapshot / "text_encoder",
        torch_dtype=torch.float16,
        variant=MODEL_VARIANT,
        use_safetensors=True,
        local_files_only=True,
    ).eval().to("cuda")
    prompt_cache: dict[str, Any] = {}
    with torch.inference_mode():
        for record in records:
            prompt = prompt_for_record(record)
            if prompt not in prompt_cache:
                tokens = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                hidden = text_encoder(tokens.input_ids.to("cuda"))[0]
                prompt_cache[prompt] = hidden.detach().cpu()
    encoder_states = [prompt_cache[prompt_for_record(record)] for record in records]
    del text_encoder
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    vae = AutoencoderKL.from_pretrained(
        snapshot / "vae",
        torch_dtype=torch.float16,
        variant=MODEL_VARIANT,
        use_safetensors=True,
        local_files_only=True,
    ).eval().to("cuda")
    latents: list[Any] = []
    features: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for record in records:
            path = image_root / record.image_filename
            with Image.open(path) as source:
                image = preprocess_pil(source, image_size, "center_crop")
            pixels = pil_to_normalized_tensor(image).unsqueeze(0).to(
                device="cuda", dtype=torch.float16
            )
            latent = vae.encode(pixels).latent_dist.mode()
            latent = latent * float(vae.config.scaling_factor)
            pooled = functional.adaptive_avg_pool2d(latent.float(), (8, 8)).flatten()
            pooled = functional.normalize(pooled, dim=0)
            latents.append(latent.detach().cpu())
            features.append(pooled.detach().cpu().numpy().astype(np.float64))
            rows.append(
                {
                    "image_id": record.image_id,
                    "cap_rank": record.cap_rank,
                    "lesion_id": record.lesion_id,
                    "anatom_site": record.anatom_site,
                    "target": record.target,
                    "prompt": prompt_for_record(record),
                    "image_sha256": sha256_file(path),
                }
            )
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    return latents, encoder_states, np.stack(features), rows


def build_gradient_bank(
    snapshot: Path,
    latents: list[Any],
    encoder_states: list[Any],
    perturbations: int,
    rank: int,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    import torch
    import torch.nn.functional as functional
    from diffusers import DDPMScheduler, UNet2DConditionModel
    from peft import LoraConfig

    torch.manual_seed(260902)
    torch.cuda.manual_seed_all(260902)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    scheduler = DDPMScheduler.from_pretrained(
        snapshot / "scheduler", local_files_only=True
    )
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
            r=rank,
            lora_alpha=rank,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
    )
    unet.enable_gradient_checkpointing()
    unet.train().to("cuda")
    trainable = [(name, parameter) for name, parameter in unet.named_parameters() if parameter.requires_grad]
    parameter_dimension = sum(parameter.numel() for _, parameter in trainable)
    bank = np.empty(
        (len(latents) * perturbations, parameter_dimension), dtype=np.float32
    )
    evaluations: list[dict[str, Any]] = []
    prediction_type = str(scheduler.config.prediction_type)
    started = time.perf_counter()

    for record_index, latent_cpu in enumerate(latents):
        latent = latent_cpu.to(device="cuda", dtype=torch.float16)
        hidden = encoder_states[record_index].to(device="cuda", dtype=torch.float16)
        for perturbation in range(perturbations):
            seed = stable_seed("gradient", record_index, perturbation)
            generator = torch.Generator(device="cuda").manual_seed(seed)
            timestep_generator = np.random.default_rng(seed)
            timestep_value = int(
                timestep_generator.integers(int(scheduler.config.num_train_timesteps))
            )
            timestep = torch.tensor([timestep_value], device="cuda", dtype=torch.long)
            noise = torch.randn(
                latent.shape,
                generator=generator,
                device="cuda",
                dtype=torch.float16,
            )
            noisy_latent = scheduler.add_noise(latent, noise, timestep)
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
            gradient_norm = float(np.linalg.norm(bank[bank_index].astype(np.float64)))
            evaluations.append(
                {
                    "record_index": record_index,
                    "perturbation_index": perturbation,
                    "seed": seed,
                    "timestep": timestep_value,
                    "loss": float(loss.detach().cpu()),
                    "gradient_norm": gradient_norm,
                    "finite": finite,
                }
            )
            if not finite:
                raise RuntimeError("non-finite LoRA gradient encountered")

    torch.cuda.synchronize()
    runtime = {
        "elapsed_sec": round(time.perf_counter() - started, 4),
        "parameter_dimension": parameter_dimension,
        "trainable_tensors": len(trainable),
        "prediction_type": prediction_type,
        "cuda": cuda_memory(torch),
    }
    del unet
    del scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return bank, evaluations, runtime


def build_summary_markdown(
    result: dict[str, Any], metrics: list[dict[str, Any]]
) -> str:
    lines = [
        "# ISIC SD 2.1 real-gradient smoke v1",
        "",
        f"- Status: **{result['status']}**",
        f"- Selected public-development patient: `{result['patient_selection']['selected_patient_id']}`",
        f"- Gradient bank: {result['gradient_bank']['gradients']} gradients x "
        f"{result['gradient_bank']['parameter_dimension']} LoRA parameters",
        f"- Gradient wall time: {result['gradient_bank']['elapsed_sec']} sec",
        "- Scope: one-patient execution smoke; not an efficacy, privacy, or novelty result",
        "",
        "## Q=4 finite-reference comparison",
        "",
        "| Method | Preclip MSE | Postclip MSE | Clip rate | Ratio to noise-only |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['method']} | {row['preclip_mse']:.10f} | "
            f"{row['postclip_mse']:.10f} | {row['clip_rate']:.4f} | "
            f"{row['preclip_mse_ratio_to_noise_only']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The ranking above is descriptive for one deterministically selected public-development patient.",
            "It must not select the paper method. The next efficacy gate requires a separately frozen",
            "multi-patient diagnostic and must retain null or adverse patients.",
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

    records = read_manifest(args.manifest, partitions={"public_development"})
    patient_id, patient_records, selection = select_patient(records)
    del patient_id
    latents, encoder_states, features, record_rows = encode_patient_inputs(
        snapshot, patient_records, args.image_root.resolve(), args.image_size
    )
    labels, anchors = farthest_first_strata(features, args.strata)
    for index, row in enumerate(record_rows):
        row["coverage_stratum"] = int(labels[index])
        row["is_anchor"] = int(index in set(int(value) for value in anchors))
    write_csv(output_dir / "selected_patient_records.csv", record_rows)
    np.save(output_dir / "vae_feature_distance_matrix.npy", np.sum(
        np.square(features[:, None, :] - features[None, :, :]), axis=2
    ))

    bank, evaluations, runtime = build_gradient_bank(
        snapshot, latents, encoder_states, args.reference_perturbations, args.rank
    )
    write_csv(output_dir / "gradient_evaluations.csv", evaluations)
    gram = np.asarray(bank @ bank.T, dtype=np.float64)
    gram = 0.5 * (gram + gram.T)
    np.save(output_dir / "gradient_gram_local_only.npy", gram)
    del bank
    gc.collect()

    eigenvalues = np.linalg.eigvalsh(gram)
    max_diagonal = float(np.max(np.diag(gram)))
    psd_tolerance = max(1e-6, max_diagonal * 1e-5)
    seeds = (260902, 260903, 260904, 260905, 260906)
    by_seed, metrics = evaluate_estimators(
        gram=gram,
        parameter_dimension=runtime["parameter_dimension"],
        labels=labels,
        perturbations=args.reference_perturbations,
        q=args.q,
        clip_norm=args.clip_norm,
        seeds=seeds,
        trials=args.trials,
    )
    write_csv(output_dir / "estimator_metrics_by_seed.csv", by_seed)
    write_csv(output_dir / "estimator_metrics.csv", metrics)

    checks = [
        {
            "name": "critical_model_hashes",
            "status": "PASS"
            if all(row["status"] == "PASS" for row in critical_hash_checks.values())
            else "FAIL",
        },
        {
            "name": "five_finite_record_perturbation_gradients",
            "status": "PASS"
            if len(evaluations) == 5 * args.reference_perturbations
            and all(row["finite"] for row in evaluations)
            else "FAIL",
        },
        {
            "name": "nonempty_coverage_strata",
            "status": "PASS"
            if set(int(value) for value in labels) == set(range(args.strata))
            else "FAIL",
        },
        {
            "name": "estimator_weight_sums",
            "status": "PASS"
            if max(row["max_weight_sum_error"] for row in metrics) <= 1e-12
            else "FAIL",
        },
        {
            "name": "gram_psd_tolerance",
            "status": "PASS" if float(np.min(eigenvalues)) >= -psd_tolerance else "FAIL",
            "minimum_eigenvalue": float(np.min(eigenvalues)),
            "tolerance": psd_tolerance,
        },
    ]
    status = "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL"
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "privacy_status": "DIAGNOSTIC_NOT_DP",
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "variant": MODEL_VARIANT,
            "repository_role": "third_party_hash_pinned_mirror",
            "critical_hash_checks": critical_hash_checks,
        },
        "patient_selection": selection,
        "configuration": {
            "partition": "public_development",
            "image_size": args.image_size,
            "rank": args.rank,
            "reference_perturbations_per_record": args.reference_perturbations,
            "q": args.q,
            "strata": args.strata,
            "clip_norm": args.clip_norm,
            "seeds": list(seeds),
            "trials_per_seed": args.trials,
            "feature": "sd21_vae_mode_scaled_adaptive_avg_pool_8x8_l2",
        },
        "coverage": {
            "labels_by_cap_rank": [int(value) for value in labels],
            "anchor_indices_zero_based": [int(value) for value in anchors],
            "stratum_sizes": {
                str(int(stratum)): int(np.sum(labels == stratum))
                for stratum in np.unique(labels)
            },
        },
        "gradient_bank": {
            "gradients": len(evaluations),
            **runtime,
            "gram_min_eigenvalue": float(np.min(eigenvalues)),
            "gram_psd_tolerance": psd_tolerance,
            "raw_gradient_saved": False,
            "gram_release_status": "LOCAL_ONLY",
        },
        "checks": checks,
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
            "feature_distance_matrix": "vae_feature_distance_matrix.npy",
            "gradient_evaluations": "gradient_evaluations.csv",
            "gradient_gram": "gradient_gram_local_only.npy",
            "estimator_metrics_by_seed": "estimator_metrics_by_seed.csv",
            "estimator_metrics": "estimator_metrics.csv",
            "summary": "summary.md",
        },
        "interpretation_limit": (
            "One public-development patient execution smoke only; no optimizer update, "
            "no DP noise, and no efficacy, privacy, medical, or novelty claim."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        build_summary_markdown(result, metrics), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--reference-perturbations", type=int, default=4)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument("--strata", type=int, default=4)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--trials", type=int, default=1024)
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
