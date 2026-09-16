#!/usr/bin/env python3
"""Disposable one-step M1/M2 LoRA conformance on public-development X-rays.

This is intentionally not private training.  It uses only frozen
``public_development`` records and deterministic TEST_ONLY Gaussian noise to
show that real SD 2.1 LoRA gradients can pass through the executable DP core
and an exact AdamW optimizer step.  The model is destroyed without a
checkpoint after each arm.
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
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "nih-cxr14-public-lora-dp-step-smoke/v1"
MODEL_ID = "Manojb/stable-diffusion-2-1-base"
MODEL_REVISION = "0094d483a120f3f33dafbd187ea4aa60d10de75c"
MODEL_VARIANT = "fp16"
K10_SHA256 = "2D749FB7B70823114A69FD55921277B0D0FF2F9AEC8FECD239159C553F3A0C06"
PROTOCOL_SHA256 = "F2757DC7EBC7488B6A9DD0F69227419EA16BB9A3E9BD4434514CB19AFD2B4018"
PREPROCESSING_CONTRACT_SHA256 = (
    "0ED9E434764BDB2D542BB16AB951F856CB5B82D4F2E6B9F25F6C554F1076669D"
)
SELECTION_SALT = "nih-cxr14-public-lora-dp-step-smoke-clip-path-v1"
CALIBRATION_DRAW_SALT = "nih-cxr14-public-clip-calibration-v1"
CALIBRATION_REPORT_SHA256 = "F9771F169BA3F6468F5AAF254A27AFFE1FB47B1ABA897C365BFFC7CDD11A73F1"
CALIBRATION_GRADIENTS_SHA256 = "9A63FB1546E08F4B4BAD2A6C8027A6A901F2CCE46EFBBC769F390FFD16DA4110"
LORA_INIT_SEED = 260_903
M1_DP_NOISE_TEST_SEED = 260_903_301
M2_DP_NOISE_TEST_SEED = 260_903_302
LORA_RANK = 8
LORA_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")
TRAINABLE_PARAMETERS = 1_659_904
IMAGE_SIZE = 256
PATIENT_MAX_IMAGES = 4
CLIP_NORMS = {"image": 0.28448700606156724, "patient": 0.1997973088974048}
OPTIMIZER = {
    "name": "AdamW",
    "learning_rate": 1e-4,
    "betas": [0.9, 0.999],
    "epsilon": 1e-8,
    "weight_decay": 0.01,
    "schedule": "constant_no_warmup",
}


def default_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parent.parent
    return {
        "root": root,
        "manifest": root
        / "_data"
        / "derived"
        / "nih_cxr14_pa_target_enriched_v1"
        / "k10_private.csv",
        "image_root": root / "_data" / "raw" / "nih_cxr14_pa_k10_plus_census_v1" / "images",
        "protocol": root / "_reports" / "nih_cxr14_dp_attack_protocol_v1_001" / "protocol.json",
        "preprocessing_contract": root
        / "_reports"
        / "nih_cxr14_sd21_ppmark_interface_v1_001"
        / "preprocessing_contract.json",
        "calibration_report": root
        / "_reports"
        / "nih_cxr14_public_clip_calibration_v1_001"
        / "report.json",
        "calibration_gradients": root
        / "_reports"
        / "nih_cxr14_public_clip_calibration_v1_001"
        / "gradient_norms_private.csv",
        "output": root / "_reports" / "nih_cxr14_public_lora_dp_step_smoke_v1_001",
    }


def parse_args() -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=paths["manifest"])
    parser.add_argument("--image-root", type=Path, default=paths["image_root"])
    parser.add_argument("--protocol", type=Path, default=paths["protocol"])
    parser.add_argument("--preprocessing-contract", type=Path, default=paths["preprocessing_contract"])
    parser.add_argument("--calibration-report", type=Path, default=paths["calibration_report"])
    parser.add_argument("--calibration-gradients", type=Path, default=paths["calibration_gradients"])
    parser.add_argument("--output-dir", type=Path, default=paths["output"])
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest().upper()


def hash_uint63(*parts: object) -> int:
    return int(stable_hash(*parts)[:16], 16) % (2**63)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def select_public_units(
    records: list[Any], calibration_rows: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select public high-gradient sentinels solely for clipping-path coverage.

    This is intentionally outcome-dependent and therefore must never be used
    for utility, attack, or generalization evaluation.  The frozen public
    calibration is allowed here because this disposable run asks only whether
    the real-vector clipping branch executes correctly.
    """

    by_patient: dict[str, list[Any]] = defaultdict(list)
    by_image: dict[str, Any] = {}
    for record in records:
        by_patient[record.patient_id].append(record)
        by_image[record.image_id] = record
    for patient_records in by_patient.values():
        patient_records.sort(key=lambda item: (item.cap_rank, item.image_id))

    image_units: list[dict[str, Any]] = []
    for target in (0, 1):
        candidates = [
            row
            for row in calibration_rows
            if row["unit_type"] == "image" and int(row["target_patient"]) == target
        ]
        require(candidates, f"no calibrated public image for target={target}")
        selected = max(
            candidates,
            key=lambda row: (float(row["gradient_l2_norm"]), row["unit_id"]),
        )
        image_ids = selected["image_ids"].split("|")
        require(len(image_ids) == 1 and image_ids[0] in by_image, "calibrated M1 image missing")
        record = by_image[image_ids[0]]
        image_units.append(
            {
                "unit_type": "image",
                "unit_id": record.image_id,
                "patient_id": record.patient_id,
                "target_patient": target,
                "records": [record],
                "calibration_gradient_l2_norm": float(selected["gradient_l2_norm"]),
            }
        )

    patient_units: list[dict[str, Any]] = []
    for target in (0, 1):
        candidates = [
            row
            for row in calibration_rows
            if row["unit_type"] == "patient"
            and int(row["target_patient"]) == target
            and int(row["image_count"]) == PATIENT_MAX_IMAGES
        ]
        require(candidates, f"no four-image calibrated public patient for target={target}")
        selected = max(
            candidates,
            key=lambda row: (float(row["gradient_l2_norm"]), row["unit_id"]),
        )
        patient = selected["patient_id"]
        image_ids = selected["image_ids"].split("|")
        require(len(image_ids) == PATIENT_MAX_IMAGES, "calibrated M2 image count mismatch")
        require(all(image_id in by_image for image_id in image_ids), "calibrated M2 image missing")
        chosen = [by_image[image_id] for image_id in image_ids]
        require(all(record.patient_id == patient for record in chosen), "calibrated M2 patient mismatch")
        patient_units.append(
            {
                "unit_type": "patient",
                "unit_id": patient,
                "patient_id": patient,
                "target_patient": target,
                "records": chosen,
                "calibration_gradient_l2_norm": float(selected["gradient_l2_norm"]),
            }
        )

    all_patients = [unit["patient_id"] for unit in image_units + patient_units]
    require(len(all_patients) == len(set(all_patients)), "smoke units overlap by patient")
    return image_units, patient_units


def unit_commitment(unit: dict[str, Any]) -> str:
    return stable_hash(
        SELECTION_SALT,
        unit["unit_type"],
        unit["unit_id"],
        *(record.image_id for record in unit["records"]),
    )


def flatten_named_tensors(named_tensors: list[tuple[str, Any]]) -> Any:
    import torch

    return torch.cat([tensor.detach().float().reshape(-1) for _, tensor in named_tensors], dim=0)


def named_tensor_digest(named_tensors: list[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        value = tensor.detach().to(device="cpu", dtype=tensor.dtype).contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest().upper()


def gradient_for_unit(
    *,
    unit: dict[str, Any],
    latent_by_image: dict[str, Any],
    hidden_by_prompt: dict[str, Any],
    model_input: Any,
    scheduler: Any,
    unet: Any,
    trainable: list[tuple[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as functional

    latents = []
    hidden = []
    noises = []
    timesteps = []
    seed_commitments = []
    commitment = unit_commitment(unit)
    for index, record in enumerate(unit["records"]):
        latent = latent_by_image[record.image_id].to(device="cuda", dtype=torch.float16)
        prompt = model_input.prompt_for_record(record)
        text = hidden_by_prompt[prompt].to(device="cuda", dtype=torch.float16)
        seed = hash_uint63(
            CALIBRATION_DRAW_SALT,
            "noise",
            unit["unit_type"],
            unit["unit_id"],
            index,
            record.image_id,
        )
        timestep = hash_uint63(
            CALIBRATION_DRAW_SALT,
            "timestep",
            unit["unit_type"],
            unit["unit_id"],
            index,
            record.image_id,
        ) % int(scheduler.config.num_train_timesteps)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        noise = torch.randn(latent.shape, generator=generator, device="cuda", dtype=torch.float16)
        latents.append(latent)
        hidden.append(text)
        noises.append(noise)
        timesteps.append(int(timestep))
        seed_commitments.append(stable_hash("public-calibration-seed", seed))

    latent_batch = torch.cat(latents, dim=0)
    hidden_batch = torch.cat(hidden, dim=0)
    noise_batch = torch.cat(noises, dim=0)
    timestep_batch = torch.tensor(timesteps, device="cuda", dtype=torch.long)
    noisy = scheduler.add_noise(latent_batch, noise_batch, timestep_batch)
    prediction_type = str(scheduler.config.prediction_type)
    if prediction_type == "epsilon":
        target = noise_batch
    elif prediction_type == "v_prediction":
        target = scheduler.get_velocity(latent_batch, noise_batch, timestep_batch)
    else:
        raise RuntimeError(f"unsupported scheduler prediction type: {prediction_type}")

    for _, parameter in trainable:
        parameter.grad = None
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        prediction = unet(noisy, timestep_batch, hidden_batch).sample
        per_image_loss = functional.mse_loss(
            prediction.float(), target.float(), reduction="none"
        ).flatten(1).mean(1)
        # This line is the M2 privacy boundary: all images selected within one
        # patient receive equal inner weight and become one gradient before the
        # outer patient clip.  For M1 the vector has length one.
        unit_loss = per_image_loss.mean()
    unit_loss.backward()

    gradients = []
    missing = 0
    for name, parameter in trainable:
        if parameter.grad is None:
            missing += 1
            gradient = torch.zeros_like(parameter, dtype=torch.float32)
        else:
            gradient = parameter.grad.detach().float()
        require(bool(torch.isfinite(gradient).all()), "non-finite unit gradient")
        gradients.append((name, gradient))
    require(missing == 0, "trainable tensor has no gradient")
    flat = flatten_named_tensors(gradients)
    # Match the frozen calibration's per-tensor float64 accumulation exactly;
    # a single flattened reduction has a harmless but nonzero rounding drift.
    norm_squared = sum(
        float(torch.sum(gradient.double() ** 2).cpu()) for _, gradient in gradients
    )
    norm = math.sqrt(norm_squared)
    summary = {
        "unit_commitment": commitment,
        "target_patient": int(unit["target_patient"]),
        "images_in_unit": len(unit["records"]),
        "loss_reduction": "mean_over_selected_images_before_gradient",
        "loss": float(unit_loss.detach().cpu()),
        "gradient_l2_norm": norm,
        "frozen_calibration_gradient_l2_norm": float(unit["calibration_gradient_l2_norm"]),
        "calibration_norm_absolute_difference": abs(
            norm - float(unit["calibration_gradient_l2_norm"])
        ),
        "gradient_sha256": named_tensor_digest(gradients),
        "timesteps": timesteps,
        "diffusion_seed_commitments": seed_commitments,
        "missing_gradient_tensors": missing,
    }
    return flat, summary


def assign_flat_gradient(trainable: list[tuple[str, Any]], flat: Any) -> None:
    offset = 0
    for _, parameter in trainable:
        count = parameter.numel()
        parameter.grad = flat[offset : offset + count].view_as(parameter).detach().clone()
        offset += count
    require(offset == flat.numel(), "flat gradient length mismatch")


def run_arm(
    *,
    units: list[dict[str, Any]],
    config: Any,
    dp_noise_seed: int,
    snapshot: Path,
    latent_by_image: dict[str, Any],
    hidden_by_prompt: dict[str, Any],
    model_input: Any,
    scheduler: Any,
) -> dict[str, Any]:
    import torch
    from diffusers import UNet2DConditionModel
    from peft import LoraConfig
    from peft.utils.other import cast_mixed_precision_params

    from dp_training.mechanism import aggregate_noised_update, tensor_sha256

    torch.manual_seed(LORA_INIT_SEED)
    torch.cuda.manual_seed_all(LORA_INIT_SEED)
    torch.cuda.reset_peak_memory_stats()
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
            target_modules=list(LORA_TARGETS),
        )
    )
    cast_mixed_precision_params(unet, dtype=torch.float16)
    unet.enable_gradient_checkpointing()
    unet.train().to("cuda")
    trainable = [(name, parameter) for name, parameter in unet.named_parameters() if parameter.requires_grad]
    require(sum(parameter.numel() for _, parameter in trainable) == TRAINABLE_PARAMETERS, "trainable count mismatch")
    require({parameter.dtype for _, parameter in trainable} == {torch.float32}, "LoRA is not fp32")
    initial_digest = named_tensor_digest(trainable)

    unit_vectors = []
    unit_summaries = []
    for unit in units:
        vector, summary = gradient_for_unit(
            unit=unit,
            latent_by_image=latent_by_image,
            hidden_by_prompt=hidden_by_prompt,
            model_input=model_input,
            scheduler=scheduler,
            unet=unet,
            trainable=trainable,
        )
        unit_vectors.append(vector)
        unit_summaries.append(summary)

    # Replay one real-gradient sentinel before any optimizer construction.
    # This is the relevant exactness check: same prepared inputs, model state,
    # and draw policy within this execution context.
    gradient_replay_vector, gradient_replay_summary = gradient_for_unit(
        unit=units[0],
        latent_by_image=latent_by_image,
        hidden_by_prompt=hidden_by_prompt,
        model_input=model_input,
        scheduler=scheduler,
        unet=unet,
        trainable=trainable,
    )
    require(
        tensor_sha256(gradient_replay_vector) == tensor_sha256(unit_vectors[0]),
        "real-gradient sentinel vector did not replay exactly",
    )
    require(
        gradient_replay_summary["gradient_sha256"] == unit_summaries[0]["gradient_sha256"]
        and gradient_replay_summary["gradient_l2_norm"] == unit_summaries[0]["gradient_l2_norm"],
        "real-gradient sentinel summary did not replay exactly",
    )
    gradient_replay = {
        "unit_commitment": unit_summaries[0]["unit_commitment"],
        "gradient_sha256": unit_summaries[0]["gradient_sha256"],
        "vector_sha256": tensor_sha256(unit_vectors[0]),
        "repeat_exact": True,
    }
    del gradient_replay_vector

    template = torch.zeros(TRAINABLE_PARAMETERS, device="cuda", dtype=torch.float32)
    noised_gradient, private_diagnostics = aggregate_noised_update(
        unit_vectors,
        template=template,
        config=config,
        generator=torch.Generator(device="cuda").manual_seed(dp_noise_seed),
    )
    replay, replay_diagnostics = aggregate_noised_update(
        unit_vectors,
        template=template,
        config=config,
        generator=torch.Generator(device="cuda").manual_seed(dp_noise_seed),
    )
    require(tensor_sha256(noised_gradient) == tensor_sha256(replay), "TEST_ONLY DP-noise replay mismatch")
    require(
        private_diagnostics["noise_sha256"] == replay_diagnostics["noise_sha256"],
        "TEST_ONLY noise digest replay mismatch",
    )
    require(
        all(summary["frozen_calibration_gradient_l2_norm"] > config.clip_norm for summary in unit_summaries),
        "a selected calibration sentinel did not originally cross the clip norm",
    )
    require(
        all(summary["gradient_l2_norm"] > config.clip_norm for summary in unit_summaries),
        "a selected real-gradient sentinel did not cross the clip norm",
    )
    require(
        private_diagnostics["clipped_unit_count"] == len(units),
        "real-gradient clipping branch was not taken for every sentinel",
    )

    before = flatten_named_tensors(trainable).detach().cpu()
    assign_flat_gradient(trainable, noised_gradient)
    assigned = flatten_named_tensors([(name, parameter.grad) for name, parameter in trainable])
    require(tensor_sha256(assigned) == tensor_sha256(noised_gradient), "assigned gradient drift")
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=OPTIMIZER["learning_rate"],
        betas=tuple(OPTIMIZER["betas"]),
        eps=OPTIMIZER["epsilon"],
        weight_decay=OPTIMIZER["weight_decay"],
    )
    optimizer.step()
    after = flatten_named_tensors(trainable).detach().cpu()
    final_digest = named_tensor_digest(trainable)
    delta = after - before
    delta_norm = float(torch.linalg.vector_norm(delta.double()))
    changed = int(torch.count_nonzero(delta).item())
    require(initial_digest != final_digest, "optimizer step did not change adapter digest")
    require(delta_norm > 0.0 and changed > 0, "optimizer step did not change adapter values")
    require(bool(torch.isfinite(after).all()), "adapter became non-finite")
    peak_memory = int(torch.cuda.max_memory_allocated())

    result = {
        "arm": config.arm,
        "privacy_unit": config.privacy_unit,
        "selected_units": len(units),
        "selected_images": sum(len(unit["records"]) for unit in units),
        "unit_summaries": unit_summaries,
        "real_gradient_sentinel_replay": gradient_replay,
        "aggregation_private_diagnostics_public_data_only": private_diagnostics,
        "noised_gradient_sha256": tensor_sha256(noised_gradient),
        "test_noise_replay_exact": True,
        "adapter_initial_sha256": initial_digest,
        "adapter_final_sha256": final_digest,
        "adapter_delta_l2_norm": delta_norm,
        "changed_trainable_scalars": changed,
        "optimizer_step": True,
        "parameters_finite_after_step": True,
        "checkpoint_written": False,
        "peak_cuda_memory_bytes": peak_memory,
    }

    del optimizer, assigned, delta, after, before, replay, noised_gradient, template
    del unit_vectors, trainable, unet
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    from dp_protocol.calibrate_xray_public_clip_norms import (
        MODEL_HASHES,
        disable_broken_optional_onnx,
        prepare_latents_and_text,
        verify_snapshot,
    )

    disable_broken_optional_onnx()
    import numpy as np
    import torch
    from diffusers import DDPMScheduler
    from huggingface_hub import snapshot_download

    from dp_training.mechanism import (
        MechanismConfig,
        append_public_trace,
        make_public_scheduled_event,
        verify_public_trace,
    )

    require(torch.cuda.is_available(), "CUDA is required for public LoRA smoke")
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(LORA_INIT_SEED)
    torch.cuda.manual_seed_all(LORA_INIT_SEED)

    manifest = args.manifest.resolve()
    protocol_path = args.protocol.resolve()
    preprocessing_path = args.preprocessing_contract.resolve()
    calibration_report_path = args.calibration_report.resolve()
    calibration_gradients_path = args.calibration_gradients.resolve()
    image_root = args.image_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    require(sha256_file(manifest) == K10_SHA256, "K10 manifest hash mismatch")
    require(sha256_file(protocol_path) == PROTOCOL_SHA256, "protocol hash mismatch")
    require(
        sha256_file(preprocessing_path) == PREPROCESSING_CONTRACT_SHA256,
        "preprocessing contract hash mismatch",
    )
    require(
        sha256_file(calibration_report_path) == CALIBRATION_REPORT_SHA256,
        "public calibration report hash mismatch",
    )
    require(
        sha256_file(calibration_gradients_path) == CALIBRATION_GRADIENTS_SHA256,
        "public calibration gradient table hash mismatch",
    )

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "data_pipeline"))
    import nih_cxr14_model_input as model_input

    records = model_input.read_manifest(manifest, partitions={"public_development"})
    require(len(records) == 4_831, "public-development image count mismatch")
    require(len({record.patient_id for record in records}) == 1_816, "public-development patient count mismatch")
    with calibration_gradients_path.open("r", encoding="utf-8", newline="") as handle:
        calibration_rows = list(csv.DictReader(handle))
    require(len(calibration_rows) == 144, "public calibration row count mismatch")
    image_units, patient_units = select_public_units(records, calibration_rows)
    require(len(image_units) == 2 and len(patient_units) == 2, "smoke selection size mismatch")
    require(sum(len(unit["records"]) for unit in patient_units) == 8, "M2 smoke image count mismatch")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    entries = protocol["accounting"]["entries"]
    image_entry = next(item for item in entries if item["arm"] == "M1-I8" and item["cap"] == 10)
    patient_entry = next(item for item in entries if item["arm"] == "M2-P8" and item["cap"] == 10)
    image_config = MechanismConfig(
        arm="M1-I8",
        privacy_unit="image",
        population=int(image_entry["population"]),
        expected_batch=int(image_entry["expected_batch"]),
        poisson_sample_rate=float(image_entry["poisson_sample_rate"]),
        clip_norm=CLIP_NORMS["image"],
        noise_multiplier=float(image_entry["noise_multiplier"]),
        fixed_denominator=float(image_entry["expected_batch"]),
        max_steps=int(image_entry["max_steps"]),
        rng_security_mode="TEST_ONLY_DETERMINISTIC",
        rng_backend="torch_cuda_seeded_public_smoke_only",
    )
    patient_config = MechanismConfig(
        arm="M2-P8",
        privacy_unit="patient",
        population=int(patient_entry["population"]),
        expected_batch=int(patient_entry["expected_batch"]),
        poisson_sample_rate=float(patient_entry["poisson_sample_rate"]),
        clip_norm=CLIP_NORMS["patient"],
        noise_multiplier=float(patient_entry["noise_multiplier"]),
        fixed_denominator=float(patient_entry["expected_batch"]),
        max_steps=int(patient_entry["max_steps"]),
        rng_security_mode="TEST_ONLY_DETERMINISTIC",
        rng_backend="torch_cuda_seeded_public_smoke_only",
    )

    snapshot = Path(
        snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    ).resolve()
    require(snapshot.name == MODEL_REVISION, "snapshot revision mismatch")
    snapshot_hashes = verify_snapshot(snapshot)
    require(set(snapshot_hashes) == set(MODEL_HASHES), "critical snapshot hash set mismatch")

    started = time.perf_counter()
    latent_by_image, hidden_by_prompt, input_summary = prepare_latents_and_text(
        snapshot,
        image_units + patient_units,
        image_root,
        model_input,
    )
    scheduler = DDPMScheduler.from_pretrained(snapshot / "scheduler", local_files_only=True)
    m1 = run_arm(
        units=image_units,
        config=image_config,
        dp_noise_seed=M1_DP_NOISE_TEST_SEED,
        snapshot=snapshot,
        latent_by_image=latent_by_image,
        hidden_by_prompt=hidden_by_prompt,
        model_input=model_input,
        scheduler=scheduler,
    )
    m2 = run_arm(
        units=patient_units,
        config=patient_config,
        dp_noise_seed=M2_DP_NOISE_TEST_SEED,
        snapshot=snapshot,
        latent_by_image=latent_by_image,
        hidden_by_prompt=hidden_by_prompt,
        model_input=model_input,
        scheduler=scheduler,
    )
    require(
        m1["adapter_initial_sha256"] == m2["adapter_initial_sha256"],
        "M1/M2 did not start from identical LoRA parameters",
    )

    public_trace: list[dict[str, Any]] = []
    append_public_trace(public_trace, make_public_scheduled_event(image_config, first_step=1, event_count=1))
    append_public_trace(public_trace, make_public_scheduled_event(patient_config, first_step=1, event_count=1))
    trace_summary = verify_public_trace(public_trace)
    elapsed = time.perf_counter() - started

    preexisting_checkpoint_files = [
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors", ".bin"}
    ]
    require(not preexisting_checkpoint_files, "checkpoint-like output exists in smoke directory")
    report = {
        "schema": SCHEMA,
        "status": "PASS_PUBLIC_DISPOSABLE_ONE_STEP_CONFORMANCE",
        "scope": "PUBLIC_DEVELOPMENT_ONLY_DISPOSABLE_MODEL_NOT_PRIVATE_TRAINING_NOT_DP_RELEASE",
        "frozen_inputs": {
            "k10_manifest_sha256": K10_SHA256,
            "protocol_sha256": PROTOCOL_SHA256,
            "preprocessing_contract_sha256": PREPROCESSING_CONTRACT_SHA256,
            "public_clip_calibration_report_sha256": CALIBRATION_REPORT_SHA256,
            "public_clip_calibration_gradients_sha256": CALIBRATION_GRADIENTS_SHA256,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "critical_model_hashes": snapshot_hashes,
            "profile": "P256",
            "lora_init_seed": LORA_INIT_SEED,
            "lora_rank": LORA_RANK,
            "lora_targets": list(LORA_TARGETS),
            "trainable_parameters": TRAINABLE_PARAMETERS,
            "trainable_dtype": "torch.float32",
        },
        "selection": {
            "partition": "public_development",
            "population_images": len(records),
            "population_patients": len({record.patient_id for record in records}),
            "salt": SELECTION_SALT,
            "rule": "public calibration maximum gradient norm within each target/control stratum; M2 restricted to exactly four-image units; all four patients disjoint",
            "role": "outcome-dependent clipping-path coverage sentinel only; prohibited for utility, privacy-attack, or generalization evaluation",
            "cross_run_norm_policy": "frozen calibration and smoke norms are both reported but need not be bit-identical because their public VAE/text preparation batch contexts differ; each smoke norm must independently exceed frozen C",
            "M1_unit_commitments": [unit_commitment(unit) for unit in image_units],
            "M2_unit_commitments": [unit_commitment(unit) for unit in patient_units],
            "M1_target_control": [int(unit["target_patient"]) for unit in image_units],
            "M2_target_control": [int(unit["target_patient"]) for unit in patient_units],
            "uses_private_train": False,
            "uses_privacy_attack_holdout": False,
            "uses_final_test": False,
        },
        "input_summary": input_summary,
        "mechanism_configs": {"M1-I8": image_config.public_dict(), "M2-P8": patient_config.public_dict()},
        "optimizer": OPTIMIZER,
        "arms": {"M1-I8": m1, "M2-P8": m2},
        "identical_initial_adapter_across_arms": True,
        "public_schedule_trace": public_trace,
        "public_schedule_trace_summary": trace_summary,
        "model_lifecycle": {
            "fresh_model_per_arm": True,
            "optimizer_steps_per_arm": 1,
            "checkpoint_written": False,
            "model_discarded_after_arm": True,
            "checkpoint_like_files_in_output": preexisting_checkpoint_files,
        },
        "pregate_revision_history": [
            {
                "report_sha256": "4B452CCA01B84CDA10F6CFEC6F7A2A72905FFF2877F34DB212B3C7A10F6A73C2",
                "status": "PASS_PUBLIC_DISPOSABLE_ONE_STEP_CONFORMANCE",
                "observed_clipped_units": {"M1-I8": 0, "M2-P8": 0},
                "reason_for_replacement": "stable-hash-selected public units validated gradients, noise, fixed denominators, and optimizer updates but did not exercise the real-gradient clipping branch",
                "disposition": "superseded before private training by public calibration sentinels selected solely for clipping-path coverage",
            }
        ],
        "randomness_boundary": {
            "classification": "TEST_ONLY_DETERMINISTIC",
            "purpose": "exact conformance replay on public data",
            "formal_privacy_evidence": False,
            "release_eligible": False,
            "test_seed_values_emitted_in_report": False,
            "note": "deterministic test seeds are public implementation constants and must not be reused for private training",
        },
        "claim_limits": [
            "This proves executable one-step compatibility, not private-run privacy.",
            "The fixed smoke selection is not a replacement for Poisson sampling; Poisson semantics are validated by the synthetic gate and will be used in private training.",
            "The sentinel selection deliberately uses public calibration gradients to cover clipping and is invalid for evaluation.",
            "The deterministic CUDA generator is not cryptographically secure and cannot support a release claim.",
            "No generated clinical image, trained adapter, or checkpoint is retained.",
        ],
        "elapsed_seconds": elapsed,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "packages": {
                "diffusers": package_version("diffusers"),
                "peft": package_version("peft"),
                "transformers": package_version("transformers"),
            },
        },
    }
    report_path = output_dir / "report.json"
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "elapsed_seconds": elapsed,
                "M1_delta_l2": m1["adapter_delta_l2_norm"],
                "M2_delta_l2": m2["adapter_delta_l2_norm"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
