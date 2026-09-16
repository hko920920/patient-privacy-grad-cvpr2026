#!/usr/bin/env python3
"""Calibrate image- and patient-gradient clip norms on public development data.

The diagnostic loads the pinned SD 2.1 model and computes real LoRA gradients,
but it creates no optimizer, takes no optimizer step, adds no DP noise, and
touches no private-train or attack-holdout record.  A fixed grid is selected by
the clipping fraction closest to 20 percent.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "nih-cxr14-public-clip-calibration/v1"
MODEL_ID = "Manojb/stable-diffusion-2-1-base"
MODEL_REVISION = "0094d483a120f3f33dafbd187ea4aa60d10de75c"
MODEL_VARIANT = "fp16"
K10_SHA256 = "2D749FB7B70823114A69FD55921277B0D0FF2F9AEC8FECD239159C553F3A0C06"
PREPROCESSING_CONTRACT_SHA256 = (
    "0ED9E434764BDB2D542BB16AB951F856CB5B82D4F2E6B9F25F6C554F1076669D"
)
MODEL_HASHES = {
    "text_encoder/model.fp16.safetensors": "681C555376658C81DC273F2D737A2AEB23DDB6D1D8E5B3A7064636D359A22668",
    "unet/diffusion_pytorch_model.fp16.safetensors": "28EC9CF3B239C0751C201B1F6FB46B551DF5862731B30A37AA1360101CB3FBAB",
    "vae/diffusion_pytorch_model.fp16.safetensors": "3E4C08995484EE61270175E9E7A072B66A6E4EEB5F0C266667FE1F45B90DAF9A",
}
SELECTION_SALT = "nih-cxr14-public-clip-calibration-v1"
LORA_INIT_SEED = 260903
LORA_RANK = 8
LORA_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")
IMAGE_UNITS_PER_CELL = 12
PATIENT_UNITS_PER_CELL = 12
PATIENT_MAX_IMAGES = 4
CLIP_GRID = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
TARGET_CLIP_FRACTION = 0.20
IMAGE_SIZE = 256
PREFREEZE_REVISION_HISTORY = [
    {
        "report_sha256": "3C2F5F05329CBB7058FCE72D12BF2B4EB9DE161B8FF95912CBB829A1FCDD1E55",
        "image_units": 72,
        "patient_units": 24,
        "image_clip_norm": 0.5,
        "image_clip_fraction": 2 / 72,
        "patient_clip_norm": 0.2,
        "patient_clip_fraction": 3 / 24,
        "reason": "the initial seven-point grid was too coarse to realize the declared 20% image clipping target, and trainable LoRA parameters were observed in fp16",
        "disposition": "rejected before private training; replaced by exact public p80 selection plus explicit fp32 trainable-parameter casting",
    },
    {
        "report_sha256": "2EEFC666AA5E9ED3910CA398AC0225E223BF82F8DA9C04CA2B1A6F516E6C4D56",
        "image_units": 72,
        "patient_units": 24,
        "image_clip_norm": 0.28448700606156724,
        "patient_clip_norm": 0.15435489901375093,
        "reason": "the corrected mechanism passed, but the patient p80 used one third as many units as the image p80",
        "disposition": "superseded before private training by a symmetric 72-image/72-patient public calibration",
    },
]


def default_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parent.parent
    return {
        "root": root,
        "manifest": root
        / "_data"
        / "derived"
        / "nih_cxr14_pa_target_enriched_v1"
        / "k10_private.csv",
        "image_root": root
        / "_data"
        / "raw"
        / "nih_cxr14_pa_k10_plus_census_v1"
        / "images",
        "preprocessing_contract": root
        / "_reports"
        / "nih_cxr14_sd21_ppmark_interface_v1_001"
        / "preprocessing_contract.json",
        "output": root / "_reports" / "nih_cxr14_public_clip_calibration_v1_001",
    }


def parse_args() -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=paths["manifest"])
    parser.add_argument("--image-root", type=Path, default=paths["image_root"])
    parser.add_argument(
        "--preprocessing-contract", type=Path, default=paths["preprocessing_contract"]
    )
    parser.add_argument("--output-dir", type=Path, default=paths["output"])
    return parser.parse_args()


def disable_broken_optional_onnx() -> None:
    original = importlib.util.find_spec

    def guarded(name: str, package: str | None = None):
        if name == "onnxruntime" or name.startswith("onnxruntime."):
            return None
        return original(name, package)

    importlib.util.find_spec = guarded


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def stable_hash(*parts: object) -> str:
    return sha256_bytes("|".join(map(str, parts)).encode("utf-8"))


def hash_uint63(*parts: object) -> int:
    return int(stable_hash(*parts)[:16], 16) % (2**63)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def contribution_bucket(count: int) -> str:
    if count == 1:
        return "n1"
    if count <= 3:
        return "n2_3"
    return "n4_10"


def select_units(records: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_patient: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        by_patient[record.patient_id].append(record)
    for patient_records in by_patient.values():
        patient_records.sort(key=lambda record: (record.cap_rank, record.image_id))

    cells: dict[tuple[int, str], list[str]] = defaultdict(list)
    for patient, patient_records in by_patient.items():
        target = patient_records[0].target_patient
        require(all(record.target_patient == target for record in patient_records), "target flag changes within patient")
        cells[(target, contribution_bucket(len(patient_records)))].append(patient)

    image_units: list[dict[str, Any]] = []
    patient_units: list[dict[str, Any]] = []
    for target in (0, 1):
        for bucket in ("n1", "n2_3", "n4_10"):
            candidates = cells[(target, bucket)]
            require(len(candidates) >= IMAGE_UNITS_PER_CELL, f"insufficient patients in {target}/{bucket}")
            ranked = sorted(
                candidates,
                key=lambda patient: (stable_hash(SELECTION_SALT, "cell", target, bucket, patient), patient),
            )
            image_patients = ranked[:IMAGE_UNITS_PER_CELL]
            patient_patients = ranked[IMAGE_UNITS_PER_CELL : IMAGE_UNITS_PER_CELL + PATIENT_UNITS_PER_CELL]
            require(len(patient_patients) == PATIENT_UNITS_PER_CELL, "patient calibration sample incomplete")
            for patient in image_patients:
                record = min(
                    by_patient[patient],
                    key=lambda item: (stable_hash(SELECTION_SALT, "image", patient, item.image_id), item.image_id),
                )
                image_units.append(
                    {
                        "unit_type": "image",
                        "unit_id": record.image_id,
                        "patient_id": patient,
                        "target_patient": target,
                        "contribution_bucket": bucket,
                        "records": [record],
                    }
                )
            for patient in patient_patients:
                chosen_records = sorted(
                    by_patient[patient],
                    key=lambda item: (stable_hash(SELECTION_SALT, "within_patient", patient, item.image_id), item.image_id),
                )[:PATIENT_MAX_IMAGES]
                patient_units.append(
                    {
                        "unit_type": "patient",
                        "unit_id": patient,
                        "patient_id": patient,
                        "target_patient": target,
                        "contribution_bucket": bucket,
                        "records": chosen_records,
                    }
                )
    require(len(image_units) == 72, "image calibration unit count mismatch")
    require(len(patient_units) == 72, "patient calibration unit count mismatch")
    require(
        {unit["patient_id"] for unit in image_units}.isdisjoint(
            {unit["patient_id"] for unit in patient_units}
        ),
        "image/patient calibration patients overlap",
    )
    return image_units, patient_units


def write_selected_units(path: Path, units: list[dict[str, Any]]) -> None:
    fields = [
        "unit_type",
        "unit_id",
        "patient_id",
        "target_patient",
        "contribution_bucket",
        "image_count",
        "image_ids",
        "selection_commitment",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for unit in units:
            image_ids = [record.image_id for record in unit["records"]]
            writer.writerow(
                {
                    "unit_type": unit["unit_type"],
                    "unit_id": unit["unit_id"],
                    "patient_id": unit["patient_id"],
                    "target_patient": unit["target_patient"],
                    "contribution_bucket": unit["contribution_bucket"],
                    "image_count": len(image_ids),
                    "image_ids": "|".join(image_ids),
                    "selection_commitment": stable_hash(
                        SELECTION_SALT,
                        unit["unit_type"],
                        unit["unit_id"],
                        *image_ids,
                    ),
                }
            )


def verify_snapshot(snapshot: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for relative, expected in MODEL_HASHES.items():
        path = snapshot / relative
        require(path.is_file(), f"missing model file: {relative}")
        actual = sha256_file(path)
        require(actual == expected, f"model hash mismatch: {relative}")
        result[relative] = {"expected": expected, "actual": actual, "status": "PASS"}
    return result


def tensor_digest(named_tensors: list[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        array = tensor.detach().float().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tuple(array.shape)).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest().upper()


def prepare_latents_and_text(
    snapshot: Path,
    units: list[dict[str, Any]],
    image_root: Path,
    model_input: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import torch
    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel, CLIPTokenizer

    record_by_id = {
        record.image_id: record
        for unit in units
        for record in unit["records"]
    }
    prompts = sorted({model_input.prompt_for_record(record) for record in record_by_id.values()})
    tokenizer = CLIPTokenizer.from_pretrained(snapshot / "tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(
        snapshot / "text_encoder",
        torch_dtype=torch.float16,
        variant=MODEL_VARIANT,
        use_safetensors=True,
        local_files_only=True,
    ).eval().to("cuda")
    hidden_by_prompt: dict[str, Any] = {}
    with torch.inference_mode():
        for start in range(0, len(prompts), 16):
            batch_prompts = prompts[start : start + 16]
            tokens = tokenizer(
                batch_prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            hidden = text_encoder(tokens.input_ids.to("cuda"))[0].detach().cpu()
            for prompt, value in zip(batch_prompts, hidden):
                hidden_by_prompt[prompt] = value.unsqueeze(0).contiguous()
    del text_encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    vae = AutoencoderKL.from_pretrained(
        snapshot / "vae",
        torch_dtype=torch.float16,
        variant=MODEL_VARIANT,
        use_safetensors=True,
        local_files_only=True,
    ).eval().to("cuda")
    records = [record_by_id[key] for key in sorted(record_by_id)]
    latent_by_image: dict[str, Any] = {}
    with torch.inference_mode():
        for start in range(0, len(records), 4):
            batch_records = records[start : start + 4]
            pixels = []
            for record in batch_records:
                image, _ = model_input.preprocess_path(
                    image_root / record.image_filename, IMAGE_SIZE
                )
                pixels.append(model_input.pil_to_normalized_tensor(image))
            pixel_batch = torch.stack(pixels).to(device="cuda", dtype=torch.float16)
            latents = vae.encode(pixel_batch).latent_dist.mode()
            latents = latents * float(vae.config.scaling_factor)
            require(bool(torch.isfinite(latents).all()), "non-finite public latent")
            for record, latent in zip(batch_records, latents.detach().cpu()):
                latent_by_image[record.image_id] = latent.unsqueeze(0).contiguous()
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    summary = {
        "unique_images": len(record_by_id),
        "unique_prompts": len(prompts),
        "latent_shape": [1, 4, IMAGE_SIZE // 8, IMAGE_SIZE // 8],
        "vae_sampling": "latent_dist.mode",
    }
    return latent_by_image, hidden_by_prompt, summary


def gradient_for_unit(
    *,
    unit: dict[str, Any],
    latent_by_image: dict[str, Any],
    hidden_by_prompt: dict[str, Any],
    model_input: Any,
    scheduler: Any,
    unet: Any,
    trainable: list[tuple[str, Any]],
    include_digest: bool,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    latents = []
    hidden = []
    noises = []
    timesteps = []
    seeds = []
    for index, record in enumerate(unit["records"]):
        latent = latent_by_image[record.image_id].to(device="cuda", dtype=torch.float16)
        prompt = model_input.prompt_for_record(record)
        text = hidden_by_prompt[prompt].to(device="cuda", dtype=torch.float16)
        seed = hash_uint63(SELECTION_SALT, "noise", unit["unit_type"], unit["unit_id"], index, record.image_id)
        timestep = hash_uint63(SELECTION_SALT, "timestep", unit["unit_type"], unit["unit_id"], index, record.image_id) % int(scheduler.config.num_train_timesteps)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        noise = torch.randn(latent.shape, generator=generator, device="cuda", dtype=torch.float16)
        latents.append(latent)
        hidden.append(text)
        noises.append(noise)
        timesteps.append(int(timestep))
        seeds.append(seed)

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
        per_image = functional.mse_loss(
            prediction.float(), target.float(), reduction="none"
        ).flatten(1).mean(1)
        loss = per_image.mean()
    loss.backward()
    norm_squared = 0.0
    missing = 0
    finite = True
    gradients: list[tuple[str, Any]] = []
    for name, parameter in trainable:
        if parameter.grad is None:
            missing += 1
            gradient = torch.zeros_like(parameter, dtype=torch.float32)
        else:
            gradient = parameter.grad.detach().float()
            finite = finite and bool(torch.isfinite(gradient).all())
            norm_squared += float(torch.sum(gradient.double() ** 2).cpu())
        if include_digest:
            gradients.append((name, gradient))
    require(finite and missing == 0, f"invalid gradient for {unit['unit_type']} {unit['unit_id']}")
    return {
        "unit_type": unit["unit_type"],
        "unit_id": unit["unit_id"],
        "patient_id": unit["patient_id"],
        "target_patient": unit["target_patient"],
        "contribution_bucket": unit["contribution_bucket"],
        "image_count": len(unit["records"]),
        "image_ids": "|".join(record.image_id for record in unit["records"]),
        "timesteps": "|".join(map(str, timesteps)),
        "noise_seed_commitments": "|".join(
            stable_hash("public-calibration-seed", seed) for seed in seeds
        ),
        "loss": float(loss.detach().cpu()),
        "gradient_l2_norm": math.sqrt(norm_squared),
        "gradient_sha256": tensor_digest(gradients) if include_digest else "",
        "finite": True,
        "missing_gradient_tensors": missing,
    }


def choose_clip(norms: list[float]) -> tuple[float, list[dict[str, Any]]]:
    import numpy as np

    table = [
        {
            "clip_norm": clip,
            "clipped_units": sum(norm > clip for norm in norms),
            "units": len(norms),
            "clip_fraction": sum(norm > clip for norm in norms) / len(norms),
        }
        for clip in CLIP_GRID
    ]
    selected = float(
        np.quantile(np.asarray(norms, dtype=np.float64), 0.80, method="higher")
    )
    return selected, table


def distribution(values: list[float]) -> dict[str, float]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25, method="linear")),
        "median": float(np.quantile(array, 0.50, method="linear")),
        "p75": float(np.quantile(array, 0.75, method="linear")),
        "p80": float(np.quantile(array, 0.80, method="linear")),
        "p90": float(np.quantile(array, 0.90, method="linear")),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def write_gradient_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "unit_type",
        "unit_id",
        "patient_id",
        "target_patient",
        "contribution_bucket",
        "image_count",
        "image_ids",
        "timesteps",
        "noise_seed_commitments",
        "loss",
        "gradient_l2_norm",
        "gradient_sha256",
        "finite",
        "missing_gradient_tensors",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    disable_broken_optional_onnx()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    require(sha256_file(args.manifest.resolve()) == K10_SHA256, "K10 manifest hash mismatch")
    require(
        sha256_file(args.preprocessing_contract.resolve()) == PREPROCESSING_CONTRACT_SHA256,
        "preprocessing contract hash mismatch",
    )

    import numpy as np
    import torch
    from diffusers import DDPMScheduler, UNet2DConditionModel
    from huggingface_hub import snapshot_download
    from peft import LoraConfig
    from peft.utils.other import cast_mixed_precision_params

    require(torch.cuda.is_available(), "CUDA is required for public gradient calibration")
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(LORA_INIT_SEED)
    torch.cuda.manual_seed_all(LORA_INIT_SEED)

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "data_pipeline"))
    import nih_cxr14_model_input as model_input

    records = model_input.read_manifest(
        args.manifest.resolve(), partitions={"public_development"}
    )
    require(len(records) == 4_831, "public-development image count mismatch")
    require(len({record.patient_id for record in records}) == 1_816, "public-development patient count mismatch")
    image_units, patient_units = select_units(records)
    selected_path = output_dir / "selected_public_units_private.csv"
    write_selected_units(selected_path, image_units + patient_units)

    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
    ).resolve()
    require(snapshot.name == MODEL_REVISION, "snapshot revision mismatch")
    snapshot_hashes = verify_snapshot(snapshot)
    latent_by_image, hidden_by_prompt, input_summary = prepare_latents_and_text(
        snapshot,
        image_units + patient_units,
        args.image_root.resolve(),
        model_input,
    )

    scheduler = DDPMScheduler.from_pretrained(
        snapshot / "scheduler", local_files_only=True
    )
    torch.manual_seed(LORA_INIT_SEED)
    torch.cuda.manual_seed_all(LORA_INIT_SEED)
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
    trainable = [
        (name, parameter)
        for name, parameter in unet.named_parameters()
        if parameter.requires_grad
    ]
    require(sum(parameter.numel() for _, parameter in trainable) == 1_659_904, "trainable parameter count mismatch")
    trainable_dtypes = sorted({str(parameter.dtype) for _, parameter in trainable})
    require(trainable_dtypes == ["torch.float32"], "LoRA trainable parameters are not fp32")
    adapter_before = tensor_digest(trainable)

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for index, unit in enumerate(image_units + patient_units):
        rows.append(
            gradient_for_unit(
                unit=unit,
                latent_by_image=latent_by_image,
                hidden_by_prompt=hidden_by_prompt,
                model_input=model_input,
                scheduler=scheduler,
                unet=unet,
                trainable=trainable,
                include_digest=index in {0, len(image_units), len(image_units) + len(patient_units) - 1},
            )
        )

    # Replay one image and one patient sentinel exactly after all other gradients.
    replay_indices = [0, len(image_units)]
    replay_checks = []
    for index in replay_indices:
        original = rows[index]
        replay = gradient_for_unit(
            unit=(image_units + patient_units)[index],
            latent_by_image=latent_by_image,
            hidden_by_prompt=hidden_by_prompt,
            model_input=model_input,
            scheduler=scheduler,
            unet=unet,
            trainable=trainable,
            include_digest=True,
        )
        require(replay["gradient_sha256"] == original["gradient_sha256"], "gradient replay digest mismatch")
        require(replay["gradient_l2_norm"] == original["gradient_l2_norm"], "gradient replay norm mismatch")
        replay_checks.append(
            {
                "unit_type": original["unit_type"],
                "unit_id": original["unit_id"],
                "gradient_sha256": original["gradient_sha256"],
                "repeat_exact": True,
            }
        )
    adapter_after = tensor_digest(trainable)
    require(adapter_before == adapter_after, "LoRA adapter changed without optimizer step")
    elapsed = time.perf_counter() - started

    gradient_path = output_dir / "gradient_norms_private.csv"
    write_gradient_csv(gradient_path, rows)
    image_norms = [row["gradient_l2_norm"] for row in rows if row["unit_type"] == "image"]
    patient_norms = [row["gradient_l2_norm"] for row in rows if row["unit_type"] == "patient"]
    image_clip, image_table = choose_clip(image_norms)
    patient_clip, patient_table = choose_clip(patient_norms)
    selection_rule = {
        "grid": list(CLIP_GRID),
        "grid_role": "reported sensitivity table only; it does not choose C",
        "target_clip_fraction": TARGET_CLIP_FRACTION,
        "choice": "exact empirical p80 with numpy quantile method='higher'; C is the selected observed norm",
        "data": "frozen K10 public_development only",
        "stratification": "target/control crossed with patient K10 contribution buckets n1, n2-3, n4-10",
        "model_state": "pinned base plus fixed freshly initialized rank-8 LoRA; no optimizer",
    }
    report = {
        "schema": SCHEMA,
        "status": "PASS_PUBLIC_CLIP_CALIBRATION",
        "scope": "PUBLIC_DEVELOPMENT_GRADIENTS_ONLY_NOT_TRAINING_NOT_DP_NOT_ATTACK",
        "frozen_inputs": {
            "k10_manifest_sha256": K10_SHA256,
            "preprocessing_contract_sha256": PREPROCESSING_CONTRACT_SHA256,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "critical_model_hashes": snapshot_hashes,
            "profile": "P256",
            "lora_init_seed": LORA_INIT_SEED,
            "lora_rank": LORA_RANK,
            "lora_targets": list(LORA_TARGETS),
            "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
            "trainable_dtypes_observed": trainable_dtypes,
        },
        "selection": {
            "salt": SELECTION_SALT,
            "image_units": len(image_units),
            "patient_units": len(patient_units),
            "image_units_per_target_contribution_cell": IMAGE_UNITS_PER_CELL,
            "patient_units_per_target_contribution_cell": PATIENT_UNITS_PER_CELL,
            "patient_max_images": PATIENT_MAX_IMAGES,
            "selected_units_file": selected_path.name,
            "selected_units_sha256": sha256_file(selected_path),
            "uses_private_train": False,
            "uses_attack_holdout": False,
            "uses_model_or_attack_outcome": False,
        },
        "input_summary": input_summary,
        "selection_rule": selection_rule,
        "prefreeze_revision_history": PREFREEZE_REVISION_HISTORY,
        "selected_clip_norms": {"image": image_clip, "patient": patient_clip},
        "image_gradient_norms": {
            "distribution": distribution(image_norms),
            "grid": image_table,
        },
        "patient_gradient_norms": {
            "distribution": distribution(patient_norms),
            "grid": patient_table,
        },
        "gradient_rows": gradient_path.name,
        "gradient_rows_sha256": sha256_file(gradient_path),
        "gradient_replay": replay_checks,
        "adapter_before_sha256": adapter_before,
        "adapter_after_sha256": adapter_after,
        "parameters_unchanged": True,
        "optimizer_created": False,
        "optimizer_step": False,
        "gradient_clipping_executed": False,
        "dp_noise_added": False,
        "privacy_accounting_executed": False,
        "elapsed_gradient_seconds": elapsed,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "packages": {
                name: package_version(name)
                for name in ("diffusers", "transformers", "peft", "safetensors")
            },
            "deterministic_algorithms": True,
            "tf32": False,
        },
        "interpretation_limit": "The selected C values are public-data hyperparameters for the frozen pilot mechanism. This is not optimizer training, an executed DP guarantee, attack resistance, utility, or release authorization.",
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("NIH_CXR14_PUBLIC_CLIP_CALIBRATION: PASS_PUBLIC_CLIP_CALIBRATION")
    print(f"image_clip_norm={image_clip}")
    print(f"patient_clip_norm={patient_clip}")
    print(f"report={report_path}")
    print(f"report_sha256={sha256_file(report_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
