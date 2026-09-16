#!/usr/bin/env python3
"""Real-X-ray SD 2.1/PP-Mark interface gate.

This diagnostic fixes and checks raw-to-model preprocessing, prompt mapping,
VAE encoding, a real-image LoRA gradient, and PP-Mark DDIM inversion.  It does
not perform an optimizer update, clipping, DP noise, accounting, training,
attack calibration, a receipt, or a release decision.
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
from collections import Counter
from pathlib import Path
from typing import Any, Callable


SCHEMA = "nih-cxr14-sd21-ppmark-interface-gate/v1"
MODEL_ID = "Manojb/stable-diffusion-2-1-base"
MODEL_REVISION = "0094d483a120f3f33dafbd187ea4aa60d10de75c"
MODEL_VARIANT = "fp16"
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
EXPECTED_INPUT_SHA256 = {
    "k10_manifest": "2D749FB7B70823114A69FD55921277B0D0FF2F9AEC8FECD239159C553F3A0C06",
    "content_inventory": "AD34344F962FAC7052114A3486D42D2B48CB5C9F27DDDC70365E07B715F68CC3",
}
EXPECTED_INVENTORY_CONTENT_SET_SHA256 = (
    "E983A21B9B8558CE38F1AD5BA7C0F6BC51787CC7CBA739399ACE1976E2FB068E"
)
SELECTION_SALT = "nih-cxr14-sd21-ppmark-interface-v1"
PROFILE_SIZES = (256, 512)
LORA_RANK = 8
LORA_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")
GRADIENT_TIMESTEP = 500
GRADIENT_SEED = 260903
PPMARK_STEPS = 2


def disable_broken_optional_onnx() -> None:
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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest().upper()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def cuda_memory(torch: Any) -> dict[str, float]:
    gib = float(1024**3)
    return {
        "allocated_gib": round(torch.cuda.memory_allocated() / gib, 4),
        "reserved_gib": round(torch.cuda.memory_reserved() / gib, 4),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / gib, 4),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / gib, 4),
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
        "inventory": root
        / "_data"
        / "raw"
        / "nih_cxr14_pa_k10_plus_census_v1"
        / "content_inventory_private.csv",
        "image_root": root
        / "_data"
        / "raw"
        / "nih_cxr14_pa_k10_plus_census_v1"
        / "images",
        "output": root / "_reports" / "nih_cxr14_sd21_ppmark_interface_v1_001",
    }


def parse_args() -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=paths["manifest"])
    parser.add_argument("--inventory", type=Path, default=paths["inventory"])
    parser.add_argument("--image-root", type=Path, default=paths["image_root"])
    parser.add_argument("--output-dir", type=Path, default=paths["output"])
    parser.add_argument("--ppmark-steps", type=int, default=PPMARK_STEPS)
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Run deterministic input checks without loading the GPU models.",
    )
    return parser.parse_args()


def verify_inputs(manifest: Path, inventory: Path) -> dict[str, Any]:
    actual = {
        "k10_manifest": sha256_file(manifest),
        "content_inventory": sha256_file(inventory),
    }
    checks = {
        key: {
            "expected_sha256": expected,
            "actual_sha256": actual[key],
            "status": "PASS" if actual[key] == expected else "FAIL",
        }
        for key, expected in EXPECTED_INPUT_SHA256.items()
    }
    failures = [key for key, row in checks.items() if row["status"] != "PASS"]
    if failures:
        raise RuntimeError(f"frozen input identity mismatch: {failures}")
    return checks


def inventory_rows(path: Path) -> tuple[dict[str, dict[str, str]], str]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = row["image_id"].strip()
            if image_id in rows:
                raise ValueError(f"duplicate inventory image: {image_id}")
            rows[image_id] = row
    content_set = hashlib.sha256()
    for image_id in sorted(rows):
        content_set.update(image_id.encode("ascii"))
        content_set.update(b"\x00")
        content_set.update(rows[image_id]["sha256"].upper().encode("ascii"))
        content_set.update(b"\n")
    return rows, content_set.hexdigest().upper()


def select_interface_records(
    records: list[Any],
    inventory: dict[str, dict[str, str]],
) -> list[tuple[str, Any, dict[str, str]]]:
    joined = [(record, inventory[record.image_id]) for record in records]
    predicates: list[tuple[str, Callable[[Any, dict[str, str]], bool]]] = [
        (
            "rgba_positive",
            lambda r, i: i["mode"] == "RGBA" and r.finding_labels != ("No Finding",),
        ),
        (
            "rgba_no_finding",
            lambda r, i: i["mode"] == "RGBA" and r.finding_labels == ("No Finding",),
        ),
        (
            "l_no_finding",
            lambda r, i: i["mode"] == "L" and r.finding_labels == ("No Finding",),
        ),
        (
            "l_pneumothorax",
            lambda r, i: i["mode"] == "L" and "pneumothorax" in r.primary_groups,
        ),
        (
            "l_pneumonia_or_consolidation",
            lambda r, i: i["mode"] == "L"
            and "pneumonia_or_consolidation" in r.primary_groups,
        ),
        (
            "l_pleural_effusion",
            lambda r, i: i["mode"] == "L" and "pleural_effusion" in r.primary_groups,
        ),
        (
            "l_mass_or_nodule",
            lambda r, i: i["mode"] == "L" and "mass_or_nodule" in r.primary_groups,
        ),
        (
            "l_multilabel",
            lambda r, i: i["mode"] == "L" and len(r.finding_labels) >= 3,
        ),
    ]
    selected: list[tuple[str, Any, dict[str, str]]] = []
    selected_patients: set[str] = set()
    for stratum, predicate in predicates:
        candidates = [
            (stable_hash(SELECTION_SALT, stratum, record.image_id), record, row)
            for record, row in joined
            if record.patient_id not in selected_patients and predicate(record, row)
        ]
        if not candidates:
            raise RuntimeError(f"no eligible record for interface stratum {stratum}")
        candidates.sort(key=lambda item: (item[0], item[1].image_id))
        _, record, row = candidates[0]
        selected.append((stratum, record, row))
        selected_patients.add(record.patient_id)
    return selected


def pixel_digest(image: Any) -> str:
    import numpy as np

    array = np.asarray(image, dtype=np.uint8)
    prefix = f"{image.mode}|{image.width}|{image.height}|".encode("ascii")
    return sha256_bytes(prefix + array.tobytes(order="C"))


def tensor_digest(tensor: Any) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    prefix = f"{array.dtype}|{array.shape}|".encode("ascii")
    return sha256_bytes(prefix + array.tobytes(order="C"))


def run_preprocessing(
    selected: list[tuple[str, Any, dict[str, str]]],
    image_root: Path,
    output_dir: Path,
    model_input: Any,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    import numpy as np

    previews = output_dir / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    commitments: dict[str, str] = {}
    for size in PROFILE_SIZES:
        commitment = hashlib.sha256()
        for stratum, record, inventory in selected:
            path = image_root / record.image_filename
            if not path.is_file():
                raise FileNotFoundError(path)
            raw_hash = sha256_file(path)
            if raw_hash != inventory["sha256"].upper():
                raise RuntimeError(f"raw image hash mismatch: {record.image_id}")
            first, metadata = model_input.preprocess_path(path, size)
            second, metadata_again = model_input.preprocess_path(path, size)
            first_pixel_hash = pixel_digest(first)
            second_pixel_hash = pixel_digest(second)
            first_tensor = model_input.pil_to_normalized_tensor(first)
            second_tensor = model_input.pil_to_normalized_tensor(second)
            first_tensor_hash = tensor_digest(first_tensor)
            if (
                metadata != metadata_again
                or first_pixel_hash != second_pixel_hash
                or first_tensor_hash != tensor_digest(second_tensor)
            ):
                raise RuntimeError(f"preprocessing is not deterministic: {record.image_id}")
            array = np.asarray(first, dtype=np.uint8)
            if not (
                np.array_equal(array[:, :, 0], array[:, :, 1])
                and np.array_equal(array[:, :, 0], array[:, :, 2])
            ):
                raise RuntimeError(f"channel replication failed: {record.image_id}")
            prompt = model_input.prompt_for_record(record)
            row = {
                "profile": f"P{size}",
                "resolution": size,
                "stratum": stratum,
                "image_id": record.image_id,
                "patient_id": record.patient_id,
                "partition": record.partition,
                "finding_labels": list(record.finding_labels),
                "primary_groups": list(record.primary_groups),
                "prompt": prompt,
                "raw_sha256": raw_hash,
                "native_mode": metadata["native_mode"],
                "preprocessing": metadata,
                "pixel_sha256": first_pixel_hash,
                "normalized_tensor_sha256": first_tensor_hash,
                "pixel_min": int(array.min()),
                "pixel_max": int(array.max()),
                "pixel_mean": round(float(array[:, :, 0].mean()), 8),
                "normalized_min": round(float(first_tensor.min()), 8),
                "normalized_max": round(float(first_tensor.max()), 8),
                "repeat_exact": True,
            }
            rows.append(row)
            commitment.update(
                (
                    f"{size}|{record.image_id}|{first_pixel_hash}|"
                    f"{first_tensor_hash}|{prompt}\n"
                ).encode("utf-8")
            )
            if stratum in {"rgba_positive", "l_pneumothorax"}:
                preview = previews / f"{stratum}_{record.image_id[:-4]}_{size}.png"
                first.save(
                    preview,
                    format="PNG",
                    optimize=False,
                    compress_level=9,
                )
                row["preview"] = preview.relative_to(output_dir).as_posix()
                row["preview_sha256"] = sha256_file(preview)
        commitments[f"P{size}"] = commitment.hexdigest().upper()
    return rows, commitments


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
    failures = [key for key, row in checks.items() if row["status"] != "PASS"]
    if failures:
        raise RuntimeError(f"critical model artifact mismatch: {failures}")
    return checks


def audit_prompts(records: list[Any], snapshot: Path, model_input: Any) -> dict[str, Any]:
    from transformers import CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(snapshot / "tokenizer", local_files_only=True)
    max_tokens = -1
    max_record = None
    unique_prompts: set[str] = set()
    for record in records:
        prompt = model_input.prompt_for_record(record)
        unique_prompts.add(prompt)
        count = len(tokenizer(prompt, truncation=False)["input_ids"])
        if count > max_tokens:
            max_tokens = count
            max_record = record
    if max_tokens > tokenizer.model_max_length:
        raise RuntimeError(
            f"prompt exceeds tokenizer limit: {max_tokens}>{tokenizer.model_max_length}"
        )
    result = {
        "policy": model_input.PROMPT_POLICY,
        "records_audited": len(records),
        "unique_prompts": len(unique_prompts),
        "model_max_length": int(tokenizer.model_max_length),
        "maximum_token_count": max_tokens,
        "maximum_token_image_id": max_record.image_id if max_record else None,
        "truncated_prompts": 0,
        "uses_age": False,
        "uses_sex": False,
        "uses_patient_id": False,
        "claim_boundary": (
            "Conditions on NIH automatically mined weak labels; prompt text is not a "
            "radiologist-adjudicated diagnosis."
        ),
    }
    del tokenizer
    return result


def encode_text(snapshot: Path, prompt: str) -> Any:
    import torch
    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(snapshot / "tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(
        snapshot / "text_encoder",
        torch_dtype=torch.float16,
        variant=MODEL_VARIANT,
        use_safetensors=True,
        local_files_only=True,
    ).eval().to("cuda")
    tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        hidden = text_encoder(tokens.input_ids.to("cuda"))[0].detach().cpu()
    del text_encoder
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return hidden


def run_vae(
    snapshot: Path,
    selected: list[tuple[str, Any, dict[str, str]]],
    image_root: Path,
    model_input: Any,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], Any]]:
    import torch
    from diffusers import AutoencoderKL

    torch.cuda.empty_cache()
    vae = AutoencoderKL.from_pretrained(
        snapshot / "vae",
        torch_dtype=torch.float16,
        variant=MODEL_VARIANT,
        use_safetensors=True,
        local_files_only=True,
    ).eval().to("cuda")
    chosen = [item for item in selected if item[0] in {"rgba_positive", "l_pneumothorax"}]
    rows: list[dict[str, Any]] = []
    latents: dict[tuple[str, int], Any] = {}
    with torch.inference_mode():
        for size in PROFILE_SIZES:
            for stratum, record, _ in chosen:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                image, _ = model_input.preprocess_path(
                    image_root / record.image_filename, size
                )
                pixels = model_input.pil_to_normalized_tensor(image).unsqueeze(0).to(
                    device="cuda", dtype=torch.float16
                )
                started = time.perf_counter()
                first = vae.encode(pixels).latent_dist.mode()
                first = first * float(vae.config.scaling_factor)
                second = vae.encode(pixels).latent_dist.mode()
                second = second * float(vae.config.scaling_factor)
                torch.cuda.synchronize()
                if not torch.equal(first, second):
                    raise RuntimeError(
                        f"VAE mode encoding is not exact on replay: {record.image_id} P{size}"
                    )
                if not bool(torch.isfinite(first).all()):
                    raise RuntimeError(f"non-finite VAE latent: {record.image_id}")
                expected_shape = (1, 4, size // 8, size // 8)
                if tuple(first.shape) != expected_shape:
                    raise RuntimeError(
                        f"unexpected VAE shape {tuple(first.shape)} != {expected_shape}"
                    )
                cpu = first.detach().cpu()
                latents[(stratum, size)] = cpu
                rows.append(
                    {
                        "profile": f"P{size}",
                        "resolution": size,
                        "stratum": stratum,
                        "image_id": record.image_id,
                        "latent_shape": list(first.shape),
                        "latent_dtype": str(first.dtype),
                        "latent_sha256": tensor_digest(cpu),
                        "latent_mean": round(float(first.float().mean()), 8),
                        "latent_std": round(float(first.float().std()), 8),
                        "finite": True,
                        "repeat_exact": True,
                        "elapsed_sec": round(time.perf_counter() - started, 4),
                        "cuda": cuda_memory(torch),
                    }
                )
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    return rows, latents


def named_tensor_digest(named_tensors: list[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        array = tensor.detach().float().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tuple(array.shape)).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest().upper()


def run_lora_gradient(
    snapshot: Path,
    primary: Any,
    prompt_hidden: Any,
    latents: dict[tuple[str, int], Any],
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    from diffusers import DDPMScheduler, UNet2DConditionModel
    from peft import LoraConfig

    torch.manual_seed(GRADIENT_SEED)
    torch.cuda.manual_seed_all(GRADIENT_SEED)
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
            target_modules=list(LORA_TARGETS),
        )
    )
    unet.enable_gradient_checkpointing()
    unet.train().to("cuda")
    trainable = [(name, parameter) for name, parameter in unet.named_parameters() if parameter.requires_grad]
    adapter_before = named_tensor_digest(trainable)
    parameter_manifest = [
        {"name": name, "shape": list(parameter.shape), "parameters": parameter.numel()}
        for name, parameter in trainable
    ]
    parameter_manifest_sha256 = sha256_bytes(
        (json.dumps(parameter_manifest, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
    )
    rows: list[dict[str, Any]] = []
    prediction_type = str(scheduler.config.prediction_type)
    for size in PROFILE_SIZES:
        latent = latents[("l_pneumothorax", size)].to(
            device="cuda", dtype=torch.float16
        )
        hidden = prompt_hidden.to(device="cuda", dtype=torch.float16)
        seed = GRADIENT_SEED + size
        generator = torch.Generator(device="cuda").manual_seed(seed)
        noise = torch.randn(
            latent.shape, generator=generator, device="cuda", dtype=torch.float16
        )
        timestep = torch.tensor([GRADIENT_TIMESTEP], device="cuda", dtype=torch.long)
        noisy_latent = scheduler.add_noise(latent, noise, timestep)
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "v_prediction":
            target = scheduler.get_velocity(latent, noise, timestep)
        else:
            raise RuntimeError(f"unsupported prediction_type: {prediction_type}")

        repeat_results: list[dict[str, Any]] = []
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(2):
            for _, parameter in trainable:
                parameter.grad = None
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = unet(noisy_latent, timestep, hidden).sample
                loss = functional.mse_loss(prediction.float(), target.float())
            loss.backward()
            gradients: list[tuple[str, Any]] = []
            finite = True
            norm_squared = 0.0
            missing = 0
            for name, parameter in trainable:
                if parameter.grad is None:
                    missing += 1
                    gradient = torch.zeros_like(parameter, dtype=torch.float32)
                else:
                    gradient = parameter.grad.detach().float()
                    finite = finite and bool(torch.isfinite(gradient).all())
                    norm_squared += float(torch.sum(gradient.double() ** 2).cpu())
                gradients.append((name, gradient))
            repeat_results.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "gradient_sha256": named_tensor_digest(gradients),
                    "gradient_l2_norm": math.sqrt(norm_squared),
                    "finite": finite,
                    "missing_gradient_tensors": missing,
                }
            )
        torch.cuda.synchronize()
        if repeat_results[0] != repeat_results[1]:
            raise RuntimeError(f"LoRA gradient is not exact on replay at P{size}")
        if not repeat_results[0]["finite"] or repeat_results[0]["missing_gradient_tensors"]:
            raise RuntimeError(f"invalid LoRA gradient at P{size}")
        rows.append(
            {
                "profile": f"P{size}",
                "resolution": size,
                "image_id": primary.image_id,
                "prompt": None,
                "noise_seed": seed,
                "timestep": GRADIENT_TIMESTEP,
                "prediction_type": prediction_type,
                **repeat_results[0],
                "repeat_exact": True,
                "elapsed_sec_two_backward": round(time.perf_counter() - started, 4),
                "cuda": cuda_memory(torch),
            }
        )
    adapter_after = named_tensor_digest(trainable)
    if adapter_before != adapter_after:
        raise RuntimeError("LoRA adapter parameters changed without an optimizer step")
    result = {
        "status": "PASS",
        "privacy_status": "DIAGNOSTIC_NOT_DP",
        "optimizer_created": False,
        "optimizer_step": False,
        "gradient_clipping": False,
        "dp_noise": False,
        "accounting": False,
        "rank": LORA_RANK,
        "alpha": LORA_RANK,
        "target_modules": list(LORA_TARGETS),
        "gradient_checkpointing": True,
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in unet.parameters()),
        "trainable_tensor_count": len(trainable),
        "parameter_manifest_sha256": parameter_manifest_sha256,
        "adapter_before_sha256": adapter_before,
        "adapter_after_sha256": adapter_after,
        "parameters_unchanged": True,
        "profiles": rows,
    }
    del unet
    del scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_ppmark(
    snapshot: Path,
    selected: list[tuple[str, Any, dict[str, str]]],
    image_root: Path,
    model_input: Any,
    steps: int,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from ppmark_v03.ddim_unet import DiffusersDDIMConfig, DiffusersDDIMInverter

    chosen = [item for item in selected if item[0] in {"rgba_positive", "l_pneumothorax"}]
    primary = next(item for item in chosen if item[0] == "l_pneumothorax")
    initial_prompt = model_input.prompt_for_record(primary[1])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    inverter = DiffusersDDIMInverter(
        DiffusersDDIMConfig(
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            variant=MODEL_VARIANT,
            use_safetensors=True,
            local_files_only=True,
            disable_safety_checker=True,
            device="cuda",
            torch_dtype="float16",
            num_steps=steps,
            guidance_scale=0.0,
            prompt=initial_prompt,
        )
    )
    load_elapsed = time.perf_counter() - load_started
    load_memory = cuda_memory(torch)
    rows: list[dict[str, Any]] = []
    for size in PROFILE_SIZES:
        for stratum, record, _ in chosen:
            image, _ = model_input.preprocess_path(
                image_root / record.image_filename, size
            )
            expected_pixel_hash = pixel_digest(image)
            array = model_input.pil_to_unit_float_array(image)
            prompt = model_input.prompt_for_record(record)
            inverter.prompt = prompt
            inverter.cfg.prompt = prompt
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            first = inverter.invert_latents(array)
            second = inverter.invert_latents(array)
            elapsed = time.perf_counter() - started
            if not np.array_equal(first, second):
                raise RuntimeError(
                    f"PP-Mark inversion is not exact on replay: {record.image_id} P{size}"
                )
            if not bool(np.isfinite(first).all()):
                raise RuntimeError(f"non-finite PP-Mark latent: {record.image_id}")
            expected_shape = (4, size // 8, size // 8)
            if tuple(first.shape) != expected_shape:
                raise RuntimeError(
                    f"unexpected PP-Mark shape {first.shape} != {expected_shape}"
                )
            latent_hash = sha256_bytes(
                f"{first.dtype}|{first.shape}|".encode("ascii")
                + first.tobytes(order="C")
            )
            rows.append(
                {
                    "profile": f"P{size}",
                    "resolution": size,
                    "stratum": stratum,
                    "image_id": record.image_id,
                    "prompt": prompt,
                    "input_pixel_sha256": expected_pixel_hash,
                    "input_array_dtype": str(array.dtype),
                    "input_array_range": [float(array.min()), float(array.max())],
                    "latent_shape": list(first.shape),
                    "latent_dtype": str(first.dtype),
                    "latent_sha256": latent_hash,
                    "latent_mean": round(float(first.mean()), 8),
                    "latent_std": round(float(first.std()), 8),
                    "finite": True,
                    "repeat_exact": True,
                    "elapsed_sec_two_inversions": round(elapsed, 4),
                    "cuda": cuda_memory(torch),
                }
            )
    result = {
        "status": "PASS",
        "scope": "INTERFACE_ONLY_NOT_CALIBRATION_NOT_ROBUSTNESS_NOT_DP",
        "module": "ppmark_v03.ddim_unet",
        "steps": steps,
        "guidance_scale": 0.0,
        "load_spec": inverter.load_spec,
        "load_elapsed_sec": round(load_elapsed, 4),
        "load_cuda": load_memory,
        "profiles": rows,
        "calibration_status": "REQUIRED_FOR_EXACT_MEDICAL_LORA_CHECKPOINT",
    }
    del inverter
    gc.collect()
    torch.cuda.empty_cache()
    return result


def write_selected_csv(
    path: Path, selected: list[tuple[str, Any, dict[str, str]]], model_input: Any
) -> None:
    fields = [
        "stratum",
        "selection_key_sha256",
        "image_id",
        "patient_id",
        "partition",
        "native_mode",
        "finding_labels",
        "primary_groups",
        "prompt",
        "raw_sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for stratum, record, inventory in selected:
            writer.writerow(
                {
                    "stratum": stratum,
                    "selection_key_sha256": stable_hash(
                        SELECTION_SALT, stratum, record.image_id
                    ),
                    "image_id": record.image_id,
                    "patient_id": record.patient_id,
                    "partition": record.partition,
                    "native_mode": inventory["mode"],
                    "finding_labels": "|".join(record.finding_labels),
                    "primary_groups": "|".join(record.primary_groups),
                    "prompt": model_input.prompt_for_record(record),
                    "raw_sha256": inventory["sha256"].upper(),
                }
            )


def main() -> int:
    args = parse_args()
    if args.ppmark_steps <= 0:
        raise ValueError("PP-Mark steps must be positive")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    disable_broken_optional_onnx()

    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from PIL import __version__ as pillow_version

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(root / "ppmark" / "src"))
    import nih_cxr14_model_input as model_input

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_checks = verify_inputs(args.manifest.resolve(), args.inventory.resolve())
    records = model_input.read_manifest(
        args.manifest.resolve(), partitions={"public_development"}
    )
    inventory, inventory_content_set = inventory_rows(args.inventory.resolve())
    if inventory_content_set != EXPECTED_INVENTORY_CONTENT_SET_SHA256:
        raise RuntimeError(
            "ordered inventory content-set commitment mismatch: "
            f"{inventory_content_set}"
        )
    missing_inventory = [record.image_id for record in records if record.image_id not in inventory]
    if missing_inventory:
        raise RuntimeError(f"manifest records absent from inventory: {missing_inventory[:5]}")
    selected = select_interface_records(records, inventory)
    if len({record.patient_id for _, record, _ in selected}) != len(selected):
        raise RuntimeError("interface sample is not patient-distinct")
    mode_counts = Counter(inventory[record.image_id]["mode"] for record in records)

    write_selected_csv(
        output_dir / "selected_interface_records_private.csv", selected, model_input
    )
    preprocessing_rows, profile_commitments = run_preprocessing(
        selected, args.image_root.resolve(), output_dir, model_input
    )
    contract = {
        "schema": model_input.SCHEMA,
        "status": "FROZEN_FOR_PILOT_INTERFACE",
        "source_population": "NIH ChestXray14 PA K10 main cohort",
        "source_geometry": [1024, 1024],
        "accepted_native_modes": {
            "L": "identity grayscale",
            "RGBA": "accepted only if R=G=B and alpha=255; use shared channel",
        },
        "geometry": "full-field square resize; no crop; no pad",
        "resampler": "PIL.Image.Resampling.LANCZOS",
        "reducing_gap": None,
        "channels": "resize one uint8 grayscale channel, then replicate to RGB",
        "normalization": "float32(x / 127.5 - 1.0)",
        "augmentation": "none",
        "private_statistics": "none",
        "profiles": {
            "P256": {
                "resolution": [256, 256],
                "role": "feasibility_and_pilot_primary",
            },
            "P512": {
                "resolution": [512, 512],
                "role": (
                    "confirmatory_extension_only_after_preregistered_medical_utility_"
                    "and_compute_gate"
                ),
            },
        },
        "resolution_selection_rule": (
            "Do not choose a profile from a preferred privacy-attack ordering. Retain P256 "
            "for feasibility; activate P512 only if preregistered medical utility and compute "
            "criteria justify it."
        ),
        "prompt_policy": model_input.PROMPT_POLICY,
        "prompt_inputs": "all automatically mined NIH finding labels; fixed canonical order",
        "prompt_excludes": ["patient_id", "age", "sex", "primary-group target flag"],
        "weak_label_boundary": (
            "No Finding means no label in the NIH mined label set, not clinically normal. "
            "Mass and Nodule are not called cancer."
        ),
        "pillow_version": pillow_version,
        "preprocessing_profile_commitments": profile_commitments,
        "full_cache_policy": "on_the_fly; no duplicate 42,423-image resized corpus",
    }
    contract_path = output_dir / "preprocessing_contract.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PREPROCESS_ONLY" if args.preprocess_only else "RUNNING",
        "scope": (
            "INPUT_AND_INTERFACE_ONLY_NOT_TRAINING_NOT_DP_NOT_UTILITY_NOT_ATTACK_"
            "CALIBRATION_NOT_RELEASE"
        ),
        "frozen_inputs": input_checks,
        "inventory_content_set": {
            "expected_sha256": EXPECTED_INVENTORY_CONTENT_SET_SHA256,
            "actual_sha256": inventory_content_set,
            "status": "PASS",
        },
        "population": {
            "public_development_k10_records": len(records),
            "public_development_k10_patients": len({r.patient_id for r in records}),
            "native_mode_counts": dict(sorted(mode_counts.items())),
        },
        "selection": {
            "salt": SELECTION_SALT,
            "rule": "minimum SHA256(salt|stratum|image_id), distinct patient per stratum",
            "strata": [stratum for stratum, _, _ in selected],
            "records": len(selected),
            "patients": len({record.patient_id for _, record, _ in selected}),
            "selected_csv": "selected_interface_records_private.csv",
            "selected_csv_sha256": sha256_file(
                output_dir / "selected_interface_records_private.csv"
            ),
            "selection_uses_model_or_attack_outcome": False,
        },
        "preprocessing_contract": contract_path.name,
        "preprocessing_contract_sha256": sha256_file(contract_path),
        "preprocessing_profiles": preprocessing_rows,
        "profile_commitments": profile_commitments,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pillow": pillow_version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "deterministic_algorithms": not args.preprocess_only,
            "cudnn_benchmark": False,
            "tf32_matmul": False,
            "tf32_cudnn": False,
            "packages": {
                name: package_version(name)
                for name in (
                    "diffusers",
                    "transformers",
                    "peft",
                    "safetensors",
                    "huggingface-hub",
                )
            },
        },
        "prohibited_operations": {
            "optimizer_created": False,
            "optimizer_step": False,
            "gradient_clipping": False,
            "dp_noise": False,
            "privacy_accounting": False,
            "model_checkpoint_written": False,
            "receipt_created": False,
            "release_image_created": False,
        },
    }
    if not args.preprocess_only:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the full interface gate")
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        snapshot = Path(
            snapshot_download(
                repo_id=MODEL_ID,
                revision=MODEL_REVISION,
                local_files_only=True,
            )
        ).resolve()
        if snapshot.name != MODEL_REVISION:
            raise RuntimeError(f"snapshot revision mismatch: {snapshot.name}")
        report["model"] = {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "variant": MODEL_VARIANT,
            "resolved_snapshot": str(snapshot),
            "repository_role": "third_party_hash_pinned_mirror",
            "critical_hashes": verify_snapshot(snapshot),
        }
        report["environment"].update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            }
        )
        report["prompt_audit"] = audit_prompts(records, snapshot, model_input)
        vae_rows, latents = run_vae(
            snapshot, selected, args.image_root.resolve(), model_input
        )
        report["vae"] = {
            "status": "PASS",
            "sampling": "latent_dist.mode",
            "profiles": vae_rows,
        }
        primary = next(record for stratum, record, _ in selected if stratum == "l_pneumothorax")
        primary_prompt = model_input.prompt_for_record(primary)
        hidden = encode_text(snapshot, primary_prompt)
        report["lora_gradient"] = run_lora_gradient(
            snapshot, primary, hidden, latents
        )
        for row in report["lora_gradient"]["profiles"]:
            row["prompt"] = primary_prompt
        del hidden
        del latents
        gc.collect()
        torch.cuda.empty_cache()
        report["ppmark"] = run_ppmark(
            snapshot,
            selected,
            args.image_root.resolve(),
            model_input,
            args.ppmark_steps,
        )
        report["status"] = "PASS_FOR_PILOT_INTERFACE"
    report["interpretation_limit"] = (
        "A PASS proves deterministic real-X-ray compatibility with the pinned SD 2.1 VAE, "
        "LoRA gradient path, and PP-Mark inversion API only. It is not model training, a "
        "privacy guarantee, medical utility, attack robustness, calibration, provenance, "
        "or release authorization."
    )
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"NIH_CXR14_SD21_PPMARK_INTERFACE: {report['status']}")
    print(f"report={report_path}")
    print(f"report_sha256={sha256_file(report_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
