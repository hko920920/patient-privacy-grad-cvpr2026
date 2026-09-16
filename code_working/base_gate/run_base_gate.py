#!/usr/bin/env python3
"""Exact-checkpoint inference and LoRA VRAM feasibility gate.

This is a diagnostic only. The single-record clipping path below is not a DP
run, does not use private noise, and cannot support a privacy claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
import traceback
from pathlib import Path
from typing import Any


MODEL_ID = "CompVis/stable-diffusion-v1-4"
MODEL_REVISION = "133a221b8aa7292a167afc5127cb63fb5005638b"
MODEL_VARIANT = "fp16"
ARTIFACT_PROVENANCE: dict[str, Any] = {
    "repository_role": "official_source",
}
EXPECTED_CRITICAL_SHA256: dict[str, str] = {
    "text_encoder/model.fp16.safetensors": (
        "77795E2023ADCF39BC29A884661950380BD093CF0750A966D473D1718DC9EF4E"
    ),
    "unet/diffusion_pytorch_model.fp16.safetensors": (
        "A35404D03EC8F977715A4D2A080DDF72E2144F2EE49BB1EE213258BC64F9CC87"
    ),
    "vae/diffusion_pytorch_model.fp16.safetensors": (
        "4FBCF0EBE55A0984F5A5E00D8C4521D52359AF7229BB4D81890039D2AA16DD7C"
    ),
}

ALLOW_PATTERNS = [
    ".gitattributes",
    "README.md",
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
    """Hide a broken inherited ONNX Runtime from Diffusers feature detection.

    The gate uses only native PyTorch. The host environment exposes an
    onnxruntime distribution whose Windows DLL cannot initialize; Diffusers
    0.30.2 otherwise treats its mere presence as a request to load ONNX code.
    """

    import importlib.util

    original_find_spec = importlib.util.find_spec

    def guarded_find_spec(name: str, package: str | None = None):
        if name == "onnxruntime" or name.startswith("onnxruntime."):
            return None
        return original_find_spec(name, package)

    importlib.util.find_spec = guarded_find_spec


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def snapshot_manifest(snapshot: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(snapshot.rglob("*")):
        if not path.is_file():
            continue
        rows.append(
            {
                "path": path.relative_to(snapshot).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def cuda_memory(torch: Any) -> dict[str, float]:
    gib = 1024.0**3
    return {
        "allocated_gib": round(torch.cuda.memory_allocated() / gib, 4),
        "reserved_gib": round(torch.cuda.memory_reserved() / gib, 4),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / gib, 4),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / gib, 4),
    }


def run_inference(snapshot: Path, output_dir: Path) -> list[dict[str, Any]]:
    import numpy as np
    import torch
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    from PIL import Image

    pipe = StableDiffusionPipeline.from_pretrained(
        snapshot,
        torch_dtype=torch.float16,
        variant=MODEL_VARIANT,
        use_safetensors=True,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")

    results: list[dict[str, Any]] = []
    prompt = "a clinical dermoscopic photograph of a skin lesion"
    for resolution in (256, 512):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        row: dict[str, Any] = {
            "resolution": resolution,
            "steps": 2,
            "guidance_scale": 7.5,
            "status": "PASS",
        }
        try:
            generator = torch.Generator(device="cuda").manual_seed(260902)
            with torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    num_inference_steps=2,
                    guidance_scale=7.5,
                    height=resolution,
                    width=resolution,
                    generator=generator,
                    output_type="np",
                )
            image = np.clip(result.images[0] * 255.0, 0, 255).astype(np.uint8)
            image_path = output_dir / f"inference_{resolution}.png"
            Image.fromarray(image).save(image_path)
            row["image"] = image_path.name
            row["image_sha256"] = sha256_file(image_path)
        except torch.cuda.OutOfMemoryError as exc:
            row["status"] = "OOM"
            row["error"] = str(exc)
            torch.cuda.empty_cache()
        except Exception as exc:  # diagnostic result must preserve the failure
            row["status"] = "ERROR"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc()
        row["elapsed_sec"] = round(time.perf_counter() - started, 4)
        row["cuda"] = cuda_memory(torch)
        results.append(row)

    del pipe
    torch.cuda.empty_cache()
    return results


def run_lora_step(snapshot: Path, resolution: int, rank: int) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    from diffusers import UNet2DConditionModel
    from peft import LoraConfig

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    row: dict[str, Any] = {
        "resolution": resolution,
        "batch_size": 1,
        "rank": rank,
        "gradient_checkpointing": True,
        "status": "PASS",
        "privacy_status": "DIAGNOSTIC_NOT_DP",
    }
    unet = None
    optimizer = None
    try:
        torch.manual_seed(260902 + resolution)
        torch.cuda.manual_seed_all(260902 + resolution)
        unet = UNet2DConditionModel.from_pretrained(
            snapshot,
            subfolder="unet",
            revision=None,
            variant=MODEL_VARIANT,
            torch_dtype=torch.float16,
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

        trainable = [parameter for parameter in unet.parameters() if parameter.requires_grad]
        row["trainable_parameters"] = sum(parameter.numel() for parameter in trainable)
        row["total_parameters"] = sum(parameter.numel() for parameter in unet.parameters())
        row["trainable_fraction"] = row["trainable_parameters"] / row["total_parameters"]

        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
        latent_side = resolution // 8
        sample = torch.randn((1, 4, latent_side, latent_side), device="cuda", dtype=torch.float16)
        target = torch.randn_like(sample)
        timestep = torch.tensor([500], device="cuda", dtype=torch.long)
        cross_attention_dim = int(unet.config.cross_attention_dim)
        encoder_hidden_states = torch.randn(
            (1, 77, cross_attention_dim), device="cuda", dtype=torch.float16
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = unet(sample, timestep, encoder_hidden_states).sample
            loss = functional.mse_loss(prediction.float(), target.float())
        loss.backward()

        # With batch size one this exercises the record-gradient clipping path.
        # It deliberately adds no DP noise and supports no privacy accounting.
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        torch.cuda.synchronize()
        row["loss"] = float(loss.detach().cpu())
        row["preclip_grad_norm"] = float(grad_norm.detach().cpu())
    except torch.cuda.OutOfMemoryError as exc:
        row["status"] = "OOM"
        row["error"] = str(exc)
    except Exception as exc:
        row["status"] = "ERROR"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()
    row["elapsed_sec"] = round(time.perf_counter() - started, 4)
    row["cuda"] = cuda_memory(torch)

    del optimizer
    del unet
    torch.cuda.empty_cache()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, default=8)
    args = parser.parse_args()

    # Prevent an implicit switch to a mutable model revision.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    disable_broken_optional_onnx()

    import torch
    from huggingface_hub import snapshot_download

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this gate")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=ALLOW_PATTERNS,
        )
    ).resolve()
    files = snapshot_manifest(snapshot)
    actual_hashes = {row["path"]: row["sha256"] for row in files}
    critical_hash_checks = {
        path: {
            "expected_sha256": expected.upper(),
            "actual_sha256": actual_hashes.get(path),
            "status": "PASS" if actual_hashes.get(path) == expected.upper() else "FAIL",
        }
        for path, expected in EXPECTED_CRITICAL_SHA256.items()
    }
    failed_hashes = [path for path, row in critical_hash_checks.items() if row["status"] != "PASS"]
    if failed_hashes:
        raise RuntimeError(f"Critical artifact hash mismatch or missing files: {failed_hashes}")
    manifest_path = output_dir / "base_snapshot_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "revision": MODEL_REVISION,
                "variant": MODEL_VARIANT,
                "artifact_provenance": ARTIFACT_PROVENANCE,
                "critical_hash_checks": critical_hash_checks,
                "snapshot_revision_directory": snapshot.name,
                "files": files,
                "total_bytes": sum(row["bytes"] for row in files),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result: dict[str, Any] = {
        "status": "DIAGNOSTIC_ONLY",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "variant": MODEL_VARIANT,
        "artifact_provenance": ARTIFACT_PROVENANCE,
        "critical_hash_checks": critical_hash_checks,
        "snapshot_manifest": manifest_path.name,
        "snapshot_manifest_sha256": sha256_file(manifest_path),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "optional_onnx_disabled": True,
            "packages": {
                name: package_version(name)
                for name in (
                    "diffusers",
                    "transformers",
                    "accelerate",
                    "peft",
                    "opacus",
                    "safetensors",
                    "huggingface-hub",
                    "torchvision",
                )
            },
        },
        "inference": run_inference(snapshot, output_dir),
        "lora_single_record_clipping_probe": [
            run_lora_step(snapshot, 256, args.rank),
            run_lora_step(snapshot, 512, args.rank),
        ],
        "interpretation_limit": (
            "This gate checks loading, inference and one batch-size-one LoRA clipping step only. "
            "It is not DP-SGD, patient-DP, a utility result or a confirmatory training run."
        ),
    }
    result_path = output_dir / "base_gate_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
