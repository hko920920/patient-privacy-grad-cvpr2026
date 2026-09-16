#!/usr/bin/env python3
"""Exercise PP-Mark DDIM inversion with an exact frozen SD artifact.

This is an adapter/interface diagnostic.  It does not reuse SD2.1/SDXL
calibration, establish PP-Mark robustness on medical images, or create a
privacy/provenance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any


MODEL_ID = "CompVis/stable-diffusion-v1-4"
MODEL_REVISION = "133a221b8aa7292a167afc5127cb63fb5005638b"
MODEL_VARIANT = "fp16"
EXPECTED_UNET_SHA256 = (
    "A35404D03EC8F977715A4D2A080DDF72E2144F2EE49BB1EE213258BC64F9CC87"
)
EXPECTED_VAE_SHA256 = (
    "4FBCF0EBE55A0984F5A5E00D8C4521D52359AF7229BB4D81890039D2AA16DD7C"
)


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


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=Path,
        default=root
        / "_data"
        / "intake"
        / "isic2020_collection70"
        / "smoke_images"
        / "ISIC_2027582.jpg",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root
        / "_reports"
        / "isic2020_split_v1_001"
        / "ppmark_exact_loader_result.json",
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--variant", default=MODEL_VARIANT)
    parser.add_argument("--expected-unet-sha256", default=EXPECTED_UNET_SHA256)
    parser.add_argument("--expected-vae-sha256", default=EXPECTED_VAE_SHA256)
    parser.add_argument(
        "--calibration-status",
        default="REQUIRED_FOR_SD14_MEDICAL_OUTPUTS",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "ppmark" / "src"))
    disable_broken_optional_onnx()
    try:
        import numpy as np
        import torch
        from huggingface_hub import snapshot_download
        from PIL import Image, ImageOps
        from ppmark_v03.ddim_unet import DiffusersDDIMConfig, DiffusersDDIMInverter

        snapshot = Path(
            snapshot_download(
                repo_id=args.model_id,
                revision=args.revision,
                local_files_only=True,
            )
        )
        if snapshot.name != args.revision:
            raise ValueError(
                f"snapshot revision mismatch: {snapshot.name} != {args.revision}"
            )
        unet_path = snapshot / "unet" / "diffusion_pytorch_model.fp16.safetensors"
        vae_path = snapshot / "vae" / "diffusion_pytorch_model.fp16.safetensors"
        component_hashes = {
            "unet": sha256_file(unet_path),
            "vae": sha256_file(vae_path),
        }
        if component_hashes["unet"] != args.expected_unet_sha256.upper():
            raise ValueError("UNet hash mismatch")
        if component_hashes["vae"] != args.expected_vae_sha256.upper():
            raise ValueError("VAE hash mismatch")

        with Image.open(args.image) as source:
            image = ImageOps.fit(
                ImageOps.exif_transpose(source).convert("RGB"),
                (256, 256),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )
        array = np.asarray(image, dtype=np.float32) / 255.0

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        config = DiffusersDDIMConfig(
            model_id=args.model_id,
            revision=args.revision,
            variant=args.variant,
            use_safetensors=True,
            local_files_only=True,
            disable_safety_checker=True,
            device="cuda",
            torch_dtype="float16",
            num_steps=args.steps,
            guidance_scale=0.0,
            prompt="a clinical dermoscopic image of malignant melanoma",
        )
        inverter = DiffusersDDIMInverter(config)
        latent = inverter.invert_latents(array)
        result: dict[str, Any] = {
            "schema": "ppmark-exact-model-loader-smoke/v1",
            "status": "PASS",
            "evidence_scope": "INTERFACE_ONLY_NOT_CALIBRATION_NOT_ROBUSTNESS_NOT_DP",
            "ppmark_module": "ppmark_v03.ddim_unet",
            "load_spec": inverter.load_spec,
            "resolved_snapshot_revision_directory": snapshot.name,
            "component_hashes": component_hashes,
            "input_image": args.image.name,
            "input_image_sha256": sha256_file(args.image),
            "ddim_steps": args.steps,
            "latent_shape": list(latent.shape),
            "latent_dtype": str(latent.dtype),
            "finite": bool(np.isfinite(latent).all()),
            "latent_mean": round(float(latent.mean()), 8),
            "latent_std": round(float(latent.std()), 8),
            "elapsed_sec": round(time.perf_counter() - started, 4),
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 4),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 4),
            "calibration_status": args.calibration_status,
        }
        if result["latent_shape"] != [4, 32, 32] or not result["finite"]:
            raise ValueError(f"unexpected PP-Mark inversion result: {result['latent_shape']}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("PPMARK_EXACT_LOADER: PASS")
        print("latent_shape=" + str(result["latent_shape"]))
        print("result_sha256=" + sha256_file(args.output))
        del inverter
        torch.cuda.empty_cache()
        return 0
    except Exception as exc:
        print(f"PPMARK_EXACT_LOADER: FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
