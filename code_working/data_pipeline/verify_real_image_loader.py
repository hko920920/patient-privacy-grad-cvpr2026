#!/usr/bin/env python3
"""Verify manifest loading and exact SD-v1.4 VAE encoding on real ISIC JPEGs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

from isic_manifest_dataset import IsicManifestDataset, PatientGroupedView


MODEL_ID = "CompVis/stable-diffusion-v1-4"
MODEL_REVISION = "133a221b8aa7292a167afc5127cb63fb5005638b"
MODEL_VARIANT = "fp16"


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
        "--manifest",
        type=Path,
        default=root
        / "_data"
        / "derived"
        / "isic2020_v2_split_v1"
        / "smoke_subset_private.csv",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=root
        / "_data"
        / "intake"
        / "isic2020_collection70"
        / "smoke_images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root
        / "_reports"
        / "isic2020_split_v1_001"
        / "real_image_loader_result.json",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--skip-vae", action="store_true")
    return parser.parse_args()


def run_vae(pixel_values: Any) -> dict[str, Any]:
    disable_broken_optional_onnx()
    import torch
    from diffusers import AutoencoderKL
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    vae = AutoencoderKL.from_pretrained(
        snapshot,
        subfolder="vae",
        torch_dtype=torch.float16,
        variant=MODEL_VARIANT,
        use_safetensors=True,
        local_files_only=True,
    ).to("cuda")
    with torch.inference_mode():
        distribution = vae.encode(pixel_values.to(device="cuda", dtype=torch.float16)).latent_dist
        # Mode, not a random sample, makes this interface check deterministic.
        latents = distribution.mode() * vae.config.scaling_factor
    result = {
        "snapshot_revision_directory": snapshot.name,
        "latent_shape": list(latents.shape),
        "latent_dtype": str(latents.dtype),
        "finite": bool(torch.isfinite(latents).all().item()),
        "latent_mean": round(float(latents.float().mean().item()), 8),
        "latent_std": round(float(latents.float().std().item()), 8),
        "elapsed_sec": round(time.perf_counter() - started, 4),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 4),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 4),
    }
    del latents, distribution, vae
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    try:
        import torch

        dataset = IsicManifestDataset(
            args.manifest.resolve(),
            args.image_root.resolve(),
            image_size=args.image_size,
            preprocessing_policy="center_crop",
        )
        grouped = PatientGroupedView(dataset)
        tensors = []
        rows = []
        for index in range(len(dataset)):
            item = dataset[index]
            tensors.append(item["pixel_values"])
            rows.append(
                {
                    "image_id": item["image_id"],
                    "patient_id": item["patient_id"],
                    "partition": item["partition"],
                    "target": item["target"],
                    "shape": list(item["pixel_values"].shape),
                    "min": round(float(item["pixel_values"].min().item()), 8),
                    "max": round(float(item["pixel_values"].max().item()), 8),
                    "finite": bool(torch.isfinite(item["pixel_values"]).all().item()),
                }
            )
        batch = torch.stack(tensors)
        result: dict[str, Any] = {
            "schema": "real-image-loader-smoke/v1",
            "status": "PASS",
            "evidence_scope": "INTERFACE_AND_MEMORY_ONLY_NOT_DP_NOT_UTILITY",
            "manifest": args.manifest.name,
            "manifest_sha256": sha256_file(args.manifest),
            "image_root_file_count": len(list(args.image_root.glob("*.jpg"))),
            "records": len(dataset),
            "patients": len(grouped),
            "patient_contributions": sorted(grouped.contribution_counts()),
            "preprocessing": {
                "policy": "center_crop",
                "image_size": args.image_size,
                "tensor_range": "[-1,1]",
            },
            "rows": rows,
        }
        if not args.skip_vae:
            result["exact_vae"] = {
                "model_id": MODEL_ID,
                "revision": MODEL_REVISION,
                "variant": MODEL_VARIANT,
                **run_vae(batch),
            }
            if not result["exact_vae"]["finite"]:
                raise ValueError("VAE latent contains non-finite values")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("REAL_IMAGE_LOADER: PASS")
        print(f"records={len(dataset)} patients={len(grouped)}")
        if "exact_vae" in result:
            print("latent_shape=" + str(result["exact_vae"]["latent_shape"]))
        print("result_sha256=" + sha256_file(args.output))
        return 0
    except Exception as exc:
        print(f"REAL_IMAGE_LOADER: FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
