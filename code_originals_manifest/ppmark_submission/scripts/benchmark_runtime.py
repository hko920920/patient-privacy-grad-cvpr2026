#!/usr/bin/env python3
"""Runtime benchmark for 7 watermark methods (generation + verification).

Measures:
  - Generation: watermark embedding + diffusion pipe() time
  - Verification: DDIM inversion (where applicable) + detection time
  - PP-Mark additionally: proof generation time, receipt verification time

Usage:
  cd anonymous_supplementary_code
  source .venv/bin/activate
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:$PYTHONPATH python scripts/benchmark_runtime.py \
      --num-images 10 --output outputs/benchmark_runtime/results.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
METHODS_ROOT = REPO_ROOT / "outputs" / "attacks" / "muller_forgery_report_prep" / "methods"
KEYS_JSON = REPO_ROOT / "outputs" / "attacks" / "muller_forgery_report_prep" / "keys.json"
THRESHOLDS_DIR = REPO_ROOT / "outputs" / "attacks" / "muller_forgery_report_prep" / "thresholds"

STABLESIG_DECODER = REPO_ROOT / "external" / "stable_signature" / "models" / "dec_48b_whit.torchscript.pt"
STABLESIG_VAE_DECODER = REPO_ROOT / "external" / "stable_signature" / "models" / "sd2_decoder.pth"
HIDDEN_CKPT = REPO_ROOT / "external" / "stable_signature" / "hidden" / "ckpts" / "hidden_replicate.pth"

MODEL_ID = "Manojb/stable-diffusion-2-1-base"
STEPS = 50
IMAGE_SIZE = 512
GUIDANCE = 7.5
DTYPE_STR = "float16"

DEFAULT_KEY_BITS = "111010110101000001010111010011010100010000100111"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_manifest(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_image_tensor(path: Path, device: str, dtype, size: int = 512):
    """Load image as normalised tensor [1,3,H,W] in [-1,1]."""
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    import torch
    t = torch.from_numpy(arr).unsqueeze(0).to(device, dtype=dtype)
    return t * 2.0 - 1.0


def _sample_images(method_dir: str, n: int) -> List[Path]:
    """Return first n eval image paths for a method."""
    eval_dir = METHODS_ROOT / method_dir / "eval" / "images"
    if not eval_dir.exists():
        return []
    # Standard layout: flat PNGs
    imgs = sorted(eval_dir.glob("*.png"))
    if imgs:
        return imgs[:n]
    # PP-Mark layout: images/img_XXXX/watermarked.png
    imgs = sorted(eval_dir.glob("*/watermarked.png"))
    return imgs[:n]


def _sample_manifest_rows(method_dir: str, n: int) -> List[dict]:
    manifest = METHODS_ROOT / method_dir / "eval" / "manifest.jsonl"
    if not manifest.exists():
        return []
    return _load_manifest(manifest)[:n]


def _stats(values: List[float]) -> Dict[str, float]:
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": len(values),
    }


def _clear_gpu():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ===========================================================================
# VERIFICATION BENCHMARKS
# ===========================================================================

def bench_verify_tree_ring(images: List[Path], device: str) -> Dict[str, Any]:
    """Tree-Ring verification: DDIM inversion + FFT p-value."""
    import torch, scipy.stats
    from types import SimpleNamespace

    tr_root = REPO_ROOT / "external" / "tree-ring-watermark"
    sys.path.insert(0, str(tr_root))
    from diffusers import DPMSolverMultistepScheduler
    from inverse_stable_diffusion import InversableStableDiffusionPipeline
    from optim_utils import get_watermarking_mask, get_watermarking_pattern

    keys = json.loads(KEYS_JSON.read_text())
    w_seed = int(keys["tree_ring"]["seed"])

    # Pipeline load (excluded from timing)
    scheduler = DPMSolverMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    try:
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            MODEL_ID, scheduler=scheduler, torch_dtype=getattr(torch, DTYPE_STR), revision="fp16")
    except Exception:
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            MODEL_ID, scheduler=scheduler, torch_dtype=getattr(torch, DTYPE_STR))
    pipe.safety_checker = None
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    helper_args = SimpleNamespace(
        w_seed=w_seed, w_channel=3, w_pattern="ring", w_mask_shape="circle",
        w_radius=10, w_measurement="l1_complex", w_injection="complex", w_pattern_const=0.0)
    gt_patch = get_watermarking_pattern(pipe, helper_args, device)
    mask = get_watermarking_mask(pipe.get_random_latents(), helper_args, device)
    gt_patch_np = gt_patch.detach().to(torch.complex64).cpu().numpy()
    mask_np = mask.detach().cpu().numpy()
    text_embeddings = pipe.get_text_embedding("")

    from calibrate_tree_ring import _p_value_from_latents

    t_inversion, t_detection, t_total = [], [], []
    for img_path in images:
        img_t = _load_image_tensor(img_path, device, pipe.unet.dtype)

        t0 = time.perf_counter()
        with torch.inference_mode():
            image_latents = pipe.get_image_latents(img_t, sample=False)
            reversed_latents = pipe.forward_diffusion(
                latents=image_latents, text_embeddings=text_embeddings,
                guidance_scale=1, num_inference_steps=STEPS)
        t1 = time.perf_counter()
        rev_np = reversed_latents.detach().float().cpu().numpy()
        _p_value_from_latents(rev_np, mask_np, gt_patch_np, scipy.stats)
        t2 = time.perf_counter()

        t_inversion.append(t1 - t0)
        t_detection.append(t2 - t1)
        t_total.append(t2 - t0)
        print(f"  [TR verify] {len(t_total)}/{len(images)} inv={t1-t0:.2f}s det={t2-t1:.4f}s")

    del pipe
    _clear_gpu()
    return {"ddim_inv_s": _stats(t_inversion), "detection_s": _stats(t_detection), "total_s": _stats(t_total)}


def bench_verify_gaussian_shading(images: List[Path], device: str) -> Dict[str, Any]:
    """Gaussian Shading verification: DDIM inversion + bit accuracy."""
    import torch

    gs_root = REPO_ROOT / "external" / "Gaussian-Shading"
    sys.path.insert(0, str(gs_root))
    from diffusers import DPMSolverMultistepScheduler
    from inverse_stable_diffusion import InversableStableDiffusionPipeline
    from watermark import Gaussian_Shading

    keys = json.loads(KEYS_JSON.read_text())
    gs_cfg = keys["gaussian_shading"]

    scheduler = DPMSolverMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    try:
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            MODEL_ID, scheduler=scheduler, torch_dtype=getattr(torch, DTYPE_STR), revision="fp16")
    except Exception:
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            MODEL_ID, scheduler=scheduler, torch_dtype=getattr(torch, DTYPE_STR))
    pipe.safety_checker = None
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    wm = Gaussian_Shading(1, 8, float(gs_cfg["detection_fpr"]), int(gs_cfg["user_pool_size"]))
    import random
    def _set_seed(s):
        torch.manual_seed(s); torch.cuda.manual_seed_all(s+2); np.random.seed(s+3); random.seed(s+5)
    _set_seed(int(gs_cfg["alice_user_index"]))
    wm.key = torch.randint(0, 2, [1, 4, 64, 64]).to(device)
    wm.watermark = torch.randint(0, 2, [1, 4 // wm.ch, 64 // wm.hw, 64 // wm.hw]).to(device)
    text_embeddings = pipe.get_text_embedding("")

    t_inversion, t_detection, t_total = [], [], []
    for img_path in images:
        img_t = _load_image_tensor(img_path, device, pipe.unet.dtype)

        t0 = time.perf_counter()
        with torch.inference_mode():
            image_latents = pipe.get_image_latents(img_t, sample=False)
            reversed_latents = pipe.forward_diffusion(
                latents=image_latents, text_embeddings=text_embeddings,
                guidance_scale=1, num_inference_steps=STEPS)
        t1 = time.perf_counter()
        wm.eval_watermark(reversed_latents)
        t2 = time.perf_counter()

        t_inversion.append(t1 - t0)
        t_detection.append(t2 - t1)
        t_total.append(t2 - t0)
        print(f"  [GS verify] {len(t_total)}/{len(images)} inv={t1-t0:.2f}s det={t2-t1:.4f}s")

    del pipe
    _clear_gpu()
    return {"ddim_inv_s": _stats(t_inversion), "detection_s": _stats(t_detection), "total_s": _stats(t_total)}


def bench_verify_ringid(images: List[Path], device: str) -> Dict[str, Any]:
    """RingID verification: DDIM inversion + distance."""
    import torch, itertools, random

    ring_root = REPO_ROOT / "external" / "RingID"
    sys.path.insert(0, str(ring_root))
    from diffusers import DPMSolverMultistepScheduler
    from inverse_stable_diffusion import InversableStableDiffusionPipeline
    from utils import (HETER_WATERMARK_CHANNEL, RING_WATERMARK_CHANNEL, WATERMARK_CHANNEL,
                       RADIUS, RADIUS_CUTOFF, fft, ifft, make_Fourier_ringid_pattern,
                       ring_mask, get_distance)

    keys = json.loads(KEYS_JSON.read_text())
    ring_cfg = keys["ringid"]

    scheduler = DPMSolverMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    try:
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            MODEL_ID, scheduler=scheduler, torch_dtype=getattr(torch, DTYPE_STR), revision="fp16")
    except Exception:
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            MODEL_ID, scheduler=scheduler, torch_dtype=getattr(torch, DTYPE_STR))
    pipe.safety_checker = None
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    base_latents = pipe.get_random_latents().to(torch.float64)
    shape = base_latents.shape
    sing_mask = torch.tensor(ring_mask(size=shape[-1], r_out=RADIUS, r_in=RADIUS_CUTOFF), device=device)

    if len(HETER_WATERMARK_CHANNEL) > 0:
        heter_mask_v = sing_mask.unsqueeze(0).repeat(len(HETER_WATERMARK_CHANNEL), 1, 1)
    else:
        heter_mask_v = None

    wm_region_mask = []
    for ch in WATERMARK_CHANNEL:
        if ch in RING_WATERMARK_CHANNEL:
            wm_region_mask.append(sing_mask)
        else:
            wm_region_mask.append(heter_mask_v[0] if heter_mask_v is not None else sing_mask)
    wm_region_mask = torch.stack(wm_region_mask)

    slot_count = RADIUS - RADIUS_CUTOFF
    per_slot_values = np.linspace(-64, 64, 2).tolist()
    key_value_list = [[list(c) for c in itertools.product(per_slot_values, repeat=len(RING_WATERMARK_CHANNEL))] for _ in range(slot_count)]
    key_value_combinations = list(itertools.product(*key_value_list))
    rng = random.Random(int(ring_cfg["seed"]))
    key_index = rng.randrange(len(key_value_combinations))
    kvc = key_value_combinations[key_index]

    watermark_pattern = make_Fourier_ringid_pattern(
        device, list(kvc), base_latents, radius=RADIUS, radius_cutoff=RADIUS_CUTOFF,
        ring_watermark_channel=RING_WATERMARK_CHANNEL, heter_watermark_channel=HETER_WATERMARK_CHANNEL,
        heter_watermark_region_mask=heter_mask_v, ring_width=1)
    watermark_pattern = fft(ifft(watermark_pattern).real)
    if len(RING_WATERMARK_CHANNEL) > 0:
        rs = watermark_pattern[:, RING_WATERMARK_CHANNEL, ...]
        rs = fft(torch.fft.fftshift(ifft(rs), dim=(-1, -2)))
        watermark_pattern[:, RING_WATERMARK_CHANNEL, ...] = rs

    text_embeddings = pipe.get_text_embedding("")

    t_inversion, t_detection, t_total = [], [], []
    for img_path in images:
        img_t = _load_image_tensor(img_path, device, pipe.unet.dtype)

        t0 = time.perf_counter()
        with torch.inference_mode():
            image_latents = pipe.get_image_latents(img_t, sample=False)
            reversed_latents = pipe.forward_diffusion(
                latents=image_latents, text_embeddings=text_embeddings,
                guidance_scale=1, num_inference_steps=STEPS)
        t1 = time.perf_counter()
        rev_fft = fft(reversed_latents)
        get_distance(watermark_pattern, rev_fft, wm_region_mask, p=2, mode="complex",
                     channel_min=True, channel=WATERMARK_CHANNEL)
        t2 = time.perf_counter()

        t_inversion.append(t1 - t0)
        t_detection.append(t2 - t1)
        t_total.append(t2 - t0)
        print(f"  [RingID verify] {len(t_total)}/{len(images)} inv={t1-t0:.2f}s det={t2-t1:.4f}s")

    del pipe
    _clear_gpu()
    return {"ddim_inv_s": _stats(t_inversion), "detection_s": _stats(t_detection), "total_s": _stats(t_total)}


def bench_verify_stable_signature(images: List[Path], device: str) -> Dict[str, Any]:
    """Stable Signature verification: TorchScript decoder inference (no DDIM)."""
    import torch

    decoder = torch.jit.load(str(STABLESIG_DECODER))
    decoder = decoder.to(device).eval()
    key_bits = np.array([int(c) for c in DEFAULT_KEY_BITS], dtype=np.int64)
    key_tensor = torch.from_numpy(key_bits).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]

    t_detection, t_total = [], []
    for img_path in images:
        img = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device, dtype=torch.float32)
        imgs = (t - mean) / std

        t0 = time.perf_counter()
        with torch.inference_mode():
            decoded = decoder(imgs)
        t1 = time.perf_counter()

        t_detection.append(t1 - t0)
        t_total.append(t1 - t0)
        print(f"  [StableSig verify] {len(t_total)}/{len(images)} det={t1-t0:.4f}s")

    del decoder
    _clear_gpu()
    return {"ddim_inv_s": None, "detection_s": _stats(t_detection), "total_s": _stats(t_total)}


def bench_verify_hidden(images: List[Path], device: str) -> Dict[str, Any]:
    """HiDDeN verification: decoder inference (no DDIM)."""
    import torch

    sys.path.insert(0, str(REPO_ROOT / "external" / "stable_signature"))
    import utils_model
    try:
        torch.serialization.add_safe_globals([argparse.Namespace])
    except Exception:
        pass
    try:
        ckpt = torch.load(str(HIDDEN_CKPT), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(HIDDEN_CKPT), map_location="cpu")
    params = ckpt["params"]
    decoder = utils_model.get_hidden_decoder(
        num_bits=params.num_bits, redundancy=params.redundancy,
        num_blocks=params.decoder_depth, channels=params.decoder_channels)
    dec_state = {k.replace("module.", "").replace("decoder.", ""): v
                 for k, v in ckpt["encoder_decoder"].items() if "decoder" in k}
    decoder.load_state_dict(dec_state, strict=False)
    decoder = decoder.to(device).eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]

    t_detection, t_total = [], []
    for img_path in images:
        img = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device, dtype=torch.float32)
        imgs = (t - mean) / std

        t0 = time.perf_counter()
        with torch.inference_mode():
            decoded = decoder(imgs)
        t1 = time.perf_counter()

        t_detection.append(t1 - t0)
        t_total.append(t1 - t0)
        print(f"  [HiDDeN verify] {len(t_total)}/{len(images)} det={t1-t0:.4f}s")

    del decoder
    _clear_gpu()
    return {"ddim_inv_s": None, "detection_s": _stats(t_detection), "total_s": _stats(t_total)}


def bench_verify_wind(images: List[Path], device: str) -> Dict[str, Any]:
    """WIND verification: DDIM inversion + pattern distance."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from wind_common import WindWatermarker, build_pipe
    from types import SimpleNamespace

    wind_args = SimpleNamespace(
        model_id=MODEL_ID, dtype=DTYPE_STR, device=device,
        general_seed=42, watermark_seed=5, ring_width=1, num_inmost_keys=2,
        ring_value_range=64, assigned_keys=-1, fix_gt=1, time_shift=1,
        time_shift_factor=1.0, channel_min=1, M=10000,
        num_inference_steps=STEPS, test_num_inference_steps=STEPS,
        guidance_scale=GUIDANCE, image_length=IMAGE_SIZE,
        precompute_cache=str(REPO_ROOT / "outputs" / "benchmark_runtime" / "wind_cache.pth"),
        rotation_angles=[0.0, 74.5, 75.5],
    )

    pipe, _ = build_pipe(MODEL_ID, DTYPE_STR, device)
    watermarker = WindWatermarker(wind_args, pipe, device)

    from wind_common import load_pil_image

    t_inversion, t_detection, t_total = [], [], []
    for img_path in images:
        img = load_pil_image(img_path)

        t0 = time.perf_counter()
        detected_n, reversed_latent = watermarker.detect_watermark_stage1(img)
        t1 = time.perf_counter()
        # Stage 2: distance matching
        min_distance = float("inf")
        for m, wm_lat in watermarker.pre_computed_latents[detected_n]:
            wm_lat = wm_lat.to(device)
            for angle in watermarker.rotation_angles:
                rot = watermarker._rotate_tensor(wm_lat, angle)
                import torch
                d = float(torch.norm(reversed_latent - rot))
                if d < min_distance:
                    min_distance = d
        t2 = time.perf_counter()

        t_inversion.append(t1 - t0)
        t_detection.append(t2 - t1)
        t_total.append(t2 - t0)
        print(f"  [WIND verify] {len(t_total)}/{len(images)} inv={t1-t0:.2f}s det={t2-t1:.2f}s")

    del pipe, watermarker
    _clear_gpu()
    return {"ddim_inv_s": _stats(t_inversion), "detection_s": _stats(t_detection), "total_s": _stats(t_total)}


def bench_verify_ppmark(images: List[Path], device: str) -> Dict[str, Any]:
    """PP-Mark verification: DDIM inversion (same pipe as TR/GS/RingID) + correlation score.
    Receipt verification timing is taken from paper (7.59s) as it requires SP1 binary.
    """
    import torch
    import hashlib

    # Use the same InversableStableDiffusionPipeline as TR/GS/RingID for fair comparison
    tr_root = REPO_ROOT / "external" / "tree-ring-watermark"
    if str(tr_root) not in sys.path:
        sys.path.insert(0, str(tr_root))
    from diffusers import DPMSolverMultistepScheduler
    from inverse_stable_diffusion import InversableStableDiffusionPipeline

    scheduler = DPMSolverMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    try:
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            MODEL_ID, scheduler=scheduler, torch_dtype=getattr(torch, DTYPE_STR), revision="fp16")
    except Exception:
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            MODEL_ID, scheduler=scheduler, torch_dtype=getattr(torch, DTYPE_STR))
    pipe.safety_checker = None
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    text_embeddings = pipe.get_text_embedding("")

    # Load PP-Mark config and reconstruct binding for score computation
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))
    from ppmark_v03.config import GlobalConfig
    from ppmark_v03.noise import build_bit_index_map
    from ppmark_v03.sampling import deterministic_sample
    from ppmark_v03.payload import bytes_to_bits

    cfg_path = REPO_ROOT / "config_alpha4_k1000.json"
    if not cfg_path.exists():
        cfg_path = REPO_ROOT / "config.json"
    cfg = GlobalConfig.load(cfg_path)

    keys = json.loads(KEYS_JSON.read_text())
    secret_hex = keys["pp_mark"]["secret_hex"]
    alpha = float(keys["pp_mark"].get("alpha", 4.0))
    N = int(keys["pp_mark"].get("N", 1000))

    meta_root = METHODS_ROOT / "pp_mark" / "eval"
    manifest = _load_manifest(meta_root / "manifest.jsonl")

    # Pre-compute binding and sample set from first image's metadata.json
    meta_dir = meta_root / "images"
    sample_meta_candidates = sorted(meta_dir.glob("*/metadata.json"))
    binding_int = None
    bits_for_samples = None
    sample_indices = None
    if sample_meta_candidates:
        sample_meta = json.loads(sample_meta_candidates[0].read_text())
        if "codeword_hex" in sample_meta:
            codeword = bytes.fromhex(sample_meta["codeword_hex"])
            bits = bytes_to_bits(codeword)
            binding_int = int(sample_meta.get("binding", "0"), 16) if isinstance(sample_meta.get("binding"), str) else int(sample_meta.get("binding", 0))
            bit_idx_map = build_bit_index_map(
                binding=binding_int, width=cfg.image.width, height=cfg.image.height,
                bit_len=len(bits), backend=cfg.zk.sample_backend)
            sample_set = deterministic_sample(
                total_pixels=cfg.image.width * cfg.image.height,
                width=cfg.image.width, height=cfg.image.height,
                key_material=codeword, count=N)
            sample_indices = sample_set.indices
            bits_for_samples = [bits[int(bit_idx_map[idx // cfg.image.width, idx % cfg.image.width])] for idx in sample_indices]

    t_inversion, t_detection, t_total = [], [], []
    for i, img_path in enumerate(images):
        img_t = _load_image_tensor(img_path, device, pipe.unet.dtype)

        # DDIM inversion — same as TR/GS/RingID
        t0 = time.perf_counter()
        with torch.inference_mode():
            image_latents = pipe.get_image_latents(img_t, sample=False)
            reversed_latents = pipe.forward_diffusion(
                latents=image_latents, text_embeddings=text_embeddings,
                guidance_scale=1, num_inference_steps=STEPS)
        t1 = time.perf_counter()

        # Correlation score computation
        if sample_indices is not None and bits_for_samples is not None:
            rev_np = reversed_latents.detach().float().cpu().numpy().squeeze()
            # Use channel 0 as reference (consistent with PP-Mark default)
            if rev_np.ndim == 3:
                aligned = rev_np[0]
            else:
                aligned = rev_np
            flat = aligned.ravel()
            ys = np.array(sample_indices, dtype=np.int64) // cfg.image.width
            xs = np.array(sample_indices, dtype=np.int64) % cfg.image.width
            samples = aligned[ys, xs]
            signs = 2.0 * np.array(bits_for_samples, dtype=np.float32) - 1.0
            watermark = alpha * signs
            numerator = float(np.dot(samples, watermark))
            denom = float(np.linalg.norm(watermark) + 1e-9)
            score = abs(numerator / denom)
        t2 = time.perf_counter()

        t_inversion.append(t1 - t0)
        t_detection.append(t2 - t1)
        t_total.append(t2 - t0)
        print(f"  [PP-Mark verify] {len(t_total)}/{len(images)} inv={t1-t0:.2f}s det={t2-t1:.4f}s")

    del pipe
    _clear_gpu()
    return {
        "ddim_inv_s": _stats(t_inversion),
        "detection_s": _stats(t_detection),
        "total_s": _stats(t_total),
        "receipt_verify_s": {"mean": 7.59, "std": 0.0, "note": "from Table 5 (N=1000, CPU verify)"},
    }


# ===========================================================================
# GENERATION BENCHMARKS
# ===========================================================================

def bench_gen_tree_ring(n: int, device: str) -> Dict[str, Any]:
    """Tree-Ring generation timing."""
    import torch
    from types import SimpleNamespace

    tr_root = REPO_ROOT / "external" / "tree-ring-watermark"
    sys.path.insert(0, str(tr_root))
    from diffusers import DPMSolverMultistepScheduler
    from inverse_stable_diffusion import InversableStableDiffusionPipeline
    from optim_utils import get_watermarking_mask, get_watermarking_pattern, inject_watermark, set_random_seed

    keys = json.loads(KEYS_JSON.read_text())
    w_seed = int(keys["tree_ring"]["seed"])

    scheduler = DPMSolverMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    try:
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            MODEL_ID, scheduler=scheduler, torch_dtype=getattr(torch, DTYPE_STR), revision="fp16")
    except Exception:
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            MODEL_ID, scheduler=scheduler, torch_dtype=getattr(torch, DTYPE_STR))
    pipe.safety_checker = None
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    helper_args = SimpleNamespace(
        w_seed=w_seed, w_channel=3, w_pattern="ring", w_mask_shape="circle",
        w_radius=10, w_measurement="l1_complex", w_injection="complex", w_pattern_const=0.0)
    gt_patch = get_watermarking_pattern(pipe, helper_args, device)

    manifest = _sample_manifest_rows("tree_ring", n)
    if not manifest:
        manifest = [{"prompt_text": "a photo of a cat", "seed": 42 + i} for i in range(n)]

    t_embed, t_pipe, t_total = [], [], []
    for i, row in enumerate(manifest[:n]):
        prompt = row.get("prompt_text", "a photo of a cat")
        seed = int(row.get("seed", 42 + i))

        set_random_seed(seed)
        init_latents = pipe.get_random_latents()
        mask = get_watermarking_mask(init_latents, helper_args, device)

        t0 = time.perf_counter()
        init_latents_w = inject_watermark(init_latents, mask, gt_patch, helper_args)
        t1 = time.perf_counter()
        result = pipe(prompt, num_images_per_prompt=1, guidance_scale=GUIDANCE,
                      num_inference_steps=STEPS, height=IMAGE_SIZE, width=IMAGE_SIZE,
                      latents=init_latents_w)
        t2 = time.perf_counter()

        t_embed.append(t1 - t0)
        t_pipe.append(t2 - t1)
        t_total.append(t2 - t0)
        print(f"  [TR gen] {i+1}/{n} embed={t1-t0:.4f}s pipe={t2-t1:.2f}s")

    del pipe
    _clear_gpu()
    return {"embed_s": _stats(t_embed), "pipe_s": _stats(t_pipe), "total_s": _stats(t_total)}


def bench_gen_gaussian_shading(n: int, device: str) -> Dict[str, Any]:
    """Gaussian Shading generation timing."""
    import torch, random

    gs_root = REPO_ROOT / "external" / "Gaussian-Shading"
    sys.path.insert(0, str(gs_root))
    from diffusers import DPMSolverMultistepScheduler
    from inverse_stable_diffusion import InversableStableDiffusionPipeline
    from watermark import Gaussian_Shading
    from image_utils import set_random_seed

    keys = json.loads(KEYS_JSON.read_text())
    gs_cfg = keys["gaussian_shading"]

    scheduler = DPMSolverMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    try:
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            MODEL_ID, scheduler=scheduler, torch_dtype=getattr(torch, DTYPE_STR), revision="fp16")
    except Exception:
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            MODEL_ID, scheduler=scheduler, torch_dtype=getattr(torch, DTYPE_STR))
    pipe.safety_checker = None
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    wm = Gaussian_Shading(1, 8, float(gs_cfg["detection_fpr"]), int(gs_cfg["user_pool_size"]))
    set_random_seed(int(gs_cfg["alice_user_index"]))
    wm.key = torch.randint(0, 2, [1, 4, 64, 64]).to(device)
    wm.watermark = torch.randint(0, 2, [1, 4 // wm.ch, 64 // wm.hw, 64 // wm.hw]).to(device)
    sd = wm.watermark.repeat(1, wm.ch, wm.hw, wm.hw)
    base_message = ((sd + wm.key) % 2).flatten().cpu().numpy()

    manifest = _sample_manifest_rows("gaussian_shading", n)
    if not manifest:
        manifest = [{"prompt_text": "a photo of a cat", "seed": 42 + i} for i in range(n)]

    t_embed, t_pipe, t_total = [], [], []
    for i, row in enumerate(manifest[:n]):
        prompt = row.get("prompt_text", "a photo of a cat")
        seed = int(row.get("seed", 42 + i))

        set_random_seed(seed)
        np.random.seed(seed + 3)

        t0 = time.perf_counter()
        w_latents = wm.truncSampling(base_message)
        t1 = time.perf_counter()
        result = pipe(prompt, num_images_per_prompt=1, guidance_scale=GUIDANCE,
                      num_inference_steps=STEPS, height=IMAGE_SIZE, width=IMAGE_SIZE,
                      latents=w_latents)
        t2 = time.perf_counter()

        t_embed.append(t1 - t0)
        t_pipe.append(t2 - t1)
        t_total.append(t2 - t0)
        print(f"  [GS gen] {i+1}/{n} embed={t1-t0:.4f}s pipe={t2-t1:.2f}s")

    del pipe
    _clear_gpu()
    return {"embed_s": _stats(t_embed), "pipe_s": _stats(t_pipe), "total_s": _stats(t_total)}


def bench_gen_stable_signature(n: int, device: str) -> Dict[str, Any]:
    """Stable Signature generation: pipe() with modified VAE decoder."""
    import torch
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=getattr(torch, DTYPE_STR))
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    state = torch.load(str(STABLESIG_VAE_DECODER), map_location="cpu")
    pipe.vae.load_state_dict(state, strict=False)

    manifest = _sample_manifest_rows("stable_signature", n)
    if not manifest:
        manifest = [{"prompt_text": "a photo of a cat", "seed": 42 + i} for i in range(n)]

    t_total = []
    for i, row in enumerate(manifest[:n]):
        prompt = row.get("prompt_text", "a photo of a cat")
        seed = int(row.get("seed", 42 + i))

        gen = torch.Generator(device=device).manual_seed(seed)
        t0 = time.perf_counter()
        result = pipe(prompt=prompt, negative_prompt="", num_inference_steps=STEPS,
                      guidance_scale=GUIDANCE, height=IMAGE_SIZE, width=IMAGE_SIZE,
                      generator=gen, output_type="np")
        t1 = time.perf_counter()
        t_total.append(t1 - t0)
        print(f"  [StableSig gen] {i+1}/{n} total={t1-t0:.2f}s")

    del pipe
    _clear_gpu()
    return {"embed_s": {"mean": 0, "std": 0, "note": "embedded in VAE decoder"}, "pipe_s": _stats(t_total), "total_s": _stats(t_total)}


def bench_gen_hidden(n: int, device: str) -> Dict[str, Any]:
    """HiDDeN generation: encoder applied post-hoc to clean images."""
    import torch

    # Load encoder
    sys.path.insert(0, str(REPO_ROOT / "external" / "stable_signature"))
    import utils_model
    try:
        torch.serialization.add_safe_globals([argparse.Namespace])
    except Exception:
        pass
    try:
        ckpt = torch.load(str(HIDDEN_CKPT), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(HIDDEN_CKPT), map_location="cpu")
    params = ckpt["params"]
    encoder = utils_model.get_hidden_encoder(
        num_bits=params.num_bits, num_blocks=params.encoder_depth, channels=params.encoder_channels)
    enc_state = {k.replace("module.", "").replace("encoder.", ""): v
                 for k, v in ckpt["encoder_decoder"].items() if "encoder" in k}
    encoder.load_state_dict(enc_state, strict=False)
    encoder = encoder.to(device).eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    key_bits = np.array([int(c) for c in DEFAULT_KEY_BITS], dtype=np.float32)
    msg_bits = torch.from_numpy(key_bits).to(device) * 2.0 - 1.0

    # Use existing clean images as input
    clean_images = _sample_images("hidden_revisited", n)
    if not clean_images:
        clean_images = _sample_images("stable_signature", n)

    t_encoder, t_total = [], []
    for i, img_path in enumerate(clean_images[:n]):
        img = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device, dtype=torch.float32)
        imgs_norm = (t - mean) / std
        msgs = msg_bits.unsqueeze(0)

        t0 = time.perf_counter()
        with torch.inference_mode():
            deltas = encoder(imgs_norm, msgs)
            imgs_w = imgs_norm + deltas
        t1 = time.perf_counter()

        t_encoder.append(t1 - t0)
        t_total.append(t1 - t0)
        print(f"  [HiDDeN gen] {i+1}/{n} encoder={t1-t0:.4f}s (post-hoc)")

    del encoder
    _clear_gpu()
    return {"embed_s": _stats(t_encoder), "pipe_s": {"mean": 0, "std": 0, "note": "post-hoc, no diffusion pipe"}, "total_s": _stats(t_total), "note": "post-hoc watermarking (no image generation)"}


def bench_gen_ringid(n: int, device: str) -> Dict[str, Any]:
    """RingID generation timing."""
    import torch, itertools, random

    ring_root = REPO_ROOT / "external" / "RingID"
    if str(ring_root) not in sys.path:
        sys.path.insert(0, str(ring_root))
    from diffusers import DPMSolverMultistepScheduler
    from inverse_stable_diffusion import InversableStableDiffusionPipeline
    from utils import (HETER_WATERMARK_CHANNEL, RING_WATERMARK_CHANNEL, WATERMARK_CHANNEL,
                       RADIUS, RADIUS_CUTOFF, fft, ifft, make_Fourier_ringid_pattern,
                       ring_mask, generate_Fourier_watermark_latents, set_random_seed)

    keys = json.loads(KEYS_JSON.read_text())
    ring_cfg = keys["ringid"]

    scheduler = DPMSolverMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    try:
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            MODEL_ID, scheduler=scheduler, torch_dtype=getattr(torch, DTYPE_STR), revision="fp16")
    except Exception:
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            MODEL_ID, scheduler=scheduler, torch_dtype=getattr(torch, DTYPE_STR))
    pipe.safety_checker = None
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    base_latents = pipe.get_random_latents().to(torch.float64)
    shape = base_latents.shape
    sing_mask = torch.tensor(ring_mask(size=shape[-1], r_out=RADIUS, r_in=RADIUS_CUTOFF), device=device)

    if len(HETER_WATERMARK_CHANNEL) > 0:
        heter_mask = sing_mask.unsqueeze(0).repeat(len(HETER_WATERMARK_CHANNEL), 1, 1)
    else:
        heter_mask = None

    wm_region_mask = []
    for ch in WATERMARK_CHANNEL:
        if ch in RING_WATERMARK_CHANNEL:
            wm_region_mask.append(sing_mask)
        else:
            wm_region_mask.append(heter_mask[0] if heter_mask is not None else sing_mask)
    wm_region_mask = torch.stack(wm_region_mask).to(device)

    slot_count = RADIUS - RADIUS_CUTOFF
    per_slot_values = np.linspace(-64, 64, 2).tolist()
    key_value_list = [[list(c) for c in itertools.product(per_slot_values, repeat=len(RING_WATERMARK_CHANNEL))] for _ in range(slot_count)]
    key_value_combinations = list(itertools.product(*key_value_list))
    rng = random.Random(int(ring_cfg["seed"]))
    key_index = rng.randrange(len(key_value_combinations))
    kvc = key_value_combinations[key_index]

    watermark_pattern = make_Fourier_ringid_pattern(
        device, list(kvc), base_latents, radius=RADIUS, radius_cutoff=RADIUS_CUTOFF,
        ring_watermark_channel=RING_WATERMARK_CHANNEL, heter_watermark_channel=HETER_WATERMARK_CHANNEL,
        heter_watermark_region_mask=heter_mask, ring_width=1)
    watermark_pattern = fft(ifft(watermark_pattern).real)
    if len(RING_WATERMARK_CHANNEL) > 0:
        rs = watermark_pattern[:, RING_WATERMARK_CHANNEL, ...]
        rs = fft(torch.fft.fftshift(ifft(rs), dim=(-1, -2)))
        watermark_pattern[:, RING_WATERMARK_CHANNEL, ...] = rs

    manifest = _sample_manifest_rows("ringid", n)
    if not manifest:
        manifest = [{"prompt_text": "a photo of a cat", "seed": 42 + i} for i in range(n)]

    t_embed, t_pipe, t_total = [], [], []
    for i, row in enumerate(manifest[:n]):
        prompt = row.get("prompt_text", "a photo of a cat")
        seed = int(row.get("seed", 42 + i))

        set_random_seed(seed)
        no_wm_latents = pipe.get_random_latents()

        t0 = time.perf_counter()
        wm_latents = generate_Fourier_watermark_latents(
            device, radius=RADIUS, radius_cutoff=RADIUS_CUTOFF,
            watermark_region_mask=wm_region_mask, watermark_channel=WATERMARK_CHANNEL,
            original_latents=no_wm_latents, watermark_pattern=watermark_pattern)
        t1 = time.perf_counter()
        result = pipe(prompt, num_images_per_prompt=1, guidance_scale=GUIDANCE,
                      num_inference_steps=STEPS, height=IMAGE_SIZE, width=IMAGE_SIZE,
                      latents=wm_latents)
        t2 = time.perf_counter()

        t_embed.append(t1 - t0)
        t_pipe.append(t2 - t1)
        t_total.append(t2 - t0)
        print(f"  [RingID gen] {i+1}/{n} embed={t1-t0:.4f}s pipe={t2-t1:.2f}s")

    del pipe
    _clear_gpu()
    return {"embed_s": _stats(t_embed), "pipe_s": _stats(t_pipe), "total_s": _stats(t_total)}


def bench_gen_wind(n: int, device: str) -> Dict[str, Any]:
    """WIND generation timing."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from wind_common import WindWatermarker, build_pipe
    from types import SimpleNamespace

    wind_args = SimpleNamespace(
        model_id=MODEL_ID, dtype=DTYPE_STR, device=device,
        general_seed=42, watermark_seed=5, ring_width=1, num_inmost_keys=2,
        ring_value_range=64, assigned_keys=-1, fix_gt=1, time_shift=1,
        time_shift_factor=1.0, channel_min=1, M=10000,
        num_inference_steps=STEPS, test_num_inference_steps=STEPS,
        guidance_scale=GUIDANCE, image_length=IMAGE_SIZE,
        precompute_cache=str(REPO_ROOT / "outputs" / "benchmark_runtime" / "wind_cache.pth"),
        rotation_angles=[0.0, 74.5, 75.5],
    )

    pipe, _ = build_pipe(MODEL_ID, DTYPE_STR, device)
    watermarker = WindWatermarker(wind_args, pipe, device)

    manifest = _sample_manifest_rows("tree_ring", n)  # borrow prompts
    if not manifest:
        manifest = [{"prompt_text": "a photo of a cat", "seed": 42 + i} for i in range(n)]

    t_total = []
    wind_images = []  # save for verification
    out_dir = REPO_ROOT / "outputs" / "benchmark_runtime" / "wind_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(manifest[:n]):
        prompt = row.get("prompt_text", "a photo of a cat")
        seed = int(row.get("seed", 42 + i))

        t0 = time.perf_counter()
        image, true_n, true_m, _ = watermarker.encode_and_generate(prompt, seed)
        t1 = time.perf_counter()

        img_path = out_dir / f"wind_{i:04d}.png"
        image.save(img_path)
        wind_images.append(img_path)
        t_total.append(t1 - t0)
        print(f"  [WIND gen] {i+1}/{n} total={t1-t0:.2f}s")

    del pipe, watermarker
    _clear_gpu()
    return {"embed_s": {"note": "included in pipe"}, "pipe_s": _stats(t_total), "total_s": _stats(t_total), "_wind_images": [str(p) for p in wind_images]}


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Runtime benchmark for watermark methods.")
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--output", default="outputs/benchmark_runtime/results.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-verification", action="store_true")
    args = parser.parse_args()

    device = args.device
    n = args.num_images
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "config": {
            "num_images": n, "model_id": MODEL_ID, "steps": STEPS,
            "image_size": IMAGE_SIZE, "device": device, "dtype": DTYPE_STR,
        },
        "generation": {},
        "verification": {},
    }

    # ---- GENERATION ----
    if not args.skip_generation:
        print("\n=== GENERATION BENCHMARKS ===\n")

        print("[1/7] Tree-Ring generation...")
        results["generation"]["tree_ring"] = bench_gen_tree_ring(n, device)

        print("[2/7] Gaussian Shading generation...")
        results["generation"]["gaussian_shading"] = bench_gen_gaussian_shading(n, device)

        print("[3/7] Stable Signature generation...")
        results["generation"]["stable_signature"] = bench_gen_stable_signature(n, device)

        print("[4/7] HiDDeN generation (post-hoc)...")
        results["generation"]["hidden"] = bench_gen_hidden(n, device)

        print("[5/7] WIND generation...")
        wind_gen = bench_gen_wind(n, device)
        results["generation"]["wind"] = wind_gen

        print("[6/7] RingID generation...")
        results["generation"]["ringid"] = bench_gen_ringid(n, device)

        print("[7/7] PP-Mark generation...")
        results["generation"]["pp_mark"] = {"note": "Requires SP1 binary for proof generation. Diffusion pipe identical to other latent-injection methods. Proof generation: 48.6s (Table 5, N=1000). Total = pipe + 48.6s."}

    # ---- VERIFICATION ----
    if not args.skip_verification:
        print("\n=== VERIFICATION BENCHMARKS ===\n")

        tr_imgs = _sample_images("tree_ring", n)
        gs_imgs = _sample_images("gaussian_shading", n)
        ri_imgs = _sample_images("ringid", n)
        ss_imgs = _sample_images("stable_signature", n)
        hd_imgs = _sample_images("hidden_revisited", n)
        pp_imgs = _sample_images("pp_mark", n)

        # WIND images from generation, or generate if skipped
        wind_dir = REPO_ROOT / "outputs" / "benchmark_runtime" / "wind_eval"
        wind_imgs = sorted(wind_dir.glob("*.png"))[:n] if wind_dir.exists() else []

        print(f"[1/7] Tree-Ring verification ({len(tr_imgs)} images)...")
        results["verification"]["tree_ring"] = bench_verify_tree_ring(tr_imgs, device)

        print(f"[2/7] Gaussian Shading verification ({len(gs_imgs)} images)...")
        results["verification"]["gaussian_shading"] = bench_verify_gaussian_shading(gs_imgs, device)

        print(f"[3/7] RingID verification ({len(ri_imgs)} images)...")
        results["verification"]["ringid"] = bench_verify_ringid(ri_imgs, device)

        print(f"[4/7] Stable Signature verification ({len(ss_imgs)} images)...")
        results["verification"]["stable_signature"] = bench_verify_stable_signature(ss_imgs, device)

        print(f"[5/7] HiDDeN verification ({len(hd_imgs)} images)...")
        results["verification"]["hidden"] = bench_verify_hidden(hd_imgs, device)

        if wind_imgs:
            print(f"[6/7] WIND verification ({len(wind_imgs)} images)...")
            results["verification"]["wind"] = bench_verify_wind(wind_imgs, device)
        else:
            print("[6/7] WIND verification SKIPPED (no images)")
            results["verification"]["wind"] = {"note": "skipped — no eval images. Run generation first."}

        print(f"[7/7] PP-Mark verification ({len(pp_imgs)} images)...")
        results["verification"]["pp_mark"] = bench_verify_ppmark(pp_imgs, device)

    # ---- SAVE ----
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
