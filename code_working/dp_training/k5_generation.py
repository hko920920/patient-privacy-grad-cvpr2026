"""Frozen, reusable generation primitives for the NIH-CXR14 K5 study.

This module contains no training code and performs no work at import time.  It
defines the exact prompt/seed validation, RGB PNG serialization, resumable
record checks, and pixel-sanity calculations shared by the B0 generator.  A
later trained-model generator can reuse the same batch primitive without
changing B0's already frozen implementation.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image


RUN_ID = "nih_cxr14_k5_feasibility_v1_001"
MODEL_ID = "Manojb/stable-diffusion-2-1-base"
MODEL_REVISION = "0094d483a120f3f33dafbd187ea4aa60d10de75c"
MODEL_VARIANT = "fp16"
GENERATION_SALT = "nih-cxr14-k5-feasibility-generation-v1"
GENERATION_TASKS_SHA256 = "B0379A2A1EBDFB8758AAD865D6F709D803C7632D2697A7BB9A5DFD9C23F2124B"
FEASIBILITY_PROTOCOL_SHA256 = "2234B3BA8701565B768A11AB5196B2F696788900D07254833997D7823489E9DA"
PRIMARY_GATE_SHA256 = "661A0F3F135C6DA4F0CAF1D000AD1A1B5A5CAC7A23786081F86D1A380199FE74"
LAUNCH_AUTHORITY_SHA256 = "A1A07C51FBFFF6462A6248D45E77D6C25A8F9041A7F5B675092EC8A0BF2551F2"
ENVIRONMENT_MANIFEST_SHA256 = "47BED599FE05BD9A7DA710BA80420D6B335943328A5E52D0188FA953E9767767"

PROTOCOL_SCHEMA = "nih-cxr14-k5-b0-generation-protocol/v1"
PROGRESS_SCHEMA = "nih-cxr14-k5-b0-generation-progress/v1"
PRIMARY_RESULT_SCHEMA = "nih-cxr14-k5-b0-primary-generation-result/v1"
B0_MANIFEST_SCHEMA = "nih-cxr14-k5-b0-generation-manifest/v1"

CONDITIONS = (
    ("no_finding", "no_finding", "with no labeled finding"),
    ("pneumothorax", "pneumothorax", "with radiographic findings of pneumothorax"),
    ("pneumonia", "pneumonia_or_consolidation", "with radiographic findings of pneumonia"),
    (
        "consolidation",
        "pneumonia_or_consolidation",
        "with radiographic findings of consolidation",
    ),
    ("pleural_effusion", "pleural_effusion", "with radiographic findings of pleural effusion"),
    ("mass_opacity", "mass_or_nodule", "with radiographic findings of mass opacity"),
    ("nodule_opacity", "mass_or_nodule", "with radiographic findings of nodule opacity"),
)

IMAGE_SIZE = 256
BATCH_SIZE = 4
INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
PNG_COMPRESS_LEVEL = 9
EXPECTED_IMAGE_COUNT = 448
MAX_DUPLICATE_FRACTION = 0.01
MAX_LOW_CONTRAST_FRACTION = 0.01
MAX_SATURATION_FRACTION = 0.01


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_paths() -> dict[str, Path]:
    root = project_root()
    return {
        "root": root,
        "protocol": root
        / "_reports"
        / "nih_cxr14_k5_b0_generation_protocol_v1_002"
        / "protocol.json",
        "protocol_build_report": root
        / "_reports"
        / "nih_cxr14_k5_b0_generation_protocol_v1_002"
        / "build_report.json",
        "protocol_preflight": root
        / "_reports"
        / "nih_cxr14_k5_b0_generation_protocol_v1_002"
        / "preflight_independent.json",
        "feasibility_protocol": root
        / "_reports"
        / "nih_cxr14_k5_feasibility_protocol_v1_001"
        / "protocol.json",
        "generation_tasks": root
        / "_reports"
        / "nih_cxr14_k5_feasibility_protocol_v1_001"
        / "generation_tasks.csv",
        "primary_gate": root
        / "_reports"
        / "nih_cxr14_k5_full_runner_gate_v1_001"
        / "primary_gate.json",
        "launch_authority": root
        / "_reports"
        / "nih_cxr14_k5_full_runner_gate_v1_001"
        / "independent_launch_authority.json",
        "environment_manifest": root
        / "_reports"
        / "nih_cxr14_k5_full_runner_gate_v1_001"
        / "environment_manifest.json",
        "base_snapshot_manifest": root
        / "_reports"
        / "base_gate_sd21_manojb_r0094_001"
        / "base_snapshot_manifest.json",
        "base_gate_result": root
        / "_reports"
        / "base_gate_sd21_manojb_r0094_001"
        / "base_gate_result.json",
        "staging_root": root / "_restricted_generation" / RUN_ID / "B0",
        "full_run_root": root / "_restricted_runs" / RUN_ID,
        "public_report_root": root / "_reports" / RUN_ID,
        "public_b0_root": root / "_reports" / RUN_ID / "B0",
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".new")
    require(not temporary.exists(), f"stale atomic JSON temporary requires audit: {temporary}")
    body = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    json.loads(temporary.read_text(encoding="utf-8"))
    os.replace(temporary, path)


def canonical_seed(condition: str, index: int) -> int:
    message = f"{GENERATION_SALT}|{condition}|{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(message).digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def load_and_validate_tasks(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), "generation task CSV is missing")
    require(sha256_file(path) == GENERATION_TASKS_SHA256, "generation task hash drift")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(
            reader.fieldnames == ["task_id", "condition", "metric_stratum", "prompt", "seed"],
            "generation task schema drift",
        )
        raw = list(reader)
    require(len(raw) == EXPECTED_IMAGE_COUNT, "generation task count drift")
    rows: list[dict[str, Any]] = []
    prefix = "a frontal posteroanterior chest radiograph"
    cursor = 0
    for condition, metric_stratum, suffix in CONDITIONS:
        for index in range(64):
            expected = {
                "task_id": f"{condition}_{index:03d}",
                "condition": condition,
                "metric_stratum": metric_stratum,
                "prompt": f"{prefix} {suffix}",
                "seed": str(canonical_seed(condition, index)),
            }
            require(raw[cursor] == expected, f"generation task drift at CSV row {cursor + 2}")
            rows.append({**expected, "seed": int(expected["seed"]), "task_index": cursor})
            cursor += 1
    return rows


def relative_image_path(task: dict[str, Any]) -> str:
    return f"images/{task['condition']}/{task['task_id']}.png"


def quantize_pipeline_image(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    require(value.shape == (IMAGE_SIZE, IMAGE_SIZE, 3), f"unexpected pipeline image shape: {value.shape}")
    require(bool(np.isfinite(value).all()), "non-finite generated pixels")
    return np.clip(value * 255.0, 0.0, 255.0).astype(np.uint8)


def encode_rgb_png(image_u8: np.ndarray) -> bytes:
    value = np.asarray(image_u8)
    require(value.dtype == np.uint8, "PNG source must be uint8")
    require(value.shape == (IMAGE_SIZE, IMAGE_SIZE, 3), "PNG source must be P256 RGB")
    buffer = io.BytesIO()
    Image.fromarray(value).save(
        buffer,
        format="PNG",
        compress_level=PNG_COMPRESS_LEVEL,
        optimize=False,
    )
    return buffer.getvalue()


def analyze_rgb_array(image_u8: np.ndarray) -> dict[str, Any]:
    value = np.asarray(image_u8)
    require(value.dtype == np.uint8 and value.shape == (IMAGE_SIZE, IMAGE_SIZE, 3), "invalid RGB array")
    # This exactly freezes PIL's 8-bit ITU-R 601 luma conversion for the study.
    grayscale = np.asarray(Image.fromarray(value).convert("L"), dtype=np.uint8)
    grayscale_01 = grayscale.astype(np.float64) / 255.0
    standard_deviation = float(np.std(grayscale_01, dtype=np.float64))
    extreme_fraction = float(np.mean((grayscale <= 1) | (grayscale >= 254), dtype=np.float64))
    return {
        "grayscale_std": standard_deviation,
        "extreme_pixel_fraction": extreme_fraction,
        "low_contrast": standard_deviation < 0.02,
        "saturated": extreme_fraction > 0.98,
    }


def analyze_png_bytes(body: bytes) -> dict[str, Any]:
    with Image.open(io.BytesIO(body)) as image:
        image.load()
        require(image.format == "PNG", "generated artifact is not PNG")
        require(image.mode == "RGB", "generated PNG is not RGB")
        require(image.size == (IMAGE_SIZE, IMAGE_SIZE), "generated PNG is not P256")
        array = np.asarray(image, dtype=np.uint8)
    return {
        "bytes": len(body),
        "sha256": sha256_bytes(body),
        "width": IMAGE_SIZE,
        "height": IMAGE_SIZE,
        "mode": "RGB",
        **analyze_rgb_array(array),
    }


def analyze_png_file(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"generated PNG is missing: {path}")
    return analyze_png_bytes(path.read_bytes())


def write_bytes_atomic(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"refusing to overwrite generated image: {path}")
    temporary = path.with_name(path.name + ".new")
    require(not temporary.exists(), f"stale atomic image temporary requires audit: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    analyze_png_file(temporary)
    os.replace(temporary, path)


def record_from_png(task: dict[str, Any], body: bytes, state: str = "prepared") -> dict[str, Any]:
    require(state in {"prepared", "committed"}, "invalid generation record state")
    return {
        "task_index": int(task["task_index"]),
        "task_id": str(task["task_id"]),
        "condition": str(task["condition"]),
        "metric_stratum": str(task["metric_stratum"]),
        "prompt": str(task["prompt"]),
        "seed": int(task["seed"]),
        "relative_path": relative_image_path(task),
        "state": state,
        **analyze_png_bytes(body),
    }


def compare_record_to_task(record: dict[str, Any], task: dict[str, Any]) -> None:
    expected = {
        "task_index": int(task["task_index"]),
        "task_id": str(task["task_id"]),
        "condition": str(task["condition"]),
        "metric_stratum": str(task["metric_stratum"]),
        "prompt": str(task["prompt"]),
        "seed": int(task["seed"]),
        "relative_path": relative_image_path(task),
    }
    for key, value in expected.items():
        require(record.get(key) == value, f"progress record drift for {task['task_id']}: {key}")


def compare_record_to_file(record: dict[str, Any], path: Path) -> None:
    actual = analyze_png_file(path)
    for key in ("bytes", "sha256", "width", "height", "mode", "low_contrast", "saturated"):
        require(record.get(key) == actual[key], f"generated file drift for {record['task_id']}: {key}")
    for key in ("grayscale_std", "extreme_pixel_fraction"):
        require(
            abs(float(record.get(key)) - float(actual[key])) <= 1e-15,
            f"generated pixel statistic drift for {record['task_id']}: {key}",
        )


def summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    require(len(records) == EXPECTED_IMAGE_COUNT, "incomplete B0 record set")
    hashes = [str(record["sha256"]) for record in records]
    duplicate_surplus = len(hashes) - len(set(hashes))
    low_contrast = sum(bool(record["low_contrast"]) for record in records)
    saturated = sum(bool(record["saturated"]) for record in records)
    count = len(records)
    summary = {
        "decode_rate": 1.0,
        "exact_duplicate_surplus_count": duplicate_surplus,
        "exact_duplicate_fraction": duplicate_surplus / count,
        "low_contrast_image_count": low_contrast,
        "low_contrast_fraction": low_contrast / count,
        "saturated_image_count": saturated,
        "saturation_fraction": saturated / count,
        "condition_counts": dict(Counter(str(record["condition"]) for record in records)),
    }
    checks = {
        "decode_rate": summary["decode_rate"] == 1.0,
        "exact_duplicate_fraction": summary["exact_duplicate_fraction"] <= MAX_DUPLICATE_FRACTION,
        "low_contrast_fraction": summary["low_contrast_fraction"] <= MAX_LOW_CONTRAST_FRACTION,
        "saturation_fraction": summary["saturation_fraction"] <= MAX_SATURATION_FRACTION,
    }
    return {**summary, "checks": checks, "status": "PASS" if all(checks.values()) else "FAIL"}


def inventory_sha256(records: Sequence[dict[str, Any]]) -> str:
    ordered = sorted(records, key=lambda item: int(item["task_index"]))
    return sha256_bytes(canonical_json(ordered))


def expected_batches(tasks: Sequence[dict[str, Any]]) -> Iterable[list[dict[str, Any]]]:
    require(len(tasks) == EXPECTED_IMAGE_COUNT, "unexpected task count")
    require(len(tasks) % BATCH_SIZE == 0, "task count is not divisible by frozen batch size")
    for offset in range(0, len(tasks), BATCH_SIZE):
        yield list(tasks[offset : offset + BATCH_SIZE])


def configure_deterministic_cuda(torch: Any) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def disable_broken_optional_onnx() -> None:
    """Hide the host's unusable optional ONNX Runtime from Diffusers.

    The base-gate virtual environment intentionally inherits system packages.
    Its host ONNX Runtime is discoverable but its Windows DLL cannot initialize.
    This study uses only the native PyTorch pipeline, so matching the already
    passed SD 2.1 base gate means masking that irrelevant optional backend from
    Diffusers feature detection before Diffusers is imported.
    """

    import importlib.util

    current = importlib.util.find_spec
    if bool(getattr(current, "_k5_masks_broken_onnx", False)):
        return

    def guarded_find_spec(name: str, package: str | None = None):
        if name == "onnxruntime" or name.startswith("onnxruntime."):
            return None
        return current(name, package)

    guarded_find_spec._k5_masks_broken_onnx = True  # type: ignore[attr-defined]
    importlib.util.find_spec = guarded_find_spec


def load_base_pipeline(snapshot: Path) -> Any:
    disable_broken_optional_onnx()
    import torch
    from diffusers import DDIMScheduler, StableDiffusionPipeline

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
    return pipe


def generate_batch_arrays(pipe: Any, tasks: Sequence[dict[str, Any]]) -> list[np.ndarray]:
    import torch

    require(len(tasks) == BATCH_SIZE, "generation must use the frozen batch size")
    generators = [torch.Generator(device="cuda").manual_seed(int(task["seed"])) for task in tasks]
    with torch.inference_mode():
        result = pipe(
            prompt=[str(task["prompt"]) for task in tasks],
            negative_prompt=None,
            num_inference_steps=INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            generator=generators,
            output_type="np",
        )
    values = np.asarray(result.images)
    require(values.shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3), "pipeline batch shape drift")
    return [quantize_pipeline_image(values[index]) for index in range(BATCH_SIZE)]


def nvidia_state() -> dict[str, Any]:
    rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
    require(len(rows) == 1, "exactly one NVIDIA GPU is required")
    name, driver, total, free, used, temperature, utilization = [part.strip() for part in rows[0].split(",")]
    return {
        "name": name,
        "driver": driver,
        "memory_total_mib": int(total),
        "memory_free_mib": int(free),
        "memory_used_mib": int(used),
        "temperature_c": int(temperature),
        "utilization_percent": int(utilization),
        "idle_detection": "WDDM-safe aggregate utilization and free-memory thresholds",
    }


def require_idle_gpu(state: dict[str, Any], expected: dict[str, Any]) -> None:
    require(state["name"] == expected["name"], "GPU name drift")
    require(state["driver"] == expected["driver"], "GPU driver drift")
    require(state["memory_total_mib"] == int(expected["memory_total_mib"]), "GPU memory drift")
    require(state["memory_free_mib"] >= 6_500, "B0 requires at least 6500 MiB free GPU memory")
    require(state["temperature_c"] < 80, "GPU is too hot to start B0")
    require(state["utilization_percent"] <= 10, "GPU utilization indicates another active workload")


def require_free_space(path: Path, minimum_gib: float = 15.0) -> float:
    import shutil

    anchor = path
    while not anchor.exists():
        require(anchor.parent != anchor, "cannot resolve disk anchor")
        anchor = anchor.parent
    free_gib = shutil.disk_usage(anchor).free / (1024.0**3)
    require(free_gib >= minimum_gib, f"free space below {minimum_gib:.1f} GiB")
    return free_gib
