#!/usr/bin/env python3
"""Independent preflight and final audit for frozen K5 B0 generation.

The verifier deliberately does not import the generation implementation.  It
reconstructs all 448 tasks, decodes and hashes every output, independently
recomputes the pixel criteria, and regenerates the first and last fixed
four-image batches before atomically publishing the manifest consumed by the
full training runner.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


RUN_ID = "nih_cxr14_k5_feasibility_v1_001"
MODEL_ID = "Manojb/stable-diffusion-2-1-base"
MODEL_REVISION = "0094d483a120f3f33dafbd187ea4aa60d10de75c"
GENERATION_SALT = "nih-cxr14-k5-feasibility-generation-v1"
GENERATION_TASKS_SHA256 = "B0379A2A1EBDFB8758AAD865D6F709D803C7632D2697A7BB9A5DFD9C23F2124B"
FEASIBILITY_PROTOCOL_SHA256 = "2234B3BA8701565B768A11AB5196B2F696788900D07254833997D7823489E9DA"
PRIMARY_GATE_SHA256 = "661A0F3F135C6DA4F0CAF1D000AD1A1B5A5CAC7A23786081F86D1A380199FE74"
LAUNCH_AUTHORITY_SHA256 = "A1A07C51FBFFF6462A6248D45E77D6C25A8F9041A7F5B675092EC8A0BF2551F2"
ENVIRONMENT_MANIFEST_SHA256 = "47BED599FE05BD9A7DA710BA80420D6B335943328A5E52D0188FA953E9767767"
BASE_SNAPSHOT_MANIFEST_SHA256 = "E38A5F8FE1695745EDD60FB6E2EEDFD35A01A65B5E981ACA28EEF026F4C02638"
BASE_GATE_RESULT_SHA256 = "0B9BF8031D0C182D4E695EA8995D6E1F27A1F6B509B38E99A5E25DEBA21EEB35"

PROTOCOL_SCHEMA = "nih-cxr14-k5-b0-generation-protocol/v1"
PROGRESS_SCHEMA = "nih-cxr14-k5-b0-generation-progress/v1"
PRIMARY_SCHEMA = "nih-cxr14-k5-b0-primary-generation-result/v1"
PREFLIGHT_SCHEMA = "nih-cxr14-k5-b0-independent-preflight/v1"
INDEPENDENT_SCHEMA = "nih-cxr14-k5-b0-independent-generation-verification/v1"
B0_MANIFEST_SCHEMA = "nih-cxr14-k5-b0-generation-manifest/v1"

IMAGE_SIZE = 256
BATCH_SIZE = 4
EXPECTED_IMAGE_COUNT = 448
INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
PNG_COMPRESS_LEVEL = 9
SENTINEL_BATCH_INDICES = (0, 111)

CONDITIONS = (
    ("no_finding", "no_finding", "with no labeled finding"),
    ("pneumothorax", "pneumothorax", "with radiographic findings of pneumothorax"),
    ("pneumonia", "pneumonia_or_consolidation", "with radiographic findings of pneumonia"),
    ("consolidation", "pneumonia_or_consolidation", "with radiographic findings of consolidation"),
    ("pleural_effusion", "pleural_effusion", "with radiographic findings of pleural effusion"),
    ("mass_opacity", "mass_or_nodule", "with radiographic findings of mass opacity"),
    ("nodule_opacity", "mass_or_nodule", "with radiographic findings of nodule opacity"),
)
SOURCE_FILES = (
    "dp_training/k5_generation.py",
    "dp_training/run_k5_b0_generation.py",
)
AUDIT_FILES = (
    "dp_training/build_k5_b0_generation_protocol.py",
    "dp_training/test_k5_generation.py",
    "dp_training/verify_k5_b0_generation_independent.py",
)
PACKAGE_NAMES = (
    "torch",
    "torchvision",
    "diffusers",
    "peft",
    "transformers",
    "safetensors",
    "accelerate",
    "huggingface-hub",
    "opacus",
    "dp-accounting",
    "numpy",
    "Pillow",
)


def root_path() -> Path:
    return Path(__file__).resolve().parent.parent


def paths() -> dict[str, Path]:
    root = root_path()
    protocol_dir = root / "_reports" / "nih_cxr14_k5_b0_generation_protocol_v1_002"
    return {
        "root": root,
        "protocol_dir": protocol_dir,
        "protocol": protocol_dir / "protocol.json",
        "build_report": protocol_dir / "build_report.json",
        "preflight": protocol_dir / "preflight_independent.json",
        "tasks": root / "_reports" / "nih_cxr14_k5_feasibility_protocol_v1_001" / "generation_tasks.csv",
        "feasibility_protocol": root / "_reports" / "nih_cxr14_k5_feasibility_protocol_v1_001" / "protocol.json",
        "primary_gate": root / "_reports" / "nih_cxr14_k5_full_runner_gate_v1_001" / "primary_gate.json",
        "launch_authority": root / "_reports" / "nih_cxr14_k5_full_runner_gate_v1_001" / "independent_launch_authority.json",
        "environment": root / "_reports" / "nih_cxr14_k5_full_runner_gate_v1_001" / "environment_manifest.json",
        "base_snapshot_manifest": root / "_reports" / "base_gate_sd21_manojb_r0094_001" / "base_snapshot_manifest.json",
        "base_gate_result": root / "_reports" / "base_gate_sd21_manojb_r0094_001" / "base_gate_result.json",
        "staging": root / "_restricted_generation" / RUN_ID / "B0",
        "full_run": root / "_restricted_runs" / RUN_ID,
        "public_root": root / "_reports" / RUN_ID,
        "public_b0": root / "_reports" / RUN_ID / "B0",
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def sha256_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest().upper()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def json_body(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"refusing to overwrite independent artifact: {path}")
    temporary = path.with_name(path.name + ".new")
    require(not temporary.exists(), f"stale independent temporary requires audit: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(json_body(value))
        handle.flush()
        os.fsync(handle.fileno())
    load_json(temporary)
    os.replace(temporary, path)


def canonical_seed(condition: str, index: int) -> int:
    message = f"{GENERATION_SALT}|{condition}|{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(message).digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def reconstruct_tasks(path: Path) -> list[dict[str, Any]]:
    require(sha256_file(path) == GENERATION_TASKS_SHA256, "independent task hash drift")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == ["task_id", "condition", "metric_stratum", "prompt", "seed"], "task schema")
        raw = list(reader)
    require(len(raw) == EXPECTED_IMAGE_COUNT, "task count")
    output = []
    prefix = "a frontal posteroanterior chest radiograph"
    cursor = 0
    for condition, stratum, suffix in CONDITIONS:
        for index in range(64):
            expected = {
                "task_id": f"{condition}_{index:03d}",
                "condition": condition,
                "metric_stratum": stratum,
                "prompt": f"{prefix} {suffix}",
                "seed": str(canonical_seed(condition, index)),
            }
            require(raw[cursor] == expected, f"task reconstruction mismatch at row {cursor + 2}")
            output.append({**expected, "seed": int(expected["seed"]), "task_index": cursor})
            cursor += 1
    return output


def relative_path(task: dict[str, Any]) -> str:
    return f"images/{task['condition']}/{task['task_id']}.png"


def analyze_png(path: Path) -> dict[str, Any]:
    body = path.read_bytes()
    with Image.open(io.BytesIO(body)) as image:
        image.load()
        require(image.format == "PNG", f"not PNG: {path.name}")
        require(image.mode == "RGB", f"not RGB: {path.name}")
        require(image.size == (IMAGE_SIZE, IMAGE_SIZE), f"not P256: {path.name}")
        rgb = np.asarray(image, dtype=np.uint8)
    grayscale = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.uint8)
    grayscale_01 = grayscale.astype(np.float64) / 255.0
    std = float(np.std(grayscale_01, dtype=np.float64))
    extreme = float(np.mean((grayscale <= 1) | (grayscale >= 254), dtype=np.float64))
    return {
        "bytes": len(body),
        "sha256": sha256_bytes(body),
        "width": IMAGE_SIZE,
        "height": IMAGE_SIZE,
        "mode": "RGB",
        "grayscale_std": std,
        "extreme_pixel_fraction": extreme,
        "low_contrast": std < 0.02,
        "saturated": extreme > 0.98,
    }


def summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    hashes = [str(record["sha256"]) for record in records]
    duplicate_count = len(hashes) - len(set(hashes))
    low_count = sum(bool(record["low_contrast"]) for record in records)
    saturation_count = sum(bool(record["saturated"]) for record in records)
    count = len(records)
    value = {
        "decode_rate": 1.0,
        "exact_duplicate_surplus_count": duplicate_count,
        "exact_duplicate_fraction": duplicate_count / count,
        "low_contrast_image_count": low_count,
        "low_contrast_fraction": low_count / count,
        "saturated_image_count": saturation_count,
        "saturation_fraction": saturation_count / count,
        "condition_counts": dict(Counter(str(record["condition"]) for record in records)),
    }
    checks = {
        "decode_rate": value["decode_rate"] == 1.0,
        "exact_duplicate_fraction": value["exact_duplicate_fraction"] <= 0.01,
        "low_contrast_fraction": value["low_contrast_fraction"] <= 0.01,
        "saturation_fraction": value["saturation_fraction"] <= 0.01,
    }
    return {**value, "checks": checks, "status": "PASS" if all(checks.values()) else "FAIL"}


def inventory_sha(records: Sequence[dict[str, Any]]) -> str:
    return sha256_bytes(canonical_json(sorted(records, key=lambda value: int(value["task_index"]))))


def validate_static() -> dict[str, Any]:
    p = paths()
    protocol = load_json(p["protocol"])
    build = load_json(p["build_report"])
    protocol_sha = sha256_file(p["protocol"])
    require(build["protocol_sha256"] == protocol_sha, "protocol/build hash mismatch")
    require(protocol["schema"] == PROTOCOL_SCHEMA, "protocol schema")
    require(protocol["run_id"] == RUN_ID and protocol["model"] == "B0", "protocol identity")
    require(protocol["expected_images"] == 448, "expected image count")
    require(protocol["sentinel_batch_indices"] == [0, 111], "sentinel indices")
    expected_generation = {
        "sampler": "DDIM",
        "inference_steps": 50,
        "guidance_scale": 7.5,
        "negative_prompt": None,
        "batch_size": 4,
        "height": 256,
        "width": 256,
        "output_type": "np",
        "uint8_quantization": "clip(x * 255.0, 0.0, 255.0).astype(uint8); truncation toward zero",
        "png": {"format": "PNG", "mode": "RGB", "compress_level": 9, "optimize": False, "metadata": "none"},
    }
    require(protocol["generation"] == expected_generation, "generation contract drift")
    require(
        protocol["pipeline_loading"][
            "broken_inherited_onnxruntime_masked_from_diffusers_feature_detection"
        ]
        is True,
        "optional ONNX masking contract",
    )

    upstream_paths = {
        "feasibility_protocol": p["feasibility_protocol"],
        "generation_tasks": p["tasks"],
        "primary_gate": p["primary_gate"],
        "launch_authority": p["launch_authority"],
        "environment_manifest": p["environment"],
        "base_snapshot_manifest": p["base_snapshot_manifest"],
        "base_gate_result": p["base_gate_result"],
    }
    expected_upstream = {
        "feasibility_protocol": FEASIBILITY_PROTOCOL_SHA256,
        "generation_tasks": GENERATION_TASKS_SHA256,
        "primary_gate": PRIMARY_GATE_SHA256,
        "launch_authority": LAUNCH_AUTHORITY_SHA256,
        "environment_manifest": ENVIRONMENT_MANIFEST_SHA256,
        "base_snapshot_manifest": BASE_SNAPSHOT_MANIFEST_SHA256,
        "base_gate_result": BASE_GATE_RESULT_SHA256,
    }
    actual_upstream = {name: sha256_file(path) for name, path in upstream_paths.items()}
    require(actual_upstream == expected_upstream == protocol["upstream_sha256"], "upstream evidence drift")

    actual_sources = {relative: sha256_file(p["root"] / Path(relative)) for relative in SOURCE_FILES}
    actual_audit = {relative: sha256_file(p["root"] / Path(relative)) for relative in AUDIT_FILES}
    require(actual_sources == protocol["generator_source_sha256"], "generator source drift")
    require(actual_audit == protocol["audit_source_sha256"], "B0 audit source drift")

    authority = load_json(p["launch_authority"])
    require(authority["status"] == "PASS_K5_FULL_RUNNER_GATE_NO_FULL_TRAINING_STARTED", "launch authority status")
    require(authority["full_K5_optimizer_execution_started"] is False, "optimizer boundary")
    require(authority["generation_tasks_sha256"] == GENERATION_TASKS_SHA256, "authority task hash")
    environment = load_json(p["environment"])
    packages = {name: importlib.metadata.version(name) for name in PACKAGE_NAMES}
    require(packages == environment["packages"], "package environment drift")
    require(platform.python_version() == environment["python"]["version"], "Python version drift")
    executable = Path(sys.executable).resolve()
    require(executable == Path(environment["python"]["executable"]).resolve(), "Python executable drift")
    require(sha256_file(executable) == environment["python"]["executable_sha256"], "Python binary drift")
    snapshot = Path(environment["frozen_inputs"]["model_snapshot"]).resolve()
    require(snapshot.is_dir() and snapshot.name == MODEL_REVISION, "model snapshot")
    require(environment["frozen_inputs"]["model_id"] == MODEL_ID, "model ID")
    for relative, check in environment["frozen_inputs"]["critical_model_hashes"].items():
        require(check["expected"] == check["actual"] == sha256_file(snapshot / relative), f"model hash: {relative}")
    tasks = reconstruct_tasks(p["tasks"])
    return {
        "paths": p,
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "environment": environment,
        "snapshot": snapshot,
        "tasks": tasks,
        "source_hashes": actual_sources,
        "audit_hashes": actual_audit,
        "upstream_hashes": actual_upstream,
    }


def run_preflight(context: dict[str, Any]) -> dict[str, Any]:
    p = context["paths"]
    require(not p["staging"].exists(), "B0 staging already exists before independent preflight")
    require(not p["full_run"].exists(), "full run already exists before B0")
    require(not p["public_root"].exists(), "public K5 run root already exists before B0")
    require(not p["preflight"].exists(), "refusing to overwrite B0 preflight")
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "dp_training.test_k5_generation", "-v"],
        cwd=p["root"],
        check=True,
        capture_output=True,
        text=True,
    )
    combined = (completed.stdout + completed.stderr).encode("utf-8")
    report = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "PASS_B0_PROTOCOL_AND_IMPLEMENTATION_BEFORE_OUTPUT",
        "run_id": RUN_ID,
        "model": "B0",
        "protocol_sha256": context["protocol_sha256"],
        "generator_source_sha256": context["source_hashes"],
        "audit_source_sha256": context["audit_hashes"],
        "verifier_source_sha256": sha256_file(Path(__file__).resolve()),
        "generation_tasks_sha256": GENERATION_TASKS_SHA256,
        "all_448_tasks_reconstructed": True,
        "fixed_batches": 112,
        "unit_tests": 7,
        "unit_test_output_sha256": sha256_bytes(combined),
        "B0_outputs_present": False,
        "full_K5_optimizer_execution_started": False,
        "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    write_json_exclusive(p["preflight"], report)
    return report


def verify_record(record: dict[str, Any], task: dict[str, Any], image_path: Path) -> dict[str, Any]:
    expected = {
        "task_index": task["task_index"],
        "task_id": task["task_id"],
        "condition": task["condition"],
        "metric_stratum": task["metric_stratum"],
        "prompt": task["prompt"],
        "seed": task["seed"],
        "relative_path": relative_path(task),
        "state": "committed",
    }
    for key, value in expected.items():
        require(record.get(key) == value, f"record/task mismatch {task['task_id']}: {key}")
    actual = analyze_png(image_path)
    for key in ("bytes", "sha256", "width", "height", "mode", "low_contrast", "saturated"):
        require(record.get(key) == actual[key], f"record/file mismatch {task['task_id']}: {key}")
    for key in ("grayscale_std", "extreme_pixel_fraction"):
        require(abs(float(record.get(key)) - float(actual[key])) <= 1e-15, f"pixel statistic mismatch {task['task_id']}: {key}")
    return record


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
    require(len(rows) == 1, "one NVIDIA GPU required for sentinels")
    name, driver, total, free, used, temperature, utilization = [part.strip() for part in rows[0].split(",")]
    return {
        "name": name,
        "driver": driver,
        "memory_total_mib": int(total),
        "memory_free_mib": int(free),
        "memory_used_mib": int(used),
        "temperature_c": int(temperature),
        "utilization_percent": int(utilization),
    }


def require_idle(state: dict[str, Any], environment: dict[str, Any]) -> None:
    frozen = environment["cuda_device"]
    require(state["name"] == frozen["name"] and state["driver"] == frozen["driver"], "sentinel GPU drift")
    require(state["memory_total_mib"] == frozen["memory_total_mib"], "sentinel GPU memory drift")
    require(state["memory_free_mib"] >= 6500, "sentinel verification requires 6500 MiB free GPU memory")
    require(state["temperature_c"] < 80, "GPU too hot for sentinel verification")
    require(state["utilization_percent"] <= 10, "another GPU workload is active")


def encode_independent(image: np.ndarray) -> bytes:
    value = np.asarray(image)
    require(value.shape == (256, 256, 3) and bool(np.isfinite(value).all()), "sentinel pipeline shape")
    u8 = np.clip(value * 255.0, 0.0, 255.0).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(u8).save(buffer, format="PNG", compress_level=9, optimize=False)
    return buffer.getvalue()


def regenerate_sentinels(context: dict[str, Any], records_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import importlib.util

    original_find_spec = importlib.util.find_spec

    def guarded_find_spec(name: str, package: str | None = None):
        if name == "onnxruntime" or name.startswith("onnxruntime."):
            return None
        return original_find_spec(name, package)

    importlib.util.find_spec = guarded_find_spec
    import torch
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    gpu = nvidia_state()
    require_idle(gpu, context["environment"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    pipe = None
    sentinels = []
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            context["snapshot"],
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            local_files_only=True,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        pipe.to("cuda")
        for batch_index in SENTINEL_BATCH_INDICES:
            offset = batch_index * BATCH_SIZE
            batch = context["tasks"][offset : offset + BATCH_SIZE]
            generators = [torch.Generator(device="cuda").manual_seed(task["seed"]) for task in batch]
            with torch.inference_mode():
                result = pipe(
                    prompt=[task["prompt"] for task in batch],
                    negative_prompt=None,
                    num_inference_steps=INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    height=IMAGE_SIZE,
                    width=IMAGE_SIZE,
                    generator=generators,
                    output_type="np",
                )
            values = np.asarray(result.images)
            require(values.shape == (4, 256, 256, 3), "sentinel batch output shape")
            for index, task in enumerate(batch):
                digest = sha256_bytes(encode_independent(values[index]))
                expected = records_by_id[task["task_id"]]["sha256"]
                require(digest == expected, f"independent sentinel mismatch: {task['task_id']}")
                sentinels.append({"batch_index": batch_index, "task_id": task["task_id"], "sha256": digest})
    finally:
        if pipe is not None:
            del pipe
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "status": "PASS",
        "fixed_batch_indices": list(SENTINEL_BATCH_INDICES),
        "images_regenerated": len(sentinels),
        "sentinels": sentinels,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_at_start": gpu,
    }


def publish_atomically(public_b0: Path, independent: dict[str, Any], manifest_base: dict[str, Any]) -> dict[str, Any]:
    temporary = public_b0.with_name(public_b0.name + ".new")
    require(not public_b0.exists(), "refusing to overwrite public B0 result")
    require(not temporary.exists(), f"stale B0 publication directory requires audit: {temporary}")
    temporary.mkdir(parents=True)
    independent_path = temporary / "independent_verification.json"
    write_json_exclusive(independent_path, independent)
    manifest = {
        **manifest_base,
        "independent_verification_file": "independent_verification.json",
        "independent_verification_sha256": sha256_file(independent_path),
    }
    write_json_exclusive(temporary / "generation_manifest.json", manifest)
    load_json(temporary / "generation_manifest.json")
    os.replace(temporary, public_b0)
    return manifest


def run_final(context: dict[str, Any]) -> dict[str, Any]:
    p = context["paths"]
    require(not p["full_run"].exists(), "full K5 run exists; cannot prove B0 preceded private training")
    require(not p["public_b0"].exists(), "public B0 result already exists")
    preflight = load_json(p["preflight"])
    require(preflight["schema"] == PREFLIGHT_SCHEMA, "preflight schema")
    require(preflight["status"] == "PASS_B0_PROTOCOL_AND_IMPLEMENTATION_BEFORE_OUTPUT", "preflight status")
    require(preflight["protocol_sha256"] == context["protocol_sha256"], "preflight protocol drift")
    require(preflight["verifier_source_sha256"] == sha256_file(Path(__file__).resolve()), "verifier changed after preflight")

    staging = p["staging"]
    progress_path = staging / "progress.json"
    primary_path = staging / "primary_generation_result.json"
    progress = load_json(progress_path)
    primary = load_json(primary_path)
    require(progress["schema"] == PROGRESS_SCHEMA and progress["state"] == "PRIMARY_COMPLETE", "progress state")
    require(primary["schema"] == PRIMARY_SCHEMA and primary["status"] == "READY_FOR_INDEPENDENT_VERIFICATION", "primary status")
    for value in (progress, primary):
        require(value["run_id"] == RUN_ID and value["model"] == "B0", "B0 result identity")
        require(value["protocol_sha256"] == context["protocol_sha256"], "B0 result protocol drift")
        require(value["generation_tasks_sha256"] == GENERATION_TASKS_SHA256, "B0 result task drift")
        require(value["source_gate_sha256"] == PRIMARY_GATE_SHA256, "B0 source gate drift")
    require(primary["generated_before_private_training"] is True, "B0 timing assertion")
    require(primary["full_K5_optimizer_steps_started"] == 0, "B0 optimizer boundary")
    require(progress["primary_result_sha256"] == sha256_file(primary_path), "progress/primary link")

    records_by_id = progress["records"]
    require(len(records_by_id) == EXPECTED_IMAGE_COUNT, "progress record count")
    records = []
    for task in context["tasks"]:
        require(task["task_id"] in records_by_id, f"missing record: {task['task_id']}")
        records.append(verify_record(records_by_id[task["task_id"]], task, staging / relative_path(task)))
    expected_files = {relative_path(task) for task in context["tasks"]}
    actual_files = {path.relative_to(staging).as_posix() for path in (staging / "images").rglob("*.png")}
    require(actual_files == expected_files, "independent exact file-set mismatch")
    require(not list((staging / "images").rglob("*.new")), "stale image temporary")
    pixel = summary(records)
    require(pixel["status"] == "PASS", "independent pixel sanity failure")
    digest = inventory_sha(records)
    require(primary["image_count"] == EXPECTED_IMAGE_COUNT, "primary image count")
    require(primary["inventory_sha256"] == digest, "primary inventory hash")
    require(primary["pixel_sanity"] == pixel, "primary/independent pixel result mismatch")

    sentinel = regenerate_sentinels(context, records_by_id)
    require(not p["full_run"].exists(), "full K5 run appeared before B0 publication")
    independent = {
        "schema": INDEPENDENT_SCHEMA,
        "status": "PASS_B0_ALL_448_FILES_AND_FIXED_SENTINELS",
        "run_id": RUN_ID,
        "model": "B0",
        "protocol_sha256": context["protocol_sha256"],
        "preflight_sha256": sha256_file(p["preflight"]),
        "primary_result_sha256": sha256_file(primary_path),
        "progress_sha256": sha256_file(progress_path),
        "generation_tasks_sha256": GENERATION_TASKS_SHA256,
        "source_gate_sha256": PRIMARY_GATE_SHA256,
        "generator_source_sha256": context["source_hashes"],
        "verifier_source_sha256": sha256_file(Path(__file__).resolve()),
        "image_count": EXPECTED_IMAGE_COUNT,
        "all_task_rows_reconstructed": True,
        "all_file_hashes_recomputed": True,
        "all_files_lossless_8bit_RGB_P256_PNG": True,
        "no_missing_extra_or_stale_images": True,
        "inventory_sha256": digest,
        "pixel_sanity": pixel,
        "sentinel_regeneration": sentinel,
        "generated_before_private_training": True,
        "full_K5_optimizer_steps_started": 0,
        "generated_images_publicly_released": False,
        "claim_boundary": "B0 execution integrity only; not radiographic fidelity, clinical utility, privacy, or release evidence",
        "verified_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    manifest_base = {
        "schema": B0_MANIFEST_SCHEMA,
        "status": "PASS_B0_FROZEN_BEFORE_PRIVATE_TRAINING",
        "run_id": RUN_ID,
        "model": "B0",
        "image_count": EXPECTED_IMAGE_COUNT,
        "generation_tasks_sha256": GENERATION_TASKS_SHA256,
        "source_gate_sha256": PRIMARY_GATE_SHA256,
        "b0_protocol_sha256": context["protocol_sha256"],
        "primary_result_sha256": sha256_file(primary_path),
        "restricted_inventory_sha256": digest,
        "pixel_sanity": pixel,
        "sentinel_batches_verified": list(SENTINEL_BATCH_INDICES),
        "generated_before_private_training": True,
        "full_K5_optimizer_steps_started": 0,
        "generated_images_release_eligible": False,
    }
    manifest = publish_atomically(p["public_b0"], independent, manifest_base)
    require(sha256_file(p["public_b0"] / "independent_verification.json") == manifest["independent_verification_sha256"], "published independent hash")
    return {
        "status": manifest["status"],
        "manifest": str(p["public_b0"] / "generation_manifest.json"),
        "manifest_sha256": sha256_file(p["public_b0"] / "generation_manifest.json"),
        "independent_verification_sha256": manifest["independent_verification_sha256"],
        "image_count": EXPECTED_IMAGE_COUNT,
        "pixel_sanity": pixel,
        "sentinel_images_regenerated": sentinel["images_regenerated"],
        "full_K5_optimizer_steps_started": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "final"))
    args = parser.parse_args()
    context = validate_static()
    result = run_preflight(context) if args.command == "preflight" else run_final(context)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
