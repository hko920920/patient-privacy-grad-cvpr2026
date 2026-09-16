#!/usr/bin/env python3
"""Fail-closed, resumable generation of the frozen 448-image B0 baseline."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterator

# Set before the first CUDA operation.  Direct script execution is the frozen
# command, but setting this here is still early enough when torch was imported
# without a CUDA context by a wrapper.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from dp_training import k5_generation as core


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
SOURCE_FILES = (
    "dp_training/k5_generation.py",
    "dp_training/run_k5_b0_generation.py",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    paths = core.default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "generate"), nargs="?", default="validate")
    parser.add_argument("--protocol", type=Path, default=paths["protocol"])
    parser.add_argument("--staging-root", type=Path, default=paths["staging_root"])
    parser.add_argument("--start-b0", action="store_true")
    parser.add_argument("--acknowledge-protocol-sha256")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    core.require(path.is_file(), f"required JSON is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_protocol_shape(protocol: dict[str, Any]) -> None:
    core.require(protocol["schema"] == core.PROTOCOL_SCHEMA, "B0 protocol schema drift")
    core.require(protocol["run_id"] == core.RUN_ID and protocol["model"] == "B0", "B0 protocol identity")
    generation = protocol["generation"]
    expected = {
        "sampler": "DDIM",
        "inference_steps": core.INFERENCE_STEPS,
        "guidance_scale": core.GUIDANCE_SCALE,
        "negative_prompt": None,
        "batch_size": core.BATCH_SIZE,
        "height": core.IMAGE_SIZE,
        "width": core.IMAGE_SIZE,
        "output_type": "np",
        "uint8_quantization": "clip(x * 255.0, 0.0, 255.0).astype(uint8); truncation toward zero",
        "png": {
            "format": "PNG",
            "mode": "RGB",
            "compress_level": core.PNG_COMPRESS_LEVEL,
            "optimize": False,
            "metadata": "none",
        },
    }
    core.require(generation == expected, "frozen generation settings drift")
    core.require(protocol["expected_images"] == core.EXPECTED_IMAGE_COUNT, "B0 expected count drift")
    core.require(protocol["sentinel_batch_indices"] == [0, 111], "sentinel batch drift")
    core.require(
        protocol["pipeline_loading"][
            "broken_inherited_onnxruntime_masked_from_diffusers_feature_detection"
        ]
        is True,
        "optional ONNX masking contract drift",
    )


def validate_static_inputs(protocol_path: Path) -> dict[str, Any]:
    paths = core.default_paths()
    root = paths["root"]
    protocol_path = protocol_path.resolve()
    protocol = load_json(protocol_path)
    protocol_sha = core.sha256_file(protocol_path)
    build_report = load_json(protocol_path.parent / "build_report.json")
    core.require(build_report["protocol_sha256"] == protocol_sha, "B0 protocol build hash mismatch")
    core.require(build_report["status"] == "FROZEN_BEFORE_ANY_B0_OUTPUT", "B0 build status")
    validate_protocol_shape(protocol)

    upstream_paths = {
        "feasibility_protocol": paths["feasibility_protocol"],
        "generation_tasks": paths["generation_tasks"],
        "primary_gate": paths["primary_gate"],
        "launch_authority": paths["launch_authority"],
        "environment_manifest": paths["environment_manifest"],
        "base_snapshot_manifest": paths["base_snapshot_manifest"],
        "base_gate_result": paths["base_gate_result"],
    }
    actual_upstream = {name: core.sha256_file(path) for name, path in upstream_paths.items()}
    core.require(actual_upstream == protocol["upstream_sha256"], "B0 upstream evidence drift")
    core.require(actual_upstream["feasibility_protocol"] == core.FEASIBILITY_PROTOCOL_SHA256, "main protocol drift")
    core.require(actual_upstream["generation_tasks"] == core.GENERATION_TASKS_SHA256, "task hash drift")
    core.require(actual_upstream["primary_gate"] == core.PRIMARY_GATE_SHA256, "primary gate drift")
    core.require(actual_upstream["launch_authority"] == core.LAUNCH_AUTHORITY_SHA256, "launch authority drift")
    core.require(actual_upstream["environment_manifest"] == core.ENVIRONMENT_MANIFEST_SHA256, "environment drift")

    actual_sources = {relative: core.sha256_file(root / Path(relative)) for relative in SOURCE_FILES}
    core.require(actual_sources == protocol["generator_source_sha256"], "frozen B0 generator source drift")

    authority = load_json(paths["launch_authority"])
    core.require(
        authority["status"] == "PASS_K5_FULL_RUNNER_GATE_NO_FULL_TRAINING_STARTED",
        "full-runner launch authority is not PASS",
    )
    core.require(authority["full_K5_optimizer_execution_started"] is False, "optimizer-start boundary drift")
    core.require(authority["primary_gate_sha256"] == core.PRIMARY_GATE_SHA256, "source gate link drift")
    core.require(authority["generation_tasks_sha256"] == core.GENERATION_TASKS_SHA256, "authority task drift")

    environment = load_json(paths["environment_manifest"])
    current_packages = {name: importlib.metadata.version(name) for name in PACKAGE_NAMES}
    core.require(current_packages == environment["packages"], "B0 Python package environment drift")
    core.require(platform.python_version() == environment["python"]["version"], "B0 Python version drift")
    executable = Path(sys.executable).resolve()
    core.require(executable == Path(environment["python"]["executable"]).resolve(), "B0 Python executable drift")
    core.require(core.sha256_file(executable) == environment["python"]["executable_sha256"], "Python binary drift")

    snapshot = Path(environment["frozen_inputs"]["model_snapshot"]).resolve()
    core.require(snapshot.is_dir() and snapshot.name == core.MODEL_REVISION, "model snapshot drift")
    core.require(environment["frozen_inputs"]["model_id"] == core.MODEL_ID, "model ID drift")
    core.require(environment["frozen_inputs"]["model_revision"] == core.MODEL_REVISION, "model revision drift")
    for relative, check in environment["frozen_inputs"]["critical_model_hashes"].items():
        core.require(check["actual"] == check["expected"], f"frozen model check is not PASS: {relative}")
        core.require(core.sha256_file(snapshot / relative) == check["expected"], f"model weight drift: {relative}")

    tasks = core.load_and_validate_tasks(paths["generation_tasks"])
    full_protocol = load_json(paths["feasibility_protocol"])
    generation_contract = full_protocol["evaluation"]["generation_tasks"]
    core.require(generation_contract["B0_timing"] == "generate and hash before any full K5 private optimizer update", "B0 timing drift")
    core.require(generation_contract["images_per_model"] == core.EXPECTED_IMAGE_COUNT, "main B0 count drift")
    core.require(generation_contract["batch_size"] == core.BATCH_SIZE, "main B0 batch drift")
    return {
        "root": root,
        "paths": paths,
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "environment": environment,
        "snapshot": snapshot,
        "tasks": tasks,
        "source_hashes": actual_sources,
        "upstream_hashes": actual_upstream,
    }


@contextlib.contextmanager
def exclusive_lock(staging_root: Path) -> Iterator[None]:
    import msvcrt

    staging_root.mkdir(parents=True, exist_ok=True)
    path = staging_root / "generation.lock"
    handle = path.open("a+b")
    try:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError("another B0 generation process holds the exclusive lock") from error
        yield
    finally:
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            handle.close()


def validate_progress_identity(progress: dict[str, Any], context: dict[str, Any]) -> None:
    core.require(progress["schema"] == core.PROGRESS_SCHEMA, "B0 progress schema drift")
    core.require(progress["run_id"] == core.RUN_ID and progress["model"] == "B0", "B0 progress identity")
    core.require(progress["protocol_sha256"] == context["protocol_sha256"], "B0 progress protocol drift")
    core.require(progress["generation_tasks_sha256"] == core.GENERATION_TASKS_SHA256, "B0 progress task drift")
    core.require(progress["source_gate_sha256"] == core.PRIMARY_GATE_SHA256, "B0 progress source gate drift")
    core.require(progress["generator_source_sha256"] == context["source_hashes"], "B0 progress source drift")
    core.require(progress["total_tasks"] == core.EXPECTED_IMAGE_COUNT, "B0 progress total drift")
    core.require(progress["state"] in {"IN_PROGRESS", "PRIMARY_COMPLETE", "PRIMARY_FAILED_PIXEL_SANITY"}, "B0 progress state")
    core.require(isinstance(progress["records"], dict), "B0 progress records")


def create_or_load_progress(staging_root: Path, context: dict[str, Any]) -> dict[str, Any]:
    path = staging_root / "progress.json"
    if path.exists():
        progress = load_json(path)
        validate_progress_identity(progress, context)
        return progress
    core.require(not (staging_root / "images").exists(), "untracked B0 images exist without progress")
    core.require(not (staging_root / "primary_generation_result.json").exists(), "untracked B0 primary result")
    now = utc_now()
    progress = {
        "schema": core.PROGRESS_SCHEMA,
        "state": "IN_PROGRESS",
        "run_id": core.RUN_ID,
        "model": "B0",
        "protocol_sha256": context["protocol_sha256"],
        "generation_tasks_sha256": core.GENERATION_TASKS_SHA256,
        "source_gate_sha256": core.PRIMARY_GATE_SHA256,
        "generator_source_sha256": context["source_hashes"],
        "total_tasks": core.EXPECTED_IMAGE_COUNT,
        "created_utc": now,
        "updated_utc": now,
        "records": {},
    }
    core.write_json_atomic(path, progress)
    return progress


def save_progress(path: Path, progress: dict[str, Any]) -> None:
    progress["updated_utc"] = utc_now()
    core.write_json_atomic(path, progress)


def reconcile_progress(staging_root: Path, progress: dict[str, Any], tasks: list[dict[str, Any]]) -> bool:
    records = progress["records"]
    tasks_by_id = {task["task_id"]: task for task in tasks}
    core.require(set(records).issubset(tasks_by_id), "progress contains an unknown task")
    changed = False
    for task_id, record in records.items():
        task = tasks_by_id[task_id]
        core.compare_record_to_task(record, task)
        path = staging_root / record["relative_path"]
        if record["state"] == "committed":
            core.compare_record_to_file(record, path)
        elif path.exists():
            core.compare_record_to_file(record, path)
            record["state"] = "committed"
            changed = True
    images_root = staging_root / "images"
    actual_pngs = set()
    if images_root.exists():
        actual_pngs = {path.relative_to(staging_root).as_posix() for path in images_root.rglob("*.png")}
        stale = list(images_root.rglob("*.new"))
        core.require(not stale, f"stale B0 image temporary requires audit: {stale[0]}")
    tracked_existing = {
        str(record["relative_path"])
        for record in records.values()
        if (staging_root / str(record["relative_path"])).is_file()
    }
    core.require(actual_pngs == tracked_existing, "untracked, missing, or extra B0 PNG")
    return changed


def records_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    first = dict(left)
    second = dict(right)
    first.pop("state", None)
    second.pop("state", None)
    return first == second


def finalize_inventory(staging_root: Path, progress: dict[str, Any], tasks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task in tasks:
        record = progress["records"].get(task["task_id"])
        core.require(record is not None and record["state"] == "committed", f"incomplete task: {task['task_id']}")
        core.compare_record_to_task(record, task)
        core.compare_record_to_file(record, staging_root / record["relative_path"])
        records.append(record)
    expected_paths = {core.relative_image_path(task) for task in tasks}
    actual_paths = {
        path.relative_to(staging_root).as_posix()
        for path in (staging_root / "images").rglob("*.png")
    }
    core.require(actual_paths == expected_paths, "final B0 file set mismatch")
    core.require(not list((staging_root / "images").rglob("*.new")), "stale B0 image temporary")
    return records, core.summarize_records(records)


def generate(args: argparse.Namespace, context: dict[str, Any]) -> dict[str, Any]:
    core.require(args.start_b0, "B0 generation requires --start-b0")
    core.require(
        args.acknowledge_protocol_sha256 == context["protocol_sha256"],
        "exact B0 protocol SHA-256 acknowledgement is required",
    )
    paths = context["paths"]
    staging_root = args.staging_root.resolve()
    core.require(staging_root == paths["staging_root"].resolve(), "non-frozen B0 staging path")
    core.require(not paths["full_run_root"].exists(), "full K5 run already initialized; B0 is too late")
    core.require(not paths["public_b0_root"].exists(), "public B0 result already exists")
    free_gib = core.require_free_space(staging_root, 15.0)
    gpu_start = core.nvidia_state()
    core.require_idle_gpu(gpu_start, context["environment"]["cuda_device"])

    with exclusive_lock(staging_root):
        progress_path = staging_root / "progress.json"
        result_path = staging_root / "primary_generation_result.json"
        progress = create_or_load_progress(staging_root, context)
        if reconcile_progress(staging_root, progress, context["tasks"]):
            save_progress(progress_path, progress)
        if result_path.exists():
            result = load_json(result_path)
            core.require(
                result["schema"] == core.PRIMARY_RESULT_SCHEMA
                and result["protocol_sha256"] == context["protocol_sha256"],
                "existing primary result drift",
            )
            records, summary = finalize_inventory(staging_root, progress, context["tasks"])
            core.require(result["inventory_sha256"] == core.inventory_sha256(records), "existing inventory drift")
            core.require(result["pixel_sanity"] == summary, "existing pixel summary drift")
            return result

        import torch

        core.configure_deterministic_cuda(torch)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        batches_generated = 0
        pipe = None
        try:
            pipe = core.load_base_pipeline(context["snapshot"])
            total_batches = core.EXPECTED_IMAGE_COUNT // core.BATCH_SIZE
            for batch_index, batch in enumerate(core.expected_batches(context["tasks"])):
                core.require(not paths["full_run_root"].exists(), "full K5 run appeared during B0")
                existing = [progress["records"].get(task["task_id"]) for task in batch]
                if all(record is not None and record["state"] == "committed" for record in existing):
                    continue
                arrays = core.generate_batch_arrays(pipe, batch)
                prepared: list[tuple[dict[str, Any], bytes, dict[str, Any]]] = []
                for task, array in zip(batch, arrays, strict=True):
                    body = core.encode_rgb_png(array)
                    candidate = core.record_from_png(task, body, state="prepared")
                    previous = progress["records"].get(task["task_id"])
                    if previous is not None:
                        core.require(records_match(previous, candidate), f"resumed output mismatch: {task['task_id']}")
                    progress["records"][task["task_id"]] = candidate
                    prepared.append((task, body, candidate))
                # Hashes are durably recorded before any final image name appears.
                save_progress(progress_path, progress)
                for task, body, record in prepared:
                    image_path = staging_root / record["relative_path"]
                    if image_path.exists():
                        core.compare_record_to_file(record, image_path)
                    else:
                        core.write_bytes_atomic(image_path, body)
                    progress["records"][task["task_id"]]["state"] = "committed"
                save_progress(progress_path, progress)
                batches_generated += 1
                if (batch_index + 1) % 8 == 0 or batch_index + 1 == total_batches:
                    state = core.nvidia_state()
                    core.require(state["temperature_c"] < 88, "GPU thermal abort threshold reached")
                    core.require_free_space(staging_root, 15.0)
                    committed = sum(record["state"] == "committed" for record in progress["records"].values())
                    print(f"[B0] committed {committed}/{core.EXPECTED_IMAGE_COUNT} images")
        finally:
            if pipe is not None:
                del pipe
            gc.collect()
            torch.cuda.empty_cache()

        records, summary = finalize_inventory(staging_root, progress, context["tasks"])
        elapsed = time.perf_counter() - started
        passed = summary["status"] == "PASS"
        result = {
            "schema": core.PRIMARY_RESULT_SCHEMA,
            "status": "READY_FOR_INDEPENDENT_VERIFICATION" if passed else "FAIL_PIXEL_SANITY",
            "run_id": core.RUN_ID,
            "model": "B0",
            "protocol_sha256": context["protocol_sha256"],
            "source_gate_sha256": core.PRIMARY_GATE_SHA256,
            "generation_tasks_sha256": core.GENERATION_TASKS_SHA256,
            "image_count": len(records),
            "inventory_sha256": core.inventory_sha256(records),
            "progress_sha256_before_final_state": core.sha256_file(progress_path),
            "pixel_sanity": summary,
            "elapsed_seconds_this_invocation": elapsed,
            "seconds_per_image_this_invocation": elapsed / max(1, batches_generated * core.BATCH_SIZE),
            "batches_generated_this_invocation": batches_generated,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "free_space_gib_at_start": free_gib,
            "gpu_at_start": gpu_start,
            "generated_before_private_training": True,
            "full_K5_optimizer_steps_started": 0,
            "images_release_eligible": False,
            "completed_utc": utc_now(),
        }
        core.write_json_atomic(result_path, result)
        progress["state"] = "PRIMARY_COMPLETE" if passed else "PRIMARY_FAILED_PIXEL_SANITY"
        progress["primary_result_sha256"] = core.sha256_file(result_path)
        save_progress(progress_path, progress)
        return result


def main() -> int:
    args = parse_args()
    context = validate_static_inputs(args.protocol)
    validation = {
        "status": "PASS_READ_ONLY_B0_GENERATOR_VALIDATION",
        "run_id": core.RUN_ID,
        "model": "B0",
        "protocol_sha256": context["protocol_sha256"],
        "generation_tasks_sha256": core.GENERATION_TASKS_SHA256,
        "generator_source_sha256": context["source_hashes"],
        "image_count": core.EXPECTED_IMAGE_COUNT,
        "optimizer_steps_started": False,
    }
    result = validation if args.command == "validate" else generate(args, context)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not str(result["status"]).startswith("FAIL") else 2


if __name__ == "__main__":
    raise SystemExit(main())
