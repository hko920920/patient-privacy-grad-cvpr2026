#!/usr/bin/env python3
"""Four-step K5 private-partition research runtime dry-run.

This is the first optimizer execution on records assigned to ``private_train``.
The underlying NIH corpus is public, but the partition is handled as restricted
to exercise the intended lifecycle.  The run is bounded by a pre-hashed
protocol, uses a non-cryptographic PyTorch backend seeded from ephemeral OS
entropy, writes no model checkpoint, and is never release-eligible.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import hmac
import importlib.metadata
import json
import math
import os
import platform
import secrets
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "nih-cxr14-k5-private-research-dryrun/v1"
PROTOCOL_SHA256 = "0509D745E8B192FC2CE7694EBDA0C36AB60005D704864D27A44681FD6E8432C8"
UPSTREAM_PROTOCOL_SHA256 = "F2757DC7EBC7488B6A9DD0F69227419EA16BB9A3E9BD4434514CB19AFD2B4018"
EXECUTABLE_GATE_SHA256 = "EA2041E282153A547EEE89BBF1D6533315D7C071B5B712907FB2DCC5E6BE7A2C"
K5_SHA256 = "DC49D82E497EAA6940DCF92C8F773179B8F7A838057A82AE0C26A69CA785BFC5"
PREPROCESSING_SHA256 = "0ED9E434764BDB2D542BB16AB951F856CB5B82D4F2E6B9F25F6C554F1076669D"
MODEL_ID = "Manojb/stable-diffusion-2-1-base"
MODEL_REVISION = "0094d483a120f3f33dafbd187ea4aa60d10de75c"
MODEL_VARIANT = "fp16"
LORA_INIT_SEED = 260_903
LORA_RANK = 8
LORA_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")
TRAINABLE_PARAMETERS = 1_659_904
IMAGE_SIZE = 256
MAX_TRAINING_STEPS = 4_000
RDP_ORDERS = [round(1.0 + index / 10.0, 1) for index in range(1, 100)]
RDP_ORDERS += list(range(12, 64)) + [64, 128, 256, 512]


def default_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parent.parent
    return {
        "root": root,
        "dryrun_protocol": root / "dp_training" / "private_research_dryrun_protocol.json",
        "upstream_protocol": root / "_reports" / "nih_cxr14_dp_attack_protocol_v1_001" / "protocol.json",
        "executable_gate": root
        / "_reports"
        / "nih_cxr14_dp_trainer_executable_gate_v1_001"
        / "independent_verification.json",
        "manifest": root
        / "_data"
        / "derived"
        / "nih_cxr14_pa_target_enriched_v1"
        / "k5_private.csv",
        "image_root": root / "_data" / "raw" / "nih_cxr14_pa_k10_plus_census_v1" / "images",
        "preprocessing": root
        / "_reports"
        / "nih_cxr14_sd21_ppmark_interface_v1_001"
        / "preprocessing_contract.json",
        "output": root / "_reports" / "nih_cxr14_k5_private_research_dryrun_v1_001",
    }


def parse_args() -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dryrun-protocol", type=Path, default=paths["dryrun_protocol"])
    parser.add_argument("--upstream-protocol", type=Path, default=paths["upstream_protocol"])
    parser.add_argument("--executable-gate", type=Path, default=paths["executable_gate"])
    parser.add_argument("--manifest", type=Path, default=paths["manifest"])
    parser.add_argument("--image-root", type=Path, default=paths["image_root"])
    parser.add_argument("--preprocessing", type=Path, default=paths["preprocessing"])
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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def derive_seed(master_secret: bytes, label: str) -> int:
    require(len(master_secret) == 32, "master secret must contain 256 bits")
    digest = hmac.new(master_secret, label.encode("utf-8"), hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def named_tensor_digest(named_tensors: list[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        value = tensor.detach().to(device="cpu", dtype=tensor.dtype).contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest().upper()


def flatten_named_tensors(named_tensors: list[tuple[str, Any]]) -> Any:
    import torch

    return torch.cat([tensor.detach().float().reshape(-1) for _, tensor in named_tensors], dim=0)


def assign_flat_gradient(trainable: list[tuple[str, Any]], flat: Any) -> None:
    offset = 0
    for _, parameter in trainable:
        count = parameter.numel()
        parameter.grad = flat[offset : offset + count].view_as(parameter).detach().clone()
        offset += count
    require(offset == flat.numel(), "flat gradient length mismatch")


def build_runtime_schedules(
    records: list[Any],
    protocol: dict[str, Any],
    master_secret: bytes,
    scheduler_timesteps: int,
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]], dict[str, int]]:
    """Sample all four steps before a model or private loss is loaded."""

    import torch

    from dp_training.mechanism import poisson_select

    by_image = {record.image_id: record for record in records}
    by_patient: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        by_patient[record.patient_id].append(record)
    for patient_records in by_patient.values():
        patient_records.sort(key=lambda record: (record.cap_rank, record.image_id))
    image_ids = sorted(by_image)
    patient_ids = sorted(by_patient)

    runtime = protocol["runtime"]
    limits = protocol["resource_fail_closed"]
    steps = int(runtime["steps_per_arm"])
    image_q = float(protocol["mechanisms"]["M1-I8"]["poisson_sample_rate"])
    patient_q = float(protocol["mechanisms"]["M2-P8"]["poisson_sample_rate"])
    image_selection_generator = torch.Generator(device="cpu").manual_seed(
        derive_seed(master_secret, "m1-shared-poisson-selection")
    )
    image_diffusion_generator = torch.Generator(device="cpu").manual_seed(
        derive_seed(master_secret, "m1-shared-diffusion-draws")
    )
    patient_selection_generator = torch.Generator(device="cpu").manual_seed(
        derive_seed(master_secret, "m2-patient-poisson-selection")
    )
    patient_inner_generator = torch.Generator(device="cpu").manual_seed(
        derive_seed(master_secret, "m2-within-patient-selection")
    )
    patient_diffusion_generator = torch.Generator(device="cpu").manual_seed(
        derive_seed(master_secret, "m2-diffusion-draws")
    )

    def draw_diffusion(generator: Any) -> tuple[int, int]:
        timestep = int(torch.randint(0, scheduler_timesteps, (1,), generator=generator).item())
        seed = int(torch.randint(0, 2**63 - 1, (1,), generator=generator, dtype=torch.int64).item())
        return timestep, seed

    image_schedule: list[list[dict[str, Any]]] = []
    for step in range(1, steps + 1):
        selected = poisson_select(image_ids, image_q, generator=image_selection_generator)
        require(
            len(selected) <= int(limits["maximum_realized_image_units_per_m1_step"]),
            f"M1 step {step} exceeded the frozen resource limit; do not resample",
        )
        units = []
        for image_id in selected:
            timestep, noise_seed = draw_diffusion(image_diffusion_generator)
            units.append(
                {
                    "unit_type": "image",
                    "unit_id": image_id,
                    "patient_id": by_image[image_id].patient_id,
                    "records": [by_image[image_id]],
                    "draws": [{"timestep": timestep, "noise_seed": noise_seed}],
                }
            )
        image_schedule.append(units)

    patient_schedule: list[list[dict[str, Any]]] = []
    for step in range(1, steps + 1):
        selected = poisson_select(patient_ids, patient_q, generator=patient_selection_generator)
        require(
            len(selected) <= int(limits["maximum_realized_patient_units_per_m2_step"]),
            f"M2 step {step} exceeded the frozen patient resource limit; do not resample",
        )
        units = []
        raw_images = 0
        for patient_id in selected:
            available = by_patient[patient_id]
            take = min(4, len(available))
            order = torch.randperm(len(available), generator=patient_inner_generator).tolist()
            chosen = [available[index] for index in order[:take]]
            draws = []
            for _ in chosen:
                timestep, noise_seed = draw_diffusion(patient_diffusion_generator)
                draws.append({"timestep": timestep, "noise_seed": noise_seed})
            raw_images += len(chosen)
            units.append(
                {
                    "unit_type": "patient",
                    "unit_id": patient_id,
                    "patient_id": patient_id,
                    "records": chosen,
                    "draws": draws,
                }
            )
        require(
            raw_images <= int(limits["maximum_realized_raw_images_per_m2_step"]),
            f"M2 step {step} exceeded the frozen raw-image resource limit; do not resample",
        )
        patient_schedule.append(units)

    schedule_counts = {
        "m1_units": sum(len(step) for step in image_schedule),
        "m2_units": sum(len(step) for step in patient_schedule),
        "m2_images": sum(len(unit["records"]) for step in patient_schedule for unit in step),
    }
    return image_schedule, patient_schedule, schedule_counts


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

    from dp_training.mechanism import tensor_sha256

    require(len(unit["records"]) == len(unit["draws"]), "unit record/draw mismatch")
    require(len(unit["records"]) >= 1, "privacy unit is empty")
    if unit["unit_type"] == "image":
        require(len(unit["records"]) == 1, "image unit has more than one record")
    else:
        require(
            unit["unit_type"] == "patient" and len(unit["records"]) <= 4,
            "patient unit boundary mismatch",
        )
        require(
            {record.patient_id for record in unit["records"]} == {unit["patient_id"]},
            "patient unit crosses patients",
        )

    latents = []
    hidden = []
    noises = []
    timesteps = []
    diffusion_noise_hashes = []
    for record, draw in zip(unit["records"], unit["draws"]):
        latent = latent_by_image[record.image_id].to(device="cuda", dtype=torch.float16)
        prompt = model_input.prompt_for_record(record)
        text = hidden_by_prompt[prompt].to(device="cuda", dtype=torch.float16)
        generator = torch.Generator(device="cuda").manual_seed(int(draw["noise_seed"]))
        noise = torch.randn(latent.shape, generator=generator, device="cuda", dtype=torch.float16)
        latents.append(latent)
        hidden.append(text)
        noises.append(noise)
        timesteps.append(int(draw["timestep"]))
        diffusion_noise_hashes.append(tensor_sha256(noise))

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
        raise RuntimeError(f"unsupported prediction type: {prediction_type}")

    for _, parameter in trainable:
        parameter.grad = None
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        prediction = unet(noisy, timestep_batch, hidden_batch).sample
        per_image = functional.mse_loss(
            prediction.float(), target.float(), reduction="none"
        ).flatten(1).mean(1)
        # For M2 this is the single protected patient loss. The one backward
        # call below happens before the outer patient clip in mechanism.py.
        loss = per_image.mean()
    loss.backward()

    gradients = []
    norm_squared = 0.0
    for name, parameter in trainable:
        require(parameter.grad is not None, f"missing gradient: {name}")
        gradient = parameter.grad.detach().float()
        require(bool(torch.isfinite(gradient).all()), "non-finite unit gradient")
        gradients.append((name, gradient))
        norm_squared += float(torch.sum(gradient.double() ** 2).cpu())
    flat = flatten_named_tensors(gradients)
    return flat, {
        "unit_type": unit["unit_type"],
        "unit_id": unit["unit_id"],
        "patient_id": unit["patient_id"],
        "image_ids": [record.image_id for record in unit["records"]],
        "image_count": len(unit["records"]),
        "timesteps": timesteps,
        "diffusion_noise_sha256": diffusion_noise_hashes,
        "loss": float(loss.detach().cpu()),
        "gradient_l2_norm": math.sqrt(norm_squared),
        "gradient_sha256": named_tensor_digest(gradients),
        "mean_before_outer_clip": unit["unit_type"] == "patient",
    }


def optimizer_is_finite(optimizer: Any) -> bool:
    import torch

    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
                return False
    return True


def run_arm(
    *,
    arm: str,
    schedule: list[list[dict[str, Any]]],
    config: Any,
    dp_noise_seed: int,
    snapshot: Path,
    latent_by_image: dict[str, Any],
    hidden_by_prompt: dict[str, Any],
    model_input: Any,
    scheduler: Any,
    optimizer_contract: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    import torch
    from diffusers import UNet2DConditionModel
    from peft import LoraConfig
    from peft.utils.other import cast_mixed_precision_params

    from dp_training.mechanism import (
        aggregate_noised_update,
        append_public_trace,
        make_public_scheduled_event,
        tensor_sha256,
    )

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
    require(sum(parameter.numel() for _, parameter in trainable) == TRAINABLE_PARAMETERS, "trainable count")
    require({parameter.dtype for _, parameter in trainable} == {torch.float32}, "LoRA dtype")
    initial_digest = named_tensor_digest(trainable)
    initial_vector = flatten_named_tensors(trainable).detach().cpu()
    previous_vector = initial_vector.clone()

    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=float(optimizer_contract["learning_rate"]),
        betas=tuple(float(value) for value in optimizer_contract["betas"]),
        eps=float(optimizer_contract["epsilon"]),
        weight_decay=float(optimizer_contract["weight_decay"]),
    )
    dp_generator = torch.Generator(device="cuda").manual_seed(dp_noise_seed)
    trace: list[dict[str, Any]] = []
    step_diagnostics = []
    started = time.perf_counter()

    for step_number, units in enumerate(schedule, start=1):
        step_started = time.perf_counter()
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
            summary["clip_factor"] = min(
                1.0,
                float(config.clip_norm)
                / max(float(summary["gradient_l2_norm"]), float.fromhex("0x1.0p-1022")),
            )
            summary["was_clipped"] = summary["gradient_l2_norm"] > config.clip_norm
            unit_vectors.append(vector)
            unit_summaries.append(summary)

        template = torch.zeros(TRAINABLE_PARAMETERS, device="cuda", dtype=torch.float32)
        noised_gradient, aggregation = aggregate_noised_update(
            unit_vectors,
            template=template,
            config=config,
            generator=dp_generator,
        )
        require(aggregation["realized_unit_count"] == len(units), "aggregation unit count")
        require(
            aggregation["clipped_unit_count"] == sum(item["was_clipped"] for item in unit_summaries),
            "aggregation clipping count",
        )
        assign_flat_gradient(trainable, noised_gradient)
        assigned = flatten_named_tensors([(name, parameter.grad) for name, parameter in trainable])
        require(tensor_sha256(assigned) == tensor_sha256(noised_gradient), "assigned gradient drift")
        optimizer.step()
        require(optimizer_is_finite(optimizer), "non-finite optimizer state")
        current_vector = flatten_named_tensors(trainable).detach().cpu()
        require(bool(torch.isfinite(current_vector).all()), "non-finite adapter")
        step_delta = current_vector - previous_vector
        require(float(torch.linalg.vector_norm(step_delta.double())) > 0.0, "zero optimizer step")
        optimizer.zero_grad(set_to_none=True)

        append_public_trace(
            trace,
            make_public_scheduled_event(config, first_step=step_number, event_count=1),
        )
        step_diagnostics.append(
            {
                "step": step_number,
                "selected_unit_count": len(units),
                "selected_raw_image_count": sum(len(unit["records"]) for unit in units),
                "empty_sample": len(units) == 0,
                "units": unit_summaries,
                "aggregation": aggregation,
                "noised_gradient_sha256": tensor_sha256(noised_gradient),
                "adapter_sha256_after_step": named_tensor_digest(trainable),
                "adapter_step_delta_l2_norm": float(torch.linalg.vector_norm(step_delta.double())),
                "optimizer_state_finite": True,
                "elapsed_seconds": time.perf_counter() - step_started,
            }
        )
        previous_vector = current_vector
        del assigned, step_delta, noised_gradient, template, unit_vectors
        gc.collect()
        torch.cuda.empty_cache()

    elapsed = time.perf_counter() - started
    final_vector = previous_vector
    total_delta = final_vector - initial_vector
    final_digest = named_tensor_digest(trainable)
    changed = int(torch.count_nonzero(total_delta).item())
    require(initial_digest != final_digest, "arm adapter digest unchanged")
    require(changed > 0, "arm adapter values unchanged")
    public_summary = {
        "arm": arm,
        "status": "PASS_FOUR_STEP_RUNTIME",
        "completed_steps": len(schedule),
        "adapter_initial_sha256": initial_digest,
        "adapter_final_sha256": final_digest,
        "adapter_total_delta_l2_norm": float(torch.linalg.vector_norm(total_delta.double())),
        "changed_trainable_scalars": changed,
        "parameters_and_optimizer_finite": True,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "checkpoint_written": False,
    }

    del total_delta, final_vector, previous_vector, initial_vector
    del optimizer, trainable, unet
    gc.collect()
    torch.cuda.empty_cache()
    return public_summary, step_diagnostics, trace


def opacus_epsilon(sigma: float, q: float, steps: int, delta: float) -> tuple[float, float]:
    from opacus.accountants.analysis import rdp

    values = rdp.compute_rdp(q=q, noise_multiplier=sigma, steps=steps, orders=RDP_ORDERS)
    epsilon, order = rdp.get_privacy_spent(orders=RDP_ORDERS, rdp=values, delta=delta)
    return float(epsilon), float(order)


def google_epsilon(sigma: float, q: float, steps: int, delta: float) -> tuple[float, float]:
    from dp_accounting import dp_event
    from dp_accounting.privacy_accountant import NeighboringRelation
    from dp_accounting.rdp import RdpAccountant

    accountant = RdpAccountant(
        orders=RDP_ORDERS,
        neighboring_relation=NeighboringRelation.ADD_OR_REMOVE_ONE,
    )
    accountant.compose(
        dp_event.PoissonSampledDpEvent(q, dp_event.GaussianDpEvent(sigma)),
        count=steps,
    )
    epsilon, order = accountant.get_epsilon_and_optimal_order(delta)
    return float(epsilon), float(order)


def validate_frozen_inputs(
    dryrun: dict[str, Any], upstream: dict[str, Any], executable_gate: dict[str, Any]
) -> None:
    require(dryrun["status"] == "FROZEN_BEFORE_PRIVATE_PARTITION_OPTIMIZER_EXECUTION", "dry-run protocol status")
    require(executable_gate["status"] == "PASS_EXECUTABLE_DP_TRAINER_GATE_PRIVATE_PILOT_STILL_BLOCKED", "executable gate status")
    require(dryrun["runtime"]["steps_per_arm"] == 4, "dry-run step count")
    require(dryrun["runtime"]["arms"] == ["M1-I8", "M1-G8", "M2-P8"], "dry-run arm order")
    entries = upstream["accounting"]["entries"]
    for arm, spec in dryrun["mechanisms"].items():
        if arm.startswith("M1"):
            matches = [entry for entry in entries if entry["arm"] == arm and entry["cap"] == 5]
        else:
            matches = [entry for entry in entries if entry["arm"] == arm and entry["cap"] == 10]
        require(len(matches) == 1, f"upstream accounting entry: {arm}")
        entry = matches[0]
        require(spec["privacy_unit"] == entry["accounting_unit"], f"unit drift: {arm}")
        require(spec["adjacency"] == entry["adjacency"], f"adjacency drift: {arm}")
        require(spec["population"] == entry["population"], f"population drift: {arm}")
        require(spec["expected_batch"] == entry["expected_batch"], f"batch drift: {arm}")
        require(spec["poisson_sample_rate"] == entry["poisson_sample_rate"], f"q drift: {arm}")
        require(spec["noise_multiplier"] == entry["noise_multiplier"], f"sigma drift: {arm}")
        require(spec["fixed_denominator"] == entry["expected_batch"], f"denominator drift: {arm}")
        require(spec["accounting_delta"] == entry["target_delta"], f"delta drift: {arm}")


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    paths = {
        "dryrun": args.dryrun_protocol.resolve(),
        "upstream": args.upstream_protocol.resolve(),
        "gate": args.executable_gate.resolve(),
        "manifest": args.manifest.resolve(),
        "image_root": args.image_root.resolve(),
        "preprocessing": args.preprocessing.resolve(),
        "output": args.output_dir.resolve(),
    }
    require(sha256_file(paths["dryrun"]) == PROTOCOL_SHA256, "dry-run protocol hash mismatch")
    require(sha256_file(paths["upstream"]) == UPSTREAM_PROTOCOL_SHA256, "upstream protocol hash mismatch")
    require(sha256_file(paths["gate"]) == EXECUTABLE_GATE_SHA256, "executable gate hash mismatch")
    require(sha256_file(paths["manifest"]) == K5_SHA256, "K5 manifest hash mismatch")
    require(sha256_file(paths["preprocessing"]) == PREPROCESSING_SHA256, "preprocessing hash mismatch")
    require(not paths["output"].exists() or not any(paths["output"].iterdir()), "output directory is not empty")

    dryrun = json.loads(paths["dryrun"].read_text(encoding="utf-8"))
    upstream = json.loads(paths["upstream"].read_text(encoding="utf-8"))
    executable_gate = json.loads(paths["gate"].read_text(encoding="utf-8"))
    validate_frozen_inputs(dryrun, upstream, executable_gate)

    from dp_protocol.calibrate_xray_public_clip_norms import (
        disable_broken_optional_onnx,
        prepare_latents_and_text,
        verify_snapshot,
    )

    disable_broken_optional_onnx()
    import numpy as np
    import torch
    from diffusers import DDPMScheduler
    from huggingface_hub import snapshot_download

    from dp_training.mechanism import MechanismConfig, verify_public_trace

    require(torch.cuda.is_available(), "CUDA is required")
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "data_pipeline"))
    import nih_cxr14_model_input as model_input

    records = model_input.read_manifest(paths["manifest"], partitions={"private_train"})
    require(len(records) == dryrun["data"]["expected_images"], "K5 private image count")
    require(len({record.patient_id for record in records}) == dryrun["data"]["expected_patients"], "K5 private patient count")
    require(all(1 <= record.cap_rank <= 5 for record in records), "K5 cap violation")

    snapshot = Path(
        snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    ).resolve()
    require(snapshot.name == MODEL_REVISION, "model revision mismatch")
    snapshot_hashes = verify_snapshot(snapshot)
    scheduler = DDPMScheduler.from_pretrained(snapshot / "scheduler", local_files_only=True)

    master_secret = secrets.token_bytes(32)
    session_commitment = sha256_bytes(b"nih-cxr14-k5-dryrun-session-v1\0" + master_secret)
    image_schedule, patient_schedule, schedule_counts = build_runtime_schedules(
        records,
        dryrun,
        master_secret,
        int(scheduler.config.num_train_timesteps),
    )
    dp_seeds = {
        arm: derive_seed(master_secret, f"{arm}-dp-gaussian-stream")
        for arm in dryrun["runtime"]["arm_order"]
    }
    del master_secret

    selected_units = [unit for step in image_schedule for unit in step]
    selected_units += [unit for step in patient_schedule for unit in step]
    require(selected_units, "all schedules are empty")
    latent_by_image, hidden_by_prompt, input_summary = prepare_latents_and_text(
        snapshot,
        selected_units,
        paths["image_root"],
        model_input,
    )

    configs = {}
    for arm, spec in dryrun["mechanisms"].items():
        configs[arm] = MechanismConfig(
            arm=arm,
            privacy_unit=spec["privacy_unit"],
            population=int(spec["population"]),
            expected_batch=int(spec["expected_batch"]),
            poisson_sample_rate=float(spec["poisson_sample_rate"]),
            clip_norm=float(spec["clip_norm"]),
            noise_multiplier=float(spec["noise_multiplier"]),
            fixed_denominator=float(spec["fixed_denominator"]),
            max_steps=MAX_TRAINING_STEPS,
            rng_security_mode="RESEARCH_ONLY_NONCRYPTOGRAPHIC",
            rng_backend="torch_default_generators_ephemerally_seeded_from_os_entropy",
        )

    arm_public = {}
    arm_private = {}
    traces = {}
    full_started = time.perf_counter()
    for arm in dryrun["runtime"]["arm_order"]:
        schedule = image_schedule if arm.startswith("M1") else patient_schedule
        public_summary, private_steps, trace = run_arm(
            arm=arm,
            schedule=schedule,
            config=configs[arm],
            dp_noise_seed=dp_seeds[arm],
            snapshot=snapshot,
            latent_by_image=latent_by_image,
            hidden_by_prompt=hidden_by_prompt,
            model_input=model_input,
            scheduler=scheduler,
            optimizer_contract=dryrun["optimizer"],
        )
        trace_check = verify_public_trace(trace)
        require(trace_check["total_events"] == dryrun["runtime"]["steps_per_arm"], f"trace events: {arm}")
        public_summary["public_trace_summary"] = trace_check
        arm_public[arm] = public_summary
        arm_private[arm] = private_steps
        traces[arm] = trace

    initial_digests = {value["adapter_initial_sha256"] for value in arm_public.values()}
    require(len(initial_digests) == 1, "arms did not begin from identical LoRA parameters")

    # Controlled M1 arms must use exactly the same data and diffusion draws.
    for step_i8, step_g8 in zip(arm_private["M1-I8"], arm_private["M1-G8"]):
        require(step_i8["selected_unit_count"] == step_g8["selected_unit_count"], "M1 count mismatch")
        for unit_i8, unit_g8 in zip(step_i8["units"], step_g8["units"]):
            for key in ("unit_id", "patient_id", "image_ids", "timesteps", "diffusion_noise_sha256"):
                require(unit_i8[key] == unit_g8[key], f"M1 shared schedule/draw mismatch: {key}")
    if arm_private["M1-I8"][0]["units"]:
        require(
            [unit["gradient_sha256"] for unit in arm_private["M1-I8"][0]["units"]]
            == [unit["gradient_sha256"] for unit in arm_private["M1-G8"][0]["units"]],
            "M1 first-step gradients differ before the first arm-specific update",
        )

    accounting = {}
    steps = int(dryrun["runtime"]["steps_per_arm"])
    for arm, spec in dryrun["mechanisms"].items():
        opacus = opacus_epsilon(
            float(spec["noise_multiplier"]),
            float(spec["poisson_sample_rate"]),
            steps,
            float(spec["accounting_delta"]),
        )
        google = google_epsilon(
            float(spec["noise_multiplier"]),
            float(spec["poisson_sample_rate"]),
            steps,
            float(spec["accounting_delta"]),
        )
        accounting[arm] = {
            "events": steps,
            "delta": float(spec["accounting_delta"]),
            "opacus_epsilon": opacus[0],
            "opacus_optimal_order": opacus[1],
            "google_epsilon": google[0],
            "google_optimal_order": google[1],
            "conservative_epsilon": max(opacus[0], google[0]),
            "accounting_only_not_release_evidence": True,
        }

    restricted = {
        "schema": "nih-cxr14-k5-private-research-dryrun-restricted-diagnostics/v1",
        "classification": "LOCAL_RESTRICTED_DO_NOT_INCLUDE_IN_RELEASE_PACKAGE",
        "source_is_public_proxy": True,
        "notice": "identifiers and realized sample counts are retained only to audit the research loader; no randomness seed is serialized",
        "seed_values_serialized": False,
        "input_summary": input_summary,
        "schedule_counts": schedule_counts,
        "arms": arm_private,
    }
    public_traces = {
        "schema": "nih-cxr14-k5-private-research-public-event-traces/v1",
        "session_commitment": session_commitment,
        "traces": traces,
    }

    paths["output"].mkdir(parents=True, exist_ok=True)
    restricted_path = paths["output"] / "restricted_runtime_diagnostics.json"
    trace_path = paths["output"] / "public_event_traces.json"
    write_json(restricted_path, restricted)
    write_json(trace_path, public_traces)
    checkpoint_suffixes = {".pt", ".pth", ".ckpt", ".safetensors", ".bin"}
    checkpoint_files = [
        str(path.relative_to(paths["output"]))
        for path in paths["output"].rglob("*")
        if path.is_file() and path.suffix.lower() in checkpoint_suffixes
    ]
    require(not checkpoint_files, "checkpoint-like output found")

    public_report = {
        "schema": SCHEMA,
        "status": "PASS_K5_PRIVATE_RESEARCH_RUNTIME_DRYRUN",
        "scope": "FOUR_STEPS_PER_DP_ARM_RUNTIME_ONLY_NOT_FEASIBILITY_NOT_UTILITY_NOT_PRIVACY_RELEASE",
        "frozen_inputs": {
            "dryrun_protocol_sha256": PROTOCOL_SHA256,
            "upstream_protocol_sha256": UPSTREAM_PROTOCOL_SHA256,
            "executable_gate_sha256": EXECUTABLE_GATE_SHA256,
            "k5_manifest_sha256": K5_SHA256,
            "preprocessing_contract_sha256": PREPROCESSING_SHA256,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "critical_model_hashes": snapshot_hashes,
            "profile": "P256",
        },
        "data_boundary": {
            "partition": "private_train",
            "cap": 5,
            "source": "public NIH ChestXray14 proxy handled as restricted",
            "identifiers_or_realized_counts_in_public_report": False,
            "selected_inputs_cached_on_disk": False,
        },
        "runtime": {
            "arms": dryrun["runtime"]["arm_order"],
            "steps_per_arm": steps,
            "M1_shared_schedule_and_diffusion_draws": True,
            "identical_initial_adapter_across_arms": True,
            "adapter_initial_sha256": next(iter(initial_digests)),
            "fresh_model_per_arm": True,
            "model_discarded_after_each_arm": True,
            "checkpoint_written": False,
            "full_elapsed_seconds": time.perf_counter() - full_started,
        },
        "mechanism_configs": {arm: config.public_dict() for arm, config in configs.items()},
        "optimizer": dryrun["optimizer"],
        "arms": arm_public,
        "accounting_at_completed_events": accounting,
        "randomness": {
            "classification": "RESEARCH_ONLY_NONCRYPTOGRAPHIC",
            "session_commitment": session_commitment,
            "master_entropy_source": "operating-system entropy pool",
            "torch_backend": "default CPU/CUDA generators",
            "private_seed_values_serialized": False,
            "release_eligible": False,
            "formal_dp_claim_allowed": False,
        },
        "artifacts": {
            "public_event_traces": trace_path.name,
            "public_event_traces_sha256": sha256_file(trace_path),
            "restricted_runtime_diagnostics": restricted_path.name,
            "restricted_runtime_diagnostics_sha256": sha256_file(restricted_path),
            "checkpoint_like_files": checkpoint_files,
        },
        "claim_limits": dryrun["prohibited_interpretations"],
        "next": dryrun["next_if_pass"],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "packages": {
                "diffusers": package_version("diffusers"),
                "peft": package_version("peft"),
                "opacus": package_version("opacus"),
                "dp-accounting": package_version("dp-accounting"),
            },
        },
    }
    public_report_path = paths["output"] / "public_report.json"
    write_json(public_report_path, public_report)
    print(
        json.dumps(
            {
                "status": public_report["status"],
                "report": str(public_report_path),
                "steps_per_arm": steps,
                "elapsed_seconds": public_report["runtime"]["full_elapsed_seconds"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
