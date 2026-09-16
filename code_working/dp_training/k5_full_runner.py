#!/usr/bin/env python3
"""Shared implementation for the frozen NIH-CXR14 K5 training runner.

The module keeps the scientific mechanism separate from the command-line
lifecycle.  It is used by both the bounded real-SD2.1 resume preflight and the
long-run launcher.  Importing it never creates a run directory and never starts
an optimizer step.
"""

from __future__ import annotations

import gc
import hashlib
import hmac
import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


PROTOCOL_SCHEMA = "nih-cxr14-k5-feasibility-execution-protocol/v1"
PROTOCOL_SHA256 = "2234B3BA8701565B768A11AB5196B2F696788900D07254833997D7823489E9DA"
RESUME_SCHEMA = "nih-cxr14-k5-full-training-resume/v1"
MATRIX_SCHEMA = "nih-cxr14-k5-full-training-matrix-root/v1"
RUN_ID = "nih_cxr14_k5_feasibility_v1_001"
ARMS = ("M0", "M1-I8", "M1-G8", "M2-P8")
DP_ARMS = ("M1-I8", "M1-G8", "M2-P8")
MODEL_ID = "Manojb/stable-diffusion-2-1-base"
MODEL_REVISION = "0094d483a120f3f33dafbd187ea4aa60d10de75c"
MODEL_VARIANT = "fp16"
K5_SHA256 = "DC49D82E497EAA6940DCF92C8F773179B8F7A838057A82AE0C26A69CA785BFC5"
PREPROCESSING_SHA256 = "0ED9E434764BDB2D542BB16AB951F856CB5B82D4F2E6B9F25F6C554F1076669D"
LORA_INIT_SEED = 260_903
LORA_RANK = 8
LORA_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")
TRAINABLE_PARAMETERS = 1_659_904
IMAGE_SIZE = 256
FULL_STEPS = 4_000
RESUME_INTERVAL = 250
ADAPTER_STEPS = (1_000, 2_000, 4_000)
TRACE_GENESIS = "0" * 64


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_paths() -> dict[str, Path]:
    root = project_root()
    return {
        "root": root,
        "protocol": root
        / "_reports"
        / "nih_cxr14_k5_feasibility_protocol_v1_001"
        / "protocol.json",
        "manifest": root
        / "_data"
        / "derived"
        / "nih_cxr14_pa_target_enriched_v1"
        / "k5_private.csv",
        "image_root": root
        / "_data"
        / "raw"
        / "nih_cxr14_pa_k10_plus_census_v1"
        / "images",
        "preprocessing": root
        / "_reports"
        / "nih_cxr14_sd21_ppmark_interface_v1_001"
        / "preprocessing_contract.json",
        "run_root": root / "_restricted_runs" / RUN_ID,
        "report_root": root / "_reports" / RUN_ID,
        "launch_authority": root
        / "_reports"
        / "nih_cxr14_k5_full_runner_gate_v1_001"
        / "independent_launch_authority.json",
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
    require(not temporary.exists(), f"stale atomic JSON temporary requires audit: {temporary.name}")
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


def derive_seed(experiment_root: bytes, label: str) -> int:
    require(len(experiment_root) == 32, "experiment root must contain 256 bits")
    digest = hmac.new(experiment_root, label.encode("utf-8"), hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def exact_equal(left: Any, right: Any) -> bool:
    import torch

    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return (
            left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            exact_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            exact_equal(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


def tree_to_cpu(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: tree_to_cpu(child) for key, child in value.items()}
    if isinstance(value, list):
        return [tree_to_cpu(child) for child in value]
    if isinstance(value, tuple):
        return tuple(tree_to_cpu(child) for child in value)
    return value


def load_frozen_protocol(path: Path | None = None) -> dict[str, Any]:
    path = (path or default_paths()["protocol"]).resolve()
    require(path.is_file(), "frozen K5 protocol is missing")
    require(sha256_file(path) == PROTOCOL_SHA256, "frozen K5 protocol hash mismatch")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    require(protocol["schema"] == PROTOCOL_SCHEMA, "K5 protocol schema mismatch")
    require(protocol["status"] == "FROZEN_BEFORE_FULL_K5_OPTIMIZER_EXECUTION", "K5 protocol status")
    require(protocol["run_id"] == RUN_ID, "K5 run ID mismatch")
    require(tuple(protocol["matrix"]["arm_order"]) == ARMS, "K5 arm order mismatch")
    require(int(protocol["matrix"]["steps_per_arm"]) == FULL_STEPS, "K5 step count mismatch")
    require(int(protocol["data_and_model"]["images"]) == 18_393, "K5 image count")
    require(int(protocol["data_and_model"]["patients"]) == 8_476, "K5 patient count")
    require(protocol["data_and_model"]["model_id"] == MODEL_ID, "model ID drift")
    require(protocol["data_and_model"]["model_revision"] == MODEL_REVISION, "model revision drift")
    return protocol


def configure_deterministic_cuda() -> None:
    import torch

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    require(torch.cuda.is_available(), "CUDA is required")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def resolve_and_verify_snapshot(protocol: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    from huggingface_hub import snapshot_download

    from dp_protocol.calibrate_xray_public_clip_norms import (
        disable_broken_optional_onnx,
        verify_snapshot,
    )

    disable_broken_optional_onnx()
    snapshot = Path(
        snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    ).resolve()
    require(snapshot.name == MODEL_REVISION, "local model revision mismatch")
    verified = verify_snapshot(snapshot)
    expected = protocol["data_and_model"]["critical_model_hashes"]
    require(verified["unet/diffusion_pytorch_model.fp16.safetensors"]["actual"] == expected["unet"], "UNet hash drift")
    require(verified["vae/diffusion_pytorch_model.fp16.safetensors"]["actual"] == expected["vae"], "VAE hash drift")
    require(verified["text_encoder/model.fp16.safetensors"]["actual"] == expected["text_encoder"], "text encoder hash drift")
    return snapshot, verified


def load_model_input_module() -> Any:
    import importlib.util
    import sys

    path = project_root() / "data_pipeline" / "nih_cxr14_model_input.py"
    spec = importlib.util.spec_from_file_location("nih_cxr14_model_input_k5_full", path)
    require(spec is not None and spec.loader is not None, "model-input module load failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_full_records(records: Sequence[Any]) -> None:
    require(len(records) == 18_393, "K5 private image population mismatch")
    require(len({record.image_id for record in records}) == 18_393, "duplicate K5 image ID")
    require(len({record.patient_id for record in records}) == 8_476, "K5 patient population mismatch")
    require(all(1 <= int(record.cap_rank) <= 5 for record in records), "K5 cap violation")
    require(all(record.partition == "private_train" for record in records), "private partition drift")


def mechanism_config(
    protocol: dict[str, Any],
    arm: str,
    *,
    max_steps: int = FULL_STEPS,
    population_override: int | None = None,
    expected_batch_override: int | None = None,
) -> Any:
    from dp_training.mechanism import MechanismConfig

    require(arm in DP_ARMS, "M0 has no DP mechanism")
    spec = protocol["DP_arms"][arm]
    population = int(population_override if population_override is not None else spec["population"])
    expected_batch = int(
        expected_batch_override if expected_batch_override is not None else spec["expected_batch"]
    )
    require(0 < expected_batch <= population, "preflight/full expected batch is invalid")
    return MechanismConfig(
        arm=arm,
        privacy_unit=str(spec["privacy_unit"]),
        population=population,
        expected_batch=expected_batch,
        poisson_sample_rate=expected_batch / population,
        clip_norm=float(spec["C"]),
        noise_multiplier=float(spec["sigma"]),
        fixed_denominator=float(expected_batch),
        max_steps=max_steps,
        rng_security_mode="RESEARCH_ONLY_NONCRYPTOGRAPHIC",
        rng_backend="torch_default_generators_seeded_from_encrypted_os_root",
    )


def _draw_diffusion(generator: Any, scheduler_timesteps: int) -> dict[str, int]:
    import torch

    timestep = int(
        torch.randint(0, scheduler_timesteps, (1,), generator=generator, dtype=torch.int64).item()
    )
    noise_seed = int(
        torch.randint(0, 2**63 - 1, (1,), generator=generator, dtype=torch.int64).item()
    )
    return {"timestep": timestep, "noise_seed": noise_seed}


def _schedule_secret_view(schedule: Sequence[Sequence[dict[str, Any]]]) -> list[Any]:
    return [
        [
            {
                "unit_type": unit["unit_type"],
                "unit_id": unit["unit_id"],
                "patient_id": unit["patient_id"],
                "image_ids": [record.image_id for record in unit["records"]],
                "draws": unit["draws"],
            }
            for unit in step
        ]
        for step in schedule
    ]


def schedule_commitment(
    experiment_root: bytes,
    arm: str,
    schedule: Sequence[Sequence[dict[str, Any]]],
) -> str:
    domain = "M1-shared" if arm.startswith("M1") else arm
    body = canonical_json({"domain": domain, "schedule": _schedule_secret_view(schedule)})
    return hmac.new(
        experiment_root,
        b"nih-cxr14-k5-schedule-commitment-v1\0" + body,
        hashlib.sha256,
    ).hexdigest().upper()


def build_arm_schedule(
    records: Sequence[Any],
    protocol: dict[str, Any],
    arm: str,
    experiment_root: bytes,
    scheduler_timesteps: int,
    *,
    steps: int = FULL_STEPS,
    fixture: bool = False,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any], Any | None]:
    """Precompute a secret in-memory schedule before loading the trainable UNet.

    Precomputation permits deterministic VAE-mode caching without persisting a
    latent bank.  The root, schedule RNG end states, and schedule commitment are
    retained only in the encrypted resume envelope.
    """

    import torch

    from dp_training.mechanism import poisson_select

    require(arm in ARMS, "unknown K5 arm")
    require(len(experiment_root) == 32, "invalid experiment root")
    require(steps >= 1 and steps <= FULL_STEPS, "invalid schedule step count")
    by_image = {record.image_id: record for record in records}
    require(len(by_image) == len(records), "schedule input has duplicate images")
    by_patient: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        by_patient[record.patient_id].append(record)
    for values in by_patient.values():
        values.sort(key=lambda record: (int(record.cap_rank), record.image_id))
    image_ids = sorted(by_image)
    patient_ids = sorted(by_patient)
    schedule: list[list[dict[str, Any]]] = []
    generators: dict[str, Any] = {}
    config = None

    if arm == "M0":
        require(len(image_ids) >= 8, "M0 requires at least eight fixture/full images")
        sampling = torch.Generator(device="cpu").manual_seed(
            derive_seed(experiment_root, "M0-private-permutation")
        )
        diffusion = torch.Generator(device="cpu").manual_seed(
            derive_seed(experiment_root, "M0-diffusion-draws")
        )
        generators = {"sampling": sampling, "diffusion": diffusion}
        permutation = torch.empty(0, dtype=torch.int64)
        cursor = len(image_ids)
        epoch = -1
        for step_number in range(1, steps + 1):
            if cursor + 8 > len(image_ids):
                permutation = torch.randperm(len(image_ids), generator=sampling)
                cursor = 0
                epoch += 1
            indices = permutation[cursor : cursor + 8].tolist()
            cursor += 8
            chosen = [by_image[image_ids[int(index)]] for index in indices]
            schedule.append(
                [
                    {
                        "unit_type": "m0_batch",
                        "unit_id": f"m0-step-{step_number:04d}",
                        "patient_id": "MULTIPLE",
                        "records": chosen,
                        "draws": [
                            _draw_diffusion(diffusion, scheduler_timesteps) for _ in chosen
                        ],
                    }
                ]
            )
        extra = {"m0_epoch_after_schedule": epoch, "m0_cursor_after_schedule": cursor}
    elif arm.startswith("M1"):
        expected = 8
        if not fixture:
            require(len(image_ids) == int(protocol["DP_arms"][arm]["population"]), "M1 population drift")
        config = mechanism_config(
            protocol,
            arm,
            max_steps=steps,
            population_override=len(image_ids) if fixture else None,
            expected_batch_override=expected if fixture else None,
        )
        sampling = torch.Generator(device="cpu").manual_seed(
            derive_seed(experiment_root, "m1-shared-poisson-selection")
        )
        diffusion = torch.Generator(device="cpu").manual_seed(
            derive_seed(experiment_root, "m1-shared-diffusion-draws")
        )
        generators = {"sampling": sampling, "diffusion": diffusion}
        maximum = 64 if not fixture else len(image_ids)
        for step_number in range(1, steps + 1):
            selected = poisson_select(
                image_ids, config.poisson_sample_rate, generator=sampling
            )
            require(
                len(selected) <= maximum,
                f"M1 step {step_number} exceeds frozen resource limit; do not resample",
            )
            schedule.append(
                [
                    {
                        "unit_type": "image",
                        "unit_id": image_id,
                        "patient_id": by_image[image_id].patient_id,
                        "records": [by_image[image_id]],
                        "draws": [_draw_diffusion(diffusion, scheduler_timesteps)],
                    }
                    for image_id in selected
                ]
            )
        extra = {}
    else:
        expected = 4
        if not fixture:
            require(len(patient_ids) == int(protocol["DP_arms"][arm]["population"]), "M2 population drift")
        config = mechanism_config(
            protocol,
            arm,
            max_steps=steps,
            population_override=len(patient_ids) if fixture else None,
            expected_batch_override=expected if fixture else None,
        )
        sampling = torch.Generator(device="cpu").manual_seed(
            derive_seed(experiment_root, "m2-patient-poisson-selection")
        )
        inner = torch.Generator(device="cpu").manual_seed(
            derive_seed(experiment_root, "m2-within-patient-selection")
        )
        diffusion = torch.Generator(device="cpu").manual_seed(
            derive_seed(experiment_root, "m2-diffusion-draws")
        )
        generators = {"sampling": sampling, "inner": inner, "diffusion": diffusion}
        maximum_patients = 32 if not fixture else len(patient_ids)
        maximum_images = 128 if not fixture else 4 * len(patient_ids)
        for step_number in range(1, steps + 1):
            selected = poisson_select(
                patient_ids, config.poisson_sample_rate, generator=sampling
            )
            require(
                len(selected) <= maximum_patients,
                f"M2 step {step_number} exceeds frozen patient limit; do not resample",
            )
            units = []
            raw_images = 0
            for patient_id in selected:
                available = by_patient[patient_id]
                order = torch.randperm(len(available), generator=inner).tolist()
                chosen = [available[index] for index in order[: min(4, len(available))]]
                raw_images += len(chosen)
                units.append(
                    {
                        "unit_type": "patient",
                        "unit_id": patient_id,
                        "patient_id": patient_id,
                        "records": chosen,
                        "draws": [
                            _draw_diffusion(diffusion, scheduler_timesteps) for _ in chosen
                        ],
                    }
                )
            require(
                raw_images <= maximum_images,
                f"M2 step {step_number} exceeds frozen raw-image limit; do not resample",
            )
            schedule.append(units)
        extra = {}

    generator_states = {
        name: generator.get_state().cpu().clone() for name, generator in generators.items()
    }
    secret_commitment = schedule_commitment(experiment_root, arm, schedule)
    meta = {
        "arm": arm,
        "steps": steps,
        "fixture": bool(fixture),
        "commitment": secret_commitment,
        "generator_end_states": generator_states,
        "unique_images": len(
            {
                record.image_id
                for step in schedule
                for unit in step
                for record in unit["records"]
            }
        ),
        "scheduled_units": sum(len(step) for step in schedule),
        "scheduled_raw_images": sum(
            len(unit["records"]) for step in schedule for unit in step
        ),
        **extra,
    }
    return schedule, meta, config


def prepare_arm_cache(
    snapshot: Path,
    schedule: Sequence[Sequence[dict[str, Any]]],
    image_root: Path,
    model_input: Any,
    *,
    first_step_index: int = 0,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from dp_protocol.calibrate_xray_public_clip_norms import prepare_latents_and_text

    units = [unit for step in schedule[first_step_index:] for unit in step]
    if not units:
        return {}, {}, {
            "unique_images": 0,
            "unique_prompts": 0,
            "latent_shape": [1, 4, IMAGE_SIZE // 8, IMAGE_SIZE // 8],
            "vae_sampling": "latent_dist.mode",
            "all_remaining_DP_steps_empty": True,
        }
    return prepare_latents_and_text(snapshot, units, image_root, model_input)


def named_trainable(unet: Any) -> list[tuple[str, Any]]:
    values = sorted([
        (name, parameter)
        for name, parameter in unet.named_parameters()
        if parameter.requires_grad
    ], key=lambda item: item[0])
    require(sum(parameter.numel() for _, parameter in values) == TRAINABLE_PARAMETERS, "LoRA trainable count")
    return values


def named_tensor_digest(named_tensors: Iterable[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest().upper()


def flatten_named_tensors(named_tensors: Iterable[tuple[str, Any]]) -> Any:
    import torch

    return torch.cat(
        [tensor.detach().float().reshape(-1) for _, tensor in named_tensors], dim=0
    )


def assign_flat_gradient(trainable: Sequence[tuple[str, Any]], flat: Any) -> None:
    offset = 0
    for _, parameter in trainable:
        count = parameter.numel()
        parameter.grad = flat[offset : offset + count].view_as(parameter).detach().clone()
        offset += count
    require(offset == int(flat.numel()), "flat gradient length mismatch")


def create_training_state(
    snapshot: Path,
    protocol: dict[str, Any],
    arm: str,
    experiment_root: bytes,
    config: Any | None,
    schedule_meta: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from diffusers import UNet2DConditionModel
    from peft import LoraConfig
    from peft.utils.other import cast_mixed_precision_params

    require(arm in ARMS, "unknown arm")
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
    trainable = named_trainable(unet)
    require({parameter.dtype for _, parameter in trainable} == {torch.float32}, "LoRA dtype")
    optimizer_contract = protocol["optimizer"]
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=float(optimizer_contract["learning_rate"]),
        betas=tuple(float(value) for value in optimizer_contract["betas"]),
        eps=float(optimizer_contract["epsilon"]),
        weight_decay=float(optimizer_contract["weight_decay"]),
        foreach=False,
        fused=False,
    )
    gaussian = None
    if arm in DP_ARMS:
        gaussian = torch.Generator(device="cuda").manual_seed(
            derive_seed(experiment_root, f"{arm}-dp-gaussian-stream")
        )
    return {
        "arm": arm,
        "step": 0,
        "unet": unet,
        "trainable": trainable,
        "optimizer": optimizer,
        "gaussian": gaussian,
        "config": config,
        "trace_head": TRACE_GENESIS,
        "trace_event_count": 0,
        "initial_adapter_sha256": named_tensor_digest(trainable),
        "schedule_commitment": schedule_meta["commitment"],
    }


def destroy_training_state(state: dict[str, Any] | None) -> None:
    import torch

    if state is None:
        return
    state.clear()
    gc.collect()
    torch.cuda.empty_cache()


def optimizer_is_finite(optimizer: Any) -> bool:
    import torch

    for values in optimizer.state.values():
        for value in values.values():
            if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
                return False
    return True


def compute_unit_gradient(
    *,
    unit: dict[str, Any],
    latent_by_image: dict[str, Any],
    hidden_by_prompt: dict[str, Any],
    model_input: Any,
    scheduler: Any,
    state: dict[str, Any],
) -> tuple[Any, dict[str, float]]:
    import torch
    import torch.nn.functional as functional

    records = unit["records"]
    draws = unit["draws"]
    require(len(records) == len(draws) and len(records) >= 1, "unit record/draw boundary")
    if unit["unit_type"] == "image":
        require(len(records) == 1, "image unit boundary")
    elif unit["unit_type"] == "patient":
        require(len(records) <= 4, "patient image cap")
        require(
            {record.patient_id for record in records} == {unit["patient_id"]},
            "patient unit crosses patients",
        )
    else:
        require(unit["unit_type"] == "m0_batch" and len(records) == 8, "M0 batch boundary")

    latents = []
    hidden = []
    noises = []
    timesteps = []
    for record, draw in zip(records, draws):
        latent = latent_by_image[record.image_id].to(device="cuda", dtype=torch.float16)
        text = hidden_by_prompt[model_input.prompt_for_record(record)].to(
            device="cuda", dtype=torch.float16
        )
        generator = torch.Generator(device="cuda").manual_seed(int(draw["noise_seed"]))
        noise = torch.randn(
            latent.shape,
            generator=generator,
            device="cuda",
            dtype=torch.float16,
        )
        latents.append(latent)
        hidden.append(text)
        noises.append(noise)
        timesteps.append(int(draw["timestep"]))

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

    trainable = state["trainable"]
    for _, parameter in trainable:
        parameter.grad = None
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        prediction = state["unet"](noisy, timestep_batch, hidden_batch).sample
        per_image = functional.mse_loss(
            prediction.float(), target.float(), reduction="none"
        ).flatten(1).mean(1)
        loss = per_image.mean()
    require(bool(torch.isfinite(loss)), "non-finite loss")
    loss.backward()
    gradients = []
    norm_squared = 0.0
    for name, parameter in trainable:
        require(parameter.grad is not None, f"missing gradient: {name}")
        gradient = parameter.grad.detach().float()
        require(bool(torch.isfinite(gradient).all()), "non-finite unit gradient")
        gradients.append((name, gradient))
        norm_squared += float(torch.sum(gradient.double() ** 2).cpu())
    return flatten_named_tensors(gradients), {
        "loss": float(loss.detach().cpu()),
        "gradient_l2_norm": math.sqrt(norm_squared),
    }


def _advance_trace(state: dict[str, Any], step_number: int) -> None:
    from dp_training.mechanism import canonical_json as mechanism_canonical_json
    from dp_training.mechanism import make_public_scheduled_event, sha256_bytes as mechanism_sha256

    if state["arm"] == "M0":
        event = {
            "schema": "nih-cxr14-k5-m0-public-scheduled-event/v1",
            "arm": "M0",
            "first_step": step_number,
            "event_count": 1,
            "batch_size": 8,
            "rng_security_mode": "RESEARCH_ONLY_NONCRYPTOGRAPHIC",
        }
    else:
        event = make_public_scheduled_event(
            state["config"], first_step=step_number, event_count=1
        )
    state["trace_head"] = mechanism_sha256(
        mechanism_canonical_json(
            {"previous_hash": state["trace_head"], "event": event}
        )
    )
    state["trace_event_count"] = int(state["trace_event_count"]) + 1


def run_one_step(
    *,
    state: dict[str, Any],
    step_units: Sequence[dict[str, Any]],
    latent_by_image: dict[str, Any],
    hidden_by_prompt: dict[str, Any],
    model_input: Any,
    scheduler: Any,
) -> dict[str, Any]:
    import torch

    from dp_training.mechanism import aggregate_noised_update

    step_number = int(state["step"]) + 1
    require(step_number <= int(state["config"].max_steps) if state["config"] is not None else step_number <= FULL_STEPS, "step overflow")
    unit_vectors = []
    summaries = []
    for unit in step_units:
        vector, summary = compute_unit_gradient(
            unit=unit,
            latent_by_image=latent_by_image,
            hidden_by_prompt=hidden_by_prompt,
            model_input=model_input,
            scheduler=scheduler,
            state=state,
        )
        unit_vectors.append(vector)
        summaries.append(summary)

    if state["arm"] == "M0":
        require(len(unit_vectors) == 1 and len(step_units) == 1, "M0 batch aggregation")
        update = unit_vectors[0]
        aggregation = {"realized_unit_count": 8, "empty_sample": False}
    else:
        template = torch.zeros(
            TRAINABLE_PARAMETERS, device="cuda", dtype=torch.float32
        )
        update, aggregation = aggregate_noised_update(
            unit_vectors,
            template=template,
            config=state["config"],
            generator=state["gaussian"],
        )
        require(
            int(aggregation["realized_unit_count"]) == len(step_units),
            "DP aggregation unit count",
        )
    require(bool(torch.isfinite(update).all()), "non-finite optimizer update")
    assign_flat_gradient(state["trainable"], update)
    state["optimizer"].step()
    require(optimizer_is_finite(state["optimizer"]), "non-finite optimizer state")
    require(
        all(bool(torch.isfinite(parameter).all()) for _, parameter in state["trainable"]),
        "non-finite adapter",
    )
    state["optimizer"].zero_grad(set_to_none=True)
    state["step"] = step_number
    _advance_trace(state, step_number)
    return {
        "step": step_number,
        "selected_units": len(step_units) if state["arm"] != "M0" else 8,
        "selected_raw_images": sum(len(unit["records"]) for unit in step_units),
        "empty_sample": len(step_units) == 0,
        "mean_unit_loss": (
            sum(item["loss"] for item in summaries) / len(summaries) if summaries else None
        ),
        "maximum_unit_gradient_l2": max(
            (item["gradient_l2_norm"] for item in summaries), default=0.0
        ),
        "adapter_sha256": named_tensor_digest(state["trainable"]),
        "trace_head": state["trace_head"],
    }


def adapter_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu().contiguous().clone()
        for name, parameter in state["trainable"]
    }


def frozen_digest_bundle(
    *,
    environment_manifest_sha256: str,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_SHA256,
        "k5_manifest": K5_SHA256,
        "preprocessing": PREPROCESSING_SHA256,
        "environment_manifest": environment_manifest_sha256,
        "sources": dict(sorted(source_hashes.items())),
    }


def capture_resume_payload(
    state: dict[str, Any],
    *,
    experiment_root: bytes,
    schedule_meta: dict[str, Any],
    frozen_digests: dict[str, Any],
) -> dict[str, Any]:
    import torch

    require(state["schedule_commitment"] == schedule_meta["commitment"], "schedule commitment drift")
    return {
        "schema": RESUME_SCHEMA,
        "run_id": RUN_ID,
        "arm": state["arm"],
        "committed_step": int(state["step"]),
        "adapter": adapter_state_dict(state),
        "optimizer": tree_to_cpu(state["optimizer"].state_dict()),
        "experiment_root": bytes(experiment_root),
        "schedule_commitment": schedule_meta["commitment"],
        "schedule_generator_end_states": tree_to_cpu(
            schedule_meta["generator_end_states"]
        ),
        "dp_gaussian_generator_state": (
            None
            if state["gaussian"] is None
            else state["gaussian"].get_state().cpu().clone()
        ),
        "global_cpu_rng": torch.get_rng_state().cpu().clone(),
        "global_cuda_rng_all": [
            value.cpu().clone() for value in torch.cuda.get_rng_state_all()
        ],
        "public_trace_head": state["trace_head"],
        "public_trace_event_count": int(state["trace_event_count"]),
        "initial_adapter_sha256": state["initial_adapter_sha256"],
        "frozen_digests": frozen_digests,
    }


def restore_training_state(
    payload: dict[str, Any],
    *,
    snapshot: Path,
    protocol: dict[str, Any],
    arm: str,
    experiment_root: bytes,
    config: Any | None,
    schedule_meta: dict[str, Any],
    frozen_digests: dict[str, Any],
) -> dict[str, Any]:
    import torch

    require(payload["schema"] == RESUME_SCHEMA, "resume schema mismatch")
    require(payload["run_id"] == RUN_ID and payload["arm"] == arm, "resume identity mismatch")
    require(bytes(payload["experiment_root"]) == experiment_root, "matrix root mismatch")
    require(payload["frozen_digests"] == frozen_digests, "resume frozen digest drift")
    require(payload["schedule_commitment"] == schedule_meta["commitment"], "resume schedule drift")
    require(
        exact_equal(
            payload["schedule_generator_end_states"],
            schedule_meta["generator_end_states"],
        ),
        "resume schedule generator-state drift",
    )
    committed = int(payload["committed_step"])
    require(0 <= committed <= int(schedule_meta["steps"]), "resume step outside schedule")
    state = create_training_state(
        snapshot, protocol, arm, experiment_root, config, schedule_meta
    )
    expected_names = [name for name, _ in state["trainable"]]
    require(list(payload["adapter"].keys()) == expected_names, "resume adapter key/order drift")
    for name, parameter in state["trainable"]:
        restored = payload["adapter"][name]
        require(tuple(restored.shape) == tuple(parameter.shape), f"resume shape drift: {name}")
        parameter.data.copy_(restored.to(device="cuda", dtype=torch.float32))
    state["optimizer"].load_state_dict(payload["optimizer"])
    if arm in DP_ARMS:
        require(payload["dp_gaussian_generator_state"] is not None, "missing DP Gaussian state")
        state["gaussian"].set_state(payload["dp_gaussian_generator_state"].cpu())
    else:
        require(payload["dp_gaussian_generator_state"] is None, "M0 has DP Gaussian state")
    state["step"] = committed
    state["trace_head"] = str(payload["public_trace_head"])
    state["trace_event_count"] = int(payload["public_trace_event_count"])
    state["initial_adapter_sha256"] = str(payload["initial_adapter_sha256"])
    require(state["trace_event_count"] == committed, "resume trace count/step mismatch")
    torch.set_rng_state(payload["global_cpu_rng"].cpu())
    torch.cuda.set_rng_state_all(
        [value.cpu() for value in payload["global_cuda_rng_all"]]
    )
    require(optimizer_is_finite(state["optimizer"]), "restored optimizer is non-finite")
    return state


def compare_resume_payloads(
    uninterrupted: dict[str, Any], resumed: dict[str, Any]
) -> dict[str, bool]:
    keys = {
        "adapter_exact": "adapter",
        "optimizer_exact": "optimizer",
        "schedule_generator_states_exact": "schedule_generator_end_states",
        "dp_gaussian_generator_exact": "dp_gaussian_generator_state",
        "global_cpu_rng_exact": "global_cpu_rng",
        "global_cuda_rng_exact": "global_cuda_rng_all",
        "trace_head_exact": "public_trace_head",
        "trace_count_exact": "public_trace_event_count",
        "schedule_commitment_exact": "schedule_commitment",
        "initial_adapter_exact": "initial_adapter_sha256",
        "frozen_digests_exact": "frozen_digests",
        "committed_step_exact": "committed_step",
        "experiment_root_exact_internal_only": "experiment_root",
    }
    result = {
        label: exact_equal(uninterrupted[key], resumed[key])
        for label, key in keys.items()
    }
    require(all(result.values()), f"exact resume mismatch: {result}")
    return result


def verify_adapter_snapshot(directory: Path, expected_digest: str) -> dict[str, Any]:
    from safetensors.torch import load_file

    manifest_path = directory / "manifest.json"
    adapter_path = directory / "adapter.safetensors"
    require(manifest_path.is_file() and adapter_path.is_file(), "adapter snapshot incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["adapter_file_sha256"] == sha256_file(adapter_path), "adapter file hash")
    tensors = load_file(str(adapter_path), device="cpu")
    digest = named_tensor_digest((name, tensors[name]) for name in sorted(tensors))
    require(digest == expected_digest == manifest["adapter_tensor_sha256"], "adapter tensor digest")
    return manifest


def save_adapter_snapshot(
    state: dict[str, Any], arm_root: Path, frozen_digests: dict[str, Any]
) -> dict[str, Any]:
    from safetensors.torch import save_file

    step = int(state["step"])
    require(step in ADAPTER_STEPS, "adapter snapshot step is not frozen")
    final = arm_root / "adapters" / f"step_{step:04d}"
    expected_digest = named_tensor_digest(state["trainable"])
    if final.exists():
        return verify_adapter_snapshot(final, expected_digest)
    temporary = final.with_name(final.name + ".new")
    require(not temporary.exists(), "stale adapter snapshot temporary requires audit")
    temporary.mkdir(parents=True)
    tensors = adapter_state_dict(state)
    adapter_path = temporary / "adapter.safetensors"
    save_file(tensors, str(adapter_path))
    with adapter_path.open("rb") as handle:
        os.fsync(handle.fileno())
    manifest = {
        "schema": "nih-cxr14-k5-lora-snapshot/v1",
        "run_id": RUN_ID,
        "arm": state["arm"],
        "step": step,
        "classification": "LOCAL_RESTRICTED_NOT_FOR_RELEASE",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "adapter_file": adapter_path.name,
        "adapter_file_sha256": sha256_file(adapter_path),
        "adapter_tensor_sha256": expected_digest,
        "trainable_parameters": TRAINABLE_PARAMETERS,
        "frozen_digests": frozen_digests,
        "quantitative_evaluation_allowed": step == FULL_STEPS,
    }
    manifest_path = temporary / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with manifest_path.open("rb") as handle:
        os.fsync(handle.fileno())
    verify_adapter_snapshot(temporary, expected_digest)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, final)
    return verify_adapter_snapshot(final, expected_digest)


def require_free_space(path: Path, minimum_gib: float = 15.0) -> float:
    existing = path
    while not existing.exists():
        require(existing.parent != existing, "cannot locate disk for free-space check")
        existing = existing.parent
    free_gib = shutil.disk_usage(existing).free / (1024**3)
    require(free_gib >= minimum_gib, f"free space {free_gib:.2f} GiB is below {minimum_gib:.2f} GiB")
    return free_gib
