#!/usr/bin/env python3
"""Bounded public Q=2 SD 2.1 + PatientSetAdapter architecture smoke."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from set_adapter import PatientSetAdapter, PatientSetMidBlock


SCHEMA = "nih-cxr14-q2-setadapter-architecture-smoke/v1"
SELECTION_SALT = "nih-cxr14-q2-setadapter-architecture-smoke-v1"
PROTOCOL_SHA256 = "C6766B87010029DB412DBD45A62E9B4FB50E5CE275AFE717EC4B9E4DFCEB5DCD"
CONFIRMATION_SELECTION_SHA256 = "E1ABC33EA68082F69086CDFB752B558580710FAD35FB70640BDAD590DFC2C6F8"
EXPECTED_LORA_PARAMETERS = 1_659_904
EXPECTED_SET_PARAMETERS = 463_872
EXPECTED_TOTAL_PARAMETERS = EXPECTED_LORA_PARAMETERS + EXPECTED_SET_PARAMETERS
DIAGNOSTIC_CLIP_NORM = 1.0
MEMORY_LIMIT_BYTES = int(7.5 * 1024**3)
PROBE_WEIGHT_STD = 0.01
PROBE_SEED = 26090339762


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def default_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parent.parent
    return {
        "root": root,
        "manifest": root / "_data" / "derived" / "nih_cxr14_pa_target_enriched_v1" / "k10_private.csv",
        "image_root": root / "_data" / "raw" / "nih_cxr14_pa_k10_plus_census_v1" / "images",
        "preprocessing_contract": root / "_reports" / "nih_cxr14_sd21_ppmark_interface_v1_001" / "preprocessing_contract.json",
        "confirmation_selection": root / "_reports" / "nih_cxr14_patient_set_premise_confirmation_v2_001" / "selected_public_patient_sets_private.csv",
        "protocol": root / "patient_set_diffusion" / "NIH_CXR14_SETADAPTER_ARCHITECTURE_SMOKE_PROTOCOL_V1.md",
        "output": root / "_reports" / "nih_cxr14_q2_setadapter_architecture_smoke_v1_001",
    }


def parse_args() -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser()
    for name in ("manifest", "image_root", "preprocessing_contract", "confirmation_selection", "protocol"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=paths[name])
    parser.add_argument("--output-dir", type=Path, default=paths["output"])
    return parser.parse_args()


def select_public_q2_patient(
    selection_path: Path,
    records: list[Any],
    stable_hash: Any,
) -> dict[str, Any]:
    with selection_path.open("r", encoding="utf-8", newline="") as handle:
        candidates = [
            row
            for row in csv.DictReader(handle)
            if int(row["target_patient"]) == 0 and int(row["record_count"]) == 2
        ]
    require(len(candidates) == 40, f"expected 40 target0_n2 confirmation patients, got {len(candidates)}")
    selected = min(
        candidates,
        key=lambda row: (stable_hash(SELECTION_SALT, "patient", row["patient_id"]), row["patient_id"]),
    )
    image_ids = selected["image_ids"].split("|")
    require(len(image_ids) == 2 and len(set(image_ids)) == 2, "selected row is not Q=2")
    record_by_id = {record.image_id: record for record in records}
    require(all(image_id in record_by_id for image_id in image_ids), "selected image absent from manifest")
    chosen_records = [record_by_id[image_id] for image_id in image_ids]
    require(all(str(record.patient_id) == selected["patient_id"] for record in chosen_records), "patient mismatch")
    require(all(int(record.target_patient) == 0 for record in chosen_records), "target mismatch")
    return {
        "patient_id": selected["patient_id"],
        "target_patient": 0,
        "records": chosen_records,
        "selection_commitment": stable_hash(SELECTION_SALT, selected["patient_id"], *image_ids),
    }


def write_private_selection(path: Path, unit: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["patient_id", "target_patient", "image_ids", "selection_commitment"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "patient_id": unit["patient_id"],
                "target_patient": unit["target_patient"],
                "image_ids": "|".join(record.image_id for record in unit["records"]),
                "selection_commitment": unit["selection_commitment"],
            }
        )


def gradient_summary(named_parameters: list[tuple[str, Any]]) -> dict[str, Any]:
    import torch

    missing = 0
    nonfinite = 0
    nonzero_tensors = 0
    squared = 0.0
    for _, parameter in named_parameters:
        if parameter.grad is None:
            missing += 1
            continue
        gradient = parameter.grad.detach().float()
        nonfinite += not bool(torch.isfinite(gradient).all())
        nonzero_tensors += bool(torch.count_nonzero(gradient).item())
        squared += float(torch.sum(gradient.double() ** 2).cpu())
    return {
        "parameter_tensors": len(named_parameters),
        "scalars": sum(parameter.numel() for _, parameter in named_parameters),
        "missing_gradient_tensors": missing,
        "nonfinite_gradient_tensors": nonfinite,
        "nonzero_gradient_tensors": nonzero_tensors,
        "gradient_l2_norm": math.sqrt(squared),
    }


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "dp_protocol"))
    import calibrate_xray_public_clip_norms as calibration

    calibration.disable_broken_optional_onnx()
    import torch
    import torch.nn.functional as functional
    from diffusers import DDPMScheduler, UNet2DConditionModel
    from huggingface_hub import snapshot_download
    from peft import LoraConfig
    from peft.utils.other import cast_mixed_precision_params

    require(torch.cuda.is_available(), "CUDA is required for the SetAdapter architecture smoke")
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(PROBE_SEED)
    torch.cuda.manual_seed_all(PROBE_SEED)

    manifest = args.manifest.resolve()
    image_root = args.image_root.resolve()
    preprocessing_contract = args.preprocessing_contract.resolve()
    confirmation_selection = args.confirmation_selection.resolve()
    protocol = args.protocol.resolve()
    require(calibration.sha256_file(protocol) == PROTOCOL_SHA256, "protocol hash mismatch")
    require(calibration.sha256_file(manifest) == calibration.K10_SHA256, "manifest hash mismatch")
    require(
        calibration.sha256_file(preprocessing_contract) == calibration.PREPROCESSING_CONTRACT_SHA256,
        "preprocessing contract hash mismatch",
    )
    require(
        calibration.sha256_file(confirmation_selection) == CONFIRMATION_SELECTION_SHA256,
        "confirmation selection hash mismatch",
    )

    sys.path.insert(0, str(root / "data_pipeline"))
    import nih_cxr14_model_input as model_input

    records = model_input.read_manifest(manifest, partitions={"public_development"})
    unit = select_public_q2_patient(confirmation_selection, records, calibration.stable_hash)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    private_selection_path = output_dir / "selected_public_q2_patient_private.csv"
    write_private_selection(private_selection_path, unit)

    snapshot = Path(
        snapshot_download(
            repo_id=calibration.MODEL_ID,
            revision=calibration.MODEL_REVISION,
            local_files_only=True,
        )
    ).resolve()
    require(snapshot.name == calibration.MODEL_REVISION, "snapshot revision mismatch")
    snapshot_hashes = calibration.verify_snapshot(snapshot)
    latent_by_image, hidden_by_prompt, input_summary = calibration.prepare_latents_and_text(
        snapshot,
        [unit],
        image_root,
        model_input,
    )
    scheduler = DDPMScheduler.from_pretrained(snapshot / "scheduler", local_files_only=True)

    latents = []
    hidden = []
    noises = []
    timesteps = []
    seed_commitments = []
    for record in unit["records"]:
        latent = latent_by_image[record.image_id].to(device="cuda", dtype=torch.float16)
        prompt = model_input.prompt_for_record(record)
        hidden.append(hidden_by_prompt[prompt].to(device="cuda", dtype=torch.float16))
        timestep = calibration.hash_uint63(SELECTION_SALT, "timestep", record.image_id) % int(
            scheduler.config.num_train_timesteps
        )
        noise_seed = calibration.hash_uint63(SELECTION_SALT, "noise", record.image_id)
        generator = torch.Generator(device="cuda").manual_seed(noise_seed)
        noises.append(torch.randn(latent.shape, generator=generator, device="cuda", dtype=torch.float16))
        latents.append(latent)
        timesteps.append(int(timestep))
        seed_commitments.append(calibration.stable_hash("public-setadapter-seed", noise_seed))
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

    del latent_by_image, hidden_by_prompt, latents, hidden, noises
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    torch.manual_seed(calibration.LORA_INIT_SEED)
    torch.cuda.manual_seed_all(calibration.LORA_INIT_SEED)
    unet = UNet2DConditionModel.from_pretrained(
        snapshot / "unet",
        torch_dtype=torch.float16,
        variant=calibration.MODEL_VARIANT,
        use_safetensors=True,
        local_files_only=True,
    )
    unet.requires_grad_(False)
    unet.add_adapter(
        LoraConfig(
            r=calibration.LORA_RANK,
            lora_alpha=calibration.LORA_RANK,
            init_lora_weights="gaussian",
            target_modules=list(calibration.LORA_TARGETS),
        )
    )
    cast_mixed_precision_params(unet, dtype=torch.float16)
    channels = int(unet.config.block_out_channels[-1])
    set_adapter = PatientSetAdapter(channels=channels, token_dim=128, num_heads=4)
    mid_wrapper = PatientSetMidBlock(unet.mid_block, set_adapter)
    unet.mid_block = mid_wrapper
    unet.enable_gradient_checkpointing()
    unet.train().to("cuda")

    trainable = [(name, parameter) for name, parameter in unet.named_parameters() if parameter.requires_grad]
    lora_named = [(name, parameter) for name, parameter in trainable if "lora_" in name]
    set_named = [(name, parameter) for name, parameter in trainable if ".set_adapter." in name]
    categorized = {name for name, _ in lora_named + set_named}
    uncategorized = [name for name, _ in trainable if name not in categorized]
    require(not uncategorized, f"uncategorized trainable parameters: {uncategorized[:3]}")
    require(sum(parameter.numel() for _, parameter in lora_named) == EXPECTED_LORA_PARAMETERS, "LoRA count")
    require(sum(parameter.numel() for _, parameter in set_named) == EXPECTED_SET_PARAMETERS, "SetAdapter count")
    require(sum(parameter.numel() for _, parameter in trainable) == EXPECTED_TOTAL_PARAMETERS, "total count")
    require({str(parameter.dtype) for _, parameter in trainable} == {"torch.float32"}, "trainable dtype")
    lora_before = calibration.tensor_digest(lora_named)
    set_before = calibration.tensor_digest(set_named)
    valid_q2 = torch.tensor([[True, True]], device="cuda", dtype=torch.bool)

    # Integrated zero-init identity against explicit mid-block bypass.
    unet.eval()
    unet.disable_gradient_checkpointing()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        mid_wrapper.configure(valid_q2, bypass=True)
        prediction_bypass = unet(noisy, timestep_batch, hidden_batch).sample
        mid_wrapper.configure(valid_q2, bypass=False)
        prediction_zero = unet(noisy, timestep_batch, hidden_batch).sample
    zero_init_max_abs = float(torch.max(torch.abs(prediction_zero.float() - prediction_bypass.float())).cpu())

    # One real public Q=2 backward, no optimizer and no DP noise.
    unet.train()
    unet.enable_gradient_checkpointing()
    mid_wrapper.configure(valid_q2, bypass=False)
    unet.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        prediction = unet(noisy, timestep_batch, hidden_batch).sample
        per_record_loss = functional.mse_loss(
            prediction.float(), target.float(), reduction="none"
        ).flatten(1).mean(1)
        loss = per_record_loss.mean()
    loss.backward()
    lora_gradients = gradient_summary(lora_named)
    set_gradients = gradient_summary(set_named)
    output_projection_named = [
        (name, parameter) for name, parameter in set_named if ".to_delta." in name
    ]
    output_projection_gradients = gradient_summary(output_projection_named)
    flat_gradient = torch.cat(
        [
            (parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).detach().float().reshape(-1)
            for _, parameter in trainable
        ]
    )
    combined_norm = float(torch.linalg.vector_norm(flat_gradient.double()).cpu())
    clip_factor = min(1.0, DIAGNOSTIC_CLIP_NORM / max(combined_norm, 1e-30))
    clipped_gradient = flat_gradient * clip_factor
    clipped_norm = float(torch.linalg.vector_norm(clipped_gradient.double()).cpu())
    gradients_finite = bool(torch.isfinite(flat_gradient).all() and torch.isfinite(clipped_gradient).all())
    del prediction, flat_gradient, clipped_gradient
    unet.zero_grad(set_to_none=True)

    # Temporary nonzero output projection: graph/interaction probe only, not an optimizer step.
    probe_generator = torch.Generator(device="cuda").manual_seed(PROBE_SEED)
    with torch.no_grad():
        set_adapter.to_delta.weight.copy_(
            torch.randn(
                set_adapter.to_delta.weight.shape,
                generator=probe_generator,
                device="cuda",
                dtype=torch.float32,
            )
            * PROBE_WEIGHT_STD
        )
        set_adapter.to_delta.bias.zero_()
    probe_projection_digest = calibration.tensor_digest(
        [("to_delta.weight", set_adapter.to_delta.weight), ("to_delta.bias", set_adapter.to_delta.bias)]
    )
    require(calibration.tensor_digest(lora_named) == lora_before, "LoRA changed without optimizer")

    unet.eval()
    unet.disable_gradient_checkpointing()
    swap = torch.tensor([1, 0], device="cuda", dtype=torch.long)
    counterfactual = noisy.clone()
    counterfactual[1] = -counterfactual[1]
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        mid_wrapper.configure(valid_q2, bypass=False)
        activated = unet(noisy, timestep_batch, hidden_batch).sample
        permuted = unet(noisy[swap], timestep_batch[swap], hidden_batch[swap]).sample
        counterfactual_prediction = unet(counterfactual, timestep_batch, hidden_batch).sample
        valid_q1 = torch.tensor([[True]], device="cuda", dtype=torch.bool)
        mid_wrapper.configure(valid_q1, bypass=False)
        singleton_active = unet(noisy[:1], timestep_batch[:1], hidden_batch[:1]).sample
        mid_wrapper.configure(valid_q1, bypass=True)
        singleton_bypass = unet(noisy[:1], timestep_batch[:1], hidden_batch[:1]).sample
    restored_permuted = permuted[swap]
    permutation_max_abs = float(torch.max(torch.abs(activated.float() - restored_permuted.float())).cpu())
    other_record_influence_l2 = float(
        torch.linalg.vector_norm(
            (activated[0].float() - counterfactual_prediction[0].float()).reshape(-1).double()
        ).cpu()
    )
    singleton_max_abs = float(
        torch.max(torch.abs(singleton_active.float() - singleton_bypass.float())).cpu()
    )
    predictions_finite = all(
        bool(torch.isfinite(value).all())
        for value in (
            prediction_bypass,
            prediction_zero,
            activated,
            permuted,
            counterfactual_prediction,
            singleton_active,
            singleton_bypass,
        )
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_memory = int(torch.cuda.max_memory_allocated())

    gate_checks = {
        "parameter_counts_exact": (
            len(uncategorized) == 0
            and sum(parameter.numel() for _, parameter in lora_named) == EXPECTED_LORA_PARAMETERS
            and sum(parameter.numel() for _, parameter in set_named) == EXPECTED_SET_PARAMETERS
        ),
        "integrated_zero_init_identity_max_abs_at_most_1e_6": zero_init_max_abs <= 1e-6,
        "q2_backward_all_gradients_present_finite_and_nonzero_paths": (
            bool(torch.isfinite(loss))
            and lora_gradients["missing_gradient_tensors"] == 0
            and set_gradients["missing_gradient_tensors"] == 0
            and lora_gradients["nonfinite_gradient_tensors"] == 0
            and set_gradients["nonfinite_gradient_tensors"] == 0
            and lora_gradients["gradient_l2_norm"] > 0.0
            and set_gradients["gradient_l2_norm"] > 0.0
            and output_projection_gradients["gradient_l2_norm"] > 0.0
        ),
        "combined_patient_vector_clip_operator_finite_and_bounded": (
            gradients_finite and combined_norm > 0.0 and clipped_norm <= DIAGNOSTIC_CLIP_NORM + 1e-5
        ),
        "activated_permutation_equivariance_max_abs_at_most_0_005": permutation_max_abs <= 0.005,
        "activated_other_record_influence_l2_greater_than_1e_6": other_record_influence_l2 > 1e-6,
        "activated_singleton_identity_max_abs_at_most_1e_6": singleton_max_abs <= 1e-6,
        "finite_predictions_and_peak_memory_below_7_5_gib": (
            predictions_finite and peak_memory < MEMORY_LIMIT_BYTES
        ),
    }
    passed = all(gate_checks.values())
    status = "PASS_Q2_SETADAPTER_ARCHITECTURE_SMOKE" if passed else "FAIL_Q2_SETADAPTER_ARCHITECTURE_SMOKE"
    report = {
        "schema": SCHEMA,
        "status": status,
        "scope": "PUBLIC_ARCHITECTURE_SMOKE_NOT_UTILITY_NOT_GENERATION_NOT_DP_NOT_CLINICAL",
        "frozen_inputs": {
            "protocol_sha256": PROTOCOL_SHA256,
            "manifest_sha256": calibration.K10_SHA256,
            "preprocessing_contract_sha256": calibration.PREPROCESSING_CONTRACT_SHA256,
            "confirmation_selection_sha256": CONFIRMATION_SELECTION_SHA256,
            "selection_salt": SELECTION_SALT,
            "model_id": calibration.MODEL_ID,
            "model_revision": calibration.MODEL_REVISION,
            "model_hashes": snapshot_hashes,
            "lora_init_seed": calibration.LORA_INIT_SEED,
            "lora_rank": calibration.LORA_RANK,
            "lora_targets": list(calibration.LORA_TARGETS),
            "probe_seed": PROBE_SEED,
            "probe_weight_std": PROBE_WEIGHT_STD,
        },
        "selection": {
            "patient_commitment": unit["selection_commitment"],
            "target_patient": unit["target_patient"],
            "record_count": len(unit["records"]),
            "image_commitments": [
                calibration.stable_hash("public-setadapter-image", record.image_id)
                for record in unit["records"]
            ],
            "private_selection_file": private_selection_path.name,
            "private_selection_sha256": calibration.sha256_file(private_selection_path),
        },
        "input": {
            **input_summary,
            "prediction_type": prediction_type,
            "timesteps": timesteps,
            "noise_seed_commitments": seed_commitments,
            "patient_loss_reduction": "mean of two per-record denoising losses before one backward",
        },
        "architecture": {
            "insertion": "wrap UNet mid block after base-mid output",
            "channels": channels,
            "token_dim": 128,
            "attention_heads": 4,
            "slot_position_embedding": False,
            "zero_initialized_output_projection": True,
            "valid_mask": True,
            "singleton_exact_identity_branch": True,
            "lora_trainable_parameters": EXPECTED_LORA_PARAMETERS,
            "set_adapter_trainable_parameters": EXPECTED_SET_PARAMETERS,
            "combined_trainable_parameters": EXPECTED_TOTAL_PARAMETERS,
            "trainable_dtypes": sorted({str(parameter.dtype) for _, parameter in trainable}),
        },
        "backward": {
            "loss": float(loss.detach().cpu()),
            "per_record_loss": [float(value) for value in per_record_loss.detach().cpu()],
            "lora": lora_gradients,
            "set_adapter": set_gradients,
            "zero_initialized_output_projection": output_projection_gradients,
            "combined_patient_gradient_l2_norm": combined_norm,
            "diagnostic_clip_norm": DIAGNOSTIC_CLIP_NORM,
            "diagnostic_clip_factor": clip_factor,
            "diagnostic_clipped_gradient_l2_norm": clipped_norm,
            "optimizer_created": False,
            "optimizer_step": False,
            "dp_noise": False,
            "accounting": False,
        },
        "functional_checks": {
            "zero_init_vs_bypass_max_abs": zero_init_max_abs,
            "temporary_probe_projection_sha256": probe_projection_digest,
            "activated_swap_permutation_max_abs": permutation_max_abs,
            "activated_other_record_influence_l2": other_record_influence_l2,
            "activated_singleton_vs_bypass_max_abs": singleton_max_abs,
            "all_predictions_finite": predictions_finite,
        },
        "execution": {
            "elapsed_model_seconds": elapsed,
            "peak_cuda_memory_bytes": peak_memory,
            "peak_cuda_memory_gib": peak_memory / 1024**3,
            "memory_limit_bytes": MEMORY_LIMIT_BYTES,
            "checkpoint_written": False,
            "latent_retained": False,
            "gradient_retained": False,
            "adapter_retained": False,
            "lora_unchanged": calibration.tensor_digest(lora_named) == lora_before,
            "set_adapter_initial_sha256": set_before,
        },
        "gate_checks": gate_checks,
        "interpretation_limit": (
            "A pass establishes only execution feasibility, invariances, cross-record connectivity, and a finite "
            "patient-vector backward. It is not evidence of generation or patient-DP utility improvement."
        ),
    }
    report_path = output_dir / "report.json"
    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": status,
                "loss": report["backward"]["loss"],
                "zero_init_vs_bypass_max_abs": zero_init_max_abs,
                "lora_gradient_l2_norm": lora_gradients["gradient_l2_norm"],
                "set_adapter_gradient_l2_norm": set_gradients["gradient_l2_norm"],
                "permutation_max_abs": permutation_max_abs,
                "other_record_influence_l2": other_record_influence_l2,
                "singleton_max_abs": singleton_max_abs,
                "peak_cuda_memory_gib": report["execution"]["peak_cuda_memory_gib"],
                "gate_checks": gate_checks,
                "report": str(report_path),
            },
            indent=2,
        )
    )

    del unet, set_adapter, mid_wrapper, trainable, lora_named, set_named
    del prediction_bypass, prediction_zero, activated, permuted, counterfactual_prediction
    del singleton_active, singleton_bypass, noisy, target, hidden_batch, latent_batch, noise_batch
    gc.collect()
    torch.cuda.empty_cache()
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
