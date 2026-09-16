#!/usr/bin/env python3
"""Confirm a strict leave-one-record-out adapter on fresh held-out patients."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cross_record_adapter import CrossRecordMidBlock, CrossRecordOnlyAdapter
import run_nih_cxr14_setadapter_context_signal as v1


SCHEMA = "nih-cxr14-cross-record-context-confirmation/v2"
SELECTION_SALT = "nih-cxr14-cross-record-context-confirmation-v2"
PROTOCOL_SHA256 = "7EF39EB5F88220971EB9F5BAB99ADFB28C6CAB208ACDDAEA175D69B7E7491943"
CONTEXT_V1_SELECTION_SHA256 = "B6951897C7270CC7A462496F82A5EC1CDDC65D7F37A15EA6E2D629809604E319"
VALIDATION_PATIENTS_PER_TARGET = 16
EXPECTED_CROSS_PARAMETERS = 447_360
RUNTIME_SEED = 2609033976202


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def default_paths() -> dict[str, Path]:
    paths = v1.default_paths()
    root = paths["root"]
    paths.update(
        {
            "context_v1_selection": root
            / "_reports"
            / "nih_cxr14_setadapter_context_signal_v1_001"
            / "selected_public_context_cohort_private.csv",
            "protocol": root
            / "patient_set_diffusion"
            / "NIH_CXR14_CROSS_RECORD_CONTEXT_CONFIRMATION_PROTOCOL_V2.md",
            "output": root / "_reports" / "nih_cxr14_cross_record_context_confirmation_v2_001",
        }
    )
    return paths


def parse_args() -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser()
    for name in (
        "manifest",
        "image_root",
        "preprocessing_contract",
        "calibration_selection",
        "ucan_selection",
        "premise_v1_selection",
        "confirmation_v2_selection",
        "context_v1_selection",
        "protocol",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=paths[name])
    parser.add_argument("--output-dir", type=Path, default=paths["output"])
    return parser.parse_args()


def load_reused_train_units(selection_path: Path, records: list[Any]) -> list[dict[str, Any]]:
    record_by_id = {record.image_id: record for record in records}
    with selection_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "train"]
    require(len(rows) == 64, f"expected 64 reused train patients, got {len(rows)}")
    units = []
    for row in rows:
        image_ids = row["image_ids"].split("|")
        require(len(image_ids) == 2 and all(image_id in record_by_id for image_id in image_ids), "train images")
        chosen = [record_by_id[image_id] for image_id in image_ids]
        require(all(str(record.patient_id) == row["patient_id"] for record in chosen), "train patient")
        require(all(int(record.target_patient) == int(row["target_patient"]) for record in chosen), "train target")
        units.append(
            {
                "split": "train_reused_v1",
                "patient_id": row["patient_id"],
                "target_patient": int(row["target_patient"]),
                "records": chosen,
            }
        )
    require(Counter(unit["target_patient"] for unit in units) == {0: 32, 1: 32}, "train balance")
    return units


def select_fresh_validation(
    records: list[Any], excluded: set[str]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_patient: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        by_patient[str(record.patient_id)].append(record)
    units = []
    eligible_counts = {}
    for target in (0, 1):
        eligible = [
            patient_id
            for patient_id, patient_records in by_patient.items()
            if patient_id not in excluded
            and len(patient_records) == 2
            and int(patient_records[0].target_patient) == target
        ]
        ranked = sorted(
            eligible,
            key=lambda patient_id: (v1.stable_hash(SELECTION_SALT, "validation", target, patient_id), patient_id),
        )
        require(len(ranked) >= VALIDATION_PATIENTS_PER_TARGET, f"only {len(ranked)} fresh target={target}")
        eligible_counts[str(target)] = len(ranked)
        for patient_id in ranked[:VALIDATION_PATIENTS_PER_TARGET]:
            patient_records = sorted(by_patient[patient_id], key=lambda record: record.image_id)
            require(all(int(record.target_patient) == target for record in patient_records), "validation target")
            units.append(
                {
                    "split": "validation_fresh_v2",
                    "patient_id": patient_id,
                    "target_patient": target,
                    "records": patient_records,
                }
            )
    require(len(units) == 32, "fresh validation size")
    require({unit["patient_id"] for unit in units}.isdisjoint(excluded), "fresh validation overlap")
    return units, eligible_counts


def write_selection(
    path: Path, train_units: list[dict[str, Any]], validation_units: list[dict[str, Any]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "patient_id", "target_patient", "image_ids", "selection_commitment"],
            lineterminator="\n",
        )
        writer.writeheader()
        for unit in train_units + validation_units:
            image_ids = [record.image_id for record in unit["records"]]
            writer.writerow(
                {
                    "split": unit["split"],
                    "patient_id": unit["patient_id"],
                    "target_patient": unit["target_patient"],
                    "image_ids": "|".join(image_ids),
                    "selection_commitment": v1.stable_hash(
                        SELECTION_SALT, unit["split"], unit["patient_id"], *image_ids
                    ),
                }
            )


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

    require(torch.cuda.is_available(), "CUDA required")
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(RUNTIME_SEED)
    torch.cuda.manual_seed_all(RUNTIME_SEED)

    paths = {name: getattr(args, name).resolve() for name in vars(args) if name != "output_dir"}
    require(calibration.sha256_file(paths["protocol"]) == PROTOCOL_SHA256, "protocol hash")
    require(calibration.sha256_file(paths["manifest"]) == calibration.K10_SHA256, "manifest hash")
    require(
        calibration.sha256_file(paths["preprocessing_contract"])
        == calibration.PREPROCESSING_CONTRACT_SHA256,
        "preprocessing hash",
    )
    selection_hashes = {
        "calibration_selection": v1.CALIBRATION_SELECTION_SHA256,
        "ucan_selection": v1.UCAN_SELECTION_SHA256,
        "premise_v1_selection": v1.PREMISE_V1_SELECTION_SHA256,
        "confirmation_v2_selection": v1.DINO_CONFIRMATION_SELECTION_SHA256,
        "context_v1_selection": CONTEXT_V1_SELECTION_SHA256,
    }
    excluded_parts = {}
    for name, expected_hash in selection_hashes.items():
        require(calibration.sha256_file(paths[name]) == expected_hash, f"{name} hash")
        excluded_parts[name] = v1.read_patient_ids(paths[name])
    excluded = set().union(*excluded_parts.values())
    require(len(excluded) == 496, f"expected 496 exclusions, got {len(excluded)}")

    sys.path.insert(0, str(root / "data_pipeline"))
    import nih_cxr14_model_input as model_input

    records = model_input.read_manifest(paths["manifest"], partitions={"public_development"})
    train_units = load_reused_train_units(paths["context_v1_selection"], records)
    validation_units, eligible_counts = select_fresh_validation(records, excluded)
    require(
        {unit["patient_id"] for unit in train_units}.isdisjoint(
            {unit["patient_id"] for unit in validation_units}
        ),
        "train/validation overlap",
    )
    schedule = v1.training_schedule(train_units)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    selection_path = output_dir / "selected_public_cross_record_cohort_private.csv"
    write_selection(selection_path, train_units, validation_units)

    snapshot = Path(
        snapshot_download(
            repo_id=calibration.MODEL_ID,
            revision=calibration.MODEL_REVISION,
            local_files_only=True,
        )
    ).resolve()
    require(snapshot.name == calibration.MODEL_REVISION, "model revision")
    snapshot_hashes = calibration.verify_snapshot(snapshot)
    latent_by_image, hidden_by_prompt, input_summary = calibration.prepare_latents_and_text(
        snapshot, train_units + validation_units, paths["image_root"], model_input
    )
    scheduler = DDPMScheduler.from_pretrained(snapshot / "scheduler", local_files_only=True)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    unet = UNet2DConditionModel.from_pretrained(
        snapshot / "unet",
        torch_dtype=torch.float16,
        variant=calibration.MODEL_VARIANT,
        use_safetensors=True,
        local_files_only=True,
    )
    unet.requires_grad_(False)
    channels = int(unet.config.block_out_channels[-1])
    cross_adapter = CrossRecordOnlyAdapter(channels=channels, token_dim=128, num_heads=4)
    mid_wrapper = CrossRecordMidBlock(unet.mid_block, cross_adapter)
    unet.mid_block = mid_wrapper
    unet.enable_gradient_checkpointing()
    unet.train().to("cuda")
    parameters = list(cross_adapter.parameters())
    require(sum(parameter.numel() for parameter in parameters) == EXPECTED_CROSS_PARAMETERS, "adapter count")
    require({str(parameter.dtype) for parameter in parameters} == {"torch.float32"}, "adapter dtype")
    require(
        sum(parameter.requires_grad for name, parameter in unet.named_parameters() if ".cross_adapter." not in name)
        == 0,
        "non-adapter trainable parameter",
    )
    base_before = v1.sentinel_digest(unet, calibration.tensor_digest)
    adapter_before = calibration.tensor_digest(list(cross_adapter.named_parameters()))

    pretraining_evaluation = v1.evaluate(
        unet=unet,
        mid_wrapper=mid_wrapper,
        validation_units=validation_units,
        latent_by_image=latent_by_image,
        hidden_by_prompt=hidden_by_prompt,
        model_input=model_input,
        scheduler=scheduler,
    )
    require(
        pretraining_evaluation["pretraining_arm_max_absolute_loss_difference"] <= 1e-7,
        "zero-init arms differ",
    )

    optimizer = torch.optim.AdamW(
        parameters,
        lr=v1.LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    training_losses = []
    gradient_norms = []
    clipping_steps = 0
    for step, patient_pair in enumerate(schedule):
        draws = []
        for unit in patient_pair:
            records_for_step = list(unit["records"])
            if v1.hash_uint63(v1.SELECTION_SALT, "slot_swap", step, unit["patient_id"]) % 2:
                records_for_step.reverse()
            draws.extend(
                (record, f"train:{step}:{unit['patient_id']}:{slot}")
                for slot, record in enumerate(records_for_step)
            )
        noisy, target, timesteps, hidden = v1.build_diffusion_batch(
            draws, latent_by_image, hidden_by_prompt, model_input, scheduler
        )
        optimizer.zero_grad(set_to_none=True)
        unet.train()
        unet.enable_gradient_checkpointing()
        mid_wrapper.configure(torch.ones((2, 2), device="cuda", dtype=torch.bool), bypass=False)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = unet(noisy, timesteps, hidden).sample
            loss = functional.mse_loss(prediction.float(), target.float(), reduction="mean")
        require(bool(torch.isfinite(loss)), f"non-finite loss step {step}")
        loss.backward()
        require(
            all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in parameters),
            f"invalid gradient step {step}",
        )
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=v1.GRADIENT_CLIP_NORM)
        norm_value = float(norm.detach().cpu())
        require(math.isfinite(norm_value), f"non-finite norm step {step}")
        clipping_steps += norm_value > v1.GRADIENT_CLIP_NORM
        optimizer.step()
        require(all(bool(torch.isfinite(parameter).all()) for parameter in parameters), "non-finite adapter")
        training_losses.append(float(loss.detach().cpu()))
        gradient_norms.append(norm_value)
        del noisy, target, timesteps, hidden, prediction, loss

    adapter_after = calibration.tensor_digest(list(cross_adapter.named_parameters()))
    base_after = v1.sentinel_digest(unet, calibration.tensor_digest)
    require(adapter_after != adapter_before, "adapter unchanged")
    require(base_after == base_before, "base changed")
    posttraining_evaluation = v1.evaluate(
        unet=unet,
        mid_wrapper=mid_wrapper,
        validation_units=validation_units,
        latent_by_image=latent_by_image,
        hidden_by_prompt=hidden_by_prompt,
        model_input=model_input,
        scheduler=scheduler,
    )

    cross_adapter.eval()
    generator = torch.Generator(device="cuda").manual_seed(RUNTIME_SEED + 1)
    pure_hidden = torch.randn((4, channels, 2, 2), generator=generator, device="cuda", dtype=torch.float32)
    pure_mask = torch.ones((2, 2), device="cuda", dtype=torch.bool)
    swap = torch.tensor([1, 0], device="cuda", dtype=torch.long)
    with torch.inference_mode():
        pure_output = cross_adapter(pure_hidden, pure_mask).reshape(2, 2, channels, 2, 2)
        permuted_hidden = pure_hidden.reshape(2, 2, channels, 2, 2)[:, swap].reshape(4, channels, 2, 2)
        permuted_output = cross_adapter(permuted_hidden, pure_mask[:, swap]).reshape(2, 2, channels, 2, 2)
        permutation_error = float(torch.max(torch.abs(pure_output - permuted_output[:, swap])).cpu())
        singleton_hidden = pure_hidden[:2]
        singleton_output = cross_adapter(
            singleton_hidden, torch.ones((2, 1), device="cuda", dtype=torch.bool)
        )
        singleton_error = float(torch.max(torch.abs(singleton_output - singleton_hidden)).cpu())

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_memory = int(torch.cuda.max_memory_allocated())
    target_ratios = posttraining_evaluation["target_ratios"]
    gate_checks = {
        "finite_192_steps_adapter_changed_base_unchanged": (
            len(training_losses) == v1.TRAIN_STEPS
            and all(math.isfinite(value) for value in training_losses + gradient_norms)
            and adapter_after != adapter_before
            and base_after == base_before
        ),
        "pretraining_zero_init_arms_identical_at_most_1e_7": (
            pretraining_evaluation["pretraining_arm_max_absolute_loss_difference"] <= 1e-7
        ),
        "post_correct_over_shuffled_at_most_0_995_and_win_fraction_at_least_0_60": (
            posttraining_evaluation["correct_over_shuffled"] <= 0.995
            and posttraining_evaluation["correct_win_fraction_over_shuffled"] >= 0.60
        ),
        "post_correct_over_shuffled_below_1_in_both_targets": all(
            target_ratios[str(target)]["correct_over_shuffled"] < 1.0 for target in (0, 1)
        ),
        "post_correct_over_bypass_at_most_0_995": (
            posttraining_evaluation["correct_over_bypass"] <= 0.995
        ),
        "posttraining_permutation_and_singleton_errors_at_most_1e_5": (
            permutation_error <= 1e-5 and singleton_error <= 1e-5
        ),
        "peak_memory_below_7_5_gib": peak_memory < v1.MEMORY_LIMIT_BYTES,
    }
    passed = all(gate_checks.values())
    status = (
        "PASS_CROSS_RECORD_CONTEXT_CONFIRMATION"
        if passed
        else "FAIL_CROSS_RECORD_CONTEXT_CONFIRMATION_CLOSE_STANDARD_IID_CONTEXT_CLASS"
    )
    report = {
        "schema": SCHEMA,
        "status": status,
        "scope": "PUBLIC_NONDP_FRESH_HELDOUT_CONTEXT_CONFIRMATION_NOT_GENERATION_NOT_PRIVACY_NOT_CLINICAL",
        "frozen_inputs": {
            "protocol_sha256": PROTOCOL_SHA256,
            "manifest_sha256": calibration.K10_SHA256,
            "preprocessing_contract_sha256": calibration.PREPROCESSING_CONTRACT_SHA256,
            "context_v1_selection_sha256": CONTEXT_V1_SELECTION_SHA256,
            "selection_salt": SELECTION_SALT,
            "training_draw_salt_reused_from_v1": v1.SELECTION_SALT,
            "excluded_unique_patients_for_validation": len(excluded),
            "selection_hashes": selection_hashes,
            "model_id": calibration.MODEL_ID,
            "model_revision": calibration.MODEL_REVISION,
            "model_hashes": snapshot_hashes,
            "runtime_seed": RUNTIME_SEED,
        },
        "selection": {
            "reused_train_patients": len(train_units),
            "reused_train_images": 2 * len(train_units),
            "fresh_validation_patients": len(validation_units),
            "fresh_validation_images": 2 * len(validation_units),
            "fresh_eligible_exact_q2_by_target": eligible_counts,
            "train_target_counts": dict(Counter(str(unit["target_patient"]) for unit in train_units)),
            "validation_target_counts": dict(
                Counter(str(unit["target_patient"]) for unit in validation_units)
            ),
            "selection_file": selection_path.name,
            "selection_sha256": calibration.sha256_file(selection_path),
        },
        "input": {
            **input_summary,
            "patient_records": 2,
            "training_timesteps_noise": "exact v1 schedule and independent hash draws",
        },
        "model": {
            "mechanism_change_from_v1": (
                "remove diagonal/self value attention and own-token residual; delta uses other-record values only"
            ),
            "standalone_novelty_claim": False,
            "base_trainable_parameters": 0,
            "lora_present": False,
            "cross_adapter_trainable_parameters": EXPECTED_CROSS_PARAMETERS,
            "channels": channels,
            "token_dim": 128,
            "attention_heads": 4,
            "base_sentinel_initial_sha256": base_before,
            "base_sentinel_final_sha256": base_after,
            "cross_adapter_initial_sha256": adapter_before,
            "cross_adapter_final_sha256": adapter_after,
        },
        "optimization": {
            "optimizer": "AdamW",
            "steps": v1.TRAIN_STEPS,
            "patients_per_step": 2,
            "records_per_patient": 2,
            "patient_exposures": v1.TRAIN_EXPOSURES_PER_PATIENT,
            "learning_rate": v1.LEARNING_RATE,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0.01,
            "gradient_clip_norm": v1.GRADIENT_CLIP_NORM,
            "clipping_steps": clipping_steps,
            "training_loss": v1.describe(training_losses),
            "first_32_step_loss": v1.describe(training_losses[:32]),
            "last_32_step_loss": v1.describe(training_losses[-32:]),
            "preclip_gradient_norm": v1.describe(gradient_norms),
        },
        "pretraining_evaluation": pretraining_evaluation,
        "posttraining_evaluation": posttraining_evaluation,
        "posttraining_invariance": {
            "permutation_max_abs": permutation_error,
            "singleton_identity_max_abs": singleton_error,
        },
        "gate_checks": gate_checks,
        "execution": {
            "elapsed_model_training_evaluation_seconds": elapsed,
            "peak_cuda_memory_bytes": peak_memory,
            "peak_cuda_memory_gib": peak_memory / 1024**3,
            "memory_limit_bytes": v1.MEMORY_LIMIT_BYTES,
            "optimizer_created": True,
            "optimizer_steps": v1.TRAIN_STEPS,
            "dp_noise": False,
            "accounting": False,
            "checkpoint_written": False,
            "latent_retained": False,
            "gradient_retained": False,
            "optimizer_state_retained": False,
            "adapter_retained": False,
        },
        "interpretation_limit": (
            "Pass licenses only a matched public generation experiment. Failure closes the current standard-IID "
            "pooled mid-block context class under the frozen stop rule."
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
                "post_correct_over_shuffled": posttraining_evaluation["correct_over_shuffled"],
                "post_correct_win_fraction": posttraining_evaluation[
                    "correct_win_fraction_over_shuffled"
                ],
                "post_correct_over_bypass": posttraining_evaluation["correct_over_bypass"],
                "target_correct_over_shuffled": {
                    key: value["correct_over_shuffled"] for key, value in target_ratios.items()
                },
                "training_loss_first32_mean": report["optimization"]["first_32_step_loss"]["mean"],
                "training_loss_last32_mean": report["optimization"]["last_32_step_loss"]["mean"],
                "permutation_max_abs": permutation_error,
                "singleton_max_abs": singleton_error,
                "peak_cuda_memory_gib": report["execution"]["peak_cuda_memory_gib"],
                "elapsed_seconds": elapsed,
                "gate_checks": gate_checks,
                "report": str(report_path),
            },
            indent=2,
        )
    )

    del optimizer, unet, cross_adapter, mid_wrapper, latent_by_image, hidden_by_prompt
    gc.collect()
    torch.cuda.empty_cache()
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
