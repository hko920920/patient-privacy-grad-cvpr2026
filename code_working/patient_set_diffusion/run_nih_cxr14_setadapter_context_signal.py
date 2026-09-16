#!/usr/bin/env python3
"""Train a public SetAdapter briefly and test correct versus shuffled context."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from set_adapter import PatientSetAdapter, PatientSetMidBlock


SCHEMA = "nih-cxr14-setadapter-context-signal/v1"
SELECTION_SALT = "nih-cxr14-setadapter-context-signal-v1"
PROTOCOL_SHA256 = "C832321AF319FDEDA06669291AC0B165F321F487087A2277DFE6D55E86584B50"
DINO_CONFIRMATION_SELECTION_SHA256 = "E1ABC33EA68082F69086CDFB752B558580710FAD35FB70640BDAD590DFC2C6F8"
CALIBRATION_SELECTION_SHA256 = "CB7E845F7B1D627BF52973ADBA9E7F3E6207F2081441DBEFBADE7E40AE261F6E"
UCAN_SELECTION_SHA256 = "13CB2054DE2A25208972590BA023585C16555D7ECE260D4FA571633CD9681AF0"
PREMISE_V1_SELECTION_SHA256 = "87B356A6175493E309DE2D84E8D0769B4764005BEE809AA8EB5CF5D5EF544AB0"
TRAIN_PATIENTS_PER_TARGET = 32
VALIDATION_PATIENTS_PER_TARGET = 16
TRAIN_STEPS = 192
TRAIN_EXPOSURES_PER_PATIENT = 6
EVALUATION_BANKS = 4
EVALUATION_PATIENT_CHUNK = 4
LEARNING_RATE = 1e-3
GRADIENT_CLIP_NORM = 1.0
EXPECTED_SET_PARAMETERS = 463_872
MEMORY_LIMIT_BYTES = int(7.5 * 1024**3)
RUNTIME_SEED = 2609033976201


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest().upper()


def hash_uint63(*parts: object) -> int:
    return int(stable_hash(*parts)[:16], 16) % (2**63)


def canonical_labels(record: Any) -> str:
    value = record.finding_labels
    if isinstance(value, str):
        return value
    return "|".join(str(item) for item in value)


def label_set(record: Any) -> set[str]:
    return {item for item in canonical_labels(record).split("|") if item}


def label_jaccard(left: Any, right: Any) -> float:
    first, second = label_set(left), label_set(right)
    union = first | second
    return 1.0 if not union else len(first & second) / len(union)


def default_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parent.parent
    return {
        "root": root,
        "manifest": root / "_data" / "derived" / "nih_cxr14_pa_target_enriched_v1" / "k10_private.csv",
        "image_root": root / "_data" / "raw" / "nih_cxr14_pa_k10_plus_census_v1" / "images",
        "preprocessing_contract": root / "_reports" / "nih_cxr14_sd21_ppmark_interface_v1_001" / "preprocessing_contract.json",
        "calibration_selection": root / "_reports" / "nih_cxr14_public_clip_calibration_v1_001" / "selected_public_units_private.csv",
        "ucan_selection": root / "_reports" / "nih_cxr14_ucan_public_diagnostic_v1_001" / "selected_public_patients_private.csv",
        "premise_v1_selection": root / "_reports" / "nih_cxr14_patient_set_premise_v1_001" / "selected_public_patient_sets_private.csv",
        "confirmation_v2_selection": root / "_reports" / "nih_cxr14_patient_set_premise_confirmation_v2_001" / "selected_public_patient_sets_private.csv",
        "protocol": root / "patient_set_diffusion" / "NIH_CXR14_SETADAPTER_CONTEXT_SIGNAL_PROTOCOL_V1.md",
        "output": root / "_reports" / "nih_cxr14_setadapter_context_signal_v1_001",
    }


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
        "protocol",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=paths[name])
    parser.add_argument("--output-dir", type=Path, default=paths["output"])
    return parser.parse_args()


def read_patient_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["patient_id"] for row in csv.DictReader(handle)}


def select_cohort(
    records: list[Any],
    excluded: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    by_patient: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        by_patient[str(record.patient_id)].append(record)
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    eligible_counts: dict[str, int] = {}
    for target in (0, 1):
        eligible = []
        for patient_id, patient_records in by_patient.items():
            if patient_id in excluded or len(patient_records) != 2:
                continue
            if int(patient_records[0].target_patient) != target:
                continue
            require(all(int(record.target_patient) == target for record in patient_records), "target changes")
            eligible.append(patient_id)
        ranked = sorted(
            eligible,
            key=lambda patient_id: (stable_hash(SELECTION_SALT, "patient", target, patient_id), patient_id),
        )
        required = TRAIN_PATIENTS_PER_TARGET + VALIDATION_PATIENTS_PER_TARGET
        require(len(ranked) >= required, f"only {len(ranked)} eligible target={target} patients")
        eligible_counts[str(target)] = len(ranked)
        for split, selected in (
            ("train", ranked[:TRAIN_PATIENTS_PER_TARGET]),
            ("validation", ranked[TRAIN_PATIENTS_PER_TARGET:required]),
        ):
            destination = train if split == "train" else validation
            for patient_id in selected:
                destination.append(
                    {
                        "split": split,
                        "patient_id": patient_id,
                        "target_patient": target,
                        "records": sorted(by_patient[patient_id], key=lambda record: record.image_id),
                    }
                )
    require(len(train) == 64 and len(validation) == 32, "cohort size mismatch")
    train_ids = {unit["patient_id"] for unit in train}
    validation_ids = {unit["patient_id"] for unit in validation}
    require(train_ids.isdisjoint(validation_ids), "train/validation overlap")
    require((train_ids | validation_ids).isdisjoint(excluded), "excluded patient selected")
    return train, validation, eligible_counts


def training_schedule(train_units: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_target = {
        target: [unit for unit in train_units if unit["target_patient"] == target]
        for target in (0, 1)
    }
    require(all(len(units) == TRAIN_PATIENTS_PER_TARGET for units in by_target.values()), "train balance")
    schedule = []
    for epoch in range(TRAIN_EXPOSURES_PER_PATIENT):
        ordered = {
            target: sorted(
                by_target[target],
                key=lambda unit: (
                    stable_hash(SELECTION_SALT, "epoch", epoch, target, unit["patient_id"]),
                    unit["patient_id"],
                ),
            )
            for target in (0, 1)
        }
        schedule.extend(zip(ordered[0], ordered[1]))
    require(len(schedule) == TRAIN_STEPS, "training schedule length mismatch")
    exposures = Counter(unit["patient_id"] for pair in schedule for unit in pair)
    require(set(exposures.values()) == {TRAIN_EXPOSURES_PER_PATIENT}, "unequal patient exposure")
    return schedule


def choose_shuffled_companion(
    unit: dict[str, Any],
    correct_companion: Any,
    validation_units: list[dict[str, Any]],
    bank: int,
) -> tuple[Any, bool, float]:
    candidates = [
        record
        for other in validation_units
        if other["target_patient"] == unit["target_patient"] and other["patient_id"] != unit["patient_id"]
        for record in other["records"]
    ]
    require(candidates, "no shuffled-companion candidates")
    exact = [record for record in candidates if canonical_labels(record) == canonical_labels(correct_companion)]
    if exact:
        pool = exact
        best_similarity = 1.0
        exact_match = True
    else:
        similarities = [label_jaccard(correct_companion, record) for record in candidates]
        best_similarity = max(similarities)
        pool = [record for record, value in zip(candidates, similarities) if value == best_similarity]
        exact_match = False
    chosen = min(
        pool,
        key=lambda record: (
            stable_hash(
                SELECTION_SALT,
                "shuffled",
                bank,
                unit["patient_id"],
                correct_companion.image_id,
                record.image_id,
            ),
            record.image_id,
        ),
    )
    require(str(chosen.patient_id) != unit["patient_id"], "shuffle retained same patient")
    return chosen, exact_match, best_similarity


def write_selection(path: Path, train: list[dict[str, Any]], validation: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "patient_id", "target_patient", "image_ids", "selection_commitment"],
            lineterminator="\n",
        )
        writer.writeheader()
        for unit in train + validation:
            image_ids = [record.image_id for record in unit["records"]]
            writer.writerow(
                {
                    "split": unit["split"],
                    "patient_id": unit["patient_id"],
                    "target_patient": unit["target_patient"],
                    "image_ids": "|".join(image_ids),
                    "selection_commitment": stable_hash(
                        SELECTION_SALT, unit["split"], unit["patient_id"], *image_ids
                    ),
                }
            )


def describe(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    require(array.size > 0 and np.all(np.isfinite(array)), "invalid summary values")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "minimum": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def sentinel_digest(unet: Any, tensor_digest: Any) -> str:
    wanted = {
        "conv_in.weight",
        "conv_out.weight",
        "mid_block.base_mid_block.resnets.0.conv1.weight",
    }
    selected = [(name, parameter) for name, parameter in unet.named_parameters() if name in wanted]
    require({name for name, _ in selected} == wanted, "base sentinel missing")
    return tensor_digest(selected)


def build_diffusion_batch(
    record_draws: list[tuple[Any, str]],
    latent_by_image: dict[str, Any],
    hidden_by_prompt: dict[str, Any],
    model_input: Any,
    scheduler: Any,
) -> tuple[Any, Any, Any, Any]:
    import torch

    latents = []
    hidden = []
    noises = []
    timesteps = []
    for record, draw_tag in record_draws:
        latent = latent_by_image[record.image_id].to(device="cuda", dtype=torch.float16)
        prompt = model_input.prompt_for_record(record)
        hidden.append(hidden_by_prompt[prompt].to(device="cuda", dtype=torch.float16))
        timestep = hash_uint63(SELECTION_SALT, "timestep", draw_tag, record.image_id) % int(
            scheduler.config.num_train_timesteps
        )
        seed = hash_uint63(SELECTION_SALT, "noise", draw_tag, record.image_id)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        noises.append(torch.randn(latent.shape, generator=generator, device="cuda", dtype=torch.float16))
        latents.append(latent)
        timesteps.append(int(timestep))
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
    return noisy, target, timestep_batch, hidden_batch


def evaluate(
    *,
    unet: Any,
    mid_wrapper: PatientSetMidBlock,
    validation_units: list[dict[str, Any]],
    latent_by_image: dict[str, Any],
    hidden_by_prompt: dict[str, Any],
    model_input: Any,
    scheduler: Any,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    correct_losses: list[float] = []
    shuffled_losses: list[float] = []
    bypass_losses: list[float] = []
    targets: list[int] = []
    exact_matches: list[bool] = []
    label_similarities: list[float] = []
    primary_input_max_abs = 0.0

    unet.eval()
    unet.disable_gradient_checkpointing()
    ordered_units = sorted(validation_units, key=lambda unit: (unit["target_patient"], unit["patient_id"]))
    with torch.inference_mode():
        for bank in range(EVALUATION_BANKS):
            rows = []
            for unit in ordered_units:
                primary_index = bank % 2
                primary = unit["records"][primary_index]
                companion = unit["records"][1 - primary_index]
                shuffled, exact_match, label_similarity = choose_shuffled_companion(
                    unit, companion, ordered_units, bank
                )
                rows.append((unit, primary, companion, shuffled, exact_match, label_similarity))
            for start in range(0, len(rows), EVALUATION_PATIENT_CHUNK):
                chunk = rows[start : start + EVALUATION_PATIENT_CHUNK]
                correct_draws = []
                shuffled_draws = []
                for unit, primary, companion, shuffled, _, _ in chunk:
                    primary_tag = f"eval:{bank}:primary:{unit['patient_id']}"
                    correct_draws.extend(
                        [(primary, primary_tag), (companion, f"eval:{bank}:companion:{companion.image_id}")]
                    )
                    shuffled_draws.extend(
                        [(primary, primary_tag), (shuffled, f"eval:{bank}:companion:{shuffled.image_id}")]
                    )
                correct_batch = build_diffusion_batch(
                    correct_draws, latent_by_image, hidden_by_prompt, model_input, scheduler
                )
                shuffled_batch = build_diffusion_batch(
                    shuffled_draws, latent_by_image, hidden_by_prompt, model_input, scheduler
                )
                noisy_correct, target_correct, timestep_correct, hidden_correct = correct_batch
                noisy_shuffled, target_shuffled, timestep_shuffled, hidden_shuffled = shuffled_batch
                primary_indices = torch.arange(0, 2 * len(chunk), 2, device="cuda")
                primary_input_max_abs = max(
                    primary_input_max_abs,
                    float(
                        torch.max(
                            torch.abs(
                                noisy_correct[primary_indices].float()
                                - noisy_shuffled[primary_indices].float()
                            )
                        ).cpu()
                    ),
                    float(
                        torch.max(
                            torch.abs(
                                target_correct[primary_indices].float()
                                - target_shuffled[primary_indices].float()
                            )
                        ).cpu()
                    ),
                )
                require(
                    bool(torch.equal(timestep_correct[primary_indices], timestep_shuffled[primary_indices])),
                    "primary timestep changed across arms",
                )
                valid_mask = torch.ones((len(chunk), 2), device="cuda", dtype=torch.bool)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    mid_wrapper.configure(valid_mask, bypass=False)
                    correct_prediction = unet(noisy_correct, timestep_correct, hidden_correct).sample
                    shuffled_prediction = unet(noisy_shuffled, timestep_shuffled, hidden_shuffled).sample
                    mid_wrapper.configure(valid_mask, bypass=True)
                    bypass_prediction = unet(noisy_correct, timestep_correct, hidden_correct).sample
                correct_cell = functional.mse_loss(
                    correct_prediction[primary_indices].float(),
                    target_correct[primary_indices].float(),
                    reduction="none",
                ).flatten(1).mean(1)
                shuffled_cell = functional.mse_loss(
                    shuffled_prediction[primary_indices].float(),
                    target_shuffled[primary_indices].float(),
                    reduction="none",
                ).flatten(1).mean(1)
                bypass_cell = functional.mse_loss(
                    bypass_prediction[primary_indices].float(),
                    target_correct[primary_indices].float(),
                    reduction="none",
                ).flatten(1).mean(1)
                require(
                    bool(
                        torch.isfinite(correct_cell).all()
                        and torch.isfinite(shuffled_cell).all()
                        and torch.isfinite(bypass_cell).all()
                    ),
                    "non-finite evaluation loss",
                )
                correct_losses.extend(float(value) for value in correct_cell.cpu())
                shuffled_losses.extend(float(value) for value in shuffled_cell.cpu())
                bypass_losses.extend(float(value) for value in bypass_cell.cpu())
                targets.extend(unit["target_patient"] for unit, *_ in chunk)
                exact_matches.extend(exact_match for *_, exact_match, _ in chunk)
                label_similarities.extend(label_similarity for *_, label_similarity in chunk)
                del correct_batch, shuffled_batch, correct_prediction, shuffled_prediction, bypass_prediction

    require(len(correct_losses) == 32 * EVALUATION_BANKS, "evaluation cell count mismatch")
    correct = np.asarray(correct_losses, dtype=np.float64)
    shuffled = np.asarray(shuffled_losses, dtype=np.float64)
    bypass = np.asarray(bypass_losses, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.int64)
    return {
        "cells": len(correct_losses),
        "correct": describe(correct_losses),
        "shuffled": describe(shuffled_losses),
        "bypass": describe(bypass_losses),
        "correct_over_shuffled": float(np.sum(correct) / np.sum(shuffled)),
        "correct_over_bypass": float(np.sum(correct) / np.sum(bypass)),
        "correct_shuffled_absolute_difference": describe((correct - shuffled).tolist()),
        "correct_bypass_absolute_difference": describe((correct - bypass).tolist()),
        "correct_wins_over_shuffled": int(np.sum(correct < shuffled)),
        "correct_win_fraction_over_shuffled": float(np.mean(correct < shuffled)),
        "ties_correct_shuffled": int(np.sum(correct == shuffled)),
        "target_ratios": {
            str(target): {
                "cells": int(np.sum(target_array == target)),
                "correct_over_shuffled": float(
                    np.sum(correct[target_array == target]) / np.sum(shuffled[target_array == target])
                ),
                "correct_over_bypass": float(
                    np.sum(correct[target_array == target]) / np.sum(bypass[target_array == target])
                ),
                "correct_win_fraction_over_shuffled": float(
                    np.mean(correct[target_array == target] < shuffled[target_array == target])
                ),
            }
            for target in (0, 1)
        },
        "pretraining_arm_max_absolute_loss_difference": float(
            max(np.max(np.abs(correct - shuffled)), np.max(np.abs(correct - bypass)))
        ),
        "primary_input_max_abs_across_correct_shuffled": primary_input_max_abs,
        "shuffled_exact_label_match_fraction": float(np.mean(exact_matches)),
        "shuffled_label_jaccard": describe(label_similarities),
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

    require(torch.cuda.is_available(), "CUDA is required for context-signal diagnostic")
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(RUNTIME_SEED)
    torch.cuda.manual_seed_all(RUNTIME_SEED)

    paths = {name: getattr(args, name).resolve() for name in vars(args) if name != "output_dir"}
    require(calibration.sha256_file(paths["protocol"]) == PROTOCOL_SHA256, "protocol hash mismatch")
    require(calibration.sha256_file(paths["manifest"]) == calibration.K10_SHA256, "manifest hash mismatch")
    require(
        calibration.sha256_file(paths["preprocessing_contract"])
        == calibration.PREPROCESSING_CONTRACT_SHA256,
        "preprocessing contract hash mismatch",
    )
    selection_sources = {
        "calibration": (paths["calibration_selection"], CALIBRATION_SELECTION_SHA256),
        "ucan": (paths["ucan_selection"], UCAN_SELECTION_SHA256),
        "premise_v1": (paths["premise_v1_selection"], PREMISE_V1_SELECTION_SHA256),
        "confirmation_v2": (paths["confirmation_v2_selection"], DINO_CONFIRMATION_SELECTION_SHA256),
    }
    excluded_parts = {}
    for name, (path, expected_hash) in selection_sources.items():
        require(calibration.sha256_file(path) == expected_hash, f"{name} selection hash mismatch")
        excluded_parts[name] = read_patient_ids(path)
    excluded = set().union(*excluded_parts.values())
    require(len(excluded) == 400, f"expected 400 unique exclusions, got {len(excluded)}")

    sys.path.insert(0, str(root / "data_pipeline"))
    import nih_cxr14_model_input as model_input

    records = model_input.read_manifest(paths["manifest"], partitions={"public_development"})
    train_units, validation_units, eligible_counts = select_cohort(records, excluded)
    schedule = training_schedule(train_units)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    selection_path = output_dir / "selected_public_context_cohort_private.csv"
    write_selection(selection_path, train_units, validation_units)

    snapshot = Path(
        snapshot_download(
            repo_id=calibration.MODEL_ID,
            revision=calibration.MODEL_REVISION,
            local_files_only=True,
        )
    ).resolve()
    require(snapshot.name == calibration.MODEL_REVISION, "snapshot revision mismatch")
    snapshot_hashes = calibration.verify_snapshot(snapshot)
    all_units = train_units + validation_units
    latent_by_image, hidden_by_prompt, input_summary = calibration.prepare_latents_and_text(
        snapshot, all_units, paths["image_root"], model_input
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
    set_adapter = PatientSetAdapter(channels=channels, token_dim=128, num_heads=4)
    mid_wrapper = PatientSetMidBlock(unet.mid_block, set_adapter)
    unet.mid_block = mid_wrapper
    unet.enable_gradient_checkpointing()
    unet.train().to("cuda")
    set_parameters = list(set_adapter.parameters())
    require(sum(parameter.numel() for parameter in set_parameters) == EXPECTED_SET_PARAMETERS, "parameter count")
    require({str(parameter.dtype) for parameter in set_parameters} == {"torch.float32"}, "adapter dtype")
    require(
        sum(parameter.requires_grad for name, parameter in unet.named_parameters() if ".set_adapter." not in name)
        == 0,
        "non-adapter trainable parameter",
    )
    base_before = sentinel_digest(unet, calibration.tensor_digest)
    set_before = calibration.tensor_digest(list(set_adapter.named_parameters()))

    pretraining_evaluation = evaluate(
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
        "zero-init pretraining arms differ",
    )

    optimizer = torch.optim.AdamW(
        set_parameters,
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    training_losses: list[float] = []
    gradient_norms: list[float] = []
    clipping_steps = 0
    for step, patient_pair in enumerate(schedule):
        draws = []
        for unit in patient_pair:
            records_for_step = list(unit["records"])
            if hash_uint63(SELECTION_SALT, "slot_swap", step, unit["patient_id"]) % 2:
                records_for_step.reverse()
            draws.extend(
                (record, f"train:{step}:{unit['patient_id']}:{slot}")
                for slot, record in enumerate(records_for_step)
            )
        noisy, target, timesteps, hidden = build_diffusion_batch(
            draws, latent_by_image, hidden_by_prompt, model_input, scheduler
        )
        optimizer.zero_grad(set_to_none=True)
        unet.train()
        unet.enable_gradient_checkpointing()
        mid_wrapper.configure(torch.ones((2, 2), device="cuda", dtype=torch.bool), bypass=False)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = unet(noisy, timesteps, hidden).sample
            loss = functional.mse_loss(prediction.float(), target.float(), reduction="mean")
        require(bool(torch.isfinite(loss)), f"non-finite training loss at step {step}")
        loss.backward()
        require(
            all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in set_parameters),
            f"invalid SetAdapter gradient at step {step}",
        )
        gradient_norm = torch.nn.utils.clip_grad_norm_(set_parameters, max_norm=GRADIENT_CLIP_NORM)
        gradient_norm_value = float(gradient_norm.detach().cpu())
        require(math.isfinite(gradient_norm_value), f"non-finite gradient norm at step {step}")
        clipping_steps += gradient_norm_value > GRADIENT_CLIP_NORM
        optimizer.step()
        require(all(bool(torch.isfinite(parameter).all()) for parameter in set_parameters), "non-finite adapter")
        training_losses.append(float(loss.detach().cpu()))
        gradient_norms.append(gradient_norm_value)
        del noisy, target, timesteps, hidden, prediction, loss

    set_after = calibration.tensor_digest(list(set_adapter.named_parameters()))
    base_after = sentinel_digest(unet, calibration.tensor_digest)
    require(set_after != set_before, "SetAdapter did not change")
    require(base_after == base_before, "frozen base sentinel changed")

    posttraining_evaluation = evaluate(
        unet=unet,
        mid_wrapper=mid_wrapper,
        validation_units=validation_units,
        latent_by_image=latent_by_image,
        hidden_by_prompt=hidden_by_prompt,
        model_input=model_input,
        scheduler=scheduler,
    )

    # Pure post-training invariance check independent of the UNet wrapper.
    set_adapter.eval()
    generator = torch.Generator(device="cuda").manual_seed(RUNTIME_SEED + 1)
    pure_hidden = torch.randn((4, channels, 2, 2), generator=generator, device="cuda", dtype=torch.float32)
    pure_mask = torch.ones((2, 2), device="cuda", dtype=torch.bool)
    swap = torch.tensor([1, 0], device="cuda", dtype=torch.long)
    with torch.inference_mode():
        pure_output = set_adapter(pure_hidden, pure_mask).reshape(2, 2, channels, 2, 2)
        pure_permuted_hidden = pure_hidden.reshape(2, 2, channels, 2, 2)[:, swap].reshape(4, channels, 2, 2)
        pure_permuted = set_adapter(pure_permuted_hidden, pure_mask[:, swap]).reshape(2, 2, channels, 2, 2)
        pure_restored = pure_permuted[:, swap]
        permutation_error = float(torch.max(torch.abs(pure_output - pure_restored)).cpu())
        singleton_hidden = pure_hidden[:2]
        singleton_output = set_adapter(
            singleton_hidden, torch.ones((2, 1), device="cuda", dtype=torch.bool)
        )
        singleton_error = float(torch.max(torch.abs(singleton_output - singleton_hidden)).cpu())

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_memory = int(torch.cuda.max_memory_allocated())
    target_ratios = posttraining_evaluation["target_ratios"]
    gate_checks = {
        "all_training_and_evaluation_finite_adapter_changed_base_unchanged": (
            len(training_losses) == TRAIN_STEPS
            and all(math.isfinite(value) for value in training_losses + gradient_norms)
            and set_after != set_before
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
        "peak_memory_below_7_5_gib": peak_memory < MEMORY_LIMIT_BYTES,
    }
    passed = all(gate_checks.values())
    status = "PASS_SETADAPTER_HELDOUT_CONTEXT_SIGNAL" if passed else "FAIL_SETADAPTER_HELDOUT_CONTEXT_SIGNAL"
    report = {
        "schema": SCHEMA,
        "status": status,
        "scope": "PUBLIC_NONDP_CONTEXT_SIGNAL_NOT_GENERATION_NOT_PRIVACY_NOT_CLINICAL",
        "frozen_inputs": {
            "protocol_sha256": PROTOCOL_SHA256,
            "manifest_sha256": calibration.K10_SHA256,
            "preprocessing_contract_sha256": calibration.PREPROCESSING_CONTRACT_SHA256,
            "selection_salt": SELECTION_SALT,
            "exclusion_selection_hashes": {
                name: expected_hash for name, (_, expected_hash) in selection_sources.items()
            },
            "excluded_unique_patients": len(excluded),
            "model_id": calibration.MODEL_ID,
            "model_revision": calibration.MODEL_REVISION,
            "model_hashes": snapshot_hashes,
            "runtime_seed": RUNTIME_SEED,
        },
        "selection": {
            "eligible_exact_q2_by_target_after_exclusion": eligible_counts,
            "train_patients": len(train_units),
            "train_images": 2 * len(train_units),
            "validation_patients": len(validation_units),
            "validation_images": 2 * len(validation_units),
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
            "training_timesteps": "independent hash-uniform per record and step",
            "training_corruption_noise": "independent Gaussian per record and step",
        },
        "model": {
            "base_trainable_parameters": 0,
            "lora_present": False,
            "set_adapter_trainable_parameters": EXPECTED_SET_PARAMETERS,
            "channels": channels,
            "token_dim": 128,
            "attention_heads": 4,
            "base_sentinel_initial_sha256": base_before,
            "base_sentinel_final_sha256": base_after,
            "set_adapter_initial_sha256": set_before,
            "set_adapter_final_sha256": set_after,
        },
        "optimization": {
            "optimizer": "AdamW",
            "steps": TRAIN_STEPS,
            "patients_per_step": 2,
            "records_per_patient": 2,
            "patient_exposures": TRAIN_EXPOSURES_PER_PATIENT,
            "learning_rate": LEARNING_RATE,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0.01,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "clipping_steps": clipping_steps,
            "training_loss": describe(training_losses),
            "first_32_step_loss": describe(training_losses[:32]),
            "last_32_step_loss": describe(training_losses[-32:]),
            "preclip_gradient_norm": describe(gradient_norms),
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
            "memory_limit_bytes": MEMORY_LIMIT_BYTES,
            "optimizer_created": True,
            "optimizer_steps": TRAIN_STEPS,
            "dp_noise": False,
            "accounting": False,
            "checkpoint_written": False,
            "latent_retained": False,
            "gradient_retained": False,
            "optimizer_state_retained": False,
            "adapter_retained": False,
        },
        "interpretation_limit": (
            "A pass supports only a held-out denoising context signal and licenses a matched public generation "
            "experiment. It is not image-quality, set-coherence, patient-DP, privacy, or clinical evidence."
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
                "pretraining_max_arm_loss_difference": pretraining_evaluation[
                    "pretraining_arm_max_absolute_loss_difference"
                ],
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

    del optimizer, unet, set_adapter, mid_wrapper, latent_by_image, hidden_by_prompt
    gc.collect()
    torch.cuda.empty_cache()
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
