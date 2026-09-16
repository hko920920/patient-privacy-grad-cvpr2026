#!/usr/bin/env python3
"""Run the frozen NIH CXR14 UCAN public full-gradient diagnostic.

This script creates no optimizer, takes no optimizer step, adds no DP noise,
and reads only the K10 public-development partition.  Full gradients exist in
RAM only long enough to compute Gram-derived error cells and are never saved.
"""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from coupling import METHODS, paired_plan, stable_hash


SCHEMA = "nih-cxr14-ucan-public-diagnostic/v1"
SELECTION_SALT = "nih-cxr14-ucan-public-diagnostic-v1"
PROTOCOL_SHA256 = "C3AEC580ADF99D026B5210BD0395875DB4C4D905E963806FE58BE7CE3498D45B"
CALIBRATION_SELECTION_SHA256 = "CB7E845F7B1D627BF52973ADBA9E7F3E6207F2081441DBEFBADE7E40AE261F6E"
PATIENT_CLIP_NORM = 0.1997973088974048
PATIENTS_PER_TARGET = 8
RECORDS_PER_PATIENT = 4
BANKS = tuple(f"bank_{index:02d}" for index in range(1, 6))
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 26090371
PRIMARY_BASELINE = "iid_pair_shared_t"
PRIMARY_TREATMENT = "antithetic_pair_shared_t"
ORDINARY_BASELINE = "iid_independent_t"


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
        "calibration_selection": root / "_reports" / "nih_cxr14_public_clip_calibration_v1_001" / "selected_public_units_private.csv",
        "protocol": root / "unit_coupled_noise" / "NIH_CXR14_UCAN_PUBLIC_DIAGNOSTIC_PROTOCOL_V1.md",
        "output": root / "_reports" / "nih_cxr14_ucan_public_diagnostic_v1_001",
    }


def parse_args() -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=paths["manifest"])
    parser.add_argument("--image-root", type=Path, default=paths["image_root"])
    parser.add_argument("--preprocessing-contract", type=Path, default=paths["preprocessing_contract"])
    parser.add_argument("--calibration-selection", type=Path, default=paths["calibration_selection"])
    parser.add_argument("--protocol", type=Path, default=paths["protocol"])
    parser.add_argument("--output-dir", type=Path, default=paths["output"])
    return parser.parse_args()


def read_excluded_patients(path: Path, calibration: Any) -> set[str]:
    require(calibration.sha256_file(path) == CALIBRATION_SELECTION_SHA256, "clip-calibration selection hash mismatch")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 144, "unexpected clip-calibration selection row count")
    return {row["patient_id"] for row in rows}


def select_units(records: list[Any], excluded: set[str]) -> list[dict[str, Any]]:
    by_patient: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        by_patient[str(record.patient_id)].append(record)

    eligible: dict[int, list[str]] = {0: [], 1: []}
    for patient_id, patient_records in by_patient.items():
        if patient_id in excluded or len(patient_records) < RECORDS_PER_PATIENT:
            continue
        target = int(patient_records[0].target_patient)
        require(all(int(record.target_patient) == target for record in patient_records), "target flag changes within patient")
        eligible[target].append(patient_id)

    units: list[dict[str, Any]] = []
    for target in (0, 1):
        ranked = sorted(
            eligible[target],
            key=lambda patient_id: (stable_hash(SELECTION_SALT, "patient", target, patient_id), patient_id),
        )
        require(len(ranked) >= PATIENTS_PER_TARGET, f"insufficient eligible target={target} patients")
        for patient_id in ranked[:PATIENTS_PER_TARGET]:
            patient_records = sorted(
                by_patient[patient_id],
                key=lambda record: (
                    stable_hash(SELECTION_SALT, "record", patient_id, record.image_id),
                    record.image_id,
                ),
            )
            chosen = patient_records[:RECORDS_PER_PATIENT]
            require(len(chosen) == RECORDS_PER_PATIENT, "selected patient lacks four records")
            units.append(
                {
                    "unit_type": "patient",
                    "unit_id": patient_id,
                    "patient_id": patient_id,
                    "target_patient": target,
                    "contribution_bucket": "n4_10",
                    "eligible_image_count": len(patient_records),
                    "records": chosen,
                }
            )
    require(len(units) == 16, "UCAN cohort size mismatch")
    require(len({unit["patient_id"] for unit in units}) == 16, "UCAN patient IDs are not unique")
    require({unit["patient_id"] for unit in units}.isdisjoint(excluded), "UCAN cohort overlaps calibration")
    return units


def write_selection(path: Path, units: list[dict[str, Any]]) -> None:
    fields = [
        "patient_id", "target_patient", "eligible_image_count", "selected_image_count",
        "image_ids", "selection_commitment",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for unit in units:
            image_ids = [record.image_id for record in unit["records"]]
            writer.writerow(
                {
                    "patient_id": unit["patient_id"],
                    "target_patient": unit["target_patient"],
                    "eligible_image_count": unit["eligible_image_count"],
                    "selected_image_count": len(image_ids),
                    "image_ids": "|".join(image_ids),
                    "selection_commitment": stable_hash(SELECTION_SALT, unit["patient_id"], *image_ids),
                }
            )


def gradient_for_plan(
    *,
    unit: dict[str, Any],
    bank: str,
    method: str,
    latent_by_image: dict[str, Any],
    hidden_by_prompt: dict[str, Any],
    model_input: Any,
    scheduler: Any,
    unet: Any,
    trainable: list[tuple[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as functional

    plan = paired_plan(
        salt=SELECTION_SALT,
        bank=bank,
        patient_id=unit["patient_id"],
        method=method,
        record_count=len(unit["records"]),
        num_train_timesteps=int(scheduler.config.num_train_timesteps),
    )
    latents = []
    hidden = []
    noises = []
    noise_cache: dict[int, Any] = {}
    for record, item in zip(unit["records"], plan):
        latent = latent_by_image[record.image_id].to(device="cuda", dtype=torch.float16)
        prompt = model_input.prompt_for_record(record)
        text = hidden_by_prompt[prompt].to(device="cuda", dtype=torch.float16)
        seed = int(item["noise_seed"])
        if seed not in noise_cache:
            generator = torch.Generator(device="cuda").manual_seed(seed)
            noise_cache[seed] = torch.randn(
                latent.shape, generator=generator, device="cuda", dtype=torch.float16
            )
        noise = noise_cache[seed]
        if int(item["noise_sign"]) == -1:
            noise = -noise
        latents.append(latent)
        hidden.append(text)
        noises.append(noise)

    latent_batch = torch.cat(latents, dim=0)
    hidden_batch = torch.cat(hidden, dim=0)
    noise_batch = torch.cat(noises, dim=0)
    timestep_batch = torch.tensor(
        [item["timestep"] for item in plan], device="cuda", dtype=torch.long
    )
    noisy = scheduler.add_noise(latent_batch, noise_batch, timestep_batch)
    prediction_type = str(scheduler.config.prediction_type)
    if prediction_type == "epsilon":
        target = noise_batch
    elif prediction_type == "v_prediction":
        target = scheduler.get_velocity(latent_batch, noise_batch, timestep_batch)
    else:
        raise RuntimeError(f"unsupported scheduler prediction type: {prediction_type}")

    for _, parameter in trainable:
        parameter.grad = None
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        prediction = unet(noisy, timestep_batch, hidden_batch).sample
        per_record = functional.mse_loss(
            prediction.float(), target.float(), reduction="none"
        ).flatten(1).mean(1)
        loss = per_record.mean()
    loss.backward()

    gradient_parts = []
    missing = 0
    for _, parameter in trainable:
        if parameter.grad is None:
            missing += 1
            gradient_parts.append(torch.zeros_like(parameter, dtype=torch.float32).reshape(-1))
        else:
            gradient_parts.append(parameter.grad.detach().float().reshape(-1))
    require(missing == 0, f"missing gradient tensors for {unit['patient_id']}/{bank}/{method}")
    flat = torch.cat(gradient_parts)
    require(bool(torch.isfinite(flat).all()), f"non-finite gradient for {unit['patient_id']}/{bank}/{method}")
    norm = float(torch.linalg.vector_norm(flat.double()).cpu())
    flat_cpu = flat.cpu().contiguous()
    metadata = {
        "patient_id": unit["patient_id"],
        "target_patient": unit["target_patient"],
        "bank": bank,
        "method": method,
        "image_ids": "|".join(record.image_id for record in unit["records"]),
        "timesteps": "|".join(str(item["timestep"]) for item in plan),
        "noise_seed_commitments": "|".join(
            stable_hash("ucan-public-seed", item["noise_seed"]) for item in plan
        ),
        "noise_signs": "|".join(str(item["noise_sign"]) for item in plan),
        "loss": float(loss.detach().cpu()),
        "gradient_l2_norm": norm,
        "would_clip": norm > PATIENT_CLIP_NORM,
        "clip_scale": min(1.0, PATIENT_CLIP_NORM / norm) if norm > 0 else 1.0,
    }
    return flat_cpu, metadata


def tensor_sha256(tensor: Any) -> str:
    import hashlib

    return hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest().upper()


def error_cells(
    vectors: dict[tuple[str, str, str], Any],
    units: list[dict[str, Any]],
    *,
    stage: str,
) -> list[dict[str, Any]]:
    import torch

    patients = [unit["patient_id"] for unit in units]
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for method in METHODS:
            for bank_left, bank_right in itertools.combinations(BANKS, 2):
                differences = torch.stack(
                    [
                        vectors[(method, bank_left, patient)]
                        - vectors[(method, bank_right, patient)]
                        for patient in patients
                    ]
                )
                gpu = differences.to(device="cuda", dtype=torch.float32)
                gram = torch.matmul(gpu, gpu.T).double().cpu().numpy()
                del gpu, differences
                for left_index, right_index in itertools.combinations(range(len(patients)), 2):
                    value = 0.125 * (
                        gram[left_index, left_index]
                        + gram[right_index, right_index]
                        + 2.0 * gram[left_index, right_index]
                    )
                    rows.append(
                        {
                            "stage": stage,
                            "method": method,
                            "bank_left": bank_left,
                            "bank_right": bank_right,
                            "patient_left": patients[left_index],
                            "patient_right": patients[right_index],
                            "b2_replicate_error": float(max(value, 0.0)),
                        }
                    )
                del gram
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"no rows for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def cell_matrix(rows: list[dict[str, Any]], method: str) -> tuple[np.ndarray, list[tuple[str, str]], list[tuple[str, str]]]:
    selected = [row for row in rows if row["method"] == method]
    bank_pairs = sorted({(row["bank_left"], row["bank_right"]) for row in selected})
    patient_pairs = sorted({(row["patient_left"], row["patient_right"]) for row in selected})
    lookup = {
        ((row["bank_left"], row["bank_right"]), (row["patient_left"], row["patient_right"])):
        row["b2_replicate_error"]
        for row in selected
    }
    matrix = np.asarray(
        [[lookup[(bank_pair, patient_pair)] for patient_pair in patient_pairs] for bank_pair in bank_pairs],
        dtype=np.float64,
    )
    require(matrix.shape == (10, 120), f"unexpected cell matrix for {method}: {matrix.shape}")
    return matrix, bank_pairs, patient_pairs


def ratio_summary(rows: list[dict[str, Any]], treatment: str, baseline: str) -> dict[str, Any]:
    treatment_matrix, bank_pairs, patient_pairs = cell_matrix(rows, treatment)
    baseline_matrix, baseline_bank_pairs, baseline_patient_pairs = cell_matrix(rows, baseline)
    require(bank_pairs == baseline_bank_pairs and patient_pairs == baseline_patient_pairs, "ratio cells do not align")
    ratio = float(np.mean(treatment_matrix) / np.mean(baseline_matrix))

    rng = np.random.default_rng(BOOTSTRAP_SEED + sum(map(ord, treatment + baseline)))
    bootstrap = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for index in range(BOOTSTRAP_DRAWS):
        sampled_banks = rng.integers(0, treatment_matrix.shape[0], size=treatment_matrix.shape[0])
        sampled_pairs = rng.integers(0, treatment_matrix.shape[1], size=treatment_matrix.shape[1])
        treatment_mean = np.mean(treatment_matrix[np.ix_(sampled_banks, sampled_pairs)])
        baseline_mean = np.mean(baseline_matrix[np.ix_(sampled_banks, sampled_pairs)])
        bootstrap[index] = treatment_mean / baseline_mean

    leave_bank = {}
    for bank in BANKS:
        keep = np.asarray([bank not in pair for pair in bank_pairs], dtype=bool)
        leave_bank[bank] = float(np.mean(treatment_matrix[keep]) / np.mean(baseline_matrix[keep]))

    patients = sorted({patient for pair in patient_pairs for patient in pair})
    leave_patient = {}
    for patient in patients:
        keep = np.asarray([patient not in pair for pair in patient_pairs], dtype=bool)
        leave_patient[patient] = float(np.mean(treatment_matrix[:, keep]) / np.mean(baseline_matrix[:, keep]))

    return {
        "treatment": treatment,
        "baseline": baseline,
        "ratio": ratio,
        "bootstrap": {
            "scheme": "independent resampling of the 10 bank-pair and 120 patient-pair axes; leave-outs separately expose dependence",
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED + sum(map(ord, treatment + baseline)),
            "ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        },
        "leave_one_bank_out": leave_bank,
        "leave_one_bank_out_wins": sum(value < 1.0 for value in leave_bank.values()),
        "leave_one_patient_out": leave_patient,
        "leave_one_patient_out_wins": sum(value < 1.0 for value in leave_patient.values()),
        "treatment_mean_error": float(np.mean(treatment_matrix)),
        "baseline_mean_error": float(np.mean(baseline_matrix)),
    }


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "dp_protocol"))
    import calibrate_xray_public_clip_norms as calibration

    calibration.disable_broken_optional_onnx()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    require(calibration.sha256_file(args.protocol.resolve()) == PROTOCOL_SHA256, "protocol hash mismatch")
    require(calibration.sha256_file(args.manifest.resolve()) == calibration.K10_SHA256, "K10 manifest hash mismatch")
    require(
        calibration.sha256_file(args.preprocessing_contract.resolve()) == calibration.PREPROCESSING_CONTRACT_SHA256,
        "preprocessing contract hash mismatch",
    )

    import torch
    from diffusers import DDPMScheduler, UNet2DConditionModel
    from huggingface_hub import snapshot_download
    from peft import LoraConfig
    from peft.utils.other import cast_mixed_precision_params

    require(torch.cuda.is_available(), "CUDA is required for the UCAN public diagnostic")
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(calibration.LORA_INIT_SEED)
    torch.cuda.manual_seed_all(calibration.LORA_INIT_SEED)

    sys.path.insert(0, str(root / "data_pipeline"))
    import nih_cxr14_model_input as model_input

    records = model_input.read_manifest(args.manifest.resolve(), partitions={"public_development"})
    require(len(records) == 4_831, "public-development image count mismatch")
    require(len({record.patient_id for record in records}) == 1_816, "public-development patient count mismatch")
    excluded = read_excluded_patients(args.calibration_selection.resolve(), calibration)
    units = select_units(records, excluded)
    selection_path = output_dir / "selected_public_patients_private.csv"
    write_selection(selection_path, units)

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
        snapshot, units, args.image_root.resolve(), model_input
    )

    scheduler = DDPMScheduler.from_pretrained(snapshot / "scheduler", local_files_only=True)
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
    unet.enable_gradient_checkpointing()
    unet.train().to("cuda")
    trainable = [(name, parameter) for name, parameter in unet.named_parameters() if parameter.requires_grad]
    require(sum(parameter.numel() for _, parameter in trainable) == 1_659_904, "trainable parameter count mismatch")
    require({str(parameter.dtype) for _, parameter in trainable} == {"torch.float32"}, "LoRA parameters are not fp32")
    adapter_before = calibration.tensor_digest(trainable)

    started = time.perf_counter()
    vectors: dict[tuple[str, str, str], Any] = {}
    gradient_rows: list[dict[str, Any]] = []
    sentinels: dict[str, str] = {}
    for bank in BANKS:
        for unit in units:
            for method in METHODS:
                vector, row = gradient_for_plan(
                    unit=unit,
                    bank=bank,
                    method=method,
                    latent_by_image=latent_by_image,
                    hidden_by_prompt=hidden_by_prompt,
                    model_input=model_input,
                    scheduler=scheduler,
                    unet=unet,
                    trainable=trainable,
                )
                key = (method, bank, unit["patient_id"])
                require(key not in vectors, f"duplicate gradient key: {key}")
                vectors[key] = vector
                gradient_rows.append(row)
                if len(sentinels) < 2 or (
                    bank == BANKS[-1] and unit is units[-1] and method == METHODS[-1]
                ):
                    sentinels["|".join(key)] = tensor_sha256(vector)

    adapter_after = calibration.tensor_digest(trainable)
    require(adapter_before == adapter_after, "adapter changed without an optimizer step")
    gradient_seconds = time.perf_counter() - started

    del unet, scheduler, latent_by_image, hidden_by_prompt
    gc.collect()
    torch.cuda.empty_cache()

    unclip_started = time.perf_counter()
    unclip_cells = error_cells(vectors, units, stage="unclipped")
    unclip_seconds = time.perf_counter() - unclip_started

    for key, vector in vectors.items():
        matching = next(
            row for row in gradient_rows
            if (row["method"], row["bank"], row["patient_id"]) == key
        )
        vector.mul_(float(matching["clip_scale"]))

    clipped_started = time.perf_counter()
    clipped_cells = error_cells(vectors, units, stage="clipped")
    clipped_seconds = time.perf_counter() - clipped_started
    del vectors
    gc.collect()
    torch.cuda.empty_cache()

    gradient_path = output_dir / "gradient_metadata_private.csv"
    clipped_path = output_dir / "b2_clipped_error_cells_private.csv"
    unclip_path = output_dir / "b2_unclipped_error_cells_private.csv"
    write_csv(gradient_path, gradient_rows)
    write_csv(clipped_path, clipped_cells)
    write_csv(unclip_path, unclip_cells)

    method_summary = {}
    for method in METHODS:
        selected = [row for row in gradient_rows if row["method"] == method]
        method_summary[method] = {
            "gradients": len(selected),
            "mean_loss": float(np.mean([row["loss"] for row in selected])),
            "mean_gradient_l2_norm": float(np.mean([row["gradient_l2_norm"] for row in selected])),
            "median_gradient_l2_norm": float(np.median([row["gradient_l2_norm"] for row in selected])),
            "clipped_gradients": sum(bool(row["would_clip"]) for row in selected),
            "clip_fraction": float(np.mean([row["would_clip"] for row in selected])),
            "mean_b2_unclipped_error": float(np.mean([
                row["b2_replicate_error"] for row in unclip_cells if row["method"] == method
            ])),
            "mean_b2_clipped_error": float(np.mean([
                row["b2_replicate_error"] for row in clipped_cells if row["method"] == method
            ])),
        }

    primary = ratio_summary(clipped_cells, PRIMARY_TREATMENT, PRIMARY_BASELINE)
    total = ratio_summary(clipped_cells, PRIMARY_TREATMENT, ORDINARY_BASELINE)
    noise_only = ratio_summary(clipped_cells, "antithetic_independent_t", ORDINARY_BASELINE)
    shared_t_only = ratio_summary(clipped_cells, "iid_pair_shared_t", ORDINARY_BASELINE)
    unclip_primary = ratio_summary(unclip_cells, PRIMARY_TREATMENT, PRIMARY_BASELINE)
    clip_delta = (
        method_summary[PRIMARY_TREATMENT]["clip_fraction"]
        - method_summary[PRIMARY_BASELINE]["clip_fraction"]
    )
    gate_checks = {
        "primary_ratio_at_most_0_95": primary["ratio"] <= 0.95,
        "primary_bootstrap_upper_below_1": primary["bootstrap"]["ci95"][1] < 1.0,
        "primary_leave_one_bank_out_wins_at_least_4_of_5": primary["leave_one_bank_out_wins"] >= 4,
        "primary_leave_one_patient_out_wins_at_least_14_of_16": primary["leave_one_patient_out_wins"] >= 14,
        "complete_ucan_better_than_ordinary_iid": total["ratio"] < 1.0,
        "clip_fraction_delta_at_most_0_05": clip_delta <= 0.05,
    }
    passed = all(gate_checks.values())
    status = "PASS_PROMOTE_TO_BOUNDED_TRAINING_PILOT" if passed else "FAIL_DO_NOT_PROMOTE_THIS_UCAN_IMPLEMENTATION"

    report = {
        "schema": SCHEMA,
        "status": status,
        "scope": "PUBLIC_DEVELOPMENT_FULL_GRADIENT_DIAGNOSTIC_ONLY_NOT_TRAINING_NOT_DP_NOT_UTILITY",
        "decision": "PROMOTE_BOUNDED_PILOT" if passed else "STOP_IMPLEMENTATION",
        "frozen_inputs": {
            "protocol_sha256": PROTOCOL_SHA256,
            "k10_manifest_sha256": calibration.K10_SHA256,
            "preprocessing_contract_sha256": calibration.PREPROCESSING_CONTRACT_SHA256,
            "calibration_selection_sha256": CALIBRATION_SELECTION_SHA256,
            "patient_clip_norm": PATIENT_CLIP_NORM,
            "selection_salt": SELECTION_SALT,
            "model_id": calibration.MODEL_ID,
            "model_revision": calibration.MODEL_REVISION,
            "critical_model_hashes": snapshot_hashes,
            "lora_init_seed": calibration.LORA_INIT_SEED,
            "lora_rank": calibration.LORA_RANK,
            "lora_targets": list(calibration.LORA_TARGETS),
            "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
            "banks": list(BANKS),
            "methods": list(METHODS),
        },
        "selection": {
            "patients": len(units),
            "target_patients": sum(unit["target_patient"] == 1 for unit in units),
            "control_patients": sum(unit["target_patient"] == 0 for unit in units),
            "records_per_patient": RECORDS_PER_PATIENT,
            "excluded_calibration_patients": len(excluded),
            "selection_file": selection_path.name,
            "selection_sha256": calibration.sha256_file(selection_path),
        },
        "input_summary": input_summary,
        "execution": {
            "optimizer_created": False,
            "optimizer_step": False,
            "dp_noise_added": False,
            "gradient_clipping_simulated_for_analysis": True,
            "private_partition_used": False,
            "attack_holdout_used": False,
            "raw_gradients_retained": False,
            "adapter_before_sha256": adapter_before,
            "adapter_after_sha256": adapter_after,
            "parameters_unchanged": adapter_before == adapter_after,
            "gradient_seconds": gradient_seconds,
            "unclipped_gram_seconds": unclip_seconds,
            "clipped_gram_seconds": clipped_seconds,
            "sentinel_gradient_sha256": sentinels,
        },
        "method_summary": method_summary,
        "ratios": {
            "primary_complete_ucan_vs_shared_t_iid": primary,
            "complete_ucan_vs_ordinary_iid": total,
            "noise_coupling_only_vs_ordinary_iid": noise_only,
            "shared_t_only_vs_ordinary_iid": shared_t_only,
            "unclipped_complete_ucan_vs_shared_t_iid": unclip_primary,
        },
        "clip_fraction_delta_treatment_minus_primary_baseline": clip_delta,
        "gate_checks": gate_checks,
        "interpretation_limit": (
            "A pass only licenses a bounded public/research training pilot. A fail closes this exact implementation. "
            "Neither outcome is a generation-quality, clinical, privacy-release, or novelty result."
        ),
        "artifacts": {
            "gradient_metadata": gradient_path.name,
            "gradient_metadata_sha256": calibration.sha256_file(gradient_path),
            "b2_clipped_error_cells": clipped_path.name,
            "b2_clipped_error_cells_sha256": calibration.sha256_file(clipped_path),
            "b2_unclipped_error_cells": unclip_path.name,
            "b2_unclipped_error_cells_sha256": calibration.sha256_file(unclip_path),
        },
    }
    report_path = output_dir / "report.json"
    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "status": status,
        "primary_ratio": primary["ratio"],
        "primary_ci95": primary["bootstrap"]["ci95"],
        "total_ratio": total["ratio"],
        "noise_only_ratio": noise_only["ratio"],
        "shared_t_only_ratio": shared_t_only["ratio"],
        "clip_fraction_delta": clip_delta,
        "gradient_seconds": gradient_seconds,
        "report": str(report_path),
    }, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

