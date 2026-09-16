#!/usr/bin/env python3
"""Generate five preregistered full cross-patient LoRA-gradient Gram banks."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from run_isic_sd21_gradient_smoke import (
    ALLOW_PATTERNS,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_VARIANT,
    cuda_memory,
    disable_broken_optional_onnx,
    encode_patient_inputs,
    package_version,
    sha256_file,
    verify_snapshot,
)
from run_isic_sd21_multipatient_diagnostic import (
    IMAGE_SIZE,
    LORA_RANK,
    load_gradient_model,
    write_csv,
)
from run_isic_sd21_two_axis_confirmatory import select_confirmatory_patients


SCHEMA = "pcm-isic-sd21-joint-crossgram/v1"
PROTOCOL = "ISIC_SD21_JOINT_BATCH_LOCALITY_PROTOCOL_V1.md"
SPEC_FILE = "ISIC_SD21_JOINT_BATCH_BANKS_V1.csv"
EXPECTED_SPEC_SHA256 = "ECF583521CEBC760E5FF4BCA2AB20B5F16FA0ABC4B89436448EF1560089D981A"
EXPECTED_TAGS = tuple(f"joint_{index:02d}" for index in range(1, 6))
PERTURBATIONS = 8
RECORDS_PER_PATIENT = 5
GRADIENTS_PER_PATIENT = RECORDS_PER_PATIENT * PERTURBATIONS
SPOT_CHECKS = 64
SPOT_SEED = 26090305
DIRECT_DOT_RELATIVE_TOLERANCE = 1e-5
NORM_DIAGONAL_RELATIVE_TOLERANCE = 1e-4


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_specs(path: Path) -> dict[str, list[dict[str, int]]]:
    grouped: dict[str, list[dict[str, int]]] = defaultdict(list)
    integer_fields = (
        "perturbation_index",
        "bin_index",
        "bin_replicate",
        "timestep",
        "timestep_bin_start",
        "timestep_bin_end_exclusive",
        "noise_seed",
    )
    for row in read_csv(path):
        grouped[row["bank_tag"]].append(
            {name: int(row[name]) for name in integer_fields}
        )
    if tuple(grouped) != EXPECTED_TAGS:
        raise ValueError(f"unexpected bank tags: {tuple(grouped)}")
    for bank_tag, rows in grouped.items():
        rows.sort(key=lambda row: row["perturbation_index"])
        if [row["perturbation_index"] for row in rows] != list(range(8)):
            raise ValueError(f"invalid perturbation indices in {bank_tag}")
        counts = {
            bin_index: sum(row["bin_index"] == bin_index for row in rows)
            for bin_index in range(4)
        }
        if set(counts.values()) != {2}:
            raise ValueError(f"invalid bin counts in {bank_tag}: {counts}")
        for row in rows:
            if not (
                row["timestep_bin_start"]
                <= row["timestep"]
                < row["timestep_bin_end_exclusive"]
            ):
                raise ValueError(f"timestep outside frozen bin in {bank_tag}: {row}")
    return dict(grouped)


def gradient_sha256(vector: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(vector).view(np.uint8)).hexdigest().upper()


def fill_patient_gradient_bank(
    output: np.ndarray,
    patient_id: str,
    patient_index: int,
    latents: list[Any],
    encoder_states: list[Any],
    unet: Any,
    scheduler: Any,
    trainable: list[tuple[str, Any]],
    specs: list[dict[str, int]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    import torch.nn.functional as functional

    if output.shape[0] != len(latents) * len(specs):
        raise ValueError("output bank shape does not match patient inputs")
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for record_index, (latent_cpu, hidden_cpu) in enumerate(
        zip(latents, encoder_states)
    ):
        latent = latent_cpu.to(device="cuda", dtype=torch.float16)
        hidden = hidden_cpu.to(device="cuda", dtype=torch.float16)
        for spec in specs:
            perturbation = int(spec["perturbation_index"])
            timestep = torch.tensor(
                [int(spec["timestep"])], device="cuda", dtype=torch.long
            )
            generator = torch.Generator(device="cuda").manual_seed(
                int(spec["noise_seed"])
            )
            noise = torch.randn(
                latent.shape,
                generator=generator,
                device="cuda",
                dtype=torch.float16,
            )
            noisy_latent = scheduler.add_noise(latent, noise, timestep)
            prediction_type = str(scheduler.config.prediction_type)
            if prediction_type == "epsilon":
                target = noise
            elif prediction_type == "v_prediction":
                target = scheduler.get_velocity(latent, noise, timestep)
            else:
                raise RuntimeError(f"unsupported prediction_type: {prediction_type}")

            for _, parameter in trainable:
                parameter.grad = None
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = unet(noisy_latent, timestep, hidden).sample
                loss = functional.mse_loss(prediction.float(), target.float())
            loss.backward()

            local_index = record_index * len(specs) + perturbation
            offset = 0
            finite = True
            for _, parameter in trainable:
                count = parameter.numel()
                if parameter.grad is None:
                    output[local_index, offset : offset + count] = 0.0
                else:
                    gradient = parameter.grad.detach().float().reshape(-1).cpu().numpy()
                    output[local_index, offset : offset + count] = gradient
                    finite = finite and bool(np.isfinite(gradient).all())
                offset += count
            if not finite:
                raise RuntimeError(f"non-finite gradient for {patient_id}")
            vector = output[local_index]
            rows.append(
                {
                    "patient_id": patient_id,
                    "patient_index": patient_index,
                    "record_index": record_index,
                    "perturbation_index": perturbation,
                    "global_gradient_index": patient_index * GRADIENTS_PER_PATIENT
                    + local_index,
                    "timestep": int(spec["timestep"]),
                    "noise_seed": int(spec["noise_seed"]),
                    "loss": float(loss.detach().cpu()),
                    "gradient_norm": float(np.linalg.norm(vector)),
                    "gradient_sha256": gradient_sha256(vector),
                    "finite": finite,
                }
            )
    torch.cuda.synchronize()
    return rows, {
        "patient_id": patient_id,
        "patient_index": patient_index,
        "gradients": len(rows),
        "elapsed_sec": round(time.perf_counter() - started, 4),
    }


def spot_pairs(size: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(SPOT_SEED)
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < SPOT_CHECKS:
        left, right = sorted(rng.integers(0, size, size=2).tolist())
        pairs.add((left, right))
    return sorted(pairs)


def bank_is_complete(
    output_dir: Path, bank_tag: str, protocol_hash: str, spec_hash: str
) -> bool:
    metadata_path = output_dir / f"{bank_tag}_crossgram.json"
    gram_path = output_dir / f"{bank_tag}_crossgram_local_only.npy"
    evaluations_path = output_dir / f"{bank_tag}_gradient_evaluations.csv"
    runtime_path = output_dir / f"{bank_tag}_patient_runtime.csv"
    if not all(path.is_file() for path in (metadata_path, gram_path, evaluations_path, runtime_path)):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return bool(
            metadata["status"] == "PASS"
            and metadata["bank_tag"] == bank_tag
            and metadata["protocol_sha256"] == protocol_hash
            and metadata["spec_sha256"] == spec_hash
            and metadata["artifacts_sha256"][gram_path.name] == sha256_file(gram_path)
            and metadata["artifacts_sha256"][evaluations_path.name]
            == sha256_file(evaluations_path)
            and metadata["artifacts_sha256"][runtime_path.name]
            == sha256_file(runtime_path)
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def build_gradient_index(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for patient_index, patient in enumerate(selected):
        for record_index in range(RECORDS_PER_PATIENT):
            record = patient["records"][record_index]
            for perturbation in range(PERTURBATIONS):
                rows.append(
                    {
                        "global_gradient_index": patient_index * GRADIENTS_PER_PATIENT
                        + record_index * PERTURBATIONS
                        + perturbation,
                        "patient_index": patient_index,
                        "patient_id": patient["patient_id"],
                        "site_group": patient["site_group"],
                        "record_index": record_index,
                        "image_id": record.image_id,
                        "perturbation_index": perturbation,
                    }
                )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    disable_broken_optional_onnx()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    import torch
    from huggingface_hub import snapshot_download

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = Path(__file__).with_name(PROTOCOL)
    spec_path = Path(__file__).with_name(SPEC_FILE)
    protocol_hash = sha256_file(protocol_path)
    spec_hash = sha256_file(spec_path)
    if spec_hash != EXPECTED_SPEC_SHA256:
        raise RuntimeError(f"frozen bank spec hash mismatch: {spec_hash}")
    bank_specs = load_specs(spec_path)
    all_noise_seeds = [
        row["noise_seed"] for rows in bank_specs.values() for row in rows
    ]
    if not len(all_noise_seeds) == len(set(all_noise_seeds)) == 40:
        raise RuntimeError("joint bank noise seeds are not all unique")

    source_report = args.cohort_source_report.resolve()
    source_summary = json.loads(
        (source_report / "summary.json").read_text(encoding="utf-8")
    )
    previous_seeds: set[int] = set()
    for report_dir in args.previous_bank_report:
        for row in read_csv(report_dir.resolve() / "perturbation_specs.csv"):
            previous_seeds.add(int(row["noise_seed"]))
    seed_overlap = sorted(set(all_noise_seeds) & previous_seeds)
    if seed_overlap:
        raise RuntimeError(f"joint noise seeds overlap previous banks: {seed_overlap}")

    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=ALLOW_PATTERNS,
            local_files_only=True,
        )
    ).resolve()
    critical_hash_checks = verify_snapshot(snapshot)
    data_pipeline_dir = Path(__file__).resolve().parents[1] / "data_pipeline"
    sys.path.insert(0, str(data_pipeline_dir))
    from isic_manifest_dataset import read_manifest  # pylint: disable=import-error,import-outside-toplevel

    all_records = read_manifest(args.manifest, partitions={"public_development"})
    selected, selection_evidence = select_confirmatory_patients(all_records)
    patient_ids = [str(patient["patient_id"]) for patient in selected]
    expected_patient_ids = [
        str(patient["patient_id"]) for patient in source_summary["selection"]["selected"]
    ]
    if patient_ids != expected_patient_ids:
        raise RuntimeError("selected cohort differs from frozen confirmatory cohort")
    write_csv(output_dir / "gradient_index.csv", build_gradient_index(selected))

    flattened_records = [record for patient in selected for record in patient["records"]]
    latents, encoder_states, _, record_rows = encode_patient_inputs(
        snapshot, flattened_records, args.image_root.resolve(), IMAGE_SIZE
    )
    patient_slices: dict[str, slice] = {}
    cursor = 0
    for patient in selected:
        patient_id = str(patient["patient_id"])
        patient_slices[patient_id] = slice(cursor, cursor + RECORDS_PER_PATIENT)
        for local_index in range(RECORDS_PER_PATIENT):
            row = record_rows[cursor + local_index]
            row["patient_id"] = patient_id
            row["site_group"] = patient["site_group"]
            row["patient_site_count"] = patient["site_count"]
            row["patient_selection_hash"] = patient["selection_hash"]
        cursor += RECORDS_PER_PATIENT
    write_csv(output_dir / "selected_patient_records.csv", record_rows)

    source_records = read_csv(source_report / "selected_patient_records.csv")
    current_record_identity = [
        (row["patient_id"], row["image_id"], row["image_sha256"]) for row in record_rows
    ]
    source_record_identity = [
        (row["patient_id"], row["image_id"], row["image_sha256"])
        for row in source_records
    ]
    if current_record_identity != source_record_identity:
        raise RuntimeError("encoded records differ from frozen source report")

    pending_tags = [
        bank_tag
        for bank_tag in EXPECTED_TAGS
        if not (
            args.resume
            and bank_is_complete(output_dir, bank_tag, protocol_hash, spec_hash)
        )
    ]
    unet = scheduler = trainable = model_metadata = None
    if pending_tags:
        unet, scheduler, trainable, model_metadata = load_gradient_model(snapshot)
        if int(model_metadata["parameter_dimension"]) != int(
            source_summary["model"]["parameter_dimension"]
        ):
            raise RuntimeError("trainable parameter dimension differs from source")

    for bank_position, bank_tag in enumerate(EXPECTED_TAGS, start=1):
        if bank_tag not in pending_tags:
            print(
                json.dumps(
                    {"progress": "joint_crossgram_bank_resume_skip", "bank_tag": bank_tag}
                ),
                flush=True,
            )
            continue
        assert unet is not None and scheduler is not None and trainable is not None
        assert model_metadata is not None
        dimension = int(model_metadata["parameter_dimension"])
        bank_started = time.perf_counter()
        full_bank = np.empty(
            (len(selected) * GRADIENTS_PER_PATIENT, dimension), dtype=np.float32
        )
        evaluation_rows: list[dict[str, Any]] = []
        runtime_rows: list[dict[str, Any]] = []
        print(
            json.dumps(
                {
                    "progress": "joint_crossgram_bank_started",
                    "bank_index": bank_position,
                    "banks_total": len(EXPECTED_TAGS),
                    "bank_tag": bank_tag,
                    "raw_bank_gib": round(full_bank.nbytes / 1024**3, 4),
                }
            ),
            flush=True,
        )
        for patient_index, patient in enumerate(selected):
            patient_id = str(patient["patient_id"])
            patient_slice = patient_slices[patient_id]
            output_slice = slice(
                patient_index * GRADIENTS_PER_PATIENT,
                (patient_index + 1) * GRADIENTS_PER_PATIENT,
            )
            evaluations, runtime = fill_patient_gradient_bank(
                output=full_bank[output_slice],
                patient_id=patient_id,
                patient_index=patient_index,
                latents=latents[patient_slice],
                encoder_states=encoder_states[patient_slice],
                unet=unet,
                scheduler=scheduler,
                trainable=trainable,
                specs=bank_specs[bank_tag],
            )
            for row in evaluations:
                row["bank_tag"] = bank_tag
            runtime["bank_tag"] = bank_tag
            evaluation_rows.extend(evaluations)
            runtime_rows.append(runtime)
            print(
                json.dumps(
                    {
                        "progress": "joint_crossgram_patient_complete",
                        "bank_tag": bank_tag,
                        "patient_index": patient_index + 1,
                        "patients_total": len(selected),
                        "patient_id": patient_id,
                        "elapsed_sec": runtime["elapsed_sec"],
                    }
                ),
                flush=True,
            )

        if not (
            len(evaluation_rows) == 640
            and all(bool(row["finite"]) for row in evaluation_rows)
            and np.all(np.isfinite(full_bank))
        ):
            raise RuntimeError(f"non-finite or incomplete gradient bank: {bank_tag}")
        direct_spots = {
            f"{left}:{right}": float(np.dot(full_bank[left], full_bank[right]))
            for left, right in spot_pairs(full_bank.shape[0])
        }
        print(
            json.dumps(
                {"progress": "joint_crossgram_matmul_started", "bank_tag": bank_tag}
            ),
            flush=True,
        )
        gram = np.asarray(full_bank @ full_bank.T, dtype=np.float64)
        gram = 0.5 * (gram + gram.T)
        maximum_spot_relative_error = 0.0
        for key, expected in direct_spots.items():
            left, right = (int(value) for value in key.split(":"))
            observed = float(gram[left, right])
            maximum_spot_relative_error = max(
                maximum_spot_relative_error,
                abs(observed - expected) / max(abs(expected), abs(observed), 1e-30),
            )
        norm_squared = np.square(
            np.asarray([row["gradient_norm"] for row in evaluation_rows], dtype=np.float64)
        )
        maximum_diagonal_relative_error = float(
            np.max(
                np.abs(np.diag(gram) - norm_squared)
                / np.maximum(np.maximum(np.abs(np.diag(gram)), norm_squared), 1e-30)
            )
        )
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(gram)))
        maximum_diagonal = float(np.max(np.diag(gram)))
        psd_tolerance = max(1e-6, maximum_diagonal * 1e-4)
        gram_path = output_dir / f"{bank_tag}_crossgram_local_only.npy"
        evaluations_path = output_dir / f"{bank_tag}_gradient_evaluations.csv"
        runtime_path = output_dir / f"{bank_tag}_patient_runtime.csv"
        np.save(gram_path, gram, allow_pickle=False)
        write_csv(evaluations_path, evaluation_rows)
        write_csv(runtime_path, runtime_rows)
        artifacts_sha256 = {
            gram_path.name: sha256_file(gram_path),
            evaluations_path.name: sha256_file(evaluations_path),
            runtime_path.name: sha256_file(runtime_path),
        }
        checks = [
            {
                "name": "all_640_gradients_finite",
                "status": "PASS"
                if len(evaluation_rows) == 640
                and all(bool(row["finite"]) for row in evaluation_rows)
                else "FAIL",
            },
            {
                "name": "full_640_by_640_gram_finite_and_symmetric",
                "status": "PASS"
                if gram.shape == (640, 640)
                and np.all(np.isfinite(gram))
                and np.allclose(gram, gram.T, rtol=0.0, atol=1e-12)
                else "FAIL",
            },
            {
                "name": "direct_dot_spot_checks",
                "status": "PASS"
                if maximum_spot_relative_error <= DIRECT_DOT_RELATIVE_TOLERANCE
                else "FAIL",
                "checks": SPOT_CHECKS,
                "maximum_relative_error": maximum_spot_relative_error,
                "relative_tolerance": DIRECT_DOT_RELATIVE_TOLERANCE,
            },
            {
                "name": "gradient_norm_diagonal_match",
                "status": "PASS"
                if maximum_diagonal_relative_error
                <= NORM_DIAGONAL_RELATIVE_TOLERANCE
                else "FAIL",
                "maximum_relative_error": maximum_diagonal_relative_error,
                "relative_tolerance": NORM_DIAGONAL_RELATIVE_TOLERANCE,
            },
            {
                "name": "full_gram_psd_tolerance",
                "status": "PASS"
                if minimum_eigenvalue >= -psd_tolerance
                else "FAIL",
                "minimum_eigenvalue": minimum_eigenvalue,
                "tolerance": psd_tolerance,
            },
        ]
        status = "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL"
        bank_metadata: dict[str, Any] = {
            "schema": "pcm-isic-sd21-joint-crossgram-bank/v1",
            "status": status,
            "bank_tag": bank_tag,
            "protocol_sha256": protocol_hash,
            "spec_sha256": spec_hash,
            "patients": len(selected),
            "gradients": len(evaluation_rows),
            "parameter_dimension": dimension,
            "full_gram_shape": list(gram.shape),
            "elapsed_sec": round(time.perf_counter() - bank_started, 4),
            "patient_elapsed_sec_sum": float(
                sum(float(row["elapsed_sec"]) for row in runtime_rows)
            ),
            "checks": checks,
            "artifacts_sha256": artifacts_sha256,
            "raw_gradients_saved": False,
            "gram_release_status": "LOCAL_ONLY",
        }
        (output_dir / f"{bank_tag}_crossgram.json").write_text(
            json.dumps(bank_metadata, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "progress": "joint_crossgram_bank_complete",
                    "bank_tag": bank_tag,
                    "status": status,
                    "elapsed_sec": bank_metadata["elapsed_sec"],
                    "minimum_eigenvalue": minimum_eigenvalue,
                }
            ),
            flush=True,
        )
        del full_bank
        del gram
        gc.collect()
        if status != "PASS":
            raise RuntimeError(f"crossgram bank failed checks: {bank_tag}")

    if unet is not None:
        del unet
        del scheduler
        del trainable
        gc.collect()
        torch.cuda.empty_cache()

    bank_metadata_rows = [
        json.loads(
            (output_dir / f"{bank_tag}_crossgram.json").read_text(encoding="utf-8")
        )
        for bank_tag in EXPECTED_TAGS
    ]
    execution_checks = [
        {
            "name": "critical_model_hashes",
            "status": "PASS"
            if all(row["status"] == "PASS" for row in critical_hash_checks.values())
            else "FAIL",
        },
        {
            "name": "frozen_cohort_and_records_match",
            "status": "PASS"
            if patient_ids == expected_patient_ids
            and current_record_identity == source_record_identity
            else "FAIL",
        },
        {
            "name": "exact_five_new_bank_tags",
            "status": "PASS" if tuple(bank_specs) == EXPECTED_TAGS else "FAIL",
        },
        {
            "name": "all_40_new_noise_seeds_unique_and_disjoint",
            "status": "PASS"
            if len(all_noise_seeds) == len(set(all_noise_seeds)) == 40
            and not seed_overlap
            else "FAIL",
        },
        {
            "name": "all_five_crossgram_banks_pass",
            "status": "PASS"
            if all(row["status"] == "PASS" for row in bank_metadata_rows)
            else "FAIL",
        },
    ]
    status = (
        "PASS" if all(row["status"] == "PASS" for row in execution_checks) else "FAIL"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "privacy_status": "DIAGNOSTIC_NOT_DP",
        "protocol": {"file": PROTOCOL, "sha256": protocol_hash},
        "bank_spec": {
            "file": SPEC_FILE,
            "sha256": spec_hash,
            "bank_tags": list(EXPECTED_TAGS),
        },
        "input_manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": sha256_file(args.manifest.resolve()),
        },
        "cohort_source_report": {
            "path": str(source_report),
            "summary_sha256": sha256_file(source_report / "summary.json"),
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "variant": MODEL_VARIANT,
            "image_size": IMAGE_SIZE,
            "rank": LORA_RANK,
            "parameter_dimension": int(source_summary["model"]["parameter_dimension"]),
            "critical_hash_checks": critical_hash_checks,
        },
        "selection": {
            **selection_evidence,
            "patient_order": patient_ids,
            "patient_pairs": math.comb(len(patient_ids), 2),
        },
        "gradient_execution": {
            "banks": len(EXPECTED_TAGS),
            "patients_per_bank": len(patient_ids),
            "gradients_per_patient": GRADIENTS_PER_PATIENT,
            "total_gradients": len(EXPECTED_TAGS)
            * len(patient_ids)
            * GRADIENTS_PER_PATIENT,
            "full_gram_shape_per_bank": [640, 640],
            "raw_gradients_saved": False,
            "gram_release_status": "LOCAL_ONLY",
            "bank_results": bank_metadata_rows,
            "cuda": cuda_memory(torch),
        },
        "execution_checks": execution_checks,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "packages": {
                name: package_version(name)
                for name in ("diffusers", "transformers", "peft", "safetensors")
            },
        },
        "artifacts": {
            "gradient_index": "gradient_index.csv",
            "selected_patient_records": "selected_patient_records.csv",
            "bank_pattern": "joint_XX_crossgram_local_only.npy",
            "bank_metadata_pattern": "joint_XX_crossgram.json",
        },
        "analysis_boundary": (
            "Five new full cross-patient finite-gradient Gram banks; no optimizer update, "
            "DP noise, training, generation, clinical, novelty, or venue claim."
        ),
    }
    (output_dir / "crossgram_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--cohort-source-report", type=Path, required=True)
    parser.add_argument(
        "--previous-bank-report", type=Path, action="append", default=[]
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
