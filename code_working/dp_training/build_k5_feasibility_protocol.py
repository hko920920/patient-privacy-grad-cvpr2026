#!/usr/bin/env python3
"""Build the frozen, outcome-independent K5 full-feasibility protocol.

This builder performs no optimizer update and consumes no model output.  It
materializes only a public generation-task list, a local reference manifest
selected from ``public_development``, and a machine-readable execution plan.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Callable


SCHEMA = "nih-cxr14-k5-feasibility-execution-protocol/v1"
REPORT_SCHEMA = "nih-cxr14-k5-feasibility-protocol-build/v1"
STATUS = "FROZEN_BEFORE_FULL_K5_OPTIMIZER_EXECUTION"

UPSTREAM_PROTOCOL_SHA256 = "F2757DC7EBC7488B6A9DD0F69227419EA16BB9A3E9BD4434514CB19AFD2B4018"
K5_MANIFEST_SHA256 = "DC49D82E497EAA6940DCF92C8F773179B8F7A838057A82AE0C26A69CA785BFC5"
PREPROCESSING_SHA256 = "0ED9E434764BDB2D542BB16AB951F856CB5B82D4F2E6B9F25F6C554F1076669D"
MECHANISM_SHA256 = "B4B1E037A3704238B06B442970EAB082433835201B750546C8281BBFFE2B969E"
DRYRUN_PROTOCOL_SHA256 = "0509D745E8B192FC2CE7694EBDA0C36AB60005D704864D27A44681FD6E8432C8"
DRYRUN_PUBLIC_REPORT_SHA256 = "77FA9097F1FA161E42E72B7E182F5E0A59CB47EC5D89209929DEE0E95767FB87"
DRYRUN_INDEPENDENT_SHA256 = "27D81FF7565659B3168CECCBA3A94DE31A92CA29154943559AC4510E32B283F9"

MODEL_ID = "Manojb/stable-diffusion-2-1-base"
MODEL_REVISION = "0094d483a120f3f33dafbd187ea4aa60d10de75c"
MODEL_HASHES = {
    "text_encoder": "681C555376658C81DC273F2D737A2AEB23DDB6D1D8E5B3A7064636D359A22668",
    "unet": "28EC9CF3B239C0751C201B1F6FB46B551DF5862731B30A37AA1360101CB3FBAB",
    "vae": "3E4C08995484EE61270175E9E7A072B66A6E4EEB5F0C266667FE1F45B90DAF9A",
}

RAD_DINO = {
    "model_id": "microsoft/rad-dino",
    "revision": "110cbc18d5133582e320b43d53bf5c44e410c936",
    "license": "MIT",
    "model_safetensors_bytes": 346_345_912,
    "model_safetensors_sha256": "DBFB9F54459C38773505DE64A6AB7807BDCB392610FE1E697166342E43FB91AE",
    "config_sha256": "89DAF9751D9576D586DEDF9543C1083211611FA3A36908DB7A799B3CE7C68EDE",
    "preprocessor_sha256": "C537FC995C30E2353F07253899618D60E9EAE3D5F82473778602C007C6523B56",
    "feature": "768-dimensional pooler_output/CLS embedding",
    "pretraining_overlap": (
        "The released RAD-DINO checkpoint was self-supervised on all 112,120 NIH-CXR images among "
        "five public datasets. It is therefore a fixed comparative feature screen, not an independent "
        "clinical validator and not privacy evidence."
    ),
}

BIOVIL_T = {
    "model_id": "microsoft/BiomedVLP-BioViL-T",
    "revision": "692f09e9be1bfe5fdd5f3efdd0e1eca7d2c10b23",
    "license": "MIT",
    "text_model_safetensors_bytes": 440_909_232,
    "text_model_safetensors_sha256": "70BED344872E0A4F4DCA2352564C44F44CB7A9CFCAC3D878BE2DD08A5FFF9ABA",
    "image_model_bytes": 109_745_561,
    "image_model_sha256": "B2399D73DC2A68B9F3A1950E864AE0ECD24093FB07AA459D7E65807EBDC0FB77",
    "config_sha256": "58A8EAD5BDBE8B71E396CD094746779DC929954841282F94B6C100F0FE009A88",
    "remote_code_hashes": {
        "configuration_cxrbert.py": "857366A973291D09EA8E19CFE749FC801E81FAE9AE9CE0C9B462C7C26CB67E1F",
        "modeling_cxrbert.py": "88F0F0D1FCF6B5C91FD3BEBEEC0C3BFEE271103DB7C13C5229E3F4807CB17AC2",
    },
    "training_boundary": "MIMIC-CXR/MIMIC-III/PubMed, not NIH ChestXray14",
}

REFERENCE_SALT = "nih-cxr14-k5-feasibility-real-reference-v1"
GENERATION_SALT = "nih-cxr14-k5-feasibility-generation-v1"
BOOTSTRAP_SALT = "nih-cxr14-k5-feasibility-bootstrap-v1"

REFERENCE_STRATA: list[tuple[str, int, Callable[[dict[str, str]], bool]]] = [
    ("pneumothorax", 64, lambda row: "pneumothorax" in tokens(row["primary_groups"])),
    (
        "pneumonia_or_consolidation",
        128,
        lambda row: "pneumonia_or_consolidation" in tokens(row["primary_groups"]),
    ),
    ("pleural_effusion", 64, lambda row: "pleural_effusion" in tokens(row["primary_groups"])),
    ("mass_or_nodule", 128, lambda row: "mass_or_nodule" in tokens(row["primary_groups"])),
    ("no_finding", 64, lambda row: row["finding_labels"] == "No Finding"),
]

CONDITIONS = [
    ("no_finding", "no_finding", "with no labeled finding"),
    ("pneumothorax", "pneumothorax", "with radiographic findings of pneumothorax"),
    ("pneumonia", "pneumonia_or_consolidation", "with radiographic findings of pneumonia"),
    ("consolidation", "pneumonia_or_consolidation", "with radiographic findings of consolidation"),
    ("pleural_effusion", "pleural_effusion", "with radiographic findings of pleural effusion"),
    ("mass_opacity", "mass_or_nodule", "with radiographic findings of mass opacity"),
    ("nodule_opacity", "mass_or_nodule", "with radiographic findings of nodule opacity"),
]


def tokens(value: str) -> set[str]:
    return {token for token in value.split("|") if token}


def root_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parent.parent
    report = root / "_reports" / "nih_cxr14_k5_feasibility_protocol_v1_001"
    return {
        "root": root,
        "report": report,
        "protocol": report / "protocol.json",
        "generation": report / "generation_tasks.csv",
        "reference": report / "real_reference_local.csv",
        "builder_report": report / "builder_report.json",
        "manifest": root / "_data" / "derived" / "nih_cxr14_pa_target_enriched_v1" / "k5_private.csv",
        "upstream": root / "_reports" / "nih_cxr14_dp_attack_protocol_v1_001" / "protocol.json",
        "preprocessing": root
        / "_reports"
        / "nih_cxr14_sd21_ppmark_interface_v1_001"
        / "preprocessing_contract.json",
        "mechanism": root / "dp_training" / "mechanism.py",
        "dryrun_protocol": root / "dp_training" / "private_research_dryrun_protocol.json",
        "dryrun_public": root
        / "_reports"
        / "nih_cxr14_k5_private_research_dryrun_v1_001"
        / "public_report.json",
        "dryrun_independent": root
        / "_reports"
        / "nih_cxr14_k5_private_research_dryrun_gate_v1_001"
        / "independent_verification.json",
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


def canonical_seed(condition: str, index: int) -> int:
    digest = hashlib.sha256(f"{GENERATION_SALT}|{condition}|{index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_generation_tasks(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prefix = "a frontal posteroanterior chest radiograph"
    for condition, metric_stratum, suffix in CONDITIONS:
        for index in range(64):
            rows.append(
                {
                    "task_id": f"{condition}_{index:03d}",
                    "condition": condition,
                    "metric_stratum": metric_stratum,
                    "prompt": f"{prefix} {suffix}",
                    "seed": canonical_seed(condition, index),
                }
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def select_references(manifest: Path, path: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["partition"] == "public_development"]
    require(len(rows) == 4_031, "K5 public-development image count drift")
    require(len({row["patient_id"] for row in rows}) == 1_816, "K5 public-development patient count drift")

    used_patients: set[str] = set()
    selected: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    for stratum, quota, predicate in REFERENCE_STRATA:
        candidates = [row for row in rows if predicate(row)]
        candidates.sort(
            key=lambda row: hashlib.sha256(
                f"{REFERENCE_SALT}|{stratum}|{row['image_id']}".encode("utf-8")
            ).hexdigest()
        )
        chosen: list[dict[str, str]] = []
        for row in candidates:
            if row["patient_id"] in used_patients:
                continue
            item = {
                "reference_stratum": stratum,
                "image_id": row["image_id"],
                "patient_id": row["patient_id"],
                "finding_labels": row["finding_labels"],
                "primary_groups": row["primary_groups"],
                "image_filename": row["image_filename"],
                "selection_key_sha256": hashlib.sha256(
                    f"{REFERENCE_SALT}|{stratum}|{row['image_id']}".encode("utf-8")
                ).hexdigest().upper(),
            }
            chosen.append(item)
            used_patients.add(row["patient_id"])
            if len(chosen) == quota:
                break
        require(len(chosen) == quota, f"insufficient globally patient-distinct references: {stratum}")
        selected.extend(chosen)
        counts[stratum] = len(chosen)

    require(len(selected) == 448, "reference total drift")
    require(len({row["image_id"] for row in selected}) == 448, "reference images are not unique")
    require(len({row["patient_id"] for row in selected}) == 448, "reference patients are not unique")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    return selected, counts


def binomial_tail_union_bound(n: int, q: float, maximum: int, steps: int) -> float:
    # P[X > maximum] evaluated without SciPy so the frozen protocol is stdlib-buildable.
    tail = 0.0
    for k in range(maximum + 1, n + 1):
        log_p = (
            math.lgamma(n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(n - k + 1)
            + k * math.log(q)
            + (n - k) * math.log1p(-q)
        )
        term = math.exp(log_p)
        tail += term
        if k > maximum + 100 and term < 1e-320:
            break
    return min(1.0, steps * tail)


def build_protocol(generation_sha: str, reference_sha: str) -> dict[str, Any]:
    m1_q = 8 / 18_393
    m2_q = 4 / 8_476
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "scope": "ONE_SEED_K5_RESEARCH_FEASIBILITY_NOT_CONFIRMATORY_NOT_RELEASE",
        "run_id": "nih_cxr14_k5_feasibility_v1_001",
        "frozen_before": [
            "any full K5 optimizer update",
            "any K5 checkpoint selection",
            "any K5 generated-image quality result",
            "any K5 privacy attack result",
        ],
        "upstream": {
            "dp_attack_protocol_sha256": UPSTREAM_PROTOCOL_SHA256,
            "k5_manifest_sha256": K5_MANIFEST_SHA256,
            "preprocessing_contract_sha256": PREPROCESSING_SHA256,
            "mechanism_source_sha256": MECHANISM_SHA256,
            "private_dryrun_protocol_sha256": DRYRUN_PROTOCOL_SHA256,
            "private_dryrun_public_report_sha256": DRYRUN_PUBLIC_REPORT_SHA256,
            "private_dryrun_independent_report_sha256": DRYRUN_INDEPENDENT_SHA256,
        },
        "data_and_model": {
            "dataset": "NIH ChestXray14 public proxy handled as restricted",
            "partition": "K5 private_train",
            "cap": 5,
            "images": 18_393,
            "patients": 8_476,
            "profile": "P256",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_variant": "fp16",
            "critical_model_hashes": MODEL_HASHES,
            "lora": {
                "initialization_seed": 260_903,
                "rank": 8,
                "alpha": 8,
                "targets": ["to_q", "to_k", "to_v", "to_out.0"],
                "trainable_parameters": 1_659_904,
                "trainable_dtype": "float32",
                "frozen_base_dtype": "float16",
            },
            "latent_policy": (
                "deterministic VAE mode latents and prompt embeddings may be cached in CPU memory within an arm; "
                "no raw-image, resized-image, latent, text-embedding, or gradient bank is persisted"
            ),
        },
        "matrix": {
            "arm_order": ["M0", "M1-I8", "M1-G8", "M2-P8"],
            "fresh_identical_initial_lora_per_arm": True,
            "steps_per_arm": 4_000,
            "one_seed_only": True,
            "K5_cannot_select_or_drop_K10_arms_from_preferred_ordering": True,
            "K5_privacy_attacks": "NOT_RUN; K10 attack protocol remains frozen and uncontaminated",
        },
        "optimizer": {
            "type": "AdamW",
            "learning_rate": 1e-4,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0.01,
            "scheduler": "constant",
            "warmup_steps": 0,
            "maximum_and_final_step": 4_000,
            "private_data_dependent_stopping_or_selection": "PROHIBITED",
        },
        "M0": {
            "privacy": "non-DP comparator; no epsilon or patient-level privacy claim",
            "sampling": (
                "sort the 18,393 image IDs; generate a fresh private-seeded permutation each epoch; "
                "consume consecutive fixed batches of 8 and drop the final incomplete batch"
            ),
            "images_per_step": 8,
            "loss": "mean of eight per-image prompt-conditioned diffusion MSE losses",
            "diffusion_draws": "one uniform scheduler timestep and one Gaussian latent-noise draw per image",
            "clipping": "none",
            "DP_noise": "none",
            "optimizer_steps": 4_000,
            "total_image_exposures": 32_000,
        },
        "DP_arms": {
            "shared": {
                "implementation": "dp_training/mechanism.py",
                "empty_poisson_step": "unconditional Gaussian-noise-only optimizer update",
                "denominator": "fixed public expected batch, never realized batch size",
                "M1_matching": (
                    "M1-I8 and M1-G8 replay the identical Poisson image schedule, diffusion timesteps, "
                    "and latent-noise draws; their DP Gaussian streams are independent"
                ),
                "research_rng_only": True,
            },
            "M1-I8": {
                "privacy_unit": "image",
                "adjacency": "add_or_remove_one_image",
                "population": 18_393,
                "expected_batch": 8,
                "q": m1_q,
                "C": 0.28448700606156724,
                "sigma": 0.4007042918650512,
                "fixed_denominator": 8,
                "ideal_4000_step_bound": {"epsilon_image": 8.0, "delta_image": 1e-5},
                "patient_interpretation": "group conversion is vacuous; blocked",
            },
            "M1-G8": {
                "privacy_unit": "image",
                "adjacency": "add_or_remove_one_image",
                "population": 18_393,
                "expected_batch": 8,
                "q": m1_q,
                "C": 0.28448700606156724,
                "sigma": 0.8350657829633824,
                "fixed_denominator": 8,
                "ideal_4000_step_bound": {
                    "epsilon_image": 1.6,
                    "delta_image": 1.3265396497483431e-8,
                },
                "ideal_group_conversion": {"epsilon_patient": 8.0, "delta_patient": 1e-5, "K": 5},
            },
            "M2-P8": {
                "privacy_unit": "patient",
                "adjacency": "add_or_remove_one_complete_patient",
                "population": 8_476,
                "expected_batch": 4,
                "q": m2_q,
                "within_patient": "uniform without replacement up to four K5 images; mean before one patient clip",
                "C": 0.1997973088974048,
                "sigma": 0.4044571458362774,
                "fixed_denominator": 4,
                "ideal_4000_step_bound": {"epsilon_patient": 8.0, "delta_patient": 1e-5},
            },
        },
        "randomness_and_resume": {
            "classification": "RESEARCH_ONLY_NONCRYPTOGRAPHIC",
            "experiment_root_entropy": "256 bits from the operating-system entropy pool",
            "domain_separation": "HMAC-SHA256 labels for sampling, diffusion, and independent DP Gaussian streams",
            "public_seed_material": "LoRA initialization and evaluation/bootstrap seeds only",
            "private_seed_material": (
                "never written in plaintext or included in a public report/receipt; the experiment root and exact "
                "generator states may exist only inside the encrypted resume envelope"
            ),
            "resume_envelope": {
                "platform": "Windows DPAPI CryptProtectData",
                "scope": "CurrentUser",
                "optional_entropy_utf8": "nih-cxr14-k5-feasibility-resume-v1",
                "payload": [
                    "schema/run/arm/current committed step",
                    "LoRA and AdamW states",
                    "experiment-root and all relevant torch generator states",
                    "global CPU/CUDA RNG states",
                    "public trace head and completed event count",
                    "frozen input and environment digests",
                ],
                "atomicity": "write encrypted .new, fsync, verify decrypt/hash, then os.replace current",
                "retention": "step 0 and every 250 steps; keep current plus one previous recoverable envelope",
                "plaintext_temporary_file": "PROHIBITED",
            },
            "restart_preflight": (
                "before full K5, separately prove exact uninterrupted-4-step versus 2+resume+2 equality for all four "
                "arm semantics on public-development/synthetic fixtures, including adapter, optimizer, RNG, and trace"
            ),
            "lost_or_corrupt_envelope": (
                "do not invent a replacement seed; retain failure and restart the complete K5 matrix under a new run ID"
            ),
            "release_boundary": (
                "no research-PRNG checkpoint can yield DIRECT or GROUP/CONVERT release authorization; a registered "
                "CSPRNG path and fresh training from scratch remain mandatory"
            ),
        },
        "checkpoint_and_artifact_policy": {
            "adapter_snapshots": {
                "steps": [1_000, 2_000, 4_000],
                "format": "LoRA-only safetensors plus canonical config/manifest",
                "classification": "LOCAL_RESTRICTED_NOT_FOR_RELEASE",
                "quantitative_evaluation_checkpoint": 4_000,
                "steps_1000_2000_role": "fixed diagnostics only; cannot select, stop, or release a model",
            },
            "run_root": "code_working/_restricted_runs/nih_cxr14_k5_feasibility_v1_001",
            "public_report_root": "code_working/_reports/nih_cxr14_k5_feasibility_v1_001",
            "generated_images": (
                "lossless 8-bit RGB P256 PNG; retain locally until extraction screening; no forced grayscale conversion"
            ),
            "public_report_excludes": [
                "patient/image identifiers",
                "realized sample counts or empty-step locations",
                "losses, norms, clip decisions, gradients, updates, and optimizer tensors",
                "private seeds/RNG states and resume envelopes",
                "generated images before near-copy/extraction review",
            ],
            "never_persist": [
                "second resized raw corpus",
                "raw-image or latent cache",
                "prompt-embedding cache",
                "per-unit gradient bank",
            ],
        },
        "evaluation": {
            "generation_tasks": {
                "file": "generation_tasks.csv",
                "sha256": generation_sha,
                "conditions": [condition for condition, _, _ in CONDITIONS],
                "seeds_per_condition": 64,
                "images_per_model": 448,
                "models": ["B0", "M0", "M1-I8", "M1-G8", "M2-P8"],
                "total_final_images": 2_240,
                "seed_rule": "uint63(first 8 bytes SHA256(salt|condition|index), big-endian)",
                "salt": GENERATION_SALT,
                "sampler": "DDIM",
                "inference_steps": 50,
                "guidance_scale": 7.5,
                "negative_prompt": None,
                "batch_size": 4,
                "same_prompt_seed_pairs_across_models": True,
                "B0_timing": "generate and hash before any full K5 private optimizer update",
                "intermediate_diagnostic": (
                    "at steps 1000 and 2000 generate only indices 0 and 1 for each of seven conditions: "
                    "14 images/checkpoint/arm, excluded from every quantitative gate"
                ),
            },
            "real_reference": {
                "file": "real_reference_local.csv",
                "sha256": reference_sha,
                "classification": "LOCAL_REFERENCE_WITH_IDENTIFIERS_NOT_PUBLIC_REPORT",
                "partition": "K5 public_development only",
                "selection": (
                    "scarcity-first fixed stratum order; within stratum ascending SHA256(salt|stratum|image_id); "
                    "one image per globally distinct patient"
                ),
                "salt": REFERENCE_SALT,
                "strata": {
                    "pneumothorax": 64,
                    "pneumonia_or_consolidation": 128,
                    "pleural_effusion": 64,
                    "mass_or_nodule": 128,
                    "no_finding": 64,
                },
                "images": 448,
                "patients": 448,
                "input_to_evaluator": (
                    "apply the frozen P256 model-input resize/channel contract to real images first, so real and "
                    "generated sets have matched 256-pixel information before evaluator preprocessing"
                ),
            },
            "feature_fidelity": {
                "primary_encoder": RAD_DINO,
                "primary_small_sample_metric": {
                    "name": "RadDINO-KID",
                    "kernel": "(x dot y / 768 + 1)^3 on raw fp32 pooler features",
                    "deterministic_subsets": 100,
                    "subset_size": 50,
                    "report": "mean and standard deviation overall and for five matched strata",
                },
                "mode_metrics": {
                    "name": "PRDC",
                    "nearest_k": 5,
                    "outputs": ["precision", "recall", "density", "coverage"],
                },
                "secondary": [
                    "RadDINO Frechet distance as a descriptive point estimate only because n=448 < d=768",
                    "feature effective rank and within-condition pairwise-distance summaries",
                ],
                "not_sufficient": "Inception-only FID or any one scalar quality score",
            },
            "condition_alignment": {
                "encoder": BIOVIL_T,
                "metric": "cosine similarity between projected image and exact generation-prompt embeddings",
                "reports": "overall and each of seven conditions, paired by prompt/seed across models",
                "preflight_validity_gate": (
                    "on the 448 fixed real references, matched exact weak-label prompts must exceed a fixed cyclic "
                    "patient-distinct prompt derangement with a positive mean difference and lower 95% paired "
                    "bootstrap bound > 0"
                ),
                "boundary": "weak-label/text-image alignment, not radiologist-confirmed diagnostic correctness",
            },
            "pixel_sanity": {
                "required_decode_rate": 1.0,
                "maximum_exact_duplicate_fraction": 0.01,
                "low_contrast_definition": "grayscale pixel standard deviation < 0.02 on [0,1]",
                "maximum_low_contrast_fraction": 0.01,
                "saturation_definition": "more than 98% of grayscale pixels are <=1/255 or >=254/255",
                "maximum_saturation_fraction": 0.01,
            },
            "uncertainty": {
                "paired_stratified_bootstrap_replicates": 2_000,
                "confidence": 0.95,
                "salt": BOOTSTRAP_SALT,
                "pairing": "same generation condition and seed across model arms",
                "patient_weighting": "one patient per fixed real reference",
            },
            "qualitative_sanity": (
                "a fixed 35-image blinded contact sheet (indices 0..4 per condition) may flag gross non-radiograph, "
                "anatomy, marker/device, text, or geometry failures; it cannot establish clinical utility or select "
                "a checkpoint"
            ),
            "downstream_utility": (
                "not estimated from this 448-image one-seed feasibility screen; augmentation classifiers and real "
                "held-out patient utility belong to the preregistered K10 confirmatory phase"
            ),
        },
        "decision_rules": {
            "TRAINING_EXECUTION_PASS": [
                "all four arms start from the same LoRA digest and complete exactly 4000 fixed steps",
                "all parameters/optimizer states remain finite; no outcome-dependent stop or retry occurs",
                "M1 arms have identical private schedule/diffusion commitments and independent Gaussian streams",
                "all three DP traces contain exactly 4000 valid events and both accountants replay the frozen values",
                "every retained adapter/config/environment digest and restricted/public artifact boundary verifies",
            ],
            "M0_DOMAIN_ADAPTATION_PASS": [
                "M0 passes all pixel-sanity thresholds",
                "RadDINO-KID(M0, real) < RadDINO-KID(B0, real) and the upper 95% paired stratified bootstrap bound of the difference is < 0",
                "BioViL-T mean alignment(M0) > alignment(B0) and the lower 95% paired stratified bootstrap bound of the difference is > 0",
                "M0 RadDINO coverage is strictly greater than B0 coverage",
            ],
            "DP_ARM_UTILITY_COLLAPSE": (
                "classify an arm as COLLAPSED if pixel-sanity fails, or if its RadDINO coverage or feature effective "
                "rank is <=10% of M0; otherwise report its full privacy-utility degradation without a pass/fail ranking"
            ),
            "K10_PROGRESSION": (
                "requires TRAINING_EXECUTION_PASS, evaluator/restart preflight PASS, and M0_DOMAIN_ADAPTATION_PASS. "
                "A poor or collapsed DP-arm utility result is retained and does not authorize dropping/substituting "
                "that K10 arm. If M0 fails, retain the K5 null and block K10 pending a separately frozen redesign."
            ),
            "P512": (
                "not auto-activated by K5 results; any P512 study requires a separate amendment before K10 and may "
                "not be selected from a preferred privacy-attack ordering"
            ),
        },
        "resource_and_failure_policy": {
            "target_hardware": "single NVIDIA GeForce RTX 3070 8 GiB, sequential arms",
            "measured_four_step_loop_only_hours_projected": {
                "M1-I8": 4.878,
                "M1-G8": 4.836,
                "M2-P8": 2.207,
                "three_DP_arms_total": 11.922,
            },
            "measured_B0_generation": {
                "batch": 4,
                "P256_DDIM_steps": 50,
                "seconds_per_image": 1.08524,
                "peak_cuda_bytes": 3_146_087_424,
                "role": "in-memory planning benchmark only, not a quality result",
            },
            "planning_envelope": {
                "training_and_checkpointing": "approximately 13-16 wall-clock hours including M0 and non-loop overhead",
                "final_and_diagnostic_generation": "approximately 1 hour at the measured base rate; allow 2 hours",
                "evaluation_and_verification": "allow 1-3 hours",
                "total": "plan 16-21 hours across resumable arm-level sessions; do not promise an exact completion time",
            },
            "realization_limits": {
                "M1_maximum_image_units_per_step": 64,
                "M2_maximum_patient_units_per_step": 32,
                "M2_maximum_raw_images_per_step": 128,
                "M1_4000_step_union_bound_above_limit": binomial_tail_union_bound(18_393, m1_q, 64, 4_000),
                "M2_4000_step_union_bound_above_limit": binomial_tail_union_bound(8_476, m2_q, 32, 4_000),
                "action": "abort without resampling; retain failure; do not loosen from observed outcomes",
            },
            "storage": {
                "conservative_new_artifact_and_model_cache_estimate_gib": 2.5,
                "minimum_free_space_before_each_arm_gib": 15,
                "current_design": "no second image corpus and no persistent latent/gradient bank",
                "if_more_storage_is_scientifically_required": (
                    "report the reason and measured projection first; do not silently discard evidence to meet a cap"
                ),
            },
            "recoverable_failures": [
                "process interruption after a verified resume boundary",
                "evaluation interruption with already hashed images",
            ],
            "hard_failures": [
                "non-finite loss/gradient/update/weight or optimizer state",
                "frozen hash/config/environment mismatch",
                "corrupt DPAPI envelope or public trace mismatch",
                "realized-unit resource limit exceeded",
                "free space below 15 GiB before starting an arm",
            ],
            "prohibited_recovery": [
                "resample an inconvenient/empty/oversized Poisson event",
                "restart only an unfavorable arm under a new private seed",
                "continue from an unverified checkpoint",
                "change C, sigma, q, step count, learning rate, data, or arm matrix from K5 outcomes",
            ],
        },
        "required_before_full_execution": [
            "this protocol and its generation/reference manifests pass an implementation-independent verifier",
            "RAD-DINO and BioViL-T exact weights/dependencies pass the public-development evaluator preflight",
            "DPAPI checkpoint/restart exact-equivalence preflight passes for M0 and all DP semantics",
            "the full runner source and environment manifest are hash-frozen and independently checked",
            "at least 15 GiB free space and an idle compatible CUDA device are confirmed",
            "report to the user before launching the long run",
        ],
        "claim_limits": [
            "K5 is a one-seed feasibility study and cannot support confirmatory superiority or clinical deployment claims",
            "public NIH data are a reproducible restricted-cohort proxy, not proof of handling confidential hospital data",
            "research-PRNG accounting is ideal-mechanism research evidence, not release authorization",
            "quality metrics and weak-label alignment do not establish diagnostic truth, radiologist equivalence, or clinical safety",
            "attack failure, when later measured under K10, cannot strengthen the formal DP guarantee",
            "B0/M0/M1/M2 are benchmark arms, not a jointly authorized multi-model release",
        ],
        "method_basis": {
            "CheXGenBench": {
                "paper": "CheXGenBench: A Unified Benchmark For Fidelity, Privacy and Utility of Synthetic Chest Radiographs",
                "venue": "Transactions on Machine Learning Research (2026)",
                "arxiv": "2505.10496v4",
                "use": "RadDINO fidelity/PRDC and BioViL-T alignment motivate the diagnostic metric suite",
            },
            "RoentGen": {
                "arxiv": "2211.12737",
                "use": "supports domain-adapted latent diffusion and joint fidelity/diversity evaluation for chest X-rays",
            },
            "boundary": "the present K5 screen adapts these ideas to a smaller DP-feasibility comparison; it does not claim benchmark equivalence",
        },
        "next_if_protocol_passes": (
            "report first; then implement and pass the bounded evaluator plus checkpoint/resume preflight. Do not start "
            "the 4000-step K5 matrix in the same unreported gate."
        ),
    }


def validate_sources(paths: dict[str, Path]) -> None:
    expected = {
        "upstream": UPSTREAM_PROTOCOL_SHA256,
        "manifest": K5_MANIFEST_SHA256,
        "preprocessing": PREPROCESSING_SHA256,
        "mechanism": MECHANISM_SHA256,
        "dryrun_protocol": DRYRUN_PROTOCOL_SHA256,
        "dryrun_public": DRYRUN_PUBLIC_REPORT_SHA256,
        "dryrun_independent": DRYRUN_INDEPENDENT_SHA256,
    }
    for key, digest in expected.items():
        require(paths[key].is_file(), f"missing source: {paths[key]}")
        require(sha256_file(paths[key]) == digest, f"source hash drift: {key}")


def main() -> int:
    paths = root_paths()
    validate_sources(paths)
    require(not paths["report"].exists(), "refusing to overwrite an existing protocol report directory")
    paths["report"].mkdir(parents=True)

    generation = write_generation_tasks(paths["generation"])
    reference, reference_counts = select_references(paths["manifest"], paths["reference"])
    generation_sha = sha256_file(paths["generation"])
    reference_sha = sha256_file(paths["reference"])
    protocol = build_protocol(generation_sha, reference_sha)
    write_json(paths["protocol"], protocol)

    disk = shutil.disk_usage(paths["root"])
    report = {
        "schema": REPORT_SCHEMA,
        "status": "PASS_PROTOCOL_BUILD_FULL_K5_NOT_STARTED",
        "protocol": paths["protocol"].name,
        "protocol_sha256": sha256_file(paths["protocol"]),
        "generation_tasks": {
            "file": paths["generation"].name,
            "sha256": generation_sha,
            "rows": len(generation),
            "conditions": dict(Counter(row["condition"] for row in generation)),
            "unique_seeds_within_condition": all(
                len({row["seed"] for row in generation if row["condition"] == condition}) == 64
                for condition, _, _ in CONDITIONS
            ),
        },
        "real_reference": {
            "file": paths["reference"].name,
            "sha256": reference_sha,
            "classification": "LOCAL_WITH_IDENTIFIERS",
            "rows": len(reference),
            "unique_images": len({row["image_id"] for row in reference}),
            "unique_patients": len({row["patient_id"] for row in reference}),
            "strata": reference_counts,
        },
        "environment_snapshot_not_frozen_result": {
            "python": platform.python_version(),
            "free_bytes": disk.free,
            "free_gib": disk.free / 1024**3,
        },
        "optimizer_or_model_generation_performed": False,
        "next": protocol["next_if_protocol_passes"],
    }
    write_json(paths["builder_report"], report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
