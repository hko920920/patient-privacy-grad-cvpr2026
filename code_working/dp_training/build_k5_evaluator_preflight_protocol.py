#!/usr/bin/env python3
"""Freeze the K5 evaluator preflight before computing encoder outputs."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA = "nih-cxr14-k5-evaluator-preflight-protocol/v1"
MAIN_PROTOCOL_SHA256 = "2234B3BA8701565B768A11AB5196B2F696788900D07254833997D7823489E9DA"
REFERENCE_SHA256 = "99972FC6BAB402F620D2F6B142348CF6A3F16B60BFD63F41EB877FFDDAAA898E"
GENERATION_SHA256 = "B0379A2A1EBDFB8758AAD865D6F709D803C7632D2697A7BB9A5DFD9C23F2124B"
PREPROCESSING_SHA256 = "0ED9E434764BDB2D542BB16AB951F856CB5B82D4F2E6B9F25F6C554F1076669D"
RAD_DINO_REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"
RAD_DINO_SHA256 = "DBFB9F54459C38773505DE64A6AB7807BDCB392610FE1E697166342E43FB91AE"
BIOVIL_REVISION = "692f09e9be1bfe5fdd5f3efdd0e1eca7d2c10b23"
BIOVIL_TEXT_SHA256 = "70BED344872E0A4F4DCA2352564C44F44CB7A9CFCAC3D878BE2DD08A5FFF9ABA"
BIOVIL_IMAGE_SHA256 = "B2399D73DC2A68B9F3A1950E864AE0ECD24093FB07AA459D7E65807EBDC0FB77"
BOOTSTRAP_SALT = "nih-cxr14-k5-feasibility-bootstrap-v1"

PROMPTS = {
    "no_finding": "a frontal posteroanterior chest radiograph with no labeled finding",
    "pneumothorax": "a frontal posteroanterior chest radiograph with radiographic findings of pneumothorax",
    "pneumonia": "a frontal posteroanterior chest radiograph with radiographic findings of pneumonia",
    "consolidation": "a frontal posteroanterior chest radiograph with radiographic findings of consolidation",
    "pleural_effusion": "a frontal posteroanterior chest radiograph with radiographic findings of pleural effusion",
    "mass_opacity": "a frontal posteroanterior chest radiograph with radiographic findings of mass opacity",
    "nodule_opacity": "a frontal posteroanterior chest radiograph with radiographic findings of nodule opacity",
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


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def matched_condition(row: dict[str, str]) -> str:
    stratum = row["reference_stratum"]
    labels = {label for label in row["finding_labels"].split("|") if label}
    if stratum == "pneumonia_or_consolidation":
        require(bool(labels & {"Pneumonia", "Consolidation"}), "missing pneumonia/consolidation label")
        return "pneumonia" if "Pneumonia" in labels else "consolidation"
    if stratum == "mass_or_nodule":
        require(bool(labels & {"Mass", "Nodule"}), "missing mass/nodule label")
        # Fixed tie rule for the six selected records carrying both labels.
        return "mass_opacity" if "Mass" in labels else "nodule_opacity"
    require(stratum in {"no_finding", "pneumothorax", "pleural_effusion"}, "unknown stratum")
    return stratum


def build_tasks(reference_path: Path, output_path: Path) -> tuple[list[dict[str, str]], int]:
    with reference_path.open("r", encoding="utf-8", newline="") as handle:
        source = list(csv.DictReader(handle))
    require(len(source) == 448, "reference row count drift")
    enriched: list[dict[str, str]] = []
    for row in source:
        condition = matched_condition(row)
        enriched.append({**row, "matched_condition": condition, "matched_prompt": PROMPTS[condition]})
    enriched.sort(key=lambda row: (row["matched_condition"], row["selection_key_sha256"], row["image_id"]))
    counts = Counter(row["matched_condition"] for row in enriched)
    shift = max(counts.values())
    require(shift == 89, "derangement shift drift")

    tasks: list[dict[str, str]] = []
    for index, row in enumerate(enriched):
        donor = enriched[(index + shift) % len(enriched)]
        require(row["patient_id"] != donor["patient_id"], "derangement patient collision")
        require(row["matched_condition"] != donor["matched_condition"], "derangement condition collision")
        tasks.append(
            {
                "eval_index": str(index),
                "reference_stratum": row["reference_stratum"],
                "image_id": row["image_id"],
                "patient_id": row["patient_id"],
                "image_filename": row["image_filename"],
                "selection_key_sha256": row["selection_key_sha256"],
                "matched_condition": row["matched_condition"],
                "matched_prompt": row["matched_prompt"],
                "deranged_donor_patient_id": donor["patient_id"],
                "deranged_condition": donor["matched_condition"],
                "deranged_prompt": donor["matched_prompt"],
            }
        )
    require(Counter(row["deranged_condition"] for row in tasks) == counts, "derangement frequency drift")
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tasks[0]))
        writer.writeheader()
        writer.writerows(tasks)
    return tasks, shift


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    source_dir = root / "_reports" / "nih_cxr14_k5_feasibility_protocol_v1_001"
    output_dir = root / "_reports" / "nih_cxr14_k5_evaluator_preflight_protocol_v1_001"
    require(not output_dir.exists(), "refusing to overwrite evaluator preflight protocol")
    paths = {
        "main_protocol": source_dir / "protocol.json",
        "reference": source_dir / "real_reference_local.csv",
        "generation": source_dir / "generation_tasks.csv",
        "preprocessing": root
        / "_reports"
        / "nih_cxr14_sd21_ppmark_interface_v1_001"
        / "preprocessing_contract.json",
        "rad_weights": Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--microsoft--rad-dino"
        / "snapshots"
        / RAD_DINO_REVISION
        / "model.safetensors",
        "biovil_text": Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--microsoft--BiomedVLP-BioViL-T"
        / "snapshots"
        / BIOVIL_REVISION
        / "model.safetensors",
        "biovil_image": Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--microsoft--BiomedVLP-BioViL-T"
        / "snapshots"
        / BIOVIL_REVISION
        / "biovil_t_image_model_proj_size_128.pt",
    }
    expected = {
        "main_protocol": MAIN_PROTOCOL_SHA256,
        "reference": REFERENCE_SHA256,
        "generation": GENERATION_SHA256,
        "preprocessing": PREPROCESSING_SHA256,
        "rad_weights": RAD_DINO_SHA256,
        "biovil_text": BIOVIL_TEXT_SHA256,
        "biovil_image": BIOVIL_IMAGE_SHA256,
    }
    for name, digest in expected.items():
        require(paths[name].is_file(), f"missing frozen source: {name}")
        require(sha256_file(paths[name]) == digest, f"hash drift: {name}")

    output_dir.mkdir(parents=True)
    task_path = output_dir / "real_evaluator_tasks_local.csv"
    tasks, shift = build_tasks(paths["reference"], task_path)
    counts = dict(Counter(row["matched_condition"] for row in tasks))
    task_sha = sha256_file(task_path)
    bootstrap_seed = int.from_bytes(
        hashlib.sha256(f"{BOOTSTRAP_SALT}|biovil-real-validity".encode("utf-8")).digest()[:8], "big"
    )
    protocol = {
        "schema": SCHEMA,
        "status": "FROZEN_BEFORE_REAL_ENCODER_OUTPUT",
        "scope": "PUBLIC_DEVELOPMENT_EVALUATOR_VALIDITY_NOT_MODEL_QUALITY_NOT_CLINICAL_VALIDATION",
        "upstream": {name: digest for name, digest in expected.items()},
        "local_tasks": {
            "file": task_path.name,
            "classification": "LOCAL_WITH_PATIENT_AND_IMAGE_IDENTIFIERS_NOT_PUBLIC_REPORT",
            "sha256": task_sha,
            "rows": 448,
            "matched_condition_counts": counts,
            "mapping": {
                "pneumonia_or_consolidation": "Pneumonia if present, otherwise Consolidation",
                "mass_or_nodule": "Mass if present (including Mass+Nodule), otherwise Nodule",
                "other_three_strata": "one-to-one",
            },
            "derangement": {
                "sort": "matched_condition, selection_key_sha256, image_id",
                "rule": "deranged prompt donor is row (i + shift) modulo 448",
                "shift": shift,
                "preserves_prompt_frequencies": True,
                "every_donor_patient_distinct": True,
                "every_deranged_condition_different": True,
            },
        },
        "runtime": {
            "device": "single CUDA device",
            "torch_inference_mode": True,
            "model_eval": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "allow_tf32": False,
            "no_encoder_feature_or_per_image_score_persistence": True,
        },
        "preprocessing": {
            "shared_information_boundary": "first reduce every real input to frozen full-field P256 grayscale with Pillow 11.3 LANCZOS and reducing_gap=None",
            "RAD_DINO": "convert the P256 grayscale image to replicated RGB, then pinned BitImageProcessor with use_fast=False, shortest-edge/crop 518 and published normalization",
            "BioViL_T": "keep P256 grayscale, then hi-ml-multimodal inference transform resize=512, center_crop=448, bilinear, ToTensor, ExpandChannels",
            "future_generated_BioViL_T": "convert retained P256 RGB PNG to Pillow L before the same BioViL-T transform",
        },
        "encoders": {
            "RAD_DINO": {
                "model_id": "microsoft/rad-dino",
                "revision": RAD_DINO_REVISION,
                "weights_sha256": RAD_DINO_SHA256,
                "output": "raw fp32 pooler_output shape [n,768]",
            },
            "BioViL_T_image": {
                "model_id": "microsoft/BiomedVLP-BioViL-T",
                "revision": BIOVIL_REVISION,
                "weights_sha256": BIOVIL_IMAGE_SHA256,
                "implementation": "hi-ml-multimodal 0.2.2 ImageModel RESNET50_MULTI_IMAGE with missing-previous embedding",
                "output": "L2-normalized projected_global_embedding shape [n,128]",
            },
            "BioViL_T_text": {
                "model_id": "microsoft/BiomedVLP-BioViL-T",
                "revision": BIOVIL_REVISION,
                "weights_sha256": BIOVIL_TEXT_SHA256,
                "implementation": "hash-pinned remote CXRBertModel code through Transformers AutoModel",
                "output": "L2-normalized projected CLS shape [7,128]",
                "known_load_boundary": "the checkpoint BERT pooler is unused by the projected-CLS architecture and must be reported, not treated as a silent failure",
            },
        },
        "validity_gates": {
            "weights_and_inputs": "all frozen hashes, 448 decodes, exact task mapping and no identifier in public output",
            "features": "exact shapes, all finite, nonzero row norms, RAD-DINO effective rank > 2, and first-eight replay maximum absolute drift <=1e-5",
            "BioViL_T_alignment": "mean cosine(image,matched)-cosine(image,deranged) > 0 and percentile-bootstrap lower 95% bound > 0",
            "decision": "all gates must pass before BioViL-T alignment can be used in K5; failure blocks full K5 pending a separately frozen redesign",
        },
        "bootstrap": {
            "replicates": 2_000,
            "confidence": 0.95,
            "strata": "five reference_stratum values; sample with replacement at original stratum size",
            "generator": "numpy.random.Generator(PCG64)",
            "seed": bootstrap_seed,
            "interval": "numpy quantile [0.025,0.975], method='linear'",
        },
        "future_metric_implementation": {
            "numeric_dtype": "float64 after encoder fp32 output",
            "KID": "unbiased polynomial MMD; kernel (x dot y / 768 + 1)^3; 100 deterministic without-replacement subsets of 50; same generated index subsets across model arms",
            "PRDC": "Euclidean distances on raw RAD-DINO features; standard precision/recall/density/coverage with nearest_k=5 and self-distance excluded for radii",
            "Frechet": "descriptive only; float64 means/covariances and scipy.linalg.sqrtm with real-part acceptance only when maximum imaginary magnitude <=1e-6",
            "effective_rank": "exp Shannon entropy of nonnegative covariance eigenvalues normalized by their positive sum",
            "pixel_sanity": "use the exact thresholds in the main K5 protocol before any feature metric",
        },
        "output_policy": {
            "public": "aggregate counts, timing, environment/weight/source hashes, overall/condition/stratum summaries and gate decision",
            "prohibited": [
                "patient/image identifiers",
                "per-image embeddings or scores",
                "raw or P256 reference images",
                "any generated-image quality or K5 optimizer claim",
            ],
        },
        "next_on_pass": "freeze source hashes, independently verify the evaluator result, then implement exact encrypted restart preflight; do not start full K5",
    }
    protocol_path = output_dir / "protocol.json"
    write_json(protocol_path, protocol)
    report = {
        "schema": "nih-cxr14-k5-evaluator-preflight-protocol-build/v1",
        "status": "PASS_PROTOCOL_BUILD_ENCODER_OUTPUT_NOT_COMPUTED",
        "protocol_sha256": sha256_file(protocol_path),
        "local_tasks_sha256": task_sha,
        "rows": len(tasks),
        "matched_condition_counts": counts,
        "derangement_shift": shift,
        "all_deranged_conditions_different": all(
            row["matched_condition"] != row["deranged_condition"] for row in tasks
        ),
        "encoder_output_computed": False,
    }
    write_json(output_dir / "builder_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
