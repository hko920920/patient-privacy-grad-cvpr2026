#!/usr/bin/env python3
"""Independent verifier for the NIH CXR SD2.1/PP-Mark interface report.

This file deliberately imports neither the model-input adapter nor the gate
implementation.  It independently rebuilds the metadata-only sample and the
exact preprocessing commitments, then checks the recorded GPU interface facts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, __version__ as pillow_version


SELECTION_SALT = "nih-cxr14-sd21-ppmark-interface-v1"
EXPECTED_MANIFEST_SHA256 = "2D749FB7B70823114A69FD55921277B0D0FF2F9AEC8FECD239159C553F3A0C06"
EXPECTED_INVENTORY_SHA256 = "AD34344F962FAC7052114A3486D42D2B48CB5C9F27DDDC70365E07B715F68CC3"
EXPECTED_CONTENT_SET_SHA256 = "E983A21B9B8558CE38F1AD5BA7C0F6BC51787CC7CBA739399ACE1976E2FB068E"
EXPECTED_MODEL_HASHES = {
    "text_encoder/model.fp16.safetensors": "681C555376658C81DC273F2D737A2AEB23DDB6D1D8E5B3A7064636D359A22668",
    "unet/diffusion_pytorch_model.fp16.safetensors": "28EC9CF3B239C0751C201B1F6FB46B551DF5862731B30A37AA1360101CB3FBAB",
    "vae/diffusion_pytorch_model.fp16.safetensors": "3E4C08995484EE61270175E9E7A072B66A6E4EEB5F0C266667FE1F45B90DAF9A",
}
LABEL_ORDER = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
)
LABEL_TEXT = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Effusion": "pleural effusion",
    "Infiltration": "infiltration",
    "Mass": "mass opacity",
    "Nodule": "nodule opacity",
    "Pneumonia": "pneumonia",
    "Pneumothorax": "pneumothorax",
    "Consolidation": "consolidation",
    "Edema": "edema",
    "Emphysema": "emphysema",
    "Fibrosis": "fibrosis",
    "Pleural_Thickening": "pleural thickening",
    "Hernia": "hernia",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest().upper()


def tokens(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split("|") if part.strip())


def prompt(labels: tuple[str, ...]) -> str:
    prefix = "a frontal posteroanterior chest radiograph"
    if labels == ("No Finding",):
        return prefix + " with no labeled finding"
    label_set = set(labels)
    ordered = [label for label in LABEL_ORDER if label in label_set]
    require(len(ordered) == len(labels), f"unknown or duplicate labels: {labels}")
    descriptions = [LABEL_TEXT[label] for label in ordered]
    if len(descriptions) == 1:
        joined = descriptions[0]
    elif len(descriptions) == 2:
        joined = " and ".join(descriptions)
    else:
        joined = ", ".join(descriptions[:-1]) + ", and " + descriptions[-1]
    return prefix + " with radiographic findings of " + joined


def read_inputs(manifest_path: Path, inventory_path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]], str]:
    records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["partition"] != "public_development":
                continue
            records.append(
                {
                    "image_id": row["image_id"],
                    "patient_id": row["patient_id"],
                    "partition": row["partition"],
                    "view": row["view"],
                    "labels": tokens(row["finding_labels"]),
                    "groups": tokens(row["primary_groups"]),
                }
            )
    inventory: dict[str, dict[str, str]] = {}
    with inventory_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            require(row["image_id"] not in inventory, "duplicate inventory image")
            inventory[row["image_id"]] = row
    digest = hashlib.sha256()
    for image_id in sorted(inventory):
        digest.update(image_id.encode("ascii"))
        digest.update(b"\x00")
        digest.update(inventory[image_id]["sha256"].upper().encode("ascii"))
        digest.update(b"\n")
    return records, inventory, digest.hexdigest().upper()


def select_records(
    records: list[dict[str, Any]], inventory: dict[str, dict[str, str]]
) -> list[tuple[str, dict[str, Any], dict[str, str]]]:
    predicates: list[tuple[str, Callable[[dict[str, Any], dict[str, str]], bool]]] = [
        ("rgba_positive", lambda r, i: i["mode"] == "RGBA" and r["labels"] != ("No Finding",)),
        ("rgba_no_finding", lambda r, i: i["mode"] == "RGBA" and r["labels"] == ("No Finding",)),
        ("l_no_finding", lambda r, i: i["mode"] == "L" and r["labels"] == ("No Finding",)),
        ("l_pneumothorax", lambda r, i: i["mode"] == "L" and "pneumothorax" in r["groups"]),
        (
            "l_pneumonia_or_consolidation",
            lambda r, i: i["mode"] == "L" and "pneumonia_or_consolidation" in r["groups"],
        ),
        ("l_pleural_effusion", lambda r, i: i["mode"] == "L" and "pleural_effusion" in r["groups"]),
        ("l_mass_or_nodule", lambda r, i: i["mode"] == "L" and "mass_or_nodule" in r["groups"]),
        ("l_multilabel", lambda r, i: i["mode"] == "L" and len(r["labels"]) >= 3),
    ]
    selected: list[tuple[str, dict[str, Any], dict[str, str]]] = []
    used_patients: set[str] = set()
    for stratum, predicate in predicates:
        candidates = []
        for record in records:
            row = inventory[record["image_id"]]
            if record["patient_id"] not in used_patients and predicate(record, row):
                candidates.append(
                    (
                        stable_hash(SELECTION_SALT, stratum, record["image_id"]),
                        record["image_id"],
                        record,
                        row,
                    )
                )
        require(bool(candidates), f"no independent candidate for {stratum}")
        candidates.sort(key=lambda item: (item[0], item[1]))
        _, _, record, row = candidates[0]
        selected.append((stratum, record, row))
        used_patients.add(record["patient_id"])
    return selected


def preprocess(path: Path, size: int) -> tuple[Image.Image, str]:
    with Image.open(path) as source:
        source.load()
        require(source.format == "PNG", f"not PNG: {path.name}")
        require(source.size == (1024, 1024), f"wrong geometry: {path.name}")
        native_mode = source.mode
        if native_mode == "L":
            gray = source.copy()
        elif native_mode == "RGBA":
            rgba = np.asarray(source, dtype=np.uint8)
            require(np.array_equal(rgba[:, :, 0], rgba[:, :, 1]), "RGBA R/G mismatch")
            require(np.array_equal(rgba[:, :, 0], rgba[:, :, 2]), "RGBA R/B mismatch")
            require(bool(np.all(rgba[:, :, 3] == 255)), "RGBA nonopaque alpha")
            gray = Image.fromarray(np.ascontiguousarray(rgba[:, :, 0]))
        else:
            raise RuntimeError(f"unsupported mode: {native_mode}")
    resized = gray.resize((size, size), resample=Image.Resampling.LANCZOS, reducing_gap=None)
    return Image.merge("RGB", (resized, resized, resized)), native_mode


def digests(image: Image.Image) -> tuple[str, str]:
    pixels = np.asarray(image, dtype=np.uint8)
    pixel_prefix = f"RGB|{image.width}|{image.height}|".encode("ascii")
    pixel_sha = hashlib.sha256(pixel_prefix + pixels.tobytes(order="C")).hexdigest().upper()
    normalized = np.ascontiguousarray(
        np.transpose(pixels.astype(np.float32) / 127.5 - 1.0, (2, 0, 1))
    )
    tensor_prefix = f"{normalized.dtype}|{normalized.shape}|".encode("ascii")
    tensor_sha = hashlib.sha256(
        tensor_prefix + normalized.tobytes(order="C")
    ).hexdigest().upper()
    return pixel_sha, tensor_sha


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    manifest = root / "_data/derived/nih_cxr14_pa_target_enriched_v1/k10_private.csv"
    inventory_path = root / "_data/raw/nih_cxr14_pa_k10_plus_census_v1/content_inventory_private.csv"
    image_root = root / "_data/raw/nih_cxr14_pa_k10_plus_census_v1/images"
    report_dir = root / "_reports/nih_cxr14_sd21_ppmark_interface_v1_001"
    report_path = report_dir / "report.json"
    output_path = report_dir / "independent_verification.json"

    require(sha_file(manifest) == EXPECTED_MANIFEST_SHA256, "manifest digest mismatch")
    require(sha_file(inventory_path) == EXPECTED_INVENTORY_SHA256, "inventory digest mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_sha256 = sha_file(report_path)
    require(report["status"] == "PASS_FOR_PILOT_INTERFACE", "interface gate did not pass")
    records, inventory, content_set = read_inputs(manifest, inventory_path)
    require(content_set == EXPECTED_CONTENT_SET_SHA256, "content-set digest mismatch")
    require(len(records) == 4831, "public-development K10 record count mismatch")
    require(len({row["patient_id"] for row in records}) == 1816, "patient count mismatch")
    modes = Counter(inventory[row["image_id"]]["mode"] for row in records)
    require(modes == Counter({"L": 4804, "RGBA": 27}), "native-mode counts mismatch")

    selected = select_records(records, inventory)
    selected_csv = list(
        csv.DictReader(
            (report_dir / "selected_interface_records_private.csv").open(
                "r", encoding="utf-8", newline=""
            )
        )
    )
    require(len(selected_csv) == len(selected) == 8, "selected-row count mismatch")
    for csv_row, (stratum, record, inv) in zip(selected_csv, selected):
        require(csv_row["stratum"] == stratum, "selected stratum mismatch")
        require(csv_row["image_id"] == record["image_id"], "selected image mismatch")
        require(csv_row["patient_id"] == record["patient_id"], "selected patient mismatch")
        require(csv_row["native_mode"] == inv["mode"], "selected native mode mismatch")
        require(csv_row["prompt"] == prompt(record["labels"]), "selected prompt mismatch")
        require(sha_file(image_root / record["image_id"]) == inv["sha256"].upper(), "raw digest mismatch")

    reported_preprocessing = {
        (row["stratum"], int(row["resolution"])): row
        for row in report["preprocessing_profiles"]
    }
    profile_commitments: dict[str, str] = {}
    verified_rows = 0
    for size in (256, 512):
        commitment = hashlib.sha256()
        for stratum, record, inv in selected:
            image, native_mode = preprocess(image_root / record["image_id"], size)
            pixel_sha, tensor_sha = digests(image)
            row = reported_preprocessing[(stratum, size)]
            require(row["image_id"] == record["image_id"], "preprocessing image mismatch")
            require(row["native_mode"] == native_mode == inv["mode"], "preprocessing mode mismatch")
            require(row["pixel_sha256"] == pixel_sha, "preprocessed pixel digest mismatch")
            require(row["normalized_tensor_sha256"] == tensor_sha, "tensor digest mismatch")
            require(row["repeat_exact"] is True, "preprocessing replay did not pass")
            expected_prompt = prompt(record["labels"])
            require(row["prompt"] == expected_prompt, "preprocessing prompt mismatch")
            commitment.update(
                f"{size}|{record['image_id']}|{pixel_sha}|{tensor_sha}|{expected_prompt}\n".encode(
                    "utf-8"
                )
            )
            verified_rows += 1
        profile_commitments[f"P{size}"] = commitment.hexdigest().upper()
    require(profile_commitments == report["profile_commitments"], "profile commitment mismatch")

    for relative, expected in EXPECTED_MODEL_HASHES.items():
        row = report["model"]["critical_hashes"][relative]
        require(row["status"] == "PASS", f"model hash status failed: {relative}")
        require(row["expected_sha256"] == expected, f"model expected hash changed: {relative}")
        require(row["actual_sha256"] == expected, f"model actual hash changed: {relative}")
    for row in report["vae"]["profiles"]:
        size = int(row["resolution"])
        require(row["latent_shape"] == [1, 4, size // 8, size // 8], "VAE shape mismatch")
        require(row["finite"] and row["repeat_exact"], "VAE deterministic/finite check failed")
    lora = report["lora_gradient"]
    require(lora["optimizer_created"] is False and lora["optimizer_step"] is False, "optimizer was used")
    require(lora["dp_noise"] is False and lora["accounting"] is False, "DP operation was used")
    require(lora["parameters_unchanged"] is True, "adapter parameters changed")
    require(lora["adapter_before_sha256"] == lora["adapter_after_sha256"], "adapter digest mismatch")
    require(lora["trainable_parameters"] == 1_659_904, "trainable parameter count mismatch")
    for row in lora["profiles"]:
        require(row["finite"] and row["repeat_exact"], "gradient deterministic/finite check failed")
        require(row["missing_gradient_tensors"] == 0, "missing LoRA gradients")
        require(len(row["gradient_sha256"]) == 64, "gradient digest malformed")

    preprocessing_pixels = {
        (row["stratum"], int(row["resolution"])): row["pixel_sha256"]
        for row in report["preprocessing_profiles"]
    }
    for row in report["ppmark"]["profiles"]:
        size = int(row["resolution"])
        require(row["latent_shape"] == [4, size // 8, size // 8], "PP-Mark shape mismatch")
        require(row["finite"] and row["repeat_exact"], "PP-Mark deterministic/finite check failed")
        require(
            row["input_pixel_sha256"] == preprocessing_pixels[(row["stratum"], size)],
            "PP-Mark did not consume the committed preprocessing pixels",
        )
    forbidden = report["prohibited_operations"]
    require(not any(forbidden.values()), "a prohibited operation is recorded as true")

    result = {
        "schema": "nih-cxr14-sd21-interface-independent-verification/v1",
        "status": "PASS",
        "verifier_independence": (
            "Does not import nih_cxr14_model_input or the interface-gate implementation."
        ),
        "verifier_sha256": sha_file(Path(__file__).resolve()),
        "report_sha256": report_sha256,
        "manifest_sha256": sha_file(manifest),
        "inventory_sha256": sha_file(inventory_path),
        "content_set_sha256": content_set,
        "public_development_records": len(records),
        "public_development_patients": len({row["patient_id"] for row in records}),
        "selected_records_rebuilt": len(selected),
        "preprocessing_rows_recomputed": verified_rows,
        "profile_commitments": profile_commitments,
        "model_component_hashes_checked": len(EXPECTED_MODEL_HASHES),
        "vae_rows_checked": len(report["vae"]["profiles"]),
        "lora_profiles_checked": len(lora["profiles"]),
        "ppmark_rows_checked": len(report["ppmark"]["profiles"]),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": pillow_version,
        },
        "scope_limit": (
            "Independent report/input/interface verification only; not a rerun of model "
            "training, DP, medical utility, attacks, PP-Mark calibration, or release."
        ),
    }
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("NIH_CXR14_SD21_INTERFACE_INDEPENDENT: PASS")
    print(f"output={output_path}")
    print(f"output_sha256={sha_file(output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
