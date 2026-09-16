#!/usr/bin/env python3
"""Implementation-independent verifier for the frozen K5 feasibility protocol.

This verifier deliberately imports neither the protocol builder nor the DP
mechanism.  It reconstructs the public generation tasks and the local,
patient-distinct reference set from independently stated constants, checks the
upstream accounting entries, and performs an in-memory Windows DPAPI probe.
It performs no model load, image generation, or optimizer update.
"""

from __future__ import annotations

import csv
import ctypes
import hashlib
import json
import math
import os
import platform
import secrets
import shutil
import subprocess
from collections import Counter
from ctypes import wintypes
from pathlib import Path
from typing import Any, Callable


SCHEMA = "nih-cxr14-k5-feasibility-protocol-independent-verification/v1"
PROTOCOL_SHA256 = "2234B3BA8701565B768A11AB5196B2F696788900D07254833997D7823489E9DA"
UPSTREAM_PROTOCOL_SHA256 = "F2757DC7EBC7488B6A9DD0F69227419EA16BB9A3E9BD4434514CB19AFD2B4018"
K5_MANIFEST_SHA256 = "DC49D82E497EAA6940DCF92C8F773179B8F7A838057A82AE0C26A69CA785BFC5"
PREPROCESSING_SHA256 = "0ED9E434764BDB2D542BB16AB951F856CB5B82D4F2E6B9F25F6C554F1076669D"
MECHANISM_SHA256 = "B4B1E037A3704238B06B442970EAB082433835201B750546C8281BBFFE2B969E"
DRYRUN_PROTOCOL_SHA256 = "0509D745E8B192FC2CE7694EBDA0C36AB60005D704864D27A44681FD6E8432C8"
DRYRUN_PUBLIC_SHA256 = "77FA9097F1FA161E42E72B7E182F5E0A59CB47EC5D89209929DEE0E95767FB87"
DRYRUN_INDEPENDENT_SHA256 = "27D81FF7565659B3168CECCBA3A94DE31A92CA29154943559AC4510E32B283F9"
GENERATION_SHA256 = "B0379A2A1EBDFB8758AAD865D6F709D803C7632D2697A7BB9A5DFD9C23F2124B"
REFERENCE_SHA256 = "99972FC6BAB402F620D2F6B142348CF6A3F16B60BFD63F41EB877FFDDAAA898E"

GENERATION_SALT = "nih-cxr14-k5-feasibility-generation-v1"
REFERENCE_SALT = "nih-cxr14-k5-feasibility-real-reference-v1"
REFERENCE_ORDER = (
    ("pneumothorax", 64),
    ("pneumonia_or_consolidation", 128),
    ("pleural_effusion", 64),
    ("mass_or_nodule", 128),
    ("no_finding", 64),
)
CONDITIONS = (
    ("no_finding", "no_finding", "with no labeled finding"),
    ("pneumothorax", "pneumothorax", "with radiographic findings of pneumothorax"),
    ("pneumonia", "pneumonia_or_consolidation", "with radiographic findings of pneumonia"),
    ("consolidation", "pneumonia_or_consolidation", "with radiographic findings of consolidation"),
    ("pleural_effusion", "pleural_effusion", "with radiographic findings of pleural effusion"),
    ("mass_opacity", "mass_or_nodule", "with radiographic findings of mass opacity"),
    ("nodule_opacity", "mass_or_nodule", "with radiographic findings of nodule opacity"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def close(actual: float, expected: float, *, atol: float = 1e-15) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=1e-13, abs_tol=atol)


def canonical_seed(condition: str, index: int) -> int:
    message = f"{GENERATION_SALT}|{condition}|{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(message).digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def verify_generation(path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    require(sha256_file(path) == GENERATION_SHA256, "generation task hash drift")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(
            reader.fieldnames == ["task_id", "condition", "metric_stratum", "prompt", "seed"],
            "generation task schema drift",
        )
        rows = list(reader)
    require(len(rows) == 448, "generation task count drift")
    prefix = "a frontal posteroanterior chest radiograph"
    cursor = 0
    for condition, metric_stratum, suffix in CONDITIONS:
        for index in range(64):
            row = rows[cursor]
            expected = {
                "task_id": f"{condition}_{index:03d}",
                "condition": condition,
                "metric_stratum": metric_stratum,
                "prompt": f"{prefix} {suffix}",
                "seed": str(canonical_seed(condition, index)),
            }
            require(row == expected, f"generation task drift at row {cursor + 2}")
            cursor += 1
    task = protocol["evaluation"]["generation_tasks"]
    require(task["sha256"] == GENERATION_SHA256, "protocol generation hash")
    require(task["images_per_model"] == 448 and task["total_final_images"] == 2_240, "generation totals")
    require(task["models"] == ["B0", "M0", "M1-I8", "M1-G8", "M2-P8"], "generation model order")
    require(task["sampler"] == "DDIM" and task["inference_steps"] == 50, "generation sampler")
    require(task["guidance_scale"] == 7.5 and task["batch_size"] == 4, "generation settings")
    return {
        "sha256": GENERATION_SHA256,
        "rows": len(rows),
        "conditions": dict(Counter(row["condition"] for row in rows)),
        "all_rows_reconstructed": True,
        "status": "PASS",
    }


def reference_predicate(stratum: str) -> Callable[[dict[str, str]], bool]:
    if stratum == "no_finding":
        return lambda row: row["finding_labels"] == "No Finding"
    return lambda row: stratum in {value for value in row["primary_groups"].split("|") if value}


def verify_references(manifest_path: Path, reference_path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    require(sha256_file(manifest_path) == K5_MANIFEST_SHA256, "K5 manifest hash drift")
    require(sha256_file(reference_path) == REFERENCE_SHA256, "reference task hash drift")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        public_dev = [row for row in csv.DictReader(handle) if row["partition"] == "public_development"]
    require(len(public_dev) == 4_031, "K5 public-development image count")
    require(len({row["patient_id"] for row in public_dev}) == 1_816, "K5 public-development patient count")

    expected: list[dict[str, str]] = []
    used_patients: set[str] = set()
    for stratum, quota in REFERENCE_ORDER:
        predicate = reference_predicate(stratum)
        candidates = [row for row in public_dev if predicate(row)]
        candidates.sort(
            key=lambda row: hashlib.sha256(
                f"{REFERENCE_SALT}|{stratum}|{row['image_id']}".encode("utf-8")
            ).hexdigest()
        )
        chosen = 0
        for row in candidates:
            if row["patient_id"] in used_patients:
                continue
            selection_digest = hashlib.sha256(
                f"{REFERENCE_SALT}|{stratum}|{row['image_id']}".encode("utf-8")
            ).hexdigest().upper()
            expected.append(
                {
                    "reference_stratum": stratum,
                    "image_id": row["image_id"],
                    "patient_id": row["patient_id"],
                    "finding_labels": row["finding_labels"],
                    "primary_groups": row["primary_groups"],
                    "image_filename": row["image_filename"],
                    "selection_key_sha256": selection_digest,
                }
            )
            used_patients.add(row["patient_id"])
            chosen += 1
            if chosen == quota:
                break
        require(chosen == quota, f"independent reference scarcity: {stratum}")

    with reference_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == list(expected[0]), "reference schema drift")
        actual = list(reader)
    require(actual == expected, "reference manifest differs from independent selection")
    require(len(actual) == 448, "reference count")
    require(len({row["image_id"] for row in actual}) == 448, "reference image uniqueness")
    require(len({row["patient_id"] for row in actual}) == 448, "reference patient uniqueness")
    strata = dict(Counter(row["reference_stratum"] for row in actual))
    require(strata == dict(REFERENCE_ORDER), "reference stratum totals")
    frozen = protocol["evaluation"]["real_reference"]
    require(frozen["sha256"] == REFERENCE_SHA256, "protocol reference hash")
    require(frozen["images"] == 448 and frozen["patients"] == 448, "protocol reference totals")
    require(frozen["partition"] == "K5 public_development only", "reference partition boundary")
    return {
        "manifest_sha256": K5_MANIFEST_SHA256,
        "reference_sha256": REFERENCE_SHA256,
        "rows": len(actual),
        "unique_images": len({row["image_id"] for row in actual}),
        "unique_patients": len({row["patient_id"] for row in actual}),
        "strata": strata,
        "all_rows_independently_reselected": True,
        "identifiers_copied_to_public_gate_report": False,
        "status": "PASS",
    }


def find_accounting_entry(
    upstream: dict[str, Any], arm: str, cap: int, execution_role: str
) -> dict[str, Any]:
    matches = [
        entry
        for entry in upstream["accounting"]["entries"]
        if entry["arm"] == arm
        and entry["cap"] == cap
        and entry["execution_role"] == execution_role
    ]
    require(len(matches) == 1, f"upstream accounting entry: {arm}")
    return matches[0]


def verify_dp_arms(protocol: dict[str, Any], upstream: dict[str, Any]) -> dict[str, Any]:
    arms = protocol["DP_arms"]
    expected = {
        "M1-I8": find_accounting_entry(upstream, "M1-I8", 5, "one_seed_feasibility"),
        "M1-G8": find_accounting_entry(upstream, "M1-G8", 5, "one_seed_feasibility"),
        "M2-P8": find_accounting_entry(
            upstream, "M2-P8", 10, "P8_primary_P4_secondary_P2_accounting_curve"
        ),
    }
    clip_image = float(upstream["clip_norms"]["image_clip_norm"])
    clip_patient = float(upstream["clip_norms"]["patient_clip_norm"])
    for arm, entry in expected.items():
        frozen = arms[arm]
        require(frozen["population"] == entry["population"], f"{arm} population")
        require(frozen["expected_batch"] == entry["expected_batch"], f"{arm} expected batch")
        require(frozen["fixed_denominator"] == entry["expected_batch"], f"{arm} denominator")
        require(close(frozen["q"], entry["poisson_sample_rate"]), f"{arm} q")
        require(close(frozen["sigma"], entry["noise_multiplier"]), f"{arm} sigma")
        require(
            close(frozen["C"], clip_patient if arm == "M2-P8" else clip_image), f"{arm} C"
        )
    require(arms["M1-I8"]["ideal_4000_step_bound"] == {"epsilon_image": 8.0, "delta_image": 1e-5}, "M1-I8 bound")
    require(close(arms["M1-G8"]["ideal_4000_step_bound"]["epsilon_image"], 1.6), "M1-G8 epsilon")
    require(close(arms["M1-G8"]["ideal_group_conversion"]["epsilon_patient"], 8.0), "M1-G8 patient epsilon")
    require(close(arms["M1-G8"]["ideal_group_conversion"]["delta_patient"], 1e-5), "M1-G8 patient delta")
    require(arms["M2-P8"]["ideal_4000_step_bound"] == {"epsilon_patient": 8.0, "delta_patient": 1e-5}, "M2-P8 bound")
    require(arms["shared"]["research_rng_only"] is True, "research RNG boundary")
    return {
        arm: {
            "population": arms[arm]["population"],
            "q": arms[arm]["q"],
            "C": arms[arm]["C"],
            "sigma": arms[arm]["sigma"],
            "upstream_entry_recomputed": True,
        }
        for arm in ("M1-I8", "M1-G8", "M2-P8")
    } | {"status": "PASS"}


def binomial_union_bound(n: int, q: float, maximum: int, steps: int) -> float:
    """P[any Binomial(n,q) > maximum], using a recurrence distinct from the builder."""
    first = maximum + 1
    log_first = (
        math.lgamma(n + 1)
        - math.lgamma(first + 1)
        - math.lgamma(n - first + 1)
        + first * math.log(q)
        + (n - first) * math.log1p(-q)
    )
    term = math.exp(log_first)
    tail = term
    for k in range(first, n):
        term *= ((n - k) / (k + 1)) * (q / (1.0 - q))
        tail += term
        if term == 0.0:
            break
    return min(1.0, steps * tail)


class DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def data_blob(value: bytes) -> tuple[DataBlob, Any]:
    buffer = (ctypes.c_ubyte * len(value)).from_buffer_copy(value)
    return DataBlob(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))), buffer


def dpapi_probe() -> dict[str, Any]:
    require(os.name == "nt", "DPAPI preflight requires Windows")
    crypt32 = ctypes.WinDLL("crypt32.dll", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
    protect = crypt32.CryptProtectData
    unprotect = crypt32.CryptUnprotectData
    protect.argtypes = [
        ctypes.POINTER(DataBlob),
        wintypes.LPCWSTR,
        ctypes.POINTER(DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(DataBlob),
    ]
    protect.restype = wintypes.BOOL
    unprotect.argtypes = protect.argtypes
    unprotect.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p

    plaintext = b"k5-resume-probe:" + secrets.token_bytes(32)
    entropy = b"nih-cxr14-k5-feasibility-resume-v1"
    plain_blob, plain_buffer = data_blob(plaintext)
    entropy_blob, entropy_buffer = data_blob(entropy)
    encrypted = DataBlob()
    flags = 0x1  # CRYPTPROTECT_UI_FORBIDDEN
    if not protect(
        ctypes.byref(plain_blob),
        "K5 feasibility in-memory preflight",
        ctypes.byref(entropy_blob),
        None,
        None,
        flags,
        ctypes.byref(encrypted),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        ciphertext = ctypes.string_at(encrypted.pbData, encrypted.cbData)
    finally:
        kernel32.LocalFree(encrypted.pbData)

    cipher_blob, cipher_buffer = data_blob(ciphertext)
    decrypted = DataBlob()
    if not unprotect(
        ctypes.byref(cipher_blob),
        None,
        ctypes.byref(entropy_blob),
        None,
        None,
        flags,
        ctypes.byref(decrypted),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        recovered = ctypes.string_at(decrypted.pbData, decrypted.cbData)
    finally:
        kernel32.LocalFree(decrypted.pbData)
    # Keep backing buffers live through both calls.
    _ = (plain_buffer, entropy_buffer, cipher_buffer)
    require(recovered == plaintext, "DPAPI CurrentUser round trip")
    require(ciphertext != plaintext, "DPAPI ciphertext differs")
    return {
        "platform": "Windows DPAPI CryptProtectData/CryptUnprotectData",
        "scope": "CurrentUser",
        "optional_entropy_applied": True,
        "ciphertext_bytes": len(ciphertext),
        "plaintext_or_ciphertext_persisted": False,
        "round_trip": True,
        "status": "PASS",
    }


def gpu_snapshot() -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    line = completed.stdout.strip().splitlines()[0]
    fields = [value.strip() for value in line.split(",")]
    require(len(fields) == 4, "nvidia-smi output schema")
    result = {
        "name": fields[0],
        "memory_total_mib": int(fields[1]),
        "memory_used_mib": int(fields[2]),
        "utilization_percent": int(fields[3]),
    }
    require("RTX 3070" in result["name"], "target GPU mismatch")
    require(result["memory_total_mib"] >= 8_000, "target GPU memory")
    return result


def verify_static_contract(protocol: dict[str, Any]) -> dict[str, Any]:
    require(protocol["schema"] == "nih-cxr14-k5-feasibility-execution-protocol/v1", "protocol schema")
    require(protocol["status"] == "FROZEN_BEFORE_FULL_K5_OPTIMIZER_EXECUTION", "protocol status")
    require(protocol["scope"] == "ONE_SEED_K5_RESEARCH_FEASIBILITY_NOT_CONFIRMATORY_NOT_RELEASE", "protocol scope")
    matrix = protocol["matrix"]
    require(matrix["arm_order"] == ["M0", "M1-I8", "M1-G8", "M2-P8"], "arm order")
    require(matrix["steps_per_arm"] == 4_000 and matrix["one_seed_only"] is True, "K5 scale")
    require(matrix["K5_privacy_attacks"].startswith("NOT_RUN"), "K5 attack boundary")

    m0 = protocol["M0"]
    require(m0["optimizer_steps"] == 4_000 and m0["images_per_step"] == 8, "M0 scale")
    require(m0["total_image_exposures"] == 32_000 and m0["DP_noise"] == "none", "M0 comparator")
    optimizer = protocol["optimizer"]
    require(optimizer["type"] == "AdamW" and optimizer["learning_rate"] == 1e-4, "optimizer")
    require(optimizer["maximum_and_final_step"] == 4_000, "optimizer steps")
    require(optimizer["private_data_dependent_stopping_or_selection"] == "PROHIBITED", "stopping rule")

    model = protocol["data_and_model"]
    require(model["profile"] == "P256" and model["cap"] == 5, "profile/cap")
    require(model["images"] == 18_393 and model["patients"] == 8_476, "private population")
    require(model["model_id"] == "Manojb/stable-diffusion-2-1-base", "model ID")
    require(model["model_revision"] == "0094d483a120f3f33dafbd187ea4aa60d10de75c", "model revision")
    require(model["lora"]["rank"] == 8 and model["lora"]["trainable_parameters"] == 1_659_904, "LoRA")

    eval_contract = protocol["evaluation"]
    feature = eval_contract["feature_fidelity"]
    rad = feature["primary_encoder"]
    require(rad["model_id"] == "microsoft/rad-dino", "RAD-DINO ID")
    require(rad["revision"] == "110cbc18d5133582e320b43d53bf5c44e410c936", "RAD-DINO revision")
    require(rad["model_safetensors_sha256"] == "DBFB9F54459C38773505DE64A6AB7807BDCB392610FE1E697166342E43FB91AE", "RAD-DINO hash")
    require(feature["primary_small_sample_metric"]["name"] == "RadDINO-KID", "primary quality metric")
    require(feature["mode_metrics"] == {"name": "PRDC", "nearest_k": 5, "outputs": ["precision", "recall", "density", "coverage"]}, "PRDC")
    bio = eval_contract["condition_alignment"]["encoder"]
    require(bio["model_id"] == "microsoft/BiomedVLP-BioViL-T", "BioViL-T ID")
    require(bio["revision"] == "692f09e9be1bfe5fdd5f3efdd0e1eca7d2c10b23", "BioViL-T revision")
    require(eval_contract["downstream_utility"].startswith("not estimated"), "K5 downstream boundary")
    require(len(protocol["decision_rules"]["M0_DOMAIN_ADAPTATION_PASS"]) == 4, "M0 gate")
    require("<=10% of M0" in protocol["decision_rules"]["DP_ARM_UTILITY_COLLAPSE"], "collapse threshold")

    checkpoints = protocol["checkpoint_and_artifact_policy"]["adapter_snapshots"]
    require(checkpoints["steps"] == [1_000, 2_000, 4_000], "checkpoint steps")
    require(checkpoints["quantitative_evaluation_checkpoint"] == 4_000, "final-only evaluation")
    require(protocol["randomness_and_resume"]["classification"] == "RESEARCH_ONLY_NONCRYPTOGRAPHIC", "RNG boundary")
    require(protocol["randomness_and_resume"]["resume_envelope"]["scope"] == "CurrentUser", "resume scope")
    require(protocol["randomness_and_resume"]["restart_preflight"].startswith("before full K5"), "restart gate")
    require(len(protocol["claim_limits"]) == 6, "claim limits")
    return {
        "matrix": "M0 -> M1-I8 -> M1-G8 -> M2-P8; one seed; 4000 fixed steps each",
        "model": "SD2.1-base P256 plus rank-8 LoRA",
        "K5_attacks": "NOT_RUN",
        "quantitative_checkpoint": 4_000,
        "research_only_boundary": True,
        "status": "PASS",
    }


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    protocol_dir = root / "_reports" / "nih_cxr14_k5_feasibility_protocol_v1_001"
    gate_dir = root / "_reports" / "nih_cxr14_k5_feasibility_protocol_gate_v1_001"
    paths = {
        "protocol": protocol_dir / "protocol.json",
        "generation": protocol_dir / "generation_tasks.csv",
        "reference": protocol_dir / "real_reference_local.csv",
        "manifest": root / "_data" / "derived" / "nih_cxr14_pa_target_enriched_v1" / "k5_private.csv",
        "upstream": root / "_reports" / "nih_cxr14_dp_attack_protocol_v1_001" / "protocol.json",
        "preprocessing": root / "_reports" / "nih_cxr14_sd21_ppmark_interface_v1_001" / "preprocessing_contract.json",
        "mechanism": root / "dp_training" / "mechanism.py",
        "dryrun_protocol": root / "dp_training" / "private_research_dryrun_protocol.json",
        "dryrun_public": root / "_reports" / "nih_cxr14_k5_private_research_dryrun_v1_001" / "public_report.json",
        "dryrun_independent": root / "_reports" / "nih_cxr14_k5_private_research_dryrun_gate_v1_001" / "independent_verification.json",
    }
    require(not gate_dir.exists(), "refusing to overwrite an existing independent gate directory")
    expected_sources = {
        "protocol": PROTOCOL_SHA256,
        "generation": GENERATION_SHA256,
        "reference": REFERENCE_SHA256,
        "manifest": K5_MANIFEST_SHA256,
        "upstream": UPSTREAM_PROTOCOL_SHA256,
        "preprocessing": PREPROCESSING_SHA256,
        "mechanism": MECHANISM_SHA256,
        "dryrun_protocol": DRYRUN_PROTOCOL_SHA256,
        "dryrun_public": DRYRUN_PUBLIC_SHA256,
        "dryrun_independent": DRYRUN_INDEPENDENT_SHA256,
    }
    for name, digest in expected_sources.items():
        require(paths[name].is_file(), f"missing source: {name}")
        require(sha256_file(paths[name]) == digest, f"source hash drift: {name}")

    protocol = load_json(paths["protocol"])
    upstream = load_json(paths["upstream"])
    require(protocol["upstream"]["dp_attack_protocol_sha256"] == UPSTREAM_PROTOCOL_SHA256, "upstream protocol link")
    require(protocol["upstream"]["k5_manifest_sha256"] == K5_MANIFEST_SHA256, "upstream manifest link")
    static_check = verify_static_contract(protocol)
    generation_check = verify_generation(paths["generation"], protocol)
    reference_check = verify_references(paths["manifest"], paths["reference"], protocol)
    dp_check = verify_dp_arms(protocol, upstream)

    limits = protocol["resource_and_failure_policy"]["realization_limits"]
    m1_bound = binomial_union_bound(18_393, 8 / 18_393, 64, 4_000)
    m2_bound = binomial_union_bound(8_476, 4 / 8_476, 32, 4_000)
    require(
        math.isclose(
            limits["M1_4000_step_union_bound_above_limit"],
            m1_bound,
            rel_tol=1e-10,
            abs_tol=0.0,
        ),
        "M1 resource bound",
    )
    require(
        math.isclose(
            limits["M2_4000_step_union_bound_above_limit"],
            m2_bound,
            rel_tol=1e-10,
            abs_tol=0.0,
        ),
        "M2 resource bound",
    )
    require(limits["M2_maximum_raw_images_per_step"] == 128, "M2 raw-image bound")

    dpapi_check = dpapi_probe()
    disk = shutil.disk_usage(root)
    free_gib = disk.free / 1024**3
    require(free_gib >= 15.0, "free space below the frozen arm-start threshold")
    gpu = gpu_snapshot()
    full_run_root = root / "_restricted_runs" / "nih_cxr14_k5_feasibility_v1_001"
    full_report_root = root / "_reports" / "nih_cxr14_k5_feasibility_v1_001"
    require(not full_run_root.exists(), "full K5 restricted run directory already exists")
    require(not full_report_root.exists(), "full K5 public report directory already exists")

    gate_dir.mkdir(parents=True)
    report = {
        "schema": SCHEMA,
        "status": "PASS_PROTOCOL_INDEPENDENTLY_VERIFIED_FULL_K5_NOT_STARTED",
        "source_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "checks": {
            "static_contract": static_check,
            "generation_tasks": generation_check,
            "real_reference": reference_check,
            "DP_arms": dp_check,
            "resource_bounds": {
                "M1_4000_step_union_bound_above_64": m1_bound,
                "M2_4000_step_union_bound_above_32": m2_bound,
                "independent_recurrence": True,
                "status": "PASS",
            },
            "DPAPI_in_memory": dpapi_check,
        },
        "environment_snapshot_not_a_frozen_training_result": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "free_bytes": disk.free,
            "free_gib": free_gib,
            "gpu": gpu,
        },
        "full_K5_optimizer_execution_started": False,
        "model_loaded_or_image_generated_by_verifier": False,
        "remaining_gates": [
            "implement and pass the pinned RAD-DINO/BioViL-T evaluator preflight",
            "implement and pass exact M0/M1/M2 DPAPI restart equivalence",
            "freeze and independently verify the full 4000-step runner and environment",
            "report to the user before starting the long run",
        ],
    }
    report_path = gate_dir / "independent_verification.json"
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report_sha256={sha256_file(report_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
