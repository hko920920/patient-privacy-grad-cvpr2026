#!/usr/bin/env python3
"""Build the frozen NIH CXR14 DP/accounting/attack protocol.

This is a pre-training gate.  It verifies the locked manifests, constructs an
outcome-independent patient membership evaluation cohort, calibrates Gaussian
noise against two RDP implementations, and writes a machine-readable contract.
It never loads a model, constructs an optimizer, or accesses attack outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable


SCHEMA = "nih-cxr14-dp-attack-protocol/v1"
REPORT_SCHEMA = "nih-cxr14-dp-attack-protocol-gate/v1"
MANIFEST_SHA256 = {
    2: "742980F7C70647C20BF9BA010DF505764259D8A8481704EBBB53D8FE58F363BE",
    5: "DC49D82E497EAA6940DCF92C8F773179B8F7A838057A82AE0C26A69CA785BFC5",
    10: "2D749FB7B70823114A69FD55921277B0D0FF2F9AEC8FECD239159C553F3A0C06",
}
PREPROCESSING_CONTRACT_SHA256 = (
    "0ED9E434764BDB2D542BB16AB951F856CB5B82D4F2E6B9F25F6C554F1076669D"
)
MODEL_REVISION = "0094d483a120f3f33dafbd187ea4aa60d10de75c"
MODEL_HASHES = {
    "unet": "28EC9CF3B239C0751C201B1F6FB46B551DF5862731B30A37AA1360101CB3FBAB",
    "vae": "3E4C08995484EE61270175E9E7A072B66A6E4EEB5F0C266667FE1F45B90DAF9A",
    "text_encoder": "681C555376658C81DC273F2D737A2AEB23DDB6D1D8E5B3A7064636D359A22668",
}
EXPECTED_IMAGES = {
    2: {
        "private_train": 12_467,
        "public_development": 2_702,
        "privacy_attack_holdout": 2_681,
        "final_test": 2_837,
        "total": 20_687,
    },
    5: {
        "private_train": 18_393,
        "public_development": 4_031,
        "privacy_attack_holdout": 4_012,
        "final_test": 5_015,
        "total": 31_451,
    },
    10: {
        "private_train": 21_756,
        "public_development": 4_831,
        "privacy_attack_holdout": 4_740,
        "final_test": 7_165,
        "total": 38_492,
    },
}
EXPECTED_PATIENTS = {
    "private_train": 8_476,
    "public_development": 1_816,
    "privacy_attack_holdout": 1_816,
    "final_test": 1_818,
    "total": 13_926,
}

IMAGE_EXPECTED_BATCH = 8
PATIENT_EXPECTED_BATCH = 4
PATIENT_IMAGES_PER_INCLUSION = 4
MAX_STEPS = 4_000
DIRECT_DELTA = 1e-5
PATIENT_TARGET_EPSILON = 8.0
MEMBER_SELECTION_SALT = "nih-cxr14-k10-patient-mia-members-v1"
BOOTSTRAP_SALT = "nih-cxr14-k10-patient-mia-bootstrap-v1"

# Opacus's dense default range, extended at high orders.  The exact list is
# serialized so later accountants cannot silently change it.
RDP_ORDERS = [round(1.0 + i / 10.0, 1) for i in range(1, 100)]
RDP_ORDERS += list(range(12, 64)) + [64, 128, 256, 512]


def default_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parent.parent
    data = root / "_data" / "derived" / "nih_cxr14_pa_target_enriched_v1"
    return {
        "root": root,
        "data": data,
        "preprocessing_contract": root
        / "_reports"
        / "nih_cxr14_sd21_ppmark_interface_v1_001"
        / "preprocessing_contract.json",
        "clip_report": root
        / "_reports"
        / "nih_cxr14_public_clip_calibration_v1_001"
        / "report.json",
        "output": root / "_reports" / "nih_cxr14_dp_attack_protocol_v1_001",
    }


def parse_args() -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=paths["data"])
    parser.add_argument(
        "--preprocessing-contract", type=Path, default=paths["preprocessing_contract"]
    )
    parser.add_argument("--clip-report", type=Path, default=paths["clip_report"])
    parser.add_argument("--output-dir", type=Path, default=paths["output"])
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def stable_hash(*parts: object) -> str:
    return sha256_bytes("|".join(map(str, parts)).encode("utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def read_and_check_manifests(data_dir: Path) -> tuple[dict[int, list[dict[str, str]]], dict[str, Any]]:
    all_rows: dict[int, list[dict[str, str]]] = {}
    report: dict[str, Any] = {}
    expected_partitions = set(EXPECTED_PATIENTS) - {"total"}
    for cap in (2, 5, 10):
        path = data_dir / f"k{cap}_private.csv"
        require(path.is_file(), f"missing K{cap} manifest: {path}")
        actual_hash = sha256_file(path)
        require(actual_hash == MANIFEST_SHA256[cap], f"K{cap} manifest hash mismatch")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        require(len(rows) == EXPECTED_IMAGES[cap]["total"], f"K{cap} row count mismatch")
        image_ids = [row["image_id"] for row in rows]
        require(len(image_ids) == len(set(image_ids)), f"K{cap} duplicate image")
        require(all(row[f"selected_k{cap}"] == "1" for row in rows), f"K{cap} flag mismatch")
        partitions = Counter(row["partition"] for row in rows)
        require(set(partitions) == expected_partitions, f"K{cap} partition names mismatch")
        patient_partition: dict[str, str] = {}
        patient_target: dict[str, str] = {}
        for row in rows:
            patient = row["patient_id"]
            previous = patient_partition.setdefault(patient, row["partition"])
            require(previous == row["partition"], f"patient {patient} crosses partitions")
            target = patient_target.setdefault(patient, row["target_patient"])
            require(target == row["target_patient"], f"patient {patient} target flag changes")
            require(1 <= int(row["cap_rank"]) <= cap, f"K{cap} cap rank out of range")
            require(row["view"] == "PA", f"K{cap} includes non-PA image")
        partition_patients = Counter(patient_partition.values())
        for partition in expected_partitions:
            require(
                partitions[partition] == EXPECTED_IMAGES[cap][partition],
                f"K{cap} {partition} image count mismatch",
            )
            require(
                partition_patients[partition] == EXPECTED_PATIENTS[partition],
                f"K{cap} {partition} patient count mismatch",
            )
        require(len(patient_partition) == EXPECTED_PATIENTS["total"], f"K{cap} patient total mismatch")
        all_rows[cap] = rows
        report[str(cap)] = {
            "path": str(path.resolve()),
            "sha256": actual_hash,
            "images": len(rows),
            "patients": len(patient_partition),
            "partition_images": dict(sorted(partitions.items())),
            "partition_patients": dict(sorted(partition_patients.items())),
            "status": "PASS",
        }
    for row2, row5, row10 in zip(all_rows[2], all_rows[5], all_rows[10]):
        # The files are differently sized, so nesting is checked below by sets;
        # this loop intentionally does nothing beyond preventing a misleading
        # assumption about row-aligned files.
        del row2, row5, row10
        break
    ids = {cap: {row["image_id"] for row in rows} for cap, rows in all_rows.items()}
    require(ids[2] < ids[5] < ids[10], "K2/K5/K10 are not strict nested views")
    return all_rows, report


def group_patients(rows: list[dict[str, str]], partition: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["partition"] == partition:
            grouped[row["patient_id"]].append(row)
    for patient_rows in grouped.values():
        patient_rows.sort(key=lambda row: (int(row["cap_rank"]), row["image_id"]))
    return dict(grouped)


def write_attack_cohorts(
    rows: list[dict[str, str]], output_dir: Path
) -> tuple[dict[str, Any], set[str], set[str]]:
    train = group_patients(rows, "private_train")
    holdout = group_patients(rows, "privacy_attack_holdout")
    holdout_by_target: dict[int, list[str]] = {0: [], 1: []}
    for patient, patient_rows in holdout.items():
        holdout_by_target[int(patient_rows[0]["target_patient"])].append(patient)
    require(len(holdout_by_target[0]) == 908, "holdout control count changed")
    require(len(holdout_by_target[1]) == 908, "holdout target count changed")

    selected_members: list[str] = []
    member_selection_rows: list[dict[str, Any]] = []
    holdout_strata = Counter(
        (int(patient_rows[0]["target_patient"]), len(patient_rows))
        for patient_rows in holdout.values()
    )
    for target in (0, 1):
        require(sum(count for (flag, _), count in holdout_strata.items() if flag == target) == 908, "holdout target stratum total changed")
        for image_count in range(1, 11):
            required = holdout_strata[(target, image_count)]
            candidates = [
                patient
                for patient, patient_rows in train.items()
                if int(patient_rows[0]["target_patient"]) == target
                and len(patient_rows) == image_count
            ]
            require(len(candidates) >= required, f"insufficient train patients in target={target}/n={image_count}")
            ranked = sorted(
                candidates,
                key=lambda patient: (
                    stable_hash(MEMBER_SELECTION_SALT, target, image_count, patient),
                    patient,
                ),
            )
            chosen = ranked[:required]
            selected_members.extend(chosen)
            for patient in chosen:
                member_selection_rows.append(
                    {
                        "role": "member",
                        "patient_id": patient,
                        "target_patient": target,
                        "image_count_k10": image_count,
                        "selection_sha256": stable_hash(
                            MEMBER_SELECTION_SALT, target, image_count, patient
                        ),
                    }
                )

    nonmembers = sorted(
        holdout,
        key=lambda patient: (int(holdout[patient][0]["target_patient"]), patient),
    )
    member_selection_rows.sort(key=lambda row: (row["target_patient"], row["selection_sha256"], row["patient_id"]))
    nonmember_rows = [
        {
            "role": "nonmember",
            "patient_id": patient,
            "target_patient": int(holdout[patient][0]["target_patient"]),
            "image_count_k10": len(holdout[patient]),
            "selection_sha256": stable_hash("all-frozen-holdout-patients-v1", patient),
        }
        for patient in nonmembers
    ]

    fieldnames = ["role", "patient_id", "target_patient", "image_count_k10", "selection_sha256"]
    paths = {
        "members": output_dir / "mia_members_private.csv",
        "nonmembers": output_dir / "mia_nonmembers_private.csv",
    }
    for key, cohort_rows in (("members", member_selection_rows), ("nonmembers", nonmember_rows)):
        with paths[key].open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(cohort_rows)

    member_set = set(selected_members)
    nonmember_set = set(nonmembers)
    require(len(member_set) == 1_816, "member evaluation size mismatch")
    require(len(nonmember_set) == 1_816, "nonmember evaluation size mismatch")
    require(member_set.isdisjoint(nonmember_set), "MIA member and nonmember patients overlap")
    return (
        {
            "selection_salt": MEMBER_SELECTION_SALT,
            "matching": "exact target_patient x K10 image-count (1..10) stratum counts from the frozen holdout",
            "members": {
                "patients": len(member_set),
                "target": sum(row["target_patient"] for row in member_selection_rows),
                "control": sum(1 - row["target_patient"] for row in member_selection_rows),
                "file": paths["members"].name,
                "sha256": sha256_file(paths["members"]),
            },
            "nonmembers": {
                "patients": len(nonmember_set),
                "target": sum(row["target_patient"] for row in nonmember_rows),
                "control": sum(1 - row["target_patient"] for row in nonmember_rows),
                "file": paths["nonmembers"].name,
                "sha256": sha256_file(paths["nonmembers"]),
            },
            "uses_model_or_attack_outcome": False,
        },
        member_set,
        nonmember_set,
    )


def opacus_epsilon(sigma: float, q: float, steps: int, delta: float) -> tuple[float, float]:
    from opacus.accountants.analysis import rdp

    values = rdp.compute_rdp(
        q=q,
        noise_multiplier=sigma,
        steps=steps,
        orders=RDP_ORDERS,
    )
    epsilon, best_order = rdp.get_privacy_spent(
        orders=RDP_ORDERS,
        rdp=values,
        delta=delta,
    )
    return float(epsilon), float(best_order)


def google_epsilon(sigma: float, q: float, steps: int, delta: float) -> tuple[float, float]:
    from dp_accounting import dp_event
    from dp_accounting.privacy_accountant import NeighboringRelation
    from dp_accounting.rdp import RdpAccountant

    accountant = RdpAccountant(
        orders=RDP_ORDERS,
        neighboring_relation=NeighboringRelation.ADD_OR_REMOVE_ONE,
    )
    accountant.compose(
        dp_event.PoissonSampledDpEvent(
            sampling_probability=q,
            event=dp_event.GaussianDpEvent(sigma),
        ),
        count=steps,
    )
    epsilon, best_order = accountant.get_epsilon_and_optimal_order(delta)
    return float(epsilon), float(best_order)


def calibrate_dual(target_epsilon: float, q: float, steps: int, delta: float) -> dict[str, Any]:
    def evaluate(sigma: float) -> tuple[tuple[float, float], tuple[float, float]]:
        return opacus_epsilon(sigma, q, steps, delta), google_epsilon(sigma, q, steps, delta)

    low, high = 0.01, 1.0
    while max(evaluate(high)[0][0], evaluate(high)[1][0]) > target_epsilon:
        high *= 2.0
        require(high <= 1024.0, "could not bracket target epsilon")
    for _ in range(60):
        middle = (low + high) / 2.0
        opacus, google = evaluate(middle)
        if max(opacus[0], google[0]) > target_epsilon:
            low = middle
        else:
            high = middle
    opacus, google = evaluate(high)
    require(max(opacus[0], google[0]) <= target_epsilon + 1e-10, "dual calibration overshot target")
    return {
        "noise_multiplier": high,
        "opacus_epsilon": opacus[0],
        "opacus_optimal_order": opacus[1],
        "google_dp_accounting_epsilon": google[0],
        "google_optimal_order": google[1],
        "conservative_epsilon": max(opacus[0], google[0]),
        "interimplementation_abs_difference": abs(opacus[0] - google[0]),
    }


def log_geometric_sum(epsilon: float, cap: int) -> float:
    if cap == 1:
        return 0.0
    maximum = (cap - 1) * epsilon
    return maximum + math.log(sum(math.exp(i * epsilon - maximum) for i in range(cap)))


def required_image_delta(target_delta: float, image_epsilon: float, cap: int) -> float:
    return math.exp(math.log(target_delta) - log_geometric_sum(image_epsilon, cap))


def convert_group(image_epsilon: float, image_delta: float, cap: int) -> dict[str, Any]:
    log_delta = math.log(image_delta) + log_geometric_sum(image_epsilon, cap)
    return {
        "patient_epsilon": cap * image_epsilon,
        "patient_delta": math.exp(log_delta) if log_delta < 0 else ">=1",
        "patient_log10_delta": log_delta / math.log(10),
        "vacuous": log_delta >= 0,
    }


def accounting_entry(
    *,
    arm: str,
    unit: str,
    cap: int,
    population: int,
    expected_batch: int,
    epsilon: float,
    delta: float,
    role: str,
) -> dict[str, Any]:
    q = expected_batch / population
    calibrated = calibrate_dual(epsilon, q, MAX_STEPS, delta)
    return {
        "arm": arm,
        "accounting_unit": unit,
        "adjacency": f"add_or_remove_one_{unit}",
        "cap": cap,
        "population": population,
        "expected_batch": expected_batch,
        "poisson_sample_rate": q,
        "max_steps": MAX_STEPS,
        "target_epsilon": epsilon,
        "target_delta": delta,
        "execution_role": role,
        **calibrated,
    }


def build_accounting(rows_by_cap: dict[int, list[dict[str, str]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    roles = {2: "accounting_only_unit_sensitivity", 5: "one_seed_feasibility", 10: "controlled_main"}
    for cap in (2, 5, 10):
        image_population = EXPECTED_IMAGES[cap]["private_train"]
        entries.append(
            accounting_entry(
                arm="M1-I8",
                unit="image",
                cap=cap,
                population=image_population,
                expected_batch=IMAGE_EXPECTED_BATCH,
                epsilon=8.0,
                delta=DIRECT_DELTA,
                role=roles[cap],
            )
        )
        group = convert_group(8.0, DIRECT_DELTA, cap)
        entries[-1]["group_conversion"] = group
        entries[-1]["patient_release_policy"] = (
            "BLOCKED_VACUOUS" if group["vacuous"] else "BLOCKED_POLICY_EPSILON_OR_DELTA"
        )

        image_epsilon = PATIENT_TARGET_EPSILON / cap
        image_delta = required_image_delta(DIRECT_DELTA, image_epsilon, cap)
        entries.append(
            accounting_entry(
                arm="M1-G8",
                unit="image",
                cap=cap,
                population=image_population,
                expected_batch=IMAGE_EXPECTED_BATCH,
                epsilon=image_epsilon,
                delta=image_delta,
                role=roles[cap],
            )
        )
        group = convert_group(image_epsilon, image_delta, cap)
        require(not group["vacuous"], f"K{cap} group-matched arm became vacuous")
        require(abs(group["patient_epsilon"] - 8.0) < 1e-12, "group epsilon mismatch")
        require(abs(float(group["patient_delta"]) - DIRECT_DELTA) < 1e-15, "group delta mismatch")
        entries[-1]["group_conversion"] = group
        entries[-1]["patient_release_policy"] = "MATHEMATICALLY_ELIGIBLE_POLICY_STILL_REQUIRED"

    patient_population = EXPECTED_PATIENTS["private_train"]
    for epsilon in (2.0, 4.0, 8.0):
        entries.append(
            accounting_entry(
                arm=f"M2-P{int(epsilon)}",
                unit="patient",
                cap=10,
                population=patient_population,
                expected_batch=PATIENT_EXPECTED_BATCH,
                epsilon=epsilon,
                delta=DIRECT_DELTA,
                role="P8_primary_P4_secondary_P2_accounting_curve",
            )
        )

    train_k10 = group_patients(rows_by_cap[10], "private_train")
    internal_images = sum(min(PATIENT_IMAGES_PER_INCLUSION, len(value)) for value in train_k10.values())
    expected_patient_arm_images = PATIENT_EXPECTED_BATCH * internal_images / len(train_k10)
    compute_match = {
        "m1_expected_images_per_step": IMAGE_EXPECTED_BATCH,
        "m2_expected_patients_per_step": PATIENT_EXPECTED_BATCH,
        "m2_max_images_per_included_patient": PATIENT_IMAGES_PER_INCLUSION,
        "m2_sum_min_m_ni": internal_images,
        "m2_mean_images_per_included_patient": internal_images / len(train_k10),
        "m2_expected_images_per_step": expected_patient_arm_images,
        "relative_image_work_m2_over_m1": expected_patient_arm_images / IMAGE_EXPECTED_BATCH,
        "interpretation": "raw-image work is approximately matched; backward-pass structure is inherently unit-specific",
    }
    require(abs(expected_patient_arm_images - IMAGE_EXPECTED_BATCH) / IMAGE_EXPECTED_BATCH < 0.01, "image work mismatch exceeds 1%")
    return entries, compute_match


def attack_power() -> dict[str, Any]:
    members = nonmembers = 1_816
    auc0 = 0.5
    q1 = auc0 / (2.0 - auc0)
    q2 = 2.0 * auc0 * auc0 / (1.0 + auc0)
    standard_error = math.sqrt(
        (
            auc0 * (1.0 - auc0)
            + (members - 1) * (q1 - auc0 * auc0)
            + (nonmembers - 1) * (q2 - auc0 * auc0)
        )
        / (members * nonmembers)
    )
    z = NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.80)
    return {
        "members": members,
        "nonmembers": nonmembers,
        "auc_null": auc0,
        "hanley_mcneil_null_standard_error": standard_error,
        "approximate_two_sided_alpha_0_05_power_0_80_auc": auc0 + z * standard_error,
        "predeclared_material_auc": 0.55,
        "interpretation": "the material AUC threshold is above the approximate detectable AUC; final uncertainty uses patient bootstrap",
    }


def clip_binding(clip_report: Path) -> dict[str, Any]:
    if not clip_report.is_file():
        return {
            "status": "BLOCKED_PENDING_PUBLIC_GRADIENT_CALIBRATION",
            "required_path": str(clip_report.resolve()),
            "selection_rule": "separate image and patient clip norms are the exact empirical public-development p80 values (numpy method='higher')",
        }
    with clip_report.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    require(report.get("status") == "PASS_PUBLIC_CLIP_CALIBRATION", "clip report did not pass")
    require(report.get("optimizer_created") is False, "clip calibration created optimizer")
    require(report.get("optimizer_step") is False, "clip calibration changed model")
    return {
        "status": "BOUND_FROM_PUBLIC_DEVELOPMENT_ONLY",
        "report": str(clip_report.resolve()),
        "report_sha256": sha256_file(clip_report),
        "image_clip_norm": report["selected_clip_norms"]["image"],
        "patient_clip_norm": report["selected_clip_norms"]["patient"],
        "selection_rule": report["selection_rule"],
    }


def write_accounting_csv(path: Path, entries: list[dict[str, Any]]) -> None:
    fields = [
        "arm",
        "accounting_unit",
        "cap",
        "population",
        "expected_batch",
        "poisson_sample_rate",
        "max_steps",
        "target_epsilon",
        "target_delta",
        "noise_multiplier",
        "opacus_epsilon",
        "google_dp_accounting_epsilon",
        "interimplementation_abs_difference",
        "execution_role",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(entries)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    require(sha256_file(args.preprocessing_contract.resolve()) == PREPROCESSING_CONTRACT_SHA256, "preprocessing contract hash mismatch")
    rows_by_cap, manifest_report = read_and_check_manifests(args.data_dir.resolve())
    cohort, members, nonmembers = write_attack_cohorts(rows_by_cap[10], output_dir)
    all_k10_patients = {
        row["patient_id"]: row["partition"] for row in rows_by_cap[10]
    }
    require(all(all_k10_patients[patient] == "private_train" for patient in members), "member not in private train")
    require(all(all_k10_patients[patient] == "privacy_attack_holdout" for patient in nonmembers), "nonmember not in holdout")

    accounting, compute_match = build_accounting(rows_by_cap)
    accounting_path = output_dir / "accounting_table.csv"
    write_accounting_csv(accounting_path, accounting)
    clips = clip_binding(args.clip_report.resolve())

    protocol: dict[str, Any] = {
        "schema": SCHEMA,
        "status": (
            "FROZEN_PRETRAINING_PROTOCOL"
            if clips["status"] == "BOUND_FROM_PUBLIC_DEVELOPMENT_ONLY"
            else "DRAFT_BLOCKED_ON_PUBLIC_CLIP_CALIBRATION"
        ),
        "scope": "CENTRAL_NIH_CXR14_LORA_FINE_TUNING_RESEARCH_PROXY",
        "claim_boundary": {
            "public_proxy_not_confidential_clinical_handling": True,
            "formal_accounting_is_per_run_and_per_released_model": True,
            "multiple_released_models_require_composition": True,
            "attack_failure_is_not_privacy_proof": True,
            "release_grade_dp_runtime_available": False,
        },
        "frozen_inputs": {
            "manifests": manifest_report,
            "preprocessing_contract_sha256": PREPROCESSING_CONTRACT_SHA256,
            "model_id": "Manojb/stable-diffusion-2-1-base",
            "model_revision": MODEL_REVISION,
            "model_hashes": MODEL_HASHES,
            "profile": "P256",
            "lora": {
                "rank": 8,
                "alpha": 8,
                "targets": ["to_q", "to_k", "to_v", "to_out.0"],
                "trainable_parameters": 1_659_904,
                "trainable_dtype": "float32",
                "frozen_base_dtype": "float16",
                "mixed_precision_cast": "peft.utils.other.cast_mixed_precision_params(model, torch.float16)",
                "initialization_seed": 260903,
            },
        },
        "optimizer": {
            "type": "AdamW",
            "learning_rate": 1e-4,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0.01,
            "scheduler": "constant",
            "warmup_steps": 0,
            "max_steps": MAX_STEPS,
            "final_checkpoint_step": MAX_STEPS,
            "private_data_dependent_checkpoint_selection": "PROHIBITED",
            "diagnostic_checkpoint_steps_not_released": [1000, 2000],
        },
        "clip_norms": clips,
        "mechanisms": {
            "M1_image_dp": {
                "privacy_unit": "image",
                "adjacency": "add_or_remove_one_image",
                "sampling": "independent Bernoulli/Poisson inclusion of every image each step",
                "expected_units_per_step": IMAGE_EXPECTED_BATCH,
                "unit_gradient": "one prompt-conditioned diffusion loss with one uniformly sampled timestep and one Gaussian latent-noise draw",
                "aggregation": "clip each full LoRA gradient vector to C_image; sum; add N(0,(sigma*C_image)^2 I); divide by fixed public expected batch 8",
                "empty_step": "perform Gaussian-noise-only update",
            },
            "M2_patient_dp": {
                "privacy_unit": "patient",
                "adjacency": "add_or_remove_one complete patient contribution",
                "sampling": "independent Bernoulli/Poisson inclusion of every patient each step",
                "expected_units_per_step": PATIENT_EXPECTED_BATCH,
                "within_patient": "take up to m=4 K-capped records by fresh uniform sampling without replacement; use all records when n_i<4",
                "unit_gradient": "mean the selected per-image diffusion losses before one patient-gradient computation; every patient has equal outer weight",
                "aggregation": "clip each full patient gradient vector to C_patient; sum; add N(0,(sigma*C_patient)^2 I); divide by fixed public expected batch 4",
                "empty_step": "perform Gaussian-noise-only update",
                "privacy_reason": "all within-patient processing occurs inside one bounded patient vector before clipping",
            },
            "shared": {
                "diffusion_timestep_distribution": "discrete uniform over scheduler training timesteps",
                "noise_realizations_per_record": 1,
                "loss_reduction_within_record": "mean",
                "optimizer_is_postprocessing_of_noised_gradient": True,
                "variable_realized_batch_denominator": "PROHIBITED",
                "loss_or_gradient_dependent_sampling": "PROHIBITED",
            },
            "compute_match": compute_match,
        },
        "accounting": {
            "method": "Poisson-sampled Gaussian RDP under add/remove adjacency",
            "orders": RDP_ORDERS,
            "orders_count": len(RDP_ORDERS),
            "calibration_rule": "binary search the smallest sigma whose maximum epsilon across Opacus and Google dp-accounting does not exceed target",
            "entries": accounting,
            "table": accounting_path.name,
            "table_sha256": sha256_file(accounting_path),
            "patient_operational_policy": {
                "maximum_epsilon": 8.0,
                "maximum_delta": DIRECT_DELTA,
                "status": "DISSERTATION_EXPERIMENT_POLICY_NOT_UNIVERSAL_CLINICAL_SAFETY_STANDARD",
            },
        },
        "execution_matrix": {
            "K2": "accounting and privacy-unit sensitivity only; no generator training in the initial matrix",
            "K5": "one-seed feasibility for M0, M1-I8, M1-G8, and M2-P8; cannot select the K10 arm from desired ordering",
            "K10_primary_three_seeds": ["M0", "M1-I8", "M1-G8", "M2-P8"],
            "K10_secondary": ["M2-P4"],
            "K10_one_seed_mechanism_controls": ["A0", "A1", "A2"],
            "research_only_curve": ["M2-P2"],
            "benchmark_models_are_not_a_multi_model_release": True,
        },
        "randomness_policy": {
            "public_reproducibility_seeds": "allowed only for nonprivate initialization, evaluation, and bootstrap",
            "dp_sampling_and_noise_seed": "private and never included in a public report or receipt",
            "research_runtime": "a conventional pseudorandom torch generator may be used only with RESEARCH_ONLY labeling",
            "release_runtime": "BLOCKED_PENDING_REGISTERED_AND_PERFORMANCE_TESTED_CSPRNG_GAUSSIAN_BACKEND",
            "current_torchcsprng_status": "not installed and official package compatibility does not cover the pinned Python 3.11/PyTorch 2.6 environment",
            "fail_closed": "no DIRECT/GROUP release statement may be produced from the research PRNG path",
        },
        "membership_attack": {
            "threat_model": "white-box released weights, scheduler, preprocessing contract, and prompt policy; no training gradients",
            "cohort": cohort,
            "primary_score": {
                "name": "fixed_noise_denoising_loss",
                "timesteps": [50, 250, 500, 750, 950],
                "noise_replicates_per_timestep": 4,
                "image_score": "negative mean epsilon-prediction MSE; higher means more member-like",
                "patient_score": "arithmetic mean of all K10 image scores for that patient",
                "unit_weighting": "one equal-weight score per patient",
            },
            "secondary_score": {
                "name": "SecMI-LDM",
                "official_repository": "https://github.com/jinhaoduan/SecMI-LDM",
                "upstream_commit": "83eff37e06aadf078a79755164a5cf531bd34b04",
                "status": "REQUIRES_PINNED_SD21_ADAPTER_CONFORMANCE_GATE_BEFORE_USE",
                "selection_from_results": "PROHIBITED",
            },
            "metrics": ["ROC_AUC", "max_TPR_minus_FPR", "TPR_at_1_percent_FPR"],
            "uncertainty": {
                "method": "10,000-replicate percentile bootstrap resampling patients within member/nonmember strata",
                "confidence": 0.95,
                "salt": BOOTSTRAP_SALT,
                "paired_arm_differences": True,
            },
            "categorical_rule_primary_score": {
                "LEAKAGE_DETECTED": "lower 95% AUC CI > 0.50 and point AUC >= 0.55",
                "NO_MATERIAL_LEAKAGE_DETECTED": "upper 95% AUC CI < 0.55 and upper 95% TPR@1%FPR CI < 0.05",
                "INCONCLUSIVE": "all other outcomes",
                "formal_dp_effect": "none; these labels never replace or strengthen the accountant bound",
            },
            "power": attack_power(),
            "B0_boundary": "untouched-base scores are overlap/floor diagnostics; ChestXray14 pretraining membership is not asserted",
        },
        "memorization_and_extraction": {
            "threat_model": "black-box generation from the released checkpoint plus published prompt policy",
            "profile": "P256",
            "sampler": "DDIM",
            "inference_steps": 50,
            "guidance_scale": 7.5,
            "negative_prompt": None,
            "canonical_prompts": [
                "no labeled finding",
                "pneumothorax",
                "pneumonia",
                "consolidation",
                "pleural effusion",
                "mass opacity",
                "nodule opacity",
            ],
            "queries_per_prompt": 350,
            "queries_per_model": 2450,
            "generation_seed_rule": "same uint63(SHA256('nih-cxr14-extraction-v1'|prompt_id|query_index)) for every arm",
            "filter_reference_sets": ["all K10 private-train images", "all K10 privacy-attack-holdout images"],
            "copy_metrics": ["exact pixel SHA256", "registered normalized RMSE", "registered SSIM", "SD2.1 VAE latent cosine distance"],
            "threshold_calibration": "freeze thresholds on public-development transformed positives and >=100,000 different-patient public-development negative pairs before any target-model generation",
            "near_copy_requires": [
                "passes the pre-frozen joint RMSE/SSIM threshold against a private-train image",
                "nearest train match is unique under the pre-frozen margin",
                "does not pass the same rule against an attack-holdout image",
                "blinded human review is reported separately and cannot overturn metric failure",
            ],
            "reporting": "exact duplicates, thresholded near-copies, per-query rates, binomial confidence intervals, and nearest-neighbor distributions",
            "claim_boundary": "a bounded 2,450-query generate-and-filter test is not exhaustive; ordinary similarity is not called reconstruction",
            "gradient_inversion": "OUT_OF_SCOPE_BECAUSE_RELEASE_THREAT_MODEL_EXPOSES_NO_TRAINING_GRADIENTS",
        },
        "release_gate": {
            "current_status": "BLOCKED_BEFORE_TRAINING",
            "required_before_optimizer": [
                "public-development clip calibration pass and hash binding",
                "DP trainer mechanism conformance tests including empty Poisson batches",
                "research runtime dry-run with accountant event trace",
            ],
            "required_before_formal_release": [
                "registered secure sampling/noise backend and full retraining from scratch",
                "runtime event trace matches accountant contract",
                "one selected model digest and model-bound receipt",
                "attack and utility reports without prohibited privacy wording",
                "PP-Mark calibration and robustness gate on that exact model",
            ],
        },
    }

    protocol_path = output_dir / "protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": (
            "PASS_PROTOCOL_FREEZE_TRAINING_STILL_BLOCKED"
            if clips["status"] == "BOUND_FROM_PUBLIC_DEVELOPMENT_ONLY"
            else "BLOCKED_PENDING_PUBLIC_CLIP_CALIBRATION"
        ),
        "protocol": protocol_path.name,
        "protocol_sha256": sha256_file(protocol_path),
        "accounting_table": accounting_path.name,
        "accounting_table_sha256": sha256_file(accounting_path),
        "cohort": cohort,
        "clip_binding": clips,
        "accounting_entries": len(accounting),
        "dual_accounting_all_within_target": all(
            entry["conservative_epsilon"] <= entry["target_epsilon"] + 1e-10
            for entry in accounting
        ),
        "prohibited_operations": {
            "model_loaded": False,
            "optimizer_created": False,
            "optimizer_step": False,
            "private_training": False,
            "attack_run": False,
            "model_checkpoint_written": False,
            "receipt_created": False,
            "release_authorized": False,
        },
        "environment": {
            "python": platform.python_version(),
            "opacus": package_version("opacus"),
            "dp_accounting": package_version("dp-accounting"),
        },
        "interpretation_limit": "This pass freezes and validates a pre-training mechanism/accounting/attack contract. It is not a trained model, an executed DP guarantee, an attack result, or release authorization.",
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"NIH_CXR14_DP_ATTACK_PROTOCOL: {report['status']}")
    print(f"protocol={protocol_path}")
    print(f"protocol_sha256={sha256_file(protocol_path)}")
    print(f"report={report_path}")
    print(f"report_sha256={sha256_file(report_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
