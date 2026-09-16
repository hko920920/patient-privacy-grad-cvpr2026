#!/usr/bin/env python3
"""Independent verifier for the frozen X-ray DP/attack protocol.

The verifier does not import the protocol builder or UnitDP helpers.  It
re-reads locked CSVs, reconstructs the MIA cohort, recomputes both accountants
and group privacy, and checks public clip selection from the emitted gradients.
It deliberately does not load the diffusion model or perform training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "nih-cxr14-dp-attack-protocol-independent-verification/v1"
MANIFEST_HASHES = {
    2: "742980F7C70647C20BF9BA010DF505764259D8A8481704EBBB53D8FE58F363BE",
    5: "DC49D82E497EAA6940DCF92C8F773179B8F7A838057A82AE0C26A69CA785BFC5",
    10: "2D749FB7B70823114A69FD55921277B0D0FF2F9AEC8FECD239159C553F3A0C06",
}
EXPECTED_TOTALS = {2: 20_687, 5: 31_451, 10: 38_492}
EXPECTED_PRIVATE_IMAGES = {2: 12_467, 5: 18_393, 10: 21_756}
EXPECTED_PARTITION_PATIENTS = {
    "private_train": 8_476,
    "public_development": 1_816,
    "privacy_attack_holdout": 1_816,
    "final_test": 1_818,
}
EXPECTED_CLIP_REPORT_SHA256 = (
    "F9771F169BA3F6468F5AAF254A27AFFE1FB47B1ABA897C365BFFC7CDD11A73F1"
)


def default_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parent.parent
    protocol_output = root / "_reports" / "nih_cxr14_dp_attack_protocol_v1_001"
    return {
        "root": root,
        "data": root / "_data" / "derived" / "nih_cxr14_pa_target_enriched_v1",
        "protocol": protocol_output / "protocol.json",
        "builder_report": protocol_output / "report.json",
        "clip_report": root
        / "_reports"
        / "nih_cxr14_public_clip_calibration_v1_001"
        / "report.json",
        "output": root / "_reports" / "nih_cxr14_dp_attack_protocol_verify_v1_001",
    }


def parse_args() -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=paths["data"])
    parser.add_argument("--protocol", type=Path, default=paths["protocol"])
    parser.add_argument("--builder-report", type=Path, default=paths["builder_report"])
    parser.add_argument("--clip-report", type=Path, default=paths["clip_report"])
    parser.add_argument("--output-dir", type=Path, default=paths["output"])
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_manifests(data_dir: Path) -> dict[int, list[dict[str, str]]]:
    result: dict[int, list[dict[str, str]]] = {}
    ids: dict[int, set[str]] = {}
    for cap in (2, 5, 10):
        path = data_dir / f"k{cap}_private.csv"
        require(sha256_file(path) == MANIFEST_HASHES[cap], f"K{cap} hash mismatch")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        require(len(rows) == EXPECTED_TOTALS[cap], f"K{cap} total mismatch")
        require(sum(row["partition"] == "private_train" for row in rows) == EXPECTED_PRIVATE_IMAGES[cap], f"K{cap} train count mismatch")
        patient_partition: dict[str, str] = {}
        for row in rows:
            require(row["selected_k" + str(cap)] == "1", f"K{cap} selected flag mismatch")
            require(row["view"] == "PA", f"K{cap} non-PA row")
            require(int(row["cap_rank"]) <= cap, f"K{cap} cap violation")
            previous = patient_partition.setdefault(row["patient_id"], row["partition"])
            require(previous == row["partition"], "patient crosses partitions")
        partition_counts = Counter(patient_partition.values())
        require(dict(partition_counts) == EXPECTED_PARTITION_PATIENTS, f"K{cap} patient partition counts mismatch")
        result[cap] = rows
        ids[cap] = {row["image_id"] for row in rows}
        require(len(ids[cap]) == len(rows), f"K{cap} duplicate image")
    require(ids[2] < ids[5] < ids[10], "nested cap sets fail")
    return result


def group_rows(rows: list[dict[str, str]], partition: str) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["partition"] == partition:
            result[row["patient_id"]].append(row)
    return dict(result)


def verify_cohorts(protocol_dir: Path, protocol: dict[str, Any], rows: list[dict[str, str]]) -> dict[str, Any]:
    train = group_rows(rows, "private_train")
    holdout = group_rows(rows, "privacy_attack_holdout")
    salt = protocol["membership_attack"]["cohort"]["selection_salt"]
    holdout_strata = Counter(
        (int(patient_rows[0]["target_patient"]), len(patient_rows))
        for patient_rows in holdout.values()
    )
    expected_members: list[str] = []
    for target in (0, 1):
        for image_count in range(1, 11):
            required = holdout_strata[(target, image_count)]
            candidates = [
                patient
                for patient, patient_rows in train.items()
                if int(patient_rows[0]["target_patient"]) == target
                and len(patient_rows) == image_count
            ]
            require(len(candidates) >= required, "train matching stratum too small")
            expected_members.extend(
                sorted(
                    candidates,
                    key=lambda patient: (
                        stable_hash(salt, target, image_count, patient),
                        patient,
                    ),
                )[:required]
            )
    expected_nonmembers = set(holdout)
    require(len(expected_nonmembers) == 1_816, "holdout patient size changed")

    cohort = protocol["membership_attack"]["cohort"]
    member_path = protocol_dir / cohort["members"]["file"]
    nonmember_path = protocol_dir / cohort["nonmembers"]["file"]
    require(sha256_file(member_path) == cohort["members"]["sha256"], "member cohort hash mismatch")
    require(sha256_file(nonmember_path) == cohort["nonmembers"]["sha256"], "nonmember cohort hash mismatch")
    with member_path.open("r", encoding="utf-8", newline="") as handle:
        member_rows = list(csv.DictReader(handle))
    with nonmember_path.open("r", encoding="utf-8", newline="") as handle:
        nonmember_rows = list(csv.DictReader(handle))
    actual_members = {row["patient_id"] for row in member_rows}
    actual_nonmembers = {row["patient_id"] for row in nonmember_rows}
    require(actual_members == set(expected_members), "member hash selection reconstruction mismatch")
    require(actual_nonmembers == expected_nonmembers, "nonmember frozen-holdout reconstruction mismatch")
    require(actual_members.isdisjoint(actual_nonmembers), "member/nonmember overlap")
    for cohort_rows in (member_rows, nonmember_rows):
        require(Counter(row["target_patient"] for row in cohort_rows) == {"0": 908, "1": 908}, "cohort target balance mismatch")
    require(
        Counter((row["target_patient"], row["image_count_k10"]) for row in member_rows)
        == Counter((row["target_patient"], row["image_count_k10"]) for row in nonmember_rows),
        "member/nonmember contribution strata are not exactly matched",
    )
    return {
        "members": len(actual_members),
        "nonmembers": len(actual_nonmembers),
        "target_control_balance": "908/908 in each role",
        "selection_reconstructed": True,
        "disjoint": True,
        "status": "PASS",
    }


def opacus_epsilon(entry: dict[str, Any], orders: list[float]) -> tuple[float, float]:
    from opacus.accountants.analysis import rdp

    values = rdp.compute_rdp(
        q=float(entry["poisson_sample_rate"]),
        noise_multiplier=float(entry["noise_multiplier"]),
        steps=int(entry["max_steps"]),
        orders=orders,
    )
    epsilon, order = rdp.get_privacy_spent(
        orders=orders,
        rdp=values,
        delta=float(entry["target_delta"]),
    )
    return float(epsilon), float(order)


def google_epsilon(entry: dict[str, Any], orders: list[float]) -> tuple[float, float]:
    from dp_accounting import dp_event
    from dp_accounting.privacy_accountant import NeighboringRelation
    from dp_accounting.rdp import RdpAccountant

    accountant = RdpAccountant(
        orders=orders,
        neighboring_relation=NeighboringRelation.ADD_OR_REMOVE_ONE,
    )
    accountant.compose(
        dp_event.PoissonSampledDpEvent(
            sampling_probability=float(entry["poisson_sample_rate"]),
            event=dp_event.GaussianDpEvent(float(entry["noise_multiplier"])),
        ),
        count=int(entry["max_steps"]),
    )
    epsilon, order = accountant.get_epsilon_and_optimal_order(float(entry["target_delta"]))
    return float(epsilon), float(order)


def log_geometric_sum(epsilon: float, cap: int) -> float:
    maximum = (cap - 1) * epsilon
    return maximum + math.log(sum(math.exp(i * epsilon - maximum) for i in range(cap)))


def verify_accounting(protocol: dict[str, Any]) -> dict[str, Any]:
    accounting = protocol["accounting"]
    orders = [float(value) for value in accounting["orders"]]
    require(len(orders) == 155 and len(set(orders)) == 155, "RDP orders changed or duplicated")
    entries = accounting["entries"]
    require(len(entries) == 9, "accounting entry count mismatch")
    max_drift = 0.0
    reconstructed: list[dict[str, Any]] = []
    for entry in entries:
        require(abs(float(entry["poisson_sample_rate"]) - float(entry["expected_batch"]) / float(entry["population"])) < 1e-18, "sample rate is not expected_batch/population")
        opacus, opacus_order = opacus_epsilon(entry, orders)
        google, google_order = google_epsilon(entry, orders)
        max_drift = max(
            max_drift,
            abs(opacus - float(entry["opacus_epsilon"])),
            abs(google - float(entry["google_dp_accounting_epsilon"])),
        )
        require(opacus <= float(entry["target_epsilon"]) + 1e-9, "independent Opacus epsilon exceeds target")
        require(google <= float(entry["target_epsilon"]) + 1e-9, "independent Google epsilon exceeds target")
        require(abs(opacus_order - float(entry["opacus_optimal_order"])) < 1e-12, "Opacus optimal order mismatch")
        require(abs(google_order - float(entry["google_optimal_order"])) < 1e-12, "Google optimal order mismatch")
        if entry["accounting_unit"] == "image":
            epsilon = float(entry["target_epsilon"])
            delta = float(entry["target_delta"])
            cap = int(entry["cap"])
            log_delta = math.log(delta) + log_geometric_sum(epsilon, cap)
            group = entry["group_conversion"]
            require(abs(float(group["patient_epsilon"]) - cap * epsilon) < 1e-12, "group epsilon mismatch")
            require(bool(group["vacuous"]) == (log_delta >= 0), "group vacuity mismatch")
            if log_delta < 0:
                require(abs(float(group["patient_delta"]) - math.exp(log_delta)) < 1e-15, "group delta mismatch")
        reconstructed.append(
            {
                "arm": entry["arm"],
                "cap": entry["cap"],
                "opacus_epsilon": opacus,
                "google_epsilon": google,
            }
        )
    k10_i8 = next(entry for entry in entries if entry["arm"] == "M1-I8" and entry["cap"] == 10)
    require(k10_i8["group_conversion"]["vacuous"] is True, "K10 image epsilon 8 must be vacuous as patient conversion")
    for entry in (item for item in entries if item["arm"] == "M1-G8"):
        require(abs(float(entry["group_conversion"]["patient_epsilon"]) - 8.0) < 1e-12, "M1-G8 patient epsilon mismatch")
        require(abs(float(entry["group_conversion"]["patient_delta"]) - 1e-5) < 1e-15, "M1-G8 patient delta mismatch")
    return {
        "entries": len(entries),
        "orders": len(orders),
        "maximum_recompute_drift": max_drift,
        "all_targets_respected": True,
        "k10_standard_image_dp_patient_conversion_vacuous": True,
        "group_matched_entries_reconstruct_patient_8_1e-5": True,
        "recomputed": reconstructed,
        "status": "PASS",
    }


def verify_clip_report(protocol: dict[str, Any], clip_report_path: Path) -> dict[str, Any]:
    require(sha256_file(clip_report_path) == EXPECTED_CLIP_REPORT_SHA256, "clip report hash mismatch")
    clip = load_json(clip_report_path)
    require(clip["status"] == "PASS_PUBLIC_CLIP_CALIBRATION", "clip report status mismatch")
    require(clip["frozen_inputs"]["trainable_dtypes_observed"] == ["torch.float32"], "clip gradients were not from fp32 LoRA")
    require(clip["parameters_unchanged"] is True, "clip calibration changed parameters")
    require(clip["optimizer_created"] is False and clip["optimizer_step"] is False, "clip calibration optimized")
    gradient_path = clip_report_path.parent / clip["gradient_rows"]
    require(sha256_file(gradient_path) == clip["gradient_rows_sha256"], "gradient-row hash mismatch")
    with gradient_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    image_norms = sorted(float(row["gradient_l2_norm"]) for row in rows if row["unit_type"] == "image")
    patient_norms = sorted(float(row["gradient_l2_norm"]) for row in rows if row["unit_type"] == "patient")
    require(len(image_norms) == 72 and len(patient_norms) == 72, "clip sample sizes mismatch")
    # NumPy method='higher' at q=.8 has index ceil((n-1)*q).
    image_c = image_norms[math.ceil((len(image_norms) - 1) * 0.8)]
    patient_c = patient_norms[math.ceil((len(patient_norms) - 1) * 0.8)]
    require(image_c == float(clip["selected_clip_norms"]["image"]), "image p80 mismatch")
    require(patient_c == float(clip["selected_clip_norms"]["patient"]), "patient p80 mismatch")
    binding = protocol["clip_norms"]
    require(binding["report_sha256"] == EXPECTED_CLIP_REPORT_SHA256, "protocol clip hash binding mismatch")
    require(float(binding["image_clip_norm"]) == image_c, "protocol image C mismatch")
    require(float(binding["patient_clip_norm"]) == patient_c, "protocol patient C mismatch")
    return {
        "image_clip_norm": image_c,
        "patient_clip_norm": patient_c,
        "image_units_above_c": sum(value > image_c for value in image_norms),
        "patient_units_above_c": sum(value > patient_c for value in patient_norms),
        "fp32_trainable_gradients": True,
        "parameters_unchanged": True,
        "status": "PASS",
    }


def main() -> int:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    builder_report_path = args.builder_report.resolve()
    protocol = load_json(protocol_path)
    builder_report = load_json(builder_report_path)
    require(protocol["schema"] == "nih-cxr14-dp-attack-protocol/v1", "protocol schema mismatch")
    require(protocol["status"] == "FROZEN_PRETRAINING_PROTOCOL", "protocol not frozen")
    require(builder_report["status"] == "PASS_PROTOCOL_FREEZE_TRAINING_STILL_BLOCKED", "builder gate status mismatch")
    require(sha256_file(protocol_path) == builder_report["protocol_sha256"], "builder report does not bind protocol")
    require(protocol["randomness_policy"]["release_runtime"].startswith("BLOCKED_"), "release RNG must remain blocked")
    require(protocol["release_gate"]["current_status"] == "BLOCKED_BEFORE_TRAINING", "release gate unexpectedly open")

    manifests = read_manifests(args.data_dir.resolve())
    cohort_result = verify_cohorts(protocol_path.parent, protocol, manifests[10])
    accounting_result = verify_accounting(protocol)
    clip_result = verify_clip_report(protocol, args.clip_report.resolve())

    compute = protocol["mechanisms"]["compute_match"]
    train = group_rows(manifests[10], "private_train")
    expected_images = 4 * sum(min(4, len(value)) for value in train.values()) / len(train)
    require(abs(expected_images - float(compute["m2_expected_images_per_step"])) < 1e-15, "M2 expected image work mismatch")
    require(abs(expected_images / 8 - float(compute["relative_image_work_m2_over_m1"])) < 1e-15, "compute ratio mismatch")
    require(abs(expected_images / 8 - 1.0) < 0.01, "raw image work differs by >=1%")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": SCHEMA,
        "status": "PASS_INDEPENDENT_PROTOCOL_VERIFICATION_TRAINING_STILL_BLOCKED",
        "protocol_sha256": sha256_file(protocol_path),
        "builder_report_sha256": sha256_file(builder_report_path),
        "manifest_verification": {
            "caps": [2, 5, 10],
            "hashes": MANIFEST_HASHES,
            "nested": True,
            "patient_disjoint_partitions": True,
            "status": "PASS",
        },
        "cohort_verification": cohort_result,
        "accounting_verification": accounting_result,
        "clip_verification": clip_result,
        "compute_match_verification": {
            "m1_expected_images": 8,
            "m2_expected_images": expected_images,
            "ratio": expected_images / 8,
            "within_one_percent": True,
            "status": "PASS",
        },
        "fail_closed": {
            "training_still_blocked": True,
            "release_rng_still_blocked": True,
            "no_attack_result": True,
            "no_privacy_receipt": True,
            "no_release_authorization": True,
        },
        "environment": {"python": platform.python_version()},
        "interpretation_limit": "This independently verifies the pre-training protocol artifacts and arithmetic. It does not rerun model gradients, train a model, execute DP, run attacks, or authorize release.",
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"NIH_CXR14_DP_ATTACK_PROTOCOL_VERIFY: {report['status']}")
    print(f"report={report_path}")
    print(f"report_sha256={sha256_file(report_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
