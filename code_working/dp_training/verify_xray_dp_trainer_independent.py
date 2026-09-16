#!/usr/bin/env python3
"""Independent verifier for the executable NIH CXR14 DP-trainer gate.

This verifier does not import the mechanism, smoke runner, protocol builder, or
UnitDP reference implementation.  It independently reconstructs commitments,
event-chain hashes, accounting, and clipping-path assertions from artifacts.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "nih-cxr14-dp-trainer-independent-verification/v1"
PROTOCOL_SHA256 = "F2757DC7EBC7488B6A9DD0F69227419EA16BB9A3E9BD4434514CB19AFD2B4018"
CALIBRATION_REPORT_SHA256 = "F9771F169BA3F6468F5AAF254A27AFFE1FB47B1ABA897C365BFFC7CDD11A73F1"
CALIBRATION_GRADIENTS_SHA256 = "9A63FB1546E08F4B4BAD2A6C8027A6A901F2CCE46EFBBC769F390FFD16DA4110"
K10_SHA256 = "2D749FB7B70823114A69FD55921277B0D0FF2F9AEC8FECD239159C553F3A0C06"
SELECTION_SALT = "nih-cxr14-public-lora-dp-step-smoke-clip-path-v1"
TRACE_GENESIS = "0" * 64
EXPECTED_TRACE_EVENT_KEYS = {
    "schema",
    "mechanism_schema",
    "arm",
    "privacy_unit",
    "adjacency",
    "population",
    "expected_batch",
    "poisson_sample_rate",
    "clip_norm",
    "noise_multiplier",
    "noise_std_before_division",
    "fixed_denominator",
    "first_step",
    "event_count",
    "rng_security_mode",
    "rng_backend",
}
FORBIDDEN_PUBLIC_KEYS = {
    "selected_ids",
    "selected_unit_ids",
    "realized_batch",
    "realized_batch_size",
    "realized_unit_count",
    "empty_batch",
    "sampling_seed",
    "noise_seed",
}
RDP_ORDERS = [round(1.0 + index / 10.0, 1) for index in range(1, 100)]
RDP_ORDERS += list(range(12, 64)) + [64, 128, 256, 512]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def stable_hash(*parts: object) -> str:
    return sha256_bytes("|".join(map(str, parts)).encode("utf-8"))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_keys(child)


def verify_trace(trace: list[dict[str, Any]]) -> dict[str, Any]:
    previous = TRACE_GENESIS
    events = 0
    for index, record in enumerate(trace):
        require(set(record) == {"previous_hash", "event", "event_hash"}, f"trace {index} record schema")
        event = record["event"]
        require(set(event) == EXPECTED_TRACE_EVENT_KEYS, f"trace {index} event schema")
        require(not (set(walk_keys(event)) & FORBIDDEN_PUBLIC_KEYS), f"trace {index} realization leak")
        require(record["previous_hash"] == previous, f"trace {index} predecessor mismatch")
        expected = sha256_bytes(canonical_json({"previous_hash": previous, "event": event}))
        require(record["event_hash"] == expected, f"trace {index} hash mismatch")
        require(
            math.isclose(
                float(event["poisson_sample_rate"]),
                int(event["expected_batch"]) / int(event["population"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            f"trace {index} q mismatch",
        )
        require(float(event["fixed_denominator"]) == float(event["expected_batch"]), f"trace {index} denominator")
        require(
            event["adjacency"] == f"add_or_remove_one_{event['privacy_unit']}",
            f"trace {index} adjacency",
        )
        require(
            math.isclose(
                float(event["noise_std_before_division"]),
                float(event["clip_norm"]) * float(event["noise_multiplier"]),
                rel_tol=1e-15,
                abs_tol=0.0,
            ),
            f"trace {index} noise scale",
        )
        events += int(event["event_count"])
        previous = expected
    return {"records": len(trace), "events": events, "head_sha256": previous, "status": "PASS"}


def opacus_epsilon(sigma: float, q: float, steps: int, delta: float) -> tuple[float, float]:
    from opacus.accountants.analysis import rdp

    values = rdp.compute_rdp(q=q, noise_multiplier=sigma, steps=steps, orders=RDP_ORDERS)
    epsilon, order = rdp.get_privacy_spent(orders=RDP_ORDERS, rdp=values, delta=delta)
    return float(epsilon), float(order)


def google_epsilon(sigma: float, q: float, steps: int, delta: float) -> tuple[float, float]:
    from dp_accounting import dp_event
    from dp_accounting.privacy_accountant import NeighboringRelation
    from dp_accounting.rdp import RdpAccountant

    accountant = RdpAccountant(
        orders=RDP_ORDERS,
        neighboring_relation=NeighboringRelation.ADD_OR_REMOVE_ONE,
    )
    accountant.compose(
        dp_event.PoissonSampledDpEvent(q, dp_event.GaussianDpEvent(sigma)),
        count=steps,
    )
    epsilon, order = accountant.get_epsilon_and_optimal_order(delta)
    return float(epsilon), float(order)


def protocol_entry(protocol: dict[str, Any], arm: str) -> dict[str, Any]:
    values = [
        entry
        for entry in protocol["accounting"]["entries"]
        if entry["arm"] == arm and int(entry["cap"]) == 10
    ]
    require(len(values) == 1, f"protocol entry count for {arm}")
    return values[0]


def verify_config(config: dict[str, Any], entry: dict[str, Any], clip_norm: float) -> None:
    require(config["arm"] == entry["arm"], "config arm drift")
    require(config["privacy_unit"] == entry["accounting_unit"], "config unit drift")
    require(config["adjacency"] == entry["adjacency"], "config adjacency drift")
    for key in ("population", "expected_batch", "max_steps"):
        require(int(config[key]) == int(entry[key]), f"config {key} drift")
    require(float(config["fixed_denominator"]) == float(entry["expected_batch"]), "config denominator drift")
    require(float(config["poisson_sample_rate"]) == float(entry["poisson_sample_rate"]), "config q drift")
    require(float(config["noise_multiplier"]) == float(entry["noise_multiplier"]), "config sigma drift")
    require(float(config["clip_norm"]) == clip_norm, "config C drift")
    require(config["rng_security_mode"] == "TEST_ONLY_DETERMINISTIC", "smoke RNG label drift")


def reconstruct_commitments(
    manifest_path: Path, calibration_path: Path
) -> tuple[list[str], list[str], list[str]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = [row for row in csv.DictReader(handle) if row["partition"] == "public_development"]
    require(len(manifest) == 4_831, "independent public manifest count")
    by_image = {row["image_id"]: row for row in manifest}
    require(len({row["patient_id"] for row in manifest}) == 1_816, "independent public patient count")
    with calibration_path.open("r", encoding="utf-8", newline="") as handle:
        calibration = list(csv.DictReader(handle))
    require(len(calibration) == 144, "independent calibration row count")

    image_commitments = []
    patient_commitments = []
    selected_patients = []
    for target in (0, 1):
        candidates = [
            row
            for row in calibration
            if row["unit_type"] == "image" and int(row["target_patient"]) == target
        ]
        selected = max(candidates, key=lambda row: (float(row["gradient_l2_norm"]), row["unit_id"]))
        image_ids = selected["image_ids"].split("|")
        require(len(image_ids) == 1 and image_ids[0] in by_image, "independent M1 selection")
        record = by_image[image_ids[0]]
        require(int(record["target_patient"]) == target, "independent M1 target")
        image_commitments.append(
            stable_hash(SELECTION_SALT, "image", selected["unit_id"], *image_ids)
        )
        selected_patients.append(record["patient_id"])

    for target in (0, 1):
        candidates = [
            row
            for row in calibration
            if row["unit_type"] == "patient"
            and int(row["target_patient"]) == target
            and int(row["image_count"]) == 4
        ]
        selected = max(candidates, key=lambda row: (float(row["gradient_l2_norm"]), row["unit_id"]))
        image_ids = selected["image_ids"].split("|")
        require(len(image_ids) == 4 and all(value in by_image for value in image_ids), "independent M2 selection")
        patients = {by_image[value]["patient_id"] for value in image_ids}
        require(patients == {selected["patient_id"]}, "independent M2 patient composition")
        require(int(by_image[image_ids[0]]["target_patient"]) == target, "independent M2 target")
        patient_commitments.append(
            stable_hash(SELECTION_SALT, "patient", selected["unit_id"], *image_ids)
        )
        selected_patients.append(selected["patient_id"])
    require(len(selected_patients) == len(set(selected_patients)), "independent sentinel patient overlap")
    return image_commitments, patient_commitments, selected_patients


def has_unconditional_randn(function: ast.FunctionDef) -> bool:
    found = []

    def visit(node: ast.AST, conditional_depth: int) -> None:
        if isinstance(node, ast.Call):
            function_value = node.func
            if (
                isinstance(function_value, ast.Attribute)
                and function_value.attr == "randn"
                and isinstance(function_value.value, ast.Name)
                and function_value.value.id == "torch"
            ):
                found.append(conditional_depth)
        next_depth = conditional_depth + int(isinstance(node, (ast.If, ast.IfExp, ast.Match)))
        for child in ast.iter_child_nodes(node):
            visit(child, next_depth)

    visit(function, 0)
    return found == [0]


def static_source_checks(mechanism_path: Path, smoke_path: Path, legacy_path: Path) -> dict[str, Any]:
    mechanism_source = mechanism_path.read_text(encoding="utf-8")
    mechanism_tree = ast.parse(mechanism_source)
    aggregate = next(
        node
        for node in mechanism_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "aggregate_noised_update"
    )
    mean_function = next(
        node
        for node in mechanism_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "mean_unit_vectors"
    )
    require(has_unconditional_randn(aggregate), "noise generation is conditional or missing")
    aggregate_text = ast.get_source_segment(mechanism_source, aggregate) or ""
    mean_text = ast.get_source_segment(mechanism_source, mean_function) or ""
    require("config.fixed_denominator" in aggregate_text, "fixed denominator missing from aggregate")
    require("torch.stack" in mean_text and ".mean(" in mean_text, "patient mean helper missing")

    smoke_source = smoke_path.read_text(encoding="utf-8")
    smoke_tree = ast.parse(smoke_source)
    calls = [node for node in ast.walk(smoke_tree) if isinstance(node, ast.Call)]
    attributes = [node.func.attr for node in calls if isinstance(node.func, ast.Attribute)]
    require("step" in attributes, "optimizer step call missing")
    require("save_pretrained" not in attributes and "save" not in attributes, "model save call found")

    legacy_source = legacy_path.read_text(encoding="utf-8")
    require("if not selected_owners:" in legacy_source, "legacy empty-sample branch not found")
    branch = legacy_source.split("if not selected_owners:", 1)[1].split("for owner in selected_owners:", 1)[0]
    require("continue" in branch, "legacy empty-sample skip not found")
    return {
        "mechanism_sha256": sha256_file(mechanism_path),
        "smoke_runner_sha256": sha256_file(smoke_path),
        "legacy_owa_sha256": sha256_file(legacy_path),
        "unconditional_gaussian_call": True,
        "fixed_denominator_reference": True,
        "patient_mean_helper": True,
        "optimizer_step_call": True,
        "model_save_call_absent": True,
        "legacy_empty_poisson_skip_confirmed": True,
        "status": "PASS",
    }


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    protocol_path = root / "_reports" / "nih_cxr14_dp_attack_protocol_v1_001" / "protocol.json"
    calibration_report_path = root / "_reports" / "nih_cxr14_public_clip_calibration_v1_001" / "report.json"
    calibration_gradients_path = root / "_reports" / "nih_cxr14_public_clip_calibration_v1_001" / "gradient_norms_private.csv"
    manifest_path = root / "_data" / "derived" / "nih_cxr14_pa_target_enriched_v1" / "k10_private.csv"
    synthetic_dir = root / "_reports" / "nih_cxr14_dp_trainer_synthetic_v1_001"
    smoke_dir = root / "_reports" / "nih_cxr14_public_lora_dp_step_smoke_v1_001"
    output_dir = root / "_reports" / "nih_cxr14_dp_trainer_executable_gate_v1_001"
    output_dir.mkdir(parents=True, exist_ok=True)

    require(sha256_file(protocol_path) == PROTOCOL_SHA256, "protocol hash mismatch")
    require(sha256_file(calibration_report_path) == CALIBRATION_REPORT_SHA256, "calibration report hash mismatch")
    require(sha256_file(calibration_gradients_path) == CALIBRATION_GRADIENTS_SHA256, "calibration gradients hash mismatch")
    require(sha256_file(manifest_path) == K10_SHA256, "K10 hash mismatch")
    protocol = load_json(protocol_path)
    synthetic = load_json(synthetic_dir / "report.json")
    smoke = load_json(smoke_dir / "report.json")
    require(synthetic["status"] == "PASS_SYNTHETIC_DP_MECHANISM_CONFORMANCE", "synthetic gate status")
    require(smoke["status"] == "PASS_PUBLIC_DISPOSABLE_ONE_STEP_CONFORMANCE", "smoke gate status")
    require("NOT_PRIVATE_TRAINING" in smoke["scope"], "smoke scope boundary")

    checks: dict[str, Any] = {}
    image_entry = protocol_entry(protocol, "M1-I8")
    patient_entry = protocol_entry(protocol, "M2-P8")
    clip_norms = {"M1-I8": 0.28448700606156724, "M2-P8": 0.1997973088974048}
    verify_config(smoke["mechanism_configs"]["M1-I8"], image_entry, clip_norms["M1-I8"])
    verify_config(smoke["mechanism_configs"]["M2-P8"], patient_entry, clip_norms["M2-P8"])
    verify_config(synthetic["mechanism_configs"]["M1_I8_K10"], image_entry, clip_norms["M1-I8"])
    verify_config(synthetic["mechanism_configs"]["M2_P8_K10"], patient_entry, clip_norms["M2-P8"])
    checks["mechanism_config_alignment"] = {"arms": ["M1-I8", "M2-P8"], "status": "PASS"}

    synthetic_trace_bundle = load_json(synthetic_dir / "public_event_trace.json")
    require(
        sha256_file(synthetic_dir / "public_event_trace.json")
        == synthetic["artifacts"]["public_event_trace_sha256"],
        "synthetic trace artifact hash",
    )
    require(
        sha256_file(synthetic_dir / "synthetic_private_diagnostics.json")
        == synthetic["artifacts"]["synthetic_private_diagnostics_sha256"],
        "synthetic diagnostics artifact hash",
    )
    synthetic_trace = verify_trace(synthetic_trace_bundle["trace"])
    smoke_trace = verify_trace(smoke["public_schedule_trace"])
    require(synthetic_trace["events"] == 8_000, "synthetic trace event count")
    require(smoke_trace["events"] == 2, "smoke trace event count")
    checks["independent_trace_verification"] = {
        "synthetic": synthetic_trace,
        "smoke": smoke_trace,
        "status": "PASS",
    }

    gaussian = synthetic["checks"]["gaussian_moments"]
    require(abs(float(gaussian["standardized_mean"])) <= float(gaussian["mean_abs_tolerance"]), "Gaussian mean")
    require(abs(float(gaussian["standardized_std"]) - 1.0) <= float(gaussian["std_relative_tolerance"]), "Gaussian std")
    poisson = synthetic["checks"]["poisson_sampling"]
    require(float(poisson["mean_standard_error_z"]) <= 5.0, "Poisson mean z")
    require(float(poisson["empty_frequency_standard_error_z"]) <= 5.0, "Poisson empty z")
    require(synthetic["checks"]["empty_noise_only_update"]["nonzero"], "empty update was zero")
    require(synthetic["checks"]["empty_noise_only_update"]["replay_exact"], "empty replay")
    require(synthetic["checks"]["public_event_trace"]["tamper_detected"], "synthetic tamper detection")
    require(
        all(item["rejected"] for item in synthetic["checks"]["fail_closed_configuration"]["cases"]),
        "malformed config accepted",
    )
    checks["synthetic_statistics_and_fail_closed"] = {"status": "PASS"}

    accountant_rows = []
    for entry in (image_entry, patient_entry):
        sigma = float(entry["noise_multiplier"])
        q = float(entry["poisson_sample_rate"])
        steps = int(entry["max_steps"])
        delta = float(entry["target_delta"])
        opacus = opacus_epsilon(sigma, q, steps, delta)
        google = google_epsilon(sigma, q, steps, delta)
        opacus_drift = abs(opacus[0] - float(entry["opacus_epsilon"]))
        google_drift = abs(google[0] - float(entry["google_dp_accounting_epsilon"]))
        require(opacus_drift <= 1e-12 and google_drift <= 1e-12, "independent accountant drift")
        accountant_rows.append(
            {
                "arm": entry["arm"],
                "opacus_epsilon": opacus[0],
                "google_epsilon": google[0],
                "maximum_absolute_drift": max(opacus_drift, google_drift),
            }
        )
    checks["independent_accountant_replay"] = {
        "orders": len(RDP_ORDERS),
        "entries": accountant_rows,
        "status": "PASS",
    }

    image_commitments, patient_commitments, selected_patients = reconstruct_commitments(
        manifest_path, calibration_gradients_path
    )
    require(image_commitments == smoke["selection"]["M1_unit_commitments"], "M1 selection commitment")
    require(patient_commitments == smoke["selection"]["M2_unit_commitments"], "M2 selection commitment")
    require(smoke["selection"]["M1_target_control"] == [0, 1], "M1 target/control")
    require(smoke["selection"]["M2_target_control"] == [0, 1], "M2 target/control")
    require(not smoke["selection"]["uses_private_train"], "private train used in smoke")
    require(not smoke["selection"]["uses_privacy_attack_holdout"], "attack holdout used in smoke")
    require(not smoke["selection"]["uses_final_test"], "final test used in smoke")
    checks["independent_public_selection"] = {
        "M1_commitments": image_commitments,
        "M2_commitments": patient_commitments,
        "disjoint_patients": len(selected_patients),
        "status": "PASS",
    }

    initial_digests = set()
    arm_checks = []
    for arm, expected_images in (("M1-I8", 2), ("M2-P8", 8)):
        result = smoke["arms"][arm]
        config = smoke["mechanism_configs"][arm]
        diagnostics = result["aggregation_private_diagnostics_public_data_only"]
        require(result["selected_units"] == 2 and result["selected_images"] == expected_images, f"{arm} sizes")
        require(len(result["unit_summaries"]) == 2, f"{arm} summary count")
        require(
            all(float(item["gradient_l2_norm"]) > float(config["clip_norm"]) for item in result["unit_summaries"]),
            f"{arm} real norm did not cross C",
        )
        require(diagnostics["clipped_unit_count"] == 2, f"{arm} clip count")
        require(diagnostics["realized_unit_count"] == 2 and not diagnostics["empty_sample"], f"{arm} realized units")
        require(float(diagnostics["fixed_denominator_used"]) == float(config["expected_batch"]), f"{arm} denominator")
        require(
            math.isclose(
                float(diagnostics["noise_std_before_division"]),
                float(config["clip_norm"]) * float(config["noise_multiplier"]),
                rel_tol=1e-15,
            ),
            f"{arm} noise std",
        )
        require(result["test_noise_replay_exact"], f"{arm} replay")
        require(result["real_gradient_sentinel_replay"]["repeat_exact"], f"{arm} gradient replay")
        require(result["optimizer_step"] and result["parameters_finite_after_step"], f"{arm} optimizer")
        require(float(result["adapter_delta_l2_norm"]) > 0.0, f"{arm} zero adapter delta")
        require(int(result["changed_trainable_scalars"]) == 1_659_904, f"{arm} changed scalars")
        require(result["adapter_initial_sha256"] != result["adapter_final_sha256"], f"{arm} digest unchanged")
        require(not result["checkpoint_written"], f"{arm} checkpoint flag")
        initial_digests.add(result["adapter_initial_sha256"])
        arm_checks.append(
            {
                "arm": arm,
                "real_units_clipped": diagnostics["clipped_unit_count"],
                "adapter_delta_l2_norm": result["adapter_delta_l2_norm"],
                "status": "PASS",
            }
        )
    require(len(initial_digests) == 1 and smoke["identical_initial_adapter_across_arms"], "initial adapter mismatch")
    checks["real_lora_one_step"] = {"arms": arm_checks, "status": "PASS"}

    checkpoint_suffixes = {".pt", ".pth", ".ckpt", ".safetensors", ".bin"}
    checkpoint_files = [
        str(path.relative_to(smoke_dir))
        for path in smoke_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in checkpoint_suffixes
    ]
    require(not checkpoint_files, "checkpoint-like smoke artifact found")
    require(not smoke["model_lifecycle"]["checkpoint_written"], "lifecycle checkpoint flag")
    require(smoke["model_lifecycle"]["model_discarded_after_arm"], "model discard flag")
    checks["artifact_minimization"] = {
        "checkpoint_like_files": checkpoint_files,
        "smoke_output_bytes": sum(path.stat().st_size for path in smoke_dir.rglob("*") if path.is_file()),
        "synthetic_output_bytes": sum(path.stat().st_size for path in synthetic_dir.rglob("*") if path.is_file()),
        "status": "PASS",
    }

    source_checks = static_source_checks(
        root / "dp_training" / "mechanism.py",
        root / "dp_training" / "run_xray_public_lora_dp_step_smoke.py",
        root / "unitdp_compiler_reference" / "src" / "unitdp" / "owa_dpsgd.py",
    )
    checks["static_source_checks"] = source_checks

    report = {
        "schema": SCHEMA,
        "status": "PASS_EXECUTABLE_DP_TRAINER_GATE_PRIVATE_PILOT_STILL_BLOCKED",
        "scope": "INDEPENDENT_ARTIFACT_AND_SOURCE_VERIFICATION_NO_MODEL_LOAD_NO_PRIVATE_TRAINING",
        "inputs": {
            "protocol_sha256": sha256_file(protocol_path),
            "calibration_report_sha256": sha256_file(calibration_report_path),
            "calibration_gradients_sha256": sha256_file(calibration_gradients_path),
            "k10_manifest_sha256": sha256_file(manifest_path),
            "synthetic_report_sha256": sha256_file(synthetic_dir / "report.json"),
            "public_smoke_report_sha256": sha256_file(smoke_dir / "report.json"),
        },
        "checks": checks,
        "legacy_reuse_decision": {
            "status": "REJECT_AS_EXECUTION_PATH",
            "reason": "the legacy owner trainer continues on an empty Bernoulli sample instead of executing the frozen Gaussian-noise-only update",
            "technical_evidence_reusable": True,
        },
        "gate_boundary": {
            "private_train_images_opened": False,
            "frozen_manifest_read_for_contract_checks": True,
            "private_optimizer_training_started": False,
            "release_grade_rng_available": False,
            "formal_dp_release_claim_allowed": False,
            "next_action": "report this gate before any private pilot",
        },
        "environment": {"python": platform.python_version()},
    }
    output_path = output_dir / "independent_verification.json"
    write_json(output_path, report)
    print(json.dumps({"status": report["status"], "report": str(output_path), "checks": len(checks)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
