#!/usr/bin/env python3
"""Independent verifier for the four-step K5 research runtime dry-run.

The verifier deliberately imports neither the dry-run runner nor the DP
mechanism. It reads the local restricted diagnostic only to verify unit
membership and execution invariants; identifiers are not copied to its output.
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


SCHEMA = "nih-cxr14-k5-private-research-dryrun-independent-verification/v1"
DRYRUN_PROTOCOL_SHA256 = "0509D745E8B192FC2CE7694EBDA0C36AB60005D704864D27A44681FD6E8432C8"
UPSTREAM_PROTOCOL_SHA256 = "F2757DC7EBC7488B6A9DD0F69227419EA16BB9A3E9BD4434514CB19AFD2B4018"
EXECUTABLE_GATE_SHA256 = "EA2041E282153A547EEE89BBF1D6533315D7C071B5B712907FB2DCC5E6BE7A2C"
K5_SHA256 = "DC49D82E497EAA6940DCF92C8F773179B8F7A838057A82AE0C26A69CA785BFC5"
TRACE_GENESIS = "0" * 64
PUBLIC_EVENT_KEYS = {
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
FORBIDDEN_TRACE_KEYS = {
    "selected_ids",
    "selected_unit_ids",
    "realized_batch",
    "realized_batch_size",
    "realized_unit_count",
    "empty_batch",
    "sampling_seed",
    "noise_seed",
}
FORBIDDEN_SERIALIZED_SECRET_KEYS = {
    "master_secret",
    "sampling_seed",
    "noise_seed",
    "dp_noise_seed",
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


def walk_string_values(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for child in value.values():
            yield from walk_string_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_string_values(child)
    elif isinstance(value, str):
        yield value


def verify_trace(trace: list[dict[str, Any]], expected_arm: str) -> dict[str, Any]:
    previous = TRACE_GENESIS
    for index, record in enumerate(trace, start=1):
        require(set(record) == {"previous_hash", "event", "event_hash"}, "trace record schema")
        event = record["event"]
        require(set(event) == PUBLIC_EVENT_KEYS, "trace event schema")
        require(not (set(walk_keys(event)) & FORBIDDEN_TRACE_KEYS), "trace realization leak")
        require(event["arm"] == expected_arm, "trace arm")
        require(event["first_step"] == index and event["event_count"] == 1, "trace step sequence")
        require(event["rng_security_mode"] == "RESEARCH_ONLY_NONCRYPTOGRAPHIC", "trace RNG label")
        require(record["previous_hash"] == previous, "trace predecessor")
        expected_hash = sha256_bytes(canonical_json({"previous_hash": previous, "event": event}))
        require(record["event_hash"] == expected_hash, "trace hash")
        require(
            float(event["fixed_denominator"]) == float(event["expected_batch"]),
            "trace fixed denominator",
        )
        require(
            math.isclose(
                float(event["poisson_sample_rate"]),
                float(event["expected_batch"]) / int(event["population"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            "trace q",
        )
        require(
            math.isclose(
                float(event["noise_std_before_division"]),
                float(event["clip_norm"]) * float(event["noise_multiplier"]),
                rel_tol=1e-15,
                abs_tol=0.0,
            ),
            "trace noise scale",
        )
        previous = expected_hash
    require(len(trace) == 4, "trace must contain four events")
    return {"records": len(trace), "head_sha256": previous, "status": "PASS"}


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


def verify_static_source(runner_path: Path) -> dict[str, Any]:
    source = runner_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_names = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    require("build_runtime_schedules" in function_names, "schedule builder missing")
    require("gradient_for_unit" in function_names and "run_arm" in function_names, "runtime functions missing")
    require("poisson_select" in source, "Poisson selection call missing")
    require("torch.randperm" in source, "within-patient uniform permutation missing")
    require("aggregate_noised_update" in source, "DP aggregate call missing")
    require("secrets.token_bytes(32)" in source, "ephemeral 256-bit entropy call missing")
    require("del master_secret" in source, "master secret lifetime boundary missing")

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    attributes = [node.func.attr for node in calls if isinstance(node.func, ast.Attribute)]
    require("step" in attributes, "optimizer step missing")
    require("save" not in attributes and "save_pretrained" not in attributes, "model save call found")
    for call in calls:
        if isinstance(call.func, ast.Attribute) and call.func.attr == "manual_seed" and call.args:
            require(
                not isinstance(call.args[0], ast.Constant) or not isinstance(call.args[0].value, int),
                "literal private generator seed found",
            )

    main_function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    main_source = ast.get_source_segment(source, main_function) or ""
    schedule_position = main_source.find("build_runtime_schedules(")
    latent_position = main_source.find("prepare_latents_and_text(")
    arm_position = main_source.find("run_arm(")
    require(0 <= schedule_position < latent_position < arm_position, "schedule is not fixed before model gradients")
    return {
        "runner_sha256": sha256_file(runner_path),
        "ephemeral_entropy": True,
        "schedule_before_model_gradient": True,
        "poisson_outer_sampling": True,
        "uniform_without_replacement_inner_sampling": True,
        "literal_private_seed_absent": True,
        "model_save_call_absent": True,
        "status": "PASS",
    }


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    protocol_path = root / "dp_training" / "private_research_dryrun_protocol.json"
    upstream_path = root / "_reports" / "nih_cxr14_dp_attack_protocol_v1_001" / "protocol.json"
    executable_gate_path = root / "_reports" / "nih_cxr14_dp_trainer_executable_gate_v1_001" / "independent_verification.json"
    manifest_path = root / "_data" / "derived" / "nih_cxr14_pa_target_enriched_v1" / "k5_private.csv"
    dryrun_dir = root / "_reports" / "nih_cxr14_k5_private_research_dryrun_v1_001"
    output_dir = root / "_reports" / "nih_cxr14_k5_private_research_dryrun_gate_v1_001"
    public_report_path = dryrun_dir / "public_report.json"
    trace_path = dryrun_dir / "public_event_traces.json"
    restricted_path = dryrun_dir / "restricted_runtime_diagnostics.json"

    require(sha256_file(protocol_path) == DRYRUN_PROTOCOL_SHA256, "dry-run protocol hash")
    require(sha256_file(upstream_path) == UPSTREAM_PROTOCOL_SHA256, "upstream protocol hash")
    require(sha256_file(executable_gate_path) == EXECUTABLE_GATE_SHA256, "executable gate hash")
    require(sha256_file(manifest_path) == K5_SHA256, "K5 manifest hash")
    protocol = load_json(protocol_path)
    report = load_json(public_report_path)
    trace_bundle = load_json(trace_path)
    restricted = load_json(restricted_path)
    require(report["status"] == "PASS_K5_PRIVATE_RESEARCH_RUNTIME_DRYRUN", "public report status")
    require(report["runtime"]["steps_per_arm"] == 4, "public report steps")
    require(report["runtime"]["checkpoint_written"] is False, "public checkpoint flag")
    require(report["randomness"]["release_eligible"] is False, "release eligibility")
    require(report["randomness"]["formal_dp_claim_allowed"] is False, "formal claim flag")
    require(report["randomness"]["private_seed_values_serialized"] is False, "seed serialization flag")
    require(sha256_file(trace_path) == report["artifacts"]["public_event_traces_sha256"], "trace artifact hash")
    require(
        sha256_file(restricted_path) == report["artifacts"]["restricted_runtime_diagnostics_sha256"],
        "restricted artifact hash",
    )
    require(not (set(walk_keys(restricted)) & FORBIDDEN_SERIALIZED_SECRET_KEYS), "serialized private seed key")

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = [row for row in csv.DictReader(handle) if row["partition"] == "private_train"]
    require(len(manifest_rows) == 18_393, "K5 private image count")
    by_image = {row["image_id"]: row for row in manifest_rows}
    by_patient: dict[str, list[dict[str, str]]] = {}
    for row in manifest_rows:
        by_patient.setdefault(row["patient_id"], []).append(row)
    require(len(by_patient) == 8_476, "K5 private patient count")

    public_strings = set(walk_string_values(report)) | set(walk_string_values(trace_bundle))
    selected_ids: set[str] = set()
    checks: dict[str, Any] = {}
    arm_private = restricted["arms"]
    protocol_arms = protocol["runtime"]["arm_order"]
    require(protocol_arms == ["M1-I8", "M1-G8", "M2-P8"], "protocol arm order")
    require(set(report["arms"]) == set(protocol_arms), "public arm set")
    require(set(arm_private) == set(protocol_arms), "restricted arm set")

    for arm in protocol_arms:
        spec = protocol["mechanisms"][arm]
        public_config = report["mechanism_configs"][arm]
        for key in (
            "privacy_unit",
            "adjacency",
            "population",
            "expected_batch",
            "poisson_sample_rate",
            "clip_norm",
            "noise_multiplier",
            "fixed_denominator",
        ):
            require(public_config[key] == spec[key], f"{arm} public config drift: {key}")
        require(public_config["max_steps"] == 4_000, f"{arm} maximum accounting steps")
        require(
            public_config["rng_security_mode"] == "RESEARCH_ONLY_NONCRYPTOGRAPHIC",
            f"{arm} public RNG classification",
        )

    restricted_checks = []
    for arm in protocol_arms:
        steps = arm_private[arm]
        config = protocol["mechanisms"][arm]
        require(len(steps) == 4, f"{arm} restricted step count")
        require(report["arms"][arm]["completed_steps"] == 4, f"{arm} completed steps")
        require(report["arms"][arm]["status"] == "PASS_FOUR_STEP_RUNTIME", f"{arm} status")
        require(report["arms"][arm]["adapter_total_delta_l2_norm"] > 0.0, f"{arm} total delta")
        require(report["arms"][arm]["changed_trainable_scalars"] > 0, f"{arm} changed scalars")
        require(report["arms"][arm]["parameters_and_optimizer_finite"], f"{arm} finite flag")
        require(not report["arms"][arm]["checkpoint_written"], f"{arm} checkpoint")
        for expected_step, step in enumerate(steps, start=1):
            require(step["step"] == expected_step, f"{arm} step sequence")
            units = step["units"]
            require(step["selected_unit_count"] == len(units), f"{arm} selected count")
            require(
                step["selected_raw_image_count"] == sum(len(unit["image_ids"]) for unit in units),
                f"{arm} raw image count",
            )
            require(step["empty_sample"] == (len(units) == 0), f"{arm} empty flag")
            unit_ids = [unit["unit_id"] for unit in units]
            require(len(unit_ids) == len(set(unit_ids)), f"{arm} duplicate unit within step")
            clipped = 0
            for unit in units:
                selected_ids.add(str(unit["unit_id"]))
                selected_ids.update(map(str, unit["image_ids"]))
                require(math.isfinite(float(unit["loss"])), f"{arm} non-finite loss")
                require(math.isfinite(float(unit["gradient_l2_norm"])), f"{arm} non-finite norm")
                expected_clipped = float(unit["gradient_l2_norm"]) > float(config["clip_norm"])
                require(unit["was_clipped"] == expected_clipped, f"{arm} clip flag")
                clipped += int(expected_clipped)
                if arm.startswith("M1"):
                    require(unit["unit_type"] == "image" and len(unit["image_ids"]) == 1, "M1 unit boundary")
                    image_id = unit["image_ids"][0]
                    require(image_id == unit["unit_id"] and image_id in by_image, "M1 manifest membership")
                    require(by_image[image_id]["patient_id"] == unit["patient_id"], "M1 patient mapping")
                    require(not unit["mean_before_outer_clip"], "M1 mean flag")
                else:
                    require(unit["unit_type"] == "patient", "M2 unit type")
                    require(1 <= len(unit["image_ids"]) <= 4, "M2 image bound")
                    require(unit["unit_id"] in by_patient, "M2 patient membership")
                    require(len(unit["image_ids"]) == len(set(unit["image_ids"])), "M2 duplicate image")
                    require(
                        all(
                            image_id in by_image
                            and by_image[image_id]["patient_id"] == unit["unit_id"]
                            and int(by_image[image_id]["cap_rank"]) <= 5
                            for image_id in unit["image_ids"]
                        ),
                        "M2 patient/image boundary",
                    )
                    require(unit["patient_id"] == unit["unit_id"], "M2 patient id")
                    require(unit["mean_before_outer_clip"], "M2 mean-before-clip flag")
                require(len(unit["timesteps"]) == len(unit["image_ids"]), "timestep count")
                require(len(unit["diffusion_noise_sha256"]) == len(unit["image_ids"]), "noise hash count")
                require(all(0 <= int(value) < 1000 for value in unit["timesteps"]), "timestep range")
            aggregation = step["aggregation"]
            require(aggregation["realized_unit_count"] == len(units), f"{arm} aggregate count")
            require(aggregation["empty_sample"] == (len(units) == 0), f"{arm} aggregate empty")
            require(aggregation["clipped_unit_count"] == clipped, f"{arm} aggregate clips")
            require(
                float(aggregation["fixed_denominator_used"]) == float(config["fixed_denominator"]),
                f"{arm} aggregate denominator",
            )
            require(
                math.isclose(
                    float(aggregation["noise_std_before_division"]),
                    float(config["noise_multiplier"]) * float(config["clip_norm"]),
                    rel_tol=1e-15,
                ),
                f"{arm} aggregate noise std",
            )
            require(step["optimizer_state_finite"], f"{arm} optimizer state")
            require(float(step["adapter_step_delta_l2_norm"]) > 0.0, f"{arm} zero step")
            if arm.startswith("M1"):
                require(
                    step["selected_unit_count"]
                    <= protocol["resource_fail_closed"]["maximum_realized_image_units_per_m1_step"],
                    f"{arm} resource cap",
                )
            else:
                require(
                    step["selected_unit_count"]
                    <= protocol["resource_fail_closed"]["maximum_realized_patient_units_per_m2_step"],
                    "M2 patient resource cap",
                )
                require(
                    step["selected_raw_image_count"]
                    <= protocol["resource_fail_closed"]["maximum_realized_raw_images_per_m2_step"],
                    "M2 image resource cap",
                )
        restricted_checks.append({"arm": arm, "steps": 4, "status": "PASS"})

    schedule_counts = restricted["schedule_counts"]
    require(
        schedule_counts["m1_units"]
        == sum(step["selected_unit_count"] for step in arm_private["M1-I8"]),
        "M1 aggregate schedule count",
    )
    require(
        schedule_counts["m2_units"]
        == sum(step["selected_unit_count"] for step in arm_private["M2-P8"]),
        "M2 aggregate unit count",
    )
    require(
        schedule_counts["m2_images"]
        == sum(step["selected_raw_image_count"] for step in arm_private["M2-P8"]),
        "M2 aggregate image count",
    )

    require(not (selected_ids & public_strings), "selected identifier leaked into public artifacts")
    require(restricted["seed_values_serialized"] is False, "restricted seed flag")
    checks["restricted_unit_and_step_invariants"] = {
        "arms": restricted_checks,
        "identifiers_copied_to_verifier_output": False,
        "status": "PASS",
    }

    # M1 schedules and diffusion inputs are matched, while the independent DP
    # Gaussian streams and later parameter states are allowed to differ.
    for step_i8, step_g8 in zip(arm_private["M1-I8"], arm_private["M1-G8"]):
        require(step_i8["selected_unit_count"] == step_g8["selected_unit_count"], "M1 matched count")
        require(len(step_i8["units"]) == len(step_g8["units"]), "M1 matched units")
        for unit_i8, unit_g8 in zip(step_i8["units"], step_g8["units"]):
            for key in ("unit_id", "patient_id", "image_ids", "timesteps", "diffusion_noise_sha256"):
                require(unit_i8[key] == unit_g8[key], f"M1 matched field: {key}")
    require(
        [unit["gradient_sha256"] for unit in arm_private["M1-I8"][0]["units"]]
        == [unit["gradient_sha256"] for unit in arm_private["M1-G8"][0]["units"]],
        "M1 first-step gradient control",
    )
    checks["m1_control_matching"] = {"four_steps": True, "first_gradient_exact": True, "status": "PASS"}

    traces = trace_bundle["traces"]
    trace_checks = {arm: verify_trace(traces[arm], arm) for arm in protocol_arms}
    require(trace_bundle["session_commitment"] == report["randomness"]["session_commitment"], "session commitment")
    checks["public_event_traces"] = {"arms": trace_checks, "status": "PASS"}

    accounting_checks = []
    for arm in protocol_arms:
        spec = protocol["mechanisms"][arm]
        recorded = report["accounting_at_completed_events"][arm]
        opacus = opacus_epsilon(
            float(spec["noise_multiplier"]),
            float(spec["poisson_sample_rate"]),
            4,
            float(spec["accounting_delta"]),
        )
        google = google_epsilon(
            float(spec["noise_multiplier"]),
            float(spec["poisson_sample_rate"]),
            4,
            float(spec["accounting_delta"]),
        )
        drift = max(
            abs(opacus[0] - float(recorded["opacus_epsilon"])),
            abs(google[0] - float(recorded["google_epsilon"])),
        )
        require(drift <= 1e-12, f"{arm} accounting drift")
        require(recorded["events"] == 4 and recorded["accounting_only_not_release_evidence"], "accounting boundary")
        accounting_checks.append({"arm": arm, "maximum_absolute_drift": drift, "status": "PASS"})
    checks["independent_accounting"] = {"orders": len(RDP_ORDERS), "arms": accounting_checks, "status": "PASS"}

    initial_digests = {report["arms"][arm]["adapter_initial_sha256"] for arm in protocol_arms}
    require(len(initial_digests) == 1, "initial adapter mismatch")
    require(report["runtime"]["identical_initial_adapter_across_arms"], "initial adapter flag")
    require(report["runtime"]["M1_shared_schedule_and_diffusion_draws"], "M1 control flag")
    checks["model_lifecycle"] = {
        "identical_initial_adapter": True,
        "fresh_model_per_arm": report["runtime"]["fresh_model_per_arm"],
        "discarded_after_each_arm": report["runtime"]["model_discarded_after_each_arm"],
        "status": "PASS",
    }

    checkpoint_suffixes = {".pt", ".pth", ".ckpt", ".safetensors", ".bin"}
    checkpoint_files = [
        str(path.relative_to(dryrun_dir))
        for path in dryrun_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in checkpoint_suffixes
    ]
    require(not checkpoint_files, "checkpoint-like artifact")
    checks["artifact_minimization"] = {
        "checkpoint_like_files": checkpoint_files,
        "dryrun_files": sum(1 for path in dryrun_dir.rglob("*") if path.is_file()),
        "dryrun_bytes": sum(path.stat().st_size for path in dryrun_dir.rglob("*") if path.is_file()),
        "status": "PASS",
    }

    checks["static_source"] = verify_static_source(
        root / "dp_training" / "run_xray_k5_private_research_dryrun.py"
    )

    report_out = {
        "schema": SCHEMA,
        "status": "PASS_K5_PRIVATE_RESEARCH_DRYRUN_INDEPENDENT",
        "scope": "LOCAL_ARTIFACT_SOURCE_AND_ACCOUNTING_VERIFICATION_NO_MODEL_LOAD_NO_IDENTIFIER_EXPORT",
        "inputs": {
            "dryrun_protocol_sha256": sha256_file(protocol_path),
            "upstream_protocol_sha256": sha256_file(upstream_path),
            "executable_gate_sha256": sha256_file(executable_gate_path),
            "k5_manifest_sha256": sha256_file(manifest_path),
            "public_report_sha256": sha256_file(public_report_path),
            "public_event_traces_sha256": sha256_file(trace_path),
            "restricted_runtime_diagnostics_sha256": sha256_file(restricted_path),
        },
        "checks": checks,
        "privacy_boundary": {
            "restricted_diagnostic_read_locally": True,
            "selected_identifiers_copied_to_output": False,
            "private_randomness_seed_available_to_verifier": False,
            "release_grade_rng": False,
            "formal_dp_release_claim_allowed": False,
        },
        "decision": {
            "runtime_dryrun": "PASS",
            "k5_4000_step_feasibility_started": False,
            "k10_main_started": False,
            "next": "report before any K5 4000-step feasibility implementation or execution",
        },
        "environment": {"python": platform.python_version()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "independent_verification.json"
    write_json(output_path, report_out)
    print(json.dumps({"status": report_out["status"], "report": str(output_path), "checks": len(checks)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
