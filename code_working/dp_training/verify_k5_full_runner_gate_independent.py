#!/usr/bin/env python3
"""Independent static/dynamic audit of the frozen K5 full-runner gate.

This verifier deliberately does not import ``k5_full_runner``.  It re-hashes
the frozen inputs, replays the accountant, inspects the source AST/text,
re-runs the test suite, and invokes the runner's read-only validation command
before atomically publishing launch authority.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


GATE_NAME = "nih_cxr14_k5_full_runner_gate_v1_001"
PRIMARY_SCHEMA = "nih-cxr14-k5-full-runner-primary-gate/v1"
ENV_SCHEMA = "nih-cxr14-k5-full-runner-environment/v1"
AUTHORITY_SCHEMA = "nih-cxr14-k5-full-runner-independent-launch-authority/v1"
INDEPENDENT_SCHEMA = "nih-cxr14-k5-full-runner-independent-verification/v1"
RUN_ID = "nih_cxr14_k5_feasibility_v1_001"
PROTOCOL_SHA256 = "2234B3BA8701565B768A11AB5196B2F696788900D07254833997D7823489E9DA"
K5_SHA256 = "DC49D82E497EAA6940DCF92C8F773179B8F7A838057A82AE0C26A69CA785BFC5"
PREPROCESSING_SHA256 = "0ED9E434764BDB2D542BB16AB951F856CB5B82D4F2E6B9F25F6C554F1076669D"
GENERATION_TASKS_SHA256 = "B0379A2A1EBDFB8758AAD865D6F709D803C7632D2697A7BB9A5DFD9C23F2124B"
REAL_REFERENCE_SHA256 = "99972FC6BAB402F620D2F6B142348CF6A3F16B60BFD63F41EB877FFDDAAA898E"
ACTUAL_PROTOCOL_SHA256 = "3921227498086C6D3E794FBE92DE12DF229E463D9F461642BDF71187B704924D"
ACTUAL_RESULT_SHA256 = "C7BC6F12B02980E3ADBF309DBD16479DC7DCCB67418B65D2F1B5FCD9D8540CD2"
EVALUATOR_PRIMARY_SHA256 = "914002B52C2EF48C551FFD9A5CD0260BA7708EA6492F5296AEF98BF6D0B4F5B5"
EVALUATOR_INDEPENDENT_SHA256 = "A55672E5D5C68491C75AAE4770F1C13A1F1B43BCCD721E7F732BF3C8D9560CAD"
SYNTHETIC_RESUME_PRIMARY_SHA256 = "89A08E9B1AAC248A3DF4F4935C82354D49D28AFC0DD8BE632DE6A80C394FCBFF"
SYNTHETIC_RESUME_INDEPENDENT_SHA256 = "720D7A9DDE7A9DD862AE84B4053DBF0F704B8100CE0C7166D28B56EFE76BA463"
ARMS = ("M0", "M1-I8", "M1-G8", "M2-P8")
RUNNER_SOURCE_FILES = (
    "dp_training/k5_full_runner.py",
    "dp_training/run_k5_full_training.py",
    "dp_training/mechanism.py",
    "dp_training/secure_resume.py",
    "dp_protocol/calibrate_xray_public_clip_norms.py",
    "data_pipeline/nih_cxr14_model_input.py",
)
AUDIT_SOURCE_FILES = (
    "dp_training/run_k5_actual_sd21_resume_preflight.py",
    "dp_training/build_k5_actual_sd21_resume_preflight_protocol.py",
    "dp_training/test_k5_full_runner.py",
    "dp_training/freeze_k5_full_runner_gate.py",
    "dp_training/verify_k5_full_runner_gate_independent.py",
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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def constants_from_ast(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value_node = node.value
            for target in targets:
                if isinstance(target, ast.Name):
                    try:
                        values[target.id] = ast.literal_eval(value_node)
                    except (ValueError, TypeError):
                        pass
    return values


def static_source_audit(root: Path) -> dict[str, Any]:
    core_path = root / "dp_training" / "k5_full_runner.py"
    cli_path = root / "dp_training" / "run_k5_full_training.py"
    mechanism_path = root / "dp_training" / "mechanism.py"
    secure_path = root / "dp_training" / "secure_resume.py"
    core_source = core_path.read_text(encoding="utf-8")
    cli_source = cli_path.read_text(encoding="utf-8")
    mechanism_source = mechanism_path.read_text(encoding="utf-8")
    secure_source = secure_path.read_text(encoding="utf-8")
    core_ast = ast.parse(core_source)
    cli_ast = ast.parse(cli_source)
    core_functions = {
        node.name for node in ast.walk(core_ast) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    cli_functions = {
        node.name for node in ast.walk(cli_ast) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    constants = constants_from_ast(core_path)
    require(constants["FULL_STEPS"] == 4_000, "source full-step constant")
    require(constants["RESUME_INTERVAL"] == 250, "source resume interval")
    require(constants["ADAPTER_STEPS"] == (1_000, 2_000, 4_000), "source adapter steps")
    require(constants["TRAINABLE_PARAMETERS"] == 1_659_904, "source trainable count")
    require(tuple(constants["ARMS"]) == ARMS, "source arm order")
    required_core_functions = {
        "build_arm_schedule",
        "create_training_state",
        "compute_unit_gradient",
        "run_one_step",
        "capture_resume_payload",
        "restore_training_state",
        "save_adapter_snapshot",
        "require_free_space",
    }
    require(required_core_functions <= core_functions, "full-runner core function missing")
    require({"initialize", "run_arm", "enforce_arm_order", "exclusive_run_lock"} <= cli_functions, "CLI lifecycle function missing")
    for token in (
        '"m1-shared-poisson-selection"',
        '"m1-shared-diffusion-draws"',
        'f"{arm}-dp-gaussian-stream"',
        "mean()",
        "aggregate_noised_update",
        "torch.Generator(device=\"cuda\").manual_seed",
        "foreach=False",
        "fused=False",
    ):
        require(token in core_source, f"core static token: {token}")
    for token in (
        'choices=("validate", "initialize", "run-arm")',
        'default="validate"',
        "--start-full-run",
        "--acknowledge-gate-sha256",
        "save_envelope_atomic(resume_path, initial_payload)",
        "step % core.RESUME_INTERVAL == 0",
        "core.save_adapter_snapshot",
        "enforce_arm_order",
        "core.require_free_space(run_root, 15.0)",
        "require_idle_compatible_gpu",
    ):
        require(token in cli_source, f"CLI static token: {token}")
    loop_index = mechanism_source.index("for index, vector in enumerate(unit_vectors)")
    noise_index = mechanism_source.index("noise = torch.randn", loop_index)
    update_index = mechanism_source.index("update = (clipped_sum + noise) / config.fixed_denominator", noise_index)
    require(loop_index < noise_index < update_index, "DP noise/fixed denominator structure")
    for token in (
        "CryptProtectData",
        "CryptUnprotectData",
        'temporary.open("xb")',
        "os.fsync",
        "os.replace(current, previous)",
        "os.replace(temporary, current)",
        "weights_only=True",
    ):
        require(token in secure_source, f"secure resume static token: {token}")
    return {
        "AST_parse": "PASS",
        "constants": {
            "full_steps": constants["FULL_STEPS"],
            "resume_interval": constants["RESUME_INTERVAL"],
            "adapter_steps": list(constants["ADAPTER_STEPS"]),
            "trainable_parameters": constants["TRAINABLE_PARAMETERS"],
            "arms": list(constants["ARMS"]),
        },
        "M1_shared_sampling_and_diffusion_labels": "PASS",
        "arm_independent_DP_Gaussian_labels": "PASS",
        "M2_mean_loss_before_outer_aggregation_clip": "PASS",
        "unconditional_Gaussian_and_fixed_denominator": "PASS",
        "DPAPI_atomic_current_previous": "PASS",
        "default_read_only_and_explicit_long_run_acknowledgement": "PASS",
    }


def replay_accounting(protocol: dict[str, Any]) -> dict[str, Any]:
    from dp_accounting import dp_event
    from dp_accounting.privacy_accountant import NeighboringRelation
    from dp_accounting.rdp import RdpAccountant
    from opacus.accountants.analysis import rdp

    orders = [round(1.0 + index / 10.0, 1) for index in range(1, 100)]
    orders += list(range(12, 64)) + [64, 128, 256, 512]
    results = {}
    for arm in ARMS[1:]:
        spec = protocol["DP_arms"][arm]
        target = spec["ideal_4000_step_bound"]
        delta_key = "delta_patient" if arm == "M2-P8" else "delta_image"
        epsilon_key = "epsilon_patient" if arm == "M2-P8" else "epsilon_image"
        delta = float(target[delta_key])
        epsilon_bound = float(target[epsilon_key])
        q = float(spec["q"])
        sigma = float(spec["sigma"])
        values = rdp.compute_rdp(q=q, noise_multiplier=sigma, steps=4_000, orders=orders)
        opacus_epsilon, opacus_order = rdp.get_privacy_spent(orders=orders, rdp=values, delta=delta)
        accountant = RdpAccountant(orders=orders, neighboring_relation=NeighboringRelation.ADD_OR_REMOVE_ONE)
        accountant.compose(dp_event.PoissonSampledDpEvent(q, dp_event.GaussianDpEvent(sigma)), count=4_000)
        google_epsilon, google_order = accountant.get_epsilon_and_optimal_order(delta)
        conservative = max(float(opacus_epsilon), float(google_epsilon))
        require(conservative <= epsilon_bound + 1e-10, f"independent accounting target: {arm}")
        results[arm] = {
            "opacus_epsilon": float(opacus_epsilon),
            "google_epsilon": float(google_epsilon),
            "conservative_epsilon": conservative,
            "epsilon_bound": epsilon_bound,
            "delta": delta,
            "opacus_order": float(opacus_order),
            "google_order": float(google_order),
            "status": "PASS",
        }
    return results


def nvidia_state() -> dict[str, Any]:
    line = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
    require(len(line) == 1, "independent one-GPU check")
    name, driver, total, free, used, temperature, utilization = [
        item.strip() for item in line[0].split(",")
    ]
    return {
        "name": name,
        "driver": driver,
        "memory_total_mib": int(total),
        "memory_free_mib": int(free),
        "memory_used_mib": int(used),
        "temperature_c": int(temperature),
        "utilization_percent": int(utilization),
        "idle_detection": "WDDM-safe aggregate utilization and free-memory thresholds",
    }


def scan_public(value: Any) -> None:
    forbidden = {
        "experiment_root",
        "generator_states",
        "global_cpu_rng",
        "global_cuda_rng_all",
        "noise_seed",
        "selected_ids",
        "patient_id",
        "image_id",
    }
    if isinstance(value, dict):
        require(not (set(value) & forbidden), "independent public artifact contains secret field")
        for child in value.values():
            scan_public(child)
    elif isinstance(value, list):
        for child in value:
            scan_public(child)
    elif isinstance(value, bytes):
        raise RuntimeError("independent public artifact contains bytes")


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    gate_root = root / "_reports" / GATE_NAME
    primary_path = gate_root / "primary_gate.json"
    environment_path = gate_root / "environment_manifest.json"
    authority_path = gate_root / "independent_launch_authority.json"
    independent_path = gate_root / "independent_verification.json"
    candidate_path = gate_root / "independent_launch_authority.candidate.json"
    require(gate_root.is_dir(), "primary full-runner gate is missing")
    require(not authority_path.exists() and not independent_path.exists() and not candidate_path.exists(), "refusing to overwrite independent full-runner gate")
    primary = load_json(primary_path)
    environment = load_json(environment_path)
    require(primary["schema"] == PRIMARY_SCHEMA, "primary schema")
    require(primary["status"] == "FROZEN_PENDING_INDEPENDENT_FULL_RUNNER_VERIFICATION", "primary status")
    require(primary["run_id"] == RUN_ID and primary["full_K5_optimizer_execution_started"] is False, "primary boundary")
    require(environment["schema"] == ENV_SCHEMA and environment["run_id"] == RUN_ID, "environment identity")
    require(primary["environment_manifest_sha256"] == sha256_file(environment_path), "environment hash")

    runner_hashes = {relative: sha256_file(root / Path(relative)) for relative in RUNNER_SOURCE_FILES}
    audit_hashes = {relative: sha256_file(root / Path(relative)) for relative in AUDIT_SOURCE_FILES}
    require(primary["source_hashes"] == runner_hashes == environment["runner_source_hashes"], "runner source freeze drift")
    require(primary["audit_source_hashes"] == audit_hashes == environment["audit_source_hashes"], "audit source freeze drift")
    static = static_source_audit(root)

    paths = {
        "protocol": root / "_reports" / "nih_cxr14_k5_feasibility_protocol_v1_001" / "protocol.json",
        "generation_tasks": root / "_reports" / "nih_cxr14_k5_feasibility_protocol_v1_001" / "generation_tasks.csv",
        "real_reference": root / "_reports" / "nih_cxr14_k5_feasibility_protocol_v1_001" / "real_reference_local.csv",
        "manifest": root / "_data" / "derived" / "nih_cxr14_pa_target_enriched_v1" / "k5_private.csv",
        "preprocessing": root / "_reports" / "nih_cxr14_sd21_ppmark_interface_v1_001" / "preprocessing_contract.json",
        "actual_protocol": root / "_reports" / "nih_cxr14_k5_actual_sd21_resume_preflight_protocol_v1_002" / "protocol.json",
        "actual_result": root / "_reports" / "nih_cxr14_k5_actual_sd21_resume_preflight_v1_002" / "public_report.json",
        "evaluator_primary": root / "_reports" / "nih_cxr14_k5_evaluator_preflight_v1_001" / "public_report.json",
        "evaluator_independent": root / "_reports" / "nih_cxr14_k5_evaluator_preflight_gate_v1_001" / "independent_verification.json",
        "synthetic_resume_primary": root / "_reports" / "nih_cxr14_k5_encrypted_resume_preflight_v1_001" / "public_report.json",
        "synthetic_resume_independent": root / "_reports" / "nih_cxr14_k5_encrypted_resume_preflight_gate_v1_001" / "independent_verification.json",
    }
    expected_hashes = {
        "protocol": PROTOCOL_SHA256,
        "generation_tasks": GENERATION_TASKS_SHA256,
        "real_reference": REAL_REFERENCE_SHA256,
        "manifest": K5_SHA256,
        "preprocessing": PREPROCESSING_SHA256,
        "actual_protocol": ACTUAL_PROTOCOL_SHA256,
        "actual_result": ACTUAL_RESULT_SHA256,
        "evaluator_primary": EVALUATOR_PRIMARY_SHA256,
        "evaluator_independent": EVALUATOR_INDEPENDENT_SHA256,
        "synthetic_resume_primary": SYNTHETIC_RESUME_PRIMARY_SHA256,
        "synthetic_resume_independent": SYNTHETIC_RESUME_INDEPENDENT_SHA256,
    }
    actual_hashes = {name: sha256_file(path) for name, path in paths.items()}
    require(actual_hashes == expected_hashes, "independent frozen evidence hash mismatch")
    protocol = load_json(paths["protocol"])
    actual_protocol = load_json(paths["actual_protocol"])
    actual_result = load_json(paths["actual_result"])
    require(actual_protocol["source_hashes"] == actual_result["source_hashes"], "actual preflight source mismatch")
    require(
        all(actual_result["source_hashes"].get(name) == digest for name, digest in runner_hashes.items()),
        "runner changed after actual SD2.1 resume proof",
    )
    require(actual_result["status"] == "PASS_ACTUAL_SD21_EXACT_ENCRYPTED_RESUME_ALL_FOUR_ARMS", "actual resume status")
    require(actual_result["all_exact_checks_pass"] is True, "actual resume aggregate")
    require(set(actual_result["arms"]) == set(ARMS), "actual resume arm set")
    initial_digests = set()
    for arm in ARMS:
        arm_result = actual_result["arms"][arm]
        require(arm_result["status"] == "PASS", f"actual resume arm: {arm}")
        require(len(arm_result["exact_checks"]) == 13 and all(arm_result["exact_checks"].values()), f"13 exact checks: {arm}")
        require(arm_result["step0_ciphertext_bytes"] > 6_000_000, f"real step0 envelope: {arm}")
        require(arm_result["step2_ciphertext_bytes"] > 20_000_000, f"real AdamW envelope: {arm}")
        require(arm_result["peak_cuda_memory_bytes"] < int(7.5 * 1024**3), f"preflight CUDA bound: {arm}")
        initial_digests.add(arm_result["initial_adapter_sha256"])
    require(len(initial_digests) == 1, "actual initial LoRA differs")
    require(
        actual_result["arms"]["M1-I8"]["schedule_commitment"]
        == actual_result["arms"]["M1-G8"]["schedule_commitment"],
        "actual M1 schedule commitment",
    )
    require(not any(actual_result["artifact_boundary"].values()), "actual preflight retained/started artifact")

    require(protocol["matrix"]["steps_per_arm"] == 4_000 and tuple(protocol["matrix"]["arm_order"]) == ARMS, "protocol matrix")
    image_exposures = int(protocol["M0"]["images_per_step"]) * 4_000
    patient_exposures = int(protocol["DP_arms"]["M2-P8"]["expected_batch"]) * 4_000
    require(image_exposures == 32_000 and patient_exposures == 16_000, "exposure arithmetic")
    require(math.isclose(primary["scale_rationale"]["M0_and_M1_expected_exposures_per_image"], 32_000 / 18_393, rel_tol=0, abs_tol=1e-15), "image exposure ratio")
    require(math.isclose(primary["scale_rationale"]["M2_expected_exposures_per_patient"], 16_000 / 8_476, rel_tol=0, abs_tol=1e-15), "patient exposure ratio")
    accounting = replay_accounting(protocol)

    model_snapshot = Path(environment["frozen_inputs"]["model_snapshot"])
    require(model_snapshot.is_dir() and model_snapshot.name == environment["frozen_inputs"]["model_revision"], "model snapshot path")
    for relative, result in environment["frozen_inputs"]["critical_model_hashes"].items():
        require(sha256_file(model_snapshot / relative) == result["actual"] == result["expected"], f"model hash: {relative}")
    package_versions = {name: importlib.metadata.version(name) for name in environment["packages"]}
    require(package_versions == environment["packages"], "independent package version drift")
    python_path = Path(environment["python"]["executable"])
    require(Path(sys.executable).resolve() == python_path and sha256_file(python_path) == environment["python"]["executable_sha256"], "Python executable drift")
    require(platform.python_version() == environment["python"]["version"], "Python version drift")

    regression = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "dp_training", "-p", "test_*.py", "-v"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    regression_output = regression.stdout + regression.stderr
    require(regression.returncode == 0 and "Ran 32 tests" in regression_output and "OK" in regression_output, "independent 32-test suite")
    gpu = nvidia_state()
    frozen_gpu = environment["cuda_device"]
    require(gpu["name"] == frozen_gpu["name"] and gpu["driver"] == frozen_gpu["driver"], "independent GPU/driver drift")
    require(gpu["memory_total_mib"] == frozen_gpu["memory_total_mib"], "independent GPU memory drift")
    require(
        gpu["memory_free_mib"] >= 6_500
        and gpu["temperature_c"] < 80
        and gpu["utilization_percent"] <= 10,
        "independent GPU readiness",
    )
    free_gib = shutil.disk_usage(root).free / 1024**3
    require(free_gib >= 15.0, "independent free-space readiness")
    restricted_root = root / "_restricted_runs" / RUN_ID
    public_root = root / "_reports" / RUN_ID
    require(not restricted_root.exists() and not public_root.exists(), "full K5 run root exists")

    primary_sha = sha256_file(primary_path)
    authority = {
        "schema": AUTHORITY_SCHEMA,
        "status": "PASS_K5_FULL_RUNNER_GATE_NO_FULL_TRAINING_STARTED",
        "run_id": RUN_ID,
        "scope": "AUTHORIZE_ONLY_EXPLICIT_POST_REPORT_B0_AND_SEQUENTIAL_ARM_COMMANDS",
        "primary_gate_sha256": primary_sha,
        "environment_manifest_file": environment_path.name,
        "environment_manifest_sha256": sha256_file(environment_path),
        "source_hashes": runner_hashes,
        "generation_tasks_sha256": GENERATION_TASKS_SHA256,
        "preflight_evidence_sha256": expected_hashes,
        "all_required_prelaunch_gates_pass": True,
        "default_runner_command_read_only": True,
        "B0_required_before_matrix_initialization": True,
        "explicit_gate_hash_and_start_flag_required": True,
        "arm_order": list(ARMS),
        "full_K5_optimizer_execution_started": False,
        "full_run_roots_created": False,
        "claim_boundary": "one-seed K5 research feasibility only; research PRNG is not release authority",
        "next": "report this gate; then generate and hash B0 before deciding whether to initialize or run M0",
    }
    scan_public(authority)
    write_json(candidate_path, authority)
    validation = subprocess.run(
        [
            sys.executable,
            "-m",
            "dp_training.run_k5_full_training",
            "validate",
            "--launch-authority",
            str(candidate_path),
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    validation_output = validation.stdout + validation.stderr
    require(validation.returncode == 0 and "PASS_READ_ONLY_FULL_RUNNER_VALIDATION" in validation_output, "frozen CLI read-only validation")
    require(not restricted_root.exists() and not public_root.exists(), "read-only validation created full-run root")
    os.replace(candidate_path, authority_path)
    authority_sha = sha256_file(authority_path)
    independent = {
        "schema": INDEPENDENT_SCHEMA,
        "status": "PASS_K5_FULL_RUNNER_GATE_INDEPENDENT_NO_FULL_TRAINING_STARTED",
        "run_id": RUN_ID,
        "primary_gate_sha256": primary_sha,
        "environment_manifest_sha256": sha256_file(environment_path),
        "launch_authority_file": authority_path.name,
        "launch_authority_sha256": authority_sha,
        "frozen_evidence_hashes": actual_hashes,
        "source_hashes": runner_hashes,
        "audit_source_hashes": audit_hashes,
        "static_source_audit": static,
        "independent_accounting": accounting,
        "actual_SD21_resume": {
            "all_four_arms": "PASS",
            "exact_checks_per_arm": 13,
            "real_LoRA_and_AdamW_envelope": True,
            "M1_schedule_commitment_shared": True,
            "temporary_envelopes_retained": False,
        },
        "regression": {
            "tests": 32,
            "status": "PASS",
            "combined_output_sha256": sha256_bytes(regression_output.encode("utf-8")),
        },
        "read_only_CLI_validation": {
            "status": "PASS",
            "combined_output_sha256": sha256_bytes(validation_output.encode("utf-8")),
            "run_roots_after_validation": "ABSENT",
        },
        "resource_readiness": {
            "free_space_gib": free_gib,
            "GPU": gpu,
        },
        "scale_recomputation": {
            "image_exposures": image_exposures,
            "expected_exposures_per_image": image_exposures / 18_393,
            "patient_unit_exposures": patient_exposures,
            "expected_exposures_per_patient": patient_exposures / 8_476,
            "interpretation": "valid feasibility budget, not a confirmatory sample-size claim",
        },
        "full_K5_optimizer_execution_started": False,
        "full_run_roots_created": False,
        "next": authority["next"],
    }
    scan_public(independent)
    write_json(independent_path, independent)
    print(
        json.dumps(
            {
                "status": independent["status"],
                "independent_report": str(independent_path),
                "independent_report_sha256": sha256_file(independent_path),
                "launch_authority": str(authority_path),
                "launch_authority_sha256": authority_sha,
                "full_K5_optimizer_execution_started": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
