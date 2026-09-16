#!/usr/bin/env python3
"""Prove exact four-step versus encrypted two-plus-resume-plus-two state equality."""

from __future__ import annotations

import gc
import hashlib
import hmac
import json
import math
import secrets
import tempfile
from pathlib import Path
from typing import Any

import torch

from dp_training.mechanism import (
    MechanismConfig,
    aggregate_noised_update,
    append_public_trace,
    canonical_json,
    make_public_scheduled_event,
    mean_unit_vectors,
    poisson_select,
    sha256_bytes,
    tensor_sha256,
    verify_public_trace,
)
from dp_training.secure_resume import load_envelope, save_envelope_atomic, sha256_file


SCHEMA = "nih-cxr14-k5-encrypted-resume-equivalence-result/v1"
PAYLOAD_SCHEMA = "nih-cxr14-k5-research-resume-payload/v1"
PROTOCOL_SHA256 = "4B4B9F20C4830C541ED857EDB751FD00D4032E0ACDB4FCE0C84DC18436808F75"
MECHANISM_SHA256 = "B4B1E037A3704238B06B442970EAB082433835201B750546C8281BBFFE2B969E"
SECURE_RESUME_SHA256 = "360D66DA4A9FAE4060C806A71FB02555E58C1E0E71188CFDDC261EDC95895E47"
DIMENSION = 257
ARMS = ("M0", "M1-I8", "M1-G8", "M2-P8")
OPTIMIZER = {
    "lr": 1e-4,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.01,
}
M0_TRACE_GENESIS = "0" * 64


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def derive_seed(root: bytes, label: str) -> int:
    require(len(root) == 32, "experiment root must contain 256 bits")
    return int.from_bytes(hmac.new(root, label.encode("utf-8"), hashlib.sha256).digest()[:8], "big") % (2**63)


def record_code(record_id: str) -> int:
    return int.from_bytes(hashlib.sha256(record_id.encode("utf-8")).digest()[:4], "big")


def exact_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape) and torch.equal(left, right)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(exact_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(exact_equal(a, b) for a, b in zip(left, right))
    return bool(left == right)


def optimizer(parameter: torch.nn.Parameter) -> torch.optim.AdamW:
    return torch.optim.AdamW([parameter], **OPTIMIZER)


def generator_names(arm: str) -> tuple[str, ...]:
    if arm == "M0":
        return ("sampling", "diffusion")
    if arm.startswith("M1"):
        return ("sampling", "diffusion", "gaussian")
    require(arm == "M2-P8", "unknown arm")
    return ("sampling", "inner", "diffusion", "gaussian")


def initialize_state(arm: str, experiment_root: bytes) -> dict[str, Any]:
    require(arm in ARMS, "unknown arm")
    parameter = torch.nn.Parameter(torch.linspace(-0.03, 0.03, DIMENSION, dtype=torch.float32))
    generators: dict[str, torch.Generator] = {}
    for name in generator_names(arm):
        generators[name] = torch.Generator(device="cpu").manual_seed(
            derive_seed(experiment_root, f"{arm}|{name}")
        )
    torch.manual_seed(derive_seed(experiment_root, f"{arm}|global-cpu"))
    torch.cuda.manual_seed_all(derive_seed(experiment_root, f"{arm}|global-cuda"))
    return {
        "arm": arm,
        "step": 0,
        "parameter": parameter,
        "optimizer": optimizer(parameter),
        "generators": generators,
        "trace": [],
        "m0_sampler": {
            "permutation": torch.empty(0, dtype=torch.int64),
            "cursor": 18,
            "epoch": -1,
        },
        "experiment_root": experiment_root,
    }


def mechanism_config(arm: str) -> MechanismConfig:
    if arm == "M1-I8":
        return MechanismConfig(
            arm=arm,
            privacy_unit="image",
            population=64,
            expected_batch=8,
            poisson_sample_rate=8 / 64,
            clip_norm=0.28448700606156724,
            noise_multiplier=0.4007042918650512,
            fixed_denominator=8,
            max_steps=4,
            rng_security_mode="RESEARCH_ONLY_NONCRYPTOGRAPHIC",
            rng_backend="torch_cpu_encrypted_resume_preflight",
        )
    if arm == "M1-G8":
        return MechanismConfig(
            arm=arm,
            privacy_unit="image",
            population=64,
            expected_batch=8,
            poisson_sample_rate=8 / 64,
            clip_norm=0.28448700606156724,
            noise_multiplier=0.8350657829633824,
            fixed_denominator=8,
            max_steps=4,
            rng_security_mode="RESEARCH_ONLY_NONCRYPTOGRAPHIC",
            rng_backend="torch_cpu_encrypted_resume_preflight",
        )
    require(arm == "M2-P8", "M0 has no DP mechanism config")
    return MechanismConfig(
        arm=arm,
        privacy_unit="patient",
        population=16,
        expected_batch=4,
        poisson_sample_rate=4 / 16,
        clip_norm=0.1997973088974048,
        noise_multiplier=0.4044571458362774,
        fixed_denominator=4,
        max_steps=4,
        rng_security_mode="RESEARCH_ONLY_NONCRYPTOGRAPHIC",
        rng_backend="torch_cpu_encrypted_resume_preflight",
    )


def synthetic_record_gradient(record_id: str, diffusion: torch.Generator) -> torch.Tensor:
    code = record_code(record_id)
    axis = torch.arange(1, DIMENSION + 1, dtype=torch.float32)
    base = torch.sin(axis * ((code % 997) + 1) * 0.00037) * 0.08
    timestep = int(torch.randint(0, 1_000, (1,), generator=diffusion, device="cpu").item())
    corruption = torch.randn(DIMENSION, generator=diffusion, dtype=torch.float32) * 0.015
    global_cpu = torch.rand((), dtype=torch.float32)
    global_cuda = torch.rand((), dtype=torch.float32, device="cuda").cpu()
    gradient = base + corruption + (timestep / 1_000_000.0) + global_cpu * 1e-5 + global_cuda * 1e-5
    require(gradient.dtype == torch.float32 and bool(torch.isfinite(gradient).all()), "synthetic gradient")
    return gradient


def append_m0_trace(trace: list[dict[str, Any]], step: int) -> None:
    event = {
        "schema": "nih-cxr14-k5-m0-public-scheduled-event/v1",
        "arm": "M0",
        "first_step": step,
        "event_count": 1,
        "batch_size": 8,
        "rng_security_mode": "RESEARCH_ONLY_NONCRYPTOGRAPHIC",
    }
    previous = M0_TRACE_GENESIS if not trace else trace[-1]["event_hash"]
    event_hash = sha256_bytes(canonical_json({"previous_hash": previous, "event": event}))
    trace.append({"previous_hash": previous, "event": event, "event_hash": event_hash})


def m0_step(state: dict[str, Any]) -> None:
    sampler = state["m0_sampler"]
    if int(sampler["cursor"]) + 8 > 18:
        sampler["permutation"] = torch.randperm(18, generator=state["generators"]["sampling"])
        sampler["cursor"] = 0
        sampler["epoch"] = int(sampler["epoch"]) + 1
    start = int(sampler["cursor"])
    selected = sampler["permutation"][start : start + 8].tolist()
    require(len(selected) == 8, "M0 fixed batch")
    sampler["cursor"] = start + 8
    gradients = [
        synthetic_record_gradient(f"m0-image-{int(index):03d}", state["generators"]["diffusion"])
        for index in selected
    ]
    update = torch.stack(gradients, dim=0).mean(dim=0)
    state["parameter"].grad = update.clone()
    state["optimizer"].step()
    state["optimizer"].zero_grad(set_to_none=True)
    state["step"] = int(state["step"]) + 1
    append_m0_trace(state["trace"], int(state["step"]))


def m1_step(state: dict[str, Any], config: MechanismConfig) -> None:
    image_ids = [f"m1-image-{index:03d}" for index in range(64)]
    selected = poisson_select(
        image_ids,
        config.poisson_sample_rate,
        generator=state["generators"]["sampling"],
    )
    vectors = [
        synthetic_record_gradient(image_id, state["generators"]["diffusion"])
        for image_id in selected
    ]
    template = torch.zeros(DIMENSION, dtype=torch.float32)
    update, _ = aggregate_noised_update(
        vectors,
        template=template,
        config=config,
        generator=state["generators"]["gaussian"],
    )
    state["parameter"].grad = update.clone()
    state["optimizer"].step()
    state["optimizer"].zero_grad(set_to_none=True)
    state["step"] = int(state["step"]) + 1
    append_public_trace(
        state["trace"],
        make_public_scheduled_event(config, first_step=int(state["step"]), event_count=1),
    )


def m2_step(state: dict[str, Any], config: MechanismConfig) -> None:
    patients = [f"m2-patient-{index:02d}" for index in range(16)]
    records = {
        patient: [f"{patient}-image-{item}" for item in range(1 + index % 5)]
        for index, patient in enumerate(patients)
    }
    selected_patients = poisson_select(
        patients,
        config.poisson_sample_rate,
        generator=state["generators"]["sampling"],
    )
    patient_vectors: list[torch.Tensor] = []
    for patient in selected_patients:
        candidates = records[patient]
        order = torch.randperm(len(candidates), generator=state["generators"]["inner"])
        chosen = [candidates[int(index)] for index in order[: min(4, len(candidates))].tolist()]
        image_vectors = [
            synthetic_record_gradient(image_id, state["generators"]["diffusion"])
            for image_id in chosen
        ]
        patient_vectors.append(mean_unit_vectors(image_vectors))
    template = torch.zeros(DIMENSION, dtype=torch.float32)
    update, _ = aggregate_noised_update(
        patient_vectors,
        template=template,
        config=config,
        generator=state["generators"]["gaussian"],
    )
    state["parameter"].grad = update.clone()
    state["optimizer"].step()
    state["optimizer"].zero_grad(set_to_none=True)
    state["step"] = int(state["step"]) + 1
    append_public_trace(
        state["trace"],
        make_public_scheduled_event(config, first_step=int(state["step"]), event_count=1),
    )


def run_steps(state: dict[str, Any], count: int) -> None:
    config = None if state["arm"] == "M0" else mechanism_config(state["arm"])
    for _ in range(count):
        require(int(state["step"]) < 4, "step overflow")
        if state["arm"] == "M0":
            m0_step(state)
        elif state["arm"].startswith("M1"):
            m1_step(state, config)
        else:
            m2_step(state, config)
        require(bool(torch.isfinite(state["parameter"]).all()), "non-finite adapter")


def capture_payload(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": PAYLOAD_SCHEMA,
        "run_id": "nih_cxr14_k5_encrypted_resume_preflight_v1_001",
        "arm": state["arm"],
        "committed_step": int(state["step"]),
        "adapter": state["parameter"].detach().cpu().clone(),
        "optimizer": state["optimizer"].state_dict(),
        "experiment_root": bytes(state["experiment_root"]),
        "generator_states": {
            name: generator.get_state().cpu().clone() for name, generator in state["generators"].items()
        },
        "global_cpu_rng": torch.get_rng_state().cpu().clone(),
        "global_cuda_rng_all": [value.cpu().clone() for value in torch.cuda.get_rng_state_all()],
        "public_trace": json.loads(json.dumps(state["trace"])),
        "m0_sampler": {
            "permutation": state["m0_sampler"]["permutation"].cpu().clone(),
            "cursor": int(state["m0_sampler"]["cursor"]),
            "epoch": int(state["m0_sampler"]["epoch"]),
        },
        "frozen_input_digests": {
            "protocol": PROTOCOL_SHA256,
            "mechanism": MECHANISM_SHA256,
            "secure_resume": SECURE_RESUME_SHA256,
        },
    }


def restore_payload(payload: dict[str, Any]) -> dict[str, Any]:
    require(payload["schema"] == PAYLOAD_SCHEMA, "resume payload schema")
    require(payload["arm"] in ARMS, "resume arm")
    require(0 <= int(payload["committed_step"]) <= 4, "resume committed step")
    require(
        payload["frozen_input_digests"]
        == {
            "protocol": PROTOCOL_SHA256,
            "mechanism": MECHANISM_SHA256,
            "secure_resume": SECURE_RESUME_SHA256,
        },
        "resume frozen input drift",
    )
    parameter = torch.nn.Parameter(payload["adapter"].detach().cpu().float().clone())
    restored_optimizer = optimizer(parameter)
    restored_optimizer.load_state_dict(payload["optimizer"])
    generators: dict[str, torch.Generator] = {}
    require(set(payload["generator_states"]) == set(generator_names(payload["arm"])), "generator set")
    for name, generator_state in payload["generator_states"].items():
        generator = torch.Generator(device="cpu")
        generator.set_state(generator_state.cpu())
        generators[name] = generator
    torch.set_rng_state(payload["global_cpu_rng"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in payload["global_cuda_rng_all"]])
    return {
        "arm": payload["arm"],
        "step": int(payload["committed_step"]),
        "parameter": parameter,
        "optimizer": restored_optimizer,
        "generators": generators,
        "trace": json.loads(json.dumps(payload["public_trace"])),
        "m0_sampler": {
            "permutation": payload["m0_sampler"]["permutation"].cpu().clone(),
            "cursor": int(payload["m0_sampler"]["cursor"]),
            "epoch": int(payload["m0_sampler"]["epoch"]),
        },
        "experiment_root": bytes(payload["experiment_root"]),
    }


def compare_final(baseline: dict[str, Any], resumed: dict[str, Any]) -> dict[str, bool]:
    checks = {
        "complete_payload_exact": exact_equal(baseline, resumed),
        "adapter_exact": exact_equal(baseline["adapter"], resumed["adapter"]),
        "optimizer_exact": exact_equal(baseline["optimizer"], resumed["optimizer"]),
        "generator_states_exact": exact_equal(
            baseline["generator_states"], resumed["generator_states"]
        ),
        "global_cpu_rng_exact": exact_equal(baseline["global_cpu_rng"], resumed["global_cpu_rng"]),
        "global_cuda_rng_exact": exact_equal(
            baseline["global_cuda_rng_all"], resumed["global_cuda_rng_all"]
        ),
        "sampler_state_exact": exact_equal(baseline["m0_sampler"], resumed["m0_sampler"]),
        "public_trace_exact": exact_equal(baseline["public_trace"], resumed["public_trace"]),
        "experiment_root_exact_internal_only": exact_equal(
            baseline["experiment_root"], resumed["experiment_root"]
        ),
        "frozen_inputs_exact": exact_equal(
            baseline["frozen_input_digests"], resumed["frozen_input_digests"]
        ),
        "committed_step_exact": baseline["committed_step"] == resumed["committed_step"] == 4,
    }
    require(all(checks.values()), f"resume mismatch: {checks}")
    return checks


def run_arm(arm: str) -> dict[str, Any]:
    experiment_root = secrets.token_bytes(32)
    baseline_state = initialize_state(arm, experiment_root)
    initial_adapter_digest = tensor_sha256(baseline_state["parameter"].detach())
    run_steps(baseline_state, 4)
    baseline = capture_payload(baseline_state)
    if arm != "M0":
        require(verify_public_trace(baseline["public_trace"])["total_events"] == 4, "baseline trace")

    interrupted_state = initialize_state(arm, experiment_root)
    require(tensor_sha256(interrupted_state["parameter"].detach()) == initial_adapter_digest, "initial adapter")
    temp_path: Path | None = None
    with tempfile.TemporaryDirectory(prefix="k5-resume-preflight-") as directory:
        temp_path = Path(directory)
        current = temp_path / f"{arm}.dpapi"
        step0_payload = capture_payload(interrupted_state)
        global_before_save = (
            torch.get_rng_state().clone(),
            [value.clone() for value in torch.cuda.get_rng_state_all()],
            {name: generator.get_state().clone() for name, generator in interrupted_state["generators"].items()},
        )
        step0_metadata = save_envelope_atomic(current, step0_payload)
        global_after_save = (
            torch.get_rng_state().clone(),
            [value.clone() for value in torch.cuda.get_rng_state_all()],
            {name: generator.get_state().clone() for name, generator in interrupted_state["generators"].items()},
        )
        require(exact_equal(global_before_save, global_after_save), "step-0 save consumed torch RNG")
        require(experiment_root not in current.read_bytes(), "experiment root visible in ciphertext")
        run_steps(interrupted_state, 2)
        step2_payload = capture_payload(interrupted_state)
        step2_metadata = save_envelope_atomic(current, step2_payload)
        previous = temp_path / f"{arm}.previous.dpapi"
        require(step2_metadata["previous_retained"] and previous.is_file(), "previous envelope retention")
        require(load_envelope(previous)["committed_step"] == 0, "previous envelope is not step 0")
        require(load_envelope(current)["committed_step"] == 2, "current envelope is not step 2")
        require(experiment_root not in current.read_bytes(), "step-2 experiment root visible")
        envelope_report = {
            "step0_ciphertext_bytes": int(step0_metadata["ciphertext_bytes"]),
            "step2_ciphertext_bytes": int(step2_metadata["ciphertext_bytes"]),
            "step0_became_previous": True,
            "plaintext_file_created": False,
            "save_consumed_torch_rng": False,
        }
        del interrupted_state, step0_payload, step2_payload
        gc.collect()
        torch.rand(17)
        torch.rand(17, device="cuda")
        restored_payload = load_envelope(current)
        resumed_state = restore_payload(restored_payload)
        run_steps(resumed_state, 2)
        resumed = capture_payload(resumed_state)
        if arm != "M0":
            require(verify_public_trace(resumed["public_trace"])["total_events"] == 4, "resumed trace")
        checks = compare_final(baseline, resumed)
        trace_head = baseline["public_trace"][-1]["event_hash"]
        final_adapter_finite = bool(torch.isfinite(resumed["adapter"]).all())
        require(final_adapter_finite, "final adapter non-finite")
    require(temp_path is not None and not temp_path.exists(), "temporary envelope directory retained")
    del experiment_root, baseline_state, baseline, resumed_state, resumed, restored_payload
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "state_checks": checks,
        "envelope": envelope_report,
        "trace_records": 4,
        "trace_head_sha256": trace_head,
        "final_adapter_finite": final_adapter_finite,
        "temporary_envelopes_retained": False,
        "status": "PASS",
    }


def scan_public(value: Any) -> None:
    forbidden = {
        "experiment_root",
        "generator_states",
        "global_cpu_rng",
        "global_cuda_rng_all",
        "sampling_seed",
        "noise_seed",
        "rng_digest",
    }
    if isinstance(value, dict):
        require(not (set(value) & forbidden), "public secret field")
        for child in value.values():
            scan_public(child)
    elif isinstance(value, list):
        for child in value:
            scan_public(child)
    elif isinstance(value, bytes):
        raise RuntimeError("bytes are prohibited in public result")


def main() -> int:
    require(torch.cuda.is_available(), "CUDA required for global CUDA RNG-state preflight")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(__file__).resolve().parent.parent
    protocol_path = (
        root
        / "_reports"
        / "nih_cxr14_k5_encrypted_resume_preflight_protocol_v1_001"
        / "protocol.json"
    )
    output = root / "_reports" / "nih_cxr14_k5_encrypted_resume_preflight_v1_001"
    require(not output.exists(), "refusing to overwrite encrypted-resume preflight result")
    require(sha256_file(protocol_path) == PROTOCOL_SHA256, "resume protocol hash drift")
    require(sha256_file(root / "dp_training" / "mechanism.py") == MECHANISM_SHA256, "mechanism drift")
    require(
        sha256_file(root / "dp_training" / "secure_resume.py") == SECURE_RESUME_SHA256,
        "secure-resume source drift",
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    require(protocol["status"] == "FROZEN_BEFORE_EXACT_RESUME_OUTPUT", "resume protocol status")
    arm_reports = {arm: run_arm(arm) for arm in ARMS}
    report = {
        "schema": SCHEMA,
        "status": "PASS_EXACT_ENCRYPTED_RESUME_EQUIVALENCE",
        "scope": protocol["scope"],
        "source_hashes": {
            "protocol": PROTOCOL_SHA256,
            "mechanism": MECHANISM_SHA256,
            "secure_resume": SECURE_RESUME_SHA256,
            "runner": sha256_file(Path(__file__).resolve()),
        },
        "environment": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "comparison": "four uninterrupted versus encrypted two-plus-resume-plus-two",
        "arms": arm_reports,
        "all_four_arms_exact": all(
            all(report["state_checks"].values()) for report in arm_reports.values()
        ),
        "DPAPI": {
            "scope": "CurrentUser",
            "optional_entropy_applied": True,
            "atomic_new_fsync_verify_replace": True,
            "current_plus_previous_rotation": True,
            "tamper_and_stale_new_fail_closed_unit_tests_required": True,
        },
        "artifact_policy": {
            "plaintext_resume_file_created": False,
            "encrypted_preflight_envelopes_retained": False,
            "experiment_root_or_rng_state_reported": False,
            "model_or_real_image_used": False,
            "full_K5_optimizer_execution_started": False,
        },
        "next": "independently verify this result, then freeze the full runner/environment and report before full K5",
    }
    scan_public(report)
    output.mkdir(parents=True)
    report_path = output / "public_report.json"
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"public_report_sha256={sha256_file(report_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
