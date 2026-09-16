#!/usr/bin/env python3
"""Independent static and dynamic audit of the encrypted-resume preflight."""

from __future__ import annotations

import ast
import ctypes
import hashlib
import io
import json
import os
import tempfile
from ctypes import wintypes
from pathlib import Path
from typing import Any

import torch


SCHEMA = "nih-cxr14-k5-encrypted-resume-independent-verification/v1"
PROTOCOL_SHA256 = "4B4B9F20C4830C541ED857EDB751FD00D4032E0ACDB4FCE0C84DC18436808F75"
PUBLIC_REPORT_SHA256 = "89A08E9B1AAC248A3DF4F4935C82354D49D28AFC0DD8BE632DE6A80C394FCBFF"
RUNNER_SHA256 = "7AB604373BF06F52EBC191846419E18E98C4D1685E9FDA0891AB51B2613B574F"
SECURE_RESUME_SHA256 = "360D66DA4A9FAE4060C806A71FB02555E58C1E0E71188CFDDC261EDC95895E47"
MECHANISM_SHA256 = "B4B1E037A3704238B06B442970EAB082433835201B750546C8281BBFFE2B969E"
MAGIC = b"NIH-CXR14-K5-RESUME-V1\0"
ENTROPY = b"nih-cxr14-k5-feasibility-resume-v1"


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


def exact(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape) and torch.equal(left, right)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(exact(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
    return bool(left == right)


class DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def blob(value: bytes) -> tuple[DataBlob, Any]:
    buffer = (ctypes.c_ubyte * len(value)).from_buffer_copy(value)
    return DataBlob(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))), buffer


def dpapi(value: bytes, *, protect: bool) -> bytes:
    require(os.name == "nt", "Windows required")
    crypt32 = ctypes.WinDLL("crypt32.dll", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
    input_blob, input_buffer = blob(value)
    entropy_blob, entropy_buffer = blob(ENTROPY)
    output = DataBlob()
    if protect:
        function = crypt32.CryptProtectData
        function.argtypes = [
            ctypes.POINTER(DataBlob),
            wintypes.LPCWSTR,
            ctypes.POINTER(DataBlob),
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(DataBlob),
        ]
        args = (
            ctypes.byref(input_blob),
            "independent K5 resume audit",
            ctypes.byref(entropy_blob),
            None,
            None,
            0x1,
            ctypes.byref(output),
        )
    else:
        function = crypt32.CryptUnprotectData
        function.argtypes = [
            ctypes.POINTER(DataBlob),
            ctypes.POINTER(wintypes.LPWSTR),
            ctypes.POINTER(DataBlob),
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(DataBlob),
        ]
        description = wintypes.LPWSTR()
        args = (
            ctypes.byref(input_blob),
            ctypes.byref(description),
            ctypes.byref(entropy_blob),
            None,
            None,
            0x1,
            ctypes.byref(output),
        )
    function.restype = wintypes.BOOL
    _ = (input_buffer, entropy_buffer)
    if not function(*args):
        raise ctypes.WinError(ctypes.get_last_error())
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    try:
        result = ctypes.string_at(output.pbData, output.cbData)
    finally:
        kernel32.LocalFree(output.pbData)
        if not protect and description:
            kernel32.LocalFree(description)
    return result


def encode(payload: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(payload, buffer, pickle_protocol=2)
    body = buffer.getvalue()
    return MAGIC + hashlib.sha256(body).digest() + body


def decode(value: bytes) -> dict[str, Any]:
    require(value.startswith(MAGIC), "independent magic")
    offset = len(MAGIC)
    expected = value[offset : offset + 32]
    body = value[offset + 32 :]
    require(hashlib.sha256(body).digest() == expected, "independent inner hash")
    payload = torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)
    require(isinstance(payload, dict), "independent payload type")
    return payload


def dynamic_equivalence() -> dict[str, Any]:
    require(torch.cuda.is_available(), "CUDA RNG audit requires CUDA")

    def initialize() -> tuple[torch.nn.Parameter, torch.optim.AdamW, torch.Generator]:
        parameter = torch.nn.Parameter(torch.linspace(-0.2, 0.2, 73, dtype=torch.float32))
        opt = torch.optim.AdamW(
            [parameter], lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
        )
        generator = torch.Generator(device="cpu").manual_seed(911)
        torch.manual_seed(177)
        torch.cuda.manual_seed_all(313)
        return parameter, opt, generator

    def steps(
        parameter: torch.nn.Parameter,
        opt: torch.optim.AdamW,
        generator: torch.Generator,
        count: int,
    ) -> None:
        for _ in range(count):
            gradient = torch.randn(73, generator=generator, dtype=torch.float32)
            gradient += torch.rand((), dtype=torch.float32) * 1e-4
            gradient += torch.rand((), dtype=torch.float32, device="cuda").cpu() * 1e-4
            parameter.grad = gradient
            opt.step()
            opt.zero_grad(set_to_none=True)

    def capture(
        parameter: torch.nn.Parameter,
        opt: torch.optim.AdamW,
        generator: torch.Generator,
        step: int,
    ) -> dict[str, Any]:
        return {
            "schema": "independent-resume-audit/v1",
            "arm": "independent",
            "committed_step": step,
            "adapter": parameter.detach().clone(),
            "optimizer": opt.state_dict(),
            "generator": generator.get_state().clone(),
            "global_cpu": torch.get_rng_state().clone(),
            "global_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
        }

    baseline_parameter, baseline_optimizer, baseline_generator = initialize()
    steps(baseline_parameter, baseline_optimizer, baseline_generator, 4)
    baseline = capture(baseline_parameter, baseline_optimizer, baseline_generator, 4)

    interrupted_parameter, interrupted_optimizer, interrupted_generator = initialize()
    steps(interrupted_parameter, interrupted_optimizer, interrupted_generator, 2)
    interrupted = capture(interrupted_parameter, interrupted_optimizer, interrupted_generator, 2)
    plaintext = encode(interrupted)
    ciphertext = dpapi(plaintext, protect=True)
    require(ciphertext != plaintext and b"independent-resume-audit" not in ciphertext, "ciphertext boundary")
    with tempfile.TemporaryDirectory(prefix="independent-resume-audit-") as directory:
        current = Path(directory) / "current.dpapi"
        temporary = Path(directory) / "current.dpapi.new"
        with temporary.open("xb") as handle:
            handle.write(ciphertext)
            handle.flush()
            os.fsync(handle.fileno())
        verified = decode(dpapi(temporary.read_bytes(), protect=False))
        require(verified["committed_step"] == 2, "independent post-write verify")
        os.replace(temporary, current)
        loaded = decode(dpapi(current.read_bytes(), protect=False))
        resumed_parameter = torch.nn.Parameter(loaded["adapter"].clone())
        resumed_optimizer = torch.optim.AdamW(
            [resumed_parameter], lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
        )
        resumed_optimizer.load_state_dict(loaded["optimizer"])
        resumed_generator = torch.Generator(device="cpu")
        resumed_generator.set_state(loaded["generator"])
        torch.set_rng_state(loaded["global_cpu"])
        torch.cuda.set_rng_state_all(loaded["global_cuda"])
        steps(resumed_parameter, resumed_optimizer, resumed_generator, 2)
        resumed = capture(resumed_parameter, resumed_optimizer, resumed_generator, 4)
        exact_result = exact(baseline, resumed)
        require(exact_result, "independent optimizer/RNG resume mismatch")
        ciphertext_bytes = current.stat().st_size
    return {
        "adapter_optimizer_generator_global_rng_exact": exact_result,
        "DPAPI_round_trip": True,
        "inner_hash_verified": True,
        "ciphertext_bytes": ciphertext_bytes,
        "plaintext_file_created": False,
        "temporary_ciphertext_retained": False,
        "status": "PASS",
    }


def static_audit(secure_path: Path, runner_path: Path) -> dict[str, Any]:
    secure_source = secure_path.read_text(encoding="utf-8")
    runner_source = runner_path.read_text(encoding="utf-8")
    ast.parse(secure_source)
    runner_tree = ast.parse(runner_source)
    required_secure_tokens = [
        "CryptProtectData",
        "CryptUnprotectData",
        "io.BytesIO()",
        "weights_only=True",
        "os.fsync",
        "os.replace",
        'temporary.open("xb")',
        "load_envelope(temporary)",
        'current.name + ".new"',
        'current.stem + ".previous"',
        "nih-cxr14-k5-feasibility-resume-v1",
    ]
    for token in required_secure_tokens:
        require(token in secure_source, f"secure-resume static token: {token}")
    function_names = {
        node.name for node in runner_tree.body if isinstance(node, ast.FunctionDef)
    }
    require(
        {
            "initialize_state",
            "run_steps",
            "capture_payload",
            "restore_payload",
            "compare_final",
            "run_arm",
            "scan_public",
        }
        <= function_names,
        "runner functions",
    )
    required_runner_tokens = [
        "secrets.token_bytes(32)",
        "run_steps(baseline_state, 4)",
        "run_steps(interrupted_state, 2)",
        "run_steps(resumed_state, 2)",
        "save_envelope_atomic(current, step0_payload)",
        "save_envelope_atomic(current, step2_payload)",
        "del interrupted_state",
        "restore_payload(restored_payload)",
        "torch.cuda.set_rng_state_all",
        "optimizer_exact",
        "public_trace_exact",
        "scan_public(report)",
    ]
    for token in required_runner_tokens:
        require(token in runner_source, f"resume runner static token: {token}")
    require("write_bytes" not in secure_source, "secure module direct write_bytes prohibited")
    return {
        "secure_resume_ast": "PASS",
        "runner_ast": "PASS",
        "ciphertext_only_atomic_sequence_present": True,
        "weights_only_deserialization_present": True,
        "four_vs_two_resume_two_present": True,
        "secret_public_scan_present": True,
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
        require(not (set(value) & forbidden), "resume public secret field")
        for child in value.values():
            scan_public(child)
    elif isinstance(value, list):
        for child in value:
            scan_public(child)
    elif isinstance(value, bytes):
        raise RuntimeError("resume public bytes field")


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    gate_dir = root / "_reports" / "nih_cxr14_k5_encrypted_resume_preflight_gate_v1_001"
    require(not gate_dir.exists(), "refusing to overwrite encrypted-resume independent gate")
    paths = {
        "protocol": root
        / "_reports"
        / "nih_cxr14_k5_encrypted_resume_preflight_protocol_v1_001"
        / "protocol.json",
        "public_report": root
        / "_reports"
        / "nih_cxr14_k5_encrypted_resume_preflight_v1_001"
        / "public_report.json",
        "runner": root / "dp_training" / "run_k5_resume_equivalence_preflight.py",
        "secure_resume": root / "dp_training" / "secure_resume.py",
        "mechanism": root / "dp_training" / "mechanism.py",
    }
    expected = {
        "protocol": PROTOCOL_SHA256,
        "public_report": PUBLIC_REPORT_SHA256,
        "runner": RUNNER_SHA256,
        "secure_resume": SECURE_RESUME_SHA256,
        "mechanism": MECHANISM_SHA256,
    }
    for name, digest in expected.items():
        require(paths[name].is_file() and sha256_file(paths[name]) == digest, f"hash drift: {name}")
    public = json.loads(paths["public_report"].read_text(encoding="utf-8"))
    require(public["status"] == "PASS_EXACT_ENCRYPTED_RESUME_EQUIVALENCE", "resume status")
    require(public["all_four_arms_exact"] is True, "all-four exact flag")
    require(set(public["arms"]) == {"M0", "M1-I8", "M1-G8", "M2-P8"}, "arm set")
    for arm, result in public["arms"].items():
        require(result["status"] == "PASS", f"{arm} status")
        require(all(result["state_checks"].values()), f"{arm} state checks")
        require(result["trace_records"] == 4, f"{arm} trace count")
        require(result["envelope"]["step0_became_previous"], f"{arm} rotation")
        require(not result["temporary_envelopes_retained"], f"{arm} temporary retention")
    scan_public(public)
    static = static_audit(paths["secure_resume"], paths["runner"])
    dynamic = dynamic_equivalence()
    report = {
        "schema": SCHEMA,
        "status": "PASS_EXACT_ENCRYPTED_RESUME_INDEPENDENT",
        "source_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "primary_report": {
            "four_arms": 4,
            "all_state_checks_true": True,
            "secret_field_found": False,
            "temporary_envelopes_retained": False,
            "status": "PASS",
        },
        "static_audit": static,
        "independent_dynamic_probe": dynamic,
        "full_K5_optimizer_execution_started": False,
        "boundary": "synthetic exact-state and encrypted-persistence evidence only; not full-model restart, privacy, utility, or release evidence",
    }
    gate_dir.mkdir(parents=True)
    report_path = gate_dir / "independent_verification.json"
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"independent_report_sha256={sha256_file(report_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
