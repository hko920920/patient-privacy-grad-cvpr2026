#!/usr/bin/env python3
"""Freeze exact encrypted-restart equivalence semantics before the preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "nih-cxr14-k5-encrypted-resume-preflight-protocol/v1"
MAIN_PROTOCOL_SHA256 = "2234B3BA8701565B768A11AB5196B2F696788900D07254833997D7823489E9DA"
EVALUATOR_REPORT_SHA256 = "914002B52C2EF48C551FFD9A5CD0260BA7708EA6492F5296AEF98BF6D0B4F5B5"
MECHANISM_SHA256 = "B4B1E037A3704238B06B442970EAB082433835201B750546C8281BBFFE2B969E"
SECURE_RESUME_SHA256 = "360D66DA4A9FAE4060C806A71FB02555E58C1E0E71188CFDDC261EDC95895E47"


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


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    output = root / "_reports" / "nih_cxr14_k5_encrypted_resume_preflight_protocol_v1_001"
    require(not output.exists(), "refusing to overwrite encrypted-resume preflight protocol")
    sources = {
        "main_protocol": (
            root / "_reports" / "nih_cxr14_k5_feasibility_protocol_v1_001" / "protocol.json",
            MAIN_PROTOCOL_SHA256,
        ),
        "evaluator_report": (
            root / "_reports" / "nih_cxr14_k5_evaluator_preflight_v1_001" / "public_report.json",
            EVALUATOR_REPORT_SHA256,
        ),
        "mechanism": (root / "dp_training" / "mechanism.py", MECHANISM_SHA256),
        "secure_resume": (root / "dp_training" / "secure_resume.py", SECURE_RESUME_SHA256),
    }
    for name, (path, digest) in sources.items():
        require(path.is_file() and sha256_file(path) == digest, f"source hash drift: {name}")
    evaluator = json.loads(sources["evaluator_report"][0].read_text(encoding="utf-8"))
    require(evaluator["status"] == "PASS_K5_EVALUATOR_PREFLIGHT", "evaluator gate is not PASS")

    protocol = {
        "schema": SCHEMA,
        "status": "FROZEN_BEFORE_EXACT_RESUME_OUTPUT",
        "scope": "SYNTHETIC_STATE_EQUIVALENCE_NOT_TRAINING_NOT_PRIVACY_NOT_UTILITY",
        "upstream_sha256": {name: digest for name, (_, digest) in sources.items()},
        "comparison": {
            "arms": ["M0", "M1-I8", "M1-G8", "M2-P8"],
            "baseline": "four uninterrupted optimizer steps",
            "resume": "save encrypted step-0 envelope, run two steps, atomically rotate step-2 envelope, destroy in-memory state, decrypt in a fresh state object, then run steps three and four",
            "required_exact_equal": [
                "fp32 adapter tensor",
                "complete AdamW state",
                "all domain-separated torch.Generator states",
                "global CPU and all CUDA RNG states",
                "M0 permutation/cursor/epoch or DP sampler state",
                "public trace records and head",
                "committed step and frozen input digests",
            ],
        },
        "synthetic_fixture": {
            "adapter_parameters": 257,
            "initialization": "same public deterministic fp32 vector for every arm",
            "optimizer": {
                "type": "AdamW",
                "learning_rate": 0.0001,
                "betas": [0.9, 0.999],
                "epsilon": 1e-8,
                "weight_decay": 0.01,
            },
            "M0": "18 ordered image units, fixed batch 8, drop incomplete epoch remainder; four steps cross an epoch boundary",
            "M1": "64 ordered synthetic image units, Poisson expected batch 8; exact K5 C and arm-specific sigma; fixed denominator 8; empty remains noise-only",
            "M2": "16 ordered synthetic patients with cyclic 1..5 image counts, Poisson expected batch 4, uniformly choose without replacement up to four images, mean before exact K5 patient clip, fixed denominator 4",
            "gradient": "deterministic synthetic record vector plus domain-separated diffusion RNG and restored global CPU/CUDA draws",
            "accounting_boundary": "synthetic populations and q stress restart semantics only; they are not K5 privacy-accounting events",
        },
        "randomness": {
            "experiment_root": "fresh 256-bit operating-system entropy, shared only between the paired baseline/resume comparisons in process memory",
            "domain_separation": "HMAC-SHA256 per arm and sampling/diffusion/inner/Gaussian/global stream",
            "backend": "ordinary PyTorch RNG; RESEARCH_ONLY_NONCRYPTOGRAPHIC",
            "public_output": "never serialize or report the root, seeds, RNG states, or their digests",
        },
        "envelope": {
            "platform": "Windows DPAPI CryptProtectData/CryptUnprotectData",
            "scope": "CurrentUser",
            "optional_entropy_utf8": "nih-cxr14-k5-feasibility-resume-v1",
            "serialization": "torch.save to memory; inner SHA-256 framing; torch.load weights_only=True",
            "write": "ciphertext-only .new opened exclusively, flush+fsync, decrypt/hash/schema verify, os.replace current to previous, os.replace .new to current",
            "retention_test": "step-0 becomes previous when step-2 becomes current",
            "tamper_and_stale_new": "fail closed",
            "preflight_cleanup": "temporary ciphertext envelopes are removed after their hashes/byte counts and exact equality are recorded; no plaintext file exists",
        },
        "decision": {
            "PASS": "all four arms are exact on every required state component, DPAPI rotation/cleanup passes, and no secret field enters the public report",
            "FAIL": "retain the failure, block full K5, and require a separately frozen correction and fresh preflight",
        },
        "next_on_pass": "independently verify result and source, then build/freeze the full runner; do not start the 4000-step matrix",
    }
    output.mkdir(parents=True)
    protocol_path = output / "protocol.json"
    write_json(protocol_path, protocol)
    report = {
        "schema": "nih-cxr14-k5-encrypted-resume-preflight-protocol-build/v1",
        "status": "PASS_PROTOCOL_BUILD_EXACT_RESUME_NOT_RUN",
        "protocol_sha256": sha256_file(protocol_path),
        "exact_resume_executed": False,
    }
    write_json(output / "builder_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
