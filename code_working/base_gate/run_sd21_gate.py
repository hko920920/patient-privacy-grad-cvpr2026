#!/usr/bin/env python3
"""Run the exact-base gate on the hash-pinned SD 2.1 mirror used by PP-Mark."""

import run_base_gate


run_base_gate.MODEL_ID = "Manojb/stable-diffusion-2-1-base"
run_base_gate.MODEL_REVISION = "0094d483a120f3f33dafbd187ea4aa60d10de75c"
run_base_gate.MODEL_VARIANT = "fp16"
run_base_gate.ARTIFACT_PROVENANCE = {
    "repository_role": "third_party_hash_pinned_mirror",
    "mirror_commit_title": "Cloned from stabilityai/stable-diffusion-2-1-base",
    "mirror_commit_created_at_utc": "2025-06-13T20:37:56Z",
    "upstream_repository": "stabilityai/stable-diffusion-2-1-base",
    "upstream_status_checked_2026_09_02": "unavailable_via_huggingface_api",
    "claim_limit": "Do not describe the accessible repository itself as Stability AI official.",
}
run_base_gate.EXPECTED_CRITICAL_SHA256 = {
    "text_encoder/model.fp16.safetensors": (
        "681C555376658C81DC273F2D737A2AEB23DDB6D1D8E5B3A7064636D359A22668"
    ),
    "unet/diffusion_pytorch_model.fp16.safetensors": (
        "28EC9CF3B239C0751C201B1F6FB46B551DF5862731B30A37AA1360101CB3FBAB"
    ),
    "vae/diffusion_pytorch_model.fp16.safetensors": (
        "3E4C08995484EE61270175E9E7A072B66A6E4EEB5F0C266667FE1F45B90DAF9A"
    ),
}


if __name__ == "__main__":
    run_base_gate.main()
