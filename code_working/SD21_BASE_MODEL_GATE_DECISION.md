# Reopened Base-Model Gate: SD 2.1

- Decision date: 2026-09-02 (Asia/Seoul)
- Status: **PASS_FOR_SKELETON_AND_PILOT; not confirmatory-training evidence**
- Selected implementation base: `Manojb/stable-diffusion-2-1-base`
- Exact revision: `0094d483a120f3f33dafbd187ea4aa60d10de75c`
- Weight variant: `fp16` safetensors
- Role of SD v1.4: verified fallback only

## Decision

Use the exact, hash-pinned SD 2.1 artifact above for the integrated medical-generator skeleton and
real-data pilot. This decision supersedes the provisional SD v1.4 selection before any generator or
DP training occurred. SD v1.4 is retained only as a recoverable fallback; it is not the current
experimental base.

The reason is substantive rather than nominal version matching: SD 2.1 is the base already used by
the historical PP-Mark implementation, retains the required 512-pixel factor-8, four-channel latent
interface, and has now passed exact-artifact, RTX 3070 LoRA, and PP-Mark inversion gates locally.
There is therefore no measured need to introduce an avoidable cross-version integration variable.

This is still a base-selection result only. It does not establish medical generation quality,
patient-level DP, attack resistance, or PP-Mark robustness after medical LoRA fine-tuning.

## Provenance boundary

The formerly referenced upstream repository,
`stabilityai/stable-diffusion-2-1-base`, was unavailable through the Hugging Face API on the decision
date. The accessible repository is therefore described exactly as a **third-party, hash-pinned
mirror**, never as a currently accessible official Stability AI repository.

The mirror has only two commits. Its exact selected commit is titled `Cloned from
stabilityai/stable-diffusion-2-1-base`. The repository metadata and all 28 file identities are frozen
in `_reports/base_gate_sd21_manojb_r0094_001/repository_metadata.json`. The same three fp16 component
hashes and the same 5.21 GB monolithic safetensors hash are independently present in the
`sd2-community/stable-diffusion-2-1-base` archive; the monolithic hash is also present in
`patrickvonplaten/v2-1-base`, whose history labels it a duplicate of the Stability AI repository.
This corroborates artifact identity but does not convert any mirror into an official source.

The executable identity is the exact revision plus every local file digest, not the mutable model
name or the mirror owner's statement. If an authoritative upstream artifact becomes accessible
later, equality must be established by hashes before relabeling provenance.

## License boundary

The pinned model card declares CreativeML Open RAIL++-M and links to the Stability AI model license.
The card describes SD 2.1-base as a 512-pixel text-to-image latent diffusion model, fine-tuned for an
additional 220,000 steps from SD 2.0-base. This is acceptable for the internal dissertation pilot
subject to the license's use restrictions and notice obligations.

This does not override the independent CC BY-NC 4.0 boundary of the aggregate ISIC release. Do not
redistribute the medical adapter, checkpoint, or derivatives until model-license notices, dataset
attribution, non-commercial limits, and release-review requirements are packaged explicitly.

## Exact-artifact gate

Only the 14 files required for the safety-checker-free PyTorch pipeline were downloaded. The local
snapshot totals 2,581,663,938 bytes (2.404 GiB); fp32 weights and four redundant 5.21 GB monolithic
checkpoints were deliberately excluded.

| Component | SHA-256 | Status |
|---|---|---|
| text encoder fp16 safetensors | `681C555376658C81DC273F2D737A2AEB23DDB6D1D8E5B3A7064636D359A22668` | PASS |
| UNet fp16 safetensors | `28EC9CF3B239C0751C201B1F6FB46B551DF5862731B30A37AA1360101CB3FBAB` | PASS |
| VAE fp16 safetensors | `3E4C08995484EE61270175E9E7A072B66A6E4EEB5F0C266667FE1F45B90DAF9A` | PASS |

Evidence digests:

- local snapshot manifest:
  `E38A5F8FE1695745EDD60FB6E2EEDFD35A01A65B5E981ACA28EEF026F4C02638`
- repository metadata:
  `8543034211987A33A8DF28A998DCF770FF9DB68B411EC2C65DF048BF794CB4B5`

Any later download or receipt construction must fail closed if these identities change.

## RTX 3070 diagnostic

Environment: RTX 3070 8 GiB, PyTorch 2.6.0+cu124, Diffusers 0.30.2, PEFT 0.12.0.

| Probe | Status | Peak allocated | Peak reserved |
|---|---|---:|---:|
| 256x256, 2-step inference, CFG 7.5 | PASS | 2.5782 GiB | 2.7559 GiB |
| 512x512, 2-step inference, CFG 7.5 | PASS | 3.0237 GiB | 3.5293 GiB |
| 256x256, rank-8 LoRA, batch 1, one backward/clip/update | PASS | 1.7205 GiB | 1.8164 GiB |
| 512x512, rank-8 LoRA, batch 1, one backward/clip/update | PASS | 1.8627 GiB | 2.0039 GiB |

The LoRA probe exposed 1,659,904 trainable parameters out of 867,570,628 total parameters
(0.191%). It used random precomputed latents and text embeddings. Its single-record gradient clip
contains no DP noise, patient aggregation, secure RNG, or accountant, so it supports no privacy
claim. The GPU remains classified as smoke/pilot-capable until the real patient-level DP path is
timed.

## PP-Mark compatibility

The exact same pinned SD 2.1 snapshot passed the revised fail-closed PP-Mark loader and a two-step
DDIM inversion of a real ISIC image:

- finite latent: `4x32x32` at 256x256 input;
- peak reserved GPU memory: 2.6504 GiB;
- result digest:
  `65C8927B082CA2BF620B8A5D001C547E4A7DAA2C3392D16F99F6A62A03CA5D24`;
- PP-Mark regression: 16 passed, 5 environment-dependent skips.

This removes the base-version mismatch, but it does **not** authorize transplanting historical
PP-Mark thresholds. Medical LoRA changes the generator checkpoint; clean thresholds, embedding,
inversion, removal/forgery attacks, geometry robustness, and quality must all be recalibrated on the
exact fine-tuned checkpoint.

## Remaining risks and fallback rule

Pretraining overlap with publicly hosted ISIC images remains unresolved. Preserve the untouched-base
control, perform exact/perceptual near-duplicate checks, and restrict membership conclusions to the
incremental fine-tuning exposure if upstream membership cannot be ruled out.

Do not fall back to SD v1.4 merely for convenience or because it produces a preferred result. A
fallback is allowed only after a recorded material failure such as corrupted/unverifiable SD 2.1
artifacts, real-data utility collapse under the frozen pilot, DP-mechanism incompatibility, or an
execution failure that cannot be resolved within available compute. Any fallback must rerun the
same exact-artifact, loader, DP, and PP-Mark gates.

## Evidence locations

- `_reports/base_gate_sd21_manojb_r0094_001/base_gate_result.json`
- `_reports/base_gate_sd21_manojb_r0094_001/base_snapshot_manifest.json`
- `_reports/base_gate_sd21_manojb_r0094_001/repository_metadata.json`
- `_reports/base_gate_sd21_manojb_r0094_001/ppmark_exact_loader_result.json`
- `base_gate/run_sd21_gate.py`
- `base_gate/capture_sd21_repo_metadata.py`

