# Step 4 — Provisional Exact Base-Model Gate Decision

> **Superseded before training on 2026-09-02.** This file preserves the verified SD v1.4 fallback
> record. The active decision is `SD21_BASE_MODEL_GATE_DECISION.md`; no generator or DP training
> occurred between the provisional and reopened decisions.

- Decision date: 2026-09-02 (Asia/Seoul)
- Status: **PASS_FOR_SKELETON; not confirmatory-training evidence**
- Selected implementation base: `CompVis/stable-diffusion-v1-4`
- Exact revision: `133a221b8aa7292a167afc5127cb63fb5005638b`
- Weight variant: `fp16` safetensors

## Decision

Use the exact CompVis Stable Diffusion v1.4 revision above for the first integrated medical
generator skeleton and real-data pilot. This is an official CompVis repository, is a 512-pixel
latent diffusion pipeline with a 4x64x64 latent at 512x512, and passed the local RTX 3070 inference
and LoRA backward gate.

This is a base selection, not a result claim. It does not authorize confirmatory DP training,
release of a medical generator, or reuse of PP-Mark thresholds measured on SD2.1/SDXL.

## Why this checkpoint was selected

| Candidate | Decision | Reason |
|---|---|---|
| `CompVis/stable-diffusion-v1-4` | Selected for skeleton/pilot | Official source, accessible exact revision, 512 latent layout, OpenRAIL-M model card, local 8 GB PASS |
| `Manojb/stable-diffusion-2-1-base` | Reference only | PP-Mark historically used it and its UNet hash matches the formerly indexed Stability AI file, but the currently accessible repository is a third-party mirror |
| `stabilityai/stable-diffusion-2-1-base` | Not currently usable | Model/API access returned unavailable even with the configured Hugging Face credential on 2026-09-02 |
| `stabilityai/stable-diffusion-xl-base-1.0` | Deferred extension | Official and PP-Mark-tested, but the 3B/1024-pixel base adds unnecessary DP and 8 GB compute risk before the lifecycle works |

The selected base may be revised only before confirmatory training and only for a recorded failure
such as material contamination, real-data utility collapse, DP-mechanism incompatibility, or
PP-Mark incompatibility. It may not be changed because another checkpoint gives a preferred attack
ordering.

## License gate

Official sources:

- Model card: <https://huggingface.co/CompVis/stable-diffusion-v1-4>
- CompVis model license: <https://github.com/CompVis/stable-diffusion/blob/main/LICENSE>
- Official LoRA training guide: <https://huggingface.co/docs/diffusers/en/training/lora>

The model card identifies CreativeML OpenRAIL-M. It permits fine-tuning subject to its use-based
restrictions and imposes notice/license obligations when distributing model derivatives. This is
compatible with the internal dissertation pilot. It does not remove the independent CC BY-NC 4.0
boundary of the aggregate ISIC release. Do not redistribute the trained adapter, checkpoint or
medical-image derivatives until both sets of obligations and attribution are packaged explicitly.

## Exact artifact gate

Only the 14 files required for a safety-checker-free PyTorch pipeline were downloaded. The local
snapshot totals 1.988 GiB. The full local manifest is
`_reports/base_gate_compvis_sd14_r133a_001/base_snapshot_manifest.json`; its digest is bound from
the result file.

Critical official LFS hashes were independently matched after download:

| File | SHA-256 | Match |
|---|---|---|
| `text_encoder/model.fp16.safetensors` | `77795E2023ADCF39BC29A884661950380BD093CF0750A966D473D1718DC9EF4E` | PASS |
| `unet/diffusion_pytorch_model.fp16.safetensors` | `A35404D03EC8F977715A4D2A080DDF72E2144F2EE49BB1EE213258BC64F9CC87` | PASS |
| `vae/diffusion_pytorch_model.fp16.safetensors` | `4FBCF0EBE55A0984F5A5E00D8C4521D52359AF7229BB4D81890039D2AA16DD7C` | PASS |

Result digest:

- `base_snapshot_manifest.json` SHA-256:
  `65942C6C213361881EE74F1BDB940DC9ECF13CE4121EB9A64E533D46B81784F0`

The eventual model-bound receipt must bind the revision plus every file in the local manifest, not
only the model name.

## RTX 3070 diagnostic

Environment: RTX 3070 8 GiB, PyTorch 2.6.0+cu124, Diffusers 0.30.2, PEFT 0.12.0. The gate disabled a
broken optional host `onnxruntime` import; the tested path is native PyTorch only.

| Probe | Status | Peak allocated | Peak reserved |
|---|---|---:|---:|
| 256x256, 2-step inference, CFG 7.5 | PASS | 2.179 GiB | 2.332 GiB |
| 512x512, 2-step inference, CFG 7.5 | PASS | 2.625 GiB | 3.086 GiB |
| 256x256, rank-8 LoRA, batch 1, one backward/clip/update | PASS | 1.713 GiB | 1.813 GiB |
| 512x512, rank-8 LoRA, batch 1, one backward/clip/update | PASS | 1.889 GiB | 1.990 GiB |

The LoRA probe exposed 1,594,368 trainable parameters out of 861,115,332 total parameters
(0.185%). It used random precomputed latents and text embeddings. Its batch-size-one gradient clip
is only a memory-path diagnostic: no DP noise, secure RNG, accountant, patient aggregation or
privacy claim was present.

Therefore the RTX 3070 is classified as **smoke/pilot-capable**. It is not yet classified as
confirmatory-capable because real image encoding, DP per-sample/group gradients, K-image patient
aggregation, repeated steps, three seeds and the full attack pipeline were not timed.

## Pretraining-contamination gate

The CompVis model card states that SD v1.4 was trained on LAION-2B(en) and subsets. ISIC 2020 was
public before this model was trained, so the absence of ISIC images from pretraining cannot be
established from the model card. The status is **UNRESOLVED_RISK**, not PASS-NO-OVERLAP.

Mandatory controls remain:

1. retain B0 untouched-base attacks and nearest-neighbor results;
2. check exact and perceptual/embedding near-duplicates against the full acquired ISIC pool;
3. compare B0 with M0/M1/M2 and restrict membership wording to incremental fine-tuning exposure;
4. replace the base under a new preregistered manifest if material overlap prevents interpretation.

This unresolved risk is acceptable for starting the skeleton because it is observable and has a
predeclared fail condition. It is not acceptable to claim that the public ISIC cohort was absent
from pretraining.

## PP-Mark compatibility gate

The selected model has the non-SDXL `StableDiffusionPipeline`, DDIM scheduler interface, VAE factor
8 and 4x64x64 latent layout expected by the existing 512-pixel PP-Mark path. Static architecture
compatibility therefore passes.

One integration gap is explicit: current PP-Mark loaders pin neither a repository revision nor an
fp16 `variant` for non-SDXL models. The next skeleton must add an adapter that accepts the exact
local snapshot/manifest and binds it into the receipt. It must then recalibrate clean thresholds and
rerun embedding/inversion/robustness tests on the fine-tuned medical model. Prior SD2.1/SDXL
thresholds are historical evidence only.

## Gate conclusion

The exact CompVis SD v1.4 revision passes the license, artifact-integrity, architecture and local
VRAM checks needed to build the next skeleton. Confirmatory training remains blocked until the
patient split/cap manifests, real-data loader, DP mechanism/accountant, patient-MIA protocol,
operational privacy budget and PP-Mark exact-revision adapter pass their own gates.
