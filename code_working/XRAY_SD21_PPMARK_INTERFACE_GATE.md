# NIH chest X-ray raw-to-SD 2.1/PP-Mark interface gate

- Decision date: 2026-09-03 (Asia/Seoul)
- Status: **PASS_FOR_PILOT_INTERFACE**
- Scope: deterministic real-X-ray input and execution-interface evidence only
- Not performed: optimizer step, gradient clipping, DP noise/accounting, generator training,
  utility or privacy attack, PP-Mark calibration/robustness, receipt, release, or active-LaTeX edit

## Decision

The exact acquired NIH ChestXray14 PA images can pass through one deterministic preprocessing
contract into the pinned SD 2.1 VAE, the rank-8 LoRA gradient path, and PP-Mark's exact DDIM
inversion path. Both 256 and 512 pixel profiles execute on the local RTX 3070 and reproduce their
VAE, gradient, and inversion outputs bit-for-bit on immediate replay.

This does not change the registered resolution decision. `P256` remains the feasibility/pilot
profile. `P512` is now a verified available interface, but it becomes a confirmatory setting only
if preregistered medical-utility and compute criteria justify it. It must not be activated merely
because one resolution produces a preferred privacy-attack ordering.

## Frozen preprocessing contract

The input adapter is `data_pipeline/nih_cxr14_model_input.py`.

1. Accept only the source-locked 1024x1024 PNG geometry.
2. Preserve native `L` directly.
3. Accept native `RGBA` only when every pixel satisfies `R=G=B` and alpha is 255; then use the
   shared intensity channel. Actual color or transparency fails closed.
4. Keep the full square field of view. Apply no crop and no padding.
5. Resize the one grayscale channel with Pillow 11.3.0 `LANCZOS`, with `reducing_gap=None`.
6. Replicate the resized grayscale channel to RGB only after resize.
7. Convert model pixels as float32 `x / 127.5 - 1.0`.
8. Apply no augmentation, histogram equalization, windowing, per-image standardization, or
   private-population-derived statistic.
9. Generate resized images on demand. Do not create a second 42,423-image cache.

The exact contract is `_reports/nih_cxr14_sd21_ppmark_interface_v1_001/preprocessing_contract.json`.
Its SHA-256 is
`0ED9E434764BDB2D542BB16AB951F856CB5B82D4F2E6B9F25F6C554F1076669D`.

The two deterministic profile commitments are:

| Profile | Role | Sample pixel/tensor/prompt commitment |
|---|---|---|
| P256 | feasibility and pilot primary | `149ED15BB149862787050C365BFE143ADA76BE69CAD1BC3A75BC4C71AD894CAD` |
| P512 | gated confirmatory extension | `1C71C0E77F36025DAC9DA471B02995FF2563A10909218BE264D5389182B7B326` |

## Prompt-conditioning contract

Every PA record uses a fixed prefix, `a frontal posteroanterior chest radiograph`. `No Finding`
maps to `with no labeled finding`; all other automatically mined NIH labels are rendered in one
fixed canonical order as `with radiographic findings of ...`. Mass and nodule are described as
opacities and are never relabeled as cancer.

The prompt excludes patient ID, age, sex, and the target-enrichment flag. It uses all mined finding
labels rather than collapsing non-target abnormalities into a nominal control condition. Across
all 4,831 K10 public-development records, 203 distinct prompts were produced; the longest used 43
of the tokenizer's 77 positions and no prompt was truncated. These remain weak-label conditions,
not radiologist-adjudicated diagnoses.

## Outcome-independent interface sample

The test used only K10 `public_development`, before any generator or attack result. That partition
contains 4,831 images from 1,816 patients: 4,804 native `L` and 27 native `RGBA` images. Eight
patient-distinct records were selected by the minimum
`SHA256(salt | stratum | image_id)` in fixed strata covering:

- positive and `No Finding` native `RGBA`;
- native-`L` `No Finding`;
- pneumothorax, pneumonia/consolidation, pleural effusion, and mass/nodule;
- a native-`L` multilabel case.

The selection uses no model-quality, privacy-attack, or PP-Mark score. Its local detailed manifest
is `selected_interface_records_private.csv`, SHA-256
`6B48AD409CFF55DAF0BCF4C38E3F07D4A8EB173156417915A95A721C7DE464CD`.

## Exact model identity

The gate reused the selected third-party, hash-pinned SD 2.1 mirror only:

- repository: `Manojb/stable-diffusion-2-1-base`;
- revision: `0094d483a120f3f33dafbd187ea4aa60d10de75c`;
- text encoder:
  `681C555376658C81DC273F2D737A2AEB23DDB6D1D8E5B3A7064636D359A22668`;
- UNet: `28EC9CF3B239C0751C201B1F6FB46B551DF5862731B30A37AA1360101CB3FBAB`;
- VAE: `3E4C08995484EE61270175E9E7A072B66A6E4EEB5F0C266667FE1F45B90DAF9A`.

No network or mutable-revision fallback was used.

## Measured real-X-ray results

Two source cases were sent through the GPU interfaces: one native-`RGBA` mass-opacity case and one
native-`L` pneumothorax/effusion/fibrosis case. The latter exact preprocessed pixels were used by
both the VAE/LoRA path and PP-Mark; the `RGBA` case additionally checks the channel-normalization
boundary end to end.

### VAE

| Profile | Expected latent | Cases | Replay | Peak reserved |
|---|---:|---:|---|---:|
| P256 | `1x4x32x32` | 2 | exact, finite | 0.3438 GiB |
| P512 | `1x4x64x64` | 2 | exact, finite | 0.7207 GiB |

The deterministic VAE statistic is `latent_dist.mode`; no VAE sampling is used.

### Real-image LoRA gradient

The UNet used rank 8 on `to_q`, `to_k`, `to_v`, and `to_out.0`, exposing 1,659,904 trainable
parameters out of 867,570,628. Fixed real-image latents, prompt embeddings, timestep 500, and noise
seeds produced:

| Profile | Loss | Gradient L2 | Gradient SHA-256 | Two backwards | Peak reserved |
|---|---:|---:|---|---:|---:|
| P256 | 0.09761695 | 0.08244545 | `73F78DB6F6A42E4CCFCB43144E5517F51FAFF81176F8011117FDD38870041CEF` | 1.1689 s | 1.9785 GiB |
| P512 | 0.06776760 | 0.00392318 | `E809C3734836477788B1C93AF35AE34738CF22E7CA0B0FE4059C45F95013FB7E` | 0.9961 s | 2.0586 GiB |

Each backward was repeated without changing weights; both gradient digests and all scalar outputs
matched exactly. The adapter before/after digests are equal. No optimizer was instantiated, no
clip was applied, and no noise was added. The different gradient magnitudes are diagnostic values,
not evidence that either resolution is more private or useful.

### PP-Mark exact DDIM inversion

The exact same committed preprocessed pixels were supplied as HWC float32 `[0,1]` arrays to
`ppmark_v03.ddim_unet`. With two inversion steps and guidance 0:

| Profile | Expected latent | Cases | Replay | Peak reserved |
|---|---:|---:|---|---:|
| P256 | `4x32x32` | 2 | exact, finite | 2.7031 GiB |
| P512 | `4x64x64` | 2 | exact, finite | 2.9746 GiB |

This proves the adapter boundary and the historical PP-Mark latent interface are executable on
real X-rays. It does not transfer an old threshold or establish watermark detectability,
geometric robustness, removal resistance, forgery resistance, or medical-image fidelity.

## Independent verification

`verify_nih_cxr14_sd21_interface_independent.py` imports neither the loader nor the gate. It
independently rebuilt the eight-record hash selection, rechecked the raw source hashes, recomputed
all 16 P256/P512 pixel and tensor digests, reproduced both profile commitments, and checked the
model/VAE/LoRA/PP-Mark report invariants. Status: **PASS**.

- main report SHA-256:
  `B5C7826BEDB0E6584A5A5346FC70F7A43BE71C64E59062E66667AB1E430AE6F0`;
- independent verification SHA-256:
  `7B03A7AEDD662F21DDF3796B70F84CF4823376981334C6CCD0B7645F2980C6B3`;
- four loader unit tests: PASS.

## Interpretation and next gate

This gate closes the raw-image/channel/geometry/intensity/prompt and SD2.1-to-PP-Mark interface
questions for the pilot. It does not show that 256 pixels preserve subtle nodules, that the base
generates acceptable chest radiographs, or that patient-level DP is affordable. Those require
numerical public-development utility and execution checks.

The next gate is therefore the DP mechanism/accountant/privacy-budget/attack-protocol freeze,
informed by this exact input contract. It must be reported before optimizer training begins. The
eventual medical LoRA checkpoint still requires fresh PP-Mark embedding, calibration, attack, and
quality experiments.
