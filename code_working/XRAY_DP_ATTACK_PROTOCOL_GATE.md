# NIH CXR14 DP mechanism, accounting, and attack-protocol gate

- Date: 2026-09-03 (Asia/Seoul)
- Decision: **PASS_PROTOCOL_FREEZE_TRAINING_STILL_BLOCKED**
- Scope: pre-training contract only
- Primary machine-readable contract:
  `_reports/nih_cxr14_dp_attack_protocol_v1_001/protocol.json`
- Protocol SHA-256:
  `F2757DC7EBC7488B6A9DD0F69227419EA16BB9A3E9BD4434514CB19AFD2B4018`

## 1. What this gate resolves

The raw X-ray and SD 2.1/PP-Mark input interface had already passed, but that did not define a DP
algorithm. This gate now freezes, before any optimizer update:

1. the exact image-DP and patient-DP adjacency, sampling, aggregation, clipping, noise, and fixed
   denominators;
2. the RDP orders, dual-accountant calibration rule, maximum steps, epsilon/delta targets, and
   image-to-patient conversion;
3. an approximately image-work-matched M1/M2 comparison;
4. outcome-independent patient membership cohorts, attack scores, uncertainty, materiality rules,
   and extraction query budget;
5. the distinction between research PRNG evidence and a release-grade secure-noise path.

No model was trained and no attack was run. A protocol pass is not an executed DP guarantee.

## 2. Final mechanism

Both mechanisms use P256, the pinned SD 2.1 revision, the same rank-8 attention LoRA, AdamW at
`1e-4`, a constant learning rate, and a hard 4,000-step final checkpoint. Diagnostic checkpoints at
steps 1,000 and 2,000 cannot be selected or released. LoRA trainable parameters are explicitly
cast to fp32 while frozen base parameters remain fp16.

### M1: image-DP

- Independently include each image with Poisson probability `q=8/N_image` each step.
- Compute one full LoRA gradient per image from one uniformly sampled diffusion timestep and one
  latent-noise draw.
- Clip each image vector to `C_image`, sum, add `N(0,(sigma*C_image)^2 I)`, and divide by the fixed
  public expected batch 8.
- A realized empty batch still receives a noise-only update. Dividing by realized batch size is
  prohibited.

### M2: native patient-DP

- Independently include each patient with Poisson probability `q=4/8,476` each step.
- For each included patient, uniformly sample without replacement at most four K-capped images,
  using all images when the patient has fewer than four.
- Average those image losses, compute one equal-weight patient gradient, clip it to `C_patient`,
  sum patient vectors, add `N(0,(sigma*C_patient)^2 I)`, and divide by fixed expected patient batch 4.
- All within-patient operations occur before the one patient vector is clipped. Thus the protected
  adjacency is addition/removal of one patient's complete bounded contribution, not one image.

K10 private train has `sum_i min(4,n_i)=17,001` over 8,476 patients. M2 therefore processes an
expected `8.023124` raw images per step, versus exactly 8 for M1, a ratio of `1.002891`. Raw-image
work is matched within 1%; backward-pass structure remains inherently unit-specific and is reported
rather than called identical.

## 3. Public-only clip calibration

Clip norms were determined only from K10 `public_development`, before private training and without
an optimizer. The final sample has 72 image units and 72 patient units, balanced across
target/control and contribution buckets `n=1`, `n=2-3`, and `n=4-10`. The selected value is the
exact empirical p80 using NumPy `method='higher'`.

| Unit | Public gradients | Selected C | Units above C | Fraction |
|---|---:|---:|---:|---:|
| Image | 72 | `0.28448700606156724` | 14 | 19.44% |
| Patient | 72 | `0.1997973088974048` | 14 | 19.44% |

All trainable LoRA tensors were observed as fp32. Three sentinel gradients replayed exactly and
the adapter digest was identical before and after all gradient calculations. No clipping/noise was
actually applied and no parameter changed.

Two pre-freeze diagnostics are retained in the final report rather than hidden:

- the first used a grid that selected image `C=0.5` but clipped only 2/72 units, and also exposed
  fp16 LoRA trainables; it was rejected;
- the corrected p80/fp32 run used 72 image but only 24 patient units; it passed mechanically but was
  superseded before private training by the symmetric 72/72 design.

Final clip report SHA-256:
`F9771F169BA3F6468F5AAF254A27AFFE1FB47B1ABA897C365BFFC7CDD11A73F1`.

## 4. Accountant and budget freeze

The accountant uses a Poisson-sampled Gaussian event under add/remove adjacency, 4,000 maximum
steps, and 155 explicit RDP orders. Noise is calibrated to the **larger** epsilon recomputed by
Opacus 1.6.0 and Google `dp-accounting` 0.6.0. The independent verifier reproduced every value with
zero serialized drift.

| Arm | K | Direct unit | Direct (epsilon, delta) | sigma | Patient interpretation |
|---|---:|---|---|---:|---|
| M1-I8 | 2 | image | `(8, 1e-5)` | `0.419767485` | converted `(16, 0.0298196)`; policy blocked |
| M1-G8 | 2 | image | `(4, 1.79862e-7)` | `0.571337588` | converted `(8, 1e-5)` |
| M1-I8 | 5 | image | `(8, 1e-5)` | `0.400704292` | converted delta `>=1`; vacuous |
| M1-G8 | 5 | image | `(1.6, 1.32654e-8)` | `0.835065783` | converted `(8, 1e-5)` |
| M1-I8 | 10 | image | `(8, 1e-5)` | `0.393338498` | converted `(80, >=1)`; vacuous |
| M1-G8 | 10 | image | `(0.8, 4.11261e-9)` | `1.155724687` | converted `(8, 1e-5)` |
| M2-P2 | 10 | patient | `(2, 1e-5)` | `0.624936852` | direct patient bound |
| M2-P4 | 10 | patient | `(4, 1e-5)` | `0.497456487` | direct patient bound |
| M2-P8 | 10 | patient | `(8, 1e-5)` | `0.404457146` | direct patient bound |

The main controlled comparison is K10 M1-I8 versus M1-G8 versus M2-P8, with M0 as the non-DP
reference. M1-G8 is deliberately retained even though its strong K10 base-image budget may damage
utility: it is the exact group-conversion reuse frontier, not a favorable-result arm. M2-P4 is a
secondary privacy-utility point and M2-P2 is a research-only curve point. A0/A1/A2 are one-seed
mechanism controls. Releasing more than one trained DP model would require cross-release privacy
composition; the benchmark matrix is not presented as one multi-model deployment.

The dissertation's operational patient policy is `(epsilon <= 8, delta <= 1e-5)`. It is an
experimental decision threshold, not a universal statement of clinical safety.

## 5. Membership, memorization, and extraction protocol

### Patient membership

- Members: 1,816 hash-selected K10 private-train patients.
- Nonmembers: all 1,816 K10 privacy-attack-holdout patients.
- Each role contains exactly 908 target and 908 control patients.
- The member cohort exactly matches the holdout's `target/control x K10 image count (1..10)` cell
  counts, preventing contribution-count imbalance from masquerading as membership leakage.
- Primary score: negative mean epsilon-prediction MSE over fixed timesteps
  `{50,250,500,750,950}` and four fixed noise replicates per timestep; image scores are averaged to
  one equal-weight patient score.
- Secondary score: the official SecMI-LDM code at commit
  `83eff37e06aadf078a79755164a5cf531bd34b04`, but only after a pinned SD 2.1 adapter-conformance
  gate. It cannot replace the primary score based on observed results.
- Metrics: ROC-AUC, maximum `TPR-FPR`, and TPR at 1% FPR, with 10,000 patient-stratified bootstrap
  replicates and paired arm differences.

With 1,816/1,816 patients, the precomputed Hanley-McNeil approximation gives 80%-power AUC
`0.526843` at two-sided alpha 0.05. The preregistered material AUC is 0.55. Final inference uses the
patient bootstrap, not that approximation.

`LEAKAGE_DETECTED` requires lower AUC CI above 0.50 and point AUC at least 0.55.
`NO_MATERIAL_LEAKAGE_DETECTED` requires upper AUC CI below 0.55 and upper TPR@1%FPR CI below 0.05.
Everything else is `INCONCLUSIVE`. None of these empirical labels alters the formal DP bound.

### Bounded generate-and-filter extraction

Each model receives the same 2,450 P256 DDIM queries: 350 seeds for each of seven canonical
conditions (`No Finding`, pneumothorax, pneumonia, consolidation, effusion, mass, nodule). Exact
pixel hashes, registered normalized RMSE/SSIM, and pinned VAE-latent similarity are compared against
both private train and attack holdout. Near-copy thresholds and uniqueness margins must first be
calibrated from transformed positives and at least 100,000 different-patient public-development
negative pairs. Ordinary radiograph similarity is not called reconstruction, and this bounded
query test is not called exhaustive.

SecMI is motivated by diffusion-specific step-wise estimation error, while the extraction test
follows a generate-and-filter threat model. Primary references are the
[SecMI paper](https://proceedings.mlr.press/v202/duan23b.html), its
[official LDM implementation](https://github.com/jinhaoduan/SecMI-LDM), and
[Carlini et al.'s diffusion extraction study](https://www.usenix.org/conference/usenixsecurity23/presentation/carlini).

## 6. Secure randomness boundary

The mathematical table is valid for the ideal registered sampled-Gaussian mechanism. Current
PyTorch research runs may use a conventional private-seeded PRNG only with `RESEARCH_ONLY`
wording. Opacus recommends disabling secure RNG for experimentation and retraining from scratch
with a secure RNG for a production-grade run; the official `torchcsprng` package is not installed
and its documented compatibility does not cover this Python 3.11/PyTorch 2.6 environment. See the
[Opacus secure-RNG FAQ](https://github.com/meta-pytorch/opacus/blob/main/docs/faq.md) and
[Google DP accounting event model](https://github.com/google/differential-privacy/blob/main/python/dp_accounting/docs/dp_event.rst).

Therefore no research-PRNG result may produce a `DIRECT` or `GROUP/CONVERT` release statement. A
registered, performance-tested secure sampling/Gaussian backend and a fresh full rerun remain
mandatory before a formal release package.

## 7. Verification and current boundary

- Builder report SHA-256:
  `265FB172B6749C32248D5159E2414598C9ACEF29F2443AD426BED1F0639886EC`.
- Accounting CSV SHA-256:
  `D7523DCFAB0DBCB54A359EAC802E63315A02D0467DDEDEC948D8DB239932629C`.
- Member/nonmember cohort SHA-256:
  `F0F5A802090BC42BA268D9BB79E49B3BC2A4501FF4D3EE24739565EEBF26B3F5` /
  `A71010809B400C9E7C1041953AD83569DBF6C55E79A9001EFC85F49AE3E018F2`.
- Independent verification SHA-256:
  `9CAB4DB57B83985A90BC2265D9FA2BBEDC122B96B3418919CB6A8B2684472FC4`.
- Five fast contract tests pass.
- All new persistent reports total only 396,155 bytes; no resized corpus, latent cache, gradient
  tensor bank, or checkpoint was stored.
- The observed C: free-space decrease is not attributable to project output: no recent project,
  `%TEMP%`, or Hugging Face cache file above 100 MiB was created, while the system-managed Windows
  pagefile is currently allocated 45,957 MiB. It was not modified or deleted.
- All seven active LaTeX source hashes still exactly match the preserved baseline; no PDF was
  rebuilt.

The next gate is an executable DP-trainer mechanism conformance smoke: synthetic unit tests,
empty-Poisson-batch behavior, per-image versus per-patient clipping/aggregation, noise scaling, and
runtime event-trace/accountant agreement. Private generator optimization remains blocked until that
gate passes and is reported.
