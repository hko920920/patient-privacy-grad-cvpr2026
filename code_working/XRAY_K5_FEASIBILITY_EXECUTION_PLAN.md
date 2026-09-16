# NIH CXR14 K5 full-feasibility execution plan

Date: 2026-09-03  
Protocol decision: `PASS_PROTOCOL_INDEPENDENTLY_VERIFIED_FULL_K5_NOT_STARTED`  
Full K5 optimizer execution: `NOT_STARTED`  
Formal release: `BLOCKED_RESEARCH_PRNG`

## 1. Question and role of this stage

The experiment models one central institution fine-tuning a shared chest-radiograph generator from
multiple patients' images. K5 is a bounded, one-seed feasibility tier. It asks whether the pinned
SD 2.1 base can adapt to the radiograph domain and whether the frozen image-DP and patient-DP
mechanisms can complete without catastrophic utility collapse. It is not the confirmatory K10
study, confidential-hospital-data evidence, clinical validation, or release authorization.

K5 will not run membership, reconstruction, or extraction attacks. Those remain frozen for K10 so
that feasibility outcomes cannot tune the attack. K5 also cannot remove or replace an unfavorable
K10 arm.

## 2. Fixed data, model, and four-arm matrix

- Data: K5 `private_train`, 18,393 PA chest radiographs from 8,476 patients, maximum five images per
  patient. NIH ChestXray14 is public but is handled as a restricted operational proxy.
- Model: `Manojb/stable-diffusion-2-1-base` at revision
  `0094d483a120f3f33dafbd187ea4aa60d10de75c`, P256, frozen fp16 base, fp32 rank-8 attention LoRA
  with 1,659,904 trainable parameters.
- Optimizer: AdamW, learning rate `1e-4`, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `0.01`,
  constant schedule, no warmup, exactly 4,000 updates.
- Order: `M0 -> M1-I8 -> M1-G8 -> M2-P8`. Every arm starts from the same freshly initialized LoRA
  digest. There is one seed only.

`M0` is the non-DP comparator. It uses a fresh private-seeded permutation each epoch, consecutive
batches of eight, drops only the last incomplete epoch batch, and therefore has 32,000 image
exposures.

| Arm | Unit and expected batch | Clip norm | Noise multiplier | Frozen interpretation |
|---|---:|---:|---:|---|
| M1-I8 | image, 8 | 0.2844870061 | 0.4007042919 | image `(8,1e-5)`; patient conversion blocked |
| M1-G8 | image, 8 | 0.2844870061 | 0.8350657830 | image `(1.6,1.32654e-8)`; ideal K5 group conversion to patient `(8,1e-5)` |
| M2-P8 | patient, 4 | 0.1997973089 | 0.4044571458 | native patient `(8,1e-5)` |

M1-I8 and M1-G8 replay the same image-Poisson schedule and diffusion draws but use independent DP
Gaussian streams. M2 Poisson-samples patients, samples at most four images uniformly without
replacement inside a patient, averages before one patient clip, and gives every patient one outer
weight. All DP arms divide by the fixed expected batch. An empty sample is a noise-only update and
is never resampled.

## 3. Checkpoints, randomness, and failure handling

LoRA-only restricted checkpoints are scheduled at steps 1,000, 2,000, and 4,000. The first two
produce only 14 fixed diagnostic images per arm and cannot select, stop, or release a model. Only
step 4,000 enters quantitative evaluation. No resized second corpus, latent/text bank, raw-image
cache, or per-unit gradient bank is persisted.

A fresh 256-bit operating-system root is domain-separated for sampling, diffusion, and Gaussian
streams. The root and exact RNG/optimizer state may be stored only inside a Windows DPAPI
`CurrentUser` encrypted envelope. Envelopes are written atomically at step 0 and every 250 steps;
the current and one previous version are retained. Losing or corrupting the envelope cannot be
repaired with a new arm seed: the entire four-arm matrix must restart under a new run ID while the
failure is retained.

Before the full run, an exact uninterrupted-four-step versus two-plus-resume-plus-two test must
match adapter, optimizer, RNG, and trace for M0 and all three DP semantics. The DPAPI in-memory
round trip has passed, but this exact restart test has not yet been implemented.

## 4. Fixed generation and reference sets

The untouched base `B0` is generated and hashed before the first full K5 optimizer update. B0 and
all four trained arms use the same 448 public prompt/seed tasks: 64 each for no finding,
pneumothorax, pneumonia, consolidation, pleural effusion, mass opacity, and nodule opacity. The
sampler is DDIM, 50 steps, guidance 7.5, batch four. Final output is 2,240 images. Public seeds are
derived deterministically from a fixed salt.

The real comparison set contains 448 globally patient-distinct K5 `public_development` images:
64 pneumothorax, 128 pneumonia/consolidation, 64 pleural effusion, 128 mass/nodule, and 64 no
finding. Selection is scarcity-first and then SHA-256 ordered inside each stratum. Identifiers stay
local; they are excluded from the public report.

## 5. Fixed evaluation and decisions

Following the current chest-radiograph evaluation evidence in
[CheXGenBench](https://arxiv.org/abs/2505.10496), the primary small-sample fidelity measure is
RAD-DINO KID. PRDC (`k=5`), descriptive RAD-DINO Fréchet distance, effective rank, and
within-condition distance supplement it. BioViL-T prompt-image cosine measures weak-label
alignment. Inception-only FID is prohibited.

RAD-DINO itself was pretrained using all NIH ChestXray14 images. It is therefore only a fixed
comparative feature screen, not independent clinical validation or privacy evidence. BioViL-T must
first pass a fixed real-image matched-prompt versus patient-distinct derangement check. Neither
encoder establishes diagnostic correctness.

`M0_DOMAIN_ADAPTATION_PASS` requires all four conditions:

1. all pixel sanity checks pass;
2. M0 RAD-DINO KID is better than B0 and the upper paired stratified-bootstrap 95% bound is below 0;
3. M0 BioViL-T alignment is better than B0 and the lower bound is above 0;
4. M0 RAD-DINO coverage is strictly greater than B0.

A DP arm is labeled `COLLAPSED` only if pixel sanity fails or its coverage/effective rank is at most
10% of M0. Otherwise its complete privacy-utility degradation is reported without inventing a
pass/fail ranking. A poor DP result is evidence and does not authorize an arm substitution. K10
progression requires execution integrity, evaluator/restart PASS, and M0 domain adaptation PASS.
P512 is not automatically activated.

K5 has no downstream diagnostic classifier: 448 generated images and one seed are insufficient for
a credible augmentation claim. Patient-held-out augmentation utility remains a preregistered K10
task.

## 6. Resource and disclosure boundary

The observed four-step loops project about 11.92 hours for the three DP arms alone. Including M0,
generation, checkpoints, evaluator work, and margin, planning is 16--21 hours across resumable
sessions, not a promised duration. At least 15 GiB must be free before each arm; the current design
allows 2.5 GiB of new cache/artifacts and avoids duplicate image corpora. If scientifically needed
storage exceeds that estimate, the measured reason is reported before changing retention.

M1 aborts without resampling above 64 realized images in a step; the 4,000-step union bound is
approximately `8.52e-33`. M2 aborts above 32 patients or 128 raw images; its union bound is
approximately `6.72e-16`. Non-finite state, hash drift, corrupt resume state, trace mismatch, or low
disk are also fail-closed events.

Public outputs exclude identifiers, realized sample counts/empty locations, losses, norms,
gradients, updates, optimizer state, private seeds, and resume envelopes. Research PyTorch RNG
means no result can authorize formal model release; a registered secure path and fresh from-scratch
training would still be required.

## 7. Frozen artifacts and present stopping point

- Protocol SHA-256: `2234B3BA8701565B768A11AB5196B2F696788900D07254833997D7823489E9DA`.
- Generation-task SHA-256: `B0379A2A1EBDFB8758AAD865D6F709D803C7632D2697A7BB9A5DFD9C23F2124B`.
- Local-reference SHA-256: `99972FC6BAB402F620D2F6B142348CF6A3F16B60BFD63F41EB877FFDDAAA898E`.
- Independent report SHA-256: `49E715F715B5634B4071EA843AF8FC4A85666DD2E9EBC51D3D51DE1D89472A1B`.
- Builder source SHA-256: `F4B85D379369E0D79DC3B3B9B41F075AB15C42AD1C5C4E18ACA26CC7FB4D133B`.
- Independent verifier source SHA-256:
  `00CB027258F7F7CF3F308AE0EC6D89643D1C5EF2BFDC7A49595BACF1327A976E`.

The verifier independently reconstructed all 448 generation tasks and all 448 patient-distinct
references, linked the three DP settings to the upstream accountant, recomputed both resource
bounds, and passed the in-memory DPAPI check. At verification time the RTX 3070 was idle and C: had
43.26 GiB free. No model was loaded, no image was generated, and no full K5 optimizer update was
performed by this protocol gate.

The first verifier invocation stopped before creating a report because the independently evaluated
M1 tail probability differed from the builder by `2.21e-44` absolute (`2.60e-12` relative) at a
value near `8.52e-33`, while the initial comparison tolerance was `1e-13` relative. The verifier
tolerance was changed to `1e-10` relative and rerun. No protocol value, resource limit, or decision
threshold changed. This was a numerical-comparison correction, not outcome-based experiment tuning.
The 15 existing mechanism/private-runtime tests then passed, and all seven active thesis/abstract
LaTeX hashes remained equal to their preserved baseline.

This paragraph records the stopping point when the execution contract was first frozen. The current
post-preflight stopping point is recorded in Section 8. The 4,000-step matrix must not start before
all gates pass and are reported.

## 8. Post-freeze evaluator and encrypted-resume gate

The previously required evaluator and restart gates have now passed both primary execution and an
independent recomputation. No full K5 optimizer run was started.

- The evaluator protocol froze the mixed-stratum weak-label mapping, an 89-position
  frequency-preserving patient-and-condition derangement, five-stratum 2,000-sample bootstrap,
  exact model revisions, preprocessing, and disabled matrix/cuDNN TF32 before encoding.
- All 448 references decoded. RAD-DINO produced finite `[448,768]` features with effective rank
  `184.65758447` and replay drift zero. BioViL-T produced finite unit-normalized `[448,128]` image
  and `[7,128]` text features with replay drift zero.
- BioViL-T matched-minus-deranged cosine was `0.06058422`, with frozen 95% CI
  `[0.03795662,0.08322361]`, so the overall weak-label alignment screen passed. Consolidation,
  no finding, pleural effusion, and pneumonia nevertheless had negative descriptive condition
  differences. The encoder therefore remains an overall screen only; condition-level clinical
  validity and diagnostic correctness remain prohibited claims.
- An initial independent encoder replay stopped on a roughly `4.27e-5` BioViL-T aggregate
  discrepancy because the verifier had disabled matrix TF32 but omitted explicit cuDNN TF32
  disablement. The setting was corrected and all 448 images were recomputed from the beginning.
  No metric, mapping, or threshold changed; the independent aggregates then matched exactly.
- DPAPI `CurrentUser` ciphertext-only atomic current/previous rotation passed tamper and stale-new
  fail-closed tests. For M0 and all three DP semantics, uninterrupted four-step execution matched
  save/two-step/rotate/delete/decrypt/restore/two-step execution exactly across adapter, full AdamW,
  explicit and global RNG, sampler, trace/head, committed step, and frozen digests.
- The resume evidence uses compact synthetic training semantics. It does not yet establish full
  SD 2.1 checkpoint equivalence, privacy, utility, or release eligibility.
- The combined mechanism, private-runtime, evaluator, and resume regression suite passed 26/26.

Primary evaluator report SHA-256:
`914002B52C2EF48C551FFD9A5CD0260BA7708EA6492F5296AEF98BF6D0B4F5B5`.
Independent evaluator report SHA-256:
`A55672E5D5C68491C75AAE4770F1C13A1F1B43BCCD721E7F732BF3C8D9560CAD`.
Primary resume report SHA-256:
`89A08E9B1AAC248A3DF4F4935C82354D49D28AFC0DD8BE632DE6A80C394FCBFF`.
Independent resume report SHA-256:
`720D7A9DDE7A9DD862AE84B4053DBF0F704B8100CE0C7166D28B56EFE76BA463`.

The next bounded stage is to connect these frozen contracts to the actual full runner, independently
freeze and verify the source/environment/launch command and per-arm time/storage plan, report that
gate, and only then decide whether to launch M0.

## 9. Actual full-training runner gate

That bounded stage is now complete and independently **PASS**. The frozen runner implements the
four-arm order, exact M1 matching, native patient-unit M2 loss/clipping, noise-only empty DP steps,
250-step encrypted current/previous resume rotation, and restricted 1,000/2,000/4,000 LoRA
snapshots. Its default command is read-only; mutating commands require both an explicit start flag
and the exact independently issued launch-authority hash.

The step budget was also recomputed rather than inherited by convention. M0/M1 each have 32,000
expected image exposures (`1.7397923123` per K5 image), while M2 has 16,000 expected patient-unit
exposures (`1.8876828693` per K5 patient). This is retained as a reasonable one-seed rank-8 LoRA
feasibility budget, not a confirmatory convergence or superiority claim.

On 16 public-development images from eight two-image patients, all four real SD 2.1 arms passed
exact four-step versus encrypted two-plus-resume-plus-two equality across 13 state fields per arm.
The step-2 encrypted state was about 20.24 MB and included the complete real LoRA and AdamW state.
The first full-gate verifier correctly stopped on a Windows WDDM `[N/A]` process-memory parse; the
idle rule was changed to aggregate free-memory/utilization/temperature thresholds, the actual test
was re-frozen and rerun as v1_002, and the first failed gate was retained as superseded evidence.

Final launch-authority SHA-256:
`A1A07C51FBFFF6462A6248D45E77D6C25A8F9041A7F5B675092EC8A0BF2551F2`.
Independent verification SHA-256:
`FB6EA9674FC5613B8FB3F7E750B6954F8C9535EDF621D042EAD9D9A201800BAB`.

No full run root exists and no B0 or M0 work has started. The next separately reported stage is the
fixed 448-image B0 generation/hash gate. Details are in `XRAY_K5_FULL_RUNNER_GATE.md`.

## 10. B0 generation gate

The B0 stage is now independently **PASS** for execution integrity and ordering before the full K5
matrix. The separate K5 four-step `RESEARCH_ONLY` dry-run predates B0, so no broader claim that B0
preceded every private-role optimizer update is allowed. Protocol
v1_002 was frozen before any image and has SHA-256
`49C18694F740053048971C88286B55C256F28FC8BB223B94A8F1132FA0686324`. All 448 fixed images were
generated in 467.17 seconds with no reroll, rejection, or replacement. Decode rate was 1.0 and the
duplicate, low-contrast, and saturation fractions were all 0.0. The restricted inventory SHA-256
is `C385BA3A0C642E9B3A62B0D8FFF11232BBCF842FDC25102C5A26EFC89DB5C984`.

The independent verifier reconstructed every task, rehashed and decoded every PNG, and regenerated
the first and last fixed four-image batches. All eight sentinel hashes matched. Independent report
SHA-256 is `A1A157A996ABBA631D4289DC18B490C894928815E4F6931DB756D9A6A1E12520`; the manifest accepted by
the full runner has SHA-256
`6CB11A45A5B99D61E5B44F7C9463A7832CF5439AE0DF7D497CF8CF752CC94962`.

The fixed 35-image visual grid nevertheless flags every reviewed B0 image as non-conventional PA
chest radiography or grossly implausible geometry. This is kept as the expected off-domain B0
comparator result, not relabeled as a quality PASS. Matrix initialization, M0, and all full-matrix
optimizer steps remain absent. The earlier separately reported four-step dry-run retained no
checkpoint. The next bounded stage is initialization plus M0 only, followed by
the already frozen domain-adaptation evaluation. A failed M0 domain gate stops scientific
progression to the DP arms. Details are in `XRAY_K5_B0_GENERATION_GATE.md`.
