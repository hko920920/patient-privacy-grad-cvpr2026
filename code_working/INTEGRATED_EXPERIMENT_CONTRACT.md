# Integrated Medical DP–Audit Experiment Contract

- Drafted: 2026-09-02 (Asia/Seoul)
- Current status: **B0 execution integrity PASS; temporal claim narrowed to before the full K5 matrix;
  matrix and M0 not started**
- Core choice: fine-tune one public pretrained latent diffusion model; do not train the dissertation
  generator from scratch

The current-state authority is `../CURRENT_STATUS.md`. Sections 1--14 preserve the ISIC-era contract and
Sections 15 onward record the X-ray supersession. Later gate results supersede earlier present-tense
`next`/`not started` sentences without deleting the historical record.

## Step-3 freeze decision

The following choices were approved on 2026-09-02 and are no longer candidates to be selected from
observed attack or utility results.

- Central, patient-linked 2D medical-image generation is the primary scenario; it is not presented
  as federated learning or as proof that confidential clinical data were handled.
- At this Step-3 freeze, ISIC 2020 v2 was the main public proxy. Section 15 later superseded the core
  modality with NIH ChestXray14 while retaining ISIC only as a separately trained extension.
- Split patients before deriving image partitions and reuse the identical patient partition for all
  matched training arms.
- Fine-tune a common latent-diffusion base with LoRA. Compare B0/M0/M1/M2 and the registered
  no-noise mechanism controls; do not replace this with from-scratch training.
- Start at 256x256 for feasibility. A 512x512 confirmatory setting is an extension only if medical
  utility, VRAM and runtime checks justify it before the confirmatory runs.
- K10 is the controlled main condition because patient contribution is the independent privacy
  variable, not because the 32,701-image source population is inadequate. Generation quality is a
  mandatory utility gate, while the primary contribution is the privacy-unit-to-provenance chain.

The implementation base is now conditionally frozen in Section 5 for skeleton/pilot work. Exact
split/cap manifests were frozen in Step 5. Privacy budgets, patient-membership protocol and
confirmatory compute were open at this historical freeze, not parameters that could be chosen after
seeing the desired model ordering. Later sections close several of those gates. “Frozen” here means
the experimental architecture was preregistered; it does not mean that an integrated system or result
already exists.

## 1. Exact objective

The experiment separates two effects that must never be merged in the paper.

1. **Training-time privacy effect.** Non-DP, image-DP, and native patient-DP fine-tuning produce
   different formal privacy guarantees, empirical attack resistance, utility, and cost.
2. **Post-execution audit effect.** Auditable Privacy does not improve a model's privacy. It checks
   whether the evidence from an already completed run permits a requested patient-level claim as
   `DIRECT`, `GROUP/CONVERT`, or a blocking decision.

The intended result is therefore not “the auditor always avoids retraining.” It is:

> identify the evidence-validated region in which an existing image-DP model can be reused under a
> nonvacuous patient-level conversion, and the region in which the audit correctly forces native
> patient-DP retraining or narrower wording.

An accepted conversion can save retraining cost. A blocked conversion demonstrates prevention of
an invalid or operationally unacceptable release, not computational savings.

## 2. Claims and non-claims

### Claims to test

- DP fine-tuning reduces empirical image/patient membership and memorization risk relative to its
  matched non-DP control, subject to confidence intervals and attack power.
- Native patient-DP gives a direct patient-unit guarantee; image-DP does not reuse the same numeric
  epsilon for a patient claim.
- A public contribution cap and complete patient-to-image mapping can support a recomputed group
  conversion when the converted parameters remain usable.
- The audit detects missing/mismatched units, sampler/accountant mismatch, private adaptive
  selection, vacuous conversion, runtime mismatch, and wrong model digest without an unsafe
  positive statement.
- In accepted cases, evidence construction and audit verification cost much less than a new
  patient-DP fine-tuning run.

### Claims prohibited

- Auditing or receipt generation makes an already trained model more private.
- Failure of a finite attack suite proves that the model cannot leak training data.
- A numerically equal image-level and patient-level epsilon represents equal privacy.
- Every image-DP model can be converted into a useful patient-DP claim.
- Public ISIC data establish handling of confidential clinical data.
- Synthetic-image quality or watermark validity establishes diagnostic truth or clinical safety.

## 3. Actors and decision boundary

| Actor | Responsibility | Outside its authority |
|---|---|---|
| Patients/data subjects | Define the requested protected unit through one patient's complete contribution | Do not operate federated clients in this central scenario |
| Central curator/trainer | Fix data policy, fine-tune models, emit runtime evidence | Cannot self-declare a patient claim without audit evidence |
| Privacy accountant | Recompute epsilon/delta for the executed mechanism and its accounting unit | Does not validate patient/image mapping or model identity |
| Auditable Privacy validator | Validate unit path, mapping, mechanism, runtime, conversion, and claim wording | Does not improve weights or attest malicious execution |
| Release authority | Apply a declared operational privacy budget to an audit-valid result | Cannot override a blocked or unverified audit |
| Model publisher | Publish the selected exact model and later signed model-bound receipt | Cannot substitute a different checkpoint |
| External relying party | Verify the evidence package and later PP-Mark linkage | Does not infer `Real` from missing evidence |
| Adversary/auditor | Run image- and patient-level privacy attacks and substitution tests | Empirical attack failure is not a formal proof |

## 4. Dataset and units

The fixed ISIC 2020 v2 release is the only split/mapping authority.

```text
raw privacy unit       = patient_id
raw lower-level unit   = lesion_id / image_name
generated train record = one image with its fixed condition
image-DP account unit  = image
patient-DP account unit= patient
requested release unit = patient
```

The official 425 duplicate pairs are resolved before any cap or split. For each pair, retain the
lexicographically smaller image ID. This leaves 32,701 images and all 2,056 patients.

### Contribution and split policy

- Pilot cap: `K=5`, giving 9,495 images.
- Main confirmatory cap: `K=10`, giving 15,924 images before patient partitioning.
- Unit-sensitivity ablation: `K=2`, giving 4,112 images.
- Full 32,701-image pool remains an acquisition/expansion reserve for a justified
  uncapped/non-private upper-bound run or a preregistered scale follow-up, but it is not a matched
  comparator for a capped model. Expansion must be triggered by feasibility, statistical-power or
  rare-class coverage criteria, not by failure to obtain a preferred model ordering.
- Within each patient, rank images by a precommitted public SHA-256 rule over dataset release ID,
  split seed, patient ID, and image ID; retain the first `K`. Do not select by diagnosis, attack
  score, observed group maximum, or later model behavior.
- Assign patients to partitions by a separate precommitted public hash. One patient cannot cross
  private-train, public-development, privacy-attack holdout, or final-test partitions.
- Publish aggregate partition counts and mapping digests, not a new patient-level disclosure.

Twenty public hash seeds were checked diagnostically. Label-independent capping retained, on
average, about 304 malignant images from 253 malignant-bearing patients at `K=5`, and about 418
malignant images from 329 patients at `K=10`. The exact confirmatory seed and counts must be frozen
before model training.

Private-data-dependent checkpoint selection is prohibited. Training steps and release-checkpoint
selection must be fixed from the smoke/public-development phase or separately accounted.

## 5. Model choice and contamination control

### Locked choice

- Use parameter-efficient fine-tuning, not from-scratch diffusion training.
- Reopened Step-4 implementation base: `Manojb/stable-diffusion-2-1-base` at exact revision
  `0094d483a120f3f33dafbd187ea4aa60d10de75c`, using fp16 safetensors. It is a third-party,
  hash-pinned mirror of the now-unavailable named Stability AI repository and is selected for the
  skeleton and real-data pilot, not yet as confirmatory evidence. Verified SD v1.4 remains fallback.
- Fine-tune a fixed LoRA target set; keep VAE, text encoder, base UNet weights, scheduler, tokenizer,
  and preprocessing fixed unless the contract is revised before confirmatory runs.

The base is not accepted merely by model name. Bind exact file digests, revision, license,
configuration, scheduler, tokenizer, and trainable-parameter manifest. The active snapshot manifest
and diagnostic are recorded in `SD21_BASE_MODEL_GATE_DECISION.md`; the earlier SD v1.4 record is
preserved in `BASE_MODEL_GATE_DECISION.md` as a superseded fallback gate.

The selected model retains the 512-pixel 4x64x64 latent interface used by the historical PP-Mark
path. The integration adapter now pins a non-SDXL fp16 variant and exact repository revision.
Nevertheless all thresholds must be recalibrated after medical LoRA fine-tuning; historical
SD2.1/SDXL thresholds are not transferred as medical-model results.

### Pretraining-overlap control

Because a web-scale base may already have seen publicly hosted ISIC images, every privacy result
must include:

1. an untouched-base control;
2. attacks and nearest-neighbor checks against that base;
3. a documented overlap/contamination audit where available;
4. wording restricted to incremental fine-tuning membership if overlap cannot be ruled out.

If material exact/near-duplicate contamination prevents interpretation, switch to a base with a
known pretraining corpus or narrow the claim. Do not conceal the confound for PP-Mark convenience.

## 6. Training arms

### Released-model arms

| ID | Model | Sampling/accounting unit | Purpose |
|---|---|---|---|
| B0 | Untouched pretrained base | none | Pretraining-overlap and attack floor |
| M0 | Non-DP LoRA fine-tune | image training records | Utility and memorization reference |
| M1 | Image-DP LoRA fine-tune | image | Valid image claim; candidate group conversion for patient wording |
| M2 | Native patient-DP LoRA fine-tune | patient | Direct patient-level reference |

### Mechanism-attribution ablations

- A0: image clipping with no DP noise.
- A1: patient aggregation/clipping with no DP noise.
- A2: matched patient-sampled non-private update where needed to separate sampling/aggregation from
  noise.

The main models share the same base checkpoint, LoRA architecture/rank, capped data, patient split,
image preprocessing, condition vocabulary, nominal step budget, and evaluation set. Sampling and
aggregation cannot be falsely called identical between image-DP and patient-DP; those differences
are mechanism fields and are explicitly registered. Each DP arm is compared with its closest
matched no-noise control.

Research pilots may use a research PRNG but can produce only `RESEARCH_ONLY`/diagnostic evidence.
Confirmatory release claims require a registered secure DP-noise path, private noise randomness,
and a matched runtime trace. Public experiment seeds and private DP-noise randomness are separate.

## 7. Privacy-budget and conversion grid

The final epsilon/delta values will be preregistered after accountant and utility feasibility, not
chosen after observing attacks. The pilot grid is:

```text
K in {2, 5, 10}
image epsilon in {0.5, 1, 2, 4}
image delta candidate in {1e-7, 1e-6}
native patient epsilon targets matched to selected converted patient bounds
```

For the current built-in black-box conversion,

```text
epsilon_patient = K * epsilon_image
delta_patient   = delta_image * sum(exp(i * epsilon_image), i=0,...,K-1).
```

Examples at `delta_image=1e-7` show the intended frontier:

| K | image epsilon | converted patient epsilon | converted patient delta | Audit math status |
|---:|---:|---:|---:|---|
| 2 | 4 | 8 | 5.56e-6 | nonvacuous; release policy still decides |
| 5 | 1 | 5 | 8.58e-6 | nonvacuous; principal reuse candidate |
| 5 | 2 | 10 | 3.45e-4 | nonvacuous but likely operationally weak |
| 5 | 4 | 20 | 0.905 | formally below 1 but operationally unusable |
| 10 | 1 | 10 | 1.28e-3 | nonvacuous but weak |
| 10 | 2 | 20 | 7.59 | `BLOCKED_VACUOUS` |

`ALLOWED` by the claim validator means evidence-valid and nonvacuous under its contract. It does not
mean an application has accepted the privacy budget. The release authority must separately apply a
preregistered maximum patient epsilon/delta. This second policy must not be invented after seeing
model quality.

A tighter RDP/PLD or structured conversion may be evaluated only as a registered, independently
recomputed attachment. It is a separate method result, not a hidden replacement for the built-in
group bound.

## 8. Evaluation matrix

### Formal privacy

- Recomputed `(epsilon, delta)` with the unit printed beside every number.
- Accountant/sampler/mechanism registry match.
- Converted patient parameters and contribution `K` where applicable.
- Direct versus group path and final release-policy decision.
- Composition for every private preprocessing or selection mechanism, if any.

### Empirical privacy

- Image-membership attack: ROC-AUC, attack advantage, TPR at fixed low FPR, confidence interval.
- Patient-membership attack: aggregate all records belonging to one patient before the decision;
  use the same metrics.
- Memorization/extraction: nearest-neighbor and copy/exposure tests against train and held-out data.
- Controlled canary audit: sensitivity diagnostic reported separately from natural-data attacks.
- Reconstruction is claimed only if a defined threat model and a working attack are implemented.

“Not broken” means only that the preregistered attack suite did not exceed its success threshold,
with an upper confidence bound. Formal DP, not attack failure, is the primary privacy guarantee.

### Utility and medical fidelity

- FID/KID plus precision/recall or density/coverage using a medical-domain feature extractor where
  justified; Inception-only scores are not sufficient.
- Downstream melanoma classifier trained on synthetic data and evaluated only on held-out real
  patients: ROC-AUC, sensitivity at fixed specificity, balanced accuracy, calibration, and class
  confidence intervals.
- Per-class diversity and duplicate/nearest-neighbor rate, with special reporting for the rare
  malignant class.
- PP-Mark distortion and shortcut-learning tests are added only after the unmarked generator
  baselines are stable.

### Replication

- One explicitly excluded feasibility pilot.
- At least three independently trained confirmatory seeds for the selected main settings.
- Bootstrap or patient-level confidence intervals and paired comparisons on a common test set.
- No claim is selected because a single seed produced the desired ordering.

## 9. Audit/retrain efficiency experiment

The fair comparison begins after an image-DP checkpoint already exists.

### Reuse path

```text
existing image-DP model
  -> construct/bind mapping, runtime and accountant evidence
  -> Auditable Privacy group conversion
  -> application privacy-budget check
  -> model-bound receipt
```

### Retrain path

```text
same public base + same capped patient cohort
  -> native patient sampling/aggregation/clipping/noising
  -> patient-DP fine-tuning
  -> direct audit
  -> model-bound receipt
```

Measure wall-clock time, GPU-hours, peak VRAM, optimizer steps, energy when reliably available,
checkpoint bytes, evidence bytes, audit construction time, verification time, and failure/retry
cost. Report the existing Auditable Privacy benchmark only as historical context; remeasure every
number on the medical generator and its large checkpoint.

Cost saving is claimed only when the converted patient claim is both evidence-valid and accepted by
the predeclared release policy. When conversion is blocked, report the audit's value as prevention
of an invalid release and the resulting need for direct patient-DP training.

## 10. Evidence and model identity

Every run must bind:

- fixed dataset, deduplication, cap and patient-split manifest digests;
- base model and trainable-adapter manifests;
- source, dependency and mechanism-registry digests;
- sampling, clipping, noising, aggregation, optimizer and schedule fields;
- exact accountant inputs and independently recomputed outputs;
- matched runtime/batch trace and secure-RNG assurance;
- canonical exact-checkpoint digest;
- model-selection status and public/private development-data policy.

The existing `released_model_sha256` field is necessary but not a signature and is insufficient for
a multi-file diffusion directory by itself. Later implementation must canonicalize the base plus
adapter/config manifest and sign the audit decision bound to that exact identity. PP-Mark then binds
released synthetic images to the signed model/receipt context; it does not prove DP.

## 11. Success, null, and failure interpretations

### Strong integrated result

- DP arms have valid unit-labelled accountant evidence.
- Patient-DP reduces patient-level attack success relative to matched controls while retaining
  useful downstream performance.
- At least one preregistered image-DP/K setting lies in the accepted reuse region.
- Weaker settings are correctly converted or blocked with zero unsafe-positive decisions.
- Audit/verification overhead is negligible relative to a confirmed patient-DP retraining cost.
- Checkpoint substitution and evidence tampering fail closed.

### Still publishable but narrower result

- Natural attacks are near chance for every arm, but formal unit differences, canary sensitivity,
  utility trade-offs, and audit correctness remain clear. Do not claim empirical attack superiority.
- Every practically relevant group conversion is rejected. The result becomes an evidence-backed
  demonstration that post-hoc wording cannot replace native patient-DP at realistic contribution
  sizes.

### Failure requiring redesign

- Patient-DP utility collapses at every useful budget.
- The selected base materially contaminates the ISIC membership experiment.
- Secure confirmatory execution cannot be bound to the reported accountant/checkpoint.
- Patient-level attacks lack enough subjects or statistical power for their stated conclusion.
- PP-Mark cannot be adapted without changing the released generator semantics beyond the contract.

In those cases, expand compute, change the base under a new frozen manifest, or add a larger
patient-rich dataset. Do not tune the claim after observing failure.

## 12. Freeze status and next gates

### Approved core gates

1. **PASS:** parameter-efficient fine-tuning and the B0/M0/M1/M2 model matrix.
2. **PASS:** public hash-based patient split/cap policy, `K=5` pilot, `K=10` main, and `K=2`
   unit-sensitivity ablation.
3. **PASS:** 256x256 feasibility first; 512x512 only as a separately justified confirmatory
   extension.
4. **PASS_FOR_SKELETON_AND_PILOT:** exact SD 2.1 mirror revision selected after provenance,
   license, artifact-hash, architecture, RTX 3070 and PP-Mark diagnostics. The accessible repository
   is not labeled official; pretraining overlap remains an explicit controlled risk. SD v1.4 is the
   verified fallback only.

### Step-5 gates now passed

1. **PASS:** exact patient split and K2/K5/K10 manifests were materialized, hashed and independently
   regenerated before model training.
2. **PASS_FOR_INTERFACE:** the real-image loader, exact SD-v1.4 VAE encoding, and exact-revision
   PP-Mark DDIM loading originally passed; the same PP-Mark gate now also passes on the selected
   exact SD 2.1 artifact. This is not calibration, robustness, utility or privacy evidence.

### Gates that must still pass before confirmatory training

1. Freeze the pilot epsilon/delta grid and operational patient-level release budget before DP
   attacks.
2. Freeze the patient-membership attack protocol, thresholds and minimum statistical power.
3. Classify the RTX 3070 as smoke/pilot/confirmatory-capable from measured traces; reserve a
   higher-memory GPU if confirmatory replication is not credible locally.
4. Reassess or replace the base if the full ISIC overlap audit finds material contamination.
5. Acquire the frozen pilot images and rerun PP-Mark calibration on the eventual exact medical
   checkpoint. Existing SD2.1/SDXL thresholds are not transferable.

## 13. Step-5 split and interface result

The public SHA-256 split seed was used once with label-independent 70/10/10/10 thresholds; no
alternative seed was selected from model or attack results. K10 contains the following exact
partition counts:

| Partition | Patients | Images | Malignant images |
|---|---:|---:|---:|
| private train | 1,415 | 10,848 | 276 |
| public development | 200 | 1,565 | 37 |
| privacy-attack holdout | 233 | 1,870 | 42 |
| final test | 208 | 1,641 | 46 |

K2 and K5 private-train contain 2,830 and 6,512 images respectively. The uncapped private-train
upper-bound contains 21,505 images. The manifest-lock digest is
`7234076A4D7AA50B261984C18C92D3D65BF19CFB7B0300D8BBD327C191864D11`.

Eight real images passed the manifest loader and exact SD-v1.4 VAE encoding. PP-Mark's non-SDXL
DDIM loader was extended to fail-closed exact revision/variant loading and updated for the current
Diffusers prompt interface. The reopened gate then hash-pinned the same SD 2.1 mirror historically
used by PP-Mark and passed 256/512 inference, rank-8 LoRA backward/update, and exact two-step
PP-Mark inversion on a real ISIC image. This removes the base-version mismatch but establishes only
structural and local execution compatibility. Medical-output calibration and robustness remain open.
The mirror provenance limitation is explicit in `SD21_BASE_MODEL_GATE_DECISION.md`.

The detailed record is `STEP5_SPLIT_LOADER_GATE.md`. No generator, DP mechanism, attack experiment,
receipt or marked release image has yet been produced.

## 14. Step-6 frozen K5 acquisition result

The exact K5 manifest was acquired before model training. All 9,495 expected dermoscopic JPEGs from
2,056 patient-disjoint subjects passed decoding and partition validation. The local corpus contains
6,224,399,586 bytes (5.796924 GiB); its ordered content-set SHA-256 is
`06F177B5C345121D3C5D876530768437A710C9C56DAD30F1225BD795C536E3F2`.
An independent offline pass reopened all files, found zero failures, and reproduced that digest and
the local inventory digest exactly.

The experiment's medical scope is now explicitly restricted to patient-linked dermoscopic
skin-lesion generation. A lesion is not automatically cancer: the K5 population contains 301
malignant and 9,194 non-malignant images. The privacy rationale is sensitive patient/cohort
membership plus possible model-mediated memorization or extraction across a patient's repeated
images. It is not a claim that every dermoscopic image directly identifies a person.

Because SIIM–ISIC 2020 is public, it is a reproducible proxy for the data structure of a restricted
clinical cohort. Formal DP results may demonstrate what the mechanism would guarantee under the
frozen adjacency and implementation; they must not be described as newly restoring confidentiality
to already released ISIC subjects. DP also does not replace raw-data access control, clinical
validation, model-bound receipts, or PP-Mark provenance.

The acquisition gate is `PASS_FOR_PILOT_DATA`; full evidence is in
`STEP6_K5_ACQUISITION_GATE.md`. No generator/DP training, attack, receipt, marked release, or active
LaTeX edit occurred in that gate. Its then-next mandatory gate was the mechanism/accountant,
privacy-budget, and attack-protocol freeze; later sections record the X-ray path.

## 15. Core-modality pivot: chest X-ray first

The user subsequently selected frontal chest X-ray as the first/core modality because the medical
and privacy setting is more immediately interpretable from the image itself. ISIC dermoscopy is
retained as a separate cross-domain extension; modalities are not pooled in one generator.

This section supersedes only the earlier dataset role and ISIC-specific scale as the primary
experiment. It does not supersede central LoRA fine-tuning, the B0/M0/M1/M2 arms, patient-disjoint
evaluation, image-versus-patient adjacency, K2/K5/K10 sensitivity, attack families, Auditable
Privacy, model-bound receipts, PP-Mark, or fail-closed release decisions. The X-ray-specific source,
view, patient cohort, partitions, and cap sizes below supersede the earlier ISIC primary scale.

NIH ChestXray14 is the exact metadata authority. All four official metadata/split files are byte and
hash locked. The source contains 112,120 images from 30,805 patients, and the frozen PA-only policy
leaves 67,310 images/28,868 patients. The official train/test lists are patient-disjoint.

The four frozen, overlapping PA evaluation strata are pneumothorax, pneumonia/consolidation, pleural
effusion, and mass/nodule, with `No Finding` only as a weak-label reference. `Mass`/`Nodule` is not
called cancer, and no X-ray label is treated as a definitive diagnosis. Cardiomegaly remains an
optional later stratum. Edema is deferred because its PA count is small and its AP concentration
risks a view shortcut. All labels remain in the local manifests. Section 16 now freezes the exact
weak-label prompt conditioning used by the pilot interface.

The target-enriched method cohort contains 13,926 patients, balanced 1:1 target/control within each
patient-disjoint partition. Nested image totals are K2=20,687 for privacy-unit sensitivity,
K5=31,451 for feasibility/pilot work, and K10=38,492 as the controlled main condition and maximum
contribution stress. Within-patient cap ranking is label-independent. The first target-enriched seed is frozen and the manifest lock is
`518D3288ACDF55BE355038BC693B3BCF7A8C44B8B3455FAC1DC493EE5E4E3ECE`.

This enrichment is not hospital prevalence. The full official PA-test census, 11,096 images from
2,647 patients, is frozen as an overlapping secondary distribution-sensitivity view. It cannot tune
the model or attacks and is not analyzed as an independent second test. The K10-plus-census image
union contains 42,423 unique images.

An ambiguous X-ray pilot cannot be silently replaced after looking at the desired attack ordering.
Quality, utility, attack-power, accounting, runtime and PP-Mark feasibility gates must be numerical
and frozen first. A failed/null pilot is retained; any return to dermoscopy is a transparent contract
amendment. Full decision: `XRAY_CORE_PIVOT_DECISION.md`.

All 12 source-locked archives have now been processed and exactly 42,423 PNGs/14,755 patients/
16.457515 GiB are materialized. The complete 112,120-name archive set matches official metadata,
and acquisition-time, offline regeneration, separately implemented full-file verification, and
PowerShell checks pass. The full content-set SHA-256 is
`E983A21B9B8558CE38F1AD5BA7C0F6BC51787CC7CBA739399ACE1976E2FB068E`. Native storage is 42,244
`L` plus 179 fully opaque grayscale-equivalent `RGBA` images; no raw conversion occurred. This is
source/acquisition evidence, not a utility or privacy result. Generator training, a DP run, receipt,
marked release, and active LaTeX edit have not occurred. Exact preprocessing and SD-2.1/PP-Mark
interface validation now pass as recorded in Section 16. Evidence: `XRAY_METADATA_MANIFEST_GATE.md`,
`XRAY_OFFICIAL_PNG_SMOKE_GATE.md`, and `XRAY_FULL_ACQUISITION_GATE.md`.

## 16. X-ray preprocessing and SD 2.1/PP-Mark interface result

The raw-to-model contract is now frozen for the pilot. Original 1024x1024 PA PNGs remain
unchanged. Native `L` is used directly; native `RGBA` is accepted only if `R=G=B` at every pixel
and alpha is 255, after which the shared grayscale channel is used. The full square field is
resized with Pillow 11.3.0 Lanczos, without crop or padding, then replicated to RGB and mapped as
float32 `x/127.5-1`. No augmentation, histogram transform, per-image standardization, or
private-population statistic is fitted. Resized inputs are produced on demand rather than cached
as a second full corpus.

`P256` remains the feasibility/pilot profile. `P512` passed the same executable interface but is a
confirmatory extension only if preregistered medical-utility and compute criteria justify it; a
preferred privacy-attack result cannot choose the resolution.

Prompt policy `nih-cxr14-weak-label-prompt/v1` conditions on all automatically mined NIH labels in
a fixed order. It excludes patient ID, age, sex, and the target-enrichment flag. `No Finding`
means no mined label rather than clinically normal, and mass/nodule are not relabeled as cancer.
All 4,831 K10 public-development records fit without tokenizer truncation.

Eight patient-distinct public-development cases were selected by a fixed stratum plus salted
SHA-256 rule, covering both native modes, `No Finding`, all four primary finding groups, and a
multilabel case. Both profiles reproduced preprocessed pixels and normalized tensors exactly. On
the exact pinned SD 2.1 revision, two real cases produced finite, replay-exact VAE latents at
`1x4x32x32` and `1x4x64x64`. The rank-8 attention-LoRA path produced finite replay-exact gradients
at both resolutions without constructing an optimizer or changing a parameter. PP-Mark consumed
the same committed pixels and produced finite replay-exact `4x32x32` and `4x64x64` DDIM latents.
Peak reserved memory was 2.0586 GiB for the LoRA gradient path and 2.9746 GiB for PP-Mark at P512.

An implementation-independent verifier rebuilt the eight-record selection and all 16 P256/P512
pixel/tensor commitments and checked the model/interface invariants. Main report SHA-256:
`B5C7826BEDB0E6584A5A5346FC70F7A43BE71C64E59062E66667AB1E430AE6F0`; independent report:
`7B03A7AEDD662F21DDF3796B70F84CF4823376981334C6CCD0B7645F2980C6B3`.

The status is `PASS_FOR_PILOT_INTERFACE`, not training or privacy evidence. It establishes neither
medical generation quality nor preservation of subtle findings at P256, patient-DP feasibility,
attack resistance, PP-Mark calibration/robustness, a receipt, or release authorization. At this
section's closure, the next gate was the DP mechanism/accountant/privacy-budget/attack-protocol
freeze before optimizer training; Section 17 records its completion. Full record:
`XRAY_SD21_PPMARK_INTERFACE_GATE.md`.

## 17. X-ray DP mechanism, budget, and attack-protocol freeze

This section supersedes the provisional budget candidates in Section 7 for the X-ray experiment.
It does not assert that DP training has occurred.

P256, rank-8 attention LoRA and a 4,000-step fixed final checkpoint are used. M1 independently
Poisson-samples an expected eight images per step, clips each image gradient at
`C_image=0.28448700606156724`, adds Gaussian noise to the sum, and divides by fixed expected batch
eight. M2 independently Poisson-samples an expected four patients per step, samples at most four
images inside each selected patient, averages them to one patient gradient, clips at
`C_patient=0.1997973088974048`, adds Gaussian noise, and divides by fixed expected patient batch
four. K10 M2 processes 8.023124 images per step in expectation, within 0.2891% of M1's image work.

Both C values are exact public-development p80 values from 72 image and 72 patient gradients,
respectively. LoRA trainables were explicitly fp32, sentinel gradients replayed exactly, no
optimizer existed, and the adapter did not change. The two superseded pre-freeze diagnostics and
their reasons are retained in the final report.

The accountant fixes 155 RDP orders and conservatively calibrates to the larger result from Opacus
1.6.0 and Google `dp-accounting` 0.6.0. At K10, ordinary image-DP M1-I8 has image
`(epsilon,delta)=(8,1e-5)`, `sigma=0.393338498`, but its patient group conversion is vacuous.
The conversion-matched M1-G8 instead needs image `(0.8,4.1126114e-9)` and
`sigma=1.155724687` to reach patient `(8,1e-5)`. Native M2-P8 directly reaches patient
`(8,1e-5)` at `sigma=0.404457146`; M2-P4 and M2-P2 are frozen secondary curve points. These are
per-run ideal-mechanism bounds, not cumulative authorization to release every benchmark model.

Patient membership uses 1,816 members and 1,816 holdout nonmembers, each 908/908 target/control and
exactly matched by target flag and K10 image count. The primary fixed-noise denoising-loss attack,
secondary gated SecMI-LDM adapter, metrics, 10,000 patient bootstrap, power, and three-way
`DETECTED`/`NO_MATERIAL_LEAKAGE_DETECTED`/`INCONCLUSIVE` decision are frozen. A separate bounded
2,450-query generate-and-filter extraction test uses thresholds calibrated only on public
development before target generation. Attack failure never strengthens the formal DP claim.

At this protocol-gate closure, private training was still blocked pending an
executable trainer-conformance/event-trace smoke. Formal release is additionally blocked because
the current environment has no registered release-grade secure sampling/Gaussian backend; a
research PRNG can yield only `RESEARCH_ONLY` evidence. Protocol SHA-256:
`F2757DC7EBC7488B6A9DD0F69227419EA16BB9A3E9BD4434514CB19AFD2B4018`; independent report:
`9CAB4DB57B83985A90BC2265D9FA2BBEDC122B96B3418919CB6A8B2684472FC4`. Full record:
`XRAY_DP_ATTACK_PROTOCOL_GATE.md`.

## 18. Executable DP-trainer conformance result

The frozen M1/M2 aggregation semantics now have an executable core and have passed synthetic,
actual-public-X-ray, and independent verification. The older owner-DP reference trainer is not used
as the X-ray execution path because it skips an empty Bernoulli sample; this would omit the required
Gaussian-noise-only event.

The new core clips one complete fp32 vector per declared privacy unit, requires M2 to average inside
the patient before its one outer clip, adds `N(0,(sigma*C)^2 I)` even for an empty sample, and divides
only by the public expected batch. Its public trace is schedule-only and hash chained; selected IDs,
realized counts, and DP seeds are excluded.

Ten unit tests and nine synthetic checks pass. The latter include analytic image/patient clipping,
an exact nonzero empty-sample replay, fixed-denominator recovery, 262,144-draw Gaussian moments,
20,000-step Poisson moments, fail-closed tamper/config tests, and exact dual-accountant replay for
K10 M1-I8 and M2-P8.

A disposable public-development SD 2.1 LoRA smoke then used two M1 images and two four-image M2
patients chosen solely from the frozen public calibration to cover the clipping branch. All four
actual gradients exceeded their frozen C and were clipped (M1 2/2, M2 2/2). Each arm began from the
same fp32 LoRA digest, reproduced a real-gradient sentinel and test Gaussian exactly, made one
finite AdamW update, and was discarded without a checkpoint. Because this public selection is
outcome-dependent branch coverage, it cannot be reused as an evaluation sample.

The independent verifier reconstructed the selection and trace without importing the mechanism,
replayed both accountants, checked source structure and artifacts, and passed. The status is
`PASS_EXECUTABLE_DP_TRAINER_CONFORMANCE`; at this gate closure private optimizer training had not
started because this result had to be reported first. Formal release remains blocked by the absence of a registered
release-grade sampling/Gaussian backend. Full record: `XRAY_DP_TRAINER_EXECUTABLE_GATE.md`.

## 19. K5 private-partition four-step runtime result

After reporting the executable mechanism gate, a separate machine-readable protocol froze a
bounded K5 runtime dry-run before any `private_train` optimizer loss was computed. The run covered
M1-I8, M1-G8, and M2-P8 for exactly four steps each. M1 arms shared an image-Poisson and diffusion
draw schedule for control matching but used independent DP Gaussian streams. M2 Poisson-sampled
patients and selected uniformly without replacement at most four K5 images inside each patient.

Fresh OS entropy seeded domain-separated ordinary PyTorch streams in process memory; no private
seed was serialized. This remains `RESEARCH_ONLY_NONCRYPTOGRAPHIC`. Public traces contain one
schedule-only event for each successfully completed optimizer step. Identifiers, realized counts,
losses, norms, and noise/update digests are confined to a local restricted diagnostic excluded from
release packaging.

All three arms completed four finite, nonzero AdamW updates from the same fp32 LoRA initialization.
The shared M1 realization contained 37 image units across the four steps. M2 contained 13 patient
units and 22 internally selected images. Both clipped and unclipped real gradients occurred; no
empty step or resource abort occurred. These small realized counts and clipping fractions are
runtime diagnostics and cannot tune or evaluate C, sigma, q, datasets, or the full matrix.

Each four-event trace and the Opacus/Google accountant replay passed. A separate verifier checked
K5 patient/image membership, M2 boundaries, M1 matching, aggregation values, source structure,
secret nonserialization, and checkpoint absence without importing the runner or mechanism. The
status is `PASS_K5_PRIVATE_RESEARCH_RUNTIME_DRYRUN`, not K5 feasibility or privacy evidence. No
model/checkpoint was retained. Full K5 4,000-step feasibility, including M0, remains unstarted and
must receive its own frozen execution/retention/utility plan. Formal release remains blocked by the
research RNG. Full record: `XRAY_K5_PRIVATE_RESEARCH_DRYRUN_GATE.md`.

## 20. K5 full-feasibility protocol freeze

Before any full K5 optimizer update, the one-seed feasibility matrix, failure handling, retention,
and evaluation criteria were separately frozen and independently verified. The order is M0,
M1-I8, M1-G8, and M2-P8, with the same P256 SD 2.1 base, fresh identical rank-8 LoRA start, fixed
AdamW settings, and exactly 4,000 steps per arm. M0 is a non-DP fixed-batch comparator. The three DP
arms use the exact K5 values already linked to the frozen accountant: image-DP M1-I8, patient-
conversion-matched image-DP M1-G8, and native patient-DP M2-P8. M1 schedules/diffusion draws are
matched while Gaussian streams remain independent; M2 averages up to four patient images before
one patient clip. Empty DP samples remain unconditional noise-only steps.

Only step 4,000 enters quantitative evaluation. Steps 1,000/2,000 are restricted fixed diagnostics
and cannot select or stop training. The final generation set is 448 prompt/seed pairs per model
across seven chest-X-ray conditions, shared by B0 and all four trained arms. The real reference is a
separately frozen set of 448 globally patient-distinct K5 public-development images across five
matched strata. K5 attacks and downstream augmentation utility are not run; both remain frozen for
K10.

The quality gate uses RAD-DINO KID as its primary small-sample feature measure, PRDC plus diversity
summaries, BioViL-T weak-label prompt alignment, and pixel sanity. Inception-only FID is not
sufficient. RAD-DINO's pretraining includes NIH ChestXray14, so it is only a comparative feature
screen and cannot be presented as independent clinical validation or privacy evidence. K10 can
progress only if execution/restart/evaluator gates pass and M0 improves on B0 under the fixed
confidence rules. An unfavorable DP arm is retained; it cannot be dropped or substituted from K5
outcomes.

Resume material must be encrypted in a Windows DPAPI CurrentUser envelope written atomically every
250 steps. Exact uninterrupted-versus-resumed equivalence for M0 and all DP semantics remains a
mandatory preflight. The in-memory DPAPI capability probe passed, as did independent reconstruction
of all generation/reference rows, DP values, and resource bounds. Full K5 training remains
unstarted pending evaluator/restart implementation, full-run source/environment freeze, and a
report. Formal release remains blocked by research RNG. Protocol SHA-256:
`2234B3BA8701565B768A11AB5196B2F696788900D07254833997D7823489E9DA`; independent report:
`49E715F715B5634B4071EA843AF8FC4A85666DD2E9EBC51D3D51DE1D89472A1B`. Full record:
`XRAY_K5_FEASIBILITY_EXECUTION_PLAN.md`.

## 21. K5 evaluator and encrypted-resume preflight result

The two pre-execution gates required by Section 20 now pass independently. No full K5 optimizer
execution was started.

The exact RAD-DINO and BioViL-T weights, revisions, dependencies, P256 information boundary, seven
prompt mappings, 448 patient-distinct references, frequency-preserving prompt derangement, and
2,000-replicate bootstrap were frozen before encoder output. RAD-DINO returned finite `[448,768]`
features with effective rank `184.65758447` and exact first-eight replay. BioViL-T returned finite
unit-normalized image/text features; matched-minus-deranged cosine was `0.06058422`, with 95% interval
`[0.03795662,0.08322361]`, so the frozen overall weak-label validity gate passed.

Four descriptive condition means were negative (consolidation, no finding, pleural effusion, and
pneumonia). Therefore BioViL-T is retained only as an overall weak-label alignment screen;
per-condition values are descriptive and cannot establish seven-condition clinical validity or
diagnostic correctness. An independent full re-encoding reproduced the aggregate results. Its
first attempt exposed a missing cuDNN TF32-disable setting; after matching the already frozen
deterministic runtime, the recomputation was exact. No metric or threshold changed.

The Windows DPAPI CurrentUser resume module writes only an inner-hash-framed ciphertext `.new`,
fsyncs and decrypt-verifies it, then atomically maintains current plus previous. Five fail-closed
tests pass. For M0, M1-I8, M1-G8, and M2-P8 synthetic semantics, uninterrupted four-step state and
two-plus-encrypted-resume-plus-two state matched exactly across adapter, AdamW, explicit and global
CPU/CUDA RNG, sampler, trace, committed step, and frozen inputs. An import-independent DPAPI/AdamW
probe plus static audit also passed. This is restart-mechanism evidence, not a full SD 2.1 checkpoint
or privacy/utility result.

The combined regression suite passes 26/26 tests. No feature bank, resized corpus, plaintext resume
file, generated image, or preflight envelope was retained. Evaluator report SHA-256:
`914002B52C2EF48C551FFD9A5CD0260BA7708EA6492F5296AEF98BF6D0B4F5B5`; evaluator independent:
`A55672E5D5C68491C75AAE4770F1C13A1F1B43BCCD721E7F732BF3C8D9560CAD`; resume report:
`89A08E9B1AAC248A3DF4F4935C82354D49D28AFC0DD8BE632DE6A80C394FCBFF`; resume independent:
`720D7A9DDE7A9DD862AE84B4053DBF0F704B8100CE0C7166D28B56EFE76BA463`. At this section's closure, the
next gate was the actual full-runner/environment freeze and a launch report before M0; Sections 22--24
record the later status. Full record:
`XRAY_K5_EVALUATOR_RESUME_PREFLIGHT_GATE.md`.

## 22. Actual full-runner and environment freeze

The actual SD 2.1 rank-8 LoRA runner, full AdamW/RNG/trace DPAPI resume state, source/environment
freeze, and read-only launch guard passed independent verification. This gate performed only bounded
public-development resume tests; it did not initialize the 4,000-step matrix. The gate-closure next
step was B0 generation. Full record: `XRAY_K5_FULL_RUNNER_GATE.md`.

## 23. B0 generation result and temporal-scope correction

B0 generated all 448 fixed outputs and passed full-file decode/hash/pixel checks plus independent
regeneration of eight fixed sentinels. The fixed 35-image visual grid was grossly off-domain, so B0
is retained only as the unadapted comparator and M0 domain adaptation is a hard gate. Full record:
`XRAY_K5_B0_GENERATION_GATE.md`.

The 2026-09-09 record audit identified an overbroad temporal statement. Section 19's K5
`private_train` four-step `RESEARCH_ONLY` dry-run occurred before B0. B0 therefore cannot establish
that no earlier private-role optimizer update occurred. It establishes ordering only before the full
4,000-step K5 feasibility matrix. The hash-frozen JSON strings are preserved as historical evidence,
but `generated_before_private_training` and `PASS_B0_FROZEN_BEFORE_PRIVATE_TRAINING` must be read as
`before full K5 matrix training`, not as `before every earlier dry-run`.

This correction does not change the 448 images, hashes, prompt/seed freeze, independent replay, or
off-domain comparator role. The current human-readable result is
`PASS_EXECUTION_INTEGRITY; TEMPORAL_CLAIM_NARROWED_TO_FULL_MATRIX`.

## 24. Current stopping point

The restricted full-run root, matrix initialization, M0 completion, and every full-matrix optimizer
step remain absent. The earlier isolated four-step dry-run had no retained checkpoint. If separately
authorized, the next mutating dissertation-experiment stage is matrix initialization followed by
**M0 only**, generation of the same 448 tasks, and the frozen domain-adaptation evaluation. A failed
M0 gate stops scientific progression to the DP arms.
