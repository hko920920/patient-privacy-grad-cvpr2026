# Core-dataset pivot — frontal chest X-ray first, dermoscopy extension

- Decision date: 2026-09-02 (Asia/Seoul)
- Direction: **APPROVED**
- Intake status at pivot closure: **METADATA/MANIFEST PASS; OFFICIAL IMAGE ACQUISITION PENDING**
- Training status at pivot closure: **NOT STARTED**

Current supersession note (2026-09-09): official acquisition and B0 later completed. The initial
K2-feasibility/K5-primary/K10-stress role labels in this historical pivot were subsequently refined
to K2 accounting sensitivity, K5 one-seed feasibility, and K10 confirmatory main plus contribution
stress. Dataset counts and nested-cap construction did not change.

## 1. Current dataset hierarchy

1. **Primary/core:** one frontal chest-radiograph generator, with NIH ChestXray14 as the first
   intake candidate.
2. **Cross-domain extension:** the already frozen SIIM–ISIC 2020 dermoscopy experiment, trained and
   evaluated as a separate model only after the X-ray core is stable or under the predeclared
   contingency branch.
3. **Not allowed:** mixing X-ray, MRI, and dermoscopy images in one unconditioned training pool.
4. **Not current scope:** 3D MRI generation or a universal medical foundation generator.

The earlier ISIC acquisition and interface gates remain valid evidence and are not deleted. They no
longer define the first/core modality after this decision.

## 2. Real deployment scenario

A central hospital or radiology organization holds repeated frontal chest radiographs from many
patients and fine-tunes a shared latent-diffusion model. The model generates synthetic chest
radiographs for methodological research, education, and carefully evaluated augmentation of
thoracic-finding models. The lifecycle remains:

```text
restricted multi-patient chest X-rays
  -> non-DP / image-DP / native patient-DP generator comparison
  -> Auditable Privacy claim decision
  -> model-bound receipt
  -> PP-Mark on released synthetic radiographs
  -> third-party evidence-package decision
```

The work does not claim that the generator diagnoses a new patient or that a generated radiograph
is clinically usable without separate validation.

## 3. Why the scientific design still holds

The privacy question depends on the contribution unit, not on RGB skin texture. NIH ChestXray14 has
substantial repeated-patient structure:

| Scope | Images | Patients | Patients with >=2 | >=5 | >=10 | Maximum images/patient |
|---|---:|---:|---:|---:|---:|---:|
| all frontal views | 112,120 | 30,805 | 13,302 | 5,759 | 2,545 | 184 |
| PA only | 67,310 | 28,868 | 10,688 | 3,245 | 953 | 100 |
| AP only | 44,810 | 9,060 | 5,506 | 2,459 | 1,009 | 183 |

The official lists contain 86,524 train/validation images from 28,008 patients and 25,596 test
images from 2,797 patients, with zero image or patient overlap. Thus image adjacency and capped
patient adjacency remain experimentally distinct, including under a homogeneous PA-only policy.

The existing structural comparison is retained:

- `B0`: untouched base;
- `M0`: matched non-DP fine-tuning;
- `M1`: image-DP fine-tuning with explicit image-unit evidence;
- `M2`: native patient-DP fine-tuning with patient aggregation;
- optional no-noise patient-aggregation ablation to isolate aggregation from privacy noise;
- patient-disjoint development, attack-holdout, and final-test partitions;
- predeclared `K=2/5/10` contribution sensitivity, with X-ray-specific cohort sizes to be recomputed
  rather than copying the ISIC image totals.

Formal unit-labelled privacy, patient/image membership attacks, memorization/extraction, utility,
and audit-versus-retrain cost remain the core outcomes. A large empirical attack gap is not promised
in advance; an attack-null result is handled under the existing formal/canary/utility interpretation.

## 4. Clinically consequential finding scope

The paper must use the term **radiographic finding**, not claim definitive disease or cancer
diagnosis from one X-ray. Preliminary PA-only counts support four intuitive, consequential groups:

| Candidate group | PA images | PA patients | Interpretation boundary |
|---|---:|---:|---|
| pneumothorax | 3,407 | 1,155 | urgent collapsed-lung finding |
| pneumonia or consolidation | 2,108 | 1,601 | infection/air-space-opacity-related finding; not pathogen confirmation |
| pleural effusion | 6,589 | 3,279 | fluid around the lung |
| pulmonary mass or nodule | 7,140 | 4,089 | cancer-suspicious finding; **not a cancer label** |
| `No Finding` reference | 39,302 | 22,452 | none of the 14 mined labels; **not adjudicated normality** |

Cardiomegaly is a possible fifth group (1,563 PA images/1,150 patients). Pulmonary edema is not
frozen merely because its name sounds severe: only 276 PA images carry that label, while 2,027 are
AP, making it tightly entangled with view and bedside-acquisition status. It is reconsidered only if
the gate selects an explicitly view-conditioned AP+PA design.

These groups overlap and are now frozen as the four primary evaluation strata. All source labels are
retained in the local manifests; the exact multi-label text/prompt conditioning policy remains a
pretraining gate and cannot be chosen after observing generator or attack results.

## 5. Frozen view and remaining exact-image policy

The primary experiment is now frozen as **PA-only core training** because it removes a large
acquisition shortcut while retaining ample patient and finding counts. AP and PA must not be
silently pooled. An AP+PA experiment requires a transparent contract amendment, explicit view
conditioning, and matched view distributions.

The intake gate must also freeze:

- portable versus standard acquisition and support-device policy;
- side marker, text, tube, lead, and postoperative-hardware handling;
- grayscale decoding and deterministic one-to-three-channel mapping for SD 2.1;
- crop/padding, orientation, intensity scaling, and 256/512 resolution policy;
- weak-label confidence, label overlap, and any expert-adjudicated evaluation subset;
- patient-level deduplication, split, and capped image selection before outcomes are visible.

## 6. Utility and provenance changes

Inception-only FID is not sufficient. X-ray utility requires a radiology-domain feature extractor,
held-out real-patient finding evaluation, diversity/nearest-neighbor analysis, and explicit checks for
anatomical and acquisition shortcuts. Label fidelity and any augmentation benefit must be evaluated
on real held-out patients and reported with patient-level confidence intervals.

PP-Mark still operates on the released 2D image, but grayscale replication, intensity-preserving
distortion, geometry, crop/resize, and exact medical-checkpoint calibration must be retested. PP-Mark
proves neither a radiographic finding nor DP.

## 7. Source and metadata preflight

The official NIH Box pages expose the following file commitments:

| File | Official bytes | Official SHA-1 | Local preflight status |
|---|---:|---|---|
| `BBox_List_2017.csv` | 92,416 | `c567ceb25277c9883c5a7fe4c2f70c2ef94b02be` | exact match |
| `train_val_list.txt` | 1,470,907 | `e3e1b677c01d28481777f3d84e10fcdaac05694c` | exact match |
| `test_list.txt` | 435,131 | `41b85e218abec560a2f5999acbcf333b0f2fa495` | exact match |
| `Data_Entry_2017_v2020.csv` | 9,003,496 | `48a9f849a8f100a0f1721b33bdbd209767656111` | exact official file acquired |

The accessible metadata mirror used only for preflight is hash-pinned at
`alkzar90/NIH-Chest-X-ray-dataset` revision
`36778e3b0e4f4b4fad31d1728d6190f3eda5b543`. Its main CSV has SHA-256
`DC1D2DF67FDC1C5A7601D48699CDA2B13DC2C4841488B4183DCF04884DBACA11`. The three-byte difference
is resolved: only the header text differs (`Patient Gender` versus official `Patient Sex`), while
all 112,120 image rows are identical. The exact official Box file, SHA-256
`C69A6DACA3549AF707CA9CACBDF9F9A7B6A9188E8C61157DF653520BB72D8EB1`, is the frozen authority.

The six previously viewed JPEGs are third-party visual renditions and are permanently excluded from
experimental input. The full official release is roughly 42 GiB; image acquisition must avoid
simultaneously retaining redundant archives and extracted copies and must leave adequate space for
checkpoints and evidence.

## 8. Predeclared ambiguity and fallback rule

The X-ray pilot is a feasibility run excluded from confirmatory claims. Before seeing its privacy
attack ordering, the next contract revision must set numerical gates for:

1. exact-source and patient-manifest validity;
2. anatomical/image-quality and diversity acceptance;
3. finding-conditioning or downstream-utility signal above the matched control;
4. a powered attack protocol with positive-control canaries;
5. DP accountant/runtime correctness and feasible replication cost;
6. PP-Mark calibration on the exact resulting checkpoint.

If X-ray results are ambiguous, the failed or null pilot is retained and reported. The response may
be a preregistered X-ray redesign, narrower claim, additional compute, or a transparent amendment
that promotes dermoscopy to the core. It is not permissible to switch modalities merely because a
preferred attack ranking failed to appear.

## 9. Manifest closure and immediate next gate

The PA-only target-enriched patient manifests now pass. They contain 13,926 patients split into
8,476 private-train, 1,816 development, 1,816 privacy-attack, and 1,818 enriched final-test patients,
with a 1:1 target/control patient balance in every partition. Nested totals are K2=20,687,
K5=31,451, and K10=38,492 images. At this pivot closure K2 was called the feasibility pilot, K5 the
primary/main condition, and K10 the contribution-unit stress condition. The later execution contract
supersedes those role names as stated at the top of this document. The manifest lock is
`518D3288ACDF55BE355038BC693B3BCF7A8C44B8B3455FAC1DC493EE5E4E3ECE`.

Because the enriched final test is not a prevalence sample, the full official PA-test census is
also frozen as an overlapping distribution-sensitivity view: 11,096 images/2,647 patients. The
K10-plus-census acquisition union is 42,423 unique images.

Before any generator or DP training:

1. map the official image archives and freeze source commitments;
2. selectively acquire the K10-plus-census union once, with K2/K5 as nested views;
3. verify official PNG decoding, patient mapping, duplicates, and content commitments;
4. freeze grayscale, crop/padding, resolution, and prompt-conditioning policy;
5. run a small exact-image SD 2.1 VAE/LoRA/PP-Mark interface gate;
6. then freeze DP mechanism/accountant/budget and attack power.

Full evidence is in `XRAY_METADATA_MANIFEST_GATE.md`. No generator/DP training or active LaTeX edit
was made by this pivot or manifest decision.

## 10. Official-PNG source-smoke update

Items 1 and the small-source portion of item 3 above now pass. All 12 official image archives are
locked by current Box file/version ID, byte count, and provider SHA-1. The verified first archive
contains 4,999 PNGs, all found in the exact metadata, with zero duplicate basenames or unsafe/special
members. A deterministic 24-image/24-patient sample covers every partition/target-status bucket,
all four primary finding groups, `No Finding`, and census-only cases; every file decodes as a
1024 x 1024 grayscale PNG. A second offline execution and an independent hash/header check exactly
reproduced the result.

Manual source-format inspection also exposed normal real-world variability in exposure,
positioning, laterality markers, tubes, and surgical clips. It did not adjudicate labels. The
preprocessing/utility gate must explicitly address device/marker shortcuts without silently erasing
clinically meaningful content.

At source-smoke closure, only those 24 PNGs were materialized and the then-immediate next gate was
full selective acquisition. That historical status is superseded by Section 11 below. Smoke
details: `XRAY_OFFICIAL_PNG_SMOKE_GATE.md`.

## 11. Full-acquisition update

Items 1–3 above now pass completely. All 12 archives were provider-hash verified and safely
processed; their 112,120 unique PNG names exactly match official metadata. The full union contains
42,423 PNGs/14,755 patients/16.457515 GiB with content-set SHA-256
`E983A21B9B8558CE38F1AD5BA7C0F6BC51787CC7CBA739399ACE1976E2FB068E`. Offline regeneration and a
separately implemented verifier both re-hashed the entire population.

The full run corrected the smoke-only native-mode assumption: 42,244 files are `L`, while 179 are
`RGBA` with pixelwise-identical R/G/B and fully opaque alpha. Raw bytes remain unchanged, and actual
color/transparency failures are zero. The remaining pretraining step is item 4–5: freeze the exact
raw-to-model preprocessing and run the SD 2.1/PP-Mark image-interface gate. Full evidence:
`XRAY_FULL_ACQUISITION_GATE.md`.
