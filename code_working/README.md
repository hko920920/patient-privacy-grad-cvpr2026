# Dissertation Code Working Tree

- Created: 2026-09-02 (Asia/Seoul)
- Baseline: adjacent `../code_originals/`
- Rule: modify code only here; do not edit `code_originals/`

## Current authority and stopping point — 2026-09-09

Read `../CURRENT_STATUS.md` first. NIH X-ray acquisition, executable DP/evaluator/resume/full-runner
gates, and B0 generation are complete. B0 has 448 integrity-verified but grossly off-domain images.
The full K5 matrix, M0, all full-matrix DP arms, attacks, model-bound receipt, medical PP-Mark
integration, and release package remain unstarted.

The earlier K5 `private_train` four-step-per-DP-arm `RESEARCH_ONLY` dry-run predates B0 and retained
no checkpoint. Accordingly, B0 is fixed before the **full 4,000-step matrix**, not before every
private-role optimizer update. If separately authorized, the next mutating stage is matrix
initialization followed by M0 only and the frozen domain-adaptation evaluation.

## Directories

- `ppmark/`: working copy of the latest PP-Mark v0.41 development source snapshot, including
  local external comparison implementations.
- `auditable_privacy/`: working copy of the locked Auditable Privacy V17 release.
- `unitdp_compiler_reference/`: inactive reference copy of the locked UnitDP V10 release. Do not
  make it a dissertation component merely because it is present.
- `patient_exposure_tail/`: frozen public NIH endpoint-feasibility diagnostic for patient-level
  biometric exposure tails; not a generator attack or DP result.
- `_diagnostics/unitdp_incomplete_first_extract/`: retained evidence of the first incomplete
  407/410-file extraction. Never use this tree as source.

## Intake verification

- PP-Mark: 16 passed, 5 environment-dependent skips.
- Auditable Privacy: manifest 20,909/20,909 and 91 tests passed.
- UnitDP reference: public verifier passed; 141 tests passed, 13 optional-backend skips, and
  180 subtests passed.

At intake, the selected Python environment lacked `poseidon-py` even though PP-Mark declared it in
`requirements.txt`, and it lacks optional CuPy. The SP1 build output was deliberately excluded and
must be rebuilt when full proof execution begins. No integrated medical DP generator, model-bound
privacy receipt, PP-Mark receipt adapter, or unified evidence-package code exists yet.

Since intake, the working tree has added the X-ray data pipeline, executable DP mechanism and trainer,
evaluator, encrypted resume, full-runner guards, and B0 generator. It still has no completed
fine-tuned medical generator, model-bound receipt, medical PP-Mark receipt adapter, unified release
package, or marked release image.

## Dataset intake

Step 2 initially selected ISIC 2020 Collection 70 as the primary 2D patient-lineage candidate and
PAPILA as the smoke dataset. The later X-ray pivot superseded ISIC as the core while retaining it as
a separately trained extension. The fixed metadata, duplicate policy, exact hashes, license
boundaries, and candidate comparison are recorded in `DATASET_INTAKE_DECISION.md`. Raw data lives
under `_data/` and must remain outside future source repositories.

## Experiment-contract status

Step 3 froze the central LoRA fine-tuning architecture, patient-disjoint comparison, B0/M0/M1/M2
matrix, and then-current ISIC contribution scales. Those ISIC counts remain extension evidence but
do not define the X-ray core. For the current X-ray plan, K2 is accounting sensitivity, K5 is the
one-seed feasibility tier, and K10 is confirmatory main plus contribution stress.

The reopened Step 4 selected `Manojb/stable-diffusion-2-1-base` revision
`0094d483a120f3f33dafbd187ea4aa60d10de75c` for the skeleton/pilot. It is recorded as a
third-party, hash-pinned mirror rather than mislabeled as an accessible official repository. Exact
fp16 hashes, license/provenance boundaries and RTX 3070 diagnostics are in
`SD21_BASE_MODEL_GATE_DECISION.md`; the earlier verified SD v1.4 gate remains a fallback record in
`BASE_MODEL_GATE_DECISION.md`. SD 2.1 passed 256/512 inference and rank-8 LoRA one-step probes, but
the GPU is still classified only as smoke/pilot-capable.

Step 5 froze a label-independent 70/10/10/10 patient split and exact K2/K5/K10 manifests. K10
private-train contains 1,415 patients and 10,848 images; final-test contains 208 patients and 1,641
images. Eight real ISIC JPEGs passed deterministic 256 preprocessing and exact SD-v1.4 VAE
encoding. PP-Mark's DDIM loader now accepts a pinned revision/variant; exact inversion passed first
on SD v1.4 and then on the selected SD 2.1 artifact. Historical PP-Mark calibration is still not
transferable to the medically fine-tuned checkpoint. Details are in `STEP5_SPLIT_LOADER_GATE.md`
and `SD21_BASE_MODEL_GATE_DECISION.md`.

Step 6 acquired and independently decoded all 9,495 frozen K5 JPEGs (2,056 patient-disjoint
subjects; 5.796924 GiB; zero failures). The ordered content-set SHA-256 is
`06F177B5C345121D3C5D876530768437A710C9C56DAD30F1225BD795C536E3F2`. The images are a
reproducible public proxy for a restricted dermoscopic cohort, not evidence that public ISIC data
became confidential. Details and claim limits are in `STEP6_K5_ACQUISITION_GATE.md`.

After the visual comparison, the user approved frontal chest X-ray as the first/core modality and
retained the completed ISIC setup as a separately trained cross-domain extension. NIH ChestXray14
is now the exact metadata authority: all four official metadata/split files are byte/hash locked, the
earlier three-byte mirror discrepancy is resolved as a header-only difference, and PA-only
patient-disjoint K2/K5/K10 manifests have been frozen. Totals are K2=20,687, K5=31,451, and
K10=38,492 images across 13,926 selected patients. The full 11,096-image/2,647-patient official PA
test census is also frozen as an overlapping distribution-sensitivity view, not an independent test
or tuning set. The manifest lock is
`518D3288ACDF55BE355038BC693B3BCF7A8C44B8B3455FAC1DC493EE5E4E3ECE`. Details are in
`XRAY_CORE_PIVOT_DECISION.md` and `XRAY_METADATA_MANIFEST_GATE.md`.

The 12 official NIH image archives are ID/version/byte/provider-SHA-1 locked and have now been fully
processed. The exact K10-plus-census union contains 42,423 PNGs from 14,755 patients and occupies
16.457515 GiB. All 112,120 archive PNG names exactly cover the official metadata. Acquisition-time,
offline byte-regeneration, independent full-file decode/hash, and PowerShell aggregate checks pass.
The ordered content-set SHA-256 is
`E983A21B9B8558CE38F1AD5BA7C0F6BC51787CC7CBA739399ACE1976E2FB068E`. Native modes are 42,244
`L` and 179 fully opaque, pixelwise grayscale-equivalent `RGBA`; raw files were not converted.
Details are in `XRAY_OFFICIAL_PNG_SMOKE_GATE.md` and `XRAY_FULL_ACQUISITION_GATE.md`.

The exact X-ray raw-to-model preprocessing and real SD-2.1 VAE/LoRA/PP-Mark interface have passed.
The subsequent pre-training gate now freezes image-DP and patient-DP mechanisms, 4,000-step dual
RDP accounting, public-only clip norms, matched patient-MIA cohorts, and bounded extraction rules.
At K10, ordinary image `(epsilon,delta)=(8,1e-5)` conversion is vacuous; the group-matched image arm
uses `(0.8,4.1126114e-9)`, while native patient-DP directly targets `(8,1e-5)`. The protocol and an
independent verifier pass, but no generator has been trained. That protocol gate required an
executable DP-trainer/event-trace conformance smoke before any private pilot, while formal release
additionally requires a registered secure RNG/noise backend. See
`XRAY_SD21_PPMARK_INTERFACE_GATE.md` and `XRAY_DP_ATTACK_PROTOCOL_GATE.md`.

That executable DP-trainer conformance gate has now passed. `dp_training/` implements and tests
per-image/per-patient clipping, patient mean-before-clip, unconditional empty-sample Gaussian noise,
fixed expected-batch normalization, Poisson selection, and a schedule-only hash-chained public
trace. Ten fast tests and nine synthetic checks pass. A disposable public-development SD 2.1 LoRA
smoke exercised actual clipping for M1 and M2 and one AdamW update per arm, then discarded both
models without a checkpoint. An independent verifier reconstructed the selection, traces, and
accounting. That conformance gate itself did not use private-role optimizer training, and formal
release remains blocked on a registered secure RNG backend. See
`XRAY_DP_TRAINER_EXECUTABLE_GATE.md`.

The next bounded gate has also passed: K5 `private_train` was exercised for exactly four optimizer
steps on each of M1-I8, M1-G8, and M2-P8. This was a `RESEARCH_ONLY` runtime dry-run, not the
4,000-step K5 feasibility matrix. Ephemeral OS entropy seeded ordinary PyTorch streams without seed
serialization; public traces remain schedule-only, while selected IDs and realized diagnostics are
kept in a local restricted JSON. All arms stayed finite, shared the same initial LoRA state, emitted
four valid events, matched both accountants, and wrote no checkpoint. Independent verification
passed. M0, full K5 feasibility, trained-arm generation, attacks, PP-Mark, receipts, and release
remain unstarted. B0 generation was completed later. See
`XRAY_K5_PRIVATE_RESEARCH_DRYRUN_GATE.md`.

The full K5 one-seed feasibility protocol is frozen and independently verified, but its
4,000-step run has not started. It fixes M0/M1-I8/M1-G8/M2-P8, final-only quantitative evaluation,
448 shared generation tasks, 448 patient-distinct public-development references, RAD-DINO
KID/PRDC, BioViL-T weak-label alignment, encrypted resume semantics, failure rules, and a 16--21
hour planning envelope. At that protocol's closure the next gate was evaluator plus exact
restart-equivalence implementation; the following paragraph records its later completion. See
`XRAY_K5_FEASIBILITY_EXECUTION_PLAN.md`.

The pinned evaluator and encrypted-resume preflights now also pass independently. RAD-DINO and
BioViL-T re-encoded all 448 fixed real references; the overall BioViL-T matched-versus-deranged gate
passed, while four negative condition summaries are retained as an explicit limit against
condition-level clinical claims. M0/M1-I8/M1-G8/M2-P8 four-step state matched exactly across an
encrypted two-plus-two restart. Twenty-six regression tests pass, and no feature bank, plaintext
resume, trained model, or generated image was retained. See
`XRAY_K5_EVALUATOR_RESUME_PREFLIGHT_GATE.md`.

The actual 4,000-step training runner is now implemented, environment-frozen, and independently
verified, but its 4,000-step matrix remains unstarted. A real pinned SD 2.1 test on
public-development records proved
bit-exact four-step versus encrypted two-plus-resume-plus-two equality for M0 and all three DP
semantics, including the complete 1,659,904-value LoRA and AdamW state. The combined regression
suite now passes 32/32. The runner defaults to read-only validation and requires a B0 manifest,
explicit start flag, exact launch-authority hash, 15 GiB disk, and WDDM-safe GPU readiness before
initialization. At that gate's closure the next stage was the fixed 448-image B0 generation/hash;
that B0 gate has since passed. No private 4,000-step arm has started. See
`XRAY_K5_FULL_RUNNER_GATE.md` and `XRAY_K5_B0_GENERATION_GATE.md`.

## B0 result and current stop

B0 protocol v1_002 generated 448/448 fixed P256 PNGs with no reroll, rejection, or replacement.
Full-file decode/hash/pixel checks and independent regeneration of the first and last batches passed.
The fixed 35-image grid was grossly off-domain, so B0 is an unadapted comparator rather than a
radiographic-quality pass. M0 must demonstrate domain adaptation before any DP arm is interpreted.

The 2026-09-09 audit corrected one temporal overstatement: the separate K5 four-step dry-run occurred
before B0. The frozen JSON artifacts remain unchanged, but their `before private training` wording is
valid only for the full 4,000-step matrix. See `../CURRENT_STATUS.md` and
`XRAY_K5_B0_GENERATION_GATE.md`.

## PCM gradient diagnostics

`pcm_diagnostic/` contains the pre-training allocation diagnostics. The synthetic positive/negative
controls passed, and a preregistered one-patient ISIC `public_development` smoke computed 20 actual
SD 2.1 rank-8 LoRA gradients without an optimizer update or DP noise. The execution gate passed, but
the frozen VAE coverage estimator was worse than uniform-distinct sampling for that one patient and
`C=1.0` produced no clipping. This adverse/null evidence is retained. It is not a method, utility,
privacy, or novelty result; a separately frozen multi-patient diagnostic is required before promotion.

## UCAN public-gradient diagnostic

`unit_coupled_noise/` tests unit-coupled antithetic corruption noise on NIH CXR14 public-development
records. Five code tests pass, but the preregistered empirical method gate failed. Across 16 fresh
patients, five banks, four arms, and 320 actual SD 2.1 rank-8 LoRA full gradients, complete
UCAN/shared-t IID had clipped actual-B2 error ratio `1.074199` with bootstrap 95% CI
`[0.965688,1.206865]`; patient leave-one-out wins were 0/16. The status is
`FAIL_DO_NOT_PROMOTE_THIS_UCAN_IMPLEMENTATION`. No optimizer, DP noise, private-role data,
checkpoint, or retained raw-gradient tensor was used. This antithetic corruption-noise branch is
closed; its unclipped outlier result must not be used to rescue the method. Protocol and report are
in `unit_coupled_noise/NIH_CXR14_UCAN_PUBLIC_DIAGNOSTIC_PROTOCOL_V1.md` and
`_reports/nih_cxr14_ucan_public_diagnostic_v1_001/report.json`.

## Patient-set diffusion premise diagnostic

`patient_set_diffusion/` begins the replacement direction, Patient-Set Private Diffusion (PSPD),
without yet implementing or training a generator. Its frozen public DINOv2 probe used 80 fresh NIH
patients and 320 images with zero overlap to prior gradient experiments. The preregistered status is
`FAIL_PATIENT_SET_DATA_PREMISE` because the absolute within-minus-cross cosine gap was `0.011060`,
below `0.05`; this threshold is not changed post hoc. Four other gates passed: balanced same-patient
AUC `0.877852`, retrieval R@1 `0.65625` (`69.78x` chance), multi-label-set patient fraction `0.7125`,
and within-patient label Jaccard distance `0.500799`. These scale-free signals justify only a fresh,
nonoverlapping 2--3-record confirmation. They do not establish a successful SetAdapter, generator,
privacy guarantee, or utility gain. See `patient_set_diffusion/NIH_CXR14_PATIENT_SET_PREMISE_PROTOCOL_V1.md`
and `_reports/nih_cxr14_patient_set_premise_v1_001/report.json`.

The v1 failure was not relabelled. A new frozen confirmation excluded all prior 240 patients and used
160 patients with exactly two or three records, balanced across target and record-count cells. All five
scale-free gates passed: pooled AUC `0.880605`, cell AUCs `0.863333--0.908472`, overall and per-cell
patient-ordering win rate `0.975`, global R@1 `0.48` (`119.7x` chance), changed-label fraction `0.75625`,
and label Jaccard distance `0.582240`. Status is
`PASS_PATIENT_SET_DATA_PREMISE_CONFIRMATION`; this licenses architecture testing only. See
`patient_set_diffusion/NIH_CXR14_PATIENT_SET_PREMISE_CONFIRMATION_PROTOCOL_V2.md` and
`_reports/nih_cxr14_patient_set_premise_confirmation_v2_001/report.json`.

The following public Q=2 SD 2.1 architecture smoke also passed. `set_adapter.py` adds one 463,872-scalar
permutation-equivariant mid-block SetAdapter to the existing 1,659,904-scalar rank-8 LoRA. Ten pure tests
pass. Integrated zero-init/bypass, swapped-record equivariance, and singleton/bypass max errors were all
`0.0`; the full Q=2 LoRA+SetAdapter backward was finite; a temporary nonzero probe demonstrated
cross-record influence L2 `0.0402214`; peak allocated CUDA was `1.81906 GiB`. No optimizer, DP noise,
checkpoint, latent, gradient, or adapter was retained. Status is `PASS_Q2_SETADAPTER_ARCHITECTURE_SMOKE`.
This licenses only a matched public non-DP generation falsification, not a utility or privacy claim. See
`patient_set_diffusion/NIH_CXR14_SETADAPTER_ARCHITECTURE_SMOKE_PROTOCOL_V1.md` and
`_reports/nih_cxr14_q2_setadapter_architecture_smoke_v1_001/report.json`.

The licensed context falsification has now been executed and failed twice.  The standard SetAdapter trained
for 192 public non-DP steps improved correct-pair loss over bypass (`0.933078`) but did not improve over a
finding-matched other-patient companion: correct/shuffled was `1.0002136004` and wins were `64/128`.  A
single predeclared causal revision removed the own-token residual and every diagonal/self value path.  On
fresh validation patients, this strict cross-record-only adapter again improved over bypass (`0.939390`) but
correct/shuffled was `1.0010702866`, with wins `65/128` and ratios above one in both target strata.  Status is
`FAIL_CROSS_RECORD_CONTEXT_CONFIRMATION_CLOSE_STANDARD_IID_CONTEXT_CLASS`.  Therefore the current
standard-IID pooled-context PSPD method is closed; full generation and patient-DP promotion are not licensed.
Protocols, runners, tests, and reports are under `patient_set_diffusion/` and
`_reports/nih_cxr14_{setadapter_context_signal_v1,cross_record_context_confirmation_v2}_001/`.

The next conditional hypothesis is a directed longitudinal residual transition whose previous input is
necessary by task construction, not another attention-mask rescue.  NIH public-development has 3,015
adjacent-available ordered pairs and 2,279 exact `Follow-up #` gap-1 pairs, but that field is only an order
proxy: local NIH metadata contains no study date/time, reports, medication, or lab events, and no MIMIC-CXR
corpus is present locally.  NIH can support only a frozen residual-premise diagnostic; any actual-time
longitudinal generation claim requires timestamp/report data access first.

That frozen NIH order-proxy premise has now also failed.  After excluding all 528 earlier patients, the
diagnostic selected 288 new patients/576 images, balanced 96/48 train/validation within changed-label and
unchanged-label exact follow-up-index pairs.  A training-only-CV 30-variable transition ridge was evaluated in
both generic DINOv2 and NIH-pretrained RAD-DINO spaces.  On changed validation pairs, true-residual/prior-copy
was respectively `1.140914543` (95% CI `[1.073630,1.236754]`, 8/48 wins) and `1.159127120`
(`[1.101906,1.245540]`, 1/48 wins).  In contrast, correct same-patient prior/finding-matched shuffled-prior
was `0.516104650` and `0.319953448`.  Thus the stable patient carrier is real, but NIH weak labels/order do
not provide a useful held-out change residual.  Status is `FAIL_NIH_ORDER_PROXY_RESIDUAL_PREMISE`; no
feature, regressor, checkpoint, latent, gradient, optimizer state, or generated image was retained.  See
`longitudinal_residual_diffusion/NIH_CXR14_ORDER_PROXY_RESIDUAL_PREMISE_PROTOCOL_V1.md` and
`_reports/nih_cxr14_order_proxy_residual_premise_v1_001/report.json`.

## Patient-exposure-tail endpoint premise

`patient_exposure_tail/` contains the next public, feature-only falsification. It used every one of the
886 K10 public-development patients with at least two records and equalized retrieval opportunity to one
first-order query and one last-order true gallery image per patient. Metadata-matched different-patient
negatives were frozen within 16 sex/target/record-count cells. Generic DINOv2 and NIH-pretrained RAD-DINO
both passed the conjunctive endpoint gate: pair AUC was `0.825795` and `0.988363`, R@1 among 886 gallery
patients was `0.239278` and `0.711061`, and rank-greater-than-10 fractions were `0.577878` and `0.102709`.
The cross-encoder patient-margin Spearman correlation was `0.505432` with 95% bootstrap CI
`[0.451656,0.555337]`. Status is `PASS_PATIENT_EXPOSURE_TAIL_ENDPOINT_PREMISE`.

This pass licenses only a separately frozen biometric-exposure endpoint addendum. It is not generator
leakage, individual-patient membership AUC, clinical identity evidence, a privacy guarantee, or authorization
for the long K5 run. No feature bank, model, optimizer, gradient, latent, or generated image was retained.
See `patient_exposure_tail/NIH_CXR14_PATIENT_EXPOSURE_TAIL_PREMISE_PROTOCOL_V1.md` and
`_reports/nih_cxr14_patient_exposure_tail_premise_v1_001/report.json`.
