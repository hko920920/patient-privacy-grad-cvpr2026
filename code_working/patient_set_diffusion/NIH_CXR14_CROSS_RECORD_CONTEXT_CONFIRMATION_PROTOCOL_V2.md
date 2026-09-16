# NIH CXR14 cross-record-only held-out context confirmation v2

- Frozen on: 2026-09-03 after v1 failed and before any v2 model outcome
- Scope: one mechanism revision with fresh held-out patients; public non-DP diagnostic only
- V1 failure retained: standard SetAdapter learned a generic residual but no same-patient companion advantage

## Causal revision

V1 reduced correct-pair loss relative to bypass (`0.9331`) but correct/shuffled was `1.000214`, with only
50% correct wins and ratios above one in both target strata.  The standard adapter contained a direct own-token
residual (`tokens + attended`), so it could improve each image without using another record.

V2 makes one predeclared change: **strict leave-one-record-out context**.  Each record's adapter residual uses
values only from other valid records; the diagonal/self value path and the own-token residual path are removed.
For Q=2, record 1's delta is therefore a learned function of record 2's feature only, and conversely.  Output
projection remains zero-initialized; padding, permutation equivariance, and singleton identity remain.

Cross-view attention and constrained cross-view interaction already exist in multi-view diffusion.  This block
is not claimed as a standalone attention invention.  Its role is an identifiability constraint for the narrower
patient-set/patient-DP question.

## Frozen data

- Reuse only the 64 training patients from context-signal v1 selection SHA-256
  `B6951897C7270CC7A462496F82A5EC1CDDC65D7F37A15EA6E2D629809604E319`.
- Do not reuse its 32 validation patients for evaluation.
- For v2 validation, exclude all 400 earlier premise/gradient patients and all 96 v1 context patients:
  496 unique patients.
- From the remaining exact-Q2 K10 public-development patients, select 16 target-0 and 16 target-1 patients by
  salt `nih-cxr14-cross-record-context-confirmation-v2`.
- V2 validation totals 32 patients/64 images and is unseen by both mechanism training and v1 evaluation.

## Frozen training and evaluation

- Exact SD 2.1 base, locked NIH preprocessing, frozen original UNet, no LoRA.
- Train only the 447,360-parameter cross-record-only adapter.
- Reuse v1's 192-step balanced training schedule, patient exposures, independent uniform timesteps/noise,
  AdamW settings (`1e-3`, `(0.9,0.999)`, `1e-8`, weight decay `0.01`), and gradient clip `1.0`.
- Four held-out banks.  Alternate the primary record; compare correct same-patient companion, different-patient
  same-target finding-matched companion, and bypass with the primary input fixed exactly.
- Shuffled matching and all aggregation rules are identical to v1.  No checkpoint or tensor artifact is retained.

## Frozen gate

Use the same substantive thresholds as v1; all must hold:

1. 192 finite optimizer steps; cross adapter changes; frozen base sentinels do not;
2. zero-init correct/shuffled/bypass loss difference at most `1e-7` before training;
3. held-out aggregate `correct / shuffled <= 0.995` and correct wins at least `60%` of 128 cells;
4. correct/shuffled ratio below `1.0` in target-0 and target-1 separately;
5. aggregate `correct / bypass <= 0.995`;
6. post-training permutation and singleton max errors at most `1e-5`;
7. peak allocated CUDA below `7.5 GiB`.

Failure closes the current class `standard IID denoising + pooled mid-block patient context` as the main
method; it is not rescued with another attention mask, threshold, patient subset, or timestep rule.  Pass
licenses a matched public generation experiment, not a utility/privacy/clinical claim.
