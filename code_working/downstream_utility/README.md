# Downstream execution

## Full non-DP development study (2026-09-17)

`run_development_v1.py` is the separate full-study entrypoint. Execution is complete: 512 images, R1 calibration1,600 updates, 21 runs x400 steps. Integrity verification passed; both S2/S3 private-synthetic efficacy gates FAILED. See `result.json` and `verification.json` in `_reports/downstream_development_20260917_v1`. Expert-final, reserved and DP remain closed. Do not rerun for a more favorable outcome.

From `code_working`, using the project Python environment:

```text
python -B -X utf8 -m downstream_utility.run_development_v1 prepare_contract
python -u -B -X utf8 -m downstream_utility.run_development_v1 execute
```

`execute` runs raw/cache rebinding and observation parity, 512 unfiltered paired images, two public R1 calibration trajectories (1,600 optimizer updates), calibration freeze, 21 development training runs, development evaluation, and independent verification. The 400/800 calibration choice is made only on classifier-selection patients. All 21 runs finish before method-development scores are computed. No expert-final, reserved, or DP phase exists.

For an interruption, rerunning `execute` skips only SHA-complete cells/runs. Partial folders stop with an explicit error and remain preserved; they are never silently overwritten. The original corrected data/training files stay byte-identical. A scoped observation hook saves intermediate states without changing training math, with exact parity checked before the study.

Individual phases: `preflight`, `generate_512`, `calibrate_R1`, `train_21_runs`, `evaluate_method_development`, `verify_all`. These retain the same dependency and source-hash checks.

## Historical bounded profile and replay

The original `run.py classifier` / `data.py` / `train.py` files are frozen historical sources of the failed source-composition profile. Do not use that classifier entrypoint for new research runs. It also refuses implicit reruns of an existing output directory.

Corrected reusable code is `data_v2.py` plus `train_v2.py`. Pool names distinguish `real/public`, `real/private`, and `synthetic/<method>`.

The profile consumed31 generation decodes and100 total optimizer updates:98 with incorrect source composition, then2 corrected single-step S1 replays. See `classifier_verification_failure.json`, `classifier_repair_v2/result.json`, and `verification_v2.json` under `_reports/downstream_profile_20260917_v3`.

The corrected seven-arm one-step replay has now PASSED14updates with independent actual pixel/array/GPU-input and final-weight checks. Use only `python -m downstream_utility.run_v2 <phase>`; supported phases are `prepare`, `classifier_corrected`, and `verify_corrected`. Each existing output is protected against implicit reruns. The legacy classifier CLI immediately fails; the safe entrypoint rejects imports of both legacy data/train modules.

The14-update record is `_reports/downstream_all_arm_replay_20260917_v1`. A logging-only interruption after the saved first update was resumed for exactly13 more, with an explicit amendment and preserved artifacts. Historical sources changed for safety are preserved under the previous profile's `frozen_source_before_safe_entry_20260917`.

The full study uses the separate entrypoint above. Expert final and reserved cohorts remain closed. The historical one-step integration result is not long-run convergence or efficacy validation.
