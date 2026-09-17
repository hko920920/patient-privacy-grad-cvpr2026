# Bounded downstream execution

The original `run.py classifier` / `data.py` / `train.py` files are frozen historical sources of the failed source-composition profile. Do not use that classifier entrypoint for new research runs. It also refuses implicit reruns of an existing output directory.

Corrected reusable code is `data_v2.py` plus `train_v2.py`. Pool names distinguish `real/public`, `real/private`, and `synthetic/<method>`.

The profile consumed31 generation decodes and100 total optimizer updates:98 with incorrect source composition, then2 corrected single-step S1 replays. See `classifier_verification_failure.json`, `classifier_repair_v2/result.json`, and `verification_v2.json` under `_reports/downstream_profile_20260917_v3`.

The corrected seven-arm one-step replay has now PASSED14updates with independent actual pixel/array/GPU-input and final-weight checks. Use only `python -m downstream_utility.run_v2 <phase>`; supported phases are `prepare`, `classifier_corrected`, and `verify_corrected`. Each existing output is protected against implicit reruns. The legacy classifier CLI immediately fails; the safe entrypoint rejects imports of both legacy data/train modules.

The14-update record is `_reports/downstream_all_arm_replay_20260917_v1`. A logging-only interruption after the saved first update was resumed for exactly13 more, with an explicit amendment and preserved artifacts. Historical sources changed for safety are preserved under the previous profile's `frozen_source_before_safe_entry_20260917`.

The full512-image/21-run development study is still unexecuted and needs explicit orchestration using the corrected kernel, not the legacy CLI. Expert final and reserved cohorts remain closed. One-step integration is not long-run convergence or efficacy validation.
