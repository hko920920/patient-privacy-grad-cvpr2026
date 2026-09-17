# Bounded downstream execution

The original `run.py classifier` / `data.py` / `train.py` files are frozen historical sources of the failed source-composition profile. Do not use that classifier entrypoint for new research runs. It also refuses implicit reruns of an existing output directory.

Corrected reusable code is `data_v2.py` plus `train_v2.py`. Pool names distinguish `real/public`, `real/private`, and `synthetic/<method>`.

The profile consumed31 generation decodes and100 total optimizer updates:98 with incorrect source composition, then2 corrected single-step S1 replays. See `classifier_verification_failure.json`, `classifier_repair_v2/result.json`, and `verification_v2.json` under `_reports/downstream_profile_20260917_v3`.

The corrected complete seven-arm runtime replay and full512-image/21-run development study have not run. The expert final and reserved cohort remain closed. Reusing the profile as an efficacy result is invalid.
