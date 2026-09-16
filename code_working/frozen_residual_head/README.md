# Frozen Residual Head Patient-DP Audit Code

This repository contains Python code for the frozen residual-head patient-DP
comparison and its audit/verifier scripts.

The repository is prepared as a code-only upload. Generated reports, cached
bytecode, model checkpoints, and NumPy result archives are ignored by Git
because they may contain run-specific or non-release data.

## Contents

- `run_capacity.py`: builds and analyzes the frozen residual capacity baseline.
- `run_public.py`: prepares public-only calibration statistics.
- `public_calibrate_dp.py`: runs public-only DP mechanism calibration.
- `run_patient_dp.py`: runs the fixed patient-DP four-cell comparison.
- `verify_capacity.py`, `verify_public_dp.py`, `verify_patient_dp.py`: audit
  and verification scripts.
- `dp_mechanisms.py`, `residual_math.py`: numerical DP and residual-head helper
  routines.

## Dependencies

The scripts use Python 3, NumPy, and PyTorch. Some runners also import the local
`u_patient_audit` package from the surrounding project checkout and expect
precomputed report directories under `code_working/_reports`.

## Running

Run modules from the parent directory of this package so relative imports
resolve correctly, for example:

```powershell
python -m frozen_residual_head.public_calibrate_dp --phase self-test
python -m frozen_residual_head.run_patient_dp --phase prepare
python -m frozen_residual_head.run_patient_dp --phase run
python -m frozen_residual_head.run_patient_dp --phase analyze
```

The full experiment depends on local data, cached model features, and prior
report outputs that are not included in this code-only upload.
