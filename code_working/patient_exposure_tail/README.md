# Patient exposure-tail diagnostic

This directory contains the frozen public NIH ChestXray14 endpoint-feasibility diagnostic for a
possible patient-level biometric exposure audit of chest-X-ray diffusion models.

Run pure tests first:

```powershell
python -m unittest test_patient_exposure_tail.py -v
```

Then run the frozen CUDA diagnostic once:

```powershell
python run_nih_cxr14_patient_exposure_tail_premise.py
```

The aggregate report is written below `code_working/_reports/`; the identifier-bearing pairing CSV
is local-only. No encoder features, checkpoint, optimizer, gradient, latent, or generated image is
retained.

## Frozen result

Status: `PASS_PATIENT_EXPOSURE_TAIL_ENDPOINT_PREMISE`.

- DINOv2: pair AUC `0.825795`, R@1 `0.239278`, rank > 10 fraction `0.577878`.
- RAD-DINO: pair AUC `0.988363`, R@1 `0.711061`, rank > 10 fraction `0.102709`.
- Cross-encoder patient-margin Spearman: `0.505432`, 95% bootstrap CI
  `[0.451656,0.555337]`.

This pass licenses only a separately frozen generator biometric-exposure endpoint addendum. It is
not evidence of generator leakage, differential privacy, clinical identity, or patient-specific
membership AUC, and it does not authorize the long K5 run.
