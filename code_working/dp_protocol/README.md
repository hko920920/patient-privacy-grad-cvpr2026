# X-ray DP and attack protocol

This directory implements the pre-training freeze for the NIH ChestXray14 image-DP versus native
patient-DP experiment. It does not contain a trained generator or a release-grade DP runtime.

Run from `code_working` with the existing base-gate environment:

```powershell
$python = ".\base_gate\.venv\Scripts\python.exe"
& $python .\dp_protocol\calibrate_xray_public_clip_norms.py
& $python .\dp_protocol\build_xray_dp_attack_protocol.py
& $python .\dp_protocol\verify_xray_dp_attack_protocol_independent.py
& $python -m unittest -v .\dp_protocol\test_xray_dp_attack_protocol.py
```

The public clip diagnostic loads the pinned model and computes gradients, but does not create an
optimizer or change parameters. The builder is CPU-only. The independent verifier does not import
the builder or UnitDP helper code. Exact decisions and claim limits are in
`../XRAY_DP_ATTACK_PROTOCOL_GATE.md`.
