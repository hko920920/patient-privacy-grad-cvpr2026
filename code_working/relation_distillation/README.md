# Patient relation distillation — preparation v1

This isolated package implements feature-level mathematics and a prospective
A/B/C/D contract. It is **not an end-to-end medical image runner yet**.

- `settings_v1.json`: fixed initial recipe, 12 banks, paired-encoder reuse and
  secondary 36-run BCE compatibility evaluation; DP remains disabled.
- `moments.py`: patient-uniform class moments, bounded joint contributions,
  private pairing control, noisy-sum normalization, optional repair and readouts.
- `objective.py`: differentiable whole-bank matching and replay VJP interface.
- `prepare.py`: metadata/public-weight hash binding, no image decoding or model.
- `verify_preparation.py`: binding verification and independent CPU array tests.

From `code_working`, use `base_gate/.venv/Scripts/python.exe -B -m unittest
relation_distillation.test_core -v`. The one-time metadata preparation command is
`base_gate/.venv/Scripts/python.exe -B -m relation_distillation.prepare`.
Then run `base_gate/.venv/Scripts/python.exe -B -m relation_distillation.verify_preparation`.
Preparation artifacts cannot be overwritten by rerunning these commands.

The authoritative design is
`CVPR 주제 탐색/research_2026-09-10/TRACK1_PATIENT_RELATION_DISTILLATION_PROTOCOL_20260918.md`.
No historical experiment code or role ledger is replaced by this package.
