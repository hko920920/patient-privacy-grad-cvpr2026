# Real-patient support diagnostic plan

This package prepares metadata only. It does not implement or run classifier training.
One expanded real-data pool is frozen before observing its model outcomes:
original public813 plus all5097 former classifier-selection images.

The additional2027 patients retain their original private_train provenance.
The planned arm is diagnostic NON-DP and is not an expanded public baseline.
Original role tables remain unchanged; a separate planned-role overlay is created.

Prepare and verify once, before updating the research state:

    python -B -m real_support_plan.prepare
    python -B -m real_support_plan.verify

Outputs: _reports/real_support_plan_20260917_v1.
These commands reject overwrites. The report-level verifier checks the completed
plan after the current research-state pointers have been updated.

Future runtime work must retain the original training kernel and pass separate
R1 provider parity and Rwide replay checks. No actual Rwide performance is available.
