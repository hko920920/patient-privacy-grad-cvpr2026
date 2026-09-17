# Frozen-classifier generalization diagnostic

Inference only: existing R1, Dreal, L_public and L_pooled classifiers, seeds11/23/37.
The runtime source, protocol, twelve checkpoints, traces and existing development
predictions are bound before new inference.

From code_working, using base_gate/.venv/Scripts/python.exe:

    python -B -m classifier_generalization.run prepare
    python -B -m classifier_generalization.run infer
    python -B -m classifier_generalization.verify

The completed output is _reports/classifier_generalization_20260917_v1.
These commands intentionally reject silent overwriting/re-execution.
Existing checkpoints, including all BatchNorm buffers, stay unchanged.
There is no optimizer, backward, generation, final/reserved access or threshold tuning.

Actually seen training images are deduplicated per checkpoint from the original trace,
and real versus synthetic sources are reported separately. Synthetic labels are
requested conditions, not expert annotations. The independent verifier reconstructs
raw/PNG input pixels and normalization and recomputes metrics without importing the
production metric function. Integrity verification is not new private utility evidence.
