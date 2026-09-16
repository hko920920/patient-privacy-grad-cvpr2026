# Patient-set diffusion premise checks

This directory evaluates whether repeated X-rays support a set-valued patient-private diffusion problem.
It is separate from the retired gradient-allocation and UCAN branches.

- `NIH_CXR14_PATIENT_SET_PREMISE_PROTOCOL_V1.md`: frozen public-data premise gate.
- `run_nih_cxr14_patient_set_premise.py`: DINOv2 geometry and finding-diversity diagnostic.
- `NIH_CXR14_PATIENT_SET_PREMISE_CONFIRMATION_PROTOCOL_V2.md`: frozen independent confirmation on
  exactly two- and three-record patients, using only scale-free ordering/retrieval criteria.
- `run_nih_cxr14_patient_set_premise_confirmation.py`: balanced-cell v2 confirmation runner.
- `test_patient_set_confirmation.py`: deterministic selection, AUC, and retrieval-chance tests.
- `NIH_CXR14_SETADAPTER_ARCHITECTURE_SMOKE_PROTOCOL_V1.md`: frozen integrated Q=2 execution gate.
- `set_adapter.py`: zero-init, permutation-equivariant, mask-aware mid-block SetAdapter.
- `test_set_adapter.py`: identity, equivariance, padding, singleton, and parameter-count tests.
- `run_nih_cxr14_setadapter_architecture_smoke.py`: real public-X-ray SD 2.1 Q=2 forward/backward,
  patient-vector clip-operator, cross-record connectivity, and CUDA-memory runner.
- `NIH_CXR14_SETADAPTER_CONTEXT_SIGNAL_PROTOCOL_V1.md`: frozen correct-vs-shuffled held-out
  learning-signal gate.
- `run_nih_cxr14_setadapter_context_signal.py`: 192-step public non-DP SetAdapter runner.
- `test_context_signal.py`: deterministic schedule, selection, shuffle, and metric tests.
- `NIH_CXR14_CROSS_RECORD_CONTEXT_CONFIRMATION_PROTOCOL_V2.md`: one-revision fresh-validation
  confirmation and stop rule.
- `cross_record_adapter.py`: strict leave-one-record-out adapter used only to isolate the v1 self-path
  explanation; not a standalone novelty claim.
- `run_nih_cxr14_cross_record_context_confirmation.py`: v2 confirmation runner.
- `test_cross_record_adapter.py`: no-self-value, zero-init, permutation, mask, and singleton tests.

The premise diagnostics and architecture smoke perform no model training and save no embeddings.  The two
context-signal diagnostics briefly train public non-DP adapters but save no checkpoint, latent, gradient,
optimizer state, prediction, or adapter.

Both held-out context gates failed.  V1 obtained correct/shuffled `1.0002136004` with `0.5000` wins; the
strict cross-record-only v2 obtained `1.0010702866` with `0.5078125` wins on fresh validation.  Both adapters
improved over bypass but did not prefer the same-patient companion.  Under the frozen stop rule, the current
`standard IID denoising + pooled mid-block patient context` method class is closed and no matched generation
experiment is licensed.  The historical data-premise and architecture PASS results remain valid only within
their original scopes.
