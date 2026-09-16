# Longitudinal residual diffusion premise checks

This directory contains falsification-first diagnostics for the conditional successor to the closed
standard-IID pooled-context PSPD method.

- `NIH_CXR14_ORDER_PROXY_RESIDUAL_PREMISE_PROTOCOL_V1.md`: frozen public feature-space premise gate.
- `run_nih_cxr14_order_proxy_residual_premise.py`: patient-disjoint DINOv2/RAD-DINO residual runner.
- `test_order_proxy_residual_premise.py`: selection, transition, ridge, derangement, bootstrap, and shuffle tests.

NIH `Follow-up #` is only an ordering proxy.  Nothing in this directory licenses actual-time, causal,
clinical, generation, or privacy claims.

The frozen v1 gate failed in both encoders.  On 48 changed-label validation pairs, true residual over prior
copy was `1.140914543` in DINOv2 and `1.159127120` in RAD-DINO.  Correct same-patient prior over a
finding-matched shuffled prior was strongly favorable (`0.516104650` and `0.319953448`), so the failure
separates a strong identity/anatomy carrier from an unsupported weak-label change direction.  Status is
`FAIL_NIH_ORDER_PROXY_RESIDUAL_PREMISE`; the present NIH PPLRD route is closed without a generation pilot.
