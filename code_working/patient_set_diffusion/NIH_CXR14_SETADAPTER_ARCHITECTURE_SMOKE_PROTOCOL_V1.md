# NIH CXR14 Q=2 SetAdapter architecture smoke v1

- Frozen on: 2026-09-03 before the integrated SD 2.1 outcomes were computed
- Prerequisite: patient-set premise confirmation v2 passed all five scale-free gates
- Scope: public architecture feasibility only; no utility, generation, privacy, attack, or clinical claim

## Frozen public input

- Verify the K10 manifest and the premise-confirmation v2 selection file, SHA-256
  `E1ABC33EA68082F69086CDFB752B558580710FAD35FB70640BDAD590DFC2C6F8`.
- Within the v2 `(target_patient=0, record_count=2)` cell, select one patient by SHA-256 salt
  `nih-cxr14-q2-setadapter-architecture-smoke-v1`; use both records.
- Use the exact locked NIH preprocessing and deterministic VAE mode latents.
- Draw each record's diffusion timestep and corruption noise independently from separate hash-derived public
  seeds.  No antithetic or shared-t rule is used.

## Frozen model and adapter

- Exact local SD 2.1 base snapshot at revision `0094d483a120f3f33dafbd187ea4aa60d10de75c`.
- Freeze the base UNet and add the existing rank-8 LoRA to `to_q`, `to_k`, `to_v`, and `to_out.0`:
  1,659,904 fp32 trainable scalars.
- Wrap the UNet mid block with one `PatientSetAdapter`: spatial mean pool, LayerNorm, 128-dimensional shared
  record token, four-head self-attention without slot positional embeddings, a 2x token FFN, and a
  token-to-1,280-channel residual projection.
- The residual projection is initialized to exactly zero.  Valid masks remove padded records; patients with
  one valid record take an exact identity path.  Slot order is randomized by data loading in later work;
  this smoke directly verifies permutation equivariance.
- Expected SetAdapter size: 463,872 fp32 trainable scalars; expected LoRA+SetAdapter total: 2,123,776.

## Frozen checks

1. The pure adapter test suite must verify exact zero-init identity, permutation equivariance after a nonzero
   probe activation, padded-slot isolation, singleton identity, deterministic cohort selection, AUC, and
   exact retrieval-chance calculations.
2. In the integrated Q=2 UNet, zero-initialized SetAdapter output must match an explicit mid-block bypass
   with max absolute prediction difference at most `1e-6`.
3. The mean two-record denoising loss must backpropagate once through the full model.  All LoRA and
   SetAdapter gradient tensors must be present and finite; LoRA norm, total SetAdapter norm, and the
   zero-initialized output-projection gradient norm must each be greater than zero.
4. Flattening LoRA+SetAdapter as one patient vector and clipping it to the diagnostic norm `1.0` must produce
   a finite vector with norm at most `1.00001`.  This is an operator smoke, not a calibrated DP clip norm.
5. Without an optimizer step, temporarily fill only the output projection with deterministic zero-mean
   Gaussian probe weights of standard deviation `0.01`.  The full UNet must remain permutation-equivariant
   under swapping the two records with max absolute error at most `0.005`, and changing only record 2 must
   change record 1's output by L2 greater than `1e-6`.
6. With that nonzero probe active, Q=1 SetAdapter output must still match explicit bypass with max absolute
   error at most `1e-6`.
7. Peak allocated CUDA memory after model load must remain below `7.5 GiB`; every prediction/loss/gradient
   must be finite.

All checks are conjunctive.  The runner performs no optimizer step, adds no DP noise, stores no gradient,
latent, adapter, or checkpoint, and discards the model.  A pass licenses only a parameter/UNet-call-matched
public non-DP falsification protocol; it does not show that the adapter improves generated patient sets.
