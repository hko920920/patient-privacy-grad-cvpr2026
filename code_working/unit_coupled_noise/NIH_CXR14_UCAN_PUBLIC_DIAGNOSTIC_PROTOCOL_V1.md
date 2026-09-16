# NIH CXR14 unit-coupled antithetic-noise public diagnostic v1

- Frozen on: 2026-09-03, before any UCA-noise X-ray gradient outcome was computed
- Scope: public-development gradients only; no optimizer, DP noise, private train, attack, generation, or release claim
- Working method name: **Unit-Coupled Antithetic Noise (UCAN)**

## 1. Question

For a patient privacy unit with repeated radiographs, can dependence among the *training corruption noises*
reduce the stochastic error of the clipped patient update without changing any per-record DDPM marginal,
the number of UNet evaluations, or the outer patient-DP mechanism?

This is not the retired timestep-allocation branch.  Pairing is only within one patient and all comparisons
use fresh NIH gradients.  No randomness is coupled across patients.

## 2. Construction

Four frozen records of patient `u` are paired as `(0,1)` and `(2,3)`.  For a pair, UCAN draws
`z ~ N(0,I)` and uses `(z,-z)`.  Each record therefore still receives a standard Gaussian marginal.
The primary implementation also shares the uniformly sampled timestep inside a pair; a factorial arm keeps
timesteps independent to separate the effects.

The patient loss remains the arithmetic mean of four per-record denoising losses.  Its gradient is computed,
clipped once at the already frozen `C_patient=0.1997973088974048`, and only then enters a patient batch.
Since every patient vector is still bounded by `C_patient`, replacing independent internal randomness by this
data-independent within-unit coupling does not change the sensitivity or accountant of native patient-DP.

## 3. Frozen cohort and compute

- Source: NIH ChestXray14 K10 `public_development` only.
- Eligible: PA-only selected patients with at least four K10 records.
- Exclude every patient used by the earlier public clip-calibration image or patient units.
- Select by SHA-256 ranking with salt `nih-cxr14-ucan-public-diagnostic-v1`.
- Cohort: 8 target and 8 control patients; four records per patient by a second frozen hash ranking.
- Model: pinned SD 2.1 base, rank-8 attention LoRA, 1,659,904 fp32 trainable scalars, P256 preprocessing.
- Banks: `bank_01` through `bank_05`.
- Every arm: exactly four record-level UNet evaluations per patient and bank.

## 4. Factorial arms

| Arm | timestep inside each record pair | corruption noise inside pair | Role |
|---|---|---|---|
| `iid_independent_t` | independent marginals | independent | ordinary native patient-gradient baseline |
| `antithetic_independent_t` | independent marginals | `(z,-z)` | isolates noise coupling without timestep coupling |
| `iid_pair_shared_t` | shared uniform marginal | independent | controls for timestep sharing |
| `antithetic_pair_shared_t` | shared uniform marginal | `(z,-z)` | complete UCAN treatment |

The same first noise draw and the same marginal timesteps are reused across matched arms where possible.
This pairing reduces comparison noise but does not change an arm's marginal law.

## 5. Outcome computed from full gradients

For method `a`, bank `b`, and patient `u`, let `h(a,b,u)` be the once-clipped full LoRA gradient.  For an
unordered patient pair `(u,v)`, the actual `B=2` pre-DP update is

`H(a,b,u,v) = (h(a,b,u) + h(a,b,v))/2`.

For every one of 120 patient pairs and 10 unordered bank pairs, estimate stochastic trace error by

`E = ||H(a,b1,u,v)-H(a,b2,u,v)||^2 / 2`.

This pairwise-replicate estimator needs no post hoc reference gradient and retains all cross-patient inner
products.  The primary ratio is the arithmetic mean `E(antithetic_pair_shared_t) / E(iid_pair_shared_t)`.
The total-pipeline ratio against `iid_independent_t`, the independent-t antithetic ratio, unclipped/clipped
patient marginal ratios, gradient norms, losses, and clipping rates are secondary diagnostics.

No raw gradient tensor is retained after Gram-based statistics are computed.

## 6. Frozen promotion gate

Promote UCAN to a short X-ray training pilot only if all conditions hold:

1. primary actual-`B=2` ratio is at most `0.95`;
2. deterministic 10,000-draw patient-pair × bank-pair bootstrap 95% upper endpoint is below `1.0`;
3. at least 4 of 5 leave-one-bank-out primary ratios are below `1.0`;
4. at least 14 of 16 leave-one-patient-out primary ratios are below `1.0`;
5. complete UCAN has actual-`B=2` point ratio below `1.0` versus ordinary `iid_independent_t`;
6. its empirical patient clipping fraction is no more than 0.05 absolute above `iid_pair_shared_t`.

Failing any item closes this implementation as a training candidate.  A shared-t-only success cannot be
reported as proof for antithetic noise.  Thresholds will not be relaxed, patients/banks will not be removed,
and secondary `C` values will not replace the primary decision.

## 7. What a pass would and would not mean

A pass would support only the mechanism-level claim that within-patient joint corruption randomness has
enough full-gradient headroom to justify a bounded training pilot.  It would not establish generation quality,
medical validity, privacy beyond the already specified patient-DP mechanism, attack resistance, novelty, or a
CVPR-ready paper.  Those claims require frozen patient-DP training, visual/medical evaluation, and an expanded
related-work audit.

