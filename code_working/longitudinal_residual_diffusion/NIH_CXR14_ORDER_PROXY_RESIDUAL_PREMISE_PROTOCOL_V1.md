# NIH CXR14 order-proxy longitudinal residual premise v1

- Frozen on: 2026-09-03 before cohort selection or feature outcomes
- Scope: public, non-DP, feature-space premise diagnostic only
- Not in scope: image generation, actual elapsed time, causal disease progression, clinical prediction,
  privacy guarantee, or model utility

## Question

On patient-disjoint held-out ordered CXR pairs, does a transition-conditioned residual learned from other
patients improve future-feature prediction beyond copying the previous record, while a same-patient previous
record remains materially better than a finding-matched previous record from another patient?

This is the minimum premise for a directed longitudinal residual transition.  It does not test a diffusion
generator.  Failure prevents a NIH-based generation pilot; success licenses only a timestamp/report data-access
gate and a separately frozen public generation protocol.

## Metadata boundary

- NIH ChestXray14 `Follow-up #` is treated only as an ordering index.
- The local metadata has no study date/time, report, medication, laboratory event, encounter type, or validated
  clinical progression interval.
- “Consecutive” below means `current Follow-up # = previous Follow-up # + 1`, not consecutive days or visits.
- NLP-derived finding labels are weak transition proxies, not radiologist-adjudicated change labels.

## Frozen exclusions and cohort

- Source: PA-only K10 `public_development` manifest.
- Exclude the union of all patients in these six prior selections: public clip calibration, UCAN, patient-set
  premise v1, patient-set confirmation v2, SetAdapter context v1, and cross-record context v2.  The expected
  union is 528 unique patients.
- For every remaining patient, enumerate locally available ordered pairs with exact follow-up-index gap one.
- A pair is `no_change` when the canonical full finding-label strings are equal and `change` otherwise.
- Within each patient and class, retain the pair with the smallest SHA-256 rank under salt
  `nih-cxr14-order-proxy-residual-premise-v1-pair`.
- Because no-change patients are scarcer, rank them first under salt
  `nih-cxr14-order-proxy-residual-premise-v1-patient-no-change` and select 144.  Remove those patients, rank
  remaining change candidates under salt `nih-cxr14-order-proxy-residual-premise-v1-patient-change`, and select
  144.
- In each class, the first 96 ranked patients are training and the next 48 are validation.
- Total: 288 unique patients, 288 ordered pairs, and 576 unique images.  Train/validation and the two transition
  classes are patient-disjoint.
- No feature, outcome, label prevalence, age, or target flag enters the hash ranking.

## Frozen encoders

Run the complete analysis independently in both feature spaces.

1. Generic DINOv2 ViT-B/14, exact locally pinned weights already used by the patient-set premise.  Use the
   existing deterministic 518 preprocessing and L2-normalize its 768-dimensional output.
2. Microsoft RAD-DINO, exact revision `110cbc18d5133582e320b43d53bf5c44e410c936`.  Decode through the locked
   P256 grayscale loader, replicate RGB, use the pinned slow image processor, take the 768-dimensional pooler
   output, and L2-normalize it.

Do not save image features.  Save only aggregate metrics, frozen commitments, and the local selection CSV.

## Frozen transition representation and regressors

- For each pair, encode the weak-label transition as 30 binary variables: 15 additions followed by 15 removals,
  using the 14 ChestXray14 findings plus `No Finding` as an ordinary explicit label.
- The response is `future_feature - prior_feature`.
- Fit a multi-output ridge regression with an intercept on all 192 training pairs.  Standardize nonconstant input
  columns using training statistics only.  Choose alpha from `{0.01, 0.1, 1, 10, 100}` by five patient-hash
  folds and minimum mean held-out per-pair residual MSE; break ties toward the larger alpha.  Refit on all train
  pairs.
- Build a deranged-label control by a fixed cyclic permutation of transition rows within each training class;
  no row may keep its own transition vector when a nonidentical derangement exists.  Apply the identical
  preprocessing, alpha grid, cross-validation, and refit procedure.
- Prediction arms on validation are:
  1. `copy`: previous feature, equivalent to a zero residual;
  2. `true_transition`: previous feature plus the true-transition ridge prediction;
  3. `deranged_training`: previous feature plus the deranged-training ridge prediction;
  4. `shuffled_prior`: a different patient's previous feature plus the true-transition prediction.
- For `shuffled_prior`, remain within the same validation transition class and prefer same sex, then maximize
  the sum of previous-label and future-label Jaccard similarities.  Resolve ties only by the frozen hash.  The
  current future feature and transition condition remain fixed.

## Frozen metrics

- Per-pair error is mean squared error over 768 future-feature coordinates.
- Ratios are ratio of arithmetic mean paired errors, not mean per-pair ratios.
- A win is strict lower paired error; exact ties are not wins.
- For each ratio, compute a 5,000-replicate patient bootstrap percentile 95% interval using fixed
  PCG64-derived encoder/contrast seeds.
- Report change and no-change validation classes separately for each encoder.

## Frozen conjunctive gate

All conditions must hold in **both** DINOv2 and RAD-DINO spaces.

For the 48 held-out `change` pairs:

1. `true_transition / copy <= 0.995`, bootstrap upper 95% bound `< 1.0`, and true-transition wins `>= 0.60`;
2. `true_transition / deranged_training <= 0.995`, bootstrap upper bound `< 1.0`, and true-transition wins
   `>= 0.60`;
3. `true_transition(correct prior) / shuffled_prior <= 0.95`, bootstrap upper bound `< 1.0`, and correct-prior
   wins `>= 0.75`.

For the 48 held-out `no_change` pairs:

4. `true_transition / copy <= 1.01`; this prevents a transition model from materially damaging the negative
   control.

Integrity conditions:

5. exact cohort counts, exclusions, pair order, feature shapes, finite values, nonzero row norms, deterministic
   eight-image replay max error `<=1e-5`, and zero patient overlap;
6. no checkpoint, learned regressor, image feature, latent, gradient, optimizer state, or generated image is
   retained; peak CUDA allocation remains below `7.5 GiB`.

If any substantive condition fails, retain the full result and mark
`FAIL_NIH_ORDER_PROXY_RESIDUAL_PREMISE`.  Do not rescue it by changing the encoder, alpha grid, pair class,
threshold, or follow-up gap.  Such a failure closes the present NIH weak-label PPLRD route; it does not prove
that richer MIMIC EHR conditioning cannot work, but without that data locally it blocks progression.

If all conditions pass, mark `PASS_NIH_ORDER_PROXY_RESIDUAL_PREMISE`.  A pass is still not a generation,
privacy, clinical, or CVPR result.  The next gate is evidence of access to timestamp+report longitudinal data.
