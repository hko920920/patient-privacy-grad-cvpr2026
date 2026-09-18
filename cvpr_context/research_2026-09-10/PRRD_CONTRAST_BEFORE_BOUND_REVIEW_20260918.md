# PRRD contrast-before-bound proposal: objective review — 2026-09-18

Status: design analysis of the user-pasted proposal; no adoption of a replacement method, no new patient experiment for this review.

The ongoing v3 public runtime did not pass W1. P-only source/receiver projections were produced, but the BioViL gradient comparison failed its declared numerical tolerance. No full128 timed update, W2 bank, Q/V pixel extraction, DP release, expert or reserved evaluation has occurred. This implementation issue is distinct from the scientific proposal below.

## Decision

The proposal improves the specificity of the failure mechanism: nonlinear endpoint bounding and within-patient contrast do not commute. Its translation-invariant relation query and sqrt(3) add/remove sensitivity are correct under fixed public feature transforms and clipping scale.

It does not yet establish a uniquely better algorithm, clinical utility, or a reason to discard all private pointwise information. Treat it as a concrete mechanism refinement to PRRD, not a proven replacement for v3. Keep prior results and v3 preparation immutable; no automatic 15-bank or new relation-only run.

## 1. Correct claims and their assumptions

Let h(x) be the selected fixed representation before endpoint-wise nonlinear normalization, and let Pi be a fixed public linear map. For a mixed patient:
d_i = Pi(mean_positive h - mean_negative h)/2,
r_i = d_i/max(R, ||d_i||), R>0 public and fixed.

Adding the same a_i to all h(x) for patient i leaves d_i and r_i unchanged. This holds for the chosen feature-space transformation, not arbitrary pixel perturbations or all forms of patient identity. A public center cancels in the difference. If h already contains L2 normalization, the property applies only to additive shifts AFTER that normalization, not to shifts of the earlier raw embedding.

q_i = b_i [1,r_i,svec(r_i r_i^T)] has norm squared <=3. The relation-only sum has add/remove sensitivity sqrt(3); replacement <=2sqrt(3). At d16 it has153 coordinates. Counts must be protected if private; normalization floors and empty mixed populations must be defined. The same sensitivity is available to any method using the same bounded relation query.

Gaussian release and postprocessing preserve this conditional shift invariance when every private dependency is through this query. Reusing nonDP private initialization, hyperparameters, per-patient scales or data-dependent output decisions would invalidate the claimed whole-pipeline statement.

## 2. The supplied symmetric example is mainly a scale example

For endpoints (M,1),(M,-1), normalization-first yields delta=(0,1/sqrt(M^2+1)). Unit-normalizing that nonzero delta BEFORE the DP release gives (0,1), exactly the same as contrast-first with R=1.

Thus this example cannot distinguish contrast-first from a strong pre-release recalibrated baseline. It can distinguish it from an uncorrected fixed-scale baseline. Rescaling a noisy release AFTER noise instead scales both signal and noise; that is a different comparison.

The user's report already includes recalibration. Its main motivating example should therefore not be presented as if it defeated that included baseline.

## 3. A stronger irreversible-loss construction

Take h+ = (11,0), h- = (9,0). Their individually unit-normalized endpoints are both (1,0). Normalization-first produces delta=0. Changing which endpoint is positive also leaves the normalized image/label feature distribution unchanged. Every algorithm that only receives these normalized endpoints has the same available information in both worlds, including after adding the same-law DP noise.

Contrast-first produces (1,0) or (-1,0). Post-normalizing zero cannot restore the missing sign.

This is a precise information collision, stronger than mere amplitude reduction, and does supply a conditional problem–repair argument. It is not a lower bound for all pointwise DP methods: access to raw norms, unclipped raw features, different feature maps, or patient-first statistics can avoid the restricted information loss.

A less degenerate constructed example uses a=(100,0), d=(1,1), endpoints a+d and a-d. Even after normalizing the endpoint-first delta, its direction differs from the raw delta by44.99427 degrees.

These checks used constructed arrays only, not NIH features. The clinical question is whether the discarded radial/directional components contain transferable label signal, rather than imaging intensity, irrelevant variation or label artifacts.

## 4. Unit normalization and bounded clipping are different

Current v1 settings explicitly used raw L2 normalization -> public PCA -> L2 normalization.

The incoming v3 master uses a/max(r_P,||a||), which is linear below the public radius. If both endpoints stay below r_P, their bounded-feature difference equals their raw difference divided by r_P, and additive common shifts cancel while remaining within that region.

The local v3 image wrapper ALSO applies F.normalize to BioViL projected_global_embedding BEFORE PCA (encoders.py raw()). Therefore endpoint nonlinearity remains in the concrete implementation, but the new proposal's DINO/v1 description is not a precise description of every v3 stage. Raw pre-L2 BioViL embeddings and their norms have not been cached as Q statistics; the stored P features are normalized. One cannot claim measured Q contrast destruction from those assets.

To compare orders fairly, fix the same pre-normalization feature layer and the same public linear transform. Changing encoder, feature layer, dimension, norming, private access and objective together would not isolate the order.

## 5. Relation-summary invariance is not single-image prediction invariance

Even perfect d_i estimates do not guarantee the desired marginal classification result.

Construct scalar features x_i+ = a_i+1 and x_i- = a_i-1. Every patient has exactly d_i=1, independent of their common background. The relation summary is perfect. For independent patient backgrounds U,V~Uniform[0,L], any positive scalar linear score has:
AUROC = P(U+1>V-1) = 1/2+2/L-2/L^2, L>=2.

At L=100 this is .5198, despite exact contrast recovery. At L=1000 it is .501998. This is a constructed limitation, not a forecast of NIH performance.

The theorem protects the TRAINING summary against a nuisance shift. It does not make w^T h(x) at TEST time invariant to w^T a_i. Within-patient ranking, prediction of change using two visits, and single-image disease ranking across patients are distinct endpoints.

Do not strengthen the theorem into causal disease isolation or universal AUROC superiority. Conditions on nuisance/task geometry or actual evidence of marginal predictive value are needed for the latter.

## 6. The closest strong alternatives already aggregate before clipping

[DP-MEPF official code](https://github.com/ParkLabML/DP-MEPF/blob/52e504d93d5f3c5bce8e627d1eb1ea8991dd29ea/code/dp_functions.py) supports both norm and clip, including per-layer variants. [Its feature extraction](https://github.com/ParkLabML/DP-MEPF/blob/52e504d93d5f3c5bce8e627d1eb1ea8991dd29ea/code/feature_matching.py) bounds features before label aggregation. A legitimate endpoint-first comparator exists, but an always-unit-normalized weak version must not stand in for all clipping alternatives.

[Dosser §3.2](https://arxiv.org/html/2508.01749v1) extracts per-image signals, clips, aggregates and privatizes them. Its source setup is not already the patient contrast problem.

[Person-level private mean estimation](https://arxiv.org/html/2405.20405v2) explicitly averages samples within each person before clipping and noise. [User-level DP training](https://arxiv.org/html/1710.06963) clips user updates, not necessarily individual images. These do not already implement this visual distillation method, but refute a blanket claim that existing DP fundamentally requires endpoint-first clipping.

For a mixed patient with class-balanced squared loss
L_i(w)=1/4 [mean_+(w^T h-1)^2+mean_-(w^T h+1)^2],
gradient at w=0 is -d_i. Clipping that patient gradient is the same bounded contrast up to sign and the fixed R scale. This identity was checked. Its first/second moments reproduce the same relation query. This is an important equivalence, not an artificially weaker competitor.

CovMatch/LGM remain relevant to moment-based relational learning and visual transfer. This review does not certify novelty against all literature. SciSpace was used to locate priors; claims above were checked against originals/code rather than relying on its abstracts alone.

## 7. Removing private pointwise statistics is an additional intervention

v3 uses Q2027 point information plus Q102 mixed relation information. The new relation-only proposal discards Q pointwise information.

Of Q2027,102 are mixed,12 have only positive-labeled images, and1913 have only negative-labeled images. Most positive patients114 are still represented in the relation subgroup, but the broader negative and nonmixed information is absent. The dimension/sensitivity reductions from459/3 to153/sqrt(3) are partly consequences of releasing less information, not proof of superior protection at unchanged utility.

Full-output shift invariance requires excluding other shift-dependent private channels. Relation-channel invariance does not: one can retain a COMMON private pointwise block across all three order variants and restrict the theorem to the relation block.

Recommended comparison if this mechanism is adopted:
1 endpoint bound -> difference;
2 endpoint bound -> difference -> relation re-bounding/recalibration;
3 raw difference -> relation bound.
Keep private point information, feature layer, public Pi, total query dimension, sensitivity/accounting, image budget and recipient contract common. A relation-only variant can be separate, with its information reduction disclosed.

This is a proposal for a prospective amendment, not a change applied to current v3 or permission to run a larger grid. Standard delta-feature matching with the same operations is method3 itself and must be acknowledged as such.

## 8. Transfer coefficients and privacy boundary

Any exported synthetic-pair coefficient must be a function only of public inputs and the released DP summary/synthetic artifacts. Exporting actual per-private-patient inverse norms or their identifiable pairing is an additional release requiring protection; synthetic coefficients must not be copied from patient records.

For transfer analysis, a source-to-recipient linear relation of deltas and shared synthetic coefficients can support a conditional moment bound. It is not guaranteed for BioViL/DenseNet, especially with independently fitted nonlinear feature bounds. Recomputing a new coefficient in the recipient changes that stated relation and must have a separate learner contract.

Public P contains only3 actual mixed patients. R may be fixed or derived from allowed public statistics, but a calibration claimed to be based on many public longitudinal contrasts would be false here. No private tuning can be treated as free public calibration.

## 9. Practical judgment

Accept the narrower mechanism insight: pre-contrast nonlinear feature bounding can irreversibly discard relation information; contrast-first avoids that specific loss under the specified assumptions.

Do not accept as established: that this explains old head/LoRA failures, that NIH task signal is substantially lost in this way, that the new method beats patient-first gradient/moment alternatives, that relation-only dominates full-Q methods, or that summary invariance implies image/recipient utility.

The contribution target can be sharpened to the task-relevant information lost BEFORE privatization, and its preservation at matched privacy/output cost. Its value must be evaluated against recalibrated/centered endpoint methods and equivalent patient-first methods, not against a baseline made unnecessarily weak.

This is stronger mechanistic reasoning than the earlier general relation-value hypothesis, but not yet a unique advantage over the strongest alternatives. It is a concrete refinement worth assessing within PRRD, not a reason to switch names or promise a better success probability.

## Artifacts and execution boundary

- [Constructed-array checks](contrast_before_bound_review_20260918.py).
- [Actual CPU results](spec_sources/contrast_before_bound_review_checks_20260918.json): shift-invariance error2.6368e-15, patient-gradient identity error0; no patient/model/DP execution in this review.
- Previous public W0/W1 technical artifacts: code_working/_reports/prrd_v3_20260918_w01.
- Last actual efficacy and Rwide result files remain unchanged.
- The 15 main banks and 45 RN18 compatibility runs, new DP, expert and reserved remain unexecuted. The primary bank count remains 15; it has not been reduced based on outcomes.
