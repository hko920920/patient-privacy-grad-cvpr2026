# Patient-Relational Distillation under User-Level Differential Privacy

Working manuscript — 2026-09-18. Methods draft; medical utility and DP experiments pending. No performance or acceptance claims are asserted.

## Abstract — results pending

Repeated medical imaging provides patient-level relationships that a flat image-label summary may not retain. We study whether private within-patient positive/negative centroid contrasts can complement pointwise information and be transferred to other visual learners through a small synthetic image set with virtual pairs. The proposed construction retains full-cohort pointwise statistics, bounds patient-local relational contributions, and uses separate pointwise and paired synthetic banks. A guarded quadratic learner limits deviation from the pointwise objective, while synthesis matches both statistics and the learned source scoring function. Experiments are specified to separate private pointwise value, true patient correspondence, normalization order, the two optimization additions, cross-encoder reuse, and patient-level privacy cost.

[RESULTS_PENDING: measured effects, uncertainty, strongest baseline, privacy and total cost. Replace this marker only with recorded evidence.]

## 1. Introduction

The intended user is an institution permitted to analyze a protected patient cohort. A recipient receives synthetic images, target labels, virtual pair indices and a learning contract, and trains in its own feature space without receiving the original patient images or unprotected moments.

The central question is whether within-patient label contrasts provide useful learning information beyond pointwise summaries, and whether that incremental value survives image realization, a change of encoder, and patient-level DP. Neither the existence of repeated examinations nor an improvement from adding real data establishes that claim. In the current development record, the real-data expansion Rwide improves a fixed classifier, but it did not use the proposed relation learner or protected synthetic images.

Three distinctions guide the study: preserving a specified relation objective is different from improving single-image prediction; source fidelity is different from cross-encoder reuse; and synthetic appearance is different from a formal privacy guarantee.

## 2. Related work — scope fixed, baseline cards pending

Feature/gradient matching, kernel-based distillation, second-order relational matching and private signal extraction are existing principles. The study does not claim their invention. LGM is relevant to frozen visual representations and transfer, CovMatch to relation statistics and learning objectives, and DP-KIP, DP-MEPF and Dosser to private distillation. Analytic Gaussian calibration supplies the standard release mechanism.

A standard patient-difference moment matcher with the same feature path, bounding, release and objective can coincide with our relation computation. Such an implementation must be acknowledged as equivalent rather than labeled an artificial competing method. R_joint is a patient endpoint joint-moment comparator, not a reproduction of image-text CovMatch.

[BASELINE_CARDS_PENDING: exact code versions, patient-level adaptations, access and tuning budgets.]

## 3. Method

### 3.1 Data and feature contract

Public P contains672patients/813images; protected Q contains2027/5097; reused development V contains2026/5047. Q has102 mixed-label patients and P has3. Expert532/810 and Reserved4213 remain closed at this draft stage. Former classifier-selection Q retains its private_train provenance.

The source is the official BioViL-T image model with P-only PCA16. DenseNet121 ImageNet1K V1 is the primary recipient with P-only PCA128; a reserved ViT-B/16 is a confirmation recipient. The point feature path is unchanged. A raw affine relation path uses the same stored public axes before the external feature normalization. Recipient features and relation scales are computed in the recipient's own space; source relation vectors are not transferred.

### 3.2 Patient-local summaries and banks

Within each patient/class, average the feature and its outer product; then aggregate with equal mass per patient possessing that class. Point moments \((m_0,A_0,m_1,A_1)\) use the entire permitted cohort. Mixed-patient relation moments \((\mu_r,C_r)\) use a separately normalized population.

Contrast-first C forms a raw class-centroid difference and then bounds it. E and E_R retain endpoint-first processing, with E_R receiving public rescaling. D deranges only Q endpoints before bounding. Its point moments stay fixed, but after nonlinear bounding both relation mean and second moment may change. R_joint bounds joint endpoints before applying the linear difference map.

A/B export128 marginal images. Relational methods export64 marginal images and32 virtual pairs. The point and relational risk use their respective banks. Every recipient reads the exported8-bit PNGs rather than the pre-export floating-point images.

### 3.3 Guarded relational learner

For point moments, define \(G=(A_0+A_1)/2\), \(v_0=(m_1-m_0)/2\), \(M=G+\lambda I\), \(\lambda=0.1\). Then
\[
w_0=M^{-1}v_0,\qquad
w_1=(M+\beta C_r)^{-1}(v_0+\beta\mu_r),\quad\beta=1.
\]
The relation term is an auxiliary quadratic objective; it is not asserted to equal the score difference of normalized point features.

Let \(d=w_1-w_0\), \(\rho^2=\kappa w_0^\top M w_0\), \(\kappa=0.1\). Set
\[
w_*=w_0+\min(1,\rho/\|d\|_M)d,
\]
with the zero-shift case defined as \(w_*=w_0\). These are prospective starting settings, not empirically selected optima. The same rule is applied to target, synthetic and recipient moments in their respective spaces.

### 3.4 Statistical and functional synthesis

From synthetic moments compute \(w_S\) with the same learning rule. Keep target weight and target point matrix fixed and minimize
\[
L_{\rm synth}=L_{\rm point}+\gamma L_{\rm relation}
+\eta(w_S-w_*)^\top M_T(w_S-w_*)+L_{\rm image},
\]
where \(\gamma=1,\eta=1\). Existing public image templates, multiresolution residuals, pixel/TV regularization and paired augmentation are retained. The functional term applies to the unaugmented learning function; the existing augmentation term remains statistical matching.

Point baselines also receive functional matching; all relation baselines receive the same guarded learning rule. The effects of these changes are separately ablated on C. Other encoders are excluded from synthesis, checkpoint choice and coefficient selection.

### 3.5 Patient-level privacy

A patient's point and relation blocks are bounded before concatenation, with full-query add/remove sensitivity at most3 (point-only bound \(\sqrt6\)). One protected Gaussian summary supports subsequent image optimization and recipients as postprocessing. Distinct releases require composition. A weak relation intervention does not refund its privacy cost.

Noisy count normalization and fixed positive-definite readout stabilization must be part of the DP contract. The guarded objective and functional metric use the same declared stabilized point system. The production sampler and accounting implementation are still pending; mathematical bounds and CPU tests do not certify a deployed DP mechanism.

## 4. Analysis

For the declared positive-definite point objective,
\[
F_0(w)-F_0(w_0)=\tfrac12\|w-w_0\|_M^2.
\]
Consequently the guard bounds the objective increase by \(\rho^2/2\), or by \(\kappa\) times the pointwise improvement over the zero vector under the relative-radius policy.

With \(e_{\rm dist}=\|w_S-w_*\|_{M_T}\),
\[
F_{0,T}(w_S)-F_{0,T}(w_0)
\le\tfrac12(\rho+e_{\rm dist})^2.
\]
These are source quadratic-objective properties, not AUROC or arbitrary-encoder guarantees. A conservative guard can suppress helpful changes. Identical point features remain indistinguishable by any linear readout even if a raw relation path retains a difference.

Common additive patient shifts cancel before nonlinear contrast bounding, conditional on the specified feature space. This is a relation-path property, not identity removal or invariance of the full image classifier. Endpoint normalization can attenuate or remove contrasts, but rescaling can repair some examples; E_R is therefore an essential comparator.

## 5. Experiments — fixed plan, results pending

The initial package contains7conditions×3initializations=21banks and9 additional C ablation banks, each128images and500updates. Source and primary-recipient evaluation includes108synthetic readout configurations and20trusted-real reference configurations; these counts are not inner linear-solve call counts. Image-only RN18 compatibility is limited to the core21banks×3seeds×400updates.

Primary evidence is recipient C−B and C−D; C−E_R assesses the proposed order change, and C−R_joint assesses the nearest relationship-preserving alternative. Patient-cluster paired bootstrap uses2,000development draws. Synthetic initialization, classifier seed, patient sampling and DP noise are reported as distinct variation sources.

Table1: [PENDING — access, learner rule and cost by arm.]
Table2: [PENDING — source/recipient AUROC and AP, all banks.]
Table3: [PENDING — cap/function2×2 ablation, alpha, fidelity and PNG loss.]
Table4: [PENDING — same-budget DP, direct/strong baselines and composition.]
Figure1: [PENDING — bounded patient summaries to two-bank recipient workflow.]
Figure2: [PENDING — paired cross-encoder effects and intervals.]
Figure3: [PENDING — privacy, utility and total measured cost.]

Actual implementation evidence available: six constructed-array CPU checks of the guarded learner pass. Actual BioViL W1 parity has not passed; no PRRD patient utility is reported. Previous Rwide numbers belong to a separate direct-real-data diagnostic and cannot populate the new method's result cells.

## 6. Limitations and scope

The cohort uses weak labels, only102private mixed patients and3public mixed patients, with reused development data. Relations summarize class centroids rather than time-ordered clinical changes. Treatment, acquisition and comorbid changes may contribute to the contrast. Source matching does not establish cross-encoder transfer or clinical validity. Final evidence must distinguish nonDP development choices from the privacy guarantee of a specified release.

## 7. Conclusion — to be completed from results

[CLAIM_PENDING: retain, narrow or reject each proposed advantage according to the frozen comparisons. Do not fill this section with anticipated positive outcomes.]

## Source references for the working draft

- [LGM](https://arxiv.org/html/2511.16674v1).
- [CovMatch](https://arxiv.org/html/2510.18583v1).
- [DP-KIP](https://arxiv.org/abs/2301.13389).
- [Dosser](https://arxiv.org/html/2508.01749v1).
- [Analytic Gaussian calibration](https://proceedings.mlr.press/v80/balle18a.html).
- [DP-MEPF code](https://github.com/ParkLabML/DP-MEPF).
- Local authority: ../PRRD_CONSOLIDATED_DESIGN_20260918.md and ../PRRD_PAPER_DELIVERY_PLAN_20260918.md.
