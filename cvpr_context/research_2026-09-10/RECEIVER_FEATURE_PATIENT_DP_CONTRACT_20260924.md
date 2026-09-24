# Patient-DP feature-mean control: frozen first comparison (2026-09-24)

One new DINO feature-mean/L2 bank against both existing DINO gradient/cosine DP banks.
This tests the complete signal, metric, class pooling and clipping recipe; not a single-factor causal effect or a full Dosser reproduction.

- Same DINO checkpoint, P-only PCA16/q95, four separate conditions, P/Q roles, patient/class visit means, seed101 public templates, 128 images, final200 updates, microbatch16, AdamW and resolution schedule.
- Feature vectors keep K4 separate. No old aug-mean target reuse. Heads are retained in provenance but do not enter a feature signal.
- P-only class-present patient full-K4 feature norm q95 (linear, floor1e-6) defines C0,C1 before Q is read.
- Clip patient/class feature vectors at Cc for P and Q; synthetic images are one-class virtual patients and use the identical map. This bounded P/synthetic rule differs from the gradient baseline and is part of the full recipe comparison.
- Query u=.5[clip(v0,C0)/C0,clip(v1,C1)/C1,present0,present1] is 130-dimensional. Norm<=1, add/remove-patient sum sensitivity1. Existing audited analytic Gaussian calibration uses epsilon8/delta1e-5. One OS-random draw, no saved coins or redraw.
- Decode nonnegative noisy class counts and project class sums into radius Cc*count; pool bounded P sums with protected Q sums using denominator max(1,Pcount+noisyQcount). Preserve class and condition axes.
- Loss=.25 sum_class,condition,coordinate ((synthetic_class_mean-target)/Cc)^2. This averages two half-squared normalized L2 values. No coordinate/condition averaging. Nominal range0..2, without claiming gradient-scale equivalence to cosine. Prior anchor.01 and TV.0001 occur once.
- Public changed-path checks use the existing loss2e-6 and gradient2e-7+5e-4*peak criteria. No GPU profile or coefficient search. CPU tests cover aggregation, sensitivity, invalid/empty counts and active/inactive/zero clipping derivatives.
- Final PNG only. Freeze the bank before scoring. Reuse cached BioViL/DenseNet/ResNet18 V features and the existing2000 patient bootstrap draws. DenseNet is primary, ResNet18 supportive and BioViL diagnostic; all are existing development readouts. No new receiver, Expert or Reserved.
- Compare feature minus EACH existing gradient-DP draw, never just the worse/better draw. Positive descriptive support requires positive DenseNet AUROC differences with both conditional CI lower bounds>0 and AP point nondecrease; otherwise report mixed/no advantage. A worse feature result without sufficient target optimization does not establish an intrinsic feature limitation.
- One feature noise, two historical gradient noises and fixed synthesis seed do not estimate population-wide noise/seed superiority. No equal-time claim. No automated repeats or changes based on utility.
- Planned total70-100min, hard total120min from03:26:46UTC; deadline05:26:46UTC. Bank worker cap3600sec. Historical DINO synthesis~35min is a reference, not a new speed measurement.
- Estimated main encoder work:200*128*4*2=204800 forward images and102400 backward images. Public verification96 forwards/64 backwards; final evaluation128 PNGs per existing readout. No new real-Q/V pixel forwards.
- One new protected summary; five summaries if jointly disclosed have basic bound(40,5e-5). This does not protect past nonDP development or evaluation reports. Nothing is externally published here.
