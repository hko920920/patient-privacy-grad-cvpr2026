# Public geometry feasibility probe — 2026-09-24

Scope fixed before the probe: public P images and public checkpoints only; no Q statistics, V efficacy, new protected release, synthesis optimization, Expert/Reserved, or external upload. Source: frozen DINOv2, existing public PCA16 and K4 transforms. Public construction models for geometry: existing DenseNet121 and ResNet18; they are NOT held-out in this probe.

8 public images: 4 per class, distinct patients, deterministic SHA256 order with tag "public-inverse-probe-v1". This is a local operator check, NOT a utility experiment or representation of the private cohort. Public positive-patient count is only six.

Image coordinates: 8 low-frequency cosine patterns per class (16 parameters), shared across the four images of a class, on logit pixel values. No coefficient optimization. Finite differences h=0.01; h/2 checks on coordinates 0,8,15. Source signal: class means of K4 features with the existing feature clipping map, normalized by public class bounds. Receiver functional: existing ridge(.1), no bias, class-balanced readout predictions on public P features. Public probes use equal-patient, equal-visit weights without labels.

Analysis: identify the Jacobian from image coefficients to source statistics A and to public receiver predictions B_r. Test the exact linearized bias-plus-noise formula, and compare a fixed budget of 8 modes selected by source singular values versus downstream risk. Receiver risks are normalized by their public total signal energy. Coefficient prior SD=0.02 in logit-pixel coordinates; illustrative query-noise SD=0.25 times median singular value times that prior SD. These are PUBLIC mathematical probe settings, NOT epsilon8 noise calibration or a claim about actual private noise.

Run at most 10 minutes; keep every result, including nonlinear approximation failures. No following GPU job automatically authorized by this probe. It assesses mathematical feasibility of a candidate, not novelty or clinical performance.

