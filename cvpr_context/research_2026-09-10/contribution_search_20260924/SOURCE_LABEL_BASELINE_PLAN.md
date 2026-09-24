# Ordinary source-label baselines for the transport candidate

Exploratory controls declared before their outcomes. Required because soft-label distillation and KIP Label Solve are established prior art.

Reuse the same two fixed feature-DP PNG banks, existing DINO K4/projection, public P cache and each bank's SAME protected target. Extract only DINO condition features of these 256 synthetic PNGs (1,024 condition forwards), not any real private/V/Reserved image. No new synthetic optimization or private release.

Fit a source-space ridge0.1 classifier using public P class-balanced patient-equal second moment and the protected class mean difference. Retain both fixed baselines:
- ordinary relabel: clipped [-1,1] DINO teacher scores on existing synthetic PNGs;
- source-only functional label solve: the same bounded least-squares compiler with the DINO teacher's scores on public P and the same fixed trace-scaled label regularizer.

Evaluate each fixed label packet on DenseNet and ResNet18 via the existing frozen linear readout. No architecture, ridge or temperature sweep; keep all outcomes. These are comparison implementations, not a claim of fully reproducing DPPL or KIP.

