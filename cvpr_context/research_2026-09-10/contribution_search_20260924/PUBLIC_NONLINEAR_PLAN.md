# Public nonlinear verification plan
2026-09-24. After the operator probe and linear risk calculation, verify the SAME four mappings on 32 newly drawn public coefficient/noise pairs, RNG 872413. No changes to rank8, tau0.02, illustrative sigma, public images or models. Methods: all16, public_signal_PCA8, source_singular_top8, receiver_risk_top8. The low-noise linear calculation predicts all16 is BEST; preserve this result.

For each public coefficient vector a, generate public image perturbations X(a), calculate the actual nonlinear source signal T(X(a)), add illustrative Gaussian noise, and reconstruct coefficients with each fixed linear mapping. Calculate the actual public-probe ridge prediction distance from the clean X(a) reference. This verifies whether the local prediction formula describes nonlinear public-network behavior. It is NOT an epsilon-calibrated private release or clinical utility test. No V/Expert/Reserved, Q image/statistic access, or image-optimization updates.

