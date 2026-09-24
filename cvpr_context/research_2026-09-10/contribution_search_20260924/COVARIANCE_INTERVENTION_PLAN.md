# Fixed covariance intervention diagnostic
2026-09-24; exploratory existing development V analysis. No new synthesis, query, source target extraction, encoder forward, or Reserved access. Ridge .1, no intercept, existing image-level AUROC/AP and patient-weighting conventions.

For all six existing DP banks (A2, DINO-gradient, feature; draws1/2) plus existing feature public-only, calculate the same readout with (i) its own second moment, (ii) public P class-balanced patient-equal second moment. No grid, no choice of public weighting, no method selection.

For each pair of DP banks from one recipe, cross the two first-moment vectors with the two second-moment matrices. Report the four AUROC/AP cells and the two-order average (Shapley) decomposition of the observed difference. This is an algebraic fixed-bank diagnostic, not a causal claim about the DP mechanism or proof of a new algorithm.

