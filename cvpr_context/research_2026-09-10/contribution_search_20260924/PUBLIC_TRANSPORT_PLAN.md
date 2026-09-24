# Public receiver transport — exploratory diagnostic, fixed before scores

Motivation: the fixed-bank covariance intervention attributes the DenseNet DP1→DP2 drops mainly to class-mean changes, not second-moment changes. Test whether a public mapping can transfer an already protected source mean into other receiver feature spaces, avoiding a second private receiver query.

Inputs: public P cached DINO K4 features and public P receiver features; existing feature public-only and two DP target files; existing DenseNet/ResNet18 V features ONLY for the final exploratory comparison. No raw Q, new release, encoder forward, synthetic bank or Reserved.

Construct patient/class visit means on public P. For each of the two existing public class clipping bounds, transform all P patient/class means with that clipping rule. Fit an affine ridge map from 64-D DINO condition vectors to 128-D receiver means, using equal patient weight (split weight across two present classes). Ridge fixed to 0.001 * trace(centered source covariance)/64. Five patient-group folds are diagnostic only; no choice of ridge from V or folds.

Translate each existing class-conditional protected mean through its public affine map. Readout uses public P class-balanced patient-equal second moment, ridge0.1, no bias. This is a SUMMARY-TRANSFER DIAGNOSTIC, not a new synthetic-image result. Compare its own translated-P baseline, real-P readout, and all two protected targets. No coefficient tuning or architecture selection.

