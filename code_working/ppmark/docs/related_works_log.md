# Related Works Draft Log

## Scope
- Methods compared in experiments: Tree-Ring (TR), Gaussian Shading (GS), RingID, Stable Signature, HiDDeN (Hidden baseline).
- Goal: write Related Works so each method is introduced and linked to why it is evaluated.

## Sources Consulted
- `external/RingID/README.md`
- `external/stable_signature/README.md`
- `external/stable_signature/hidden/README.md`
- `external/semantic-forgery/README.md` (Muller CVPR 2025)
- `docs/work_log.md`, `docs/worklog_2025-12-28.md`

## Notes by Method
- Tree-Ring (TR): latent-space watermarking via structured initial noise; basis for semantic watermarking lineage.
- Gaussian Shading (GS): semantic watermark family used in Muller attacks and robustness discussions; treated as a strong semantic baseline in our evals.
- RingID: extends Tree-Ring for multi-key identification with enhanced robustness/distinguishability.
- Stable Signature: latent diffusion decoder fine-tuning + hidden watermark extractor (ICCV 2023); provides decoder-rooted baseline.
- HiDDeN (Hidden): neural post-hoc watermark baseline (encoder/decoder), used here via Stable Signature hidden module.

## Attack Context
- Zhao et al. (2024): regeneration attack on post-hoc watermarking (noise + diffusion denoise/regenerate).
- Muller et al. (CVPR 2025): black-box imprint forgery/removal for semantic watermarking.

## Drafting Plan
- Structure: (1) post-hoc invisible watermarking + Zhao, (2) semantic watermarking + Tree-Ring + Muller, (3) robust/variant methods (RingID, Stable Signature, HiDDeN), with explicit tie-in to experiments.
- Emphasize why each baseline is included: lineage (TR/GS), robustness extension (RingID), decoder-rooted (Stable Signature), post-hoc neural (HiDDeN).
