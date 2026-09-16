# AI Virtual Review v4 — PP-Mark (Post Circuit Fix)

**Saved**: 2026-04-22
**Target**: Paper version after v3 circuit P4 strengthening + clarity fixes
**Result**: Recommend acceptance

---

## Summary

The paper introduces PP-Mark, a provenance watermarking framework that combines a context-bound latent-space watermark with a publicly verifiable zero-knowledge proof (ZKP) to achieve unforgeability without centralized verification. The system commits a lightweight trace of the embedding process into a Merkle root and proves, via SP1 zkVM, that the published binding and sampled trace are consistent with a secret key and deterministic embedding; acceptance requires both a statistical detection score and proof verification. The authors provide a formal unforgeability guarantee under standard cryptographic assumptions, and empirical results on Stable Diffusion 2.1 and SDXL show strong resistance to black-box imprint transfer, white-box optimization, and regeneration-based removal, with practical decentralized verification but non-trivial proving overhead.

## Strengths

### Technical novelty and innovation
- Anchoring statistical detection to a ZKP is a compelling reformulation of watermark verification that moves beyond purely statistical or secret-detector approaches; the composite predicate (P1–P4) and image-bound partial opening are well thought out.
- The commit–open–verify design with a challenge derived from the image hash, context hash, and Merkle root prevents proof replay and ties verification to the specific published content without exposing the secret key or full trace.
- The scheme cleanly separates two operational modes—score (fast, robust to transforms) and accept (cryptographically strong, decentralized verification)—and explains their complementary threat coverage.

### Experimental rigor and validation
- Evaluations span black-box imprint forgery (direct and transfer), white-box gradient-based attacks, regeneration removal, and standard image transformations across two architectures (SD2.1 and SDXL).
- Thresholds are calibrated at a fixed 1% FPR across methods, with confidence intervals and statistical tests reported for transfer experiments; ablations cover N vs. variance/proving time, and headroom/assumptions for DDIM inversion residuals are empirically examined.
- Runtime and proof-size measurements (including a sweep over N) provide useful engineering context for deployment trade-offs.

### Clarity of presentation
- The threat model, key hierarchy, and verification pipeline are clearly articulated; figures helpfully convey generation, publishing, and verification flows.
- Formalization of the circuit constraints, notation, and a precise security theorem with a reduction-style proof sketch (and full appendix) aid reproducibility and scrutiny.

### Significance of contributions
- The work addresses an important and timely need: public, decentralized, and robust provenance verification for generative AI, relevant to C2PA and regulatory compliance.
- Demonstrating that public verifiability can coexist with resistance to state-of-the-art forgery attacks meaningfully advances the watermarking/provenance field.

## Weaknesses

### Technical limitations or concerns
- The ZKP ensures consistency of a sampled trace with the binding, but the link between the committed trace and the actual published image is statistical (via the score) rather than cryptographic; the theorem therefore relies on a hybrid (crypto + statistics) guarantee with a parameter-dependent residual risk.
- Partial opening security with k_open = 32 provides modest detection probability for small-q cheating; the formal bound is weak in the small-q regime and offloaded to the score threshold without a joint, end-to-end analytical bound.
- The acceptance proof binds to the exact pixel hash (after canonicalization), so any benign pixel-level change invalidates the receipt; this is a deliberate design but limits practical "public verifiability" once assets are re-shared or recompressed.

### Experimental gaps or methodological issues
- White-box results show PP-Mark(score) can be driven above threshold with high success (FAR 0.76) under a modest l∞ budget; while PP-Mark(accept) blocks this, it is fragile to benign edits and incurs notable overhead, raising deployment trade-offs that are not fully explored in real-world dissemination scenarios.
- Assumption 3.3 (inversion residuals) is only tested on SD2.1/SDXL under one inversion schedule; broader stress tests (alternative schedulers/autoencoders/step counts) would strengthen claims about generality.
- Several experiments rely on relatively small sample sizes (n=100 per condition), especially for low-FAR regimes; larger-scale tests would increase confidence in tail behaviors.

### Clarity or presentation issues
- Some implementation-specific choices (e.g., precise LUT quantization fidelity and tolerance margins) could be discussed more systematically with sensitivity analyses.
- The practical guidance for selecting k_open and N is helpful but still leaves open how to target a desired cryptographic detection probability vs. proving bandwidth/latency for different operating environments.

### Missing related work or comparisons
- The paper positions itself against semantic and post-hoc watermarks and mentions CLUE-Mark, but does not discuss recent instance-specific forgery-resistant approaches (e.g., ISTS: instance-specific injection plus two-sided detection), which would be a natural counterpoint given PP-Mark's security goals.
- Discussion of alternatives to strict cryptographic hashes for image binding (e.g., robust hashing or fuzzy commitments) is brief; a more thorough analysis of their security–robustness trade-offs would contextualize the design choice.

## Detailed Comments

### Technical soundness evaluation
The composite predicate (P1–P4) is sensible: P1 binds b to k, P2 commits the trace, P3 uses an image-derived challenge to prevent adaptive openings or proof replay, and P4 checks deterministically derived baselines and bits with exact fixed-point embedding consistency on opened indices. Using the LUT hash in public inputs improves determinism and auditability.

Witness extraction by zkVM soundness, reduction to hash collision/preimage for binding or Merkle forgeries, and a combinatorial bound for partial openings. The split between large-q (cryptographic detection) and small-q (statistical threshold) is reasonable, albeit with a formal gap in the joint analysis; this is acknowledged and partially addressed by calibration/ablation.

The choice to avoid bringing image latents into the circuit (relying on the score outside) is pragmatic and consistent with performance constraints; it does, however, shift some of the burden from cryptographic to statistical guarantees.

### Experimental evaluation assessment
Black-box imprint forgery experiments are thoughtfully designed with fixed cover–reference pairings, multiple steps, and calibrated thresholds, and they include both direct and cross-method (transfer) settings; the results consistently show PP-Mark(score) at or near the FPR floor and PP-Mark(accept) at zero FAR.

White-box PGD results demonstrate the vulnerability of public detectors; it is a notable and honest strength that the paper reports PP-Mark(score) at FAR 0.76 rather than relying solely on accept-mode. The accept-mode result (FAR 0) is compelling but practically constrained by fragility to benign edits.

Regeneration/steganalysis removal results favor PP-Mark but are summarized at a high level; including more detailed numbers (beyond appendices) would help verify consistency across noise levels and reference counts.

Runtime and receipt-size measurements are valuable; they underscore that accept-mode verification is currently more suitable for archival/evidence use cases than high-throughput pipelines, pending SNARK wrapping or aggregation.

### Comparison with related work
- Recent surveys on AI-generated media defenses (2407.10575) emphasize the need for trustworthy, auditable mechanisms; PP-Mark's public ZKP directly addresses these calls by offering decentralized verification with formal soundness.
- Müller et al. (2025) show strong black-box imprint forgery against semantic watermarks; the presented experiments specifically target this vector and show clear advantages of PP-Mark relative to Tree-Ring/Gaussian Shading and other baselines.
- ISTS (2604.06662) proposes instance-specific injection and two-sided detection to reduce forgery/remove vulnerabilities. While PP-Mark pursues a different axis (cryptographic anchoring rather than adaptive injection), a brief comparison would help contrast resilience mechanisms and deployment trade-offs (e.g., ZKP overhead vs. learned selectors).
- CLUE-Mark targets provable undetectability (complementary to PP-Mark's unforgeability). The paper correctly positions the guarantees as orthogonal and potentially composable.

### Discussion of broader impact and significance
The work proposes a path to decentralized, open verification that could enhance transparency and trust in public platforms and legal contexts; it is timely and likely to influence both industry practices (e.g., C2PA-compatible metadata) and academic follow-up on crypto-anchored watermarking.

Key practical caveat: binding proofs to exact pixel hashes makes receipts brittle under benign edits. The authors acknowledge this and discuss re-proving; nonetheless, realistic content lifecycles often involve recompression/cropping, so future work on secure-but-robust binding (e.g., verifiable robust hashing) would broaden applicability.

## Questions for Authors

1. Can you release code, precomputed receipts, and evaluation scripts to facilitate reproducibility, especially for the imprint-forgery pipelines and the SP1 circuits?
2. What are the projected prove/verify costs after a SNARK wrapping stage (e.g., SP1 STARK → Groth16/Plonk-ish)? Any preliminary measurements or engineering blockers?
3. How sensitive are the detection margins and Assumption 3.3 to different inversion schedules, step counts, and autoencoder variants (e.g., V-prediction, DiT backbones)? Could you report results beyond DDIM-50?
4. What is accept-mode usage in realistic dissemination settings where images are frequently recompressed or resized? Are there mechanisms (e.g., batched re-proving, publisher caches/CDNs) to make this practical at scale?
5. Did you consider robust binding variants (e.g., fuzzy commitments or robust hashing over learned embeddings) and quantify the security–robustness trade-off vs. the strict SHA-256 approach?
6. For small-q cheating, can you provide a tighter joint bound that combines the partial-opening probability with the calibrated τ (beyond empirical FAR), perhaps under mild residual assumptions?
7. Have you tested semantic-level optimization objectives that modify higher-level content (e.g., small semantic changes) rather than direct pixel-space perturbations, and do those affect PP-Mark(score) differently?
8. What is the security/quality impact of increasing k_open (e.g., 64 or 128) and N (e.g., 1500) on acceptance rates, latency, and receipt size? Any guidance for selecting (N, k_open) for different deployment targets?
9. Does committing (g_i, z_i^(0)) for N=1000 positions risk leaking information about the secret payload over many publicly verifiable receipts (e.g., across multiple images from the same key)? How is cross-proof privacy handled?
10. How does PP-Mark interact with instance-specific watermarking (e.g., ISTS)? Could your ZKP anchor be applied to those schemes to combine adaptive injection with provable unforgeability?

## Overall Assessment

This paper presents a well-motivated and technically novel approach to public watermark verification by anchoring detection to a zero-knowledge proof of the embedding process. The security argument is sound within its hybrid crypto–statistical framing, the experimental evaluation is broad and, in critical respects (imprint transfer, accept-mode resilience), compelling, and the presentation is clear. The main limitations are practical: substantial proving overhead, large receipt sizes, and a strict binding to the exact pixel hash that makes accept-mode fragile under benign edits. Furthermore, the white-box vulnerability of score-mode underscores that the cryptographic layer is essential but currently costly to deploy at scale. Despite these trade-offs, the paper makes a meaningful and timely contribution by showing that decentralized, public verification with strong anti-forgery guarantees is achievable in practice and by providing a concrete, auditable design. I recommend acceptance, with the caveat that future work should focus on more scalable proofs and robust-yet-secure binding to broaden operational viability.
