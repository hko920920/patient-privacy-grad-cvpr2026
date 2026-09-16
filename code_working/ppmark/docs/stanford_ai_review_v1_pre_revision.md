# Stanford AI Review — PP-Mark (Pre-Revision / v1)

**Saved**: 2026-04-21
**Target**: Paper version as submitted to ICML 2026 (pre-revision)
**Purpose**: Baseline AI review to compare against post-revision AI review for gap analysis.

---

## Summary (AI)
This paper introduces PP-Mark, a provenance framework that combines semantic watermarking in latent diffusion models with zero-knowledge proofs (ZKPs) to enable publicly verifiable, decentralized verification that is resistant to black-box imprint and white-box optimization forgery attacks. The key idea is to bind a statistical detection rule to a cryptographic commitment over a context-conditioned latent trace, and to require a ZKP receipt tied to the content's generation context and an image-derived challenge. Experiments on Stable Diffusion v2.1 suggest improved resistance to watermark forging (both transfer imprint and gradient-based optimization) relative to recent baselines, while proof artifacts enable public verification without access to secret decoders.

## Strengths

### Technical novelty and innovation
- Tackles an important and underexplored tension between public verifiability and robustness against adaptive adversaries by cryptographically anchoring a latent-space watermark to the generation context via ZKPs.
- Design couples deterministic, context-bound watermark embedding and a Merkle-committed latent trace with a publicly verifiable SP1 zkVM proof, which, to my knowledge, is novel in the generative watermarking literature.
- The separation between statistical detection and cryptographic binding is a compelling architectural move that reframes verification as a cryptographic consistency check rather than a purely statistical test.

### Experimental rigor and validation
- Evaluations include both black-box imprint forgery (direct and transfer) and white-box PGD-style attacks, as well as watermark removal attempts (regeneration, steganalysis), providing a meaningful stress-test beyond common image transforms.
- The authors report calibration at a fixed operating point (1% FPR) and provide confidence intervals and Fisher exact tests for some comparisons, which helps contextualize observed differences.
- Ablations on the number of sampled positions N and ZK proving/verification overheads offer initial insight into throughput strategies (batch proving, proof aggregation, reducing N) for realistic deployment.

## Questions / Weaknesses (AI)

1. **h_img definition and robustness** — if `h_img` is perceptual, what are its properties? If SHA-256, any modification breaks the commitment.
2. **ZK circuit scope** — does the ZK circuit enforce the full watermark generation relations, or only partial consistency?
3. **End-to-end verification cost** — what are practical throughput strategies (batch proving, proof aggregation, reducing N) for realistic deployment?
4. **Opening subset k_open and ε tolerance sensitivity** — could an attacker exploit a weak opening policy (small k, permissive ε) to craft ambiguous receipts?
5. **C2PA interoperability** — would PP-Mark artifacts be embedded as Content Credentials assertions? How does the system behave when re-encodes or benign crops occur—does the receipt remain valid if `h_img` is perceptual?
6. **Signed-image baseline comparison** — can you provide a comparison or ablation against a purely signed-image baseline (digital signature over pixels + metadata) to isolate the practical advantage of watermark+ZKP (e.g., acceptance after light edits if `h_img` is robust)?
7. **Attack surfaces around the normalized image hash** — adversarial collisions for a perceptual hash, and how to mitigate.
8. **CLUE-Mark-style formal schemes** — missing comparison/discussion.

## Overall Assessment (AI)

This paper addresses an important and timely problem—publicly verifiable, forgery-resistant provenance for generative models—by innovatively combining latent watermarking with ZK proofs. Conceptually, binding statistical detection to a cryptographic receipt is a promising path to defeating optimization-based forgery in public settings. The empirical results on imprint and white-box attacks, together with ablations, are encouraging and suggest real advantages over several baselines when the score alone is used.

However, critical details that determine the real security and practicality are underspecified:
- Definition and robustness of the image hash `h_img`
- Whether the ZK circuit enforces the full watermark generation relations
- End-to-end verification cost
- Acceptance behavior under benign transformations when proofs are required

Some comparisons (e.g., to signed provenance baselines or CLUE-Mark-style formal schemes) and clarifications of security semantics would make the contribution more compelling and complete.

**Recommendation**: Weak Reject (in pre-revision state). AI indicated: "With stronger formalization of the cryptographic statement, rigorous specification/evaluation of `h_img`, and fair, disentangled comparisons (including proof survival under benign edits and signed baselines), I would be inclined to recommend acceptance."
