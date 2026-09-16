# ICML 2026 Reviews & Rebuttals — Full Record

## Submission: PP-Mark (Submission #21743)
Scores: yoou=4(WA), GiP6=2(R), Y5S2=3(WR), T6eo=3(WR) → Reject

---

## Reviewer yoou (Score 4 — Weak Accept, Confidence 2)

### Summary
Common issue in watermarking for generative AI where public watermark can be vulnerable under adaptive optimization attacks. Authors carry out experiments on SD v2.1, PP-Mark achieves lower FAR over baselines.

### Strengths
- Decent motivation — public detectors enable adversarial optimization
- Clear explanation of why purely statistical detection insufficient
- Empirical analysis generally convincing — several threat models covered

### Weaknesses
- **W1**: SP1 circuit only checks consistency, not full embedding. Scope of formal guarantee could be clarified.
- **W2**: Only SD v2.1. Additional models would strengthen generality.

### Key Questions
- **Q1**: White-box score-only FAR still high. Is crypto layer the major contribution?
- **Q2**: Proving takes 48.6s at N=1000. How practical for large-scale deployment?

### Rebuttal Response
- W1: SP1 circuit is bounded by design. b fully determines sign pattern (b→c→bits→sign). Guaranteeing b + trace = anchors score check.
- W2: SD v2.1 for fair comparison. Security guarantee is model-independent. Future work for newer models.
- Q1: Core contribution is crypto layer (accept FAR=0.00). Score-only still better than baselines (0.76 vs 1.00). ZKP is the structural solution.
- Q2: Dual-mode: score mode at baseline speed for throughput, accept mode for high-stakes only. Proof once per image by producer, verify lightweight (7.59s).

### Acknowledgement: (a) Fully resolved
"Helpful clarifications on SP1 circuit design and deployment. However, limitations in empirical generalization and moderate novelty still remain, so I will keep my current score."

---

## Reviewer GiP6 (Score 2 — Reject, Confidence 3)

### Summary
PP-Mark tries to solve spoofing vulnerability when detector is public. Binds watermark to generation context via crypto hashes + ZKP for public verification.

### Weaknesses
- **W1 (soundness)**: h_img never specified. SHA-256 → any modification breaks. Perceptual hash → no guarantees, forging moved to fooling hash. "Major flaw."
- **W2 (significance)**: Model open-sourcing required for DDIM inversion. Anyone can generate without watermark.
- **W3 (significance)**: Metadata storing not explained. Social networks strip metadata → "practically useless."
- **W4 (significance)**: Why ZKP when digital signature suffices? Orders of magnitude speedup.
- **W5 (significance/soundness)**: Only SD2.1. Assumption 3.3 not verified.
- **W6 (significance)**: Robustness/removal ignored. Rotation weakness buried in appendix. "Major issue should be prominently discussed."
- **W7 (significance)**: Visual quality evaluation missing. Single metric insufficient.

### Rebuttal Response
- W1 & W4: h_img uses SHA-256 to prevent proof replay. ZKP prevents fabricated binding + compliance bypass.
- W2: "Public verifiability" = independence from proprietary APIs. Standard assumption for inversion-based methods.
- W3: Universal challenge (C2PA same). Appendix F: IPFS + ledger registries. Two-stage: fetch via visual features → verify.
- W5: ZKP framework model-independent. Future work.
- W6: Appendix G has comprehensive benchmark. Crop+resize=70% TPR. Rotation acknowledged as limitation.
- W7: BRISQUE p95 (fidelity) + KID (diversity) = two axes.

### Acknowledgement: (b) Partially resolved
"Authors did not convince me why simple digital signature is not enough. After rebuttal, even stronger concerns about practicality — lossy compression breaks SHA-256."

### Authors' Reply to Follow-up
Two concrete failure scenarios of pure signature:
1. Forged Binding: attacker fabricates b' to maximize score. ZKP enforces b derivation.
2. Compliance Bypass: generator signs without embedding. ZKP proves procedure executed.

SHA-256 + lossy: signature has identical failure. PP-Mark has graceful degradation via score layer. Designed for dual-mode (accept=exact, score=robust).

### Final Justification (post-rebuttal, 3 killer points)
1. PP-Mark(accept) is overly complicated with close to zero practical applicability (any upload kills watermark).
2. No benefit over digital signature — authors didn't prove why additional info from PP-Mark(accept) is practically valuable.
3. PP-Mark(score) is standard watermarking without provable guarantees, NOT properly evaluated against SOTA (TrustMark, InvisMark, VideSeal) with detailed per-attack results and comparisons.

---

## Reviewer Y5S2 (Score 3 — Weak Reject, Confidence 4)

### Summary
PP-Mark binds watermark to generation trajectory of diffusion process. Merkle tree of intermediate states. ZKP via SP1 zkVM proves image originates from model.

### Strengths
- S1: Interesting combination of watermarking + cryptographic provenance.
- S2: Merkle commitments ensure trajectory cannot be tampered.
- S3: Conceptually novel direction.

### Weaknesses
- **W1**: High system complexity and runtime overhead. No runtime comparison with baselines.
- **W2**: Verification fragility — proof lost → image unverifiable even if watermark present.
- **W3**: Low robustness contradicts fundamental watermarking goal. Any modification invalidates proof.
- **W4**: Limited applicability in realistic image-sharing (compression, re-upload).

### Key Questions
- Q1: Robustness to JPEG, resize, crop?
- Q2: Why zkVM over simpler commitments?
- Q3: Runtime comparisons needed.
- Q4: Possible to design robust + verifiable variant?

### Limitations noted
- L1: Robustness limitations
- L2: Computational overhead
- L3: Model access security
- L4: Privacy/surveillance potential

### Rebuttal Response
- W1/Q3/L2: Runtime comparison table provided. Dual-mode: score at baseline speed, accept for high-stakes.
- W2/W3/W4/Q1/L1: PP-Mark(accept) rejecting modified images is a FEATURE. Score mode handles transforms (JPEG q25: 94%, blur: 100%, noise: 86%). Metadata stripping addressed via external registry.
- Q2/Q4: Signature doesn't prove embedding. Perceptual hash → proof replay vulnerability. Future work.
- L3: Unwatermarked images = "unverified." Can't forge provenance without key.
- L4: Per-provider key, not per-user. One-way hctx.

### Acknowledgement: (a) Fully resolved
"Thanks for the rebuttal." (No score change despite saying fully resolved.)

---

## Reviewer T6eo (Score 3 — Weak Reject, Confidence 3)

### Summary
Publicly verifiable watermarking for diffusion images. Producer derives binding b, generates spread-spectrum watermark, commits traces to Merkle tree, produces ZKP. Verifier: DDIM-invert, reconstruct sign pattern, compute correlation, check ZKP.

### Strengths
- Problem well-motivated (C2PA, EU AI Act)
- Attack evaluation thorough (imprint, transfer, WB PGD, regen, steg, SPSA)
- Full ZKP pipeline implemented with concrete times — more than most crypto+ML papers

### Weaknesses
- **W1**: Unclear what attack ZKP prevents that simpler approach wouldn't. Alternative: sign(perceptual_hash(img), metadata) — public verifiability, non-transferability, transform robustness. Microseconds vs 48s/7s. No watermark/DDIM/Merkle needed.
- **W2**: Two modes don't leverage ZKP for robustness together.
  - PP-Mark(accept): FAR=0 only because SHA-256 breaks on any modification → functionally identical to digital signature on exact image.
  - PP-Mark(score): survives transforms but doesn't use ZKP. FAR advantage (0.76 vs 1.00) comes from per-image randomized sampling, orthogonal to ZKP.
  - "There is no mode where ZKP and watermark's transform robustness work together."

### Key Questions
- Q1: Concrete scenario where sign(perceptual_hash(img), metadata) fails but PP-Mark ZKP succeeds with practical significance?
- Q2: Would replacing SHA-256 with perceptual hash in challenge seed make accept robust? Why not explored?

### Rebuttal Response
- W1/Q1: Two failure scenarios of signature:
  1. Forged Binding: attacker searches fabricated b' to maximize score. ZKP enforces b = H(ctx || k).
  2. Compliance Bypass: generator signs without embedding. ZKP proves embedding executed.
- W2: Keyed sampling is strictly keyed by b. Without ZKP guaranteeing b, score check meaningless against adaptive attacker. ZKP secures foundation of keyed sampling.
- Q2: Proof Replay vulnerability. Perceptual hash collisions → attacker reuses proof for manipulated image. Dual-mode: accept=SHA-256 for exact, score for robust.

### Acknowledgement: (a) Fully resolved
"Security model is unclear. Authors should justify clearly why a simple, production-ready solution is not ok and why their setting requires a much more complicated solution."

### Authors' Reply to Follow-up
1. Metadata stripping: signature = zero fallback. PP-Mark watermark remains embedded.
2. Post-retrieval: re-encoded image → SHA-256 fails for signature. PP-Mark(score) succeeds (soft correlation).
3. Regulatory: EU AI Act requires proving watermarking procedure was executed, not just data authorship.

---

## Authors' Confidential AC Comment
Concerns about GiP6:
1. Arbitrary redefinition of research scope (penalizing for not solving stripping instead of forgery)
2. Failure to review supplementary material (W6/W7 claims evaluations "missing" but they're in appendices)
3. Inconsistency: Originality/Presentation=3 but Reject based on "technical flaws" addressed in appendix
