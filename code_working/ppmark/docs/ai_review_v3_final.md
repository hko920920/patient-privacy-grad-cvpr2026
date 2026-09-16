# AI Virtual Review v3 — PP-Mark (Final, Post Tier 1+2 Fixes)

**Saved**: 2026-04-22
**Target**: Paper version after all Tier 1+2 fixes + Abstract/Intro/Related Works tightening
**Input**: 15-page PDF with essential appendices only (Proof, Ablation, Assumption 3.3, SDXL)
**Result**: Lean toward rejection

---

## Summary

This paper proposes PP-Mark, a provenance framework for generative image watermarking that combines a lightweight latent-space statistical detector with a publicly verifiable zero-knowledge proof (ZKP) receipt. The key idea is to bind the watermark to a generation context via a secret binding key and to commit sampled embedding traces to a Merkle root, then prove in zero-knowledge that the binding and trace are consistent, enabling decentralized verification resistant to forged bindings and adversarial optimization. The authors present a formal unforgeability argument under standard cryptographic assumptions and evaluate robustness against black-box imprint-forgery transfer, white-box gradient attacks, and regeneration/removal attacks on SD 2.1 and SDXL, reporting strong empirical resistance while keeping verification practical.

## Strengths

### Technical novelty and innovation
- Introduces a compelling hybrid of semantic watermarking and cryptographic verifiable computation, anchoring a public statistical test to a ZKP receipt to enable open verification without exposing a detector oracle.
- Formalizes a composite predicate and partial-opening check over a Merkle-committed trace, yielding a clean unforgeability bound that explicitly captures the open-sampling trade-off.
- Clear articulation of the context-binding mechanism (b = H(ctx_hash||h_ctx||k)) and image-binding (h_img in the challenge seed) to prevent proof replay and cross-context transfer.

### Experimental rigor and validation
- Evaluates across two diffusion architectures (SD 2.1 and SDXL), multiple attack types (imprint forgery, transfer, white-box PGD, regeneration), and several baselines spanning semantic and post-hoc methods.
- Uses controlled FPR calibration (1%) and fixed pairing protocols for fairness in transfer experiments, with confidence intervals and statistical tests in places.
- Provides ablations for sample count N, ZKP runtime/size scaling, watermark-strength effects, and robustness to common transforms.

### Clarity of presentation
- Clear system overview with end-to-end workflow and verification modes (score vs accept), explicit threat model and key hierarchy, and detailed notation.
- Theorem statement with an intuitive probability bound decomposition and a proof sketch; appendices extend the reductions and empirical checks of residual assumptions.

### Significance of contributions
- Addresses a critical, timely tension in provenance: public auditability vs robustness to optimization attacks. A credible path for decentralized verification could materially impact standards (e.g., C2PA) and regulatory compliance.

## Weaknesses

### Technical limitations or concerns
- **Potential gap between stated "anchoring" claims and the actual circuit constraints**: the listed constraints (P1–P4) do not appear to enforce that the per-index Gaussian baseline g_i and payload bits are deterministically derived from b via F^-1 ∘ H and RS-encoding; nor do they check consistency of opened latents with the image-derived inversion. This weakens the cryptographic link between the public binding and the statistical detector.
- **The ZKP currently does not prove any relation between the published image and the committed trace** beyond including h_img in the seed that selects which (private) leaves are digested; no opened values are revealed or compared to the image latent, even inside the circuit. This undermines the claim that the proof "proves the embedding procedure was executed for the published image."
- **Security and detection reliability hinge on Assumption 3.3** (properties of inversion residuals), which is empirically plausible but not theoretically guaranteed across models/schedulers/encoders; distribution shifts could degrade margins.
- **Accept-mode's strict pixel-hash binding** means benign post-processing (e.g., JPEG recompression, resizing) invalidates the proof; the system then requires re-proving, which is computationally heavy.

### Experimental gaps or methodological issues
- The imprint-forgery transfer uses TR/GS as imprint sources; while useful for comparability, it may understate adaptive attacks tuned to PP-Mark's correlation score. White-box PGD is constrained to l∞ = 2/255, which may not model a strong content-preserving but watermark-seeking adversary.
- Some baselines (e.g., TrustMark/InvisMark) are used via public APIs with bit-accuracy surrogates; comparability might be affected by differences in detector design and score ranges.
- Evaluation datasets are modest (mostly n=100 per condition), and FPR calibration is done on clean, same-model images; more diverse real-world degradations and cross-model mismatches would strengthen claims of public auditability.

### Clarity or presentation issues
- **Inconsistency between Algorithm 1 and the circuit description**: Algorithm 1 suggests VerifyReceipt uses I, but the circuit constraints P1–P4 do not encode any check against the image latent; the paper earlier mentions "opening_ok with ε tolerance measured in latent units," but later sets ε=0 and constructs values solely from (g_i, bit_i) in-digest, seemingly dropping any image-latent comparison.
- The role of RS coding and deterministic sampling in the circuit is described conceptually but not enforced in the listed constraints; "proving correctness of b and Merkle consistency" alone does not cryptographically bind the detector's expected signs back to b unless the sign bits and baselines are constrained accordingly.

### Missing related work or comparisons
- The commit-and-open, sampling-based verification literature (e.g., protocols that sample neural execution traces with Merkle-commitments) is only partially connected; given the similar probabilistic soundness trade-offs (e.g., related work 2603.19025), a deeper comparison of sampling soundness vs PP-Mark's partial-opening bound would help.
- Discussion of public-verification systems for provenance (e.g., beyond C2PA signatures, and closer to ZK proof-of-execution variants) could be expanded to situate PP-Mark among cryptographic alternatives.

## Detailed Comments

### Technical soundness evaluation

The high-level architecture is well-motivated: statistical detection supplies robustness to benign perturbations, while a ZKP receipt is intended to remove the attack surface of public detectors by cryptographically binding verification inputs (b, sample_root, opening_digest) to secret state. However, the security-critical details in the circuit are under-specified or seemingly insufficient:

- **P1** proves b corresponds to some k; good for provenance identity.
- **P2** proves Merkle consistency; standard.
- **P3**: the "openings" are not actually opened to the verifier; instead, equality to opening_digest is proved in zero-knowledge. Without revealing opened tuples or checking them against the image latent, P3 alone does not link the trace to the image.
- **P4** asserts z_i^(0) = g_i + α_eff(2bit_i−1) for i in K, but does not enforce that g_i = F^-1(H(b||i+1)/2^32) or that bit_i equals the RS-derived bit from b at index idx(x,y). The paper explicitly notes they avoid re-simulating inverse-CDF in the circuit for cost reasons; there is also no listed constraint tying the bits/indices to b. This leaves a gap: the committed trace could be arbitrary, satisfying P4 tautologically, yet not reflect the deterministic b→c→bits mapping or the public LUT. As written, the ZKP proves "some" internally consistent trace exists for "some" b, not that the trace induces the exact public detector used by verifiers.

Because accept = (score ≥ τ) ∧ (proof_ok), the actual linkage to the image is still statistical (score). The proof_ok component currently seems to primarily prevent forged bindings (b without k) and proof replay across images (via h_img in the seed), but does not, as specified, prove that the published image contains the specific b-governed signs beyond what the score already tests.

The unforgeability theorem's term (1−q/N)^{k_open} presumes a nonzero number of "cheating positions" violating the honest embedding relation; yet if the circuit does not tie g_i and bit_i to b, an adversary can set q = 0 trivially by choosing witness tuples satisfying P4 for all i∈S. The theorem then essentially reduces to hash and SP1 soundness. If the intended model is that only a holder of k can generate b and thus the entire sign pattern, this must be enforced in-circuit by deriving (or at least checking a PRG relation for) bits and g_i from b and i; otherwise the "anchoring" to the public detector is incomplete.

### Experimental evaluation assessment

The empirical exploration is broad and carefully controlled in many respects (constant FPR, fixed pairings, CI reporting). The imprint-forgery findings (score ≈ FPR floor) and white-box PGD outcomes (PP-Mark_accept at 0 FAR) are encouraging and align with intuition: public optimization against the score alone is blocked by the binding, and proof replay is blocked by h_img.

Runtime measurements are transparent. The 48 s proving time and ≈40 MB receipt at N=1000 are significant; the authors acknowledge this and reasonably argue for SNARK wrapping or aggregation as future work.

The robustness claims under common image transforms are helpful; however, accept-mode's strict pixel-hash binding undoes this in practice unless re-proving is available. In scenarios where generators are untrusted and accept-mode is required, users will likely face frequent re-proving costs for benign edits, which should be discussed more quantitatively.

### Comparison with related work (using the summaries provided)

- The commit-with-sampling paradigm mirrors ideas from verifiable inference via activation-trace sampling (related work 2603.19025): both use Merkle-style commitments and probabilistic checking to avoid full-circuit ZK. PP-Mark's opening bound plays a similar role to the "good test" property: small random checks can detect deviations with non-negligible probability. It would strengthen the paper to formalize this parallel and clarify differences (e.g., PP-Mark's zero-knowledge choice, image-binding via h_img, and the specific watermark embedding constraints).
- ZeroMark (2601.12136) also uses Merkleized commitments and ZK for inclusion/exclusion proofs under regulatory demands; the conceptual alignment (privacy-preserving public verification) is relevant background for provenance ecosystems and could be cited as complementary.
- The paper frames CLUE-Mark and C2PA appropriately (undetectability vs unforgeability; signatures vs proofs of process). A deeper discussion of the trade-offs with signature-only provenance under public audit would help position PP-Mark's cost/benefit envelope.

### Discussion of broader impact and significance

If the cryptographic anchoring is completed (i.e., circuit enforces derivations from b and/or compares to image latents), PP-Mark would meaningfully advance public verifiability for AI provenance. This could enable transparent third-party auditing without trusted APIs—a significant step for policy and industry practice. The current proof-size and re-proving requirements may be challenging for large-scale deployments; nonetheless, archival or on-demand verification is plausible.

Key management and revocation are appropriately discussed. The strict pixel-hash dependence avoids perceptual-collision replay but raises usability concerns that merit further HCI/policy discussion.

## Questions for Authors

1. Do the SP1 circuit constraints enforce that (i) the per-index baselines g_i are derived from b and i via the published LUT F^-1 ∘ H, and (ii) the bit_i values correspond to the RS-encoded payload deterministically derived from b at the indices idx(x,y)? If not, how is the claimed "anchoring" to the public detector realized cryptographically rather than heuristically?

2. Does VerifyReceipt(π; md, I) check any equality between the committed latent values and the image-derived latents at the opened positions (e.g., within tolerance ε)? The text suggests this earlier, but P1–P4 as written do not involve the image beyond challenge_seed. Please clarify what, if anything, is checked against the image.

3. In accept mode, Table 1 claims detection of "generator fraud (no embedding)." Since acceptance requires score ≥ τ, wouldn't a no-embed image simply fail the score check regardless of ZKP? What additional fraud mode does the ZKP detect that score alone would not?

4. How sensitive are the results to the inversion scheduler/parameters and to different VAEs? Could a mismatch between the verifier's model M and the generator's affect TPR/FPR significantly? What is your recommendation for closed-source models where M is not publicly accessible?

5. What are the quantitative implications of re-proving for benign edits in typical content workflows (e.g., social-media recompression or thumbnails)? Can you share amortized costs or caching/aggregation strategies that make accept-mode feasible at platform scale?

6. The white-box PGD budget is l∞=2/255. Have you evaluated unconstrained or perceptual-metric-constrained attacks that specifically optimize PP-Mark's correlation score while preserving perceptual similarity? How do results change with larger budgets or score-targeting objectives?

7. Could you share ablations where N and k_open vary concurrently to illuminate the soundness/runtime trade-off, and whether there is an optimal frontier for typical deployment targets?

## Overall Assessment

This paper tackles an important and timely problem—publicly verifiable, forgery-resistant watermarking—and introduces an original and promising architecture that blends semantic watermarking with zero-knowledge proofs. The empirical results are strong and suggest substantially improved resistance to both black-box and white-box attacks compared to representative baselines; the public-verification framing is impactful for standards and policy. However, the current specification of the SP1 circuit appears to omit critical constraints that would cryptographically bind the public detector to the proof: namely, enforcing that the baseline values and sign bits are derived deterministically from the public binding, and/or comparing opened latent values to the image-derived latents. Without these, the ZKP seems to prove only that b corresponds to some secret key and that a committed trace is internally consistent—not that the trace corresponds to the specific detector or to the published image—leaving the statistical score as the sole linkage to the image. This discrepancy undermines the central "anchoring" claim and the unforgeability theorem's meaningfulness (beyond hash and ZK soundness) in its current form.

If the authors can clarify and, ideally, strengthen the circuit to enforce the missing derivations (e.g., check g_i via a committed LUT with a public hash, prove bits are RS-encoded from b, or perform a lightweight in-circuit consistency check against the image latent for the opened subset), the contribution would be significantly more compelling. **As written, I view the conceptual framework and empirical evidence as valuable but the cryptographic core as under-specified. I lean toward rejection in its current form** due to these foundational concerns, with the caveat that a revised version addressing them could be a strong fit for NeurIPS.
