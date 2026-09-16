We thank the reviewer and address all five points below.

**Q1 (notation and Figure 1).** We will restore the condensed notation table to the main text, grouping recurring symbols by context/binding, payload/sampling, embedding, commitment/opening, proof, and detection/decision. Figure 1 will show three aligned flows: (i) context/key -> binding -> payload/sampling -> latent embedding -> trace commitment; (ii) public/private inputs -> P1 binding, P2 Merkle commitment, P3 image-bound opening, P4 embedding consistency -> proof; and (iii) inversion -> score, receipt -> proof verification. Public, private, and derived fields will be separated. Score returns the statistical decision; Accept requires both score >= threshold and a valid receipt. These presentation changes do not alter the mechanism or claims.

**Q2 (cross-method and quality comparisons).** Sections 4.2-4.4 (Tables 3-4; details in Appendices F-K) already compare TrustMark, InvisMark, RingID, Stable Signature, HiDDeN, WIND, and PP-Mark at the same 1% clean FPR. Baseline transfer FARs (TR/GS) span .0100-.0400/.0100-.0267, versus .0033/.0067 for PP-Mark(score) and 0/0 for PP-Mark(accept); white-box FAR is 1.00 for every baseline, .76 for Score, and 0 for Accept. We will consolidate these dispersed results.

The submission reports PP-Mark BRISQUE p95 = 34.14 (n=100) and KID = 1.101 +/- .202 x 10^-3 (n=1000). We have now evaluated all baselines on the matched pool:

|Method|B95|KID x10^3|CLIP x100|
|---|---:|---:|---:|
|PP-Mark|34.14|1.101 +/- .202|31.64|
|Gaussian Shading|27.35|.621 +/- 1.096|32.42|
|HiDDeN|105.86|7.239 +/- 1.474|31.30|
|InvisMark|31.36|-.900 +/- .895|32.53|
|RingID|34.64|-.989 +/- .701|32.41|
|Stable Signature|32.69|-.814 +/- .878|32.50|
|Tree-Ring|41.90|-.090 +/- .922|32.30|
|WIND|27.42|-.386 +/- .879|32.28|
|TrustMark|29.44|-.174 +/- 1.001|32.54|

We use KID rather than FID because the 2,048-dimensional FID covariance is rank-deficient at this sample size, whereas KID is an unbiased finite-sample estimator.

**Q3 (SEAL and semantic binding).** PP-Mark is not a pixel-space watermark or merely a proof over pixels and a key: it embeds a context-keyed pattern in the initial latent. The exact image hash enters only P3 to make the opening image-specific. SEAL instead captions the image, maps its semantic embedding through secret-salt SimHash to patch keys, and statistically compares reconstructed noise with the inverted latent. This supports semantic-change tolerance, but not a public receipt proving a context/key/embedding relation for one artifact.

A semantic hash is not a drop-in receipt identifier: exact equality remains brittle, while fuzzy equality lets distinct images share an acceptance neighborhood and reuse a receipt. SEAL avoids this issue by still testing each image's inverted latent; it arises only if fuzzy semantics identify a transferable receipt. PP-Mark assigns transformation tolerance to Score and exact-origin verification to Accept.

For clean/rotation/JPEG/crop-resize/blur/noise/brightness, SEAL's published accuracies are .980/.774/.945/.668/.938/.976/.992; submitted PP-Mark(score) TPRs are 1/.85/.94/.87/1/.86/.98. This is behavioral, not a strict ranking: SEAL uses tuned patch/match thresholds, whereas PP-Mark reports TPR at FPR <= 1%. We will add SEAL and state this distinction.

**Q4 (exact binding and deployment).** JPEG-compressed or resized copies do not incur a 48.6 s re-proof: Score is the transformation-tolerant screening path (0.72 s verification), while Accept is deliberately scoped to bit-exact originals. A derivative needs a fresh receipt only if it must itself receive cryptographic Accept. Replacing SHA-256 with a perceptual identifier would create a receipt-replay neighborhood. For re-Accept, the submitted N knob reduces prove/verify time from 48.572/7.589 s at N=1000 to 32.309/4.468 s at N=200, with score SD changing from .1301 to .1409. Appendix M identifies succinct wrapping and aggregation as future size/batch optimizations; we will clarify that neither is demonstrated here as a single-image latency reduction.

**Q5 (DGS).** DGS protects a managed decoder by redirecting or suppressing gradients used to train box-free removal models. PP-Mark targets offline public verification, where an adversary can run the verifier locally and no mediated decoder can alter gradients. Its receipt instead makes optimization of the statistical score insufficient for Accept by requiring a valid image-, context-, and trace-bound proof. Thus DGS protects the removal optimization channel, whereas PP-Mark protects the public provenance acceptance rule; they are complementary. We will add and discuss both DGS works.

If these clarifications and results resolve your concerns, we would appreciate reconsideration of the score.
