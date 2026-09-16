We thank the reviewer and address all four points below.

**Q1 (sequential transformations).** We agree that isolated transforms miss cumulative degradation. We evaluated the requested three-operation pipeline on the same 100 watermarked images, with no survivor filtering, using the submitted FP32 DDIM-100 alignment and frozen threshold. We use a conservative persistent criterion: success requires detection after crop-resize-JPEG; failure at either checkpoint counts as failure.

|Pipeline|Status|Detection|
|---|---|---:|
|JPEG q25|Submitted|94/100|
|75% crop + bicubic resize|Submitted|87/100|
|75% crop + resize + JPEG q25|New, persistent|80/100|

Thus 80/100 remained detectable throughout the complete pipeline. We will add this protocol and result.

**Q2 (bare-signature control).** We agree that for canonical-image integrity under a trusted signer, a standard signature over the image hash has the same modification-rejection role as the image-hash component of Accept and is simpler and preferable. Our submitted Sig-only baseline authenticates metadata, so we implemented the requested Ed25519 control with a fixed provider public key. Bare-Sig signs H(canonical pixels); the stronger Context-Sig signs H(canonical pixels), the context hash, and public binding b.

|Test (n=100)|Bare-Sig|Context-Sig|PP-Mark evidence|
|---|---:|---:|---:|
|Original verifies|100/100|100/100|--|
|Cross-image replay accepted|0/100|0/100|--|
|Context substitution accepted|100/100|0/100|--|
|Authorized signer signs unwatermarked image|100/100 accepted|100/100 accepted|0/100 Score/Accept|
|JPEG q25 after signing|0/100 remain valid|0/100 remain valid|94/100 Score detections|
|75% crop-resize after signing|0/100 remain valid|0/100 remain valid|87/100 Score detections|

The signature controls are new; PP-Mark entries are the separately submitted compliance-bypass and Score evaluations. Context-Sig isolates the distinction: even after binding image, context, and b, an authorized signer can validly sign 100/100 images in which no watermark was embedded. A signature authenticates the signer's assertion; it does not verify that embedding occurred. PP-Mark accepted 0/100 such images because Accept also requires the watermark score and a proof enforcing P1-P4, including sampled embedding consistency. Conversely, exact signatures reject lossy derivatives, while Score retains an in-image signal. We will narrow the claim: PP-Mark is not a replacement for a signature when exact integrity alone suffices; its added value is embedding-compliance verification plus exact Accept and transform-tolerant Score.

**Q3 (matched quality metrics).** We agree and computed BRISQUE and KID for every evaluated method:

|Method|BRISQUE p95|KID x10^3 (mean +/- SD)|
|---|---:|---:|
|PP-Mark|34.14|1.101 +/- .202|
|Gaussian Shading|27.35|.621 +/- 1.096|
|HiDDeN|105.86|7.239 +/- 1.474|
|InvisMark|31.36|-.900 +/- .895|
|RingID|34.64|-.989 +/- .701|
|Stable Signature|32.69|-.814 +/- .878|
|Tree-Ring|41.90|-.090 +/- .922|
|WIND|27.42|-.386 +/- .879|
|TrustMark|29.44|-.174 +/- 1.001|

The PP-Mark row retains the submitted measurements: BRISQUE uses 100 images; KID uses 1,000 images, subset 500, and 20 splits. New baseline measurements use the common 100 prompt-seed pairs and a disjoint pool of 900 clean images with 1,000 reference resamples. Unbiased KID can be slightly negative at finite sample sizes, so near-zero negative estimates are not negative distances. We do not claim the lowest BRISQUE: PP-Mark's 34.14 is comparable to RingID's 34.64, below Tree-Ring's 41.90 and HiDDeN's 105.86, while KID remains low. This quality profile accompanies transfer FAR .0033/.0067 for Score and 0 for Accept. We will add the complete table and qualify the quality-robustness claim.

**Q4 (architecture scope).** We agree that SD2.1 and SDXL share a latent-diffusion/DDIM-inversion interface and do not establish architecture-agnostic generalization. We will soften line 711, remove "architecture-agnostic," replace "two architectures" with "two latent-diffusion models," and rename the appendix section. The scoped claim is that the circuit verifies the committed embedding trace without re-simulating the generator; end-to-end Score still requires a compatible initial-latent embedding and recovery interface.

If these new results and scope corrections resolve your concerns, we would be grateful if you would reconsider your assessment.
