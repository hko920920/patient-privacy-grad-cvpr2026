# SP1 Detect / Attest Design Snapshot

## 1. Fixed protocol inputs

- Latent grid: the width/height declared by the supported latent-diffusion
  embedding and recovery interface.
- Payload: the 32-byte context/key binding, encoded with RS(64,32) into a
  64-byte, MSB-first bitstream.
- Sample set: deterministic indices keyed by the RS codeword; both its ordered
  index hash and ordered sample Merkle root are public commitments.
- Gaussian source: `tables/invcdf_gaussian.bin`, pinned by SHA-256.
- Fixed-point formats: Q18 for Gaussian/combined samples and Q30 for the
  normalization scale.
- Image identifier: SHA-256 of canonical sRGB pixels after EXIF transpose and
  the declared bicubic resize/rounding procedure. This identifier deliberately
  denotes the exact canonical artifact, not a perceptual neighborhood.

## 2. Embedding and public statement

1. Canonically hash prompt/model/seed/grid/deployment context into `ctx_hash`.
2. Compute `binding = SHA256("ctx_hash" || ctx_hash || producer_secret)` and
   `producer_key_commitment = SHA256("ppmark_key_v1" || producer_secret)`.
3. RS-encode the 32-byte binding and derive the deterministic sample indices.
4. For latent index `i`, derive the LUT position and payload-bit position from
   the binding. The canonical normalized relation is

   `combined_q18 = round(gaussian_q18 * scale_q30 / 2^30)
                   + alpha_effective_q18 * (2 * bit - 1)`,

   where `scale = 1/sqrt(1+alpha^2)` and
   `alpha_effective = alpha * scale`.
5. Commit the ordered fixed-point samples with a SHA-256 Merkle root and commit
   the ordered index set separately.
6. Compute
   `challenge = SHA256("ppmark_opening_v1" || image_hash || ctx_hash || root)`
   and derive the deterministic partial-opening digest.
7. Publish one canonical `AttestationStatement` containing the protocol
   version, context/binding, producer-key commitment, exact image hash,
   root/index/challenge/opening commitments, LUT hash, normalization values,
   sample/grid dimensions, and RS parameters.

The prover witness is separate: it contains the producer secret, RS codeword,
and sampled trace. It is never a verifier input and is removed after successful
proving unless private-debug retention is explicitly requested.

## 3. SP1 predicates

- **P1:** recompute the context/key binding and producer-key commitment.
- **P2:** recompute the ordered sample Merkle root and sampled-index-set hash.
- **P3:** recompute the image-bound challenge and partial-opening digest.
- **P4:** re-encode RS(64,32), derive each payload bit and inverse-CDF Gaussian,
  and enforce the normalized Q18/Q30 embedding relation for every sample.

The guest commits the canonical statement as its only public value. The host
first verifies the receipt cryptographically and then byte-compares those
public values with the verifier-supplied statement. Altering public JSON cannot
reuse an otherwise valid receipt.

## 4. Verification levels

### Detect

Recover/align the latent signal, compute the soft correlation score over the
deterministic sample set, and compare it with a frozen threshold. Detect is the
transform-tolerant screening result. It does not claim that a generator obeyed
the embedding protocol.

### Attest

Attest requires all of the following:

1. the statement's producer-key commitment equals a verifier-configured trust
   anchor (it cannot be learned solely from the submitted artifact);
2. the received artifact has the exact canonical image hash in the statement;
3. the SP1 receipt verifies and commits exactly the expected statement;
4. P1--P4 hold; and
5. Detect passes the frozen threshold.

A transformed derivative can still be screened with Detect when its latent
signal survives, but it cannot reuse the exact-artifact Attest receipt.

## 5. Remaining empirical work

- Re-run full N=1000 H100 timing and artifact-size measurements whenever the
  guest statement or constraints change.
- Keep per-model threshold calibration frozen and versioned.
- Expand end-to-end image inversion tests across supported latent-diffusion
  models and realistic sequential transformations.
