We thank the reviewer and address all three points below.

**Q1 (Other Circuits?)**

Routine high-throughput screening uses PP-Mark(score) (0.72 s), and generated receipts can be cached. The 48.6-s proving cost applies only to the one-time generation of a PP-Mark(accept) receipt for exact-image legal or regulatory verification. (paper line 202)

As shown in App. D, RISC0 was substantially slower (180.9 s). We also investigated Halo2, PLONK, and Groth16. Halo2 produced kilobyte-scale proofs on small configurations, but full-size runs timed out and did not reach a validated end-to-end proof.

| Proof configuration (N=1000) | Prove time | Serialized artifact |
|---|---:|---:|
| SP1 (paper) | 48.572 s | 40.68 MB |
| RISC0 | 180.9 s | 40.51 MB |
| PLONK | 312.677 s | 3,874 B |
| Groth16 | 438.380 s | 1,499 B |

PLONK and Groth16 substantially reduced the serialized artifact size but required longer proving times (4.03x, 312.677 s and 5.65x, 438.380 s, respectively). Thus, among the evaluated configurations, SP1 provided the best practical latency-size trade-off for our submitted implementation.

**Q2 (how can the verifier obtain the required generation context for Score mode?)**

Thank you. As already stated in the Appendix under Reliance on Explicit Proof Artifacts, PP-Mark requires public verification artifact, and an image whose artifact is detached, deleted, or corrupted becomes unverified. We agree that the main text did not connect this condition clearly enough to Score-mode use on social platforms.

PP-Mark(score) is claim-conditioned: the verifier receives b and the declared model/inversion configuration from the separately supplied artifact, rather than recovering the original prompt or seed from the image. Metadata stripping removes one transport path, not the latent signal. The same b can therefore evaluate the re-encoded image; our submitted JPEG (Q=25) experiment retains 94/100 detections on SD2.1.

Concretely, detached signatures require the signature and public key, while C2PA requires an embedded or externally resolved manifest, optionally located through a Link, sidecar, or soft binding. If neither the credential nor a resolution path remains, no provenance claim can be validated. PP-Mark likewise returns unverified: context unavailable, not a negative watermark result. We will make this input and failure state explicit in the main text.

**Q3 (Allow an adversary to link these images to the same user?)**

Thank you for raising this privacy question. The core PP-Mark protocol does not include an end-user identifier, account identifier, or public account key in either the Score or Accept statement. The binding key k remains part of the private witness.

For generation j, the public context and binding are h_ctx,j = H(h_p,j || M_j || s_j || W_j || H_j) and b_j = H("ctx_hash" || h_ctx,j || k). Therefore, when the generation context changes, h_ctx,j and b_j also change even if the same secret key k is reused. Under the cryptographic assumptions stated in the paper, the binding values do not expose a public equality test for a shared secret key.

Accept mode additionally uses an image-specific Merkle root, image-bound challenge, and opening digest. Reusing the identical generation context and binding key reproduces the same h_ctx and b, allowing those artifacts to be linked to the same context-key pair, although this does not reveal the end user's identity. Across different contexts, the core protocol exposes no stable user-level identifier.

There is one separate deployment-level qualification. The key-lifecycle guidance in the appendix allows metadata to carry a producer key identifier and rotation epoch for revocation. This intentionally permits linking at the producer/key-epoch level. If a deployment assigns a stable key identifier to each end-user account, it would also make that user's images linkable; a provider-scoped identifier would reveal only the provider and key epoch.

We will add this distinction to the appendix's key-lifecycle discussion, the core receipt exposes no stable user-level identifier across different contexts, whereas deployment key identifiers intentionally enable producer/key-epoch linkage and become user-linking only when assigned per account.
