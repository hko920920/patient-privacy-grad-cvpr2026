# NeurIPS 2026 PP-Mark Rebuttal Worklog

## Response 2 reconstructed final: notation and Figure 1 clarity

The original final prose was not persisted in this worklog. The following
reconstructs the agreed response from the final notation table, the submitted
method, and the agreed Figure 1 revision scope.

Thank you; we agree with both presentation concerns. We had condensed an
earlier notation table to meet the page limit, but this made the transitions
among context binding, latent embedding, commitment, and proof verification
unnecessarily difficult to follow. We will restore the following compact core
notation table in the main paper; symbols local to individual experiments or
the security reduction will continue to be defined at their point of use.

| Stage | Symbol | Meaning |
|---|---|---|
| Generation context | p, s, M, W, H | Prompt, generation seed, model identifier, and output width and height. |
| Binding | k, H, h_ctx, h_img, b | Producer-held binding key; SHA-256; context hash; hash of canonicalized image pixels; and public context-key binding. |
| Payload | c, bits | Reed-Solomon codeword derived from Trunc(b) and its expanded binary watermark sequence. |
| Sampling | S, N, i, idx(i) | Deterministic latent sample set, its size, a sampled latent position, and the payload-bit index assigned to that position. |
| Embedding | F^{-1}, g_i | Committed inverse-CDF lookup table and the binding-derived Gaussian baseline at sampled position i. |
| Embedding | s_i, alpha, alpha_eff, z_i^(0) | Binding-derived sign, raw and effective watermark strengths, and the variance-normalized embedded latent value. |
| Commitment | T, leaf_i, paths_i, sample_root | Sampled trace, its Merkle leaf and authentication path, and the Merkle root committing the complete trace. |
| Opening | K, k_open, challenge_seed, opening_digest | Deterministically opened subset, its size, the image-bound opening challenge, and the digest committing the selected trace entries. |
| Proof interface | x, w, pi, C | Public statement, private witness, SP1 proof receipt, and verification circuit. |
| Proof predicates | P1, P2, P3, P4 | Predicates enforcing binding-key consistency, Merkle commitment, image-bound opening selection, and embedding consistency on K. |
| Detection | zhat_i^(0), w_i | DDIM-recovered latent value and expected watermark component w_i = alpha_eff s_i. |
| Decision | score, tau | Normalized correlation score and its clean-calibrated acceptance threshold. |

We also agree that the current Figure 1 is too high-level. It was intended to
separate generation from the two verification modes, but it does not expose
where the cryptographic predicates enter. We will revise it to show three
aligned flows: (1) generation and commitment, from context and key through b,
the payload, sampled latent embedding, trace T, and sample_root; (2) proof
construction, with P1 checking key/context binding, P2 checking the Merkle
commitment, P3 deriving the image-bound opening challenge and opening digest,
and P4 checking the b-derived baseline, payload bit, and embedding relation on
the opened set K; and (3) verification, where DDIM inversion produces the
correlation score while the receipt independently verifies P1-P4. Public
statement fields, private-witness fields, and derived values will be visually
separated, and the final branch will explicitly show that Score mode returns
the statistical decision whereas Accept mode requires both score >= tau and a
valid receipt. These changes are presentational and do not alter the mechanism
or claims.


Date: 2026-07-24
Submission: 15860, "PP-Mark: Provable and Publicly Verifiable Watermarking for Generative AI"

## Response 3: Reviewer iryC Q5, Decoder Gradient Shield comparison

### Reviewer request

Compare PP-Mark with Decoder Gradient Shield (DGS) defenses that actively
modify a watermark decoder to prevent gradient-based remover training from
converging:

- An et al., CVPR 2025, "Decoder Gradient Shield: Provable and High-Fidelity
  Prevention of Gradient-Based Box-Free Watermark Removal."
- An et al., IEEE TDSC 2026, "Decoder Gradient Shields: A Family of Provable
  and High-Fidelity Methods Against Gradient-Based Box-Free Watermark
  Removal."

### Verified technical boundary

- DGS assumes verification queries pass through a defender-operated black-box
  watermark decoder.
- DGS-O, DGS-I, and DGS-L perturb the decoder at its output, input, or
  intermediate layers, respectively. Their target is the gradient channel used
  to train a watermark-removal network.
- The DGS guarantee concerns preventing remover convergence under its stated
  managed-decoder threat model. The TDSC extension explicitly leaves attacks
  that bypass the decoder outside its scope.
- PP-Mark targets independently executable public provenance verification.
  Its inversion-based score and proof verifier can be run locally, so its
  design cannot rely on a mediated API that alters adversarial gradients.
- PP-Mark(score) must not be presented as providing a DGS-style convergence
  guarantee. PP-Mark(accept) instead makes score optimization insufficient:
  acceptance also requires a receipt bound to the image, generation context,
  and committed embedding trace under the stated cryptographic assumptions.
- The methods therefore address different layers. DGS protects the
  optimization channel against watermark removal; PP-Mark's cryptographic
  layer protects the acceptance condition against provenance forgery and
  cross-image proof transfer.

Primary sources checked:

- https://arxiv.org/html/2502.20924
- https://arxiv.org/html/2601.11952

### Decision

- Response type: clarification and related-work positioning.
- New experiment required: no.
- Do not claim that DGS is a drop-in wrapper for PP-Mark's inversion-based
  score. No such integration is implemented or evaluated.
- Do not dismiss DGS or imply that PP-Mark supersedes its removal guarantee.
- Add both DGS citations and this access-model/guarantee distinction to the
  revised related-work discussion.

### Final rebuttal text

> Thank you for highlighting this missing comparison. DGS and PP-Mark
> protect different interfaces and address different failure modes. DGS
> targets box-free watermark removal through a defender-operated black-box
> decoder. DGS-O, DGS-I, and DGS-L perturb the decoder output, input, or
> intermediate layers to redirect or suppress the gradients used to train a
> watermark-removal network. Its guarantee therefore concerns remover
> convergence while queries pass through the protected decoder.
>
> PP-Mark targets independent public provenance verification. Its
> inversion-based score and proof verifier can be executed locally, so the
> design cannot rely on a mediated API to alter an adversary's gradients.
> Instead, PP-Mark(accept) makes optimization of the statistical score
> insufficient for full acceptance: the verifier also requires a valid
> receipt bound to the image, generation context, and committed embedding
> trace. Accordingly, we do not equate PP-Mark(score)'s empirical robustness
> with DGS's convergence guarantee.
>
> DGS therefore protects the optimization channel against removal, whereas
> PP-Mark's cryptographic layer protects the acceptance rule against
> provenance forgery and cross-image proof transfer. DGS is appropriate when
> a managed decoder service is available; PP-Mark addresses the setting in
> which third parties must verify provenance without trusting such a service.
> We will add this distinction and cite both DGS works.

### Status

Finalized for rebuttal drafting. Revisit only if the reviewer asks for a
numerical comparison or an integrated DGS and PP-Mark experiment.

## Response 4: Reviewer Ef3K Q3, multi-image linkability

### Reviewer request

Clarify whether Accept receipts or Score-mode contexts share deterministic
elements, such as user identifiers or account keys, that allow multiple
images to be linked to the same user.

### Scope of this response

This response is based only on the protocol, public statement, and key
lifecycle described in the submitted paper. Undocumented implementation
options must not be introduced in the rebuttal.

### Verified paper-level boundary

- Neither the Score nor Accept statement defined in the paper contains an
  end-user identifier, account identifier, or public account key.
- The binding key `k` remains in the private witness.
- For generation `j`, the context and binding are
  `h_ctx,j = H(h_p,j || M_j || s_j || W_j || H_j)` and
  `b_j = H("ctx_hash" || h_ctx,j || k)`.
- When the context changes, `h_ctx,j` and `b_j` change even if the same
  secret `k` is reused. Under the paper's cryptographic assumptions, the
  binding values do not expose a public equality test for a shared secret
  key.
- Accept mode additionally uses an image-specific Merkle root, image-bound
  challenge, and opening digest.
- Reusing the identical context and binding key reproduces the same `h_ctx`
  and `b`. This allows the artifacts to be linked to the same context-key
  pair, but does not by itself reveal the end user's identity. Across
  different contexts, the core protocol exposes no stable user-level
  identifier. The paper must not claim universal receipt unlinkability or a
  formally proved user-level unlinkability property.
- The appendix's key-lifecycle guidance permits a producer key identifier
  and rotation epoch in deployment metadata. This intentionally enables
  producer/key-epoch-level linking.
- If a deployment maps a stable key identifier one-to-one to an end-user
  account, that deployment makes the user's images linkable. If the
  identifier is provider-scoped, it reveals only the provider and key epoch.

### Decision

- Response type: privacy-boundary clarification.
- New experiment required: no.
- Do not claim complete receipt unlinkability.
- Do not claim that shared use of `k` creates a stable public pseudonym.
- Do not introduce implementation-only fields or unsubmitted privacy
  mechanisms.
- Clarify the distinction among image-specific cryptographic values,
  producer-level accountability, and end-user unlinkability.

### Final rebuttal text

> Thank you for raising this privacy question. The core PP-Mark protocol
> does not include an end-user identifier, account identifier, or public
> account key in either the Score or Accept statement. The binding key k
> remains part of the private witness.
>
> For generation j, the public context and binding are
> h_ctx,j = H(h_p,j || M_j || s_j || W_j || H_j) and
> b_j = H("ctx_hash" || h_ctx,j || k). Therefore, when the generation
> context changes, h_ctx,j and b_j also change even if the same secret key
> k is reused. Under the cryptographic assumptions stated in the paper,
> the binding values do not expose a public equality test for a shared
> secret key. Accept mode additionally uses an image-specific Merkle root,
> image-bound challenge, and opening digest. Reusing the identical generation
> context and binding key reproduces the same h_ctx and b, allowing those
> artifacts to be linked to the same context-key pair, although this does not
> reveal the end user's identity. Across different contexts, the core
> protocol exposes no stable user-level identifier.
>
> There is one separate deployment-level qualification. The key-lifecycle
> guidance in the appendix allows metadata to carry a producer key
> identifier and rotation epoch for revocation. This intentionally permits
> linking at the producer/key-epoch level. If a deployment assigns a stable
> key identifier to each end-user account, it would also make that user's
> images linkable; a provider-scoped identifier would reveal only the
> provider and key epoch. We will add this distinction to the appendix's
> key-lifecycle discussion: the core receipt exposes no stable user-level
> identifier across different contexts, whereas deployment key identifiers
> intentionally enable producer/key-epoch linkage and become user-linking
> only when assigned per account.

### Excluded from the rebuttal

- Undocumented or implementation-only metadata fields.
- Claims about the settings used by scripts or experiments that are not
  stated in the paper.
- New anonymous-credential, one-time-key, or membership-proof designs.
- Any claim of a formal unlinkability theorem.

### Status

Finalized for rebuttal drafting. Revisit only if the reviewer asks for a
formal unlinkability construction or deployment-specific identity policy.

## Response 5: Reviewer Ef3K Q1, proof size and alternative proof modes

### Reviewer request

The reviewer asks whether the approximately 48-second proof-generation time
and 40-MB receipt can be reduced by implementing the predicates in another
circuit or proof backend.

### Verified local measurements

The submitted appendix reports only the SP1 core-receipt sweep and mentions a
succinct wrapper qualitatively. Existing artifacts show that compact proof
modes were in fact evaluated but their measured trade-off was omitted.

Earlier same-build, same-shape SP1 backend sweep at `sample_count = 1003`:

| Proof mode | Logged prover-command time | Serialized receipt |
|---|---:|---:|
| Core | 77.537 s | 42,665,513 B |
| PLONK | 312.677 s | 3,874 B |
| Groth16 | 438.380 s | 1,499 B |

Artifact sources:

- `artifacts/outputs/out_sp1_1080_s1000_gpu_core/`
- `artifacts/outputs/out_sp1_1080_s1000_gpu_plonk/`
- `artifacts/outputs/out_sp1_1080_s1000_gpu_groth16/`

The compact modes reduce the serialized artifact by approximately
11,013x (PLONK) and 28,463x (Groth16) relative to the matched core run, but
increase the logged prover-command time by approximately 4.03x and 5.65x.

A separate matched `N = 200` SP1 comparison gives:

| Proof mode | Logged prover-command time | Serialized receipt |
|---|---:|---:|
| Core | 39.296 s | 9,221,531 B |
| Compressed | 45.308 s | 1,315,626 B |

Here, compressed mode reduces receipt size by 85.7% (7.01x) while increasing
the logged command time by 15.3%.

Important timing qualification:

- `sp1_prover_sec` is wall time around the SP1 host `prove` command, not an
  isolated kernel-only proving measurement.
- The submission's 48.572-second core result comes from a later optimized
  core timing sweep. The PLONK and Groth16 artifacts above were not rerun in
  that later sweep.
- Therefore, compare PLONK and Groth16 only with the matched 77.537-second
  core artifact, not directly with 48.572 seconds.

### Technical interpretation

- No evaluated mode reduces proof size and proof-generation time
  simultaneously.
- Core is the minimum-latency mode among the measured full-scale SP1 runs but
  produces a large execution-proportional receipt.
- Compressed mode gives a moderate size reduction with a smaller latency
  penalty.
- PLONK and Groth16 make public transfer/storage practical, but recursive
  compression and SNARK wrapping move the system to a much higher
  proof-generation latency point.
- This is a latency-versus-bandwidth Pareto frontier, not a free compression
  improvement.
- The revised paper should replace the unsupported qualitative
  "approximately 20 KB" statement with the measured serialized sizes and
  clearly distinguish proof modes and setup assumptions.

Official SP1 documentation confirms that core, compressed, Groth16, and PLONK
are distinct proof types with different proof-generation, verification, and
size trade-offs:

- https://docs.succinct.xyz/docs/sp1/generating-proofs/proof-types
- https://docs.succinct.xyz/docs/sp1/security/security-model

### Evidence excluded from the rebuttal

- Do not use the RISC0 timing comparison until reconciled. The submitted
  figure visually reports a value near 181 seconds, while the retained
  `out_full_gpu_sample1000` metadata records 865.358 seconds. These are not a
  trustworthy matched pair as currently documented.
- Do not claim a full-scale Halo2 comparison. The retained Halo2 paths are
  legacy or incomplete and do not establish a reliable matched benchmark.
- Do not describe the approximately 20-KB wrapper size in the submitted text
  as measured. The retained measured SP1 wrapper artifacts are 3,874 B
  (PLONK) and 1,499 B (Groth16).
- Do not imply that compact proof modes also improve proving latency.

### Correctness follow-up outside this individual answer

SP1's official security documentation distinguishes the native STARK modes
from the Groth16/PLONK zero-knowledge wrappers. Before finalizing the full
rebuttal, audit every use of "ZKP" and "zero knowledge" in the submission so
that the 48.6-second core artifact is not assigned a property that belongs
only to the wrapped proof modes.

### Final rebuttal text

> Thank you for raising this. We agree that the appendix was incomplete: it
> reported only the native SP1 core-receipt sweep and mentioned succinct
> wrapping qualitatively, although we had also evaluated compact proof modes.
> The measurements reveal a clear latency--bandwidth trade-off rather than a
> backend that improves both objectives.
>
> In an earlier same-build SP1 sweep at 1,003 sampled positions on one H100,
> the core, PLONK, and Groth16 modes produced serialized receipts of
> 42,665,513 B, 3,874 B, and 1,499 B, respectively. Their recorded
> end-to-end prover-command times were 77.54 s, 312.68 s, and 438.38 s.
> Thus, PLONK and Groth16 reduced the transferred artifact by over four
> orders of magnitude, but increased proof-generation latency. In a separate
> matched 200-position comparison, SP1 compressed mode reduced the receipt
> from 9,221,531 B to 1,315,626 B (85.7%) while increasing the command time
> from 39.30 s to 45.31 s (15.3%).
>
> The 48.57-s core number in the submission comes from a later optimized core
> sweep; because the compact modes were not rerun in that sweep, we do not
> compare their times directly against 48.57 s. The matched results support
> the following deployment distinction: core minimizes generation latency,
> compressed mode offers an intermediate off-chain point, and PLONK/Groth16
> are appropriate when transfer or storage size dominates. We will replace
> the qualitative wrapper-size statement with these measured results and
> state the proof-mode and setup assumptions explicitly.

### Status

Draft finalized from retained measured artifacts. Before posting, verify that
the OpenReview response budget can accommodate the two compact-mode result
sentences and complete the separate proof-mode/zero-knowledge terminology
audit.

## Response 5 re-audit: complete backend history and evidence boundary

The preceding "finalized" status is superseded by this re-audit. The compact
SP1 artifacts are real, but they predate the April 2026 P1--P4 guest
strengthening. They establish the proof-mode trade-off and backend-development
history; they are not direct timing measurements of the final P1--P4 guest.

### A. Complete implementation inventory

| Family / mode | Predicate generation | Scale | Prover-command time | Serialized proof | Verification / scale status | Rebuttal use |
|---|---|---:|---:|---:|---|---|
| SP1 Core, three repeats | Pre-P1--P4, Dec. 2025 | N=200 | 39.296 / 40.218 / 39.420 s | 9,221,531 B | Receipts retained; H100 path completed | Yes, only as development-stage Core baseline |
| SP1 Compressed | Pre-P1--P4, Dec. 2025 | N=200 | 45.308 s | 1,315,626 B | Receipt retained; H100 path completed | Yes, paired by workload shape with the N=200 Core runs |
| SP1 Core | Pre-P1--P4, Dec. 2025 | N=1003 | 77.537 s | 42,665,513 B | Receipt retained; H100 path completed | Yes, paired by workload shape with wrappers below |
| SP1 PLONK | Pre-P1--P4, Dec. 2025 | N=1003 | 312.677 s | 3,874 B | Receipt retained; H100 path completed | Yes, as compact-wrapper trade-off |
| SP1 Groth16 | Pre-P1--P4, Dec. 2025 | N=1003 | 438.380 s | 1,499 B | Receipt retained; H100 path completed | Yes, as compact-wrapper trade-off |
| Submitted SP1 Core sweep | Pre-P1--P4, Jan. 2026 paper asset | N=1000 | 48.572 s; verify 7.589 s | 40.68 MB in submitted table | Reported in submission; exact local artifact lineage not recovered | Retain as the submitted number, but do not pair its time directly with the older wrappers |
| Final SP1 Core guest | P1--P4, Apr. 2026 | Small 64x64 smoke, about N=205 | 35.712 s | 10.9 MB | Internal SP1 generation and receipt verification succeeded; realistic DDIM verifier roundtrip was not completed | Yes, only as final-guest smoke evidence |
| Final SP1 compact modes | P1--P4, Apr. 2026 | N=1000 target | Not run | Not run | No retained Compressed/PLONK/Groth16 receipt for the final guest | No numerical claim without rerun |
| RISC0 v3 CPU/GPU fixture | Legacy predicate, Dec. 2025 | Tiny fixture | about 2.79 / 1.28 s | 244,266 B | Both receipts retained; GPU receipt verified; not a realistic workload | Development history only |
| RISC0 full retained runs | Legacy predicate, Dec. 2025 | N=1003 | 811.577 s CPU; 865.358 s GPU-named; 4471.918 s no-GPU | 42,478,550 / 42,478,550 / 50,110,826 B | Receipts retained, but the "GPU" full path was not a clean matched H100 result | Do not use as a quantitative rebuttal baseline |
| Submitted RISC0 figure | Unknown source, Jan. 2026 asset | N=1000 label | Vector bars imply about 180.9 s prove and 2.39 s verify | Not shown | No matching metadata or generation script; conflicts with the retained 865.358-s run | Exclude |
| Halo2/KZG CPU artifacts | Legacy predicates, Dec. 2025 | N=205 | 22.068--32.228 s for representative retained runs | 1,376--3,680 B, plus about 8.39 MB params | Some small configurations eventually verified, but exact timed artifacts are not tied cleanly to the post-fallback verified state; full k=20/21 timed out | Development history only |
| Halo2 GPU: PSE/Scroll/Axiom/zkonduit | Legacy predicates | Build stage | No valid timing | None | No compatible CUDA feature / package mismatch | Exclude numerically |
| Halo2 GPU: halo2-arithmetic/libfam | Legacy predicates | Build stage | No valid timing | None | Required private CUDA submodule unavailable | Exclude numerically |
| Halo2 GPU: ICICLE | Legacy predicates | Build / kernel stage | No valid timing | None | Missing BN254 Poseidon specification and sm_90 kernel failures | Exclude numerically |
| Halo2 GPU: PSE 0.4 + ec-gpu-gen | Tightened legacy circuit | Small H100 path | No valid end-to-end timing | None | GPU FFT matched; strict GPU MSM mismatched; fallback used CPU; later witness assignment failed | Exclude numerically |

The many `out_halo2_small_new*` directories are iterations of the same small
legacy Halo2 bring-up, not independent circuit families. Reporting each as a
separate successful alternative would inflate the evidence and is therefore
not appropriate.

### B. Recomputed SP1 proof-mode trade-off

At N=200, Compressed versus Core:

- Size reduction: 85.7 percent, or 7.01x smaller.
- Prover-command increase: 15.3 percent.

At N=1003, relative to the 77.537-s Core run:

- PLONK: 11,013x smaller and 4.03x slower.
- Groth16: 28,463x smaller and 5.65x slower.

The runs have the same sample-count and workload shape and were generated on
the same development day, but use distinct prompts/witnesses. Call them
"matched workload-shape measurements," not "identical-input measurements."

### C. Defensible engineering conclusion

1. Several proof systems and proof modes were genuinely attempted.
2. Direct Halo2 produced compact small-circuit artifacts, but the full
   realistic and H100 paths did not become stable and validated.
3. The retained realistic RISC0 path was materially slower and the submitted
   RISC0 plot is not traceable to a retained run.
4. SP1 was the only family that completed the realistic H100 development path
   and exposed a measured latency--size frontier within one toolchain.
5. Core minimized measured generation latency. Compressed occupied an
   intermediate point. PLONK and Groth16 reduced the transferred artifact by
   more than four orders of magnitude at substantially higher proving latency.
6. Therefore the Core choice was evidence-backed for experiment iteration and
   minimum latency; it was not a claim that Core also minimizes bandwidth.
7. No evaluated alternative reduced both proving time and proof size.

### D. Mandatory proof-type terminology correction

The current official SP1 security documentation states that individual SP1
STARK proofs are not zero-knowledge, while the Groth16 and PLONK wrappers are
zero-knowledge. Therefore:

- Do not call the 48.572-s / 40.68-MB Core artifact itself a
  "zero-knowledge receipt."
- The unforgeability argument uses computational soundness and can be
  discussed for the native execution receipt.
- Witness confidentiality / zero knowledge must be attached to the
  Groth16/PLONK deployment mode, with its measured extra latency.
- The paper's qualitative `approximately 20 KB` wrapper statement is not the
  retained measurement. The retained v5.2.3 serialized artifacts are 3,874 B
  for PLONK and 1,499 B for Groth16.
- Before posting, verify version-specific SP1 v5.2.3 security wording or use
  the conservative proof-type distinction above.

### E. Minimal reviewer-facing table available without a new H100 run

| SP1 mode | Workload | Time | Serialized artifact | Interpretation |
|---|---:|---:|---:|---|
| Core | N=200 | 39.30 s | 9,221,531 B | Lowest-latency native receipt |
| Compressed | N=200 | 45.31 s | 1,315,626 B | 7.01x smaller; 15.3 percent slower |
| Core | N=1003 | 77.54 s | 42,665,513 B | Matched-shape wrapper baseline |
| PLONK | N=1003 | 312.68 s | 3,874 B | 11,013x smaller; 4.03x slower |
| Groth16 | N=1003 | 438.38 s | 1,499 B | 28,463x smaller; 5.65x slower |

Caption qualification required if posted:

> Development-stage SP1 v5.2.3 measurements on one H100. Rows are matched by
> sample count and workload shape, not by identical witness. This sweep
> predates the final P1--P4 strengthening and characterizes the backend
> latency--size frontier rather than the exact final-guest runtime.

### F. Honest fallback response if no current-P1--P4 rerun is available

> Thank you for raising this. We did evaluate both alternative proof systems
> and compact proof modes; the appendix was incomplete because it reported
> only the native SP1 Core sweep. Direct Halo2/KZG produced kilobyte-scale
> proofs on small configurations, but its full-size CPU runs timed out and
> the H100 path did not reach a validated end-to-end proof. The retained
> realistic RISC0 path was slower. SP1 was the only backend that completed
> our realistic H100 development path, so we then evaluated its Core,
> Compressed, PLONK, and Groth16 modes.
>
> In development-stage SP1 v5.2.3 measurements, Core and Compressed at
> N=200 required 39.30 s / 9,221,531 B and 45.31 s / 1,315,626 B,
> respectively: compression reduced size by 7.01x at a 15.3 percent time
> cost. At N=1003, Core, PLONK, and Groth16 required
> 77.54 / 312.68 / 438.38 s and produced
> 42,665,513 / 3,874 / 1,499 B artifacts. Thus the succinct modes reduce
> transfer size by over four orders of magnitude, but take 4.03x and 5.65x
> longer than the matched Core run. No tested mode improved both objectives.
>
> These backend measurements predate the final P1--P4 strengthening, so we
> use them to characterize the proof-mode frontier rather than as exact
> final-guest timings. The strengthened guest separately passed an
> end-to-end Core smoke test. We will revise the paper to distinguish the
> native Core execution receipt from the zero-knowledge PLONK/Groth16
> deployment receipts, replace the unsupported qualitative wrapper-size
> statement with measured values, and state the proof mode and predicate
> generation explicitly. Core was selected for the experiments because it
> minimized measured proving latency; a succinct wrapper is the appropriate
> deployment choice when transfer or storage dominates.

### G. Stronger response requiring one matched rerun

The cleanest rebuttal evidence is a final-P1--P4, same-input, same-build,
same-H100 sweep of Core, Compressed, and at least one of PLONK/Groth16 at
N=1000. Without that rerun, the fallback response above is accurate but
necessarily carries a development-generation qualification.

### Status after re-audit

Not ready to post as final. The evidence inventory and fallback answer are
complete. Decide whether a matched final-P1--P4 H100 rerun is feasible; then
either replace the fallback table with current numbers or post the qualified
development-history answer.

## Response 6: iryC - benign transformations and proof regeneration

### Reviewer question

> In real-world deployment, benign pixel perturbations induced by lossy
> operations (e.g., JPEG compression or rescaling) are ubiquitous. The
> framework's heavy reliance on the deterministic rigidity of cryptographic
> hashes (like SHA-256) causes minor visual edits to disrupt the inversion
> reconciliation chain, thereby necessitating costly and high-latency ZKP
> re-generations (~48 seconds per image). To mitigate this overhead, what
> potential solutions or architectural optimizations could bridge the gap
> between strict cryptographic rigidity and practical robustness?

### Correct interpretation and source map

- The question correctly observes that JPEG/resizing changes `h_img` and
  invalidates the exact Accept receipt.
- It overstates the operational consequence. The submitted paper does not
  require Accept or a 48.6-second re-proof for every distributed JPEG/resized
  copy:
  - Main-text line 355 says Score degrades gracefully under lossy transforms.
  - Line 359 defines Score for large-scale routine screening and Accept for
    legal evidence, regulatory compliance, or an untrusted generator.
  - Lines 421--422 measure Score verification at 0.72 s and identify 48.6 s as
    a one-time Accept proving cost.
  - Lines 1286--1287 explicitly scope Accept to bit-exact originals and Score
    to general lossy distribution.
- The transform table is specifically a Score-mode experiment. It reports TPR
  0.94/1.00 after JPEG q=25 and 0.87/0.74 after crop+resize on SD2.1/SDXL.
- Lines 362 and 665 say a benign edit requires a new proof, but this means only
  when the edited derivative must itself receive cryptographic Accept. The
  appendix wording should be revised to state this qualifier explicitly.
- SHA-256 should remain the Accept binding: replacing it with a perceptual hash
  turns the tolerated perceptual neighborhood into a receipt-replay
  neighborhood.

### What can actually reduce the special-case Accept cost

1. **Avoid unnecessary proving through the existing dual mode.** Routine
   transformed copies use Score; the authoritative bit-exact original retains
   the cached Accept receipt.
2. **Use the measured sample-count knob when a lower-latency Accept profile is
   acceptable.** The submitted sweep reduces prove/verify time from
   48.572/7.589 s at N=1000 to 32.309/4.468 s at N=200. The cost is weaker
   score concentration: standard deviation changes from 0.1301 to 0.1409.
   N=1000 remains the submitted default knee, so N=200 is a trade-off, not a
   free improvement.
3. **Use the future directions already stated in the appendix for their
   correct cost targets.** A STARK-to-SNARK wrapper reduces receipt size at
   additional proving cost, while proof aggregation can amortize batch
   verification and transfer.
4. **Do not present either future direction as a measured single-image
   latency fix.** The paper has not shown that either reduces the 48.6-second
   per-image proving time.
5. **If an edited derivative itself requires exact Accept, the current
   construction still needs a fresh receipt.** Removing that requirement by
   making the hash fuzzy would weaken replay resistance.

### Revised rebuttal text

> Thank you. This trade-off is precisely why PP-Mark exposes two operational
> modes, although we agree that the qualifier in our appendix should be more
> explicit. Routine JPEG-compressed or resized copies do not incur a
> 48.6-second re-proof. `PP-Mark(score)` is the high-throughput path for
> trusted-generator screening (0.72 s verification), and our submitted
> Score-mode results retain TPR 0.94/1.00 after JPEG \(q=25\) and 0.87/0.74
> after crop+resize on SD2.1/SDXL. `PP-Mark(accept)` is intentionally
> exact-hash and is scoped to bit-exact originals for legal evidence,
> regulatory audit, or settings with an untrusted generator. Thus an edited
> copy requires a new proof only if that derivative must itself receive
> cryptographic Accept.
>
> We retain SHA-256 for that exact mode because replacing it with a perceptual
> hash would turn every tolerated collision neighborhood into a proof-replay
> neighborhood. For the narrower re-Accept case, the submitted implementation
> already exposes a measured latency--reliability knob: reducing \(N\) from
> 1000 to 200 changes prove/verify time from 48.572/7.589 s to
> 32.309/4.468 s, while score standard deviation changes from 0.1301 to
> 0.1409; we selected \(N=1000\) as the default knee rather than claiming this
> as a free speedup.
>
> The appendix also identifies STARK-to-SNARK wrapping and proof aggregation
> as future deployment optimizations. We will clarify their targets:
> succinct wrapping reduces receipt size at additional proving cost, whereas
> aggregation can amortize batch verification and transfer; neither has been
> demonstrated here as a single-image proving-latency reduction. Under the
> current exact-binding construction, an edited derivative that itself
> requires cryptographic Accept still needs a fresh receipt. We will revise the
> appendix sentence to state this condition explicitly and avoid implying that
> routine transformed copies must be re-proved.

### Status

Revised after checking the exact dual-mode scope in the submitted paper. This
answer now corrects the reviewer's over-broad re-proving premise before
answering the narrower optimization question. It uses only the submitted
dual-mode design, transform results, measured N sweep, and the future-work
directions already stated in the appendix.

## Response 7: P58K - secure image-hash signature baseline

### Reviewer question

> Since accept mode binds to the image hash and rejects any modification, it
> is closely related to simply signing the image hash with a standard bit
> sequence key. Please include this baseline and show what the watermark adds
> over a bare signature of the bit sequence.

### Exact mapping

- The reviewer is correct for the narrow canonical-image integrity objective:
  a standard producer signature over the same `norm_v1` canonical image hash
  has the same modification-rejection role as PP-Mark Accept's image binding.
- The submitted `Sig-only` column is not this requested control. It signs
  metadata under a key that the attack model permits the adversary to control.
  It must not be presented as a secure image-hash signature baseline.
- Add two domain-separated Ed25519 controls:
  - `Bare-Sig`: sign `H(canonical_pixels(I))`.
  - `Context-Sig`: sign `H(canonical_pixels(I)) || h_ctx || b`.
- The controls isolate the additional PP-Mark guarantees:
  1. A signature authenticates the canonical image or asserted context chosen
     by the signer, but does not establish that the watermark was embedded. A
     signer can sign a clean image.
  2. PP-Mark Accept combines the score with P1--P4, checking context/key
     consistency, committed-trace membership, image-bound opening selection,
     and the sampled embedding relation without revealing `k`.
  3. A transform that changes the canonical image hash invalidates all exact
     modes, but the watermark supports transform-tolerant Score detection
     after exact signatures fail.
- Narrow the claim explicitly: for canonical-image integrity alone, use a
  standard signature. PP-Mark is valuable when embedding-policy compliance and a
  transform-tolerant in-image signal are also required.

### Implemented controls

- Code:
  - `src/ppmark_v03/signature_controls.py`
  - `scripts/experiment_secure_signature_controls.py`
  - `tests/test_signature_controls.py`
- `Bare-Sig` signs
  `Ed25519("PPMARK_BARE_SIG_V1" || H(canonical_pixels(I)))`.
- `Context-Sig` signs
  `Ed25519("PPMARK_CONTEXT_SIG_V1" || H(canonical_pixels(I)) || h_ctx || b)`.
- Both controls use the submitted `norm_v1` image canonicalization:
  EXIF transpose, RGB conversion, bicubic resize to 512 by 512, uint8 rounding,
  and SHA-256.
- The provider public key is fixed for the run. The private key is not exported.
- No latency or size benchmark is reported because P58K asked for the security
  baseline and the watermark's additional function, not a runtime comparison.

### Verified control results

The full paired experiment used 100 PP-Mark positive images with opening
metadata and 100 clean SD2.1 images. All 100 positive image hashes, context
hashes, and bindings were unique; all 100 clean hashes were unique and
disjoint from the positive hashes.

| Test (`n=100` per row) | Bare-Sig | Context-Sig |
|---|---:|---:|
| Original verifies | 100/100 | 100/100 |
| Cross-image replay verifies | 0/100 | 0/100 |
| Context/binding substitution verifies | 100/100 | 0/100 |
| Intentionally signed clean image verifies | 100/100 | 100/100 |
| JPEG q=25 verifies under original signature | 0/100 | 0/100 |
| 75% crop+resize verifies under original signature | 0/100 | 0/100 |

All transformed and cross-image inputs changed the canonical hash in 100/100
cases. All seven predeclared expected checks passed. Four deterministic tests
also passed, including fail-closed rejection of stale image hashes and fallback
to the submitted `metadata_opening.json` format.

The decisive comparison is the signed-clean row. Even Context-Sig, which closes
the obvious context-substitution gap, accepts 100/100 clean images when the
signer intentionally signs them. In the submitted compliance-bypass evaluation,
PP-Mark Score and Accept accepted 0/100 clean images. Separately, the submitted
Score-mode transform evaluation retained 94/100 detections after JPEG q=25 and
87/100 after 75% crop+resize on SD2.1, while both exact signature controls
rejected all transformed canonical images under their original signatures.

### Compact capability comparison

| Capability | Bare-Sig | Context-Sig | PP-Mark |
|---|---:|---:|---:|
| Canonical-image integrity | Yes | Yes | Yes, via Accept |
| Authenticates claimed context | No | Yes | Yes |
| Checks sampled watermark-embedding relation | No | No | Yes, via Score + P1--P4 |
| Transform-tolerant in-image detection | No | No | Yes, via Score |

### Rebuttal draft

> Thank you; we agree that this is the correct control. For integrity of the
> canonical image under a trusted signing authority, a
> standard signature over the canonical image hash provides the same
> modification-rejection role as the image-hash component of
> `PP-Mark(accept)`. For that narrow objective, the signature is preferable.
> Our submitted
> `Sig-only` baseline signs metadata and is therefore not the secure image-hash
> signature baseline requested here. We corrected this distinction and
> implemented two domain-separated Ed25519 controls: `Bare-Sig`, which signs
> \(H(\mathrm{canonical\_pixels}(I))\), and `Context-Sig`, which additionally
> signs \(h_{\mathrm{ctx}}\) and \(b\).
>
> On \(n=100\) images, both controls verified 100/100 originals and rejected
> 100/100 cross-image replays, JPEG q=25 copies, and 75% crop+resize copies.
> Context-Sig also rejected 100/100 context/binding substitutions, whereas
> Bare-Sig did not. However, when the signer intentionally signed clean,
> unwatermarked images, both Bare-Sig and the stronger Context-Sig verified
> 100/100. In the submitted compliance-bypass evaluation, PP-Mark Score and
> Accept accepted 0/100 clean images because acceptance additionally requires
> the watermark score and, for Accept, a proof of P1--P4 checking the sampled
> embedding relation.
>
> The two mechanisms therefore serve different scopes. PP-Mark is not a
> replacement for a standard signature when canonical-image integrity alone is
> sufficient. Its additional value is sampled embedding-compliance checking
> plus a transform-tolerant in-image Score signal: in the separately submitted
> Score evaluation, detection remained 94/100 after JPEG q=25 and 87/100 after
> 75% crop+resize on SD2.1, after the exact signatures had become invalid.

### Status

Implemented and fully run on 100 positive and 100 clean images. The empirical
results directly satisfy the requested baseline and preserve the narrow
concession that standard signatures are preferable for canonical-image
integrity alone. Both controls should be shown: Context-Sig preempts the
objection that signing the context closes the gap, while its 100/100
signed-clean acceptance isolates the sampled embedding-compliance value.

## Response 8: Ef3K Q2 - context recovery after metadata stripping

### Reviewer question

> Some modern social media platforms aggressively strip embedded metadata and
> re-encode uploaded images. In this realistic scenario, how can the verifier
> obtain the required generation context for Score mode?

### Paper-grounded answer boundary

- Score computation does not require disclosure or recovery of the raw prompt
  or seed. Given the public binding `b`, it reconstructs the codeword, payload
  bits, deterministic sample set, and expected sign pattern.
- It still requires an authenticated public verification record containing
  `b` and versioned verification parameters. PP-Mark cannot infer `b` from
  pixels alone.
- Embedded metadata is one carrier for that record, not the only possible
  carrier. A deployment can retain the same record in an external signed
  manifest and resolve it using a trusted platform asset mapping or an
  independent soft binding.
- C2PA 2.4 explicitly permits manifest stores external to an asset and defines
  soft-binding recovery from a manifest repository after a platform strips an
  embedded manifest. This is a compatible deployment path, not functionality
  implemented by the current PP-Mark artifact.
- Re-encoded pixels can be evaluated by Score after the record is recovered.
  The original Accept receipt is intentionally invalid because its hard image
  hash no longer matches. An authenticated derivative needs a newly issued
  manifest and receipt.
- If neither an embedded record nor a trusted external association is
  recoverable, the only sound output is `UNVERIFIED`; the method must not turn
  missing provenance into a negative provenance claim.

### Final rebuttal text

> Thank you; we agree that the current text conflates the carrier of the
> public verification record with the record itself. `PP-Mark(score)` does
> not require disclosure or recovery of the raw prompt or seed after
> publication. Its score computation requires the public binding \(b\) and
> versioned verification parameters; \(b\) deterministically regenerates the
> codeword, payload bits, sample set, and expected sign pattern.
>
> We envision two delivery paths. If metadata survives, this public record can
> travel with the image. If a platform strips and re-encodes it, the same
> authenticated record can be stored externally and resolved through a
> trusted platform asset mapping or an independent soft binding into a
> manifest repository. This is compatible with C2PA's external-manifest and
> soft-binding recovery model, but is a deployment integration rather than a
> component implemented in our current artifact. After recovering \(b\), the
> verifier can run the transform-tolerant Score path on the received pixels.
> In contrast, the original Accept receipt is hard-bound to canonical image
> pixels and therefore intentionally fails after re-encoding; an authorized
> derivative requires a newly issued manifest and receipt.
>
> If neither an embedded record nor a trusted external association can be
> recovered, the current system cannot infer \(b\) from the image alone. The
> correct result is `UNVERIFIED`, not a negative provenance judgment. We will
> add an explicit `ResolvePublicRecord` step and fail-closed branch before
> Algorithm 1, list the minimum public fields, and clarify this distinction
> between transformed-image scoring and exact-image acceptance.

### Why this answer is structured this way

1. The first paragraph answers what Score actually needs and avoids implying
   that sensitive raw generation context must be disclosed.
2. The second paragraph gives a concrete metadata-stripping workflow while
   clearly labeling C2PA/external storage as deployment integration, not a
   completed artifact contribution.
3. The final paragraph states the hard impossibility boundary and prevents
   missing metadata from being misreported as a negative detection.

### Status

Ready as a paper-grounded rebuttal draft. Before posting, keep the C2PA
reference textual rather than adding an external link, and do not claim that
the current artifact already implements manifest discovery.

## Response 9 forensic audit: BRISQUE and KID baseline comparison

### Reviewer question

> Image quality metrics such as BRISQUE and KID are reported only for PP-Mark.
> Providing the same metrics for baseline methods would enable a fairer
> assessment of the quality-robustness trade-offs across approaches.

### Root cause of the initially anomalous PP-Mark result

- The first rebuttal evaluator used
  `methods/pp_mark/eval/manifest.jsonl`, a stale 100-row manifest.
- The 1,000-image PP-Mark extension later overwrote `img_0000` through
  `img_0099` in the full clean-manifest order, but did not rewrite that stale
  manifest. As a result, 99/100 declared PP-Mark prompt/seed pairs disagreed
  with the image-local metadata. The image set contained only 10 actual
  prompts while the stale manifest declared 100.
- The history is explained by the interrupted 1,000-image run. The original
  run stopped on disk exhaustion before `build_pos_dataset.py` wrote its
  manifest. The resume used `--append-manifest`, preserved the old first 100
  rows, and appended later rows.
- `manifest_fixed.jsonl` is also not a canonical identity manifest. It was
  originally created for an attack experiment by replacing sample 16 with
  sample 72. It therefore has 1,000 rows but only 999 unique image paths,
  duplicates `img_0072`, and omits `img_0016`.
- The actual 1,000 image directories are intact. Exhaustive metadata checks
  show 1,000/1,000 prompt, seed, model, guidance, step-count, scheduler, and
  alpha matches against the clean 1,000-row generation manifest.

### Corrective implementation

- `scripts/eval_rebuttal_quality_baselines.py` now derives PP-Mark keys from
  each per-image `metadata.json`, rather than trusting either stale manifest.
- Inception feature caches now include an image-set fingerprint; matching
  sample counts alone can no longer cause stale feature reuse.
- The original invalid output is explicitly marked with
  `outputs/rebuttal_quality_baselines_n100/INVALID_MANIFEST_DIAGNOSIS.md`.
- Corrected common-100 output:
  `outputs/rebuttal_quality_baselines_n100_corrected`.
- Independent full-1,000 validation:
  `outputs/rebuttal_quality_ppmark_full1000_corrected`.

### Numerical confirmation

- Invalid common-100 PP-Mark KID: `0.020818158512`.
- Corrected common-100 PP-Mark KID: `0.000969855578`.
- Corrected full-1,000 PP-Mark KID, subset 500 and 20 splits:
  `0.001133875617 +/- 0.000212540597`.
- Submitted full-1,000 PP-Mark KID under the same subset/split protocol:
  `0.001101357399 +/- 0.000202021377`.
- The two full-1,000 estimates differ by about 3% and their uncertainty
  intervals overlap. The anomalous KID was therefore a manifest-assignment
  error, not a PP-Mark image-quality failure.

### BRISQUE interpretation

- Recomputing the submitted alpha-4 sweep with the current `pyiqa==0.1.16`
  reproduces every saved value to within `4.93e-7`, including mean `9.924519`
  and p95 `34.144476`. There is no BRISQUE implementation drift.
- The submitted sweep used 25 prompts with four seeds each. The rebuttal
  comparison uses 100 prompts with one common seed each; the full validation
  uses 100 prompts with ten seeds each. BRISQUE is no-reference and
  content-sensitive, so absolute tails differ across these prompt sets.
- On the full metadata-aligned 1,000 set, clean BRISQUE is mean `8.052336`,
  p95 `29.385443`; PP-Mark is mean `13.676499`, p95 `40.402222`, with paired
  mean delta `+5.624163`.
- The submitted alpha-4 p95 bootstrap interval `[27.116, 49.514]` overlaps
  the corrected common-100 interval `[33.718, 50.254]`; the two p95 estimates
  are not statistically contradictory.

### KID protocol caveat for nearly identity-preserving baselines

- Computing unbiased KID on exact clean/watermarked prompt-seed pairs creates
  a negative finite-sample bias for post-hoc methods whose output is nearly
  identical to its paired clean image. The KID estimator assumes independent
  samples, while diagonal cross-pairs are unusually similar.
- Recomputing against the disjoint remaining 900 clean seeds removes the
  large negative values (approximately `-0.007`) and leaves small estimates
  around zero, with standard deviations of roughly `0.0007` to `0.0012`.
- For the rebuttal table, use matched BRISQUE for per-image fidelity and KID
  against a common disjoint clean reference. Report the KID uncertainty and
  do not interpret a small negative unbiased estimate as a negative distance.

### Persisted independent-reference KID protocol

- Added `scripts/build_rebuttal_quality_tables.py`.
- The script maps cached Inception features to the validated prompt/seed
  manifests rather than assuming row order.
- Each method contributes the same 100 common prompt/seed examples.
- The KID reference pool is the 900 clean examples in the validated
  1,000-image clean set whose keys are disjoint from the common 100.
- Each estimate uses all 100 method examples and 1,000 deterministic draws of
  100 clean reference examples. The reported standard deviation is the
  reference-resampling standard deviation, not a claim of 1,000 independent
  method datasets.
- Persisted artifacts:
  - `independent_reference_kid.json`
  - `independent_reference_kid_split_scores.csv`
  - `controlled_quality_table.csv`
  - `controlled_quality_table.tex`
  - `submitted_ppmark_quality_anchor.csv`
  - `submitted_anchored_quality_table.tex`
  - `rebuttal_quality_tables.md`
  - `sample_size_assessment.md`
  under `outputs/rebuttal_quality_baselines_n100_corrected`.

### Submitted PP-Mark anchor

- Do not replace the submitted PP-Mark measurements:
  - BRISQUE mean/p95: `9.924519 / 34.144476`,
    `n=100` (`25 prompts x 4 seeds`).
  - KID: `0.001101357399 +/- 0.000202021377`,
    `n=1000`, subset `500`, `20` splits.
- The controlled common-100 PP-Mark values are a distribution-shift and
  matching check, not a correction to the submitted values.

### Controlled common-100 quality results

KID values below are multiplied by 1,000.

| Method | BRISQUE mean | BRISQUE p95 | KID x 1,000 |
|---|---:|---:|---:|
| PP-Mark | 14.57 | 45.16 | -0.438 +/- 0.771 |
| Gaussian Shading | 9.67 | 27.35 | 0.621 +/- 1.096 |
| HiDDeN | 67.33 | 105.86 | 7.239 +/- 1.474 |
| InvisMark | 8.89 | 31.36 | -0.900 +/- 0.895 |
| RingID | 9.99 | 34.64 | -0.989 +/- 0.701 |
| Stable Signature | 9.26 | 32.69 | -0.814 +/- 0.878 |
| Tree-Ring | 11.06 | 41.90 | -0.090 +/- 0.922 |
| WIND | 8.27 | 27.42 | -0.386 +/- 0.879 |
| TrustMark | 8.29 | 29.44 | -0.174 +/- 1.001 |

Interpretation:

- PP-Mark does not dominate every baseline on BRISQUE and the rebuttal must
  not claim that it does.
- HiDDeN is clearly separated as the largest-distortion baseline.
- PP-Mark and the remaining low-distortion methods form a near-zero KID
  group at this sample size. Their fine ordering is not statistically
  meaningful.
- The correct trade-off claim is that PP-Mark retains near-clean
  distributional fidelity while incurring a measurable no-reference quality
  cost in exchange for proof-backed provenance and the submitted forgery
  resistance. This is stronger and more accurate than a best-quality claim.

### Sample-size decision

- No additional generation is required for this rebuttal response.
- `n=100` matches the submitted BRISQUE and attack-evaluation sample sizes and
  is sufficient to add the missing descriptive baseline quality axis.
- It is not sufficient to claim fine KID ranking or precise p95 superiority:
  at `n=100`, only about five observations determine the upper five-percent
  tail.
- Increase to at least `n=500` per method, preferably `n=1000` for KID, only
  if the response intends to claim statistically resolved visual-quality
  superiority. That stronger claim is unnecessary and unsupported by the
  present result.

### Response 9 draft

> Thank you; we agree that the submitted comparison should expose the quality
> axis for every method, not only PP-Mark. The submitted PP-Mark measurements
> remain BRISQUE mean/p95 \(9.92/34.14\) (\(n=100\), 25 prompts with four
> seeds) and KID \(1.101\pm0.202\times10^{-3}\) (\(n=1000\), subset 500,
> 20 splits). We additionally evaluated every baseline on the same 100
> prompt--seed pairs used by our cross-method attack evaluation. BRISQUE
> mean/p95 and KID \(\times10^3\), respectively, are: Gaussian Shading
> \(9.67/27.35,\ 0.621\pm1.096\); HiDDeN
> \(67.33/105.86,\ 7.239\pm1.474\); InvisMark
> \(8.89/31.36,\ -0.900\pm0.895\); RingID
> \(9.99/34.64,\ -0.989\pm0.701\); Stable Signature
> \(9.26/32.69,\ -0.814\pm0.878\); Tree-Ring
> \(11.06/41.90,\ -0.090\pm0.922\); WIND
> \(8.27/27.42,\ -0.386\pm0.879\); and TrustMark
> \(8.29/29.44,\ -0.174\pm1.001\). Baseline KID uses a disjoint pool of 900
> clean images and 1,000 reference resamples; finite-sample unbiased KID can
> be slightly negative, so near-zero negative estimates are not interpreted
> as better-than-zero distances.
>
> These results make the trade-off more explicit. PP-Mark is not the
> lowest-BRISQUE method, and we will not claim otherwise; its distributional
> KID remains near the clean reference, while HiDDeN exhibits substantially
> larger distortion. Read together with the submitted attack tables
> (PP-Mark score transfer FAR \(0.0033/0.0067\) for TR/GS sources and Accept
> FAR \(0\)), the result shows that PP-Mark obtains proof-backed provenance
> and stronger forgery resistance with a measurable but bounded visual
> quality cost. We will add the complete baseline-quality table and this
> qualification.

### Rebuttal-use recommendation

- If space permits, show the controlled common-100 table and state the
  submitted PP-Mark anchor in prose. This is the strongest scientific
  presentation because it preserves the submission while exposing the
  prompt-distribution sensitivity of BRISQUE.
- If only one compact table can be shown, use
  `submitted_anchored_quality_table.tex` and retain its source/protocol
  footnotes. Do not describe its rows as a single paired experiment.

## Response 10: quantitative SOTA comparison and visual quality

### Reviewer question

> The experimental section lacks quantitative comparisons with existing
> state-of-the-art public watermarking methods. Furthermore, it fails to
> evaluate the impact of the watermark on image quality, leaving the
> potential degradation of visual fidelity unassessed by standard metrics
> such as FID and CLIP Score.

### Exact submitted-paper mapping

- Main lines 250--259: common SD2.1 setup, same clean-calibrated 1% FPR, and
  runtime pointer.
- Main lines 260--280, Table 2: Sig-only, Sig+Score, PP-Mark(score), and
  PP-Mark(accept) under forged-binding and compliance-bypass attacks.
- Main lines 281--317, Table 3 and Figure 3: cross-method imprint-forgery
  comparison with TrustMark, InvisMark, RingID, StableSig, HiDDeN, WIND, and
  PP-Mark.
- Main lines 318--349, Table 4 and Figure 4: white-box PGD FAR, breach step,
  and PSNR across the same target methods.
- Main lines 350--358: removal-attack comparison summary.
- Appendix D.2, lines 596--602, Table 6: per-method runtime.
- Appendix D.4, lines 611--620, Figure 8: submitted PP-Mark BRISQUE p95.
- Appendix F, lines 628--665, Tables 8--9: direct TR/GS and step-wise transfer
  FAR.
- Appendix G, lines 667--712, Tables 10--12 and Figures 9--10: SDXL
  quantitative comparisons.
- Appendices H--K, lines 713--753, Tables 13--16: regeneration,
  steganalysis, SPSA, and image-transform results.
- Appendix L, lines 754--758, Table 17: submitted PP-Mark KID.

The absolute statement that there is no quantitative SOTA comparison or no
quality evaluation is therefore inaccurate. The valid underlying concerns
are that the comparisons are distributed across main/appendix tables, the
main text does not prominently point to the quality results, and the
submitted quality metrics cover PP-Mark but not the baselines.

### New matched CLIPScore evaluation

- Added `scripts/eval_rebuttal_clipscore_baselines.py`.
- Input: the validated common-100 prompt/seed manifest used by all methods.
- Metric: cosine similarity between normalized image/text embeddings from
  `openai/clip-vit-base-patch32`, multiplied by 100.
- Every method uses the same 100 prompts.
- Output:
  `outputs/rebuttal_quality_baselines_n100_corrected/clipscore`.
- Results:

| Method | CLIPScore x100 | 95% CI of mean | Paired delta vs clean |
|---|---:|---:|---:|
| Clean | 32.55 | 31.96--33.13 | 0.00 |
| PP-Mark | 31.64 | 30.91--32.35 | -0.91 [-1.55, -0.29] |
| Gaussian Shading | 32.42 | 31.71--33.14 | -0.12 [-0.68, 0.40] |
| HiDDeN | 31.30 | 30.69--31.90 | -1.25 [-1.58, -0.94] |
| InvisMark | 32.53 | 31.94--33.10 | -0.02 [-0.06, 0.02] |
| RingID | 32.41 | 31.72--33.11 | -0.13 [-0.80, 0.51] |
| Stable Signature | 32.50 | 31.91--33.06 | -0.05 [-0.10, 0.00] |
| Tree-Ring | 32.30 | 31.60--32.98 | -0.25 [-0.92, 0.40] |
| WIND | 32.28 | 31.58--32.98 | -0.27 [-0.85, 0.30] |
| TrustMark | 32.54 | 31.98--33.12 | -0.00 [-0.08, 0.07] |

Interpretation: PP-Mark has a measurable semantic-alignment cost relative to
clean images, but the absolute reduction is modest (`0.91` on the x100
scale), and it remains above HiDDeN. Do not claim best CLIPScore.

### FID decision

- Do not report a newly computed cross-method FID from the current baseline
  pool.
- Each baseline has only 100 images while standard FID uses a 2,048-
  dimensional Inception covariance; the covariance is singular and the
  estimate is strongly finite-sample biased.
- KID is already the submitted distributional-fidelity metric and is an
  unbiased finite-sample estimator. The new disjoint-reference KID table plus
  matched CLIPScore addresses distributional and semantic quality without
  introducing an underpowered FID.

### Consolidated submitted robustness table

All values below come from submitted Tables 3 and 4.

| Method | Transfer FAR, TR | Transfer FAR, GS | White-box FAR |
|---|---:|---:|---:|
| TrustMark | 0.0367 | 0.0267 | 1.00 |
| InvisMark | 0.0300 | 0.0167 | 1.00 |
| RingID | 0.0400 | 0.0133 | 1.00 |
| Stable Signature | 0.0333 | 0.0167 | 1.00 |
| HiDDeN | 0.0233 | 0.0233 | 1.00 |
| WIND | 0.0100 | 0.0100 | 1.00 |
| PP-Mark(score) | 0.0033 | 0.0067 | 0.76 |
| PP-Mark(accept) | 0.0000 | 0.0000 | 0.00 |

Tree-Ring and Gaussian Shading are the imprint sources in the transfer
table; submitted Appendix F, Table 8 reports their direct FAR averages as
`0.953` and `1.00`.

### Response 10 draft

> Thank you. We agree that the cross-method evidence and quality assessment
> were too dispersed in the submission and should have been consolidated.
> Quantitative public-watermark comparisons are present in the submitted
> paper: Tables 2--4 report verification-ablation, cross-method imprint
> transfer, and white-box PGD results; Table 6 reports runtime; and
> Tables 8--16 provide the step-wise, SDXL, regeneration, steganalysis, SPSA,
> and transformation breakdowns. The central submitted comparisons are:
>
> | Method | Transfer FAR (TR/GS) | White-box FAR |
> |---|---:|---:|
> | TrustMark | 0.0367 / 0.0267 | 1.00 |
> | InvisMark | 0.0300 / 0.0167 | 1.00 |
> | RingID | 0.0400 / 0.0133 | 1.00 |
> | Stable Signature | 0.0333 / 0.0167 | 1.00 |
> | HiDDeN | 0.0233 / 0.0233 | 1.00 |
> | WIND | 0.0100 / 0.0100 | 1.00 |
> | PP-Mark(score) | 0.0033 / 0.0067 | 0.76 |
> | PP-Mark(accept) | 0.0000 / 0.0000 | 0.00 |
>
> The submission also reports PP-Mark quality in Appendix D.4
> (BRISQUE p95 34.14, n=100) and Appendix L (KID
> \(1.101\pm0.202\times10^{-3}\), n=1000, subset 500, 20 splits).
> The actual omission was the same quality evaluation for the baselines. We
> have now computed BRISQUE, KID, and matched CLIPScore for all methods:
>
> | Method | BRISQUE p95 | KID x 1,000 | CLIPScore x100 |
> |---|---:|---:|---:|
> | PP-Mark | 34.14 | 1.101 +/- 0.202 | 31.64 |
> | Gaussian Shading | 27.35 | 0.621 +/- 1.096 | 32.42 |
> | HiDDeN | 105.86 | 7.239 +/- 1.474 | 31.30 |
> | InvisMark | 31.36 | -0.900 +/- 0.895 | 32.53 |
> | RingID | 34.64 | -0.989 +/- 0.701 | 32.41 |
> | Stable Signature | 32.69 | -0.814 +/- 0.878 | 32.50 |
> | Tree-Ring | 41.90 | -0.090 +/- 0.922 | 32.30 |
> | WIND | 27.42 | -0.386 +/- 0.879 | 32.28 |
> | TrustMark | 29.44 | -0.174 +/- 1.001 | 32.54 |
>
> PP-Mark retains the submitted BRISQUE/KID values; the new baseline
> BRISQUE/KID and all CLIPScores use the common 100-prompt evaluation pool.
> Baseline KID uses a disjoint 900-image clean reference. Unbiased KID can be
> slightly negative at finite n, so near-zero negative estimates are not
> interpreted as negative distances. We use KID rather than an n=100 FID:
> at this sample size, a 2,048-dimensional FID covariance is singular and
> strongly biased, whereas KID provides an unbiased finite-sample
> distributional comparison.
>
> The expanded results make the trade-off explicit rather than claiming that
> PP-Mark has the best visual quality. PP-Mark incurs a modest CLIPScore
> reduction relative to clean images (31.64 vs. 32.55) and is not the
> lowest-BRISQUE method, while retaining low distributional KID. In return,
> it provides substantially stronger resistance under the evaluated forgery
> attacks and proof-backed public provenance. We will consolidate these
> comparisons in the experimental section and qualify the quality claim
> accordingly. Would this consolidated comparison and explicit
> quality--robustness characterization address your concern?

## Response 11 preparation: sequential transformation pilot (2026-07-26)

Reviewer P58K accepts the existing robustness evidence but asks for a realistic
sequential transformation pipeline rather than isolated operators. This is a
targeted strengthening request, not a correctness failure. The staged protocol
is therefore:

1. Pilot two fixed pipelines on ten positive/negative pairs:
   `crop 90% area -> resize 512 -> JPEG q75` and
   `crop 75% area -> resize 512 -> JPEG q75`.
2. Freeze the submitted score threshold at `2.4665623073926706`; do not retune
   it after transformation.
3. Inspect positive pass counts and score margins, transformed-negative score
   inflation, and runtime before expanding to `n=100`.
4. Escalate to JPEG q50 only if the q75 stress condition remains saturated.
   Do not begin with the unnecessarily harsh crop75+JPEG25 composition.
5. Report at most one realistic and one stress pipeline. SDXL and cross-method
   reruns are not required for this reviewer question.

Preparation artifacts:

- `scripts/prepare_rebuttal_sequential_pilot.py`: deterministic CPU-only image
  and manifest preparation. Positive indices 100,110,...,190 are disjoint from
  the first 100 examples used by the submitted transformation table.
- `scripts/run_rebuttal_sequential_pilot_fast.sh`: bounded 3x3+5x5 crop search,
  DDIM-50/bfloat16, for direction and runtime.
- `scripts/run_rebuttal_sequential_pilot_full.sh`: submitted high-recovery
  9x9+9x9 crop search, DDIM-100/float32, for confirmation only.
- `scripts/summarize_rebuttal_sequential.py`: frozen-threshold TPR/FPR, exact
  Clopper--Pearson intervals, and score-margin summaries.

Pilot negatives intentionally come from the calibration split and are only a
gross FPR-inflation diagnostic. Any final reported FPR must use a disjoint
held-out negative set. No GPU evaluation was run during preparation.

### Response 11 preparation correction and validation

The first preparation pass incorrectly attached each clean calibration image
to metadata from a positive evaluation example with the same split-local
`prompt_id`. Prompt identifiers are not global across the eval and calibration
splits. The prepared manifests were regenerated so that:

- every positive image uses its own `methods/pp_mark/eval` metadata;
- every clean calibration negative uses the corresponding
  `methods/pp_mark/calib` metadata with exactly matching prompt and seed;
- metadata paths are repository-relative, allowing mixed eval/calib metadata
  in one manifest while the instance runner uses the repository root as
  `--metadata-root`.

Independent post-generation validation passed:

- 40/40 metadata prompt/seed matches;
- 40/40 eval/calib partition checks;
- 40/40 transformed-image SHA-256 checks;
- 40/40 RGB 512x512 image checks;
- 20 rows per condition (10 positive, 10 negative);
- deterministic force-regeneration hashes:
  `ED544E8A205A5C2C6C53243374AB9A776E1C3E9A334BD3EA333E18896FE126B5`
  for crop90 and
  `198189AB788F7668169AC01F4DC1D0ED54F1D0651C543BB9763A02D1216EB436`
  for crop75.

The prepared directory contains no detector scores; GPU evaluation remains
pending on the instance.

### Response 11 fast pilot result (A100 instance, 2026-07-26)

The prepared pilot was run in parallel on two NVIDIA A100 80GB PCIe GPUs with
the frozen submitted threshold `tau=2.4665623073926706`. The environment was
torch 2.5.1+cu121, diffusers 0.30.2, transformers 4.41.2, accelerate 0.33.0,
NumPy 1.26.4, and SciPy 1.13.1. A one-image sanity check passed with score
3.2267 in 28.6 seconds. The two 20-row conditions completed in 690 seconds
wall time when run concurrently.

| Condition | Positive TPR (n=10) | Calibration-negative diagnostic FPR (n=10) | Positive median | Positive p10 |
|---|---:|---:|---:|---:|
| crop 90% -> resize -> JPEG q75 | 0.30 | 0.50 | 2.044 | 1.594 |
| crop 75% -> resize -> JPEG q75 | 0.40 | 0.40 | 2.296 | 1.396 |

These values are not suitable for the rebuttal and must not be expanded to
`n=100` as-is. The fast crop search maximizes over multiple alignment
candidates, but the submitted 1%-FPR threshold was calibrated without that
search. This caused severe null-score inflation: pilot AUC was 0.48 for crop90
and 0.43 for crop75, so the positive and negative score distributions were not
separated. The calibration negatives are not an independent final test set,
but they are sufficient to reject this detector configuration as a reporting
candidate. A matched-search calibration or a different, explicitly scoped
sequential pipeline is required before any final robustness number is used.

Local result files are under
`outputs/rebuttal_sequential_pilot_v1/eval_fast/`. No full-search or `n=100`
expansion was started.

### Response 11 historical-artifact reuse audit

Before running further controls, the submitted transformation artifacts were
indexed against the paper table. The historical data already contain the
following results:

| Existing artifact | n | Pass rate at submitted tau |
|---|---:|---:|
| Clean calibration, no search | 100 | FPR 0.01 |
| Watermarked baseline, no search | 100 | TPR 0.98 |
| JPEG q25 | 100 | TPR 0.94 |
| Box blur 8x8 | 100 | TPR 1.00 |
| Additive noise sigma 0.1 | 100 | TPR 0.86 |
| Brightness jitter 0.6 | 100 | TPR 0.98 |
| Random crop 75%, fast search | 100 | TPR 0.70 |
| Random crop 75%, full search | 100 | TPR 0.87 |

The values above were independently recomputed from these submitted-result
sources at `tau=2.4665623073926706`:

- `outputs/attacks/muller_forgery_report_prep/methods/pp_mark/calib/accept_check_calib/clean_scores.csv`;
- `outputs/attacks/muller_forgery_report_prep/methods/pp_mark/eval/accept_check_wm_full/scores.csv`;
- `outputs/attacks/muller_forgery_report_prep/geom_eval_rerun_full100/eval_geom4_100/scores.csv`;
- `outputs/attacks/muller_forgery_report_prep/geom_eval_rerun_full100/eval_crop_shift_refine_100/scores.csv`;
- `outputs/attacks/muller_forgery_report_prep/geom_eval_rerun_full100/eval_crop_full100_maxlevers/scores.csv`.

The recomputation produced, respectively, clean FPR `1/100`, untransformed
watermarked TPR `98/100`, JPEG/blur/noise/brightness TPRs
`94/100`, `100/100`, `86/100`, and `98/100`, and crop TPRs `70/100`
(fast search) and `87/100` (full search). All positive-result files contain
exactly the same unique indices `0,...,99`.

The baseline and every SD 2.1 single-transform manifest use the same evaluation
indices 0--99, so these are paired controls rather than unrelated datasets.
No historical artifact applies multiple transformations sequentially, and no
historical geometric-transform score file contains transformed negatives under
the same crop search. Those are the only substantive gaps for Response 11.

The first new composite pilot used disjoint positive indices 100,110,...,190.
That was conservative but unnecessary for a paired extension of the submitted
table and makes direct comparison to the historical single-transform scores
weaker. The separately prepared four-condition diagnostic set has therefore
not been transferred or executed. Existing baseline and single-transform
artifacts should be reused; only a matched sequential composition and its
search-matched negative calibration should be generated next.

### Response 11 exact submitted-crop recompression pilot

The sequential experiment was restarted from the actual submitted SD 2.1
random-crop artifacts rather than newly generated crops. The inputs were
`geom_eval_rerun_full100/img_000--002/crop_075.png`, which already contain the
submitted random 75% area crop followed by bicubic resize. A single JPEG
roundtrip at qualities 75, 50, and 25 was added to each image. The same three
indices and exact metadata were used for every quality.

Evaluation reused the submitted crop alignment configuration without a fast
approximation: FP32 DDIM inversion with 100 steps, `max-shift=16`, crop keep
ratio 0.75, a 9x9 crop grid, and 9x9 refinement at step 0.03. The frozen
threshold was `tau=2.4665623073926706`.

| idx | Submitted crop-only | + JPEG q75 | + JPEG q50 | + JPEG q25 |
|---:|---:|---:|---:|---:|
| 0 | 3.0340 (pass) | 2.6245 (pass) | 3.2875 (pass) | 3.2039 (pass) |
| 1 | 2.2848 (fail) | 2.7260 (pass) | 2.5231 (pass) | 2.4242 (fail) |
| 2 | 3.2314 (pass) | 2.5926 (pass) | 2.9409 (pass) | 3.4446 (pass) |

Pilot pass counts were crop-only 2/3, q75 3/3, q50 3/3, and q25 2/3. Mean
paired score changes relative to crop-only were -0.2024, +0.0671, and +0.1741
for q75, q50, and q25, respectively. JPEG quality does not induce a monotone
score change at this sample size because recompression can perturb the
alignment-search optimum in either direction. Source, recompressed-image, and
metadata hashes were checked, and every result file contains exactly indices
0, 1, and 2. This is a go/no-go pilot only; no n=100 expansion has been run.

The Elice SSH tunnel terminated foreground child processes during the first
attempt even though the instance remained powered on. The successful run used
`nohup` plus `setsid`, after which all nine exact-setting evaluations completed
independently of the tunnel session.

### Response 11 q25 extension to n=20

The exact submitted-crop experiment was extended only at the strongest tested
recompression setting, JPEG q25, from indices 0--2 to the fixed index set
0--19. Indices 3--19 were evaluated once on two A100 GPUs and then merged with
the preserved three-image pilot; no pilot row was rerun. The transform chain is
the submitted random 75% crop followed by bicubic resize, followed by JPEG q25
recompression. Evaluation again used FP32 DDIM-100, `max-shift=16`, the 9x9
crop grid and 9x9 refinement at step 0.03, and the frozen submitted threshold
`tau=2.4665623073926706`.

| Condition on the same 20 images | Pass count | TPR | Exact 95% CI | Mean score | Median | p10 | Minimum |
|---|---:|---:|---:|---:|---:|---:|---:|
| Submitted crop 75% -> resize | 16/20 | 0.80 | [0.563, 0.943] | 2.9638 | 2.9243 | 2.3465 | 2.1505 |
| Crop 75% -> resize -> JPEG q25 | 19/20 | 0.95 | [0.751, 0.999] | 3.0008 | 2.8871 | 2.5113 | 2.4242 |

The paired transitions were 16 pass-to-pass, three fail-to-pass, one
fail-to-fail, and zero pass-to-fail. Scores increased on 12 images and
decreased on eight; the mean and median paired changes were +0.0370 and
+0.1546. These search-maximized scores do not support a claim that JPEG
improves detection. The defensible conclusion is narrower: on this n=20
paired check, adding severe q25 recompression to the submitted crop-and-resize
pipeline did not cause detection to collapse.

The merged result is
`outputs/rebuttal_existing_crop_recompression_q25_n20/scores.csv`. The 20
rows have unique indices 0--19. Their crop-only controls exactly match the
corresponding rows of the submitted 100-image crop CSV. SHA-256 verification
of all 20 submitted crop inputs, recompressed outputs, and copied metadata
found zero mismatches. The 17-image extension is under
`outputs/rebuttal_existing_crop_recompression_q25_idx3_19/`; its merged CSV
contains exactly indices 3--19. No n=50 or n=100 expansion was started.

### Response 11 planned q25 extension to n=50

After reviewing the n=20 result (19/20 passes), the next fixed extension is to
evaluate indices 20--49 at JPEG q25. Existing indices 0--19 and their outputs
will be preserved and will not be rerun. The 30 new images must reuse the same
submitted crop-75%-then-resize artifacts, metadata, frozen threshold, and exact
FP32 DDIM-100 alignment configuration used above. The extension will be
sharded across the additional instances supplied for this run. Final n=50
statistics will be computed only after verifying unique indices 0--49 and all
source/output/metadata hashes.

### Response 11 q25 extension result at n=50

Indices 20--49 were evaluated on four A100 80GB GPUs in fixed shards of
8/8/7/7 and merged with the preserved indices 0--19. The resulting file
`outputs/rebuttal_existing_crop_recompression_q25_n50/scores.csv` contains
exactly 50 unique rows, indices 0--49. The downloaded 30-row extension CSV
matched the remote SHA-256 digest, and all prepared source, recompressed-image,
and metadata hashes matched.

| Condition on the same 50 original images | Pass count | TPR | Exact 95% CI | Mean score | Median | p10 | Minimum |
|---|---:|---:|---:|---:|---:|---:|---:|
| Submitted crop 75% -> resize | 41/50 | 0.82 | [0.686, 0.914] | 2.9938 | 2.9151 | 2.3502 | 2.1505 |
| Crop 75% -> resize -> JPEG q25 | 44/50 | 0.88 | [0.757, 0.955] | 3.0002 | 2.9315 | 2.4468 | 2.1271 |

The paired transitions were 37 pass-to-pass, four pass-to-fail, seven
fail-to-pass, and two fail-to-fail. Thus final composite TPR was 44/50, while
retention conditional on passing the submitted crop condition was 37/41
(0.902). The strict joint pass rate was 37/50 (0.74). Scores increased on 27
images and decreased on 23, with mean paired change +0.0064. The discordant
counts do not support an improvement claim (exact two-sided McNemar
`p=0.5488`). The appropriate interpretation is no detected additional
degradation at this sample size, not that JPEG improves the detector.

### Response 11 q25 extension to n=100

The final fixed extension uses all remaining original indices 50--99, without
filtering on the submitted crop verdict. The 50 rows were prepared at q25,
verified with zero source/output/metadata hash errors, and split across four
A100 GPUs as 13/13/12/12. Existing indices 0--49 are preserved and will not be
rerun. The primary final statistic will be composite TPR over all 100 original
images; the four paired transition counts will be reported as secondary
interpretation.

The four-GPU run completed and the two preserved halves were merged into
`outputs/rebuttal_existing_crop_recompression_q25_n100/scores.csv`. The file
contains exactly 100 unique rows, indices 0--99. The downloaded second-half
CSV matched its remote SHA-256 digest. Across all four preparation batches,
all 100 source crop images, q25 outputs, and metadata copies matched their
recorded SHA-256 hashes, with zero mismatches.

| Condition on the same 100 original images | Pass count | TPR | Exact 95% CI | Mean score | Median | p10 | Minimum |
|---|---:|---:|---:|---:|---:|---:|---:|
| Submitted crop 75% -> resize | 87/100 | 0.87 | [0.788, 0.929] | 2.9825 | 2.9295 | 2.4325 | 2.1505 |
| Crop 75% -> resize -> JPEG q25 | 91/100 | 0.91 | [0.836, 0.958] | 3.0833 | 3.0264 | 2.5044 | 2.1271 |

Paired transitions were 80 pass-to-pass, seven pass-to-fail, 11 fail-to-pass,
and two fail-to-fail. Composite final-output TPR was therefore 91/100. The
strict joint rate, which counts an image as a success only if it passes both
the submitted crop condition and the final recompressed condition, was
80/100. Conditional retention among the 87 crop-pass images was 80/87
(0.920). The exact two-sided McNemar test on the 18 discordant pairs gave
`p=0.4807`, so the four-point difference does not support a claim that JPEG
improves detection. The correct claim is that TPR remained comparable under
the three-operation composite pipeline.

Scores increased on 59 images and decreased on 41. Mean and median paired
changes were +0.1008 and +0.1666. A separate pixel-level check confirmed that
q25 recompression was actually applied: across all 100 pairs, mean PSNR was
31.46 dB and mean absolute pixel error was 5.01 (8-bit scale). Thus the
non-monotone score movement is not attributable to accidentally evaluating
the unrecompressed source images.
## Response 12: SEAL comparison and semantic-vs-exact binding (2026-07-27)

- Reviewer request: distinguish PP-Mark from SEAL, justify not replacing the
  exact image hash with a semantic input, analyze the robustness/security
  trade-off, and provide a comparative experiment.
- Primary-source audit:
  - SEAL does not use a semantic hash as a ZKP or receipt-binding input. At
    generation, it captions a proxy image, embeds the caption, and uses
    secret-salt SimHash patches to sample the initial diffusion noise. At
    detection, it captions the suspect image, reconstructs semantic SimHash
    noise using the secret salt, inverts the image, and counts matching noise
    patches. Its goal is database-free, semantic-conditioned statistical
    watermark detection and tamper/forgery resistance.
  - PP-Mark(score) is the transform-tolerant statistical layer; PP-Mark(accept)
    additionally uses the exact SHA-256 pixel digest to bind a proof receipt to
    one canonical image artifact and its generation context. This targets
    public verification and cross-image proof-transfer resistance, which SEAL
    does not claim.
  - Consequently, SEAL-style semantic matching is not a drop-in replacement
    for `h_img`. Exact equality of a semantic bit string remains brittle; fuzzy
    semantic acceptance creates a non-singleton collision neighborhood in
    which a receipt can be replayed onto a distinct image.
- Existing same-suite evidence (not yet a strictly matched recalibration):
  SEAL reports detection accuracies on SD2.1 for Clean/Rotate75/JPEG25/
  CropScale75/Blur8/Noise0.1/Brightness of
  0.980/0.774/0.945/0.668/0.938/0.976/0.992. PP-Mark's submitted score-mode
  rates are 1.00/0.85/0.94/0.87/1.00/0.86/0.98. The seven-condition means are
  0.896 and 0.929, respectively. These numbers are only preliminary because
  SEAL's paper figure uses its tuned detector threshold whereas PP-Mark reports
  TPR at FPR <= 1%.
- Recommended rebuttal experiment:
  1. Run the official SEAL implementation on the same 100-prompt SD2.1 pool.
  2. Recalibrate both methods at 1% FPR and report Clean, JPEG q=25, and random
     75% crop+resize (optionally the full six-transform suite).
  3. Add a semantic-binding ablation on distinct same-prompt regenerations:
     exact SHA equality versus SEAL-style semantic patch acceptance. This
     quantifies the benign robustness gained and the cross-image receipt-reuse
     neighborhood introduced by fuzzy semantic binding.
- Response framing: acknowledge SEAL as complementary; do not claim that
  semantic hashing is categorically insecure or that PP-Mark dominates SEAL.
  State that the methods optimize different guarantees, retain SHA-256 only in
  exact `accept` mode, and retain transform tolerance in `score` mode.
- Sources inspected: ICCV 2025 main paper and supplement, and official SEAL
  repository commit `92d31b31b93a6e373fe88a584c640beafb68c6fa` (2026-05-10).

### Response 12 draft

Thank you for pointing us to SEAL. We agree that it is an important
comparison, but semantic information enters SEAL and PP-Mark at different
interfaces. SEAL captions a proxy/suspect image, maps its semantic embedding
through secret-salt SimHash, and statistically compares the resulting
semantic-conditioned noise patches with the inverted latent. In PP-Mark, raw
pixels do not determine the watermark key or payload: the binding
`b = H(h_ctx || k)` is derived from the generation context and producer key.
The exact image hash enters only P3, where it makes the opening challenge
image-specific; P1--P4 additionally check key/context consistency, the Merkle
commitment, opening selection, and the committed latent-embedding relation.
Thus, SEAL targets database-free semantic-conditioned detection, whereas
PP-Mark separates transform-tolerant statistical detection (`score`) from
public verification of an exact artifact against its associated receipt
(`accept`).

A SEAL-style semantic representation is therefore not a drop-in replacement
for `h_img`. If its bit string is required to match exactly, a changed caption
or embedding still invalidates the receipt and provides little robustness
benefit. If similarity is accepted fuzzily, multiple distinct images occupy
the same accepted semantic neighborhood, so a receipt for one image can be
reused within that neighborhood. This is not a weakness of SEAL: its full
detector also checks the inverted latent and does not use the semantic
representation as a cryptographic receipt identifier. It is instead the
reason we retain SHA-256 for `accept` and use the correlation score for
transformed copies.

The existing results also permit the following comparison on the common
SD2.1/50-step transformation suite:

| Transformation | SEAL, published detection accuracy | PP-Mark(score), submitted TPR |
|---|---:|---:|
| Clean | 0.980 | 1.00 |
| Rotation 75 deg | 0.774 | 0.85 |
| JPEG q=25 | 0.945 | 0.94 |
| Crop 75% + resize | 0.668 | 0.87 |
| Blur 8x8 | 0.938 | 1.00 |
| Noise sigma=0.1 | 0.976 | 0.86 |
| Brightness | 0.992 | 0.98 |

SEAL values are from Arabi et al., Fig. 6; PP-Mark values are from the
submitted transformation appendix. We present this as a behavioral comparison,
not a strict ranking: SEAL reports detection accuracy under its tuned
patch/match thresholds, whereas PP-Mark reports TPR at FPR <= 1%. The table
nevertheless shows that exact receipt binding does not remove practical
post-processing tolerance, because transformed copies are handled by the
score layer. We will add the SEAL discussion and this calibration caveat, and
clarify that PP-Mark deliberately offers two verification semantics: robust
statistical screening of transformed copies and exact cryptographic audit of
the original artifact.

## Final OpenReview posting record (2026-07-29)

The three final reviewer-thread comments were posted and archived. The
authoritative index, AC meta-review, timestamps, rating context, response
coverage, and score-request strategy are in:

- `docs/neurips_2026_openreview_rebuttal_record_2026-07-29.md`

Final response files:

- `docs/reviewer_P58K_response_5000.md`
- `docs/reviewer_iryC_response_5000.md`
- `docs/reviewer_Ef3K_response_final_2026-07-29.md`

Closing strategy actually posted:

- P58K: soft request to reconsider the assessment.
- iryC: direct request to reconsider the score, matching the reviewer's
  explicit statement that a score increase was possible.
- Ef3K: no score request.

The AC-wide synthesis comment was posted on 2026-07-29 at 01:47 and modified
at 01:49. Its final text is archived at
`docs/ac_GDXR_official_comment_5000.md`. It maps the response to the AC's
three decision axes and makes the routine Score versus high-assurance Accept
deployment distinction the lead point.
