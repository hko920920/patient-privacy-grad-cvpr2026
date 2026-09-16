# PP-Mark NeurIPS 2026 OpenReview Rebuttal Record

Submission: 15860, PP-Mark: Provable and Publicly Verifiable Watermarking for Generative AI

Record date: 2026-07-29 (Asia/Seoul)

This file records the final author comments posted to the three reviewer threads. OpenReview-rendered tables are preserved as Markdown in the linked files; whitespace from the copied web view is normalized.

## Area-chair meta-review

Area Chair: GDXR

Posted: 2026-07-20 13:42; modified 2026-07-24 02:56

> The paper has received all borderline scores, with the majority tending the paper toward rejection in the initial round. There are several concerns raised on the clarity, comparisons and practicality of the method.
>
> P58K (4 / Borderline Accept) - questions the generality of the approach beyond SD and the reporting of the visual quality metrics. Suggests simpler engineering baseline for comparison.
>
> IrYC (3 / Borderline Reject) - concerns over the practicality of the system particularly the ZKP aspects (including proof time and robustness to real world platform manipulations vs. SHA hash which will break under the same), and its motivation vs. semantic watermarking. Lacking comparison to SOTA baselines and related methods such as gradient manipulation. No use of standard metrics for visual fidelity.
>
> Ef3K (3 / Borderline Reject) - shares the concerns over practicality (ZKP, transformations in real world) voiced by IrYC, and additional queries on the accommodation of multiple images.

## Final thread responses

| Reviewer | Initial rating | Confidence | Author comment | Score-request tone |
|---|---:|---:|---|---|
| P58K | 4, Borderline Accept | 3 | 2026-07-29 01:27 | Soft: reconsider the assessment |
| iryC | 3, Borderline Reject | 3 | 2026-07-29 01:08 | Direct: reconsideration of the score, matching the reviewer's explicit offer to raise it |
| Ef3K | 3, Borderline Reject | 4 | 2026-07-29 00:10; modified 01:26 | No score request |

Final AC-wide author comment:

- Posted 2026-07-29 01:47; modified 01:49.
- Authoritative text: `docs/ac_GDXR_official_comment_5000.md`.
- Organized around the meta-review's three axes: practicality, quantitative/engineering comparisons, and clarity/scope.
- Foregrounds the operational distinction between routine Score verification (0.72 s per image) and one-time, per-origin-artifact Accept proving (48.6 s) for high-assurance legal, regulatory, or untrusted-generator verification.
- Requests assessment on the clarified contribution without directly requesting a numerical AC score change.

Authoritative response files:

- P58K: `docs/reviewer_P58K_response_5000.md`
- iryC: `docs/reviewer_iryC_response_5000.md`
- Ef3K: `docs/reviewer_Ef3K_response_final_2026-07-29.md`

## Response coverage

P58K:

- New persistent three-operation transform result: 80/100 on crop-resize-JPEG, no survivor filtering.
- New Ed25519 Bare-Sig and Context-Sig controls, including the signed-unwatermarked-image compliance distinction.
- Matched baseline BRISQUE and KID table.
- Explicit withdrawal of architecture-agnostic wording and latent-diffusion scope correction.
- Closing: `If these new results and scope corrections resolve your concerns, we would be grateful if you would reconsider your assessment.`

iryC:

- Main-text notation table restoration and Figure 1 P1-P4 flow revision.
- Consolidated SOTA FAR evidence plus BRISQUE/KID/CLIPScore table.
- SEAL mechanism, guarantee, semantic-binding, and transformation comparison.
- Exact Accept versus transform-tolerant Score deployment distinction and measured N trade-off.
- DGS comparison as a complementary managed-decoder defense.
- Closing: `If these clarifications and results resolve your concerns, we would appreciate reconsideration of the score.`

Ef3K:

- SP1/RISC0/Halo2/PLONK/Groth16 proof-system results and latency-size trade-off.
- Context resolution after metadata stripping, with explicit unverified failure state.
- Core cross-context unlinkability scope and deployment key-identifier qualification.
- No score-update request was added because the review contained no explicit upward-reconsideration signal.

## Archival note

The final OpenReview web copy flattened Markdown and MathJax tables into line-oriented text. The linked Markdown files preserve their intended rows and columns. The Ef3K proof-system table, omitted by plain-text extraction between the Halo2 and PLONK paragraphs, is restored from the posted response source and the measured values used during rebuttal preparation.

## Post-rebuttal rating update (2026-08-01 KST)

- Reviewer `iryC` modified the official review at 11:00 KST and raised the
  rating from **3 (Borderline Reject)** to **5 (Accept)**, with confidence
  **4**. The public follow-up states that the response left no further
  questions and that the rating was raised accordingly.
- The visible reviewer-rating set is now **P58K 4 / iryC 5 / Ef3K 3**, with
  confidences **3 / 4 / 4**, respectively. The arithmetic mean is 4.0 and
  two of three ratings are on the accept side.
- The AC meta-review displayed above predates the rebuttal and this score
  change. Its statement that the majority tended toward rejection describes
  the initial 4/3/3 configuration; it is no longer an accurate description of
  the visible post-rebuttal rating majority.
- This is a material positive update, not a final decision. Final disposition
  still depends on the AC/SAC assessment, including the remaining confidence-4
  Borderline Reject and the venue-wide decision threshold.

## Second post-rebuttal rating update (2026-08-04 KST)

- Reviewer `P58K` modified the official review at **2026-08-04 01:27 KST**
  and raised the rating from **4 (Borderline Accept)** to **5 (Accept)**;
  confidence remains **3**.
- The public follow-up says: `I have no further questions or concerns`, states
  that all comments were addressed satisfactorily, and explicitly says the
  recommendation is being increased to Accept.
- The response items accepted as satisfactory are exactly the four issues in
  the original review: the persistent crop-resize-JPEG test, Bare-Sig and
  Context-Sig controls, matched baseline quality metrics, and withdrawal of
  architecture-agnostic generalization beyond two latent-diffusion models.
- The visible post-rebuttal rating set is now **P58K 5 / iryC 5 / Ef3K 3**,
  with confidences **3 / 4 / 4**. The arithmetic mean is **4.33**, and two of
  three reviewers now give an explicit Accept rating.
- Both reviewers who publicly responded to the rebuttal (`P58K` and `iryC`)
  report no remaining questions and upgraded to Accept. The only remaining
  negative visible rating is `Ef3K`'s confidence-4 Borderline Reject; that
  thread has no recorded post-rebuttal score change.
- The AC meta-review's initial description of a rejection-leaning majority was
  based on the pre-rebuttal **4/3/3** ratings and now materially understates the
  visible **5/5/3** reviewer support. The AC/SAC can still reject on the
  remaining practicality or significance judgment, so this is a strong
  positive decision signal rather than an acceptance guarantee.

## Contingency revision strategy if rejected (2026-08-01 KST)

- Preserve the distinction between a fast statistical signal and a
  cryptographic acceptance guarantee. The submitted SD2.1 white-box result is
  FAR 0.76 for `Score` and 0.00 for `Accept`; `Score` therefore cannot inherit
  the ZKP unforgeability claim or be described as secure without a named threat
  model.
- The viable repositioning is a **Score-first, proof-on-demand two-tier
  workflow**: use `Score` for routine, transformation-tolerant screening and
  invoke `Accept` only for exact-origin adjudication in high-assurance cases.
  `Accept` is an escalation verifier, not an ablation and not merely an
  appendix curiosity.
- The current mode names are not parallel: `score` names an internal statistic
  while `accept` names a final decision, which can make the routine path appear
  auxiliary. Naming remains open pending an author decision; the earlier long
  `Robust Verification / Proof-backed Origin Verification` proposal was
  rejected as cumbersome. The current compact candidate is **Detect / Attest**:
  Detect names transform-tolerant statistical detection, whereas Attest names
  proof-backed exact-origin attestation. Keep `score` as an internal scalar and
  `accept` as an algorithmic Boolean rather than using either as a mode name.
- Literature-usage check: `detection` is the standard watermarking term for a
  thresholded score evaluated by TPR/FPR and robustness tests (e.g.,
  Tree-Ring, Stable Signature, and watermark benchmarks). `Attestation` is
  established primarily in systems security and content provenance for
  cryptographic evidence about a device, program, or execution; C2PA uses it
  in that platform-oriented sense, while ZK-IMG provides a directly relevant
  precedent for "attesting to image transformations" with zero-knowledge
  proofs. Thus `Detect / Attest` is defensible but not a conventional paired
  watermarking taxonomy.
- If adopted, define `Attest` narrowly as **proof-backed embedding-process
  attestation**: evidence that the declared context/binding, committed trace,
  sampled embedding predicates, and exact-image binding verify. Explicitly
  disclaim hardware/TEE attestation and full diffusion-execution attestation.
  Report Detect with TPR/FPR, attack FAR, transformation robustness, and
  latency; report Attest with proof-generation/verification time, receipt
  size, soundness/forgery acceptance, replay, and substitution controls.
- Reviewer evidence shows that the dual-mode distinction was not simply
  overlooked. `iryC` explicitly praised the split between sub-second Score and
  high-assurance Accept, and `Ef3K` accurately restated that transformed images
  retain Score while exact Accept is lost. The residual objection was a value
  judgment: routine distribution appeared to use the statistical path while
  the cryptographic guarantee appeared confined to a costly niche. The
  submitted text states the operational split explicitly, but the abstract,
  introduction, Figure 1 caption, and main acceptance equation first frame
  full verification as Score-plus-proof; the two independent deployment paths
  become explicit only later. A revision should therefore change first-page
  information architecture and mode naming, not merely repeat the same
  explanation more often.
- Keep the decision flow, mode-specific guarantees, and one core proof result
  in the main paper. Move circuit engineering, alternative proof-system
  benchmarks, extended derivations, and detailed case studies to the
  supplement. Present 48.6 s as one-time producer-side proof generation for a
  cacheable origin receipt, separately from 0.72 s routine Score verification
  and Accept verification cost.
- A revised evaluation should compare `Score` only, signature controls, and
  `Score+Accept` under attacks that cross the statistical threshold; measure
  end-to-end latency as a function of escalation rate and show precisely which
  false claims are resolved by escalation.
- High-assurance case studies require a bit-exact origin and its trace/receipt.
  Do not imply that a transformed image can be proved post hoc without the
  required witness, or that `Score` certifies generator compliance, prevents
  proof replay, or provides cryptographic white-box-forgery resistance.
- A genuinely Score-centric standalone paper would require a different central
  claim and new score-side security evidence. Demoting ZKP to a supplemental
  ablation while retaining PP-Mark's current unforgeability motivation would
  recreate the prior criticism that the cryptographic layer is bolted on.

### Detailed structural redesign blueprint

This blueprint is inactive unless the submission is rejected. It does not
authorize edits to the manuscript under review.

#### 1. Replace the central thesis

- Current reader model: a public detector is unsafe, so PP-Mark adds a ZKP;
  full PP-Mark is Score plus proof, while Score is a fallback.
- Revised reader model: public provenance has two distinct questions that no
  single brittle decision rule answers well: (D) whether a distributed image
  carries the claimed transform-tolerant signal, and (A) whether an exact
  origin artifact is backed by a valid registered embedding execution.
- Present PP-Mark as one shared context-bound mark with two public interfaces,
  tentatively `Detect` and `Attest`. Detect owns statistical presence evidence;
  Attest owns proof-backed embedding-process provenance. Do not transfer
  Attest's theorem to Detect.

#### 2. Rewrite the title, TL;DR, abstract, and contributions

- Candidate title direction: `PP-Mark: Robust Detection and Proof-Backed
  Provenance for Generative Images`. Do not finalize a title until the venue
  and claim audit are fixed.
- TL;DR structure: one shared context-bound watermark supports fast detection
  of transformed images and selective ZK attestation of exact-origin claims.
- Abstract order: (i) the robustness/assurance tension; (ii) the two interfaces
  and their distinct questions; (iii) Detect evidence and 0.72 s verification;
  (iv) Attest guarantee, exact-origin boundary, and one-time/cached receipt;
  (v) attacks, models, and scoped conclusion.
- Contributions must be parallel: context-bound mark and Detect interface;
  Attest predicate/theorem; evaluation of each interface under its own threat
  model; workload-level cost/coverage of using them together.
- Never write an unqualified `PP-Mark is unforgeable`. Write that Attest has the
  stated unforgeability bound and Detect has measured/calibrated resistance in
  the declared attack setting.

#### 3. Replace Figure 1 with a branch-first architecture

- Shared left-hand generation spine: context and secret key -> binding and
  payload -> sampled latent embedding -> image, trace commitment, and optional
  receipt generation.
- Upper/right Detect branch: original or transformed derivative + public claim
  artifact -> inversion/alignment -> scalar score -> `DETECTED` or
  `NOT DETECTED`; label 0.72 s and transformation tolerance.
- Lower/right Attest branch: exact-origin artifact + public statement + receipt
  -> score check plus P1--P4 receipt verification -> `ATTESTED` or
  `UNVERIFIED`; label proof-generation and verification costs separately.
- Visually state that a transformed derivative can remain Detected without
  being Attested. There is no silent fallback and `UNVERIFIED` is not a claim
  that an image is natural or forged.
- State that a receipt is generated at origin or while the full witness/trace
  is retained; only its later verification is unconditionally on demand.

#### 4. Put an interface contract on page 1 or 2

- Add one compact table with rows Detect and Attest and columns: question,
  required inputs, output, transformation behavior, threat model, guarantee,
  producer cost, and verifier cost.
- Detect question: `Does this image carry the signal for the supplied claim?`
- Attest question: `Does this exact artifact have a proof-consistent registered
  embedding trace for the supplied context and binding?`
- Detect does not certify honest generator execution. Attest does not tolerate
  arbitrary pixel modification, attest hardware/TEE state, or prove the full
  diffusion execution.

#### 5. Reorder the method around interfaces, not implementation chronology

1. System contract, actors, artifacts, typed outcomes, and threat boundaries.
2. Shared generation: binding, payload, sampling, and latent embedding.
3. Detect: inversion, alignment, score, calibration, and statistical scope.
4. Attest: trace commitment, image-bound challenge, P1--P4, proof creation,
   and public verification.
5. Relationship: `ATTESTED` requires both Detect success and valid receipt;
   `DETECTED` alone remains a legitimate routine result with a narrower claim.
6. Deployment/caching policy and exact conditions for re-attestation.
7. Formal guarantees, explicitly indexed to the relevant interface.

- Move the Merkle-path illustration and low-level circuit schemas out of the
  main flow unless they are needed to understand P1--P4.
- Rename the main predicate from a generic `Accept` to an Attest-specific
  predicate only after checking every theorem, algorithm, code identifier, and
  artifact reference; do not perform a blind textual replacement.

#### 6. Separate the security statements

- Detect: calibrated FPR, statistical detection behavior, measured robustness,
  and empirical attack FAR. Its SD2.1 white-box FAR 0.76 and SDXL FAR 0.11 are
  boundaries, not cryptographic guarantees.
- Attest: proof soundness, binding/commitment/image-replay conditions, and the
  partial-opening bound. The theorem title and statement must say `Attest`, not
  unqualified PP-Mark.
- Define outcomes so missing artifacts, invalid proofs, and absent signals are
  not collapsed into one negative claim.

#### 7. Reorder experiments by research question

- RQ1, Detect utility: clean calibration, TPR/FPR, quality, model scope,
  isolated and sequential transformations, and 0.72 s runtime.
- RQ2, Detect adversarial boundary: direct/transfer imprint, black-box and
  white-box optimization, regeneration, and removal. Present the 0.76/0.11
  white-box results as the measured reason an assurance escalation exists.
- RQ3, Attest security: honest-original success, authorized signed-but-
  unwatermarked controls, fabricated binding/generator fraud, cross-image
  replay, context substitution, altered-image rejection, and P1--P4 ablations.
- RQ4, Attest systems cost: N sweep, 48.6 s generation, 7.59 s verification,
  receipt size, caching, SP1/RISC0/Halo2/PLONK/Groth16 trade-offs, and the exact
  redistribution boundary.
- RQ5, two-interface workload: apply a prespecified risk policy to a mixed
  workload and report coverage, error, and cost as the Attest fraction changes.

#### 8. Add the killer two-interface evaluation

- Build the workload from existing clean originals, benign derivatives,
  external forgeries, Score-crossing white-box forgeries, and generator-
  noncompliance controls. Freeze categories and routing before measuring.
- Routing must be application-risk-based, not cherry-picked after observing
  whether an attack succeeded: routine derivative screening uses Detect;
  exact-origin disputes, audited producer samples, and other declared
  high-assurance claims require Attest.
- Compare Detect only, Bare-Sig, Context-Sig, and Detect+Attest. Report which
  claims each accepts, rejects, or leaves unverified.
- For verifier-side accounting, if fraction `p` uses Attest after Detect, the
  measured average is `0.72 + 7.59p` seconds under the current implementation:
  0.796 s at 1 percent, 1.100 s at 5 percent, and 1.479 s at 10 percent. Report
  producer proof cost separately; do not hide it in the verifier average.
- Reuse the existing Score-crossing attack set to show the concrete marginal
  value of a valid receipt. New GPU work is required only where the required
  per-instance pairings or retained witness artifacts are absent.

#### 9. Stop mixing the two interfaces in every baseline table

- Watermark benchmark tables should compare Detect with watermark detectors
  using detection/attack metrics on equal footing.
- One main security table should compare Detect, signature controls, and Attest
  on generator fraud, replay, exact modification, and third-party audit.
- Do not repeat an Attest FAR of 0.00 beside statistical watermark baselines in
  every table; it is a different predicate and repeated zeroes look like a
  by-construction result rather than an evaluated system contribution.
- Keep one interface/cost table, one Detect result table, one Attest security
  table, and one two-interface workload table in the main paper. Extended
  attack grids, proof-system sweeps, and per-transformation details belong in
  the supplement.

#### 10. Rebuild related work around the residual contribution

- Statistical/semantic watermark detection: robust presence but no proof of
  honest embedding under the public/adaptive threat model.
- Signatures and C2PA: signed provenance assertions and exact integrity, but no
  verification that the watermark embedding relation was followed.
- ZK image/provenance systems and attestation: proof-backed process claims, but
  not a transform-tolerant in-image detection channel.
- SEAL and DGS: semantic robustness and managed-decoder gradient protection,
  respectively; state why their guarantees and access models are complementary.
- Residual claim: PP-Mark shares one context-bound construction across a robust
  detection interface and an exact proof-backed embedding-attestation
  interface. Do not claim to supersede any of the above families.

#### 11. Main/supplement allocation under the same body budget

- Page 1: problem split, two-interface thesis, contribution bullets, and branch
  figure.
- Page 2: interface contract and closest-work positioning.
- Pages 3--4: common construction, Detect, and Attest definitions.
- Page 5: threat model and one core Attest theorem/proof sketch.
- Pages 6--8: RQ1--RQ5 core results, including the killer workload result.
- Final page: exact limitations, deployment boundary, and conclusion.
- Supplement: full theorem proof, circuit internals, proof-system comparisons,
  complete attacks/transforms, quality tables, model details, artifacts,
  commands, and extended case studies.

#### 12. Claims and wording firewall

- Replace global `forgery-resistant` language with mode-qualified statements.
- Replace `legal evidence` or `regulatory compliance` as guaranteed outcomes
  with `high-assurance exact-origin workflows`; legal admissibility is outside
  the evaluated claim unless independently established.
- Retain participating-generator and latent-diffusion scope. Do not revive
  `architecture-agnostic`, universal AI detection, or post-hoc proof claims.
- A transformed copy may be Detected; only an exact artifact with the required
  receipt/witness path may be Attested.

#### 13. Evidence reuse versus new work

- Reusable: all submitted calibration, attacks, transformation tests, quality
  measurements, two-model evidence, signature controls, sequential transform,
  proof-system timing/size, and replay/substitution checks.
- Required synthesis: interface-contract table, branch figure, mode-qualified
  claim ledger, and workload/cost table.
- Potentially new experiment: prespecified mixed-workload routing and any
  missing per-instance Detect/Attest pairing needed to compute joint outcomes.
- Optional scope expansion only after the core redesign: a non-latent-
  diffusion family or substantially faster proof system. Neither is required
  merely to repair the presentation and evaluation architecture.

## Cross-venue review synthesis and technical hardening audit (2026-08-01 KST)

This section combines the four ICML reviews with the three NeurIPS reviews and
then audits the current paper/code at the exact points repeatedly questioned by
reviewers. It is an internal contingency record. The live under-review
submission remains unchanged.

### 1. What the combined reviews actually establish

- The core problem is accepted as real. Reviewers across both venues recognized
  the public-detector/adaptive-forgery motivation, and several described the
  watermark-plus-cryptographic-provenance direction as interesting or novel.
- The empirical attack effort and concrete proof implementation are genuine
  strengths. The current NeurIPS reviewer `iryC` moved from 3 to 5 after the
  response, showing that the added comparisons, quality metrics, semantic-
  watermark positioning, and deployment clarification materially worked.
- The repeated negative signal is not simply that reviewers failed to read the
  two modes. It is that the practical mode appeared to retain only statistical
  evidence, while the expensive exact mode appeared insufficiently distinct
  from a signature and insufficiently connected to ordinary distribution.
- Therefore a future revision needs both better interface architecture and a
  tighter technical contract. Renaming `Score / Accept` to `Detect / Attest`
  fixes the presentation hierarchy, but cannot by itself close the formal and
  implementation issues below.

### 2. Cross-review concern matrix

| Concern | ICML evidence | NeurIPS evidence | Post-rebuttal status | Revision priority |
|---|---|---|---|---|
| Why not a simple signature? | GiP6, T6eo | P58K; also AC comparison concern | New Bare-Sig/Context-Sig data helps, but the distinction is still not the paper's organizing security property | Mandatory centralization |
| 48.6 s / 40.7 MB practicality | yoou, Y5S2, T6eo, GiP6 | iryC, Ef3K, AC | Dual-mode explanation and proof-system trade-offs help; workload value and final-predicate timing remain open | Mandatory systems evidence |
| Robust watermark and exact proof do not operate on the same transformed artifact | Y5S2, T6eo, GiP6 | iryC, Ef3K, P58K | Sequential Detect evidence is stronger, but the cryptographic/derivative bridge remains a workflow boundary | Mandatory framing; optional new primitive |
| Exact scope of the ZK guarantee | yoou; implicit in GiP6/T6eo | Ef3K's generator-fraud objection | Still high risk; see implementation/theorem audit below | Mandatory formal repair |
| Model/generalization scope | yoou, GiP6 | P58K | SDXL helps, but both models share a latent-diffusion/inversion interface; architecture-agnostic wording was withdrawn | Scope honestly; expansion optional |
| SOTA, quality, and realistic transforms | GiP6 | iryC, P58K, AC | Largely closed by matched baselines, BRISQUE/KID/CLIPScore, and 80/100 sequential transforms | Integrate into main paper |
| Metadata/artifact discoverability | GiP6, Y5S2 | Ef3K | Sidecar/resolver explanation is plausible but not implemented as an evaluated end-to-end path | High-impact operational experiment |
| Notation and first-pass clarity | Some ICML confusion | iryC, AC | Planned notation table and branch-first Figure 1 are appropriate | Mandatory presentation repair |
| Positive provenance, not universal AI detection | GiP6 | Ef3K | Scope correction is sound and should appear on page 1 | Mandatory claim boundary |
| Privacy/linkability | Y5S2 | Ef3K | Response substantially closes it, subject to explicit provider/key-epoch metadata | Main-text boundary, not a new research axis |

### 3. Newly identified mandatory technical blockers

These were exposed by following the reviewers' guarantee and signature questions
through the submitted paper, SP1 guest, host verifier, and retained experiment
provenance. They are more important than adding another baseline.

#### A. Paper P4 and the implemented SP1 predicate do not currently match

- The paper states that P4 checks all three relations on opened positions:
  `bit_i = RS(b)[idx(i)]`, `g_i = F^{-1}[H(b || i+1)]`, and
  `z_i = g_i + alpha * sign(bit_i)`.
- The current SP1 guest's `check_embedding_relation` checks only the third
  arithmetic equality. The witness `codeword` is not used by the guest, and no
  inverse-CDF LUT hash or `b`-derived Gaussian check appears in the guest.
- The copy in `anonymous_supplementary_code/sp1/guest/src/lib.rs` is byte-for-
  byte identical to the working-tree guest (SHA-256
  `6A2C208881B05C186E2898266DB286ECC811A9C29E9B1272C755F0B7C1291C9F`),
  so this is not merely an obsolete private copy.
- Required resolution: either implement the full stated P4 and rerun all
  proof-system evidence, or narrow the paper/theorem to the predicate actually
  implemented. The first option is preferable if Attest remains the claimed
  security contribution.

#### B. Receipt verification does not bind the receipt's public values to the
expected image/metadata statement

- The guest commits `challenge_seed`, `opening_digest`, `alpha`,
  `sample_merkle_root`, and `binding` as public outputs.
- The host `verify` path currently calls `client.verify(&proof, &vk)` and returns
  success without extracting and comparing those committed values against an
  externally expected statement.
- The Python wrapper invokes the host as `verify --receipt <path>`; it does not
  pass the expected public-input file. It separately recomputes image/metadata
  checks, but never establishes that those values are the ones inside the
  verified receipt.
- The guest also receives `challenge_seed` as an input rather than recomputing
  it from `h_img`, and `h_img` is absent from the guest's public-input struct.
  Consequently, the paper's end-to-end `VerifyReceipt(pi; x)` contract is not
  implemented by the current verification path.
- Required resolution: construct a canonical expected statement from the image
  and declared context, verify the receipt, decode its committed public values,
  and require byte-exact equality. Add negative tests for swapped receipt,
  binding, context, root, opening digest, image, and verification key.

#### C. The producer identity / authorized-key trust anchor is underspecified

- P1 currently proves knowledge of *some* `k` satisfying
  `b = H(domain || h_ctx || k)`. The public statement contains no registered
  provider key commitment or certified producer public key that distinguishes
  the legitimate producer's `k` from an adversary-chosen `k*`.
- The theorem's second-preimage argument works only when the target `b` and
  producer identity are fixed in advance. It does not by itself prevent a
  prover from choosing a fresh `k*`, fresh `b*`, and fresh statement.
- Required resolution: add a provider/epoch trust anchor, for example a
  certified public key or public commitment `K_epoch = H(domain || k)`, include
  it in the statement, and enforce the corresponding relation in-circuit.
  State separately how the provider identity is authenticated. A signature or
  certificate is therefore a complementary outer identity layer, not something
  PP-Mark should claim to replace.

#### D. The actor and key-access model currently mixes incompatible adversaries

- The mode table says Detect/Score trusts honest embedding, while the verification
  ablation says Score resists forged binding because `b` is fixed by `k`. Without
  proof or an authenticated registry, a public verifier cannot know that a
  supplied `b` was derived from the authorized `k`.
- The paper describes Attest/Accept as useful when the generator is untrusted,
  but Theorem 1 assumes the adversary does not possess `k`. An actually
  untrusted generator/operator may control `k` unless the key and prover are
  isolated in a separate trusted service or HSM.
- The current `compliance bypass` therefore models a party controlling the
  metadata-signing key but not PP-Mark's binding/proving key. It is not yet a
  general malicious-generator result.
- Required resolution: define an actor/key matrix for external forger,
  publisher/signing authority, embedding service, proof service/HSM, platform,
  and public verifier. Rename the attack to match the modeled credential
  compromise, or strengthen the construction to cover the broader actor.

#### E. The theorem/game is broader than the reduction currently supports

- The game gives the adversary one honest public statement but lets it output
  `x*`; the proof cases analyze reuse of the honest `b` or Merkle root and do not
  exhaust the fresh-statement case.
- The partial-opening theorem concerns consistency of a committed sampled trace.
  It does not, by itself, prove that the full diffusion execution produced the
  pixel image. Image hashing changes the opening challenge and blocks direct
  receipt replay, but it is not a proof of the diffusion trace-to-image
  computation.
- The proof sketch assumes the trace root is fixed before the image hash/challenge
  is known. The current non-interactive artifact format does not itself enforce
  that temporal ordering; the precise Fiat-Shamir/commitment game and any
  grinding allowance must be stated and proved.
- Required resolution: formulate a target-identity, target-statement security
  game with exhaustive cases. Use the narrow claim `registered-key sampled-
  embedding attestation bound to an exact artifact` unless an actual
  trace-to-image or trusted-execution link is added. Do not claim that the full
  generator execution is proved.

#### F. Final-predicate runtime and end-to-end evidence must be rerun

- The retained audit states that the submitted 48.572 s / 7.589 s N=1000 SP1
  numbers and compact PLONK/Groth16 artifacts predate the April P1--P4
  strengthening.
- The post-change guest was tested only in a small roughly N=205 smoke run. Its
  realistic pixel-image DDIM round-trip ended in
  `opening_combined_mismatch`; a final-predicate, realistic-configuration
  end-to-end run was not completed.
- Therefore the future paper must not combine old timing/size numbers with the
  strengthened predicate as though they were one measured configuration.
- Required resolution: after the predicate and verifier are frozen, rerun honest
  image-to-receipt-to-public-verification at N=1000, all negative security tests,
  the N sweep, and every proof-system comparison under the same predicate. Only
  then quote final proving, verification, and receipt-size numbers.

#### G. Concrete manuscript consistency corrections

- The small-q appendix says gradient-forgery FAR remains `0--0.01` while citing
  the white-box table, but the submitted Score FAR is 0.76 on SD2.1 and 0.11 on
  SDXL. The text must distinguish trace-position cheating from image-space
  white-box optimization and cite the correct experiment.
- `in milliseconds` should be `sub-second` for the measured 0.72 s verifier.
- Remove the remaining `architecture-agnostic` statement and use `two latent-
  diffusion models` consistently.
- The transform table caption says every cell uses n=100, while the accompanying
  SDXL crop text records n=50. Resolve from the authoritative artifact and state
  per-cell sample sizes accurately.
- Replace claims of legal evidence or regulatory compliance with evaluated
  high-assurance workflow language unless the legal claim is independently
  established.
- Treat the claimed causal explanation that larger latent dimensionality causes
  the lower SDXL white-box FAR as a hypothesis unless supported by a controlled
  dimensionality ablation.

### 4. Strengthening order for a revised paper

The correct order is specification and implementation first, evidence second,
and framing third.

1. Freeze a one-page security contract: actors, credentials, public registry,
   artifacts, typed outcomes, target identity, and exact claims/non-claims.
2. Add the registered producer/epoch key anchor and canonical context check.
3. Implement the complete P4 or reduce the claim; make the receipt verifier
   compare all receipt public values to a canonical expected statement.
4. Decide the exact Attest claim. The recommended bounded version proves the
   registered-key sampled embedding relation and exact-artifact binding; proving
   full diffusion execution is a separate, much larger paper-sized extension.
5. Rewrite the security game and theorem around that claim, including fresh-
   statement, replay, substitution, key compromise, and challenge-grinding
   cases.
6. Rerun the frozen final implementation end to end at realistic scale before
   reusing any runtime, size, or alternative-proof-system result.
7. Recast signatures as the identity/assertion layer and Attest as relation-
   compliance evidence. Put Bare-Sig and Context-Sig in the main security table.
8. Then apply the Detect/Attest first-page redesign and the mixed-workload
   escalation evaluation already specified above.
9. Implement and measure one authenticated resolver/registry workflow: exact
   origin Attests, transformed derivatives may Detect and resolve to the origin,
   and missing artifacts return `UNVERIFIED`. Do not call the derivative itself
   cryptographically Attested.
10. Integrate the rebuttal-only evidence into the main paper: sequential
    transforms, matched quality, signature controls, SEAL/DGS positioning,
    proof-system caveats, and scoped model claim.
11. Only after all mandatory items are closed should a genuinely different
    generative family or a new robust-proof primitive be considered.

### 5. Priority classification

- **P0, mandatory correctness blockers:** full P4/code alignment; receipt public-
  statement comparison; registered producer key anchor; coherent actor/key
  model; corrected theorem/game; final-predicate end-to-end rerun and evidence
  provenance.
- **P1, acceptance-signal upgrades:** Detect/Attest information architecture;
  signature-as-complement positioning; mixed-workload escalation result;
  implemented artifact resolver; rebuttal evidence moved into the main paper.
- **P2, optional scope expansion:** a genuinely different model family, a new
  transform-robust cryptographic binding primitive, user study, or deeper proof-
  system optimization.

Bottom line: the combined reviews reveal more than a naming problem. The paper
has a defensible research core, and the rebuttal data materially improved it,
but a future submission must first make the theorem, SP1 predicate, public
verification path, key trust anchor, and measured final implementation exactly
the same object. Once that P0 closure is real, the Detect/Attest framing can turn
the existing two branches into a substantially clearer and stronger paper.

## Scope clarification for the next audit pass (2026-08-01 KST)

- Keep implementation/code discrepancies as a separate P0 backlog. The current
  paper-logic review asks whether the manuscript is coherent assuming the
  described method and reported results are implemented as written. Do not mix
  code-level findings into every paper-framing judgment.
- Under that paper-only assumption, the four remaining logical repairs are:
  (i) make the actor/key threat model consistent, especially `untrusted
  generator` versus the theorem's unknown-`k` adversary; (ii) position
  signatures as authenticating an identity/assertion and Attest as proving the
  declared sampled embedding relation; (iii) restrict the guarantee to the
  registered-key/context/committed-trace/sampled-embedding/exact-artifact
  relation rather than full diffusion execution or real-world provenance; and
  (iv) present transformed derivatives as Detectable objects that may resolve
  to an Attested exact origin, not as cryptographically Attested derivatives.
- Together with the rebuttal evidence already obtained and integrated into a
  revised main paper, these repairs substantially close the recurring logical
  objections across the ICML and NeurIPS reviews: simple-signature sufficiency,
  mode confusion, exact-hash fragility, guarantee ambiguity, transformed-image
  workflow, missing SOTA/quality/sequential-transform evidence, artifact
  availability semantics, and positive-provenance scope.
- The residual risks then become ordinary research-value judgments rather than
  obvious logical gaps: whether selective high-assurance Attest justifies its
  cost, whether the two-interface system is significant despite not providing
  transform-robust cryptographic acceptance in one predicate, whether evidence
  from two related latent-diffusion models is broad enough, and whether the
  integration has sufficient novelty for the venue.
- Final paper-only verdict: the research direction and independent contribution
  remain defensible. The Detect/Attest redesign plus exact threat/guarantee
  boundaries can remove most easy reviewer rejection arguments, although it
  cannot eliminate subjective rejection based on cost, scope, or perceived
  significance.

## P0 implementation-alignment pass (2026-08-01 KST)

This pass acted on the code backlog identified above. It did not edit the
submitted manuscript or reinterpret old measurements. The source of truth is
`PP-Mark/`; the already existing `anonymous_supplementary_code/` directory was
updated mechanically after verification. No new version directory was made,
and the existing supplementary ZIP was intentionally not overwritten.

### Closed implementation mismatches

1. Added one canonical `AttestationStatement` shared at the byte level by
   Python, the SP1 host, and the SP1 guest. It includes the context/binding,
   producer-key commitment, exact image hash, ordered sample root/index hash,
   image-derived challenge/opening digest, LUT hash, Q18/Q30 normalization
   values, sample/grid dimensions, and RS parameters. Python and Rust reject
   missing or unknown fields.
2. Replaced the prior partial guest with explicit P1--P4 enforcement:
   - P1 recomputes the context/key binding and producer-key commitment.
   - P2 recomputes the ordered SHA-256 Merkle root and sampled-index-set hash.
   - P3 recomputes the image/context/root challenge and partial-opening digest.
   - P4 re-encodes RS(64,32), maps MSB-first codeword bits, reproduces the
     pinned inverse-CDF Gaussian, and enforces the normalized fixed-point
     embedding relation at every committed sample.
3. The SP1 host now cryptographically verifies the receipt and byte-compares
   its committed public values with the verifier-supplied canonical statement.
   A receipt cannot be accepted under altered public JSON.
4. Attest now requires an external trusted producer-key commitment by default.
   A self-asserted artifact key is allowed only behind an explicit test-only
   flag and is labeled unauthenticated.
5. Public artifacts and the private prover witness are separated. The producer
   secret, full trace, and latent diagnostics are not verifier inputs. The
   serialized witness is deleted after successful proving unless explicit
   private-debug retention is requested, and secret export is refused inside
   the public artifact directory.
6. The implementation now exposes the paper-level distinction directly:
   `Detect` returns the statistical signal and does not claim generator
   compliance; `Attest` requires trusted key + exact artifact + P1--P4 receipt
   + Detect. Legacy SP1 metadata is refused instead of falling through to the
   old verifier, and the dead legacy SP1 prover/verifier branches were removed.
7. CPU uniform-to-LUT indexing and Q18/Q30 normalization were aligned with the
   guest's integer implementation. The old raw `g + alpha*s` versus normalized
   `(g + alpha*s)/sqrt(1+alpha^2)` mismatch is no longer present in Attest.

Primary implementation files: `src/ppmark_v03/attestation.py`,
`src/ppmark_v03/sp1_runner.py`, `src/ppmark_v03/cli.py`,
`src/ppmark_v03/noise.py`, `src/ppmark_v03/tables.py`,
`sp1/guest/src/lib.rs`, and `sp1/host/src/main.rs`. Model-free regression and
tamper fixtures were added under `scripts/` and `tests/`.

### Verification evidence from this pass

- Python source tree: **14 passed, 5 skipped**.
- Rust guest: **3 passed** (reedsolo-compatible RS vector, deterministic
  openings, Python/Rust LUT and fixed-point vector).
- Rust host: **2 passed** (strict hex parsing and all-statement-field public-
  value sensitivity).
- SP1 v5.2.3 release host rebuilt successfully; Rust formatting, Python compile,
  shell syntax, and scoped whitespace checks passed.
- Honest model-free guest execution passed.
- Ten coherent tamper cases were rejected at their intended predicates:
  P1 binding/key; P2 root/index; P3 image/opening; P4 codeword/bit/Gaussian/LUT.
- A real SP1 Core receipt was generated and verified. Reusing it with a changed
  public image-hash statement was rejected by the host's public-value match.
- Full CLI Attest passed only with the exact fixture image, trusted producer
  commitment, valid receipt, and Detect score. A different image failed at P3;
  a wrong trusted commitment failed before receipt acceptance. Detect-only
  passed the supplied diagnostic latent while explicitly reporting that
  generator compliance was not attested.
- The 22 selected source/document/test files copied into the existing anonymous
  supplementary directory match the source tree by SHA-256. Its independent
  test subset reports **7 passed**.

The two measured CPU Core fixture proofs in this pass took approximately 44.8 s
and 41.9 s, with receipt verification around 5 s. These are model-free, small
fixture engineering checks and are **not** paper performance measurements.

### Evidence that remains mandatory before a revised submission

The final predicate has not yet been rerun at the paper's full N=1000 setting
on H100 with the real SD2.1/SDXL image-generation and inversion workflow.
Therefore the submitted 48.572 s proving, 7.589 s verification, 40.68 MB SP1
artifact, and PLONK/Groth16 comparisons must not be relabeled as measurements of
this strengthened implementation. Rerun honest end-to-end generation,
N=1000/H100 timing and size, the N sweep, all negative tests, and alternative-
proof-system comparisons after the final circuit is frozen.

## Pre/post-alignment SP1 execution-cost audit (2026-08-01 KST)

This audit isolates the deterministic SP1 guest execution cost before asking
an H100 prover for wall-clock timing. The legacy guest was rebuilt from Git
commit `b5572d5c250542d8938456e2ec73d3f1f8e438a9`; the current guest is the
P1--P4-aligned working-tree implementation. Both were compiled with SP1 v5.2.3
and the Succinct Rust 1.91.1 toolchain. The same honest canonical statement and
witness values were used at each N. A non-production diagnostic serialized the
same samples into the legacy bincode shape and derived the legacy per-sample
Merkle paths; production verification continues to reject legacy statements.

| N | guest | instructions | syscalls | SP1 weighted gas |
|---:|:------|-------------:|---------:|-----------------:|
| 32 | legacy | 438,501 | 795 | 1,909,726 |
| 32 | current P1--P4 | 239,685 | 636 | 1,643,846 |
| 200 | legacy | 4,048,825 | 6,891 | 8,584,353 |
| 200 | current P1--P4 | 890,074 | 2,204 | 3,328,611 |
| 1,000 | legacy | 24,873,425 | 42,091 | 47,550,322 |
| 1,000 | current P1--P4 | 2,774,702 | 6,970 | 9,034,677 |

- At N=32, the current guest reduces instructions by **45.34%**, syscalls by
  **20.00%**, and weighted gas by **13.92%**.
- At N=1,000, it reduces instructions by **88.84%**, syscalls by **83.44%**,
  and weighted gas by **81.00%**. Equivalently, the legacy guest uses 8.964x
  the instructions and 5.263x the weighted gas of the current guest.
- The reason is structural: the legacy guest verifies a full Merkle path for
  every sample, giving O(N log N) repeated hashing. The current guest builds
  the ordered sample tree once in O(N), which more than offsets the new P1--P4
  binding, challenge/opening, RS, LUT, and embedding-relation checks.

ELF SHA-256 identities are
`76a481aadb37b02152893e5e963789dc3896f1d3618e011e4e4552a694d15316`
(legacy) and
`88a5f19ef24636fb23d7fb57188c99c5e2aa4dd049bf8b4076e62dd94f6fa47a`
(current). The diagnostic is `sp1/host/examples/cycle_compare.rs`; the fixture
builder now accepts explicit grid, N, and opening-K parameters while retaining
its old N=32 defaults.

These figures establish the direction and magnitude of circuit execution-cost
change, not H100 wall-clock proving time. GPU proving/verification time, shard
behavior, and receipt size still require the frozen current guest to be proved
on the target H100 environment. The first supplied Elice SSH endpoint accepted
TCP connections but closed them before SSH key exchange, so no H100 measurement
was attributed to this audit.

### H100 time inference while the replacement instance is unavailable

The submitted/rebuttal legacy H100 measurements provide two calibration
points under the old guest: N=200 took 32.309 s to prove and 4.468 s to verify;
N=1,000 took 48.572 s to prove and 7.589 s to verify. Current N=1,000 has
9.035M weighted gas and 6,970 syscalls, only 5.25% and 1.15% above legacy
N=200's 8.584M gas and 6,891 syscalls. It also has 31.47% fewer instructions.
Thus legacy N=200 is the closest directly measured H100 workload analogue.

Linear interpolation/extrapolation of the two legacy H100 points gives the
following current-N=1,000 predictions:

| Cost predictor | Prove prediction | Verify prediction |
|:---------------|-----------------:|------------------:|
| instructions | 31.314 s | 4.277 s |
| syscall count | 32.345 s | 4.475 s |
| SP1 weighted gas | 32.497 s | 4.504 s |

The working estimate is therefore **about 32--33 s proving** and **about 4.5 s
SP1 verification**, with a conservative same-stack planning interval of
**30--36 s** and **4.2--5.2 s**, respectively. This predicts roughly a 33%
wall-clock proving reduction from 48.572 s, not the full 81% gas reduction,
because the two old H100 observations imply about 29 s of fixed/startup/shard
overhead. If the 0.72 s Detect path is sequentially included in full Attest
verification, the corresponding central estimate is approximately 5.2 s.
These remain calibrated estimates, not measurements of the strengthened guest.

## Further Attest proving-time optimization audit (2026-08-01 KST)

This continuation asked whether the strengthened Attest relation can be made
materially faster through program structure, newer proof systems, proof
compression, or deployment changes. The paper was not edited. All temporary
cost-isolation guest changes were reverted after their ELF files were built;
the production guest remains the P1--P4 Merkle implementation.

### Measured bottleneck

At N=1,000, the current guest executes 2,774,702 instructions, 6,970 syscalls,
and 9,034,677 weighted gas. The detailed report contains 3,327 SHA-256
compression/extension pairs. Removing only P2 while retaining input decoding,
P1, P3, P4, and public-value commitment yields 479,415 instructions, 750
syscalls, and 1,910,970 gas. Consequently, P2 accounts for 2,295,287
instructions (82.72%) and 7,123,707 gas (78.85%). It invokes 3,110 SHA blocks:
1,000 leaf hashes, 2,046 blocks for 1,023 two-block internal-node hashes, and
64 blocks for the sample-index transcript. The cost is therefore not the LUT,
Reed--Solomon encoding, bincode alone, or the k=32 embedding checks; it is the
complete binary Merkle reconstruction inside every receipt.

### Safe compiler-only result

Building the unchanged relation with release `lto = "fat"` and
`codegen-units = 1` produced the following N=1,000 execution report:

| build | instructions | syscalls | weighted gas |
|:------|-------------:|---------:|-------------:|
| current default release | 2,774,702 | 6,970 | 9,034,677 |
| current fat-LTO / one codegen unit | 2,245,151 | 6,970 | 8,121,446 |

This is a semantics-preserving 19.08% instruction and 10.11% gas reduction.
It is the lowest-risk code optimization and should be retained in the next
frozen implementation after reproducible-build and tamper-test confirmation.

### Full-trace streaming-commitment diagnostic

A temporary guest replaced only the binary Merkle reconstruction with a
domain-separated, length-prefixed SHA-256 commitment over the same complete,
ordered canonical sample vector. It still decoded all 1,000 private entries,
checked every entry's shape, retained the independent sample-index hash, and
ran unchanged P1, P3, and P4. For cost isolation, its newly computed digest was
consumed but was not substituted into the fixture's public statement; this ELF
is not a valid production protocol implementation.

| N=1,000 diagnostic | instructions | syscalls | weighted gas | SHA pairs |
|:-------------------|-------------:|---------:|-------------:|----------:|
| current binary Merkle | 2,774,702 | 6,970 | 9,034,677 | 3,327 |
| streaming trace commitment | 721,533 | 1,536 | 2,846,228 | 610 |
| streaming + fat LTO | 631,504 | 1,536 | 2,807,884 | 610 |

The combined diagnostic reduces instructions by **77.24%**, SHA pairs by
**81.67%**, and weighted gas by **68.92%** relative to the current guest. Fat
LTO adds a further 12.48% instruction reduction after streaming, but only
1.35% gas because accelerated SHA syscalls then dominate.

This is the strongest conservative structural candidate. A collision-resistant
canonical vector digest is still a binding commitment to the entire ordered
trace and still proves knowledge of that trace inside the same receipt. It
does not provide independently verifiable selective Merkle paths. Because the
current receipt does not expose those paths to a separate verifier, the change
may preserve the operational security relation more cleanly than supplying
only opened leaves and a multiproof. It nevertheless changes the public
statement, challenge input, terminology, Figure 1, proof description, and
security theorem, so it requires a protocol-version bump, canonical-encoding
specification, fresh negative tests, and a paper revision. It must never be
silently swapped into the submitted semantics.

### Multiproof comparison

For a 1,024-leaf tree and k=32 uniformly selected positions, 100,000 simulated
position sets required a mean 166.08 internal hashes and 135.08 sibling nodes
(about 4.3 KiB), with p5/median/p95 internal-hash counts 156/166/175. Including
leaf hashing gives approximately 364 P2 SHA blocks, versus 3,110 today. The
resulting complete guest was estimated at roughly 2.5--2.9M gas, close to the
measured 2.808M streaming result.

The multiproof is therefore not the first structural choice: it adds path
parsing, position/direction constraints, and a precommitment requirement while
offering little speed beyond the simpler full-vector digest. Use it only if
small private input or externally reusable selective openings are a genuine
deployment requirement. If used, the Merkle root must be fixed before the
image-bound challenge (for example by an authenticated registration or fresh
verifier randomness), and the theorem must account for Fiat--Shamir grinding.
The same grinding/query-budget issue should be quantified for the current
commit-then-challenge construction; it is not introduced by streaming.

### What execution-cost reduction means for H100 wall time

The two legacy H100 gas/time points imply a cold one-shot fit of approximately
`28.726 s + 0.417 s * gas_in_millions`. Under that deliberately conservative
same-stack fit, current N=1,000 predicts 32.497 s and the 2.808M streaming/LTO
guest predicts 29.898 s. Thus a 69% guest-gas reduction alone does not justify
claiming a 69% wall-clock reduction: the old workflow has a roughly 29-second
fixed/startup/minimum-shape component.

The current v5 host explains an important part of that floor. For each one-shot
CLI invocation it constructs `ProverClient::builder().cuda().local()`, which
starts a managed Docker Moongate server; dropping the client stops that server.
The host already caches PK/VK by ELF digest, but not the CUDA service/client
across processes. The SDK also supports an external `.cuda().server(endpoint)`.
The first no-security-change deployment experiment must therefore compare:

1. cold one-shot local CUDA;
2. one long-lived CUDA server/client with cached PK/VK and one warm-up proof;
3. producer latency with redundant dry execution and local self-verification
   timed separately; and
4. recipient verification through a CPU/light verifier, not by initializing a
   CUDA prover merely to verify a receipt.

`SP1_SKIP_EXECUTE=true` can remove the host's redundant pre-proof execution
after the frozen guest has been exhaustively tested. `SP1_SKIP_VERIFY=true`
can remove producer-side self-verification from the critical path only if the
proof is still verified before publication or by the recipient; it is an
operational latency choice, not a cryptographic speedup. Exact warm H100 times
remain unmeasured.

### Current external-system audit

- **SP1 v6.3.1 is the first system pilot.** It is the latest release as of this
  audit. Hypercube reports up to 5x for compute-heavy programs and up to 2x for
  precompile-heavy programs; PP-Mark is SHA-precompile-heavy, so 2x is an
  optimistic vendor upper bound, not a prediction. V6 uses an incompatible
  64-bit toolchain, async APIs, a native `sp1-gpu-server` instead of Docker,
  and explicitly recommends retaining a `ProverClient` in an `Arc` because
  initialization is expensive. Migrate in an isolated test branch and rerun
  statement-serialization, honest-proof, and every tamper test.
- **OpenVM 2.0 is the strongest independent equal-workload pilot.** Its 10 July
  2026 production release is externally audited, reports 100-bit security and
  proofs below 300 KiB, and has formally verified SHA-2/Keccak extensions plus
  GPU proving. Its published Ethereum cluster numbers on 8/16 RTX 5090 GPUs
  cannot be transferred to this single-H100 SHA workload. Port the exact
  canonical PP-Mark relation before comparing.
- **Pico is a third pilot, not an inferred winner.** Pico reports a Sherlock
  review and production recommendation and supports SHA-2 coprocessors. A 2025
  vendor benchmark reported about 25% average advantage over then-current SP1
  on one RTX 4090, but it predates SP1 v6 and is not this workload. Pico's
  `--fast`/`prove_fast` mode uses one FRI query and its own documentation says
  never to use it in production; only the full-security path is admissible.
- **RISC Zero v3.0.6 is lower priority.** It is newer than the retained PP-Mark
  pilot, but there is no primary equal-workload evidence overturning the much
  slower retained result. Retest only after SP1 v6/OpenVM/Pico if resources
  remain.
- **Jolt is excluded from a high-assurance receipt.** Its official repository
  labels it alpha, unaudited, and unsuitable for production.
- **Flock is a valuable future SHA coprocessor, not a drop-in replacement.** It
  reports 42,100 SHA-256 compressions/s on one M4 Max core and explicitly
  targets hash batches, but the authors call it a research prototype and do
  not recommend production use. It also gives up generality; P1, RS, P4,
  public serialization, composition, and zero-knowledge integration remain.
  After streaming leaves only 610 SHA blocks, composition overhead may erase
  much of its benefit. Revisit if its ideas enter audited SP1/OpenVM support.

Primary sources consulted:

- SP1 v6.3.1 release: https://github.com/succinctlabs/sp1/releases/tag/v6.3.1
- SP1 v6 migration: https://docs.succinct.xyz/docs/sp1/getting-started/migration
- SP1 GPU workflow: https://docs.succinct.xyz/docs/sp1/generating-proofs/hardware-acceleration
- SP1 optimization/I/O: https://docs.succinct.xyz/docs/sp1/optimizing-programs/basics
- SP1 proof types: https://docs.succinct.xyz/docs/sp1/generating-proofs/proof-types
- SP1 aggregation warning: https://docs.succinct.xyz/docs/sp1/writing-programs/proof-aggregation
- Hypercube performance scope: https://blog.succinct.xyz/sp1-hypercube/
- OpenVM 2.0 release: https://blog.openvm.dev/2.0-production
- Pico production/security status: https://github.com/brevis-network/pico
- Pico full-security warning: https://pico-docs.brevis.network/writing-apps/proving.html
- Flock status/benchmarks: https://blog.succinct.xyz/introducing-flock/
- Jolt status: https://github.com/a16z/jolt

### Ranked implementation/measurement order

1. Freeze and benchmark the current default and current fat-LTO guests on one
   persistent H100 service; report cold first, warm median/p95, and component
   times separately.
2. Implement a protocol-versioned streaming trace commitment in an isolated
   branch, preserve full-vector knowledge and k-opening/P4 checks, update the
   challenge to bind the new commitment, and rerun all positive/negative tests.
3. Benchmark that exact guest on SP1 v5 warm and SP1 v6.3.1 warm. The key gate
   is measured end-to-end Attest latency, not guest gas alone.
4. Keep Core as the latency path. If small artifacts are required, issue/cache
   Core first and asynchronously produce a compressed/Groth16 wrapper or a
   batch wrapper; do not advertise wrapping as a single-image latency win.
5. Port the frozen equal-workload relation to OpenVM 2.0, then Pico full
   security. Promote an alternative only if it passes the same tamper suite and
   beats warm SP1 on the same H100.
6. Use multi-image batching only for throughput/amortization. Multi-GPU,
   recursive splitting, and proof aggregation are unlikely to reduce latency
   for a 2.8--9.0M-gas single artifact and add orchestration overhead.

The defensible conclusion is that substantial additional reduction is
available, but not from proof compression. The measured structural ceiling is
about 2.808M gas while retaining the full private trace; the largest plausible
wall-clock gain requires combining that change with persistent prover state
and SP1 v6. No single-digit H100 latency should be claimed until that exact
combination is measured.

## Five-condition H100 timing campaign directive (2026-08-02 KST)

The author requested direct evaluation of every condition listed in the
realistic-time outlook, rather than leaving the last conditions as estimates.
The campaign is frozen as follows:

| ID | guest/relation | compiler/runtime | required report |
|:--:|:---------------|:-----------------|:----------------|
| T1 | submitted legacy relation | submitted SP1 v5 lineage, cold one-shot | reproduce or explicitly retain the historical 48.572/7.589 s with provenance |
| T2 | current strengthened P1--P4 Merkle relation | SP1 v5 default and fat-LTO, cold one-shot | init/setup/execute/prove/self-verify/size plus total |
| T3 | full-trace streaming commitment | fat-LTO, SP1 v5 cold one-shot | same components and exact public-relation version |
| T4 | identical T3 relation | SP1 v5 persistent CUDA server/client | first warm-up excluded; at least five measured warm runs, median and p95 |
| T5 | identical T3 relation | SP1 v6.3.1 persistent native GPU server/client | same warm protocol and full tamper suite |

All conditions must use the same honest N=1,000 statement/witness semantics,
opening k=32, fixed LUT, RS(64,32), and H100 instance. Each result must record
the guest ELF SHA-256, SP1 version/toolchain, proof mode, CUDA/driver/GPU,
whether PK/VK were cached, whether dry execution/self-verification were on the
critical path, proof byte size, and raw per-run timings. Cold and warm numbers
must never be pooled. T3--T5 require a protocol-versioned streaming commitment;
an earlier cost-isolation ELF that consumed but did not publish its digest is
not an admissible proof implementation.

At 2026-08-02 KST the supplied Elice endpoint
`central-01.tcp.tunnel.elice.io:29099` still accepted the TCP connection but
closed it before SSH key exchange (`kex_exchange_identification: Connection
closed by remote host`). No H100 timing was attributed to that attempt. Work
continues on reproducible ELFs, fixtures, cold/warm harnesses, and the v6 port
so that only remote measurement remains when a live endpoint is available.

## Five-condition campaign results and calibrated outlook (2026-08-02 KST)

### Measurement boundary

The five requested conditions were implemented and evaluated as far as the
available hardware permits. The paper source was not edited. T1 is a retained
historical H100 measurement. T2--T5 have exact relation costs and local CPU
measurements; their H100 wall-clock entries below are explicitly marked as
two-point interpolation or sensitivity analysis, never as measurements. The
Elice endpoint was retried after all local work and again closed before SSH key
exchange, so inventing an H100 result would be scientifically invalid.

All new runs use one honest N=1,000 fixture, opening k=32, RS(64,32), the fixed
inverse-CDF LUT, and Core proofs. V5 and v6 use different zkVM architectures,
so v6 `normalized_gas` is not numerically interchangeable with v5 gas.

### Frozen guest artifacts

| condition/artifact | bytes | SHA-256 |
|:-------------------|------:|:--------|
| T1 submitted legacy v5 guest | 150,428 | `76a481aadb37b02152893e5e963789dc3896f1d3618e011e4e4552a694d15316` |
| T2 strengthened v5 default | 437,228 | `88a5f19ef24636fb23d7fb57188c99c5e2aa4dd049bf8b4076e62dd94f6fa47a` |
| T2 strengthened v5 fat-LTO | 394,720 | `d4af244fc4c3a2ffed0550c4b6294c23623310d1a8352cece4b3097bf12cd902` |
| T3/T4 streaming-v2 v5 fat-LTO | 391,840 | `b11c91f936be3cf534a5f05fcf51f29460df62a6ff0b06fd9156a8d1a7277ca0` |
| T5 streaming-v2 SP1 v6.3.1 | 381,384 | `1a9eb1a3b100ac7de792cb44865946d44301330574922a187c7ce1b237bb1459` |

The v1 production guest was restored after the campaign to the SP1 v5.2.3
toolchain and a default non-LTO build. Its post-format build hash is
`6b417190f6bdd0456a2919bb4909a49855fdd39a2a19e73fdb0fca3ee4f4052f`;
it reproduces the same 9,034,677 gas as the frozen T2 default artifact. The v6
toolchain and guest remain isolated under cache paths and do not replace the
repository default.

### Exact relation cost

| relation | instructions | syscalls | gas/cost |
|:---------|-------------:|---------:|---------:|
| legacy N=1,000 | 24,873,425 | 42,091 | 47,550,322 v5 gas |
| legacy N=200 calibration point | 4,048,825 | 6,891 | 8,584,353 v5 gas |
| T2 strengthened default | 2,774,702 | 6,970 | 9,034,677 v5 gas |
| T2 strengthened fat-LTO | 2,245,151 | 6,970 | 8,121,446 v5 gas |
| T3/T4 streaming-v2 | 644,072 | 1,534 | 2,807,884 v5 gas |
| T5 streaming-v2 | 663,355 | 1,218 | 966,008 v6 normalized gas |

Streaming-v2 reduces v5 gas by 94.095% and instructions by 97.411% relative
to the submitted legacy N=1,000 guest. Relative to the already strengthened
fat-LTO relation, it still reduces gas by 65.426% and instructions by 71.313%.
Fat-LTO alone reduces the strengthened relation's gas by 10.108% and
instructions by 19.085%.

This reduction does not delete the security relation. Protocol v2 commits to
the complete ordered private trace with a domain-separated, length-bound
streaming SHA-256 commitment and binds the image/context challenge to that
commitment. Both the v5 and v6 guests passed the honest input and rejected all
ten targeted corruptions: P1 binding, P1 producer key, P2 trace root, P2 sample
index, P3 image, P3 opening, P4 codeword, P4 bit, P4 Gaussian value, and P4
LUT. Thus the cost reduction is not an artifact of silently dropping P1--P4.

### Local exact wall-clock measurements

Hardware was the same i7-11700 machine (8 cores/16 threads, approximately
15 GiB WSL memory). Times below are seconds; proof sizes are MiB. A cold row is
one measured proof. A warm row excludes one warm-up and reports five measured
runs.

| condition | protocol | prove median | prove p95 | verify median | verify p95 | proof MiB |
|:----------|:---------|-------------:|----------:|--------------:|-----------:|----------:|
| T2 default cold | v5 CPU | 289.332 | -- | 1.537 | -- | 17.289 |
| T2 fat-LTO cold | v5 CPU | 275.155 | -- | 1.336 | -- | 15.681 |
| T3 streaming cold | v5 CPU | 76.766 | -- | 0.612 | -- | 7.083 |
| T4 streaming warm | v5 CPU | 75.369 | 76.665 | 0.605 | 0.616 | 7.083 |
| T5 streaming cold | v6.3.1 CPU | 90.817 | -- | 0.098 | -- | 2.677 |
| T5 streaming warm | v6.3.1 CPU | 91.616 | 92.511 | 0.101 | 0.102 | 2.677 |

The five v5 warm proof times were 76.665, 75.167, 76.043, 74.883, and
75.369 s. The five v6 warm proof times were 92.465, 91.407, 89.520, 91.616,
and 92.511 s. Persistent state therefore amortizes initialization/setup but
does not remove the per-proof cryptographic work: v5's warm median is only
1.819% below its single cold run.

On this equal CPU, v6 is not automatically a proving-latency win: its warm
median is 21.556% slower than v5. It is, however, 83.387% faster to verify and
its serialized Core proof is 62.208% smaller. This is why T5 must remain a
range until it runs on the actual H100 GPU stack.

The v6 native-CUDA path was also exercised locally. The server loaded the CUDA
runtime and reached its explicit hardware check, then rejected the RTX 3070
because it has 8 GiB and the prover requires at least 24 GiB. This distinguishes
a real GPU-memory blocker from a build or API failure; the requested H100 80GB
clears that capacity gate.

### H100 two-point interpolation

The only comparable actual H100 points are the submitted legacy measurements:

| legacy point | v5 gas | H100 prove | H100 verify | artifact |
|:-------------|-------:|-----------:|------------:|:---------|
| N=200 | 8,584,353 | 32.309 s | 4.468 s | not retained |
| N=1,000 | 47,550,322 | 48.572 s | 7.589 s | 40.68 MB reported |

The exact line through those two points is

`prove_seconds = 28.7261984831 + 4.1736418771e-7 * v5_gas`,

`verify_seconds = 3.7804316833 + 8.0095531565e-8 * v5_gas`,

equivalently 28.726 s plus 0.417364 proving seconds and 3.780 s plus
0.080096 verification seconds per million v5 gas. These are two-point
engineering interpolations, not statistical regressions or confidence
intervals. Their main finding is the approximately 28.7 s proving floor in the
submitted v5 pipeline.

| condition | H100 status | prove outlook | verify outlook | measured proof artifact |
|:----------|:------------|--------------:|---------------:|------------------------:|
| T1 submitted N=1,000 | **actual** | 48.572 s | 7.589 s | 40.68 MB reported |
| T2 strengthened default | v5 interpolation | 32.497 s | 4.504 s | 18,128,550 B |
| T2 strengthened fat-LTO | v5 interpolation | 32.116 s | 4.431 s | 16,442,457 B |
| T3 streaming-v2 cold | v5 interpolation | 29.898 s | 4.005 s | 7,427,300 B |
| T4 streaming-v2 persistent | v5 interpolation plus warm/cold sensitivity | median 29.354 s; p95 29.859 s | median 3.959 s; p95 4.031 s | 7,427,300 B |
| T5 streaming-v2 v6.3.1 persistent CUDA | cross-version sensitivity, **not measured** | 29.129--36.343 s; midpoint 32.736 s | approximately 0.665 s sensitivity anchor | 2,806,931 B |

T4 scales the T3 interpolation by the measured v5 CPU warm/cold ratios. T5's
optimistic endpoint naively inserts v6 normalized gas into the v5 line; its
conservative endpoint instead scales T3 by the observed v6/v5 warm CPU proving
ratio. Neither endpoint is an H100 error bar, and actual v6 GPU behavior may
fall outside it. They are transparent sensitivity bounds used only until the
same ELF runs on H100. T5's verification anchor similarly scales the v5
streaming verification interpolation by the measured same-CPU v6/v5 median
ratio; it is not an H100 measurement. The new artifact byte counts were
measured from successfully generated Core proofs and are not wall-clock
estimates.

The most defensible near-term expectation is therefore about 29--30 s for the
v5 persistent streaming path, a roughly 39% reduction from the submitted
48.572 s. The large 65.4% guest-gas reduction from strengthened fat-LTO to
streaming translates to only about a 6.9% modeled H100 wall-clock reduction
because fixed proof-system work dominates. A single-digit H100 claim remains
unsupported.

### Official v6 evidence boundary

The official v6.3.0 release identifies a DAG-native zerocheck prover and a GPU
performance change that keeps JaggedMle metadata device-resident; v6.3.1 is a
small subsequent release. Neither release publishes an equal-workload v5/v6
H100 timing table. The official end-to-end benchmark tooling records results
locally, and its documentation says the measurement directory is gitignored.
Consequently no vendor number was substituted for the missing PP-Mark H100
run.

Primary sources:

- SP1 v6.3.1: https://github.com/succinctlabs/sp1/releases/tag/v6.3.1
- SP1 v6.3.0: https://github.com/succinctlabs/sp1/releases/tag/v6.3.0
- SP1-GPU end-to-end benchmark: https://github.com/succinctlabs/sp1/blob/v6.3.1/sp1-gpu/crates/perf/README.md
- SP1 persistent benchmark/measurement behavior: https://github.com/succinctlabs/sp1/blob/v6.3.1/crates/perf/README.md

### Reproducibility and final state

Permanent code supporting this campaign is in
`sp1/host/examples/timing_campaign.rs`,
`sp1/host/examples/cycle_compare.rs`,
`scripts/build_sp1_attest_fixture.py`, and the protocol-v2 guest entrypoint.
The harness records SP1 version, backend, ELF hash/size, hardware metadata,
client initialization, setup, execution, proving, verification, serialization,
proof bytes, gas/instructions, warm-up exclusions, raw runs, median, and p95 in
JSON.

Final validation after restoring the repository default:

- `cargo fmt --all -- --check`: pass.
- guest Rust tests: 4 passed.
- Python attestation tests: 7 passed.
- host examples, including the timing harness: compile-check passed. The local
  command explicitly used `/usr/bin/gcc` and `/usr/bin/g++` because the
  H100-oriented `.cargo/config.toml` names gcc-12/g++-12, which are absent on
  this WSL installation.
- restored production v1 execution: honest N=1,000 succeeds at 9,034,677 gas.
- paper directory: untouched by this campaign.

The last Elice retry again returned
`kex_exchange_identification: Connection closed by remote host`. The sole
remaining measurement gap is exact T2--T5 H100 wall-clock time; all code,
fixtures, security checks, local baselines, and honest uncertainty bounds are
ready for a live endpoint.

## A100 native-GPU closure of the timing campaign (2026-08-02 KST)

### Outcome

The replacement Elice endpoint `central-02.tcp.tunnel.elice.io:28835` was live
and exposed one full NVIDIA A100 80GB PCIe GPU. The frozen T5 streaming-v2
relation was successfully proved and self-verified with the SP1 v6.3.1 native
GPU server. Two independent 20-run campaigns, each excluding two warm-ups,
give a combined **40-run prove median of 1.665 s and nearest-rank p95 of
3.551 s**. Median self-contained receipt generation plus serialization and
self-verification was 1.846 s; its p95 was 3.742 s. The serialized Core proof
was 2,806,931 B (2.677 MiB).

This is the first direct native-GPU measurement of the optimized PP-Mark
relation. It overturns the earlier 29--36 s cross-version sensitivity range,
which was necessarily based on a v5 two-point line with a large fixed floor.
That line does not transfer to SP1 v6's native prover architecture. The new
number is an A100 measurement, not an H100 measurement; it must not be relabeled
as H100 without rerunning the same frozen artifact there.

The paper source was not edited during this campaign.

### Frozen measurement identity

| item | identity |
|:-----|:---------|
| guest relation | streaming full-trace commitment, protocol v2, N=1,000, k=32, RS(64,32) |
| guest ELF | 381,384 B; SHA-256 `1a9eb1a3b100ac7de792cb44865946d44301330574922a187c7ce1b237bb1459` |
| exact relation cost | 663,355 instructions; 1,218 syscalls; 966,008 v6 normalized gas |
| SP1 runtime | v6.3.1 Core proof, native `sp1-gpu-server` |
| native GPU server | 232,747,920 B; SHA-256 `7e9c5be8030a5ccf0285eee40a2f54ea6906a32d5bdb5b06c069e64da0cd08fe`; reports `6.3.1` |
| final timing harness | SHA-256 `24837788f546078148acca7a735f711d33b07068ddd2170459b410060d896e09` |
| user-local CUDA runtime | SHA-256 `9335f6a29ca91010e2da9f40e82fb4b28e3a4ae22fd385e1293e93bf3c46c9e6` |
| honest fixture | the frozen `ppmark_attest_streaming_n1000` public statement/private witness |
| proof artifact | 2,806,931 B in every successful run |

The native v6 server contains an explicit `sm_80` target, in addition to newer
GPU targets. This matters for distinguishing the successful v6 path from the
v5 A100 incompatibility below.

### Hardware and container boundary

| property | observed value |
|:---------|:---------------|
| GPU | NVIDIA A100 80GB PCIe, compute capability 8.0, 81,920 MiB |
| driver / reported CUDA capability | 535.183.06 / CUDA 12.2 |
| GPU power limit | 300 W |
| host CPU | Intel Xeon Gold 6338; two physical sockets visible |
| effective CPU allocation | cgroup `cpu.max = 1600000 100000`, i.e. 16 CPU equivalents |
| effective CPU set | `8,12,16,20,24,28,32,36,72,76,80,84,88,92,96,100` |
| memory limit | 206,158,430,208 B (192 GiB) |
| OS / libc | Ubuntu 22.04 container; glibc 2.35; Linux 5.15.0-97 |
| Rust used for remote harness | rustc 1.97.1 |

This A100 host is not the submitted H100 host: GPU architecture, GPU bandwidth,
CPU allocation, container image, and runtime all differ. Comparisons to the
submitted result are therefore system-level before/after endpoints, not a
single-factor hardware ablation.

### Cold and persistent timing semantics

Times are milliseconds in this table. `total` is the sum of measured client
initialization, setup, dry execution, proof, serialization, and self-verify.

| path | client init | setup | execute | prove | verify | serialize | total |
|:-----|------------:|------:|--------:|------:|-------:|----------:|------:|
| first installation/start | 20,416.175 | 24,054.457 | 80.851 | 2,098.984 | 159.971 | 3.856 | 46,814.293 |
| cached server, fresh process | 883.450 | 14,886.354 | 60.552 | 1,626.339 | 186.300 | 2.969 | 17,645.964 |

The first row includes downloading/starting the native server and is not a
per-artifact steady-state number. The second still regenerates setup state for
the ELF. A persistent service retains one client and proving key; its
per-artifact critical path is the proof (and, if desired, serialization and
self-verification), not either cold total. Dry execution is separately timed
for diagnostics and need not be duplicated before every production proof.

The first successful proof wrote valid JSON but the synchronous diagnostic
process subsequently exited nonzero because the v6 CUDA client's destructor
attempted an asynchronous cleanup after its Tokio reactor had closed. The
harness was repaired to drop the proving key and client inside a small live
Tokio runtime. The clean run and every distribution run then exited zero. This
was a host-lifecycle bug after proof verification, not a relation or GPU-proof
failure.

### Primary persistent distribution

The primary result pools only two runs with the same protocol: one process,
one setup, two excluded warm-ups, and 20 measured proofs. Cold and diagnostic
runs are not mixed into it.

| metric, n=40 | median | p95 | min | max | mean |
|:-------------|-------:|----:|----:|----:|-----:|
| dry execute (ms) | 80.126 | 101.768 | 53.415 | 104.064 | 81.905 |
| prove (ms) | **1,665.342** | **3,551.223** | 1,532.611 | 3,750.414 | 2,018.868 |
| verify (ms) | **183.269** | **197.594** | 163.735 | 205.888 | 182.578 |
| serialize (ms) | 2.227 | 3.301 | 2.109 | 3.327 | 2.303 |
| prove + serialize + self-verify (ms) | **1,845.804** | **3,742.221** | 1,723.343 | 3,940.298 | 2,203.749 |

Thirty-two of 40 prove runs were below 2 s; eight were at or above 3 s. The
raw prove times, retained to prevent favorable-run selection, were:

- campaign A: 1.637, 1.629, 1.534, 1.608, 1.612, 1.533, 1.667, 1.610,
  1.629, 1.689, 1.621, 1.596, 1.615, 1.542, 1.654, 1.610, 1.911, 3.551,
  3.404, 3.489 s;
- campaign B: 1.678, 1.634, 1.649, 1.653, 1.717, 1.766, 1.704, 1.720,
  1.664, 1.769, 1.655, 1.707, 1.736, 1.670, 1.603, 3.146, 3.416, 3.683,
  3.294, 3.750 s.

The two campaign setup times were 33.322 s and 28.937 s. These one-time setup
values were variable on the cloud host and are reported separately rather than
being hidden inside the per-proof number.

### Variability diagnosis

Several auxiliary sequences were run to test whether the 3 s tail was caused
by a deterministic leak or thermal throttling:

- three fresh-process proofs were 1.565, 1.568, and 3.402 s;
- an initial five-run sequence was 1.622, 1.658, 2.786, 3.453, and 3.237 s;
- after an idle interval, five runs moved in the opposite direction: 3.635,
  3.393, 2.607, 1.624, and 1.669 s;
- a dedicated ten-run diagnostic moved from 2.80--2.96 s to 1.54--1.58 s.

During the instrumented diagnostic:

- cgroup `nr_throttled` and `throttled_usec` both increased by zero;
- GPU temperature stayed at or below 42 C;
- SM clock stayed at 1,410 MHz;
- GPU memory plateaued near 13.64 GiB instead of growing across proofs;
- fast proof intervals had higher sampled GPU utilization than slow intervals,
  while the guest, public input, witness, gas, and proof size were identical.

An earlier 20-run trace similarly stayed below 44 C, held 1,410 MHz, and
plateaued near 13.61 GiB. These observations rule out thermal throttling,
monotonic device-memory leakage, and the measured container CPU quota as the
immediate cause. They are consistent with cloud/GPU runtime scheduling
variability, but do not identify its internal cause. The defensible report is
therefore median plus p95, not a claim that every proof takes 1.6 s.

### Relation integrity after optimization

The exact v6 ELF that generated the successful GPU proofs was also executed
against the complete protocol-v2 tamper suite on the remote host. The honest
case exited zero. Each of ten targeted corruptions exited nonzero with a public
value mismatch:

`P1 binding`, `P1 producer key`, `P2 sample index`, `P2 trace root`,
`P3 image`, `P3 opening`, `P4 bit`, `P4 codeword`, `P4 Gaussian value`, and
`P4 LUT`.

This execute-only suite does not substitute for cryptographic soundness; it
checks that the optimized guest still enforces the intended relation. Combined
with successful Core proof generation and self-verification of the honest case,
it closes the engineering risk that the speedup came from silently deleting a
P1--P4 predicate.

### Why SP1 v5 could not be measured on this A100

The v5 path was not abandoned at the missing-Docker boundary. The pinned
official image `public.ecr.aws/succinct-labs/moongate:v5.0.8` was pulled by
digest, exported into user space, and its server was started directly against
the exposed GPU:

| item | identity/result |
|:-----|:---------------|
| image digest | `sha256:40d8f1c1cb6d58072d119d8986e68e945805c63c6bb6440f849d9e54e684d74a` |
| server binary | 47,184,784 B; SHA-256 `1e34c3505a5cf346ccac9e20d84b8553c9a76ef0a30e0fd1216c540fafad8bf1` |
| server initialization | successful in approximately 5.51 s |
| host/client | SP1 5.2.3 timing harness rebuilt under remote glibc 2.35; SHA-256 `47cdaf91304c53f3003f0aab51803d66ec36c9978c6407674735a79a3fca9d70` |
| request result | `/Setup` reached the server, then the server aborted with `Rust cannot catch foreign exceptions` |

The architecture cause was then established with the CUDA image's own
`cuobjdump`: the server contains cubins only for `sm_86` and `sm_89`, plus PTX
targeting `sm_89`. The provided A100 is `sm_80`. Retrying once with the host
driver libraries and once with the image's CUDA 12.5 forward-compatibility
libraries produced the same setup abort. A binary compiled for the higher
targets cannot run its kernels on `sm_80`; no public v5 image tag advertised an
A100-specific build, and the SP1 5.2.3 client itself pins `v5.0.8`.

Therefore T1--T4 cannot receive honest A100 GPU timings from this image. The
correct entries are **not measurable on this A100 with the pinned official v5
server**, not estimated values and not failed PP-Mark proofs. Rebuilding the v5
GPU server for `sm_80` would require server source/build support outside the
released client/image boundary. Existing CPU measurements and the historical
H100 v5 measurements remain the valid v5 evidence.

### Calibrated comparison and revised engineering conclusion

Against the submitted legacy H100 endpoint, the new optimized A100 endpoint is:

| endpoint | prove | verify | proof artifact |
|:---------|------:|-------:|---------------:|
| submitted legacy v5 on H100 | 48.572 s | 7.589 s | 40.68 MB reported |
| streaming-v2 v6.3.1 on A100, persistent median | 1.665 s | 0.183 s | 2,806,931 B |
| streaming-v2 v6.3.1 on A100, persistent p95 | 3.551 s | 0.198 s | 2,806,931 B |

As a system-level before/after endpoint, the measured prove reduction is
96.57% at the median and 92.69% at p95; self-verification is 41.4x faster at
the median. Treating the reported 40.68 MB as decimal MB, the serialized
artifact is 93.10% smaller. These percentages combine relation, compiler,
runtime, and hardware changes and must not be described as any one component's
ablation.

The practical conclusion is stronger than the earlier estimate:

1. A full P1--P4 Attest receipt no longer has an inherent approximately 48 s
   per-artifact proving cost. On the measured A100, a persistent v6 service
   produced it in 1.665 s median / 3.551 s p95.
2. Setup and first-install costs still matter operationally, but they are
   service/ELF lifecycle costs and can be amortized; they must remain separate
   from per-artifact proof latency.
3. Detect should remain the transform-tolerant, low-latency routine path, while
   Attest remains the stronger exact-origin path. The new timing removes the
   need to frame Attest as nearly unusable, but it does not change the semantic
   split or make exact receipts tolerant to JPEG/resizing.
4. The same v6 ELF should still be rerun on H100 before publishing an H100
   number. Until then, the A100 median/p95 are the measured claims; any H100
   expectation is explicitly prospective.
5. Porting to another proving system is no longer the highest-leverage latency
   task. The immediate priorities are H100 replication, persistent-service
   packaging, and understanding/reducing the observed cloud-runtime tail.

### Evidence preservation

All remote result JSON, raw per-run timings, CPU/GPU telemetry, v5 server logs,
container manifest, CUDA target listings, and tamper status files were copied
back before instance teardown and archived as
`docs/ppmark_a100_campaign_evidence_2026-08-02.zip` (42,020 B; SHA-256
`e09c636f0dec1d115019241de83b2c2819c844e1c04b67fe5f5ac0f7e67bab99`).
The uncompressed working copy is also retained at
`C:\Users\SOGANG\AppData\Local\Temp\ppmark_a100_results_20260802`.

### H100 SXM sensitivity projection from the measured A100 (not measured)

For planning only, the measured A100 PCIe persistent distribution was mapped
to the earlier H100 80GB SXM class with an explicit fixed-cost model,

`T_H100 = F + (T_A100 - F) / r`.

The hardware sensitivity spans the approximately 1.73x memory-bandwidth ratio
(3.35/1.935 TB/s) and the approximately 3.44x FP32 ratio (67/19.5 TFLOP/s).
Because the SP1 proof mixes GPU kernels, transfers, and host orchestration,
neither endpoint is asserted as the true scaling factor. Fixed host/runtime
cost `F` is deliberately left between 0.30 and 0.60 s in the scenarios below.

| sensitivity case | assumed `r` | assumed `F` | projected prove median | projected prove p95 | plus measured verify median/p95 |
|:-----------------|------------:|------------:|-------------------------:|----------------------:|---------------------------------:|
| memory-bound / conservative | 1.731 | 0.60 s | 1.215 s | 2.305 s | 1.399 / 2.502 s |
| balanced planning point | 2.300 | 0.45 s | **0.978 s** | **1.798 s** | **1.162 / 1.996 s** |
| compute-bound / optimistic | 3.436 | 0.30 s | 0.697 s | 1.246 s | 0.881 / 1.444 s |

The planning estimate is therefore approximately **1.0 s median and 1.8 s
p95 for proving**, or approximately **1.16 s median and 2.0 s p95 including
self-verification**. The sensitivity envelope is 0.70--1.22 s median and
1.25--2.31 s p95. This is not a confidence interval. Moreover, the A100's
cloud-runtime tail may not scale with raw GPU specifications, so an operational
upper budget should retain the measured A100 p95 of 3.551 s until the identical
ELF is run on H100.

The one-time setup interval is not converted with the GPU ratios because it is
substantially host/runtime dependent. Proof size remains exactly 2,806,931 B
regardless of GPU. The historical 48.572 s H100 v5 result is not used as a
calibration point here because it changes the guest relation and proof-system
generation simultaneously.

## Quantitative speedup provenance audit (2026-08-02 KST)

This section supersedes any earlier shorthand that could be read as treating
`48.572 s -> 1.665 s` as a controlled, like-for-like prover ablation. Both
endpoint values stand as observations, but their ratio is descriptive only:
the guest relation, compiler, SP1 generation, hardware, process lifetime, and
timer boundary all changed.

### Revalidated execution work

The retained pre-alignment ELF (150,428 B, SHA-256
`76a481aadb37b02152893e5e963789dc3896f1d3618e011e4e4552a694d15316`)
was rerun on the retained N=1,000 fixture during this audit. It reproduced
exactly 24,873,425 instructions, 42,091 syscalls, and 47,550,322 v5 gas. The
optimized A100 proof ELF remains 381,384 B with SHA-256
`1a9eb1a3b100ac7de792cb44865946d44301330574922a187c7ce1b237bb1459`.

The controlled v5 execution-cost chain is:

| stage | instructions | syscalls | v5 gas | change from preceding stage |
|:------|-------------:|---------:|-------:|:----------------------------|
| legacy N=1,000, one depth-10 path per sample | 24,873,425 | 42,091 | 47,550,322 | baseline |
| strengthened P1--P4, one complete Merkle build | 2,774,702 | 6,970 | 9,034,677 | gas -81.000% (5.263x) |
| same relation, fat LTO | 2,245,151 | 6,970 | 8,121,446 | gas -10.108% (1.112x) |
| protocol-v2 full-trace streaming commitment | 644,072 | 1,534 | 2,807,884 | gas -65.426% (2.892x) |

The product of those reductions is a real **16.935x v5-gas reduction** from
legacy to streaming-v2 (94.095%), together with 38.619x fewer instructions
and 27.439x fewer syscalls. This is the defensible computation reduction. The
v6 report, 663,355 instructions / 1,218 syscalls / 966,008 normalized gas, is
kept separate because v5 gas and v6 normalized gas use different zkVM cost
models and must not be divided.

The dominant source-level cause is also numerically accounted for:

- Legacy verified 1,000 independent depth-10 Merkle paths. Each sample used
  one one-block leaf hash plus ten 64-byte internal hashes, each of which needs
  two SHA-256 compression blocks after padding: approximately 21,000 SHA
  blocks, with shared tree nodes redundantly recomputed across paths.
- The strengthened v1 guest built the 1,024-leaf padded tree once. P2 used
  exactly 3,110 SHA blocks: 1,000 leaves, 2,046 blocks for 1,023 internal
  nodes, and 64 blocks for the sample-index transcript. The full program used
  3,327 SHA compression/extension pairs.
- Streaming-v2 replaced only that tree reconstruction with one
  domain-separated, length-bound SHA-256 over the same ordered 1,000-entry
  trace. The full program used 610 SHA pairs. Thus the measured whole-program
  SHA count fell 81.67% from strengthened v1 and approximately 97.10% from
  legacy (about 34.4x).

This is why the reduction is large rather than mysterious: the legacy guest
performed O(N log N) repeated membership work even though all N leaves were
already private inputs to one receipt; v1 changed this to one O(N) tree build,
and v2 changed the same full-vector binding to one streaming hash. Fat LTO is
a secondary improvement, not the main cause.

### Wall-clock timer audit

The historical Python timer surrounded the entire SP1 host `prove` subprocess.
At the submitted sweep settings (`SP1_SKIP_EXECUTE=false` and
`SP1_SKIP_VERIFY=false`), that command included process/client initialization,
key loading or setup, dry execution, proof generation, internal verification,
public-output handling, serialization, and receipt writing. The exact local
receipt/ELF lineage for the later 48.572 s paper row was not recovered; it
remains a reported submission value rather than a reproducible frozen timing
artifact.

The v6 A100 campaign deliberately separated those phases:

| timing boundary | median | p95 | status |
|:----------------|-------:|----:|:-------|
| persistent `prove()` only | 1.665 s | 3.551 s | measured, 40 runs |
| prove + serialize + self-verify | 1.846 s | 3.742 s | measured, 40 runs |
| cached server, fresh process including init/setup/execute/prove/verify/serialize | 17.646 s | -- | measured once |
| first install/server start plus the same phases | 46.814 s | -- | measured once; deployment cold start |

Accordingly:

- `48.572 / 1.665 = 29.17x` is only a cross-generation endpoint ratio, not a
  controlled proof-kernel speedup.
- Even `48.572 / 1.846 = 26.31x` still mixes old fresh-subprocess behavior with
  a new persistent service.
- The most conservative observed full-command-style comparison is
  `48.572 / 17.646 = 2.75x`, but it too is not controlled because the old
  artifact and setup/cache state are unavailable and the hardware differs.
- Independent same-host CPU evidence confirms that the structural change has
  a substantial time effect: strengthened fat-LTO v5 took 275.155 s cold,
  while streaming-v2 v5 took 76.766 s cold (3.584x faster). This is the
  strongest available wall-clock isolation of the relation change.

SP1 v6 is not itself a universal latency win. On the same local CPU and the
same streaming-v2 relation, the v6 warm median was 91.616 s versus 75.369 s
for v5, i.e. v6 was 21.56% slower to prove. It was 83.39% faster to verify and
reduced the serialized Core proof from 7,427,300 B to 2,806,931 B (62.21%).
The A100 result therefore comes from the small optimized relation combined
with v6's native CUDA prover, not from a blanket claim that v6 reduces guest
work. Succinct's official documentation likewise separates client/setup
initialization from proof latency and recommends retaining a persistent client:
https://docs.succinct.xyz/docs/sp1/generating-proofs/hardware-acceleration and
https://docs.succinct.xyz/docs/sp1/getting-started/migration.

### Relation-equivalence boundary

Streaming-v2 still reads and binds every ordered trace entry, validates the
sample-index set, enforces P1 key/context binding, recomputes RS(64,32), derives
the image/context/trace-bound k=32 challenge, and checks the P3/P4 opening,
Gaussian/LUT/bit/embedding relations. The exact GPU ELF passed the honest case
and rejected all ten targeted P1--P4 corruptions. The speedup is therefore not
explained by a missing predicate.

It is nevertheless a protocol change, not a byte-for-byte implementation
optimization. A flat full-trace digest is collision-resistant binding to the
complete ordered vector, but unlike a Merkle root it does not support reusable
selective membership proofs. PP-Mark's current receipt did not expose or reuse
such external paths, so the operational full-trace claim can be retained; any
revised paper must still name protocol v2 and restate the commitment/security
argument rather than silently substituting it into the submitted theorem.

### H100 projection audit

The earlier H100 figures remain a sensitivity calculation, not measurements.
NVIDIA's published H100-SXM/A100-80GB-PCIe ratios are 1.731x in memory
bandwidth (3.35/1.935 TB/s) and 3.436x in FP32 throughput (67/19.5 TFLOP/s).
Applying those endpoints with an explicit 0.30--0.60 s non-scaling floor to
the measured A100 distribution yields 0.697--1.215 s median and 1.246--2.305 s
p95. The balanced 0.978/1.798 s point is mathematically consistent, but it is
not empirically validated and is not a confidence interval.

For internal planning, use approximately **1.0--1.2 s median and 1.8--2.3 s
p95** on H100. For a paper or reviewer response, use only the measured A100
1.665 s median / 3.551 s p95 until the identical frozen ELF is actually rerun
on H100. Raw GPU specifications do not determine how the cloud-runtime tail or
host orchestration will scale.

### Full-process-boundary reconciliation

The 40 retained A100 runs were recomputed per run rather than by adding
independently reported distribution summaries. Two per-artifact boundaries are
therefore available exactly:

| A100 v6 persistent boundary | median | p95 | interpretation |
|:----------------------------|-------:|----:|:---------------|
| prove + verify + serialize | 1.845804 s | 3.742221 s | receipt issuance without redundant dry execution |
| dry execute + prove + verify + serialize | 1.923244 s | 3.842936 s | phases performed by the historical host `prove` command after client/key availability |

The second row is the closest measured persistent per-receipt analogue to the
historical command semantics. It still excludes process launch, client
initialization, and key setup/loading. The measured cached-server fresh-process
phase sum was 17.645964 s: 0.883450 s client initialization, 14.886354 s setup,
0.060552 s dry execution, 1.626339 s proof, 0.186300 s verification, and
0.002969 s serialization. It is not a literal externally timed subprocess
wall clock because fixture parsing, executable launch, and final file/JSON
handling were outside the component timers; those were not measured on the
A100. It also regenerates setup state and is therefore not the fastest
persistent deployment.

Applying the same H100 sensitivity model to the complete persistent
per-receipt phases, while conservatively leaving execute, verification, and
serialization unscaled, gives:

| H100 sensitivity | receipt issuance median/p95 | historical-command phases median/p95 | fresh-process phase sum if all non-prove phases stay fixed |
|:-----------------|-----------------------------:|----------------------------------------:|----------------------------------------------------------:|
| memory-bound / conservative | 1.400945 / 2.505819 s | 1.481071 / 2.607587 s | 17.212542 s |
| balanced | **1.163906 / 1.999253 s** | **1.244032 / 2.101021 s** | **16.981077 s** |
| compute-bound / optimistic | 0.882860 / 1.447118 s | 0.962986 / 1.548886 s | 16.705638 s |

These are exact outputs of the stated sensitivity assumptions, not exact H100
measurements. In particular, setup cannot be responsibly scaled from GPU peak
specifications; the fresh-process column is only a hold-all-other-phases-fixed
calculation.

The legacy two-point H100 v5 model provides an important negative check. Its
fit from N=200 and N=1,000 was

`prover-command seconds = 28.7261984831 + 4.1736418771e-7 * v5_gas`.

Substituting streaming-v2's 2,807,884 v5 gas gives **29.898109 s**, not about
1.24 s. Even the unrealistic zero-fixed-overhead proportional calculation
`48.572 * 2,807,884 / 47,550,322` gives **2.868215 s**. Therefore the old v5
workload/timing data do **not** independently predict or validate the new H100
approximately 1.24 s full-phase projection. That projection depends on the
measured v6 native-CUDA A100 endpoint, persistent lifecycle, and assumed
A100-to-H100 scaling. It must remain a forecast until measured on H100.

### Final operational interpretation retained for future revision

- The submitted legacy 48.572 s result is internally consistent with the old
  v5 H100 calibration and its measured 47,550,322-gas workload. The missing
  exact artifact lineage limits forensic reproduction, but the value is not
  contradicted by the retained computation evidence.
- The optimized implementation is supported independently by exact v5 work
  reduction (16.935x gas), repeated v6 A100 proofs, self-verification, stable
  proof size, and the full P1--P4 tamper suite.
- The measured A100 cached-server fresh-process component sum is **17.645964
  s**. Under the balanced hold-non-prove-phases-fixed conversion, the H100
  counterpart is **16.981077 s**, which must be communicated as approximately
  **17.0 s projected**, not as an H100 measurement. The sensitivity range is
  16.705638--17.212542 s, excluding unmeasured outer process/file overhead.
- With setup/client state amortized, the closest complete persistent
  per-receipt path (`execute + prove + verify + serialize`) is **1.923244 s
  median / 3.842936 s p95 measured on A100** and **1.244032 s median /
  2.101021 s p95 projected on H100** under the balanced model.
- Future reporting must state the lifecycle explicitly: use the approximately
  17 s number for a cold fresh process that regenerates setup, and the
  approximately 1.24 s number only for a persistent H100 service. Never mix
  either projection with the legacy 48.572 s endpoint as a controlled
  like-for-like speedup.

## Production-code consolidation audit (2026-08-02 KST)

The current implementation was consolidated without editing the paper. The
production path now uses SP1 `6.3.1` and protocol-v2 full-trace streaming from
artifact generation through host execution, proof serialization, exact public-
statement comparison, and verification. The protocol-v1 Merkle relation is
retained only as an explicitly labelled historical diagnostic for the
controlled cycle comparison; it is not selected by the production host.

The release profile is pinned to fat LTO and one code-generation unit. A fresh
local build from the current source reproduced the exact A100 campaign guest
ELF: 381,384 B, SHA-256
`1a9eb1a3b100ac7de792cb44865946d44301330574922a187c7ce1b237bb1459`.
The release host was rebuilt around that ELF. Consequently, the retained A100
campaign is evidence for the same proving program now selected by production,
not merely a source-similar program.

The main repository, anonymous supplementary directory, and nested
supplementary directory were audited file-by-file. All 29 implementation files
shared by all three locations are byte-identical after consolidation. The audit
also found and repaired two prior tool-output truncation artifacts in the
supplementary `cli.py` and `Cargo.lock`, plus three stale nested-copy numerical
alignment edits (`ddim_unet.py`, `noise.py`, and `tables.py`). No truncation
sentinel remains.

Validation after repair:

- Python targeted tests: 9 passed, 0 failed.
- Rust formatting check: passed.
- Rust SP1 v6.3.1 all-target check: passed; diagnostic examples emit only
  non-fatal dead-code warnings.
- Rust host unit tests: 2 passed, 0 failed; all examples compiled.
- Exact frozen-ELF mock prove/serialize/verify and exact public-values check:
  passed. This is a control-flow check, not a cryptographic proof.
- Exact frozen-ELF honest CPU execution: passed; representative P1 tampering
  was rejected. The separately retained A100 campaign supplies the real proof,
  self-verification, 40-run timing distribution, and all ten P1--P4 targeted
  rejection results.

The anonymous code archive was rebuilt in place, excluding generated caches,
`pyc`, `target`, `.git`, and key files. Final archive: 55 entries, 399,597 B,
SHA-256
`200a1dfa0e09c94722a2c0879973b618c8f5481cd934ff1fedc681b1dfbf05b6`.

### Current circuit-selection conclusion

Within PP-Mark's required P1--P4 relation, full private trace availability,
single-receipt workflow, and the evaluated SP1 backends, protocol-v2 streaming
is the strongest current implementation choice: it preserves binding to every
ordered trace entry, removes redundant Merkle membership work that the receipt
does not expose, has an exact frozen build, and has real A100 proof/tamper
evidence. This is not a claim that it is universally optimal among every ZK
system or future backend. A different choice may become preferable if external
selective openings, recursive aggregation, materially smaller public artifacts,
or another deployment constraint becomes a primary requirement.

No new H100 proof was run in this consolidation. The approximately 17.0 s cold
fresh-process and approximately 1.24 s persistent-service H100 values remain
projections only; the measured values remain the A100 results recorded above.

## Detect-latency boundary after circuit optimization (2026-08-04 KST)

- The submitted **0.72 s Score/Detect verification** was measured on H100 PCIe,
  SD2.1 at 512x512 with DDIM-50, n=10, with model/pipeline loading excluded.
  The benchmark measured inversion separately and found all DDIM-based methods
  near 0.72 s because inversion dominates.
- The protocol-v2 streaming-commitment, fat-LTO, and SP1 v6.3.1 changes optimize
  the **Attest guest/proof path**. Detect does not execute the SP1 guest or
  generate/verify a receipt; it performs image inversion, alignment, expected-
  sign reconstruction, and correlation scoring. Therefore no Detect speedup
  may be inferred from the new circuit timings.
- The current Detect implementation does perform inexpensive canonical metadata,
  RS, LUT-hash, and sample-set consistency checks before scoring. Their cost has
  not been isolated, but it is not the source of the circuit speedup and is
  expected to be small relative to DDIM inversion.
- Any lower Detect number requires a separate controlled benchmark: same GPU,
  persistent loaded inverter, explicit end-to-end versus inversion-only timer,
  and a DDIM-step/batching/precision optimization sweep followed by threshold,
  FPR, TPR, transform, and attack revalidation. Until then retain 0.72 s as the
  measured Detect value and update only the Attest cost.
