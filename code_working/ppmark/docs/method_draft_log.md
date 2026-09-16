# Method Draft Worklog (PP-Mark)

## 2025-02-14

### Request summary
- User asked for a 3–4 A4 page Method draft based on the previously agreed SP1-only design.
- Scope exclusions: implementation details, computational complexity, failure modes/comparisons (former sections 8/9) are removed.
- Deliver content sequentially (“from the top, one by one”) with concise, core explanations and minimal low-level details.

### Key constraints confirmed
- ZKP backend is SP1 only (no RISC0/Halo2 in mainline method).
- Acceptance rule: `score ≥ τ` AND `SP1 receipt valid`.
- Merkle root and RS decode are diagnostic only; they must not gate acceptance.
- Avoid GPU/command-level or code-specific implementation details.

### Draft structure to deliver
1) Overview & Threat Model  
2) Notation  
3) Binding & Payload  
4) Deterministic Sampling & Bit Mapping  
5) Watermark Embedding  
6) SP1 Proof Generation  
7) Verification (Score + Proof Gate)  
8) Synchronization & Security Rationale

### Action taken
- Delivered Section 1 (Overview & Threat Model).
- Refined verification rule wording to “score + SP1 receipt” (no soft/hard phrasing).
- Delivered Section 2 (Notation).
- Prepared Table 1 (notation/parameter summary).
- Provided a concise Table 1 variant with essential symbols only.
- Removed optional tags from the table after user requested essentials only.
- Delivered Section 3 (Binding and Payload) in a minimal form.
- Delivered Section 4 (Deterministic Sampling and Bit Mapping).
- Delivered Section 5 (Watermark Embedding).
- Prepared Section 6 (SP1 Proof Generation) for delivery.
- Delivered Section 6 (SP1 Proof Generation).
- Delivered Section 7 (Extraction and Verification).
- Delivered Section 8 (Synchronization and Security Rationale).
- Prepared Figure 1 (end-to-end pipeline), Figure 2 (Merkle tree + leaf format), and Algorithm 1 (Verify + Extract) LaTeX snippets.
- Clarified diagram label formatting: italics for math symbols, plain text for labels.
- Provided insertion text explaining decentralized/public verification and no central decoder/API requirement.
- Refined Binding and Payload section language (domain separation, length-prefix encoding, RS role).
- Expanded SP1 proof + verification sections with public-verification/anti-forgery messaging and tightened terminology.
- Prepared a verification-rule diagram (score + proof gate) as an optional figure.
- Drafted an Embed+Commit algorithm (Algorithm 2) to complement Verify+Extract.
- Expanded detection score explanation to clarify it as latent alignment/consistency rather than perceptual similarity.

### Pending
- Continue Sections 3–8 sequentially on request.
