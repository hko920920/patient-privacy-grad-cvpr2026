# Method Draft Review Log

## Scope
Reviewed the provided Method draft (up to Deterministic Sampling) against the current SP1-only, score+proof detection rule.

## Key Findings
- The Framework Overview section still describes Merkle-root consistency as a hard acceptance condition. This conflicts with the current rule: acceptance is `score >= tau AND SP1 receipt valid`; root recomputation is diagnostic only.
- The verification description should not claim that root consistency is required. It can mention root recomputation as optional diagnostics.
- The binding equations are aligned with SP1 (SHA-256 ctx-hash and binding), but should clarify that tags are optional and that width/height are fixed-width encodings.

## Required Edits (Minimal)
1) Framework Overview: remove “checks root consistency” as a gate; replace with “optional diagnostic.”
2) Hard-check definition: should refer only to SP1 receipt verification; do not gate on root recomputation.
3) Add explicit acceptance rule: `score >= tau` AND `receipt valid`.

## Optional Clarifications
- Sampling detail: sampling seed derived from RS codeword; bit-index map is hashed over (binding, x, y) to avoid spatial resonance.
- Mention that public inputs contain the committed root, but the verifier does not need to recompute it to accept.
