# Privacy-Claim Audit Truth Table v2.0

Status: normative companion to `privacy_claim_contract_v2_0.md`  
Frozen: 2026-07-24

## Field Meanings

- `K`: validated worst-case transformation distance in the accountant's input
  metric, not observed incidence alone.
- `unit_path`: `DIRECT`, `CONVERT`, `GROUP`, or `UNDERSPECIFIED`.
- `release`: certificate `release_status`.
- `assurance`: certificate `assurance_status`.

Every row assumes finite `epsilon >= 0`, `0 < delta < 1`, and a successfully
recomputed registered base accountant unless the row explicitly tests one of
those obligations. Rows mentioning a registered extension are conditional
contract cases: if no matching checker is installed, they resolve to
`UNDERSPECIFIED / BLOCKED_UNVERIFIED`.

## Core Truth Table

| ID | Claim and evidence condition | K | Unit path | Release | Assurance | Required action or wording |
| --- | --- | ---: | --- | --- | --- | --- |
| T01 | Window claim; window accountant; exact adjacency, sampler, schedule, clipping/noising, and runtime match | 1 | `DIRECT` | `ALLOWED` | `EVIDENCE_VALIDATED` | State window-level DP and the accountant adjacency. |
| T02 | Same as T01 but runtime binding is missing | — | `UNDERSPECIFIED` | `BLOCKED_UNVERIFIED` | `DIAGNOSTIC_ONLY` | Bind a matched runtime trace before emitting any copy-ready statement about the concrete run. |
| T03 | Window claim with sampler/accountant mismatch | — | `UNDERSPECIFIED` | `BLOCKED_INVALID` | `DIAGNOSTIC_ONLY` | No positive DP sentence. Name the sampler mismatch. |
| T04 | Window claim with non-finite/negative epsilon or delta outside `(0,1)` | — | `UNDERSPECIFIED` | `BLOCKED_INVALID` | `DIAGNOSTIC_ONLY` | Reject the report before unit reasoning. |
| T05 | Event claim; observed event multiplicity one; no transformation-stability proof | — | `UNDERSPECIFIED` | `BLOCKED_UNVERIFIED` | `DIAGNOSTIC_ONLY` | Observed non-overlap is not sufficient. |
| T06 | Event claim; validated `K=1`; raw and accountant adjacency metrics match | 1 | `DIRECT` | `ALLOWED` | `EVIDENCE_VALIDATED` or conditional | Reuse the accountant number and state the stability condition. |
| T07 | Replace-one raw event changes one generated record; accountant uses add/remove adjacency | 2 | `CONVERT` | `ALLOWED` if nonvacuous | `EVIDENCE_VALIDATED` or conditional | Apply the built-in `K=2` group conversion; do not reuse the same number. |
| T08 | Event claim; validated `K>1`; built-in conversion has `delta_K<1` | K | `CONVERT` | `ALLOWED` | `EVIDENCE_VALIDATED` or conditional | Print recomputed `epsilon_K`, `delta_K`, and the metric used for K. |
| T09 | Event claim; validated `K>1`; built-in conversion has `delta_K>=1` | K | `CONVERT` | `BLOCKED_VACUOUS` | `DIAGNOSTIC_ONLY` | Show the diagnostic conversion but emit no copy-ready DP claim. |
| T10 | Owner claim; native owner sampler, one owner vector before clipping/noising, owner accountant, matched owner adjacency | 1 | `DIRECT` | `ALLOWED` | `EVIDENCE_VALIDATED` or conditional | State owner-level DP and every native-owner premise. |
| T11 | Owner claim; window accountant; owner removal changes one generated record and no other pipeline state | 1 | `DIRECT` | `ALLOWED` | `EVIDENCE_VALIDATED` or conditional | Reuse only under the validated add/remove owner-to-window metric. |
| T12 | Owner claim; window accountant; validated owner distance `K>1`; nonvacuous group conversion | K | `GROUP` | `ALLOWED` | `EVIDENCE_VALIDATED` or conditional | Print recomputed group parameters and the contribution bound. |
| T13 | Owner claim; group conversion is vacuous | K | `GROUP` | `BLOCKED_VACUOUS` | `DIAGNOSTIC_ONLY` | Recommend native owner accounting, a public contribution bound, or narrower wording. |
| T14 | Multi-owner record with complete conservative attribution and a registered relational stability checker | plugin | According to checker | According to checker | According to evidence | Charge every attributed owner; the built-in owner-partition checker rejects attribution arity above one. |
| T15 | Multi-owner or ambiguous record encoded as one arbitrary owner or pseudo-owner | — | `UNDERSPECIFIED` | `BLOCKED_INVALID` | `DIAGNOSTIC_ONLY` | Require complete attribution or a registered relational analysis. |
| T16 | Private global normalization or private adaptive selection is unaccounted | — | `UNDERSPECIFIED` | `BLOCKED_INVALID` | `DIAGNOSTIC_ONLY` | Use public/local preprocessing, separately DP preprocessing, or a validated global stability bound. |
| T17 | Separately DP preprocessing with a registered release/composition graph | plugin | According to registered graph | According to registered graph | `EVIDENCE_VALIDATED` | Use only the registered composition or post-processing rule. |
| T18 | Separately DP preprocessing is named but its release/composition graph is absent | — | `UNDERSPECIFIED` | `BLOCKED_UNVERIFIED` | `EXTERNAL_UNVERIFIED` | Do not guess composition. |
| T19 | Fixed mapping produced by a public deterministic policy and bound to the run | K | According to K | According to conversion | According to runtime | Normal fixed-mapping path. |
| T20 | Mapping is merely the observed result of a private adaptive policy | — | `UNDERSPECIFIED` | `BLOCKED_UNVERIFIED` | `DIAGNOSTIC_ONLY` | An observed branch cannot certify worst-case stability. |
| T21 | `union` mode has an executable complete branch-union construction | K | According to K | According to conversion | `EVIDENCE_VALIDATED` or conditional | Record the union-construction id and digest. |
| T22 | `union` is only a user-selected label over an observed mapping | — | `UNDERSPECIFIED` | `BLOCKED_UNVERIFIED` | `DIAGNOSTIC_ONLY` | Reject the claimed union bound. |
| T23 | `stochastic_bound` has only user integers or Monte Carlo evidence | — | `UNDERSPECIFIED` | `BLOCKED_UNVERIFIED` | `DIAGNOSTIC_ONLY` | Retain diagnostics only. |
| T24 | `stochastic_bound` has a registered uniform theorem, verified beta, and delta rule | plugin | According to plugin | According to plugin | `EVIDENCE_VALIDATED` | Print the bound, beta, and exact unconditional/conditional convention. |
| T25 | Arbitrary external-conversion JSON has complete-looking fields but no registered checker | — | `UNDERSPECIFIED` | `BLOCKED_UNVERIFIED` | `EXTERNAL_UNVERIFIED` | Record it as an attachment but emit no positive wording. |
| T26 | Registered same-mechanism conversion matches every bound field and recomputes successfully | plugin | `CONVERT` or `GROUP` | According to plugin | `EVIDENCE_VALIDATED` | Print the registered method and bound pipeline digest. |
| T27 | Registered native raw-unit mechanism matches sampling, clipping, noising, adjacency, and accounting units | 1 | `DIRECT` | `ALLOWED` | `EVIDENCE_VALIDATED` or conditional | State the native raw-unit mechanism, not the old window mechanism. |
| T28 | Mapping digest is missing or differs from the privacy report | — | `UNDERSPECIFIED` | `BLOCKED_INVALID` | `DIAGNOSTIC_ONLY` | Regenerate the mapping/report pair. |
| T29 | CSV hash matches but transform, split, code, or support parameters are not bound | — | `UNDERSPECIFIED` | `BLOCKED_UNVERIFIED` | `DIAGNOSTIC_ONLY` | Require a pipeline digest. |
| T30 | Sample rate or step count changes under raw adjacency and no accountant covers the change | — | `UNDERSPECIFIED` | `BLOCKED_INVALID` | `DIAGNOSTIC_ONLY` | Fix a public schedule/population or attach a matching analysis. |
| T31 | Model selection uses private data but is not public, composed, or separately accounted | — | `UNDERSPECIFIED` | `BLOCKED_INVALID` | `DIAGNOSTIC_ONLY` | Narrow the claim to the covered run or account for selection. |
| T32 | Accountant contract passes but DP noise uses a research PRNG | K | According to K | `RESEARCH_ONLY` | `DIAGNOSTIC_ONLY` | Report accountant diagnostics; do not emit evidence-validated wording for the concrete run. |
| T33 | Public deterministic DP-noise seed, or missing/unknown randomness evidence | — | `UNDERSPECIFIED` | `BLOCKED_INVALID` for public deterministic; otherwise `BLOCKED_UNVERIFIED` | `DIAGNOSTIC_ONLY` | Rerun with a registered private CSPRNG mechanism whose exact implementation and runtime evidence are bound. |
| T34 | Provenance metadata is classified `private_internal` | K | Underlying path retained | According to underlying validation | According to underlying validation | Emit only the redacted public certificate; omit scenario, digests, counts, and multiplicities. |
| T35 | Base accountant identifier has no registered recomputation checker | — | `UNDERSPECIFIED` | `BLOCKED_UNVERIFIED` | `DIAGNOSTIC_ONLY` | Retain the accountant output as an attachment but authorize no privacy sentence. |
| T36 | Registered accountant version or recomputed epsilon differs from the report | — | `UNDERSPECIFIED` | `BLOCKED_INVALID` | `DIAGNOSTIC_ONLY` | Regenerate with the registered version and exact accountant inputs. |
| T37 | Mapping `generated_unit`, manifest generated-record unit, and accounting unit differ | — | `UNDERSPECIFIED` | `BLOCKED_INVALID` | `DIAGNOSTIC_ONLY` | Export the actual generated records; do not relabel a window mapping as a native raw-unit mechanism. |

## Precedence Rules

When several rows apply, use the most restrictive result:

```text
BLOCKED_INVALID
  > BLOCKED_UNVERIFIED
  > BLOCKED_VACUOUS
  > RESEARCH_ONLY
  > ALLOWED
```

An external attachment cannot override an unrelated failure. For example, a
valid event conversion cannot repair a sampler/accountant mismatch.

## Required Golden Cases

The implementation test suite MUST contain at least:

1. T01: fully matched window-level direct case;
2. T06: exact-metric event direct case;
3. T07: replacement-to-add/remove factor-two case;
4. T08: nonvacuous event conversion;
5. T10: native owner direct case, if retained in the paper;
6. T12: nonvacuous owner group conversion;
7. T13: vacuous owner group conversion;
8. T25: plausible but unverified external attachment;
9. T28: stale mapping hash;
10. T32: mathematically accounted but research-only RNG case.
11. T35--T36: unregistered and numerically mismatched base-accountant cases.
12. T37: generated-record/accounting-unit mismatch.

## Current-Paper Consequences

Under this truth table:

- current `safe_equal_units` rows do not survive merely because observed
  multiplicity is one;
- the WISDM non-overlap event example requires an explicit raw-adjacency and
  transformation-stability proof;
- the current sampler mismatch certificate is T03 and must be blocking;
- existing UCI/WISDM global-normalization examples are T16 until repaired;
- current arbitrary external attachments are T25;
- current owner-accounted examples are T10 only if their preprocessing,
  population, schedule, RNG, and owner mechanism all pass;
- current stochastic integer bounds are T23 unless backed by a registered
  theorem checker.
