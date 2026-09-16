# AAAI-27 Audit Paper: 200-Case Conformance Expansion Protocol V1

Date frozen: 2026-07-28 KST  
Status: **DESIGN FROZEN; NOT IMPLEMENTED OR EXECUTED**  
Scope: Table 2 contract-level conformance/ablation suite only. WISDM, UCI HAR,
ExtraSensory, Sepsis, Backblaze, TensorFlow Privacy, the five fixed deletion
witnesses, the ten lineage incidents, Figure 10, and the main paper are not
changed by this protocol.

## 1. Why this expansion is needed

The retained suite has 31 authored deterministic cases: 6 allowed and 25
non-allow. It is logically adequate for fixed counterexamples, but the six-case
allowed denominator is visibly thin for an AAAI reviewer and the suite does not
systematically cross path type, mapping boundary, and evidence representation.
This protocol replaces an arbitrary target with two explicit products:

- **100 allowed cases** = 5 registered path strata x 5 boundary/scale levels x
  4 mapping-evidence contexts;
- **100 non-allow cases** = 25 frozen semantic mutation operators x 4 paired
  parent contexts.

The result is a deterministic coverage experiment, not an IID sample, an error
rate estimate, prevalence evidence, or an independently sourced corpus. More
cases are scientifically useful here only because every cell is assigned a
predeclared path/boundary/context or mutation/parent role.

## 2. Frozen baseline

No baseline file is modified in this stage.

| Artifact | SHA-256 |
|---|---|
| `scripts/run_validator_ablation_v2.py` | `adf44933ee7cfda60830d5aaa2c79f588ee3f0f6e78a106540deb7732d0b44a8` |
| `scripts/verify_validator_ablation_v2.py` | `83adf054faf556818e4e4d0c3157ac5a829810c56b95c29bff980a51155be5c7` |
| `scripts/privacy_claim_validator.py` | `24d75408f656fcf1d586884ae5229ab8422186450d53cf68fc86aefb6d309f1a` |
| `scripts/mapping_evidence.py` | `8acefe2d4b7c170f132ac6cbebcdf91f542d33493c88ca5bdaa4c55680ac07ca` |
| `tests/test_privacy_claim_validator.py` | `36acb525dbbe026cb45374f324e2ca2edf4bf002f18c922b33147876db7e9049` |
| retained 31-case `cases.json` | `ae3dc26d4c546dfe06cce8b309e2e936c8ec830af6713cada494af88ffc68655` |
| retained 31-case `predictions.csv` | `45198d46ba75e29edece9c0861dba95da52cf6f82b5287f0e4b79a089760dd60` |
| retained 31-case `aggregate_metrics.csv` | `e87af19a16fe551d242f51a3b38b758c08d8d9e1c37a2adabed5aaa39c22b422` |
| current main V19 TeX | `8d36993a1bc321bce239d4b2a3253c7993673124b7d25d52b6e485ebff1ade16` |

The retained 31-case result remains a legacy regression and is never deleted or
silently relabeled. It is not added to the new denominator; the new experiment
has exactly 200 newly materialized case records.

## 3. Allowed-case design: 5 x 5 x 4 = 100

Case identifiers are deterministic:
`P3_<stratum>_L<level>_C<context>`. Every case has an explicit coverage record
containing stratum, level, context, generated unit, requested unit, accountant
metric, candidate K, expected path, expected status, and parent-free provenance.

### 3.1 Five registered path strata

| Stratum | Generated/accounted unit | Requested unit | Expected path | Cases |
|---|---|---|---|---:|
| S1 `DIRECT_WINDOW` | window | window | `DIRECT` | 20 |
| S2 `DIRECT_EVENT` | event | event | `DIRECT` | 20 |
| S3 `DIRECT_OWNER` | owner | owner | `DIRECT` | 20 |
| S4 `CONVERT_EVENT` | window | event | `CONVERT` | 20 |
| S5 `GROUP_OWNER` | window | owner | `GROUP` | 20 |

This yields 20 window requests, 40 event requests, and 40 owner requests. It
covers all five positive route/unit combinations currently registered; it does
not claim a second accountant/mechanism family.

### 3.2 Five levels per stratum

- S1--S3 direct strata use selected-record counts `1, 2, 4, 8, 12`; their
  privacy distance remains `K=1` because generated and requested units match.
- S4 uses target event incidences `kappa_event = 1, 2, 3, 4, 6`, hence
  generated add/remove distances `K = 2, 4, 6, 8, 12`.
- S5 uses public owner caps `K = 2, 3, 5, 8, 13` under owner partitioning.

For the frozen base pair `epsilon=1.1053169171072421`, `delta=1e-6`, the
largest allowed owner level is intentionally near the vacuity boundary:

| K | converted delta |
|---:|---:|
| 12 | `0.285105318010` |
| 13 | `0.861070796835` |
| 14 | `2.600591060080` |

Thus every positive level through K=13 is nonvacuous, whereas K=14 provides a
checked boundary outside the positive grid. No oracle label is copied from FULL.

### 3.3 Four mapping-evidence contexts per level

| Context | Frozen construction purpose |
|---|---|
| C1 `CANONICAL` | selected `gold` rows only, canonical input order |
| C2 `IRREGULAR` | gapped/irregular half-open supports with the same declared target count or K |
| C3 `DECOY_SCENARIO` | the selected mapping plus non-selected scenario rows, separating full-file and selected-record digests |
| C4 `PERMUTED` | reversed/permuted selected-row file order with invariant selected semantics, again separating file bytes from canonical selected evidence |

The generator must construct exact target kappa/cap values rather than accept
them as user integers. All case input triples must be byte-distinct. C3/C4 may
intentionally share a canonical selected digest with their semantic parent, but
their full mapping-file digest and bound report must differ.

## 4. Non-allow design: 25 x 4 = 100 paired mutations

Every non-allow case is paired one-to-one with a distinct allowed parent. Every
allowed parent is used exactly once. The negative record stores
`parent_case_id`, `operator_id`, direct mutation paths, derived rebind paths,
required issue code(s), normative/legacy anchors, and expected status/path/
assurance. A negative may cause deterministic cascading diagnostics, but only
the declared direct premise may be mutated.

### 4.1 Frozen operator catalog

| ID | Direct semantic mutation | Required status | Required primary diagnostic | Applicability |
|---|---|---|---|---|
| O01 | runtime trace missing | `BLOCKED_UNVERIFIED` | `RUNTIME_TRACE_NOT_BOUND` | global |
| O02 | owner conversion made vacuous | `BLOCKED_VACUOUS` | path `GROUP`, no positive wording | owner-group |
| O03 | mapping bytes changed after binding | `BLOCKED_INVALID` | `ACTUAL_MAPPING_DIGEST_MISMATCH` | global |
| O04 | Boolean in integer `steps` | `BLOCKED_INVALID` | `REPORT_SCHEMA_INVALID` | global |
| O05 | executed sampler/accountant mismatch | `BLOCKED_INVALID` | `SAMPLER_ACCOUNTANT_MISMATCH` | global |
| O06 | runtime noise differs from report | `BLOCKED_INVALID` | `RUNTIME_NUMERIC_FIELD_MISMATCH` | global |
| O07 | reported epsilon differs from recomputation | `BLOCKED_INVALID` | `ACCOUNTANT_EPSILON_MISMATCH` | global |
| O08 | secure assurance with secure mode false | `BLOCKED_INVALID` | `RELEASE_RNG_SECURE_MODE_MISMATCH` | global |
| O09 | research PRNG on an otherwise valid route | `RESEARCH_ONLY` | `RNG_RESEARCH_ONLY` | global |
| O10 | private-global preprocessing unaccounted | `BLOCKED_INVALID` | `PRIVATE_PREPROCESSING_UNACCOUNTED` | global |
| O11 | incomplete influence support | `BLOCKED_UNVERIFIED` | `INFLUENCE_SUPPORT_MANIFEST_INCOMPLETE` | event-convert |
| O12 | private adaptive schedule | `BLOCKED_UNVERIFIED` | `SCHEDULE_MODE_UNREGISTERED` | event-convert |
| O13 | unregistered external conversion attachment | `BLOCKED_UNVERIFIED` | `EXTERNAL_CONVERSION_UNVERIFIED` | event-convert |
| O14 | private model selection not covered | `BLOCKED_INVALID` | `MODEL_SELECTION_NOT_COVERED` | global |
| O15 | accountant identifier also violates exact registered mechanism semantics | `BLOCKED_INVALID` | both `REGISTERED_MECHANISM_SEMANTICS_MISMATCH` and `ACCOUNTANT_CHECKER_UNREGISTERED` | global, explicitly coupled diagnostic |
| O16 | manifest generated unit differs from mapping/accounting | `BLOCKED_INVALID` | `ACTUAL_GENERATED_UNIT_MISMATCH` | global |
| O17 | event stability status unverified | `BLOCKED_UNVERIFIED` | `STABILITY_STATUS_UNVERIFIED` | event-convert |
| O18 | code digest differs from trusted registry | `BLOCKED_INVALID` | `PIPELINE_CODE_REGISTRY_DIGEST_MISMATCH` | global |
| O19 | RNG backend differs from registered semantics | `BLOCKED_INVALID` | `REGISTERED_MECHANISM_SEMANTICS_MISMATCH` | global |
| O20 | one-draw Gaussian differs from registered hardening | `BLOCKED_INVALID` | `REGISTERED_MECHANISM_SEMANTICS_MISMATCH` | global |
| O21 | data-dependent normalization differs from registered mechanism | `BLOCKED_INVALID` | `REGISTERED_MECHANISM_SEMANTICS_MISMATCH` | global |
| O22 | observed owner incidence substituted for an absent public cap | `BLOCKED_UNVERIFIED` | `OWNER_PUBLIC_CAP_UNVERIFIED` | owner-group |
| O23 | generic event replace-one omits the registered raw domain | `BLOCKED_INVALID` | `RAW_ADJACENCY_NOT_SUPPORTED` | event-convert |
| O24 | executed exact rational sample rate differs from report | `BLOCKED_INVALID` | `EXACT_SAMPLE_RATE_MISMATCH` | global |
| O25 | runtime artifact digest differs from manifest-bound release | `BLOCKED_INVALID` | `RUNTIME_ARTIFACT_DIGEST_MISMATCH` | global |

O15 is deliberately not mislabeled as a pure instance of truth-table T35. Under
the current exact mechanism registry, changing the accountant identifier also
violates the registered mechanism entry and therefore produces an invalid,
two-diagnostic result. A pure T35 case would require a separately registered
mechanism whose accountant lacks a checker; this protocol does not fabricate
such a registry entry.

Expected non-allow status counts are fixed by the operator catalog:
68 `BLOCKED_INVALID`, 24 `BLOCKED_UNVERIFIED`, 4 `BLOCKED_VACUOUS`, and
4 `RESEARCH_ONLY`.

### 4.2 Exact parent assignment

Cells inside each stratum are numbered 1--20 in level-major, context-minor
order: `(L1,C1)` through `(L5,C4)`.

- Event-only operators consume all S4 parents:
  - O11 -> S4 L1 C1--C4
  - O12 -> S4 L2 C1--C4
  - O13 -> S4 L3 C1--C4
  - O17 -> S4 L4 C1--C4
  - O23 -> S4 L5 C1--C4
- Owner-only operators consume eight S5 parents:
  - O22 -> S5 L1 C1--C4 (`K=2`)
  - O02 -> S5 L5 C1--C4 (`K=13`, then made vacuous)
- Twelve cross-path global operators are, in order,
  `O01,O03,O04,O05,O06,O07,O08,O09,O10,O14,O15,O25`.
  Operator i receives S1, S2, and S3 cell i plus the i-th remaining S5 cell
  from S5 L2--L4 in level-major/context-minor order.
- Six remaining global operators are, in order,
  `O16,O18,O19,O20,O21,O24`.
  Operator j receives S1, S2, and S3 cell `12+j`; its fourth parent is:
  - O16 -> S1 cell 19
  - O18 -> S1 cell 20
  - O19 -> S2 cell 19
  - O20 -> S2 cell 20
  - O21 -> S3 cell 19
  - O24 -> S3 cell 20

This schedule consumes every positive parent exactly once and gives the
negative suite the same parent-stratum distribution: 20 cases in each of the
five registered positive path strata.

## 5. Oracle and verifier independence requirements

The new implementation must be versioned (`run_validator_ablation_v3.py` and
`verify_validator_ablation_v3.py`); V2 files and retained outputs remain
unchanged.

The runner may invoke the production validator, but its expected labels come
only from this frozen protocol/operator catalog. The independent verifier must
not import `privacy_claim_validator`, `mapping_evidence`, test helpers, or the
runner. Using Python's standard library and reading the public JSON Schema is
allowed. It must independently:

1. reopen every mapping/report/audit artifact and check its SHA-256 binding;
2. parse half-open intervals and recompute selected record count,
   `kappa_event`, `kappa_owner`, generated unit, and selected-record digest;
3. recompute the positive K rule and group conversion from raw inputs;
4. recompute all prediction flags and aggregate fractions from CSV rows;
5. check each negative's parent and direct/derived mutation closure;
6. require each declared primary diagnostic, allowing only documented cascades;
7. verify all coverage counts, parent uniqueness, operator counts, and input
   triple uniqueness;
8. verify artifact-index hashes without trusting runner-produced booleans.

## 6. Frozen execution gates

Implementation is not allowed to delete, replace, or relabel a failing cell to
make FULL pass. If a code defect is exposed, the failed run is retained, the
validator fix is separately versioned, and the same frozen 200 cases are rerun.

Required structural gates:

- exactly 100 allowed and 100 non-allow cases;
- exactly 20 allowed parents in every S1--S5 stratum;
- exactly four cases for every O01--O25 operator;
- every allowed parent used by exactly one non-allow case;
- all case IDs and input triples unique;
- all expected-positive K values nonvacuous and independently recomputed;
- legacy 31-case suite and existing validator unit/regression tests still pass.

Required FULL outcome gates:

- unsafe-positive case fraction `0.00`;
- valid-exact case fraction `1.00`;
- exact-tuple case fraction `1.00`;
- every required diagnostic present;
- no forbidden positive wording on any non-allow case.

Required ablation gates:

- every B0--B4 projection must expose at least one unsafe positive;
- A1 must expose at least one runtime/artifact-dependent unsafe positive;
- A2 must expose at least one vacuity-dependent unsafe positive;
- family/context results and raw numerators are retained, even if fractions are
  less visually favorable than the 31-case results.

## 7. Manuscript reporting rule after execution

No main-paper number changes until all gates pass. If they pass, Table 2 will
use two-decimal **case fractions**, not accuracy claims:

- `Unsafe-positive fraction` (lower is better);
- `Valid-exact fraction` (higher is better);
- `Exact-tuple fraction` (higher is better).

The caption must still disclose the denominator and construction:

> Fixed designed suite with 100 allowed and 100 non-allow paired contract
> cases; fractions are deterministic coverage outcomes, not statistical
> estimates. B0--B4 are information projections and A1--A2 are FULL
> obligation ablations, not competitor systems.

Raw numerators/denominators, the complete 5 x 5 x 4 coverage matrix, the 25 x 4
operator-parent matrix, hashes, and independent-verifier results belong in the
supplement/code artifact. The presentation must not hide the authored,
non-independent nature of the cases.

## 8. Known residual limits after a successful 200-case run

Even a perfect 200-case result would not establish population error rates,
external independence, arbitrary-code DP, cross-accountant generality, or
execution attestation. It would specifically reduce the present small-
denominator and context-coverage risk for Table 2. External claim bundles and a
second mechanism family remain separate possible extensions rather than claims
silently attributed to this suite.

## 9. Stage boundary

The frozen parent-assignment schedule was independently enumerated before
closing this stage. It yields 25 operators, 100 negative cases, 100 unique
parents, no missing or duplicate parent, and exactly 20 parents from each
S1--S5 stratum. The status arithmetic is 68 + 24 + 4 + 4 = 100.

This file closes **Stage 1 only**. No generator, oracle, verifier, result,
supplement, table, TeX, PDF, or upload artifact has been changed. Stage 2 may
begin only after explicit author instruction.
