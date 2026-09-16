# Privacy-Claim Audit Contract v2.0

Status: normative repair contract for the AAAI-27 audit paper  
Frozen: 2026-07-24

## 1. Purpose and Assurance Boundary

This contract defines when a unit-aware audit may emit a positive privacy
statement for a windowed sequential-learning pipeline.

The audited release is modeled as

```text
raw dataset D
  -> transformation T
  -> generated dataset G = T(D)
  -> randomized accounted mechanism A
  -> released output Y = A(G)
```

The audit checks whether the evidence bundle supplied for `T` and `A` supports
the requested neighboring relation and claimed privacy unit. It does not prove
that a malicious party executed the supplied code or exported a complete
mapping. Positive wording is therefore conditional on the validated evidence
and the stated execution-binding status.

The contract uses `MUST`, `MUST NOT`, `SHOULD`, and `MAY` normatively.

## 2. Domains, Metrics, and Units

Let:

- `D` be a raw dataset;
- `T(D; rho)` be the generated multiset used by training, conditional on public
  or otherwise accounted transformation randomness `rho`;
- `A` be the mechanism whose privacy report is supplied;
- `d_raw^c` be the dataset metric for requested claim type `c`;
- `d_acc` be the input metric and adjacency used by the accountant.

The certificate MUST distinguish:

- `generated_unit`: the record supplied to the private optimizer;
- `accounting_unit`: the unit adjacent under the accountant;
- `claimed_unit`: the unit named in the public privacy statement;
- `raw_adjacency`: the requested raw neighboring relation;
- `accountant_adjacency`: the neighboring relation used by the accountant.

Allowed adjacency identifiers are:

```text
add_remove
replace_one
owner_add_remove
owner_replace_one
```

Free-text descriptions MAY accompany these identifiers but MUST NOT substitute
for them.

### 2.1 Metric compatibility

An add/remove accountant uses a symmetric insertion/deletion distance. A
replace-one accountant uses a change-one or Hamming-style distance. The metrics
are not interchangeable.

In particular:

- one insertion or deletion has add/remove distance one;
- one replacement has replace-one distance one;
- one replacement generally has add/remove distance two because it can be
  represented as one deletion and one insertion.

Consequently, a raw replacement that changes one generated record is not
same-number `DIRECT` under an add/remove accountant unless a separate theorem
or accountant establishes that relation.

## 3. Transformation Stability, Not Observed Incidence

For a requested claim type `c`, define the transformation stability bound

```text
K_c = sup d_acc(T(D; rho), T(D'; rho)),
```

where the supremum is over every raw-neighbor pair
`d_raw^c(D, D') <= 1` and every transformation branch or public random coin
covered by the declared schedule contract.

`K_c` is the quantity used by the privacy verdict.

The observed mapping statistics

```text
kappa_event_observed
kappa_owner_observed
```

are diagnostics. They equal a certified `K_c` only when the evidence proves
that:

1. the generated-record universe and record identities are stable under the
   raw adjacency;
2. changing the raw unit cannot shift, create, delete, relabel, reweight, or
   reselect additional generated records outside the exported support;
3. preprocessing does not introduce additional cross-record dependencies;
4. the selected population, sample-rate rule, and step schedule remain covered
   under the same raw adjacency;
5. every schedule branch required by the declared mode is covered.

The validator MUST NOT infer `K_c` from observed incidence alone.

## 4. Influence Support

For generated record `w`, its influence support `M(w)` contains every raw unit
whose change can affect any of the following:

- presence or absence of `w`;
- any input feature of `w`;
- the label, target, weight, or owner attribution of `w`;
- eligibility or selection of `w`;
- a preprocessing parameter used by `w`;
- a pairing, grouping, augmentation, or imputation decision affecting `w`.

An interval such as `[start, end)` is sufficient only when it covers all such
dependencies. It is not sufficient merely because those timestamps appear in
the final feature tensor.

For multi-owner data, `A(w)` MUST contain every owner whose protected data can
affect `w`. Ambiguous attribution MUST use a justified conservative superset or
the owner claim is `UNDERSPECIFIED`.

## 5. Preprocessing and Selection

Each preprocessing or selection operation MUST be classified as one of:

```text
public_fixed
record_local
owner_local
separately_dp
private_global
private_adaptive
```

### 5.1 Supported preprocessing

- `public_fixed` parameters are fixed independently of the private dataset.
- `record_local` dependencies are included in the record's influence support.
- `owner_local` dependencies are included in the owner stability bound.
- `separately_dp` preprocessing supplies a registered privacy report and an
  explicit release/composition graph. The validator MUST NOT guess whether the
  downstream mechanism is post-processing of that release or an additional
  access to the raw data.

### 5.2 Unsupported preprocessing

Unaccounted `private_global` or `private_adaptive` preprocessing blocks a
positive raw-unit claim. Examples include:

- global mean or variance fitted on private training data;
- private-data-dependent vocabulary or imputation statistics;
- label-balanced or loss-adaptive selection without a public rule or privacy
  analysis;
- a split, cap, or population definition that changes on neighboring data and
  is not covered by the stability bound.

The repair is to use public fixed parameters, local preprocessing, separately
DP preprocessing, or an explicit worst-case stability/accounting argument.

## 6. Schedule Modes

### 6.1 `fixed_mapping`

`fixed_mapping` is valid only when the mapping and selection policy are fixed
by public inputs or otherwise covered by the declared raw adjacency. A mapping
observed after private adaptive selection is not a valid fixed mapping.

The evidence MUST bind the selected mapping, split, transformation parameters,
and execution identity used by the privacy report.

### 6.2 `union`

`union` requires an executable or independently checkable construction covering
every generated record and influence edge that any allowed non-private branch
can create. Relabeling an observed mapping as `union` is invalid.

### 6.3 `stochastic_bound`

A user-supplied integer and Monte Carlo observations are not a proof.
`stochastic_bound` is diagnostic-only unless a registered attachment supplies:

- a theorem or verified bound;
- the randomness and schedule to which it applies;
- a bound uniform over raw-neighbor pairs;
- failure probability `beta`;
- the exact rule incorporating `beta` into the final privacy statement.

The built-in validator MUST NOT create an unconditional DP statement from an
unverified stochastic bound.

### 6.4 `adaptive_private`

An observed adaptive schedule cannot certify a raw-unit claim. The claim remains
`UNDERSPECIFIED` unless a deterministic support covers every branch or a
registered privacy analysis covers the adaptive construction.

## 7. Accountant and Runtime Contract

A positive statement requires the following fields and consistency checks:

```text
mechanism_id
mechanism_version
accountant_id
accountant_version
accounting_unit
accountant_adjacency
sampling_unit
sampler_law
accountant_sampler_law
sample_rate
sample_rate_numerator
sample_rate_denominator
sampling_implementation
steps
noise_multiplier
secure_mode
secure_rng_backend
noise_generation
noise_hardening
clipping_unit
clipping_norm
noising_unit
gradient_aggregation
update_normalization
optimizer_step_size
privacy_convention
epsilon
delta
delta_convention
population_definition
population_size_policy
model_selection_status
rng_assurance
noise_seed_policy
adapter_registry_entry_sha256
code_artifact_id
code_sha256
accountant_checker_artifact_id
accountant_checker_sha256
```

The validator MUST establish:

1. `epsilon` is finite and nonnegative;
2. `delta` is finite and strictly between zero and one;
3. the executed sampler law matches the accountant;
4. clipping, noising, sampling, and accounting units match the declared
   mechanism;
5. the rational sample-rate numerator/denominator, floating representation,
   and step count match the executed schedule;
6. the sampling implementation, RNG backend, Gaussian construction,
   aggregation, normalization, and public optimizer step match one exact
   registered mechanism;
7. any population-dependent denominator or schedule is absent from the release
   mechanism; `population_size_policy` is only `public_fixed` or
   `not_used_in_release_mechanism` on the built-in positive path;
8. the privacy report binds the selected mapping, raw domain, executable source,
   accountant checker, model, batch trace, and pipeline contract;
9. model selection is public, separately accounted, composed, or explicitly
   excluded from the release claim.

The generic `accountant_covered` population token is retained by the input
schema as a negative regression value but is not accepted by the built-in
positive path. Likewise, an owner grouping route derives its stability bound
from a public, precommitted and enforced contribution cap; the observed owner
maximum is only a compliance diagnostic and MUST NOT be substituted for `K`.

For an unconditional `ALLOWED` base-accountant result, `accountant_id` and
`accountant_version` MUST select a registered checker and that checker MUST
recompute or independently verify the reported privacy parameters. A
well-formed but unregistered accountant report is `BLOCKED_UNVERIFIED`; string
agreement alone is not sufficient.

A Poisson accountant paired with shuffled or fixed-size batches is a hard
validation failure unless a registered accountant explicitly supports that
sampler.

### 7.1 Runtime binding

Runtime binding is recorded separately from unit alignment:

```text
matched
missing
mismatched
not_applicable
```

`mismatched` is always `BLOCKED_INVALID`. `missing` and `not_applicable` are
`BLOCKED_UNVERIFIED` for a post-execution claim: neither may emit a copy-ready
positive statement about the concrete run. A hypothetical mechanism statement
that is explicitly detached from the run is outside this validator's
post-execution output contract.

### 7.2 Randomness assurance

The certificate distinguishes:

```text
known_attack_hardened_secure
research_prng
unknown
```

`known_attack_hardened_secure` means only that the exact registered private
CSPRNG/noise implementation mitigates the named known attack class. It is not
a universal side-channel, cryptographic-attestation, or faithful-execution
claim. `research_prng` and `unknown` may carry accountant diagnostics but do
not support copy-ready privacy wording for the concrete run. Experiment seeds
and DP-noise randomness MUST be recorded separately; public deterministic
noise seeds are blocking for a released-model DP claim.

For registered secure-path evidence, `secure_mode` is an executed mechanism
field, not prose metadata. `known_attack_hardened_secure` requires
`secure_mode=true` in both the mechanism contract and matched runtime trace;
`research_prng` records `secure_mode=false`.

The JSON Schema retains the legacy token `release_grade_secure` solely as a
negative/adversarial input. The semantic validator always rejects it with
`RNG_ASSURANCE_OVERCLAIM`; no positive certificate may emit that value.

`noise_seed_policy` uses one of:

```text
secure_private_unrecorded
secure_private_internal
public_deterministic
unknown
```

Only the first two are compatible with
`known_attack_hardened_secure`.

## 8. Provenance Contract

The pipeline digest MUST bind, at minimum:

- dataset and split identifiers;
- generated-record unit, identities, and complete influence support;
- owner attribution;
- transformation parameters and their public/private classification;
- contribution policy;
- schedule mode;
- sampler and accountant configuration;
- clipping/noising units;
- mechanism and code version;
- model-selection protocol.

A CSV content hash alone is not a pipeline digest.

The certifiable mapping export MUST also state the generated unit on every row.
The selected scenario may not mix units, and that unit MUST match sampling,
clipping, noising, and accounting units. This prevents a window mapping from
being relabeled as a native owner-level mechanism.

Hash agreement detects drift between supplied artifacts. It does not prove that
the exporter was complete or honest. The certificate MUST state this assurance
boundary.

## 9. External Attachments

External attachments have one of three assurance classes:

```text
registered_verified
external_unverified
diagnostic_only
```

Only `registered_verified` attachments may authorize positive copy-ready
wording. Such an attachment MUST:

- declare whether it is a same-mechanism conversion or native raw-unit
  mechanism;
- bind the pipeline digest, mapping digest, adjacency, sampler, steps, sample
  rate, clipping/noising unit, and contribution bound;
- validate all numeric domains;
- be recomputed or verified by a registered checker;
- identify the theorem, implementation, and version used.

An arbitrary JSON object or a reference string is `external_unverified` and
cannot override a built-in blocked verdict.

## 10. Group Conversion

Suppose the supplied mechanism is `(epsilon, delta)`-DP at accountant distance
one and the validated transformation stability is integer `K >= 1`. The
built-in black-box conversion is

```text
epsilon_K = K * epsilon
delta_K   = delta * sum(exp(i * epsilon), i=0,...,K-1).
```

The implementation MUST compute the delta term in the log domain, reject
non-finite inputs, and block public DP wording when `delta_K >= 1`.

If the raw change is a replacement and the accountant uses add/remove
adjacency, `K` is measured in add/remove distance and therefore normally
includes the factor of two for each changed generated record.

A tighter RDP, PLD, sampler-specific, or structured conversion is accepted only
through a `registered_verified` attachment.

## 11. Verdict and Release Decision

The certificate uses three separate fields:

```text
unit_path
release_status
assurance_status
```

### 11.1 `unit_path`

```text
DIRECT
CONVERT
GROUP
UNDERSPECIFIED
```

- `DIRECT`: the claimed raw metric maps to accountant distance at most one and
  all mechanism/accountant obligations match.
- `CONVERT`: an event or other single raw-unit claim uses a validated
  conversion with `K > 1`, or a metric mismatch makes same-number reuse
  invalid.
- `GROUP`: an owner, multi-record, or multi-attribution claim uses a validated
  group conversion.
- `UNDERSPECIFIED`: a required premise is missing, inconsistent, unverified, or
  unsupported.

Native owner accounting is `DIRECT` only when sampling, aggregation, clipping,
noising, adjacency, and accounting all operate at the owner unit.

### 11.2 `release_status`

```text
ALLOWED
BLOCKED_INVALID
BLOCKED_VACUOUS
BLOCKED_UNVERIFIED
RESEARCH_ONLY
```

The unit path alone never authorizes wording. For example, a mathematically
identified `GROUP` path with `delta_K >= 1` has
`release_status=BLOCKED_VACUOUS`.

### 11.3 `assurance_status`

```text
EVIDENCE_VALIDATED
EXTERNAL_UNVERIFIED
DIAGNOSTIC_ONLY
```

`EVIDENCE_VALIDATED` requires a matched, content-bound runtime trace. Missing
runtime evidence produces `BLOCKED_UNVERIFIED` and `DIAGNOSTIC_ONLY`, never an
allowed conditional concrete-run statement. `EXTERNAL_UNVERIFIED` is reserved
for an otherwise plausible but unregistered external derivation and is also
non-positive.

## 12. Decision Procedure

For each requested claim:

1. parse and strictly validate every required field;
2. validate public/private metadata and certificate-release policy;
3. validate mapping completeness evidence and the pipeline digest;
4. validate transformation stability `K` in the accountant's input metric;
5. validate sampler, schedule, clipping/noising, accountant, adjacency, and
   runtime consistency;
6. validate model selection and randomness assurance;
7. select `DIRECT`, built-in group conversion, or a registered external path;
8. compute the final privacy parameters rather than trusting supplied converted
   values;
9. apply vacuity and assurance checks;
10. emit allowed wording only when `release_status=ALLOWED`.

Any failure before step 7 defaults to `UNDERSPECIFIED` and a blocking release
status. There is no permissive fallback.

## 13. Soundness Statement

Let `T` be a transformation from raw datasets with metric `d_raw` to generated
datasets with accountant metric `d_acc`. Suppose the validated evidence proves
that `T` is `(1, K)`-stable: for every raw-neighbor pair,

```text
d_raw(D, D') <= 1
  implies
d_acc(T(D), T(D')) <= K.
```

Suppose `A` is `(epsilon, delta)`-DP for `d_acc` distance one under the validated
sampler, clipping/noising, schedule, adjacency, and release assumptions. Then
`A o T` supports:

- the same `(epsilon, delta)` statement when `K=1`;
- the black-box group conversion in Section 10 when `K>1`;
- a tighter statement only when a registered attachment proves it.

This statement is conditional on mapping completeness and execution evidence.
It is not malicious-execution attestation.

## 14. Public Certificate Policy

The internal manifest may contain exact counts, hashes, labels, seeds, and
diagnostics. The public certificate MUST be generated from an allowlist.

Exact metadata may be public only when:

- it comes from an explicitly public benchmark or public release;
- it has been separately released under an appropriate privacy mechanism; or
- the data controller explicitly classifies it as public.

Raw mapping hashes, exact private population sizes, label distributions,
membership-sensitive diagnostics, and private seeds remain internal by default.

## 15. Required Regression Properties

The implementation of this contract MUST include tests showing:

- invalid epsilon/delta never yields positive wording;
- missing or mismatched digests block stronger claims;
- sampler/accountant and schedule/accountant mismatches block validation;
- replace-one versus add/remove mismatch changes `K` or blocks `DIRECT`;
- global private preprocessing expands support or blocks the claim;
- fabricated external attachments remain unverified;
- observed adaptive mappings cannot certify raw-unit privacy;
- public certificate redaction excludes non-public fields;
- every valid golden case remains reproducible.

## 16. Source Basis

This contract follows:

- the transformation-stability and metric-chaining framework used by OpenDP;
- standard group privacy for approximate DP;
- the sampler/accountant distinction established for shuffled versus Poisson
  DP-SGD;
- Opacus documentation requiring Poisson-compatible accounting and separating
  experimental PRNG use from cryptographically strong release settings;
- established implementation-audit guidance that DP claims require explicit
  neighboring relations, randomness, testing, and end-to-end mechanism review.

The manuscript bibliography and related-work section must cite the corresponding
primary sources. This contract itself is the implementation specification, not
a new mathematical theorem.

Primary source links used to freeze this contract:

- OpenDP transformation stability and metric chaining:
  <https://docs.opendp.org/en/stable/theory/a-framework-to-understand-dp.html>
- OpenDP transformation and preprocessing guidance:
  <https://docs.opendp.org/en/stable/api/user-guide/transformations/index.html>
- Dwork and Roth, *The Algorithmic Foundations of Differential Privacy*:
  <https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf>
- Chua et al., *How Private are DP-SGD Implementations?*:
  <https://proceedings.mlr.press/v235/chua24a.html>
- Opacus `PrivacyEngine` documentation:
  <https://opacus.ai/api/privacy_engine.html>
- Kifer et al., *Guidelines for Implementing and Auditing Differentially
  Private Systems*:
  <https://arxiv.org/abs/2002.04049>
