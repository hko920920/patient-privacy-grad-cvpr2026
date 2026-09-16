# Validator Controlled-Probe and Scope-Comparison Matrix v2.0

Status: execution specification  
Purpose: isolate which obligations are necessary for the full audit. B0--B4
and A1--A2 are deliberately incomplete controlled probes, not competing
privacy systems or overall library rankings.

## Primary evaluation question

For the same candidate evidence bundle and requested privacy unit, does a
method emit exactly the strongest wording justified by the frozen v2.0
contract, without an unsafe positive, wrong adjacency, wrong epsilon/delta,
vacuous conversion, or unsupported external attachment?

The primary outcome is the complete tuple:

```text
(release_status, unit_path, reported_epsilon, reported_delta, issue class)
```

Binary allow/block accuracy is insufficient because a method can allow a valid
claim while printing the wrong unit path or reusing the wrong numerical
privacy parameters.

## Controlled probes and external scope comparison

| ID | Method | Information checked | Expected blind spot |
| --- | --- | --- | --- |
| B0 | Prose checklist / field-presence probe | Required field names appear in a human-readable record | Cannot establish types, hashes, runtime equality, accountant math, stability, or conversion vacuity |
| B1 | JSON Schema only | Structure, enums, types, required fields | Accepts semantically inconsistent but well-typed evidence |
| B2 | Multiplicity only | Observed event/owner kappa from an exported mapping | Confuses incidence with stability; misses replace-one/add-remove factor; trusts incomplete or stale exports |
| B3 | Hash binding only | Mapping/report/runtime digests agree | A consistently wrong mechanism or adjacency still passes |
| B4 | Accountant only | Recomputes base epsilon for sampler/noise/steps/delta | Establishes only the accounting-unit guarantee, not raw-unit wording |
| B5 | TensorFlow Privacy statement | Official `compute_dp_sgd_privacy_statement` over scalar-equivalent population, batch, epochs, noise, delta, and optional group bound | Correctly generates a conditional example/user statement, but does not reopen an external mapping or bind runtime, preprocessing, schedule, identity, or evidence hashes |
| A1 | Full validator without runtime binding | Schema, mapping, stability, accountant; runtime omitted | Quantifies the value of execution binding |
| A2 | Full validator without vacuity gate | Full conversion but allows delta at least one | Demonstrates why numerical group output is not automatically reportable |
| FULL | Authoritative v2.0 validator | All registered checks and fail-closed wording | Reference implementation |

`B5` is evaluated in a companion scope-comparison bundle rather than forced
into the exact-tuple table. Its official function performs fresh target-delta
user accounting and prints values to three decimals, whereas FULL validates a
fixed external report and its fixed-base conversion. Treating those different
numerical contracts as the same tuple would be an unfair comparison. B5 uses the
original 23-case scalar-compatible subset; it is not assigned predictions for
the eight later mechanism/evidence repair regressions.

## Frozen case families

At minimum, the executable comparison must include:

1. valid window DIRECT;
2. valid event conversion under replace-one to generated add/remove;
3. valid nonvacuous owner grouping;
4. vacuous owner grouping;
5. observed event kappa one with adjacency factor two;
6. mapping bytes changed after report creation;
7. selected-scenario or row-count mismatch;
8. generated-unit mismatch;
9. sampler/accountant mismatch;
10. runtime noise or clipping mismatch;
11. accountant version or epsilon drift;
12. secure-path assurance with `secure_mode=false`, plus rejection of the
    universal `release_grade_secure` overclaim token;
13. research PRNG;
14. private-global preprocessing;
15. label-balanced/private-data-dependent selection;
16. private-adaptive schedule;
17. uncovered private model selection;
18. fabricated external conversion;
19. unknown accountant/mechanism adapter;
20. forged executable-source identity;
21. false RNG backend semantics;
22. one-draw Gaussian substituted for the registered hardened construction;
23. data-dependent update normalization;
24. observed owner maximum substituted for a public stability cap;
25. generic event `replace_one` substituted for the fixed owner-slot payload
    domain;
26. rational sample-rate mismatch;
27. forged runtime artifact binding;
28. Backblaze ordered-snapshot versus calendar-contiguous semantics.

Each case has a manually frozen expected tuple tied to a truth-table row or an
explicit P1 evidence invariant. The full validator does not create its own
ground-truth label during evaluation.

Items 1--27 form the executable 31-case validator matrix (six positive and 25
non-allow variants, including repeated adapters and native-unit routes).
Item 28 is evaluated separately by the secondary-evidence math regression and
two independent archive scans because it has no executed
mechanism/accountant report. Treating that mapping-only semantic check as a
validator privacy case would itself violate the audit boundary.

## Metrics

Report:

- unsafe-positive rate on cases whose release status must be blocked;
- exact tuple accuracy;
- wrong-same-number rate, where base window epsilon/delta are printed for an
  event/owner metric requiring conversion;
- vacuous-positive rate;
- stale-evidence detection rate;
- issue-family localization accuracy;
- valid-claim retention on the positive golden cases;
- median and maximum validation wall time;
- peak memory separately for ordinary mappings and the Backblaze analytic
  diagnostic.

The headline metric is unsafe-positive rate, followed by exact tuple accuracy.
The paper must not use the number of emitted warnings as a substitute.

## Required output

The executable experiment will write:

```text
reports/validator_ablation_hardened_002/
  cases.json
  predictions.csv
  aggregate_metrics.csv
  confusion_by_failure_family.csv
  summary.md
  independent_verification.json
```

Every output must bind the case file, implementation hashes, dependency
versions, and the authoritative contract/truth-table hashes.

The B5 companion writes:

```text
reports/tfprivacy_statement_baseline_v2_001/
  requests.json
  official_statements.json
  predictions.csv
  aggregate_metrics.json
  summary.md
  artifact_index.json
  independent_verification.json
```

The worker executes the official TensorFlow Privacy 0.9.0 analysis module,
SHA-256
`8e713fc815c9005b9933df08acb9efc9df32dd4ea17bfb54348c6b40386cba25`,
with `dp-accounting==0.4.3` in an isolated environment. It does not install or
exercise the unrelated TensorFlow training stack.

## B5 scope-comparison result

On the frozen matrix:

- valid reference cases receiving a finite conditional statement: `7/7`;
- non-allow reference bundles still receiving a finite conditional statement:
  `15/16`;
- non-allow bundles detected by scalar-domain rejection or an infinite bound:
  `1/16` (the Boolean `steps` case);
- independent checks: `78/78`.

Call the `15/16` result an **unvalidated-positive scope gap**, not a TensorFlow
Privacy error rate. The function is designed to trust its scalar arguments.
For the vacuity case, it performs fresh target-delta accounting and obtains a
finite alternative; this is not the fixed-base conversion rejected by FULL
and would require its own registered, execution-bound adapter before the
validator could accept it.

## Acceptance gate

The ablation is complete only if:

1. FULL has zero unsafe positives;
2. FULL matches every frozen expected tuple;
3. at least one controlled probe fails each central failure family: mapping,
   stability/adjacency, runtime/mechanism, accountant, private dependency, and
   conversion vacuity;
4. all valid golden cases remain allowed;
5. an independent verifier recomputes aggregate metrics from `cases.json` and
   `predictions.csv`.
6. the official B5 worker is module-hash pinned, reproduces every statement,
   and reports scope differences without describing them as TensorFlow bugs.
