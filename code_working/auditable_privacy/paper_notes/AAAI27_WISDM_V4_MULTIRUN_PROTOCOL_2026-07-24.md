# AAAI-27 Audit paper: frozen WISDM multi-run protocol

**Protocol status:** frozen before the confirmatory executions  
**Freeze date:** 2026-07-24 (Asia/Seoul)  
**Purpose:** test whether the reported decision paths and utility description
are stable across fresh randomness in the one fully registered end-to-end
WISDM mechanism route.

## 1. Scope and non-scope

This is a confirmatory stochastic-robustness check of one registered
mechanism, dataset, split, preprocessing rule, and privacy configuration. It
is not a hyperparameter search, a state-of-the-art utility study, evidence of
cross-dataset generality, or evidence that a submitted bundle is faithful to
an unobserved execution.

The earlier directory
`reports/wisdm_v4_multirun_001/noise2_run01` was produced as a feasibility
pilot before this protocol was frozen. It is excluded from every confirmatory
aggregate and must remain visibly labelled as a pilot.

## 2. Fixed confirmatory design

Execute exactly five runs, named:

1. `confirmatory_run01`
2. `confirmatory_run02`
3. `confirmatory_run03`
4. `confirmatory_run04`
5. `confirmatory_run05`

Every run uses a fresh invocation of
`scripts/wisdm_v2_gold_pipeline.py`. The implementation draws sampling and
Gaussian-noise randomness from `secrets.SystemRandom`; no fixed seed is
introduced.

The input and all public configuration values are fixed:

```text
data path: data/wisdm/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt
sample-rate numerator / denominator: 1 / 50
steps: 100
noise multiplier: 2
clipping norm: 1
optimizer step size: 0.006666666666666667
delta: 1e-6
```

No run may be selected, discarded, or replaced based on accuracy, macro-F1,
privacy output, decision status, or any other substantive result.

## 3. Primary acceptance criteria

All five named runs must satisfy all of the following:

- the pipeline exits successfully;
- the independent verifier reopens the raw WISDM input and passes 38/38
  checks;
- dataset, selected mapping, implementation code, mechanism registry
  binding, mechanism configuration, accountant configuration, split, and
  record counts are identical across runs;
- the public decisions are identical and equal to:
  - generated window: `ALLOWED`, `DIRECT`, `K=1`;
  - raw event: `ALLOWED`, `CONVERT`, `K=4`;
  - owner: `BLOCKED_VACUOUS`, `GROUP`, `K=600`;
- the deterministic privacy values are identical across runs;
- all gold assertions pass; and
- all five released-model digests are distinct, as a duplication guard.

A primary criterion failure is reported as a failure; it is not hidden by an
average.

## 4. Secondary descriptive outcomes

For test accuracy, test macro-F1, and train accuracy, report every run plus
the arithmetic mean, sample standard deviation, minimum, and maximum. Report
pipeline and independent-verifier wall-clock time descriptively. Do not run
a significance test, rank runs, choose a representative run by utility, or
claim that five executions establish population-level generality.

## 5. Stopping and technical-failure rule

The runner attempts all five executions in the declared order and stops only
if a command returns a nonzero exit code. It preserves that run's directory,
stdout, stderr, exit code, and timing. A failed or interrupted directory is
not overwritten. Any retry requires a separately dated protocol amendment
that names the technical reason and retains the original failed record.

The execution manifest records this protocol's SHA-256 before the first
confirmatory run, the exact commands, timings, return codes, and completion
state. This file is the pre-result analysis rule for the confirmatory
aggregate.

