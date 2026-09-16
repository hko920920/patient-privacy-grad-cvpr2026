# Fail-Closed Audit Quickstart v2.0

Status: current P0 validator and P1 evidence implementation  
Date: 2026-07-24

## Install Core Dependencies

```powershell
python -m pip install -r requirements-core.txt
```

The registered base-accountant semantics are pinned to Opacus 1.6.0. The
dependency-light source-bound checker recomputes epsilon, and the WISDM
independent verifier cross-checks it with `dp-accounting==0.6.0`. Identifier,
version, formula, or source drift is blocking.

## Run the Regression Gate

```powershell
.\scripts\run_v2_regression.ps1
```

This validates the published JSON Schema and runs the truth-table mutation,
golden, numeric, redaction, and CLI tests. The coverage map is
`tests/TRUTH_TABLE_COVERAGE.md`.

For the complete portable review surface (regression, all frozen evidence
families, and byte-identical appendix-table regeneration), run:

```powershell
python scripts\verify_frozen_artifacts_v2.py
```

## Generate Mapping Diagnostics

```powershell
python scripts\unit_audit.py `
  --mapping <mapping.csv> `
  --batch-size <batch-size> `
  --epochs <epochs> `
  --claim-units window,event,user `
  --schedule-mode fixed_mapping `
  --output-dir <audit-dir>
```

`unit_audit.py` is diagnostic-only. Its `verdict` column is always
`diagnostic_only`. Observed kappa never authorizes wording.

A certifiable v2.0 mapping needs:

```text
scenario,window_id,generated_unit,start,end
```

plus `owner_id`, `owner_ids`, or `owners`. The selected scenario may contain
only one exact generated unit: `window`, `event`, or `owner`.

## Build a Certificate

```powershell
python scripts\build_unit_certificate.py `
  --audit-csv <audit-dir>\unit_audit.csv `
  --mapping-csv <mapping.csv> `
  --privacy-report-json <privacy-report-v2.json> `
  --scenario <scenario> `
  --claim-unit <window|event|owner> `
  --output-dir <certificate-dir>
```

The builder reopens the mapping, recomputes mapping statistics and digests,
validates the JSON Schema, recomputes the registered accountant epsilon, checks
runtime, exact rational sampling, source/checker, RNG/noise, normalization, and
execution-artifact bindings, and then evaluates the registered stability
method.

Exit status is zero only for `release_status=ALLOWED`. Blocked and
research-only decisions exit two and contain an empty `supported_statement`.
The unsuffixed outputs are public/redacted; `_internal` outputs retain detailed
issues.

## Reproduce the WISDM Gold Pipeline

```powershell
python scripts\wisdm_v2_gold_pipeline.py
python scripts\verify_wisdm_v2_gold.py
```

The reference run binds its public raw file and parser-defined raw domain,
complete mapping, public owner cap 600, exact `1/50` Bernoulli sampling,
discarded-first four-draw `SystemRandom` Gaussian construction, public
constant-step noisy-sum update, source/checker hashes, RDP recomputation,
released model, and three unit-specific certificates. Its RNG label is
`known_attack_hardened_secure`, a named-attack mitigation boundary rather than
a universal side-channel or faithful-execution claim. It is an integration
baseline, not a state-of-the-art utility claim.

## Regenerate Mapping-Only Evidence

```powershell
python scripts\sepsis2019_v2_evidence.py
python scripts\verify_sepsis2019_v2_evidence.py `
  --skip-input-rescan

python scripts\backblaze_v2_evidence.py
python scripts\verify_backblaze_v2_evidence.py `
  --skip-archive-rescan
```

These portable verifier commands use the bundled mappings or sufficient
statistics. Remove the skip flags to reconstruct Sepsis from all 5,000 bound
source files or to rescan all 27.8 million Backblaze Q1 2025 rows; the latter
takes several minutes. The Backblaze analysis deliberately distinguishes
windows over available ordered snapshots from windows over
calendar-contiguous days.

These two bundles are `DIAGNOSTIC_ONLY`. They contain no epsilon/delta and
cannot authorize a privacy statement because they bind no executed DP
mechanism or accountant report. They also report candidate event stability in
the generated add/remove metric: a fixed-slot raw replacement contributes the
factor of two that observed incidence alone omits.

## Run the Validator Baseline/Ablation

```powershell
python scripts\run_validator_ablation_v2.py
python scripts\verify_validator_ablation_v2.py
```

The frozen matrix contains 31 cases: six valid and 25 blocked, vacuous,
research-only, or unverified. It combines the original conformance mutations
with eight repair regressions found during this project's re-audit. Each case
links to the normative truth table or a repair invariant. It compares field
presence, JSON-Schema-only, multiplicity-only, hash-only, accountant-only, two
targeted ablations, and the full validator. These are authored conformance and
regression cases, not a prevalence sample.

## Historical Artifact Warning

All v1.x certificates, reports, PDFs, and ZIPs are stale. The historical
reproduction and packaging scripts are disabled by default. They must not be
used for AAAI-27 evidence.
