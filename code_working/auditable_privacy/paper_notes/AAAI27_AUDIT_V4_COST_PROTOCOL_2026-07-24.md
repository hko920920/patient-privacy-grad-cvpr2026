# AAAI-27 Audit paper: frozen cost and integration protocol

**Status:** frozen before the portable timing trials  
**Date:** 2026-07-24 (Asia/Seoul)

## Objective

Measure the practical review cost and artifact burden of the registered WISDM
path without presenting the integration baseline as a throughput benchmark.

## Fixed sources

- Reuse the already recorded wall-clock times for all five confirmatory
  pipelines and all five independent raw-input verifiers from
  `reports/wisdm_v4_multirun_001/execution_manifest.json`.
- Use `confirmatory_run01` for portable verification timing.
- Execute the portable verifier exactly five times as five new Python
  processes, in declared order, with `--skip-input-rescan`.
- Include process startup and JSON-output writing in every timing.
- Do not discard a slow trial or add a warm-up after observing results.
- A nonzero return code or anything other than 35/35 is a protocol failure.

## Reported summaries

For complete pipeline, full raw-input verifier, and portable verifier times,
report every observation plus mean, sample standard deviation, minimum,
maximum, and median. These are machine-specific descriptive timings, not
complexity claims.

For artifact burden, report:

1. the complete per-run directory size and file count;
2. the exact eleven per-run JSON/CSV files consumed by the independent
   verifier;
3. the four shared schema/registry/source/checker files bound by that route;
4. the raw dataset size separately; and
5. the fact that plots, Markdown/CSV renderings, logs, and metrics are useful
   derivatives but are not part of the eleven-file verifier input.

The integration surface is described as evidence categories, not reduced to a
misleading single field count: requested claim; actual mapping; pipeline and
preprocessing manifest; mechanism/accountant declaration; runtime trace;
versioned registry/source/checker; and public/internal decision outputs.

