# Graduation Thesis and CVPR Context Bundle

This repository is a curated context bundle for the graduation thesis work and
its CVPR-oriented extension on patient privacy, patient-level exposure
assessment, and DP protection comparisons.

It is prepared for code and research-context review. It is not a full raw-data
release and is not expected to reproduce every experiment from a clean machine.

## What Is Included

- `project_context/`
  - Current project state, work log, workspace README, and local project
    instructions that explain the active research framing.
- `thesis/`
  - Thesis LaTeX source, chapter files, bibliography, figures, tables, and the
    compiled thesis PDF.
- `thesis_proposal/`
  - Latest proposal PDFs that provide thesis-level framing before the CVPR
    extension.
- `cvpr_context/`
  - CVPR topic exploration notes, stage-2 research framework, closest-prior
    reviews, design-gap analyses, patient audit rationale, track-1/track-2
    redesign notes, and selected generated HTML/JSON/CSV context artifacts.
- `code_working/`
  - Working code and protocol documents for the X-ray privacy pipeline,
    PP-Mark-related work, DP protocol/training utilities, patient audit tools,
    and the frozen residual-head patient-DP comparison.
- `code_originals_manifest/`
  - Source manifests and selected lightweight original/source-reference files.

## What Is Excluded

The bundle intentionally excludes raw or heavy experiment artifacts:

- patient/image datasets and local data caches
- `_data`, `_reports`, `_diagnostics`, `_restricted_generation`
- model checkpoints, NumPy arrays, generated run outputs, and local cache files
- downloaded paper PDFs, paper text dumps, browser profiles, and cloned external
  source dumps used only for literature inspection
- Python bytecode, virtual environments, build outputs, and dependency folders

## Research Scope

The thesis context centers on valid patient-privacy and reuse decisions,
including evidence binding and PP-Mark-related evaluation. The CVPR extension
centers on patient exposure assessment when auditor-held images differ from
training images, plus a later track-1 direction on frozen residual-head
patient-DP protection comparisons.

The latest track-1 code and documents connect:

1. frozen residual-head capacity baselines,
2. public-only calibration,
3. patient-level DP mechanism comparison, and
4. audit/verifier scripts for provenance and arithmetic checks.

The research notes preserve negative, inconclusive, and redesign results as
part of the context. They should not be read as claims of validated novelty,
medical efficacy, or paper acceptance.
