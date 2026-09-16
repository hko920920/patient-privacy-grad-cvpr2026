# Storage Policy for the Integrated Dissertation Work

- Fixed: 2026-09-02
- Current `code_originals/`: 3.694 GiB
- Current `code_working/`: 3.678 GiB
- Current C-drive free space at intake: 63.61 GiB (13.4%)

## Review triggers, not absolute caps

The numbers below are default review points. They do not prohibit a scientifically necessary
dataset, model, experiment, or proof campaign. When a review point is reached, first document why
the additional material is needed, its expected peak and retained size, and whether another drive
or a compact retention strategy is preferable. Proceed after that decision.

1. Review the storage plan before new downloads, training, generation, or SP1 batch proving when
   C-drive free space approaches 50 GiB.
2. Review the layout when `code_working/` approaches 15 GiB during the feasibility pilot.
3. Review the retention plan when newly generated pilot outputs approach 10 GiB in total.
4. Do not copy PP-Mark `outputs/`, Rust/SP1 `target/`, virtual environments, model caches, or
   private `secrets/` into `code_originals/` or duplicate them between projects.
5. Before downloading a dataset or model larger than 5 GiB, record its license, subject lineage,
   experiment role, expected peak size, and expected retained size. A justified item may be larger.
6. Begin with a 2D medical-image subset by default. A larger 2D or 3D corpus remains possible when
   the scientific need and storage/compute plan justify it.

## Storage layout

- `code_originals/` remains immutable.
- New integration source will live in one separate repository under `code_working/`; do not
  initialize `code_working/` itself as a Git repository because it contains nested external Git
  repositories.
- Raw datasets, downloaded weights, run outputs, checkpoints, proofs, and receipts must remain
  untracked.
- Preserve configs, seeds, manifests, metrics, hashes, and a small representative artifact set.
  Preserve large raw populations only when a declared experiment requires them.
- Keep no more than the baseline checkpoint, the selected final checkpoint, and an explicitly
  justified failure checkpoint for a pilot.

## Existing duplication

The current 7.372 GiB intake is bounded. Most of it is the intentionally isolated copy of PP-Mark's
11 external comparison implementations and calibration data. No further duplicate copy is allowed.
If space pressure develops, the working copies of `ppmark/external/` and `ppmark/datasets/` can be
restaged from `code_originals/`, recovering about 3.57 GiB. Do not remove them without a fresh hash
check and an explicit cleanup decision.

Run `tools/check_storage.ps1` before dataset acquisition and every training or proving campaign.
Its result is advisory: `REVIEW_REQUIRED` means that a storage decision is needed, not that the
research direction is forbidden.
