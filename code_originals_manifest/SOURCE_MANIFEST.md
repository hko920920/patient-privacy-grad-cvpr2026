# Research Code Source Manifest

- Recorded: 2026-09-02 (Asia/Seoul)
- Status: immutable source baseline; do not edit files under `code_originals/`
- Working tree: use the adjacent `../code_working/` directory for all future changes
- Scope: source-oriented preservation for the dissertation rebuild, not a copy of every
  historical raw output or build cache

## 1. Canonical contents

| Path | Role | Files | Bytes |
|---|---|---:|---:|
| `_archives/` | byte-exact upstream/release ZIP files | 3 | 22,026,240 |
| `ppmark_submission/` | clean extraction of the anonymous PP-Mark supplementary ZIP | 55 | 920,013 |
| `ppmark_dev/` | latest PP-Mark v0.41 source-oriented working snapshot | 2,850 | 3,757,614,029 |
| `audit_v17/` | clean extraction of the Auditable Privacy V17 code/data release | 2,015 | 175,600,929 |
| `unitdp_v10/` | clean, prefix-stripped extraction of the UnitDP compiler V10 release | 410 | 7,772,036 |
| `paper_sources/auditable_privacy_v24_v15/` | active Audit main V24 and supplement V15 sources and dependencies | 32 | 2,262,533 |

The active 104-page dissertation currently imports the Auditable Privacy chapter and the
PP-Mark chapter. UnitDP is retained as a small reference surface because it may become useful
for the new privacy-unit buildup, but it is not an active chapter or the initial modification
target.

## 2. Source authority and versions

### PP-Mark

- Development source:
  `C:\Users\SOGANG\Documents\PP-Mark-v0.4\PP-Mark`
- Branch: `pp_mark_v0.41`
- Last committed ancestor: `b5572d5c250542d8938456e2ec73d3f1f8e438a9`
- State: many scientifically relevant tracked modifications and untracked experiment files;
  therefore the old Git commit alone is not the current implementation.
- Exact submission artifact:
  `anonymous_supplementary_code.zip`, created 2026-08-02.
- Current implementation is SP1-first. RISC0 and Halo2 paths are retained as legacy/reference
  code.

The `ppmark_dev/` snapshot includes root configs and environment manifests, `src/`, `scripts/`,
`sp1/`, `tests/`, `docs/`, `tables/`, `paper/`, `assets/`, `datasets/`, the legacy `risc0/` and
`halo2_prover/` source, and all 11 local external watermark/attack implementations. All 2,850
copied paths were found at the source location. Every research-code/data byte matched the source.
The only 11 later differences were repository-local `external/*/.git/config` files to which the
editor automatically added `vscode-merge-base`; these are Git metadata, not research code.

The following large or unsafe material was intentionally not copied into this dissertation
snapshot:

- `outputs/` (about 70.9 GiB): historical images, traces, and receipts;
- top-level `target/` (about 15.7 GiB) and nested build caches: rebuildable Rust/SP1 artifacts;
- `.envs/` and other environment caches: machine-specific installations;
- `secrets/`: private key material must not be duplicated into the thesis tree;
- top-level `artifacts/` (about 0.64 GiB): predominantly legacy RISC0 proof/runtime artifacts;
- raw logs, temporary outputs, the 4 GiB CUDA installer, and duplicate historical packages.

The compact A100/SP1 evidence archive needed for the current runtime/tamper claims is retained at
`ppmark_dev/docs/ppmark_a100_campaign_evidence_2026-08-02.zip`. The original 92 GiB project remains
untouched at its source path if an omitted raw artifact is later needed.

External comparison repositories retained in `ppmark_dev/external/`:

| Repository | Commit | Dirty entries at intake |
|---|---|---:|
| Gaussian-Shading | `09c678fadc7545acf7be12647ddf2a5e66f6a9dc` | 55 |
| InvisMark | `8d5ce55705ada1c2daf642c97f7ff3ae6dbd1825` | 2 |
| PRC-Watermark | `3f372c1d9cafcd0732331c0d00f0430ee822a389` | 8 |
| RingID | `45631a59aecd7d63ccdb640aaaf3e616fdb89fb9` | 3 |
| semantic-forgery | `ca68950cd7b53b8438a49580f134b2a0db28d778` | 8 |
| stable_signature | `48261686883ba86f533836ad47c35040afcfe37b` | 6 |
| StegaStamp | `c984446048b826587ca1875027b0a1dc1885fb30` | 3 |
| tree-ring-watermark | `3015283d9cf82e90b628f02ad2121bd37408ca9a` | 5 |
| TrustMark | `d38aee3da479860411598a9eed56348e36c9171f` | 19 |
| WatermarkAttacker | `2637dd2c3b84a3037dd3940b401090e8bcebe1f6` | 0 |
| WIND | `e923df7f74c7e41745c071d831e4b76d04665a5a` | 5 |

### Auditable Privacy

- Source release:
  `C:\Users\SOGANG\Documents\DP-SGD\dist\audit_code_and_data_supplement_aaai27_v17.zip`
- Active paper reference: main V24, supplement V15.
- This is the directly relevant privacy-claim validator for the active dissertation.
- The public package intentionally contains no raw WISDM, UCI HAR, PhysioNet, Backblaze, or
  ExtraSensory dataset.

### UnitDP compiler reference

- Source release:
  `C:\Users\SOGANG\Documents\DP-SGD-algorithm\dist\compiler_code_and_data_supplement_aaai27_v10.zip`
- Active paper bound by the package: main V43, supplement V41, checklist V40.
- The ZIP was extracted with its top-level prefix stripped. Retaining the long archive prefix in
  the already long thesis path caused the first extraction to omit three preprocessing JSON files.
  The canonical `unitdp_v10/` tree contains all 410 files and passes the official verifier.

## 3. Verification performed

- Auditable Privacy package manifest: PASS; 2,014 indexed files, 600 raw case inputs, and
  20,909/20,909 independent checks.
- Auditable Privacy official test scope (`tests/`): 91 passed.
- UnitDP V10 public-supplement verifier: PASS; package file count 409 plus the manifest.
- UnitDP portable tests: 141 passed, 13 expected optional-backend skips, and 180 subtests passed.
- PP-Mark development tests: 16 passed and 5 skipped. Four skips require the declared
  `poseidon-py` dependency, which is not installed in the current Python environment; one CUDA
  test requires optional CuPy. These are environment gaps, not observed test failures.
- The first broad Audit pytest command also collected third-party ExtraSensory's training helper
  named `test` as if it were a pytest test. The correct package scope is `tests/`, which passed
  91/91.

## 4. What is present and what is still missing

Present and reusable:

1. Auditable Privacy's fail-closed privacy-unit claim validator, schemas, evidence records, and
   model-hash fields.
2. PP-Mark's deterministic watermarking, Detect/Attest split, exact-image/context binding, SP1
   host/guest code, attacks, quality/runtime scripts, and comparison baselines.
3. UnitDP's registered owner-sampled routes as a reference if the redesign later needs a native
   contributor/owner mechanism.

Not yet implemented in these codebases:

1. A DP-SGD medical-image diffusion training or fine-tuning pipeline with patient identifiers.
2. The planned image-level versus patient-level DP experiment and membership-inference,
   memorization/extraction, or reconstruction evaluation for that generator.
3. A signed model-bound receipt that cryptographically binds an Auditable Privacy decision to the
   exact DP generator checkpoint. Existing model SHA-256 fields are useful inputs but are not that
   receipt.
4. A PP-Mark payload/statement adapter that binds a released synthetic medical image to that
   generator and privacy receipt.
5. A single evidence-package builder and third-party fail-closed release/intake policy.
6. Medical-image compatibility and utility validation; current PP-Mark assets target SD 2.1/SDXL,
   not a selected DP medical generator.

Therefore the source recovery is sufficient to reuse both prior contributions, but it is not yet
the integrated dissertation system. Future implementation must begin in `../code_working/` and
must not silently present the six missing items above as completed.

