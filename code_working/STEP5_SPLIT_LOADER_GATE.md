# Step 5 patient split, real-image loader, and PP-Mark adapter gate

- Date: 2026-09-02 (Asia/Seoul)
- Status: **PASS_FOR_REAL_DATA_ACQUISITION_AND_PILOT_PREPARATION**
- Scope: manifests and interface compatibility only; no generator training or privacy claim

Subsequent role-label note (2026-09-09): this gate's K2/K5 unit-sensitivity and K10-main wording is
historical. Later runtime and feasibility contracts retain the same frozen counts, splits, and cap
views but refine the roles to K2 accounting sensitivity, K5 one-seed feasibility, and K10
confirmatory main plus contribution stress. Current authority: `../CURRENT_STATUS.md`.

## Objective and interpretation lock

The dissertation is not primarily a diffusion-quality paper. Its core experiment compares
image-level DP with native patient-level DP under a controlled per-patient contribution and then
connects the exact trained checkpoint to an audited claim, model-bound receipt, and PP-Mark output
provenance. Generation quality remains a mandatory utility gate: a private but unusable generator
cannot support the lifecycle claim.

K10 is the controlled main condition because `K` is an experimental privacy/contribution variable,
not because the 32,701-image source population is too small. K2 and K5 are unit-sensitivity
conditions. An uncapped run is a separately labelled realism/scale extension and cannot replace a
capped comparator after results are observed.

## Frozen split and cap algorithm

- Lineage authority: fixed `ground_truth_v2.csv`; the mutable API snapshot is cross-check only.
- Deduplication: official 425 pairs; retain the lexicographically smaller image ID.
- Split unit: patient.
- Split policy: label-independent SHA-256 threshold, 70/10/10/10.
- Partitions: private train, public development, privacy-attack holdout, final test.
- Cap policy: label-independent SHA-256 rank within each patient.
- Public seeds, algorithms, aggregate counts and private-manifest commitments are in
  `_reports/isic2020_split_v1_001/manifest_lock.json`.
- Detailed patient/image mappings remain local-only under
  `_data/derived/isic2020_v2_split_v1/`.

The fixed split passed the preregistered metadata-only adequacy checks without trying alternative
seeds. No model, attack, or utility result existed when the split was frozen.

## Exact counts

| Condition | Partition | Patients | Images | Malignant images | Selected malignant-bearing patients |
|---|---|---:|---:|---:|---:|
| K2 | private train | 1,415 | 2,830 | 111 | 108 |
| K5 | private train | 1,415 | 6,512 | 210 | 169 |
| K10 | private train | 1,415 | 10,848 | 276 | 216 |
| K10 | public development | 200 | 1,565 | 37 | 30 |
| K10 | privacy-attack holdout | 233 | 1,870 | 42 | 34 |
| K10 | final test | 208 | 1,641 | 46 | 36 |
| Full | private train | 1,415 | 21,505 | 388 | 289 |

Across all four partitions the cap totals remain K2=4,112, K5=9,495 and K10=15,924. The exact
manifest-lock SHA-256 is
`7234076A4D7AA50B261984C18C92D3D65BF19CFB7B0300D8BBD327C191864D11`.

The fixed v2 lineage also exposed a correction to the earlier intake wording: after official
deduplication there are 32,701 image rows but 32,693 unique lesion IDs. Eight lesion IDs have two
retained images, always within one patient. They are not in the official duplicate list, so they
remain; patient ID is still the protected unit.

## Loader and exact-model results

- Four manifest/loader tests passed, including K nesting, cap enforcement, disjoint patient
  partitions, and real-pixel preprocessing.
- Eight real JPEGs from four patients loaded as finite 3x256x256 tensors.
- The exact SD-v1.4 fp16 VAE encoded them as finite 8x4x32x32 latents.
- VAE smoke peak reserved memory: 1.2227 GiB.
- Result:
  `_reports/isic2020_split_v1_001/real_image_loader_result.json`, SHA-256
  `77C942E3E0A3CD17DF7F0CD486124CC92FBB52EFF43AFAFFE5B886E80A9A784A`.

## PP-Mark exact-loader adaptation

The existing PP-Mark non-SDXL inversion loader did not pin a revision/variant and preferred a
deprecated prompt API. It now accepts exact `revision`, `variant`, safetensors and local-only
controls, fails closed if an explicitly requested variant is unavailable, and can disable the
irrelevant generation safety checker for DDIM inversion.

The final diagnostic loaded official `CompVis/stable-diffusion-v1-4` revision
`133a221b8aa7292a167afc5127cb63fb5005638b`, verified the locked UNet/VAE hashes, and produced a
finite 4x32x32 two-step inversion latent from one real ISIC image. Peak reserved memory was
2.2129 GiB. Result:
`_reports/isic2020_split_v1_001/ppmark_exact_loader_result.json`, SHA-256
`E9E0B25907EE23AB6E2E24AA126F11B31C0C8D90DB1C4F994FF8A7C397E37851`.

The first exact-loader attempt failed because PP-Mark implicitly required an unpinned safety-image
processor; the second exposed its deprecated `_encode_prompt` path. Both causes were corrected and
the final path passed. The existing PP-Mark suite still reports 16 passed and 5 environment skips.

This is only structural compatibility. Existing SD2.1/SDXL thresholds and robustness numbers are
historical and cannot be reused for a medically fine-tuned checkpoint, even when its base version is
also SD 2.1. After this Step-5 result, the base gate was formally reopened: the historical PP-Mark
mirror was pinned by exact revision and hashes, its third-party provenance was recorded, and SD 2.1
passed the license/artifact/GPU/PP-Mark interface checks. It is now the executable base; SD v1.4 is
the verified fallback. See `SD21_BASE_MODEL_GATE_DECISION.md`.

## Remaining gates before DP training

1. Acquire only the images required by the frozen smoke/pilot stage, preserving official hashes and
   license evidence; full acquisition remains permitted if scientifically triggered.
2. Freeze the DP epsilon/delta grid, operational patient release budget, secure-noise mechanism and
   accountant evidence schema.
3. Freeze patient-membership, memorization and extraction attack protocols and power thresholds.
4. Run a real-data non-DP smoke and representative K5 feasibility pilot; do not count it as
   confirmatory evidence.
5. Recalibrate PP-Mark on the eventual exact medical checkpoint and generated-image protocol.

Active LaTeX, abstracts and dissertation PDF remain unchanged.
