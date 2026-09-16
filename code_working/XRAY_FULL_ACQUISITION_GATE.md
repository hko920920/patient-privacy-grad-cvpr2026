# NIH chest-X-ray full selective-acquisition gate

- Decision date: 2026-09-02 (Asia/Seoul)
- Gate status: **PASS_FULL_ACQUISITION_INDEPENDENTLY_VERIFIED**
- Frozen union: **42,423/42,423 PNGs materialized**
- Generator/DP training: **NOT STARTED**
- Active LaTeX/abstracts: **UNCHANGED**

This gate connects the exact official NIH release to every image in the frozen PA-only
K10-plus-official-test-census union. It establishes source identity, complete filename coverage,
patient/partition linkage, raw-file content commitments, PNG decode integrity, and native-channel
semantics. It does not establish radiologist-adjudicated labels, model utility, differential
privacy, attack resistance, PP-Mark robustness, or release authorization.

## 1. Archive-by-archive acquisition result

All 12 archives in `official_image_archives_v1.json` were processed in catalog order. For each
archive the implementation:

1. resumed or downloaded the current Box file-ID route;
2. required the exact provider byte count and SHA-1;
3. computed and retained a local SHA-256;
4. rejected unsafe paths, special members, non-PNG regular files, duplicate basenames, and names
   absent from the exact NIH metadata;
5. extracted only members in the frozen 42,423-image union;
6. decoded and hashed every selected PNG;
7. atomically committed the archive's complete source-name list and selected inventory;
8. re-hashed the selected files from disk;
9. removed the temporary archive only after that commit passed.

The complete 12-archive source-name union contains 112,120 unique PNG basenames and is exactly equal
to the 112,120 official metadata rows. Its ordered name-set SHA-256 is
`9E749A0B70E21F2803B792B4E4D4C2D7BF7560E2C4BD75780A5B83904DF79B7A`. There are zero missing,
extra, or cross-archive duplicate source names.

All temporary `.tar.gz` files were removed after commit and can be recovered by re-downloading the
locked official Box IDs. No `.part` file remains. The local per-archive commit records preserve
provider identity, local archive SHA-256, all-source-name commitments, and selected-content
commitments.

## 2. Exact materialized population

The raw image directory now contains exactly:

| Measure | Result |
|---|---:|
| PNG files | 42,423 |
| Unique patients | 14,755 |
| Total bytes | 17,671,122,456 |
| Total GiB | 16.457515 |
| Missing / extra / duplicate files | 0 / 0 / 0 |

Partition coverage is:

| Partition | Images | Patients |
|---|---:|---:|
| private train | 21,756 | 8,476 |
| public development | 4,831 | 1,816 |
| privacy-attack holdout | 4,740 | 1,816 |
| enriched final test | 7,165 | 1,818 |
| official PA-test census only | 3,931 | 1,113 |

The 3,931 census-only images are outside K10, but patient identity is intentionally not treated as
disjoint: their 1,113 patients include 284 patients already represented by other K10 images and 829
additional patients. Thus the union has 14,755 unique patients rather than the 13,926-patient K10
method cohort. The full official PA-test census remains an overlapping distribution-sensitivity
view, not a tuning set or an independent second test.

Aggregate weak-label coverage in the materialized union is 3,161 pneumothorax, 1,993
pneumonia/consolidation, 6,144 pleural effusion, 6,821 mass/nodule, and 20,388 `No Finding` images.
Groups overlap. These counts do not turn weak labels into diagnoses or establish attack power.

## 3. Native PNG mode issue found and resolved fail-closed

The 24-image source smoke happened to contain only mode `L`, so the first full-acquisition attempt
correctly stopped on the first mode-`RGBA` file rather than silently converting it. A diagnostic of
all 2,042 already-extracted archive-1 targets found 2,015 `L` and 27 `RGBA` files. Every `RGBA` file
had pixelwise `R = G = B` and alpha `(255, 255)`: an opaque grayscale image stored with replicated
channels, not actual color or transparency.

The raw contract was therefore amended before any archive commit or later download:

- preserve the official file bytes and native mode;
- accept `L`, or multi-channel encodings only when all color channels are exactly equal;
- require alpha to be fully opaque when present;
- record native mode, grayscale equivalence, alpha extrema, and grayscale pixel statistics;
- fail on actual color, partial transparency, unsupported modes, or non-1024 x 1024 images.

The final population contains 42,244 native-`L` PNGs and 179 native-`RGBA` PNGs. All 179 are exactly
grayscale-equivalent and fully opaque. Actual-color images, nonopaque-alpha images, unsupported
modes, dimension failures, and decode failures are all zero. No raw image was converted.

## 4. Label-order issue found by independent verification

The first independent verification attempt stopped at `00003440_000.png` because the official
metadata stores `Mass|Atelectasis` while the frozen manifest stores `Atelectasis|Mass`. A full
comparison showed 32 raw string-order differences, zero label-set differences, and zero deviations
from the manifest builder's alphabetically sorted canonical serialization.

The independent check was corrected to require both exact label-set equality and exact canonical
manifest serialization. It does not ignore missing, added, or duplicated labels. This was a verifier
assumption correction, not a data or manifest change; all frozen manifest and inventory hashes
remain unchanged.

## 5. Cryptographic commitments

- Full local-only content inventory SHA-256:
  `AD34344F962FAC7052114A3486D42D2B48CB5C9F27DDDC70365E07B715F68CC3`
- Ordered image/content-set SHA-256:
  `E983A21B9B8558CE38F1AD5BA7C0F6BC51787CC7CBA739399ACE1976E2FB068E`
- Archive commit-set SHA-256:
  `A77C36673D5F929003FE96CF424AA7C18F283D78F98A42402885594C5265EFBB`
- Acquisition aggregate report SHA-256:
  `A76611CF1A11A38502685627453611B7BCAE6BE998AAA05E8EAEBB6578944311`
- Independent verification report SHA-256:
  `A1E34DE71BB8F048F6CA44732B20FBF2041DFF3723C6633BFE2D3328B764AB92`
- Acquisition implementation SHA-256:
  `B39192BAD2B3926A52F4F3A016FF44DA7C0B331AB7266A09BD2E0F133B592594`
- Independent verifier SHA-256:
  `D1DC19914985C38C54C56638F1A3ABC0D9313A8522A476135BE5D3C0AB38AC81`

Detailed filenames, patient IDs, labels, archive-member paths, and per-image hashes remain local
under `_data/raw/nih_cxr14_pa_k10_plus_census_v1/`. Only aggregate reports and commitments should be
published.

## 6. Verification layers

Four different checks passed:

1. acquisition-time provider hash, tar, extraction, decode, and per-archive commit verification;
2. an offline `acquire_nih_cxr14_union.py --verify-only` pass that re-hashed all selected files and
   regenerated the full inventory and aggregate report byte-for-byte;
3. `verify_nih_cxr14_union_independent.py`, which imports none of the acquisition/source-smoke code,
   independently rebuilds the union and re-hashes/re-decodes all 42,423 PNGs;
4. a separate PowerShell check of inventory rows, unique images/patients, disk file count/byte sum,
   native-mode/alpha constraints, staging emptiness, and partial-file absence.

The independent verification has zero hash, mapping, decode, actual-color, nonopaque-alpha,
missing, extra, or duplicate failures.

## 7. Storage and next gate

After archive cleanup and final verification, C: has approximately 43.1 GiB free. The 42,423 raw
PNGs are the single K10-plus-census superset; K2 and K5 remain nested manifest views and do not need
duplicate image copies.

The next gate is to freeze and test the exact raw-to-model interface:

1. deterministic native-`L`/opaque-grayscale-`RGBA` conversion without changing raw files;
2. intensity scaling and grayscale-to-model channel mapping;
3. crop, padding, resize, and pilot resolution;
4. device/marker shortcut sensitivity and prompt-conditioning policy;
5. exact SD 2.1 VAE/LoRA compatibility;
6. PP-Mark exact-image inversion/calibration compatibility.

Only after that interface passes will the DP mechanism, accountant, privacy budget, and attack
protocol be frozen. No generator, DP mechanism, receipt, PP-Mark release image, or active LaTeX
revision was produced in this acquisition gate.
