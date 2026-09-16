# NIH chest-X-ray official-archive and PNG source-smoke gate

- Decision date: 2026-09-02 (Asia/Seoul)
- Gate status: **PASS_SOURCE_SMOKE**
- Full 42,423-image acquisition: **SUBSEQUENTLY PASSED; SEE SECTION 6**
- Generator/DP training: **NOT STARTED**
- Active LaTeX/abstracts: **UNCHANGED**

This gate answers a narrow question: can the frozen PA-only patient manifests be connected to the
actual official NIH PNG release through exact, reproducible source commitments? The answer is yes
for the mapped source and the first-archive smoke. It does not establish full-population integrity,
clinical label validity, generator utility, differential privacy, attack resistance, PP-Mark
robustness, or release approval.

## 1. Official archive map

The official NIH ChestXray14 Box image folder is
`https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737`. The 12 current image archives are
frozen by archive name, Box file ID, Box file-version ID, byte count, and provider SHA-1 in:

`_data/intake/nih_chestxray14_metadata/official_image_archives_v1.json`

- Catalog SHA-256:
  `3CD87825C2F7B604FF1E599A988A68D7D4B7C27AB8DB984B94F3139837572B41`
- Archive count: 12
- Total compressed bytes: 45,079,862,784 (41.983894 GiB)
- Smallest/largest archives: 2,008,470,987 / 4,187,084,020 bytes

The NIH-provided historical `batch_download_zips.py` is separately locked at 1,320 bytes and SHA-1
`8D61C8A042DFFA32A0B16FD91D418739629293E5`, but its embedded static download URLs currently return
404. It is historical provenance only. Acquisition uses the current Box shared-file ID route with
HTTP range resume and refuses any byte-count or SHA-1 mismatch.

## 2. Frozen smoke protocol

`data_pipeline/smoke_nih_cxr14_official_archive.py` locks all of the following before inspecting an
image:

- the official 12-archive catalog;
- official `Data_Entry_2017_v2020.csv`;
- the 38,492-image K10 manifest;
- the 11,096-image official PA-test census manifest;
- their 42,423-image union;
- the deterministic seed
  `unified-private-image-assurance|nih-cxr14-official-png-smoke-v1|2026-09-02`.

The default smoke archive is `images_001.tar.gz`. The selector requires at least one candidate for
each K10 partition crossed with target/non-target status, each of the four primary weak-finding
groups, two `No Finding` references, and two census-only images, then fills to 24 by salted SHA-256.
The buckets may overlap scientifically, but one image is materialized only once. The script fails
closed if any required bucket is absent.

Tar members are never passed to `extractall`. Absolute paths and `..` traversal are rejected, every
PNG basename must occur in the exact 112,120-row NIH metadata, and duplicate basenames fail the
gate. Only the preselected 24 images are extracted.

## 3. Actual source-smoke result

The downloaded archive identity is:

| Field | Verified value |
|---|---|
| Archive | `images_001.tar.gz` |
| Box file ID / version | `219764235225` / `232165503961` |
| Bytes | 2,008,470,987 |
| Provider and verified SHA-1 | `FEF95A7A789BCB0013FBF966CB92C4D92C90BECD` |
| Local SHA-256 | `FD8E3542DB6351AE9377779033F5D5C5F32FE50EB0830B519FBF1A7E791354B1` |

The tar scan found 4,999 regular files, all PNGs, with zero non-PNG files, duplicate PNG basenames,
unsafe paths, special members, or metadata-missing filenames. Of those PNGs, 2,042 belong to the
frozen K10-plus-census union.

The 24-image materialized smoke contains 24 unique patients and includes every required diversity
bucket. Aggregate observed composition is 16 target-patient and 8 non-target-patient images, 13
`No Finding` weak-label references, 2 census-only images, and at least one image in every primary
finding group. Because selection requirements overlap and the hash fill is unrestricted, these
aggregate counts are not intended to be balanced evaluation counts.

All 24 files independently pass:

- exact per-file SHA-256 regeneration;
- PNG signature and decode;
- 1024 x 1024 dimensions;
- single-channel grayscale mode;
- nonempty byte and finite pixel-statistic checks.

The ordered content-set SHA-256 is
`A802BBDAF084D4B0CA51B2E2791A36E6E76985C245C24A1588436BFD4726765A`.

## 4. Reproduction and independent checks

The aggregate report is
`_reports/nih_cxr14_official_png_smoke_v1_001/report.json`, SHA-256
`E74DCEE48BC57343F265FBC5B78D96719E6E40FAAAF6F42F01EB0D7DD6F944C9`. The local-only detailed
inventory is
`_data/raw/nih_cxr14_pa_k10_plus_census_v1/smoke_inventory_private.csv`, SHA-256
`EA457F3E58D79F8FAE29E792ECF7E98E504B14BD904EE4DD318D2E068B64F72A`.

A second `--verify-only` execution re-scanned the archive, reselected the same 24 images, and
regenerated the inventory and report byte-for-byte. A separate PowerShell pass recomputed archive
SHA-1/SHA-256, every selected image SHA-256, PNG headers, dimensions, color type, row uniqueness,
and image/patient counts; all mismatch counts were zero.

Four deterministic examples were also viewed manually. They are recognizable frontal chest
radiographs and show real-world heterogeneity in exposure, positioning, laterality markers, tubes,
and surgical clips. This is only a source/format sanity check, not radiologist adjudication and not
confirmation of any weak label.

## 5. Consequence for the next gate

The visual heterogeneity is expected, but devices and markers can become shortcuts. Before
training, the exact preprocessing and utility protocol must therefore freeze intensity handling,
resize/crop/padding, grayscale-to-model channel mapping, and device/marker sensitivity checks. The
workflow must not silently remove clinically meaningful devices merely to improve a metric.

The verified 1.871-GiB first archive is retained to avoid an immediate re-download. At final
verification, C: has approximately 58.7 GiB free. The preferred full-acquisition implementation
will process the 12 archives one at a time, materialize only the 42,423-image union, verify each
archive and selected output, and remove a temporary archive only after its contribution is
committed. This is an operational storage plan, not a scientific sample-size cap; additional
retention or storage may be used if later verification genuinely requires it.

The 24-file mean gives a provisional 16.239-GiB union estimate; adding the largest one-at-a-time
archive gives roughly 20.138 GiB of incremental peak space. This is capacity planning only, not a
promise about the unseen archives; the acquisition script must keep checking actual free space.

The next gate is the **full selective 42,423-image acquisition and independent content inventory**.
Only after that passes will preprocessing and the exact SD 2.1 VAE/LoRA/PP-Mark image interface be
frozen. No model training begins at this source-smoke gate.

## 6. Subsequent status

The full selective acquisition and independent verification subsequently passed. The completed
population contains 42,423 PNGs/14,755 patients/16.457515 GiB with content-set SHA-256
`E983A21B9B8558CE38F1AD5BA7C0F6BC51787CC7CBA739399ACE1976E2FB068E`. The full run also showed
that the 24-image smoke's all-`L` observation was not population-wide: 179 files are natively
`RGBA`, but all are pixelwise grayscale-equivalent with fully opaque alpha. No raw conversion was
performed. See `XRAY_FULL_ACQUISITION_GATE.md`.
