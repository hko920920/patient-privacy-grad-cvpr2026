# Step 2 — 2D Medical Dataset Intake Decision

> **Superseded for the first/core modality on 2026-09-02.** The user approved frontal chest X-ray
> as the core and retained ISIC dermoscopy as a separately trained cross-domain extension. This
> document remains the historical ISIC gate; the active decision is
> `XRAY_CORE_PIVOT_DECISION.md`, with the exact X-ray metadata/manifests recorded in
> `XRAY_METADATA_MANIFEST_GATE.md`.

- Decision date: 2026-09-02 (Asia/Seoul)
- Status: dataset gate passed; primary and smoke datasets selected
- Scope: central training of a 2D medical-image generator, not federated learning

## Decision

1. **Primary experiment candidate:** SIIM–ISIC 2020 Challenge Training, Collection 70.
2. **Smoke/integration dataset:** PAPILA v1.1.
3. **Secondary reserve:** BreakHis; do not download unless an independent histopathology-domain
   replication is scientifically needed.
4. **Large external-validity reserve:** CheXpert Plus; do not make it the first implementation
   target because it adds DICOM/grayscale/report and access-policy work before the core lifecycle is
   tested.

This selection is based on subject lineage and experiment strength, not on a hard storage cap.
The complete 23 GB ISIC JPEG release remains allowed if the pilot shows that the full population is
needed. If it is acquired, prefer direct per-image download or another-drive storage so that a
23 GB archive and a second extracted copy do not coexist without a reason.

## Required dataset properties

The main dataset must satisfy all of the following.

- A stable subject/patient identifier connects every image to the privacy unit.
- Multiple images per subject remain after documented duplicate removal.
- Train, validation, and test partitions can be made strictly by subject.
- Images are 2D and can enter an RGB generator and the current PP-Mark image path without inventing
  a 3D claim.
- A fixed metadata release, license, attribution, and acquisition URL can be cited and hashed.
- Labels support at least one downstream utility test; they are not treated as a privacy guarantee.
- Public-benchmark use is described as a proxy for a restricted clinical cohort. It does not prove
  that real confidential patient data were handled.

## Primary: ISIC 2020

Official sources:

- Dataset and fixed downloads: <https://github.com/ImageMarkup/isic-challenge-2020/blob/master/index.md>
- Current Collection 70 page: <https://api.isic-archive.com/collections/70/>
- Dataset DOI: <https://doi.org/10.34970/2020-ds01>
- Official selective-download client: <https://github.com/ImageMarkup/isic-cli>

### Fixed files acquired

| Local file | Bytes | SHA-256 |
|---|---:|---|
| `_data/intake/isic2020_collection70/ground_truth_v2.csv` | 2,387,418 | `A1C7C97C59E5F49232DFE3B6F6817EFAAEE5DBF9BCB4CE584C1889E641BE5C4E` |
| `_data/intake/isic2020_collection70/duplicates.csv` | 11,500 | `DDFD6C2D42430BD3581F3B2116BDB1433462A7109C715A02BF7F51115FCFABCE` |
| `_data/intake/isic2020_collection70/metadata.csv` | 7,829,945 | `0F6D0518FDAA7F5D7A005C97871C1D6487FB27B1A485CFFB6BD87AE4D1A3C442` |

`ground_truth_v2.csv` and `duplicates.csv` are the fixed 2020 source of truth. The last file is a
2026-09-02 snapshot of the mutable current API and is retained only for comparison and download
routing.

### Subject-lineage audit

- Raw images: 33,126.
- Patient IDs: 2,056, with no missing patient ID.
- Lesion ID fields: 32,701 non-missing rows. After official duplicate removal there are 32,693
  unique lesion IDs; eight lesion IDs each have two retained images, always within one patient.
  These 16 images are not listed in the official duplicate pairs and therefore remain.
- Official duplicate pairs: 425, covering 850 distinct images.
- Deterministic duplicate rule: retain the lexicographically smaller image ID in each official
  pair and exclude the other image before any split.
- After that rule: 32,701 images and all 2,056 patients remain.
- Images per patient after duplicate removal: minimum 2, median 12, mean 15.91, 95th percentile 44,
  maximum 115.
- Patients with at least 5 images: 1,570; at least 10 images: 1,140.
- Malignant target images after duplicate removal: 581; patients with at least one malignant image:
  428. This severe imbalance must be handled in conditional-generation and downstream-utility
  evaluation, not hidden by an image-wise split.

The current API and fixed v2 release contain the same 33,126 image IDs and patient IDs, but two
lesion IDs differ. Therefore every reproducible split and claim must derive from the fixed v2 file,
not silently from the mutable API.

The aggregate 2020 release is governed by CC BY-NC 4.0 and requires the official attribution. Some
current API rows expose CC-BY while others expose CC-BY-NC; the aggregate non-commercial license is
the conservative rule for this dissertation package.

Eight real JPEGs from four patients have been downloaded to `smoke_images/`. They cover a
malignant-bearing and a benign-only patient cohort, preserve two images per selected patient, and
range from 640×480 to 2592×1936. Their exact selection and hashes are in `smoke_manifest.json`.

## Smoke dataset: PAPILA v1.1

Official sources:

- Figshare record and download: <https://figshare.com/articles/dataset/PAPILA/14798004>
- Dataset paper: <https://doi.org/10.1038/s41597-022-01388-1>

The complete archive has been acquired as `_data/intake/papila/PAPILA.zip`.

- Bytes: 590,857,635.
- Provider MD5 and local MD5: `7D73FC5157D44B221A62ED5A4BAF3779` — exact match.
- Local SHA-256: `15B053DFF496BC8E53EB8A8D0707EF73BA3D56C988EEA92B65832C9C82852A7D`.
- Fundus images: 488 JPEGs.
- Subject lineage: 244 anonymized subjects, exactly one right-eye and one left-eye image per
  subject; no unmatched image or clinical subject ID.
- Eye-level diagnosis counts are 333 healthy, 87 glaucoma, and 68 suspect images. Seven subjects
  have different right/left-eye diagnosis codes, so diagnosis is eye-level while privacy remains
  subject-level.

The Figshare dataset API reports `GPL 3.0+`; the Scientific Data article itself is CC BY 4.0. These
licenses attach to different published objects and must not be casually substituted for one
another. PAPILA is approved for internal academic smoke testing, but the archive or generated
derivatives will not be redistributed until the applicable dataset-license obligations are stated
explicitly.

PAPILA is not the main patient-DP evidence because 244 subjects with exactly two images each provide
far less variation in subject contribution than ISIC.

## Reserves, not current downloads

### BreakHis

The official UFPR page reports 7,909 700×460 RGB PNG images from 82 patients and encodes the patient
identifier in each filename. The official archive reports 4,273,561,758 bytes. It is technically
convenient but statistically weaker than ISIC for a patient-level privacy-unit study. The page also
contains a 9,109/7,909 count inconsistency, so a later use would require an archive-level audit.

Source: <https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/>

### CheXpert Plus

The official Stanford page reports 223,462 chest-X-ray/report pairs, 187,711 studies, and 64,725
patients, with multiple studies per patient and possibly multiple radiographs per study. It is a
strong scale/format extension, but DICOM, grayscale handling, reports, a much larger acquisition,
and Stanford/Redivis terms would add a second problem before the central chain works.

Source: <https://aimi.stanford.edu/datasets/chexpert-plus>

## Locked split and claim rules

1. Remove only the 425 official ISIC duplicate counterparts using the fixed deterministic rule.
2. Split subjects first; derive image partitions from the subject split. A patient must never cross
   train, validation, attack, calibration, or test partitions.
3. Record the maximum allowed images per patient and the actual contribution histogram in every DP
   training receipt. Do not convert image-level DP to patient-level wording merely because patient
   IDs exist.
4. Use the same subject partition across non-DP, image-DP, and patient-DP comparisons.
5. Separate generation quality, downstream diagnostic utility, privacy attacks, and PP-Mark
   robustness. None substitutes for another.
6. Call all results on ISIC/PAPILA a public-benchmark proxy. A real-confidentiality claim requires a
   controlled dataset or an explicitly bounded external validation later.

## Step-2 exit decision

The dataset gate is passed: legal/research use has a documented basis, subject lineage is explicit,
repeated per-patient contribution is real rather than synthetically invented, and actual image
access has been verified. The next implementation contract may therefore use `patient_id` as the
requested privacy unit and `image/lesion` as lower-level records. Full ISIC acquisition is a
scientific scaling decision after the loader and contribution-cap pilot, not a prohibition caused
by storage.
