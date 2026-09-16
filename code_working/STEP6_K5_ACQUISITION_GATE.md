# Step 6 — Frozen K5 image acquisition and integrity gate

> **Post-gate role change:** this gate remains valid, but ISIC is now the separate dermoscopy
> extension rather than the first/core modality. See `XRAY_CORE_PIVOT_DECISION.md` and
> `XRAY_METADATA_MANIFEST_GATE.md`.

- Decision date: 2026-09-02 (Asia/Seoul)
- Decision: **PASS_FOR_PILOT_DATA**
- Scope: acquisition, patient-partition consistency, JPEG integrity, and local storage only
- Explicit non-claims: model utility, differential privacy, attack resistance, PP-Mark robustness,
  receipt validity, or release authorization

## 1. Dataset role and scientific scope

At the time of this historical gate, the scoped experiment was a **patient-linked dermoscopic
skin-lesion generator**, not a general medical-image generator. SIIM–ISIC 2020 remains suitable for
the later cross-domain extension because it supplies
2D RGB dermoscopic images, patient identifiers needed for patient-disjoint splitting and bounded
multi-image contributions, and a clinically meaningful benign/malignant task. It also fits the
selected SD 2.1 and PP-Mark image pipeline without introducing a separate 3D medical-imaging
architecture.

This choice does not mean that every skin lesion is skin cancer. In the acquired K5 population,
301 of 9,495 images have the malignant target and 9,194 do not. Both malignant and benign lesion
records can still be sensitive health information when they belong to an identifiable or linkable
patient.

The privacy threat must be stated precisely:

- patient-level membership inference can reveal whether a person's whole contribution was used;
- memorization, nearest-neighbor leakage, or reconstruction can expose distinctive lesion content;
- repeated images from one patient make image-level and patient-level privacy units materially
  different;
- patient-level DP is intended to bound the effect of adding or removing **all capped images from
  one patient** from training.

DP does not secure the raw image store, conceal a diagnosis that a user deliberately discloses,
prove image provenance, or make every generated image clinically safe. Those are separate access
control, governance, clinical-validation, and PP-Mark/receipt responsibilities. Dermoscopic images
usually contain fewer direct identifiers than faces or full-body photographs, so the paper must not
overstate direct visual re-identification. The defensible primary risks are sensitive cohort
membership and model-mediated memorization/extraction.

SIIM–ISIC 2020 is already public. Therefore, the experiment uses it as a reproducible proxy for a
restricted multi-patient clinical collection; it must not claim that the experiment newly protects
the confidentiality of the released ISIC subjects.

## 2. Frozen input

The downloader accepts only the already frozen K5 manifest:

- manifest: `_data/derived/isic2020_v2_split_v1/k5_private.csv`
- rows: 9,495 images from 2,056 patients
- manifest SHA-256:
  `721CBA21612D23A368086F4A0E8E5DE3173333DB3896A417102739A1DB63DC23`
- source release: SIIM–ISIC 2020 Challenge Training / ISIC Collection 70
- source DOI: `10.34970/2020-ds01`
- aggregate license boundary: CC BY-NC 4.0

No outcome from a model or privacy attack was used to select these files.

## 3. Acquisition result

| Partition | Patients | Images | Malignant images | Bytes |
|---|---:|---:|---:|---:|
| private train | 1,415 | 6,512 | 210 | 4,334,346,043 |
| public development | 200 | 919 | 26 | 531,300,218 |
| privacy-attack holdout | 233 | 1,093 | 30 | 722,135,437 |
| final test | 208 | 971 | 35 | 636,617,888 |
| **Total** | **2,056** | **9,495** | **301** | **6,224,399,586** |

All 9,495 expected JPEGs were acquired. The total is 5.796924 GiB. There are no failed images,
unfinished `.part` files, quarantined files, duplicate image IDs, or patients crossing partitions.

The acquisition report is
`_reports/isic2020_k5_acquisition_v1_001/acquisition_report.json`. Its SHA-256 is
`7C58BDCB5ACFF2C372CE0492B9B62BF1D5CB005F1B276DD0DFE3172F75CCF0D3`.

## 4. Integrity and independent re-verification

The integrity commitments are:

- ordered content-set SHA-256:
  `06F177B5C345121D3C5D876530768437A710C9C56DAD30F1225BD795C536E3F2`
- local detailed inventory SHA-256:
  `87AFBF37D1439806CAD5DC09CFD7A249D417EB16B047A403B2D236CCE08415FF`

An offline `--verify-only` pass independently reopened and decoded all 9,495 local files. It found
zero failures and reproduced the exact byte total, content-set digest, inventory digest, dimensions,
and partition counts. The verification report is
`_reports/isic2020_k5_verification_v1_001/acquisition_report.json`; its SHA-256 is
`38BDB0A0C1BC63C4DA5F94773F067DD8CB346E8CC7A386637CC702B745289D24`.

The downloader smoke test initially exposed that the ISIC S3 endpoint sometimes serves valid JPEGs
as `binary/octet-stream`. The policy was corrected to treat MIME only as a preliminary allow-list
and Pillow JPEG decoding/verification as authoritative. The same 12-file smoke set then passed both
network acquisition and offline verification with an identical content-set digest. This is retained
as diagnostic evidence rather than hidden as a clean first attempt.

## 5. Image composition and visual audit

The full source population and K5 subset contain dermoscopic skin-lesion images only; they do not
contain MRI, X-ray, CT, or multiple unrelated medical modalities. A deterministic visual check of
four malignant and four non-malignant K5 images showed expected dermoscopy variation, including
lesion color and boundary differences, hair, ruler marks, bubbles, vignetting, and one low-contrast
case. These are realistic acquisition artifacts, not a reason for outcome-adaptive exclusion.

This audit supports the narrow dermoscopy scope and detects obvious file/modality errors. It does
not establish diagnostic representativeness, fairness, generator quality, or clinical validity.

MRI remains a possible later cross-modality extension, but substituting it now would require a new
data gate plus explicit 3D/sequence representation, preprocessing, generator, and PP-Mark protocol.
It is not needed to test the present privacy-unit and evidence-chain hypotheses.

## 6. Storage and retention

- raw K5 directory: approximately 5.798 GiB
- complete `code_working` tree after acquisition: approximately 10.1 GiB
- free space on `C:` at final verification: approximately 59.01 GiB
- unfinished download files: 0
- quarantine files: 0

The storage gate remains healthy. The 9,495 frozen images are required experimental input and should
be retained locally. Raw images and the patient-level inventory must remain outside any source-code
or public artifact release; only aggregate reports, rules, and cryptographic commitments are public
evidence.

## 7. Gate conclusion and next action

The frozen K5 pilot population is locally complete, patient-isolated, hash-committed, restartably
acquired, and independently verified. **No generator or DP training was run**, no privacy claim was
created, and no active LaTeX source was changed.

Before any DP run, the next gate is to freeze the mechanism, patient/image adjacency definitions,
clipping and sampling semantics, secure-noise/accountant implementation, epsilon/delta grid,
operational release budget, and attack protocol/statistical power. Training begins only after that
gate is reported and accepted.
