# NIH chest-X-ray metadata and patient-manifest gate

- Decision date: 2026-09-02 (Asia/Seoul)
- Gate status: **PASS_METADATA_AND_MANIFEST**
- Image-acquisition status: **42,423-IMAGE FULL ACQUISITION AND INDEPENDENT VERIFICATION PASS**
- Generator/DP-training status at gate closure: **NOT STARTED**
- Active-LaTeX status: **UNCHANGED**

Current supersession note (2026-09-09): this metadata gate originally labeled K2 as feasibility,
K5 as primary/main, and K10 as unit stress. Later runtime and feasibility contracts supersede only
those role labels: K2 is now accounting sensitivity, K5 is one-seed feasibility, and K10 is the
confirmatory main/contribution-stress tier. Counts, splits, nesting, and hashes are unchanged.

This gate freezes source metadata, PA view policy, clinically consequential weak-finding strata,
patient selection, patient-disjoint partitions, and nested contribution caps before any generator
or privacy result exists. It does not establish image integrity, clinical validity, generator
utility, differential privacy, attack resistance, PP-Mark robustness, or release authorization.

## 1. Exact source authority

The active authority is the NIH ChestXray14 official Box release. Every required file is locked by
byte count, SHA-1, and SHA-256:

| File | Bytes | Box file ID | SHA-1 | SHA-256 |
|---|---:|---:|---|---|
| `Data_Entry_2017_v2020.csv` | 9,003,496 | 219760887468 | `48A9F849A8F100A0F1721B33BDBD209767656111` | `C69A6DACA3549AF707CA9CACBDF9F9A7B6A9188E8C61157DF653520BB72D8EB1` |
| `BBox_List_2017.csv` | 92,416 | 219760940956 | `C567CEB25277C9883C5A7FE4C2F70C2EF94B02BE` | `0BBFEA9D4C4E9771481B3023B1BC9F0DF9DEA924453B12986BEB29B0C4D0C95B` |
| `train_val_list.txt` | 1,470,907 | 256056636701 | `E3E1B677C01D28481777F3D84E10FCDAAC05694C` | `61FBE896321C1C1C8B75F3E4F3A08E4FEF6486D95EF8A667C31D4D60DCA6CB81` |
| `test_list.txt` | 435,131 | 256055473534 | `41B85E218ABEC560A2F5999ACBCF333B0F2FA495` | `38CA5EF7F756092946F57C1A59FACA882ED589A1AB1F72590B45DC06C6D5E1CC` |

The earlier 9,003,499-byte mirror copy is preserved as
`Data_Entry_2017_v2020.mirror_preflight.csv`, SHA-256
`DC1D2DF67FDC1C5A7601D48699CDA2B13DC2C4841488B4183DCF04884DBACA11`. The discrepancy is fully
resolved: exactly one header differs (`Patient Sex` officially versus `Patient Gender` in the
mirror), while all 112,120 image data rows are identical. The mirror is not the authority.

## 2. Source population and fixed view policy

- Full metadata: 112,120 unique images from 30,805 patients.
- Official train/validation: 86,524 images from 28,008 patients.
- Official test: 25,596 images from 2,797 patients.
- Official image and patient overlap: zero.
- Frozen primary view: **PA only**, leaving 67,310 images from 28,868 patients.
- PA official-test census: 11,096 images from 2,647 patients.

AP images are not silently pooled. An AP+PA experiment would require a later explicit contract
amendment, view conditioning, and matched acquisition distributions.

## 3. Finding semantics and cohort policy

The four primary, overlapping evaluation groups are:

1. pneumothorax;
2. pneumonia or consolidation;
3. pleural effusion;
4. mass or nodule.

These are weak radiographic findings mined from reports, not adjudicated diagnoses. `Mass/Nodule`
is not a cancer label, and `No Finding` means none of the 14 mined labels rather than confirmed
clinical normality. All source labels are retained in the private manifests. The exact text/prompt
conditioning policy is still a pretraining gate.

A patient is a target patient if at least one PA image has one of the four primary findings. The
method cohort contains every available target patient plus an equal number of non-target controls
selected by a fixed hash. Every partition is therefore 1:1 at patient level. This enrichment was
chosen from metadata to power rare-finding method evaluation; it is explicitly **not** an estimate
of hospital prevalence.

The first general hash-random 8,000-patient design was retained in the record and rejected because
its K5 private-training partition contained only 215 pneumothorax and 225
pneumonia/consolidation images. The policy was changed once to target enrichment before training;
the first target-enriched seed was retained without reseeding:

`unified-private-image-assurance|nih-cxr14-pa-target-enriched-v1|2026-09-02`

Its exact hash serialization is `seed|tag|id`; that serialization is part of the frozen policy.

## 4. Patient partitions

The official test boundary is preserved. The official train/validation patients are split 70/15/15
within target/control strata; the final-test patients come only from the official test population.

| Partition | Patients | Target | Controls | Role |
|---|---:|---:|---:|---|
| private train | 8,476 | 4,238 | 4,238 | generator training only |
| public-development proxy | 1,816 | 908 | 908 | preprocessing, tuning, and calibration only |
| privacy-attack holdout | 1,816 | 908 | 908 | privacy attacks only |
| enriched final test | 1,818 | 909 | 909 | locked finding-stratified final evaluation |

`public-development proxy` refers to this reproducible public NIH proxy. It does not imply that a
real hospital should publish its development patients.

## 5. Nested contribution caps

Patient is the privacy unit. Within each patient, images are ordered by a label-independent SHA-256
rank, so K2 is a subset of K5 and K5 is a subset of K10. No image is consumed or removed by the
pilot; all caps can reference one acquired K10 superset.

| Cap | Role at metadata-gate freeze | Private train | Development | Attack holdout | Enriched final | Total |
|---|---|---:|---:|---:|---:|---:|
| K2 | feasibility pilot | 12,467 | 2,702 | 2,681 | 2,837 | 20,687 |
| K5 | primary/main | 18,393 | 4,031 | 4,012 | 5,015 | 31,451 |
| K10 | contribution-unit stress | 21,756 | 4,831 | 4,740 | 7,165 | 38,492 |

The K5 metadata-adequacy gates pass. Actual group counts are:

| Partition | Pneumothorax | Pneumonia/consolidation | Pleural effusion | Mass/nodule |
|---|---:|---:|---:|---:|
| private train | 792 / 510 | 894 / 785 | 2,544 / 1,778 | 3,211 / 2,320 |
| privacy-attack holdout | 192 / 134 | 181 / 151 | 532 / 368 | 709 / 523 |
| enriched final test | 497 / 248 | 275 / 238 | 814 / 478 | 801 / 462 |

Each cell is `images / patients`; groups overlap. This is a metadata adequacy result, not statistical
power for a particular attack effect size and not generator utility.

## 6. Distribution-sensitivity guard

Because the enriched final test cannot support prevalence-like claims, the full official PA test
census is separately frozen: 11,096 images from 2,647 patients. It is an overlapping secondary
view, not an independent second test. It may not select preprocessing, hyperparameters, attack
thresholds, or claims.

The K10 enriched final set is a subset of this census. The union of the K10 experiment and the PA
test census is 42,423 unique images, only 3,931 more than K10 alone.

## 7. Cryptographic commitments and verification

- Manifest lock SHA-256:
  `518D3288ACDF55BE355038BC693B3BCF7A8C44B8B3455FAC1DC493EE5E4E3ECE`
- Aggregate report SHA-256:
  `035FB07CA2F1D7958FEBA5F214ADC21F88697CEEA807452CEEC966A7493B8735`
- K2 manifest SHA-256:
  `742980F7C70647C20BF9BA010DF505764259D8A8481704EBBB53D8FE58F363BE`
- K5 manifest SHA-256:
  `DC49D82E497EAA6940DCF92C8F773179B8F7A838057A82AE0C26A69CA785BFC5`
- K10 manifest SHA-256:
  `2D749FB7B70823114A69FD55921277B0D0FF2F9AEC8FECD239159C553F3A0C06`
- Official PA-test census SHA-256:
  `7896B9FF931040CEC319EF755BADF098962B008EBDD6E531B5EEE33AD7AA74FE`

`build_nih_cxr14_manifests.py --verify` reproduced every output byte exactly. Six unit tests passed.
An independent PowerShell pass confirmed 13,926 unique patients, zero duplicate image IDs, zero
patient/partition mapping errors, PA-only rows, correct official split provenance, nested caps, and
the K10-final/census subset relation.

Detailed patient and image CSVs are local-only. The aggregate report is under
`_reports/nih_cxr14_pa_split_v1_001/report.json`.

## 8. Full-acquisition closure and immediate next gate

The full acquisition subsequently passed without changing this manifest. All 12 provider-hash
verified archives were processed one at a time, their complete 112,120-name source union exactly
matched the official metadata, and exactly 42,423 target PNGs/14,755 patients/16.457515 GiB were
materialized. Temporary archives were removed only after per-archive commits; none remain.

All files are 1024 x 1024 PNGs. Native modes are 42,244 `L` and 179 `RGBA`; every `RGBA` file has
pixelwise-equal R/G/B channels and fully opaque alpha, so actual-color and transparent images are
zero. Raw files were preserved without conversion. Acquisition-time checks, offline byte-exact
report regeneration, a separately implemented 42,423-file hash/decode verification, and an
independent PowerShell aggregate check all passed.

The full inventory SHA-256 is
`AD34344F962FAC7052114A3486D42D2B48CB5C9F27DDDC70365E07B715F68CC3`; the ordered content-set
SHA-256 is `E983A21B9B8558CE38F1AD5BA7C0F6BC51787CC7CBA739399ACE1976E2FB068E`. Full evidence and
the fail-closed native-mode/label-order findings are in `XRAY_FULL_ACQUISITION_GATE.md`.

The remaining interface sequence is:

1. freeze deterministic native-mode handling, intensity scaling, channel mapping, crop/padding,
   pilot resolution, and prompt conditioning;
2. include device/marker shortcut sensitivity prompted by the real-image visual checks;
3. run exact SD-2.1 VAE/LoRA and PP-Mark image-interface tests;
4. report before freezing the DP mechanism, accountant, privacy budget, and attack power.

No generator checkpoint, privacy result, receipt, or marked release image exists, and active LaTeX
remains unchanged.
