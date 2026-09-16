# Data Acquisition and Claim Authority

Primary-source URLs and terms statements below were rechecked on 2026-07-25.
Source-site terms may change; users should recheck the linked primary source
before downloading or redistributing data.

All raw datasets are excluded from the anonymous submission package.  The
package includes hashes, manifests, derived mappings, and frozen verification
records.  Portable verification does not download data or retrain a model.

## WISDM Activity Prediction v1.1

Authoritative acquisition page:

`https://www.cis.fordham.edu/wisdm/dataset.php`

The official page identifies the Activity Prediction dataset, its v1.1
files, and the associated Kwapisz--Weiss--Moore paper.  It requests citation
of that paper and requests that `readme.txt` be included if the dataset is
shared or redistributed.  The submission package does not redistribute the
raw dataset.

Local full-rescan inputs:

```text
archive path
data/wisdm/WISDM_ar_latest.tar.gz
bytes
11404612
SHA-256
e573791269aab4a629721cf04b6031b7cbf8e14261cd2bc86e65887a2bc592d7

raw path
data/wisdm/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt
bytes
50326282
SHA-256
9aba4b1eaece56ab6d9187b16fc010c176e464750a4ec44c9d833514ecc1fa2e

readme.txt SHA-256
3f36b96a7b9ded114c8017654ef46f0d63afb003e4474693ee31e12ed1e6ae51
```

Full verification reopens the raw file.  Portable verification omits only
the raw-input rescan:

```console
python scripts/verify_wisdm_v2_gold.py --evidence-dir reports/wisdm_v4_multirun_001/confirmatory_run05 --data-path data/wisdm/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt --output-json reproduced_verification/wisdm_full.json
python scripts/verify_wisdm_v2_gold.py --evidence-dir reports/wisdm_v4_multirun_001/confirmatory_run05 --skip-input-rescan --output-json reproduced_verification/wisdm_portable.json
```

## UCI Human Activity Recognition Using Smartphones

Authoritative repository page:

`https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones`

Persistent identifier:

`https://doi.org/10.24432/C54S4K`

The official UCI page identifies 10,299 published windows, 561 features, 30
subjects, the fixed subject-partitioned train/test split, and a CC BY 4.0
license.  Attribution remains required under that source license.  The
submission package contains no source archive or published feature matrix.

Local full-rescan archive:

```text
path
data/uci_har/UCI HAR Dataset.zip
bytes
60999314
SHA-256
2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0
```

The UCI route binds eight extracted files.  Their combined declared-content
digest is:

```text
0d10319685c36554d12b88c91d5608805a6bfd95969de59de48c41c4d6ad7487
```

Full verification reopens all eight files, regenerates the 7,352-row train
mapping, and independently cross-checks accounting.  Portable verification
uses the bound manifest and generated evidence without the raw rescan:

```console
python scripts/verify_uci_har_v5.py --evidence-dir reports/uci_har_v5_multirun_001/confirmatory_run01 --data-root "data/uci_har/extracted/UCI HAR Dataset"
python scripts/verify_uci_har_v5.py --evidence-dir reports/uci_har_v5_multirun_001/confirmatory_run01 --skip-input-rescan
```

The route authorizes only published generated-window add/remove wording.
Raw-event and owner queries are deliberately unsupported and fail closed.

## ExtraSensory DP-HAR external-source intake

Fixed public code repository:

`https://github.com/jatanloya/extrasensory-dp`

```text
commit
167e0a899cb73c6e20398e8af886517cc9357c3e
Git tree
1106aa01d2f8806e72d4f99f5fc60814b3fdc9c7
```

The repository supplies an MIT license, which is retained with the minimal
source snapshot in the anonymous supplement.  The fixed code README identifies
the source data files at:

```text
http://extrasensory.ucsd.edu/data/primary_data_files/ExtraSensory.per_uuid_features_labels.zip
http://extrasensory.ucsd.edu/data/cv5Folds.zip
```

The submission package does not redistribute the 60 raw
`*.features_labels.csv.gz` files.  The local full-rescan inventory is:

```text
files
60
bytes
225325871
canonical name/size/SHA-256 inventory digest
36ac7814eb5e4710582a9b8d34881f0fe1e0e8bf17d033ef7bc23c644d16064d
```

The fixed read-only training mapping contains 250,906 selected examples from
47 nonempty owners, with maximum owner contribution 9,404 and ordered mapping
digest:

```text
a3fd2d0fa6c370ca88660a601b941cbf659d8ecfda8392a6d218e5095a82bd64
```

Both execution attempts failed before a data record or model output was
produced.  The 18/18 full intake verifier therefore authenticates a typed
non-positive result, not a completed training or privacy result.  Full
verification requires the separately obtained raw files:

```console
python scripts/verify_audit_external_intake_v12.py --output-json reproduced_verification/external_intake_full.json
```

The anonymous package's portable verifier authenticates the MIT source
snapshot, split/config/source hashes, both retained attempts, the frozen full
verification record, empty positive wording, and the declared raw inventory
without claiming to rescan absent raw data.

## PhysioNet/Computing in Cardiology Challenge 2019

Authoritative release:

`https://physionet.org/content/challenge-2019/1.0.0/`

Persistent identifier:

`https://doi.org/10.13026/v64v-d857`

The source release includes its own `LICENSE.txt`.  Users must follow the
current PhysioNet release terms.  The artifact's 5,000-patient evidence is
mapping-only: it does not bind a trained mechanism or authorize epsilon/delta
wording.

## Backblaze Drive Stats Q1 2025

Authoritative dataset and usage conditions:

`https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data`

Bound source archive:

```text
https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q1_2025.zip
bytes
1020483699
SHA-256
7a230565f72fdb3882c05fdcfab361c801e9a545ba18543899fc7618174171ec
```

The official page asks users to cite Backblaze, states that users are
responsible for their use, permits sale of derivative works, and disallows
selling the data itself.  The artifact's Backblaze evidence is mapping-only
and contains no epsilon/delta claim.

## Claim-authority boundary

Acquisition authority and privacy-claim authority are separate:

- source URLs, dataset licenses/terms, hashes, and raw rescans establish data
  identity and permitted-use context;
- mapping diagnostics establish selected identities and observed
  multiplicities;
- only a registered mechanism, matched runtime evidence, validated
  transformation path, nonvacuous conversion, and the fail-closed validator
  can authorize a public privacy sentence; and
- hashes and independent recomputation expose inconsistency but do not
  cryptographically attest that an artifact producer executed the declared
  program.
