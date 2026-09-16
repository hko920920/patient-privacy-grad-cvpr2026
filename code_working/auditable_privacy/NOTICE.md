# Notice

This file records the rights and data-source boundary for the Audit code and
data supplement.  It is not itself a software or content license.

## Project-authored material

No project-wide license has been selected.  Unless a particular file contains
its own license notice, the project-authored source code, documentation,
schemas, tests, reports, and generated tables are supplied for anonymous
research evaluation and reproducibility review without an additional license
grant.  Rights remain with their respective rights holders.

The absence of a project license is intentional and is reported as `partial`
in the reproducibility checklist.  This notice must not be interpreted as
silently applying an open-source license.

## Third-party software

Third-party Python packages are identified in `requirements-core.txt` and
`requirements-tfprivacy-baseline.txt`.  They are not relicensed by this
project and remain governed by their own licenses.  Installing a dependency
is a separate transaction between the user and that dependency's
distributor/rightsholder.

The optional TensorFlow Privacy comparison invokes the official installed
package and records its version and loaded-module digest.  It does not
incorporate that package into the project-authored source files.

The anonymous supplement includes a minimal fixed snapshot of the
`jatanloya/extrasensory-dp` source repository under `third_party/`, together
with its MIT `LICENSE`.  Those files remain third-party material governed by
that license and are not relicensed as project-authored work.

## Separately governed datasets

Public raw datasets are not redistributed in the submission package.  The
package contains derived mappings, hashes, manifests, and verification
records.  Any rights that apply to the source data continue to apply
independently of this notice.

- WISDM Activity Prediction v1.1 is obtained from the official WISDM Lab
  dataset page.  That page requests citation of the associated paper and
  requests that `readme.txt` accompany any redistribution of the dataset.
  The submission package does not redistribute the raw WISDM file.
- UCI Human Activity Recognition Using Smartphones is identified by DOI
  `10.24432/C54S4K`.  The UCI repository labels the dataset CC BY 4.0 and
  requires attribution under that license.  The submission package does not
  redistribute the source archive.
- ExtraSensory per-UUID feature/label files and source cross-validation data
  remain governed by their source terms.  The package retains only the
  MIT-licensed external code snapshot, four fixed split lists needed to
  reproduce the observed CRLF portability failure, hashes, and mapping
  summaries; it does not redistribute the 60 raw feature/label files.
- PhysioNet/Computing in Cardiology Challenge 2019 data remains governed by
  the license and terms supplied with the official PhysioNet release.
- Backblaze Drive Stats remains governed by the usage conditions stated on
  the official Drive Stats download page, including source attribution and
  its restriction on selling the data itself.

Exact authoritative URLs, acquisition hashes, and the distinction between
raw-source verification and portable verification are recorded in
`DATA_AND_AUTHORITY.md`.

## No transfer of authority

A data or software license controls permission to use material; it does not
establish a differential-privacy claim.  Only the registered evidence
contract and `scripts/privacy_claim_validator.py` can authorize public
privacy wording within this artifact.
