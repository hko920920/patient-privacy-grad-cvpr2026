# Anonymous Code and Data Supplement

Paper: **Evidence-Sealed Registration of Owner-Sampled DP-SGD Routes**

This archive is the anonymous code/data surface matched to the active V4.3
main manuscript, synchronized V4.1 supplement, and V4.0 reproducibility
checklist. Version numbers retained in dependency locks, table files, scripts,
or evidence directories identify inherited frozen materials; they do not
identify another active manuscript.

The registered training routes are P (Bernoulli owner sampling under
add/remove adjacency), F (fixed-size sampling without replacement under
replace-one adjacency), and A (ownerwise exact-k-of-T random allocation under
add/remove adjacency). The H stress test replaces only A's accounting handler
with an isolated externally maintained PLD implementation. H is not a fourth
mechanism, an open-world plugin result, or an external privacy certificate.

## Quick verification

The archive was built with Python 3.11.5. The verifier disables bytecode before
importing package modules, so the documented plain command is safe in a clean
extraction and does not create a forbidden `__pycache__` entry.

```text
python -m pip install -r requirements-aaai27-v38.txt
python scripts/verify_public_supplement_v41.py
python -B -m pytest -q -p no:cacheprovider
```

The portable clean-extraction suite collects 154 tests: 141 pass and 13 skip
when optional external numerical backends are unavailable. The public verifier
authenticates every manifest row, reruns the finite P/F/A/H gates, binds the
active manuscript sources and Figure 7, checks the path-neutral local
regression summary, and scans identity, private-trace, backend-tree, transient,
and historical-submission boundaries.

## Deterministic archive rebuild

From a clean extraction, the following creates a byte-identical ZIP and
manifest under `_rebuild/`:

```text
python scripts/build_compiler_code_data_package_v10.py --paper-dir paper --output _rebuild/compiler_code_and_data_supplement_aaai27_v10 --zip _rebuild/compiler_code_and_data_supplement_aaai27_v10.zip
```

The output directory's basename must remain
`compiler_code_and_data_supplement_aaai27_v10` because it is the deterministic
ZIP prefix.

## Active-paper-only boundary

The `paper/` directory contains exactly one main source,
`main_aaai27_v43_candidate.tex`, together with the synchronized supplement,
checklist sources, five bibliography files, official AAAI style files, the one
referenced table source, and `figures/figure7.png`. It contains no prior main
manuscript, PDF, upload manifest, or nested submission archive. PDFs are bound
separately by the four-file upload manifest.

## Regression evidence

`reports/v42_public_regression_summary_20260728/` contains an anonymous,
machine-readable summary of the frozen local base-suite run: 161 collected,
154 passed, and seven raw-data-scoped skips. It binds the original transcript
by SHA-256 but does not redistribute that transcript because its warning output
contains workstation-specific absolute paths. This local surface is distinct
from the base-plus-H 173/180 run and the portable 141/154 clean-extraction run.

## External PLD backend

The 110,964,413-byte backend environment is deliberately not redistributed.
The adapter binds `PLD_accounting==0.5.0` and
`random-allocation==1.0.5`, including their versions, MIT license declarations,
wheel hashes, and installed-tree hashes. Recorded full-backend calls are
expensive; the portable verifier authenticates the sealed reports without
silently substituting another backend. See
`requirements-external-pld-v38.txt` for the isolated target dependencies.

## Public/private evidence boundary

Raw UCI HAR, WISDM, and PhysioNet Sepsis files are not redistributed. Public
mappings and fixed preprocessing artifacts are included. The public H
execution record and manifest are included, but the nested private Route-A
allocation trace is not. No third-party wheel, installed backend tree, private
allocation trace, or research randomizer is present.

## Evidence scope

The package supports fixed-case and finite-lattice conformance, matched-shell
ablation, and fixed-seed integration checks. These checks do not prove privacy,
arbitrary third-party extensibility, semantic completeness, malicious-code
safety, formal refinement, utility or population generalization, independent
external reproduction, or cryptographic attestation. The mechanisms and
accountants are prior work.

## Layout and license status

```text
configs/          frozen P/F/A contracts and preprocessing artifacts
paper/            active V4.3/V4.1/V4.0 manuscript-supporting sources
reports/          public execution and finite conformance evidence
scripts/          deterministic builders and portable verifiers
specs/            declarative Route-P oracle specification
src/              frozen P/F/A production and oracle packages
tests/            portable predecessor tests
v36_candidate/    isolated H adapter, contracts, and tests
```

`PUBLIC_PACKAGE_MANIFEST.json` binds every other file. This anonymous archive
is supplied for confidential peer review and does not itself grant a public
software license. Any later public release and license choice are outside the
submission's evidentiary claims.
