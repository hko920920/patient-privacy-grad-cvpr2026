# Anonymous Code and Data Supplement

This bundle supports *Post-Execution Evidence-Bound Validation of Privacy-Unit
Claims in Windowed DP-SGD*. The main paper, supplementary document, and
reproducibility checklist are uploaded separately as PDFs.

## Designed conformance evidence

The current controlled surface contains 100 allowed cases on a complete
5 x 5 x 4 path/level/context grid and 100 paired non-allow cases. Twenty-five
frozen one-premise mutation operators each use four distinct parents, and every
allowed parent is used exactly once. These are deterministic authored coverage
cases, not prevalence data, random samples, or statistical estimates.

The separate standard-library verifier passes 20,909/20,909 predicates and
binds all 200 input triples, the frozen oracle, 627 indexed artifacts, 600 raw
case-input files, the aggregate metrics, and forbidden-positive checks. The
legacy 31-case surface remains only as a 394/394 regression.

From the package root, the current surface can be checked and its supplementary
tables regenerated with:

```console
python scripts/verify_package_manifest_v14.py
python scripts/verify_validator_ablation_v3_2.py
python scripts/build_validator_ablation_v3_1_supplement_tables.py --repo-root . --paper-root paper
```

The FULL contract has unsafe-positive, valid-exact, and exact-tuple fractions
0.00, 1.00, and 1.00. The B0--B4 rows are deliberately incomplete information
projections. A1 assumes runtime evidence matched without repairing static
mapping, schema, mechanism, or accountant premises; A2 removes only the
vacuity obligation.

## Other evidence boundaries

- WISDM and UCI HAR bind completed runs and exact claim decisions; utility is
  diagnostic, not competitive.
- Sepsis and Backblaze are mapping-only rescans and authorize no privacy pair.
- ExtraSensory records one author-operated source-path completion after five
  failures; it is not independent operation or privacy validation.
- TensorFlow Privacy is a scalar-statement scope comparison, not a quality
  ranking or external-evidence authenticator.
- Retrospective incidents are authenticated development lineage, not an
  independently sampled corpus.

Hashes and separate recomputation expose inconsistency but are not
cryptographic execution attestation. `scripts/privacy_claim_validator.py`
remains the sole public-wording authority.

## Layout and data rights

- `docs/`: normative contract, schemas, registry, truth table, and claim ledger.
- `scripts/`: generators, validators, and separately implemented verifiers.
- `tests/`: package regression tests and coverage notes.
- `reports/`: frozen machine-readable evidence, including all 200 case inputs.
- `paper/`: byte-reproducible generated appendix fragments and provenance.
- `third_party/extrasensory-dp/`: the minimal hash-bound MIT source snapshot;
  no raw ExtraSensory data.
- `MANIFEST.json`: byte length and SHA-256 for every bundled file.

No raw WISDM, UCI HAR, PhysioNet, Backblaze, or ExtraSensory dataset is
redistributed. `NOTICE.md` and `DATA_AND_AUTHORITY.md` state the source,
licensing, and authority boundaries.
