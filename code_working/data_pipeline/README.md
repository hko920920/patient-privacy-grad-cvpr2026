# Patient-manifest and real-image loader gate

This directory materializes the frozen ISIC 2020 patient partition and K2/K5/K10
contribution caps before any generator training.

```powershell
$python = "..\base_gate\.venv\Scripts\python.exe"
& $python .\build_isic_manifests.py
& $python .\build_isic_manifests.py --verify
& $python .\verify_real_image_loader.py
& $python .\verify_ppmark_exact_loader.py
& $python .\acquire_isic_k5.py
& $python .\build_nih_cxr14_manifests.py
& $python .\build_nih_cxr14_manifests.py --verify
& $python .\smoke_nih_cxr14_official_archive.py
& $python .\smoke_nih_cxr14_official_archive.py --verify-only
& $python .\acquire_nih_cxr14_union.py
& $python .\acquire_nih_cxr14_union.py --verify-only
& $python .\verify_nih_cxr14_union_independent.py
& $python .\verify_nih_cxr14_sd21_ppmark_interface.py
& $python .\verify_nih_cxr14_sd21_interface_independent.py
```

The detailed CSVs under `_data/derived/isic2020_v2_split_v1/` contain patient
mappings and are local-only evidence. The report under
`_reports/isic2020_split_v1_001/` contains aggregate counts and SHA-256
commitments suitable for the dissertation evidence package.

The loader smoke test uses eight real JPEGs and the exact frozen SD-v1.4 VAE.
It proves only manifest/pixel/latent interface compatibility and GPU memory
feasibility. It is not DP training, a privacy claim, or a utility result.

By default, the PP-Mark exact-loader test reproduces the historical SD-v1.4
fallback gate. It also accepts explicit model/revision/variant/hash arguments.
The selected SD 2.1 revision passed this same fail-closed test and is recorded in
`../SD21_BASE_MODEL_GATE_DECISION.md`. Existing SD2.1/SDXL thresholds remain
historical and must be recalibrated on the eventual medical LoRA checkpoint and
release-image protocol.

`acquire_isic_k5.py` refuses a changed K5 manifest, validates patient-partition isolation, downloads
JPEGs through the ISIC Archive S3 endpoint with retry/range-resume support, verifies every image,
and emits a local detailed inventory plus an aggregate hash-committed report. Use `--verify-only`
to recheck a completed acquisition without network fallback.

`build_nih_cxr14_manifests.py` implements the later X-ray-core pivot. It refuses metadata that does
not match the exact NIH Box byte/SHA locks, keeps PA views only, forms a target-patient-enriched 1:1
method cohort, preserves the official test patient boundary, creates label-independent nested
K2/K5/K10 caps, and emits local-only mappings plus an aggregate report. This cohort is not an
estimate of hospital prevalence and the weak labels are not definitive diagnoses. It also freezes
the full PA official-test census as an overlapping, secondary distribution-sensitivity set; it is
not an independent second test and cannot be used for tuning.

`smoke_nih_cxr14_official_archive.py` is the narrow official-PNG source gate. It locks the current
12-archive Box catalog and the K10/census manifests, range-resumes one exact archive, verifies the
provider byte count and SHA-1, rejects unsafe or metadata-unknown tar members, and extracts only a
deterministic 24-image diversity sample. `--verify-only` requires the archive and outputs to exist
locally and reproduces the private inventory and aggregate report byte-for-byte. It is not the full
42,423-image acquisition and does not run a model.

`acquire_nih_cxr14_union.py` processes all 12 locked archives one at a time. It commits each
archive's full source-name list and selected contribution before removing the verified temporary
archive, making the run restartable without storing 41.98 GiB of archives beside the extraction.
It materialized exactly 42,423 PNGs and preserves native bytes. Native `RGBA` is accepted only when
R/G/B are exactly equal and alpha is fully opaque; actual color or transparency fails closed.
`--verify-only` works offline from archive commits and re-hashes the full selected population.

`verify_nih_cxr14_union_independent.py` deliberately imports none of the acquisition/source-smoke
implementation. It independently reconstructs the frozen union and archive mapping and re-hashes,
re-decodes, and rechecks the channel semantics of all 42,423 PNGs. The verifier compares finding
labels as exact sets plus the manifest builder's canonical sorted serialization; it does not require
the irrelevant source-token order to remain unchanged.

`nih_cxr14_model_input.py` is the frozen on-demand X-ray input adapter. It accepts only 1024-square
PNG sources, maps native `L` and fully opaque grayscale-equivalent `RGBA` to one grayscale channel,
keeps the full field, applies Pillow-Lanczos resize, replicates to RGB, and normalizes to the SD
`[-1,1]` interface. It performs no crop, pad, augmentation, histogram transform, per-image
standardization, or private-statistic fit. Its weak-label prompt mapping excludes patient identity,
age, sex, and cohort-enrichment flags.

`verify_nih_cxr14_sd21_ppmark_interface.py` selects eight patient-distinct public-development cases
by a result-independent hash rule and checks P256/P512 preprocessing, exact SD 2.1 VAE mode
encoding, two identical real-image rank-8 LoRA gradients without any optimizer step, and PP-Mark
two-step DDIM inversion of the same committed pixels. It writes no checkpoint and performs no DP,
calibration, attack, receipt, or release. `verify_nih_cxr14_sd21_interface_independent.py` imports
neither implementation and independently reconstructs the sample and all preprocessing
commitments. The decision record is `../XRAY_SD21_PPMARK_INTERFACE_GATE.md`.
