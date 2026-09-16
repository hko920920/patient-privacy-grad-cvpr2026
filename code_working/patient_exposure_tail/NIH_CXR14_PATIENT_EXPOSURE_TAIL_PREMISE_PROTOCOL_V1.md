# NIH CXR14 patient-exposure-tail endpoint premise protocol v1

- Freeze date: 2026-09-03 (Asia/Seoul)
- Stage: public, non-private, feature-only diagnostic before any generator training
- Decision unit: all eligible patients, never images
- Output scope: aggregate metrics only; patient/image identifiers remain local

## 1. Question

This diagnostic asks a deliberately narrow question:

> After equalizing every patient's retrieval opportunity to exactly one query and one true gallery
> image, do two fixed feature encoders expose a reproducible distribution of patient-specific chest
> X-ray re-identification difficulty, including both a readily exposed tail and a difficult tail?

The purpose is to decide whether a later differentially private generator comparison can reasonably
add a patient-exposure survival endpoint. This run is not a generator membership attack, does not
estimate a patient's membership-AUC, and provides neither differential privacy nor clinical identity
evidence.

## 2. Fixed population and pairing

- Source: frozen NIH ChestXray14 K10 manifest, `public_development` partition, PA view only.
- Manifest SHA-256:
  `2D749FB7B70823114A69FD55921277B0D0FF2F9AEC8FECD239159C553F3A0C06`.
- Include every patient with at least two K10 records: expected 886 patients.
- No patient is selected or removed using an image feature, similarity, rank, or model outcome.
- Order records by `(followup_no, image_id)`.
- Query: the first ordered record. True gallery: the last ordered record.
- Thus every patient contributes exactly one query, one true gallery item, and one equal-weight
  endpoint value regardless of whether they have 2 or 10 K10 records.
- `followup_no` is only an ordering field. It is not elapsed time or a clinical progression label.

Expected K10 record-count cells are frozen as:

| K10 records | Patients |
|---:|---:|
| 2 | 290 |
| 3 | 171 |
| 4 | 117 |
| 5 | 79 |
| 6 | 39 |
| 7 | 27 |
| 8 | 42 |
| 9 | 24 |
| 10 | 97 |

For descriptive subgroup reporting, record count is mapped to `2`, `3`, `4-5`, or `6-10` and
crossed with patient sex and the frozen target-patient flag. All 16 expected cells contain at least
12 patients. No subgroup metric selects the overall conclusion.

## 3. Fixed negative control

For each query, construct exactly one different-patient negative gallery item. Within each
`sex x target flag x record-count bucket` cell, order patients by SHA-256 of
`nih-cxr14-patient-exposure-tail-v1-negative|patient_id` and cyclically shift the true gallery by
one position. This is a fixed-point-free derangement that preserves the three registered metadata
fields. It does not match exact pathology labels and must not be described as a causal identity
control.

## 4. Encoders and preprocessing

Both encoders must pass; neither may be selected after observing results.

1. Generic DINOv2 ViT-B/14 using the already pinned local weights with SHA-256
   `0B8B82F85DE91B424ADED121C7E1DCC2B7BC6D0ADEEA651BF73A13307FAD8C73` and the existing
   deterministic 518-pixel DINO preprocessing.
2. `microsoft/rad-dino` revision `110cbc18d5133582e320b43d53bf5c44e410c936`, with pinned
   weight/config/processor SHA-256 values
   `DBFB9F54459C38773505DE64A6AB7807BDCB392610FE1E697166342E43FB91AE`,
   `89DAF9751D9576D586DEDF9543C1083211611FA3A36908DB7A799B3CE7C68EDE`, and
   `C537FC995C30E2353F07253899618D60E9EAE3D5F82473778602C007C6523B56`.

Features are L2-normalized. Matrix TF32 and cuDNN TF32 are disabled; deterministic cuDNN is
enabled. The first eight features are replayed and must agree within `1e-5`. RAD-DINO was trained
using NIH ChestXray14, so it cannot be the sole supporting encoder.

## 5. Metrics

For each encoder, compute:

1. cosine similarity of each query to its true gallery item and to its registered matched-negative
   gallery item;
2. tie-aware rank AUC distinguishing the 886 true pairs from the 886 negative pairs;
3. worst-tie retrieval rank of the true gallery item among all 886 gallery items, defined as
   `1 + count(impostor_similarity >= true_similarity)`;
4. Recall@1, Recall@5, Recall@10, median rank, the fraction with rank greater than 10, and the
   true-minus-maximum-impostor cosine margin distribution;
5. an empirical survival table of the patient margin at the fixed thresholds
   `{-0.05,-0.02,-0.01,0,0.01,0.02,0.05}`;
6. descriptive results for all 16 registered metadata cells.

Across encoders, compute the Spearman correlation of the 886 patient margins. Report 5,000
patient-bootstrap percentile intervals for AUC, Recall@1, rank-greater-than-10 fraction, and the
cross-encoder Spearman correlation. Seeds are domain-separated SHA-256 integers under
`nih-cxr14-patient-exposure-tail-v1-bootstrap`.

## 6. Conjunctive gate fixed before feature extraction

The endpoint premise passes only when all of the following hold:

1. each encoder's true-versus-matched-negative AUC is at least `0.80` and its bootstrap lower bound
   is greater than `0.75`;
2. each encoder's Recall@1 is at least `0.15` and its bootstrap lower bound is greater than `0.10`;
3. for each encoder, at least `0.10` of patients have true-gallery rank greater than 10, showing
   that the endpoint is not saturated at the exposed end;
4. for each encoder, the 90th margin percentile is positive and the 10th margin percentile is
   negative, so both sides of the exposure boundary are populated without a chosen threshold;
5. the cross-encoder patient-margin Spearman correlation is at least `0.20`, with bootstrap lower
   bound greater than `0.10`;
6. peak allocated CUDA memory remains below 7.5 GiB and all integrity checks pass.

Status is `PASS_PATIENT_EXPOSURE_TAIL_ENDPOINT_PREMISE` only for the complete conjunction;
otherwise it is `FAIL_PATIENT_EXPOSURE_TAIL_ENDPOINT_PREMISE`. A failure is retained and cannot be
rescued by changing encoders, pair ordering, subgroups, thresholds, or bootstrap seeds in v1.

## 7. Interpretation boundary and next action

A pass licenses only a separate, pre-generation addendum that specifies a patient-biometric
exposure attack for the existing K10 generator matrix. It does not authorize the 16--21 hour K5
run, establish a novel method, prove privacy, or show that any generated image leaks identity.

A later generator endpoint must distinguish:

- formal add/remove-one-patient DP from record-level DP and group conversion;
- aggregate membership success from patient-specific biometric exposure;
- a one-model exposure-score survival function from the multi-target-model patient-specific MIA AUC
  used in prior privacy-audit work.

No feature matrix, encoder checkpoint, optimizer, gradient, latent, or generated image is retained.
The local selection file may retain identifiers only to make the pairing auditable.
