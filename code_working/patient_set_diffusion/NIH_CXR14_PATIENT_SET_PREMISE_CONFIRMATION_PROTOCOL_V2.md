# NIH CXR14 patient-set premise confirmation v2

- Frozen on: 2026-09-03 before computing any outcome on the v2 cohort
- Scope: public-development geometry only; no training, generation, DP, attack, or clinical claim
- Candidate licensed on pass: a bounded public Q=2 Patient-Set Private Diffusion architecture smoke

## Why a new confirmation is required

The v1 four-record cohort formally failed one of five preregistered conditions: the absolute DINO cosine
gap was `0.011060`, below `0.05`.  The same cohort is not re-labelled as a pass and its threshold is not
changed.  V1 nevertheless showed scale-free evidence (AUC, retrieval, and record-label variation), which
licenses one independent confirmation with scale-free ordering criteria fixed before looking at its outcomes.

## Frozen independent cohort

- NIH ChestXray14 K10 `public_development`, PA-only.
- Exclude all patients appearing in the 144-patient public clip calibration, 16-patient UCAN diagnostic,
  and 80-patient premise v1 cohort: 240 unique excluded patients in total.
- Use only patients with exactly two or exactly three available K10 public-development records.  This tests
  the common low-multiplicity setting rather than another high-repeat cohort.
- Select 40 patients in each `(target_patient, record_count)` cell for `(0,2)`, `(0,3)`, `(1,2)`, and
  `(1,3)` by SHA-256 with salt `nih-cxr14-patient-set-premise-confirmation-v2`.
- Total: 160 patients and 400 images.  All eligible records of a selected patient are used.
- No patient, image, threshold, probe, or subgroup is changed after outcome computation.

## Frozen visual probe

- The same locally cached public DINOv2 ViT-B/14 checkpoint as v1, SHA-256
  `0B8B82F85DE91B424ADED121C7E1DCC2B7BC6D0ADEEA651BF73A13307FAD8C73`.
- Native full-field grayscale, replicated to RGB, Lanczos 518 x 518 resize, ImageNet normalization.
- L2-normalized CLS embeddings and cosine similarity.  No feature vector is retained.
- DINO is a generic visual probe, not a medical identity oracle.  A pass cannot distinguish anatomy from
  acquisition/site artifacts and cannot establish clinical coherence.

## Frozen scale-free outcomes

1. Balanced same-patient similarity AUC.  Positives are all within-patient pairs.  Within each of the four
   cells, an equal number of cross-patient pairs is selected by the frozen SHA-256 salt; cell contributions
   are then pooled.
2. The same balanced AUC separately in all four `(target_patient, record_count)` cells.
3. Patient ordering wins.  For each patient, its mean within-patient cosine is compared with its mean cosine
   to all records from other patients in the same cell.  Record the overall and four cell-specific win rates.
4. Global same-patient retrieval Recall@1 and its exact random-candidate chance multiple.  Also compute
   Recall@1 inside each cell and compare it with `(record_count-1)/(cell_images-1)`.
5. Fraction of patients with more than one exact finding-label set and mean within-patient label Jaccard
   distance.  Retrieval@5 and label-conditioned cosine summaries are secondary diagnostics only.

## Frozen confirmation gate

The v2 premise passes only if all five conditions hold:

1. pooled balanced similarity AUC is at least `0.75`;
2. balanced similarity AUC is at least `0.65` in every one of the four cells;
3. patient ordering win rate is at least `0.75` overall and at least `0.60` in every cell;
4. global Recall@1 is at least `10x` exact chance and cell-specific Recall@1 is at least `5x` exact chance
   in every cell;
5. at least `30%` of patients have multiple exact finding-label sets and mean within-patient label Jaccard
   distance is at least `0.10`.

All five conditions are conjunctive.  Failure is retained without subgroup removal or threshold relaxation.
A pass licenses only a public Q=2 architecture/forward-backward smoke.  It is not evidence that a SetAdapter
improves generation, that patient-DP has acceptable utility, or that synthetic sets are unlinkable.
