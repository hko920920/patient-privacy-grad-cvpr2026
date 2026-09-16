# NIH CXR14 patient-set premise diagnostic v1

- Frozen on: 2026-09-03 before computing the selected cohort's DINOv2 outcomes
- Scope: public-development data geometry only; no training, DP claim, generation, attack, or clinical claim
- Candidate direction: **Patient-Set Private Diffusion (PSPD)**

## Question

Does a public chest-X-ray cohort actually contain the two signals required for set-valued patient-private
generation: (1) stable within-patient visual structure and (2) nontrivial record-to-record finding variation?

If repeated records are visually no more related than records from different patients, or if they are near
duplicates with no finding variation, jointly generating a synthetic patient set has no empirical premise.

## Frozen sample

- NIH ChestXray14 K10 `public_development`, PA-only.
- Patients must have at least four selected records.
- Exclude all 144 clip-calibration patients and all 16 UCAN diagnostic patients.
- Select 40 target and 40 control patients by SHA-256 with salt
  `nih-cxr14-patient-set-premise-v1`.
- Select four records per patient by a separate SHA-256 ranking.
- Total: 80 patients and 320 images.

## Frozen visual probe

- Public DINOv2 ViT-B/14 weights already cached locally.
- Weight SHA-256: `0B8B82F85DE91B424ADED121C7E1DCC2B7BC6D0ADEEA651BF73A13307FAD8C73`.
- Full-field grayscale-to-RGB Lanczos resize to 518 x 518; ImageNet normalization.
- L2-normalized CLS embedding; cosine similarity only.
- The generic DINOv2 probe is deliberately not called a medical identity model.  A later paper would require
  a separate radiograph re-identification evaluator and external validation.

## Outcomes

1. Mean/median cosine for all 480 within-patient image pairs.
2. Cross-patient comparisons restricted to the same target/control stratum.
3. Exact-finding-string comparisons to show whether similarity is explained only by pathology labels.
4. Same-patient retrieval Recall@1 and Recall@5 among the other 319 images.
5. Fraction of patients with more than one exact finding-label set, mean number of distinct label sets, and
   mean within-patient label Jaccard distance.
6. Balanced similarity AUC using all within-patient pairs and an equal number of outcome-blind hash-selected
   cross-patient/same-target pairs.

No feature vector is retained.

## Frozen premise gate

The patient-set direction retains data support only if all conditions hold:

1. within-patient mean cosine exceeds cross-patient/same-target mean cosine by at least `0.05`;
2. balanced same-patient similarity AUC is at least `0.75`;
3. Recall@1 is at least five times its exact random-candidate chance;
4. at least `30%` of patients have more than one exact finding-label set among their four records;
5. mean within-patient label Jaccard distance is at least `0.10`.

A pass licenses architecture/protocol design only.  It does not establish that a patient-set diffusion model
can be trained well, that the stable signal is anatomy rather than acquisition artifact, that generated sets
are clinically coherent, or that patient-level DP has acceptable utility.

