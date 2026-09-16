# NIH CXR14 SetAdapter held-out context-signal diagnostic v1

- Frozen on: 2026-09-03 before selecting the cohort or computing model outcomes
- Prerequisites: patient-set premise confirmation and Q=2 architecture smoke passed
- Scope: public non-DP learning-signal diagnostic; not generation, privacy, or clinical evidence

## Question

After bounded training on correctly paired patient records, does the SetAdapter improve denoising of a fixed
primary X-ray more when its companion is from the same patient than when the companion is replaced by a
finding-matched record from another patient?

This separates a patient-context signal from a generic capacity effect.  If correct and shuffled companions
behave the same on held-out patients, full generation training is not licensed.

## Frozen cohort

- NIH ChestXray14 K10 `public_development`, PA-only.
- Exclude the 144 clip-calibration, 16 UCAN, 80 premise-v1, and 160 premise-confirmation-v2 patients:
  400 unique patients in total.
- Use only remaining patients with exactly two public-development records.
- Within each target stratum, rank patients by SHA-256 salt
  `nih-cxr14-setadapter-context-signal-v1`.
- Per target stratum choose the first 32 patients for training and the next 16 for validation.
- Total: 64 training patients/128 images and 32 validation patients/64 images.  Train and validation patients
  are disjoint; no DINO score or model outcome enters selection.

## Frozen model and optimization

- Exact local SD 2.1 base at revision `0094d483a120f3f33dafbd187ea4aa60d10de75c` and locked NIH
  preprocessing/VAE mode latents/text prompts.
- Freeze every original UNet parameter.  Insert the architecture-smoke SetAdapter at the mid block and train
  only its 463,872 fp32 parameters.  No LoRA is added in this diagnostic so any correct-vs-shuffled effect is
  attributable to the set branch; full LoRA matching is reserved for the next gate.
- 192 AdamW steps, learning rate `1e-3`, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `0.01`.
- Each step contains one target-0 and one target-1 patient, Q=2 each.  Hash permutations give every training
  patient six exposures.  Each record receives an independent uniform diffusion timestep and independent
  Gaussian corruption noise derived from the public step hash.
- Mean the four record losses; clip the non-private SetAdapter gradient norm to `1.0`; take one optimizer step.
- No checkpoint, latent, gradient, optimizer state, prediction, or adapter is retained.

## Frozen held-out evaluation

- Four public evaluation banks; both records alternate as the primary across banks.
- For every validation patient/bank, keep the primary record, its timestep/noise, target, and text exactly
  fixed across three arms:
  1. `correct`: the other record from the same patient is the companion;
  2. `shuffled`: a record from a different validation patient in the same target stratum;
  3. `bypass`: the correct input pair with SetAdapter bypassed.
- For `shuffled`, prefer candidates whose exact finding-label set equals the correct companion.  If none
  exists, maximize finding-label Jaccard similarity; break ties only by the frozen SHA-256 salt.
- Score denoising MSE for the primary record only.  Correct/shuffled/bypass comparisons are paired on the
  identical primary input.
- Run the same evaluation before training.  Zero initialization must make all three arms identical then.

## Frozen gate

All conditions must hold:

1. all 192 optimizer steps and all evaluation cells are finite; SetAdapter parameters change and no other
   model parameter changes;
2. before training, correct/shuffled/bypass primary losses have maximum absolute difference at most `1e-7`;
3. after training, aggregate `correct / shuffled` primary-loss ratio is at most `0.995` and correct wins at
   least `60%` of the 128 patient-bank cells;
4. `correct / shuffled < 1.0` separately in target-0 and target-1 validation strata;
5. aggregate `correct / bypass` primary-loss ratio is at most `0.995`;
6. the SetAdapter remains exactly permutation-equivariant and singleton-identical after training at the pure
   module level, with max absolute errors at most `1e-5`;
7. peak allocated CUDA memory is below `7.5 GiB`.

Failure is retained without removing patients, banks, target strata, or changing thresholds.  A pass only
licenses a parameter/UNet-call-matched public LoRA generation experiment.  It does not demonstrate image
quality, patient-set coherence, patient-DP utility, privacy protection, or clinical validity.
