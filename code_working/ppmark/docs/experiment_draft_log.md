# Experiments Draft Worklog (PP-Mark)

## 2025-02-14

### Request summary
- User asked to rebuild the Experiments section from the top, one subsection at a time.
- Emphasize SP1-only verification and the score+proof acceptance rule.
- Align baselines and metrics with the current evaluation set.

### Action taken
- Drafted a cleaned Experimental Setup subsection (model, watermark params, SP1 verification, datasets, baselines, calibration, metrics).
- Removed placeholder text and aligned terminology (SP1 receipt, score threshold tau).

### Pending
- Draft Imprint Forgery Attack subsection next (direct + transfer; reprompt set if needed).
- Continue remaining experiment subsections sequentially.

## 2025-02-14 (continued)

### Fact-checking sources reviewed
- `config_alpha4_k1000.json` (alpha, sample count, SD2.1 proxy model id, RS params, SP1 backend, SHA-256 sampling).
- `datasets/neg_calib_1000_sd21/manifest.jsonl` and `prompts.json` (clean calibration set, dataset source, DDIM 50, guidance 7.5, 512x512).
- `datasets/thresholds/tau_fpr1_alpha4_sd21_fast.json` (tau_score and FPR target).
- `outputs/calib_pos_alpha4_k1000/metadata.json` (alpha, scheduler, steps, guidance, grid size).
- `outputs/quality_a4_1k/manifest.jsonl` and `outputs/quality_a0_1k/manifest.jsonl` (1k clean/wm quality sets).
- `outputs/pos_wm_a4_k1000_sd21_200_combined/manifest.jsonl` (200-row proof-backed PP-Mark set).
- `outputs/attacks/muller_forgery_report_prep/README.md` (prompt/covers protocol for forgery report prep).
- `docs/work_log.md` (negative calibration run details and SD2.1 proxy note).

### Verified parameters (for Experimental Setup)
- Model/scheduler: SD2.1 base, DDIM scheduler, 50 steps, guidance scale 7.5, 512x512 (latent 64x64x4).
- PP-Mark config: alpha=4.0, sample_count=1000, RS(64,32), msg_bits=256, SP1 backend with SHA-256 sampling.
- Calibration set: 1,000 clean SD2.1 images from `datasets/neg_calib_1000_sd21` (Gustavosta/Stable-Diffusion-Prompts).
- Tau: `datasets/thresholds/tau_fpr1_alpha4_sd21_fast.json` with tau_score=2.372129... (p99, FPR target 1%).
- Attack/eval sets referenced in repo: PP-Mark proof-backed 200-image set; quality comparison sets of 1,000 clean and 1,000 watermarked images.

### Corrections vs earlier draft
- Prompt source is Gustavosta/Stable-Diffusion-Prompts (not MS-COCO) for the standard SD2.1 prompt pools.
- Evaluation sizes seen in repo are 200 (proof-backed PP-Mark) and 1,000 (quality sets); no 5,000-image SD2.1 eval set found.

### Direct forgery PSNR (TR/GS)
- Computed PSNR for direct imprint forgery outputs in `outputs/attacks/muller_forgery_report_prep/forged/{TR,GS}/step{050,100,150}`.
- PSNR is measured between the forged image and the corresponding cover image; COCO covers are resized to 512x512 with PIL default resample to match the attack's cover resize in `run_imprint_forgery.py`.
- Results (mean +/- std, n=100 per step):
  - TR: step50 24.3707 +/- 3.3115, step100 22.9144 +/- 3.1389, step150 21.9925 +/- 3.0197.
  - GS: step50 24.3174 +/- 3.3078, step100 22.8575 +/- 3.1341, step150 21.9389 +/- 3.0275.
- Combined across all steps (n=300): TR 23.0926 +/- 3.3072, GS 23.0379 +/- 3.3069.

## 2025-02-14 (continued)

### User questions addressed
- Explained why our PSNR values differ from Muller and whether 1% FPR calibration affects PSNR.
- Clarified proxy vs target model differences and implications for comparability.

### Clarifications recorded
- Our PSNR is computed between forged images and their covers (512x512; COCO covers resized with PIL default). It measures cover fidelity; it is not affected by the 1% FPR calibration (which only sets the detector threshold tau).
- Muller uses a proxy SD2.1 attacker against multiple target models (SD2.1, SDXL, PixArt, FLUX), whereas our direct TR/GS sanity-check uses proxy=target=SD2.1. Differences in target model distribution, cover pool, preprocessing, and attack hyperparameters can shift PSNR by several dB, so absolute PSNR values are not directly comparable.
- The trend is consistent across both: increasing attack steps maintains or increases vulnerability (TPR), while PSNR decreases as the optimization proceeds.

### Pending
- Decide whether to include PSNR in the direct-attack table (with a caveat about non-comparability) or re-run under a Muller-matched protocol.

## 2025-02-14 (continued)

### Cover pool rationale (inferred)
- The repo does not explicitly state why the cover pool mixes COCO reals and SD2.1 clean generations.
- Likely intent is evaluation fairness: imprint forgery targets arbitrary covers, so mixing real + generated covers reduces distribution bias and tests transfer across both domains.
- This is not a training choice; it is a test set construction decision.

## 2025-02-14 (continued)

### Adaptive histogram asset rename + retitle
- Created a new adaptive histogram image with an overlaid title: “Adaptive Score Distribution (Instance-Specific Targeting)”.
- New assets: `outputs/attacks/muller_forgery_report_prep/figures/adaptive_score_hist_tau_shaded_instance.png` and `.pdf` (source was `1adaptive_score_hist_tau_shaded_per_image.png`).
- Updated `outputs/attacks/muller_forgery_report_prep/report_0_5_ko.md` to replace “per-image” phrasing with “instance-specific” in the adaptive section, and to reference the new figure filename and labels.

## 2025-02-14 (continued)

### Adaptive histogram title fix (no overlay bar)
- Rebuilt `adaptive_score_hist_tau_shaded_instance.png/.pdf` by editing the original histogram’s title text region (no overlay banner) to read “Adaptive Score Distribution (Instance-Specific Targeting)”.
