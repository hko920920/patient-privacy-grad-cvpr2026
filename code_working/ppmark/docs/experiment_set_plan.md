# Experiment Set Plan (SP1 + Score/Proof Gate)

## Context
- Scope: compile the next experiment sets based on recent work logs and current code.
- Current detection gate: `score >= tau` AND SP1 receipt verification succeeds.
- Merkle-root recomputation and payload decode are diagnostic only (not gate conditions).
- SP1 is the only supported ZKP backend for current evaluation; legacy backends are out of scope.

## Sources Reviewed
- `docs/work_log.md` (robustness runs, IQA sweeps, neg calibration, oracle attacks, quality dataset build).
- `docs/worklog_2025-12-28.md` (Muller/Zhao wrappers, proof-backed dataset builder, HardPass/AcceptPass rules).
- Attack and evaluation scripts:
  - `scripts/run_muller_attacks.py`, `scripts/run_zhao_attacks.py`, `scripts/eval_attack_manifest.py`.
  - `scripts/attack_oracle_forge.py`, `scripts/attack_oracle_remove.py`.
  - `scripts/build_pos_dataset.py`, `scripts/build_muller_forgery_plan.py`, `scripts/build_forgery_pairs.py`.
  - `scripts/run_geom_robustness.py`.
  - `scripts/run_sp1_sample_sweep.sh`, `tables/sp1_sweep_results.md`.
  - `scripts/build_quality_manifest.py`, `scripts/generate_quality_images.py`, `scripts/eval_kid_clip.py`.
- External attacker code: `external/semantic-forgery/` (Imprint-Forgery/Removal/Reprompt).

## Global Inputs (current defaults from logs)
- Model: `Manojb/stable-diffusion-2-1-base` (SD2.1 proxy).
- Resolution: 512px (latent 64x64) unless overridden.
- Sample count: 1000 (alpha=4.0 for most detection runs).
- Tau file: `datasets/thresholds/tau_fpr1_alpha4_sd21_fast.json` (tau_score ~2.3721).
- Configs: `config.json` (baseline), `config_alpha4_k1000.json` (alpha=4.0), `config_alpha3_k1000.json` (alpha=3.0).

---

## Experiment Sets to Build

### 1) Imprint Forgery Attack (Direct)
**Goal:** Forge PP-Mark detection by transferring a watermark imprint from a reference WM image onto a cover image.

**Data Inputs**
- Proof-backed WM images + metadata from `scripts/build_pos_dataset.py`.
- Cover images from `datasets/neg_calib_1000_sd21/manifest.jsonl` (or a curated eval split).

**Plan Build**
- Build reference WM set:
  - `scripts/build_pos_dataset.py --config config_alpha4_k1000.json --manifest <wm_manifest> --output <out_dir> --secret-hex <ppmark_secret>`
- Build forgery plan:
  - `scripts/build_muller_forgery_plan.py --pos-manifest <wm_pos_manifest> --neg-manifest <neg_manifest> --output <plan.jsonl>`

**Attack Run**
- `scripts/run_muller_attacks.py --forgery-plan <plan.jsonl> --skip-reprompt --skip-removal`
  - Calls `external/semantic-forgery/run_imprint_forgery.py` internally.

**Evaluation**
- `scripts/eval_attack_manifest.py --manifest <forgery_manifest> --config config_alpha4_k1000.json --tau-file <tau_file> --verify-proof`

**Metrics**
- `score_pass`, `proof_ok`, `accept_pass` (primary).
- Optional: PSNR/LPIPS from the underlying attack logs if captured.

**Outputs**
- `outputs/attacks/muller/forgery/<split>/images` + `scores.csv` + `summary.json`.

---

### 2) Imprint Forgery Attack (Transfer)
**Goal:** Transfer an imprint forged using a different watermark method (cross-method transfer) onto PP-Mark covers to measure vulnerability to imprint reuse.

**Data Inputs**
- Pairing index: `pairing_index.jsonl` with cover image paths + WM prompt/seed pairs.
- Method manifests from a multi-method prep dir (e.g., `outputs/attacks/muller_forgery_report_prep/methods/<method>/<split>/manifest.jsonl`).

**Plan Build**
- `scripts/build_forgery_pairs.py --pairing-index <pairing_index.jsonl> --methods-root <methods_root> --methods tree_ring,gaussian_shading,ringid,stable_signature,trustmark_hidden,pp_mark --output <forgery_pairs.jsonl>`

**Attack Run**
- Use the resulting `forgery_pairs.jsonl` as the forgery plan for `scripts/run_muller_attacks.py`.

**Evaluation**
- Same as direct: `scripts/eval_attack_manifest.py --tau-file <tau_file> --verify-proof`.

**Metrics**
- `accept_pass` rate by method (transfer success).
- Score distributions per method.

**Outputs**
- `outputs/attacks/muller/forgery/<split>` per method + summary JSON.

---

### 3) Adaptive Forgery Attack (Score Maximization)
**Goal:** Adaptive black-box forging to maximize the PP-Mark score (SPSA). This is the "adaptive forgery" track.

**Data Inputs**
- Cover images (negative set or arbitrary images) + a fixed metadata path (Alice proof) OR per-row metadata.
- Tau file: `datasets/thresholds/tau_fpr1_alpha4_sd21_fast.json`.

**Attack Run**
- `scripts/attack_oracle_forge.py --manifest <covers_manifest> --metadata-path <fixed_meta.json> --config config_alpha4_k1000.json --tau-file <tau_file> --steps <N> --epsilon <eps> --step-size <eta> --psnr-min 30 --lpips-max <optional>`

**Metrics**
- `score_pass` and best-score trajectory; quality gates (PSNR/LPIPS).
- Success rate vs. query budget (steps).

**Outputs**
- `outputs/attacks/ppmark_adaptive/` (per-image folders + CSV summary).

---

### 4) Removal Attack

#### 4.1 Regeneration (Zhao/DiffWMAttacker)
**Goal:** Regenerate watermarked images with a different model to suppress PP-Mark detection.

**Data Inputs**
- Proof-backed WM manifest (pos split) + metadata root.
- Tau file for evaluation.

**Attack Run**
- `scripts/run_zhao_attacks.py --manifest <wm_manifest> --output-root outputs/attacks/zhao --strengths weak,med,strong --run-eval --verify-proof --tau-file <tau_file>`
  - Strength mapping: weak=30, med=60, strong=100 steps.

**Metrics**
- `accept_pass` / `proof_ok` / `score_pass` per strength.

**Outputs**
- `outputs/attacks/zhao/<strength>/<split>/images`, `scores.csv`, `summary.json`.

#### 4.2 Imprint Removal (Muller)
**Goal:** Remove watermark imprint from a reference WM image (Imprint-Removal).

**Attack Run**
- `scripts/run_muller_attacks.py --skip-reprompt --skip-forgery` (removal only).

**Evaluation**
- `scripts/eval_attack_manifest.py --tau-file <tau_file> --verify-proof`.

#### 4.3 Steg Removal (Ongoing)
**Status:** Not wired for PP-Mark yet. Candidate directions:
- Integrate a steganographic denoising/removal baseline (external TrustMark or hidden watermark pipelines) into PP-Mark eval harness.
- Define consistent input/output manifests and use `scripts/eval_attack_manifest.py` for score/proof evaluation.

#### 4.4 Oracle Removal (Upper Bound)
**Goal:** Black-box score-query removal as a worst-case bound.

**Attack Run**
- `scripts/attack_oracle_remove.py --manifest <wm_manifest> --tau-file <tau_file> --epsilon-grid ... --step-size-grid ... --budget 200/1000 --psnr-min 30 --lpips-max 0.15`

**Metrics**
- Success rate under fixed query budgets and quality constraints.
- Best-of-K vs. fixed-parameter comparisons.

---

### 5) Image Transformation Robustness
**Goal:** Evaluate detection under geometric/compression perturbations.

**Primary Attack Set (from logs)**
- Rotation 75 degrees.
- JPEG quality 25.
- Random crop+scale (keep ratio 0.75).
- Gaussian blur (8x8 kernel).
- Gaussian noise (sigma=0.1).
- Brightness jitter (factor 0.6).

**Tools**
- `scripts/run_geom_robustness.py` (quick harness).
- `scripts/eval_attack_manifest.py` for standardized scoring with `tau`.

**Metrics**
- `score_pass`, `accept_pass` rates with sync search on/off.
- Per-attack score distributions.

---

### 6) ZKP Performance (SP1)
**Goal:** Prover/verification timing across sample counts and proof modes.

**Tools**
- `scripts/run_sp1_sample_sweep.sh` (sample_count sweep, metadata-only verification).
- `tables/sp1_sweep_results.md` for aggregated results.
- `scripts/run_sp1_sweetspot.py` for score distribution vs. k.

**Metrics**
- `sp1_prover_sec`, `receipt_verify_sec`, proof size, and sample_count vs. runtime.

**Outputs**
- `outputs/sp1_sweep/` per sample count.
- `tables/sp1_sweep_results.md` updated with timing summaries.

---

### 7) Generation Distributional Fidelity
**Goal:** Measure distribution shift between watermarked and clean generations.

**Data Inputs**
- Watermarked: `outputs/quality_a4_1k/images`.
- Clean: `outputs/quality_a0_1k/images`.
- Manifests: `outputs/quality_a4_1k/manifest.jsonl`, `outputs/quality_a0_1k/manifest.jsonl`.

**Evaluation**
- `scripts/eval_kid_clip.py` (KID + CLIP), currently blocked by torchvision `aux_logits` issue; requires a small loader fix before running.
- Optional: BRISQUE/NIQE summaries via existing IQA sweeps (see `scripts/alpha_iqa_sweep.py`).

**Metrics**
- KID mean/std; CLIP score mean/std across watermarked vs. clean.
- IQA deltas (optional).

---

## Open Dependencies / Notes
- Proof-backed WM datasets are required for `proof_ok` gating (use `scripts/build_pos_dataset.py`).
- Tau calibration depends on the negative calibration set; current file is `datasets/thresholds/tau_fpr1_alpha4_sd21_fast.json`.
- Any run using image-based detection must include `--tau` or `--tau-file` and an image/latent path.

---

## Section 4 Outline (Paper Draft)

### 4.1 Experimental Setup
- Model: SD2.1 proxy (`Manojb/stable-diffusion-2-1-base`), 512px, latent 64x64.
- Watermark: alpha=4.0, sample_count=1000 (unless specified).
- Detection gate: `score >= tau` AND SP1 receipt verification success.
- Tau: `datasets/thresholds/tau_fpr1_alpha4_sd21_fast.json`.

### 4.2 Imprint Forgery Attack (Direct + Transfer)
- Direct: Muller imprint-forgery on PP-Mark images.
- Transfer: cross-method imprint transfer using `build_forgery_pairs.py`.
- Report: `accept_pass` by attack variant and method (transfer).

### 4.3 Adaptive Forgery Attack (PP-Mark target)
- Variant A: per-image metadata (unique proof per target).
- Variant B: fixed metadata (proof reuse across covers).
- Report: success vs. query budget with PSNR/LPIPS gates.

### 4.4 Removal Attack
- Regeneration: Zhao/DiffWMAttacker (weak/med/strong).
- Imprint removal: Muller imprint-removal.
- Steganalysis: ongoing (not yet wired into PP-Mark harness).

### 4.5 Image Transformations (6 types)
- Rotation (75 deg), JPEG(25), crop+scale(0.75), blur(8x8), noise(sigma=0.1), brightness(0.6).
- Report: score and accept rates with sync search.

### 4.6 Public Verification ZKP Performance
- SP1: sample_count {600, 1000, 1500} timing/verification.
- Optional legacy appendix: RISC0 1000 sample comparison (if requested).

### 4.7 Generation Distributional Fidelity (KID)
- Watermarked vs. clean: `outputs/quality_a4_1k` vs `outputs/quality_a0_1k`.
- KID (and optional CLIP) once `eval_kid_clip.py` loader is fixed.
