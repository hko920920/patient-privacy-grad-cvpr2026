# Work Log

## 2025-02-15
- Implemented sync/extraction utilities with coarse alignment, deterministic DDIM inversion placeholder, and spread-spectrum demodulation plus RS decode diagnostics (`src/ppmark_v03/sync.py`).
- Added bit packing helper `bits_to_bytes` and exported it for recovery paths (`src/ppmark_v03/payload.py`, `src/ppmark_v03/__init__.py`).
- Added round-trip extraction tests covering clean and noisy/resized latents (`tests/test_sync.py`).
- Updated implementation plan to note extraction scaffold completion (`pp_mark_v0.3_implementation_plan.md`).

## 2025-02-16
- Added minimal DDIM encode path with pluggable epsilon predictor and deterministic schedule (`ddim_encode`, `DDIMSchedule`, `make_ddim_schedule`) while keeping backward-compatible normalization in `ddim_invert` (`src/ppmark_v03/sync.py`).
- Extended extraction entrypoint to pass DDIM options and added unit tests for the encode path (`tests/test_ddim_encode.py`).

## 2025-02-17
- Switched default configs to SP1 backend.
- Added SP1 ctx_hash/binding flow (`ctx_hash(prompt_hash, model_id, seed, w, h, usecase_tag, user_tag)`; binding = SHA256("ctx_hash"||ctx_hash||secret)); public inputs/metadata now carry ctx_hash/model_id/resolution/sample_count/lut_path/lut_hash.
- Updated SP1 guest/host to verify ctx_hash-based binding and accept expanded public inputs; `cargo check -p sp1-genguard-host` passes.
- Added sync search (translation+optional rotation) and sample-root recomputation in verifier; CLI flags `--image`, `--max-shift`, `--max-rotation`, `--rotation-step`.
- Added LUT hash check during verification.
- Added a geometric/compression robustness harness (`scripts/run_geom_robustness.py`) to run transformed images through the verifier with sync search.
- Added optional diffusers-based DDIM inversion hook (`ddim_unet.py` + verifier flags `--ddim-*`) and Zhao regen harness (`scripts/run_zhao_attack.py`).
- SP1 host now supports VK/PK override via `SP1_VK_PATH`/`SP1_PK_PATH`.
- Added VK registry support (`vk_path`/`vk_registry` in config) and model_id-based VK selection; sync search now gated (zero-offset check first, search on mismatch, or forced via `--sync-mode`).
- Pending experiment plan: pick a diffusers UNet/DDIM model id (e.g., `stabilityai/stable-diffusion-xl-base-1.0`), watermark an image via `cli prover` + diffusers decode, then run geometry/JPEG/Zhao harness with sync-mode auto; log pass/total, timings, and failure stages (DDIM/align/RS/proof). CPU/GPU choice and DDIM steps to be set based on provided model/device.

## 2025-02-18
- Locked experiment baseline to SDXL base 1.0 (`stabilityai/stable-diffusion-xl-base-1.0`) with DDIM 50 steps on CUDA (`float16`), starting with 1024×1024 pixel inputs (latent 128×128×4) before extending to 1080×1080 pixels (latent 135×135×4).
- Clarified lattice: sampling/Merkle/proof use the latent 2D grid (H×W) as the index space; watermark noise is generated on H×W and broadcast uniformly across 4 channels. Extraction will compute per-channel soft bits and average them (channel-mean) to recover the payload.
- Proof/metadata fields to include to reduce config coupling: `grid_w`, `grid_h`, `channels=4`, `sample_count`, `alpha`, `lut_id/hash`, `ctx_hash`, `model_id`, `version` (plus existing LUT path/hash).
- Updated default config (`config.json`) to match the baseline: pixel_resolution 1024×1024, latent resolution 128×128, fixed `sample_count=1000` (override sample_rate), alpha 2.0; SP1 host command set to `./target/sp1/release/sp1-genguard-host`; added optional pixel resolution and sample_count override to config loader for clarity.
- Existing GPU proofs (reference): SP1 full 1080px run (`artifacts/outputs/out_sp1_1080_s1000_gpu_core_full/`, sp1_prover_sec≈91.4s, sample_count≈1000) and RISC0 full 1080px run (`artifacts/outputs/out_full_gpu_sample1000/`, risc0_prover_sec≈865s).
- SDXL/DDIM E2E pending: attempted to run SDXL base 1.0 + DDIM 50 (CUDA fp16) with 1024px→128×128 latent but blocked because Python toolchains (torch/cupy) currently report `cudaErrorNoDevice`.
- Env setup blockers: conda env creation fails (NoWritableEnvsDir + DNS lookup failure for `repo.anaconda.com`; libmamba plugin errors). `getent hosts repo.anaconda.com` fails (DNS). pip/conda package installs unreachable. Need either network access or pre-provisioned pytorch-cuda12.x + diffusers wheels and a writable env dir to proceed.

## 2025-12-23
- Last work focus: fixed DDIM inversion errors, switched to SHA256-based sampling, expanded rotation/crop ranges in the 6-set geometry attacks, and tests were in progress.
- Scope/context: SHA256-based sampling, DDIM inversion correctness, sync/search behavior, and SP1 sweet-spot experiments.
- Major code changes: SHA256 hash helper and selectable hash backend (default `zk.sample_backend = "sha256"`), consistent bit index mapping via `build_bit_index_map`; files touched include `src/ppmark_v03/crypto.py`, `src/ppmark_v03/noise.py`, `src/ppmark_v03/embedding.py`, `src/ppmark_v03/cuda.py`, `src/ppmark_v03/sync.py`, `src/ppmark_v03/cli.py`, `scripts/run_sp1_sweetspot.py`.
- Tests/results: latent-only decode (SHA256 backend) OK with BER 0.015625 (`outputs/sp1_time_sha256_k1000/latent_noise.npy`, `outputs/sp1_time_sha256_k1000/z0_latents.npy`); clean DDIM10 smoke (syncoff) decode OK with BER 0.01953125 and bit_flip_applied True (`outputs/sp1_sweetspot/smoke_clean_ddim10_k1000/`).
- Tests/results: SP1 timing sanity (k=1000) sp1_prover_sec ~ 94.031s, receipt verify ~6.52s (manual verify); k=600 attacks (DDIM10, sync on, jpeg/resize/geom/chained legacy) all decode False, BER ~0.47-0.53 (random), recovery margin insufficient at k=600.
- Tests/results: rotation 75 smoke (coarse-to-fine) best_rotation_deg = -76.0 (top-3: -76, -75, -74) with search working (`outputs/sp1_sweetspot/smoke_rot75/sweetspot_results.json`); crop&scale 0.75 smoke (3x3 grid, no refine) best_crop_box = [68, 68, 955, 955], candidates = 9, score ~1.02, decode False, BER ~0.496 (random), search working but recovery weak (`outputs/sp1_sweetspot/smoke_crop075/sweetspot_results.json`).
- Open issues/blockers: verifier metadata-only mode hits UnboundLocalError (latent variable used without image path); repo write-protected (could not write new outputs); CUDA unavailable in /tmp fallback run (cudaErrorNoDevice), `nvidia-smi` failed.
- Next steps (pending permission/GPU recovery): re-run crop&scale 0.75 with DDIM steps 20; continue k=600/1000/1500 paper-set attack groups after confirming write access and GPU availability.

## 2025-12-26
- Enforced strict proof/image consistency in verifier CLI: root mismatch now throws `RuntimeError` instead of logging a warning; metadata-only mismatch also hard-fails (`src/ppmark_v03/cli.py`).
- Crop-only robustness retest (DDIM inversion, sync on) recorded under `outputs/sp1_sweetspot/`:
  - `smoke_crop075_ddim20_fast`: steps=20, dtype=float16, crop_grid=2, k=1000 → decoded=False, BER=0.5078125, score=0.0, best_crop_box=[0, 0, 887, 887] (`outputs/sp1_sweetspot/smoke_crop075_ddim20_fast/sweetspot_results.json`).
  - `smoke_crop075_ddim20_fp32_fast`: steps=20, dtype=float32, crop_grid=2, k=1000 → decoded=False, BER=0.49609375, score=0.6307673, best_crop_box=[137, 0, 1024, 887] (`outputs/sp1_sweetspot/smoke_crop075_ddim20_fp32_fast/sweetspot_results.json`).
  - `smoke_crop075_k1500_fp32_grid3`: steps=20, dtype=float32, crop_grid=3, k=1500 → decoded=False, BER=0.484375, score=0.8321127, best_crop_box=[137, 137, 1024, 1024] (`outputs/sp1_sweetspot/smoke_crop075_k1500_fp32_grid3/sweetspot_results.json`).
- Added negative calibration scripts to build an FPR=1% threshold from non-watermarked images:
  - `scripts/neg_calib_generate.py` creates 1000 negatives from Gustavosta/Stable-Diffusion-Prompts (50 prompts × 20 seeds), writes `datasets/neg_calib_1000/manifest.jsonl` and images.
  - `scripts/neg_calib_score.py` runs detector pipeline (DDIM inversion + rotation/crop search + k samples), writes `datasets/neg_calib_1000/scores.csv` and threshold summary `datasets/thresholds/tau_fpr1.json`.
- Added alpha IQA sweep script for BRISQUE/NIQE across multiple prompts/seeds with SDXL decode and watermark injection: `scripts/alpha_iqa_sweep.py`.
  - Features: resume/partial runs via `--start-index`, `--max-images`, `--append-metrics`, `--skip-summary`, `--summary-only`; outputs `prompts.json`, `manifest.jsonl`, `metrics.csv`, `summary.json` in `outputs/alpha_iqa_100/`.
  - Initial run produced prompt sampling and partial images (alpha=0.0 only); `outputs/alpha_iqa_100/a0.0` contains 22 images; `outputs/alpha_iqa_100/metrics.csv` is empty (run stopped early).
- IQA sanity on existing `outputs/run18/.../watermarked.png` images was computed with BRISQUE/NIQE (pyiqa): a0.0=10.2961/4.7154, a2.0=14.8144/4.5070, a2.5=13.4077/4.4149, a3.0=11.5157/4.4542, a3.5=19.7308/3.8543, a4.0=20.3804/4.0891, a4.5=17.2749/3.9675, a5.0=20.7986/3.8793.
- Environment changes: installed `pyiqa` + `piq` for BRISQUE/NIQE; pinned `huggingface_hub==0.25.2` to restore diffusers compatibility.
- Implemented round-robin alpha IQA sweep support in `scripts/alpha_iqa_sweep.py`: new CLI flags `--round-batch-size`, `--rounds`, `--state-path`; per-round state saved to `outputs/alpha_iqa_100/round_state.json` on completion/interruption/error.
- Round summaries now write `outputs/alpha_iqa_100/summary_roundX.json` and append `outputs/alpha_iqa_100/summary.csv` with BRISQUE/NIQE mean/median/p90 computed from metrics accumulated so far; metrics header creation now handles an empty `metrics.csv` safely.
- Refactored per-row generation/metric logic into `_process_manifest_row`, and applied manifest alpha/start/max filtering before round-robin runs to honor `--start-index`/`--max-images` in both modes.
- Ran round-robin sweep round 1 (`--round-batch-size 10 --rounds 1`) with `PYTHONPATH=src`; it processed 10 images at alpha 0.0 only because `outputs/alpha_iqa_100/manifest.jsonl` currently contains 100 rows for alpha 0.0 and no other alphas. Outputs updated: `outputs/alpha_iqa_100/metrics.csv` now has 10 entries, `outputs/alpha_iqa_100/summary_round1.json` created, and `outputs/alpha_iqa_100/summary.csv` appended.
- Extended `outputs/alpha_iqa_100/manifest.jsonl` using `outputs/alpha_iqa_100/prompts.json` to include the full alpha set [0.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]; manifest now has 800 rows (100 per alpha) with metadata matching the original 0.0 entries.
- Removed `outputs/alpha_iqa_100/summary_round1.json` and `outputs/alpha_iqa_100/summary.csv` to reset per-round summaries before re-running round 1 after the manifest expansion.
- Re-ran round 1 (`--round-batch-size 10 --rounds 1`) but the run was aborted mid-round; `outputs/alpha_iqa_100/metrics.csv` now has 22 entries total (alpha 0.0: 10, alpha 2.0: 10, alpha 2.5: 2), and no round summary was produced for this attempt.
- Attempted to resume round 1 again; run aborted mid-round after adding 14 more entries. Current metrics counts: alpha 0.0 = 10, alpha 2.0 = 10, alpha 2.5 = 10, alpha 3.0 = 6 (no summary written yet).
- Created a one-shot alpha IQA run in `outputs/alpha_iqa_1shot` using a single prompt (prompt_id 0, seed 0) across all alphas with SDXL base 1.0, DDIM 50 steps, 1024px; generated 8 images and wrote `metrics.csv` + `summary.json` (alpha_max_by_brisque_p90 = 2.0 for this 1-sample sweep).
- Generated alpha=3.0 calibration metadata via `python -m ppmark_v03.cli prover` using `config_alpha3_k1000.json` (alpha set to 3.0, sample_count=1000, SDXL base 1.0, DDIM 50, 1024px); output at `outputs/calib_pos_alpha3_k1000/metadata.json` with SP1 receipt.
- Started negative calibration generation in `datasets/neg_calib_1000` but aborted early; `manifest.jsonl` + `prompts.json` were written and 5 images exist under `datasets/neg_calib_1000/images/`.
- Switched default config to SD2.1 base (512px) by updating `config.json` (pixel_resolution 512, latent 64x64, model base `stabilityai/stable-diffusion-2-1-base`) and regenerated `config_alpha3_k1000.json` with alpha=3.0 on the new base.
- Added SDXL/SD2.1 pipeline detection to `scripts/alpha_iqa_sweep.py`, `scripts/neg_calib_generate.py`, `src/ppmark_v03/cli.py`, and `src/ppmark_v03/ddim_unet.py` so SD2.1 512 runs use `StableDiffusionPipeline` while SDXL keeps `StableDiffusionXLPipeline`; SD2.1 is now the default model id in the IQA/negative generation scripts.
- Ran SD2.1 (512px) alpha IQA sweep with 10 images per alpha in `outputs/alpha_iqa_10_sd21` (alphas 0, 2, 2.5, 3, 3.5, 4, 4.5, 5) using prompts copied from the first 10 items in `outputs/alpha_iqa_100/prompts.json`; generated 80 images and wrote `metrics.csv` + `summary.json`.
- Prepared `outputs/alpha_iqa_100_sd21` for the 3.0/3.5/4.0 sweep by writing `prompts.json` + `manifest.jsonl` from cached prompts (25 prompts × 4 seeds), but the run failed immediately because CUDA was unavailable (`torch.cuda.is_available() == False`), so no `metrics.csv` was produced yet.
- Attempted to start the SD2.1 alpha sweep and verified CUDA failure: `nvidia-smi` fails to initialize NVML, `torch.cuda.is_available()` reports False (device_count 0) even with `CUDA_VISIBLE_DEVICES=0` and `NVIDIA_VISIBLE_DEVICES=all`; `/dev/nvidia*` devices exist but GPU remains inaccessible, blocking further generation.

## 2025-12-27
- Confirmed CUDA/H100 availability; completed SD2.1 alpha IQA sweep for 3.0/3.5/4.0 (25 prompts x 4 seeds) with round-robin runs; outputs stored under `outputs/alpha_iqa_100_sd21` with `metrics.csv`, `summary_round1..10.json`, `summary.csv`, and `round_state.json`.
- Ran alpha=4.5 IQA quick check (30 images) in `outputs/alpha_iqa_30_sd21_a45`, then expanded to 100 images using a manifest cloned from alpha=3.0 rows; final results in `outputs/alpha_iqa_100_sd21_a45/metrics.csv` and `summary.json`.
- Added `--prompt-filter simple` to `scripts/neg_calib_generate.py` (first-clause extraction, bracket removal, keyword/object filter) and fixed the bracket regex; generated `datasets/neg_calib_1000_sd21` with prompts/manifest and 1000 images.
- Added `config_alpha4_k1000.json`; generated SD2.1 alpha=4.0 calibration metadata via prover with SP1 receipt (`outputs/calib_pos_alpha4_k1000/metadata.json`).
- Updated `scripts/neg_calib_score.py` to ignore manifest prompt/meta and instead fix DDIM inversion inputs to metadata prompt/guidance/negative_prompt/scheduler for stable FPR calibration.
- Ran fp16 50-image sanity check (`datasets/neg_calib_1000_sd21/scores_50.csv`), confirmed no zero/NaN scores.
- Ran full negative calibration on GPU1 (DDIM50, fp16, no search): `datasets/neg_calib_1000_sd21/scores.csv` and `datasets/thresholds/tau_fpr1_alpha4_sd21_fast.json` written; tau_score=2.372129..., fp_count=10/1000 with p99 matching tau.
- Noted GPU0 was occupied by `/home/mimic/binance_trading/scripts/train_sac_nofilter.py` (PID 488763), which blocked parallel scoring attempts.
- Defined the six primary robustness attacks to focus on: rotation 75 deg, JPEG quality 25, random crop+scale keep ratio 0.75, Gaussian blur (8x8 kernel), Gaussian noise (sigma=0.1), and brightness jitter (factor 0.6).
- Built a one-image geometric robustness pilot set from the alpha=4.0 SD2.1 sweep by copying the original image into `outputs/geom_eval_a4_pilot`, applying `rot_75` + `crop_075`, and reconstructing per-image metadata from the sweep manifest (prompt/seed/secret); both attacks scored above tau=2.3721 (pass).
- Created a 10-image evaluation set from `outputs/alpha_iqa_100_sd21/a4.0` into `outputs/geom_eval_a4_10/img_***` with per-image metadata plus `rot_75.png` and `crop_075.png` attack variants.
- Ran detection with search ON (rotation coarse_to_fine max_rotation=90, crop keep_ratio=0.75 grid=3, max_shift=0) using tau=2.3721; results in `outputs/geom_eval_a4_10/scores.csv`: rot_75 pass 1/10, crop_075 pass 3/10.
- Verified base images (no attack) for the same 10 samples: 10/10 pass with scores in ~13-19 range (`outputs/geom_eval_a4_10_basecheck/scores.csv`), confirming metadata reconstruction is correct and failures are due to attack strength.
- Re-ran detection with shift search ON (max_shift=4) and crop refine ON (grid=5, refine enabled): rot_75 pass 1/10 unchanged; crop_075 pass 5/10 improved (`outputs/geom_eval_a4_10_shift_refine_1/scores.csv`), at a much higher per-image runtime for crop search.
- Tested a rotation-only variant using `expand=True` during rotate (then resized to 512) plus a denser rotation sweep (coarse 5-degree grid, fine +/-3 degrees); results were worse with 0/10 passing tau (`outputs/geom_eval_a4_10_rot_expand/scores.csv`), so this approach was discarded.

## 2025-12-31
- Agreed to drop the weak A1 baseline and focus on Muller-style deletion plus an oracle worst-case experiment to avoid "meaningless attack" reviews.
- Defined a Muller-style surrogate removal plan that builds an image-derived surrogate from a single watermarked image (no secret, no detector oracle), then uses it to suppress the learned pattern across target images; target steps include DDIM inversion, surrogate construction, and constrained optimization under PSNR/LPIPS.
- Defined an oracle-based removal experiment (black-box score queries) as the reviewer-friendly upper bound: score minimization via SPSA/NES-style updates with a fixed query budget and quality constraints, reporting success rate vs budget and distortion metrics.
- Mapped potential implementation hooks to the existing attack harness (`scripts/run_muller_attacks.py`, `scripts/run_attack_suite.py`) and noted likely extensions (surrogate reuse, score-query loop, manifest outputs) before wiring in code.
- Investigated SD2.1 weights provenance: metadata records `stabilityai/stable-diffusion-2-1-base`, but no SD2.1 weights were found on host (including archives); HF download returned 404 (gated), and the run used `Manojb/stable-diffusion-2-1-base` as the fallback (confirmed).
- A2 one-shot surrogate removal (Manojb SD2.1 proxy) shows inversion-induced degradation: even beta=0 inversion→decode yields severe distortion, while direct decode from stored latents preserves quality; sweeping beta does not meet LPIPS <= 0.15 or PSNR >= 30, so one-shot A2 is limited under quality constraints.
- Ran a 10-image A2 one-shot sweep with LPIPS on GPU1 using the fallback SD2.1 proxy (`Manojb/stable-diffusion-2-1-base`, DDIM 50, fp16) at beta multipliers [0.0, 0.1, 0.2, 0.3, 0.5]; outputs in `outputs/attacks/a2_surrogate_10_manoj_lpips/` with `a2_metrics.csv`.
- A2 10-image quality summary (mean/median; min/max in parentheses): beta=0.0 PSNR 4.79/4.65 (3.89–6.02), LPIPS 0.842/0.820 (0.767–0.956); beta=0.1 PSNR 6.87/7.14 (4.71–8.07), LPIPS 0.882/0.863 (0.809–0.979); beta=0.2 PSNR 7.45/7.48 (5.88–8.83), LPIPS 0.867/0.860 (0.797–0.956); beta=0.3 PSNR 8.04/8.22 (6.44–9.41), LPIPS 0.891/0.893 (0.841–0.937); beta=0.5 PSNR 8.71/8.84 (7.24–9.90), LPIPS 0.935/0.932 (0.865–1.022).
- Conclusion reaffirmed: even with 10 images and multiple betas, no runs meet LPIPS <= 0.15 or PSNR >= 30; the one-shot A2 removal remains quality-limited due to inversion, consistent with the 1-image pilot.
- Clarified threat model: key recovery/replication is out of scope (cryptographic secret); A2 uses no secret and no detector oracle, while the oracle experiment is a separate black-box score-query upper bound.
- Implemented an oracle score-query attack harness (`scripts/attack_oracle_remove.py`) using SPSA-style updates in image space with score queries, LPIPS (checked every N iters) and PSNR gates, and query budgets (default 200/1000) for the worst-case upper bound experiment.
- Added best-score tracking to oracle metrics (`best_score_est`, `best_step`, `best_queries`, `best_psnr`, `best_lpips`) to separate "best-so-far" from final evaluation (`scripts/attack_oracle_remove.py`).
- Oracle sensitivity check on idx79 (Manojb SD2.1 proxy, DDIM50, fp16) showed score responds to random perturbations: eps=0.01 reduced score to ~6.8-7.1 with PSNR ~40; eps=0.05 reduced score to ~3.7-6.0 with PSNR ~26, confirming the objective is sensitive but quality gates are binding.
- Oracle Q=200 gated sweep on idx79 (step-size 0.05, LPIPS<=0.15, PSNR>=30, eps in [0.005, 0.03]) produced non-monotonic results; eps=0.025 was the strongest in this band (final_score 2.624, PSNR 30.76, LPIPS 0.055) but still above tau=2.372; several eps values resulted in no update (PSNR=inf, score=baseline).
- Oracle no-gate spot checks (eps=0.02/0.025, step-size 0.05, psnr-min=0, lpips-max=1.0) dropped score below tau in a few steps but destroyed quality (PSNR ~5.2-5.6, LPIPS ~1.59), confirming quality constraints are the limiting factor.
- Oracle Q=50 coarse sweep (step-size 0.05, eps in [0.005, 0.10], LPIPS<=0.15, PSNR>=30) on idx79 identified stronger candidates beyond 0.025: eps=0.08 (final_score 1.522, PSNR 31.54, LPIPS 0.045), eps=0.09 (final_score 0.439, PSNR 31.59, LPIPS 0.044), eps=0.045 (final_score 2.355, PSNR 34.95, LPIPS 0.018); results were non-monotonic, reinforcing the need for a bounded search space.
- Fixed oracle evaluation protocol to avoid unbounded optimization: predefine epsilon range [0.005, 0.10] and step-size grid {0.03, 0.05, 0.07}, use bounded query budgets (Q=200 main, Q=1000 subset), and report both fixed-parameter results (realistic) and per-image best-of-K within the fixed grid (upper bound) under the success definition score<tau AND LPIPS<=0.15 AND PSNR>=30.
- Recorded finalized oracle design: success is score<tau AND LPIPS<=0.15 AND PSNR>=30 under a fixed search space (epsilon in [0.005, 0.10], step-size in {0.03, 0.05, 0.07}) with query budgets Q=200 (main) and Q=1000 (subset upper-bound demo).
- Budget-limited best-of-K is the main protocol: two-stage coarse-to-fine with a fixed total query budget per image (stage-1 Q=50 across the grid to rank candidates, stage-2 Q=200 for the top M candidates; report total queries per image explicitly).
- Per-candidate upper bound is reported only on a small subset: each grid configuration receives Q queries (total K*Q per image), to show a worst-case upper bound.
- Ran budget-limited oracle pilot on idx79 (GPU1): Stage-1 Q=50 across eps {0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10} x step-size {0.03,0.05,0.07} under LPIPS<=0.15/PSNR>=30 (`outputs/attacks/oracle_stage1_q50_idx79`). Best candidates by final_score: (eps=0.09, step=0.05, score=0.4396, PSNR=31.59, LPIPS=0.044) and (eps=0.07, step=0.03, score=1.2722, PSNR=46.32, LPIPS=0.00064).
- Stage-2 Q=200 re-evaluation on idx79 confirmed both candidates succeed under quality constraints: (0.09,0.05) final_score=0.4396, queries=31; (0.07,0.03) final_score=1.2722, queries=13 (`outputs/attacks/oracle_stage2_q200_idx79`).
- Clarified interpretation of the oracle results: the oracle experiment is a hypothetical worst-case where the attacker can query the detector; adaptive attacks use random probes only to estimate a gradient and then update toward lower scores, not a one-shot random attack.
- Observed that the coarse Q=50 pass filters many candidates (low hit rate), but best-of-K adaptive refinement can break Soft under the fixed quality constraints on a single image, at a high query cost; this supports separating "realistic" budget-limited results from an "upper bound" best-of-K.
- Decided to present oracle results as: main text includes a short summary of the realistic budget-limited attacker and a note that the upper bound can succeed at high cost, while full best-of-K grid details and per-candidate upper bound tables move to the appendix.

## 2026-01-08
- Built a 100-prompt quality list offline by merging two local 50-prompt JSONs and preserving the original 25 prompt IDs; wrote `datasets/quality_prompts_sd21_100/prompts.json` plus a deterministic `seed_map.json`.
- Added `scripts/build_quality_prompts.py` and `scripts/build_quality_manifest.py` to generate 1,000-row manifests for alpha=4.0 and alpha=0.0; manifest rows include prompt/seed, meta settings, and `source_image_path` for reusing the existing 200 watermarked images.
- Added `scripts/generate_quality_images.py` to generate images without proofs from a manifest, with optional linking/copying of existing images.
- Switched SD2.1 defaults to the proxy model `Manojb/stable-diffusion-2-1-base` across configs and scripts due to gated SD2.1 weights, and updated the manifest builder to preserve original model_id for reused images via metadata lookup; regenerated `outputs/quality_a4_1k/manifest.jsonl` and `outputs/quality_a0_1k/manifest.jsonl`.
- Ran image generation on GPU1 using `.venv_a2` (reedsolo available): completed `outputs/quality_a4_1k/images` (200 linked + 800 new) and `outputs/quality_a0_1k/images` (1,000 new); observed CLIP truncation warnings for long prompts.
- Verified output counts: 1,000 images in each folder.
- Started a KID+CLIP evaluation script (`scripts/eval_kid_clip.py`) using torchvision Inception features plus CLIP (transformers); run failed due to a torchvision `aux_logits` restriction and needs a small loader fix before use.

## 2026-01-27
- Session context: white-box attack prep for the muller forgery report; continuing the 10-image PP-Mark run after an OOM failure.
- Planned run parameters: eps=1/255, steps=150 with checkpoints at 50/100/150, early-stop enabled, float16 inference, `--ppmark-opening-metadata`, GPU1; resume sequence is `start_index=10`, `max_images=40` (i.e., 10 images after the first batch).
- Previous attempt status: 10-image PP-Mark attack failed with CUDA OOM in UNet/attention before outputs were written; PyTorch reported fragmentation hints; no artifacts were produced in `outputs/attacks/muller_forgery_report_prep/forged/wb_pgd_10img_ppmark_eps1_steps150_early`.
- Mitigation for rerun: set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and explicitly pass `--ddim-dtype float16` while keeping all other options unchanged; output directory will be overwritten on rerun.
- Compliance note: searched for `AGENTS.md` under the target output directory and repo root and found none, so default session instructions apply.
- Status check (prep only, no GPU job launched): `nvidia-smi` shows GPU0 ~1.3 GiB used (0% util) and GPU1 ~3 MiB used (0% util), both H100 PCIe; GPU1 is effectively idle and safe to use.
- Process scan: no active PP-Mark/PGD/attack runs detected (only unrelated long-running Python services outside this repo).
- Output dir inspection: `outputs/attacks/muller_forgery_report_prep/forged/wb_pgd_10img_ppmark_eps1_steps150_early/` contains the directory scaffold plus a single cover image at `covers/real/coco_000000419974.jpg`.
- Checkpoints: empty `step010/`, `step015/`, `step017/` subfolders exist under `checkpoints/` with no files inside; indicates the prior run initialized checkpoints but aborted before writing artifacts.
- Prior-run verification: reviewed `outputs/attacks/muller_forgery_report_prep/figures/work_log.md` entries 216–222 to confirm the intended WB PP-Mark 10-image settings.
- Confirmed settings to reproduce: manifest `outputs/attacks/muller_forgery_report_prep/pairs/ppmark_adaptive_99.jsonl`, range idx 1–10 (`start_index=0`, `max_images=10`), steps=150, eps=1/255, early-stop, checkpoints at 50/100/150, trace_idx=0, output root `outputs/attacks/muller_forgery_report_prep/forged/wb_pgd_10img_ppmark_eps1_steps150_early`.
- Config/threshold: `outputs/attacks/muller_forgery_report_prep/ppmark_config_alpha4_k1000.json` and `outputs/attacks/muller_forgery_report_prep/thresholds/pp_mark_score_fpr1.json`.
- Paths: `--image-root outputs/attacks/muller_forgery_report_prep`, `--metadata-root outputs/attacks/muller_forgery_report_prep/methods/pp_mark/eval`, `--ppmark-opening-metadata` enabled (as in the 1-image command).
- Runtime environment: use `/data/venvs/ppmark/bin/python` with `PYTHONPATH=src` and `CUDA_VISIBLE_DEVICES=1`; for the retry, add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and explicitly set `--ddim-dtype float16`.
- Planned change (trace-all): add a `--trace-all` flag to `scripts/attack_whitebox_forge.py` so every image records per-step trace entries (score/objective/psnr/linf).
- Trace semantics: when `--trace-all` is set, record a trace row for every image at every step (0..steps), using the manifest-provided `idx` value for the `idx` column (not the local_idx).
- Precedence: keep existing `--trace-idx` behavior for single-image tracing; if both are provided, `--trace-all` takes priority.
- Output impact: trace CSV size scales as (#images × (#steps+1)) rows; for 10 images and 150 steps, expect ~1510 rows (manageable).
- CLI usage example to be used for the 10-image PP-Mark rerun:
  - add `--trace-all` to the existing command, keep all other options unchanged.
- Implemented trace-all: added `--trace-all` flag to `scripts/attack_whitebox_forge.py` and made it log per-step traces for every image (idx column uses manifest idx when trace-all is enabled); when both `--trace-all` and `--trace-idx` are provided, trace-all takes precedence.
- OOM re-run (PP-Mark 10 images, trace-all, GPU1) failed again at the first DDIM inversion step in UNet attention; CUDA error shows only ~24.81 MiB free on the mapped GPU (CUDA_VISIBLE_DEVICES=1), with this process holding ~79.08 GiB (78.36 GiB allocated by PyTorch).
- Folder cleanup note: `rm -rf` is blocked by the execution policy, so the output dir was moved aside to `outputs/attacks/muller_forgery_report_prep/forged/wb_pgd_10img_ppmark_eps1_steps150_early_bak_20260127_061029` and the target folder recreated.
- GPU status check after OOM: `nvidia-smi` shows GPU0 ~1.3 GiB used, GPU1 ~3 MiB used (both 0% util), so no external workload appears to be occupying GPU1.
- Memory-mitigation patch (planned + applied):
  - `ppmark_v03/ddim_unet.py`: added DDIM config flags for attention slicing, VAE slicing, UNet gradient checkpointing, and xformers attention; pipeline applies these best-effort during initialization.
  - `scripts/attack_whitebox_forge.py`: added CLI flags `--ddim-attn-slicing`, `--ddim-vae-slicing`, `--ddim-grad-checkpointing`, `--ddim-xformers` and threaded them through the inverter cache key and config so PP-Mark inversion can enable them.
- Memory-mitigated rerun (PP-Mark 10 images, trace-all, GPU1) still OOMed during the first DDIM inversion; even with attention/vae slicing + grad checkpointing + xformers enabled, the process exhausted GPU memory (~79.09 GiB used, ~8.81 MiB free; 20 MiB allocation failed).
- Gradient-checkpointing fix: in `ppmark_v03/ddim_unet.py` (invert_latents_torch), temporarily switch UNet to train mode when `grad_checkpointing` is enabled and then restore eval mode afterward; this ensures PyTorch checkpointing actually activates for the inversion loop.
- PP-Mark 10-image run (with memory flags + grad-checkpointing fix) was started but the CLI timed out after 120s (no completion log); output dir contains only scaffold + one cover image (`covers/real/coco_000000419974.jpg`) and empty checkpoint folders (step010/step018), indicating early-stage progress before interruption.
- Process note: the 120s CLI timeout was misaligned with the 12–20 min runtime estimate; this caused a premature abort. Going forward, long GPU runs should either use a timeout aligned to the estimate (plus slack) or run in the background with logs captured.

### Next session prompt
```
You are in /home/mimic/PP-Mark-v0.4/PP-Mark. Continue the alpha IQA sweep.
Use scripts/alpha_iqa_sweep.py, but modify or wrap it to support round-robin batches:
- Alpha set: [0.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
- Batch size: 10 images per alpha per round (total 80 images/round).
- After each round, compute BRISQUE/NIQE mean/median/p90 and save to outputs/alpha_iqa_100/summary_roundX.json and append to summary.csv.
- Must be resumable; on error, stop cleanly with state saved and allow restart.
Keep generation settings fixed (SDXL base 1.0, DDIM 50 steps, 1024px, guidance 7.5). Use existing prompts.json/manifest.jsonl in outputs/alpha_iqa_100.
Current state: outputs/alpha_iqa_100/a0.0 has 22 images; outputs/alpha_iqa_100/metrics.csv is empty.
```

## 2026-01-27 (StableSig/HiDDeN 50img eps=2/255 + RingID OOM)

- Action: Started full 50-image eps=2/255 white-box PGD runs (early-stop, steps=150, checkpoints at 50/100/150, trace-all enabled).
- Requirement: Record per-step scores + PSNR; log earliest success step per image; keep settings consistent with prior eps2 runs.

### Stable Signature (50 images, eps=2/255)
- Output: `outputs/attacks/muller_forgery_report_prep/forged/wb_pgd_50img_stable_signature_eps2_steps150_early`
- Command (GPU1):
  - `PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  - `scripts/attack_whitebox_forge.py --method stable_signature`
  - `--manifest outputs/attacks/muller_forgery_report_prep/methods/stable_signature/eval/manifest_fixed.jsonl`
  - `--image-root outputs/attacks/muller_forgery_report_prep/clean/eval`
  - `--metadata-root outputs/attacks/muller_forgery_report_prep/methods/stable_signature/eval`
  - `--threshold-file outputs/attacks/muller_forgery_report_prep/thresholds/stable_signature_fpr1.json`
  - `--steps 150 --eps 0.007843137 --checkpoint-steps 50,100,150 --save-checkpoint-images --trace-all --early-stop`
  - `--start-index 0 --max-images 50 --overwrite`
- Status: Completed successfully (run.log shows 10/50 ... 50/50).
- Artifacts: `metrics.csv`, `checkpoints.csv`, `trace_scores.csv`, `summary.json`, `checkpoints/`.
- Summary (summary.json): detected_rate=1.0; score_mean=0.6192; score_median=0.6042; psnr_median=64.29 dB (psnr_mean=Inf due to unchanged images).
- Metrics indicate early success in ~1–9 steps for most samples.

### HiDDeN (50 images, eps=2/255)
- Output: `outputs/attacks/muller_forgery_report_prep/forged/wb_pgd_50img_hidden_eps2_steps150_early`
- Command (GPU1):
  - `PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  - `scripts/attack_whitebox_forge.py --method hidden`
  - `--manifest outputs/attacks/muller_forgery_report_prep/methods/hidden_revisited/eval/manifest.jsonl`
  - `--image-root outputs/attacks/muller_forgery_report_prep/clean/eval`
  - `--metadata-root outputs/attacks/muller_forgery_report_prep/methods/hidden_revisited/eval`
  - `--threshold-file outputs/attacks/muller_forgery_report_prep/thresholds/hidden_fpr1.json`
  - `--steps 150 --eps 0.007843137 --checkpoint-steps 50,100,150 --save-checkpoint-images --trace-all --early-stop`
  - `--start-index 0 --max-images 50 --overwrite`
- Status: Completed successfully (run.log shows 10/50 ... 50/50).
- Artifacts: `metrics.csv`, `checkpoints.csv`, `trace_scores.csv`, `summary.json`, `checkpoints/`.
- Summary (summary.json): detected_rate=1.0; score_mean=0.6925; score_median=0.6875; psnr_median=62.38 dB (psnr_mean=Inf due to unchanged images).
- Metrics indicate early success in ~0–10 steps for most samples.

### RingID (50 images, eps=2/255)
- Output: `outputs/attacks/muller_forgery_report_prep/forged/wb_pgd_50img_ringid_eps2_steps150_early`
- Command (GPU1):
  - `PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  - `scripts/attack_whitebox_forge.py --method ringid`
  - `--manifest outputs/attacks/muller_forgery_report_prep/methods/ringid/eval/manifest_fixed.jsonl`
  - `--image-root outputs/attacks/muller_forgery_report_prep/clean/eval`
  - `--metadata-root outputs/attacks/muller_forgery_report_prep/methods/ringid/eval`
  - `--threshold-file outputs/attacks/muller_forgery_report_prep/thresholds/ringid_fpr1.json`
  - `--steps 150 --eps 0.007843137 --checkpoint-steps 50,100,150 --save-checkpoint-images --trace-all --early-stop`
  - `--start-index 0 --max-images 50 --overwrite`
- Status: Failed quickly with CUDA OOM in `_score_ringid` during `vae.encode()` (no metrics.csv; only one checkpoint file).
- Error excerpt: `torch.OutOfMemoryError: CUDA out of memory ... Tried to allocate 32 MiB ... GPU 0 has a total capacity of 79.11 GiB ...`
- Next: verify dtype/flags in run.log and re-run with explicit memory-saving flags (`--ddim-dtype float16`, `--ddim-vae-slicing`, `--ddim-attn-slicing`, `--ddim-xformers`, `--ddim-grad-checkpointing`) after confirming GPU1 is clear.

### RingID eps=2/255 50-image rerun (memory-saver flags) — failed
- Attempt: Re-ran RingID 50-image eps=2/255 with explicit memory flags.
- Command included: `--ddim-dtype float16 --ddim-vae-slicing --ddim-attn-slicing --ddim-xformers --ddim-grad-checkpointing`.
- Status: Immediate CUDA OOM during `_score_ringid` in `vae.encode()` (no metrics.csv; only one checkpoint file).
- Error excerpt: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 32.00 MiB ...`
- Note: DDIM-related flags likely do not affect RingID scoring path; VAE encode still full-precision/memory heavy.
- Next: adjust RingID scoring to enforce fp16 + VAE/attention slicing for the scoring encode path, then re-run.

### RingID memory-mitigation patch (2026-01-27)
- Added RingID-specific memory flags to attack script:
  - `--ringid-attn-slicing`, `--ringid-vae-slicing`, `--ringid-xformers`, `--ringid-grad-checkpointing`.
- Applied in `_build_ringid_context()` to the RingID pipeline:
  - `pipe.enable_attention_slicing()`
  - `pipe.enable_vae_slicing()`
  - `pipe.enable_xformers_memory_efficient_attention()` (best-effort)
  - `pipe.unet.enable_gradient_checkpointing()` (best-effort)
- Goal: reduce VAE encode / UNet memory usage during RingID scoring to avoid OOM.

### RingID grad-checkpointing fix (2026-01-27)
- Bugfix: `_score_ringid()` now reads checkpoint flag from `context` (dict) and toggles UNet train/eval to activate gradient checkpointing during forward diffusion.
- `_build_ringid_context()` now stores `ringid_grad_checkpointing` in context.
- Goal: reduce memory footprint in RingID scoring without changing experimental conditions.

### RingID 1-image eps=2/255 exact-repro reconstruction (2026-01-27)
- Inspected `outputs/attacks/muller_forgery_report_prep/forged/wb_pgd_1img_ringid_eps2_steps150_early` (successful 1-image run).
- Artifacts present: `metrics.csv`, `trace_scores.csv`, `checkpoints.csv`, `summary.json`, `checkpoints/step002_manifest.jsonl`, and the forged image under `covers/real/`.
- `metrics.csv` and `trace_scores.csv` show early-stop at step 2 with PSNR 74.4663 dB and Linf 0.00020915 (eps=2/255). Trace rows use `idx=0`, which implies `--trace-idx 0` (not `--trace-all`).
- `checkpoints/step002_manifest.jsonl` confirms:
  - `source_path`: `.../covers/real/coco_000000419974.jpg`
  - `metadata_path`: `.../methods/pp_mark/eval/images/img_0001/metadata.json`
  - `eps`: 0.007843137, `steps`: 150, `step_size`: 0.00010457516
- The resolved paths above require:
  - `--image-root outputs/attacks/muller_forgery_report_prep` (so `covers/real/...` resolves correctly)
  - `--metadata-root outputs/attacks/muller_forgery_report_prep/methods/pp_mark/eval`
- Manifest inference: among `outputs/attacks/muller_forgery_report_prep/pairs/*.jsonl`, only `ppmark_adaptive_99.jsonl` has `idx=1` as the *first* row for this cover; others (pilot5/10/100 perimage) would require `--start-index 1`. This makes `ppmark_adaptive_99.jsonl` the most likely original manifest when using default `--start-index 0` and `--max-images 1`.
- Retest dirs checked: `wb_pgd_1img_ringid_eps2_steps150_early_retest` and `..._retest_cover` exist but are empty (no logs, no artifacts).
- Proposed exact-repro command (no new RingID memory flags, default float32), with a fresh output root to avoid overwriting the successful baseline:
  - `PYTHONPATH=src CUDA_VISIBLE_DEVICES=1`
  - `scripts/attack_whitebox_forge.py --method ringid`
  - `--manifest outputs/attacks/muller_forgery_report_prep/pairs/ppmark_adaptive_99.jsonl`
  - `--image-root outputs/attacks/muller_forgery_report_prep`
  - `--metadata-root outputs/attacks/muller_forgery_report_prep/methods/pp_mark/eval`
  - `--threshold-file outputs/attacks/muller_forgery_report_prep/thresholds/ringid_fpr1.json`
  - `--steps 150 --eps 0.007843137 --checkpoint-steps 50,100,150 --save-checkpoint-images --early-stop`
  - `--trace-idx 0 --start-index 0 --max-images 1`
  - `--output-root outputs/attacks/muller_forgery_report_prep/forged/wb_pgd_1img_ringid_eps2_steps150_early_retest_exact`

### RingID 1-image eps=2/255 exact-repro run (executed, 2026-01-27)
- Goal: reproduce the successful 1-image RingID eps=2/255 run using the inferred original manifest and path roots (covers/real + PP-Mark metadata), without any new RingID memory flags.
- Output: `outputs/attacks/muller_forgery_report_prep/forged/wb_pgd_1img_ringid_eps2_steps150_early_retest_exact`
- Command (GPU1):
  - `PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 /data/venvs/ppmark/bin/python scripts/attack_whitebox_forge.py --method ringid`
  - `--manifest outputs/attacks/muller_forgery_report_prep/pairs/ppmark_adaptive_99.jsonl`
  - `--image-root outputs/attacks/muller_forgery_report_prep`
  - `--metadata-root outputs/attacks/muller_forgery_report_prep/methods/pp_mark/eval`
  - `--threshold-file outputs/attacks/muller_forgery_report_prep/thresholds/ringid_fpr1.json`
  - `--steps 150 --eps 0.007843137 --checkpoint-steps 50,100,150 --save-checkpoint-images --early-stop`
  - `--trace-idx 0 --start-index 0 --max-images 1`
  - `--output-root outputs/attacks/muller_forgery_report_prep/forged/wb_pgd_1img_ringid_eps2_steps150_early_retest_exact`
- Runtime: ~22s end-to-end (pipeline load + 1 image).
- Status: Success; early-stop at step 2.
- Artifacts present:
  - `metrics.csv`, `trace_scores.csv`, `checkpoints.csv`, `summary.json`, `checkpoints/step002_manifest.jsonl`, forged image under `covers/real/`.
- Key metrics (from `metrics.csv` / `summary.json`):
  - step=2, detected=1, score=-73.39788344596454, psnr=74.46667869173815, linf=0.0002091526985168457.
- Comparison to baseline (`wb_pgd_1img_ringid_eps2_steps150_early`):
  - Same early-stop step (2) and identical Linf; score/PSNR differ only at ~1e-3 scale (expected float noise), indicating the exact-repro path matches the original conditions.

### RingID 50-image eps=2/255 exact-repro attempt (trace-all) — failed OOM (2026-01-27)
- Goal: run the 50-image RingID attack under the *same conditions as the successful 1-image exact-repro* (covers/real + PP-Mark metadata), but with `--trace-all` to capture full per-step traces.
- Output target: `outputs/attacks/muller_forgery_report_prep/forged/wb_pgd_50img_ringid_eps2_steps150_early_retest_exact`
- Command (GPU1):
  - `PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 /data/venvs/ppmark/bin/python scripts/attack_whitebox_forge.py --method ringid`
  - `--manifest outputs/attacks/muller_forgery_report_prep/pairs/ppmark_adaptive_99.jsonl`
  - `--image-root outputs/attacks/muller_forgery_report_prep`
  - `--metadata-root outputs/attacks/muller_forgery_report_prep/methods/pp_mark/eval`
  - `--threshold-file outputs/attacks/muller_forgery_report_prep/thresholds/ringid_fpr1.json`
  - `--steps 150 --eps 0.007843137 --checkpoint-steps 50,100,150 --save-checkpoint-images --early-stop`
  - `--trace-all --start-index 0 --max-images 50`
  - `--output-root outputs/attacks/muller_forgery_report_prep/forged/wb_pgd_50img_ringid_eps2_steps150_early_retest_exact`
- Status: Failed immediately with CUDA OOM during RingID `vae.encode()` (VAE conv2d) before any metrics/trace were written.
  - Error excerpt (stderr): `torch.OutOfMemoryError: ... Tried to allocate 64.00 MiB ...` in `_encode_latents()`.
- Artifacts:
  - `run.log` is empty because stderr was not captured by `tee`.
  - Only one cover image written (`covers/real/coco_000000419974.jpg`).
  - No `metrics.csv`, `trace_scores.csv`, `checkpoints.csv`, or `summary.json`.
- Consequence: per-step score/PSNR deltas and success-step PSNR summaries could not be computed (missing trace/metrics).

### RingID 50-image eps=2/255 exact-repro attempt (trace-all + stderr log) — failed OOM (2026-01-27)
- Retried with identical conditions, adding `--overwrite` to avoid skipping the first image and `2>&1 | tee run.log` to capture stderr.
- Output target: `outputs/attacks/muller_forgery_report_prep/forged/wb_pgd_50img_ringid_eps2_steps150_early_retest_exact`
- Command (GPU1):
  - `PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 /data/venvs/ppmark/bin/python scripts/attack_whitebox_forge.py --method ringid`
  - `--manifest outputs/attacks/muller_forgery_report_prep/pairs/ppmark_adaptive_99.jsonl`
  - `--image-root outputs/attacks/muller_forgery_report_prep`
  - `--metadata-root outputs/attacks/muller_forgery_report_prep/methods/pp_mark/eval`
  - `--threshold-file outputs/attacks/muller_forgery_report_prep/thresholds/ringid_fpr1.json`
  - `--steps 150 --eps 0.007843137 --checkpoint-steps 50,100,150 --save-checkpoint-images --early-stop`
  - `--trace-all --start-index 0 --max-images 50 --overwrite`
  - `--output-root outputs/attacks/muller_forgery_report_prep/forged/wb_pgd_50img_ringid_eps2_steps150_early_retest_exact`
- Status: Immediate CUDA OOM in `_encode_latents()` (VAE conv2d), same failure point as prior attempt.
  - Error captured in `run.log` (traceback shows OOM during `vae.encode()`; attempted 64 MiB allocation).
- Artifacts: no `metrics.csv`, `trace_scores.csv`, or `summary.json` generated; only `run.log` and a single cover image remain.
- Result: unable to compute per-step score/PSNR deltas or success-step PSNR summaries due to missing traces.

### RingID 50-image eps=2/255 exact-repro attempt (trace-all, clean GPU) — failed OOM (2026-01-27)
- Per user request, verified GPU1 was idle (nvidia-smi shows ~3 MiB used, 0% util) before rerun.
- Re-ran identical command (same as prior trace-all attempt) with `--overwrite` and stderr captured in `run.log`.
- Status: immediate CUDA OOM at `_encode_latents()` / `vae.encode()` before any metrics or traces were written.
- Error excerpt (from `run.log`): `torch.OutOfMemoryError: ... Tried to allocate 64.00 MiB ...`.
- Artifacts: only `run.log` and a single cover image; no `metrics.csv`, `trace_scores.csv`, or `summary.json`.

### RingID 5-image eps=2/255 exact-repro attempt (trace-all) — failed OOM (2026-01-27)
- Goal: run a smaller batch (5 images) under identical conditions to the successful 1-image exact-repro, with `--trace-all` enabled.
- Output: `outputs/attacks/muller_forgery_report_prep/forged/wb_pgd_5img_ringid_eps2_steps150_early_retest_exact`
- Command (GPU1):
  - `PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 /data/venvs/ppmark/bin/python scripts/attack_whitebox_forge.py --method ringid`
  - `--manifest outputs/attacks/muller_forgery_report_prep/pairs/ppmark_adaptive_99.jsonl`
  - `--image-root outputs/attacks/muller_forgery_report_prep`
  - `--metadata-root outputs/attacks/muller_forgery_report_prep/methods/pp_mark/eval`
  - `--threshold-file outputs/attacks/muller_forgery_report_prep/thresholds/ringid_fpr1.json`
  - `--steps 150 --eps 0.007843137 --checkpoint-steps 50,100,150 --save-checkpoint-images --early-stop`
  - `--trace-all --start-index 0 --max-images 5 --overwrite`
  - `--output-root outputs/attacks/muller_forgery_report_prep/forged/wb_pgd_5img_ringid_eps2_steps150_early_retest_exact`
- Status: immediate CUDA OOM at `_encode_latents()` / `vae.encode()` before any metrics or traces were written.
  - Error captured in `run.log` (attempted 64 MiB allocation).
- Artifacts: only `run.log` and a single cover image; no `metrics.csv`, `trace_scores.csv`, or `summary.json`.

### RingID 50x1-image eps=2/255 exact-repro batch (trace-all) — success (2026-01-27)
- Goal: avoid OOM by running the 50 images one-at-a-time (fresh process per image) under the exact same conditions as the successful 1-image run, with `--trace-all` enabled, then aggregate traces/metrics.
- Batch runner: looped over `start-index=0..49` with `max-images=1`, separate output root per index, and stderr captured to per-index `run.log`.
- Output root: `outputs/attacks/muller_forgery_report_prep/forged/wb_pgd_1img_ringid_eps2_steps150_early_retest_exact_batch/`
  - Per-image folders: `idx000` ... `idx049`
  - Each folder contains: `metrics.csv`, `trace_scores.csv`, `checkpoints.csv`, `summary.json`, `covers/real/*.jpg`, `run.log`.
- Command template (GPU1):
  - `PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 /data/venvs/ppmark/bin/python scripts/attack_whitebox_forge.py --method ringid`
  - `--manifest outputs/attacks/muller_forgery_report_prep/pairs/ppmark_adaptive_99.jsonl`
  - `--image-root outputs/attacks/muller_forgery_report_prep`
  - `--metadata-root outputs/attacks/muller_forgery_report_prep/methods/pp_mark/eval`
  - `--threshold-file outputs/attacks/muller_forgery_report_prep/thresholds/ringid_fpr1.json`
  - `--steps 150 --eps 0.007843137 --checkpoint-steps 50,100,150 --save-checkpoint-images --early-stop --trace-all`
  - `--start-index {i} --max-images 1 --output-root {outdir}`
- Status: all 50 runs completed successfully (no OOM).

#### Aggregated outputs
- `metrics_all.csv` (50 rows)
- `trace_scores_all.csv` (163 rows)
- `checkpoints_all.csv` (50 rows)
- `trace_scores_deltas.csv` (163 rows; per-step score/PSNR deltas)
- `success_psnr_by_image.csv` (50 rows; success step PSNR/score/detected per image)
- Files saved under: `.../wb_pgd_1img_ringid_eps2_steps150_early_retest_exact_batch/`

#### Quick stats (from metrics_all)
- All images detected (detected=1 for all 50).
- Early-stop step distribution: {0:1, 1:19, 2:18, 3:4, 4:1, 5:2, 6:3, 7:2}.
- Min step=0, max step=7.

### Reproducible procedure: RingID 50x1-image exact-repro batch (trace-all) (2026-01-27)
This is the exact, working method used to avoid OOM while keeping *identical experimental conditions* to the 1-image success case.

#### Preconditions
- GPU: use GPU1 only. Confirm GPU1 is idle: `nvidia-smi` shows ~0% util and ~3 MiB used.
- Python env: `/data/venvs/ppmark/bin/python`.
- CWD: `/home/mimic/PP-Mark-v0.4/PP-Mark`.
- Do NOT change model/steps/eps/image-root/metadata-root/threshold-file.

#### Why this method
- Running 50 images in one process OOMs on the first image during `vae.encode()`.
- Running each image in a fresh process avoids the memory cliff; all 50 succeed.

#### Core command template (single image)
```
PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 /data/venvs/ppmark/bin/python scripts/attack_whitebox_forge.py --method ringid \
  --manifest outputs/attacks/muller_forgery_report_prep/pairs/ppmark_adaptive_99.jsonl \
  --image-root outputs/attacks/muller_forgery_report_prep \
  --metadata-root outputs/attacks/muller_forgery_report_prep/methods/pp_mark/eval \
  --threshold-file outputs/attacks/muller_forgery_report_prep/thresholds/ringid_fpr1.json \
  --steps 150 --eps 0.007843137 --checkpoint-steps 50,100,150 --save-checkpoint-images --early-stop --trace-all \
  --start-index {i} --max-images 1 --output-root {outdir}
```

#### Exact batch runner used (50x1, with logs + fail-fast)
```
base="outputs/attacks/muller_forgery_report_prep/forged/wb_pgd_1img_ringid_eps2_steps150_early_retest_exact_batch"
mkdir -p "$base"
for i in $(seq 0 49); do
  outdir="$base/idx$(printf '%03d' $i)"
  mkdir -p "$outdir"
  echo "[batch] idx $i" > "$outdir/run.log"
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 /data/venvs/ppmark/bin/python scripts/attack_whitebox_forge.py --method ringid \
    --manifest outputs/attacks/muller_forgery_report_prep/pairs/ppmark_adaptive_99.jsonl \
    --image-root outputs/attacks/muller_forgery_report_prep \
    --metadata-root outputs/attacks/muller_forgery_report_prep/methods/pp_mark/eval \
    --threshold-file outputs/attacks/muller_forgery_report_prep/thresholds/ringid_fpr1.json \
    --steps 150 --eps 0.007843137 --checkpoint-steps 50,100,150 --save-checkpoint-images --early-stop --trace-all \
    --start-index $i --max-images 1 --output-root "$outdir" 2>&1 | tee -a "$outdir/run.log"
  status=${PIPESTATUS[0]}
  if [ $status -ne 0 ]; then
    echo "[batch] idx $i failed with status $status" | tee -a "$outdir/run.log"
    echo "$i" > "$base/failed_idx.txt"
    exit $status
  fi
done
```

#### Aggregation steps (post-run)
- Aggregate all per-image CSVs into unified outputs and compute deltas.
```
python3 - <<'PY'
from pathlib import Path
import csv
from collections import defaultdict

base = Path('outputs/attacks/muller_forgery_report_prep/forged/wb_pgd_1img_ringid_eps2_steps150_early_retest_exact_batch')
idx_dirs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith('idx')])

def read_csv(path):
    with path.open('r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fieldnames})

metrics_all = []
trace_all = []
checkpoints_all = []
for p in idx_dirs:
    metrics_all.extend(read_csv(p/'metrics.csv'))
    trace_all.extend(read_csv(p/'trace_scores.csv'))
    checkpoints_all.extend(read_csv(p/'checkpoints.csv'))

if metrics_all:
    write_csv(base/'metrics_all.csv', metrics_all, list(metrics_all[0].keys()))
if trace_all:
    write_csv(base/'trace_scores_all.csv', trace_all, list(trace_all[0].keys()))
if checkpoints_all:
    write_csv(base/'checkpoints_all.csv', checkpoints_all, list(checkpoints_all[0].keys()))

# per-step deltas
trace_by_idx = defaultdict(list)
for row in trace_all:
    step = int(float(row.get('step', 0)))
    row['_step'] = step
    trace_by_idx[row.get('idx')].append(row)

trace_delta_rows = []
for idx, rows in trace_by_idx.items():
    rows.sort(key=lambda r: r['_step'])
    prev_score = None
    prev_psnr = None
    for r in rows:
        score = float(r.get('score'))
        psnr = float(r.get('psnr'))
        if prev_score is None:
            score_delta = 0.0
            psnr_delta = 0.0
        else:
            score_delta = score - prev_score
            psnr_delta = psnr - prev_psnr
        trace_delta_rows.append({
            'idx': r.get('idx'),
            'step': r.get('step'),
            'objective_score': r.get('objective_score'),
            'score': r.get('score'),
            'psnr': r.get('psnr'),
            'linf': r.get('linf'),
            'score_delta': f"{score_delta:.10f}",
            'psnr_delta': f"{psnr_delta:.10f}",
        })
        prev_score = score
        prev_psnr = psnr

if trace_delta_rows:
    write_csv(base/'trace_scores_deltas.csv', trace_delta_rows,
              ['idx','step','objective_score','score','psnr','linf','score_delta','psnr_delta'])

# success-step PSNR per image
success_rows = []
for r in metrics_all:
    success_rows.append({
        'idx': r.get('idx'),
        'step': r.get('step'),
        'psnr': r.get('psnr'),
        'score': r.get('score'),
        'objective_score': r.get('objective_score'),
        'detected': r.get('detected'),
        'linf': r.get('linf'),
        'image_path': r.get('image_path'),
        'source_path': r.get('source_path'),
    })
if success_rows:
    write_csv(base/'success_psnr_by_image.csv', success_rows,
              ['idx','step','psnr','score','objective_score','detected','linf','image_path','source_path'])
PY
```

#### Expected outputs (after aggregation)
- `metrics_all.csv`
- `trace_scores_all.csv`
- `checkpoints_all.csv`
- `trace_scores_deltas.csv`
- `success_psnr_by_image.csv`

#### Failure handling
- If any run OOMs, the batch stops and writes `failed_idx.txt` under the batch root.
- Restart by re-running the loop starting at that index (or delete only the failed idx folder and rerun).

### PP-Mark 50x1-image eps=2/255 exact-repro batch (trace-all) — completed (2026-01-27)
- Goal: run PP-Mark under the same eps=2/255, steps=150, early-stop settings as the 1-image baseline, using 1-image per process to avoid OOM.
- Output root: `outputs/attacks/muller_forgery_report_prep/forged/wb_pgd_1img_ppmark_eps2_steps150_early_retest_exact_batch/`
  - Per-image folders: `idx000` ... `idx049`
  - Each folder contains: `metrics.csv`, `trace_scores.csv`, `checkpoints.csv`, `summary.json`, `covers/real|gen/...`, `run.log`.
- Command template (GPU1):
  - `PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 /data/venvs/ppmark/bin/python scripts/attack_whitebox_forge.py --method ppmark`
  - `--manifest outputs/attacks/muller_forgery_report_prep/pairs/ppmark_adaptive_99.jsonl`
  - `--image-root outputs/attacks/muller_forgery_report_prep`
  - `--metadata-root outputs/attacks/muller_forgery_report_prep/methods/pp_mark/eval`
  - `--ppmark-opening-metadata`
  - `--ppmark-config outputs/attacks/muller_forgery_report_prep/ppmark_config_alpha4_k1000.json`
  - `--threshold-file outputs/attacks/muller_forgery_report_prep/thresholds/pp_mark_score_fpr1.json`
  - `--steps 150 --eps 0.007843137 --checkpoint-steps 50,100,150 --save-checkpoint-images --early-stop --trace-all`
  - `--start-index {i} --max-images 1 --output-root {outdir}`

#### Aggregated outputs
- `metrics_all.csv` (50 rows)
- `trace_scores_all.csv` (2512 rows)
- `checkpoints_all.csv` (77 rows)
- `trace_scores_deltas.csv` (2512 rows; per-step score/PSNR deltas)
- `success_psnr_by_image.csv` (50 rows; success step PSNR/score/detected per image)
- Files saved under: `.../wb_pgd_1img_ppmark_eps2_steps150_early_retest_exact_batch/`

#### Quick stats (from metrics_all)
- Early-stop step distribution includes 12 images that never reached detection by step 150 (detected=0, step=150).
- Min step=1, max step=150.
- Detected flags include both 1 and 0.

### WIND 50x1 batch — stopped early per user request (2026-01-27)
- User requested to stop the WIND run; background batch was terminated.
- Progress at stop:
  - Output root: `outputs/attacks/muller_forgery_report_prep/forged/wb_pgd_1img_wind_eps2_steps150_stage1obj_early_retest_exact_batch/`
  - `idx000`–`idx010` completed with `metrics.csv` present (11 images done).
  - `idx011` started but has no `metrics.csv` yet (incomplete).
- No `attack_whitebox_forge.py --method wind` process remains running after stop.
- Resume plan: rerun from `start-index 11` (or delete `idx011` and rerun that index first).

### WIND 50x1-image eps=2/255 stage1 exact-repro batch — completed (2026-01-27)
- Output root: `outputs/attacks/muller_forgery_report_prep/forged/wb_pgd_1img_wind_eps2_steps150_stage1obj_early_retest_exact_batch/`
- Per-image folders: `idx000` ... `idx049` with `metrics.csv`, `trace_scores.csv`, `checkpoints.csv`, `summary.json`, `run.log`.
- Aggregated outputs written:
  - `metrics_all.csv` (50 rows)
  - `trace_scores_all.csv` (1160 rows)
  - `checkpoints_all.csv` (53 rows)
  - `trace_scores_deltas.csv` (1160 rows; per-step score/PSNR deltas)
  - `success_psnr_by_image.csv` (50 rows; success step PSNR/score/detected per image)

#### Quick stats (from metrics_all)
- detected_rate = 1.0 (50/50 detected)
- Step stats: mean=22.2, median=19, min=0, max=69
- Score stats: mean=156.784, median=160.046, min=127.129, max=167.836
- PSNR finite stats: mean=55.768, median=56.384, min=44.513, max=58.739
- PSNR inf count: 2
- Linf stats: mean=0.002322, median=0.001987, min=0.0, max=0.007216

### Fairness note: WIND stage1 vs PP-Mark comparison (2026-01-27)
- User concern: using WIND stage2 would change computational cost and make the comparison unfair.
- Decision: keep comparison on identical compute budgets/conditions (same eps=2/255, steps=150, early-stop, trace-all, one-image-per-process) and use WIND stage1 objective only.
- Interpretation: under these *equal-cost* conditions, WIND appears easier to forge than PP-Mark (WIND 50/50 success vs PP-Mark 38/50 success), but this should not be conflated with full-stage (stage2) robustness.

### Forge table (eps=2/255) — initial aggregation for 5 methods (2026-01-27)
- Generated table for: stable_signature, hidden, ringid, wind_stage1, ppmark_score.
- Source metrics:
  - stable_signature: `.../wb_pgd_50img_stable_signature_eps2_steps150_early/metrics.csv`
  - hidden: `.../wb_pgd_50img_hidden_eps2_steps150_early/metrics.csv`
  - ringid: `.../wb_pgd_1img_ringid_eps2_steps150_early_retest_exact_batch/metrics_all.csv`
  - wind_stage1: `.../wb_pgd_1img_wind_eps2_steps150_stage1obj_early_retest_exact_batch/metrics_all.csv`
  - ppmark_score: `.../wb_pgd_1img_ppmark_eps2_steps150_early_retest_exact_batch/metrics_all.csv`
- Output table:
  - `outputs/attacks/muller_forgery_report_prep/forged/summary_tables/forge_table_eps2.csv`
- Columns:
  - detected_rate (FAR@1%FPR), avg_step_detected, avg_psnr_detected_finite, psnr_inf_count_detected.

### Forge table updated with fail_rate (eps=2/255) (2026-01-27)
- Added column: `fail_rate = 1 - detected_rate`.
- Updated file: `outputs/attacks/muller_forgery_report_prep/forged/summary_tables/forge_table_eps2.csv`

### Forge table updated (A: failures counted as 150 for Avg Steps) (2026-01-27)
- Updated table to use Avg Steps to FA with failures counted at the budget cap (150).
- Columns now: FAR@1%FPR, Avg Steps to FA (fail=150), Avg PSNR over accepted samples.
- Updated file: `outputs/attacks/muller_forgery_report_prep/forged/summary_tables/forge_table_eps2.csv`

### Interpretation note: why WIND avg steps > PP-Mark (success-only) with PP-Mark higher variance (2026-01-27)
- PP-Mark objective is image-specific: `_ppmark_score()` builds a per-image sign_map and sample indices from the image's binding/codeword and metadata. This changes the optimization landscape per sample and produces a heavy-tailed distribution (some images cross the threshold in 1–3 steps; others require 100+ steps or fail within eps=2/255).
- WIND stage1 objective is more uniform: `_score_wind_stage1()` optimizes a fixed pattern list + fixed masks/channels, then takes a min-distance over precomputed latents. Because the target structure is consistent across images, successful steps cluster around a narrow band (~19–21), yielding higher mean but much lower variance.
- Therefore, comparing success-only mean steps yields WIND > PP-Mark, while PP-Mark shows substantially larger variance and non-trivial failure rate.

## 2026-01-28
- Documented the single-panel FAR-vs-step plot design for WB forgery comparison (StableSig, Hidden, RingID, WIND, PP-Markscore, plus PP-Markaccept as an explicit zero baseline).
- Wrote a reproducible plot spec with data sources, curve definition, axis/tick/margin rules, and marker cadence so the figure can be redrawn identically or modified later: `outputs/attacks/muller_forgery_report_prep/forged/summary_tables/wb_forge_far_plot_spec.md`.

## 2026-01-28 (continued)
- Implemented a reproducible plotting script for the WB FAR-vs-step curves: `outputs/attacks/muller_forgery_report_prep/forged/summary_tables/plot_wb_forge_far_curves.py`.
  - Script locates the repo root by walking upward until it finds both `src/` and `scripts/` to avoid hardcoded paths.
  - Inputs:
    - StableSig metrics: `.../wb_pgd_50img_stable_signature_eps2_steps150_early/metrics.csv`
    - Hidden metrics: `.../wb_pgd_50img_hidden_eps2_steps150_early/metrics.csv`
    - RingID metrics: `.../wb_pgd_1img_ringid_eps2_steps150_early_retest_exact_batch/metrics_all.csv`
    - WIND metrics: `.../wb_pgd_1img_wind_eps2_steps150_stage1obj_early_retest_exact_batch/metrics_all.csv`
    - PP-Markscore metrics: `.../wb_pgd_1img_ppmark_eps2_steps150_early_retest_exact_batch/metrics_all.csv`
  - Curve definition matches spec: FAR(s) = (# images with detected==1 and step<=s) / N; failed images never contribute to FAR.
  - PP-Markaccept is forced as a constant zero line for all steps.
  - Plot layout: single panel, xlim (-5,155), xticks 0/25/50/75/100/125/150, ylim (0,1), yticks 0/0.25/0.5/0.75/1.0; markers at 10-step intervals.
- Generated reproducible data + figures:
  - `wb_forge_far_curves.csv` (step-by-step FAR values for all 6 lines)
  - `wb_forge_far_curves.png`
  - `wb_forge_far_curves.pdf`

## 2026-01-28 (continued)
- Updated WB FAR plot styling per request:
  - Y-axis label changed to "FAR@1%FPR".
  - Added light gray major gridlines at the x/y major ticks.
  - Marker cadence tightened to every 5 steps (0, 5, 10, …, 150).
  - Added top margin above 1.00 (ylim 0.0–1.02) while keeping the same major ticks.
- Regenerated outputs with the updated styling:
  - `wb_forge_far_curves.csv`
  - `wb_forge_far_curves.png`
  - `wb_forge_far_curves.pdf`

## 2026-01-28 (continued)
- Increased the top margin above FAR=1.00 to double the previous padding (ylim 0.0–1.04) and regenerated the WB FAR curves.

## 2026-01-28 (continued)
- Adjusted WB FAR plot per request:
  - Marker cadence returned to every 10 steps.
  - Added symmetric y-axis margin (ylim -0.04 to 1.04).
  - Raised legend slightly above the lower-right corner.
- Regenerated `wb_forge_far_curves.png` and `.pdf` with the updated settings.

## 2026-01-28 (continued)
- Changed PP-Markaccept line to solid (no dashed style) to match other methods and regenerated the FAR curves.

## 2026-01-28 (continued)
- Updated the WB FAR curves figure caption to focus on setup/conditions rather than PP-Markaccept, using: “Cumulative FAR@1%FPR vs. optimization step (0–150) for all WB methods under the same attack budget and evaluation setup.”
