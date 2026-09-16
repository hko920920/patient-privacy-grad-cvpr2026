# SDXL Crop Rescue Plan — 2026-04-19

## Baseline
- **Previous SDXL run** (no shift, no rotation, no crop-refine):
  - `crop_scale_0.75` soft_pass = 0/100 (0%)
  - scores: min=0.01, max=2.20, mean=0.73
  - tau = 2.4666
- **SD 2.1 reference** (with shift=4, rot=90, refine):
  - soft_pass = 70/100 (70%)
  - scores: min=2.00, max=4.23, mean=2.67

Gap: SDXL is fundamentally ~4× worse than SD 2.1. Missing levers:
- `--max-shift 4` (SD 2.1 used)
- `--max-rotation 90` (SD 2.1 used, though we now know grid step 5 is better)
- `--crop-refine` (maybe not applied on previous SDXL run despite dir name)

## Stage 1 Smoke (20 imgs, ~6 min expected)
**Match SD 2.1 minimum + fixed rotation grid**
```
--crop-refine \
--max-shift 4 \
--max-rotation 15 --rotation-step 5 --rotation-strategy grid
```
Keep bf16 + 50 DDIM steps. Expected runtime ~3.8s per image (current baseline) × 20 = ~1–2 min.
If pass rate ≥ 50% → proceed to Stage 4 full directly.
If pass rate 10–49% → Stage 2.
If pass rate < 10% → Stage 3 immediately.

## Stage 2 Smoke (20 imgs, ~10 min)
**Add precision: fp32 + 100 steps**
```
--crop-refine \
--max-shift 4 \
--max-rotation 15 --rotation-step 5 --rotation-strategy grid \
--ddim-dtype float32 --ddim-steps 100
```

## Stage 3 Smoke (20 imgs, ~15 min)
**Add dense crop search**
```
--crop-refine \
--max-shift 4 \
--max-rotation 15 --rotation-step 5 --rotation-strategy grid \
--ddim-dtype float32 --ddim-steps 100 \
--crop-grid 5 --crop-refine-grid 9 --crop-refine-step 0.025
```

## Stage 4 Full (100 imgs)
Runtime estimate based on smoke timing × 5.
Output: `outputs/sdxl_experiments/geom_eval_sdxl_full100/eval_crop_rescue/`

## Target
- pass@tau=2.4666 ≥ 0.30 (acceptable narrative)
- pass@tau=2.4666 ≥ 0.50 (strong narrative)

## Sequencing
GPU 0 only (GPU 1 busy with WB PGD).
After Rotation full (PID 4057911, ~35 min remaining) completes:
1. Smoke Stage 1
2. Smoke Stage 2 if needed
3. Smoke Stage 3 if needed
4. Full run with best combo
