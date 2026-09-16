# Steg Experiment Check Log

## 2025-02-14

### Goal
- Verify whether steganalysis (steg removal) experiments were completed.
- Locate outputs, summaries, and plots.
- Note any documentation mismatches.

### Commands executed
- `rg -n "steg|stegan|steganalysis" -S docs outputs scripts src`
- `ls -l outputs/attacks/muller_forgery_report_prep/steg_removal`
- `ls -l outputs/attacks/muller_forgery_report_prep/figures | rg -n "steg_removal"`
- `rg -n "Steg|steg" -n docs/experiment_set_plan.md`

### Findings
- Steg-removal outputs exist under `outputs/attacks/muller_forgery_report_prep/steg_removal/`.
- Per-method directories present: `pp_mark`, `stable_signature`, `hidden_revisited`, `tree_ring`.
- Aggregated summary exists: `outputs/attacks/muller_forgery_report_prep/steg_removal/steg_removal_summary.csv`.
- Plots exist: `outputs/attacks/muller_forgery_report_prep/figures/steg_removal_{tpr,psnr,lpips}_vs_k.{png,pdf}`.
- The planning doc still marks steganalysis as ongoing in `docs/experiment_set_plan.md`.

### Quality analysis (PSNR/LPIPS)
- Parsed `steg_removal_summary.csv` to read `psnr_mean` and `lpips_mean` by method and K.
- Observed monotonic quality improvements as K increases for all methods (PSNR up, LPIPS down).
- Stable Signature shows the highest PSNR and lowest LPIPS across all K.
- HiDDeN shows mid-high PSNR and low LPIPS; PP-Mark starts with lower quality at K=10 but improves sharply by K=1000.
- Tree-Ring remains the lowest PSNR and highest LPIPS even after K=1000, indicating the most visible distortion under removal.

### Metric definition check
- Reviewed `outputs/attacks/muller_forgery_report_prep/scripts/run_steg_removal_suite.py`.
- PSNR/LPIPS are computed between the attacked image and the *watermarked* image (`wm`), not the clean original.
- The attack applies `attacked = wm - residual`, where `residual` is the average `(wm - clean)` estimated from paired data.
- Therefore high PSNR means the removal operation makes only a tiny change to the watermarked image (low distortion), not necessarily higher absolute fidelity to the clean image.

### Conclusion
- The steganalysis removal experiment appears to have been run and summarized.
- Documentation should be updated to reflect completion if desired.
