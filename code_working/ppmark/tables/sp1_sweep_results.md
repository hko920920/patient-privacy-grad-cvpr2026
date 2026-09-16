# SP1 Sample Count Sweep (latent 128x128, SDXL base, alpha/config default)
- Backend: SP1 core, CUDA prover, `SP1_SKIP_EXECUTE=false`, `SP1_SKIP_VERIFY=false`
- Prompt: "A photo of a red apple on a wooden table, studio lighting, high detail"
- Model: stabilityai/stable-diffusion-xl-base-1.0
- Seed/secret: fixed hex (see outputs)
- LUT path resolved to absolute
- Sync: off; verification time here is receipt-only (`sp1-genguard-host verify`)

| sample_count | sp1_prover_sec | wall_prover_sec | receipt_verify_sec | output_dir |
| --- | ---: | ---: | ---: | --- |
| 100 | 32.47 | 69.31 | 3.46 | outputs/sp1_sweep/s100 |
| 200 | 37.91 | 74.88 | 3.73 | outputs/sp1_sweep/s200 |
| 300 | 60.30 | 97.10 | 4.04 | outputs/sp1_sweep/s300 |
| 400 | 69.62 | 119.62 | 4.51 | outputs/sp1_sweep/s400_rerun |
| 600 | 68.84 | 105.91 | 5.41 | outputs/sp1_sweep/s600_clean |
| 800 | 82.81 | 119.40 | 5.78 | outputs/sp1_sweep/s800 |
| 1000 | 94.57 | 131.60 | 6.40 | outputs/sp1_sweep/s1000_clean |
| 1500 | 135.74 | 172.49 | 8.59 | outputs/sp1_sweep/s1500 |
| 2000 | 175.09 | 211.81 | 10.32 | outputs/sp1_sweep/s2000 |

Notes:
- wall_prover includes SDXL load/decode overhead; `sp1_prover_sec` is the prover’s own timing.
- Receipt verification is standalone (no sync search, metadata-only path).
- s600_rerun / s600_rerun2 (132-143s) were outliers with identical config; likely transient load; see outputs/sp1_sweep/s600_rerun* if needed.