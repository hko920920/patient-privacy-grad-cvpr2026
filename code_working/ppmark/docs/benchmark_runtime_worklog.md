# Runtime Benchmark Work Log

## 2026-03-26

### Goal
Create a runtime benchmark comparing generation and verification times for 7 watermark methods, to answer Reviewer Y5S2 Q3 ("Can the authors provide runtime comparisons with existing watermarking baselines?").

### Design
- **Methods**: Tree-Ring, Gaussian Shading, RingID, Stable Signature, HiDDeN, WIND, PP-Mark
- **Images**: 10 per method
- **Hardware**: GPU 1 (H100 PCIe), SD2.1 (Manojb/stable-diffusion-2-1-base), DDIM 50 steps, 512x512, float16
- **Generation**: Measure watermark embedding + pipe() time. PP-Mark additionally measures proof generation separately.
- **Verification**: Use existing eval images. Measure DDIM inversion separately from method-specific detection.
- **HiDDeN**: Post-hoc (encoder on clean image), marked separately.
- **WIND**: No pre-existing eval images; generate 10 and reuse.
- **Pipeline loading**: Excluded from timing.
- **Timer**: time.perf_counter()
- **Output**: JSON with mean±std per method per metric.

### Progress

#### Step 1: Examine each method's generation and verification entry points
- Status: DONE
- Read all 7 batch scripts (run_*_batch.py) and calibrate scripts (calibrate_*.py)
- Identified generation/verification patterns per method

#### Step 2: Write benchmark script
- Status: DONE
- Created scripts/benchmark_runtime.py (1008 lines)
- Initial version had 3 bugs: PP-Mark VAE-only, RingID gen skipped, .cuda() hardcoded
- All 3 fixed and re-verified by code-paper-checker agent
- All 10 design checkpoints PASS

#### Step 3: Execute benchmark
- Status: DONE (after 6 attempts)
- Fixes applied: datasets pkg, pycryptodome pkg, RingID heter_mask None, RingID device mismatch, PP-Mark image path glob, PP-Mark config Path type, PP-Mark RS codec removal, deterministic_sample signature
- Generation: 7/7 methods completed (from 4th run log)
- Verification: 7/7 methods completed (6th run)
- Output: outputs/benchmark_runtime/results.json

#### Results Summary
**Verification (n=10, H100 PCIe, SD2.1 512x512 DDIM 50 steps):**
- DDIM-based methods (TR, GS, RingID, PP-Mark): all ~0.72s (inversion dominates)
- WIND: ~1.10s (different inversion pipeline)
- Decoder-based (StableSig): ~0.008s, HiDDeN: ~0.001s
- PP-Mark_accept adds +7.59s for receipt verification
- Key finding: PP-Mark_score verification is identical to other DDIM-based methods

**Generation (n=10):**
- All latent-injection methods (TR, RingID, WIND, PP-Mark): ~0.89s pipe
- GS: 2.21s embed + 0.89s pipe = 3.10s (truncated sampling)
- StableSig: 0.91s (modified VAE decoder)
- HiDDeN: 0.001s (post-hoc encoder only)
- PP-Mark adds +48.6s for proof generation
