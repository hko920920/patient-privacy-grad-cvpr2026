# EGPU Work Summary (current snapshot)

## Context
- Project: PP-Mark v0.31 (`halo2_prover` with PSE Halo2 0.4 stack, ec-gpu-gen 0.7.1, BN256).
- Goal: correct GPU FFT/MSM integration (ecgpu) and keep CPU-safe fallbacks.
- Initial note was “no nvcc”; later confirmed `conda run -n cuda-dev nvcc 12.9` is available on the H100 box and tests were run there.

## Changes completed
1) **Field constants (BN256)**
   - Fixed `GpuField::one()` for `Fr` and `Fq` to return Montgomery `1` (`R mod modulus`) in little-endian limbs.
   - Regenerated kernel sources are expected to pick up correct constants when built with `nvcc`.

2) **GPU runtime hardening**
   - `halo2_prover/src/ecgpu_runtime.rs` and `/tmp/pse_halo2/halo2_backend/src/ecgpu_runtime.rs` now:
     - Use a single GPU device for determinism.
     - Wrap MSM with CPU validation: GPU result is compared to CPU multiexp; on mismatch or GPU error the CPU result is returned. Opt-out via `ECGPU_TRUST_GPU_MSM=1`.
     - Experimental scalar bit-reversal (LSB↔MSB) added before GPU MSM to match kernel bit extraction; parity still fails.
     - Handle GPU errors by cleanly falling back to CPU instead of bubbling an error by default.

3) **Tests**
   - `halo2_prover/tests/ecgpu_smoke.rs` enabled. Covers 8-point FFT and MSM.
   - Added strict GPU parity knob: `ECGPU_ENFORCE_GPU_MSM=1` forces a GPU-only MSM check (internally sets `ECGPU_TRUST_GPU_MSM=1`) and will fail on any GPU mismatch.
   - Default behavior passes by relying on CPU-validated MSM.

4) **Docs updated**
   - `docs/legacy/gpu_halo2_status.md` and `docs/legacy/work_log_ecgpu.md` reflect: FFT now matches CPU, MSM uses CPU fallback unless trusted, strict mode instructions, remaining risks.

## Status / known gaps
- FFT: passes the smoke vector.
- MSM: GPU still mismatches CPU in strict mode (bit-reversal attempt didn’t fix it). Default path passes via CPU fallback.
- Tests run on H100 (`conda run -n cuda-dev`); strict GPU MSM fails with differing affine outputs.

## How to verify on a CUDA host
```bash
cd halo2_prover
export EC_GPU_FRAMEWORK=cuda
# Optional: tune nvcc flags; example for H100:
export EC_GPU_CUDA_NVCC_ARGS="--fatbin --gpu-architecture=sm_90 --generate-code=arch=compute_90,code=sm_90"

cargo test --release --features ecgpu ecgpu_fft_and_msm_smoke -- --nocapture
# Strict GPU-only MSM check (fails on mismatch):
ECGPU_ENFORCE_GPU_MSM=1 cargo test --release --features ecgpu ecgpu_fft_and_msm_smoke -- --nocapture
```

## Notes for follow-up
- Strict mode currently fails: capture GPU vs CPU MSM outputs and inspect scalar encoding/bit order or kernel windowing.
- Consider trying Montgomery-scalar conversion for exponents or kernel-side bit ordering adjustments.
- Keep `ECGPU_TRUST_GPU_MSM` unset in production unless GPU parity is fully confirmed.
