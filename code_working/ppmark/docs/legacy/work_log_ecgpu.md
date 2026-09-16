# EGPU Integration Work Log (English)

## Current tree and toolchain
- PP-Mark `halo2_prover` now targets the PSE Halo2 0.4 split stack (frontend/backend/curves) pinned to the local checkout at `/tmp/pse_halo2` with `ecgpu` features enabled.
- Poseidon gadget comes from `/tmp/poseidon-gadget` and is patched to use the same PSE workspace (frontend/proofs/curves). ICICLE is **not** used anywhere in the current build.
- CUDA/NVCC is available (`nvcc 12.9.86` via conda); `ec-gpu-gen` 0.7.1 compiles kernels successfully with `EC_GPU_FRAMEWORK=cuda`.
- `halo2_prover` features: `ecgpu` (pulls ec-gpu-gen + PSE ecgpu paths) and `cuda` (enables the rust-gpu-tools/cuda flags in ec-gpu-gen). ICICLE remains gated behind its own feature.

## Code changes (Feb 2025)
1) **Poseidon gadget alignment**
   - Unified error types to `halo2_frontend::plonk::Error`.
   - Fixed `Any` import (now taken from `halo2_proofs::plonk::Any`).
   - Cargo patched to force `halo2_frontend`, `halo2_proofs`, and `halo2curves` to `/tmp/pse_halo2`, preventing mixed 0.3/0.4 deps.

2) **halo2_prover deps & features**
   - `Cargo.toml`: added `ecgpu`/`cuda` feature wiring; ec-gpu-gen 0.7.1 optional dep; PSE halo2 stack + poseidon-gadget pinned via `[patch]`.
   - `build.rs`: ec-gpu-gen kernel build hook present to generate FFT/MSM binaries.
   - Binaries (`src/main.rs`, `src/bin/new_prover.rs`, `src/bin/new_verifier.rs`) updated to the 0.4 API: instances use `Vec<Vec<Vec<Fr>>>`, `SingleStrategy::new(&verifier_params)`, and verifier calls use `halo2_backend::plonk::verifier::verify_proof_with_strategy`.
   - Watermark circuit (`src/new/watermark.rs`): error type consolidated to frontend `Error`; Poseidon helpers updated; verify path uses backend verifier API.

3) **GPU runtime (ec-gpu-gen 0.7.1)**
   - `src/ecgpu_runtime.rs`: wraps `FftKernel` and `MultiexpKernel` with thread-local caching, truncates to a single GPU, and validates MSM against the CPU path (falling back unless `ECGPU_TRUST_GPU_MSM` is set). GPU errors also fall back to CPU.
   - `halo2_backend` is brought in via path patch with its own ecgpu runtime mirroring the CPU validation logic; warnings from `program!` macro about unexpected cfg remain but are non-fatal.
   - `tests/ecgpu_smoke.rs`: GPU FFT/MSM parity test is enabled; FFT matches CPU DFT on the smoke vector. MSM asserts equality with the CPU result and can enforce GPU-only parity with `ECGPU_ENFORCE_GPU_MSM=1` (forces `ECGPU_TRUST_GPU_MSM=1` inside the test).

4) **Arithmetic/test scaffolding**
   - `bn254_poseidon.rs` added for Poseidon params/spec.
   - Added Poseidon debug helpers and watermark prover/verifier binaries aligned with the new stack.

## Build/Test status
- Command: `cd halo2_prover && cargo test --release --features ecgpu -- --nocapture`
- Result: **PASS** (with CUDA toolchain present). All unit tests green; `new::watermark::binding_matches_poseidon_hash_real_proof_k20` ignored (explicit). `tests/ecgpu_smoke` runs by default and passes via CPU-validated MSM; set `ECGPU_ENFORCE_GPU_MSM=1` to fail fast on any GPU MSM mismatch. Upstream warnings remain (unexpected cfg in `program!`, `#[must_use]` on trait methods in PSE crates).
- Command (compile-only, user request):  
  `conda run -n cuda-dev --no-capture-output bash -lc 'export EC_GPU_FRAMEWORK=cuda; export EC_GPU_CUDA_NVCC_ARGS="--fatbin --gpu-architecture=sm_90 --generate-code=arch=compute_90,code=sm_90"; export PATH="$CONDA_PREFIX/bin:$PATH"; export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"; cd /home/mimic/PP-Mark-v0.31/PP-Mark/halo2_prover; cargo test --release --features ecgpu --no-run'`
- Result: **PASS (build only)**. All crates compile; same upstream warnings as above.
- Kernels were built with nvcc; no ICICLE code paths executed.

## Outstanding items / risks
- GPU MSM parity still needs to be rechecked on real hardware with `ECGPU_ENFORCE_GPU_MSM=1`; default path uses CPU fallback on mismatch/error.
- `program!` macro emits cfg warnings (`opencl`), but build is unaffected; could be silenced by adjusting cfgs in ec-gpu-gen/halo2_backend if desired.
- Upstream PSE crates emit numerous warning noise (unused #[must_use], unexpected cfg). Not harmful but may require patching if warnings-as-errors is enabled.

## How to reproduce
```bash
cd halo2_prover
export EC_GPU_FRAMEWORK=cuda
export EC_GPU_CUDA_NVCC_ARGS="--fatbin --gpu-architecture=sm_90 --generate-code=arch=compute_90,code=sm_90"
cargo test --release --features ecgpu -- --nocapture
```
