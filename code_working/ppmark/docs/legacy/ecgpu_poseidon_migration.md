# Poseidon Gadget Swap & ECGPU Build Status (March 2025)

This log records the steps taken to replace the missing `halo2_gadgets` dependency with `poseidon-gadget` and the current ECGPU build status.

## What was changed
- Cloned Poseidon gadget repo: `git clone https://github.com/privacy-scaling-explorations/poseidon-gadget /tmp/poseidon-gadget`.
- Swapped dependencies in `halo2_prover/Cargo.toml`:
  - Removed `halo2_gadgets` dependency; added `halo2_poseidon = { path = "/tmp/poseidon-gadget" }`.
  - Added `[patch."https://github.com/privacy-scaling-explorations/halo2"]` entries to force all Halo2 crates (proofs/backend/frontend/curves) to use the `/tmp/pse_halo2` workspace, matching the existing `[patch.crates-io]`.
- Updated Poseidon imports from `halo2_gadgets::poseidon::…` to `halo2_poseidon::poseidon::…` in:
  - `halo2_prover/src/bn254_poseidon.rs`
  - `halo2_prover/src/new/watermark.rs`
  - `halo2_prover/src/main.rs`
  - `halo2_prover/src/bin/new_prover.rs`
  - `halo2_prover/src/bin/poseidon_debug.rs`
  - `halo2_prover/tests/kzg_integration.rs`
- Poseidon gadget crate adjustments (`/tmp/poseidon-gadget`):
  - `Cargo.toml` now pins `halo2_proofs`, `halo2_frontend`, and `halo2curves` to `/tmp/pse_halo2` via direct paths and `[patch.crates-io]`.
  - Error type unified to `halo2_frontend::plonk::Error` in `src/poseidon.rs`, `src/poseidon/pow5.rs`, `src/utilities.rs`, and `src/utilities/cond_swap.rs` to match the frontend layouter API.

## Current build status
- Command attempted: `cargo check -p halo2_prover` (from `halo2_prover/`).
- Result: fails in `halo2_backend` build script (`ec-gpu-gen`) because `nvcc` is missing:
  - Panic: `Cannot run nvcc. Install the NVIDIA toolkit or disable the cuda feature.`
- Interpretation: ECGPU path requires CUDA toolkit (nvcc). Without it, the build stops before reaching the Halo2 prover crate.
- `halo2_poseidon` build was not re-run to completion; the default toolchain pulled `cargo` 1.72 (no edition 2024 support). Use `cargo +1.91.1 check` (or newer) after ensuring CUDA/nvcc if ECGPU remains enabled.

## Next steps
1) Install CUDA/nvcc (include sm_90 flags for H100, e.g., `EC_GPU_CUDA_NVCC_ARGS="--gpu-architecture=sm_90"`), then rerun `cargo +nightly build --release --features ecgpu`.
2) If GPU toolchain is unavailable, disable the `ecgpu`/`cuda` path temporarily (CPU-only build) to allow `halo2_prover` to compile.
