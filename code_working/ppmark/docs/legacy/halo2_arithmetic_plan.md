# Plan: Replace ICICLE with halo2-arithmetic (CUDA) for H100

## Goal
Drop ICICLE entirely and use an open GPU backend (halo2-arithmetic) to accelerate MSM/FFT/graph evaluation for Halo2 (BN254 + KZG) on H100.

## Target stack
- `halo2_proofs`: PSE upstream (or a pinned fork) with BN254/KZG.
- `halo2-arithmetic`: https://github.com/dompute/halo2-arithmetic with `features = ["cuda"]`, built for `sm_90`.
- BN254 curve, KZG commitment, Blake2b transcript (unchanged at the prover layer).

## Phase 1: Playground (validate H100 CUDA path)
1) Clone repos
   - `git clone https://github.com/privacy-scaling-explorations/halo2 /tmp/pse_halo2`
   - `git clone https://github.com/dompute/halo2-arithmetic /tmp/halo2-arithmetic`
2) Add H100 arch to `halo2-arithmetic` NVCC flags
   - Edit `/tmp/halo2-arithmetic/build.rs`, ensure NVCC args include `-gencode=arch=compute_90,code=sm_90` (and keep existing sm_80/86 if needed).
3) Create `halo2_gpu_playground` (separate crate)
   - `halo2_proofs = { path = "/tmp/pse_halo2/halo2_proofs" }`
   - `halo2-arithmetic = { path = "/tmp/halo2-arithmetic", features = ["cuda"] }`
   - Ensure `halo2curves/ff/group` are single-sourced (use `[patch.crates-io]` if versions diverge).
4) Wire MSM/FFT/Graph to halo2-arithmetic
   - In the vendored `halo2_proofs` (playground), re-export or call `halo2_arithmetic::best_msm`, `best_fft`, and `GraphEvaluator`.
   - If signatures differ, add thin adapters inside `halo2_proofs` to keep the public API stable.
5) Smoke test on H100
   - Simple BN254 circuit (Poseidon hash of two inputs), `ParamsKZG::setup(k=20)`, `create_proof/verify_proof`.
   - Run with `LD_LIBRARY_PATH=/usr/local/cuda/lib64` and `nvidia-smi` monitoring; confirm GPU Util spikes and no “no kernel image” errors.

## Phase 2: Integrate into PP-Mark
1) Swap dependencies in `halo2_prover/Cargo.toml`
   - Remove Ingonyama/ICICLE halo2_proofs dependency.
   - Add path/git deps to the validated `halo2_proofs` and `halo2-arithmetic` (with `features = ["cuda"]`).
2) Apply the same MSM/FFT/Graph mapping
   - Mirror the playground changes into the PP-Mark halo2_proofs copy (or use the vendored upstream).
   - Keep existing prover code (BN254, KZG, Blake2b) unchanged.
3) Dependency unification
   - Use `[patch.crates-io]` to pin `halo2curves`, `ff`, `group` to the same source as `halo2_proofs`/`halo2-arithmetic` to avoid type splits.
4) Validation on PP-Mark
   - Run `cargo +nightly-2024-07-18 test --release --test kzg_integration -- --nocapture` (or k=20 integration test) with `nvidia-smi` monitoring.
   - Compare CPU vs GPU wall-clock times; ensure GPU Util is non-zero.

## Risks / Watchouts
- **Version splits**: Ensure `halo2curves/ff/group` are single-version across all crates; patch if needed.
- **Signature mismatches**: If `best_msm`/`best_fft` signatures differ, wrap them inside `halo2_proofs` to avoid touching prover code.
- **CUDA arch**: Must include `sm_90` in NVCC flags for H100 to avoid “no kernel image” errors.
- **Stability**: `halo2-arithmetic` is experimental; keep a pre-change tag/branch to roll back if needed.

## Attempt log (status updates)
- Tried `halo2-arithmetic` → blocked: `libfam` submodule is private; cannot fetch CUDA sources, so unusable.
- Switched to PSE `halo2` + `ec-gpu`/`ec-gpu-gen` path:
  - Added `ecgpu` feature, `ec-gpu-gen 0.7.1` (with cuda/rust-gpu-tools), `build.rs` kernel generation (`add_fft::<Fr>()`, `add_multiexp::<G1Affine,Fq>()`), and pinned `halo2curves` rev.
  - Cargo/toolchain: upgraded to 1.91.1 (edition2024 OK); earlier nightly was too old.
  - Current blocker: `halo2curves::bn256::{Fr,Fq,G1Affine}` lack `ec_gpu::GpuField`/`GpuName` impls, so `ec_gpu_gen::add_fft/add_multiexp` fails. Need to implement these traits for BN256 types, then rebuild with `EC_GPU_CUDA_NVCC_ARGS="--fatbin --gpu-architecture=sm_90 --generate-code=arch=compute_90,code=sm_90"` and verify GPU utilization.
  - Update: vendored `halo2curves` (rev b753a83) under `halo2_prover/halo2curves` with optional `ecgpu` feature (`ec-gpu 0.2.0`) and added `GpuField`/`GpuName` impls for `bn256::{Fr,Fq,G1Affine}`; `halo2_prover/Cargo.toml` now patches dependencies to this local copy and enables the feature.
