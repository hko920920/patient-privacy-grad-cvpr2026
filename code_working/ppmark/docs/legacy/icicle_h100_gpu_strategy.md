# ICICLE H100 GPU Strategy

This note captures how to bring the ICICLE CUDA backend up on NVIDIA H100 (Hopper, sm_90) for the PP-Mark stack. It consolidates the hardware rationale, required toolchain versions, and the exact build/run knobs we need for Halo2.

## Why target sm_90
- Compute capability 9.0 unlocks Hopper SASS (improved INT32/FP32 units → faster finite-field math).
- Enlarged L2 (≈60MB on SXM) reduces NTT/MSM cache misses.
- HBM3 delivers ~3 TB/s; Hopper memory controllers reward coalesced access (ICICLE implements this, legacy ec-gpu does not).
- Tensor Memory Accelerator (TMA) supports async global↔shared moves without tying up registers; ICICLE’s roadmap experiments with this, ec-gpu cannot use it.

## Toolchain matrix (H100)
- OS: Ubuntu 22.04+ recommended (20.04 minimal).
- NVIDIA driver: 535/545+; CUDA Toolkit: 12.2+.
- CMake: 3.24+ for `CMAKE_CUDA_ARCHITECTURES` native handling.
- Host compiler: GCC 11/12 (CUDA 12.x compatible).

## Build the CUDA backend for sm_90
Use the helper script so we stop shipping sm_80 cubins that panic on H100 (`no kernel image is available...`).

```bash
# From repo root
CUDA_ARCH=90-real scripts/build_icicle_h100.sh
```

What the script does:
- Configures `icicle/icicle` with `-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH:-90-real}`, `-DBUILD_TESTS=ON`, `-DBUILD_SHARED_LIBS=ON`.
- Builds/installs into `icicle/build/h100-sm90/install`.
- If `icicle/icicle/backend/cuda` is present, sets `-DCUDA_BACKEND=local`; otherwise it warns (CPU-only frontend) so you can drop the Ingonyama CUDA backend sources or point `ICICLE_BACKEND_INSTALL_DIR` at prebuilt binaries.
- Runs `cuobjdump` when available to confirm `sm_90` sections exist.

Key outputs:
- Frontend libs: `icicle/build/h100-sm90/install/lib/`
- CUDA backend: `icicle/build/h100-sm90/install/lib/backend/`

If you rebuild elsewhere, align `CMAKE_CUDA_ARCHITECTURES` (include `90` or `90-real`) and verify with `cuobjdump <lib>.so | grep sm_90`.

## Wiring to Halo2 / Rust
- Environment for prover/verifier runs:
  - `ICICLE_BACKEND=CUDA`
  - `ICICLE_BACKEND_INSTALL_DIR=/abs/path/to/icicle/build/h100-sm90/install/lib/backend`
  - `LD_LIBRARY_PATH=/abs/path/to/icicle/build/h100-sm90/install/lib:${LD_LIBRARY_PATH}`
- If building via `icicle-cuda-runtime`/`build.rs`, pass `CUDA_ARCH=90` (or `90-real`) to `cargo build --release` so the crate compiles kernels for Hopper.
- Select the GPU explicitly at runtime when using the Rust bindings: `icicle_cuda_runtime::set_device(0)?;`.

## Performance/leads
- Aim for MSM sizes ≥2^18 or batch small circuits so Hopper occupancy stays high.
- Prefer pinned host buffers for transfers; H100 over PCIe Gen5 benefits strongly from `cudaHostAlloc`/`HostSlice` when moving SRS/witness data.
- Check PCIe link: `lspci -vv | grep -i LnkSta` → expect `Speed 32GT/s (Gen5)`; Gen4 halves the throughput.

## Troubleshooting
- `no kernel image is available for execution on the device`: rebuild with `CMAKE_CUDA_ARCHITECTURES=90` (script above) and rerun.
- `symbol lookup error`: ensure Rust crates and C++ backend are built from the same commit/tag; mismatch between .so and crate ABI causes this.
- GPU idle: confirm `ICICLE_BACKEND=CUDA` and `ICICLE_BACKEND_INSTALL_DIR` point to the sm_90 build; otherwise the CPU backend will be selected.

See `docs/legacy/halo2_alignment_notes.md` for the historical H100 debug log and how this replaces the old sm_80 binaries.
