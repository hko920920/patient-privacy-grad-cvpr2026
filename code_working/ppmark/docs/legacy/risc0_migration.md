# RISC Zero + H100 Migration Notes (ZK-GenGuard v0.3)

> For the fully updated H100/Ubuntu 24.04 GPU plan targeting RISC0 v3.0, see `docs/legacy/risc0_v3_h100_plan.md`. The log below captures the original 1.2.x bring-up notes and known gaps.

This repository now ships a RISC Zero zkVM path alongside the legacy Halo2 flow. The goal is to unblock proof generation on CPU/GPU (H100) while we integrate Poseidon precompiles.

## Layout
- `risc0/`: Cargo workspace.
  - `common/`: Shared structs (`PublicInputs`, `SampleEntry`, `GuestInputs`) used by host/guest via `serde`.
  - `methods/guest`: no_std guest verifying binding + combined relation + Merkle root with SHA-256 (fixed-point sanity checks with `fixed`).
  - `methods/`: embeds guest ELF/ID via `risc0-build`.
  - `host/`: CLI (`zk-genguard-host prove|verify`) that reads `public_inputs.json`/`witness.json`, builds an `ExecutorEnv`, and saves a `receipt.bin`. CUDA is auto-detected when built with `features = ["cuda","prove"]`.

## Build
```bash
cd risc0
# Install RISC Zero toolchain (r0vm + riscv target) first via rzup/cargo-risczero.
# CPU build:
cargo build --release -p zk-genguard-host
# GPU build on H100:
cargo build --release -p zk-genguard-host --features cuda
# Optional H100 tuning at runtime:
#   export RISC0_SEGMENT_LIMIT_PO2=24   # try 25 if VRAM allows (80GB)
#   export RISC0_GPU_DEVICE=0
```

## CUDA build diary (conda CUDA 12.9 on H100 host)
- Toolchain: `conda` env `cuda-dev` (nvcc 12.9 at `$CONDA_PREFIX/bin/nvcc`), libs under `$CONDA_PREFIX/targets/x86_64-linux/lib`.
- Shim setup used repeatedly:
  - `CUDA_SHIM=$CONDA_PREFIX/cuda`; symlinked `cuda/bin -> ../bin`, `cuda/include -> targets/x86_64-linux/include`, `cuda/lib64 -> targets/x86_64-linux/lib`.
  - `libcuda.so` links placed in `cuda/lib64` and `cuda/lib64/stubs` from `/usr/lib/x86_64-linux-gnu/libcuda.so*`.
  - Static libs symlinked: `libcudadevrt.a`, `libcudart_static.a` into `cuda/lib64`, `$CONDA_PREFIX/lib`, `$CONDA_PREFIX/lib64`.
  - Env vars tried (various combinations): `CUDA_PATH/HOME/TOOLKIT_ROOT_DIR=CUDA_SHIM`, `CUDA_LIBRARY_PATH` including `lib64` and `lib64/stubs`, `PATH` with shim/bin, `LD_LIBRARY_PATH` and `LIBRARY_PATH` including `targets/x86_64-linux/lib`, `lib`, `lib64`, stubs, plus `/usr/lib/x86_64-linux-gnu`. `CUDACXX/NVCC` set to shim nvcc. `NVCCFLAGS` with `-gencode=arch=compute_90,code=sm_90` and extra `-L` pointing to target libs.
- Outcome:
  - `find_cuda_helper` issue resolved after exposing `libcuda.so`; nvcc detected.
  - Build now fails in `sppark` linking stage: `ld: cannot find -lcudadevrt` and `-lcudart_static` despite symlinks and `-L` flags. Static runtime files exist at `$CONDA_PREFIX/targets/x86_64-linux/lib/libcudadevrt.a` and `libcudart_static.a`.
- Repo tweak: vendored `sppark-0.1.14` build script now forces CUDA lib search paths from `CUDA_HOME`/`CUDA_PATH`/`CUDA_TOOLKIT_ROOT_DIR`/`CUDA_SHIM`/`CUDACXX`/`CONDA_PREFIX` (adds `-L...` to nvcc device-link and `cargo:rustc-link-search`). Point `CUDA_HOME=$CONDA_PREFIX/cuda` or rely on `CONDA_PREFIX` to surface `targets/x86_64-linux/lib` so `libcudadevrt.a`/`libcudart_static.a` are found.
- Next fix to try: ensure nvcc/ld picks the target lib dir by forcing `LIBRARY_PATH`/`LDFLAGS`/`NVCCFLAGS` to include `$CONDA_PREFIX/targets/x86_64-linux/lib` (already attempted), or place those `.a` files into a path ld searches by default (e.g., `/usr/lib/x86_64-linux-gnu` if permissible). Alternatively, update `sppark`/`cust` to newer versions that honor `CUDA_LIBRARY_PATH` more reliably.

## Python integration
- Default `zk.backend` in `config*.json` is now `risc0`; `prover_cmd`/`verifier_cmd` default to `./risc0/host/target/release/zk-genguard-host`.
- `src/ppmark_v03/risc0_runner.py` serializes the existing witness/public payloads, calls the host binary, and records timings + receipt path in `metadata.json`.
- Merkle + binding hashing for the RISC0 path use SHA-256 to match the guest (Poseidon precompile TODO).

## Guest checks
- Binding: `SHA-256(prompt_hash || seed || secret_key)` vs. public binding.
- Combined relation: `combined = gaussian + alpha * (2*bit - 1)` for each sample (all values are fixed-point integers from Python).
- Merkle root: SHA-256 leaf over `(index, gaussian_fixed, combined_fixed, bit)` folded with SHA-256 siblings; depth must match `sample_merkle_depth`.
- Deterministic RNG: guest runs a short ChaCha20 sequence (seeded by `seed`) and commits a u64 to exercise the PRNG path.

## Poseidon roadmap
- Replace SHA-256 with a Poseidon implementation that targets the RISC Zero BigInt accelerator (Application-Defined Precompile). A `poseidon-rs` no_std fork or a custom implementation over the target field is required.
- Once available, swap the hashing in `risc0_runner.build_sample_merkle_sha` and guest `hash_*` helpers; keep the public/witness JSON format unchanged (32-byte digests).

## Known deviations
- Binding and Merkle root currently use SHA-256, not Poseidon. This keeps the pipeline live in zkVM while the precompile is integrated.
- Halo2 remains available by setting `zk.backend="halo2-kzg"`; config defaults now favor the RISC0 path.

## Status log
- 2025-12-07: Inside non-bind-mounted container. Driver 535.274.02 on H100; CUDA Toolkit 12.2 (nvcc 12.2.91, cicc present); gcc 13.3.0; rustc/cargo 1.91.1. CPU build/flow works (`zk-genguard-host` prove/verify with `artifacts/risc0_sample_cpu` in ~5s). GPU build still fails at link: undefined symbol `sppark_calc_prefix_operation`. crates.io sppark 0.1.14/0.1.10 lack that symbol; RISC0 1.2.6 expects a forked/patch variant. Need to vendor the matching sppark (from RISC0 release-1.2) and patch Cargo to use it; also add sm_90 gencode once unblocked. Container is not bind-mounted to host, so file updates must be copied in explicitly.
- 2025-12-05: CUDA 12.9 in conda; added symlink/shim/env vars; `find_cuda_helper` patched to use `CONDA_PREFIX`; RISC0 CPU build succeeded; GPU link failed to find `-lcudadevrt`/`-lcudart_static`; planned next steps were adding CUDA lib paths or upgrading CUDA.
