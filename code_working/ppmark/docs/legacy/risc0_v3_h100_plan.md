# RISC0 v3.0 GPU Migration Plan (H100, Ubuntu 24.04, Driver 535)

This plan upgrades the PP-Mark v0.3 RISC0 path from the current 1.2.x toolchain to the H100-ready RISC0 v3.0 stack. It is scoped to the existing repo layout (`risc0/` Rust workspace, `src/ppmark_v03/risc0_runner.py` call-out) and assumes Ubuntu 24.04 with NVIDIA driver 535 on H100.

## Target software stack
| Component | Version | Rationale |
| --- | --- | --- |
| RISC0 SDK / crates | 3.0.4 | Latest stable with GPU-first prover architecture. |
| sppark | 0.1.10 (bundled) | Fixed by RISC0; no manual install when `features = ["cuda"]` is enabled. |
| CUDA Toolkit | 12.2.2 | Native match for driver 535.274.02; supports SM90/90a. |
| GCC / G++ | 12.x | Max GCC officially supported by CUDA 12.2; avoids Ubuntu 24.04 default gcc-13. |
| Rust | 1.81+ | Satisfies RISC0 v3.0 crate requirements (managed via `rzup`/`rustup`). |
| LLVM/Clang | 16+ | Needed by `bindgen` during `risc0-sys` build. |

GPU tuning defaults for H100:
- `RISC0_CUDA_ARCH=90a` (fallback to `90` if a compiler rejects `90a`).
- `RISC0_SEGMENT_LIMIT_PO2=24` (raise to 25 if VRAM allows).
- `RISC0_GPU_DEVICE=0` to pin to the first H100.

## Step-by-step host setup
Run on Ubuntu 24.04 (no reboot required).

```bash
# 0) Base tools
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential curl git libssl-dev pkg-config

# 1) GCC 12 (keeps system gcc-13 untouched)
sudo apt install -y gcc-12 g++-12
gcc-12 --version

# 2) CUDA 12.2.2 toolkit only (preserves driver 535)
wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run
sudo sh cuda_12.2.2_535.104.05_linux.run --toolkit --silent --override
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 3) RISC0 toolchain
curl -L https://risczero.com/install | bash
source ~/.bashrc
rzup install               # installs rustc + cargo-risczero
cargo risczero --version   # expect v3.0.4
```

## Build matrix (repo-local)
From repo root (`PP-Mark/`):

```bash
# CPU prover
pushd risc0
cargo build --release -p zk-genguard-host
popd

# GPU prover (H100)
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.2}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CUDA_HOST_COMPILER=/usr/bin/gcc-12
export RISC0_CUDA_ARCH=${RISC0_CUDA_ARCH:-90a}
pushd risc0
cargo build --release -p zk-genguard-host --features cuda
popd
```

Shortcut: `./scripts/build_risc0_h100.sh [cpu|cuda]` applies the env vars above and runs the correct `cargo build` invocation from the repo root.
The script also sets `CARGO_TARGET_DIR=target-h100` to avoid permission conflicts with the existing `target/` directory and falls back to `/usr/bin/gcc` if `gcc-12` is unavailable (still prefer gcc-12 for CUDA 12.2).

Notes:
- Do **not** change the system-wide `gcc` symlink; the env vars above scope GCC 12 to CUDA/RISC0 only.
- sppark is pulled automatically through `risc0-zkp`; the workspace patches (`vendor/`) stay in place unless the upstream SDK bundles a newer variant.
- If NVCC reports missing `libcudadevrt`/`libcudart_static`, ensure `${CUDA_HOME}/lib64` is on both `LIBRARY_PATH` and `LD_LIBRARY_PATH`.
- If `protoc` is not installed system-wide, download a portable release (e.g., `protoc-25.3-linux-x86_64.zip` from protobuf GitHub), extract it under `~/.local/tools/protoc-25.3`, and set `PROTOC=~/.local/tools/protoc-25.3/bin/protoc` before building.

## Runtime checks
- Proven receipt on CPU (sample fixtures):  
  `./risc0/host/target/release/zk-genguard-host prove --public artifacts/risc0_sample_cpu/public_inputs.json --witness artifacts/risc0_sample_cpu/witness.json --receipt artifacts/risc0_sample_cpu/receipt.bin`
- GPU sanity: rebuild with `--features cuda` + `RISC0_CUDA_ARCH=90a`, then run the same command; GPU usage can be confirmed with `nvidia-smi` or by comparing prover time.
- RISC0 upstream benchmark (optional, outside repo): `cargo run --release --example datasheet --features cuda` after cloning the RISC0 repo and checking out `v3.0.4`.

## Integration points inside PP-Mark
- Python CLI already targets `zk.backend="risc0"` by default and calls `./risc0/host/target/release/zk-genguard-host prove|verify`. Building the host with `--features cuda` is sufficient to switch the prover to GPU.
- Set `RISC0_CUDA_ARCH`/`RISC0_GPU_DEVICE` in the environment before invoking `python -m ppmark_v03.cli prover ... --device-backend cuda` to keep the CUDA arch aligned with the host binary.
- The witness/public JSON layout remains unchanged (SHA-256 binding + sample Merkle). Poseidon precompile integration stays a follow-up task.

## Migration tasks (to execute in the repo)
1) **Crate bump**: Move `risc0-zkvm`/`risc0-build`/`risc0-zkp` to `=3.0.4`, re-run `cargo update -p sppark` to pin the SDK-bundled CUDA backend, and regenerate `Cargo.lock`.  
2) **API alignment**: Update `risc0/host/src/main.rs` to the v3 prover API (session/prover opts) and ensure `ExecutorEnv`/`receipt.verify` calls match the new interfaces.  
3) **CUDA pathing**: Keep the `[patch.crates-io] find_cuda_helper` override; verify `CUDA_HOME`/`CUDACXX` detection works with CUDA 12.2.2.  
4) **H100 arch defaults**: Add `RISC0_CUDA_ARCH` export to build scripts and CI (if present) so sm_90a cubins are produced.  
5) **Bench + artifacts**: Re-run the `artifacts/risc0_sample_cpu` flow on CPU and GPU, capture prover timings in `metadata.json`, and store receipts for regression.  
6) **Docs/README**: Keep this plan linked from `README.md` and `docs/legacy/risc0_migration.md` so future contributors follow the same stack.

## Performance switches
- Huge pages to reduce TLB pressure: `sudo sysctl -w vm.nr_hugepages=1024`
- Persistent mode: `sudo nvidia-smi -pm 1`
- If `sm_90a` fails to compile on a specific toolkit, fall back to `RISC0_CUDA_ARCH=90` (still Hopper-native).
