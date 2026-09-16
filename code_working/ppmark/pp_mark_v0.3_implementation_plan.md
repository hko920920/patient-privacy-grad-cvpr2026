# ZK-GenGuard V2 (pp_mark_v0.3) Implementation Plan

This document captures the requirements and progress for the pp_mark_v0.3 implementation. All metrics assume 1080×1080 SDXL latent grids unless specified.

---

## 1. Objectives & Success Targets
- Watermark embed → extract → SP1 proof/verify chain completes within 1.5s on RTX 4090 (GPU path).
- Pure lookup/addition-friendly circuits (Poseidon binding, spread spectrum) without FFT-heavy ops.
- Optimistic verification: metadata fast path with fallback to blockchain/IPFS when absent.
- Statistical security via deterministic 0.5–2% sampling (default 1%).

## 2. System Overview
1. **Setup**: register EdDSA(BN254) keys + SP1 VK on chain/IPFS (future work).
2. **Payload**: prompt/seed/secret → Poseidon binding → RS(64,32) encoding.
3. **Embedding**: Poseidon CSPRNG → inverse CDF LUT → spread spectrum noise injected into SDXL latent grid.
4. **Proving**: sample ~11k coordinates, log uniforms/gaussian/combined values, run SP1 guest/host to produce a receipt.
5. **Verification**: compute soft score and verify SP1 receipt; recovery path replays extraction + score gate + receipt verification.

## 3. Configuration & Tables
- `config.py` provides dataclasses with JSON load/dump.
- `tables/invcdf_gaussian.bin` stores the inverse CDF lookup; generated via `scripts/generate_inverse_cdf.py` (2^16 points).

## 4. Modules (under `src/ppmark_v03`)
- `crypto.py`, `keys.py`, `payload.py`, `rs.py`: Poseidon binding, BN254 keygen, RS encoding, helpers.
- `sampling.py`, `tables.py`, `noise.py`, `embedding.py`, `cuda.py`: deterministic sampling, LUT lookups, CPU/CUDA kernels.
- `sp1_runner.py`: JSON packaging + SP1 host invocation (current).
- `halo2_interface.py`, `halo2_runner.py`: legacy JSON packaging for Halo2.
- `cli.py`: prover/verifier entry points; metadata includes blockchain/IPFS placeholders.

## 5. Scripts
- `scripts/halo2_stub.py`: legacy placeholder prover/verifier.
- `scripts/run_attack_suite.py`: enumerates Müller/Zhao/geometry/quality/ZKP tests (marked skipped until GPU/external tooling is available).

## 6. Roadmap
1. **CPU reference**: ✅ config, Poseidon, RS, sampling, noise, CLI, SP1 runner.
2. **GPU integration**: CuPy RawKernel fused spread+inverse CDF kernel plus sample-log kernel; parity + xorshift tests added (`tests/test_cuda_kernel.py`). Default CUDA path now uses GPU xorshift uniforms (Poseidon-on-CPU retained for parity); NVRTC + CUDA headers required.
3. **Legacy Halo2 Rust prover**: archived for reference only; see `docs/legacy/README.md`.
4. **Blind Sync + DDIM extraction**: coarse alignment + deterministic DDIM inversion placeholders and demodulation helpers are now in `src/ppmark_v03/sync.py`; swap in model-backed inversion later.
5. **Blockchain/IPFS hooks**: metadata placeholders ready; implement upload + registry update later.
6. **Legacy Halo2 sample constraints (WIP)**: archived alongside Halo2 notes under `docs/legacy/`.

## 7. Risks & Testing
- **CUDA**: NVRTC + CUDA headers are required for CuPy kernels; GPU testing should run on a CUDA 12.x box with matching runtime.
- **Attack harness**: `scripts/run_attack_suite.py` logs tests as `skipped` until GPU/external tools are wired.
- **Deterministic sampling leakage**: only expose hashed commitments; indices stay private until proof.
- **DDIM inversion failure**: to be addressed alongside sync module build-out.
- **Testing**: once GPU/SP1 pipeline is stable, rerun Müller/Zhao/geometry suites and SP1 benchmarks.

---

This plan will be updated as new modules (CUDA kernels, SP1 host/guest changes, sync/extraction) land.
