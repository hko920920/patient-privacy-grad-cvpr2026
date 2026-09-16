# GPU + Halo2 Status (Dec 2025, H100 target)

This log captures the current GPU/ZKP state for PP-Mark v0.3 (H100 box).

## Environment
- Hardware: 2× H100 PCIe, driver CUDA 12.2.
- Toolchains: nvcc 12.9.86 (`/home/mimic/miniconda3/envs/cuda-dev/bin/nvcc`), Rust 1.91.1.
- Python entry: `src/ppmark_v03/` (BN254 Poseidon, RS, sampling, Halo2 runner). CPU path works on small k (k=17) in ~30–40s.
- Halo2 crate: `halo2_prover/` uses PSE halo2 0.4 + ec-gpu-gen 0.7.1 (`--features ecgpu`). Local patches to `/tmp/pse_halo2`, `/tmp/poseidon-gadget`. ICICLE/halo2-arithmetic/libfam attempts abandoned (see below).

## Poseidon alignment (resolved)
- BN254 Poseidon binding is now computed via the Rust helper `halo2_prover/target/release/poseidon_debug` (Pow5, ConstantLength<3>, rate=2). Python `bind_payload` calls this path (`src/ppmark_v03/crypto.py`).
- Verified: given prompt_hash/seed/secret from public/witness, `poseidon_debug` recomputes the same binding as in `public_inputs.json`.
- Removed redundant binding pre-check in `new_prover` (circuit still enforces binding = Poseidon(prompt_hash, seed, secret)).

## Remaining blockers
1) **ec-gpu MSM parity still under watch**
   - Test: `cargo test --release --features ecgpu ecgpu_fft_and_msm_smoke -- --nocapture` (in `halo2_prover/`).
   - Result: GPU FFT now matches the CPU DFT (8-point smoke test). MSM is validated against the CPU multiexp and falls back transparently when a GPU mismatch/error is detected. Strict parity can be forced via `ECGPU_ENFORCE_GPU_MSM=1` (which sets `ECGPU_TRUST_GPU_MSM=1` temporarily) to surface GPU-only errors.
   - Impact: Default path is safe (returns CPU result on mismatch); strict mode still needs to be re-run on a working GPU to confirm true parity.

2) **Halo2 witness assignment fails**
   - Running `python3 -m ppmark_v03.cli prover --config config_small.json --legacy-poseidon ...` (PYTHONPATH=src) produces public/witness files with consistent lengths (205 samples, depth=8, alpha_fixed present, bindings match `poseidon_debug`).
   - `new_prover` still panics: `AssignError(WitnessMissing { func: "assign_advice", desc: "load input_1" })` (halo2_frontend/src/circuit.rs:350). Binding mismatch is gone; the layouter now refuses because an instance/advice cell is missing.
   - Impact: proof generation halts before circuit synthesis; no end-to-end run (CPU or GPU) is possible with the current circuit configuration.

## Abandoned GPU backends
- **halo2-arithmetic/libfam**: libfam submodule is private → CUDA sources not available.
- **Ingonyama ICICLE**: sm_90 build/licensing issues and missing BN254 Poseidon Spec; runtime “no kernel image” on sm_90.
- Therefore ec-gpu-gen (Filecoin-style) is the active path despite current correctness issues.

## What’s needed next
- ec-gpu: re-run `ecgpu_fft_and_msm_smoke` on a GPU box with `ECGPU_ENFORCE_GPU_MSM=1` to verify MSM parity now that FFT is fixed. If mismatches remain, capture GPU vs CPU MSM outputs for debugging.
- Halo2 circuit: add debug logging to `new_prover` to dump parsed prompt_hash/seed/secret/alpha/instances and see which instance/advice row is considered missing (“input_1”); adjust instance layout/phase to satisfy halo2_frontend.

Until these two issues are fixed, GPU proof generation/verification timing on H100 is not meaningful (proof either fails or uses incorrect arithmetic).
