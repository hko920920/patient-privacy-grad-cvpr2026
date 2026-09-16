# Halo2 Alignment Worklog (Dec 1, 2024)

This document captures what was changed while aligning the Python and Rust Halo2 stacks, what still fails, and next debugging steps.

## Update (latest)
- **Hex parsing fixed:** `new_prover`/`new_verifier` now parse bare hex strings (no `0x` prefix) so `sample_merkle_root`/paths/seeds are no longer misread as zero. Long numeric strings are treated as hex and negatives are handled.
- **I/O serialization tightened:** Python `halo2_runner` writes public inputs and Merkle paths with `0x` prefixes for consistency.
- **Binding constraint restored:** Rewired Poseidon inputs to use dedicated advice columns (`AssignedCell` flow end-to-end) and re-enabled the in-circuit binding = Poseidon(prompt_hash, seed, secret) check. Binding is also prechecked host-side for sanity.
- **Regression test added:** `binding_matches_poseidon_hash` `MockProver` unit test verifies Poseidon hash parity against artifacts/outputs/out_halo2_small_check5 vectors.
- **Outcome:** `new_prover`/`new_verifier` now succeed on the saved `out_halo2_small_check5` witness/public pair; the previous permutation errors are resolved.

## Changes made
- **Python Poseidon parity**
  - Reimplemented Pallas Poseidon with the exact `P128Pow5T3` parameters, the correct Pallas modulus (`0x4000…094cf91b992d30ed00000001`), and domain separation matching `ConstantLength<L>` (length encoded in capacity, rate=2).
  - Added `src/ppmark_v03/pallas_poseidon_constants.py` (round constants, MDS) generated from `halo2_poseidon` Rust crate.
  - Added `poseidon_hash_pair` helper to mirror the circuit’s 2-ary hash chaining.
  - Added unit test `tests/test_pallas_poseidon.py` with the reference permutation vector.
- **Merkle hashing parity**
  - Updated `build_sample_merkle` to hash leaves and path folds as consecutive 2-input Poseidon hashes (same as the circuit).
  - Paths are now padded to a uniform depth; Python recomputation of roots matches the witness/public files.
- **Halo2 circuit tightening**
  - In `halo2_prover/src/new/watermark.rs`, re-enabled constraints:
    - Binding = Poseidon(prompt_hash, seed, secret) checked in-circuit.
    - Merkle root constrained to the public instance.
  - Kept boolean/combined constraints as before.
- **Verifier behavior**
  - Removed Python verifier stub fallback; any Halo2 verifier failure now surfaces as an error.
- **Debug tooling**
  - Added `halo2_prover/src/bin/poseidon_debug.rs` (uses `halo2_gadgets::poseidon::primitives::Hash<ConstantLength<3>>`) to emit circuit Poseidon outputs for given inputs (hex). Run with three hex args to get the BE hex of the result; note the binary prints little-endian bytes, so reverse to compare with Python.
  - Built `new_prover`/`new_verifier` after the above changes (`cargo build --release` succeeds).

## Current status
- `config_small.json` (k=17) end-to-end passes with the real Halo2 prover/verifier after binding rewiring (Poseidon constraint active).
- Full `config.json` (k=20, 1080×1080) prover attempts on this box timed out; raising to `k=21` also timed out (no constraint errors observed; likely compute-bound). Expect to rerun on a larger host or with a higher timeout budget to confirm full-config success.
- k=17 “full” runs: sample_rate 0.01% (~116 samples) and 0.02% (~233 samples) succeed (proof ≈35–36s, verify ≈3–4s). Increasing to 0.04%/0.05% at k=17 hits `NotEnoughRowsAvailable`; k must be raised for higher sample rates.
- GPU prover experiments (H100 target):
  - Tried PSE/Scroll/Axiom/zkonduit forks: no `cuda` feature or branch/package mismatch, all failed to build.
  - Ingonyama ICICLE fork (ingonyama-zk/halo2) vendored locally (`/tmp/ingy_halo2`); CUDA + cmake + nightly-2024-07-18 builds ICICLE deps. CPU→BN254 타입 전환 중이나, Poseidon 가젯이 파스타용 스펙(P128Pow5T3 over Fp/Fq)만 노출되어 있어 BN254에서는 `Spec<Fr>` 부재로 컴파일 실패. 해결책: BN254 Poseidon Spec을 직접 추가(파라미터는 P128Pow5T3를 Fr에 적용)하거나, Spec을 포함한 포크 확보가 필요. 공개 포크 중 “GPU+Pasta+Poseidon Spec for bn256” 조합은 발견 못 함.

## Artifacts
- Passing: `artifacts/outputs/out_halo2_small_check5` (legacy), `/tmp/out_cli_small` (fresh `config_small.json` run), both verified.
- To re-run full: use `config.json` and point outputs under `artifacts/outputs/` or `/tmp` on a beefier machine.

## ICICLE CUDA backend (H100) status — Jan 2025
- Env: H100 PCIe, CUDA 12.2. ICICLE backend .so files under `/tmp/ingy_halo2/icicle/backend/cuda`.
- License: default server reachable (`5053@license.icicle.ingonyama.com`) when `ICICLE_LICENSE` is unset; local `icicle.lic` absent.
- Backend load: with `ICICLE_BACKEND=CUDA`, `ICICLE_BACKEND_INSTALL_DIR=/tmp/ingy_halo2/icicle/backend`, backend loads but CUDA kernels fail with `no kernel image is available for execution on the device` during NTT → likely built for older arch (e.g., sm80), not sm90.
- Consequence: GPU path falls back/crashes; `kzg_integration` test only runs CPU. GPU Util stays ~0%.
- Next steps (updated):
  1) Build sm_90 cubins via `scripts/build_icicle_h100.sh` (uses `CMAKE_CUDA_ARCHITECTURES=90-real`); outputs live under `icicle/build/h100-sm90/install`.
  2) Export `ICICLE_BACKEND=CUDA`, `ICICLE_BACKEND_INSTALL_DIR=<install>/lib/backend`, and prepend `<install>/lib` to `LD_LIBRARY_PATH` before running the Halo2 prover.
  3) If the CUDA backend sources are absent, drop the Ingonyama CUDA backend under `icicle/icicle/backend/cuda` or place sm_90 binaries under the install dir; otherwise the script will build CPU-only.
  4) See `docs/legacy/icicle_h100_gpu_strategy.md` for the Hopper rationale and troubleshooting. Until sm_90 binaries are in place, stick to `ICICLE_BACKEND=CPU` to avoid panics.

## Hypotheses
- The Python hash now matches the Rust primitive, but the circuit layout may still differ:
  - Gadget expects state ordering and absorption identical to `Hash<ConstantLength<L>, width=3, rate=2>`. If the circuit manually wires the state differently (e.g., row/chunk packing or init constants), copy constraints can fail despite matching arithmetic.
  - Witness generation may be placing Poseidon inputs across regions in a way that violates the chip’s permutation/copy constraints (e.g., reusing advice cells incorrectly when binding and Merkle leaves are hashed).

## Next debugging steps
1. **Minimal MockProver reproduction in Rust**: Add a Rust unit/integration test that feeds a tiny witness (e.g., 2–4 samples) directly into `WMCircuit` with known values from Python (binding, prompt_hash, alpha, leaves/paths) and run `MockProver::run().verify()`. This will localize the failing gate/region without the Python CLI.
2. **Reduce sample_count further**: For a dedicated debug config (e.g., 8×8 image, sample_rate very small), regenerate witness/public from Python and rerun `new_prover` to shrink the permutation trace and make the failing region easier to inspect.
3. **Instrument Poseidon chip usage**: In `watermark.rs`, consider:
   - Hash binding via a dedicated Poseidon chip invocation with explicit state loading (rather than chaining assignments + `hash_pair` abstraction), matching the gadget’s expected state usage.
   - For Merkle folds, reuse a single chip instance per level or ensure assigned cells are copy-constrained exactly as the chip expects; avoid multiple regions unless required.
4. **Cross-check absorption order**: Verify that `poseidon_hash_pair` + chaining in Python matches `Hash<ConstantLength<2>>` (which sets capacity = `2<<64`). If the circuit uses a different domain or initializes the sponge differently, replicate that exact flow in Python (or vice versa).
5. **Add path-depth sanity in witness**: Ensure `sample_merkle_depth` matches the path length (it currently does); if the circuit assumes fixed depth, force padding to that depth in Python.

## Commands run (for reference)
- Build: `cargo build --release` (in `halo2_prover/`).
- Prover attempts (fail): `python -m ppmark_v03.cli prover --config config_small.json --prompt "poseidon parity check" --output out_halo2_small_check* --seed-hex ...`
- Debug Poseidon (Rust): `cargo run --release --bin poseidon_debug <a_hex> <b_hex> <c_hex>`; reverse output bytes to compare with Python hex.
- Python recompute checks: recomputed binding/root matched public/witness; Merkle path verification succeeded in Python.
