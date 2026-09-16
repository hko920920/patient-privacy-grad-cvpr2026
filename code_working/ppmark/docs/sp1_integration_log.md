# SP1 Integration Log (v5.2.x) — Status as of Today

## What was added
- **SP1 workspace scaffold**: `sp1/` workspace with guest + host crates, target directory set to `target/sp1`.
- **Guest (sp1/guest)**: SHA-256 Merkle verification ported from the RISC0 logic.
  - Leaf hash: `index || gaussian_fixed || combined_fixed || bit` (LE encoding) hashed with SHA-256.
  - Verifies each sample’s Merkle path against `sample_merkle_root`; panics on mismatch.
  - Commits `sample_merkle_root`, `binding`, `prompt_hash`.
- **Host (sp1/host)**: CLI mirrors RISC0 (`prove|verify --public --witness --receipt`).
  - Parses public/witness JSON, serializes to guest input, runs `ProverClient::from_env()` with `compressed()` proof, verifies, and writes receipt (bincode).
  - Uses `sp1-build` in build.rs to include the guest ELF.
- **Python integration**:
  - Added `backend="sp1"` path with `sp1_runner.py` for prove/verify.
  - Added configs `config_sp1_small.json` (64×64) and `config_sp1_1080_sample1000.json` (1080p, ~1000 samples) pointing to `./sp1/target/release/sp1-genguard-host`.
- **Version/paches**:
  - SP1 stack: `sp1-sdk = 5.2.3`, `sp1-build = 5.2.3` (transitive sp1-* at 5.2.3).
  - Patches: sha2/sha3 (`patch-sha2/sha3-0.10.8-sp1-4.0.0`), p256/k256 (`patch-*-sp1-5.0.0`), curve25519-dalek (`patch-4.1.3-sp1-5.0.0`), RSA (`patch-0.9.6-sp1-5.0.0`).

## Current blockers
- **Build fails due to toolchain**:
  - `sp1-build` invokes `rustc +succinct` (1.82.0-dev); this rustc rejects injected LLVM flags (`-misched-prera-direction`).
  - Many deps (alloy/icu, base64ct, etc.) require rustc ≥ 1.85–1.88; `+succinct` is too old. Using `stable`/`nightly` hits the same minimum-version errors on these deps.
- **Patch warnings**: Patches declared in `sp1/guest/Cargo.toml` are ignored (non-root); RSA/curve patches unused in the current graph (warnings only).
- **Result**: No successful build of the SP1 host/guest yet; no prove/verify run for SP1.

## What succeeds
- Workspace compiles up to the point of the guest/host codegen; RISC0/Halo2 paths remain unaffected.
- Python `backend="sp1"` wiring is in place (metadata, runner), but cannot be exercised until the SP1 build completes.

## Next steps to unblock
1) **Update SP1 toolchain**: Install a rustc toolchain ≥ 1.88 that `sp1-build` will use (either update the `succinct` toolchain via sp1up or point `RUSTUP_TOOLCHAIN` to a newer toolchain recognized by `sp1-build`).
2) **Rebuild**: `rm -f sp1/Cargo.lock` and rerun `RUSTUP_TOOLCHAIN=<new> ./scripts/build_sp1_h100.sh cpu|cuda`.
3) **Clean patches**: Move patches to workspace root (already in `sp1/Cargo.toml`), remove guest-local patch block to silence warnings.
4) **Test**: After a successful build, run Python CLI with `backend="sp1"` (e.g., `config_sp1_small.json`) to produce/verify a real SP1 receipt.

## Notes
- Guest/host logic is implemented (Merkle/SHA validation in guest; ProverClient flow in host). The remaining issue is purely build toolchain compatibility.***

## Session log — 2025-12-08
- Removed the duplicate `[patch.crates-io]` block from `sp1/guest/Cargo.toml` so only the workspace-level patches apply, silencing the guest-local patch warnings.
- Confirmed the `succinct` toolchain is still `rustc 1.82.0-dev`; default `rustc` is `1.91.1`. SP1 binaries in `target/sp1/release/` were built with the old toolchain and need a rebuild after upgrading.
- Next steps unchanged: upgrade the SP1 toolchain to ≥1.88 via `sp1up`, `rm -f sp1/Cargo.lock`, rebuild with `RUSTUP_TOOLCHAIN=succinct ./scripts/build_sp1_h100.sh cpu|cuda`, then run the Python `backend="sp1"` path to generate/verify a receipt.
- Upgraded the `succinct` toolchain via `sp1up -v 5.2.3`; `rustup run succinct rustc --version` now shows `1.91.1-dev`.
- Fixed the guest packaging so `include_elf!` receives the ELF env var: moved the entrypoint into `sp1/guest/src/main.rs` (binary target) that calls `guest_main` from `lib.rs`.
- Rebuilt successfully with the new toolchain: `RUSTUP_TOOLCHAIN=succinct ./scripts/build_sp1_h100.sh cpu` → `target/sp1/release/sp1-genguard-host`.
- Proved and verified using the sample public/witness from `artifacts/risc0_sample_cpu/`; new receipt at `artifacts/outputs/out_sp1_sample/receipt.bin`.
- Default proof mode switched to `core` to reduce overhead on small circuits; set `SP1_PROOF_MODE=compressed|plonk|groth16` to override. Host prints the selected mode at runtime. CPU/CUDA builds remain at `target/sp1/release/sp1-genguard-host`.
- Core-mode sanity check with the 1-sample fixture: `prove` ≈ 7.3s, `verify` ≈ 3.5s (receipt at `artifacts/outputs/out_sp1_sample/receipt_core.bin`).
- Fix: Merkle path direction now uses the sample position (not pixel index), aligning guest verification with host-generated paths.
- Mid-size GPU core run (≈200 samples, config `config_sp1_1080_sample200.json`): `prove` wall-clock 117.5s (host-reported prover elapsed 39.3s), `verify` ~26.8s. Receipt: `artifacts/outputs/out_sp1_1080_s200_gpu_core/sp1/receipt.bin`.
- Optional flags: `SP1_SKIP_EXECUTE=true` to skip simulate/execute; `SP1_SKIP_VERIFY=true` to skip in-host verify. PK/VK cached at `target/sp1/keys/<elf-digest>.{pk,vk}`.
- Sample 1000 (1080p, GPU core):
  - With execute+verify: prover 77.5s (wall 155.9s), verify ~29.8s (`artifacts/outputs/out_sp1_1080_s1000_gpu_core/`).
  - With execute/verify skipped: prover 71.7s (wall 150.1s); verify separate ~30.2s (`artifacts/outputs/out_sp1_1080_s1000_gpu_core_skip/`).
- Sample 200 (1080p, GPU):
  - Core: prover 39.3s (wall 117.8s), verify ~26.3s (`artifacts/outputs/out_sp1_1080_s200_gpu_core/`).
  - Compressed: prover 45.3s (wall 123.5s), verify ~25.7s (`artifacts/outputs/out_sp1_1080_s200_gpu_compressed/`).
- Sample 1000 (1080p, SP1_PROVER=cpu, core, skip execute): prover 223.7s (wall 302.0s); verify on CPU ~6.4s (wall 7.3s) → `artifacts/outputs/out_sp1_1080_s1000_cpu_core/`.
- Cross-verify receipts: GPU receipt verified on CPU: ~6.8s (wall 7.6s); CPU receipt verified on GPU: ~29.0s (wall 29.8s). Conclusion: run verify with `SP1_PROVER=cpu` for speed; use GPU only for proving.
- Small fixture timing (for reference):
  - CPU core: prove ~7.3s / verify ~3.5s.
  - CUDA core: prove ~28.9s / verify ~27.8s.
  - CUDA compressed: prove ~96s / verify ~26s.
