# PP-Mark v0.3 ??ZK-GenGuard V2 Reference

This repository contains the v0.3 implementation of PP-Mark (a.k.a. ZK-GenGuard V2). The code focuses on deterministic sampling, Poseidon/RS payload binding, the CPU-to-CUDA watermark embedding pipeline, and an SP1 zkVM prover path tuned for H100-class GPUs. Legacy v0.2 artifacts are kept only for backward-compatibility (`src/ppmark_v02/`).

## Features
- Deterministic 1% sampling (`src/ppmark_v03/sampling.py`) and spread-spectrum embedding (`src/ppmark_v03/embedding.py`).
- Payload binding + RS(64,32) coding (Pallas Poseidon for image-side payloads; SHA256 ctx-hash for SP1 binding).
- Canonical protocol-v2 full-trace Attest statement plus private SP1 prover witness
  (`src/ppmark_v03/attestation.py`, `src/ppmark_v03/sp1_runner.py`).
- Separate Detect and Attest assurance levels (`src/ppmark_v03/cli.py`):
  Detect reports the transform-tolerant statistical signal; Attest additionally
  requires the trusted producer key, exact artifact, P1--P4 receipt, and Detect.
- Attack/evaluation scaffolding (`scripts/run_attack_suite.py`, `scripts/eval_attack_manifest.py`).

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run the prover (CPU path):
```bash
python -m ppmark_v03.cli prover \
    --config config.json \
    --prompt "PP-Mark v0.3 demo" \
    --output out_demo \
    [--device-backend cuda --uniform-backend gpu_xorshift]
```

Run high-assurance Attest verification (the expected producer-key commitment is
an external trust anchor, not a value learned from the submitted artifact):
```bash
python -m ppmark_v03.cli verifier \
    --config config.json \
    --metadata out_demo/metadata.json \
    --image out_demo/watermarked.png \
    --tau-file datasets/thresholds/tau_fpr1_alpha4_sd21_fast.json \
    --mode attest \
    --expected-key-commitment 0x<TRUSTED_PRODUCER_KEY_COMMITMENT>
```

Timings: metadata includes `timings.sp1_prover_sec`; verifier prints receipt verification time.

Attack suite placeholder:
```bash
python scripts/run_attack_suite.py \
    --config config.json \
    --metadata out_demo/metadata.json \
    --output attack_logs
```

## SP1 Notes (Current)
- Default `zk.backend` is `sp1`; the prover/verifier calls `./target/sp1/release/sp1-genguard-host` to generate/verify receipts.
- The production host is pinned to SP1 v6.3.1 and accepts only the
  protocol-v2 streaming/full-trace commitment relation. Protocol v1 remains a
  diagnostic comparison path, not a production fallback.
- `--mode detect` returns only the statistical decision and explicitly does not
  attest generator compliance.
- `--mode attest` passes only when the trusted producer commitment, exact image
  hash, canonical public statement, SP1 receipt, P1--P4 predicates, and Detect
  threshold all pass.
- The receipt verifier requires the expected public statement and compares it
  byte-for-byte with the statement committed by the proof. The producer's
  private witness and full sample trace are not verifier inputs.
- Prover defaults to `SP1_PROVER=cuda` and `SP1_PROOF_MODE=core`; verifier defaults to `SP1_PROVER=cpu` for speed.

## Legacy Notes (RISC0/Halo2)
- RISC0/Halo2 paths remain in the codebase for reference only. See `docs/legacy/README.md` for historical notes.
- To run a legacy backend, pass `--allow-legacy-backend` to the CLI (not recommended for current evaluations).

## Repo Layout
```
README.md                    # current file
pyproject.toml               # Python package metadata
scripts/                     # CLI helpers, attack/eval harnesses
src/ppmark_v03/              # v0.3 implementation
src/ppmark_v02/              # legacy v0.2 LiteVAE stack (reference only)
sp1/                         # SP1 host/guest workspace
docs/architecture.md         # v0.3 architecture notes (SP1-first)
docs/legacy/README.md        # legacy RISC0/Halo2 notes
out_demo*/, out_stub/        # sample outputs (included intentionally)
attack_logs/                 # attack suite reports
```

## Next Steps
- Improve DDIM inversion quality for extraction and robustness.
- Refresh tau calibration for new datasets/models.
- Expand sync/search heuristics for rotation/crop resilience.

