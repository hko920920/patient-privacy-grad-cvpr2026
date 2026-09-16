# PP-Mark v0.3 Architecture Notes (SP1-first)

This document summarizes the current ppmark_v03 codebase (files under `src/ppmark_v03/`) with the SP1 pipeline as the default. RISC0/Halo2 paths are legacy and documented under `docs/legacy/`.

## Module Breakdown

| Path | Purpose |
| ---- | ------- |
| `config.py` | Defines strongly-typed dataclasses (`ImageConfig`, `ZKConfig`, etc.) and JSON load/dump helpers. |
| `crypto.py` | Hash helpers, prompt hashing, and deterministic sampling seeds. |
| `keys.py` | BN254 EdDSA keypair generation via `py-ecc`. |
| `payload.py` | Builds the payload binding, RS(64,32) codeword, and bitstream. |
| `rs.py` | Thin wrapper over `reedsolo` with validation helpers. |
| `sampling.py` | Deterministic sample coordinate selection + binary trace logger. |
| `tables.py` | Inverse CDF lookup loader; consumes `tables/invcdf_gaussian.bin`. |
| `noise.py` | CPU reference for spread-spectrum noise synthesis (hash uniforms + LUT lookup). |
| `embedding.py` | End-to-end watermark injection over the latent grid + sample trace extraction. |
| `cuda.py` | Kernel interface abstraction; CPU reference + CuPy placeholder. |
| `attestation.py` | Canonical Attest public statement, domain-separated commitments, image challenge/opening derivation, and fixed-point P4 helpers. |
| `halo2_interface.py` | Public/witness structs shared between Python and ZK backends. |
| `sp1_runner.py` | Separately serializes the canonical public statement and private prover witness, invokes the SP1 host, and removes the witness after a successful proof by default. |
| `cli.py` | CLI entrypoint (`python -m ppmark_v03 ...`) for prover/verifier flows. |

## Prover Flow (`ppmark_v03.cli`)

1. Load `config.json` via `GlobalConfig` and parse CLI arguments.
2. Generate or resolve EdDSA keys and seed.
3. Build SP1 context hash `ctx_hash` and binding (SHA256) plus RS codeword payload.
4. Deterministically select sample coordinates using the RS codeword as key material.
5. Load the inverse CDF table and obtain a `WatermarkKernel` (CPU by default, CUDA in GPU setups).
6. Apply the normalized watermark relation and collect the sampled fixed-point trace.
7. Build the canonical public statement (context/binding, producer-key
   commitment, exact image hash, Merkle/index commitments, image-derived
   challenge/opening digest, pinned LUT/normalization parameters, and shape).
8. Serialize that statement separately from the private witness (producer key,
   RS codeword, and samples), then run the SP1 host to produce a receipt.
9. Delete the serialized private witness after successful proving unless an
   explicit private-debug retention flag is set, and emit public metadata.

## Verifier Flow (Detect or Attest)

1. Load metadata + `config.json`.
2. Reconstruct and validate the canonical statement from the public artifact;
   reject inconsistent context, sampling, LUT, normalization, or shape fields.
3. Invert/align the image to the latent grid (or load an explicitly supplied
   latent in a test harness), compute the soft correlation score, and compare
   it with the frozen `tau`.
4. In **Detect** mode, return that statistical decision and label it as a
   signal-only result; it does not certify generator compliance.
5. In **Attest** mode, additionally require an externally trusted producer-key
   commitment, the exact canonical image hash, and cryptographic receipt
   verification against the expected public statement.

**Attest rule:** pass iff trusted producer key AND exact image AND expected
public statement/receipt AND P1--P4 AND `Detect >= tau` all pass.

## SP1 Notes
- P1 recomputes the context/key binding and producer-key commitment.
- P2 recomputes the ordered sample Merkle root and sampled-index-set hash.
- P3 derives the image-bound challenge and checks the opening digest.
- P4 re-encodes RS(64,32), maps payload bits, reproduces the pinned inverse-CDF
  Gaussian values, and enforces the normalized fixed-point embedding relation.
- The host verifies the receipt and then byte-compares its public values with
  the verifier-supplied canonical statement; a valid proof cannot be replayed
  under altered public JSON.
- Prover typically runs with `SP1_PROVER=cuda`; verifier defaults to `SP1_PROVER=cpu` for speed.

## Legacy (RISC0/Halo2)
See `docs/legacy/README.md` for historical RISC0/Halo2 notes and GPU bring-up logs.
