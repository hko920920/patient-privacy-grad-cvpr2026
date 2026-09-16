# PP-Mark SP1 Attestation (SP1 v6.3.1, protocol v2)

This workspace contains the production SP1 host/guest path for PP-Mark
Attest. It is not a receipt stub. The guest recomputes the canonical public
protocol-v2 statement and enforces P1--P4; the host cryptographically verifies the receipt
and byte-compares the committed public values with the verifier-supplied
statement.

## What the guest enforces

- **P1 -- context/key binding:** recompute the binding from `ctx_hash` and the
  private producer key, and expose a producer-key commitment for an external
  trust-anchor check.
- **P2 -- committed samples:** recompute one domain-separated SHA-256
  commitment to the complete ordered trace and the sampled-index-set hash.
- **P3 -- exact artifact:** derive the challenge from the canonical image hash,
  context hash, and full-trace commitment, then check the partial-opening digest.
- **P4 -- embedding relation:** re-encode RS(64,32), derive MSB-first payload
  bits, reproduce the pinned inverse-CDF Gaussian values, and check the
  normalized fixed-point embedding relation at every committed sample.

The private witness contains the producer key, RS codeword, and sampled trace.
It is needed only by the prover. Verifiers receive the canonical public
statement and receipt; they do not receive the private witness or full trace.
The submitted protocol-v1 Merkle relation remains in the source only for
historical cycle comparisons; the production host accepts protocol v2 only.

## Prerequisites

- Ubuntu 22.04/24.04
- SP1 toolchain v6.3.1 (`sp1up --version 6.3.1`)
- `protoc` with the standard Google protobuf include files (Ubuntu package
  `protobuf-compiler`)
- A working C/C++ compiler; the build helper prefers GCC 12 and falls back to
  the system `gcc`/`g++`
- CUDA/SP1 GPU dependencies only when building or proving in CUDA mode

## Build

From the repository root:

```bash
./scripts/build_sp1_h100.sh cpu
# or
./scripts/build_sp1_h100.sh cuda
```

The host binary is written to
`target/sp1/release/sp1-genguard-host`.
The workspace pins fat LTO and one codegen unit for release builds. The
validated protocol-v2 streaming ELF is 381,384 bytes with SHA-256
`1a9eb1a3b100ac7de792cb44865946d44301330574922a187c7ce1b237bb1459`;
changing the guest, dependency lock, toolchain, or release profile changes the
verifying key and requires a new measurement/validation campaign.

## Execute, prove, and verify

```bash
# Constraint-only execution, useful for fixture and tamper tests.
target/sp1/release/sp1-genguard-host execute \
  --public attestation_statement.json \
  --witness private_witness.json

# Generate a receipt. Use SP1_PROVER=cuda on a configured GPU host.
SP1_PROVER=cpu SP1_PROOF_MODE=core \
target/sp1/release/sp1-genguard-host prove \
  --public attestation_statement.json \
  --witness private_witness.json \
  --receipt attest_receipt.bin

# Verification requires only the expected public statement and receipt.
SP1_PROVER=cpu target/sp1/release/sp1-genguard-host verify \
  --public attestation_statement.json \
  --receipt attest_receipt.bin
```

`SP1_PROOF_MODE` accepts `core` (default), `compressed`, `plonk`, or
`groth16`. The Python prover
deletes its serialized private witness after a successful proof unless the
explicit private-debug retention flag is used.

SP1 v6 proving keys are backend-specific and remain in the process that ran
setup. `SP1_PK_PATH` from the v5 host is therefore rejected. A verifier may
still load a bincode-serialized v6 verifying key through `SP1_VK_PATH`; when
proving, the host checks that a configured verifying key matches the active
protocol-v2 guest. Reuse a persistent prover process (or the timing harness)
when setup amortization matters.

## Model-free regression tests

```bash
PYTHONPATH=src python scripts/build_sp1_attest_fixture.py \
  --output /tmp/ppmark-fixture --commitment-scheme streaming
PPMARK_SP1_HOST="$PWD/target/sp1/release/sp1-genguard-host" \
  bash scripts/test_sp1_attest_tamper.sh
```

The tamper suite checks independent failures for P1 binding/key, P2
trace-commitment/index,
P3 image/opening, and P4 codeword/bit/Gaussian/LUT changes.
