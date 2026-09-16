# Audit v5 UCI HAR confirmatory execution protocol

Frozen: 2026-07-24 22:23 KST

This protocol was fixed after the permitted development pilot and before any
confirmatory execution.

## Route and source snapshot

```text
pipeline
scripts/uci_har_v5_pipeline.py
0c226ad3acb1d0fcc7a08904288611877269a173dfe30e6d3bcee75fa2b9d0bf

production-import-free verifier
scripts/verify_uci_har_v5.py
14193a2f8faffe7d1998ecbb62307556ceb4cc67eb3793161c4e52f567e08b26

registry
docs/mechanism_registry_v2_0.json
9344d1549f3db3265b45e4c0f10a19155ea20e77a14fd914c3d20a12a09e2834

verifier tamper tests
tests/test_uci_har_v5_verifier.py
4a3a0f30588c45852dd2069acce7dccb4a8beb8a9651c69eed17f25cd79be290

UCI HAR eight-file bundle digest
0d10319685c36554d12b88c91d5608805a6bfd95969de59de48c41c4d6ad7487
```

Any source, registry, verifier, or input hash change invalidates this protocol
and requires a new protocol/version before further confirmatory execution.

## Fixed mechanism parameters

```text
sample-rate numerator/denominator: 1/50
steps: 100
noise multiplier: 2.0
window clipping norm: 1.0
public optimizer step: 1/200
delta: 1e-6
sampler: exact-rational SystemRandom randrange Bernoulli
noise: discarded draw plus four normal arrays divided by two
update: clipped sum plus Gaussian noise, public constant step, no dataset-size denominator
```

## Authorization boundary

For every run:

- generated-window/add-remove must be `ALLOWED / DIRECT / K=1`;
- raw-event wording must be `BLOCKED_UNVERIFIED / UNDERSPECIFIED`;
- owner wording must be `BLOCKED_UNVERIFIED / UNDERSPECIFIED`;
- blocked certificates must contain no positive wording.

## Pilot disclosure

`uci_har_v5_pilot_001` is an incomplete implementation smoke: training
executed, but an invalid influence-support vocabulary caused the certificate
schema to block.  It is not a route result and is excluded from all aggregates.

`uci_har_v5_pilot_002` is the single disclosed complete development pilot.  It
passes 49/49 full and 38/38 portable verification, but is excluded from the
confirmatory aggregate because it predates this protocol.

Pilot utility must not be used to select, replace, or terminate confirmatory
runs.

## Confirmatory design

Run exactly, in this order:

```text
confirmatory_run01
confirmatory_run02
confirmatory_run03
confirmatory_run04
confirmatory_run05
```

Each run uses fresh private unrecorded secure randomness.  There is no
user-controlled seed.  Every run is retained regardless of utility.

For every run:

1. execute the frozen pipeline;
2. execute the full independent verifier with all eight UCI files reopened;
3. execute the portable verifier;
4. retain all metrics, reports, models, traces, certificates, and verifier
   records; and
5. bind output hashes in the execution manifest.

## Adoption gate

The route may enter the manuscript only if:

- all five full verifiers pass every check;
- all five portable verifiers pass every check;
- all five authorization boundaries are identical;
- all five released-model SHA-256 values are distinct;
- every metric is finite and every run is included in the mean, sample
  standard deviation, and range; and
- the aggregate is reproduced byte-for-byte by a separate summarizer.

The result reduces dataset concentration only.  It does not establish
cross-mechanism universality, raw-event certification for UCI HAR,
owner-level privacy, state-of-the-art utility, or malicious-producer
attestation.
