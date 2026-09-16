# Audit v10 TensorFlow Privacy Setup Rebinding Protocol

Date: 2026-07-25

Purpose: rebind the already frozen TensorFlow Privacy 0.9.0 statement
comparison to the current cross-platform environment bootstrap without changing
the requests, official outputs, predictions, aggregate metrics, or scientific
interpretation.

The v2 artifact index named an older monolithic PowerShell setup file. The
current setup surface consists of a short PowerShell wrapper plus
`scripts/setup_tfprivacy_baseline.py`. The v10 index therefore:

1. binds the current wrapper's exact byte length and SHA-256;
2. adds the Python bootstrap's exact byte length and SHA-256;
3. retains the pinned requirements, official module hash, verifier, worker,
   requests, outputs, predictions, metrics, and summary unchanged; and
4. reruns the official-wheel worker in the isolated environment.

Success criteria:

- all indexed files match byte length and SHA-256;
- the official TensorFlow Privacy worker reproduces the frozen statement JSON;
- all 23 comparison rows and aggregate metrics recompute exactly; and
- the versioned verifier reports 79/79 passing checks.

This repair concerns artifact completeness only. It does not change the
comparison's authority: the official function produces statements conditional
on caller-supplied scalars, whereas the paper's validator checks whether
completed-run evidence licenses a requested sentence.
