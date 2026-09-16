# Executable DP training semantics

Current-state authority: `../../CURRENT_STATUS.md`. In particular, the K5 four-step research
dry-run predates B0; B0 ordering is asserted only before the full 4,000-step matrix.

This directory turns the frozen NIH CXR14 accounting contract into a small executable mechanism.
It covers per-unit clipping, mean-before-clip patient aggregation, unconditional Gaussian noise,
fixed-denominator normalization, Poisson sampling, and a schedule-only public event trace.

It does **not** make private training release-ready. Deterministic and ordinary PyTorch generators are
test/research backends only; a separately audited CSPRNG backend remains a release gate.

The older `unitdp_compiler_reference` owner trainer is not reused as the execution path because it
skips empty Bernoulli batches. The frozen sampled Gaussian mechanism requires a noise-only optimizer
update for those events.

Run from `code_working` with the pinned environment:

```powershell
$python = ".\base_gate\.venv\Scripts\python.exe"
& $python -m unittest -v dp_training.test_mechanism
& $python -m dp_training.run_synthetic_conformance
& $python -m dp_training.run_xray_public_lora_dp_step_smoke
& $python -m dp_training.verify_xray_dp_trainer_independent
& $python -m unittest -v dp_training.test_private_research_dryrun
& $python -m dp_training.run_xray_k5_private_research_dryrun
& $python -m dp_training.verify_xray_k5_private_research_dryrun_independent
```

The public LoRA smoke is disposable branch coverage, not evaluation: it deliberately uses frozen
public-calibration sentinels whose gradients cross C, writes no checkpoint, and cannot be cited as a
utility or privacy result. See `../XRAY_DP_TRAINER_EXECUTABLE_GATE.md`.

The K5 private-partition command is a one-time, four-step-per-arm runtime gate bound to
`private_research_dryrun_protocol.json`. It uses ephemeral private randomness and refuses to
overwrite a nonempty default report directory. Do not rerun it as an evaluation or expose
`restricted_runtime_diagnostics.json`; its result and boundaries are recorded in
`../XRAY_K5_PRIVATE_RESEARCH_DRYRUN_GATE.md`.

The subsequent full K5 protocol is materialized by
`build_k5_feasibility_protocol.py` and checked by
`verify_k5_feasibility_protocol_independent.py`. Both are one-time, refuse-overwrite gates; do not
rerun them over their existing report directories. They perform no model update. The frozen output
and remaining evaluator/restart prerequisites are documented in
`../XRAY_K5_FEASIBILITY_EXECUTION_PLAN.md`. Passing this protocol gate does not authorize the
4,000-step command. The later full-runner gate has now implemented and independently frozen that
command, but no full optimizer step has started.

The evaluator/restart prerequisite is now implemented by `k5_evaluator.py`,
`run_k5_evaluator_preflight.py`, and `secure_resume.py`. Their result protocols are one-time and
already materialized; do not overwrite or reinterpret them as trained-model quality. Run the fast
regression suite with:

```powershell
$python = ".\base_gate\.venv\Scripts\python.exe"
& $python -m unittest -v dp_training.test_mechanism `
  dp_training.test_private_research_dryrun `
  dp_training.test_k5_evaluator dp_training.test_secure_resume
```

The evaluator's overall weak-label validity passes, but consolidation, no-finding, effusion, and
pneumonia condition summaries are negative and remain descriptive only. Exact encrypted restart
passes for all four synthetic arm semantics; it is not yet an actual full SD 2.1 checkpoint. See
`../XRAY_K5_EVALUATOR_RESUME_PREFLIGHT_GATE.md`.

The fixed B0 baseline has now been generated before full-matrix initialization. The earlier
four-step-per-DP-arm `RESEARCH_ONLY` dry-run used a separate ephemeral state and retained no
checkpoint; therefore B0 is not described as preceding every private-role optimizer update. B0 uses the reusable
`k5_generation.py` primitives, the fail-closed resumable `run_k5_b0_generation.py` runner, and an
implementation-independent final verifier. Protocol v1_002 is the authority; v1_001 stopped during
pipeline loading with zero images because it omitted the already documented base-gate mask for the
host's broken, unused optional ONNX Runtime. The failed evidence is retained. V1_002 generated all
448 files and passed full-file and eight-image sentinel replay verification. Do not rerun or
overwrite the completed B0 tree. See `../XRAY_K5_B0_GENERATION_GATE.md`.

The current stop is before matrix initialization and M0. The next mutating action, if explicitly
continued after reviewing B0, is initialization followed by **M0 only**. The 35-image B0 visual
sanity grid is grossly off-domain, as expected for unadapted general-image SD 2.1; M0 domain
adaptation is therefore a hard gate before interpreting or proceeding to DP arms.
