# PRRD bank runtime

This entry point implements the frozen non-DP recipe. It does not authorize it.
The original objective, encoder, renderer, patient weighting, coefficients and
30-bank list are unchanged.

## Boundaries

- Public technical checks use four P images, C, 84 successful updates. A separate
  process stops at82, exits, and a fresh process resumes at83. No main bank is
  created and every artifact is marked TECHNICAL_TEST_ONLY.
- Main banks require an explicit authorization JSON bound to the original
  contract, current implementation verification, allowed bank IDs, prepared
  target receipts, output root and numeric cumulative time cap.
- The synthesis loader opens only P template pixels. Q preparation is a separate
  authorized task; this runner consumes its bound NPZ target, not Q/V pixels.
- No recipient, DP, Expert or Reserved entry point is provided.

## Main command interface (not authorized or run)

From code_working, using the existing environment:

    python -B -m prrd_v3.run main-bank \
      --contract <PRRD_FIRST_NONDP_EXECUTION_CONTRACT_20260918.md> \
      --authorization <explicit-main-authorization.json> \
      --verification <public-runtime-verification.json> \
      --bank <allowed-bank-id> --output-root <authorized-root>

Use --resume for a paused bank. An already completed bank is rejected rather
than overwritten. Relative paths must be resolved consistently across resumes.

The authorization schema is prrd.main-authorization/v1. Required fields:

    user_authorization: true only after actual permission
    contract_sha256
    allowed_run_ids
    maximum_seconds
    output_root
    verification_sha256
    implementation_sha256
    target_receipts: {bank_id: {path, sha256}}
    allow_private_prepared_targets
    Q_V_pixel_access: false
    recipient_execution: false
    DP: false
    expert_reserved: false

The implementation digest is the SHA256 of canonical JSON of source_hashes()
(sort_keys=True, separators=(',', ':'), ensure_ascii=True).

Each target receipt has schema prrd.prepared-nondp-target/v1, base contract hash,
target_generation_rule_sha256, arm, population P or P+Q, npz_path, npz_sha256,
prefix, and input_sha256 bindings. Q-derived targets additionally need
Q_preparation_authorization_sha256 and Q_feature_sha256; D requires its fixed
permutation_seed. Actual target preparation/authority remains pending.

## State and finalization

The bank signature binds the job, objective policy, target, templates, code,
encoder/projections and frozen recipe. Checkpoints contain all renderer
parameters/buffers, optimizer state, Python/NumPy/PyTorch CPU+CUDA RNG,
completed/next step, active pyramid levels and requires_grad flags. Restore
must match the saved state exactly before the next update.

Checkpoints are immutable files. Their hash receipt is published before the
latest pointer; a partial/orphaned write cannot replace the last committed state.
The bank has an OS process lock, released on process exit. Successful steps are
logged; the checkpoint interval remains25 with additional saves on controlled
stop and finalization.

Only the declared final step exports an artifact. PNG, labels, bank membership,
virtual pairs and all file hashes are verified before the artifact directory is
published. COMPLETED is published last. Recovery after artifact publication but
before COMPLETED validates and retains the existing bytes. It does not overwrite
them. A partial export directory is never a completed bank.

Budget accounting reserves remaining authorized time before starting a process
and reconciles elapsed time on normal exit. An unclean exit leaves the reservation
consumed/fail-closed; the saved bank can resume after budget reconciliation or
a separately bound allowance. Numeric time enforcement occurs between complete
updates, with a margin for saving. It is not a hard real-time OS kill guarantee.

## Evidence scope

The bounded public C test verifies the production execution/checkpoint/export
path. Constructed CPU checks additionally cover A/C PNG layout, corrupt/changed
bindings, incomplete output and overwrite rejection. This is not a500-step,
128-image main-bank result, a private-data test, or an efficacy evaluation.

