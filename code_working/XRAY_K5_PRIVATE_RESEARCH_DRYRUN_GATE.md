# NIH CXR14 K5 private-partition research runtime dry-run

Date: 2026-09-03  
Decision: `PASS_K5_PRIVATE_RESEARCH_RUNTIME_DRYRUN`  
Independent verification: `PASS_K5_PRIVATE_RESEARCH_DRYRUN_INDEPENDENT`  
K5 4,000-step feasibility status: `NOT_STARTED_REPORT_FIRST`  
Formal release status: `BLOCKED_RESEARCH_PRNG`

Subsequent record-audit note (2026-09-09): this four-step dry-run occurred **before** the later B0
generation gate. It retained no checkpoint, but it is still a private-role optimizer execution.
Accordingly, B0 may be described as frozen before the full 4,000-step K5 matrix, not before every
private-role optimizer update. Current cross-document status authority: `../CURRENT_STATUS.md`.

## 1. Exact scope frozen before execution

This was the first optimizer execution using images assigned to the experiment's `private_train`
partition. The source is still the public NIH ChestXray14 proxy, but identifiers, sample
realizations, and diagnostics were handled as restricted to test the intended private-image
lifecycle.

The scope was frozen before any private-partition loss was computed:

- K5 feasibility tier, not K2 accounting-only or K10 main;
- M1-I8, M1-G8, and M2-P8 in that order;
- exactly four optimizer steps per arm;
- P256, the pinned SD 2.1 revision, identical rank-8 attention-LoRA initialization, and exact frozen
  AdamW settings;
- M1-I8 and M1-G8 share one image-Poisson schedule and one diffusion timestep/noise schedule for a
  controlled comparison, but use separate DP Gaussian streams;
- M2 Poisson-samples patients and uniformly chooses without replacement up to four K5 images inside
  each selected patient;
- an empty sample must produce a Gaussian-noise-only optimizer step and must never be resampled;
- no checkpoint, adapter, latent bank, gradient bank, generated image, or private seed may be saved.

Frozen dry-run protocol SHA-256:
`0509D745E8B192FC2CE7694EBDA0C36AB60005D704864D27A44681FD6E8432C8`.

Four steps were chosen only to exercise optimizer state after the first update and to create a
multi-event runtime trace. They cannot be interpreted as convergence, feasibility, utility, attack
resistance, or a privacy result. M0 is intentionally deferred to the separately frozen full K5
feasibility matrix because it does not exercise the DP runtime.

## 2. Randomness and disclosure boundary

The run obtained a fresh 256-bit master value from the operating-system entropy pool and kept it in
process memory only. Domain-separated 63-bit values seeded ordinary PyTorch CPU/CUDA generators for
sampling and Gaussian draws. This improves accidental seed reuse control but does **not** make the
PyTorch generator a release-grade CSPRNG. The result is therefore `RESEARCH_ONLY`.

No sampling, diffusion, or DP-noise seed was serialized. A one-way session commitment binds the
three arm artifacts to the same ephemeral session. The public event traces contain only scheduled
mechanism fields and one completed event per successful optimizer step. Raw selected identifiers,
realized counts, losses, norms, and noise/update digests are confined to
`restricted_runtime_diagnostics.json`, which is excluded from any release package.

## 3. Actual loader realization

The result-independent Poisson schedules were built completely before model gradients were loaded.
The local restricted audit observed:

| Schedule | Step 1 | Step 2 | Step 3 | Step 4 | Total |
|---|---:|---:|---:|---:|---:|
| Shared M1 image units | 11 | 11 | 9 | 6 | 37 |
| M2 patient units | 3 | 4 | 3 | 3 | 13 |
| M2 raw images after inner sampling | 3 | 8 | 6 | 5 | 22 |

No step was empty and no resource abort threshold was reached. The run used 59 unique selected
images and 19 unique prompts in an ephemeral P256 latent/text preparation. No selected input or
latent was written to disk.

The M1 arms matched exactly on every selected image ID, patient mapping, timestep, and diffusion
noise digest at all four steps. Their first-step per-unit gradient digests were also exactly equal,
as required before their arm-specific DP updates caused their model states to diverge.

These realized counts and the clipping observations below are runtime diagnostics only. They were
not used to change C, sigma, q, dataset, step count, or the later K5/K10 matrix.

## 4. Gradient, clipping, noise, and optimizer result

Every sampled image or patient produced one finite fp32 full-LoRA vector. M2 used one mean loss over
the patient's selected images and performed one backward call before its outer patient clip.

| Arm | Clip counts by step | Total clipped/units | Fixed denominator | Four-step adapter delta L2 |
|---|---|---:|---:|---:|
| M1-I8 | 1, 5, 1, 3 | 10/37 | 8 | 0.2972447111 |
| M1-G8 | 1, 5, 1, 3 | 10/37 | 8 | 0.2972437096 |
| M2-P8 | 2, 1, 2, 2 | 7/13 | 4 | 0.2971245063 |

All three arms began from the exact same adapter digest
`16C4F33A2A86C52667B7AFE1513877EAB74F9C54DD55D25AFB4D9AA47EF84172`.
Each completed four nonzero optimizer updates; all adapter and optimizer tensors stayed finite; and
the three final adapter digests were distinct. No parameter tensor was retained after its arm.

The observed clipping fractions are too small and outcome-dependent to judge calibration quality.
They are recorded solely to show that both clipped and unclipped real private-role units passed
through the frozen code path.

## 5. Runtime and four-event accounting

| Arm | Four-step loop time | Peak allocated CUDA memory | Conservative epsilon at recorded delta |
|---|---:|---:|---:|
| M1-I8 | 17.5620 s | 1,981,990,400 bytes | 4.36737943 at `delta=1e-5` |
| M1-G8 | 17.4108 s | 1,982,506,496 bytes | 1.49500466 at `delta=1.32653965e-8` |
| M2-P8 | 7.9454 s | 2,088,136,704 bytes | 4.33243715 at `delta=1e-5` |

The full three-arm model phase took 49.0114 seconds. A purely linear loop-only projection is about
4.88 hours for M1-I8, 4.84 hours for M1-G8, and 2.21 hours for M2-P8, or 11.92 hours total on this
RTX 3070. This is only a planning estimate from four warm-cache steps; it excludes M0, input
preparation, evaluation, attacks, checkpoints, failures, and replication, and is not a runtime
commitment.

Each arm emitted exactly four chained schedule-only events after successful optimizer steps.
Opacus 1.6.0 and Google `dp-accounting` 0.6.0 independently recomputed the four-event values over the
frozen 155 orders. These numbers are accountant/runtime consistency checks, not release evidence;
the backend remains non-cryptographic.

## 6. Independent verification

The independent verifier imported neither the runner nor `mechanism.py`. It:

- rechecked all frozen file hashes and K5 counts of 18,393 images/8,476 patients;
- verified every restricted selected image belongs to K5 `private_train`, every M2 image belongs to
  its one declared patient, and every patient unit contains one to four distinct K5 records;
- recomputed clip flags, aggregation counts, fixed denominators, Gaussian scales, resource limits,
  trace chains, and all three accountants;
- confirmed the four-step M1 schedule/diffusion matching and first-step gradient equality;
- verified that no raw selected identifier occurs as a value in the public report or trace;
- statically confirmed schedule construction precedes model gradients, Poisson outer sampling and
  within-patient `randperm` exist, OS entropy is used, no literal private generator seed exists, and
  there is no model-save call;
- scanned the output for checkpoint-like files and found zero.

Independent report SHA-256:
`27D81FF7565659B3168CECCBA3A94DE31A92CA29154943559AC4510E32B283F9`.

Fifteen CPU/conformance tests pass: ten mechanism tests and five new private-runtime schedule,
boundary, domain-separation, deterministic-test, and resource-fail-closed tests.

## 7. Artifacts and storage

- Public report SHA-256:
  `77FA9097F1FA161E42E72B7E182F5E0A59CB47EC5D89209929DEE0E95767FB87`.
- Public event traces SHA-256:
  `FB0C25EFD601C56871655C08F4F4BE0DD51AA8EE093A89E8C00A5D2F89C6CBC1`.
- Local restricted diagnostic SHA-256:
  `085A518417279A07D2D2E3D3D5E60A3732262345BE5D60274473F52FA6B2DC3A`.

The dry-run and independent report directories contain four JSON files totaling 108,309 bytes.
There are no model/checkpoint files. C: had 46,782,050,304 free bytes (43.57 GiB) at the final
check. All seven active thesis/abstract LaTeX hashes still exactly match the preserved baseline; no
PDF was rebuilt.

## 8. Decision and next gate

This gate establishes that the complete private-role runtime path works for multiple steps:
Poisson selection, patient-bounded inner sampling, real X-ray preprocessing, SD 2.1 LoRA gradients,
unit-correct clipping, Gaussian noise, fixed-denominator aggregation, AdamW state, event traces, and
accounting remain connected.

It does not show that a 4,000-step model trains well, that K5 is sufficiently powered, that M1 or M2
has better privacy/utility, or that any model may be released. No attack, generation, medical
utility, PP-Mark, receipt, or release decision has been executed.

The next step is to report this gate, then design and freeze the full 4,000-step K5 one-seed
feasibility execution including M0, logging/checkpoint retention, failure rules, utility criteria,
and estimated overnight resource schedule. No full feasibility run should start from these four
steps automatically.
