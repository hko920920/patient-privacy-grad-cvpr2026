# NIH ChestXray14 K5 full-training runner gate

Date: 2026-09-03

Gate-closure status: **PASS — training runner/source/environment independently frozen; B0 and every 4,000-step arm were unstarted**

Subsequent status: B0 later completed and passed execution-integrity verification. The 2026-09-09
record audit narrows B0's temporal claim to `before the full 4,000-step K5 matrix`; see
`XRAY_K5_B0_GENERATION_GATE.md` and `../CURRENT_STATUS.md`.

## 1. Decision

At this gate's closure, the actual K5 training runner was ready for the next staged operation. It
implements the frozen `M0 -> M1-I8 -> M1-G8 -> M2-P8` order, but this gate did not initialize the
matrix, generate B0, or perform any optimizer update on `private_train`. Both designated full-run
roots remain absent.

The then-next operation was B0 generation and hashing; that later gate passed. The matrix may be
initialized and M0 considered only after the B0 output has been inspected and reported, which is now
complete. Running the CLI with no mutating command is read-only and returns
`PASS_READ_ONLY_FULL_RUNNER_VALIDATION`.

## 2. Scale and step rationale

The K5 private-training cohort contains 18,393 images from 8,476 patients. Four thousand steps
give M0 and each M1 arm 32,000 image exposures, or an expected `1.7397923123` exposures per image.
M2 has 16,000 expected patient-unit exposures, or `1.8876828693` per patient. This is a defensible
one-seed rank-8 LoRA feasibility budget: it gives roughly two passes at the protected-unit level
without pretending to be a confirmatory sample-size or convergence proof. K10 remains the planned
confirmatory stage.

## 3. Implemented long-run boundaries

- Fresh identical rank-8 fp32 LoRA over the pinned fp16 SD 2.1 base for each arm.
- M0: private-seeded canonical permutations, fixed batches of eight, no clipping and no DP noise.
- M1-I8/M1-G8: identical Poisson image schedule, timesteps, and latent-noise draws; independent
  arm-specific DP Gaussian streams.
- M2-P8: Poisson patient sampling, up to four same-patient images sampled without replacement,
  mean loss before one patient-gradient clip.
- Empty DP samples still take a noise-only AdamW step with the public fixed denominator.
- One arm per invocation; the previous arm must have a verified 4,000-step completion record.
- Windows DPAPI `CurrentUser` ciphertext-only resume state at step 0 and every 250 steps, retaining
  current plus one previous envelope.
- Restricted LoRA-only safetensors at steps 1,000, 2,000, and 4,000. Only step 4,000 is eligible for
  quantitative evaluation; intermediate checkpoints cannot select or stop training.
- Fail closed on source/environment/data/model drift, non-finite state, invalid resume/trace,
  frozen realization-limit violation, less than 15 GiB disk space, less than 6.5 GiB free VRAM,
  GPU utilization above 10%, or temperature at/above 80 C.

The WDDM GPU-idle rule intentionally uses aggregate utilization and free memory. Windows reports
ordinary desktop/C+G processes through `--query-compute-apps` with `[N/A]` memory, so treating that
field as a numeric CUDA-job list is invalid on this machine.

## 4. Actual SD 2.1 encrypted-resume result

Before execution, v1_002 froze 16 `public_development` images from eight multi-image patients,
two images per patient. For all four arm semantics, the test compared:

1. four uninterrupted real SD 2.1 LoRA/AdamW optimizer steps; and
2. two steps, encrypted DPAPI save/rotation, live-state destruction, fresh UNet/LoRA construction,
   restore, and two further steps.

All 13 compared state fields matched bit-for-bit in every arm: all named LoRA tensors, complete
AdamW state, schedule and DP Gaussian generator states, global CPU/CUDA RNG states, trace head and
count, commitment, root, step, and frozen digests. Step-2 ciphertext sizes were 20,237,070--
20,242,638 bytes, confirming that the actual 1,659,904-value LoRA and full AdamW moments were
serialized rather than a compact proxy. Peak CUDA allocation was at most about 2.15 GiB. The
temporary encrypted envelopes, model state, optimizer state, and caches were destroyed.

- v1_002 preflight protocol SHA-256:
  `3921227498086C6D3E794FBE92DE12DF229E463D9F461642BDF71187B704924D`
- v1_002 actual result SHA-256:
  `C7BC6F12B02980E3ADBF309DBD16479DC7DCCB67418B65D2F1B5FCD9D8540CD2`

The first v1_001 actual test also passed, but the subsequent full-gate verifier stopped because its
WDDM process parser attempted to convert `[N/A]` to an integer. The parser was corrected, the
protocol and actual four-arm test were rerun as v1_002, and the failed primary gate was preserved at
`_reports/_superseded_failed_nih_cxr14_k5_full_runner_gate_v1_001_wddm_parser_20260903`.

## 5. Independent source/environment gate

The final verifier did not import the runner core. It independently re-hashed sources and inputs,
parsed the runner AST, checked the DPAPI and sampled-Gaussian source structure, replayed both DP
accountants, re-ran all 32 tests, verified the real SD 2.1 resume evidence and model files, checked
GPU/disk readiness, and invoked the frozen CLI in read-only validation mode. The full run roots
remained absent before and after validation.

- Primary gate SHA-256:
  `661A0F3F135C6DA4F0CAF1D000AD1A1B5A5CAC7A23786081F86D1A380199FE74`
- Environment manifest SHA-256:
  `47BED599FE05BD9A7DA710BA80420D6B335943328A5E52D0188FA953E9767767`
- Independent verification SHA-256:
  `FB6EA9674FC5613B8FB3F7E750B6954F8C9535EDF621D042EAD9D9A201800BAB`
- Launch-authority SHA-256:
  `A1A07C51FBFFF6462A6248D45E77D6C25A8F9041A7F5B675092EC8A0BF2551F2`

Frozen training-source SHA-256 values:

- `dp_training/k5_full_runner.py`:
  `D152C59ABA4087809A4AC2251E183306C7541EDDAC4FD0D64A7D46554DE66505`
- `dp_training/run_k5_full_training.py`:
  `5B572136D39DBDC092C298403CB7098BE343341A278537A86AF3CC42E9A9C819`
- `dp_training/mechanism.py`:
  `B4B1E037A3704238B06B442970EAB082433835201B750546C8281BBFFE2B969E`
- `dp_training/secure_resume.py`:
  `360D66DA4A9FAE4060C806A71FB02555E58C1E0E71188CFDDC261EDC95895E47`

## 6. Current stopping point

At gate freeze, C: had approximately 43.1 GiB free. A later final audit saw 36.51 GiB after a
separate, out-of-scope `run_heterogeneous_panel_full_development_e0g4.py qwen` Python process began
using substantial private memory/pagefile; that process was not altered. The seven active
thesis/abstract LaTeX hashes still matched the Section 25.5 baseline exactly. No B0 image,
private-training optimizer step, adapter snapshot, receipt, PP-Mark output, or release artifact was
created in this gate. Resource readiness must be checked again immediately before B0 and each arm.

At this gate's closure, the exact next gate was to implement/freeze the B0 generator, create the fixed
448 images before any **full K5 matrix** optimizer update, verify all files against
`generation_tasks.csv`, and report the B0 result. That B0 gate subsequently passed. The current next
mutating stage, if separately authorized, is matrix initialization followed by M0 only. Only the
passing B0 manifest can satisfy the full runner's matrix-initialization guard.
