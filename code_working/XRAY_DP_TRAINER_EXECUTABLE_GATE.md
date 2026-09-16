# NIH CXR14 executable DP-trainer conformance gate

Date: 2026-09-03  
Decision: `PASS_EXECUTABLE_DP_TRAINER_CONFORMANCE`  
Private pilot status: `NOT_STARTED_REPORT_FIRST`  
Formal release status: `BLOCKED_NO_REGISTERED_RELEASE_GRADE_RNG`

## 1. What this gate answers

This gate asks whether the frozen M1 image-DP and M2 patient-DP definitions can be executed without
silently changing their privacy unit, clipping boundary, Gaussian scale, denominator, or accountant
event. It does not train a private generator and does not establish medical utility, attack
resistance, or a release claim.

The upstream protocol remains byte-identical at SHA-256
`F2757DC7EBC7488B6A9DD0F69227419EA16BB9A3E9BD4434514CB19AFD2B4018`.

## 2. Reuse decision

The older `unitdp_compiler_reference/src/unitdp/owa_dpsgd.py` cannot be the X-ray execution path as
written. It computes one mean gradient per owner and clips at the owner boundary, but its Bernoulli
loop executes `continue` when no owner is selected. That skips both noise and optimizer
post-processing. The frozen Poisson-sampled Gaussian mechanism requires a Gaussian-noise-only update
for an empty sample.

The old code remains technical reference evidence; it was not modified or silently presented as
conformant. A small X-ray-specific core was added in `dp_training/mechanism.py`.

## 3. Executable invariants

The new core enforces the following before an update can be returned:

1. Every privacy-unit gradient is one flat, finite fp32 vector.
2. M1 clips each complete image vector once.
3. M2 averages the selected image losses/gradients inside one patient, then clips the complete
   patient vector once. Clipping images before that average is a tested illegal comparator.
4. Gaussian noise has pre-division standard deviation `sigma*C` and is generated unconditionally,
   including a zero-unit Poisson sample.
5. The result is divided by the public expected batch, eight images for M1 or four patients for M2,
   never by the realized sample size.
6. `q` must equal `expected_batch/population`, and adjacency must match the declared image or patient
   unit.
7. The public event trace contains only the scheduled mechanism fields and a hash chain. Sampled
   identities, realized batch sizes, and private sampling/noise seeds are prohibited.
8. Deterministic test and ordinary research RNGs are labeled non-release paths. This gate does not
   claim that a release-grade Gaussian backend now exists.

## 4. Synthetic conformance result

Ten fast unit tests and nine integrated checks passed.

- The analytic `(3,4)` vector clipped at `C=2` to `(1.2,1.6)`.
- The patient mean-before-clip sentinel produced `(1,0)`; the prohibited clip-before-mean operation
  produced `(0.5,0)`.
- A zero-unit sample produced a nonzero, exact-replay noise-only update.
- The same-noise difference test recovered the clipped signal divided by fixed denominator four
  exactly, even with only two realized units.
- 262,144 standardized Gaussian samples had mean `-0.0005089652` and standard deviation
  `1.000676144`, within the predeclared 0.01 tolerances.
- Across 20,000 synthetic Poisson steps with `N=100,q=0.04`, mean sample size was `3.98205` versus
  four expected and empty frequency was `0.01725` versus `0.0168703`; both were within five standard
  errors.
- Wrong `q`, variable denominator, and invalid privacy-unit configurations failed closed.
- The two 4,000-event scheduled traces chained correctly, contained none of the prohibited public
  fields, and detected a changed noise multiplier.
- Opacus and Google accounting independently replayed the frozen K10 M1-I8 and M2-P8 4,000-step
  values with maximum serialized drift zero over the same 155 orders.

Synthetic report SHA-256:
`FB33CF43F6C5C0F6D4B60879A22E8EC97C4BD87BE933747171970210E12E94F5`.

## 5. Disposable public X-ray LoRA step

The actual-model smoke used only K10 `public_development`. Each arm loaded a fresh copy of the exact
pinned SD 2.1 UNet, initialized the same rank-8 attention LoRA at the same seed, kept 1,659,904 LoRA
scalars in fp32, and discarded the model after one AdamW step without writing a checkpoint.

For clipping-path coverage only, the final smoke selected the maximum frozen public-calibration
gradient in each target/control stratum. Patient sentinels were additionally restricted to exactly
four images. This is deliberately outcome-dependent public diagnostic selection and is prohibited
for utility, privacy-attack, or generalization evaluation.

The calibration and smoke preprocessing batch contexts are not identical, so cross-run gradient
norms are reported rather than required to be bit-identical. Every smoke norm independently had to
remain above the already frozen C. Exact replay was instead required inside the same prepared smoke
context for one real-gradient sentinel in each arm.

| Arm | Real units/images | Frozen C | Smoke unit norms | Units actually clipped | Adapter delta L2 |
|---|---:|---:|---:|---:|---:|
| M1-I8 | 2 images | 0.2844870061 | 1.193793684, 0.341953597 | 2/2 | 0.128836508 |
| M2-P8 | 2 patients / 8 images | 0.1997973089 | 1.122650613, 0.277754529 | 2/2 | 0.128837037 |

Both arms began from adapter digest
`16C4F33A2A86C52667B7AFE1513877EAB74F9C54DD55D25AFB4D9AA47EF84172`.
All 1,659,904 trainable scalars changed finitely after the noised gradient was assigned and AdamW
stepped once. Test-noise replay and one real-gradient replay per arm were exact. Peak allocated CUDA
memory was 1,914,889,728 bytes for M1 and 2,073,194,496 bytes for M2.

The first otherwise successful public smoke used stable-hash-selected target/control units. It
validated gradient construction, Gaussian noise, denominators, and optimizer updates, but happened
to clip 0/2 units in both arms. Its report hash
`4B452CCA01B84CDA10F6CFEC6F7A2A72905FFF2877F34DB212B3C7A10F6A73C2` and replacement reason are
retained in the final report. It was superseded before any private training so that the real-vector
clipping path was also covered.

Final public smoke report SHA-256:
`34FCD5792F7615FAF6B3742AA18F57E8B4FB890CEEF923EEB42DE953183122D5`.

## 6. Independent verification

The independent verifier imports neither the mechanism nor either builder. It reconstructed the
public sentinel commitments from the frozen manifest and calibration table, rebuilt both trace
chains, replayed both accountants, checked every real norm against C, checked two clips per arm,
confirmed equal initial adapters and finite nonzero optimizer changes, scanned for checkpoint-like
files, and statically verified that Gaussian generation is not inside a conditional branch.

It also independently confirmed the exact empty-sample `continue` defect in the legacy owner
trainer. Status:
`PASS_EXECUTABLE_DP_TRAINER_GATE_PRIVATE_PILOT_STILL_BLOCKED`.

Independent verification SHA-256:
`EA2041E282153A547EEE89BBF1D6533315D7C071B5B712907FB2DCC5E6BE7A2C`.

## 7. Storage and source boundary

The three new report directories contain five small JSON files totaling about 30 KiB. No resized
image corpus, latent bank, gradient tensor bank, trained adapter, model checkpoint, or generated
image was retained. The smoke used only selected public-development X-rays for its model inputs; the
verifier read the frozen manifest for contract reconstruction but opened no private-train image.

All seven active thesis/abstract LaTeX source hashes still match the preserved baseline exactly.
No LaTeX source or PDF was changed.

## 8. What is and is not cleared

Cleared:

- the frozen M1/M2 mechanism now has a tested executable aggregation core;
- empty Poisson samples, fixed denominators, patient mean-before-clip, public trace minimization, and
  the two accountant schedules have executable evidence;
- actual SD 2.1 LoRA vectors cross C, are clipped, noised, assigned, and consumed by one disposable
  optimizer step on public data.

Not cleared:

- no private partition model has been trained, even for one step;
- the fixed public sentinel selection does not test the full private Poisson loader or long-run
  training stability;
- no generator quality, patient membership, memorization/extraction, PP-Mark, receipt, or release
  result exists;
- deterministic test RNG and ordinary research RNG remain ineligible for formal release evidence.

The next step, only after this report, is to integrate this exact core into the bounded
research-only private pilot loop and run a separately logged minimal end-to-end pilot before any
4,000-step arm. A full pilot must still fail closed on unit/config/hash drift and remain labeled
`RESEARCH_ONLY` until an audited release-grade RNG path exists.
