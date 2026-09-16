# NIH-CXR14 K5 B0 generation gate

Date: 2026-09-03  
Run ID: `nih_cxr14_k5_feasibility_v1_001`  
Current human-readable status (2026-09-09 audit):
`PASS_EXECUTION_INTEGRITY; TEMPORAL_CLAIM_NARROWED_TO_FULL_MATRIX; M0_NOT_STARTED`

## Record-audit correction

The K5 `private_train` four-step `RESEARCH_ONLY` dry-run documented in
`XRAY_K5_PRIVATE_RESEARCH_DRYRUN_GATE.md` occurred earlier on 2026-09-03. Therefore B0 was not
generated before every private-role optimizer update. B0 was generated before initialization and
every optimizer update of the **full 4,000-step K5 feasibility matrix**.

The hash-frozen protocol/result/manifest JSON files are not rewritten. Their strings
`B0_must_precede: ... every private optimizer update`, `generated_before_private_training: true`,
and `PASS_B0_FROZEN_BEFORE_PRIVATE_TRAINING` are retained as historical artifacts but must be
interpreted only at the full-matrix scope. They cannot be cited as evidence that the earlier bounded
dry-run did not occur. B0 file integrity and comparator validity are unaffected because the dry-run
used separate ephemeral arm states and retained no model or checkpoint.

## Purpose and boundary

B0 is the hash-pinned, unadapted SD 2.1 comparison baseline. Its 448 outputs must be generated and
hashed before full-matrix initialization or any full K5 feasibility optimizer update. Passing this
gate proves execution integrity and that narrower temporal baseline boundary only. It does not prove radiographic
fidelity, clinical utility, privacy, near-copy safety, or release eligibility.

Generated PNGs remain under the local restricted-generation staging tree. The public report may
contain hashes and aggregate checks, but not the images. The separate full-training run root must
remain absent throughout this gate.

## Frozen task and generation contract

- Task manifest: seven conditions x 64 fixed prompt/seed pairs = 448 images.
- Task-manifest SHA-256:
  `B0379A2A1EBDFB8758AAD865D6F709D803C7632D2697A7BB9A5DFD9C23F2124B`
- Base model: `Manojb/stable-diffusion-2-1-base`, revision
  `0094d483a120f3f33dafbd187ea4aa60d10de75c`, local fp16 safetensors.
- Sampler: DDIM, 50 steps, guidance 7.5, no negative prompt, batch size 4.
- Output: `np` pipeline output, `clip(x*255, 0, 255).astype(uint8)`, lossless P256 RGB PNG,
  compression level 9, no optimization or metadata.
- Every image uses its own CUDA generator and frozen seed. All 448 rows are retained; rerolling,
  rejection, replacement, result-dependent seed changes, and checkpoint selection are prohibited.

The result-independent B0 protocol was frozen before any B0 output:

- Protocol SHA-256:
  `49C18694F740053048971C88286B55C256F28FC8BB223B94A8F1132FA0686324`
- Generation primitives SHA-256:
  `CC677A914CF60728DD798B8809F5267249DC5BE381C9009DA70AB2D8A838C1EB`
- Resumable runner SHA-256:
  `B74F3A36BD3DA426CFFDBC6568212E0A8F7090F8013650783AFC4AE54808B5AF`
- Independent verifier SHA-256:
  `11D752C9E078A07659F1481FC8282FACF7D1D181C9CAFBE305B3DCA75303276A`

The virtual environment inherits a discoverable but unusable host ONNX Runtime. The already passed
SD 2.1 base gate masked that unused optional backend from Diffusers feature detection. B0 protocol
v1_001 omitted the same compatibility action and stopped during pipeline loading with zero images
and zero optimizer steps. Its protocol and empty progress evidence were retained; no threshold,
prompt, seed, model, batch, sampler, or output rule was changed. V1_002 added only the existing
base-gate ONNX mask and was frozen before any image output. The superseded v1_001 protocol SHA-256
is `6DE349EF03C8F85BD3C7B67CF7AE09AB4C6ED8110166CB1479019B62814AAF0D`.

## Resume and independent-verification rules

Before an image obtains its final filename, the runner atomically records the task identity, encoded
PNG hash, and pixel statistics. Existing final files are never overwritten. A prepared-but-missing
file requires regeneration of the same complete four-row batch and an exact encoded-hash match;
unknown, extra, or stale temporary files fail closed.

The independent verifier does not import the generator. It reconstructs all 448 tasks from the
frozen salt, re-decodes and re-hashes every PNG, recomputes every pixel statistic, checks the exact
file set, and independently reloads SD 2.1 to regenerate batches 0 and 111 (eight images total).
Only that verifier may atomically publish the full-runner-compatible B0 PASS manifest.

The frozen pixel checks are decode rate 1.0 and at most 0.01 for duplicate-surplus, low-contrast,
and saturated-image fractions. Low contrast means population standard deviation below 0.02 after
PIL 8-bit RGB-to-L conversion. Saturation means more than 98% of L pixels are at most 1 or at least
254. Duplicate fraction is `(448 - unique encoded-PNG SHA-256 count) / 448`.

## Preflight result

All seven B0-specific unit tests and the full 39-test DP-training regression suite passed. The
independent preflight reconstructed every task, rechecked environment/model/upstream/source hashes,
and confirmed that no B0 output, full-run root, or full-matrix optimizer step existed. It did not
establish absence of the earlier separately reported four-step dry-run.

- Independent preflight status: `PASS_B0_PROTOCOL_AND_IMPLEMENTATION_BEFORE_OUTPUT`
- Frozen full-runner primary gate SHA-256:
  `661A0F3F135C6DA4F0CAF1D000AD1A1B5A5CAC7A23786081F86D1A380199FE74`
- Full-runner launch authority SHA-256:
  `A1A07C51FBFFF6462A6248D45E77D6C25A8F9041A7F5B675092EC8A0BF2551F2`

## Execution result

An unrelated `jailmeter/Qwen` workload initially used approximately 5.2 GiB of the RTX 3070. It was
not modified; B0 waited until that workload finished and the GPU passed the frozen idle gate. The
first post-workload start was rejected by the runner when its own second utilization check observed
a transient busy GPU. This occurred before staging creation. V1_002 then started at 7,661 MiB free,
8% utilization, and 36 C.

The runner generated and atomically committed all 112 fixed batches (448 images) in 467.17 seconds,
or 1.04279 seconds/image including pipeline loading and progress writes. Peak allocated CUDA memory
was 3,172,514,816 bytes (2.9546 GiB). The 448 PNGs occupy approximately 38.31 MiB.

Primary and independent pixel results agreed exactly:

- Decode rate: `1.0`.
- Exact duplicate-surplus count/fraction: `0 / 0.0`.
- Low-contrast count/fraction: `0 / 0.0`.
- Saturated count/fraction: `0 / 0.0`.
- Every condition has exactly 64 images.
- Restricted inventory SHA-256:
  `C385BA3A0C642E9B3A62B0D8FFF11232BBCF842FDC25102C5A26EFC89DB5C984`.

The independent verifier reconstructed all 448 task rows, recomputed every file hash and statistic,
confirmed the exact file set, and independently regenerated batches 0 and 111. All eight regenerated
PNG hashes matched. The full runner then accepted the B0 manifest read-only. At publication, the full
private run root, matrix initialization, M0 completion, and every full-matrix optimizer step remained
absent. The prior isolated four-step dry-run is not included in that statement.

- Primary result SHA-256:
  `E4E0C0429FFB781110975F913B6FE298BB43188E74DD5BB27B8E594CDE2E4B47`.
- Independent verification SHA-256:
  `A1A157A996ABBA631D4289DC18B490C894928815E4F6931DB756D9A6A1E12520`.
- Full-runner-compatible B0 manifest SHA-256:
  `6CB11A45A5B99D61E5B44F7C9463A7832CF5439AE0DF7D497CF8CF752CC94962`.

## Qualitative sanity flag

The fixed indices 0--4 for all seven conditions were assembled into a local, unlabeled 35-image
grid after the integrity manifest was published. All 35 were visibly non-conventional frontal chest
radiographs or had grossly implausible geometry. This does not reverse the B0 execution PASS: B0 is
the deliberately unadapted general-image comparator. It does establish that M0 radiograph-domain
adaptation is a hard scientific gate, and that no DP-arm result can be interpreted if M0 fails to
produce a credible chest-X-ray distribution.

This was an implementation-agent sanity review, not blinded radiologist review, disease accuracy,
or clinical validation. It cannot select a checkpoint. The contact sheet remains restricted and is
not release-eligible.

- Restricted contact-sheet SHA-256:
  `1DF156B25D7929D6B1AE27C70EE12E7D3368EC8E3E8E2900AAB8B33C3C9ACD45`.
- Qualitative report SHA-256:
  `71571EB82B52A30D49A05566D819CA81186DFC2C53B1FC53576866C5F8680919`.

## Stopping point

B0 is complete and valid for use as the fixed off-domain comparator and as a baseline fixed before the
full K5 matrix. M0 has not been initialized or started. The next decision is whether to initialize the
matrix and run only M0 first. M0 must then be
generated with these same 448 prompt/seed pairs and pass the already frozen radiograph-domain gate
before any DP arm is interpreted or continued on scientific grounds.
