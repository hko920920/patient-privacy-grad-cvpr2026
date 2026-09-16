# NIH CXR14 K5 evaluator and encrypted-resume preflight gate

Date: 2026-09-03  
Evaluator decision: `PASS_K5_EVALUATOR_PREFLIGHT_INDEPENDENT`  
Resume decision: `PASS_EXACT_ENCRYPTED_RESUME_INDEPENDENT`  
Full K5 optimizer execution: `NOT_STARTED_REPORT_FIRST`  
Formal release: `BLOCKED_RESEARCH_PRNG`

## 1. Scope

This gate implements and checks the two prerequisites left by the full-K5 execution contract:

1. exact RAD-DINO/BioViL-T loading and a real K5 `public_development` validity test; and
2. exact four-step versus encrypted two-plus-resume-plus-two state equivalence for M0, M1-I8,
   M1-G8, and M2-P8 semantics.

It performs no K5 `private_train` optimizer update, model generation, attack, checkpoint selection,
PP-Mark embedding, receipt construction, or release decision.

## 2. Evaluator contract frozen before encoder output

The previously selected 448 images/448 patients were mapped to the seven exact generation prompts
before any encoder output was computed. Pneumonia takes precedence when the weak label is present;
otherwise the pneumonia/consolidation stratum maps to consolidation. Mass takes precedence for the
six Mass+Nodule cases; otherwise the mass/nodule stratum maps to nodule. Resulting counts are 89
consolidation, 48 mass, 64 no finding, 80 nodule, 64 effusion, 39 pneumonia, and 64 pneumothorax.

For the required negative control, rows are sorted by matched condition, selection hash, and image
ID, then prompt donors are rotated by 89 positions. This preserves all prompt frequencies while
making every donor patient and condition different. Bootstrap details are fixed at 2,000
five-stratum PCG64 resamples and a linear percentile 95% interval.

Real images first pass the frozen full-field P256 LANCZOS boundary. RAD-DINO then uses its pinned
518-pixel processor with `use_fast=False`. BioViL-T uses P256 grayscale followed by the official
512-resize/448-center-crop/three-channel transform. Both matrix and cuDNN TF32 are disabled.

The exact downloaded files passed their frozen hashes:

- RAD-DINO weights, 346,345,912 bytes:
  `DBFB9F54459C38773505DE64A6AB7807BDCB392610FE1E697166342E43FB91AE`;
- BioViL-T text weights, 440,909,232 bytes:
  `70BED344872E0A4F4DCA2352564C44F44CB7A9CFCAC3D878BE2DD08A5FFF9ABA`;
- BioViL-T image weights, 109,745,561 bytes:
  `B2399D73DC2A68B9F3A1950E864AE0ECD24093FB07AA459D7E65807EBDC0FB77`.

The BioViL-T checkpoint's two BERT pooler tensors are not used by its projected-CLS architecture.
This known load boundary is explicitly checked and reported; no required text-model key is missing.

## 3. Actual evaluator result

All 448 real images decoded. RAD-DINO produced finite `[448,768]` fp32 features with effective rank
`184.65758447`, total feature variance `159.71734024`, and exact zero drift on the first-eight
replay. Its model phase took 30.69 seconds and peaked at 480,300,032 allocated CUDA bytes.

BioViL-T produced finite unit-normalized `[448,128]` image and `[7,128]` text features. Image replay
drift was exactly zero. The matched-prompt mean cosine was `0.19212072`, versus `0.13153649` for the
frequency-preserving derangement. The paired difference was `0.06058422`, with frozen-bootstrap 95%
interval `[0.03795662, 0.08322361]`; therefore the preregistered overall validity gate passed.

This PASS is deliberately narrow. Four descriptive condition means were negative: consolidation
`-0.11776463`, no finding `-0.05493903`, pleural effusion `-0.12947672`, and pneumonia
`-0.06193671`. Positive conditions were mass, nodule, and pneumothorax. Consequently BioViL-T may
be used as the frozen **overall weak-label alignment screen**, with per-condition values reported
descriptively. It cannot support a claim that all seven conditions are clinically validated or
diagnostically correct.

The independent verifier rebuilt all 448 mappings and the derangement, re-encoded every image, and
recomputed RAD-DINO effective rank and BioViL-T mean/CI. The values matched the primary report. No
per-image feature or score was persisted by either execution.

The first independent attempt correctly stopped because it disabled matrix TF32 but omitted
`torch.backends.cudnn.allow_tf32=False`; BioViL-T's convolutional result differed by about
`4.27e-5` in the aggregate mean. The missing deterministic setting was added and the full
independent pass was rerun. No feature, threshold, prompt mapping, or decision rule was changed.

## 4. Metric implementation prepared for the full run

The evaluator core now provides tested float64 post-encoder implementations of unbiased
degree-three polynomial KID, deterministic KID subsets, PRDC, effective rank, descriptive Fréchet
distance, exact P256 real-image loading, and stratified bootstrap. Six focused tests cover the
formulas, deterministic replay, finite PRDC, effective rank, bootstrap, and native-mode rejection.

This gate does not yet compute B0/M0/DP quality because those images do not exist. It establishes
that the exact evaluator and its real reference validity are usable before training outcomes are
available.

## 5. Encrypted resume implementation

`secure_resume.py` serializes the complete research state only in memory, frames it with an inner
SHA-256, encrypts it with Windows DPAPI `CurrentUser` and the frozen optional entropy, and writes
only ciphertext. It opens `.new` exclusively, flushes and fsyncs it, decrypts and verifies the
temporary, rotates current to previous, and atomically replaces current with `.new`.

Deserialization uses `torch.load(..., weights_only=True)`. A corrupt ciphertext, changed inner hash,
or stale `.new` fails closed. Five tests cover byte/envelope round trip, absence of a plaintext
marker, current/previous rotation, tampering, and stale-temporary refusal.

## 6. Exact restart result

A result-independent protocol fixed a 257-parameter fp32 synthetic adapter and exact AdamW settings.
The four fixtures exercise M0 fixed-batch epoch rollover, both M1 Poisson/Gaussian configurations,
and M2 patient Poisson plus within-patient sampling/mean-before-clip. They use fresh 256-bit OS
entropy and restored domain-separated PyTorch streams. Synthetic populations stress state behavior
only and are not privacy-accounting events.

For every arm, the uninterrupted four-step result exactly equals the state obtained after step-0
save, two updates, atomic step-2 rotation, destruction of the live object, decrypt/restore, and two
more updates. Equality covers:

- the complete fp32 adapter and AdamW state;
- all explicit sampling/diffusion/inner/Gaussian generator states;
- global CPU and all CUDA RNG states;
- M0 permutation/cursor/epoch state;
- public trace records/head and committed step;
- frozen input digests and the internally recovered experiment root.

Step 0 was retained as the previous envelope when step 2 became current. Saving consumed no Torch
RNG. No plaintext resume file was created, and all temporary encrypted preflight envelopes were
removed. An independent implementation then performed its own DPAPI+inner-hash+AdamW/generator/
global-RNG resume and obtained exact equality, while a separate AST audit verified the primary
atomic and four-versus-two-plus-two control flow.

This remains synthetic state-equivalence evidence. It is not a full SD 2.1 checkpoint result,
privacy evidence, utility evidence, or release-grade secure randomness.

## 7. Tests, storage, and artifacts

The combined mechanism, private-runtime, evaluator, and encrypted-resume suite passes 26/26 tests.
New code/protocol/report files occupy only 292,310 bytes (`0.2788 MiB`). BioViL-T added
550,898,615 bytes to the Hugging Face cache; the already present RAD-DINO cache occupies
346,348,659 bytes. No duplicate resized image corpus or encoder-feature bank was created. C: had
40.94 GiB free at the final pre-documentation check.

Key immutable artifacts:

- evaluator protocol:
  `C1283BD37077258F1997BE60FDEE3F49F8C782B0AF3110FFA226EACF11D8F2C9`;
- evaluator task mapping:
  `F82C4513EB64FAC2C17B31686BA18EAC4650C7A2EAA5871BE3C5D0C7CC3889FA`;
- evaluator report:
  `914002B52C2EF48C551FFD9A5CD0260BA7708EA6492F5296AEF98BF6D0B4F5B5`;
- evaluator independent report:
  `A55672E5D5C68491C75AAE4770F1C13A1F1B43BCCD721E7F732BF3C8D9560CAD`;
- resume protocol:
  `4B4B9F20C4830C541ED857EDB751FD00D4032E0ACDB4FCE0C84DC18436808F75`;
- secure-resume source:
  `360D66DA4A9FAE4060C806A71FB02555E58C1E0E71188CFDDC261EDC95895E47`;
- resume result:
  `89A08E9B1AAC248A3DF4F4935C82354D49D28AFC0DD8BE632DE6A80C394FCBFF`;
- resume independent report:
  `720D7A9DDE7A9DD862AE84B4053DBF0F704B8100CE0C7166D28B56EFE76BA463`.

## 8. Decision and next boundary

Both prerequisites pass for the stated research scope. The full K5 run still cannot start
automatically. The next bounded stage is to implement the actual 4,000-step runner, integrate these
resume/evaluator contracts, freeze its complete source and environment manifest, independently
inspect it, and report the launch command, storage projection, and arm-by-arm schedule. Only after
that report may M0 begin.

The investigator subsequently reviewed the negative descriptive condition results and classified
them as a disclosed, non-blocking evaluator limitation for this study. Disease-specific clinical
accuracy, diagnostic replacement, and augmentation efficacy are outside the K5 claim. The central
claims concern privacy-unit alignment, privacy-attack resistance versus general fidelity/diversity,
and model-to-output provenance. This scope decision does not remove any condition result, relax the
frozen overall alignment gate, or turn BioViL-T into clinical evidence. Any later disease-specific
or clinical-utility claim would require a separately designed expert-labelled downstream or
radiologist evaluation.
