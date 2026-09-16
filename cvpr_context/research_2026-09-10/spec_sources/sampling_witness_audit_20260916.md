# Sampling wrapper 정합성: 저장된 공개 witness 감사
2026-09-16. 읽기 전용 감사. Public32의 manifest와 기존4개 witness NPZ, 공통 projection, cache summary, 원 extractor/입력 함수를 확인했다. GPU·새 noise 생성·신경망 재실행·학습·성능 계산은 하지 않았다. 최종 calibration/test의 영상·latent·평가 결과는 열지 않았다.

## 1. 바로 쓸 수 있는 최소안
**Public records 0,7,16**을 최소 witness로 권장한다. 같은 영상의 낮은/높은 timestep 두 개와 다른 영상/조건 하나를 포함한다. Record23까지 포함하면 이미 저장된 public4개 witness를 모두 쓰므로 작은 추가 비용으로 두 영상 각각 low/high를 확인할 수 있다. 성능을 보고 고른 영상이 아니며 기존 projection 검산에 쓰던 고정 witness다.

공통 directory: `code_working/_reports/frozen_residual_public32_20260916_v1/`.

| record | raw 경로 | image ID | draw | timestep | CPU noise seed |
|---:|---|---|---:|---:|---:|
| 0 | raw/000000.npz | 00000200_000.png | 0 | 116 | 505183250027978868 |
| 7 | raw/000007.npz | 00000200_000.png | 7 | 942 | 567099780825388136 |
| 16 | raw/000016.npz | 00000887_001.png | 0 | 97 | 643267576184694072 |
| 23 | raw/000023.npz | 00000887_001.png | 7 | 958 | 361892271071434184 |

Raw SHA를 실제 다시 계산했으며 manifest와4개 모두 일치했다.
- 000000: `15875b6ace9afffa75564d7d6d604316b5b80a0e1e79362114bc2dd19b63db7a`
- 000007: `62bf3411e48dc2960121ad4efa6a555a9c580a06b1b1b62663ff97009465cac9`
- 000016: `cb08ff5d8d8f320c59189680b8c8c1e606331d89ba0bd4a80a7d91c11d8ad1ec`
- 000023: `004944df15631c92814da80de393f45c48a32994912c79d094fbcf0ee1182e1c`

## 2. 저장되어 있는 것과 없는 것
4개 모두 다음 key/shape/dtype를 실제 확인했다.

| key | shape | dtype | 의미 |
|---|---|---|---|
| features | [1024,16] | FP32 | 상수1+GPU에서 계산한15개 projected channel |
| hidden | [1024,320] | FP32 | conv_out 직전, norm/activation 뒤의 원래 hidden |
| base | [1024,4] | FP32 | UNet epsilon 예측을 공간 순서로 flatten |
| target | [1024,4] | FP32 | 이 epsilon 모델에서는 실제 주입한 noise |
| basis | [4] | FP64 | 해당 timestep의 log-SNR Legendre basis |
| timestep | scalar | int64 | manifest와 같은 timestep |

원 clean latent, noisy latent, text conditioning embedding은 NPZ에 직접 저장되어 있지 않다. 그러나 원 캐시와 noise seed/target가 있어 정확한 재구성 경로가 있다. `hidden`은 모든512개 raw에 있는 것이 아니라 위4개 witness에 추가 저장한 값이다.

Base/target/hidden/features의 공간 flatten 순서는 `[C,H,W]→[H,W,C]→[1024,C]`다. Target를 다시 [1,4,32,32]로 만들려면 `reshape(32,32,4).permute(2,0,1).unsqueeze(0)` 순서를 사용한다. 채널 우선 flatten으로 잘못 해석하면 noise/예측이 모두 달라진다.

## 3. 원 입력의 재구성 경로
원형은 `run_public.py:95–115`, `u_patient_audit/models.py:43–53`에 있다.

1. `cache["latents"][image_id]`의 저장 posterior-mode latent [1,4,32,32]를 FP32로 올린다. VAE를 다시 encode하거나 sample하지 않는다.
2. `torch.Generator()`의 **CPU** generator에 manifest.noise_seed를 넣고 `torch.randn(z.shape,dtype=torch.float32)`로 epsilon을 만든다. CUDA RNG로 바꾸지 않는다. 재생성한 epsilon을 원 raw.target의 역변환과 exact 대조할 수 있다.
3. 원 pinned `DDPMScheduler`를 사용하여 **CPU FP32**의 `scheduler.add_noise(z,epsilon,torch.tensor([t]))`를 호출한다. 원 extractor가 CPU에서 noising 후 CUDA로 옮겼으므로, 직접 FP64 sqrt(alpha) 수식이나 GPU에서 먼저 noising하는 경로를 “원 입력 exact replay”로 부르지 않는다.
4. Manifest.prompt와 `cache["train_prompts"][image_id]`가 같은지 확인한 뒤 `cache["hidden"][prompt].float()` [1,77,1024]를 사용한다. 원 token encoder를 다시 실행할 필요가 없다.
5. Fresh base의 동일 fp16 weight artifact를 FP32로 올리고 eval/frozen/noTF32 설정을 유지한다. **기존 LoRA helper의 adapter=None을 fresh no-LoRA base와 혼동하지 않는다.** 원 `run_capacity.new_model`은 direct UNet loader를 쓴다.
6. 신경망을 실행할 때 t는 CUDA int64 [1], conditioning은 FP32이고 batch1이다. 원 raw.base/hidden/features에 대조한다.

이 문서에서는 위 경로를 실행하지 않았으며 code/저장 shape를 확인했다. Cache 전체를 로드해 최종 cal/test 자료를 접근하지 않고, cache summary와 기존 공개 검산의 binding을 확인했다. 실제 새 runner는 필요한 두 공개 image ID의 latent/condition만 소비하면 된다.

**RNG 추가 확인:** seed salt는 `frozen-residual-capacity-20260916-v1`이며 SHA256 문자열의 첫15hex를 정수로 쓴다. 인자는 `("public", image_id, draw, "noise")` 또는 `"timestep"`이다. 원 role이 quality라고 해서 split seed를 quality로 바꾸면 안 된다. Timesteps는 `125*draw+default_rng(t_seed).integers(0,125)`다.

## 4. Head 정합성에서 확인할 것
- projection.npz의 P는 **FP64[320,15]**, alphas는 **FP32[1000]**이다. 실제 extractor는 P를 FP32로 변환해 GPU에서 hidden@P를 계산한 후 상수1을 앞에 붙였다. FP64 projection 재계산을 원 features와 무조건 bitwise 비교하면 안 된다.
- Full64 feature는 basis-major이다. `phi=(basis[:,None]*features[:,None,:])`의 방송 shape를 [1024,4,16]으로 만든 뒤 [1024,64]로 flatten한다. 즉 시간항 바깥, spatial16 안쪽이다. 저장 basis는 FP64이고 head W도 FP64이므로 offline 기준은 FP64로 정의할 수 있다. Online FP32 head와 비교할 때 허용 오차·dtype를 실행 전에 정하고 실제 차이를 보고한다.
- Offline epsilon 보정은 `base + phi @ W_full` 또는 `base + features @ W_static`이다. 원 target/noise는 head feature가 아니라 정합성·loss 기준에만 사용한다.
- W=0일 때 wrapper의 기반 출력이 원 모델과 같아야 한다. 기존 hook의 hidden을 보관하는 동작이 다음 호출에 stale 값으로 남거나 CFG batch를 잘못 섞지 않는지 확인한다.
- Witness는 조건부 단일 UNet 입력의 일치 검증이다. 실제 DDIM sampling timestep 선택·update·VAE decode·CFG null branch까지 검증한 것으로 확대하지 않는다. Generation scheduler의 `scale_model_input`과 원 witness의 noising 경로도 별개로 명시한다.
- 이 public witness는 같은 feature extractor/projection을 사용한 기존 static/full/pooled 가중치 모두에 쓸 수 있다. 필요한 가중치는 각 별도 model NPZ의 hash로 결속한다. 어떤 W를 적용했는지 안 적은 head-output 비교는 충분하지 않다.

## 5. 필요한 입력 binding
Public 계약과 capacity 계약의 feature/projection 일치는 [앞선 감사](pooled_reference_provenance_20260916.md)에서 확인했다. 새 witness protocol은 적어도 다음을 묶으면 된다.

| 입력 | SHA256 |
|---|---|
| public contract.json | 785379be961ed4d1417a74f7764816cb4fe75468d57c982f6328bfde75f66ea2 |
| public manifest.json | d496b207d1730c5c2c88955b001c0c275c6aaf7b27f4e3693b0ce4a7c35de368 |
| public images.json | e9942b361145d13076797f117669873368508fce17566f3f876f91c4a464ff72 |
| projection.npz | 5096130345df5ffb376f099d985ddd9eb7aeafb31c1abe5d9f22a39a1da8801e |
| public independent_verification.json | c058ae8847286e5de6a25b6d8a93eb23896a507e6524a0978647a70b6cfd99bf |
| 原 cache/cache.pt | d1aa72a3c06570b9e25bc589b778448f679ab4e9f2e67073b8413d607becf275 |
| 原 cache/summary.json | a24eddd8d6f911458022d5591446d99b0aeb4ba721ad9c8bd56c9a800e7cf9e4 |
| UNet fp16 safetensors | 28ec9cf3b239c0751c201b1f6fb46b551df5862731b30a37aa1360101cb3fbab |
| UNet config.json | ce0c6d379e3b1d3e1f79338de70c80a1e36f17a6372439a8249b6fb0dfa1b608 |
| scheduler config.json | 11ac5627d7df0fa344b875c4b5722b1767a8a2aa1684c2cf8b4d614300127234 |

원 cache 경로는 `code_working/_reports/cvpr_u_pilot_v1_001/cache/`다. Snapshot은 `models--Manojb--stable-diffusion-2-1-base/snapshots/0094d483a120f3f33dafbd187ea4aa60d10de75c`. Public5개파일과 raw4개는 이번에 실제 hash를 확인했다. 큰 cache/model artifact의 hash는 이번에 다시 전체 읽어 계산한 것이 아니라 frozen public contract/cache summary에서 확인한 기존 binding이다. 새 실행은 이 binding을 직접 확인할 책임을 가진다.

원 public 검산은 저장 산술과4개 projection witness를 확인했으며 UNet을 독립 재실행하지 않았다. 따라서 새 sampling correctness의 neural replay가 그 기존 검산과 구별되는 실제 추가 확인이다. 이 메모 자체는 replay 성공을 주장하지 않는다.

