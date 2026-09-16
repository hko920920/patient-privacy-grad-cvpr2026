# 고정 residual head의 sampling 연결: 원형 코드 감사

2026-09-16. `run_capacity.py`, `run_public.py`, `residual_math.py`, 캐시 생성 코드, 동결 SD2.1 snapshot 설정, 설치된 diffusers 0.30.2 코드를 읽었다. 이번 메모는 **정적 소스 검토**다. 새 GPU 실행·원시 특징 재추론·생성 품질 검증을 수행하지 않았고 기존 코드를 변경하지 않았다. 새로운 wrapper가 아래 조건을 실제로 만족하는지는 별도의 witness 및 실행 검산이 필요하다.

아래 `CODE`는 `고한경_박사학위논문_작업본/code_working`, `DF`는 `CODE/base_gate/.venv/Lib/site-packages/diffusers`이다. 상대 경로 뒤 숫자는 이번에 읽은 파일의 행 번호다.

## 1. 보정층의 정확한 입력과 출력

| 항목 | 원형 계약·근거 | 연결 요구사항·실수 위험 |
|---|---|---|
| 기반 모델 | `frozen_residual_head/run_capacity.py:169–179`: FP16 artifact를 읽고 `.float().eval()`, LoRA 없음, 모든 parameter frozen | `u_patient_audit.models.load_unet()`는 기본적으로 LoRA를 추가하므로 이 용도의 동일 loader가 아니다. 원형처럼 fresh base를 직접 읽는다. FP32 저장본 등 다른 artifact로 교체하지 않는다 |
| tap | `run_capacity.py:194–199,222`; `DF/models/unets/unet_2d_condition.py:1299–1303` | `conv_out` **forward_pre_hook 입력**, 즉 최종 GroupNorm와 SiLU 이후의 `[1,320,32,32]`. conv_out 출력 4채널이나 activation 전 특징이 아니다 |
| projection | `run_capacity.py:194,222–225` | hidden을 HWC 순서 `[1024,320]`로 펴고 저장 P의 FP32 버전과 **FP32 행렬곱**한다. P는 `[320,15]`. 그 앞에 상수1을 붙여 `[1024,16]`. 다시 QR 생성하거나 정규화·스케일 추가하지 않는다 |
| 시간 basis | `residual_math.py:77–103` | 저장된 1,000개 `alphas_cumprod[t]`를 FP64로 읽어 log-SNR를 계산하고 원 계약 min/max로 [-1,1]에 선형변환. `b=[1,u,(3u²−1)/2,(5u³−3u)/2]` |
| full64 순서 | `residual_math.py:106–120`; `run_capacity.py:313–315` | `[b0*h16,b1*h16,b2*h16,b3*h16]`, **basis-major**. `[channel0*b0..b3,...]` 순서가 아니다. W는 `[64,4]`이며 static W는 `[16,4]` |
| target·부호 | `u_patient_audit/models.py:50–53`; `run_capacity.py:311`; `residual_math.py:3–7` | 현재 prediction_type은 epsilon. 학습 residual은 `epsilon−base_prediction`; 출력은 **base_prediction+phi@W**. latent/x0/영상 공간 보정이 아니고 epsilon 예측 공간 보정이다 |
| 공간 복원 | `run_capacity.py:222,227–228` | `[1024,4]→[32,32,4]→[1,4,32,32]`. 채널 우선으로 바로 reshape하면 픽셀·채널이 섞인다. 보정 자체는 공간별 1×1 선형 head이며 기존 conv_out을 교체하지 않는다 |
| loss scaling | `residual_math.py:6–7,loss_from_statistics` | `/4`는 출력채널 SUM 목적을 MSE로 보고할 때만 쓴다. **sampling correction에 /4를 적용하지 않는다** |

`residual_math.make_features()`는 입력과 P를 FP64로 바꾸어 projection까지 수행한다. 수학적으로 같은 표현이지만 학습 때 저장한 FP32 projection과 동일한 부동소수 경로는 아니다. wrapper에서는 먼저 원형 FP32 `h@P.float32`를 보존한 뒤, 시간 곱·W 곱·base 합산의 dtype을 명시해야 한다. 저장 W와 basis는 FP64다. GPU FP32로 내려 계산한다면 offline FP64 보정과의 차이를 측정해야 하며 bitwise 동일성을 가정하지 않는다. W=0에서 base 출력 복원과 nonzero W에서 수치 일치는 별개 확인이다.

## 2. 현재 snapshot과 DDIM30의 정확한 대응

동결 모델은 `Manojb/stable-diffusion-2-1-base`, revision `0094d483a120f3f33dafbd187ea4aa60d10de75c`이다. 경로는 `C:/Users/SOGANG/.cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/snapshots/0094d483a120f3f33dafbd187ea4aa60d10de75c`이다.

`scheduler/scheduler_config.json`에는 `_class_name=PNDMScheduler`가 기록되어 있으나 학습 특징 추출은 `DDPMScheduler.from_pretrained(...)`를 사용했다. 확인된 값은 train timesteps1000, epsilon prediction, beta_start .00085, beta_end .012, scaled_linear, clip_sample=false, set_alpha_to_one=false, steps_offset=1이다. 모델 이름만으로 v_prediction이라고 추론하면 틀린다.

DDIM 사용은 **샘플러 교체**이며 beta/alpha 학습 좌표는 같아야 한다. 동결 config에서 DDIM을 만들고 저장 `projection.npz`의 1,000개 alpha와 대조하는 것이 올바른 연결이다. 기본 `DDIMScheduler()`는 beta 범위·schedule·clip·offset 등이 달라 동일 계약이 아니다.

설치 소스 `DF/schedulers/scheduling_ddim.py:185–230,297–339`에서 default spacing은 leading이다. 30단계, 1000 train steps, offset1이면 소스 식으로 유도되는 timestep은

`[958,925,892,859,826,793,760,727,694,661,628,595,562,529,496,463,430,397,364,331,298,265,232,199,166,133,100,67,34,1]`

이다. 이 목록은 이번 GPU 결과가 아니라 소스 식 `33*k+1`의 역순이다. 실제 생성 계약에는 scheduler가 반환한 목록을 저장한다. 999부터 시작하거나 linspace/trailing으로 바꾸는 것은 별도 sampling 설정이다. **head log-SNR min/max를 이 30개 timestep으로 다시 잡지 않는다.** 학습 때의 전체 1000단계 범위를 유지한다.

`scale_model_input()`는 현재 DDIM에서 항등 연산이다(`scheduling_ddim.py:236–251`). 그래도 pipeline과 같은 위치에서 호출한다. 학습 noisy latent는 DDPMScheduler.add_noise가 만든 값이며 임의의 추가 표준화는 없다. sampling에서는 `scheduler.scale_model_input(z_t,t)`를 UNet에 넣고, 보정한 epsilon과 **원래 z_t**를 `scheduler.step`에 전달한다(`pipeline_stable_diffusion.py:995–1020`).

DDIM `step`은 epsilon으로 x0를 계산한다. eta=0이면 추가 전이 noise가 없고, clip_sample=false/thresholding=false이면 x0를 clamp하지 않는다. 마지막 t=1의 previous timestep은 -32이며 `set_alpha_to_one=false`이므로 마지막 alpha는 alpha[0]이다(`scheduling_ddim.py:402–448`). 이 부분을 수작업으로 alpha=1로 바꾸지 않는다. 최종 이미지 clamp와 denoising 중 x0 clamp는 서로 다르다.

## 3. 조건부 입력, 해상도, VAE

- **guidance1은 conditional UNet 한 번**이다. null branch를 섞지 않는다. 설치 pipeline은 guidance_scale>1에서만 CFG를 켠다(`pipeline_stable_diffusion.py:729–734,1010–1017`). 현재 head는 weak-label 조건으로 학습했으므로 CFG7.5와 양 branch 보정을 기본으로 추가하면 검증 대상이 바뀐다.
- 정확한 cached-input witness에는 캐시의 weak-label hidden을 그대로 FP32로 올려 사용한다. 캐시는 CLIPTextModel FP16, max_length padding/truncation, 반환값[0]을 저장했다(`build_cache.py:20–32`; `run_capacity.py:209–214`). 별도 FP32 text encoding은 모델이 같더라도 exact replay가 아니다. inference prompt는 학습 template에서 사전 고정하되 사진의 hidden을 임의 조합하지 않는다.
- 현재 UNet config sample_size=64이므로 pipeline의 기본 생성 크기는 64×VAE8=512다. **height=width=256을 명시**해야 latent `[1,4,32,32]` 및 학습 공간 범위와 맞는다(`pipeline_stable_diffusion.py:266,664–683,885–886`). architecture가 다른 해상도를 허용한다는 사실은 그 해상도에서 이 head가 검증됐다는 뜻이 아니다.
- 캐시 latent는 FP16 VAE의 posterior **mode**에 `vae.config.scaling_factor`를 곱한 값이다(`build_cache.py:38–48`). 이를 FP32로 올린 것이 denoising 학습 입력이다. witness 재생성에서 posterior sampling이나 새 FP32 VAE encode로 대체하지 않는다.
- snapshot VAE config는 scaling_factor 필드를 생략한다. 설치 `AutoencoderKL` 기본값은 **0.18215**다(`DF/models/autoencoders/autoencoder_kl.py:87`). 실제 로드된 config 값을 기록·확인한다. sampling 최종 latent는 `vae.decode(latents/scaling_factor)`로 복원하며 다시 곱하지 않는다(`pipeline_stable_diffusion.py:1040`). 초기 생성 noise에는 VAE scaling_factor를 곱하지 않고 scheduler.init_noise_sigma=1을 사용한다.
- VAE의 공간 축소배수8과 진폭 scaling_factor .18215를 혼동하지 않는다. VAE decode의 dtype·variant를 명시한다. head 연결 검산에 사용하는 cached latent와 최종 decode 구현의 정밀도 차이는 다른 문제다.
- 영상 저장의 `(decoded/2+.5).clamp(0,1)`는 시각화 범위 변환이다(`DF/image_processor.py:159–163`). clamp 전 decode tensor의 finite/range도 남기면 시각화가 수치 발산을 가리는지 구별할 수 있다.

## 4. 최소 연결 검증에서 구별해야 할 주장

1. **W=0 동일성:** 같은 모델·prompt embedding·t·입력 latent에서 wrapper가 base epsilon을 복원해야 한다. 같은 초기 noise와 DDIM 설정을 쓴 전체 trajectory도 확인할 수 있다. RNG만 같은 숫자를 지정하는 것보다 초기 latent 자체의 동일성을 확인하는 것이 명확하다.
2. **저장 입력 재생:** 캐시 z와 원 CPU RNG, 원 `add_noise` 순서를 보존해야 한다. FP32 CPU에서 만든 noisy latent를 GPU에서 다시 생성하면 부동소수 순서 차이가 추가된다. 저장 projection·basis·base·target과 source witness를 대조한다. 이는 선택된 witness의 재생이며 과거 모든 원시 데이터의 새 재검산은 아니다.
3. **nonzero head:** 동일 noisy input에서 저장 h16/basis/W로 직접 계산한 offline 보정과 wrapper 보정을 대조한다. 부호·basis 순서·공간 복원·dtype 효과를 이 단계에서 구분한다. scheduler endpoint와 decoded image가 같아 보여도 중간 epsilon 동일성 검사를 대체할 수 없다.
4. **실제 sampling trajectory:** 모델별 궤적이 달라지면 해당 모델 자신의 현재 latent에서 매 단계 h를 새로 뽑는다. base trajectory의 h를 여러 head에 공통 재사용하면 원래의 closed-loop 보정 모델과 다른 연산이다. 한 forward 내부에서 base prediction과 h를 함께 얻는 것은 정확한 재사용이다.
5. **해석 경계:** 저장 noisy input은 real-image corruption이고 생성 궤적 latent는 모델이 만든 값이다. 연결 검산 PASS는 생성 분포에서의 유용성이나 사적 추가 효용을 보장하지 않는다. 이번 올바른 다음 산출물은 구현 정합성 기록이며, 생성 효용 판단은 별도다.

hook은 성공·예외 모두에서 제거하고, 새로운 모델을 평가할 때 이전 capture가 남아 있지 않아야 한다. 최초 패킷은 batch1/guidance1이므로 batch broadcast/CFG 대응을 구현됐다고 확대하지 않는다. 원형 UNet은 tuple 반환(`return_dict=False`)과 `.sample` 반환을 모두 지원하므로 wrapper도 호출자가 요구한 반환 계약을 지켜야 한다(`unet_2d_condition.py:1309–1312`).

## 5. 이번에 읽은 주요 소스 SHA256

| CODE 상대 경로 | SHA256 |
|---|---|
| frozen_residual_head/run_capacity.py | `59a5f27a2b70f708fae3d7ba6ca27380a4446293a2705b695039ee2e141db5f1` |
| frozen_residual_head/run_public.py | `9601d1b18df35155f7c7cc7b60be660207dcf800c4050a50ddaef2ee8f31b894` |
| frozen_residual_head/residual_math.py | `59db490cb427c99df4d6f2eed96618bbffd769821cf8bd8e6b5eba7c83b0d412` |
| u_patient_audit/models.py | `cd1a0b0eca3f4460eadc247d62960eb50f60d09cc4c39c2ed279e5a3afb2fd91` |
| u_patient_audit/build_cache.py | `043dd5d6e9ba2319a1749445c93224551e6fec28e9cad9e15d3ac131bdf8fc7a` |
| base_gate/.venv/Lib/site-packages/diffusers/schedulers/scheduling_ddim.py | `c9029dd1b0dd9ac2fa77f82bfbd77b63cab26db6ce6e144e916f749fabe068dd` |
| base_gate/.venv/Lib/site-packages/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py | `b57979897002c399938a0d928464b1982a30983fbd63b6bb3a7b6fcff6cba267` |
| base_gate/.venv/Lib/site-packages/diffusers/models/unets/unet_2d_condition.py | `d6f44351686baa00c3d003910e1452fe8e3b564b5be77cb91b2a492ebcf64627` |
| base_gate/.venv/Lib/site-packages/diffusers/models/autoencoders/autoencoder_kl.py | `dec8d6110b8d0cdff278624112812dfd5087c1806f7200fa2d605c3cb2f72fb7` |
