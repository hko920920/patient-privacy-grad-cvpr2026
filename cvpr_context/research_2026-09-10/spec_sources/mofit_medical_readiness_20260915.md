# MoFit 의료 모델 실행 준비 상태 — 2026-09-15

현재 2단계는 가까운 기존 방법을 실제 환자 U 조건에 적용하고 관측의 원인을 확인하는 단계다. 이 문서는 원형 소스와 현재 로컬 모델·캐시를 CPU에서 읽은 준비 검토다. **의료 MoFit 공격, 새 target 호출, caption 모델 다운로드, target 학습을 실행하지 않았다.** 진행 중인 CDI 추출의 코드·계약도 변경하지 않았다.

## 1. 현재 판단

현재 target에서 MoFit의 입력·전체 conditioning 차원을 연결할 자산은 있다. 빈 문자열 conditioning과 generic conditioning이 실제 캐시에 존재하며, 전체 77×1024 embedding 최적화는 코드 차원상 가능하다. 그러나 **원형의 pixel→VAE gradient 경로, posterior sample, 의료 VLM 초기 caption, 실제 메모리·시간을 검증한 의료 MoFit 실행은 아직 없다.**

새 target을 학습해야 할 이유가 확인된 것은 아니다. 지금 부족한 것은 먼저 원형 연산의 current-backend 접합과 의료 caption 준비다. 단순히 빈 prompt가 없거나 1024차원이어서 실행할 수 없다는 상황은 아니다.

## 2. 이번에 직접 확인한 근거

- 원문: `texts/X12.txt`의 방법 pp.5–7, 구현 p.15, 비용 pp.20–21, 의료 Appendix A.10 p.23. 저자 코드 commit `91e4b5edc153bac84b0b4209f70d1b2b94e653b2`.
- `code_sources/X12/COCO/MoFit_COCO.py`, `MoFit_COCO.sh`, 수정 pipeline `COCO/diffusers_0_18_2/pipelines/stable_diffusion/pipeline_stable_diffusion.py`, DDIM scheduler, README.
- 현재 `code_working/u_patient_audit/build_cache.py`, `train_coverage.py`, `models.py`, `data_pipeline/nih_cxr14_model_input.py`.
- `_reports/cvpr_u_pilot_v1_001/cache/cache.pt`는 summary hash 검사 후 CPU로 읽었다. CUDA는 초기화하지 않았다. 모델 inference는 하지 않았다.
- 실제 HF cache 경로와 target snapshot config·weight 파일 존재, caption 관련 로컬 파일 이름을 확인했다. 원격 저장소의 현재 상태나 컴퓨터의 모든 임의 폴더까지 조사한 것은 아니다.

원저자 공개 COCO scalar replay는 앞서 완료된 별도 작업이다(`replays/mofit_coco_fixture_20260914/report.md`). 그것은 우리 의료 모델의 추론·최적화 재현이 아니다.

## 3. null/text/latent 자산의 실제 상태

| 항목 | CPU 확인 결과 | 의미 |
|---|---|---|
| target snapshot | Manojb/stable-diffusion-2-1-base, revision 0094d483a120f3f33dafbd187ea4aa60d10de75c | 기존 두 LoRA target과 동일한 base |
| UNet config | in/out channels4, cross_attention_dim1024 | 현재 4×32×32 latent와 77×1024 hidden 연결 가능 |
| text encoder | CLIPTextModel, hidden_size1024, max_position_embeddings77 | 77×768은 원저자 SD1.x 예제 주석이며 현재 필수 shape가 아님 |
| cache hidden | 145개 prompt entry | 임의 의료 VLM caption entry가 있다는 뜻은 아님 |
| empty key `""` | [1,77,1024], FP16, finite | null conditioning 사용 가능 |
| generic key | `a frontal chest radiograph`, [1,77,1024], FP16 | empty hidden과 값이 다름 |
| latent cache | 2,304개, [1,4,32,32], FP16 | posterior mode; pixel gradient나 posterior variance는 저장하지 않음 |
| train_prompts | 2,304개 image→prompt mapping, 빈 문자열 값0개 | 실제 train subset은 이 mapping을 사용; 캐시 전체가 학습됐다는 뜻은 아님 |
| VAE weights | 해당 snapshot의 diffusion_pytorch_model.fp16.safetensors, 167,335,342 bytes | fresh pixel encoder 경로 구현 자산은 로컬에 있음 |
| text encoder weights | model.fp16.safetensors, 680,821,096 bytes와 tokenizer 파일 | 의료 caption 문자열이 생기면 현재 1024-width text encoder로 인코딩 가능 |

캐시 SHA256는 `d1aa72a3c06570b9e25bc589b778448f679ab4e9f2e67073b8413d607becf275`, cohort lock SHA256는 `187fdbcc6ea8b883e6b82d97f293d7a746df5408632e6f8824a9769866e1dfc4`이다.

`build_cache.py:24–32`가 weak-label prompt 집합에 generic과 빈 문자열을 추가해 hidden을 계산한다. 현재 캐시는 FP16 결과이므로 FP32로 promote해도 새 FP32 text encoding과 bitwise 같아지는 것은 아니다. cached-hidden adaptation과 fresh-FP32 encoding을 구분해야 한다.

`train_coverage.py:53–62`는 각 image의 weak-label hidden을 항상 선택하며 null-caption dropout 분기가 없다. 원문 p.5의 “member는 unconditional loss도 직접 줄이는 학습에 참여했다”는 설명을 현재 LoRA에 그대로 적용할 수 없다. 다만 shared UNet/LoRA parameter 때문에 conditional 학습이 null response를 간접적으로 바꿀 수 있다. 따라서 **dropout 없음→null 반응 불변 또는 MoFit 불가능**은 성립하지 않는다. base 사전학습의 의료 영상 포함 여부도 여기서 확인한 것이 아니다.

## 4. 원형의 어느 연산을 유지해야 하는가

### 4.1 이미지 surrogate

`MoFit_COCO.py:158–234`는 원본 RGB를 [-1,1]로 놓고 U[-.3,.3] 초기 perturbation을 더한 뒤, null-conditioned denoising MSE를 이미지 픽셀에 대해 미분한다. t=140, 고정 diffusion noise, sign-gradient update 1,000회, 선형 감소 step size(.15에서 .0015 방향), 매회 [-1,1] clipping이다. 초기 범위 .3을 원본 중심 projection constraint로 바꾸면 원형이 아니다.

매 iteration의 `vae.encode(adv_img).latent_dist.mean`과 scaling이 gradient graph 안에 있다. 현재 고정 mode latent를 직접 최적화하는 것으로 대체할 수 없다. 우리 전처리는 full-field grayscale Lanczos P256→동일 RGB 채널(`nih_cxr14_model_input.py:223–269`), 원형은 RGB P512 bilinear Resize+CenterCrop이다. P256 적용은 명시적인 입력 adaptation이다.

원형은 0.18215를 상수로 쓴다. 현재 VAE config 파일에는 scaling_factor 항목이 없고, 설치된 AutoencoderKL의 기본값이 0.18215다(`diffusers/models/autoencoders/autoencoder_kl.py:75–87`). 실행에서는 실제 생성된 VAE config 값을 계약에 저장해야 한다.

원형 `mtcnp_adv`는 CFG batch2를 만든 뒤 uncond prediction을 loss에 사용한다(`pipeline_stable_diffusion.py:1060–1094`). guidance_scale7.5로 혼합한 prediction 자체가 stage1 loss 대상은 아니다. 혼합 후 DDIM step 반환은 이 caller가 loss에 사용하지 않는다. 현재 DDPMScheduler라는 클래스 이름을 원형 DDIM과 같다고 표현하지 말고, 해당 noising alpha와 scale_model_input 동작을 대조해야 한다. 원형 DDIM scale_model_input은 identity다(`scheduling_ddim.py:243–255`).

### 4.2 전체 embedding 최적화

`MoFit_COCO.py:277–289`는 초기 embedding을 clone해 **행렬 전체**에 requires_grad를 설정하고 Adam(lr=.06)을 적용한다. 코드의 “(1,77,768)”은 주석이며, 실제 변수 shape는 encoder 출력에서 온다. 1024를 막는 hard-coded reshape는 해당 optimization 경로에 없다.

현재 변수 수는 78,848개다. FP32 embedding 한 개는 약0.301MiB이며 parameter·gradient·Adam 1/2차 moment 네 배열은 약1.203MiB다. 이것만 보고 8GB에서 전체 MoFit이 된다고 결론내리면 안 된다. 큰 메모리는 UNet/VAE weights와 activation·backward에서 발생한다.

이 경로에는 R의 저차원 basis, .05 radius, active-token mask, padding-token 고정이 없다. 기존 `token_mask`로 일부 토큰만 최적화하면 원형 전체 embedding과 다른 변형이다.

Stage2는 surrogate와 원본의 VAE posterior **sample을 각각 한 번** 추출해 고정한다(`:279–284`). 원형 stage1의 mean 및 현재 cache의 mode와 구별해야 한다. mode cache에는 variance가 없어 posterior sample을 복원할 수 없다.

### 4.3 원본에서의 점수

Stage2는 surrogate에 fitting하지만 최종 h는 원본에서
h = L(original, phi_star) − L(original, phi_null)
로 계산한다. 보조 u=L(original,phi_null), v=L(original,phi_init)−u도 저장한다(`:342–355`). fitting에 쓴 surrogate의 낮은 loss를 membership score로 보고하면 다른 공격이 된다.

원형 1,000+300회 objective를 줄여 얻은 결과와 단순 source-update conformance를 구별해야 한다. 방향 h와 auxiliary scaling/결합은 fit 역할의 고정 규칙이며, 한 이미지 conformance에서 membership threshold를 선택하지 않는다.

## 5. 의료 caption 준비의 실제 공백

원문 Appendix A.10(p.23)은 **Prompt2MedImage(SD1.4를 ROCO에 fine-tune한 모델)**를 평가하고, 의료 caption을 `Siddartha01/blip-medical-captioning-roco`로 생성했다고 명시한다. 현재 우리의 NIH CXR14 SD2.1 LoRA와 별도 target이다. 표의 MoFit AUC54.44/TPR@1%2.00은 우리 모델의 결과가 아니다.

이번에 실제로 확인한 HF_HUB_CACHE는 `C:\Users\SOGANG\.cache\huggingface\hub`이다. 해당 cache에 Siddartha01 의료 BLIP, Nihirc/Prompt2MedImage, 일반 BLIP/BLIP2 caption 모델 snapshot은 발견되지 않았다. 연구 run의 cache keys와 관련 파일 이름 검색에서도 현재 후보 영상별 의료 VLM caption·pinned revision·generation-policy cache를 찾지 못했다. 검색된 transformers의 blip Python 파일은 모델 weights가 아니다.

로컬에 BiomedCLIP·BioViL·rad-dino가 있다는 사실은 의료 caption generator가 준비됐다는 뜻이 아니다. 확인한 BiomedCLIP `open_clip_config.json`은 ViT와 BiomedBERT encoder/contrastive projection 구조이며, 그것을 BLIP의 caption 생성 대신 사용했다고 할 수 없다.

X12 README는 `data/*.npy`와 `data/Captions` 제공을 설명하지만, **현재 code_sources/X12 로컬 트리에는 data 디렉터리가 없고** COCO/Pokemon/Eval/Results가 있다. 저장 scalar txt는 존재한다. 원형 코드의 NAS 절대 caption/noise 경로(`MoFit_COCO.py:105–114,183–185`)는 현재 workspace 파일이 아니다.

실제 의료 baseline 전에 필요한 것은:
1. 의료 caption checkpoint의 정확 revision/weights 및 processor·generation 설정 고정.
2. 같은 후보 영상별 caption 문자열·input hash·generation metadata 캐시.
3. 그 caption의 현재 SD2.1 77×1024 hidden. weak-label training prompt는 생성 caption이 아니다.
4. P256의 [1,4,32,32] 고정 noise와 posterior draw 정책. 저자 P512 noise 파일을 모양만 맞춘다고 원형 동일-noise 재현이 되지는 않는다.

이 문서에서는 다운로드 크기·라이선스·모델 품질을 원격으로 확인하지 않았고 어떤 모델도 다운로드하지 않았다. 일반 prompt는 수치 접합 검사에 쓸 수 있으나, 그 결과를 “의료 VLM 초기화를 포함한 MoFit baseline”으로 이름 붙이지 않는다.

## 6. 비용: 측정된 것과 아직 측정되지 않은 것

F는 UNet forward-example, B는 backward/autograd 호출이며 VAE와 text encoder 비용을 별도로 센다. 실제 root GPU의 CDI kernel은 UNet latent-input gradient를 실행한 자료이고, **VAE 픽셀 gradient나 MoFit full embedding 시간으로 전환할 근거는 없다.**

| full COCO 원형 N1=1000,N2=300 경로 | UNet F | B | VAE encoder F/B |
|---|---:|---:|---:|
| source stage1 CFG batch2 + source stage2 매회3개 원본 monitor | 2N1+4N2=3200 | N1+N2=1300 | N1+2=1002 / N1=1000 |
| 중복 branch/monitor를 제거한 별도 equivalent-backend 후보 | N1+N2+3=1303 | 1300 | 1002 / 1000 |

두 번째 행은 아직 동등성을 확인하지 않았다. batch2→batch1의 수치 차이, gradient sign, optimizer trajectory가 누적될 수 있으므로 조용히 바꿔 원형 재현이라고 하지 않는다. source는 stage1 매회 prompt를 재인코딩하므로 literal text-encoder 비용도 있다. 동일 text hidden의 사전 계산·모델 offload는 값/precision 정책을 고정한 뒤 별도로 표시한다.

논문 Table8의 P512 실험은 surrogate 약20,667MB, embedding 약14,799MB를 보고한다. 그 값은 현재 frozen-weight P256 구현의 필요 VRAM이 아니다. 반대로 현재 RTX3070 8GB의 CDI peak약4.1GB는 MoFit이 8GB에 들어간다는 보장도 아니다. 기존 target의 FP16-variant weights를 FP32로 promote하는 것과 원저자의 본래 FP32 checkpoint는 구분한다.

저자의 이미지당7–9분이나 Table9 full358.96+116.38초는 원저자 조건의 측정값이다. **현재 의료 model+VAE pixel backward의 시간·peak memory는 미측정**이다. 현재 F/B당 시간을 단일 상수로 두어 1,300번 곱한 확정 ETA를 제시하지 않는다.

## 7. 다음에 할 한 가지 source-conformance 검사

**한 기존 fit E 영상(14393 환자의 첫 고정 E 영상)·model_1 step1000에서 source-update 경로를 대조한다.** 새 환자, U/selection 튜닝, 추가 학습, medical caption 생성은 포함하지 않는다. generic-init임을 이름에 명시하고 의료 MoFit 효능 실험으로 세지 않는다. 현재 CDI GPU 작업이 끝난 뒤 별도 frozen contract로만 실행할 제안이며, 이번에 실행하지 않았다.

최소 범위:
- 현재 고정 P256 raw pixel, empty hidden, generic initial hidden, 실제 alpha[t140]를 사용한다. 원형 initial perturbation·diffusion noise와 stage2 두 posterior draws를 고정해 저장한다.
- 원형 nominal N1=1000 step-size schedule의 첫 **pixel update 2회**를 실행한다. 전체 1,000회 수렴을 수행했다고 쓰지 않는다.
- 얻은 surrogate로 **전체77×1024 Adam update 2회**를 실행한다. 300회짜리 공격 성능을 대신 평가하지 않는다.
- source가 반환하는 원본 h/u/v와 surrogate conditional loss를 구별해 저장한다.
- 실제 gradient를 저장하고 CPU에서 sign-update/clamp, Adam의 두 moment/update를 독립 계산한다. 각 변수 공간에서 사전에 정한 한 방향의 양·음 perturbation loss를 추가 측정해 directional derivative를 대조한다. 이는 단순히 source가 출력한 gradient와 그 복사본을 비교하는 검사가 아니다.
- pixel/VAE/UNet/conditioning dtype·shape, finite gradient, model/VAE weight 값·version·grad 불변, 실제 F/B·VAE F/B·시간·peak memory를 기록한다. feature·score threshold의 성공 조건은 두지 않는다.

원형 CFG2/monitor 경로를 유지한 이 작은 검사 자체는 N1=2,N2=2이므로 12F/4B, VAE4F/2B다. 한 pixel 방향 양·음 loss 검사는 CFG batch2 두 호출로 4F와 VAE2F, embedding 방향 검사는 2F를 더한다. **합계18 UNet F/4B, VAE6F/2B**이며 text 초기화 비용은 별도다. 추가 replay 구현을 동시에 실행하면 그 호출 수를 별도로 더해야 한다. runtime은 아직 측정하지 않았다. gradient-check step 크기·수치 허용치는 해당 precision에서 사전 고정하고 결과 후 완화하지 않는다.

이 한 검사로 구별할 수 있는 경쟁 설명은 **“원형의 필요한 픽셀/conditioning gradient 경로를 잘못 대체·차단하거나 update/shape를 잘못 연결했다”**와 **“현재 backend에서 원형 update 연산은 실제로 연결된다”**이다. source endpoint의 기능 확인에 해당한다.

구별할 수 없는 것은 **“target이 약하게 기억했다” 대 “MoFit이 이 환자 U 설정에서 약하다”**이다. 의료 caption·full1000+300 실행·분리된 E/U 환자 비교가 아직 없기 때문이다. E 한 장의 objective 감소가 U membership 민감도를 보장하지 않으며, 실패도 U 불가능의 증명이 아니다.

## 8. 남은 실행 전제

수치 접합 검사에 필요한 raw image·VAE/UNet·empty/generic hidden은 로컬에 있다. 다음 한 검사에는 의료 BLIP 다운로드가 필수는 아니다. 반면 원문에 가까운 **의료 VLM 초기화 MoFit baseline**에는 caption checkpoint/revision/cache가 실제로 부족하다.

현재 판단은 “MoFit 미실행의 이유가 모델 학습 부족으로 확정됐다”가 아니다. **차원·기본 자산은 연결 가능하고, 원형 pixel/VAE 경로와 full conditioning optimization의 실제 수치·메모리 확인 및 의료 caption 준비가 남아 있다.**

## 9. 추가 확인: 학습 조건 매핑과 이미 수행한 generic/matched 대조

이 절의 상대 경로는 졸업논문 작업본 루트를 기준으로 한다. 기존 파일과 저장 tensor를 CPU로 확인한 기록이며, 새로운 target 호출·GPU 실험·조건 선택을 수행한 것이 아니다. 현재 진행 중인 고정 full-U CDI 비교의 조건과 계약은 변경하지 않았다.

### 9.1 정확한 영상 → 문자열 → hidden 매핑

`code_working/u_patient_audit/build_cache.py:19,24–32,81–82`는 source manifest의 각 영상에 `mi.prompt_for_record(record)`를 적용해 `cache['train_prompts'][image_id]`를 만들고, 그 문자열을 key로 하는 `cache['hidden'][prompt]`를 저장한다. tokenizer는 `padding='max_length'`, 모델의 최대 길이, truncation을 사용한다. `code_working/u_patient_audit/train_coverage.py:53–62`는 학습 영상별로 이 hidden을 조회한다. 이 학습 경로에는 empty/null caption dropout이 없다.

`code_working/data_pipeline/nih_cxr14_model_input.py:154–172`의 weak-label template는 다음과 같다.

- 공통 prefix: `a frontal posteroanterior chest radiograph`
- No Finding: ` with no labeled finding`을 붙인다.
- 나머지: ` with radiographic findings of ` 뒤에 고정 `NIH_LABEL_ORDER`의 `LABEL_TEXT`들을 붙인다. 한 항목은 그대로, 두 항목은 `and`, 세 항목 이상은 쉼표와 마지막 `and`로 연결한다. 예를 들어 공개 표준 용어 조합은 `pleural effusion and pneumonia`이다.
- patient ID·나이·성별은 이 prompt에 들어가지 않는다. 현재 감사용 generic prompt는 `a frontal chest radiograph`이다.

`code_working/_reports/cvpr_u_pilot_v1_001/cohort/evaluation_images.csv`와 `cache/cache.pt`를 대조한 결과, **fit80 + selection40의 E2/U2 총 480개 고유 영상** 모두 source template와 cache의 매핑이 일치했다. 해당 영상들은 서로 다른 weak-label 문자열 70개를 사용하며, 모든 hidden이 FP16 `[1,77,1024]`로 존재한다. 이 70개 문자열 중 generic 문자열과 같은 것은 없다. 매핑 확인에 사용한 cache SHA256은 `d1aa72a3c06570b9e25bc589b778448f679ab4e9f2e67073b8413d607becf275`, cohort lock SHA256은 `187fdbcc6ea8b883e6b82d97f293d7a746df5408632e6f8824a9769866e1dfc4`이다.

**`train_prompts`라는 cache key와 실제 학습 노출은 구별한다.** 특정 target에서 member인 환자의 E 영상은 manifest/step1000 노출 기록에 포함되므로 이 문자열이 실제 사용된 학습 조건이다. 반면 모든 U 영상은 두 target 모두에서 직접 학습되지 않았다. 따라서 U의 matched weak-label prompt는 **같은 학습 template를 그 U 영상의 weak label에 적용한 가상 학습 조건**이다. U 영상 자체에서 실제 사용된 학습 prompt가 아니다. nonmember target의 E 영상도 그 target에는 직접 학습되지 않았다. 환자의 E 영상과 U 영상이 같은 weak-label 문자열을 가졌다고 일괄 가정할 수도 없다.

실험자가 source manifest의 weak label을 보유한다는 사실이 영상만 가진 외부 감사자의 동일 정보 접근권을 뜻하지는 않는다. matched weak-label 결과는 별도 정보 가정을 가진 유리한 조건 대조다. 또한 weak-label template는 MoFit 원문의 의료 VLM 생성 caption과 다르다.

### 9.2 기존 fit8 generic/matched 실험은 이미 존재한다

실제 저장 결과는 `code_working/_reports/cvpr_u_pilot_v1_001/verification_20260914/basic_controls/` 아래 `protocol.json`, `report.json`, `cell_losses.json`, `image_scores.json`, `patient_scores.json`, `analysis.json`, `verification.json`이다. 조건 선택은 `code_working/u_patient_audit/run_basic_controls.py:220–224`에서 generic `PROMPT` 또는 `cache['train_prompts'][image_id]`를 조회한다.

범위는 원래 fit8 환자, 환자별 E2/U2, 두 step1000 target과 각 target의 adapter-off base 대조, generic/matched 두 조건이다. 동일 query seed 구성 `audit-noise/query/image_id/timestep/draw`, timesteps `[50,250,500,750,950]`, 각 2 noise draw를 사용했다. 1,280 target/base loss-pair cell, 128 image score, 64 조건별 patient score가 저장되어 있다. target/base를 각각 세면 2,560 UNet forward-example이며 backward는 0이다.

`analysis.json`의 사전 고정 방향 patient AUC는 다음과 같다. 각 칸은 `model_1 / model_2` 순서이며, `loss_mean`은 높은 값이 member가 되도록 **음의 target loss 평균**, `base_difference_mean`은 **base loss − target loss 평균**이다.

| 시나리오·점수 | generic | matched weak label |
|---|---:|---:|
| E, loss_mean | 0.5625 / 0.5625 | 0.5625 / 0.5625 |
| E, base_difference_mean | 0.2500 / 0.8125 | 0.2500 / 0.6875 |
| U, loss_mean | 0.7500 / 0.2500 | 0.7500 / 0.2500 |
| U, base_difference_mean | 0.6250 / 0.4375 | 0.6250 / 0.4375 |

이 비교에서 matched 조건이 위 U 점수들의 AUC를 높이지 않았다는 것이 관측 사실이다. AUC가 같다는 것은 loss 값·모든 반응·모든 공격이 같다는 뜻이 아니다. E의 model_2 base-difference AUC는 오히려 낮아졌다. 유리한 열만 고르거나 E 결과로 U 주장을 구제하지 않는다.

### 9.3 정밀도·mode·표본·접근권 한계

해당 `protocol.json`에는 **FP16 model/latent/conditioning, autocast FP16, loss reduction FP32, `model_mode='train'`, batch4, inference_mode, 가중치 동결**이 명시되어 있다. train mode라는 표시는 추가 학습을 했다는 뜻이 아니다. gradient checkpointing 설정은 켜져 있었으나 이 forward-only 실행에서 재계산은 하지 않았다. 원래 generic U batch1과 이 batch4의 저장 loss에는 수치 차이가 존재했고, 해당 대조의 member/nonmember pair 순위와 paired delta 부호는 유지됐다. 검산 기록도 bitwise batch equivalence를 선언하지 않았다.

각 target의 member/nonmember는 4명/4명이고, 두 target은 공통 seed와 묶음 참여 교환으로 만들어졌다. 독립 seed 반복이나 low-FPR 성능 검증이 아니다. 별도의 source-feature 학습·threshold fitting·conditioning optimization도 이 controls에는 없다.

후속 `code_working/_reports/cvpr_u_pilot_v1_001/verification_20260914/endpoint_precision_U8_v1/`의 FP32/batch1 endpoint 보정은 **generic 조건만** 평가했다. 기존 fit8 fixed-loss의 generic/matched 대조를 현재 **FP32/eval/batch1 CDI 26개 feature, SecMI 또는 의료 MoFit 전체 경로의 조건 대조**로 바꾸어 해석하면 안 된다. 현재 수행한 source 검사와 저장 결과에는 그 full-feature matched 대조가 없다.

따라서 “학습 조건과 generic 감사 조건의 불일치를 아직 한 번도 확인하지 않았다”는 표현은 부정확하다. 기존 작은 fixed-loss 대조는 존재한다. 동시에 **불일치가 현재 낮은 신호의 원인이라고 입증되지 않았고, 그 가능성이 모든 방법·정밀도·표본에서 배제된 것도 아니다.** null dropout이 없었다는 코드 사실만으로 null 반응이 학습 중 변하지 않았다거나 MoFit이 실패한다고 결론내릴 수 없다. 이번 기록에서는 새로운 GPU 대조 일정을 추가하지 않았다. 사용자의 기존 지시에 따라 2번의 필요한 조사를 계속하며 현재 full-U 고정 비교가 우선이다.
