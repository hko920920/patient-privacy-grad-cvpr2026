# MoFit 원형 기반 환자 비교 명세를 위한 근거

작성: 2026-09-14. 현재 **2번: 가까운 기존 방법의 실패 조건·원인 확인**. 이번 질문은 “MoFit 원형을 유지한 채 E/U 환자 비교를 어떻게 구현하고 정상 작동 여부를 확인할 것인가”이다. 이 문서는 비교 명세와 미확정 실행 계약을 구별한다. 새 설계의 타당성이나 2번 완료를 뜻하지 않는다. 이번 작업에서는 문서 하나만 작성했으며 GPU·학습·분석 main을 실행하지 않았다.

## 1. 확인한 출처와 현재 비교 단위

- 논문: `pdfs/X12.pdf`, ICLR 2026 MoFit. 방법 pp.3–7, ablation pp.9–10, 구현 pp.15–16, LoRA·비용 pp.19–21, 의료 p.23. `texts/X12.txt`와 페이지별 `texts/X12.json`으로 원문 대조.
- 저자 코드: `code_sources/X12`, repository `JoonsungJeon/MoFit`, commit `91e4b5edc153bac84b0b4209f70d1b2b94e653b2` (`code_acquisition.json:11851–11855`). 소스 다운로드 상태이며 공격 실행 재현은 아님.
- 실제 실행 entry는 `COCO/MoFit_COCO.py/.sh`, `Pokemon/MoFit_Pokemon.py/.sh`, `Eval/calculate_pred.py` 및 두 `mia_th_*.py`이다. 공개 main 코드와 README에서 **LoRA·ROCO·Prompt2MedImage 전용 경로는 발견되지 않았다**. 논문 부록 성능과 공개 재현 코드 범위를 혼동하지 않는다.
- 원형은 이미지 하나의 membership을 판단한다(Eq.3, p.3). 환자별 사진 두 장을 독립 공격한 뒤 pooling하는 비교를 **MoFit-derived patient pooling**으로 명명한다. 한 사진에서 최적화하고 다른 사진으로 평가하는 cross-photo 방식은 별도 변형이다. 현재 R과 8D 직접 fitting은 MoFit 원형 실행으로 세지 않는다.
- 현재 cohort metadata에는 fit 80명, selection 40명, calibration 140명, test 140명이 있다(`code_working/_reports/cvpr_u_pilot_v1_001/cohort/evaluation_images.csv`). 앞서 실제 최적화 진단에 사용한 fit 8명과 fit 전체 80명을 구별한다. 여기서 새 환자 query나 추가 역할 사용을 승인·실행한 것은 아니다.

## 2. 원형에서 보존해야 하는 세 단계

| 단계 | 원형 연산·옵션 | 실제 코드 근거 |
|---|---|---|
| 입력 | RGB 512, bilinear Resize+CenterCrop, tensor를 [-1,1]로 정규화. FP32 pipeline. VLM caption embedding 초기화 | `COCO/MoFit_COCO.py:38–65,103–122,155`; Pokemon `:35–61` |
| 1: surrogate | x에 U[-.3,.3] 초기 perturbation. null conditioning에서 고정 ε·t의 MSE 최소화. 이미지 픽셀 gradient-sign 1,000회, t=140. step_i=.15−(.15−.0015)i/1000. 매회 이미지 [-1,1] clipping | COCO `:158–234`; shell `:34–52` |
| 2: embedding | surrogate의 conditional loss를 같은 ε·t로 최소화. 초기 VLM embedding의 **전체 token×hidden 행렬**을 Adam(lr=.06)으로 최적화. COCO/Flickr 300회, Pokemon 200회, SD1.5 1,000회 | COCO `:273–339`; Pokemon `:247–311`; 논문 p.15 |
| 3: 원본 점수 | surrogate 대신 원본 x를 사용. h=L(x,φ*)−L(x,φ_null), member-high. raw auxiliary로 u=L(x,φ_null), v=L(x,φ_VLM)−u. Eq.9: γR(h)+(1−γ)R(−aux) | pp.6–7 Eq.7–9; COCO `:342–355`; `Eval/calculate_pred.py:726–807` |

확정 원형 shell 옵션은 `--type Uncond --eps 0.3 --step_size 0.15 --OptimIter 1000 --anchor 14 --until 13 --adv_rnd 1 --emb_rnd 1 --lr 0.06`이며, COCO `--iters 300 --init blip2`, Pokemon `--iters 200 --init clip_interrogator`이다. Python argparse 기본값(예: anchor20/until10/100회/50회)만 실행하면 이 원형 shell과 다르다(COCO `:375–395`). gen_seed 기본값은 0(`:379`).

δ=.3은 **초기 perturbation 범위**다. COCO는 원본 주변 L∞ ball로 다시 투영하지 않고 [-1,1]만 clipping한다. Pokemon `:216`의 `min(max(adv,adv−eps),adv+eps)`는 eps>0이면 항등 연산이다. 원본±.3 투영을 새로 추가하고 “원형 그대로”라고 부르면 안 된다.

Stage 1은 VAE posterior **mean**을 사용하지만(COCO `:206–207`), stage 2는 surrogate와 원본 각각 posterior **sample**을 한 번 뽑고 고정한다(`:279–284`). text embedding만의 8D 제한, 원본 이미지 직접-fit, stage1 생략, 매회 다른 diffusion noise, 마지막 surrogate의 loss 자체를 공격 점수로 사용하는 것은 모두 원형과 다른 방법이다.

## 3. 현재 환자 모델에 옮길 명세

### 모델·이미지·dtype

1. 현재 target은 고정 `Manojb/stable-diffusion-2-1-base` revision `0094d483a120f3f33dafbd187ea4aa60d10de75c` + rank8 LoRA step1000 두 개다. `code_working/u_patient_audit/common.py:11–13`, `models.py:22–46`, `train_coverage.py:14–26`. 새 target 학습을 요구하지 않는다.
2. 실제 frozen scheduler config는 **epsilon**, T=1000이다. SD2.1이라는 이름으로 v-prediction을 추정하지 않는다. `.cache/huggingface/hub/models--Manojb--stable-diffusion-2-1-base/snapshots/0094d483a120f3f33dafbd187ea4aa60d10de75c/scheduler/scheduler_config.json:8–9`. text encoder config의 hidden_size=1024, max_position_embeddings=77을 직접 확인했다. 현재 full embedding 최적화 변수는 **77×1024=78,848**개이며 SD1.x 원형의 77×768과 구조적으로 대응한다.
3. 현 연구 입력은 기존 full-field grayscale→RGB P256 전처리를 유지한다(`build_cache.py:43–48`, `data_pipeline/nih_cxr14_model_input.py:223–269`). 원형 P512→P256 해상도 적응은 명시한다. 따라서 논문 SD1.4/P512 숫자의 직접 복제라고 부르지 않는다.
4. 공격 중 UNet/LoRA/VAE/text encoder weight는 고정한다. adapter enable 이후 requires_grad=False를 재적용하고 값·버전·grad를 확인한다. 입력 픽셀 또는 full embedding에 대한 autograd만 허용한다. `loss.backward`는 **공격 입력 최적화**이며 target 재학습이 아니다.
5. 본문과 가까운 수치 계약은 UNet·VAE 계산을 FP32로 수행하는 것이다. 현 target의 FP16-variant 로드 후 FP32 promotion은 기존 frozen target weight 값을 보존하지만, 그것을 원저자 checkpoint의 원래 FP32 weight와 동일하다고 하지 않는다. 원형 pixel gradient가 필요하므로 기존 FP16 mode latent cache만으로 stage1을 대체할 수 없다.
6. 모형 입출력 해상도·VAE sampling을 원형에 맞춘 **수치 검증용 baseline 경로**와, 기존 학습 latent(mode cache)에 맞춘 별도 mode-adapted 진단은 이름·계약을 구분해야 한다. 본문에 가까운 경로를 원형 우선 비교로 삼되 8GB GPU에서 가능 여부를 먼저 측정한다. FP16로 몰래 낮추거나 stage1을 제거하지 않는다.

### Caption·noise·보조자료

- primary 원형 적용의 초기 caption은 **후보 영상에서 생성한 의료 VLM caption**이다. 논문 의료 실험은 ROCO BLIP `Siddartha01/blip-medical-captioning-roco`를 사용(p.23 footnote12). 모델 revision·출력 caption·생성 옵션을 별도 hash로 고정해야 한다. 현재 해당 revision과 의료 caption cache가 확보됐다는 증거는 확인하지 못했다.
- 기존 generic caption(`a frontal chest radiograph`) 또는 weak-label training prompt를 바로 사용하면 각기 **generic-init 변형** 또는 **추가 label 정보 접근 변형**이다. 둘 중 하나를 원형 VLM baseline으로 대체해 약한 결과를 MoFit 실패라고 평가하지 않는다. 비교 공격 모두 동일 후보 이미지·동일 생성 caption·동일 보조 label 접근권을 받아야 한다.
- 현재 training은 cache.train_prompts를 항상 사용하며 null-caption dropout 분기가 없다(`train_coverage.py:53–62`). 빈 문자열 hidden이 cache에 존재한다는 사실은 null conditioning으로 추가 학습했다는 뜻이 아니다(`build_cache.py:24`). MoFit p.5의 “member가 unconditional loss도 낮추도록 학습됐다”는 설명이 현재 LoRA에 얼마나 전달되는지는 미확인이다. 무효·불가능 원인으로 확정하지 않는다.
- **논문 수식 정합 경로:** 이미지마다 하나의 ε와 t=140을 고정하고 stage1·stage2·최종 h/u/v 계산에 재사용. 원본 조건들끼리도 같은 z0·ε·t를 사용. target1/2 사이 같은 이미지는 공통 noise draw를 사용하면 paired Monte Carlo 잡음을 줄일 수 있으나 정책을 명시한다. 평가에 target training noise나 membership label을 입력하지 않는다.
- 원저자 `rnd_noise_1.npy`는 전체 이미지에 공통인 4×64×64 텐서다. P256의 4×32×32에 그대로 넣을 수 없다. shape에 맞는 사전 고정 Gaussian draw를 새로 생성·저장하는 adaptation이 필요하다. seed, generator device, dtype, draw hash를 고정하고 결과에 따른 재추출을 금지한다. 원형 literal 재현에서는 upstream npy를 확보한다.
- **코드·본문 차이:** `Eval/calculate_pred.py:774–777`의 embedding 경로는 고정 rnd_noise_1을 사용하지만, VLM 경로 `:687`는 `:838`에서 별도 생성한 global Noise를 사용한다. 본문 Eq.7–8은 같은 hat-ε로 표기한다. 따라서 공개 artifact literal 재계산은 원래 두 stream을 보존하고, 환자 baseline primary는 위의 본문 정합 동일-noise 계약을 명시한다. 이 선택을 숨겨서 완전한 코드 재현이라고 주장하지 않는다.

### 환자 pooling·학습 역할

- 각 환자 E2장 또는 U2장에 대해 **각 이미지를 독립적으로 세 단계 공격**한다. target마다 별도 φ*이며 다른 환자의 최적화 결과를 공유하지 않는다. 그 뒤 이미지 h/u/v 또는 최종 image score를 mean/max pooling한다. m=2에서 top-1=max, top-2 mean=mean, median=mean이므로 중복 baseline을 늘리지 않는다.
- 최소 비교는 (a) raw h의 mean/max, (b) 원형 auxiliary 결합 image score의 mean/max, (c) 동일 h/u/v 정보만 쓰는 학습형 환자 집계이다. (b)의 γ·robust scaling과 (c)의 가중치가 추가 정보이므로 같은 fit 환자 label 예산을 모든 해당 비교군에 부여한다. 모델별 fitting은 허용 여부를 사전에 일치시킨다. patient-disjoint fit/selection 분리, 환자 단위 threshold 보정, 같은 E/U 보유영상 수가 필요하다.
- 논문 auxiliary 선택은 Pokemon/SD1.5에서 Luncond, COCO/Flickr에서 LVLM이며 γ∈{0,.05,…,1}, τ는 ASR 최대화(p.15). 의료 전용 auxiliary의 공개 코드 근거는 없다. 자연의료 영상에 COCO의 LVLM 결합을 적용하는 기본안을 명시하고, Luncond를 포함한 보조 후보 선택은 fit 역할에서만 수행한다. selection에서 γ·부호·aux를 고르지 않는다.
- `Eval/mia_th_COCO.py:216–288` 공개 script는 loaded 전체에서 scaler·γ·τ를 선택해 같은 자료 성능을 보고한다. 원저자 artifact를 설명할 literal 실행과 우리 환자 평가의 분리된 fitting은 구분한다. 원형 script의 평가 누출 가능성을 그대로 복제하지 않는다. `:140`의 FPR≥1% 첫 점 규칙 역시 운영 FPR≤1% 보정으로 옮길 때 명시해야 한다.
- fit 역할의 몇 명을 query하고 calibration/test를 언제 사용하는지는 root의 실행 계약에서 정할 항목이다. 이 명세는 80명 fitting·140명 calibration/test 실행을 자동 시작하지 않는다. selection40은 이미 관측된 개발 자료이며 untouched test로 표현하지 않는다.
- target1/2 응답을 동시에 사용하는 paired score는 analyst-only 진단이다. 독립 auditor의 attack feature로 넣지 않는다. cross-photo variant는 예를 들어 `MoFit-derived cross-photo transfer`로 분리하고 원형 patient pooling과 동일 자료·계산량에서 별도로 비교해야 한다.

## 4. 계산량: 원형 literal과 수학적으로 정리한 실행을 분리

F는 **UNet forward-example 수**, B는 backward/autograd 호출 수로 정의하며 VAE와 text encoder는 별도 센다. batch2 호출을 F=1로 세지 않는다. gradient checkpointing 재계산은 추가 forward로 기록한다.

| P256 환자 적용에서 쓸 기본 N1=1000/N2=300 | UNet F | B | VAE encoder F/B |
|---|---:|---:|---:|
| 저자 COCO stage1 literal: 매회 CFG batch2를 만든 뒤 uncond branch만 loss에 사용 | 2000 | 1000 | 1000 / 1000 |
| 저자 COCO stage2 literal: 최적화1F + 매회 원본 uncond/φ*/VLM monitor3F | 1200 | 300 | 시작 시 surrogate·원본 2 / 0 |
| 위 두 단계 합계(외부 최종 evaluation 별도) | **3200** | **1300** | **1002 / 1000** |
| 수학동일성 확인 후 중복 branch·매회 monitor를 생략한 핵심 연산+마지막 h/u/v3F | **1303** | **1300** | **1002 / 1000** |

근거: COCO `:214`의 mtcnp_adv 호출, 수정 pipeline `diffusers_0_18_2/pipelines/stable_diffusion/pipeline_stable_diffusion.py:1060–1078`의 2배 batch; COCO `:331–352`의 iteration당 4회 UNet. 임베딩 단계 optimizer는 embedding만 업데이트하지만 원본 코드는 UNet parameters를 명시적으로 freeze하지 않아 불필요 parameter gradient 계산 가능성이 있다. 안전한 frozen-target implementation과 bitwise 같다고 자동 가정하지 말고 입력-gradient/endpoint 동등성을 확인한다.

중복 branch/monitor 제거는 목적함수·optimizer update를 보존하는 실행 최적화이다. 다만 batch2→batch1의 부동소수 차이와 checkpointing 재계산 영향을 기록해야 한다. 연산 상한이 부족하다고 무조건 8D fitting이나 stage1 생략으로 대체하지 않는다.

환자당 2장, target당 한 E 또는 U 조건이면 위 비용의 2배. E2+U2와 target2개는 8 image-target 공격이다. public VLM 1회/image, text encoding 비용, CUDA loading·VAE encoder 비용, γ/집계 fitting 비용을 추가 보고한다. 같은 image-target raw h/u/v를 여러 pooling에 재사용하므로 mean/max마다 재공격하지 않는다.

저자 보고 시간은 RTX4090/P512에서 이미지당 약 7–9분(pp.10,15), COCO full stage1 358.96초·stage2 116.38초(p.20 Table9). 현재 RTX3070 8GB/P256/1024-hidden에서는 **실측 시간과 peak memory가 없으며 이전 R 또는 PFAMI runtime으로 추정할 수 없다**. full loss/gradient를 보존하는 한 이미지 benchmark 전에는 총 GPU시간을 확정하지 않는다. 저자 early-stop22.32초는 stage1만의 값이며 stage2가 남는다(pp.20–21). 임의 early-stop을 기본으로 사용하면 원형 full 기준과 별도 조건이다.

## 5. 정상 작동 확인과 다음 실행 하나

### 확인된 양성 조건과 비교 한계

- 원저자 positive 기준: Pokemon416 train images/15k training steps, MoFit AUC97.30·TPR@1%50.48; COCO2500/150k steps, AUC94.17·TPR47.00(pp.7,16). 현재 912 images, 4–5회 노출, rank8 step1000 LoRA와 다르다.
- 같은 논문 LoRA 표는 MoFit54.35/0, SecMI53.50/1, PFAMI77.50/1(p.10). 의료 ROCO는 MoFit54.44/2(p.23). 이들은 현 LoRA의 음성을 구현 버그나 U 불가능으로 바로 확정할 수 없음을 보여준다. PFAMI77.50의 AUC를 유용한 low-FPR 성과로 바꾸어 말하지 않는다.
- 공개 `Results/COCO/COCO_{emb,blip}_500images_{train,test}_t_[140].txt` 네 파일에는 각각 **500개 수치 행**이 현재 존재한다. local inference 없이 score/scaler/γ/방향/AUC 계산의 literal fixture 확인이 가능하다. 단 embedding 결과 header `:1`에는 `OptimIter1000_iters1000`가 남아 있어 논문 COCO300회와 provenance가 불일치/미확정이다. 예제 npy/노이즈·caption은 upstream tree에 있지만 현재 로컬 다운로드와 실제 checkpoint 확보는 별도로 확인해야 한다.
- fixture 재계산 성공은 **저장된 결과의 수치 해석 재현**이다. 이미지→surrogate→embedding inference 또는 Table2 재실험 성공이 아니다. 새로운 adapter의 기능 확인에는 실제 저자 positive checkpoint와 matching images/captions에서 loss trajectory·gradient·endpoint를 확인해야 한다. 개별 이미지 성공만으로 AUC 재현이라고 하지 않는다.

### 추천하는 다음 한 실행: 공개 COCO scalar fixture의 CPU 대조

**질문:** 지금 확보한 원저자 네 score 파일과 evaluator가 제안한 patient baseline의 점수 방향·aux 결합·분리된 평가 구현의 신뢰할 출발점인가?

범위는 위 네 파일을 read-only로 읽고 500/500 row alignment, source column `condNull_dif=Luncond−Lcond` 부호(`Eval/calculate_pred.py:804`), γ21값과 source threshold2000점의 literal 결과, 독립 rank-AUC 및 FPR 경계 차이를 비교하는 것이다. 저자 header의 iters1000/본문300 불일치도 결과에 남긴다. source-result alignment에 patient/image ID가 없으므로 row-order 보존 이상을 증명했다고 하지 않는다.

새 GPU F/B=0, 새 환자 query=0, fitting은 **저자 공개 fixture의 원래 검색 절차 재현**에만 한정한다. 우리 환자 γ·threshold·부호를 선택하지 않는다. 전체 구현·검산 5–15분은 작업시간 예상이며 CPU 실행시간은 아직 측정하지 않았다. 결과가 약하거나 논문 수치와 다르더라도 원인 확인을 수행하고 자동으로 R 수정·새 학습을 시작하지 않는다.

그 다음 필요한 실질 GPU 단계는 별도 계약의 한 이미지/한 target 원형 full-gradient benchmark이다. 후보 영상·caption·노이즈·dtype·1000+300회·모든 weights freeze를 먼저 고정하고, 8GB에서 memory·runtime·숫자 안정성을 확인한다. E 또는 U 소수로 공격 효능을 확정하지 않는다. 메모리 실패는 구현 가능성 자료이며 연구 목표나 원형의 성능 반증이 아니다.

## 6. 이번에 끝난 범위와 남은 항목

**확인:** 원형 세 연산·실제 옵션·noise/VAE/code-paper 차이·현재 모델의 epsilon/77×1024 구조·공개 scalar fixture 존재·LoRA/의료 전용 공개 main 부재·환자 pooling의 정의와 비용 단위.

**미확인:** 현재 모델에서 MoFit 원형의 작동 성능, 의료 VLM caption의 품질·pinned revision, 8GB full-gradient 실행 가능성과 시간, public checkpoint inference 재현, patient γ·학습형 pooling의 안정성, U 실패의 원인.

**다음:** 공개 scalar fixture CPU 대조 하나를 먼저 수행할 수 있다. root가 전체 CDI/집계 명세와 함께 우선순위를 정한다. 이번 명세 작성은 2번의 일부 완료이며 3번 새 설계나 4번 검증에 진입했다는 뜻이 아니다.
