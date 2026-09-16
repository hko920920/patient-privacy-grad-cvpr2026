# 공개 전용 의료 LoRA 학습·새 head 연결의 코드 감사

2026-09-16. 현재 큰 단계2·방향1의 **실행 전 설계 검토**다. 기존 코드, 학습 trace/report, manifest 메타데이터를 읽었다. GPU·새 학습·생성·효용 계산은 하지 않았고 기존 파일은 바꾸지 않았다. 아래 CODE는 `고한경_박사학위논문_작업본/code_working`이다.

**판정:** 기존 학습 연산을 보존한 fresh public-only LoRA를 구현할 수 있다. 다만 원 runner를 그대로 다른 CSV에 연결하는 것은 안전하지 않다. 출력·자동 resume·912장·batch 나눗셈 가정을 분리해야 한다. 이 준비 가능성은 공개 LoRA의 생성 품질이나 작은 patient-private head의 효용을 보장하지 않는다.

## 1. 실제 M1의 기준값

| 항목 | 확인한 값·근거 |
|---|---|
| 대상 | `u_patient_audit/train_coverage.py`의 coverage v2. 초기 50step smoke인 `train_pair.py`의 with-replacement 학습과 다름 |
| 자료 | M1 manifest 456환자 × 2장 = 912장. background256, fit40, selection20, calibration70, test70환자. 이 M1은 새 공개 모델로 재분류할 수 없음 |
| 초기화 | 같은 pinned SD2.1 FP16 artifact에서 `load_unet(None, training=True)`. seed260914, rank8/alpha8, Gaussian 초기화, to_q/to_k/to_v/to_out.0. base FP16, LoRA FP32, trainable1,659,904개 (`models.py:23–40`) |
| optimizer | AdamW lr1e-4, betas(.9,.999), eps1e-8, weight_decay.01, foreach=False, fused=False. LR scheduler 없음 (`train_coverage.py:36–39`) |
| 수치 | batch4, accumulation1, FP16 latent/hidden/noise, CUDA FP16 autocast, residual 차·제곱·mean은 FP32, GradScaler 초기1024, 전체 LoRA gradient norm cap1 (`:53–66`) |
| 학습 목적 | per-image weak-label conditioning, 모든 채널·공간·batch의 epsilon MSE. null-caption dropout 분기 없음. t는 step/ordinal hash mod1000; noise는 step별 CUDA Generator (`:53–62`) |
| sampler | epoch마다 새 deterministic randperm, 무복원. 원형은 N%4==0을 요구 (`:8–12`). 환자 sampler가 아니라 영상 sampler이며 당시에는 모두2장이라 균등도가 일치 |
| 실제 노출 | 1000 committed update, attempts1000, 4000 image exposures. 912장 모두4–5회, 평균4.3859649회. 250step checkpoint는 각1–2회 |
| 저장 | step250/1000. adapter뿐 아니라 Adam state, scaler, step/attempt, image exposure ledger, losses, contract 저장 (`train_pair.py:18–23`; `train_coverage.py:84–85`) |
| 실측 | M1 report의1000step segment496.7311173초, peak1.930654GiB. trace step250=129.5669453초, step251=130.1949501초, step1000=496.5899549초로 같은 연속 실행 기록. report 측정은 모델 로드 이후 시작하며 마지막 checkpoint 저장을 포함; 새 자료 준비/특징 캐시/생성 검증 전체 시간이 아님 |

원 근거: `CODE/_reports/cvpr_u_pilot_v1_001/training_coverage_v2/{protocol.json,model_1/report_1000.json,model_1/training_trace.jsonl,model_1/verification_0250.json,model_1/verification_1000.json}` 및 `cohort/model_1_train.csv`. step1000 SHA는 `ae789c32ce9ebaca3eb568fda968c28d5b9cbdf39682d7a498615b26cbb08435`다. verification은 상태·노출 무결성 확인이며 품질·DP 인증이 아니다.

## 2. 새 공개 runner의 최소 변경과 sampler 경계

1. **별도 manifest·출력·계약:** 원 `RUN/training_coverage_v2/model_1`을 재사용하지 않는다. 원 코드는 해당 출력 폴더의 최신 checkpoint를 자동 resume하고 `model_1_train.csv`를 읽는다(`train_coverage.py:30–47`). 새 runner는 공개 train manifest와 새 output을 명시적으로 받아야 하며 fresh 초기화에 사적 M1/M2 adapter·optimizer·scaler를 넣지 않는다. 재시작도 같은 새 계약의 checkpoint만 허용한다.
2. **연산 보존:** 위 optimizer·loss·noise·t·AMP·clip·overflow retry 규칙은 유지하고, 자료 선택/출력/노출 기반 checkpoint만 새로 고정한다. 공개 LoRA는 비DP 학습이며 norm clip1 자체를 환자DP로 부르지 않는다. 이후 private 단계의 보장은 공개 backbone과 선택 과정이 보호 대상 자료와 분리됐다는 조건을 필요로 한다.
3. **원 sampler는 임의 N용 연속 stream이 아님:** `assert n%4==0` 뒤 한 epoch의 `[offset:offset+4]`만 읽는다. N이4의 배수가 아니면 그대로 쓸 수 없다. 새 연속 full-batch 정의는 전역 위치 `k=4*step+j`, `epoch=k//N`, `offset=k%N`에 대해 그 epoch의 deterministic permutation[offset]을 읽는 것이다. 경계 batch는 두 permutation을 이어 붙이고 tail을 버리거나 가짜 padding하지 않는다. 경계에서 우연히 같은 영상이 두 번 들어갈 수 있는 것은 각 epoch 무복원과 모순되지 않는다.
4. **노출량 고정:** parent가 제안한 공개 train640/selection128/confirmation128/reserve9환자와 E4/E8 두 checkpoint는 새 명세이며, 이 메모에서 해당 새 manifest의 실제 image 수를 검산한 것은 아니다. 학습 영상 수를 N이라 두면 B4에서 `S_E=ceil(E*N/4)`; E4/E8은 각각 **S4=N, S8=2N**이라 checkpoint에서 매 영상이 정확히4/8회 노출된다. 환자별 영상 수가 다르면 image-uniform은 patient-uniform이 아니며 이를 명시한다. 공개 비DP 사전적응에 image-uniform을 쓰는 것 자체는 오류가 아니다.
5. **크기 비교:** 공개32명×2장에1000step을 복사하면62.5회/영상이다. M1의4.386회에 가까운 것은 약70step(4.375회)이며, 이것은 노출 대응일 뿐 품질 대응이 아니다. 새640환자 구성에32명 기준 수치를 적용하지 않는다.
6. **검증 자료:** 최초 base/adapter 초기 상태와 train manifest hash, 단계별 batch ID·t/noise seed, attempted/committed updates, overflow, 실제 image/patient exposures, optimizer/scaler 상태, checkpoint hash를 저장한다. 공개 train/selection/confirmation과 현재 private train/dev/cal/test의 환자 및 가능한 image 중복을 사전 확인한다. 새 자료를 encode해야 하면 그 VAE/text 처리 비용도 전체 비용에 포함한다.

시간의 현재 근거는 기존 약0.4967초/update뿐이다. 같은 GPU/batch/연산이라는 가정에서 E8 학습시간의 거친 기준은 `2*N*0.4967초`이며 새 실측값이 아니다. 로딩·저장·validation 생성·encoding·코드 검토 시간은 별도다. 새 image 수가 확정되기 전에 전체 시간이나 품질을 단정하지 않는다.

## 3. P는 재사용 가능하고, 새로 추출해야 하는 것은 다르다

`frozen_residual_head/residual_math.py:61–74`의 P는 `default_rng(seed).standard_normal((C,k))`의 reduced QR와 R 대각 부호 보정만으로 만들어진다. 데이터·특징·backbone weight를 읽지 않는다. `run_capacity.py:80`은 C320/k15/seed26091601을 고정했다. 따라서 **새 LoRA에서 기존 P를 재사용하는 것은 가능**하다. P 재사용과 기존 W 이식은 다른 문제다.

| 자료 | 공개 LoRA만 추가하고 VAE/text/scheduler를 그대로 둔 경우 |
|---|---|
| 저장 P320×15 | 그대로 고정 재사용 가능. 같은320채널 conv_out 입력 tap·flatten 순서·FP32 h@P·상수1/basis-major 순서를 유지. 재QR보다 저장 P 자체의 hash를 결속하는 편이 정확함. 새 backbone에서 통계적으로 적합하다는 보장은 없지만 데이터 의존 변환은 아님 |
| 1000 alpha·logSNR basis | scheduler가 같으면 재사용 가능. scheduler가 바뀌면 공개 설정에서 alpha/bounds를 다시 정의. 30개 sampling timestep 범위로 정규화를 다시 잡지 않음 |
| VAE latent·text hidden 캐시 | 해당 이미지/전처리/FP16 VAE posterior mode/scaling 및 text encoder/tokenizer/prompt가 같으면 재사용 가능. UNet LoRA 변경만으로 재인코딩할 이유는 없음. 새 이미지·새 encoder면 해당 부분을 새로 encode. 공개 runner가 읽는 ID/prompt를 whitelist로 제한하고 새 역할별 cache manifest를 결속 |
| DINO/RAD-DINO 등 독립 encoder 특징 | encoder·이미지·전처리가 같다면 UNet 교체만으로 수치가 바뀌지는 않음. 역할 누출/참조 집합의 중복을 별도로 확인; 기존 quality 역할을 새 훈련에 쓰면 독립 quality 표본으로 다시 세지 않음 |
| 새 UNet h320/h16·base epsilon·residual | **재추출 필수.** frozen backbone이 달라져 값이 바뀜. 같은 noisy inputs/seed를 재사용하면 backbone 차이의 비교는 가능하지만 결과 특징을 재사용할 수는 없음 |
| A/B/Q·public/private/pooled W·DP 보정 | **재계산/재학습 필수.** 새 residual=`epsilon−새 base_prediction`과 새 phi로 계산. 기존 clip/scales/floor/optimizer 보정 값의 성능 적합성을 그대로 가정하지 않음 |
| witness·zero-head·sampling 패킷 | **새 backbone에서 새로 확인.** 과거 witness는 원형 구현의 근거로 보존하지만 새 checkpoint 수치의 검산을 대신하지 않음 |

근거: `build_cache.py:19–32,38–48,81–90`; `run_capacity.py:170–179,193–199,209–238,307–315`; `residual_math.py:3–7,77–103,106–128`. 새 backbone에서도 P가 같다는 이유로 h나 최적 W가 같다고 추론하지 않는다. 반대로 P를 새로 뽑는 것을 필수라고 하면 불필요한 무작위 변인이 늘어난다.

## 4. 공개 validation과 head 운용 조건

- 공개 backbone 선택은 공개 selection에서만 하고 private80/dev40의 denoising loss·영상으로 checkpoint/훈련량/CFG를 고르지 않는다. parent의 E4/E8 두 후보·가장 이른 통과 선택을 사용한다면 통과 기준,4prompt×4seed,reference mixture,실패 처리와 동률 규칙을 **생성 전** 고정한다. confirmation16은 선택 이후 한 번만 확인하며, 실패 시 같은 confirmation을 보고 다른 checkpoint를 다시 고르면 그 자료는 더 이상 독립 확인이 아니다.
- 현재 head 학습은 weak-label **conditional** prediction이며 null 학습 분기가 없다. primary validation은 최종 head 운용과 같은 FP32 승격·cache conditioning·256/DDIM30·CFG1을 포함해야 한다. CFG7.5 이미지가 더 좋아도 CFG1 head의 적용 성공을 대신하지 않는다. CFG7.5를 보조 기준으로 표시하는 것은 가능하지만 선택 목적/운용 조건을 구분한다.
- CFG>1은 null/conditional 두 branch의 조합이다. 조건부로만 학습한 W를 null branch에도 바로 적용하거나, 조건부 correction을 CFG 밖에 단순 가산하면 검증한 모델과 달라질 수 있다. CFG를 primary로 바꾸려면 head 적용 위치와 유효한 학습/추출 조건을 별도 명세해야 한다.
- 공개 backbone의 최소 형태 확인은 필요한 적용성 점검이며 의학적 품질·질환 정확성 검증이 아니다. 이전의 잘 나온2장 재현은 진단으로 보존하고 새 selection/confirmation의 성공률로 세지 않는다. backbone을 동결한 뒤 새 head를 만들고 **공개 backbone 단독 및 public-only head**를 기준선에 남겨 private 추가 효용을 분리한다.

이 메모는 새 공개 학습을 실행하거나 통과를 예측하지 않는다. 현재 확정할 수 있는 것은 구현 변경 범위, 재사용 가능한 데이터 독립 연산, 자료/학습량/조건 일치를 검증할 방법이다.
