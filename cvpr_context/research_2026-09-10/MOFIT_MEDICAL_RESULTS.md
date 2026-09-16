# MoFit 의료 모델 실제 실행·미분 검증

2026-09-15. 현재 [고정 연구틀](RESEARCH_FRAMEWORK.md)의 **2번: 가까운 기존 방법의 실패 조건·원인 확인**이다.

**RTX3070에서 같은 환자의 E2/U2 영상4장을 두 target에 적용한 MoFit 핵심 연산8건과 두 장 집계를 완료했다.** 각 건은 pixel1000회+전체 embedding300회이며 원시 저장 산술 검산을 통과했다. 최초두실행은923.982/924.053초, 추가4건은1,877.598초였고, full8건 실행기 합은3,725.633초(62분6초)다. 처음 pixel 미분 차이21–25%는 작은 간격 검증을 추가해 국소 일관성을 확인한 뒤 진행했다.

**가장 구체적인 관측은 U의 max 집계가 한 사진의 큰 양의 모델 간 차이를 버린 사례다.** U 두 장의 h 차이는+.0097275/−.0005336이고, mean에서는+.0045969, max에서는−.0005336이었다. 두 target 모두 두 번째 U를 최댓값으로 골랐기 때문이다. 그러나 기존 mean으로도 달라지는 결과이며 환자1명·두전체학습군의대조다. 강한 기존 대안에도 남는 MIA 실패나 새 해결책의 필요성은 아직 입증하지 못했다.

## 1. 어떤 MoFit을 실행했는가

정확한 명칭은 **공개 COCO 1000+300 스케줄을 NIH P256/SD2.1에 적용한 medical-BLIP 초기화 MoFit adaptation**이다. 공개 코드 commit은 91e4b5edc153bac84b0b4209f70d1b2b94e653b2이다. 논문 의료 실험의 ROCO/Prompt2MedImage와 현재 NIH LoRA target은 다르다. [원문 pp.5–7,15,23](pdfs/X12.pdf), [공개 COCO 코드](code_sources/X12/COCO/MoFit_COCO.py), [사전 조건 검토](spec_sources/mofit_medical_readiness_20260915.md)

| 항목 | 고정한 조건 |
|---|---|
| Target | 기존 model1/model2 step1000, SD2.1 계열 epsilon prediction |
| 영상 | fit 환자14393, E002/E006와U001/U004 총4장; 기존 NIH full-field grayscale→RGB P256 |
| Pixel 단계 | 초기 U[-.3,.3] perturbation, 초기 clamp 없음, VAE posterior mean×.18215, t140 고정 noise, null loss sign update1000회, 매 update 후[-1,1] clamp |
| Pixel step | .15−(.15−.0015)×i/1000. 원영상 중심의 별도 perturbation ball로 변경하지 않음 |
| Embedding 단계 | 최종 surrogate와 원영상의 posterior를 각각 한 번 sample하여 고정; 전체77×1024 좌표를 Adam(lr=.06)으로300회 갱신 |
| Literal 연산 | Pixel의 CFG batch2와 embedding 매회 원영상 null/현재/초기 embedding monitor 유지 |
| 정밀도 | FP32 UNet/VAE 연산. Base weight artifact는 FP16→FP32; checkpoint LoRA는 FP32 값 유지 |
| Caption | Siddartha01/blip-medical-captioning-roco, revision f8a8378a9013e8e6330a930dc0b405798d326c56; greedy caption 및 fresh FP32 CLIP encoding |
| 학습·변수 | UNet/LoRA/VAE weight는 고정. Pixel 및 padding을 포함한 전체78,848개 embedding 좌표만 최적화 |

난수는 초기 pixel perturbation, diffusion noise, surrogate/original posterior noise를 **실제 CPU FP32 draw tensor로 각각 저장**하고 두 target에 재사용했다. 원 저자 CUDA RNG나 외부 noise 파일과 bitwise 동일한 재현은 아니다. 준비된 E2/U2 네 영상의 의료 BLIP caption은 모두 동일했다. 임상 정확성은 검증하지 않았으며 결과에 맞춰 수동 교정하지 않았다. Caption은 target 학습의 weak-label prompt와도 구별한다. [입력 준비 protocol](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260915/mofit_medical_inputs_v1/preparation_protocol.json), [입력 summary](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260915/mofit_medical_inputs_v1/summary.json)

### 이번 점수 범위: h/u/v

원영상 x에서 `u=L(x, null)`, `h=L(x, 최적화 embedding)−u`, `v=L(x, 초기 VLM embedding)−u`를 계산한다. 최적화 embedding은 surrogate에서 구한 것이다. **최종 h는 surrogate fitting loss 자체가 아니다.**

논문의 결합은 `s=γ·R(h)+(1−γ)·R(−aux)`이며 aux는 u 또는 v, R은 median/IQR scaling이다. 이번에는 **core h/u/v만 계산**했으며 의료 gamma·auxiliary·scaler·threshold를 선택하지 않았다. 공개 COCO replay의 gamma=.55는 의료 최적값이 아니다. 의료 Appendix에는 해당 의료 gamma가 명시되지 않았다. [원문 식7–9](pdfs/X12.pdf#page=6), [공개 scalar replay](replays/mofit_coco_fixture_20260914/report.md)

이번 kernel은 최적화 loop의 최종 원영상 monitor를 사용한다. 공개 별도 평가기가 fresh posterior를 sample하는 경로까지 bitwise 재현한 것은 아니다. 공개 평가기의 낮은 점수 방향과 논문의 높은 membership 점수 방향은 전체 부호 변환 관계이며, 결과를 보고 부호를 고르지 않는다.

## 2. 처음 미분 간극과 추가 검증

Model1 E002에서 pixel2회+embedding2회, 각 첫 지점의 중심차분과 CFG1 대조를 먼저 실행했다. 전체24.052448초, 계측6.020028초, 독립 CPU 검산4.357276초였다. 주 packet125개 및 CFG16개 산술 항목의 PASS는 **미분 오차까지 자동 통과시킨 판정이 아니다.** [Benchmark 실행](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260915/mofit_medical_benchmark_v1/execution.json), [검산](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260915/mofit_medical_benchmark_v1/verification.json)

Pixel AD 방향미분은 **−.0773342279997**였다. 처음 간격 .01/.005의 저장 FP32 loss FD는 AD와20.853%/25.283% 차이가 났다. 따라서 full 실행을 보류하고, 같은 pixel 지점·gradient·방향(seed1731 Rademacher RMS1)·noise·hidden·target을 고정해 다섯 작은 간격을 추가했다. 추가 backward나 공격 파라미터 튜닝은 없었다.

아래 η는 FD 간격으로 MoFit 점수 h와 다르다. 상대차이는 `|FD−AD| / max(|FD|, |AD|)`이다.

| η | 저장 FP32 loss FD | 저장 예측의 FP64 MSE로 재계산한 FD | FP64-MSE FD 상대차이 | 상쇄 heuristic |
|---:|---:|---:|---:|---:|
| .01 | −.0977098942 | −.0977085254 | 20.852118% | 1.68194e−5 |
| .005 | −.1035034657 | −.1035052642 | 25.284739% | 3.36241e−5 |
| .002 | −.0758096576 | −.0758089763 | 1.972285% | 8.39882e−5 |
| .001 | −.0765621662 | −.0765576751 | 1.004152% | 1.67958e−4 |
| .0005 | −.0768899918 | −.0769292236 | .523706% | 3.35890e−4 |
| .00025 | −.0771880150 | −.0771574961 | .228530% | 6.71749e−4 |
| .0001 | −.0773370266 | −.0773927838 | .075661% | 1.67932e−3 |

**FP64는 저장 FP32 neural prediction의 residual/MSE를 CPU double로 다시 합산한 것이다. FP64 neural inference가 아니다.** 마지막 간격의 FP32 loss만 보면 차이가 .0036%까지 줄지만 그 값만 골라 통과 근거로 삼지 않는다. 일곱 간격 전체를 보존했다. [추가 결과](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260915/mofit_pixel_fd_refinement_v1/results.json), [독립 FP64 검산](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260915/mofit_pixel_fd_refinement_v1/verification.json)

추가 다섯 간격에서는 AD에 가까워지는 경향이 관측된다. 196,608차원의 RMS1 방향에서 η=.01의 총 L2 이동은 약4.434여서 픽셀당 작은 간격이라는 이유로 국소성이 보장되지 않는다. 이 결과는 해당 지점·방향의 국소 미분 일관성을 뒷받침한다. 초기 차이가 **오직 곡률 때문이라고 확정하거나 모든 step/Jacobian의 정확성으로 확대하지 않는다.** 상쇄 heuristic \(8\epsilon_{32}\max(|L_+|,|L_-|,|L_0|)/(2\eta)\)도 신경망 전체의 엄밀한 오차 상한은 아니다.

추가 FD는 전체19.473504초(계측3.386795초), CPU 검산3.554088초였다. 기존4개와 추가10개의 저장 prediction을 대조했다. Embedding 첫 지점 AD=.0145964104493, .01/.005 FD=.0145941972733/.0145912170410로 상대차이는 .015162%/.035580%였다. [추가 실행 기록](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260915/mofit_pixel_fd_refinement_v1/execution.json)

CFG2와 unconditional-only CFG1은 같은 초기 loss를 보였지만 gradient는 bitwise 같지 않았다(max absolute difference6.42613e−7, L2 difference3.79766e−6). 이를1000회 sign update 동일성으로 해석하지 않고 **full은 CFG2를 유지했다.** [CFG 대조](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260915/mofit_medical_benchmark_v1/cfg_control.json)

## 3. 두 target의 full 실행: 같은 환자 E/U 각 한 장

| Target | 환자 member | E002 image member/노출 | U001 image member/노출 | 전체 실행기 시간 | 계측 시간 | 독립 CPU 검산 |
|---|---:|---:|---:|---:|---:|---:|
| Model1 | 1 | 1 / 4회 | 0 / 0회 | 923.981635초 | 908.095158초 | 4.277914초 |
| Model2 | 0 | 0 / 0회 | 0 / 0회 | 924.052725초 | 908.003058초 | 4.292894초 |

Model2의 실제 manifest912개 영상·학습 노출4000회에서 환자14393 부재를 확인했다. **Model2에서도 E는 공통 train_candidate 사진 역할이며 “model2가 본 사진”이라는 뜻이 아니다.** Membership은 현재 NIH 추가 학습 기준이고 base 전체 사전학습 포함 여부는 모른다. 각 target은 전체 A/B 환자군이 달라지므로 한 환자의 포함만 바꾼 인과 실험이 아니다. [M1 실행](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260915/mofit_medical_full_v1/execution.json), [M2 실행](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260915/mofit_medical_target2_full_v1/execution.json)

| Target·영상 | Pixel objective 초기 → 최종 | Surrogate embedding objective 초기 → 최종 |
|---|---:|---:|
| M1 E002 | .352171897888 → .033042341471 | .077315635979 → .000263459922 |
| M1 U001 | .325068235397 → .034710623324 | .084242306650 → .000262639805 |
| M2 E002 | .352192640305 → .034519616514 | .092280745506 → .000352335803 |
| M2 U001 | .325922012329 → .037515379488 | .091031961143 → .000540865702 |

네 경우 모두 각 단계가 최적화한 surrogate objective의 **시작–끝 값이 감소했다.** 모든 중간 step의 단조 감소나 membership 성공을 뜻하지 않는다. Pixel의 VAE mean과 embedding의 posterior sample도 구별한다. [M1 raw](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260915/mofit_medical_full_v1/raw.pt), [M2 raw](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260915/mofit_medical_target2_full_v1/raw.pt)

| Target·영상 | u | L(original, optimized) | h | L(original, initial) | v |
|---|---:|---:|---:|---:|---:|
| M1 E002 | .303051650524 | .329739689827 | .026688039303 | .290988147259 | −.012063503265 |
| M1 U001 | .259994983673 | .286074459553 | .026079475880 | .247229799628 | −.012765184045 |
| M2 E002 | .304328531027 | .334025979042 | .029697448015 | .292002171278 | −.012326359749 |
| M2 U001 | .261809021235 | .278160989285 | .016351968050 | .249188750982 | −.012620270252 |

Member target M1−nonmember target M2의 h 차이는 **E002 −.003009408712, U001 +.009727507830**이다. 같은 환자에서도 두 사진의 방향이 다르다. U 한 장의 양의 차이만으로 MoFit 성공을 선언하거나 E 한 장의 반대 방향으로 실패를 확정하지 않는다. 각 target에서 최적화를 따로 수행했으므로 target 반응과 최적화 경로 차이가 함께 들어간 사례다. [M1 결과](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260915/mofit_medical_full_v1/results.json), [M2 결과](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260915/mofit_medical_target2_full_v1/results.json)

## 4. 실제 비용과 검산 범위

F=UNet forward-example 수, B=backward/autograd 호출 수다. CFG batch2의 한 호출은2F이며 B의 비용이 batch1/2에서 같다고 가정하지 않는다.

| 실행 | UNet forward 호출 | UNet F | UNet B | VAE F | VAE B |
|---|---:|---:|---:|---:|---:|
| Benchmark2+2 및 FD | 18 | 24 | 4 | 8 | 2 |
| 별도 CFG1 대조 | 1 | 1 | 1 | 1 | 1 |
| 추가 pixel FD5개 간격 | 10 | 20 | 0 | 10 | 0 |
| M1 full 두 영상 | 4404 | 6406 | 2600 | 2006 | 2000 |
| M2 full 두 영상 | 4404 | 6406 | 2600 | 2006 | 2000 |
| 추가 E006/U004 × 두 target | 8808 | 12812 | 5200 | 4012 | 4000 |

한 영상 core3200F/1300B에 최종 surrogate pixel/embedding endpoint 진단3F·VAE1F를 추가했다. 원영상 h/u/v는 기존 monitor에서 가져왔다. Caption/CLIP 준비는 이 UNet 비용에 포함하지 않는다.

M1 full의 RTX3070 peak allocated는 **5,490,651,648 bytes(5.491 GB)**, reserved는 **5,953,814,528 bytes(5.954 GB)**였다. 10진 GB이며 모델 적재 후 counter를 초기화한 PyTorch 계측 구간이다. 전체 시스템 사용량이나 다른 해상도에 대한 보장이 아니다.

M1은 영상당2336개 scalar/array 항목과1300개 scalar/hash trace를 검산했다. 상세 update packet은 영상당 pixel5개(index0,1,99,499,999), embedding5개(0,1,29,149,299)다. M2는 동일 동결 수치 검산 함수를 재사용하고 실제 model2 checkpoint/manifest/노출 검사를 별도 수행했다. 저장 prediction의 FP64 MSE, posterior/noising, snapshot의 sign/clamp 및 Adam moments/update, embedding shape, trace/hash 연결, 호출 수와 frozen 입력·코드를 대조했다. 가중치 fingerprint·LoRA tensor·parameter version/no-weight-gradient guard도 통과했다. [M1 독립 검산](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260915/mofit_medical_full_v1/verification.json), [M2 독립 검산](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260915/mofit_medical_target2_full_v1/verification.json)

**전체 neural backward를 별도 구현으로 재실행한 검증은 아니다.** FD는 M1 E002 첫 pixel/embedding 지점에서 수행했으며, 나머지 영상·target의 전 gradient 경로를 독립 검증했다고 하지 않는다. 생성 당시 execution의 “검산 대기” 문자열은 보존되어 있으므로 후속 verification을 함께 읽는다. 상세 source/input/checkpoint hash는 각 protocol에 고정되어 있다.

## 5. 원문에 보고된 실패 조건과의 관계

저자 의료 ROCO 결과도 AUC54.44%,TPR@1%FPR2.00%로 절대 탐지력이 제한적이다. A.13은 시각적 복잡도와 오판의 연관성을 극단 사례에서 논의하고, Table14의 noise4개는 높은 AUC와 달리 낮은FPR의 TPR가18–41%로 달라졌다. 그러나 이를 현재 NIH의 실패 원인으로 옮길 수는 없다. [원문 pp.23–25 직접 대조와 한계](spec_sources/mofit_failure_cases_scope_20260915.md)에 저자 보고·현재 관측·미확인 원인을 구분했다.

## 6. 같은 환자의 E2/U2 집계까지 실제 완료

최초 E002/U001에 나머지 E006/U004를 추가해 **환자1명·영상4장·두 target의8 records**를 완료했다. 각 영상·target에서 동일1,000+300회로 독립 최적화한 뒤, 사전 고정한 네 점수의 mean/max를 모두 계산했다. 환자군 AUC·오탐률·개인 인과효과의 추정은 아니다.

E도 한 장의차이−.0030094와다른장+.0102229가섞여있었고, 두 장 mean/max는각각+.0036067/+.0026156이었다. 따라서 첫 E 한 장의 반대 방향을 E 전체의 실패로 읽으면 안 된다. U의 max는U004만선택해U001의양의차이를출력에보존하지않았지만, 이 관측은 임계값에 따른 false negative 판정이 아니다. 평균이라는 이미 있는 대안으로 결과가 달라지므로 이 손실 자체를 새 알고리즘 기여로 삼지 않는다.

| target | 영상 | 역할 | h | u | v |
|---|---|---|---:|---:|---:|
| model_1 | 00014393_002.png | E | 0.0266880393 | 0.3030516505 | -0.0120635033 |
| model_1 | 00014393_006.png | E | 0.0323130488 | 0.2697883844 | -0.0152073801 |
| model_1 | 00014393_001.png | U | 0.0260794759 | 0.2599949837 | -0.0127651840 |
| model_1 | 00014393_004.png | U | 0.0266759098 | 0.2830108404 | -0.0132931471 |
| model_2 | 00014393_002.png | E | 0.0296974480 | 0.3043285310 | -0.0123263597 |
| model_2 | 00014393_006.png | E | 0.0220901966 | 0.2708853483 | -0.0174591541 |
| model_2 | 00014393_001.png | U | 0.0163519681 | 0.2618090212 | -0.0126202703 |
| model_2 | 00014393_004.png | U | 0.0272095203 | 0.2857056260 | -0.0127360821 |

표의 차이는 **참여 target M1 − 비참여 target M2**다. 양수는 이 고정 사례의 점수 방향이며 참여의 인과효과나 검출 성공 판정이 아니다. `max`는 각 영상에 고정 부호를 적용한 뒤 계산했다.

| 점수 | 사진 조건·집계 | M1 | M2 | M1−M2 |
|---|---|---:|---:|---:|
| h: MoFit core | E mean | 0.0295005441 | 0.0258938223 | +0.0036067218 |
| h: MoFit core | E max | 0.0323130488 | 0.0296974480 | +0.0026156008 |
| h: MoFit core | U mean | 0.0263776928 | 0.0217807442 | +0.0045969486 |
| h: MoFit core | U max | 0.0266759098 | 0.0272095203 | -0.0005336106 |
| −u: null 원시 loss | E mean | -0.2864200175 | -0.2876069397 | +0.0011869222 |
| −u: null 원시 loss | E max | -0.2697883844 | -0.2708853483 | +0.0010969639 |
| −u: null 원시 loss | U mean | -0.2715029120 | -0.2737573236 | +0.0022544116 |
| −u: null 원시 loss | U max | -0.2599949837 | -0.2618090212 | +0.0018140376 |
| −v: 식8의 보조 차이 | E mean | 0.0136354417 | 0.0148927569 | -0.0012573153 |
| −v: 식8의 보조 차이 | E max | 0.0152073801 | 0.0174591541 | -0.0022517741 |
| −v: 식8의 보조 차이 | U mean | 0.0130291656 | 0.0126781762 | +0.0003509894 |
| −v: 식8의 보조 차이 | U max | 0.0132931471 | 0.0127360821 | +0.0005570650 |
| −(u+v): VLM 조건부 원시 loss | E mean | -0.2727845758 | -0.2727141827 | -0.0000703931 |
| −(u+v): VLM 조건부 원시 loss | E max | -0.2545810044 | -0.2534261942 | -0.0011548102 |
| −(u+v): VLM 조건부 원시 loss | U mean | -0.2584737465 | -0.2610791475 | +0.0026054010 |
| −(u+v): VLM 조건부 원시 loss | U max | -0.2472297996 | -0.2491887510 | +0.0019589514 |

추가4건 실행기는 **1877.598초**, 계측 구간은1848.953초였다. 최초두실행까지 full8건의 실행기 합은3725.633초, 총25,624F/10,400B이며 benchmark·FD는 별도다.

각 원시 실행의 저장산술 검산 PASS 뒤 기존4건과 추가4건을 결속했다. 이미지 점수32행·집계32행·target 차이32행을 별도 산술 항등식으로 대조했다. 이 집계 검산은 원시 신경망 역전파의 재실행이 아니다.

[사진별 점수 CSV](mofit_two_image_results_artifacts/v1/image_scores.csv) · [전체 집계 CSV](mofit_two_image_results_artifacts/v1/aggregates.csv) · [사진·집계의 전체 target 차이 CSV](mofit_two_image_results_artifacts/v1/target_differences.csv) · [출처 manifest](mofit_two_image_results_artifacts/v1/manifest.json)

**명칭 정정:** 동결 분석 계약은 `L_VLM`을 원시 조건부 loss의 이름으로 사용해 논문 식8의 기호와 혼동되는 설명을 포함했다. 실제 `−v`와 `−(u+v)` 계산은 사전 선언대로다. 위 표에서 `−v`는 식8의 보조 차이, `−(u+v)`는 별도 원시 조건부 loss로 구분했다. [원계약을 보존한 명칭 정정](spec_sources/mofit_score_naming_clarification_20260915.md).

**남은 범위:** 의료 Eq9의 fusion·보정, 정책 학습에서 분리한 환자군 평가, 개인 참여에 특이적인 실패 원인 설명은 미완료다. 한 환자의 E2/U2 집계를 마친 사실을2번 전체 완료나 새 설계의 근거로 올리지 않는다.
