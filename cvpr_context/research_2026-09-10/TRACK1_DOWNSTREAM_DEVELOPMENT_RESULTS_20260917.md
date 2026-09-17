# 비DP downstream 본 개발 결과 — 2026-09-17

**판정: 사적 합성자료 효용에는 나쁜 결과다. 사전 지정한 개발 관문을 통과하지 못했다.**
실행·수치 검산의 통과와 방법의 효용 판정은 구분한다. 이번에는 준비만 한 것이 아니라 512장 생성, R1 calibration 두 trajectory, 7군×3seed의 실제 학습과 개발 평가를 완료했다. Expert final·reserved와 DP는 실행하지 않았다.

[실행 전 명세](TRACK1_DOWNSTREAM_DEVELOPMENT_PROTOCOL_20260917.md)와 [master protocol](TRACK1_DOWNSTREAM_MASTER_PROTOCOL_20260917.md)을 적용했다. 큰 단계2·방향1을 유지한다.

## 실제 성능

영상별 NIH weak label을 사용하는 method-development 2,026명·5,047장(양성223장)의 결과다. AUROC/AP는 seed11/23/37 각각 계산한 뒤 평균했다. 음성은 기흉 label0이며 정상 영상만을 뜻하지 않는다.

| Arm | 역할 | AUROC 평균 | AP 평균 | AUROC seed11 /23 /37 |
|---|---|---:|---:|---|
| R0 | 공개 real 기본 | 0.554745 | 0.055087 | 0.576790 / 0.538013 / 0.549431 |
| R1 | 공개 real + 일반 증강 | 0.544610 | 0.051669 | 0.568315 / 0.556552 / 0.508962 |
| S0 | backbone 합성 | 0.553782 | 0.052479 | 0.540455 / 0.560813 / 0.560078 |
| S1 | public-head 합성 | 0.536259 | 0.051542 | 0.517554 / 0.536933 / 0.554290 |
| S2 | private-only 합성 | 0.549112 | 0.052830 | 0.551877 / 0.532186 / 0.563273 |
| S3 | pooled 합성 | 0.528546 | 0.049065 | 0.547533 / 0.514158 / 0.523947 |
| Dreal | private real 비DP 진단 | 0.596172 | 0.078445 | 0.633540 / 0.545057 / 0.609920 |

| 비교 | 평균 ΔAUROC | 평균 ΔAP | 양의 AUROC seed | 관문 |
|---|---:|---:|---:|---|
| S2−S1 | +0.012853 | +0.001289 | 2/3 | 전체 미통과 |
| S2−R1 | +0.004502 | +0.001161 | 1/3 | 전체 미통과 |
| S3−S1 | -0.007713 | -0.002476 | 1/3 | 전체 미통과 |
| S3−R1 | -0.016063 | -0.002604 | 1/3 | 전체 미통과 |

S2 또는 S3가 S1과 R1 **각각**에 대해 평균AUROC +.01 이상, 양의 차이 최소2/3seed, 평균AP 비감소를 모두 만족해야 통과다. 이 .01은 추가 DP 개발 투자 기준이며 임상적 유의성 또는 final 검정 기준이 아니다.

## 환자 군집을 고려한 보조 불확실성

환자 단위2000회 paired cluster bootstrap에 같은 draw를 모든 arm/seed에 적용했다. 매 draw에서 seed별 metric을 계산한 뒤 평균했다. 아래95% percentile 구간은 개발용이며, 한 생성bank·세 classifier에 조건부다. 반복촬영 영상을 독립 환자로 취급하지 않았다.

| 비교 | ΔAUROC 95% 구간 | ΔAP 95% 구간 |
|---|---|---|
| S2-S1 | [-0.008040, +0.032386] | [-0.004438, +0.006395] |
| S2-R1 | [-0.044948, +0.053823] | [-0.007084, +0.011501] |
| S3-S1 | [-0.029287, +0.013376] | [-0.008333, +0.001930] |
| S3-R1 | [-0.052685, +0.025458] | [-0.009110, +0.005315] |
| Dreal-R1 | [+0.005193, +0.098900] | [+0.010691, +0.054032] |

## 해석과 결정

Dreal−R1의 평균AUROC는 +0.051563, AP는 +0.026776다. Dreal은 private real을 직접 사용한 비DP 진단 비교이며 수학적 상한이나 보호 방법이 아니다.

S2는 S1과의 비교만 보면 점추정 +.012853, 양의 차이2/3seed, AP 비감소를 만족한다. 하지만 R1보다 +.004502에 그쳤고 R1을 이긴 것은 seed37 하나뿐이다. 나머지 두 seed에서는 R1보다 낮았다. S2−S1의 환자 cluster 구간도0을 포함하므로, 부분적인 점추정을 확증된 개선으로 부르지 않는다. S3는 두 기준보다 AUROC/AP 평균이 모두 낮았다.

Dreal은 AUROC에서2/3seed, AP에서3/3seed가 R1보다 높다. 사적 실자료가 현재 task에 유용할 수 있다는 진단 근거는 있지만, 같은 NIH 자료의 역할 분할이므로 사적 자료만의 고유 정보나 외부 병원 domain adaptation을 증명하지 않는다. Dreal 자체의 평균AUROC도 .596으로 높지 않다. 공개 R1 selection 최고AUROC .547과 전체적으로 낮은 downstream 성능은 이 pilot의 중요한 한계다.

또한 S1은 R1과 backbone-synthetic S0보다 낮고, S2도 plain-real R0보다 낮다. 과거 public head의 일반 KID 개선이 이번 기흉 downstream 효용까지 보장하지 않았다. 이번에는 효용을 실제 측정했으며, 현재 구성의 사전 운영 기준에 미달했다.

현재 고정 head·생성bank·classifier 설정은 DP 개발 투자 기준에 미달했다. DP로 확대하지 않는다. 기준·seed·prompt·head scale을 결과 후 바꾸어 이 실행을 성공으로 만들지 않는다. 이 결과를 private data 전체의 무용성이나 모든 생성 방법의 불가능성으로 확대하지 않는다.

## R1 calibration과 실행량

Classifier-selection 2,027명·5,097장만 사용했다. 공개 R1 seed11을 LR1e-4/3e-4 각각800step까지 연속 학습하고400/800 checkpoint를 비교했다. 중간 AdamW 초기화는 없다.

| LR | step | Selection AUROC | AP |
|---:|---:|---:|---:|
| 0.0001 | 400 | 0.547047 | 0.064292 |
| 0.0001 | 800 | 0.538445 | 0.063611 |
| 0.0003 | 400 | 0.525704 | 0.048102 |
| 0.0003 | 800 | 0.528068 | 0.048865 |

최고AUROC와 .002 이내 후보 중 적은 step, 작은 LR 우선 규칙으로 **LR0.0001, 400step**을 선택했다. 이후21run 모두에 동일하게 적용했다. Method-development 성능은21run이 끝난 뒤에만 계산했다.
실제 update는 calibration1600 + development8400 + 별도 관찰기 parity4 = **10004회**다. 과거 잘못된98회와 이번 학습을 섞지 않았다.

## 입력·노출량·비용

새 latent64개에 기흉64/비기흉64 요청을 대응시켜 방법별128장, 총512장을 만들었다. 음성 요청은 No Finding22/Effusion21/Cardiomegaly21이다. 모든 방법의 latent·prompt mapping은 같고 생성 후 filtering은 없었다. 요청 라벨은 생성영상의 전문가 판독 정답이 아니다.

R0/R1은 public real32장, S0–S3은 public real16+synthetic16장, Dreal은 public16+private real16장이다. 따라서 결과는 **동일 계산량에서 공개 real batch 절반을 대체**한 효과이며, 동일한 real 학습량에 합성을 단순 추가한 효과가 아니다.

| Arm | 공개 양성6장 최대 노출 (seed11/23/37) | 공개 양성6장 중앙 노출 | 총 학습초(3seed) |
|---|---|---|---:|
| R0 | 1087 / 1107 / 1113 | 1070 / 1069.5 / 1063 | 125.4 |
| R1 | 1087 / 1107 / 1113 | 1070 / 1069.5 / 1063 | 161.8 |
| S0 | 566 / 550 / 571 | 530 / 536 / 530 | 162.4 |
| S1 | 566 / 550 / 571 | 530 / 536 / 530 | 163.5 |
| S2 | 566 / 550 / 571 | 530 / 536 / 530 | 163.1 |
| S3 | 566 / 550 / 571 | 530 / 536 / 530 | 161.8 |
| Dreal | 566 / 550 / 571 | 530 / 536 / 530 | 162.3 |

512장 생성·저장·모델 적재 총 1593.9초, calibration 학습 217.9초,21run 학습 1100.2초다. 평가 추론·검산·원본 재결속·구현 시간은 별도다.

| 생성 방법 | 평균 연산초/장 | 평균 저장초/장 | 최대 allocated GiB |
|---|---:|---:|---:|
| backbone | 2.592 | 0.292 | 3.958 |
| public | 2.625 | 0.467 | 3.958 |
| private_only | 2.628 | 0.466 | 3.958 |
| pooled | 2.629 | 0.467 | 3.958 |

각 run의 exposure.json에 모든 public/private/synthetic image 및 patient/cell별 횟수, unique coverage, 최대·중앙 노출을 남겼다. Calibration의400/800 exposure도 별도 보존했다. 이 짧은 단일 장비 측정을 patient-DP LoRA 대비 비용 우위로 주장하지 않는다.

`patient_label_exposure.json`에는 각 환자·class별 노출도 추가 집계했다. 원 exposure의 patient 횟수는 해당 환자의 모든 class 방문을 포함하므로 이를 양성 영상 사용 횟수와 혼동하지 않는다. Synthetic의 patient 칸은 실제 환자가 아니라 생성 cell이다.

## 실행·독립 검산

검증된 data_v2/train_v2는 변경하지 않고 별도 runner에 SHA를 결속했다. 중간 저장용 관찰 hook의2step plain/observed final state·입력·logit·loss가 exact였고,1step 저장state도 과거 검증과 일치했다. 원본11,277장의 SHA·decode·cache를 다시 대조했다. Legacy import와 허용 명부 밖 raw pixel 접근을 차단했다.
독립 검산 114,538항목은 성능 표본 수가 아니다.512PNG/trajectory,15,360DDIM 전이, 모든 학습 source draw·label·array binding·BCE·exposure, 실제 최종state, calibration 선택과21개 AUROC/AP를 다시 계산했다. 최대 metric 차이는 1.11e-16, 최대 DDIM 차이는 9.54e-07였다. 독립 검산기는 생산 data/train 수치 함수를 import하지 않는다.

완료 cell/run은 SHA marker로만 재사용한다. Expert final532명·810장과 reserved4213명은 pixel/prediction을 열지 않았다. 기존192장 pooled 미통과, PadChest·CheXzero 실패, 과거98회 무효 학습은 그대로 보존했다. Final-ready=false다.

이번 본 실행에는 중단·재개나 재생성이 없었다. 선택된 calibration R1 checkpoint와 본 R1 seed11의 모든 state tensor도 exact였다. 검산 전 생산 결과 packet의 `UNVERIFIED` 표기는 원본 보존을 위해 바꾸지 않았고, 그 SHA에 결속된 별도 verification.json이 최종 검산 통과를 기록한다.

## 근거

- [계약](../../code_working/_reports/downstream_development_20260917_v1/contract.json), [R1 선택](../../code_working/_reports/downstream_development_20260917_v1/calibration_choice.json).
- [실제 성능 전체](../../code_working/_reports/downstream_development_20260917_v1/result.json), [독립 검산·환자 cluster 구간](../../code_working/_reports/downstream_development_20260917_v1/verification.json).
- [전체 생성 manifest](../../code_working/_reports/downstream_development_20260917_v1/generation_manifest.json), [안전 본실험 runner](../../code_working/downstream_utility/run_development_v1.py).

이 문서는 로컬 실행·검산 결과다. 원격 main SHA 확인이나 push 완료를 뜻하지 않는다.
