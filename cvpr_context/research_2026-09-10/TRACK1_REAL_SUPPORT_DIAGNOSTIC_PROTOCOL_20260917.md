# 실환자 학습 범위 확대: 고정 분류기의 일반화 대조 명세

2026-09-17. **설계의 근거는 있으나 새 성능 결과는 없다.** 학습영상에서는 높은 판별력, 다른 개발 환자에서는 낮은 판별력을 보인 [고정 분류기 진단](TRACK1_CLASSIFIER_GENERALIZATION_RESULTS_20260917.md)에 따른 별도 비DP 개발 대조다. 현재 full64·LoRA 실패와 DP 중단을 유지한다.

질문: **같은 ResNet18 학습 절차와 계산량에서, 추출 가능한 실환자 범위를 넓히면 같은 개발 실영상의 판별력이 개선되는가?**

## 1. 비교 하나와 변경 하나

| 조건 | 학습 가능 실자료 | 환자/영상 | 양성 환자/영상 |
|---|---|---:|---:|
| R1, 기존 고정 대조 | 기존 공개 real | 672명 / 813장 | 6명 / 6장 |
| Rwide, 새 비DP 진단 | 기존 공개 real + 기존 classifier-selection 전체 | 2,699명 / 5,910장 | 120명 / 236장 |

추가되는 classifier-selection은2,027명·5,097장, 기흉 양성114명·230장이다. 이 집합 전체를 사용하고 점수·loss·영상 외관으로 일부를 고르지 않는다. 자료량별 learning curve나 여러 subset 중 선택은 이번에 포함하지 않는다.

**Rwide는 공개자료 baseline이나 privacy-preserving arm이 아니다.** 추가 환자의 원 졸논 역할은 private_train이며, 이번에는 이미 비잠금 CVPR 개발 역할로 분리된 자료를 진단 학습에 쓰는 계획이다. 실제 private80은 추가하지 않는다. 기존 public813, head public32/64, private80/320과 구분한다.

R1과 Rwide 모두 실자료32장, 양성16/음성16, 같은400 optimizer step이다. 변경하는 것은 두 half-batch가 추출하는 **허용 실자료 pool**이다. 합성자료, 생성기, adapter, classifier 구조·학습률·증강·step은 바꾸지 않는다.

## 2. 자료 사용 이력과 보존

원본 역할 CSV는 변경하지 않는다. 별도 overlay에2027명 전원의 original_thesis_role, previous_cvpr_role, planned_role, 과거 소비, 실제 optimizer 소비 여부를 기록한다.

- 이 환자들은 이미 R1 calibration의 네 LR/step 후보 평가와 데이터 전처리에 사용됐다. 미사용 환자나 새 독립 validation이라고 부르지 않는다.
- 이번 설계 단계에서 새 학습은0이며 overlay는 planned_not_yet_optimizer_consumed다.
- 향후 실행 시 이 pool 전체를 해당 모델/후속 분기의 학습 자원으로 관리하고, 실제 등장 환자·영상·횟수도 별도로 남긴다. 해당 모델의 validation이나 독립 confirmation으로 다시 사용하지 않는다.
- 기존 R1 calibration 기록은 그대로이며, LR1e-4/400step 선택도 재실행하지 않는다. 이제 학습할 집합에 대해 과거 hyperparameter를 선택한 이력이 있다는 점을 명시한다.
- Method-development2,026명·5,047장은 계속 평가 전용이다. 다만 이미 반복적으로 결과를 본 개발자료이며 독립 최종 평가가 아니다.
- Expert final532명, reserved4,213명, 원 졸논 final/calibration/test는 계속 잠근다. Metadata로 환자 교집합만 검사하며 expert pixel·prediction을 열지 않는다.

## 3. 정확한 추출 규칙

기존 R1과 같은 patient-uniform within label → image-uniform within patient/label 규칙을 쓴다. 환자·영상 정렬, seed hash, NumPy PCG64, slot 순서도 유지한다.

매 step 앞16장과 뒤16장은 각각 음성8/양성8이다. 기존 salt의 (classifier seed, zero-based step, public-shared)와 (seed, step, public-extra) 난수 stream을 그대로 사용한다.

- R1: 두 half 모두 기존 public813 pool.
- Rwide: 두 half 모두 public813 + former-selection5097의 합친 pool.
- 공통 난수 규칙을 쓰지만 pool 크기가 달라 선택 ID가 같을 필요는 없다. **앞16장을 기존 public로 고정하는50:50 source 혼합은 아니다.**
- 실제 record에는 real/public 또는 diagnostic_real/former_classifier_selection의 출처를 보존한다. 추가자료를 real/public으로 다시 이름 붙이지 않는다.
- 혼합 양·음성 방문이 있는 환자의 영상 label은 그대로 유지한다. 환자 any-positive로 모든 영상을 양성으로 바꾸지 않는다.

확대 pool의 양성 영상 보유 환자는120명, 음성 영상 보유 환자는2,684명이며105명은 두 class에 모두 등장한다. Class-balanced sampling 때문에 source별 총 질량은 단순히 환자수672:2027이 아니다. 실제 예정 slot·환자·영상별 노출을 seed별로 계산해 저장한다.

같은400step에서 public 양성의 반복은 줄고 고유 환자·영상 범위는 늘어난다. 그와 함께 질환·동반소견 구성, 환자당 영상 수, source 질량도 달라진다. **양성 수 하나만의 인과 효과를 분리하는 실험은 아니다.**

## 4. 고정 학습·평가 조건

| 항목 | 고정값 |
|---|---|
| 초기화 | ImageNet ResNet18, 기존 동일 seed별 initialization SHA; 새 classifier head 한 출력 |
| seed | 11,23,37 |
| 학습 | FP32, 전체 parameter, BCEWithLogitsLoss, AdamW(lr1e-4,weight_decay1e-4,betas.9/.999,eps1e-8,foreach=False) |
| 계산량 | batch32 ×400step ×3seed =1,200 main update |
| 전처리 | 기존 grayscale, 종횡비 유지224 letterbox, 3채널·ImageNet normalization |
| 증강 | 기존 affine ±5도, translation±2%, scale.95–1.05, flip 없음 |
| checkpoint | 마지막400step 하나; early stopping·중간 성능 선택 없음 |
| BatchNorm | 기존 학습 동작 유지, 평가에서는 저장 통계 사용; 별도 재보정 없음 |
| 평가 | 기존 method-development 전체, eval()/inference_mode(), 증강 없음, raw logit |

분류기3개는 모두 같은 seed의 과거 R1과 동일한 초기 state에서 시작한다. 기존 R1·Dreal 또는 LoRA 분류기의 최종 가중치로 warm start하지 않는다.

학습 완료 후 실제 본 고유 학습영상도 같은 eval 경로에서 source별로 평가한다. 이는 일반화 차이의 보조 진단이며 모델 선택에 사용하지 않는다. 최종400step 모델 세 개를 전부 저장한 뒤 개발 metric을 계산한다.

## 5. 비교·통계·판정

Primary는 mean-seed AUROC(Rwide)−mean-seed AUROC(R1)다. AP 차이, 각 seed의 차이, 개발 class별 BCE·logit, 실제 본 학습자료 metric·노출량을 함께 보고한다. 예측을 먼저 평균한 ensemble AUROC로 바꾸지 않는다.

R0 대비 AUROC/AP는 사전 secondary 비교다. R1보다만 좋아지고 R0에는 못 미치면 실자료 기준 전반을 개선했다고 말하지 않는다. 기존 Dreal은 참고값으로만 두고 추가 primary arm이나 성공 조건으로 만들지 않는다.

환자2,026명 단위 paired cluster bootstrap2,000회는 기존 저장 patient_counts를 그대로 사용한다. 같은 환자의 모든 영상과 재표집 배수를 포함하고, 모든 arm/seed에 같은 draw를 적용한다. 두 비교의95% percentile 구간을 전부 보고한다. 개발 단계 기술 통계이며 다중 검정으로 보정된 최종 확증·학습 반복 전체의 모집단 구간이 아니다.

계속 검토할 개선 후보 표시는 다음으로 고정한다: R1 대비 평균 AUROC +0.01 이상, 세 seed 중2개 이상에서 양의 차이, 평균 AP 비감소. 이는 이전과 동일한 제한적 투자 판단용 규칙이며 임상적 유의성, 충분한 절대 성능, 원인 확정, DP 진입 gate가 아니다. 절대 성능과 불확실성도 반드시 같이 제시한다.

| 결과 | 허용되는 해석 |
|---|---|
| 같은 학습에서 확대 pool이 반복적인 개발 개선을 보임 | 이 자료 범위 변화의 효과 후보. 양성6명만이 유일한 원인이라는 결론은 아님 |
| 개선되나 절대 성능은 낮거나 구간이 넓음 | 일반화 문제가 해결됐다는 증거는 부족 |
| seed 방향 불일치·AP 악화·작은 차이 | 혼합/불확실. threshold·seed·budget을 자동 추가하지 않음 |
| 미통과 | 이 pool과400step 설정으로 개선 근거 미확보. 자료 확대 일반의 무효나 classifier만의 원인 증명은 아님 |
| 실행 대응 실패 | 기술 문제로 멈추며 방법 성능 결과와 분리 |

어느 경우도 private synthetic utility의 성공이 아니다. 기존 head·LoRA 실패를 사후 변경하거나 DP·expert final을 자동 실행하지 않는다.

## 6. 실행 전 연결 검사와 비용

본 문서 작성 시 **새 runtime은 미구현**이다. 다음 실행 패키지에서 새 namespace/pool 공급자만 붙이고 기존 data_v2/train_v2 source는 유지해야 한다.

1. 모델 실행 전에 선택5910장과 method-development5047장의 실제 파일·cache pixel binding을 확인하고 환자·영상·기록 SHA 분리를 다시 검사한다. 계획 단계에서는 metadata와 과거 검증 기록만 사용한다.
2. 새 공급자에 public-only pool을 넣으면 기존 R1의1step과 선택 ID·GPU 입력·logit·loss·최종 state가 exact해야 한다. 원 경로1회+새 경로1회,2update다.
3. Rwide1step을 같은 초기상태에서 두 번 실행해 실제 출처·배열·입력·update exact replay를 확인한다.2update다.
4. 위 별도4update 관문 통과 후 Rwide3seed×400=1200update를 실행한다. 본학습은 probe를 이어받지 않고 fresh initialization/optimizer로 시작한다.
5. 기존 R1 결과를 대조로 재사용할 때 고정 개발첫64장×3seed의 raw logit도 재현한다. Parity 실패 시 과거 결과를 동일 조건 대조로 조용히 재사용하지 않는다.
6. 세 checkpoint·실제 노출 기록을 고정한 뒤 개발 분석과 독립 수치 검산을 수행한다. 완료 산출물 hash를 확인한 재개만 허용하며 결과를 본 뒤 학습조건을 바꾸지 않는다.

기존 R1 세 run의 순수 학습시간 합계는161.78초였다. 새 pool의 cache/IO·검산 비용은 별도다. 향후 구현·연결검사·실행·보고 전체 예상은30–50분이며, 아직 실측한 Rwide 시간은 없다. 이번 설계는 새 GPU·학습·추론·생성·raw decode 모두0이다.

계획·명부·예정 노출·검증 근거는 [계획 JSON](spec_sources/real_support_diagnostic_plan_20260917.json)과 [설계 검증](spec_sources/real_support_diagnostic_plan_verification_20260917.json)에 연결한다.
