# 환자-DP feature-mean/L2 단일 대조 결과 (2026-09-24)

## 완료 범위

새 ε8/δ1e-5 보호 요약 한 번, DINO feature-mean/L2 bank 한 개(128장·200회·seed101·microbatch16)를 완료했다. 최종 PNG 고정 후 기존 V 특징과 2,000회 환자 bootstrap으로 기존 gradient-DP 두 bank와 비교했다. 판정: `MIXED_OR_NO_CLEAR_RECIPE_ADVANTAGE`.

이번은 신호 표현·정합 거리·class 처리·clipping을 포함한 전체 구성 비교다. 단일 요소의 인과 효과나 Dosser 전체 재현/우위를 주장하지 않는다.

주평가 DenseNet에서 feature AUROC는 기존 gradient DP1보다 낮고 DP2보다 높았으며, 두 차이의 환자 구간이 모두0을 포함했다. 따라서 현재 gradient/cosine 구성이 이 feature/L2 대조보다 더 유용하다는 근거를 이번 한 bank에서 확보하지 못했다. 두 방식의 동등성이나 feature 방식의 일반적 우위를 입증한 것도 아니다.

## 고정한 차이

| 항목 | 기존 DINO gradient-DP | 이번 feature 대조 |
|---|---|---|
| 학습신호 | 네 조건의 선형 head CE gradient | 네 조건의 class별 특징 평균 |
| 정합 | class 균등 gradient 결합 후 조건별 cosine 평균 | class별 half-squared normalized L2 평균; 조건·좌표 합산 |
| 제한 | Q 환자/class gradient 제한 | P/Q 환자/class 및 합성 virtual patient의 full-K4 특징 제한 |
| 보호 query | 274좌표 | 130좌표 |
| 개인정보 예산 | 각 ε8/δ1e-5 | ε8/δ1e-5 |

DINO checkpoint·P-only PCA16/q95·변환 조건·head provenance·초기 template·optimizer·200회 해상도 일정은 재사용했다. Feature 경로는 head를 계산에 사용하지 않는다. 초기 renderer/optimizer/RNG/활성 pyramid 상태는 기존 A2/DINO seed101과 정확히 일치한다.

공개 class별 clipping 기준 C0=1.90613299511, C1=1.5917066732. P의 class-present 환자 full-K4 norm q95로 Q 열람 전에 고정했다. 공개 양성 환자6명이라는 보정자료의 한계는 남는다.

`L = 0.25 * sum_c,k,j ((mu_S[c,k,j]-target[c,k,j])/C_c)^2`. 공개·합성에도 같은 제한을 적용하는 것은 기존 gradient 경로와의 추가 차이로 명시한다. 명목 손실 범위0~2는 같은 이미지 gradient 크기를 보장하지 않는다. Anchor0.01·TV0.0001은 한 번 적용했다.

## 개발 성능

| 수신자 | 구성 | AUROC | AP |
|---|---|---:|---:|
| DenseNet | Gradient DP1 (기존) | 0.672935 | 0.080814 |
| DenseNet | Gradient DP2 (기존) | 0.624128 | 0.067322 |
| DenseNet | Feature DP1 (신규) | 0.650018 | 0.080832 |
| ResNet18 | Gradient DP1 (기존) | 0.610225 | 0.066695 |
| ResNet18 | Gradient DP2 (기존) | 0.621542 | 0.061495 |
| ResNet18 | Feature DP1 (신규) | 0.601865 | 0.072737 |
| BioViL | Gradient DP1 (기존) | 0.707482 | 0.085203 |
| BioViL | Gradient DP2 (기존) | 0.647386 | 0.065002 |
| BioViL | Feature DP1 (신규) | 0.726545 | 0.089627 |

차이는 모두 **feature − 기존 gradient**이며, 어느 잡음 하나를 골라 비교하지 않았다.

| 수신자 / 대조 | ΔAUROC (95% 환자 구간) | ΔAP (95% 환자 구간) |
|---|---|---|
| DenseNet / DINO_DP1 | -0.022917 [-0.070324, +0.024144] | +0.000018 [-0.022595, +0.024648] |
| DenseNet / DINO_DP2 | +0.025890 [-0.031004, +0.081657] | +0.013510 [-0.014208, +0.042894] |
| ResNet18 / DINO_DP1 | -0.008361 [-0.062735, +0.045237] | +0.006042 [-0.018262, +0.032903] |
| ResNet18 / DINO_DP2 | -0.019677 [-0.056914, +0.015563] | +0.011242 [-0.004088, +0.032863] |
| BioViL / DINO_DP1 | +0.019063 [-0.014840, +0.052949] | +0.004424 [-0.016575, +0.021166] |
| BioViL / DINO_DP2 | +0.079159 [+0.044526, +0.113773] | +0.024625 [+0.010781, +0.038793] |

한 feature 잡음 대 두 gradient 잡음, 같은 합성seed의 고정 bank 비교다. 위 구간은 환자 표집에 대한 조건부 구간이며 잡음·초기화 전체의 방법 우위/동등성 구간이 아니다. DenseNet 주개발, ResNet18 추가개발, BioViL 진단이라는 역할을 유지했다. Feature 방식 자체의 공개전용 대조를 새로 만들지 않았으므로 사적 Q의 순수 추가효과까지 새로 판정하지 않는다.

## 구현·최적화 검산

- 공개 실제 DINO one-pass/two-pass microbatch1/4 손실 차이0, gradient 최대 7.45e-09; 기존 허용 기준 유지.
- P813/Q5097 cache 재사용. 환자/class 평균 독립 재계산 오차 0, Q bounded-sum 상대화 최대오차 1.16e-14. 새 Q/V 픽셀 forward0.
- 환자 추가/제거 감도1, class 분모0/음수 잡음/비정상 입력, 제한 활성·비활성·0에서 FP64 gradient 검사 통과. 기존 분리·저장·재개 검사를 전체 반복하지 않았다.
- 정합 loss: 첫 update 전 0.029923811 → 200회 뒤 float 2.1417541e-05 / PNG 3.2634635e-05. PNG 기준 감소율 99.89%. PNG−float loss +1.12e-05.
- 위 감소는 선언한 목표의 정합 진단이다. 전역 최적해 도달·다른 L2 스케일의 불필요성·feature 표현 자체의 한계까지 보장하지 않는다.
- 최종 합성 feature 제한 활성: class0 63/64, class1 64/64. 손실 감소는 제한 이후 평균의 정합이며, 제한 전 특징 분포 전체의 복제를 의미하지 않는다. 결과에 따라 제한이나 계수를 변경하지 않았다.
- 최초 연결 오류: 함수 adapter에서 keyword-only 기본 인자를 누락했다. Python 인자 결속 단계에서 중단되어 실제 난수 추출0이었다. 실패 traceback/소스를 보존하고 기본 인자 복사만 수리했다. 구성 배열의 모의 sampler 검사 후 최초 실제 draw를 한 번 수행했다. 손실·기준·잡음·checkpoint를 결과에 맞춰 바꾸지 않았다.

## 실제 계산량·보호 범위

- 합성·PNG 저장 35.07분, 기존 수신자 평가 0.21분, 최종 float/PNG 정합 진단 14.89초. 요청 착수부터 본 기록까지 53.38분. 전체 상한120분 이내.
- 학습 DINO forward204,800 / backward102,400; 공개 gradient 검사96/64; 수신자 PNG 평가384 forward; 최종 정합 진단1,024 forward. 학습 peak allocated 2.584GiB.
- 고정 encoder/target 불변, 실제 합성 parameter 갱신, 최종128 PNG·label·virtual pair·hash·초기 상태 일치 검사 통과. 중간 checkpoint로 완료 처리하지 않았다.
- analytic Gaussian σ=0.600229072258974; add/remove 한 환자의 모든 방문/class/조건을 포함. 비공개 class count도 함께 보호한다. 실제 DP seed·실현 잡음은 저장하지 않았다.
- 이번 한 요약은(8,1e-5), 기존 네 요약과 모두 공동 공개할 경우 기본 합성 상한은(40,5e-5)다. 과거 비DP 개발·V 예측·성과 보고 전체를 보호했다는 뜻이 아니다. 외부 공개는 수행하지 않았다.
- `protected_png`만 보호된 합성 산출물 후보다. cache·검산·예측을 포함하는 상위 실행 폴더는 비공개 내부 기록이다.

## 기록과 다음 범위

새 코드: `receiver_distillation/feature_control.py`(신호·query·L2), `feature_runtime.py`(기존 실행기 adapter), `verify_feature_control.py`(변경 경로 검사), `run_feature_control.py`(한 release/bank 연결), `evaluate_feature_control.py`(기존 평가 재사용). 기존 gradient 구현과 bank는 수정하지 않았다.

원본 gradient rule의 `loss`·head tuple 등 유래 metadata는 보존돼 있다. 실제 실행은 `feature_objective` 및 class bounds를 사용한다. 이 구분은 `effective_objective_receipt.json`에 별도로 기록했다.
실행 산출물: [receiver_feature_dp8_20260924_v1](../../code_working/_reports/receiver_feature_dp8_20260924_v1). 실행 계약·기존 결과·실제 실패·수정 기록은 보존했다. 추가 bank·잡음·계수 변경·새 수신자·Expert·Reserved는 실행하거나 예약하지 않았다.

[실행 계약](RECEIVER_FEATURE_PATIENT_DP_CONTRACT_20260924.md)
