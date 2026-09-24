# 환자-DP 감사 보고서 검토 — 2026-09-24

이번 작업은 사용자가 전달한 감사·CVPR 전략 보고서를 실제 구현, 기존 결과 및 가까운 선행과 대조한 검토다. 새 특징 추출, GPU 학습, 합성 bank, Q 보호 요약, 평가 실행은 모두 0이다. 현재 실제 효용 결과는 RECEIVER_RESNET18_REUSE_RESULTS_20260924.md로 유지한다. 연구 단계 2는 진행 중이며 논문 기여·최종 검증 완료를 뜻하지 않는다.

## 판단

결과 해석의 큰 방향은 채택한다. 환자-DP 아래 공개전용 합성 대비 Q의 추가 효용은 관측됐지만, A2가 DINO보다 추가 제작비용을 정당화할 만큼 우수하다는 근거는 현재 부족하다. 반면 보고서가 제안한 중심 주장으로 표현을 바꾸는 것만으로 CVPR 기여가 확보되지는 않는다. 한 번 보호한 요약의 재사용과 합성 최적화 후처리는 이미 직접적인 선행이 있다.

현재 성과는 작동하는 보호 합성의 개발 근거다. 강한 선행보다 어떤 사용 조건에서 어떤 효용·비용·분석상의 추가 가치가 있는지는 미확정이다. 기존 구성요소의 조합이나 환자 단위 적용이 기여가 될 가능성을 배제하지 않지만, 그것을 검증된 차별성으로 승격하지 않는다.

## 결과 해석에서 유지·보완할 내용

- DenseNet의 두 DP draw AUROC 기술평균은 A2 약 0.651733, DINO 약 0.648531이다. 네 교차 비교의 환자 구간은 모두 0을 포함한다. 동등성이나 일반적 안정성 우위의 증명은 아니다.
- ResNet18 기술평균은 A2 0.603760, DINO 0.615884다. A2의 약 두 배 제작비용을 정당화할 우위는 확인되지 않았다.
- 보고서가 ResNet18의 Q 추가 효과를 점추정만으로 기술한 부분은 보완한다. 실제 결과에는 A2 DP1−공개전용 AUROC +0.106076, 95% CI [0.059324, 0.149142], DP2−공개전용 +0.081757, CI [0.027835, 0.131380]가 있다. 두 AP 차이 구간은 0을 포함한다.
- 네 보호 bank 모두 ResNet18 공개 실영상 readout 대비 AUROC/AP 차이 구간은 0을 포함한다. 약한 공개전용 합성보다 좋다는 결과와 강한 공개 실영상 이용법보다 좋다는 결과는 구분한다.
- 환자 bootstrap은 해당 bank를 고정한 평가 환자 표집 불확실성이다. 독립 합성 seed 또는 DP mechanism 전체의 분산을 대신하지 않는다.
- non-DP 결과는 참고 비교이며 DP 결과의 수학적 상한이 아니다.

## 실제 clipping은 보고서의 일반 예제와 다르다

확인한 파일:
- ../../code_working/receiver_distillation/patient_dp.py
- ../../code_working/receiver_distillation/patient_dp_dino.py

현재 구현은 환자·class별로 모든 encoder×condition 신호를 묶어 공개 P의 class별 q95 경계 C_c로 먼저 제한한다. 이를 v_pc라고 하면 환자 기여는 다음과 같다.

u_p = 0.5 [v_p0/C_0 ; v_p1/C_1 ; b_p0 ; b_p1]

각 class 신호의 정규화된 norm은 1 이하이고 b_pc는 class 존재 지표다. 따라서 ||u_p||² ≤ (1+1+1+1)/4 = 1이다. 마지막 joint norm 제한은 FP64 경계의 수치적 안전장치다.

그러므로 “raw gradient와 denominator가 마지막 global clipping에서 예산을 경쟁하므로 denominator가 DINO를 밀어낸다”는 설명을 실제 구현의 확인된 실패 원인으로 채택하지 않는다. class 내부에서 두 encoder가 함께 제한되며 신호의 방향·크기가 변할 가능성은 남는다. 이 가능성과 실제 판별 효용의 원인을 동일시하지 않는다.

공개 clipping 기준과 실제 noise scale은 이미 저장되어 있다.

| 항목 | A2 | DINO |
|---|---:|---:|
| query 차원 | 546 | 274 |
| 정규화된 query 감도 | 1 | 1 |
| Gaussian sigma | 0.6002290722589745 | 0.6002290722589745 |
| 공개 class0 경계 | 2.7412545531339387 | 2.0639112795622627 |
| 공개 class1 경계 | 2.689519212597726 | 1.7797955296524304 |
| 복원 class0 gradient sum noise SD | 3.2907613545065475 | 2.4776391051129796 |
| 복원 class1 gradient sum noise SD | 3.228655243600441 | 2.136570039147897 |

A2의 복원 단위 noise SD는 DINO의 약 1.328배/1.511배다. 이는 공개 경계와 후처리에서 산출되는 사실이며 새 Q 분석이 아니다. 추가 신호에 따른 잡음 부담의 구체적인 후보 설명이지만 AUROC 차이의 인과적 증명은 아니다.

546/274의 제곱근 약 1.412는 동일 좌표 잡음에서 RMS L2 norm의 비율이다. 기대 제곱 norm, 즉 잡음 에너지의 비율은 약 1.993이다. 어느 쪽도 epsilon이 차원에 비례한다는 뜻이나 task-relevant error의 직접 예측은 아니다.

복원 시 private mass를 0 이상으로 처리하고 공개 P count와 합쳐 분모를 만든다. P의 class count는 669/6이므로 이 구성의 min=1 분모 floor는 발동하지 않는다. noisy denominator의 비율 효과는 가능하지만 floor가 현재 실패를 만들었다는 설명은 맞지 않는다.

## 이미 있는 감사 산출물을 다시 만들 필요는 없다

보고서의 “미확인”과 저장소의 “미구현”은 다르다.

- P/Q/V manifest와 환자 분리 검사, patient-equal class 집계가 존재한다. P는 672명/813장, Q는 2,027명/5,097장, V는 2,026명/5,047장이다.
- 보호 mechanism metadata에 감도, 차원, 공개 경계, sigma, 복원 잡음 크기, analytic Gaussian 보정 정보가 기록되어 있다.
- test_patient_dp.py에는 인접성·전체 norm·누락 class·반복 방문·잡음 없는 재현·안정화·잘못된 입력 검사가 있다. Gaussian calibration은 별도 dp_accounting 및 고정밀 계산으로 대조하는 경로가 있다.
- DP noise는 OS 기반 난수를 사용하며 재구성 가능한 seed와 noise vector를 공개 payload에 기록하지 않는다. 보고서의 seed 비공개 원칙은 맞지만 새 패치가 필요한 결함으로 확인된 것은 아니다.
- 평가에는 고정된 2,000회 paired patient-cluster percentile bootstrap, index hash, 예측 배열 및 metric 검산 기록이 있다.

논문용 요약 manifest를 편집할 가치는 있으나, 이 항목들을 처음부터 구현·검산해야 한다는 이유로 2–4시간 관문을 다시 추가하지 않는다. 이번 검토에서 테스트 suite나 bootstrap을 새로 실행하지는 않았다.

## privacy 범위

네 독립 epsilon8/delta1e-5 보호 산출물을 같은 Q에 대해 함께 공개하면 기본 합성의 보수적 상한 (32,4e-5)을 사용할 수 있다. 이는 조건부 상한이며 “최소 epsilon32”, 실제 외부 공개가 확인됐다는 진술 또는 전체 연구개발 transcript가 보호됐다는 진술이 아니다. 더 타이트한 회계 가능성과 과거 비DP 접근의 별도 범위는 유지한다.

같은 보호 summary로 합성 seed만 바꾸는 것은 Q를 다시 보지 않는 후처리다. 그러나 그것이 측정하는 것은 해당 summary에 조건부인 합성 변동이며 새 잡음에 대한 반복성을 대신하지 않는다. private V에서의 평가 공개 또한 Q 보호와 별개의 문제다.

raw patient prediction, 환자 식별자의 hash, 비DP 내부 진단을 공개 재현 artifact라고 자동으로 배포하지 않는다. 공개 가능한 코드·스키마와 접근이 제한된 실제 환자 자료를 구분한다. 새 final-Q 확보나 Reserved 역할 변경도 이번 검토에서 채택하지 않는다.

## 가까운 선행과 기여 경계

| 선행 | 이미 제공하는 요소 | 현재 남은 비교 질문 |
|---|---|---|
| DP-NTK | finite feature summary를 한 번 보호하고 생성 최적화에서 재사용 | 현재 환자/class/condition gradient 신호가 같은 접근 조건에서 더 유용하거나 효율적인가? |
| Dosser | 보호된 학습신호를 미리 수집하고 합성 최적화와 분리, 신호 잡음 효율 개선 | 현재 frozen pretrained 경로에 맞춘 강한 보호 신호 대안보다 무엇이 남는가? |
| LGM | pretrained encoder의 선형 gradient matching 및 다른 pretrained 모델로의 전이 | 현재 조건 신호·환자 집계·보호 설계가 공정한 적용보다 무엇을 개선하는가? |

이는 선행의 전체 재현 결과나 완전한 신규성 검색이 아니다. 수치 비교를 선택할 때는 원형과 adaptation의 차이, 공개 모델 접근, 보호 단위·예산, 합성 개수 및 계산량을 명시해야 한다.

출처:
- DP-NTK: https://arxiv.org/abs/2303.01687
- Dosser: https://arxiv.org/abs/2508.01749
- LGM: https://arxiv.org/abs/2511.16674
- Analytic Gaussian calibration: https://proceedings.mlr.press/v80/balle18a.html

## 다음 범위에 대한 권고

8개 합성 seed, A2/DINO 각 20개 public-shadow 잡음, clipping grid, diffusion 비교를 지금 일괄 채택하지 않는다. 반복 수 8의 정밀도·비용 근거가 없고, P에는 양성 class 환자가 6명뿐이다. P를 복제해 Q와 같은 2,027명의 독립 환자로 취급할 수 없으며, 작은 공개 shadow에서의 좋은 설정이 Q에서도 최적이라는 보장은 없다.

가장 먼저 정리할 것은 가까운 선행 대비 현재 방법의 정확한 차이와 그것을 판별하는 강한 대조 하나다. 이 비교로 검증할 주장을 명확히 한 뒤 필요한 실행량을 정한다. 외부 baseline 또는 더 많은 반복이 있다는 사실 자체를 독자적인 방법 기여로 삼지 않는다.

A2 우위를 만들기 위한 자동 조정, DINO를 새 이름으로 바꿔 기여로 선언하는 것, 새 Q 잡음 반복, Expert/Reserved/확인용 수신자 사용은 채택하지 않는다. 현재 보호 합성의 긍정적인 개발 결과와 다중-source의 mixed 결과를 함께 보존한다.

