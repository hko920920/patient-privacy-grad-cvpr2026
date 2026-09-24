# Feature 공개전용·독립 DP 잡음 반복 결과 (2026-09-24)

승인된 두 bank를 모두 새로 학습하고 마지막200회 PNG를 고정한 뒤 평가했다. 기존 Feature DP1과 DINO gradient DP1/DP2는 저장된 예측과 bootstrap을 재사용했다.
단계2의 제한된 개발 비교이며 일반적인 신호 복잡도 원리, DP interaction, 동등성 또는 CVPR 기여의 입증이 아니다.

**이번 묶음의 판단:** 같은 feature 제작 방식에서 보호된 Q의 AUROC 추가효용은 두 잡음·두 개발 수신자에서 관측됐다. 네 대응 차이의 환자 95% 구간이 모두 양수다. AP는 점추정이 모두 상승했지만 DenseNet DP1을 제외한 추가효용 구간은 0을 포함한다.

새 Feature DP2의 DenseNet AUROC0.596743은 기존 DP1의0.650018보다0.053274 낮고, 그 고정 bank 차이 구간[-0.088949,-0.017824]은 음수다. Feature DP2는 Gradient DP1보다 AUROC/AP 모두 낮은 조건부 구간을 보이며, Gradient DP2와의 차이 구간은0을 포함한다. 따라서 '단순 feature로 성능 손해 없이 충분하다' 또는 feature의 우위·일반적인 잡음 안정성은 확보하지 못했다. ResNet18에서는 Feature DP2 AUROC0.610507로 DP1과의 차이가 명확하지 않다. 두 관측으로 방법의 잡음 분산 우열을 판정하지 않는다.

## 실행 범위

- DINO, P-only PCA16/q95, 개별 K4 변환, 환자/class 집계와 clipping, L2, seed101, 128장,200회,microbatch16 및 AdamW/해상도 일정은 기존 feature DP1과 동일.
- 공개전용 target은 P class 분모669/6만 사용하며 이전 검산 fixture와 정확히 일치. Q 통계/분모 혼입 없음.
- DP2는 독립 OS-random draw1회, 환자 add/remove 감도1,130좌표,ε8/δ1e-5. 이전 공개 class bounds와 분모 복원·후처리 유지.
- 두 최종 PNG 봉인 후 기존 DenseNet/ResNet18 V 특징과2000회 동일 환자 bootstrap 사용. 새 Q/V 픽셀 추출0, 새 수신자0.

## 고정 bank 평가

| 구성 | DenseNet AUROC | DenseNet AP | ResNet18 AUROC | ResNet18 AP |
|---|---:|---:|---:|---:|
| Feature 공개전용(신규) | 0.511293 | 0.048806 | 0.521656 | 0.051309 |
| Feature DP1(기존) | 0.650018 | 0.080832 | 0.601865 | 0.072737 |
| Feature DP2(신규) | 0.596743 | 0.064188 | 0.610507 | 0.063724 |
| Gradient DP1(기존) | 0.672935 | 0.080814 | 0.610225 | 0.066695 |
| Gradient DP2(기존) | 0.624128 | 0.067322 | 0.621542 | 0.061495 |

## 대응 비교

구간은 해당 완성 bank와 보호 잡음을 고정한 환자 표집 불확실성이다. 두 잡음 번호를 방법 간 대응된 난수로 취급하지 않는다.

| 수신자 / 대조 | ΔAUROC (95% 환자 구간) | ΔAP (95% 환자 구간) |
|---|---|---|
| DenseNet_FEATURE_DP2_minus_FEATURE_DP1 | -0.053274 [-0.088949, -0.017824] | -0.016644 [-0.036037, +0.000003] |
| DenseNet_FEATURE_DP1_minus_FEATURE_PUBLIC | +0.138724 [+0.056689, +0.224339] | +0.032026 [+0.007929, +0.064566] |
| DenseNet_FEATURE_DP1_minus_DINO_DP1 | -0.022917 [-0.070324, +0.024144] | +0.000018 [-0.022595, +0.024648] |
| DenseNet_FEATURE_DP1_minus_DINO_DP2 | +0.025890 [-0.031004, +0.081657] | +0.013510 [-0.014208, +0.042894] |
| DenseNet_FEATURE_DP2_minus_FEATURE_PUBLIC | +0.085450 [+0.014013, +0.167720] | +0.015382 [-0.001903, +0.037795] |
| DenseNet_FEATURE_DP2_minus_DINO_DP1 | -0.076191 [-0.123990, -0.030048] | -0.016626 [-0.033873, -0.000791] |
| DenseNet_FEATURE_DP2_minus_DINO_DP2 | -0.027385 [-0.085247, +0.029113] | -0.003134 [-0.025186, +0.017674] |
| ResNet18_FEATURE_DP2_minus_FEATURE_DP1 | +0.008642 [-0.048987, +0.068168] | -0.009013 [-0.033473, +0.015564] |
| ResNet18_FEATURE_DP1_minus_FEATURE_PUBLIC | +0.080208 [+0.007208, +0.155725] | +0.021428 [-0.005946, +0.049557] |
| ResNet18_FEATURE_DP1_minus_DINO_DP1 | -0.008361 [-0.062735, +0.045237] | +0.006042 [-0.018262, +0.032903] |
| ResNet18_FEATURE_DP1_minus_DINO_DP2 | -0.019677 [-0.056914, +0.015563] | +0.011242 [-0.004088, +0.032863] |
| ResNet18_FEATURE_DP2_minus_FEATURE_PUBLIC | +0.088850 [+0.033001, +0.147693] | +0.012415 [-0.007704, +0.031458] |
| ResNet18_FEATURE_DP2_minus_DINO_DP1 | +0.000282 [-0.030912, +0.031763] | -0.002971 [-0.021587, +0.014617] |
| ResNet18_FEATURE_DP2_minus_DINO_DP2 | -0.011035 [-0.058449, +0.036205] | +0.002229 [-0.015140, +0.023295] |

## 질문별 관측

- DenseNet: 두 feature DP bank 모두 공개전용 대비 AUROC 구간 전체 양수=True; AP 구간 전체 양수=False.
  Feature DP1의 각 gradient bank 대비 AUROC 구간 전체 양수 여부=[False, False]. 차이 미검출을 동등성으로 해석하지 않는다.
  Feature DP2의 각 gradient bank 대비 AUROC 구간 전체 양수 여부=[False, False]. 차이 미검출을 동등성으로 해석하지 않는다.
- ResNet18: 두 feature DP bank 모두 공개전용 대비 AUROC 구간 전체 양수=True; AP 구간 전체 양수=False.
  Feature DP1의 각 gradient bank 대비 AUROC 구간 전체 양수 여부=[False, False]. 차이 미검출을 동등성으로 해석하지 않는다.
  Feature DP2의 각 gradient bank 대비 AUROC 구간 전체 양수 여부=[False, False]. 차이 미검출을 동등성으로 해석하지 않는다.

두 noise의 평균/범위/표본SD는 evaluation/result.json의 기술통계다. 잡음 모집단의 안정성, 앙상블 성능이나 synthesis 초기화 변동을 추정한 것으로 쓰지 않는다. 결과별 논문 기여 자동 전환은 수행하지 않았다.

## 검증·비용

- 변경 경로: P target 기존 fixture exact, loader loss/feature gradient exact, 잘못된job6개 거절. 기존 이미지 gradient·저장·재개 검증은 재사용.
- 각 bank의 실제 초기 renderer/optimizer/RNG/활성수준 상태가 기존 seed101과 exact. 최종128PNG, labels/virtual pairs/hash, encoder/target 불변 및 실제 합성 parameter 갱신 검증 통과.
- 환자 bootstrap 독립 sklearn 검산 최대오차 1.11e-16.
- 공개전용 합성·저장 34.99분, DP2 35.11분, 새 평가 10.97초.
- 요청 착수부터 본 보고서 생성까지 80.43분. 준비 실행 자체 7.49초(코드 연결·문서 작성 시간 별도 포함).
- main DINO image forward409600/backward204800; 새 PNG receiver forward512. 추가 profile/계수 탐색/초기화 반복 없음.
- GPU peak allocated: 공개 2.584GiB, DP2 2.584GiB.

## 보호와 보존

- 새 보호 요약은1개, 각 DP 산출물은고정된 공개 구성에 대해 Q의 환자-DP(8,1e-5). 공개전용은Q 접근에 따른 추가 비용0.
- 여섯 요약을 모두 같은Q에 대해 공동 공개하면 basic composition 상한(48,6e-5). 이전 비DP 개발·V 예측·평가 보고서 전체가 보호됐다는 뜻은 아니다.
- random coins/실제noise/unnoised query는 저장하지 않음. 보호된 noisy query는 안전한 같은-release 후처리를 위해 보존.
- protected_png는 이번 DP2 PNG 공개 후보, public_png는 공개자료 전용 PNG. 상위 실행폴더·검산·예측은 사적 내부자료이며 외부 업로드 없음.
- 기존 코드·target·bank·결과는 해시로 보존 검증. Expert/Reserved/확인용 수신자 미개방.

## 완료와 남은 범위

이번두-bank 개발 묶음은 완료. 추가 DP noise·seed·학습량·receiver·interaction 분석은 실행하거나 예약하지 않았다. 독립 확인, 강한 선행 대비 추가 가치와 논문 기여는 이번 완료와 별개다.

[실행 계약](RECEIVER_FEATURE_FOLLOWUP_CONTRACT_20260924.md)
[내부 실행 산출물](../../code_working/_reports/receiver_feature_followup_20260924_v1)

## 저장된 학습 trace의 목표 정합

끝값은 마지막200회 update 직전의 float 정합 손실이다. 최종PNG 재추출 loss나 제한 전 실자료 분포 복원으로 해석하지 않는다.
- PUBLIC: 0.03208394898 -> 2.207771307e-05, 감소율 99.9312%.
- DP8: 0.02923882996 -> 2.187159368e-05, 감소율 99.9252%.
