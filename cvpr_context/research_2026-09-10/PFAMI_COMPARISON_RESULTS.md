# 1번 안의 PFAMI 고정 비교 결과

2026-09-14. **PFAMI-style 정규화가 단순 손실 차이보다 환자 판별을 개선한다는 근거를 확보하지 못했다.** 핵심 U의 주점수 AUC는 모델 1/2에서 0.3550/0.6575이고, 정규화에 따른 ΔAUC는 +0.0125/−0.0150으로 두 구간 모두 0을 포함한다. 계산과 통계의 독립 검산은 통과했다.

이 작업은 큰 순서의 **1번, 기존 방법이 무엇을 잡고 놓치는지 확인하는 단계**에 속한다. 앞선 “첫 단계 완료”는 그 안의 원문·코드 대조 완료를 뜻했다. 이번에는 고정 비교 실행까지 완료했으며, 새 방법의 설계·우위 입증 또는 1번 전체의 완료로 표시하지 않는다. 기존 R 확대 보류는 유지한다.

[실행 전 계약](PFAMI_COMPARISON_PROTOCOL.md) · [원문·코드 대조](METHOD_PREMISE_AUDIT_2026-09-14.md) · [앞선 SecMI 진단](U_BASELINE_SCREEN.md)

## 이번 비교가 답한 질문

U의 관측 영상 자체는 두 모델 모두 학습하지 않았다. 해당 모델의 member 환자는 다른 영상이 학습에 포함되고, nonmember 환자는 다른 영상도 포함되지 않는다. 이 조건에서 원본/crop 손실의 **상대 변화**가 **단순 차이**보다 환자 순위를 잘 구분하는지 확인했다. 같은 손실 관측값으로 만든 점수 두 가지의 비교다. 개선이 나와도 새로운 정보원이나 암기 메커니즘을 증명하는 것은 아니다.

기존 selection 40명(A/B 각 20명), 환자별 E 2장·U 2장, 기존 두 의료 LoRA 모델을 사용했다. E 후보 영상은 해당 모델의 member 환자일 때 실제 학습에 포함되고 nonmember 환자일 때 제외되는 별도 진단이다. 환자 수는 40명이며 두 모델 결과를 80명의 독립 표본으로 합치지 않는다. fit 환자 14393은 전처리·시간 확인에만 사용했고 아래 통계에서 제외했다.

## 핵심 U 결과

| 점수 | 모델 1 AUC [95% 구간] | 모델 2 AUC [95% 구간] |
|---|---|---|
| 상대 변화 — 주점수 | 0.3550 [0.1775, 0.5225] | 0.6575 [0.4900, 0.8301] |
| 손실 차이 — 같은 비용 대조 | 0.3425 [0.1824, 0.5150] | 0.6725 [0.5025, 0.8325] |
| 음의 원본 손실 — 보조 | 0.4150 [0.2425, 0.6050] | 0.5875 [0.3974, 0.7600] |
| 공개 base 상대 변화 차감 — 보조 | 0.4375 [0.2525, 0.6250] | 0.5775 [0.3924, 0.7550] |

실행 전에 고정한 주비교는 AUC(relative)−AUC(difference)다. 두 점수 모두 이미지당 원본/crop × 10시점 = 20 forward를 사용한다.

| 모델 | ΔAUC | 환자 bootstrap 95% 구간 |
|---|---:|---|
| 모델 1 | +0.0125 | [−0.0700, +0.0925] |
| 모델 2 | −0.0150 | [−0.1000, +0.0650] |

**추가 이득 미확인**이 정확한 판정이다. 주점수의 두 AUC 구간도 모두 0.5를 포함하고, 고정된 member 방향의 일관된 판별을 확보하지 못했다. 결과를 본 뒤 부호를 뒤집지 않았다.

모델 2의 difference AUC **0.6725 [0.5025, 0.8325]**는 실제 관측이며 생략하지 않는다. 다만 반대 모델에서는 0.3425이고, 이미 사용한 개발자료에서의 탐색적 결과다. 이를 정규화 우위, 두 모델에서의 재현 또는 최종 환자 감사 성공으로 처리하지 않는다.

## E 진단 결과

| 점수 | 모델 1 AUC [95% 구간] | 모델 2 AUC [95% 구간] |
|---|---|---|
| 상대 변화 — 주점수 | 0.4450 [0.2625, 0.6225] | 0.5500 [0.3750, 0.7300] |
| 손실 차이 — 같은 비용 대조 | 0.4475 [0.2724, 0.6275] | 0.5600 [0.3800, 0.7400] |
| 음의 원본 손실 — 보조 | 0.4650 [0.2950, 0.6476] | 0.5425 [0.3625, 0.7125] |
| 공개 base 상대 변화 차감 — 보조 | 0.4300 [0.2525, 0.6125] | 0.5725 [0.3949, 0.7475] |

E의 정규화−차이 ΔAUC는 모델 1 **−0.0025 [−0.0975, 0.0875]**, 모델 2 **−0.0100 [−0.0975, 0.0675]**다. E의 결과로 U 결론을 대신하지 않는다.

## 실행 조건과 출처

- [PFAMI 공식 저장소](https://github.com/wjfu99/MIA-Gen)의 DDPM loss/statistic 원본 AST를 재사용했다. 원저자 64px 경로의 점수를 256px 의료 latent diffusion에 적용한 **PFAMI-style adaptation**이며 원 논문의 SD 실험 전체 재현이 아니다.
- 원본 FP16 VAE posterior-mode cache를 유지했다. crop은 canonical 256px 영상의 tensor 중앙 207px를 bilinear, antialias=True로 256px로 되돌려 같은 FP16 VAE mode로 인코딩했다. latent 자체를 crop하지 않았다.
- FP32 UNet·loss, 실제 epsilon scheduler, 일반 흉부 X-ray prompt, timestep 0/50/…/450을 고정했다. 원본과 crop, 각 시점, target과 base는 독립 noise이고 두 target 모델의 동일 셀에만 같은 noise를 사용했다.
- 주점수는 시점별 `(L_crop−L_original)/L_original`의 평균, 이후 환자 영상 2장의 평균이다. member=1이며 큰 값이 member 방향이다. 손실 평균끼리 나누는 식과 다르다.
- target-only는 원저자 코드의 calibration=False 경로다. 원저자 기본값 True와의 차이를 명시한다. 공개 base는 별도의 disjoint 의료자료로 학습한 reference generator가 아니므로 base 차감은 보조 진단이다.
- 분모 clipping/epsilon을 추가하지 않았다. 원본 손실 최솟값은 0.06527144이었다.
- denoiser·latent에 접근한 실행이다. 완성 이미지만 반환하는 일반 생성 API의 black-box 감사로 부르지 않는다. 추론 대상은 이번 의료 미세조정 참여이며 공개 base의 사전학습 membership은 알 수 없다.

## 독립 검산과 비용

독립 검산은 **PASS_INTEGRITY_RESIDUAL_AND_STATISTICAL_RECOMPUTATION**이다.

- FP32 잔차 파일 480개에서 MSE 9,600개를 CPU float64로 재계산하고 noise seed/hash를 확인했다. 최대 MSE 절대 차이 8.43379e-8, 상대 차이 1.53357e-7다.
- 160개 환자·모델·조건 기록의 점수 640개, sklearn AUC 16개, 별도 구현한 환자 bootstrap과 paired 비교 8개를 검산했다.
- float64 잔차 재계산으로 AUC·구간·판정은 바뀌지 않았다. 이는 저장 잔차의 산술 검증이며 GPU 전체를 FP64로 재실행하거나 독립 모델로 재현한 것이 아니다.
- 실제 E/U 노출, cohort·cache·source·동결 코드 hash 및 보호 원본 11개를 확인했다. 실행기는 모델별 가중치 불변·gradient 없음도 기록했다.
- 본 측정은 160개 고유 영상, target 320건 + base 160건 = 480건이다. **target 6,400F + base 3,200F = 9,600F/0B**, crop VAE 160장 인코딩은 별도다. 음의 원본 손실은 독립 실행 시 10F이므로 같은 비용의 우위로 주장하지 않는다.

시간은 정의·연결 검토 15–25분, fit1 확인 1–3분, 본 GPU 실행 17–20분, 검산·정리 5–10분을 사전에 안내했다. 정의·연결 검토는 약 21분이었다. 본 실행기 실측은 **895.17초(14분 55초)**, 독립 CPU 검산은 **4.89초**였다. 실행기 시간은 로딩·인코딩·내부 저장을 포함하며 순수 GPU kernel 시간과 구분한다. 전체 작업의 시작은 21:05 KST이고 종료 시각은 WORKLOG에 기록한다.

fit1 벤치마크 v1은 계산 240F를 마친 뒤, root가 검산 파일의 최종 저장 전에 실행을 시작한 탓에 코드 변경 감지가 종료를 거부했다. 해당 기록을 [실패 기록](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/pfami_benchmark_v1/failure.json)으로 보존했다. 코드를 모두 동결한 v2는 38.00초에 완료하고 독립 검산을 통과했다. 두 실행의 손실 240개·영상 점수 12개는 정확히 같았으며 원본 4장 latent 재인코딩도 기존 cache와 정확히 일치했다. 두 벤치마크 합계 **480F/0B**는 본 측정 9,600F와 별도 계상한다.

## 결론의 범위와 종료

selection 40명은 앞선 개발에서도 사용한 자료다. A/B 층화 환자 bootstrap 2,000회(seed 260914)의 구간은 고정된 모델·개발자료에 조건부인 탐색적 구간이며, 이전 방법 선택에 대한 보정 또는 최종 시험의 확증 구간이 아니다. 두 모델은 초기화·배경 환자·일부 noise를 공유하고 여러 환자의 포함 여부를 함께 교환했다. 독립 학습 seed 두 번이나 한 환자만 바꾼 DP 이웃 비교가 아니다.

이번 결과로 **정규화의 추가 이득을 주장할 수 없다.** 두 모델의 반대 방향 원인을 환자 특징이나 split bias로 확정할 수도 없다. SecMI와 이번 고정 PFAMI-style 결과만으로 전체 MIA의 실패, 누출 부재, 연구 불가능 또는 추가 학습의 필요성을 결론 내리지 않는다. 시점별 noise 한 번의 고정 예산 진단이라는 범위도 유지한다.

이번 고정 측정은 여기서 닫았다. 결과에 맞춘 부호·crop·시점 변경, R 튜닝, 새 환자 평가, 추가 target 학습, calibration/test 실행은 하지 않았다. **새 방법이 강한 기존 방법보다 좋아질 이유는 아직 실험으로 뒷받침되지 않았다.** 원시 결과와 이 한계를 다음 설계 판단의 근거로 남긴다.

## 원시 기록

- [동결 실행 protocol](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/pfami_v1/protocol.json)
- [실행 보고·계산량·시간](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/pfami_v1/report.json)
- [원시 이미지 점수·시점별 손실](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/pfami_v1/image_scores.json)
- [환자 점수·AUC·paired 구간](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/pfami_v1/analysis.json)
- [독립 검산](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/pfami_v1/verification.json)
- [잔차별 float64 재계산](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/pfami_v1/residual_recomputation.json)
- [fit1 독립 검산](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/pfami_benchmark_v2/independent_benchmark_residual_check.json)

동결 contract SHA256: `cedbd07e45c20ba0c81e437976a6efb1d446dc280f731faf72afc7b66c7e1aca`  
실행 protocol SHA256: `3d31da3b8e303736a370dc9a6d5bc0e524d5750635bae8c37c2326aec42748cc`  
image_scores SHA256: `c439ee7faf68b2e6d31dc495fa4861b4d95e854c90a67436326f2c66019eaafa`
