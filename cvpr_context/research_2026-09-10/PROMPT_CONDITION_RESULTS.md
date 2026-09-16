# t140 조건 대조 결과

작성일: 2026-09-15. 실제 측정·원시 산술 검산·표 수치 대조를 완료한 고정 조건 진단이다. 원본 측정·검산·분석 파일은 보존했다. 연구 2단계 전체의 완료나 통과 판정을 뜻하지 않는다.

현재 측정에서는 **null 조건과 실제 학습의 weak-label 조건이 다르다는 점이 기존 검출 약함의 원인이었다는 근거를 찾지 못했다.** 사전에 정한 prompt 간 AUC 차이 24개의 95% paired bootstrap 신뢰구간이 모두 0을 포함한다. 이는 조건 효과가 정확히 0이라는 증명이나 다른 관측 방식에서도 효과가 없다는 결론이 아니다.

조건을 바꿨을 때 절대 loss가 달라지는 것과, member 환자를 nonmember 환자보다 높은 점수로 배열하는 것은 다른 질문이다. 이 진단의 판단 대상은 후자의 분리력이다. 이번 결과만으로 전체 MoFit 점수나 다른 공격의 성공 가능성을 판정하지 않는다.

## 무엇을 측정했는가

고정된 NIH CXR14 개발 환자 120명(fit 80명, selection 40명)의 E 영상 2장과 U 영상 2장, 총 480장에 대해 세 모델과 세 조건의 denoising MSE를 측정했다.

| 항목 | 고정 내용 |
|---|---|
| target | 기존 model_1, model_2의 step 1000 LoRA checkpoint |
| base | 동일한 공개 배포 base UNet의 fp16 weight artifact를 FP32로 올린 모델. 추가 NIH 학습·LoRA 없음 |
| 영상·조건 입력 | 기존 FP16 VAE posterior-mode latent와 FP16 text hidden을 FP32로 올려 재사용 |
| 모델 계산 | FP32, eval mode, batch 1, 가중치 동결, gradient checkpointing 비활성 |
| timestep·noise | t=140, 영상별 고정 CPU FP32 epsilon 1개. 같은 영상의 모든 모델·조건에서 공유 |
| 조건 | null, generic, weak_label |
| 환자 점수 | 같은 환자의 해당 시나리오 영상 2장의 loss 평균 |
| 측정 비용 | 1,440 model/image packet, 4,320 forward, backward 0. 새 VAE·text encoder 계산 0 |
| 평가 구분 | selection 40명 결과를 비교하고, fit 80명 AUC는 기술값만 별도 보존. 새 공격 fitting·prompt 선택 없음 |

null은 빈 문자열, generic은 “a frontal chest radiograph”다. weak_label은 cache.train_prompts[image_id]가 가리키는 원래 영상별 NIH finding-label template과 그 캐시 hidden이다. 이것은 새 VLM caption이나 환자 ID를 포함한 prompt가 아니다.

E는 train_candidate로 고정된 영상 집합이라는 **역할 이름**이다. 해당 target에서 member인 환자의 E 영상만 실제 학습에 사용됐으며, nonmember 환자의 E 영상은 사용되지 않았다. U 영상은 두 target 모두에 개별 영상으로 사용되지 않았다. U에서 patient member는 같은 환자의 다른 E 자료가 해당 target 학습에 사용됐다는 뜻이다.

따라서 member target의 E에 대한 weak-label은 실제 사용된 학습 조건과 대응하지만, U와 nonmember target의 E에 대해서는 **그 영상의 라벨로 만든 가상 학습 template**이다. 모든 weak-label 조건을 “실제 사용된 prompt”라고 부르면 부정확하다.

모델 1의 positive label은 A 배정 환자, 모델 2의 positive label은 B 배정 환자다. 이는 실제 checkpoint 노출 원장과 대조한 추가 NIH 학습 참여 라벨이다. 두 target은 초기화 등 조건을 공유하므로 독립적인 두 학습 seed 반복으로 해석하지 않는다.

## 환자 단위 AUC

두 점수 모두 값이 클수록 member라는 방향을 사전에 고정했다.

- negative_target_loss: 영상 2장 평균의 음수, 즉 −L_target.
- base_minus_target_loss: 같은 영상·prompt·epsilon에서 L_base − L_target의 영상 2장 평균.

각 셀은 selection AUC [95% CI]다. 표시는 소수 넷째 자리로 반올림했으며, [전체 30개 metric CSV](prompt_condition_results_artifacts/v1/metrics_30.csv)에 원래 정밀도의 CI와 fit 기술 AUC를 보존했다. AUC가 0.5보다 작다는 이유로 결과를 본 뒤 점수 방향을 뒤집지 않았다.

| 모델 | 시나리오 | 점수 | null | generic | weak_label |
|---|---|---|---|---|---|
| model_1 | E | −L_target | 0.3700 [0.2100, 0.5500] | 0.3850 [0.2175, 0.5725] | 0.3825 [0.2175, 0.5676] |
| model_2 | E | −L_target | 0.6350 [0.4574, 0.7950] | 0.6200 [0.4350, 0.7826] | 0.6200 [0.4350, 0.7850] |
| model_1 | E | L_base − L_target | 0.5125 [0.3325, 0.6950] | 0.5175 [0.3375, 0.6975] | 0.5350 [0.3525, 0.7176] |
| model_2 | E | L_base − L_target | 0.4825 [0.3000, 0.6625] | 0.4950 [0.3100, 0.6751] | 0.4900 [0.3075, 0.6700] |
| model_1 | U | −L_target | 0.4600 [0.2800, 0.6326] | 0.4725 [0.2925, 0.6500] | 0.4800 [0.2999, 0.6576] |
| model_2 | U | −L_target | 0.5425 [0.3725, 0.7200] | 0.5400 [0.3649, 0.7226] | 0.5375 [0.3599, 0.7175] |
| model_1 | U | L_base − L_target | 0.4800 [0.3000, 0.6725] | 0.4925 [0.3000, 0.6850] | 0.5100 [0.3224, 0.6976] |
| model_2 | U | L_base − L_target | 0.5350 [0.3399, 0.7175] | 0.5250 [0.3375, 0.7100] | 0.5100 [0.3200, 0.7001] |

E의 단순 −loss는 model_1에서 null/generic/weak-label 순서로 0.3700/0.3850/0.3825, model_2에서 0.6350/0.6200/0.6200이었다. weak-label로 바꾸는 것만으로 두 모델의 분리력이 일관되게 개선되는 양상은 보이지 않았다. base 차감 후 E AUC도 각각 0.5125/0.5175/0.5350, 0.4825/0.4950/0.4900이었다.

## Prompt 차이와 불확실성

selection의 A 20명과 B 20명 안에서 각각 복원 추출한 2,000개 환자 bootstrap 표본을 사용했다(seed 260914). 같은 재표집 인덱스를 두 target, 두 시나리오, 모든 prompt와 점수에 공유하여 **paired ΔAUC**를 계산했다. 신뢰구간은 percentile 2.5%–97.5%다.

사전 비교는 각 모델·E/U·점수에서 weak_label − null, generic − null, weak_label − generic의 세 가지이며, 총 24개다. **24개 모두 CI가 0을 포함한다.** [전체 paired 차이 24개 CSV](prompt_condition_results_artifacts/v1/prompt_contrasts_24.csv)에 모든 점추정과 CI를 보존했다.

핵심 질문에 해당하는 E의 weak_label − null 결과는 다음과 같다.

| 모델 | 점수 | ΔAUC [95% paired CI] |
|---|---|---|
| model_1 | −L_target | +0.0125 [−0.0301, +0.0550] |
| model_2 | −L_target | −0.0150 [−0.0650, +0.0375] |
| model_1 | L_base − L_target | +0.0225 [−0.0751, +0.1176] |
| model_2 | L_base − L_target | +0.0075 [−0.0825, +0.1051] |

이 CI는 고정된 target과 이미 여러 진단에서 관측한 개발 환자에 조건부인, 다중비교 보정 전의 구간이다. 새 test set, 학습 seed 불확실성을 포함한 구간, 동등성 검정, 실용적인 낮은 FPR 검증으로 해석하지 않는다.

## 공개 base의 A/B 배정 진단

아래는 −L_base로 **A 배정 환자와 B 배정 환자를 구분하는 AUC**다. 공개 base의 학습 membership AUC가 아니다. 추가 NIH 학습에 대한 base membership은 0이지만, 공개 모델의 원래 사전학습에 해당 환자 또는 영상이 포함됐는지는 알 수 없다.

| 시나리오 | 조건 | A 배정 positive AUC [95% CI] |
|---|---|---|
| E | null | 0.3975 [0.2250, 0.5750] |
| E | generic | 0.3950 [0.2200, 0.5825] |
| E | weak_label | 0.3975 [0.2249, 0.5825] |
| U | null | 0.5000 [0.3150, 0.6776] |
| U | generic | 0.4875 [0.3075, 0.6675] |
| U | weak_label | 0.4800 [0.3000, 0.6550] |

L_base − L_target에는 한 환자의 포함 효과 외에도 전체 NIH 추가 학습에 따른 domain 적응과 다른 학습 자료의 효과가 함께 들어간다. 이를 특정 환자의 인과적인 학습 영향이나 정확한 개인별 누출량으로 등치하지 않는다. base 쪽의 배정 AUC만으로 현재 결과의 원인을 확정하지도 않는다.

## 검산과 실제 비용

원시 검산은 1,440 packet의 4,320 prediction MSE를 저장된 FP32 prediction과 epsilon의 FP64 잔차 계산으로 대조했다. 이미지별 seed를 1,440회 재생성하여 서로 다른 epsilon은 480개임을 확인하고, 동일 영상의 모델·조건 간 입력 일치, E/U 실제 노출, checkpoint·cache·source hash 및 모델 상태 불변 기록을 검사했다.

분석에서는 pair-credit AUC와 sklearn AUC, 표본빈도 가중 bootstrap과 직접 인덱스 재표집 bootstrap을 대조했다. 30개 metric 각각의 selection 20×20 비교쌍에서 GPU MSE 기반 점수와 저장 prediction의 FP64 잔차 기반 점수 사이의 순위/동점 credit 차이는 **0개**였다. 이것은 독립 UNet 재실행이나 FP64 모델 추론을 뜻하지 않는다.

| 작업 | 실제 시간 |
|---|---:|
| 기존 GPU 측정 실행 | 400.4875초 (약 6분 40초), 4,320 F / 0 B |
| 원시 CPU 검산 | 15.1491초 |
| scalar 분석·두 계산 경로 대조 | 0.3733초 |
| 이 보고서·CSV 생성 | 추가 모델 호출·fitting 없음 |

서로 다른 점수의 단독 계산 비용이 같다는 주장은 하지 않는다. base 차감 점수는 base 응답도 필요하며, 이번 4,320회 예산은 모든 조건과 모델을 한 번씩 측정해 재사용한 전체 비용이다.

## 현재 해석 범위

이 관측은 t=140과 영상별 epsilon 1개에서의 고정 loss 진단이다. VAE posterior-mode와 text hidden은 기존 FP16 캐시를 FP32로 올린 것이며, MoFit의 fresh FP32 VAE posterior sample, 픽셀 1,000회 및 전체 text embedding 300회 최적화, 최종 점수를 재현한 측정이 아니다.

또한 이것은 latent·timestep·noise와 denoiser 반응을 사용할 수 있는 white-box 연구 진단이다. weak-label 조건에는 후보 영상의 finding-label metadata가 추가 정보로 들어간다. 환자 사진만 가진 제한 접근 검증자가 언제나 이 정보를 확보하거나 같은 질의를 할 수 있다고 가정하지 않는다.

학습 중 null dropout이 없었다는 사실만으로 unconditional 반응이 변하지 않아야 한다는 결론은 나오지 않는다. 공유 UNet 가중치의 업데이트가 여러 조건의 응답에 영향을 줄 수 있다. 이번 결과가 지지하는 제한된 판단은 **“이 고정 loss 관측에서 prompt를 맞추면 기존 검출 약함이 해소된다”는 설명을 뒷받침하지 못했다**는 것이다. 다른 timestep·noise·관측 통계에서의 신호, 전체 MoFit 성능, 환자 단위 누출의 부재나 U 연구의 불가능성까지 결론 내리지 않는다.

## 기록과 재검토 자료

- [원래 정밀도의 전체 metric 30개](prompt_condition_results_artifacts/v1/metrics_30.csv)
- [전체 paired prompt 차이 24개와 CI](prompt_condition_results_artifacts/v1/prompt_contrasts_24.csv)
- [분석 JSON 사본](prompt_condition_results_artifacts/v1/analysis.json)
- [원시 검산 JSON 사본](prompt_condition_results_artifacts/v1/raw_verification.json)
- [추출 실행 기록 사본](prompt_condition_results_artifacts/v1/extraction_execution.json)
- [원본 경로·source/code·artifact SHA256 manifest](prompt_condition_results_artifacts/v1/manifest.json)

원본 위치는 code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/prompt_loss_t140_v1이다. 원본 execution.json의 COMPLETE_RAW_PENDING_INDEPENDENT_VERIFICATION은 변경하지 않았으며, 별도 verification.json의 PASS와 analysis_v1/analysis.json의 PASS가 후속 완료 근거다.

분석·검산 코드 SHA256은 7cf4b95d46636369dd0c0a63629e7cc3436bf1fda332567a75985342b7d18b63이다. 이 문서와 artifact export는 완료된 결과를 정리한 것이며 새로운 실험, prompt 선택, 학습, 연구 2단계의 자동 통과 판정을 추가하지 않는다.
