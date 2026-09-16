# 2번 — CDI 기존 특징과 환자 집계의 실제 U 비교

2026-09-15 KST. **기존 meanmax 판별기는 두 사진의 특징 차이를 실제 순위에 활용했다. 그러나 그 구분과 실제 환자 학습 참여의 관계는 아직 분리되지 않았다.** U 비교·순위 분해·실제 base 대조·평균/차이 항 제거와 독립 검산까지 완료했다. [고정 네 단계](RESEARCH_FRAMEWORK.md)의 2번이며, 새 공격이나 3번 설계의 성공이 아니다.

U fit80/selection40, 환자당 학습되지 않은 사진2장, 기존 두 target에서480건을 실제 계산했다. 18개 지정 변형과 기존 DINO 대조군을 모두 보고한다. extraction·12,480개 특징·저장 예측/통계의 독립 검산이 통과했다.

## 가장 먼저 읽을 결과

| 방법 | 모델1 selection AUC [95% 구간] | 모델2 selection AUC [95% 구간] |
|---|---|---|
| SecMI 점수 평균 | .4300 [.2500,.6150] | .5600 [.3850,.7375] |
| CDI 영상 scorer→평균, fit 내부 C 선택 | .4350 [.2625,.6101] | .5475 [.3650,.7225] |
| 환자 mean+max52→LR, fit 내부 C 선택 | .5175 [.3275,.7000] | .6050 [.4200,.7875] |
| CDI 영상 scorer→평균, 고정 C=1 | .4475 [.2749,.6250] | .5475 [.3650,.7225] |
| 환자 mean+max52→LR, 고정 C=1 | .6325 [.4600,.7925] | .6700 [.4950,.8301] |
| NO objective 추가 mean+max54, 고정 C=1 | .6150 [.4400,.7800] | .6950 [.5250,.8550] |
| 기존 DINO encoder 관계 특징 | .5475 [.3724,.7325] | .5475 [.3724,.7325] |

**모든 결과가 음성이었다고 말하면 틀린다.** 고정 C의 표준 환자 집계 개선과 NO27 모델2의 AUC 구간 하한>.5는 실제 관측이다. 동시에 이 선택40명은 반복 사용한 개발자료이며 두 target은 독립 반복이 아니다. 구간은 고정된 scorer·target에 조건부인 탐색적 구간이다.

## 원래 정한 비교를 그대로 판정

| 대조 | 모델1 ΔAUC [구간] | 모델2 ΔAUC [구간] | 두 target 평균 Δ [구간], 기술 통계 |
|---|---|---|---|
| **주 비교: CDI image mean tuned − SecMI** | +.0050 [−.2375,.2450] | −.0125 [−.2725,.2251] | −.00375 [−.1725,.1588] |
| **주 비교: patient meanmax52 tuned − CDI image mean tuned** | +.0825 [−.0126,.1825] | +.0575 [−.0750,.1975] | +.0700 [−.03625,.1750] |
| 보조: patient meanmax52 fixed − CDI image mean fixed | +.1850 [.02994,.3525] | +.1225 [.0000,.2575] | +.15375 [.02625,.28625] |
| NO objective 추가, fixed | −.0175 [−.0650,.0250] | +.0250 [−.0475,.0950] | +.00375 [−.0425,.04628] |
| NO objective 추가, tuned | −.0025 [−.0350,.0300] | +.0425 [−.0150,.1125] | +.0200 [−.0150,.0625] |

CDI 영상 scorer가 기존 SecMI의 약한 결과를 해결했다는 주 비교 근거는 없다. 표준 mean+max 집계의 고정 C 개선은 남지만, 사전 주 비교인 fit-only 튜닝 정책에서는 구간이0을 포함한다. 고정 C만 나중에 주 비교로 올리지 않는다.

고정 meanmax52의 fit AUC는 .9600/.969375이며 selection에서는 .6325/.6700이다. fit 내부 5fold의 **patient log-loss**로 선택한 C는 .01/.1이고, 선택 후 selection AUC는 .5175/.6050이다. CV가 AUC를 직접 최적화한 것은 아니므로 AUC 하락을 구현 오류로 부르지 않는다. 표본·학습 복잡도·목적함수 선택의 영향은 미분리다.

NO의 최적화 objective와 공개 코드 final feature의 차이는 앞선 [kernel](CDI_KERNEL_RESULTS.md)에서 실제 확인했다. 그러나 objective를 추가한 54D와 기존52D의 ΔAUC 구간은 모두0을 포함한다. 반환 경로 차이가 현재 병목을 설명했다는 주장은 아직 성립하지 않는다. NO27의 모델2 AUC가 높은 관측과, NO27 자체의 추가 이득 검증은 다른 질문이다.

## 동일 scorer로 모델만 바꾼 진단

각 target의 fit 자료로 학습한 scaler·LR·C를 그대로 고정하고, 같은 selection 환자의 두 target 특징을 넣었다. 아래 Δ는 참여하는 target 점수−참여하지 않는 target 점수이다. 이 분석에서만 두 모델을 함께 사용하며 실제 공격 입력으로 추가하지 않는다.

| 고정한 scorer | 평균 참여방향 Δ [구간] | 양수 환자 | A 집단 평균 / B 집단 평균 |
|---|---|---:|---|
| 모델1 fit, meanmax52 C=1 | −.044108 [−.096698,.012050] | 15/40 | −.183191 / +.094975 |
| 모델2 fit, meanmax52 C=1 | +.013161 [−.042252,.072512] | 21/40 | +.112172 / −.085849 |
| 모델1 fit, meanmax52 tuned | −.008342 [−.019142,.002848] | 17/40 | −.037236 / +.020553 |
| 모델2 fit, meanmax52 tuned | +.018385 [−.014868,.052340] | 25/40 | +.089097 / −.052326 |

전체20개 scorer의 참여방향 평균 Δ 구간은 모두0을 포함한다. 모델1에서 fit한10개는 A 평균 음수/B 평균 양수, 모델2에서 fit한10개는 A 양수/B 음수다. 고정 scorer마다 평균 출력이 한 target 쪽으로 움직이는 성분이 있다는 관측이다. 모델 반응 자체가 없었다고 말할 수 없으며, 그 반응을 일관된 개인 참여 효과라고도 할 수 없다.

**이 부호 패턴만으로 AUC 개선을 cohort bias 때문이라고 확정하지 않는다.** 모든 환자의 점수에 같은 상수를 더하는 것은 AUC를 바꾸지 않기 때문이다. 아래 후속은 공통 점수와 target 변화가 실제400개 A/B쌍의 순위를 각각 어떻게 바꿨는지 계산한 것이다. 두 target은 전체 학습 환자 집단을 교환했으므로 개인 한 명의 인과효과는 식별하지 못한다.

## 후속 CPU 순위 분해: 개선을 참여 반응으로 설명할 수 있는가

위 결과를 본 뒤 수행한 **사후 기술적 분석**이다. scorer 하나를 고정하고 동일 환자의 두 모델 점수를 s1/s2, 공통 점수를 c=(s1+s2)/2로 정의했다. 각 scorer의400개 A/B쌍에서 동점은 반점으로 처리하고, 자기 target의 AUC를 ‘c의 AUC + 자기 target에서 달라진 순위의 기여’로 정확히 나눴다. 20개 scorer 모두8000쌍을 계산했으며 새 fitting/GPU는0이다.

| meanmax52 − image mean 비교 | 관측한 AUC 이득 | 공통 점수 c에서의 이득 | target 변화의 상대 기여 |
|---|---:|---:|---:|
| 모델1 tuned | +.0825 | +.0875 | −.0050 |
| 모델1 fixed | +.1850 | +.1925 | −.0075 |
| 모델2 tuned | +.0575 | +.0875 | −.0300 |
| 모델2 fixed | +.1225 | +.1500 | −.0275 |

모든 행은 **첫 열 = 둘째 열 + 셋째 열**인 정확한 순위 기여 분해다. 네 비교 모두 공통 점수의 이득이 더 컸고, target 변화의 상대 기여는 음수였다. 따라서 이번 표준 집계의 개선을 “참여에 따른 target 변화가 환자 순위를 더 잘 개선해서 생긴 이득”으로 설명할 근거는 없다.

이는 target 변화의 기여 자체가 항상0 또는 음수라는 뜻은 아니다. 모델2 meanmax52의 자기 target 대비 c 기여는 fixed+.0075/tuned+.0050이고, 대응 image mean은+.0350이다. 둘 사이의 **추가 이득에 대한 기여 차이**가 음수인 것이다. 이 상대 기여 차이의 조건부 구간은 네 경우 모두0을 포함하므로 모집단에서 음수라고 확정하지 않는다.

또한 c는 두 학습 모델의 수학적 평균이지 학습 효과가 제거된 nuisance나 데이터 편향으로 식별된 변수가 아니다. **현재 확인한 것은 AUC 개선을 참여 반응 개선으로 연결한 설명의 공백**이다. 그 공통 구분력이 원래 영상 특성에서 왔는지, 학습 모델들이 공유하는 반응에서 왔는지는 추가로 분리해야 한다.

[분해 protocol](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/cdi_u_cohort_v1/rank_decomposition_v1/protocol.json) · [20개 scorer와4개 대조](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/cdi_u_cohort_v1/rank_decomposition_v1/summary.json) · [8000쌍의 정수 순위 증거](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/cdi_u_cohort_v1/rank_decomposition_v1/pairs.json) · [별도 독립 검산](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/cdi_u_cohort_v1/rank_decomposition_v1/independent_verification.json)

분해 내부 약.52초, 독립 검산 .618초다. 원래 저장 예측에서 c/d와20개 scorer·8000쌍의 모든 credit을 재계산했고, sklearn AUC와 별도 직접 bootstrap으로구간·4개 항등식을 대조했다. 코드/입력 hash는 일치했다. 준비·구현·검산·기록 시간과 CPU 산술 시간을 구분한다.

## 완료한 대조: 의료 추가학습 전 base

같은 선택40명 U80장을 **의료 추가학습 전의 실제 pretrained base UNet**에 넣고, 이미 고정된20개 target scorer를 그대로 적용했다. 새로운 scorer fitting·C 선택은 없었다. 실제 추출 420.510초(7분),4,594F/1,314B였고,80장·2,080개 특징·noise640개·20개 scorer의800개 예측/8,000쌍 순위 검산을 통과했다.

base에서는 이번 의료 추가학습 membership이 모두0이므로 이때의 A/B AUC를 ‘base membership 공격 성능’이라고 부르지 않는다. 기존 A/B 배정을 구분하는 **대조 지표**다. base의 원래 사전학습 데이터 포함 여부는 미확인이다. scorer 자체는 target의 fit80 정보로 학습했으므로 전체 방법이 추가학습 정보 없이 가능하다는 주장도 아니다.

| 고정 scorer | base A/B AUC | 자기 target AUC | target−base [95% 구간] |
|---|---:|---:|---|
| 모델1 image mean, fixed | .4975 | .4475 | −.0500 [−.2276,.1300] |
| 모델1 meanmax52, fixed | .5550 | .6325 | +.0775 [−.0976,.2625] |
| 모델2 image mean, fixed | .6175 | .5475 | −.0700 [−.2526,.1025] |
| 모델2 meanmax52, fixed | .5700 | .6700 | +.1000 [−.0751,.2850] |
| 모델1 meanmax52, tuned | .4250 | .5175 | +.0925 [−.1025,.2900] |
| 모델2 meanmax52, tuned | .5250 | .6050 | +.0800 [−.1325,.2900] |

전체20개 base A/B AUC 구간은 모두.5를, 개별 target−base AUC 차이 구간은 모두0을 포함했다. 따라서 **‘추가학습 전부터 있던 동일 구분력이 전부였다’는 원인을 확보하지 못했다.** 그렇다고 구분력 부재나 추가학습의 개인별 인과효과를 입증한 것도 아니다.

| meanmax52−image mean | base에서의 이득 | target에서의 이득 | 이득의 차이 [95% 구간] |
|---|---:|---:|---|
| 모델1 fixed | +.0575 | +.1850 | +.1275 [−.0500,.3025] |
| 모델1 tuned | −.0075 | +.0825 | +.0900 [−.0276,.2150] |
| 모델2 fixed | −.0475 | +.1225 | +.1700 [.0275,.3225] |
| 모델2 tuned | −.0925 | +.0575 | +.1500 [−.0150,.3275] |

모델2 fixed의 조건부 구간이0을 넘는 관측도 보존한다. **+.1700은 meanmax의+.1000과 image mean의−.0700이 함께 만든 상대 변화**이며, 전부 환자 신호 생성으로 해석하지 않는다. 네 대조는 개발자료에서의 후속 분석이고 fixed는 원래 보조 정책이다. 두 target은 독립 반복이 아니며 다중 비교·이미 본 selection40의 한계가 있다.

이번 점추정에서는 base에서 같은 집계 이득이 유지되지 않았고, base만으로 설명된다는 근거를 확보하지 못했다. 다만 domain adaptation, 특징 분포와 scaler의 정합성, 두 모델이 공유하는 반응, 환자 참여 영향은 여전히 분리되지 않았다.

추출 전후 가중치는 실제 pretrained artifact의 FP32 변환과686개 tensor가 일치했고 LoRA는 없었다. 최초 검산은 `noised` 함수 import 누락으로 실패했다. v1 소스·실패 기록을 보존하고 수치 함수는 그대로인 별도 v2로 수정하여 검산했다. 수정 경로는 기존 원시 U2로 실제 실행 확인했고 v2 검산20.350초, 분석.426초였다. 독립 GPU Jacobian 재실행은 아니다.

[base 실행](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/cdi_base_selection_v1/execution.json) · [20개 결과·4개 대조 전체](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/cdi_base_selection_v1/analysis_v1/analysis.json) · [v2 독립 검산](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/cdi_base_selection_v1/verification_v2.json) · [보존한 최초 검산 오류](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/cdi_base_selection_v1/verification.json)

## 완료한 원인 점검: 표준 집계가 쓰는 영상 간 차이

meanmax52의 고정 LR 점수를 평균 특징 항과 두 영상의 절대차 항으로 정확히 나눠, 차이 항을 제거했을 때 순위 개선이 남는지 확인했다. 사진2장이면 `max = mean + |x1−x2|/2`이다. 이미 fit한 scaler·계수와 기존 target/base 특징만 사용했고 새 학습·GPU는0이었다.

실제 식은 기존 fit scaler에서 평균 기준 μm, half-range 기준 μh=μmax−μm를 구하고, `logit = 원 intercept + M + H`로 중심화한 것이다. M은 평균 특징 항, H는 half-range 항이다. 항 제거는 그 항을 fit 기준으로 고정한 feature 조작이며 실제 환자 사진의 인과적 개입은 아니다. 포화 때문에 순위를 잘못 읽지 않도록 logit AUC를 사용하고 full의 저장 확률 순위와의 일치를 검산했다.

| scorer / 응답 출처 | 전체 AUC | 평균 항만 | 차이 항만 | 전체−평균 항만 [95% 구간] |
|---|---:|---:|---:|---|
| 모델1 fixed / 자기 target | .6325 | .5450 | .6275 | +.0875 [−.0575,.2375] |
| 모델1 tuned / 자기 target | .5175 | .4750 | .5875 | +.0425 [−.0325,.1226] |
| 모델2 fixed / 자기 target | .6700 | .5775 | .6800 | +.0925 [−.0225,.2200] |
| 모델2 tuned / 자기 target | .6050 | .5750 | .6675 | +.0300 [−.0475,.1025] |
| 모델1 fixed / 반대 target | .7025 | .5675 | .7150 | +.1350 [.0224,.2551] |
| 모델1 tuned / 반대 target | .5375 | .4925 | .6925 | +.0450 [−.0025,.1050] |
| 모델2 fixed / 반대 target | .6425 | .5675 | .6675 | +.0750 [−.0450,.2026] |
| 모델2 tuned / 반대 target | .5825 | .5325 | .6625 | +.0500 [−.0275,.1350] |
| 모델1 fixed / base | .5550 | .5800 | .5550 | −.0250 [−.1350,.0825] |
| 모델1 tuned / base | .4250 | .4500 | .5175 | −.0250 [−.0950,.0400] |
| 모델2 fixed / base | .5700 | .6150 | .4350 | −.0450 [−.1226,.0375] |
| 모델2 tuned / base | .5250 | .5675 | .4450 | −.0425 [−.1175,.0225] |

**표의 방향은 scorer를 fit한 target의 원래 A/B 배정으로 고정했다.** 자기 target에서만 그 방향이 실제 membership과 같다. 반대 target·base 행은 A/B 구분 진단이며 해당 모델의 membership 성능표가 아니다.

확인된 사실은 다음과 같다.

- 고정 scorer 안에서 차이 항을 제거하면 자기/반대 target의8개 문맥 모두 AUC 점추정이 내려갔다. 자기 target fixed에서 실제 순위 개선/악화는 모델1의67/32쌍,모델2의53/16쌍이었다. **자기 target의 네 차이 구간은 모두0을 포함**하므로 모집단에서 일관된 이득을 확정하지 않는다.
- 이 사진 간 차이 정보는 이미 **기존 표준 meanmax 집계가 이용하는 정보**다. 이것을 새 공격의 독자적 기여로 가져갈 수 없다. 같은 scorer 내부의 항 제거만으로 별도 학습한 image mean·mean26과의 성능 차이 원인이 전부 설명됐다고도 할 수 없다. 독립적으로 학습한 mean26fixed의 기존 AUC .4275/.5825와 위 평균 항만 .5450/.5775는 서로 다른 계수이므로 같은 baseline처럼 섞지 않는다.
- 모델1fixed scorer는 model2의 응답에서도 A 집단을 AUC .7025로 구분했다. 그러나 model2의 실제 추가학습 member는 B이므로 **같은 고정 방향의 실제 membership AUC는 .2975**다. 모델2에 맞춰 다시 학습한 별도 scorer의 결과와 혼동하지 않는다. 이 관측은 해당 점수를 target에 관계없이 member-positive로 해석하는 데 대한 반례이며, target별 CDI 전체 실패의 증명은 아니다.
- 따라서 현재 가장 구체적인 미해결점은 **영상 간 차이로 생긴 환자군 구분을 실제 학습 참여 신호와 구별하는 것**이다. 원래 A/B에 대한 우연한 연관, domain adaptation, scorer 전이/분포 변화, 실제 참여 영향의 역할은 아직 분리되지 않았다. 이는 개선 원인 설명의 한계이며 기존 MIA 성능 비교 자체가 모두 무효라는 뜻도 아니다.

원점수 순위 불일치0,4개 scorer×3출처의480행·4,800쌍·36AUC·24개 차이 구간이 독립 검산을 통과했다. 분해 내부.498초,검산.821초,원52D logit 최대 차이7.11e−15였고 bootstrap 구간 차이는0이었다. 전체36점수와 full−spread 대조까지 [원시 분석](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/cdi_u_cohort_v1/meanmax_feature_decomposition_v1/analysis.json)에 남겼다. [분해 전 protocol](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/cdi_u_cohort_v1/meanmax_feature_decomposition_v1/protocol.json) · [독립 검산](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/cdi_u_cohort_v1/meanmax_feature_decomposition_v1/verification.json)

A/B 배정 코드와400명 label도 [별도 재확인](spec_sources/cohort_assignment_recheck_20260915.md)했다. 원래 hash 배정과 불일치0이지만 나이·세부 질환·follow-up 수 등의 교란 가능성까지 배제한 것은 아니다.

## 전체19개 결과와 원시 표

![19개 지정 비교의 환자 AUC와 구간](cdi_u_results_artifacts/v1/selection_auc_all19.png)

[SVG 원본](cdi_u_results_artifacts/v1/selection_auc_all19.svg) · [전체 CSV](cdi_u_results_artifacts/v1/selection_auc_all19.csv) · [수치·비용 JSON](cdi_u_results_artifacts/v1/report_table.json)

파란색은 사전 주 대조에 포함된 방법을 표시한다. 성능 순으로 정렬하거나 승자를 골라 강조한 그림이 아니다. 그림에는 저장된 통계만 옮겼다.

## 정보·비용·검산 범위

- 이번 것은 **CDI 공개26D의 의료 환자 MIA adaptation**이다. 원형 전체 reference-set 통계검정까지 재현한 것은 아니다. U 영상의 image membership은 모두0이며 환자 참여 정답으로 fit80의 scorer를 학습했다. 알려진 보조 환자의 참여 정답을 사용하므로 자기 사진만 가진 환자 단독 접근과 구분한다.
- 같은26D를 쓰는 영상 집계와 환자 mean/meanmax 비교는 동일한 target 추출값을 사용한다. SecMI 단일 점수와 CDI 전체는 계산량·보조 정답 사용 조건이 다르므로, CDI−SecMI를 같은 비용 우위라고 부르지 않는다. NO objective는 이미 수행한 동일 계산에서 보관한 추가 값이다.
- 전체480건의 독립 실행 비용은28,058 UNet forward-example/8,378 input backward다. 기존 kernel U2를 재사용해 이번 신규는27,942F/8,344B,478건이다. VAE/text encoder는 기존 cache를 사용해 이번 호출0이며 standalone encoding이 무료라는 뜻은 아니다.
- 이번 전체 실행기 **2,916.393초(48분36초)**, 신규 score 측정2,875.006초, 재사용을 포함한 score 합2,886.511초다. model1/model2 peak CUDA allocated는4,096,370,688/4,087,547,904 bytes였다. 추출 안내45–55분 범위다.
- NO actual objective 평가3,578회이며 이번에는 SciPy nfev 합도3,578이다. 모두 source의5-iteration 제한에서 status1로 끝났고 계산 오류로 중단된 기록은 없다. 완전 수렴을 주장하지 않는다.
- 480개 원시 tensor·12,480개 feature를 CPU float64로 검산하고 noise3,840개를 독립 재생성했다. 기존 SecMI와 대응하는192건은 중간값이 정확히 일치했다. 나머지288건에는 비교할 과거 SecMI가 없어 재현 일치를 주장하지 않았다.
- 원시 검산 내부58.666초, 분석기 내부0.923초, 학습 경고0개다. 저장 scaler/계수의 예측, CV 저장확률의 log-loss와 C 선택, AUC와 공통 bootstrap을 별도로 검산했다. CV fold마다 LR를 새로 최적화하거나 UNet Jacobian을 독립 GPU로 재계산한 검산은 아니다.
- selection40, target별 nonmember20으로 1% FPR를 정밀 검증하지 않았다. calibration/test·R 확대·추가 target 학습은 이번 범위에 포함하지 않았다. 약한 결과가 모든 patient MIA의 불가능이나 연구 목표 폐기를 뜻하지 않는다.

## 근거와 현재 위치

[실행 전 비교 명세](PATIENT_BASELINE_SPEC.md) · [분석 계약](spec_sources/cdi_u_analysis_contract.json) · [결과 전 해석 메모](spec_sources/cdi_u_interpretation_before_results.md) · [MoFit 실행 준비·기존 conditioning 대조](spec_sources/mofit_medical_readiness_20260915.md)

[실행](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/cdi_u_cohort_v1/execution.json) · [전체 특징](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/cdi_u_cohort_v1/results.json) · [성능·대조](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/cdi_u_cohort_v1/analysis_v1/analysis.json) · [독립 검산](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/cdi_u_cohort_v1/verification.json)

**현재2번 진행 중.** 실제 U 비교·순위 분해·base 대조·평균/차이 항 진단을 완료했다. 사진 간 차이 항의 순위 역할까지 확인했지만, 가까운 기존 방법이 실패하는 원인을 완전히 식별한 것은 아니다. 다음 조사는 관측한 환자군 구분과 실제 참여를 구별할 대조를 고정하고, 기존 E/U 전이·MoFit 의료 적용 등 미완료 비교와 연결하는 것이다. 기존 A/B를 독립적으로 교차하는 모델 대조는 가능한 진단이지만, 추가학습의 필요성·한계·비용을 검토하기 전 필수 실험으로 승격하지 않는다. 새 설계 준비·2번 완료 또는 연구 방향 폐기를 선언하지 않는다.
