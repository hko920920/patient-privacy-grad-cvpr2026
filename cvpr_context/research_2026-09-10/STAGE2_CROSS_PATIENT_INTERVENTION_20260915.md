# 2번 실마리 확장: 누구의 학습 기여인지에 따라 U 반응이 달라지는가

2026-09-15 20:22 KST 시작. 사용자의 “이 실마리를 제대로 확장해보자, 이게 되는지 안 되는지”에 따라 진행한다. [네 단계](RESEARCH_FRAMEWORK.md)의 **2번**이다. [앞선 한 환자 대조](STAGE2_MEASUREMENT_AUDIT_20260915.md)는 자기 U와 다른 환자의 점수가 함께 변함을 확인했지만, 이 자체로 기존 방법의 일반적인 실패나 새 기여를 확정하지 않았다.

**실제 확장·독립 검산 완료:** 두 환자의 기여 제거를 교차 대조한 결과, 기존 U79 고정 C의 `patient mean+max52`는 U에서 두 환자의 자기-교차 차이를 모두 양수로 나타냈다(R_p=+.17249118, R_q=+.03427000). 다른 주3개는 p 양수/q 음수였다. 실마리를 일괄 기각할 결과는 아니지만 **강한 기존 방법이 이미 잡는 대응 반응**이므로 새 공격법의 필요성이 입증된 것은 아니다. q의 교차차 h는 다른39명의 IQR 안이고 p보다 큰 다른 환자의 h도 있다. 이 두 개입만으로 환자 특이성·환자군 판별·새 기여를 확정하지 않는다. 현재 GPU 실행은 끝났고2번은 진행 중이다.

## 먼저 확인한 선행연구 경계

이번 질문에 맞춰 아래 원문의 해당 절을 직접 대조했다. 이전79편 장부의 전부를 새로 완독했다는 의미가 아니며, 기존 읽기 범위를 변경하지 않는다.

| 원문·확인 범위 | 이미 다루는 것 | 이번 조사에 적용하는 경계 |
|---|---|---|
| [Kandpal 등, EMNLP2024](https://aclanthology.org/2024.emnlp-main.1014.pdf), §3·명제1 | 미사용 같은 사용자 자료, reference likelihood 비의 평균, 기여율·사용자 분포 차이 | U 설정·참조 평균·사용자별 자료 유사성 자체는 새롭지 않다. 이론의 이상적 likelihood와 현재 diffusion 손실은 다르다. |
| [Zhang 등, NeurIPS2023](https://arxiv.org/pdf/2112.12938v2), §3·§6(PDF pp.3,7–8) | 포함/제외에 따른 counterfactual memorization, 다른 미사용 자료에 대한 영향 | 자기 자료 외의 모델 반응을 개입으로 조사하는 것 자체에 선행이 있다. 해당 논문의 기대값 추정과 우리 한 학습 경로의 개입을 구분한다. |
| [Meeus 등, 2025](https://arxiv.org/html/2506.20481v1), §§2,4–5 | 전체 influence matrix, near-duplicate에 퍼지는 영향, self-influence와 추출 위험의 간극, top-1/top-2 영향 비 | 교차 영향 행렬이나 타 자료의 영향까지 보는 것 자체는 새 기여가 아니다. record/near-duplicate의 결과를 환자 U의 실증으로 옮기지 않는다. |
| [FACE-AUDITOR, USENIX Security2023](https://www.usenix.org/system/files/usenixsecurity23-chen-min.pdf), §§4.1–4.5·Appendix A | 동일인의 새로운 사진, target 유사도와 영상 reference 유사도 결합 | 같은 사람을 잘 알아보는 것과 학습 참여를 구별한다. 관계·응집도·reference만 추가했다고 신규성이 생기지 않는다. |
| [RAPID, CCS2024](https://arxiv.org/pdf/2409.00426v2), §§3–4.2(PDF pp.3–5) | 난이도 보정의 오차, 원점수와 보정점수의 결합 | 단순 차감과 원점수/차감점수 결합도 강한 기존 대안이다. sample membership의 주장을 환자의 U에 그대로 적용하지 않는다. |

[이번 확보 PDF·URL·SHA256](spec_sources/primary_cross_intervention_20260915/acquisition.json). NeurIPS2020 Feldman–Zhang은 공식 초록과 위 원문의 연결만 추가 확인했으며 이번에 전체 본문을 읽었다고 세지 않는다.

## 이번 최소 대조와 구체적인 예측

환자 p=14393은 기존 실마리를 얻은 사례로 고정한다. 새 q=4092는 **기존 selection A 중 E 두 장이 각각4번 노출된 환자의 숫자순 첫 번째**다. q의 새 개입 결과를 보지 않고 정했으며 결과가 좋지 않다고 교체하지 않는다. q는 최초37번째 갱신부터 기여하므로 기존43단계 checkpoint에서 제거를 시작할 수 없다.

원래 M1과 검증된 p 기여 제거모델은 재사용한다. q의8개 loss 기여만0으로 하는 모델 한 개를 원래 초기화에서1,000단계 학습한다. 순서·난수·optimizer·loss 분모를 유지한다. 두 제거모델 모두 예정4,000슬롯 중 비영기여3,992개지만 실제 제거 위치·환자 내용은 다르다. 이것은 명부에서 영상을 지우고 batch를 재구성하는 재학습과 같지 않다.

| 새 계산 | 범위 | 목적 |
|---|---|---|
| q 기여 제거학습 | 1,000단계, 초기화부터 | 누구의 기여를 없앴는지 바꾸는 실제 개입 |
| q control의 원형CDI26D | 기존 U82장 + p/q의 E4장 | 같은 관측 사진에서 두 개입의 차이 비교 |
| p control의 추가CDI26D | p/q의 E4장 | U뿐 아니라 정확 학습사진의 반응도 같은 대조로 확보 |
| 고정 기존 판별기 적용 | 원fit80 18방법·대상제외U79 18방법 | 강한 기존 방법이 이미 대응 신호를 쓰는지 확인 |

신규 feature 계산은90행, 원형·기존 p control168행은 저장 검산 자료를 재사용한다. 기존 U79 판별기의 직접 보조학습에는 p와 q 둘 다 없다. 고정C 주4개를 전부 보고하며 과거 tuned C·NO27 등은 보조로 보존한다. 새 fitting·C 선택·R 수정·보정/시험 분할 사용은 없다. E에도 같은 U-fit 판별기를 적용하는 것이므로 E에 최적화한 공격 성능 시험으로 부르지 않는다.

## 비교식과 허용되는 해석

같은 고정 점수 함수 g와 환자 i의 관측 사진 집합 X_i에서 다음을 계산한다. 양수는 학습 기여가 있을 때 더 높은 공격 점수를 의미한다.

```
D[p,i] = g(M1, X_i) - g(M_minus_p, X_i)
D[q,i] = g(M1, X_i) - g(M_minus_q, X_i)

R_p = D[p,p] - D[q,p]
R_q = D[q,q] - D[p,q]
K   = R_p + R_q
```

R_p는 p의 사진에 대해 p 제거와 q 제거를 비교하고, R_q는 q의 사진에 대해 반대로 비교한다. **K가 양수여도 둘 모두 자기 환자에 대응하는 양의 차이가 있다는 뜻은 아니다.** R_p/R_q를 반드시 함께 보고한다. U와 E에서 같은 식을 적용한다.

D[j,i]=a[j]+b[i]라는 단순 가산 설명에서는 K=0이다. 따라서 이 대조는 개입별 offset과 모든 개입에서 동일한 query 특성만으로 설명되는 경우를 분리한다. 그러나 개입 크기×사진 민감도의 곱, 병리/촬영 유사성, 비선형 optimizer 경로는 이 식으로 제거되지 않는다. 평균 차감만으로 환자 순위가 바뀐다는 주장도 하지 않는다.

공통 비교 환자는 U41명에서 p/q를 제외한39명(A19/B20)이다. 기존 selection40에는 q가 있으므로 q 개입의 “다른40명”이라고 잘못 표시하지 않는다. p의 다른40명과 q의 다른40명, 기존 selection군은 따로 기술한다. 다른 환자의 수를 독립 개입 반복 수로 세어 신뢰구간을 만들지 않는다.

| 관측 | 이번에 내릴 수 있는 판단 |
|---|---|
| 기존 점수에서도 U의 두 자기-교차 차이가 양수 | 이 고정 환자쌍에서 기존 방법이 이미 대응 반응을 잡는 근거. 새 공격 필요성으로 바꾸지 않는다. |
| E에는 대응 차이가 있지만 U에는 일관되게 없다 | 이 조건에서 U 전달을 지지하는 근거가 약해진다. U 전체 불가능을 뜻하지 않는다. |
| 변화는 크지만 대응 차이는 섞이거나 반대다 | 현재의 단순한 “공통 반응을 빼면 자기 환자 신호가 나온다” 설명을 지지하지 않는다. 좋은 scorer/환자로 교체하지 않는다. |
| 어떤 결과든 | 두 checkpoint 차이는 내부 연구자의 원인 진단이다. 외부 감사자의 허용 입력이나 새 공격 점수로 제안하지 않는다. |

이 사전 정책은 **새 q 학습 전에** [JSON 계약](spec_sources/patient_cross_intervention_analysis_contract_20260915.json)으로 동결했다(SHA `b0855a8bdec2e1a5f738d35ee12930bf0dfd7a21c603cf55f343ba99ccf5bc70`). 판정은 고정된 모든 수치의 기술이며 새 AUC·p-value·효능 임계값을 만들지 않는다. 강한 보정/결합까지 넘는 제한 접근의 새 정보와 해결 원리는 아직 확보하지 않았다.

## 실행 상태와 시간

원문·기록 대조와 최소 설계에15–20분, 이어 새 GPU 계산19–21분·준비 및 독립 검산 포함30–40분을 고지했다. 원래 control학습522.295초, U82추출562.766초가 예상의 근거다. q 학습 실행기는20:49:02–20:57:47 KST, **524.910초**에 완료됐다. 학습 loop만512.865초이며 준비·저장을 포함한 실행기 시간과 구별한다. 독립 CPU 검산은13.189초였다. 후속90행 추출·최종 산술 검산의 실측은 아래 완료 기록에 추가한다.

q 학습은 입력1,000쌍·36단계까지의 trace·8개 mask packet·원래 초기 adapter/optimizer/scaler·고정 base를 대조했다. 실제1,000업데이트·4,000 forward examples·1,000 backward calls이고 q의 기여0, p의 기여8, 총 비영기여3,992다. 36단계 원형 full checkpoint는 없으므로 trace 일치를 full-state 일치라고 쓰지 않는다. 최종 q checkpoint SHA는 `d952aa55221fa5c0eb8c0ac2539427457a33adc4edc9750f8f2f58b11ab02068`이다.

새90행의 추출 계약은 기존168행의 원시 파일·원행과 원형/두 control 학습 근거를 포함한528파일에 결속했다. SHA는 `41d299858f2206bfdeeee6ed3de98195edf5182c33a62ec6642d4e5ee1d977c8`이다. 원형 공개26특징을 유지하고 결과에 따라 NO 경로나 점수 방향을 바꾸지 않는다.

사전 교차 검토에서 DINO의 과거 보조 LR에 p가 포함됐다는 메타데이터 누락을 발견했다. 아직 새 결과를 분석하기 전에 [별도 정정 기록](spec_sources/cross_intervention_dino_metadata_correction_20260915.json)과 분석기v2를 만들었다. DINO 점수·모든 수치 함수는 동일하며 원본v1·주계약은 보존했다. 원시 검산기의 full-state fingerprint 메타필드와 실제 hash/텐서수의 구분도 검산 실행 전에 바로잡았다.

## 대응 차이를 발견해도 남는 식별 문제

[독립 해석 검토](spec_sources/cross_intervention_scope_review_20260915.md)에 결과 전 반례와 허용 범위를 기록했다. `D[j,i]=a_j+lambda_j*b_i`처럼 개입 offset과 영상 민감도의 곱만 있어도 두 R이 모두 양수가 될 수 있다. 따라서 두 양수는 그 고정쌍의 대응 반응이며, 환자 정체성이 원인이라는 증명은 아니다. K는 선택한 점수 척도의 가산 대조이지 AUC와 같은 순위불변 지표도 아니다. 어떤 값이 나오든 이 효과를 외부 감사자의 새 공격 입력이나 곧바로 CVPR 기여라고 부르지 않는다.

## 이번 실제 결과

아래는 새 환자를 점수기 보조학습에서 제외한 기존 U79 고정 C 주4개 전부다. E도 같은 U-fit 점수기로 계산했다. 양수/음수는 계산 오차를 넘는 산술 부호이며 통계적 유의성이나 오탐률 판정이 아니다.

| 기존 방법 | U R_p | U R_q | U K | E R_p | E R_q | E K |
|---|---:|---:|---:|---:|---:|---:|
| Image LR, mean | +0.01374491 | -0.02918168 | -0.01543677 | +0.03804987 | -0.08925030 | -0.05120043 |
| Image LR, max | +0.02724411 | -0.08383623 | -0.05659212 | -0.04002299 | -0.10420844 | -0.14423142 |
| Patient mean26 LR | +0.02550302 | -0.07133215 | -0.04582913 | +0.07499318 | -0.14173608 | -0.06674290 |
| Patient mean+max52 LR | +0.17249118 | +0.03427000 | +0.20676118 | -0.01818445 | -0.07776992 | -0.09595437 |

![기존 주4방법의 두 개입 U 반응](cross_intervention_artifacts/v1/primary_U_response_scatter.png)

회색 점은 공통 배경39명이다. p가 대각선 아래면 R_p>0, q가 위면 R_q>0이다. 두 색 점 사이의 관계를 보이며 회색 환자39명을 독립 개입 반복39회로 세지 않는다.

[모든73개 대비 CSV](cross_intervention_artifacts/v1/all_existing_contrasts.csv) · [주4방법의 U164점 CSV](cross_intervention_artifacts/v1/primary_U_response_points.csv) · [벡터 그림](cross_intervention_artifacts/v1/primary_U_response_scatter.svg)

| 기존 방법 | p의 h | q의 h | 공통39 h 중앙값 | 공통39 h Q25 | 공통39 h Q75 |
|---|---:|---:|---:|---:|---:|
| Image LR, mean | +0.01374491 | +0.02918168 | -0.01979913 | -0.03879755 | +0.00561867 |
| Image LR, max | +0.02724411 | +0.08383623 | -0.02240944 | -0.04289786 | +0.01214918 |
| Patient mean26 LR | +0.02550302 | +0.07133215 | -0.02084247 | -0.06107542 | +0.01242748 |
| Patient mean+max52 LR | +0.17249118 | -0.03427000 | +0.00459100 | -0.04644826 | +0.06578309 |

## 무엇이 좁혀졌고 무엇이 남았는가

첫째, 기존 강한 집계를 적용해도 모든 U 대응 반응이 사라진다는 설명은 이번 사례에 맞지 않는다. mean+max52는 원fit80에서도 두 R이 양수였고, 직접 보조학습에서 p/q를 제외한 U79에서도 양수였다. 이미 정한 보조 DL max·PIAN mean·NO objective 추가 fixed54에서도 두 R 양수가 관측됐다. 이들을 서로 독립된 환자 반복으로 세지 않고 모두 CSV에 보존했다. DL max의 q 차이+.00085449, PIAN mean의 q 차이+.00413036은 해당 원점수 척도 값이지 효과가 크거나 안정하다는 판정이 아니다.

둘째, 이 대응 차이와 환자 특이적 구별은 아직 다르다. mean+max52에서 p의 h=.17249118은 공통39명의 상위 사분위 .06578309보다 크지만 다른 환자의 최대 .35354462보다 작다. q의 h=−.03427000은 IQR [−.04644826,.06578309] 안이다. 이39명은 비참여 환자39명이 아니라 기존 A19/B20의 관측 대상이다. 이 분포에서 임계값을 정해 FPR·개별 검출 성공으로 바꾸지 않는다.

특히 큰 p 대비를 자기 기여만 커진 것으로 해석하면 안 된다. mean+max52의 실제2×2 변화는 아래와 같다. p의 R_p=.17249118은 자기 제거 변화+.05879570에서 **q 제거 변화−.11369548**를 뺀 값이다. 음의 교차 반응이 크게 기여하며, 자기만의 양의 신호를 분리한 값은 아니다.

| 학습 기여 제거 | p의 U 점수 변화 | q의 U 점수 변화 |
|---|---:|---:|
| p 제거 | +.05879570 | +.07370663 |
| q 제거 | −.11369548 | +.10797664 |

셋째, 같은 U-fit 주4개를 E에 적용하면 두 R이 모두 양수인 방법은 없었다. 그러나 보조 DL mean/max, SecMI mean/max, PIAN mean은 E에서 두 R이 양수다. 따라서 “정확 학습사진에도 어떤 기존 방법도 작동하지 않는다”라는 결론은 틀리다. U-fit의 E 결과와 U 결과가 다르다는 사실을 보존하며, E에서 성공한 새 공격을 확보한 뒤 U에서만 실패했다는 인과 이야기로 바꾸지 않는다.

**현재 판단은 기존 방법이 일부 대응 반응을 이미 잡으며, 환자 특이적 위험을 다른 환자 변화와 구분하는 문제는 남는다는 것이다.** 이전 한 환자의 자기 변화가 다른40명 IQR 안이라는 관측을 이 새 결과로 지우지 않는다. 질문이 한 제거의 변화 크기에서 두 제거 사이의 대응 차이로 바뀌었기 때문에 함께 읽어야 한다. 다음에 필요한 것은 mean+max52가 포착한 기존 정보와 영상 민감도·비선형성이라는 대안 설명을 구분할 구체적 근거다. “공통 반응만 빼면 새 공격이 된다”는 후보는 준비된 설계로 채택하지 않는다. 새 R·대규모 학습·보정/시험 분할은 추가하지 않았다.

## 완료 검산·시간·원자료

| 완료 작업 | 실측 | 확인 범위 |
|---|---:|---|
| q 기여 제거학습 | 524.910초 | 원형 초기화·1,000입력·8 loss 기여만 제거, 총3,992 |
| q 학습 독립 검산 | 13.189초 | 저장 state·원장·36 trace·8 mask 미분 산술 |
| 신규90행 CDI 추출 | 630.095초 | 5,267F/1,577B, NO objective677회, 기존168행 재사용 |
| 원시 독립 검산 | 53.398초 | 2,340특징·720 noise draw, 실제 두 checkpoint full FP32 hash, 이전 p-control E DL20 exact |
| 고정 기존 scorer 분석 | 8.407초 | 1,589환자행·73대비·333배경분포, 새 fitting0 |
| 별도 점수 산술 감사 | 7.456초 | 4,767 branch점수·1,032영상확률행·73대비·333분포 |

두 GPU 실행기 시간 합계는 **1,155.006초(19분15초)**다. 원시 검산은 저장된 수치·입력·가중치 출처 확인이며 neural Jacobian/전체 학습을 독립 재실행한 것은 아니다. 기존1,517환자행의4,551개 스칼라가 모두 exact 복원됐고 독립 branch 점수 계산의 최대 차이는3.89e−16이었다. 반응/대비를 새로 선택하거나 점수 방향·C를 바꾼 횟수는0이다. 최초 읽기부터 코드·계약·실행·검산·기록까지의 전체 경과는 WORKLOG110에 별도로 남긴다.

- [q 학습 실행·시간](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260915/masked_patient_4092_training_v1/execution.json) · [독립 학습 검산](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260915/masked_patient_4092_training_v1/verification.json)
- [90행 추출 실행](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260915/cdi_cross_patient_intervention_v1/execution.json) · [원시 독립 검산](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260915/cdi_cross_patient_intervention_v1/verification.json)
- [전체 고정 분석](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260915/cdi_cross_patient_intervention_v1/existing_scorer_analysis_v2/analysis.json) · [환자별 세 모델 점수](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260915/cdi_cross_patient_intervention_v1/existing_scorer_analysis_v2/patient_scores.json) · [별도 점수 감사](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260915/cdi_cross_patient_intervention_v1/existing_scorer_analysis_v2/independent_fixed_scorer_audit_v1/verification.json)
- [그림·CSV 원자료 결속](cross_intervention_artifacts/v1/manifest.json) · [독립 해석 검토](spec_sources/cross_intervention_scope_review_20260915.md)
