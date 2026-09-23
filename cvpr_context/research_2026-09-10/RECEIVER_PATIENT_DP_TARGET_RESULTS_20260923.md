# A2 환자-DP 목표 경로 구현·검산 결과

2026-09-23. 큰 연구 단계2의 목표 생성 구현을 완료했다. **Q에서 새 잡음을 뽑거나 DP 합성 bank를 실행한 결과는 아니다.** 기존 두 초기화에서 확인한 결합 이득·Q 추가 효용은 그대로 보존한다.

## 완료 범위

- 두 encoder·K4 head/변환·P-only projection·기존 class별 환자 균등 목적을 유지했다.
- 환자별 gradient와 private class 분모를 하나의 bounded query로 만들고, Gaussian 보호 후 기존 A2 target을 생성하는 경로를 구현했다.
- 구성 배열 15검사, 실제 P/Q 캐시의 독립 재집계, 공개 자료로 만든 모의 보호 target의 기존 objective 연결을 검증했다.
- 새 Q noise draw/release, encoder forward/backward, 합성 update, V 효용 평가, Expert/Reserved/확인용 수신자 접근은 모두0이다.

주요 산출물:

- [실행 전 명세](RECEIVER_PATIENT_DP_TARGET_SPEC_20260923.md)
- [핵심 연산](../../code_working/receiver_distillation/patient_dp.py), [목표 입출력](../../code_working/receiver_distillation/patient_dp_io.py)
- [검산 프로그램](../../code_working/receiver_distillation/verify_patient_dp.py), [구성 배열 검사](../../code_working/receiver_distillation/test_patient_dp.py)
- [최종 검산 JSON](../../code_working/_reports/receiver_patient_dp_design_20260923_v2/verification.json)
- [첫 비교 DRAFT](../../code_working/_reports/receiver_patient_dp_design_20260923_v2/first_dp_comparison_DRAFT.json)

## 보호하는 정확한 값

인접성은 **Q의 환자 한 명 전체 추가/제거**다. 한 환자의 방문 수나 양성·음성 혼합 여부에 관계없이, 그 환자의 모든 class·encoder·condition을 함께 보호한다. P는 공개자료이며 Q의 환자 수를 공개된 고정 분모로 가정하지 않는다.

먼저 기존 CE gradient를 환자/class의 방문 내에서 평균한다. 각 class의 두 encoder×K4×34좌표를 연결한 벡터를 \(g_{pc}\in\mathbb R^{272}\), 해당 class 존재를 \(b_{pc}\)라 한다. class가 없으면 둘 다0이다.

\[
v_{pc}=g_{pc}\min(1,C_c/\|g_{pc}\|),\qquad
u_p=\tfrac12[v_{p0}/C_0;v_{p1}/C_1;b_{p0};b_{p1}].
\]

따라서 \(\|u_p\|_2^2\le(1+1+1+1)/4=1\)이다. 총546좌표의 합계 \(\sum_{p\in Q}u_p\)에 대해 add/remove 감도1, replace-one 상계2를 사용한다. 이번 계약은 add/remove이며 이전 PRRD 모멘트 감도를 가져오지 않았다.

첫 보호 비교의 설정은 ε=8, δ=1e−5, full-vector Gaussian 한 번이다. ε=8은 첫 개발 비교를 위한 사전 시작값이며 효용을 보고 고른 최적값이 아니다. 기존 patient-DP 계획의 시작 budget을 유지했다. 조건8개나 합성200회마다 새 query를 만들지 않는다.

| 값 | 확정한 규칙·수치 |
|---|---|
| 음성 clipping \(C_0\) | P 해당 class 환자 norm q95 = 2.7412545531339387 |
| 양성 clipping \(C_1\) | P 해당 class 환자 norm q95 = 2.689519212597726 |
| 분위수 | NumPy linear interpolation, 최소1e−6 |
| Q를 읽기 전 선정·결속 | 이번 검산의 공개 calibration 파일로 기록 |
| 합계 query 감도 | 1 |
| 정규화 좌표 noise std | 0.6002290722589745 |
| 복원한 class count noise std | 1.200458144517949 |
| 복원한 gradient 합계 좌표 noise std | 음성3.2907613545065475 / 양성3.228655243600441 |

공개 P의 양성 class 환자는6명이다. 이 기준이 최적이거나 충분히 안정적으로 추정됐다는 주장은 하지 않는다. Q/V에 따라 clipping 기준을 바꾸지 않았다.

Gaussian 표준편차는 Balle–Wang analytic 식으로 계산하고 보수적1e−10 상대 여유를 더했다. 독립 Google `dp_accounting`과70자리 계산을 대조했다. ε=1/4/8, δ=1e−5 모두 계산된 δ가 요구값 이하였고, Google 표준편차와의 최대 차이는2.95e−8이었다. 이는 명시한 실수 Gaussian 메커니즘의 보정 검사이며 유한정밀도 구현의 별도 보안 증명은 아니다. [원문 Theorem8](https://proceedings.mlr.press/v80/balle18a/balle18a.pdf).

## 분모·후처리·실행 경계

원래 class별 방문 평균과 환자 균등 평균을 유지한다. clipping은 환자/class gradient 크기를 줄이며, class 존재 count를 임의로 재정의하지 않는다. 최종 unit-ball 제한은 부동소수점 경계 초과 방지용으로 전체 기여를 함께 제한한다. 이 경로가 활성화되면 count도 미세한 가중 질량이 되지만, 이번 실제 Q에서는 활성화0이었다.

보호 후 Q class 질량은 `max(0, noisy_count)`로 만들고, noisy gradient 합계는 반경 \(C_c\,\widehat n_{Q,c}\) 안으로 투영한다. 그 뒤 P의 정확한 공개 합계·count와 결합한다.

\[
t_{mk}=\tfrac12\sum_{c=0}^1
\frac{S_{P,mkc}+\widehat S_{Q,mkc}}
{\max(1,n_{P,c}+\widehat n_{Q,c})}.
\]

이 안정화는 DP 값과 P만 사용한다. Q의 참 분모로 다시 나누거나, 부정적인 noisy count가 나왔다는 이유로 잡음을 다시 뽑지 않는다. cosine target norm≤1e−12이면 해당 조건의 공개 P target으로 대체한다. 보호 요약 이후의 동일 합성 최적화는 후처리이며, 독립 보호 요약을 추가 생성하면 별도 합성 계산이 필요하다. [후처리·합성의 기준](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf).

실제 release API는 고정 seed를 받지 않으며 운영체제 난수를 사용한다. 한 좌표당 두 독립 정규표본을 결합하는 구현은 [IBM Gaussian 구현](https://github.com/IBM/differential-privacy-library/blob/main/diffprivlib/mechanisms/gaussian.py)을 참고했다. 고정 seed는 공개/구성 배열의 모의검사에만 사용했다.

향후 release 실행기는 고유 output directory에 예약 파일을 먼저 저장하고, 같은 경로를 재실행해 잡음을 다시 뽑지 못하게 한다. 공개 규칙·보호 target·메커니즘 metadata만 handoff에 넣는다. Q 환자 ID·원분모·원합계·clipping 진단·Q cache hash·실현 noise/seed는 보호 산출물 패키지에 넣지 않는다.

이번 결과 보고서와 clipping 진단은 **비DP 내부 개발 기록**이다. 과거 Q/V 기반 개발·방법 선택까지 소급해 DP라고 주장하지 않는다. 고정한 공개 설정에서 수행하는 한 Q→보호요약→합성 실행의 조건부 보증과 전체 연구 과정의 보호를 구분한다.

## 실제 검산 결과

| 검사 | 결과 |
|---|---|
| 구성 배열·감도·빈 class·잘못된 분모·입력·직렬화 | 15개 PASS |
| 기존 P 합계 재현 | 최대5.68e−14 |
| 기존 Q 합계 재현 | 최대2.27e−13 |
| 기존 pooled BioViL/DINO target 재현 | 최대1.11e−16 / 5.55e−17 |
| P/Q 영상·환자·label 순서 및 분리 | 기존 명부와 일치 |
| 실제 Q 최대 환자 기여 norm | 0.9969521873≤1 |
| 제한만 적용한 target의 독립 계산 | 최대3.61e−16 |
| 제한만 적용한 class count 오차 | 0 |
| 공개 모의 target→기존 loader→objective | target exact, feature gradient 차이0 |
| 기존 A2 코드·목표·입력 hash | 변경 없음 |

최초 검산에서는 Q 합계의 NumPy 순차 합산 오차1.364e−12가 기존1e−12 기준을 넘었다. 기존 target 자체의 오차는3.33e−16이었고, 독립 `math.fsum` 대조로 합산 오차임을 확인했다. 신규 환자 합계 함수를 보정 합산으로 고쳐2.274e−13으로 낮췄다. **허용오차와 학습 정의는 바꾸지 않았다.** 최초 코드·공개 calibration·실패 진단은 [v1 폴더](../../code_working/_reports/receiver_patient_dp_design_20260923_v1/)에 보존했다. 최종 v2의 공개 clipping 값도 최초 값과 동일하다.

저장 특징을 사용했으므로 새로운 모델 추출은 없다. 실제 검산 프로그램은2.875초, 구성 배열 검사는0.110초였다(각 Python 시작·라이브러리 로딩 제외). 실제 이미지까지의 gradient를 이번에 다시 실행한 것은 아니다. 기존 검증된 encoder/renderer/목적을 유지하고, 바뀐 target 직렬화와 feature gradient 연결만 확인했다.

Q의 clipping-only 내부 진단은 다음과 같다.

| class | 대상 환자 | 제한된 환자 | 비율 |
|---|---:|---:|---:|
| 음성 | 2,015 | 118 | 5.856% |
| 양성 | 114 | 5 | 4.386% |

조건별 target의 상대 L2 변화는 BioViL0.158~0.185%, DINO0.278~0.451%였다. 이는 잡음 전 벡터 변화이며 AUROC 손실이나 DP 잡음 내성을 예측하는 수치가 아니다. 이번에는 Q에 Gaussian noise를 적용하지 않았다.

## 다음 첫 비교의 구체적인 범위

기존 A2 공개전용101과 비DP P+Q101을 재사용한다. 새 비교는 **clipping-only/no-noise 한 bank와 ε8 DP 한 bank**, 둘 다128장·200회·seed101·microbatch16이다. source/head/조건/projection/optimizer/초기 template/해상도 일정은 기존대로다. 새로운 보호 query는 단 하나이며 같은 query의 반복 초기화를 추가하지 않는다.

두 새 최종 PNG를 모두 고정한 뒤 기존 BioViL/DenseNet V 특징과2000개의 환자 bootstrap draw로 나란히 비교한다. 주 질문은 DP(P+Q)−P의 추가 효용이고, clipping-only−비DP 및 DP−clipping-only로 제한과 잡음의 영향을 나눈다. Q signal의 벡터 차이만으로 성능 성공을 선언하지 않는다.

기존 A2 실측74~75분/bank 기준 합성·저장 합계 약148분이다. 준비·job 연결·검산·평가를 포함한 계획 범위는 **약2시간35분~3시간**, 별도 시간 상한 제안은 bank당120분·묶음5시간이다. 이번 CPU 준비 때문에 P/Q encoder 추출을 다시 할 필요는 없다. 이 시간은 실행 보장이 아니며 추가 profile을 요구하지 않는다.

남은 필수 항목은 실행할 한 release·두 bank·시간 상한의 범위 동결, 새 target hash를 별도 job에 결속하는 일, 기존 비DP 전용 실행 기록을 clipping-only/DP로 정확하게 구분하는 adapter 연결이다. 기존 저장·재개·PNG 검증을 전부 반복할 이유는 없다. 현재 DRAFT는 실제 실행을 허가하거나 예약하지 않는다.

환자-DP 효용, 강한 선행 대비 가치, 독립 수신자·최종 환자 평가는 아직 남아 있다. 이번 완료는 그 첫 보호 비교를 정확하게 정의하고 목표 생성 경로를 검산한 것이다.
