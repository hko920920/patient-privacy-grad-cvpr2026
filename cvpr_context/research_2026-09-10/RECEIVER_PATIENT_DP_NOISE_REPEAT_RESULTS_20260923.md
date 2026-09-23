# A2 환자-DP 독립 잡음 반복 결과

2026-09-23 UTC. 두 번째 독립 잡음에서도 사전 개발 기준을 충족했다.

새 DP2−공개전용 DenseNet: AUROC +0.102614,95% 환자 구간 [+0.058154, +0.150166]; AP +0.021108,95% 구간 [+0.002135, +0.041632].

합성 초기화와 학습 seed101을 고정하고, 같은 메커니즘의 새로운 독립 Gaussian 잡음 1회로 bank 하나를 새로 제작했다. 기존 공개전용·비DP·제한-only·DP1을 재학습하거나 좋은 결과만 선택하지 않았다.

| 조건 | DenseNet AUROC | DenseNet AP | BioViL AUROC | BioViL AP |
|---|---:|---:|---:|---:|
| 공개전용 | 0.542669 | 0.053838 | 0.714645 | 0.077790 |
| P+Q 비DP | 0.671734 | 0.078565 | 0.725230 | 0.076155 |
| 제한-only | 0.634691 | 0.068850 | 0.729101 | 0.079249 |
| DP 잡음1, 기존 | 0.658184 | 0.076739 | 0.730769 | 0.080103 |
| DP 잡음2, 이번 | 0.645283 | 0.074946 | 0.724648 | 0.076498 |

| DenseNet 대응 비교 | ΔAUROC [95% 환자 구간] | ΔAP [95% 환자 구간] |
|---|---:|---:|
| DP1−공개전용 | +0.115515 [+0.068791, +0.165723] | +0.022901 [+0.003706, +0.043453] |
| DP2−공개전용 | +0.102614 [+0.058154, +0.150166] | +0.021108 [+0.002135, +0.041632] |
| DP2−DP1 | -0.012902 [-0.037665, +0.011864] | -0.001793 [-0.012614, +0.010399] |
| DP2−비DP | -0.026451 [-0.055188, +0.003185] | -0.003619 [-0.014392, +0.009196] |
| DP2−제한-only | +0.010592 [-0.026893, +0.049336] | +0.006096 [-0.008446, +0.022150] |

두 DP bank의 기술적 평균은 DenseNet AUROC 0.651733/AP 0.075842다. 두 분류기를 합친 ensemble의 성능이나 잡음 모집단 평균의 신뢰구간이 아니다.

개발 기준은 첫 실행과 동일하게 AUROC 점차이>0,환자95% 구간 하한>0,AP 점차이≥0이다. 각 결과를 이 기준으로 별도 판정했다.

DP1 판정: POSITIVE_DEVELOPMENT_SIGNAL. DP2 판정: POSITIVE_DEVELOPMENT_SIGNAL. 두 실행 모두 충족: True.

독립 잡음 두 번의 관측이며 합성 초기화 반복을 추가한 것은 아니다. V와 DenseNet은 이미 사용해 온 개발 자원이다. 같은 2,000회 환자 bootstrap 구간은 고정된 bank/잡음에 조건부이며 잡음 변동·합성 초기화 변동을 포함하지 않는다. 두 결과로 일반적 성공률이나 잡음 분산을 확정하지 않는다. 차이 구간이 0을 포함하는 결과는 동등성/비열등성 증거로 바꾸지 않는다.

**보호 및 누적 회계**

각 실행은 Q 환자 한 명의 모든 방문 추가/제거 인접성, ε8/δ1e−5,546좌표 합계 감도1,analytic Gaussian σ0.6002290722589745를 유지했다. P-only class clipping 기준과 class별 환자 균등 집계,보호된 class 분모/합계 안정화는 이전과 같다.

이번은 별도 Q release02이다. 이 A2 메커니즘 계열에는 총2개의 독립 보호 요약이 있다. 같은 요약의 반복 후처리에는 추가 비용이 없지만 새 요약은 별도 release다. 두 요약 또는 그 후처리 산출물을 함께 공개할 경우 기본 합성의 보수적 상한은 (ε16,δ2e−5)이다. 이를 공동 ε8 보장이나 최적 회계값으로 부르지 않는다. [Dwork–Roth,Theorem3.16](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf).

회계 범위는 고정된 현재 A2 release01/02와 그 후처리다. 과거 비DP Q/V 기반 설계 선택·비DP 대조·원 평가 예측·내부 분석의 공개까지 소급하여 DP라고 주장하지 않는다. 유한 정밀도 구현은 기존 수치 검산 범위이며 별도의 유한 정밀도 보안 증명을 추가한 것은 아니다.

새 protected_png에는 DP 유래 PNG128장·합성 label·공개 학습 규칙 및 hash만 보존했다. Q 원통계/분모/cache hash·내부 checkpoint/provenance·실현 잡음/비밀 난수 상태·V 예측은 넣지 않았다. 기존 DP1 산출물도 보존했고 외부 게시/배포는 하지 않았다.

**실행과 비용**

- 128장·200회·microbatch16·seed101. 기존 AdamW/source/조건/head/projection/해상도 일정을 유지했다.
- 공개 template·renderer·optimizer·난수 초기상태가 기존 seed101과 일치했다. 이전 checkpoint로부터 warm start하지 않았다.
- 새 요약은 운영체제 독립 난수로 1회 생성했다. 결과를 본 재추출·계수/중간 checkpoint 선택·추가 release는 없다.
- 기존 학습/replay 코드를 재사용하고 새 target/job 연결만 확인했다. 동결 source/target 불변,parameter 갱신,PNG/label/hash와 이전 결과 보존 검증을 통과했다.
- 기존 V 2,026명/5,047장과 환자 bootstrap 배열을 재사용했다. sklearn 독립 검산 최대 오차 1.11e-16. 새 Q/V encoder forward는0회다.
- 목표 재구성/새 잡음/실행 연결 2.81초. 새 bank 합성·저장 74.07분. 평가 11.84초.
- 코드/계약 연결을 포함한 작업 시작→평가 완료 82.53분. Peak allocated GPU 5.921GiB. bank120분/전체180분 상한을 지켰다.
- 새 합성200 update/PNG128장. Expert·Reserved·확인용 수신자·다른 seed·학습 연장·HPO·세 번째 잡음은 실행하지 않았다.

**산출물**

- [동결 계약](RECEIVER_PATIENT_DP_NOISE_REPEAT_CONTRACT_20260923.md)
- [평가와 모든 차이/구간](../../code_working/_reports/receiver_dp8_noise_repeat_20260923_v1/evaluation/result.json)
- [독립 release 회계](../../code_working/_reports/receiver_dp8_noise_repeat_20260923_v1/privacy_release_ledger.json)
- [실행 결과](../../code_working/_reports/receiver_dp8_noise_repeat_20260923_v1/campaign_result.json)
- [보호 PNG 패키지](../../code_working/_reports/receiver_dp8_noise_repeat_20260923_v1/protected_png/learning_rule.json)

이번 허용된 한 bank 반복은 완료했다. 연구 단계2 전체·강한 선행 비교·독립 확인까지 완료된 것은 아니며,후속 실행은 예약하지 않는다.
