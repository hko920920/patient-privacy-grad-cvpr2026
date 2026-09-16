# 두 방향 심화 검토: 선행 환원·비용·최초 구현의 결과

**후속 연산 재설계:** [두 방향의 새로운 후보와 직접 선행 대조](TWO_TRACK_OPERATION_REDESIGN_20260916.md). 두 목표를 유지하고 현재 v0와 다른 학습·평가 연산을 구체화했다. 아래 CPU 결과와 해석 범위는 그대로 보존한다.

2026-09-16. 현재 전체 **2번: 근접 선행과 논문 설계의 구체화** 안에서, [직전 두 후보](TWO_TRACK_PAPER_DESIGNS_20260916.md)의 논리와 작은 구현을 확인했다. 의료 모델의 효능 검증이나 논문 기여 완성 단계로 넘어간 것은 아니다.

**현재 두 구현 v0를 그대로 의료 본학습으로 확대할 근거는 부족하다.** 1번의 핵심 오차 계수는 가까운 기존 방법에 환원되고, 예비 계산 비용과 작은 표본의 오판 때문에 고정 반복보다 대부분 불리했다. 2번의 불확실성 구간은 타당한 표준 도구지만, 제안한 계산 배분이 균등 배분보다 비용과 보류율에서 불리했다. 보호 설계·효율과 보호 평가 개선이라는 두 목표는 유지한다.

이번 결과는 **우리 초안의 성립 여부를 확인한 결과**다. 이를 기존 보호 SOTA 전체의 공통 실패를 발견했다는 성과로 세지 않는다. 유리한 결과를 얻기 위한 C·κ·예산·분포 조정은 하지 않았다.

## 1. 이번에 끝낸 범위

| 작업 | 완료물 | 증거의 범위 |
|---|---|---|
| 직접 선행 추가 대조 | 방향 분산 식의 정확한 대응, clipping bias·nested MC·순차 quantile·maximin·공유 관측의 점유 범위 | 지정한 원문 절의 검토. 전수 신규성 확인 아님 |
| 방향 1 CPU 비교 | 사전 7개 합성 분포 × 23개 규칙 = 161행, paired 비교 42개 | 같은 synthetic population의 독립 calibration/test. 환자·영상 품질 실험 아님 |
| 방향 2 CPU 비교 | 사전 4조건 × 3규칙 × 64반복 = 768회 | 알려진 점수 분포, 고정 3공격·2 threshold와 항상 음성 규칙. 실제 DP 보호모델 비교 아님 |
| 독립 검산 | 방향1 비용·MSE 161행, 방향2 저장 행동 61,852개와 최종 위험 구간·결정 768회 | 실행 모듈을 import하지 않는 별도 코드로 저장 결과 재구성 |

방향1 본 계산 **8.041초**, 방향2 **15.611초**였다. 이는 합성 배열의 CPU 실행시간이다. 설계·구현·원문 대조·해석·기록 시간이나 향후 GPU 학습시간과 다르다. 새 GPU·의료 학습·공격 실행은 0이다.

원자료: [방향1 protocol](spec_sources/track1_cpu_allocation_20260916/protocol.json), [결과 CSV](spec_sources/track1_cpu_allocation_20260916/results.csv), [상세 해석](spec_sources/track1_cpu_allocation_20260916/해석.md), [방향2 protocol](spec_sources/track2_cpu_audit_20260916/protocol.json), [결과 CSV](spec_sources/track2_cpu_audit_20260916/results.csv), [상세 해석](spec_sources/track2_cpu_audit_20260916/해석.md), [독립 검산 JSON](spec_sources/two_track_deeper_independent_verification_20260916.json).

## 2. 방향 1: 왜 그럴듯한 계산 배분이 아직 방법 기여가 아닌가

### 2.1 새로 확인한 직접 선행과의 대응

Bollapragada–Byrd–Nocedal의 *Adaptive Sampling Strategies for Stochastic Optimization*은 gradient의 방향에 직교하는 변동을 측정해 필요한 표본수를 정한다. §3.1의 Eq. (3.2)–(3.3)을 현재 외부 clipping 영역의 식과 대조하면, **방향 분산 계수와 두 pilot의 plug-in이 상수배까지 같다.** 원 논문 전체 알고리즘이 현재 controller와 동일하다는 뜻은 아니지만, 이 계수를 새롭게 발견했다고 주장할 수는 없다. [원문](https://arxiv.org/pdf/1710.11258)

기호를 `v=μ/||μ||`, `P=I−vvᵀ`로 두면 다음과 같다.

```text
기존 방향 분산: tr(PΣP)/(n ||μ||²)
현재 외부 clip의 국소 오차: C² tr(PΣP)/(n ||μ||²)
```

Xiao et al.의 IEEE S&P 2023 논문은 clipping 전 평균과 clipping bias를 이미 분석한다. Giles–Haji-Ali의 nested simulation은 비선형 결과의 경계와 inner-sample 수를 함께 다룬다. 따라서 반복·방향·경계라는 큰 발상도 각각 직접 선행이 있다. 단, 서로 다른 보호 record를 묶는 연산과 같은 user 내부에서 반복하는 연산의 개인정보 비용은 구분해야 한다. [Clipping Bias](https://people.csail.mit.edu/devadas/pubs/Clipping_Bias.pdf), [Nested Simulation](https://arxiv.org/pdf/1802.05016)

정확한 식 대응·읽은 절·적용 경계는 [추가 신규성 메모](spec_sources/two_track_deeper_novelty_20260916.md)에 적었다.

### 2.2 예비 계산의 비용부터 이겨야 한다

국소 오차를 `a/r`, user당 전체 예산을 B, 예비 계산을 p라고 놓자. `a`를 완벽히 아는 연속 oracle도 비용을 포함하면 다음 조건이 필요하다.

```text
고정 반복 오차 = E[a]/B
oracle 오차 = (E[sqrt(a)])²/(B−p)
oracle / 고정 = B/(B−p) / (1+CV(sqrt(a))²)
개선 조건: CV(sqrt(a))² > p/(B−p)
```

p=2일 때 B=4에서는 `CV(sqrt(a))>1`, B=8에서는 약 `.577`보다 커야 한다. 즉, user 사이에 추가 계산의 가치가 충분히 달라야 예비 비용을 만회할 수 있다. 실제 규칙에는 최소 반복·정수화·상한·추정오판까지 붙는다. 이 계산은 표준 배분 원리와 비용 항등식이며 새 정리가 아니다. **a/r라는 국소 모형 안의 조건**으로 한정하며 실제 유한-R clipping에 대한 보편적 필요조건은 아니다.

### 2.3 고정한 21개 조건 중 20개에서 고정 반복보다 오차가 컸다

C=1, κ=1, pilot=2, 최종 R∈{1,2,4,8,16}, 전체 기대비용 예산 4/8/16을 고정했다. λ와 좋은 고정 R 혼합은 독립 calibration에서만 선택했다. 고정 방법도 adaptive의 calibration 기대비용에 맞췄으며, test 실현 비용의 차이는 따로 저장했다.

아래는 예산8에서 **geometry MSE / 비용을 맞춘 고정 반복 MSE**다. 1보다 작아야 제안에 유리하다.

| 사전 합성 조건 | 오차 비율 |
|---|---:|
| 이산 radial/tangential, clip 바깥 | 3.203 |
| Gaussian radial/tangential, clip 바깥 | 5.050 |
| clip 안쪽, 서로 다른 분산 | 0.994 |
| clipping 경계 | 2.651 |
| 같은 분포 두 타입 | 1.501 |
| 상관된 radial/tangential | 2.132 |
| 128차원 큰 MC 잡음 | 1.307 |

유일한 작은 개선은 약0.6%였고, 그 조건에서는 기존 raw-variance 배분과 제안 geometry 배분이 모든 draw에서 같은 R를 골랐다. 따라서 제안 고유의 성과가 아니다. 전체 예산·MC 구간·실현 비용은 원자료에 남겼다. 이 MC 구간을 의료 환자 성능의 신뢰구간으로 해석하지 않는다.

### 2.4 무엇이 막혔는지 결과와 계산이 연결된다

**작은 pilot의 오판.** 이산 tangential 예시에서 pilot 두 개가 같은 부호이면 표본 분산이 0이 된다. v0는 추가 반복이 유용한 타입의 약 절반을 R1에 남겨 놓았다. 타입과 실제 유한-R 위험을 아는 oracle은 pilot2 비용까지 부과해 예산8에서 고정 R8보다 약19% 낮은 MSE를 얻었다. 이는 그 합성 분포에 배분 기회가 있다는 진단이며, 제안의 실용 성능이 아니다.

**국소 Jacobian이 놓치는 tail.** Gaussian radial 예시에서는 평균이 clip 바깥에 있어 Jacobian의 radial 계수가0이어도, 최종 표본 평균이 안쪽으로 들어올 수 있다. v0는 그 타입의 89.7%를 R1로 보냈다. 정확한 Gaussian 기대값과 저장 R 빈도로 계산한 오차 `.0012936911`은 관측 `.0012949201`을 가깝게 재현했다. 독립 최종 표집의 드문 큰 오차를 pilot 평균의 국소 판단이 충분히 다루지 못했다.

**반복을 늘리면 항상 postclip MSE가 줄어드는 것도 아니다.** C=1, 평균0, X가 확률1−p로0, p/2씩 ±L, L>2를 갖는 정확한 예에서는 R1 오차가 p, R2 오차가 `2p−1.5p²`다. p=.1이면 `.1→.185`로 커진다. 유한-R에서 a/r를 전역 법칙으로 쓸 수 없다는 반례이며 실제 의료 분포라는 주장은 아니다.

### 2.5 보호 효율 목표에서 남는 질문

현재 개인별 clipped-estimate MSE를 줄이는 목표는 전체 DP 업데이트를 좋게 만드는 목표와도 같지 않다. 고정 user 집합, 공개 분모 M, 독립 user-local 난수에 조건부로, exact clipped contribution을 hᵢ, 추정 편향을 bᵢ, covariance를 Vᵢ라 하면 다음은 정확한 항등식이다.

```text
exact clipped-user 평균에 대한 noisy aggregate의 MSE
= ||Σ bᵢ||²/M² + Σ tr(Vᵢ)/M² + d σ² C²/M².
```

개인별 MSE의 합에는 없는 **user 사이 편향의 교차항**이 있다. 원래 unclipped gradient를 목표로 하면 deterministic clipping bias도 남고, AdamW와 user sampling까지 포함하면 별도의 분석이 필요하다. 이 식 자체는 새 기여가 아니다.

따라서 큰 방향1에서 남기는 질문은 **같은 patient-DP와 실제 계산비용 아래, 어떤 bounded 학습 연산을 바꿔 편향·표집 오차·DP noise의 전체 손실을 줄일 것인가**다. 기존 directional sampling과 clipping-bias 방법이 이 부분을 이미 상당히 다루므로, 목적 이름을 바꾸는 것으로 새 방법이 되지는 않는다. 현재 v0의 κ·R를 조금 조정하는 일은 우선 작업으로 두지 않는다.

## 3. 방향 2: 통계적으로 타당한 구간과 효율적인 계산 배분은 다르다

### 3.1 비교 조건과 결과

고정 점수3개, threshold2개와 항상 음성 규칙, 환자 FPR α=.05, 공동 오류 예산 β=.05를 사용했다. 회원·비회원 각각 최대2048명, 128명씩16회 관측, 추상 계산비용 최대100000이다. 두 점수 분포와 공유 특징 유무를 교차했다. 이는 **5% FPR의 합성 검사**이며 1% FPR 의료 감사 성능을 확인한 것이 아니다.

모든 방법에 같은 환자 stream, Clopper–Pearson/union-bound 구간, 특징 cache, 비용, 탐색·정지 규칙을 주었다. `width`는 투명한 maximin-width/shared-cost adaptation이며 **원형 M-LUCB나 TaS-FG의 완전한 재현이 아니다.** v0는 앞선 한 관측 규칙을128명 batch에 맞춰 사전 고정한 버전이다.

각 셀은 **올바른 결정 수/64 · 보류 포함 평균비용**이다.

| 조건 | 균등 배분 | width adaptation | v0 |
|---|---:|---:|---:|
| 회원 추정 병목, 공유 없음 | 64 · 23,260 | 53 · 60,416 | 52 · 60,272 |
| 회원 추정 병목, 공유 있음 | 64 · 17,748 | 64 · 47,008 | 64 · 52,774 |
| 비회원 FPR 병목, 공유 없음 | 30 · 85,038 | 11 · 97,816 | 1 · 99,728 |
| 비회원 FPR 병목, 공유 있음 | 38 · 75,934 | 37 · 90,704 | 22 · 96,756 |

768회에서 잘못된 결정은0이었다. 하지만 결정하지 못한 경우가 많으므로 이를 성능 성공으로 세지 않는다. 알려진 분포별64개의 IID cohort를 재사용해 총128개 cohort이며, 768개 독립 모집단 실험이라는 뜻도 아니다. 점수들은 같은 환자 latent를 공유해 상관되어 있다. 위 비용은 특징1/4/16이라는 합성 단위이며 실제 GPU 초가 아니다.

### 3.2 v0가 병목을 제대로 보지 못한 구체적 이유

FPR 제약은 단순한 구간 폭과 다르다. **상한이 기준 안으로 들어오는지**가 threshold 사용 가능성을 바꾼다. 국소적으로 폭만 줄이는 v0는 그 경계를 넘어 얻을 가치를0으로 평가할 수 있다.

사전 오류 배분을 적용한 동일 CP 계산에서 128명 중 오탐0이면 상한은 `.072543`이다. 다음128명에서도 오탐0이면 실제 상한은 `.036954`가 되어 α=.05를 통과한다. 그런데 v0의 가상 폭축소는 `.051395`를 예측해 여전히 통과하지 못한다. 이는 추가 점수 분포 실험이 아닌 저장 규칙의 정확한 산술 예다.

비회원 병목·공유 없음의 저장 trace에서는, **가능한 모든 비회원 행동의 가상 개선값이0인데 회원 행동의 값은 양수여서 회원을 선택한 경우가2,111번**이었다. 비회원 기준이 결정을 막고 있어도 회원 쪽 정밀도에 계산이 쏠릴 수 있음을 확인했다. 이 결과로 모든 적응 배분이 나쁘다고 일반화할 수는 없다. 현재 v0의 효율 예측을 지지하지 못한다는 좁은 판단이다.

### 3.3 선행이 이미 해결한 것과 남는 것

고정 FPR에서의 비교와 구간은 NP-ROC, 시간·분위수 동시 구간과 quantile arm 식별은 Howard–Ramdas, 최악 공격을 고려한 선택 구조는 maximin, 공유 관측은 feedback-graph pure exploration에 직접 선행이 있다. [NP-ROC](https://stat.arizona.edu/sites/default/files/2023-09/Journal%20Paper%20091823.pdf), [Sequential Quantiles](https://arxiv.org/abs/1906.09712), [Maximin](https://proceedings.mlr.press/v49/garivier16b.html), [Feedback Graphs](https://arxiv.org/abs/2503.07824)

같은 환자의 여러 점수는 상관되고 ROC 위험에는 FPR feasibility가 개입하므로, 기존 독립 reward 모형의 최적성 정리를 그대로 가져올 수는 없다. 그렇다고 그 가정 차이만으로 새 기여가 생기는 것도 아니다. 현재의 marginal 구간과 union bound는 이 상관에도 적용할 수 있는 기본 대안이다.

큰 방향2에 남기는 질문은 **어떤 정보·공격·표본으로 평가했을 때 실제 보호 결론이 어디까지 정당화되고, 어떤 개선이 중요한 결정을 더 정확하게 만드는가**다. 새 순차 allocator가 필수는 아니다. 현재 구간·보류·cache는 평가 도구로 재사용하고, 효율적인 새 알고리즘이라는 주장은 보류한다. 유한 공격군의 위험 상한을 전체 프라이버시나 DP 상한 보장으로 부르지 않는다.

## 4. 다음 작업을 어떻게 바꾸는가

이번에는 두 **구현 후보**의 큰 학습 진입을 보류한다. 두 **연구 방향**을 폐기하지 않는다. 직전 문서의 ‘1번 v0를 우선 실제 공개 gradient에 적용’ 일정은 이 결과로 갱신한다.

| 큰 방향 | 이어서 좁힐 구체적인 질문 | 다음 설계에 필요한 연결 |
|---|---|---|
| 1. 보호 설계·효율 | user-DP 고정 계산량 학습, DPDM/DP-LoRA, clipping-bias 방법에서 실제 효용을 제한하는 연산은 무엇인가 | 기존 연산과 바꿀 연산, 유지할 privacy/objective, 사라지는 오차와 새 비용을 같은 식에서 설명 |
| 2. 보호 평가 개선 | 강한 기존 평가를 적용해도 어떤 정보 제한에서 보호 결론이 미확정되며, 그 결론을 개선할 수 있는가 | 구체적 보호 선택 또는 주장 범위, 기존 적응 공격·NP-ROC가 다룬 것, 새 정보/평가 절차가 바꾸는 결정 |

현재 확보한 것은 원문 환원과 v0의 실패 원인이다. **그 실패를 해결할 새 연산·증명·의료 효능은 아직 확보하지 못했다.** 다음 산출물은 v0 이름을 바꾼 세 번째 휴리스틱이나 더 큰 실험 목록이 아니라, 위 두 큰 질문 중 특정 기존 방법과 실제 변경을 연결한 설계여야 한다. 모든 강한 baseline이 먼저 실패해야 제안을 할 수 있다는 과도한 조건도 다시 붙이지 않는다.

## 5. 재현·기록

- [원문 추가 대조와 정확한 수식 대응](spec_sources/two_track_deeper_novelty_20260916.md)
- [방향1 사전 명세·실행](spec_sources/track1_cpu_allocation_20260916/run_cpu_allocation.py), [별도 산술 검산](spec_sources/track1_cpu_allocation_20260916/arithmetic_audit.json)
- [방향2 사전 명세·실행](spec_sources/track2_cpu_audit_20260916/run_cpu_audit.py), [저장 trace 검산](spec_sources/track2_cpu_audit_20260916/verification.json)
- [root 독립 검산 코드](spec_sources/verify_two_track_deeper_20260916.py), [결과](spec_sources/two_track_deeper_independent_verification_20260916.json)

protocol과 실행 코드의 사전 SHA 결속을 확인했고, 결과 후 조건·코드를 바꾸어 다시 시험하지 않았다. 기존79편 장부, 과거 E/U·clipping 결과, K5/K10 계약은 그대로다. 이번 추가 원문 검토를 과거 전수 분석 이력으로 소급하지 않는다.
