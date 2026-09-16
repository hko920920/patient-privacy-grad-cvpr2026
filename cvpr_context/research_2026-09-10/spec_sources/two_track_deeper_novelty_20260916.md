# 두 방향 추가 원문 대조: 현재 v0의 환원과 남은 문제

작성: 2026-09-16. 범위: 선택한 근접 원문을 직접 읽은 독립 신규성·논리 점검. 새 GPU, gradient 수집, 학습, 공격 fitting, 성능 통계는 실행하지 않았다. 아래 반례와 항등식은 수학적 설명이며 모델 관측값이 아니다. 이번 점검은 보호 설계 효율과 보호 평가 개선이라는 두 목표를 유지한다. 특정 v0의 약한 신규성을 전체 연구 방향의 불가능성으로 확대하지 않는다.

## 1. 먼저 내려야 할 좁은 판단

| 방향 | 이번에 더 명확해진 직접 충돌 | 현재 남는 질문 |
|---|---|---|
| 보호 설계 효율 | 외부 clipping 영역의 방향 분산 계수는 Bollapragada–Byrd–Nocedal의 orthogonality test와 상수배까지 같다. 두 pilot의 plug-in도 같다. sqrt 배분도 표준이다. | 이 구성의 DP diffusion 적용이 실제 비용과 optimizer까지 개선하는가. 특히 단순 per-user clipped MSE가 전체 업데이트의 오차·편향을 잘 대변하는가. 아직 새로운 연산과 효능을 입증한 것은 아니다. |
| 보호 평가 개선 | 고정 FPR에서 classifier 비교, 시간·quantile 동시 신뢰구간, maximin 식별, shared feedback 기반 관측 배분은 각각 직접 선행이 있다. | 같은 환자에서 나온 상관된 여러 점수와 불확실한 FPR 제약을 실제로 활용하여 강한 결합 기준보다 적은 비용으로 보호 선택 오류를 줄이는가. 현재 조합 설명만으로 방법 기여를 인정하기 어렵다. |

따라서 두 목표를 닫거나 임의의 κ·cap 변경으로 방법을 구제할 단계가 아니다. **현재 v0에서 새롭다고 생각했던 연산의 일부가 직접 선행으로 환원됨을 인정하고, 남은 기술적 간격을 근거 없이 이미 해결한 것처럼 쓰지 않는 것**이 이번 점검의 결론이다. 모든 관련 방법이 먼저 실험에서 실패해야 후보를 제안할 수 있다는 조건도 붙이지 않는다.

## 2. 방향 1: Bollapragada Eq. (3.2)–(3.3)와 현재 계수의 정확한 대응

원문: Bollapragada, Byrd, Nocedal, *Adaptive Sampling Strategies for Stochastic Optimization*, SIAM Journal on Optimization 2018. 직접 읽은 범위: 서론, §2의 inner-product test, §3.1 PDF p.8의 Eq. (3.1)–(3.3), §3.2의 수렴 가정, §4의 practical algorithm 일부. [원문 PDF](https://arxiv.org/pdf/1710.11258), [출판 DOI](https://doi.org/10.1137/17M1154679).

원 논문은 norm 자체를 정확히 추정하는 대신, descent에 유용한 방향을 얻는 데 필요한 표본수를 조절한다. true gradient를 μ≠0, 개별 stochastic gradient를 G, 공분산을 Σ, v=μ/||μ||, P=I−vvᵀ라고 쓰자. E[G]=μ이므로 E[PG]=0이다. Eq. (3.2)는

```text
E ||G − (Gᵀμ/||μ||²) μ||² / n ≤ ν² ||μ||²
⇔ tr(PΣP) / (n ||μ||²) ≤ ν².
```

현재 후보의 user clip을 k_C(z)=z·min(1,C/||z||)라 하자. ||μ||>C인 매끄러운 외부 영역에서는

```text
J_C(μ) = (C/||μ||) P
a_clip = tr(J_C Σ J_Cᵀ) = C² tr(PΣP)/||μ||²
local clipped MSE ≈ a_clip/n.
```

따라서 **기존 Eq. (3.2)의 정규화된 좌변은 현재 local postclip MSE 근사를 C²로 나눈 값과 같다.** 기존 논문이 clip Jacobian이라는 이름을 붙이지 않았다는 사실은 새로운 방향 오차 계수의 근거가 되지 않는다. Eq. (3.2)는 exact variance 조건이고, 실제 clipped MSE와의 동일성은 주장하지 않는다. 현재 쪽에서도 이는 clip 경계를 자주 넘지 않는 국소 선형화이지 모든 분포·n에서의 등식이 아니다.

Eq. (3.3)는 μ와 공분산을 같은 sample로 추정한다. pilot 두 개 g₁,g₂, m=(g₁+g₂)/2, d=g₁−g₂이면 표본 공분산은 Σhat=ddᵀ/2이다. m≠0에 대해 P_m=I−mmᵀ/||m||²라 두면

```text
현재 â = ||J_C(m)d||²/2
       = C² ||P_m d||² / (2||m||²).

기존 Eq. (3.3)의 sample orthogonal-variance 계수
       = ||P_m d||² / (2||m||²)
       = â/C².
```

기존 검사에는 pilot sample 수 n_pilot으로 나누는 항이 있고, 현재는 이 계수로 별도의 fresh-final sample 수 n_final을 결정한다. **계수와 두 표본 plug-in의 환원**을 말하는 것이지, 전체 알고리즘·최종 sample 재사용·stopping rule까지 동일하다는 말은 아니다. 기존은 sufficient-sample test, 현재는 계수에 sqrt 배분·cap을 붙인 별도 controller다. 그러나 sqrt allocation은 표준 MC/Neyman 원리이므로 그 차이만으로 강한 방법 신규성을 확보하기 어렵다.

기존의 정확한 variance 조건에 대한 수렴 정리를 current pilot-2 plug-in에 자동 적용할 수도 없다. 현재 m과 d가 같은 두 draw에서 나오므로 μ 방향 추정의 오차가 크고, â의 불편성이나 안전한 boundary 판단을 보장하지 않는다. clipping inactive 영역에서 J=I가 되는 경우도 일반 variance/norm sample-size 설계와 겹친다.

**판정:** 현재 v0를 “raw variance 대신 새 clipping 방향 분산을 발견한 방법”으로 제시하면 직접 반론을 받는다. 이 직접 선행은 강한 directional adaptive-sampling baseline에 포함시킬 근거이며, 보호 효율 목표 자체를 폐기할 근거는 아니다.

## 3. clipping bias와 nonlinear MC도 이미 별개의 직접 선행이다

### 3.1 preclip 평균·분산 감소와 clipping 편향

Xiao et al., *A Theory to Instruct Differentially-Private Learning via Clipping Bias Reduction*, IEEE S&P 2023. 직접 읽은 범위: abstract/서론, §3.1–3.3 특히 PDF pp.7–8의 p-average, §4 끝–§5 도입 PDF p.11. [저자 원문](https://people.csail.mit.edu/devadas/pubs/Clipping_Bias.pdf).

이 논문은 stochastic-gradient noise가 clipping/normalization bias를 만들 수 있고 반복 학습만으로 그 항이 없어지지 않는 구조를 분석한다. §3.3은 p개 gradient를 먼저 평균하고 clip하여 분산·bias 항을 낮추는 연산을 다룬다. 또한 inner–outer momentum과 normalization 기반 접근을 제시한다. §4 끝은 importance sampling이 aggregate non-clipped gradient 분산을 줄이는 것과 실제 clip되는 per-sample gradient에 미치는 효과를 구분한다.

따라서 “raw variance 감소가 clipping 이후의 유용성과 다를 수 있다”, “clip 전에 더 평균하면 bias가 바뀐다”, “gradient 방향을 고려해야 한다”는 수준은 이미 점유됐다. 다만 그 p-average는 서로 다른 보호 record를 묶는 경우의 privacy 비용을 함께 다룬다. **동일 보호 user 안의 추가 noise draw와 여러 다른 user/record의 grouping은 동일한 sensitivity·sampling 비용이 아니다.** 원문의 grouping privacy penalty를 현재 user-local 반복에 그대로 옮기면 안 된다. 읽은 범위에서 현재 diffusion용 user-local pilot/fresh-final controller 전체와 같은 알고리즘을 확인한 것은 아니다.

### 3.2 nonlinear output에 따라 inner sample 수를 바꾸는 원리

Giles & Haji-Ali, *Multilevel Nested Simulation for Efficient Risk Estimation*, SIAM/ASA Journal on Uncertainty Quantification 2019. 직접 읽은 범위: 서론, §2.2 PDF pp.5–6, §2.3.1 PDF pp.7–8. [원문](https://arxiv.org/pdf/1802.05016).

이 연구는 E[H(E[X|Y])] 같은 threshold를 거친 inner expectation에서, |conditional mean|/conditional standard deviation으로 boundary 근처의 어려움을 판단해 inner sample 수를 적응시킨다. doubling으로 pilot 비용을 관리하고, 정규화된 higher moment와 boundary 부근 density 조건을 둔다. 따라서 “비선형 결과를 바꾸지 않는 MC 오차에는 덜 계산하고 경계에 더 계산한다”는 큰 원리도 새롭지 않다.

차이는 scalar threshold/MLMC 목적과 bounded vector user contribution이다. 해당 정리를 현재 vector clipping·두 pilot·AdamW에 자동 적용할 수 없다. 반대로 boundary guard나 smoothness 조건을 추가하는 것만으로 새 원리를 만들었다고 할 수도 없다. 이 원문은 정확한 동일 DP controller의 존재를 증명하는 자료가 아니라, 큰 설계 아이디어의 신규성 경계를 정하는 자료다.

### 3.3 a/n은 유한 n의 실제 clipped MSE를 일반적으로 대변하지 않는다

다음은 모델 실험이 아닌 정확한 설명용 반례다. C=1, μ=0이고 X가 확률 1−p로 0, 확률 p/2씩 ±L을 취하며 L>2라고 하자. target k_C(μ)=0이다.

```text
n=1: E[k_C(X)²] = p.
n=2: E[k_C((X₁+X₂)/2)²]
   = 2p(1−p) + p²/2
   = 2p − 1.5p².
```

0<p<2/3에서는 n=2의 오차가 n=1보다 크다. 둘 중 하나만 극단값일 때 평균 이후에도 clipping되어 오차 사건의 확률이 늘기 때문이다. 유한 분산 분포이고 μ는 clip 내부인데도 나타난다. 이는 current a/n을 전역 단조성·최적 배분 법칙으로 해석할 수 없음을 보여준다. **현재 문서가 이미 국소 근사라고 부른 것은 맞으며**, 이 반례로 모든 실제 분포에서 실패한다고 주장해서도 안 된다. 임의 κ를 조정해 이 반례를 피해갔다는 사실만으로 paper 기여가 생기는 것도 아니다.

## 4. 방향 1에서 정확히 무엇이 아직 환원되지 않았나

### 4.1 단순히 목적 이름을 전체 optimizer utility로 바꾸는 것으로 충분하지 않다

고정 sampled-user 집합에 조건부로 h_i=k_C(μ_i), b_i=E[ĥ_i]−h_i, V_i=Cov(ĥ_i)라 하자. user-local 난수 독립, 공개 분모 M, 합에 N(0,σ²C²I_d)를 더하는 경우, clipped exact-user reference g=(Σh_i)/M에 대한 한 단계 입력 오차는

```text
E || ĝ_DP − g ||²
 = ||Σ_i b_i||²/M² + Σ_i tr(V_i)/M² + σ²C²d/M².
```

현재 per-user 목표의 합은 Σ_i{tr(V_i)+||b_i||²}이다. 전체 입력에는 **서로 다른 user의 b_iᵀb_j 교차항**이 생기므로 동일 목적이 아니다. 또한 unclipped 학습목표와 비교하면 deterministic clip bias도 남고, 실제 AdamW update에서는 기존 moments와 preconditioning이 개입한다. user sampling까지 평균내려면 별도 sampling 항을 다뤄야 한다. 위 식은 그 전체 학습 성능 정리가 아니다.

이 간격은 실제 연구 문제를 더 구체화한다. 그러나 Bollapragada는 원래 expected descent를 목표로 표본수를 정하고 Xiao는 clip noise·bias와 최적화 수렴을 함께 분석한다. 따라서 “한 단계 utility를 본다”라는 목적 교체만으로 기존보다 새로운 방법이 되지는 않는다. 현재 v0는 이 교차 bias·DP noise·AdamW 효과를 측정하거나 최적 배분하는 새 연산을 아직 제시하지 않았다. Gaussian noise가 지배적인 상태에서는 inner-MC 개선이 작을 수 있다는 부정적 경우도 보존해야 한다.

### 4.2 shared compute와 fixed user adjacency도 문제이지 해결된 기여가 아니다

현재 controller가 각 user의 자료와 독립 난수만 쓰고 마지막에 norm C로 제한하면, 고정 q·공개 분모의 user add/remove sensitivity C를 유지할 수 있다는 구조는 타당하다. replacement에는 2C 구분이 필요하다. 이 bounded user mapping은 기존 user-DP 설계 원리에 속한다.

반면 private batch 전체의 pilot을 보고 남은 budget을 경쟁 배분하거나 중간에 user를 버리면, 한 user 변경이 다른 user들의 출력·포함 여부까지 바꿀 수 있다. 단순 C sensitivity 논거가 더 이상 충분하지 않을 수 있다. 실행시간/할당 로그가 공개되면 출력 모델 DP와 별개의 관측 경로도 있다. **그 문제를 지적하는 것과, 유용한 coupled allocator 및 올바른 보장을 실제로 만든 것은 다르다.** 아직 그런 새 알고리즘·증명을 확보하지 않았다.

### 4.3 현재 남길 수 있는 연구 상태

- 확정: directional variance coefficient·pilot-2 plug-in·sqrt allocation 자체로는 새 방법 주장을 하기 어렵다.
- 미확인: 이 계수의 user-DP diffusion 적용이 좋은 fixed multiplicity와 기존 directional sampler보다 실제 비용·품질에서 유용한가.
- 미해결: 실제 유한-n postclip bias/variance, 전체 update의 비분리성, 공개/DP controller와 batching 비용을 동시에 다룰 실질 연산이 있는가.
- 판정하지 않은 것: 보호 효율이라는 목표의 불가능성, 새로운 적절한 bounded estimator/controller의 부재, 모든 선행의 실험적 실패.

현재 단계에서 κ나 cap을 바꾸어 새 후보 이름을 붙이는 것보다, **v0가 기존 directional adaptive sampling의 DP 적용인지, 그 이상으로 해결해야 할 구체 문제가 무엇인지**를 먼저 인정하는 것이 맞다. 적용 실험이 양성이어도 그 자체로 새 연산의 발견을 뜻하지 않는다.

## 5. 방향 2: NP-ROC + maximin + feedback graph의 현재 위치

### 5.1 기존 원문이 이미 처리한 구성요소

| 구성요소 | 직접 원문과 읽은 범위 | 현재 후보에서 새롭다고 할 수 없는 부분 |
|---|---|---|
| 같은 FPR에서 classifier 비교·미결정 | Tong, Feng, Li, *Neyman–Pearson classification algorithms and NP receiver operating characteristics*, Science Advances 2018, PDF pp.3–5와 p.8. [원문](https://stat.arizona.edu/sites/default/files/2023-09/Journal%20Paper%20091823.pdf) | class-0 order-statistic threshold, FPR violation 확률 통제, NP-ROC bands, 한 band가 우세하지 않으면 결론을 유보하는 비교. |
| 시간 및 quantile 전체에 유효한 구간, quantile arm 식별 | Howard & Ramdas, *Sequential estimation of quantiles with applications to A/B testing and best-arm identification*, Bernoulli 2022. PDF pp.1–2의 명제·설명 확인. [저자 원문](https://www.stevehoward.org/media/BEJ1388.pdf) | time-uniform DKW/quantile confidence sequences, 모든 quantile 동시 추론, LUCB 계열 quantile identification. 이번 재독은 앞부분이며 전체 proof를 다시 검산하지 않았다. |
| 여러 공격 중 최악 위험을 갖는 defense 선택 | Garivier, Kaufmann, Koolen, *Maximin Action Identification: A New Bandit Framework for Games*, COLT 2016. 기존 프로젝트 검토와 공식 원문 재대조. [공식 논문](https://proceedings.mlr.press/v49/garivier16b.html) | maximin action 구조, M-LUCB/Racing, fixed-confidence stopping. ROC calibration으로 reward를 구성했다고 이 틀이 새로 생기지 않는다. |
| 한 query가 여러 arm에 정보를 제공할 때 배분 | Russo, Song, Pacchiano, *Pure Exploration with Feedback Graphs*, AISTATS 2025. §2.2, §3.2, §4 선택 부분. [원문](https://arxiv.org/pdf/2503.07824) | graph feedback의 optimal allocation, forced exploration, likelihood-ratio stopping과 비용 효율적 pure exploration. |

추가로 unknown constraint를 가진 best-arm identification 자체도 선행이다. [Wang et al., AISTATS 2022](https://proceedings.mlr.press/v151/wang22h.html)는 안전 제약과 reward를 함께 학습하며, [Yang et al., Automatica 2025](https://www.sciencedirect.com/science/article/abs/pii/S0005109825001153)는 vector performance의 constrained best-arm 문제를 다룬다. 여기서는 공식 abstract/intro 범위만 확인했으며, 현재 ROC 문제와 완전히 동일하다고 단정하지 않는다. 특히 offline 공격을 관측하는 것은 unsafe action의 실제 실행과 같은 문제는 아니다.

### 5.2 현재 ROC risk 구간은 타당하지만 새 정리로 제시할 수 없다

모든 threshold와 시간에 대해 class-0/1 tail probability의 동시 bands L₀≤P₀≤U₀, L₁≤P₁≤U₁가 성립하면

```text
R_da(α) = sup_{τ: P₀,da(τ)≤α} P₁,da(τ)
L_da = sup_{τ: U₀,da(τ)≤α} L₁,da(τ)
U_da = sup_{τ: L₀,da(τ)≤α} U₁,da(τ)
⇒ L_da ≤ R_da(α) ≤ U_da.
```

이는 feasible-set 포함 관계로 따른다. finite portfolio에서 max_a를 적용하고, 한 defense의 upper bound가 다른 모든 defense의 lower bound보다 작을 때 선택하는 것도 표준 interval decision 원리다. ties, always-nonmember classifier, fixed score orientation은 필요하지만 새 공격·새 감사 보장은 아니다. 원래 NP-ROC의 표본 분할과 IID 가정도 생략할 수 없다.

### 5.3 feedback graph 선행의 theorem을 그대로 가져오면 안 되는 지점

읽은 TaS-FG 모형은 arm마다 one-parameter canonical exponential family의 reward를 두고, graph가 제공하는 observations의 independent-draw 구조를 명시한다. 현재 한 환자·동일 모델 응답에서 계산한 여러 공격 점수는 일반적으로 상관되어 있고, risk는 두 arbitrary score distribution과 FPR feasibility에 의존한다. 따라서 다음을 구분해야 한다.

- 기존 TaS-FG의 관측 배분 아이디어를 실제 cost-aware baseline으로 적용하는 것: 가능성을 검토할 수 있다.
- 현재 correlated patient-score setting에서 기존 asymptotic optimality·fixed-confidence proof가 그대로 성립한다고 말하는 것: 근거 없음.
- 이 가정 차이를 지적했으므로 자동으로 새 방법이라고 말하는 것: 역시 근거 없음.

공유 observation의 각 marginal에 유효한 구간을 만들고 union bound를 쓰는 것은 correlation이 있어도 가능한 기본 대안이다. 상관을 실제 활용한 더 좁은 joint uncertainty, query bundle의 비용, unknown ROC-feasibility가 allocation·stopping을 어떻게 바꾸는지 구체적으로 보여야 결합 이상의 기여를 검토할 수 있다. 단순히 셀마다 구간 폭/비용 점수를 주는 rule은 현재까지 새 최적성이나 실질 이득을 확보하지 못했다.

### 5.4 보호 평가 개선이라는 목표와 현재 방법의 기여를 분리한다

강한 공격 portfolio가 고정되어 있고, 어느 defense가 낮은 portfolio risk를 가지는지 비용을 줄여 결정하는 연구 질문은 의미가 있다. 다만 weak-attack non-detection을 privacy로 오해하면 안 된다는 사실, defense마다 calibration이 달라야 한다는 사실, low-FPR uncertainty 때문에 순위가 바뀔 수 있다는 사실만으로는 새 기여가 되지 않는다. Aerni의 adaptive privacy evaluation, Quantile MIA, NP-ROC와 이미 겹친다.

지금 문헌 대조에서 **별도의 구조적 대안을 새 방법으로 자신 있게 제시할 근거까지 확보하지 못했다.** 이름만 바꾼 세 번째 후보를 추가하지 않는다. 현재 v0에 남은 과제는 fixed-budget joint observation 선택/정지 연산을 닫고, 동일 bands·동일 cache·동일 cost를 주는 NP-ROC + M-LUCB/Racing + graph-aware 배분과 비교해 어떤 연산 차이가 실제 보호 선택 오류를 줄이는지 명확히 하는 것이다. 이는 아직 설계·효능의 미완료 항목이지, 관련 방법을 전부 먼저 실패시켜야 한다는 연구 시작 조건이 아니다.

## 6. 누락과 증거 경계

이번 파일은 clipping geometry, adaptive sampling, nested MC, NP-ROC, sequential quantile, maximin, feedback graph에 대해 지정한 절의 직접 대조를 기록한다. 모든 adaptive clipping/Adam bias-correction/DP utility allocation 논문을 전면 검토한 것은 아니다. 다음도 완료했다고 주장하지 않는다.

- pilot-2 controller의 실제 diffusion gradient 성능 또는 새로운 optimum 증명.
- 전체 DP optimizer의 biased MC allocation을 푸는 새 controller.
- correlated patient ROC에 대한 graph-bandit 최적성 정리.
- 유한 공격 portfolio가 전체 privacy risk나 DP upper guarantee를 포괄한다는 보장.
- 의료 환자 데이터에서 quality–cost 또는 보호 선택 개선.

현재의 근거 있는 진전은 **방향 1의 핵심 proxy가 기존 orthogonality test에 정확히 대응함을 확인한 것**, 그리고 **방향 2의 불확실성·maximin·shared-query 구성요소와 기존 theorem을 그대로 적용할 수 없는 관측 가정을 구분한 것**이다. 원래 두 연구 목적을 유지하면서 현재 v0의 신규성을 과장하지 않는 판단 자료로 사용한다.

## 7. 현재 v0를 더 수정하지 않을 때 남는 두 질문

추가 상황: root가 별도 CPU 검사에서 두 v0의 좋은 결과를 얻지 못했으며, 이를 임의 수정하거나 의료 GPU 실험으로 확대하지 않을 방침이라고 전달했다. 이 독립 문헌 메모는 해당 CPU 원시 결과를 직접 검산하지 않았으므로 수치·범용 실패 판정을 재서술하지 않는다. 아래는 그 결과와 무관하게 성립하는 **다음 연구 질문의 경계**이다.

| 큰 질문 | specific prior가 실제 다룬 것 | 아직 확인하지 못한 경계 | 새 변경이 없다면 어떻게 남길 것인가 |
|---|---|---|---|
| 같은 user-DP 아래 어떤 내부 계산을 줄여도 학습 효용을 유지할 수 있는가? | Bollapragada는 descent 방향에 필요한 sample 수, Xiao는 stochastic noise·preclip 평균·clipping bias와 수렴, DPDM은 noise multiplicity, ELS/ULS는 보호 단위 아래 gradient allocation을 다룬다. | 국소 per-user 오차 개선이 실제 aggregate update 개선을 대변하지 않을 때, 어느 오차를 저렴하게 측정하여 어떤 bounded 연산을 바꿀 수 있는가. bias 합의 제곱은 기존 clip-bias 문제와 연결되며, 이 식을 썼다는 것 자체로 새로운 문제를 최초 제기한 것은 아니다. | 현재 geometry allocator는 기존 directional sampling의 적용·진단 도구로 남긴다. 전체 update 목표를 다시 명명하거나 κ/cap만 바꾼 경우 새 방법으로 승격하지 않는다. |
| 제한된 감사 비용으로 보호 방법의 선택을 얼마나 신뢰할 수 있게 할 수 있는가? | NP-ROC는 고정 FPR 비교와 불확실성, Howard–Ramdas는 sequential quantile bounds, maximin 논문은 최악 항의 action 선택, TaS-FG는 shared feedback 배분을 다룬다. | 실제 모델·환자 관측에서 여러 공격이 공유하는 정보가 어느 결정 오류를 만들며, 기존 joint/marginal 구간 및 maximin 배분이 낭비하는 구체 관측은 무엇인가. 현재 correlated-ROC 가정 차이만으로 더 좋은 연산을 이미 얻은 것은 아니다. | 현재 audit selector는 유효한 비교·보류를 위한 기존 통계 도구의 조합으로 남긴다. 새 실질 관측/교정/배분 연산이나 고정 예산에서의 충분한 개선이 없으면 별도 방법 논문으로 주장하지 않는다. |

두 질문 모두 보호 설계 효율·보호 평가 개선이라는 원래 축 안에 있다. **큰 질문이 중요하다는 사실, 특정 v0가 새롭거나 잘 작동한다는 사실, 논문 기여가 성립한다는 사실은 세 가지 다른 판단**이다. 현재는 첫 질문들을 유지하면서 나머지 두 판단을 미확인으로 남기는 것이 맞다. 그 간격을 메우기 위해 즉석에서 새 미검증 알고리즘이나 임의 성공 조건을 추가하지 않는다.
