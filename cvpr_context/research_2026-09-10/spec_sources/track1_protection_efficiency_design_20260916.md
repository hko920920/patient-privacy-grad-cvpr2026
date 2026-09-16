# 방향 1: user-level DP 아래 clipping 이후에 남는 추정 오차에 계산을 배분하는 후보

작성: 2026-09-16. 작업 위치: 2단계의 원문 대조와 구체 설계. 새 GPU 실행·학습·모델 평가 없음. 이 파일만 새로 작성했다.

## 1. 제안과 판단

**제안 후보는 “환자마다 똑같이 여러 번 평균내는 대신, 최종 user clipping 뒤에도 남는 gradient 방향 오차가 큰 환자에만 추가 Monte Carlo 계산을 쓰는 것”이다.** 공개 자료로 고정한 controller가 각 sampled user의 독립적인 짧은 pilot으로 필요한 반복 수를 정하고, 별도의 fresh draws로 최종 user mean을 만든 뒤 한 번 clip한다. 사용자 Poisson sampling, clipping norm, Gaussian noise, optimizer step 수는 유지한다.

이것은 새 DP 정의·새 user clipping·새 noise multiplicity가 아니다. 아직 발견되지 않은 방법이라고 확정하지도 않는다. 남는 설계 차이는 **raw gradient variance 최소화와 최종 bounded user contribution 오차 최소화가 다르다는 점을 계산 배분의 기준으로 사용하는 것**이다. 그 차이가 실제 diffusion LoRA gradient에서 존재하고, pilot 비용과 AdamW를 거친 뒤에도 유용한지가 미검증 연결이다.

현재까지 가장 설득력 있는 좁은 후보라는 판단이지, 완성된 CVPR 기여나 효능 판정은 아니다. 의료 이미지는 가능한 평가 도메인일 뿐 필수 가정이 아니다. E/U membership 공격 개선을 성공 조건으로 삼지 않는다. 목표는 같은 user-level 보장 아래 품질–실제 계산비용 곡선을 개선하는 것이다.

## 2. 실제 확인한 가까운 선행과 이미 점유된 부분

아래는 원문에서 확인한 범위를 명시한 것이다. 문헌 전체의 exhaustive novelty audit은 아니다.

| 원문 / 직접 읽은 범위 | 확인한 사실 | 이 후보에 주는 제약 |
|---|---|---|
| [Learning with User-Level Differential Privacy Under Fixed Compute Budgets / 원 arXiv 2407.07737](https://arxiv.org/abs/2407.07737), 로컬 B04 pp.3–5, Algorithms 1–2, Theorems 1–2, Eqs.4–7 | ELS와 ULS의 sampling/clipping 단위, user 내 example 수와 user batch 수의 고정 계산비용 절충, tight ELS accounting을 다룬다. Eq.4 이후의 단순 비교에는 clipping을 비활성화하는 bound 선택도 사용된다. | user 평균 clipping, G 조정, generic group bound보다 나은 회계는 새 기여가 아니다. 전체 보호 성능을 주장하면 tight ELS도 비교해야 한다. |
| [Mind the Privacy Unit!](https://arxiv.org/html/2406.14322v3), 로컬 B05 pp.3–4 Algorithm 2 및 p.6 | user 내부 mean을 clip하는 학습과 user별 자료 수, 자료 선택 방법을 다룬다. | “환자별 여러 장을 평균낸다”와 “가진 자료 중 몇 장을 고른다”만으로는 부족하다. |
| [Differentially Private Diffusion Models](https://arxiv.org/html/2210.09929v3), §3.2 Eqs.6–7 / Theorem 1, Appendix D.2–D.4; [공식 프로젝트](https://research.nvidia.com/labs/toronto-ai/DPDM/) | noise/timestep multiplicity를 **example clipping 전에** 평균한다. 목적함수 MC variance의 1/K 감소, gradient variance 관측, 추가 계산비용, augmentation과의 차이를 설명한다. high-noise training distribution도 이미 구성 요소다. | 평균 전에 반복하여 clipping 손실을 줄인다는 동기는 선행이다. multiplicity=1만 이기는 비교는 약하다. |
| [Differentially Private Fine-Tuning of Diffusion Models, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Tsai_Differentially_Private_Fine-Tuning_of_Diffusion_Models_ICCV_2025_paper.pdf), 접근된 abstract/introduction; [저자 코드](https://github.com/EzzzLi/DP-LORA) | pretrained diffusion의 LoRA fine-tuning과 DP를 결합하며 rank/trainable modules/noise multiplicity를 평가한다고 명시한다. 이번 조사에서 상세 표 전체를 읽지는 못했다. | 작은 trainable parameter 집합, 공개 pretraining, multiplicity 조합 자체는 현재 강한 기반선이다. |
| [Efficient Differentially Private Fine-Tuning of Diffusion Models, DP-LoDA](https://arxiv.org/abs/2406.05257), [원문 p.4 §3.1](https://openreview.net/pdf?id=Lra7Hn7Jcn) | public pretraining 위의 효율적 adaptation을 제안한다. 효율적 adaptation이 full DP fine-tuning보다 모든 품질 지표에서 항상 좋다는 결과는 아니다. | PEFT의 비용 이익과 품질 이익을 구분한다. 이번 후보도 LoRA의 이익을 자기 기여로 합산하면 안 된다. |
| [RAPID, ICLR 2025](https://arxiv.org/html/2502.12794v1), §3.3–3.4, Algorithm 2 / Eq.5 | 공개 diffusion trajectory 지식을 활용하고 private reconstruction 학습에 결합한다. | public trajectory/latent 재사용 자체는 새로운 효율화가 아니다. 원형의 record privacy와 user adaptation은 구분한다. |
| [dp-promise, USENIX Security 2024](https://www.usenix.org/system/files/usenixsecurity24-wang-haichen.pdf), §5.1–5.2 / Algorithm 1, PDF pp.6–8 | timestep 구간을 나눈 두 단계 학습, 자체 privacy 논리, private 단계 noise augmentation, public pretraining을 사용한다. | 일부 timestep에 집중하거나 diffusion noise를 활용한다는 일반 설명은 이미 있다. 해당 논문의 회계 주장을 여기서 새로운 user-level 증명으로 그대로 옮기지 않는다. |
| [CARV: Variance Reduction for Expectations with Diffusion Teachers, 2026](https://arxiv.org/html/2605.21489v1), §3, §4.2, Appendix E.6 / F.2; [저자 프로젝트](https://research.nvidia.com/labs/sil/projects/CARV/) | hierarchical MC, 비용이 큰 upstream 연산 재사용, timestep IS/stratification을 결합한다. DMD에서는 큰 gradient variance 감소에도 FID 이익이 없었다. clipping 같은 biased 연산과의 상호작용은 추가 문제로 남긴다. 저자 프로젝트의 게재 표기는 ICML 2026 SPIGM workshop이다. | image×noise 계층화와 variance-per-compute 자체는 새 기여가 아니다. clipping 상호작용을 언급한 open question도 우리 방법의 novelty/효능 증명은 아니다. |
| [DP-FinDiff, 2025 preprint](https://arxiv.org/html/2512.00638v1), §3.3 / Appendix B.2 | adaptive timestep sampling을 논의한다. Appendix에서 unbiased importance weighting을 유도하지만 실험은 correction 없이 weighted objective를 쓴다고 구분한다. | timestep 재배분만으로 새 기여를 주장하지 않는다. 같은 objective라는 주장을 하려면 sampling correction/분모를 실제로 지켜야 한다. |
| [Adaptive Sampling for Private Worst-Case Group Optimization, 2026 preprint](https://arxiv.org/html/2602.10820v1), §3 / Algorithm 2 / Theorem 1 | subgroup별 sampling을 조절하며 private group loss의 추가 privacy 비용과 clipping/sampling을 함께 다룬다. 보호 단위와 subgroup는 동일 개념이 아니다. | private adaptivity가 자동으로 무료인 것은 아니다. 우리의 controller는 외부로 공개되지 않는 한 user 내부의 bounded mapping에만 둔다. |
| [DP-aware AdaLN-Zero, 2026 preprint](https://arxiv.org/html/2602.22610v1), §3.4 Proposition 3.1 및 ablation | conditional time-series diffusion에서 conditioning-induced gradient tails를 구조적으로 제한하며 DP-SGD를 유지한다. | clipping 전 gradient 분포를 개선한다는 넓은 발상도 새롭지 않다. 여기서는 architecture 변경 대신 추가 MC 계산의 위치만 바꾼다. |

기존 clipping-bias 감소 이론/inner momentum 계열도 근접하다. [A Theory to Instruct Differentially-Private Learning via Clipping Bias Reduction](https://people.csail.mit.edu/devadas/pubs/Clipping_Bias.pdf)은 이번 접근에서 PDF fetch가 반복 실패하여 상세 theorem 대조를 끝내지 못했다. 이 계열까지 미확인인 상태에서 “첫 clipping-aware variance allocation”이라고 쓰면 안 된다.

## 3. 원인: raw variance를 많이 줄여도 bounded contribution은 안 바뀔 수 있다

고정된 모델 상태 θ와 user u의 empirical objective를 다음과 같이 둔다.

- x는 해당 user의 기록에서 균등 선택한다.
- t, ε의 법칙과 loss weighting은 원래 학습 objective와 같다.
- g_u(x,t,ε)=∇θ ℓθ(x,t,ε), μ_u=E[g_u].
- k_C(z)=z min(1,C/||z||).
- 우리가 직접 개선하려는 국소 양은 E||k_C(μhat_u)−k_C(μ_u)||²이다.

마지막 양은 실제 최적화 목표나 모델 품질 그 자체가 아니다. 이상적인 user mean의 clipped contribution을 얼마나 잘 근사하는지를 보는 중간 지표다.

### 3.1 clipping 밖에서 radial variance와 tangential variance가 다르다

μ가 clipping 경계에서 떨어져 있고 작은 MC 오차를 가정할 때:

- ||μ||<C이면 J_C(μ)=I.
- ||μ||>C이면 J_C(μ)=(C/||μ||)(I−uuᵀ), u=μ/||μ||.

따라서 iid 평균 R개에 대한 **일차 근사**는

    E||k_C(μhat_R)−k_C(μ)||² ≈ tr(J_C Σ J_Cᵀ)/R.

clipping 밖에서는 radial 오차가 일차에서 제거되고 tangential 오차가 남는다. raw trace(Σ)가 큰 user가 반드시 더 많은 MC 계산을 필요로 하지는 않는다. clipping 경계에서는 이 Jacobian 근사를 안전한 보장으로 쓰면 안 된다. 비선형 bias·큰 오차·heavy tail도 남는다.

### 3.2 정확한 2차원 설명용 반례

C=1, μ=(2,0)으로 고정한다.

- User A: ε=±(0.5,0). Raw variance trace=0.25. 어떠한 반복 평균도 x좌표가 [1.5,2.5]이므로 clipped output은 항상 (1,0). 최종 오차는 정확히 0이다.
- User B: ε=±(0,0.1). Raw variance trace=0.01로 A보다 작지만 clipped 방향은 달라진다.
- B의 R=1 mean-square error는 2−4/√4.01 = 0.002495322244310705.
- B의 R=2에서는 평균 noise가 −0.1,0,+0.1이고 확률이 1/4,1/2,1/4이므로 MSE는 정확히 절반인 0.0012476611221553524.
- Raw variance 비례/제곱근 배분은 A에 더 많은 계산을 주지만, 이 상황에서 A의 추가 계산은 bounded contribution을 전혀 개선하지 않는다.

이는 arithmetic witness이며 의료 모델·품질·DP utility 결과가 아니다. 표준 clipping 기하에서 나온다. 새 이론으로 포장하지 않는다.

경계 반례도 있다. μ=(1,0), ε=±(0.5,0)이면 E[k_C(μ+ε)]=(0.75,0)이지만 k_C(μ)=(1,0)이다. “radial variance는 항상 버려도 된다”는 규칙은 틀리다.

## 4. 실제 바꾸는 연산

### 4.1 최소 controller 후보

각 sampled user마다 다음을 수행한다. 이 명세는 아직 실행 계약이 아닌 설계다.

1. fresh pilot gradient p1,p2 두 개를 원래 user objective의 iid joint law에서 얻는다. μtilde=(p1+p2)/2, δ=p1−p2, Vtilde=||δ||²/2.
2. 공개 자료로 미리 고정한 boundary guard를 적용한다. 예를 들어 | ||μtilde||−C |가 κ√Vtilde 이하이면 총분산 Vtilde를 보수적 proxy로 사용한다.
3. 경계에서 충분히 떨어졌다는 pilot 판정에서는 atilde=||J_C(μtilde)δ||²/2를 쓴다. 단, 이것은 μtilde와 δ가 같은 pilot에서 왔으므로 **불편 추정량이라고 보장하지 않는 plug-in heuristic**이다.
4. 공개 λ와 capped set {1,2,4,8,16}으로 R_u를 정한다. 예를 들어 clip/discretize(λ√atilde)를 사용한다. λ를 해당 private batch의 다른 user 추정값으로 정규화하지 않는다.
5. Pilot과 독립인 fresh final R_u draws로 mean을 계산한다. 해당 user mean만 한 번 k_C로 clip한다.
6. user Poisson sample의 clipped contributions를 더하고 N(0,σ²C²I)를 더한 뒤 고정 공개 분모 M으로 나누어 AdamW에 전달한다.

sqrt-variance 배분 자체는 표준 Neyman/MC 원리다. 제안의 판별 가능한 차이는 총분산 대신 **clipping 뒤 실제로 남을 오차의 proxy**를 쓰는 것이다. 작은 pilot이 틀린 결정을 할 비용을 포함해야 하며, pilot=2가 충분하다는 주장은 없다.

유저 내 동일 기록을 반복할 때는 records를 먼저 균등하게 cover하는 고정 permutation-cycle 방식도 쓸 수 있다. R 결정과 final permutation을 독립으로 두고 모든 위치의 marginal이 균등하도록 해야 한다. 첫 원리 검증에서는 더 단순한 iid record/noise law로 비교하고, coverage 효과는 별도 강한 baseline으로 분리하는 것이 명료하다.

### 4.2 왜 pilot과 final을 분리하는가

Pilot P로 R(P)를 정하더라도 fresh final draw law가 원래 objective와 같으면

    E[ μhat_{R(P)} | P ] = μ, 따라서 E[ μhat_{R(P)} ] = μ.

반대로 pilot 결과가 좋을 때만 멈추고 같은 draws를 최종 mean에 넣는 방식은 일반적으로 optional-stopping/selection bias를 만든다. 이번 후보에서는 그런 재사용을 기본으로 허용하지 않는다.

이 논리는 **preclip unbiasedness만** 보장한다. E[k_C(μhat)] = k_C(μ)라는 뜻이 아니며, 두 방법의 postclip bias가 같다는 뜻도 아니다.

### 4.3 왜 user-level sensitivity를 유지할 수 있는가

공개/이전 DP 상태 θ에 조건부로, 각 user가 자기 자료와 자기 새 난수만 이용해 만들고 마지막에 norm C로 제한한 출력은 임의의 내부 계산량과 무관하게 norm≤C이다. 고정 user Poisson q와 공개 분모 M 아래 user add/remove sensitivity는 합 기준 C이다. Gaussian σC와 같은 T의 표준 user-sampled composition을 사용할 수 있다. Replacement adjacency라면 2C 차이를 별도로 반영해야 한다.

다음 조건은 필수다.

- controller가 한 user의 private pilot으로 다른 user의 포함 여부·R·clip 값을 바꾸지 않는다.
- batch 전체 private 총비용에 도달하면 뒤의 user를 버리는 early stop을 하지 않는다.
- private pilot/variance/R/record count를 모델 DP로 보호됐다고 외부 공개하지 않는다.
- user-local private pilot을 다음 step까지 보관하여 다른 사용자의 동작에 영향을 주는 전역 상태로 만들지 않는다. 초기 후보에서는 매 step fresh로 계산한다.
- private user별/전체 가변 실행시간이 adversary에게 관측되는 환경이면 별도의 leakage 분석 또는 padding이 필요하다. Padding은 wall-clock 이익을 없앨 수 있다. 모델만 release하는 trusted central training과 interactive/untrusted execution을 구분한다.
- 실행 편의를 위한 packing이 BatchNorm·cross-sample loss·global stopping 같은 경로로 user contribution을 결합하지 않게 한다.

즉 “모든 private adaptive 계산은 무료”라는 주장이 아니다. **최종 bounded per-user mapping 안에 넣고, 외부 transcript와 다른 user의 계산을 건드리지 않는 특수한 설계**다. 내부 pilot에도 비용은 들며 privacy cost와 compute cost는 다른 축이다.

## 5. image 수와 noise 수를 함께 최적화한다는 과장도 피해야 한다

한 user에 n개 기록이 있고, g개를 without replacement로 뽑아 각 기록에 r개 독립 noise를 쓰는 균형 설계를 생각하자. B는 기록별 expected gradient의 finite-population covariance(분모 n−1), W는 기록 내부 noise covariance의 평균이면

    Cov(μhat) = [(1−g/n)/g] B + W/(gr).

총 gradient evaluation L=gr와 evaluation당 비용이 같을 때, W/L은 변하지 않고 between-record 항은 g가 클수록 감소한다. 이 단순 조건에서는 **가능한 서로 다른 기록을 먼저 쓰는 것이 낫다.** 설명을 만들기 위해 noise-vs-record 사이에 항상 interior optimum이 있다고 하면 틀린다.

내부 최적점이 생기려면 실제로 서로 다른 기록의 encoding/load 비용이 크거나, covariance가 다른 구조이거나, 목적/제약이 달라야 한다. 현재 cached-latent LoRA에서는 추가 VAE/렌더 비용이 이미 사라져 CARV의 큰 upstream amortization 이익을 그대로 기대할 수 없다.

따라서 초기 후보의 핵심은 복잡한 3축 최적화가 아니다. 이미 강한 coverage-first/고정 multiplicity 위에서도 clipping 이후 오차가 user별로 충분히 달라 계산 낭비가 남는지를 묻는다.

## 6. 개선이 예상되는 조건과 비용 경계

기하 근사가 유효한 user별 a_u=tr(JΣJᵀ)를 알고 있다고 가정하면, 동일 inner evaluation 비용에서 총 근사 오차 Σ a_u/R_u를 줄이는 oracle allocation은 R_u∝√a_u이다. 이는 알려진 수학이며 우리 기여가 아니다. 실제 controller는 a_u를 모르는 상태에서 pilot 비용까지 지불한다.

유리할 수 있는 조건:

- 강한 고정 multiplicity에서도 최종 clipped contribution의 MC 오차가 실제 optimizer 오차의 중요한 부분이다.
- user별 raw variance와 postclip 방향 오차의 순서가 자주 다르다.
- 일부 user는 큰 radial noise 때문에 raw variance가 크지만 추가 평균이 거의 불필요하고, 다른 user는 방향/경계 오차 때문에 반복이 유효하다.
- 이 이질성이 pilot 오차와 추가 gradient 2개의 비용을 상쇄할 만큼 크다.
- 배치 packing/IO가 가변 R의 nominal F/B 절감을 실제 시간 절감으로 연결한다.

불리하거나 차이가 사라지는 조건:

- 대부분 clipping inactive이면 J=I라 geometry controller는 일반 variance allocation과 같아진다.
- user 간 a_u가 비슷하면 고정 R가 pilot을 낭비하지 않아 더 좋을 수 있다.
- 최적 고정 R=1이고 MC error가 이미 중요하지 않으면 pilot 2개는 구조적으로 비싸다.
- Gaussian noise, optimizer dynamics, data diversity, clipping 자체의 deterministic bias가 주된 병목이면 MC 개선이 품질로 이어지지 않을 수 있다.
- 작은 pilot이 heavy tails/경계 위치를 잘못 판단하면 오히려 중요한 user에 부족한 R를 배정한다.
- AdamW의 moment/preconditioner 때문에 gradient-space 오차 개선이 parameter-update 오차 개선과 다를 수 있다.

같은 q,T,C,σ,parameterization이면 기본 ε는 동일하다. 이것은 실제 privacy leakage가 같다는 주장도 아니며, 동일 품질을 보장하는 것도 아니다. 실제 walltime를 줄이려고 T/q까지 바꾸면 회계·최적화 비교를 다시 해야 한다.

단계당 예상 trainable-network gradient 비용은 sampled user M에 대해

    B ≈ M [2 + E(R_u)]     (pilot+final, 각 draw 1 backward),
    F ≈ 같은 수의 gradient-bearing forward + backend recomputation.

고정 R0 기준은 MR0이다. Pilot 비용을 빼고 R만 세면 안 된다. 일반 비용 모델은 c_pilot·2+c_draw·E(R_u)+packing/IO이다. 이 수식으로 실제 초 단위를 주장할 수는 없다. 이번 작업에서 새 runtime을 측정하지 않았다.

## 7. 가장 강한 근접 비교군과 무엇을 분리하는지

| 비교군 | 맞춰야 할 조건 | 판별할 질문 |
|---|---|---|
| ULS + 현재 최선의 DP-LoRA + public-selected fixed multiplicity R/G | 같은 base, trainable parameters, optimizer, objective, user ε/δ, 실제 총비용 | 약한 R=1 비교가 아니라 좋은 고정 계산량도 넘어서는가 |
| 같은 pilot/fresh-final 구조의 raw-variance allocation | 같은 pilot 수, 허용 R set, λ 선택 자료, cap, 실제 비용 | 새 clipping-aware 기준 자체가 필요한가 |
| 균등 record coverage-first + fixed/MC variance allocation | user objective 동일, record sampling의 차이를 명시 | record 다양성 효과를 새 controller 효과로 착각했는가 |
| 고정 timestep distribution / importance-corrected schedule / stratified schedule | objective-preserving 방식과 weighted-objective 방식을 분리 | 기존 noise/time variance reduction을 단순 재포장한 것인가 |
| tight-accounted ELS | 같은 user ε/δ와 총계산비용; 원형 sampling 차이 유지 | ULS 안의 개선을 전체 보호 설계의 최고 성능으로 확대할 근거가 있는가 |
| 공개 pretraining/PEFT 및 적용 가능한 public-trajectory 방법 | 별도 backbone/접근권 비용을 명시 | 공용 자산의 이익을 새 방법 이익으로 계산하지 않는가 |

모든 줄을 먼저 대규모 실험에서 실패시켜야 후보를 제안할 수 있다는 뜻이 아니다. 작은 원리 검증에는 첫 세 줄이 직접적이다. 최종 주장 범위가 넓어질 때 baseline을 넓힌다. 같은 회계 비교와 같은 walltime 비교를 혼동하지 않는다.

## 8. 논리를 판별하는 최소 검증: 새 대학습부터 시작하지 않는다

### A. 기존 저장 자료로 먼저 할 수 있는 것

기존 protection_observation_20260915의 local gradient vectors는 기하 연산·AdamW state restore와 controller 계산 비용의 CPU 검산에 쓸 수 있다. 그러나 user/image별 반복 noise gradient bank가 아니므로 within-image covariance, a_u, 추가 평균 이득을 추정하는 자료로 충분하지 않다. 저장된 CDI/DL의 반복 **scalar loss**도 parameter-gradient 방향 covariance의 대체물이 아니다.

기존 공개 calibration은 norm 위주이며 full vector 반복 bank가 아니다. 기존 public fixture의 원래 C=.284487에서는 모든 local gradient가 비활성화되어 A=B였다는 관측을 보존한다. post-hoc C=.01의 activation control을 이 후보의 정당한 운영 C로 채택하지 않는다.

공개 gradient가 주어진 상태라면 toy counterexample, 실제 clip Jacobian, pilot/final independence, 분모·norm·cost를 먼저 CPU로 검사할 수 있다. 이것이 아직 실제 diffusion 효능 검증은 아니다.

### B. 이후 필요한 가장 작은 새 증거의 명세

다음은 **제안만이며 이번에 실행하지 않았다.**

- 기존 public fixture의 8 user, source-defined objective와 공개적으로 정당화한 C 하나, 이미 저장된 비영 AdamW step1000 상태 하나만 사용한다.
- user당 joint record/timestep/noise의 사전 고정 16 gradient draws를 얻으면 총128 forward/backward가 최소 bank 규모다. 더 큰 학습/새 checkpoint는 필요 없다.
- 순서/seed/pilot 구분을 사전에 고정한다. 각 반복 비교에서 pilot과 final용 draw는 겹치지 않게 한다. 비교 규칙 선택용 자료와 보고용 자료도 나눈다.
- 이 작은 bank의 전체 평균도 참 μ가 아니다. 독립 reference 일부를 두거나 split-half stability를 함께 보고하고, 이를 유한 bank 원리 진단이라고 명시한다.
- fixed R, raw-variance allocation, clipping-aware allocation을 **pilot을 포함한 동일 사용 gradient 수**에서 비교한다. 128개는 benchmark bank를 만드는 비용이며 실제 proposed step 비용과 구분한다.
- 고정 C에서 raw variance 순위와 clipped-error 순위가 다른지, proxy가 추가 반복의 실제 유한-bank 감소를 예측하는지, pilot 비용 후에도 이득이 남는지를 본다.
- 저장된 AdamW moment에 같은 가우시안 난수를 사용하여 한 step의 parameter-update 차이도 CPU로 계산할 수 있다. 품질을 말하려면 별도의 held-out loss/model query가 필요하며, 이를 gradient 오차만으로 대체하지 않는다.
- 처음 자료에 clipping이 거의 없거나 proxy와 실제 이득이 연결되지 않으면 “더 작은 C를 찾아 활성화”하는 대신 현재 후보의 적용 근거가 약하다고 기록한다. 이를 전체 DP 연구 불가능성으로 해석하지 않는다.

128회로 paper efficacy를 판정할 수 있다는 뜻이 아니다. **현재 후보의 핵심 연결, 즉 기존 분산 지표의 계산 오배분과 새 기준의 비용 후 개선이 실제 gradient에서도 나타나는지**를 검토하는 제한된 단계다. 긍정적이면 quality–cost 학습 검증을 설계하고, 불확실하면 어떤 연결이 불확실한지에 맞춰 판단한다.

## 9. 탈락시키는 넓은 후보와 남기는 좁은 주장

탈락: “환자마다 여러 영상·noise·timestep을 평균하고 자동으로 G/R를 고르면 같은 DP에서 효율적이다.”

이 설명은 ULS의 G 배분, DPDM/DP-LoRA의 noise multiplicity, CARV의 hierarchical MC/cost accounting, 기존 timestep 재배분을 합친 수준이고, 동일비용에서는 distinct records부터 쓰는 단순 대안도 놓친다.

남기는 좁은 설계 주장:

> 동일한 user sampling·Gaussian 회계를 사용하는 clipped user-gradient 학습에서, 최종 clipping이 제거할 variance까지 똑같이 줄이려는 계산 배분은 불필요한 MC 비용을 만들 수 있다. user 내부 독립 pilot과 fresh final estimator로 bounded contribution 이후에 남는 오차를 겨냥하면, 이 이질성이 실제 존재하는 조건에서 좋은 고정 multiplicity보다 나은 품질–비용 절충을 얻을 가능성이 있다.

기존 구성 요소만으로 무조건 기각할 이유도 없고, 이 문장만으로 novelty를 인정할 이유도 없다. 방법의 작은 새 연결은 명확하다. 남은 핵심은 **실제 clipping geometry의 이질성 → pilot의 유효한 판별 → 추가 비용 후 개선 → AdamW와 학습 품질로의 전이**이며, 아직 어느 연결도 이번 작업에서 모델 실험으로 입증하지 않았다.

## 10. 이번 작업의 완료 범위

- 완료: 주요 근접 원문의 지정 절 직접 대조, 2025–2026 인접 후보 탐색, 하나의 구체 계산 변경, privacy/selection-bias/cost 조건, 정확한 설명용 반례, 강한 baseline과 최소 증거 명세.
- 미완료: 모든 clipping-bias/adaptive-MC 문헌의 exhaustive novelty 확인, private runtime transcript의 별도 DP 설계, 실제 controller 적합성·효율·품질 검증.
- GPU/학습/새 환자 평가: 0.
- 공용 상태·기존 동결 결과·실험 코드 수정: 0.

