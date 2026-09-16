# ULS/ELS의 clipping 순서와 E/U 관측: 원문·실제 trainer 대조

2026-09-15. 현재 **2번: 보호 방식 × 기존 공격 × E/U를 연결할 원리 후보를 좁히는 작업**이다. B04/B05/B06 원문과 현재 졸논 trainer를 직접 읽고, 작은 벡터의 CPU 산술만 수행했다. 학습·GPU·공격 실행·동결 파일 변경은 없다. 이 노트는 `PAPER_LEVEL_REDESIGN_20260915.md`의 설명용 a/b를 실증된 분해로 취급하지 않는다.

핵심 판단: **두 방법 모두 평균을 낸다. 차이를 만들 수 있는 것은 clipping 계수가 이미지마다 달라지면서 gradient 방향과 결합하는 효과, 샘플링·환자별 가중치, 그리고 실제 업데이트에 들어가는 noise다.** 이 중 clipping 순서의 비가환성은 정확히 보일 수 있다. 그러나 그것이 현재 AdamW 의료 생성모델에서 환자 공통 성분을 더 보존하거나 E/U 공격 순위를 반전시킨다는 사실은 아직 확인되지 않았다.

## 1. 직접 읽은 원문의 정확한 경계

페이지는 로컬 PDF의 1-based 페이지다. PDF에서 추출된 `texts/B04.json`, `B05.json`, `B06.json`의 해당 페이지 원문을 읽었다.

| 자료 | 직접 확인한 식·절 | 이번 판단에 주는 근거 |
|---|---|---|
| [B04: Fine-Tuning Large Language Models with User-Level Differential Privacy](../pdfs/B04.pdf) | pp.3–4, Algorithms 1–2, Theorems 1–2, Eqs.(2)–(3) | ELS는 record Poisson sampling과 per-example clipping, ULS는 user Poisson sampling과 user-mean clipping이다. 환자 보장 회계는 각각 Binom(G_ELS,p), Bern(q) mixture를 사용한다. |
| B04 | p.5, Eqs.(4)–(7), Conjecture 1 | 사용자 평균 gradient의 norm bound가 줄어들 수 있고, 같은 gradient-compute에서 noise가 어떻게 달라지는지 이미 분석했다. 이 절은 C를 전역 gradient bound로 둬 **clipping이 inactive인 단순화**다. Conjecture 1은 정리가 아니다. |
| B04 | pp.6,8–9 | 사용자 내 다양성·compute·ε에 따른 utility 차이, records/user와 users/batch 선택을 다룬다. E/U membership 전이나 clipping이 공통 성분을 항상 보존한다는 정리는 없다. p.4 footnote 2는 이론의 Poisson과 일부 실험의 shuffle 차이도 경고한다. |
| [B05: Mind the Privacy Unit!](../pdfs/B05.pdf) | p.3 Figure 2, Algorithm 1 lines 9–15, Algorithm 2 lines 5–12; p.4 | group/privacy 방식의 record clipping과 user-mean clipping을 명시한다. userwise DP는 회계상 초기 record cap이 필수가 아니며, 내부 records 수가 사용자마다 달라도 된다. |
| B05 | p.4 footnotes 3–4; p.6 | 초기 naive group bound를 tighter accounting으로 바꾸었다고 명시한다. 사용자 내 표본 수·선택 전략의 효과도 이미 다룬다. |
| [B06: Tight Group-Level DP Guarantees for DP-SGD with Sampling via Mixture of Gaussians Mechanisms](../pdfs/B06.pdf) | p.4 Theorem 3.1; p.6 Theorem 3.3, §3.3 | Poisson의 group sensitivity는 Binom(k,q), fixed-size sampling의 결과는 **2×Hypergeom**이다. tightness 예는 민감 record의 gradient가 정렬된 worst case이며 모든 iterate를 보는 설정이다. 실제 환자의 E/U leakage를 동일시하는 결과가 아니다. |

같은 user 보장과 compute에서 B04 Eq.(7)은

`Var(noise_ELS) ≤ Var(noise_ULS) iff L_ELS σ_ELS ≤ G_ULS L_ULS σ_ULS`

를 준다. 여기서 B=G_ULS M이고 noise 표준편차는 ELS의 `σ_ELS L_ELS/B`, ULS의 `σ_ULS L_ULS/M`이다. 원문도 sampling 차이 때문에 이것만으로 utility를 완전히 예측하지 못한다고 적는다. **norm 감소·diversity·평균화·같은 ε 비교 자체는 선행이 이미 가진 내용**이다.

## 2. 실제 졸논 trainer와의 대응

아래 경로는 thesis root 기준이다.

- `code_working/dp_training/k5_full_runner.py:407–424`: M1은 image IDs의 독립 Poisson sampling, 한 privacy unit은 이미지 한 장이다.
- 같은 파일 `:428–473`: M2는 기대 환자 batch 4, 환자 Poisson sampling 후 환자별 fresh uniform without-replacement 최대 4장을 택한다. 매 이미지의 timestep/noise draw가 따로 있다.
- 같은 파일 `:672–678`, `:722–727`: patient unit 경계를 확인하고, per-image MSE를 먼저 평균한 loss의 gradient를 한 번 계산한다. 개별 gradient를 clip한 뒤 평균하는 코드가 아니다.
- `code_working/dp_training/mechanism.py:161–175`: 한 전체 LoRA gradient 벡터의 L2 norm으로 clip factor를 계산한다. `:179–188`은 mean-before-clip 의미를 명시하지만, 실제 caller는 loss 평균의 backward로 이를 구현한다.
- 같은 파일 `:191–203`, `:220–235`: 독립 Bernoulli sampling, 단위별 clipping 후 합산, empty batch에도 Gaussian을 넣고 **고정 expected batch**로 나눈다. realized batch로 나누지 않는다.
- `code_working/dp_training/k5_full_runner.py:608–615`, `:804–816`: 그 결과는 **AdamW에 전달할 gradient**다. clipping된 벡터가 곧 최종 parameter displacement는 아니다.
- `code_working/dp_protocol/build_xray_dp_attack_protocol.py:648–665`: M1의 분모는 8 images, M2의 분모는 4 patients이고, 별도 C_image/C_patient를 쓴다.
- `code_working/dp_protocol/calibrate_xray_public_clip_norms.py:465`, `:655–656`: C는 public-development gradient norm의 empirical p80(method='higher')이다. B04 Eq.(4)의 모든 θ에 대한 전역 Lipschitz bound가 아니다. private training 내 clipping이 inactive라고 가정할 근거도 아니다.

**회계 비교의 별도 주의:** `build_xray_dp_attack_protocol.py:397–403`, `:459–477`의 동결 M1-G8은 `ε_image=8/K`, 일반 group δ 기하합으로 환자 목표에 맞춘 arm이다. B04/B06의 mechanism-specific tight mixture accounting을 구현한 arm과 같지 않다. 졸논의 기존 동결 목적은 보존하되, 이 arm만으로 CVPR에서 “최강 ELS와 ULS의 차이”를 주장하면 회계의 보수성 차이가 섞인다. 사후 회계 개선은 기존 가중치를 바꾸지 않으며, 더 작은 noise로 새 학습하는 선택과 구별해야 한다.

## 3. 같은 환자·같은 이미지 집합에서 무엇이 달라지는가

고정 θ와 같은 k개 gradient `g_j`에 대해 `μ=(1/k)Σ_j g_j`, `clip_C(g)=a_C(g)g`, `a_C(g)=min(1,C/||g||)`로 둔다. sampling/outer weight/noise를 잠시 제거하면

`v_A=(1/k)Σ_j clip_CE(g_j)`, `v_B=clip_CU(μ)`.

이 두 식은 **clipping 순서만** 비교한다. 아래 A를 원형 record-Poisson ELS와 동일하다고 부르면 안 된다.

| 조건 | 정확한 결론 |
|---|---|
| 모든 image clipping과 patient clipping이 inactive | 같은 집합에서는 `v_A=v_B=μ`. 평균 방식만으로 차이가 생기지 않는다. |
| 모든 이미지의 clip factor가 같은 α | `v_A=αμ`, `v_B=βμ`. 환자별로 같은 방향이거나 둘 다 0이다. 같은 norm은 충분조건이고, norm이 달라도 모두 inactive면 α=1이다. |
| image clipping inactive, patient clipping만 active | 방향은 같고 크기만 달라진다. |
| 모든 g_j가 같은 ray에 있음 | norm이 달라도 방향은 같다. 반대 방향을 포함한 단순 공선은 같은 ray 조건과 다르다. |
| clip factor가 다르고 방향도 다름 | 방향 회전이 **가능**하다. 반드시 회전하는 것은 아니며 대칭 상쇄 예외가 있다. |

환자마다 A/B의 배율이 다르면 환자별로 공선이더라도 전체 환자 합은 공선일 필요가 없다. 따라서 per-patient 경계를 전체 minibatch·전체 학습 경로로 확대하지 않는다.

`g_j=s+r_j`, `s=μ`, `Σ_j r_j=0`로 **유한 집합의 항등식**을 쓰면, `a_j=a_CE(g_j)`, `ā=mean(a_j)`에 대해

`v_A = ā s + (1/k)Σ_j (a_j−ā)r_j`, `v_B=βs`.

두 번째 항은 clip factor와 residual 방향의 결합이다. 이것을 환자 특이 정보라고 미리 부를 수 없다. 여기의 s는 선택한 train gradient 평균일 뿐, unseen 이미지에도 동일한 생물학적/정체성 성분이라는 사실은 별도의 가정이다. 환자 공통 성분과 사진별 성분을 실제로 식별한 분해도 아니다.

동일한 k장의 평균에 대해 residual variance가 줄어드는 효과는 A/B 모두에 존재한다. 독립 residual이면 평균 분산이 Σ/k가 되지만, 상관이 있으면 `Var(mean r)=k^-2 Σ_ij Cov(r_i,r_j)`다. “여러 장이므로 사진별 성분은 사라진다”도 자동 결론이 아니다.

## 4. 같은 환자 sampling의 bridge: 가능한 것의 정확한 예

같은 patient Poisson q, 같은 고정 분모, 같은 C와 Gaussian `σC`를 두고 A/B를 비교할 수 있다. 두 환자 기여 모두 norm≤C이므로 add/remove user의 같은 sampled-Gaussian **상계**를 적용할 수 있다. replace-user adjacency나 데이터에 따라 달라지는 분모를 몰래 쓰지 않는다. 이는 새 DP 방법 제안이 아니라 clipping 순서만 분리하는 표준적 기제 대조다. A의 per-image gradient 처리는 B의 mean-loss backward보다 실제 메모리/호출 overhead가 클 수 있어, 동일 image-gradient 개수와 동일 wall-clock은 구별한다.

`g1=(4,0), g2=(0,1), C=1`이면

`v_A=(.5,.5)`, `v_B=(4,1)/sqrt(17)=(.9701425001,.2425356250)`.

같은 두 E gradient의 평균 query 방향 `h_E=(2,.5)`와, 하나의 U query 방향 `h_U=(0,1)`를 택하면 다음은 정확한 내적이다.

| 고정 관측 | A | B |
|---|---:|---:|
| `h_E·v` | 1.25 | 2.0615528128 |
| `h_U·v` | .5 | .2425356250 |
| `h_E·v/||h_E||` | .6063390626 | 1 |
| `h_U·v/||h_U||` | .5 | .2425356250 |

**정확한 FPR 비교가 가능한 제한된 toy:** 고정 배경 기여 0, 초기 θ 고정, one-step SGD/gradient release, member 환자 선택 `I~Bern(q)`, `Y=Iv+Z`, `Z~N(0,σ²C²I)`를 둔다. 선형 score `S_h=hᵀY`에서 nonmember FPR α로 threshold를 잡으면

`TPR_α=(1−q)α+q Φ(hᵀv/(σC||h||)−Φ^-1(1−α))`.

따라서 이 toy에서는 같은 α에서도 E는 B>A, U는 A>B다. CPU `math`/`statistics.NormalDist`로 확인한 예시 q=.1,σ=C=1,α=.01의 E TPR은 A=.013271542/B=.018236225, U는 A=.012389894/B=.010858862다. 수치는 임의로 고정한 toy의 계산값이며 학습·의료 성능 결과가 아니다.

여기서 정확한 것은 **고정 선형 관측의 순서가 다를 수 있다는 가능성**이다. 기존 공격 전부, 최적 membership test, 여러 step의 마지막 모델, nonlinear diffusion loss의 TPR 순서가 아니다. 전체 Y를 보는 최적 공격은 이 단일 이동 toy에서는 ||v||에 의해 달라지므로 norm 1인 B가 norm sqrt(.5)인 A보다 잘 구분될 수 있다. E/U 관측 방향 제한을 지우면 위 반전이 말하던 질문 자체가 달라진다.

추가 경계: `g1=(2,0),g2=(-1,0),g3=(-1,0),C=1`이면 μ=0이지만 v_A=(-1/3,0), v_B=0이다. clipping이 특정 평균 성분을 새로 만들 수도 있음을 보이며, 어느 쪽이 항상 “공통 성분 보존”인지 정하는 예가 아니다.

## 5. 현재 AdamW와 기존 공격으로 연결할 때의 제한

실제 trainer의 AdamW 때문에 위 Gaussian projection 식을 그대로 parameter update에 적용하면 틀린다. 같은 0 moment와 noise 없는 첫 step에서 bias-corrected Adam gradient 항은 `v/(|v|+eps)`다. 위 A/B는 양 좌표 sign이 같으므로 eps=1e-8이면 각각

`A: (.9999999800,.9999999800)`, `B: (.9999999897,.9999999588)`

로 거의 같아진다. 공통 초기 θ에서 weight decay 항은 같지만, DP noise 후의 비선형 변환과 이후 moment·θ 변화는 이 단순 식을 더 바꾼다. 따라서 gradient-release/SGD toy는 실제 의료 optimizer의 예측을 대신하지 못한다.

고정 query loss의 한 step 변화는 실제 displacement에 대해 `L_x(θ+Δθ)−L_x(θ)=∇L_x(θ)ᵀΔθ+O(||Δθ||²)`로 접근할 수 있다. 이때 Δθ는 **AdamW 전체 결과**여야 한다. 작은 명목 learning rate만으로 잔여항이 작다고 증명한 것도 아니다.

기존 diffusion loss 기반 점수에는 이 연결을 검토할 여지가 있지만, training timestep/noise/prompt의 gradient와 공격의 고정 timestep/noise/prompt gradient는 같다고 보장되지 않는다. CDI의 여러 특징이나 MoFit 최적화 점수는 단일 loss와 같지 않다. 더욱이 query의 절대 loss 감소와 member/nonmember 점수 차이, FPR에서의 판별력은 서로 다른 양이다.

## 6. equal patient ε / noise / compute로 말할 수 있는 것

1. 같은 `(ε,δ)`·같은 환자 adjacency·유효 accountant는 공통의 worst-case 상한 조건이다. 실제 누출, 최적 공격, E/U 공격 성능이 같다는 뜻이 아니다.
2. 같은 noise multiplier σ도 coordinate noise가 같다는 뜻이 아니다. 실제 gradient noise std는 `σC/B_expected`이며 M1/M2는 C와 분모가 다르다. ε·σ·compute를 임의로 독립 고정할 수도 없다. q,T,그룹 크기와 회계 관계를 함께 정해야 한다.
3. image sampling은 clipping이 없을 때 전체 record 평균을 향하고, uniform patient sampling+patient mean은 전체 patient 평균을 향한다. 환자별 record 수 k_u가 다르면 전자는 k_u 가중치, 후자는 동일 환자 가중치다. 같은 수를 뽑아 평균해도 원형 ELS/ULS의 표본 상관구조는 다르다.
4. 같은 q·같은 환자/내부 이미지 집합을 쓰는 bridge는 이 중 sampling·outer weighting을 고정할 수 있다. 그러나 **실제 ELS/ULS 전체 비교**는 아니며, bridge의 Gaussian release 결과와 현재 AdamW의 결과도 나눠야 한다.

## 7. 현재 남길 수 있는 원리 후보

검토할 만한 구체적 연결은 다음 조건부 문장이다.

> 환자 내 active clipping 계수의 차이가 서로 다른 gradient 방향과 결합하고, 그 차이가 실제 optimizer 이후에도 E 및 U query 방향에 다르게 투영된다면, E 관측에서 얻은 특정 기존 공격의 보호방식 비교가 U 관측으로 그대로 이전되지 않을 수 있다.

앞 절은 가능한 벡터 관계와 Gaussian 관측 반례를 보였다. 현재 의료 gradient의 계수/방향 결합, AdamW 이후의 차이, 실제 기존 공격의 E/U 비교는 미확인이다. 이것은 새로운 방법·전체 선행의 미해결·CVPR 성과를 이미 확보했다는 주장이 아니다. 관련 선행이 이미 user inference와 DP/clipping의 관계를 다루므로, 신규성은 구체적인 조건과 평가 판단의 적용 범위에서 검토해야 한다.

이 원리 후보는 설계 논의를 시작할 수 있는 근거다. 모든 baseline의 실패나 완전한 인과 식별을 먼저 요구하지 않는다. 동시에 작은 toy 하나를 실제 trainer의 예측으로 승격하지 않는다. 다음 논문 수준 작업은 **실제 optimizer까지 포함했을 때 어떤 조건의 관측 순서가 유지/변경될지와 기존 공격에서 이를 어떻게 읽는지**를 명시하는 것이다. 이 노트는 새 GPU·학습을 예약하거나 동결 K5/K10 계약을 변경하지 않는다.

## 8. 원자료 식별과 확인 범위

- B04 PDF SHA256: `e2d2a359616e891f9a21eec81bbd358fdc5044e6c1dbb276d9686128718b5ba4`
- B05 PDF SHA256: `8dc5bc1298740ba983fb14e3fa05d2f2eece3e3efe779bcd5503b982254effd8`
- B06 PDF SHA256: `7f5e71eaceaace5e20e478135850033d71b15c6dbb7fdc9789d79d58b85f78bb`
- `mechanism.py` SHA256: `b4b1e037a3704238b06b442970eab082433835201b750546c8281bbffe2b969e`
- `k5_full_runner.py` SHA256: `d152c59aba4087809a4ac2251e183306c7541eddac4fd0d64a7d46554de66505`

확인 범위는 위 원문 페이지·구현 위치·CPU 벡터 계산이다. 모든 관련 논문을 다시 읽었거나, DP 본학습을 수행했거나, AdamW에서 의료 E/U 순위를 확인했다는 뜻이 아니다.
