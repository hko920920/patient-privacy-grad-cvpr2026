# 공개 모델 전달·라벨 최적화: 다음 실험 계획

작성일: 2026-09-28 (KST)  
상태: **설계 완료 / 실행 전**. 큰 단계 2 유지. 이 문서 작성 중 학습·모델 forward·V 재평가·Q 접근·새 DP release·Expert/Reserved 접근은 모두 0이다.

## 1. 이번에 결정한 것

현재 후보를 유지하되, 다음 한 묶음은 **기존 도구에 같은 기회를 줬을 때 남는 가치**를 확인한다.

1. 일반적인 DINO teacher 예측 전달에도 같은 공개 ResNet18과 라벨 최적화를 허용한다.
2. 기존 DP 합성 PNG와 공개 실영상 128장에 같은 목표·라벨 solver를 적용한다.
3. DINO만 맞추는 대조도 같은 solver로 비교한다.
4. 이 비교에서 추가 가치가 나타난 경우에만 새 초기화 반복과 독립 평가로 진행한다.

새 손실·encoder·DP 잡음·pixel 학습을 추가하는 계획이 아니다. 기존 두 feature-DP release를 모두 사용한다. 한 release에서 좋은 구성을 골라 다른 release에만 적용하지 않는다.

**현재 가장 설득력 있는 가설:** 한 source에서 보호한 요약으로 만든 학습 기능을 고정된 이미지/라벨에 옮길 때, source 하나의 기능만 맞추는 것으로는 다른 학습자의 기능을 정하지 못할 수 있다. 공개자료로 얻은 다른 모델의 목표를 함께 맞추는 것이 이 자유도를 유용하게 제한할 가능성이 있다. 이는 검증할 가설이며 새 원리로 확정하지 않았다.

## 2. 출발점과 이미 확보한 근거

원 실행: [CONTRIBUTION_SEARCH_RESULTS_20260924.md](CONTRIBUTION_SEARCH_RESULTS_20260924.md).  
현재 전략: [PUBLIC_TRANSFER_LABEL_STRATEGY_20260928.md](PUBLIC_TRANSFER_LABEL_STRATEGY_20260928.md).

| 기존 라벨 방식 | DenseNet DP1 AUROC/AP | DenseNet DP2 AUROC/AP |
|---|---:|---:|
| 원래 hard label | 0.650018 / 0.080832 | 0.596743 / 0.064188 |
| DINO 직접 pseudo-label | 0.653683 / 0.074752 | 0.614055 / 0.062433 |
| DINO-only label solve | 0.671617 / 0.083324 | 0.613363 / 0.060382 |
| DINO+공개 전달 ResNet18 joint | 0.686370 / 0.081264 | 0.663404 / 0.074463 |

Joint−DINO-only AUROC는 DP1 +0.014753, 조건부 95% CI [-0.002636,+0.032335]; DP2 +0.050041, [+0.024707,+0.076099]. AP는 DP1 우위 미확정, DP2 양의 구간이다.

이를 동기로 삼되 독립 재현으로 세지 않는다. DenseNet은 재사용한 개발 receiver이고 ResNet18은 목표를 구성하는 모델이다. 지표는 **영상별 AUROC/AP**, 불확실성은 **환자 cluster paired bootstrap**이다.

## 3. 코드·수식 대조에서 지금 확정한 부분

### 3.1 현재 평균 전달은 공개 가중 라벨 학습으로도 정확히 표현된다

코드 근거: public_transport_probe.py의 fit/run, validate_public_transport.py의 signed-weight 검산, joint_compiler.py의 목표 계산.

공개 환자/class 행 j의 source 특징을 x_j, receiver 특징을 y_j라 한다. 환자에게 총 1/672의 무게를 주며 두 class가 있는 환자는 행 사이에 나눈다. 이 행 무게를 a_j라 한다. 각 class c의 공개 clipping C_c를 **모든 공개 행**에 적용한 행렬을 X_c라 한다. 양성 6명만으로 양성용 회귀를 fit하는 코드가 아니다.

열벡터 표기에서:

\[
\bar x_c=X_c^\top a,\quad
\bar y=Y^\top a,\quad
X_{c,0}=X_c-\mathbf1\bar x_c^\top,
\]
\[
G_c=X_{c,0}^\top\operatorname{diag}(a)X_{c,0},\quad
\lambda_c=10^{-3}\operatorname{tr}(G_c)/64,
\]
\[
T_c=(G_c+\lambda_c I)^{-1}X_{c,0}^\top\operatorname{diag}(a)(Y-\mathbf1\bar y^\top).
\]

보호 source 평균 \(\tilde\mu_c\)의 전달값은:

\[
\hat\mu_{R,c}
=\bar y+T_c^\top(\tilde\mu_c-\bar x_c)
=Y^\top \alpha_c,
\]
\[
\alpha_c=a+\operatorname{diag}(a)X_{c,0}
(G_c+\lambda_c I)^{-1}(\tilde\mu_c-\bar x_c).
\]

이 signed public-weight 동치는 9월 24일 이미 최대 오차 3.22e−15로 검산됐다. 이번에는 수식과 기존 코드를 대조했으며 검산을 다시 실행하지 않았다. \(\alpha_c\)는 확률이 아니며 음수가 가능하다.

공개 영상 i가 행 j에 속하고 그 행의 방문 수가 n_j이면,
\[
v_i=(\alpha_{1,j}-\alpha_{0,j})/(2n_j)
\]
로 풀어 쓸 수 있다. 공개 영상의 class-balanced patient-equal 무게를 \(\omega_i>0\), receiver 영상 특징행렬을 Z_R라 하면 q_i=v_i/\omega_i에 대해:

\[
w_R^{MT}
=(Z_R^\top\Omega Z_R+0.1I)^{-1}Z_R^\top v
=(Z_R^\top\Omega Z_R+0.1I)^{-1}Z_R^\top\Omega q.
\]

**따라서 현재 receiver 목표는 공개영상에 특수한 signed 회귀 라벨 q를 붙인 ridge 학습과 정확히 같은 계산이다.** 이를 별도 신규 방법 두 개처럼 비교하지 않는다. q는 source 공개 특징과 보호 평균으로 정해지며, 이 구성에서는 receiver별로 새 private 정보가 들어가지 않는다.

단, 이는 통상적인 teacher 점수 \(Z_Dw_D\)와 항상 같다는 뜻은 아니다. 현재 코드의 class별 clipping, affine centering, 공개 회귀 regularization 및 평균 복원이 차이를 만들 수 있다. 이 차이가 실제로 유용한지가 다음 비교다. 동치식은 새 정리로 주장하지 않는다.

### 3.2 일반 teacher-score KD 대조의 정확한 정의

같은 공개 영상 무게 \(\Omega\), 동일한 보호 평균으로:

\[
H_D=Z_D^\top\Omega Z_D,\quad
w_D=(H_D+0.1I)^{-1}\tfrac12(\tilde\mu_1-\tilde\mu_0),
\]
\[
q^{KD}=Z_Dw_D,\quad
w_R^{KD}=(Z_R^\top\Omega Z_R+0.1I)^{-1}Z_R^\top\Omega q^{KD}.
\]

DINO 목표는 모든 방식에서 동일하다. KD와 현재 평균 전달(MT)은 **ResNet18 목표를 만드는 규칙만** 다르며, 이후 solver·이미지·학습규칙은 같다.

- KD에도 P의 labels를 class/patient weighting에 사용할 권한을 준다.
- teacher 점수를 sigmoid나 [-1,1]로 미리 자르지 않는다. 기존 목표도 회귀 점수다. 최종 배포 라벨에만 공통 [-1,1] 제한을 적용한다.
- 새 temperature·regularization 탐색은 하지 않는다. 지금 비교는 위에 정의한 두 전체 목표 생성 구성의 비교다.
- 이것을 POST 또는 KIP 전체 재현이라고 부르지 않는다.
- MT의 signed q를 허용한 공개 가중 라벨 baseline은 MT와 같은 연산이므로, 그것을 “이겼다”는 주장은 불가능하다.

### 3.3 source-only의 불충분성을 측정 가능한 가설로 바꾼다

carrier C의 특징 Z_{C,m}에서 라벨을 ridge 계수로 바꾸는 연산:
\[
A_m=(Z_{C,m}^\top Z_{C,m}/128+0.1I)^{-1}Z_{C,m}^\top/128.
\]

\(A_Dd=0\)이면 DINO의 **전체 선형 함수**가 유지된다. 동시에 \(Z_{P,R}A_Rd\ne0\)이면 ResNet18의 공개 예측은 변한다. \(Z_{P,D}A_Dd=0\)만으로는 P 밖 DINO 예측 보존을 보장할 수 없으므로 진단에는 A_D의 nullspace를 사용한다.

128개 라벨/64차원 DINO에서 nullspace의 차원은 적어도 64이나, 실제 유용성은 별도다. 라벨 box [-1,1], 목표 residual과 방향의 정렬, 공개·평가 분포 차이가 제약한다. 기존 라벨 85/87개가 경계에 있었다는 사실도 고려한다. 라벨 regularizer까지 포함한 전체 objective가 이 방향에서 일정하다는 주장은 하지 않는다.

다음 실행에서 P/고정 PNG 특징만으로:
- A_D의 rank와 nullspace, ResNet18에 보이는 nullspace 방향을 계산한다.
- source-only 해에서 DINO 계수를 유지하는 box-constrained 라벨 변경으로 RN 목표 residual이 얼마나 감소 가능한지 계산한다.
- 이는 구성 가능성 진단이다. DenseNet 효용이나 신규성의 증명으로 쓰지 않는다. V를 보고 방향·모델·허용오차를 고르지 않는다.

## 4. 다음 한 묶음의 실행표

### 4.1 공통 비교: 3개 목표 × 2개 이미지 carrier × 기존 release 2개

| 목표 생성 | 기존 feature-DP PNG 128장 | 공개 실영상 128장 | 질문 |
|---|---|---|---|
| S: DINO-only functional label solve | DP1/DP2 | DP1/DP2 | 추가 receiver 목표가 필요한가? |
| K: DINO + 일반 teacher-score KD의 RN 목표 | DP1/DP2 | DP1/DP2 | 같은 label solve를 가진 통상적 전달은 어디까지 되는가? |
| M: DINO + 현재 평균 전달의 RN 목표 | DP1/DP2 | DP1/DP2 | 현재 목표 생성에 추가 가치가 있는가? |

**최대 12개의 128변수 bounded 선형 최소제곱 계산이다. 새로운 pixel 학습 bank 12개가 아니다.** 각 release의 DP 이미지와 해당 release 요약을 묶으며 DP1 PNG/DP2 요약처럼 교차하지 않는다. 공개128은 모든 행/두 release에서 동일하다.

nullspace 진단은 source-only 해당 4조건에서만 소규모 계산한다. 추가 V 성능 후보를 만들거나 최고값을 고르지 않는다.

이 표를 한 번에 고정해 실행한다. 결과를 본 뒤 빈칸을 유리하게 추가하지 않는다. 대수적으로 같은 셀은 alias로 기록해 중복 계산/성공 횟수로 세지 않는다.

### 4.2 공정한 carrier 비교를 위한 라벨 regularizer

기존 joint 코드는 64개 -1/64개 +1인 원래 라벨에 가까운 해를 선호한다. 공개 P는 양성 환자가 6명이므로, 이 기준을 그대로 적용하면 공개 carrier를 임의로 재라벨링하거나 양성 이미지를 중복해야 한다.

이번 공통 비교는:
\[
\min_{\ell\in[-1,1]^{128}}\|\mathcal A\ell-t\|_2^2+\eta\|\ell\|_2^2,\qquad
\eta=10^{-3}\|\mathcal A\|_F^2/128.
\]

즉 **모두 같은 zero-centered regularizer**를 쓴다. 기존 hard-label anchor만 제거하고 regularization은 유지한다.

각 모델 block은 공개 \(\omega\)로 가중하고 해당 목표의 weighted RMS로 정규화한다. 두 모델이면 각각 1/2, 단일 모델이면 1의 무게다. 목표 RMS가 1e−12 이하이면 수치 실패로 보고하며 임의 재보정하지 않는다. scipy lsq_linear, tol=1e−12, max_iter=500, float64를 유지한다.

이는 공정 비교용 공통 변형이다. 과거의 0.686370/0.663404가 이 변형에서도 재현된다고 미리 쓰지 않는다. 기존 hard-anchor 코드·라벨·성능은 역사적 결과로 보존한다. 이번 변형이 낮아져도 anchor를 바꿔 재시도하지 않는다.

### 4.3 공개128의 선정: 약한 무작위 대조로 만들지 않는다

공개 P 안에서만 선정한다. Q·V·DenseNet 특징·DP 목표·실험 결과는 선정에 쓰지 않는다.

- 환자 128명에서 영상 1장씩, 중복 없이 선택한다.
- 양성 방문이 있는 공개 환자 6명은 각각 양성 영상 한 장을 포함한다.
- 각 환자의 후보 영상은 고정 salt “public-carrier-20260928-v1” + patient/image ID의 SHA256 정렬 첫 영상으로 정한다. 양성 환자는 양성 영상 안에서, 나머지는 자신의 영상 안에서 정한다.
- 나머지 122명은 고정된 DINO/RN 공개 특징 공간에서 greedy farthest-first로 선택한다. 두 feature block은 전체 후보의 P-only RMS norm으로 맞추고 동일 무게로 연결한다. 양성 6개에서 시작하고 이미 선택한 집합까지의 최소 거리가 가장 큰 후보를 하나씩 추가한다. 동점은 같은 SHA256 순서로 해결한다.
- 이 선택 규칙 하나만 쓴다. 무작위/coreset/다른 seed를 V에서 비교하지 않는다.
- 64:64 hard labels나 class quota를 downstream 학습에 강제하지 않는다. 최종 라벨은 공통 solver가 정한다.

선정 manifest와 순서를 목표 계산·V 평가 전에 고정한다. 실제 배포 이미지와 feature cache의 decode/resize/condition 경로가 같다는 것을 확인해야 한다. 같지 않으면 선정된 P128만 공통 경로로 다시 encode한다. 이미지 128장 복사 자체는 pixel synthesis가 아니다.

공개128+DP 라벨도 Q에 관한 DP 후처리 artifact다. **완전한 public-only 대조가 아니다.** 이 비교는 사적 정보의 유무가 아니라 이미지 제작의 필요성을 묻는다.

### 4.4 공통 입력·평가 계약

- 기존 feature-DP1/DP2, 각 ε8/δ1e−5의 보호 target만 재사용한다. 새 raw Q 또는 noise 없음.
- DINO checkpoint·P-only projection·K4 조건, RN/DenseNet checkpoint·전처리·P projection 유지.
- carrier 크기 128, 최종 frozen-feature ridge=0.1, 균등 1/128, intercept 없음.
- 라벨은 회귀 supervision이며 보정된 질병확률이 아니다.
- DINO와 ResNet18만 construction에 쓴다. DenseNet은 개발 receiver. RN 평가는 construction fidelity/참고 효용이다.
- V는 라벨/선정/목표가 모두 고정된 뒤 **전체 표를 한 번 평가**한다.
- 기존 evaluator의 영상 AUROC/AP와 같은 2,000회 paired patient-cluster resampling을 사용한다.
- Expert·Reserved·새 receiver·full-network/CE 학습은 이 묶음에서 사용하지 않는다.
- 기존 보호회계 (여섯 release 공동 공개 기본 상한48/6e−5)를 유지한다. 이번 추가 Q 회계 0이며 과거 개발 전체를 보호했다는 뜻이 아니다.

## 5. 사전에 정한 분석과 다음 투자 결정

한 release를 다른 방법의 별도 noise와 비교하지 않는다. 이번 방식들은 모두 **같은 DP1, 같은 DP2**를 공유하므로 release별 대응 비교가 가능하다.

세 핵심 대비:
1. 같은 carrier에서 M−K: 목표 생성의 추가 가치.
2. 같은 carrier에서 K−S 및 M−S: 추가 construction model의 가치.
3. 같은 목표 생성에서 synthetic−public128: 기존 합성영상 제작의 가치.

각 DP1/DP2 지표·차이·조건부 구간을 모두 보여준다. 2-release 평균은 **지표의 기술 평균**이며 예측 앙상블이 아니다. bootstrap draw마다 두 대응 차이의 평균을 구해 조건부 구간을 계산한다. 이는 release 분포 전체나 adaptive development의 불확실성을 포함하지 않는다.

**주 지표 DenseNet AUROC, AP는 필수 보조 지표.** RN 상승을 독립 transfer 성공으로 합산하지 않는다. 사전에 정한 이 세 질문 밖 pairwise p-value를 전부 나열해 좋은 비교를 고르지 않는다.

다음 단계에 비용을 더 쓰기 위한 기준은 아래처럼 고정한다. 이는 임상/통계적 자연법칙이 아니라 제한된 개발 투자 기준이다.
- 해당 대비의 두 release AUROC 점차이가 모두 >0.
- 두 release 평균 ΔAUROC ≥0.01.
- 평균 AP 점차이가 ≥0.
- 평균 AUROC의 조건부 95% CI가 0보다 높으면 “반복 투자 근거 있음”.
- 앞 세 조건은 맞지만 CI가 0을 포함하면 “유망하나 미확정”. 예산을 확대하거나 유의할 때까지 반복하지 않고 아래 원인별 다음 한 작업만 판단한다.
- AP에서 조건부 차이 구간이 뚜렷하게 음수이거나 두 release 방향이 엇갈리면 복합적인 절충으로 기록한다.

계층은 **입출력/수치 검증 → 실질적 효과 → 조건부 불확실성 → 다음 투자**다. CI가 0을 포함한다는 이유로 동등성/무효를 선언하지 않으며, 점차이0.01을 넘었다는 이유만으로 방법 우위를 선언하지 않는다.

| 관측 | 연구 결정 |
|---|---|
| M과 강한 공개 가중라벨 방식이 수식상 동일 | 동일 연산으로 기록. 이를 독립 기여/서로 다른 baseline으로 세지 않음 |
| M−K가 뚜렷하지 않음 | 특수 평균 전달의 우위를 주장하지 않음. 표준 KD가 설명하는 효용을 인정 |
| K−S와 M−S도 뚜렷하지 않음 | 현재 functional multi-model 제약의 필요성이 미확인. 자동으로 encoder/계수 탐색하지 않음 |
| 공개128이 비슷하거나 우세 | 현재 데이터에서 비싼 pixel 제작의 추가 가치가 미확인. 공개 재라벨링을 새 CVPR 기여라고 자동 전환하지 않음 |
| M−K와 M−S가 양수, 공개128에서도 유지 | 표준 teacher 전달과 다른 **목표 구성의 사용상 가치** 후보. pixel 기여와 분리 |
| synthetic carrier에 추가 이득 | 해당 목표 생성/라벨 설정에서 carrier 가치 후보. 일반적 영상 품질 우위로 확대하지 않음 |
| 원래 hard-anchor만 잘되고 공통 설정에서 사라짐 | 기존 성과가 anchor를 포함한 구성에 의존함. 원래 성과를 삭제하지 않고 깨끗한 carrier/목표 분리 근거는 부족하다고 기록 |

목표 residual 감소는 구현/표현 가능성 증거다. 그것만으로 V 효용이나 논문 기여를 판정하지 않는다. 이 표는 결과마다 새 제목을 만들어내는 자동 논문 전환표가 아니다.

## 6. 후속 실험: 필요한 주장에만 연결한다

다음 묶음 결과와 코드를 검토한 뒤 순서를 정하며 자동 실행하지 않는다.

**A. 공통 비교에서 추가 가치가 있으면 재현한다.**
- 주장과 가장 강한 대조 하나를 고정한다.
- 이미지 제작 가치가 남았으면 기존 한 DP 요약을 고정하고 synthesis init202의 bank **한 개**만 새로 만든다. 동일 bank에 후보/대조 라벨을 모두 적용한다. 새 DP 잡음은 없다. 참고 합성35분, 준비·평가 포함45~70분.
- 공개 carrier가 충분한 방향이면 같은 선정규칙의 hash salt만 202로 바꾼 공개128 한 집합에서 후보/대조를 같이 확인한다. 새 합성 없이 수분 계산. 이를 자동 신규성으로 해석하지 않는다.
- 불확실성이 환자 수에서 왔다면 이미지 seed를 더 뽑아 그 불확실성을 해결했다고 말하지 않는다.

**B. 그 다음, 실증적 차이와 선행 대비 차이를 따로 닫는다.**
- POST식 public transfer, KIP/LS, DP KME에서 알려진 연산과 같은 부분은 계속 명시한다.
- 현재 이득이 bounded labels, public target geometry, multiple learner objective 중 무엇을 요구하는지 하나의 가설로 좁힌다. 위 nullspace 진단이 유효하지 않으면 그 설명을 폐기한다.
- 전체 방법 비교와 요소 제거실험을 구분한다. 지금 표는 이 논문들의 공식 end-to-end 재현을 대체하지 않는다.
- formal external baseline 하나를 재현할지, 현재 방법이 알려진 특수형이라 독립 알고리즘 주장을 포기할지 먼저 판정한다. novelty 부족을 seed 증가로 해결하지 않는다.

**C. 이후에만 사용 범위와 독립 평가를 고정한다.**
- 현재 입증 범위는 frozen-feature ridge다. full-network/CE까지 주장하려면 별도 readout 예산과 대조를 먼저 정한다.
- untouched receiver와 Expert/Reserved의 정확한 역할·주가설·후보/대조·환자 bootstrap을 freeze한 뒤 평가한다.
- 독립 평가가 성공해도 표준 연산이 새로워지는 것은 아니다. 실패했다고 다른 final receiver로 교체하지 않는다.
- patient-DP 여러 ε 또는 새 cohort는 그 다음 논문 주장에 필요할 때만 설계한다. 현재부터 광범위한 noise/ε sweep은 하지 않는다.

## 7. 시간·저장공간·구현 범위

| 항목 | 계획 비용 / 한도 | 근거 |
|---|---|---|
| 비교 adapter·manifest·변경 경로 검산 | 약40~75분 | 새 implementation 작업에 대한 추정 |
| 12개 label solve + 제한된 geometry 진단 | 보통 수분; 계산 경로30분 한도 | 과거4개 solve≈1.4초. 예외적인 수렴 지연은 별도 기록 |
| 선정 P128 특징 경로 확인/필요시 encode | 수분; 새 전체 P/Q 추출 없음 | 과거기존PNG K4 1,024 forward 포함≈13.9초 |
| 평가·bootstrap·기록 | 약15~30분 | 과거joint+bootstrap≈13.5초, 나머지는 연결/보고 |
| **첫 실행 묶음 전체** | **약60~120분, 150분 작업 상한** | 기존 cache 유효 가정; GPU 상태/디버깅에 따른 보장값 아님 |
| 추가 disk | 목표≤250MiB, 한도500MiB | 모델/데이터/checkpoint 복제 없음 |

시간 한도에 걸리면 완료된 동일성 검사와 미완료 항목을 보고한다. hyperparameter를 바꾸거나 예산을 자동 증액하지 않는다. 정상 계산·저장·평가 사이에 승인 질문을 반복하지 않는다.

현재 17.62GiB 정도의 C 여유 공간을 고려해 다음을 지킨다.
- 기존 PNG는 path+SHA256 참조, 원본 복제하지 않는다.
- P128은 decode 일치가 보장되면 원본 참조, 필요시 작은 128-image export만 만든다.
- feature/label/prediction 배열은 압축 저장한다. 모델과 checkpoint 새 다운로드 없음.
- raw Q·비공개 noise seed·patient prediction을 GitHub로 자동 업로드하지 않는다.

## 8. 구현/검산과 산출물

별도 작업 폴더 제안: contribution_controls_20260928. 아직 생성하거나 실행하지 않았다.

입력:
- contribution_search_20260924의 기존 labels, DINO K4 PNG features, 공개 maps, protected target 경로, manifests.
- code_working/_reports의 P condition features, P DN/RN projections, 기존 DP1/DP2 PNG features.
- 기존 V features/2,000 bootstrap은 **별도 평가 단계에서만** 읽는다.

변경 경로만 확인한다.
1. protected target·checkpoint·조건·PNG hash 결속; P rows/image/patient 순서 일치.
2. MT 원래 구현과 공개 가중라벨 ridge의 계수/예측 동치 검산. 임의의 모의 feature 배열에서도 대수식 확인.
3. KD ridge의 normal equation과 별도 least-squares 계산 일치.
4. solver bounds/성공·KKT optimality·목표 RMS·입출력 shape 확인. 원래 hard-anchor 경로는 과거 결과 수치 재현용으로만 검산하고 새 V 성과 비교에 섞지 않는다.
5. 공개128 선택에 Q/V/DenseNet이 들어가지 않음과 환자·영상 중복 없음.
6. 모든 라벨을 고정하고 hash한 뒤 evaluator를 호출한다. construction 모듈에서 V import/read 금지.
7. 코드/입력 provenance, 허용 output 크기, 모든 실패 포함.

예정 산출물:
- contract.json / input_manifest.json / public128_manifest.json
- operation_equivalence.json / geometry_diagnostic.json
- labels.npz / label CSVs / label_seal.json
- results.json / predictions_private.npz / paired_bootstrap.npz
- PUBLIC_TRANSFER_LABEL_CONTROLS_RESULTS_20260928.md

이번 설계 산출물은 본 문서이며, 위 실험 산출물은 아직 없다. 현재 actual result pointer는 9월24일 contribution search를 유지한다.

## 9. 정식 선행과 이번 비교의 관계

| 선행 | 확인된 범위 | 이번 계획에 반영 |
|---|---|---|
| KIP / Label Solve, ICLR2021 | 고정 이미지의 label solve, ridge 학습 결과 정합 | 단순 pseudo-label만 대조하지 않고 source-only LS에도 같은 solver 허용 |
| Balog et al., ICML2018 | DP kernel mean embedding과 공개/합성 가중 표현 | 평균 전달·signed-weight 동치를 새 기여로 세지 않음 |
| POST, ICML2025 | LLM soft prompt의 public-data 기반 private transfer | 원리 중복 인정. 이 vision ridge 대조를 POST 전체 재현이라고 부르지 않음 |
| Hard Truths about Soft Labels, CVPR2026 | soft label 조건에서 image/subset 가치와 계산비용 혼입 분석 | public carrier에 같은 label solve 허용. public baseline 자체를 새 발명으로 세지 않음 |

정식 출판 근거:
- [KIP, ICLR2021 저자 소속기관 기록](https://research.google/pubs/dataset-meta-learning-from-kernel-ridge-regression/) · [본문/Label Solve 식](https://arxiv.org/html/2011.00050v2)
- [Balog et al., ICML2018 정식 proceedings](https://proceedings.mlr.press/v80/balog18a.html)
- [POST, ICML2025 정식 proceedings](https://proceedings.mlr.press/v267/wang25ds.html)
- [Hard Truths, CVPR2026 정식 proceedings](https://openaccess.thecvf.com/content/CVPR2026/html/Dey_Rethinking_Dataset_Distillation_Hard_Truths_about_Soft_Labels_CVPR_2026_paper.html)

2026-09-28 조회: KIP 공식 landing은 확인, OpenReview PDF 직접 열기는 browser challenge라 원문 companion HTML을 사용. Hard Truths의 직접 open은403이었지만 CVF 정식 검색 결과에서 CVPR2026 pp178–187 및 본문을 확인했다. arXiv를 별도 미심사 논문으로 취급한 것이 아니라 동일 출판 연구의 접근 가능한 원문으로 구분했다.

## 최종 현재 상태

**기여가 확정된 것이 아니라, 기여 후보를 겨냥한 실행 계획이 구체화됐다.** 현재 개선 결과를 유지하면서, 표준 전달·label solve·공개 carrier라는 강한 설명을 먼저 허용한다. 이들을 제어한 후 남는 차이의 기능·전이·비용을 확인한다.

이 문서 작성은 code/literature inspection + algebra + 계획 기록만 수행했다. 새 실험은 시작하지 않았고 새 연구 효용 결과는 없다.

