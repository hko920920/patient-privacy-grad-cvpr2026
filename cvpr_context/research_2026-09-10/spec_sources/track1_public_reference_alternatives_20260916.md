# 공개 기준 통계가 있을 때의 강한 대안: 중심 이동과 공개 Gram
2026-09-16. 읽기·수식 검토만 수행했다. 현재 4-cell 실행·공개 선택 규칙은 변경하지 않는다. Private/eval 통계·성능을 열지 않았으며 새 실험·GPU·구현은 없다. 이 메모는 결과에 따라 현재 실험을 구제하는 조정안이 아니라, 공개 reference가 있는 비교에서 빠뜨리기 쉬운 대안을 사전에 설명한다.

## 1. 판단
**공개 A/B 중심의 joint innovation clipping은 매우 직접적인 강한 대안이다.** 현재 zero-centered SSP를 “공개 정보를 충분히 활용한 최선의 one-shot 방법”으로 부르기 전에 비교해야 한다. 동일 좌표·같은 Gaussian 보장 안에서 clipping 편향과 공개 정보 활용을 바꾸는 표준 연산이며 새로운 DP 원리로 주장할 근거는 없다.

**공개 Gram 고정 + B만 보호**도 직접적인 대안이다. d=64,o=4이면 보호 좌표를 2,336에서 256으로, d=16이면 200에서 64로 줄인다. 그러나 일반적으로 private denoising ridge의 목적을 그대로 풀지 않는다. 목적 변경과 통계 잡음 감소의 tradeoff를 설명하는 비교여야 한다. 더 나아가 **공개 ridge W0에서 private gradient 한 번만 보호하고 공개 Hessian으로 갱신**하는 방법도 같은 좌표 수의 가까운 대안이다.

어느 대안도 현재 자료에서 효용이 더 좋다고 확인하지 않았다. 공개32와 private80의 분포 차이·조건수·공개 추정 오차가 결과를 바꿀 수 있다.

## 2. 표기와 현재 목적
고정 특징/denoising 정답으로 만든 환자 평균 통계를 A_u,B_u라 하고 공개 고정 분모를 N0라 한다. 공개 side information A0,B0 및 a0,b0,C는 인접 private 데이터셋 사이에서 변하지 않아야 한다.

s_u = concat(svec(A_u)/a0, vec(B_u)/b0),   s0 = concat(svec(A0)/a0, vec(B0)/b0).

svec는 비대각에 sqrt(2)를 곱한다. 원 목적의 데이터 의존 부분은
L(W)=tr(Wᵀ A W)−2tr(Wᵀ B)+lambda||W||²,
A=Σ A_u/N0, B=Σ B_u/N0.
이미지·공간·noise를 환자 내부에서 평균했다는 기존 정의를 유지한다. 실제 환자 수 n과 공개 N0를 같은 변수로 몰래 치환하지 않는다.

## 3. 중심 이동 SSP의 정확한 DP 조건
다음 메커니즘은 사실 수준으로 표준 Gaussian 증명이 가능하다.

r_u=clip_C(s_u−s0)
M(D)=s0+[Σ_(u present) r_u + Z]/N0,
Z~N(0,sigma² C² I).

공개 중심은 합계 밖에서 한 번 더한다. 빠진 환자의 innovation 기여는 **0**이다. 다른 환자의 r은 해당 환자가 추가·제거되어도 변하지 않는다. 동일 공개 정보에 조건부로,
||M_noiseless(D∪{u})−M_noiseless(D)||≤C/N0.
따라서 add/remove 민감도 C/N0, replace 민감도 상계 2C/N0이다. 현재와 같은 one-shot Gaussian 회계를 적용할 수 있고, noisy A/B 복원·PSD projection·ridge는 후처리다. 공개 중심 추정이 틀려도 DP는 유지되지만 효용은 나빠질 수 있다.

### 3.1 “동일 목적 유지”의 정확한 경계
Clipping과 noise가 없으면
M(D)=Σs_u/N0 +(1−n/N0)s0.
따라서 **현재 관측된 n=N0에서만** 원 통계와 정확히 같다. 제거 이웃에서는 비어 있는 공개 슬롯을 s0로 보충하는 목적이다. 이는 유효한 add/remove DP 메커니즘이지만 “모든 크기의 데이터에서 원래 zero-padded 목적을 그대로 유지”한다고 해서는 안 된다.

원 평균을 모든 n에서 유지하려고 n*s0/N0를 더하면 한 환자 추가 시 s0도 함께 증가한다. 기여는 s0+clip(s_u−s0)이 되어 일반 상계는 (||s0||+C)/N0이고, 기존 C/N0 증명을 그대로 사용할 수 없다. 실제 n으로 나누거나 n을 그대로 공개하는 것도 별도 문제이다.

공개 registry의 N0개 슬롯을 고정해 present/absent adjacency로 정의하면, absent 슬롯이 “private statistic 0”인 목표와 “public reference s0”인 목표가 다르다는 점을 명시할 수 있다. 알려진 고정 n의 replace adjacency만 선택하면 중심 항은 양쪽에서 상쇄되지만 그 경우 표준 상계 2C/N0로 회계를 맞춰야 한다.

### 3.2 clipping 뒤 목적
n=N0일 때 w_u=min(1,C/||s_u−s0||)라 두면
A_center=(1/N0)Σ[w_u A_u+(1−w_u)A0],
B_center=(1/N0)Σ[w_u B_u+(1−w_u)B0].
이것은 강하게 잘린 환자를 공개 통계 쪽으로 대체하는 이차 목적이다. 원 zero-centered clipping의 A_clip=Σw_u A_u/N0, B_clip=Σw_u B_u/N0와 다르다. 동일 환자 weight라는 구조는 남지만 “원 목적의 불편 추정”은 아니다.

A0와 A_u가 PSD이고 n≤N0이면, centered clipping의 A는 noise 전 PSD이다. 일반적으로 n>N0까지 허용하면 이 단순 PSD 주장에는 추가 조건이 필요하다. Gaussian 이후는 어느 쪽이든 PSD 보정된 surrogate로 구분한다.

### 3.3 무엇이 좋아질 수 있는가
환자 통계가 공개 중심에 집중되어 있으면 공개 자료에서 정한 더 작은 C로 같은 clipping 비율을 얻어 σC를 줄일 수 있다. 같은 C를 유지하면 좌표 수와 Gaussian 절대 크기는 그대로이고 주로 clipping 편향만 변한다. 공개 중심이 target과 멀면 반대로 clipping이 커질 수 있다. 중심, scales, C를 private/eval 결과에서 골라 무료로 바꿀 수 없다.

## 4. 공개 Gram 고정+B 보호
공개 A0를 사용하고
Bhat=B0+[Σ clip_Cb(vec(B_u−B0))+Z_b]/N0,
What=(A0+lambda I)^−1 Bhat
으로 두면, 같은 absent/N0 조건 아래 B-vector add/remove 민감도 Cb/N0이다. 원형 B를 zero-centered로 보호해도 가능하다. Gram에 잡음을 넣지 않아 PSD 잡음과 d² 보호 좌표를 없애지만, 공개 Gram의 추정·분포 불일치를 새 오차로 가져온다.

Clipping/noise가 없고 n=N0라 하자. 원 ridge W*=(A+lambda I)^−1 B와 비교하면
W_publicGram−W*=(A0+lambda I)^−1(A−A0)W*.
이는 정확한 항등식이다. 따라서 private A가 public A와 가까운 경우 이점이 가능하지만, 희귀/시간별 feature 방향의 covariance shift가 크면 편향이 남는다. A0를 “공개로 계산했으므로 정확한 Hessian”이라고 부르면 안 된다.

고정 A0만으로 원 private quadratic을 정확하게 풀 수 있는 것은 A0=A이거나 위 오차 항이 해당 W*에서 사라지는 특수 경우다. 이 대안을 동일 목적을 푸는 SSP라고 이름 붙이지 않는다.

## 5. 더 가까운 표준 대안: 공개 W0에서 한 번의 보호된 갱신
공개 ridge W0=(A0+lambda I)^−1 B0를 먼저 만들고,
t_u=B_u−A_u W0
라는 private residual moment만 보호할 수 있다. 공개 중심 t0=B0−A0W0=lambda W0에서 innovation을 clip하고, 공개 고정 N0로
that=t0+[Σclip(t_u−t0)+Z]/N0,
Wnew=W0+(A0+lambda I)^−1(that−lambda W0)
로 갱신한다. t-vector 민감도만으로 표준 DP가 가능하다. 이는 공개 Hessian을 이용한 한 번의 gradient/Newton형 보정으로 해석할 수 있으며, 256/64차원만 보호한다.

Clipping/noise가 없고 n=N0이면
Wnew−W*=(A0+lambda I)^−1(A0−A)(W0−W*).
직접 B-only의 오차가 W*에 비례했던 것과 달리 이 오차는 공개 초기해와 target 해의 차이에 비례한다. **공개 W0가 이미 유용하다는 조건에서** 더 강한 비교가 될 수 있다. 이 식은 정확한 선형대수이며 현재 의료 자료의 조건 충족을 입증하지 않는다. 반복해서 새로운 private residual을 질의하면 추가 DP 회계가 필요하다.

W=W0+U라는 재매개화만 하면 원 ridge는 lambda||U+W0||²이다. 이를 설명 없이 lambda||U||²로 바꾸면 공개 W0를 향한 다른 prior가 된다. 공개 초기화, feature centering/whitening, residual target 자체는 표준 연산이며 신규성의 중심으로 삼지 않는다.

## 6. 직접 선행과 읽은 범위
- **DOPE-SGD**, [공식 ICML/PMLR PDF](https://proceedings.mlr.press/v202/nasr23a/nasr23a.pdf), PDF p.3 §3.1/Algorithms 1–2, p.4: 공개 정보로 clipping 중심을 이동한다. 이 원문을 이번에 다시 직접 열어 확인했다. 본 메모의 통계 중심화는 이 선행을 넘어서는 새 privacy principle이 아니다. 다만 원문의 sampling/합산 표현을 현재의 add/remove·고정 N0 설정에 그대로 복사하지 않고 §3의 민감도를 별도로 증명했다.
- **AdaSSP/BoostedAdaSSP**, [저자 PDF 링크](https://openreview.net/pdf/5dafc02af37718cf092472a357459a9500fabadb.pdf), 이전 operation note에서 확인한 §III Algorithm 1/App.B Algorithm 2: noisy Gram/response 통계의 회귀 및 Gram 재사용·새 residual 질의라는 직접 경계이다. 이번 재접속은 OpenReview browser challenge여서 상세 방법을 다시 읽었다고 하지 않는다. §5를 해당 논문의 원형 알고리즘이라고 부르지는 않는다.
- **DP-MERF**, [공식 AISTATS/PMLR 페이지](https://proceedings.mlr.press/v130/harder21a.html): 고정 특징 평균을 한 번 보호하고 재사용하는 생성학습이 이미 있다. 이번에는 공식 페이지를 확인했으며 “공개 B-centered ridge와 동일한 알고리즘”이라는 주장은 하지 않는다.
- **Where the Score Lives**, [공식 AISTATS 2026 페이지](https://proceedings.mlr.press/v300/finn26a.html): denoising의 고정 특징·모멘트·ridge 연결 자체는 이전 operation note에서 읽은 직접 선행이다. 이번에는 해당 상세 본문을 새로 열지 않았다.
- 이전 상세 문헌 검토는 [operation note](track1_operation_redesign_20260916.md)에 보존되어 있다. 전체 최신 문헌에 이러한 구성이 없다는 신규성 검토를 수행한 것은 아니다.

## 7. 현재 비교에 주는 의미
현재 4-cell 비교는 zero-centered joint SSP와 같은 head의 user-DP-SGD가 각각 무엇을 하는지 확인하는 유효한 고정 실험이다. 그대로 완료한다. 다만 그 결과만으로 “공개 reference를 활용하는 one-shot 보호 설계 전체”를 판정하지 않는다.

다음 논문 수준 비교를 설계한다면 centered joint SSP, public-Gram B-only, public-ridge one-step correction이 직접적인 강한 대안이다. 새 backbone 추출 없이 같은 내부 환자 통계로 구현 가능하다는 사실은 비용상의 장점이지만, 공개 calibration 선택·private 질의·결과 공개에는 각각 올바른 범위와 회계가 필요하다. 이 메모는 해당 추가 실험을 자동 승인하거나 실행하지 않는다.

공정한 비교는 공개32 접근권·원 head·ridge 정의·환자 보장·고정 N0를 맞춘 뒤 목적 변경의 정도와 clipping/noise/공개-domain bias를 구분한다. Public-only W0는 항상 함께 두어 “공개 적응 자체”를 private 적응의 이득으로 오인하지 않게 한다. 새로운 논문 기여가 될 수 있는 부분은 이러한 표준 대안들 중 어떤 연산이 현재의 작은 denoising 표현과 결합될 때 왜 유리한지이며, 공개 중심을 뺀다는 사실 자체가 아니다.

