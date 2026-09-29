# 2-4A: 기존 HR 실험 중단과 새 보호 신호 설계 후보

2026-09-29. 상위2의 **새 설계 항목 2-4A**다. 미실행·권고 철회된 2-1R3를 실행한 것으로 바꾸지 않는다. 연구 단계2와 상위3 미착수를 유지한다.

## 결정

**현재 HR의 가중치·기준값·대응식 대조를 더 추가하는 경로는 중단한다. 기존 HR의 효용 결과는 보존하되, 그 결과만으로 CVPR 주력 기여를 확보했다고 하지 않는다.**

이번에 선택한 개발 후보는 다음 하나다.

> **기존 모델의 예측 순위를 보존하면서 고정된 공개 영상·라벨로 다른 모델에 전달할 수 있는 신호를 먼저 계산하고, 그 성분을 환자별 clipping 전에 분리해 보호한다.**

이는 기존 HR의 원인을 더 설명하는 대조가 아니라 **보호할 query와 라벨 제약을 함께 바꾸는 별도 설계**다. 현재 고정된 DINO 요약만으로 원하는 모든 receiver 정보를 복구할 수 있다고 가정하지 않는다. 필요한 receiver 신호가 기존 요약 밖에 있다면 실제 Q에서 새 신호를 보호해야 한다. **이번에는 그 조회·release·라벨 생성·성능평가를 실행하지 않았다.**

판정은 **수식과 공개자료에서 성립 가능성을 확인한 개발 후보**다. 선행보다 유용함, 독자적 기여 충분성, unseen receiver 개선, CVPR 채택 가능성을 확인한 결과가 아니다. 알려진 원리의 조합이라는 이유만으로 폐기하지도 않고, 새 사용 조건을 이름 붙였다는 이유로 신규성을 인정하지도 않는다.

## 이번에 바꾼 판단 근거

- HR가 강화된 내부 KD 대조보다 좋았던 두 공개 영상 선정의 결과는 유효하다. 최신 가중치 교차는 **HR와 ACKD가 모두 환자균등 가중치에서 좋아진다**는 가설을 지지하지 않았다. 이 실패를 일반적인 가중치 기여로 바꾸지 않는다.
- 현재 HR는 공개 기준값과 source 변화량의 공개 선형 전달로 분해되며, 공개 가중자료 표현과도 동치다. 이미 검산한 분해를 또 검산해 신규성을 찾는 작업은 하지 않는다.
- 유한 source 평균만으로 다른 표현의 평균을 일반적으로 식별할 수 없다. 예를 들어 같은 표본 수·같은 평균을 가진 x=−1,+1과 x=0,0은 source feature x의 요약은 같지만 receiver feature x²의 요약은 다르다. 잡음 전부터 같은 source 요약이므로 후처리로 둘을 항상 구분할 수 없다. **표준 식별 가능성의 예시이며 새 정리가 아니다.** 현재 HR 성능 차이의 원인이 이것이라고 입증한 것도 아니다.
- 원래 DINO-only 라벨 계산의 128개 중 DP1은100개, DP2는96개가 box 끝값에 있다. 따라서 기존 box 안에 source 보존 변경 방향이 충분하다고 가정할 수 없다. 아래 설계는 이 제약을 명시적으로 처리한다.

## 같은 정보를 허용한 선행 대응

| 정식 선행 | 이미 알려진 연산 | 이번 후보에서 달리 검증할 부분 / 경계 |
|---|---|---|
| Balog et al., ICML2018, DP Kernel Mean Embeddings | 보호 평균을 공개자료의 가중 표현으로 변환하고 재사용 | 공개 라벨·요약 재사용 자체는 신규성이 아니다. 현재 HR의 해석에 직접 해당한다. |
| Nguyen et al., ICLR2021, KIP / Label Solve | kernel ridge를 거친 예측을 맞추도록 고정 support의 라벨을 계산. 해 공간·투영 분석도 존재 | source 보존 선형 제약과 라벨 최소제곱만으로 새 알고리즘을 주장하지 않는다. 후보의 추가 질문은 **그 라벨 학습 연산에서 보호 query를 역으로 정하는 것**이다. |
| Yu et al., ICLR2021, GEP | 공개 anchor subspace로 gradient를 나누고 embedding·residual을 각각 제한·보호 | 투영 후 clipping, 작은 부분공간의 보호 자체는 선행. 같은 표현 차원·환자 예산의 공개 PCA 기준을 허용해야 한다. |
| Zheng et al., ICCV2025, Dosser | 보호 신호 재사용과 보조자료 기반 informative subspace, 투영 손실과 잡음의 절충 | 본 후보는 신호의 큰 분산보다 **고정 영상의 라벨이 기존 모델을 보존하면서 바꿀 수 있는 학습 결과의 공간**을 기준으로 삼는다. SER에 이 공간을 넣어 같은 계산이 되면 별도 알고리즘 두 개처럼 비교하지 않는다. 차이는 공간 선택 기준의 가치로만 평가한다. |
| Xiao et al., PVLDB2023, Common Mechanism | 기존 Gaussian query에서 공통 정보를 재사용하고 추가 residual mechanism으로 새 query에 답함 | 단순히 ‘부족한 잔차만 추가 보호’하는 안도 새로운 원리가 아니다. 기존 보호 count의 재사용·추가 예산 회계는 이 선행의 관점을 허용한다. |
| Matrix mechanism / ResidualPlanner+ | workload에 맞춰 측정·잡음 구조를 설계 | 가장 중요한 중복 위험이다. **workload-aware DP의 일반 원리와 다르다고 주장하지 않는다.** 현재 artifact·source 보존 제약에서 도출한 workload가 실제로 다른 기준보다 유용한지가 남는다. |

공식 근거: [DP-KME, ICML2018](https://proceedings.mlr.press/v80/balog18a/balog18a.pdf), [KIP 정식 출판 기록](https://research.google/pubs/dataset-meta-learning-from-kernel-ridge-regression/) 및 [저자 본문](https://arxiv.org/html/2011.00050v3), [GEP 정식 ICLR2021 기록](https://www.microsoft.com/en-us/research/publication/do-not-let-privacy-overbill-utility-gradient-embedding-perturbation-for-private-learning/) 및 [본문](https://openreview.net/pdf?id=7aogOj_VYO0), [Dosser ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Zheng_Improving_Noise_Efficiency_in_Privacy-preserving_Dataset_Distillation_ICCV_2025_paper.pdf), [Common Mechanism, PVLDB2023](https://www.vldb.org/pvldb/vol16/p1883-xiao.pdf), [ResidualPlanner+, VLDB Journal2026](https://link.springer.com/article/10.1007/s00778-026-00981-9).

OpenReview 일부 본문은 직접 열기에서 browser challenge가 있었고 일부 공식 PDF도 도구에서 재조회에 실패했다. 공식 게재 기록, 검색 도구가 반환한 원문 부분, 접근 가능한 저자 본문과 기존 저장소 대조를 구분해 사용했다. 전수 문헌조사나 모든 논문의 전체 구현 재현을 했다고 하지 않는다. CLP-DD의2026 저자 페이지도 확인했으나 해당 페이지는 preprint로 표시되어 정식 게재 논문으로 세지 않았다.

**따라서 신규성 후보는 ‘nullspace’, ‘잔차’, ‘차원 축소’가 아니다. 기존에 사용하던 자료의 모델별 기능을 유지해야 하는 조건에서, 자료가 전달할 수 있는 신호에 맞춰 환자-DP query를 결정하는 설계와 그 실제 가치다. 이 좁은 차이조차 기존 workload 설계의 단순 적용 이상인지 아직 미확정이다.**

## 설계: 보존할 기능과 추가할 기능을 수식으로 분리

현재와 같은 frozen-feature, signed-label, ridge readout에 한정한다. n=128, 기존 source의 공개 carrier 특징 X_S, 추가 receiver 특징 X_R를 두고

\[
B_m=(X_m^TX_m/n+\lambda I)^{-1}X_m^T/n
\]

로 쓴다. 기존 보호 요약으로 이미 만든 source-only 라벨을 y₀라 한다. 이것은 원래 teacher 전체를 정확히 재현한다는 뜻이 아니라 **보존할 기존 artifact**다.

### 1. Source를 유지하는 라벨 변경

N의 열을 ker(X_Sᵀ)의 정규직교 기저로 두고

\[
y=y_0+Nv
\]

로 제한한다. 그러면 B_S y=B_S y₀다. 같은 carrier·특징·label-independent 전처리라면 모든 양의 ridge λ에 대해 성립한다. 새 시험 입력에서도 source의 점수 함수가 동일하다.

box 문제는 양의 공통 배율로 처리하는 후보를 명시한다.

\[
\alpha=1/\max(1,\|y\|_\infty),\qquad \ell=\alpha y.
\]

모든 frozen linear ridge의 점수는 α배가 된다. 따라서 source의 AUROC/AP는 수학적으로 기존 y₀와 같다. 실제 구현에서는 수치 정밀도와 동점 처리를 검사해야 한다. **확률 보정, 고정 threshold 성능, cross-entropy 학습, backbone fine-tuning에 관한 보장이 아니다.** 소비자는 현재의 signed-label readout 규칙을 사용해야 한다.

양의 배율 정규화는 강한 일반 Label Solve 대조에도 동일하게 허용한다. 기존 bounds를 제거해 생기는 효과를 후보의 고유 이득으로 세지 않는다. Source 보존을 단독 기여로 주장하지도 않는다. 아래 target 제곱오차의 공간 분해는 배율 적용 전 y에 대한 것이다. 배율 적용 후 확률·계수 크기까지 같은 목표를 재현한다고 주장하지 않는다.

### 2. 추가 receiver에 전달할 수 있는 target 성분

공개 P로 고정한 receiver curvature를 H_R=R_PᵀΩR_P+0.1I로 둔다. 환자/class 균등 목표의 contrast를 b_R=(μ₁−μ₀)/2라 하면 목표 head는 w*=H_R⁻¹b_R다.

\[
M=H_R^{1/2}B_RN,\qquad U=\operatorname{orth}(\operatorname{range}(M)).
\]

source 보존 라벨을 계산하는 제곱 목적

\[
\min_v\|H_R^{1/2}[B_R(y_0+Nv)-w^*]\|^2
\]

은 private target 중

\[
q=U^TH_R^{-1/2}b_R
\]

만 있으면 된다. 직교 여공간의 target은 이 목적의 최솟값에는 상수를 더하지만 v의 해를 바꾸지 않는다. 선택된 최소 norm 해 또는 v의 정규화도 같은 공간 안에서 처리한다.

이는 **해당 제한된 목적에 불필요한 신호를 버려도 최적 라벨 변경이 같다는 선형대수 결과**다. 미래의 모든 모델·손실에 불필요하다는 뜻은 아니다. Source/receiver에 쓰인 모델은 construction model이며 독립 전이 평가로 세지 않는다.

### 3. 핵심 변경: 환자별 clipping 전에 이 성분을 선택

환자 i의 class c 방문 평균 receiver 특징을 r_ic라 하고 a_ic=H_R⁻¹ᐟ²r_ic로 둔다. 같은 공개 cap C에서 두 연산은

\[
\text{full control}: U^T\operatorname{clip}_C(a_{ic}),
\qquad
\text{candidate}: \operatorname{clip}_C(U^Ta_{ic})
\]

다. U가 정규직교 열을 가지므로

\[
\|\operatorname{clip}_C(U^Ta)-U^Ta\|
\leq
\|U^T\operatorname{clip}_C(a)-U^Ta\|.
\]

증명은 두 결과가 Uᵀa의 양의 배율이고, 투영 후 norm이 커지지 않으므로 뒤 방식의 축소율이 더 작다는 것이다. 가령 a=(1,L), U=(1,0)ᵀ일 때 두 번째 좌표는 라벨 변경에 기여하지 않지만 full clipping에서는 첫 번째 좌표까지 줄일 수 있다.

**환자 하나의 투영 성분에 대한 clipping 오차 부등식이다. 여러 환자 bias가 상쇄될 수 있으므로 aggregate bias·최종 AUROC의 우위를 보장하지 않는다. 이 부등식 자체도 새로운 DP 정리라고 주장하지 않는다.**

또한 차원만 줄여서 retained Gaussian noise가 작아진다고 설명하면 틀리다.

\[
U^T\eta\sim\mathcal N(0,\sigma^2I_k),\quad
\eta\sim\mathcal N(0,\sigma^2I_d).
\]

같은 cap·sensitivity·coordinate noise라면 full release를 나중에 투영한 결과와 reduced release의 retained noise 법칙은 같다. **차이를 겨냥하는 곳은 clipping 순서와, 명시적인 bias 대조를 동반한 공개 cap 선정이다.** clipping이 없고 cap도 같으면 두 방법이 같은 분포가 된다는 대조가 필수다.

### 4. 환자 보호 계약의 초안

한 환자가 두 class에 모두 기여할 수 있다. 각 class의 projected contribution을 clip_Cc하고

\[
t_i=2^{-1/2}[v_{i0}/C_0;v_{i1}/C_1]
\]

로 결합하면 norm≤1이다. 존재하지 않는 class는0이다. add/remove patient adjacency에서 sum sensitivity≤1이며, 한 Gaussian mechanism으로 추가 신호를 보호할 수 있다.

class denominator는 **동일 환자집합·정의의 기존 보호 count를 재사용**하는 안을 우선한다. 기존 count와 다른 환자 집합/정규화를 섞지 않는다. source release가 제공하지 않는 count라면 별도 보호와 감도 재계산이 필요하다. 분모 floor·공개 P 결합은 기존 명세와 대응해 실행 전에 고정한다.

noisy denominator와 source 라벨은 서로 의존할 수 있다. 최종 비율을 독립 Gaussian mean처럼 취급하지 않는다. 위 간단한 noise 등식은 **분모 복원 전 projected numerator**에 관한 것이다.

**추가 Q query는 추가 개인정보 비용이다.** 기존 ε8 source를 쓰고 새 query를 했는데 전체를 ε8이라 쓰면 안 된다. 실제 incremental 비교에서는 같은 기존 transcript와 같은 추가 예산을 양쪽에 허용한다. 향후 total ε8을 주장하려면 source와 receiver query에 예산을 나누는 새 계약이 필요하며, 기존 ε8 bank의 성능을 그 결과로 대체할 수 없다. 이번에 새 예산값·release 일정은 채택하지 않았다.

## 이번 공개자료 검산

현재 고정 public128, P672환자/813영상/675환자-class 행만 사용했다. Q·V·기존 DP 값·새 라벨·DenseNet 특징·Expert/Reserved는 읽지 않았다. 코드는 `contribution_signal_design_20260929_v1/public_geometry.py`, 결과는 같은 폴더 `public_geometry.json`이다.

| 검산 | 값 | 의미 |
|---|---:|---|
| carrier DINO feature rank |64|128개 라벨에 source 보존 변경 공간64차원 존재 |
| source 보존 공간에서 receiver를 바꾸는 rank |64|그 공간이 RN에 전혀 보이지 않는 퇴화 상황은 아님 |
| M condition number |9.2786|이 공개 행렬에서 극단적인 역산 불안정성은 관측되지 않음 |
| 전체 a 중 retained norm energy, 환자균등 평균 |0.72709|구체적인 공간 차이가 존재. Q/task relevance 근거는 아님 |
| 공개 전체 norm의95% quantile |2.77756|설명용 기준. 실제 class cap 채택이 아님 |
| 공개 projected norm의95% quantile |2.47272|설명용. 이 값으로 자동 변경하지 않음 |
| 같은 C=2.77756에서 full→clip 행 수 |34/675|공개 기하 진단 |
| 같은 C에서 project→clip 행 수 |1/675|순서 차이가 실제 공개 행에 존재 |
| 환자 가중 per-row projected clipping 오차 평균 |0.0046103→0.00004785|개별 행 오차의 평균. 집계 오차·분류 효용이 아님 |
| X_SᵀN 최대 오차 |6.19e−16 미만|source 보존 대수 검산 |
| M의 range 투영 재현 오차 |2.09e−17 미만|목표 공간 검산 |
| retained noise covariance 단위행렬 오차 |2.56e−15 미만|차원만 줄인 noise 이득을 주장하지 않음 |

입력3개 SHA256은 기존 계약과 대조했고 실행 전후 동일했다. 수치 계산은0.125초였다. 전체 소요시간은 문헌 조사·설계·기록을 포함한 별도 실행 기록에 남긴다.

**이 결과는 설계 공간이 실제로 존재한다는 근거다. 새 방법의 성능이 잘 나왔다는 결과가 아니다.** P의 양성 환자는6명뿐이므로 public clipping 진단을 Q의 양성 utility/noise 분포로 일반화하지 않는다. 동일한 공개 입력으로 사전에 효과가 없음을 걸러낼 수 있다는 의미도 있지만, 이것을 논문 성공 gate로 쓰지 않는다.

## 다음 하나의 범위: 2-4B 공개 입력 구현·방법 비교

**예상60~90분, CPU 캐시 사용. 이번에는 실행하지 않았다.** 신규 query/labels/prototype을 구현하는 다음 묶음이며 HR 원인 대조의 재개가 아니다. 실험 번호를 더 쪼개 자동 대기열로 만들지 않는다.

이 묶음에서 한꺼번에 닫을 것은 다음이다.

1. 위 projector·patient query·공유 count 후처리·source 보존 라벨의 최소 구현. 공개 입력과 이미 보호된 정보만 허용하며, 실제 Q 신호를 추출하지 않는다. Gaussian accountant를 새로 발명하지 않고 기존 구현을 재사용한다.
2. **full→clip→project, compatible project→clip, 같은 차원의 공개 PCA/GEP/SER식 project→clip**을 같은 numerator 예산·count·최종 라벨 학습 연산에서 비교한다. 같은 U를 사용하는 표준 workload mechanism은 동치이면 중복 실행하지 않는다.
3. clipping을 없애면 retained noise/해가 같아지는 귀무 대조, source 순위 보존 조건, labels의 양의 공통 배율 처리, 환자 adjacency의 상계를 확인한다. 일반 Label Solve에도 같은 bounds 처리·공개 모델·예산을 허용한다.
4. 공개 데이터의 noise 전 target은 공개이므로 projected target 복원 오차를 분해할 수 있다. 평균적 잡음 영향은 가능한 부분은 선형 연산의 analytic bias/variance로 계산한다. 분모·배율의 비선형 효과가 필요한 경우에만 **한 번 정한 공개 모의 잡음 묶음**을 모든 방법에 공통 사용하고, 크기와 범위를 실행 전에 명시한다. P를 Q와 같은 대규모 의료 cohort로 복제하지 않는다.

**출구는 명확하다.** 같은 차원의 표준 공개 부분공간/동일 자원 workload 설계로 설명되는 것 외에 실용적인 차이가 없거나, 요구하는 source 보존이 receiver 효용을 제한해 추가 비용을 정당화할 여지가 약하면 이 후보는 주력으로 채택하지 않는다. 그때 바로 공간 수·cap·encoder를 바꾸는 후속을 예약하지 않는다. public prototype이 좋아도 CVPR 기여 완료로 쓰지 않고, 실제 Q 추가 조회가 필요한 다음 계약과 비용을 별도 보고한다.

대조 결과의 엄격한 통계적 유의성을 공개 소수 양성에서 억지로 요구하는 계획도 아니다. 먼저 구현 가능한 실제 연산 차이, 기존 방법과의 동치 여부, 보호·보존 보장, 공개 신호에서의 오차 구조를 한 묶음으로 판단한다. 임상 성능과 독립 평가로 가는 것은 그 이후다.

## 현재 번호와 실제 완료 범위

- 상위1: 이전에 정한 기본 대조 완료·혼합 결과 보존.
- 2-1/2-2 및 R1/R2: 기존 결과 그대로 보존. 2-1R3는 미실행·권고 철회 유지.
- **2-4A: 선행 재검토·새 후보 설계·공개 기하 검산 완료.**
- **2-4B: 공개 입력 prototype/대조 묶음, 계획만. 실행하지 않음.**
- 상위2의 기여 검증: 미완료. 상위3: 미착수. Expert/Reserved 보존.
- 실제 최신 효용 결과는 계속2-2R2다. 이번 문서를 새 효용 결과 보고서로 바꾸지 않는다.

이번에 찾은 것은 **검증할 수 있는 새 설계 후보 하나**다. 독자적 논문 기여를 찾았다고 확정하지 않는다. 반대로 기존 구성의 미세 조정만 계속하면 된다는 권고도 하지 않는다.
