# 2-1 결과: 공개 목표 전달의 연산·근접연구 비교

작성: 2026-09-28. 연구 단계2 유지. 실행계획 상위2의 세부2-1.

**판정: 현재 MT의 핵심 연산은 공개 가중표현과 label solve의 알려진 구조에 속한다. 일반 teacher-score KD와의 실증적 차이는 있지만, 그 차이를 새로운 정보 전달 원리의 우위라고 해석할 수 없다. 다음2-2는 ‘추가 class 평균 정보’와 ‘동일 teacher 정보의 공개 재구성’을 구분하는 두 라벨 대조로 좁힌다.**

1-5의 혼합 결과를 보존한다. 현재 MT의 DINO-only 대비 필요성, 결합 필요성, 합성 pixel 필요성은 일관되게 확인되지 않았다. 이 보고서는 연구 전체가 불가능하다고 결론내리거나 새 제목으로 기여를 확정하지 않는다.

## 같은 정보 접근 조건에서의 비교표

공통 자원은 기존 두 DINO feature-DP 결과, 공개 P의 고정 DINO/RN 특징, 고정 공개128, 같은 ridge0.1·bounded label solver다. Q 재접근과 새 DP release는 없다. DenseNet은 개발 평가에만 사용하며, RN은 목표 제작 모델이다.

| 연산 | 보호 정보의 사용 | 공개자료의 사용 | 현재 MT와의 관계 | 현재 근거 / 남은 질문 |
|---|---|---|---|---|
| DINO-only label solve | 두 class 평균의 차이로 만든 DINO ridge head | P에서 head 함수, 공개128의 학습 연산 | target encoder를 DINO 하나로 둔 같은 라벨 계산 | 1-5에서 강한 대안. MT의 일관된 추가 가치 미확인 |
| 일반 teacher-score KD + 같은 label solve | DINO head가 P에서 낸 점수 | RN ridge로 그 점수를 회귀하고 공개128 라벨로 재표현 | 목표를 만드는 연산이 다름 | 1-4에서 MT보다 낮고,1-5에서 DINO-only보다도 낮음. 일반 KD 전체/POST 전체의 실패가 아님 |
| 현재 class별 affine MT | 보호된 class 평균 두 개를 따로 사용 | class별 제한을 적용한 환자/class 행으로 DINO→RN affine ridge 지도 구성 | 현재 구현 | 지도 가중치·중심화·제한·regularizer가 KD와 함께 다름 |
| 공개 signed-weight 표현 + RN ridge | 같은 class 평균 두 개 | 동일 지도에서 유도한 signed public weights/labels | **MT 목표와 정확히 동치** | 같은 정보·같은 연산을 새 baseline 이름으로 다시 학습하지 않음 |
| teacher-head 재구성 대조(HR, 다음2-2) | DINO head에서 class 차이만 복구. class 공통 평균은 P 값으로 대체 | 현재 MT와 동일 지도·RN ridge·라벨 계산 | 정보 한 부분만 제거한 연산 대조 | MT−HR은 teacher head에 없는 class 공통 평균의 추가 가치. HR−KD는 같은 teacher 정보의 공개 처리 차이 |

여기서 HR은 신규 방법 이름이나 정식 선행의 전체 재현이 아니다. 추가 정보와 공개 처리의 기여를 분리하기 위한 대조 식별자다.

## 코드에 실제로 구현된 식

공개 환자/class 행은675개(672환자,813영상). 각 행은 해당 환자/class 방문 평균이다. 지도 가중치 a는 환자마다 총1/672이며 두 class에 기여한 환자는 그 무게를 나눈다. 각 class c의 지도는 그 class 환자만으로 학습하지 않는다. **모든675행에 해당 class의 공개 clipping C_c를 적용**한다.

X_c = clip_Cc(공개 DINO 환자/class 평균), Y = 공개 RN 환자/class 평균이라 하면:

\[
\bar x_c=a^TX_c,\quad \bar y=a^TY,\quad
G_c=X_{c,0}^T\operatorname{diag}(a)X_{c,0},\quad
\lambda_c=10^{-3}\operatorname{tr}(G_c)/64,
\]
\[
T_c=(G_c+\lambda_c I)^{-1}X_{c,0}^T\operatorname{diag}(a)Y_0,\qquad
h_c=\bar y-T_c^T\bar x_c.
\]

이 문서의 평균·head는 열벡터 표기다. 코드의 행벡터 구현과 전치만 다르다. 보호 평균 μ_c를 보내는 목표는:

\[
b_R=\tfrac12(T_1^T\mu_1+h_1-T_0^T\mu_0-h_0),\quad
w_R=H_R^{-1}b_R.
\]

H_R=R^TΩR+0.1I. Ω는 지도 가중치 a와 다르며, 공개 **class 균등 → class 안 환자 균등 → 방문 균등** 영상 가중치다.

일반 KD는 d=(μ_1−μ_0)/2, H_D=D^TΩD+0.1I에 대해:

\[
w_D=H_D^{-1}d,\quad q_{KD}=Dw_D,\quad
w_R^{KD}=H_R^{-1}R^T\Omega Dw_D.
\]

현재 KD에는 sigmoid, temperature, 점수 clipping이 없다. MT와 KD는 단순히 ‘평균/점수’ 하나만 바뀐 통제 실험이 아니다. 환자/class 행 대 영상행, a 대 Ω, class별 clipping, 중심화/intercept, 지도 ridge 대 head ridge가 함께 다르다.

## 공개 가중표현과의 정확한 동치

\[
\alpha_c=a+\operatorname{diag}(a)X_{c,0}(G_c+\lambda_c I)^{-1}(\mu_c-\bar x_c)
\]

이면 Y^Tα_c=T_c^Tμ_c+h_c다. 각 환자/class 행 j의 값을 방문 수 n_j로 나눠

\[
v_i=\frac{\alpha_{1j}-\alpha_{0j}}{2n_j},\qquad q_i=v_i/\omega_i
\]

로 두면:

\[
w_R^{MT}=H_R^{-1}R^T\Omega q.
\]

기존9월24일 동치 검산을 재사용하고, 이번에는 **실제 공개 영상 가중라벨까지 펼친 RN ridge**와 현재 목표를 비교했다. 최대 계수 차이 **1.03×10⁻¹⁴**. 새 V 평가 없이 P와 기존 DP 산출물만 사용했다.

이 q는 확률 라벨이 아니다. DP1 범위 약[−15.94,11.24], DP2[−8.02,7.56]이고 signed weights도 가능하다. 현재 공개128 라벨은 별도의 bounded solve로[−1,1]에 제한한다. 따라서 ‘동치’는 먼저 **813개 공개 영상의 weighted ridge 목표**에 관한 것이며, 다른 가중치/128장/box 제약까지 무조건 같은 성능이라는 뜻은 아니다.

## 강한 선행과 실제 남는 차이

| 정식 선행 | 실제 겹침 | 현재 구현과 다른 범위 | 이번 판단 |
|---|---|---|---|
| Balog et al., ICML2018, *Differentially Private Database Release via Kernel Mean Embeddings* | 보호한 평균의 weighted data 표현. 이미 공개된 작은 자료에 private 정보로 가중치를 부여하는 설정을 직접 다룸 | 원 논문의 kernel·일관성 가정·보호 query는 현재 patient/class·finite DINO 표현과 같지 않음 | 평균→공개 가중자료 원리를 신규성으로 세지 않음. 현재 MT의 finite-feature ridge 동치는 우리 코드에 대한 대수 분석이지 그 논문의 모든 보장을 상속한다는 뜻이 아님 |
| Nguyen et al., ICLR2021, *Dataset Meta-Learning from Kernel Ridge-Regression* | §3, Eq.(7),(8): 고정 support 영상의 라벨을 KRR 후 예측이 목표를 맞추도록 계산하는 Label Solve | 현재는 공개 P의 보호 유래 목표, 여러 frozen-feature 연산, box와 zero-centered penalty | ridge를 통해 라벨을 계산한다는 원리는 선행. 추가 제약 자체를 새 기여로 확정하지 않음 |
| Wang et al., ICML2025, *Efficient and Privacy-Preserving Soft Prompt Transfer for LLMs* (POST) | §4.3: 보호한 source prompt를 public data로 다른 모델에 전달. 예측뿐 아니라 prompt로 생긴 변화도 정합 | LLM prompt, target에서 먼저 KD한 source, 두 loss 및 prompt optimization. 현재 단순 KD와 다름 | 현재 KD를 이겼다고 POST를 이겼다고 쓰면 안 됨. public transfer 자체도 신규 원리 아님 |
| Dey et al., CVPR2026, *Rethinking Dataset Distillation: Hard Truths About Soft Labels* | §3.2: 라벨 학습을 selection baseline에도 허용해야 영상 제작의 가치를 비교할 수 있음을 분석 | 자연영상·다른 학습 regime. 환자-DP frozen-feature ridge의 결과를 대신하지 않음 | 1-3의 공정한 public carrier 대조는 타당하나, pixel보다 라벨이 중요했다는 일반 관찰도 새 발견이라고 할 수 없음 |

출처: [ICML2018 공식 본문](https://proceedings.mlr.press/v80/balog18a/balog18a.pdf), [ICLR2021 저자기관 출판 기록](https://research.google/pubs/dataset-meta-learning-from-kernel-ridge-regression/)와 [저자 공개 본문 §3](https://arxiv.org/html/2011.00050v2), [POST 공식 ICML2025 기록](https://proceedings.mlr.press/v267/wang25ds.html)와 [저자 본문 §4](https://arxiv.org/html/2506.16196v1), [CVPR2026 공식 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Dey_Rethinking_Dataset_Distillation_Hard_Truths_about_Soft_Labels_CVPR_2026_paper.pdf)와 [저자 공개 본문 §3.2](https://arxiv.org/html/2604.18811v1).

OpenReview와 일부 공식 PDF의 직접 열기는 이번 도구에서 실패했다. 정식 게재 상태는 공식 기록/공식 PDF 검색 결과로 확인하고, 해당 본문은 저자 arXiv 공개본으로 대조했다. arXiv에만 존재하는 미게재 논문으로 취급하지 않으며, 읽지 못한 공식 PDF를 전부 직접 읽었다고 하지 않는다. 이4편 비교는 관련 문헌의 전수 신규성 조사도 아니다.

## 수학적으로 남는 차이: class 평균의 공통 성분

H_D는 공개이고 양의 ridge로 가역이다. 따라서 **teacher head w_D에서 d=H_Dw_D를 정확히 복구할 수 있다.** 현재 KD가 ridge를 쓴다는 사실만으로 d의 정보가 소실됐다고 주장하면 틀리다. 점수만 제한적으로 제공하는 별도 API 설정도 여기서는 가정하지 않는다.

s=(μ_1+μ_0)/2를 쓰면:

\[
b_R^{MT}=\tfrac12(T_1+T_0)^Td+\tfrac12(T_1-T_0)^Ts+\tfrac12(h_1-h_0).
\]

현재 T_1≠T_0이므로 head에 없는 s는 MT 목표에 영향을 줄 수 있다. 두 평균에 같은 h를 더하면 d와 w_D는 그대로지만 MT 목표는(T_1−T_0)^Th/2만큼 바뀐다. 이는 명확한 식의 차이이며, 그 변화가 환자 판별에 유용하다는 증거는 아니다. 강한 공개 가중표현 대안은 s까지 사용할 수 있으므로 이 차이가 선행 전체의 한계도 아니다.

공개자료에서 class0/1 clipping 활성 행은34/675,124/675. 지도 차이 spectral norm0.17211. 실제 보호 평균의 s만 공개 P의 s_P로 바꿔 본 **P상 목표 예측 차이 RMS**는 전체 MT 목표 RMS의 **2.53% / 3.51%**다. head class 차이 복구 오차2.78×10⁻¹⁷, 분해식 오차1.41×10⁻¹⁷. 이는 작은 추가 성분이라는 단서지만 downstream 효과를 미리 0으로 단정할 수 없다.

**현재 가장 구체적인 대조 가설:** MT−KD 개선의 상당 부분은 teacher head에 없는 정보 자체보다, 같은 d를 공개 지도에 맞게 사용하는 연산 차이일 수 있다. 이 가설은 아직 효용으로 검증하지 않았다.

## 다음2-2: teacher-head 재구성 대조 두 라벨

같은 고정 공개128·기존 DP1/DP2의 DINO head만으로 d를 복구한다. P-only target에서 s_P를 고정하고 μ'_0=s_P−d, μ'_1=s_P+d를 현재 class별 지도에 통과시킨다. pseudo 평균은 새 private query/DP target이 아니며, 이 대조에서 다시 clipping하지 않는다.

- **변경 하나:** MT가 사용한 보호 class 공통 평균 s를 공개 s_P로 대체. 지도·head ridge·공개 가중치·zero-anchor·bounds·eta 규칙은 유지.
- RN-only label2개만 새로 계산. Joint 추가 없음. 두 라벨을 고정한 뒤 DenseNet 주평가/RN 참고평가 readout4개. 기존 MT/KD/DINO-only20개 예측 재사용.
- 주 비교 MT RN-only−HR: 추가 s의 필요성. 보조 HR−KD RN-only, HR−DINO-only: 같은 head 정보의 공개 처리 효과와 단독 대안 대비 가치.
- 동일2000회 환자-cluster bootstrap, 영상 AUROC 주·AP 보조. ‘유의하지 않음’을 동등성으로 바꾸지 않음.
- 공개128·두 release·V에 조건부인 개발 ablation이다. 기존 KME/POST/KIP 전체 성능 비교나 새로운 원리의 검증으로 이름 붙이지 않음.
- s의 실질 추가 가치 후보는 두 release AUROC 점차이 양수, 평균≥0.01, 평균 조건부 CI 하한>0, 평균 AP≥0 및 release별 유의 AP 악화 없음일 때만 유지. 나머지는 미확인. 기존 투자 hurdle을 유지하며0.01은 임상 보편 기준이 아님.
- HR가 MT와 가까워도 정식 동등성을 주장하지 않음. HR−KD가 양성이면 현재 KD 비교의 범위를 더 좁힌다.
- 다음2-3 선정 반복은 MT 또는 HR가 DINO-only 대비 위 반복 투자 기준을 만족할 때만 진행한다. HR가 통과해도 이를 곧바로 제안법/신규기여로 바꾸지 않고 고정 비교의 재현으로만 다룬다. 둘 다 미충족이면2-3·독립 평가를 자동으로 돌리지 않고 기여 설계의 남은 문제를 기록한다.
- 새Q/noise/release/pixel/encoder forward/수신자 탐색 없음. 예상 **20~40분**, 연결·검산·기록 포함. 실제 수치 계산은 수초~수분이다.

2-1에서 모든 연산이 완전히 같은 것은 아니었으므로 이 **한 가지 정보 성분 대조**는 성립한다. 다만 표준 가중표현과 동치인 별도 baseline을 만들거나, 이름만 바꿔 ‘선행 우위’를 주장하지 않는다.

## 실행·보존 기록

핵심 코드: `contribution_search_20260924/public_transport_probe.py`, `validate_public_transport.py`, `fixed_image_compilation.py`; `contribution_kd_control_20260928_v1/prepare_kd_control.py`; `contribution_public128_control_20260928_v1/construct_labels.py`; `contribution_public128_source_20260928_v1/run_source_control.py`.

이번 공개 대수 검산: `contribution_operation_review_20260928_v1/audit_operations.py`, 결과 `operation_algebra.json`. 공개 P와 기존 DP 산출물7개만 사용했으며 수치 계산0.078초. 기존 target/maps와 동치 확인, 입력 불변 확인 통과. 새 라벨·학습·V 평가·private release는0. 전체 문헌·코드 분석/기록 실측은 같은 폴더의 `finalization.json`에 기록한다.

실측: 예상60~120분에 대해 분석 시작부터 기록까지 **13.28분**. 정식 선행4편의 관련 절, 현재 핵심 연산, 공개/기존DP 대수 검산으로 범위를 닫았다. 관련 논문 전체 재현이나 전수 문헌조사로 부르지 않는다.
