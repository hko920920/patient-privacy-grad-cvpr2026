# 2-1 보완(2-1R1): 공개 기준값과 ridge 전달 연산의 분리

**결론: HR의 KD 대비 이득을 새로운 사적 정보 때문이라고 설명할 수 없다. HR은 공개 기준값과 같은 DINO head의 변화량을 전달하는 affine 연산이다. 이 구조와 가장 가까운 표준적인 보정을 기존 KD에도 허용한 대조가 필요하다. 이번 산출물은 그 대조의 수식·규모·판정을 고정한 설계이며, 새 기여나 새 성능 결과가 아니다.**

기존 2-1/2-2 완료 상태와 2-3 보류를 유지한다. 이번 보완을 과거 실험의 재명명으로 사용하지 않는다. 사용자 연속 진행 지시에 따른 상위2의 설계 보완이다.

## 이번에 확인한 연산

D와 R은 P의 DINO64 / ResNet18 128 특징, Ω는 기존 class 균등·환자 균등·방문 평균 가중이다.

```text
G = D' Ω D
H_D = G + 0.1 I
H_R = R' Ω R + 0.1 I
C_RD = R' Ω D
d = (mu_1 - mu_0) / 2
w_D = H_D^(-1) d
```

공개 class별 clipped 평균을 mu_Pc, d_P=(mu_P1-mu_P0)/2, w_DP=H_D^(-1)d_P로 쓴다. 기존 class별 공개 지도 T_c, h_c는 환자/class 행을 각각 C_c로 제한하고 중심화한 ridge 지도다.

```text
T_average = (T_1 + T_0)/2
w_RP_map = H_R^(-1) [(T_1' mu_P1 + h_1 - T_0' mu_P0 - h_0)/2]
A_MT = H_R^(-1) T_average' H_D
w_R_HR = w_RP_map + A_MT (w_D - w_DP)
```

이 식은 2-2 HR 목표와 최대 3.61e-16 오차로 일치한다. HR−KD를 공개 기준값 차이와 변화량 전달 연산 차이로 정확히 나눌 수 있다.

```text
A_KD = H_R^(-1) C_RD
w_R_KD = A_KD w_D
w_R_HR - w_R_KD
  = (w_RP_map - A_KD w_DP)
  + (A_MT - A_KD)(w_D - w_DP)
```

두 항의 공개 예측 RMS는 단순히 더해지지 않는다. 교차항이 음수다. 이 분해는 효용의 인과적 기여 비율을 뜻하지 않는다.

| 공개 예측 공간 진단 | DP1 | DP2 |
|---|---:|---:|
| HR−KD RMS | 0.076899 | 0.068566 |
| 기준값 보정 항 RMS | 0.073030 | 0.073030 |
| 전달 연산 변경 항 RMS | 0.088832 | 0.062063 |

G의 고유값 범위는 [0.00010181, 0.38163921]이다. 같은 source 표현에서 source-score를 다시 회귀하는 연산의 필터 g/(g+0.1)는 [0.001017, 0.792376]이다. 공개 공분산의 작은 ridge λ=0.001 trace(G)/64=3.0478093e-5를 적용하면 g/(g+λ)는 [0.769603, 0.999920]이다.

이것은 공개 방향별 전달 차이가 클 수 있다는 대수적 근거다. 서로 다른 모델 사이의 AUROC가 반드시 좋아진다는 보장이 아니며, 작은 ridge가 기존 DP 잡음을 증폭할 가능성도 있다. w_D와 공개 H_D에서 d는 정확히 복구된다. **일반 KD가 head 안의 정보를 영구적으로 잃었다는 설명은 하지 않는다.**

## 직접 선행과 남는 질문

| 선행 | 공식 상태 / 직접 겹침 | 현재 설정에 곧바로 적용할 수 없는 부분 |
|---|---|---|
| [POST](https://proceedings.mlr.press/v267/wang25ds.html), §4.3 Eq.3–5 | ICML2025. 공개자료에서 절대 예측뿐 아니라 source의 기준 모델 대비 변화도 전달한다. | LLM soft prompt / KL 목적의 전체 알고리즘이다. 아래 선형 residual-KD를 POST 재현이라고 부르지 않는다. |
| [Self-Distillation Amplifies Regularization in Hilbert Space](https://proceedings.neurips.cc/paper/2020/hash/2288f691b58edecadcc9a8691762b4fd-Abstract.html) | NeurIPS2020. 반복 증류가 함수공간 정규화를 강화하는 현상을 분석한다. | 동일 모델의 self-distillation 정리다. 현재 이종 모델·DP·bounded carrier 효용에 대한 정리가 아니다. |
| [Conditional Mean Embeddings as Regressors](https://icml.cc/Conferences/2012/papers/898.pdf), §3 Eq.9–11 | ICML2012. 조건부 평균 전달과 vector-valued ridge regression의 동치를 보인다. | 현재 P→Q 조건부 관계의 보존이나 의료 효용은 보장하지 않는다. 공개 inverse-covariance 전달 자체는 새 원리가 아니다. |
| [Fantastic Gains and Where to Find Them](https://www.eml-munich.de/publication/Fantastic-Gains-and-Where-to-Find-Them) | ICLR2024. 단순 KD가 수신 모델의 기존 지식을 덮어쓸 수 있음을 분석하고 보존·선택적 전달을 다룬다. | ImageNet full-network transfer이고 현재 patient-DP frozen-head/128-label 설정과 다르다. |
| KME / KIP Label Solve | 2-1에서 확인한 ICML2018 / ICLR2021 선행. 공개 가중표현·고정 영상의 라벨 최적화와 겹친다. | 기존 연산 조합과 현재 의료 사용 조건의 추가 가치를 별도로 확인해야 한다. |

POST 본문은 [저자 공개 PDF](https://adam-dziedzic.com/static/assets/papers/post.pdf)로, 정규화 논문 본문은 [공식 PDF](https://proceedings.neurips.cc/paper/2020/file/2288f691b58edecadcc9a8691762b4fd-Paper.pdf)로 확인했다. 이번 표의 네 편은 단순 미심사 arXiv 후보가 아니라 정식 학회 논문이다.

**이번에 겨냥할 질문은 하나다.** 현재 head-only 전달의 KD 대비 이득이, KD에 공개 모델 기준값 보존과 공개 공분산 보정을 허용해도 남는가? 표준 보정으로 설명될 수 있는 이득을 독자적 기여로 세지 않기 위한 대조다. 이 대조가 실패한다고 전체 연구가 불가능해지는 것도, 통과한다고 CVPR 기여가 완성되는 것도 아니다.

## 다음 2-2 보완(2-2R1)의 고정 설계

기존 KD 한 칸을 재사용하고 2×2의 나머지 세 칸만 만든다.

```text
lambda_C = 0.001 trace(G)/64
A_C = H_R^(-1) C_RD (G + lambda_C I)^(-1) H_D
w_RP_real = H_R^(-1) R' Ω (2y_P - 1)

KD   = A_KD w_D                                      [기존]
AKD  = w_RP_real + A_KD (w_D - w_DP)                 [새]
CKD  = A_C w_D                                       [새]
ACKD = w_RP_real + A_C (w_D - w_DP)                  [새]
```

- AKD/ACKD에는 P의 실제 RN 지도학습 head를 기준값으로 허용한다. HR의 추정된 공개 기준값을 강제로 쓰게 하지 않는다.
- source 기준 w_DP는 동일 protected signal schema에 맞춘 P-only clipped class 목표를 사용한다. 실제 RN 기준값은 P의 class/patient 가중 supervised ridge다. 서로 다른 encoder에서 같은 숫자의 head를 요구하지 않는다.
- λ는 기존 공개 지도와 같은 trace-scaled 규칙을 쓴다. V를 보며 선택하거나 ridge sweep하지 않는다.
- HR와 새 방식은 공개 지도 구성·기준값도 다르므로 전체 연산 대조다. 이 결과만으로 HR 내부의 한 요소가 유일한 원인이라고 주장하지 않는다.
- KD와 AKD 사이, CKD와 ACKD 사이의 기준값 보존 효과는 각각 같은 전달 연산 아래의 대조다. 두 연산 아래 효과가 같다고 가정하지 않는다.
- 128장 선정·기존 DP1/DP2·RN-only bounded label solver·zero anchor·readout·2000회 환자 bootstrap은 유지한다.
- 새 라벨 3종×2release=6개, DenseNet/RN readout12개. 기존24개는 그대로 재사용한다.
- 모든 라벨을 봉인한 다음에만 V 평가한다. 새 Q/DP summary/영상학습/모델 forward/새 receiver/Expert/Reserved는 없다.
- 다음 실행 전 예상 **20~35분**, 전체 상한60분. 핵심 수치 계산은 캐시 행렬로 수행하므로 GPU bank 제작 시간과 다르다.

### 판정

주 비교는 HR−ACKD, 보조 비교는 HR−AKD 및 HR−CKD, 각 보정−KD와 2×2의 조건부 효과다. 모든 결과를 함께 기록한다. 결과가 좋은 대안 하나만 사후 선택해 선행 전체를 대표하게 하지 않는다.

HR의 추가 차별 효용을 다음 반복 투자 근거로 세려면, 세 새 대안 각각에 대해 기존 개발 hurdle(두release AUROC 점차이 양수, 평균≥0.01, 평균조건부CI하한>0, 평균AP≥0, 개별release AP 유의 악화 없음)을 만족해야 한다. 이는 실용적 개발 규칙이며 equivalence/임상기준/최종 검정이 아니다.

반대로 새 표준 보정이 잘되면 그 성과는 알려진 보정의 유용성으로 기록한다. 새 이름을 붙여 기여로 전환하지 않는다. 신뢰구간이0을 포함하는 것만으로 “기존방법과 같음” 또는 “원인이 완전히 설명됨”이라고 결론내리지 않는다.

**원래 2-3의 source-only 두release 개선 gate는 유지한다.** 새 보정의 성능을 보고 과거 gate를 통과했다고 고쳐 쓰지 않는다. 다음 반복 또는 새로운 방법 연구는 이번 결과가 남긴 구체적인 한계에 근거해 별도로 설계한다.

## 이번 실행 범위와 검산

P 캐시와 이미 존재하는 DINO head/공개 지도/HR 목표만 읽었다. 새 DP class 평균이나 V 예측은 읽지 않았다. 새 labels/효용 평가0이다. 대수 일치 최대오차1.09e-14. 입력 hash와 기존 보고서 hash를 보존했다.

상세 수치·접근 기록: `contribution_operation_followup_20260928_v1/public_algebra.json`, `algebra_inputs.json`, `algebra_access.jsonl`.

보조 dtype 확인의 최초 shell 호출에서 Python stdin 표시(-)를 누락해 실행되지 않았고 즉시 수정했다. 실제 대수 계산이나 결과를 변경한 실패는 아니다.


실행 전 예상40~60분, 실제 문헌·대수·설계·기록 **12.96분**. 대수 계산 0.812초.
