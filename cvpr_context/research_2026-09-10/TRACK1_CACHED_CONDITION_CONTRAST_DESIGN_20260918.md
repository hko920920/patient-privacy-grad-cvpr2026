# 환자별 조건 대조를 캐시에서 학습하는 생성 적응 설계

작성: 2026-09-18. 상태: **구체적 제안 설계**, 채택된 실행 명세나 효용 결과가 아니다.
사용자의 지적은 타당하다. 직전의 “효용을 유지하며 비용을 줄인다”는 표현은 연구 목표였고, 그 목표를 이루는 연산을 정하지 못했다. 이 문서는 진단 목록을 추가하지 않고 한 가지 설계를 명시한다. 현재 큰 단계는 2다.

## 1. 주장할 대상을 먼저 정한다

제안: **고정 공개 diffusion의 조건별 denoising energy를 영상별 작은 이차식으로 저장하고, 그 이차식의 차이를 사용하는 조건 대조 학습을 환자 단위 DP로 수행한다.**

검증하려는 기여는 다음이다.

> 질환 조건을 구별하는 생성 적응을 공개 backbone의 반복 역전파 없이 수행하고, 환자 DP 아래 합성자료의 추가 downstream 효용과 적응 비용의 절충을 개선한다.

새 DP 정리, “대조 손실 최초 제안”, 단순 parameter 수 감소를 기여로 선언하지 않는다. 조건 대조·캐시·환자 보호의 연결이 실제 효용을 유지하는지가 중심이다. 효용이나 비용 우위가 이미 확인됐다는 뜻은 아니다.

## 2. 현재 결과와 연결되는 이유

- 현재 head는 실제 sampling에 반영되지만 평균 MSE/KID 개선이 사적 downstream 추가 효용으로 이어지지 않았다.
- 기존 LoRA 대조도 주로 올바른 영상–조건 쌍의 noise MSE를 줄이는 적응이었다. 다른 조건과의 구별을 직접 목적에 넣지는 않았다.
- 양 조건의 오차가 같은 만큼 낮아지면 평균 denoising은 개선돼도 조건 간 차이는 그대로일 수 있다. 이는 목적함수의 성질이며, 우리 실패의 주원인이 확인됐다는 뜻은 아니다.
- Rwide의 양성 결과는 더 넓은 실자료의 직접 사용이 도움이 됐음을 보여준다. 이 설계의 성공 근거나 새 공개 학습자료로 전환하지 않는다.
- 모든 원인을 먼저 분리해야 설계를 제안할 수 있는 것은 아니다. 여기서는 “상대 조건을 직접 학습하는가”라는 구체적인 변경을 선택한다.

## 3. 실제 변경 연산

### 3.1 같은 이미지에 두 조건

각 학습 영상 i에 기흉 라벨 y_i와 반대 조건 1-y_i를 적용한다. 두 조건은 같은 noisy latent, timestep, noise draw를 공유한다. 음성 조건은 “기흉 없음”이며 “모든 질환이 없는 정상”이 아니다. NIH 음성/양성은 현재 약한 학습 라벨이며 전문의 확정 라벨로 승격하지 않는다.

공개 E4, feature projection, timestep basis, CFG는 고정한다. 기존 conditional-only 출력 적용 형식을 사용한다.

\[
 f_{i,c}(W)=g_{0,i,c}+X_{i,c}W,\qquad
 g_0=\epsilon_u+7.5(\epsilon_c-\epsilon_u),\quad X=7.5\phi_c .
\]

따라서 생성 시 7.5 배율을 학습에서 빠뜨리는 변경이 아니다. Null branch는 고정한다. 초기 W=0은 E4 자체이며, 새 실험의 MSE-only 대조와 조건 대조 모델은 같은 초기화에서 출발해야 한다.

### 3.2 평균 오차와 조건 구별을 함께 학습

고정 noise/timestep bank에 대한 영상별·조건별 평균 오차를 D_{i,c}(W)라 하자. 한 영상의 목적은

\[
 \ell_i(W)=D_{i,y_i}(W)
 +\beta\tau\log\left[1+\exp\left(
 \frac{D_{i,y_i}(W)-D_{i,1-y_i}(W)+m}{\tau}
 \right)\right].
\]

정답 조건의 복원 오차를 낮추면서, 반대 조건보다 상대적으로 더 잘 설명하도록 요구한다. 무제한으로 음의 MSE를 보상하는 목적과 달리 대조항은 0 아래로 내려가지 않는다. 공개 영상에서 E4 출력과의 거리를 제한하는 기능적 anchor 및 작은 ridge를 더한다.

\[
 R_{\rm pub}(W)=\mathbb E_{\rm public,c}\|X_{i,c}W\|^2,\qquad
 \lambda R_{\rm pub}(W)+\lambda_W\|W\|_F^2 .
\]

Anchor는 private 결과로 선택한 방향이 아니며 공개자료에만 의존한다. beta/tau/m/lambda 및 학습량은 아직 성능을 보고 고른 값이 아니다. 완성된 실행 protocol처럼 수치가 동결됐다고 쓰지 않는다.

중요한 제한: 이 loss도 틀린 조건을 나쁘게 만들어 margin을 높일 수 있다. 올바른 조건의 생성 효용은 별도로 판정해야 한다. CFG energy는 임상 확률이나 정확한 log-likelihood가 아니며, 이것을 새로운 자동 생성 평가기로 채택하지 않는다.

### 3.3 비선형 조건 대조 loss도 작은 통계에서 계산

고정 backbone + 선형 출력 W이면 각 영상·조건에서

\[
 D_{i,c}(W)=Q_{i,c}-2\langle B_{i,c},W\rangle+
 \operatorname{tr}(W^\top A_{i,c}W),
\qquad
 \nabla_W D_{i,c}=2(A_{i,c}W-B_{i,c}).
\]

A=X^T X, B=X^T r, Q=r^T r에 해당하며 모두 같은 평균 정규화를 포함한다. **이 식의 softplus 조합도 저장한 A/B/Q만으로 값과 gradient를 정확히 구할 수 있다.** 정확성은 고정된 유한 noise bank와 이 출력 head 안에서의 등가성이다. UNet 전체를 학습하는 것과 동등하다는 뜻은 아니다.

현재 full64라면 W는 64×4다. 전체 UNet 활성화를 매 update 보관하거나 역전파하지 않고, 작은 이차식에 대해서만 반복 학습한다. Feature extraction의 새 forward 비용은 여전히 필요하다. 기존 통계는 정답 조건/환자 평균 중심이므로 그대로 재배열하는 것만으로 새 목적이 만들어지지 않는다. 반대 조건의 새 forward와 **영상별** 통계가 필요하다.

Softplus를 적용하기 전에 여러 영상을 환자 A/B/Q 하나로 합치면 다른 목적이 된다. 먼저 영상별 loss를 계산한 다음 환자별 평균을 낸다. 새로운 noise bank를 사용할 때는 새 추출 비용도 포함한다.

## 4. 환자 DP 연결과 범위

\[
 \ell_u(W)=|I_u|^{-1}\sum_{i\in I_u}\ell_i(W).
\]

한 환자의 모든 영상에서 계산한 gradient를 하나로 평균하고 patient clipping 및 Gaussian noise를 적용한다. 환자 sampler와 정규화는 사전에 고정하며, 비보호 private 양성 수나 gradient norm으로 선택하지 않는다. 공개 anchor gradient는 공개 항으로 별도 처리한다.

대조의 상대는 다른 환자의 영상이 아니라 같은 영상에 적용하는 고정 텍스트 조건이다. 따라서 한 환자의 변경이 다른 환자의 negative-example 선택까지 바꾸는 배치 구조를 만들지 않는다. 이는 표준 환자 DP-SGD를 적용하기 쉬운 구성이지 새로운 privacy theorem은 아니다. 실제 구현에는 add/remove adjacency, 고정 normalizer, sampling 및 composition의 기존 accountant 검증이 필요하다.

- 원영상과 A/B/Q cache는 모두 보호 경계 안의 사적 자료다. Cache 자체를 공개하지 않는다.
- Cache를 재사용해도 각 DP update의 privacy 비용이 사라지지 않는다. One-shot SSP가 아니다.
- 초기 public model/feature/anchor는 private 학습에 의존하지 않아야 한다.
- Private 결과를 이용한 hyperparameter/model 선택과 여러 출력 공개도 privacy 처리에 포함돼야 한다. 이전 비DP 실험이 소급해 보호되지는 않는다.
- 초기 generic SSP의 불안정한 noisy Gram inversion을 그대로 재사용하지 않는다. 그렇다고 일반 DP-SGD보다 낫다고 미리 주장하지 않는다.

## 5. 가장 가까운 선행과 차이

| 선행 | 이미 한 것 | 이 설계에서 확인할 차이 |
|---|---|---|
| Diffusion Classifier, ICCV 2023 | 조건별 denoising 오차로 분류 | 오차 비교 자체는 선행. 이를 생성 출력 head의 학습 목적에 연결 |
| HardNeg-DiffusionITM, NeurIPS 2023 | 맞는/틀린 영상–텍스트 조건을 사용한 LoRA 조정 | 대조 학습 자체도 선행. 고정 출력 head의 영상별 이차 energy로 학습을 캐시하고 환자 DP에 연결 |
| DPT, CVPR 2024 | discriminative adapter와 LoRA로 생성 정합성 조정 | DPT의 tuning은 backbone 경로의 활성화가 바뀐다. 여기서는 고정 backbone과 작은 출력만으로 대조 학습이 유용한지 검증 |
| DP-LoRA, ICCV 2025 | diffusion LoRA의 DP fine-tuning | 동일 자료·환자 보호에서 조건 대조 효용과 반복 backbone 연산/전체 비용을 비교 |
| 기존 프로젝트 cached DP-SGD | 같은 head의 MSE를 A/B로 학습 | 캐시 자체도 프로젝트 내 기존 기능. 새 부분은 영상별 두 조건 energy를 유지해 비선형 조건 목적까지 계산하는 연결 |

“기존 연구가 전부 이 문제를 못 풀었다”는 주장이 아니다. 문헌 전체의 신규성 부재/존재를 증명한 것도 아니다. 이 표는 확인한 가까운 선행과 설계 차이이며, 논문 가치는 실증된 효용–비용에서 결정된다.

주요 원문:
- https://arxiv.org/abs/2303.16203
- https://arxiv.org/html/2305.16397v3
- https://arxiv.org/html/2403.04321v2
- https://openaccess.thecvf.com/content/ICCV2025/html/Tsai_Differentially_Private_Fine-Tuning_of_Diffusion_Models_ICCV_2025_paper.html

## 6. 논문을 결정할 비교는 하나의 목적에 묶는다

아래는 실험을 지금 승인/실행했다는 뜻이 아니라 이 설계가 답해야 할 비교 구조다.

1. **같은 출력 구조에서 MSE-only vs 조건 대조:** 자료, 두 조건, noise bank, 초기화와 sampling을 같게 두고 beta=0 vs beta>0를 비교한다. 기존 과거 MSE 결과는 역사적 참조다. 새 prompt/관측 bank가 다르면 같은 조건의 ablation인 척 재사용하지 않는다.
2. **같은 조건 대조 목적에서 캐시 head vs LoRA:** 약한 원래 MSE-LoRA만 기준으로 두지 않는다. LoRA에도 같은 정보/목적을 허용해 표현 제한의 손해와 비용 절감을 함께 본다.
3. **각 구성 안의 public-only vs public+private:** 적응 자체의 효과와 사적 자료의 추가 효용을 나눈다. 원래 public32/64 및 private80/320 접근을 유지하며 Rwide의 추가2027명을 몰래 넣지 않는다.
4. **환자 DP의 matched utility–cost:** 사적 추가 효용이 실제 생성자료에서 관측될 때 그 손실과 전체 비용을 동일 epsilon/delta 및 환자 단위로 비교한다. “DP 연산이 가볍다”만으로 논문 성공이 아니다.

주 결과는 synthetic → 고정 real-development downstream 효용이다. 내부 condition margin이나 denoising loss만 좋아지면 이 설계의 전달 주장을 지지하지 않는다. 현재 좁은 public-real classifier의 일반화 한계는 이 설계가 해결했다고 가정하지 않는다.

비용은 추출+학습+생성 전체와 adaptation만을 둘 다 표시한다. 반복 학습 구간에서는

\[
 C_{\rm pair\ cache}+T C_{\rm small\ update}
 \quad\text{vs}\quad
 T C_{\rm LoRA\ forward/backward}
\]

를 비교하지만, cache 추출이 비싸거나 생성이 총비용을 지배하면 전체 이득이 작을 수 있다. 캐시를 쓸 수 있는 same-head DP-SGD에는 가상의 UNet 반복 비용을 부과하지 않는다. Private Evolution처럼 학습 역전파를 요구하지 않는 방법도 있으므로 “backbone backward 없음” 자체를 독점 기여로 쓰지 않는다.

실패하면 이 구체적 연결을 지지하지 못한 것으로 판단한다. Margin만 증가, public-only에 못 미침, 비용 우위 없음 등을 다른 진단의 자동 실행 근거로 바꿔 무기한 연장하지 않는다. 그것이 모든 후속 연구를 금지하거나 문제 전체를 반증한다는 뜻도 아니다.

## 7. 이번에 실제로 한 일

문헌과 기존 프로토콜을 읽고 위 연산을 도출했다. 임의 CPU FP64 tensor에서 직접 잔차 계산과 이차식 계산의 energy/loss/gradient를 비교했다. 최대 energy 차이 2.22e-16, loss 차이 0, gradient 차이 6.94e-17이었다. 이는 단순 대수 구현 확인이며 의료 성능·privacy 실행 검증이 아니다.

환자영상, 실제 checkpoint, GPU, 새 생성, DP 학습 실행은 모두 0이다. 기존 실패/원자료역할과 expert532·reserved4213을 보존했다. 마지막 실제 모델 결과는 Rwide이고 마지막 사적 생성 효용 결과는 LoRA 음성이다. 이 제안을 이미 사용자 승인된 방법, 구현 완료, 실행 예정으로 바꾸지 않는다.

