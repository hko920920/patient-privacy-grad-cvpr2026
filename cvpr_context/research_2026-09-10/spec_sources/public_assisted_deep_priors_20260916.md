# 공개 보조 최적화 선행과 현재 환자 보호 생성 적응의 비교

2026-09-16. 최신 `../REALISTIC_RESEARCH_PLAN_20260916.md`를 기준으로 한 원문 확인 메모다. 새 실험·학습·후보 선택을 수행하지 않았다. 아래 PDF 쪽수는 파일의 첫 페이지를 1쪽으로 센다.

**기존 공개 보조 원리가 있다는 사실은 새 solver라는 주장을 제한한다. 그 원리를 이용한 의료 생성 적응 구성과 실제 품질–전체 비용 기여까지 이미 해결됐다는 뜻은 아니다.** 앞선 `track1_replan_prior_review_20260916.md`의 “기존 해법으로 설명되면 중단”은 불필요한 새 residual solver 발명·우월성 주장에 한정해서 읽어야 한다. 기존 solver가 잘 작동하면 채택할 수 있으며, 응용·통합 기여는 별도로 판단한다.

## 1. 논문 정체와 직접 확인 범위

| 논문 | 공식 원문 | 이번에 확인한 부분 |
|---|---|---|
| Ehsan Amid 등, **Public Data-Assisted Mirror Descent for Private Model Training**, ICML 2022, PMLR 162:517–535. 방법명 PDA-DPMD | [공식 논문 페이지](https://proceedings.mlr.press/v162/amid22a.html), [PDF](https://proceedings.mlr.press/v162/amid22a/amid22a.pdf) | §§1.1–1.2, §3 Algorithm 1, §4 Assumption 4.1/Theorems 4.2–4.3, §5.1–5.3 및 Table 1. PDF pp.3–9. App.C.1의 근사 유도 시작 부분도 확인했으나 부록 전체 증명 재검토는 아님 |
| Zhiqi Bu, Xinwei Zhang, Sheng Zha, Mingyi Hong, George Karypis, **Pre-training Differentially Private Models with Limited Public Data**, NeurIPS 2024 | [공식 논문 페이지](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ac04e54e0a2d1927d60709019e4e7870-Abstract-Conference.html), [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/ac04e54e0a2d1927d60709019e4e7870-Paper-Conference.pdf) | §§1.3–1.5, §§2–4, §§5–6, Tables 2–8, App.D Algorithm 1. PDF pp.3–10,26. 원문 증명 부록 전체 및 저자 코드 실행은 하지 않음 |

NeurIPS 공식 PDF는 네트워크에서 메모리로 읽어 확인했다. 확인한 바이트의 SHA256은 `245f8df60d4dd4789d2217cf00ef9659c8e80614271e748f72137fad543baabf`이다. 원본 연구 파일은 변경하지 않았다.

## 2. 실제 연산과 결과: 확인한 사실

**PDA-DPMD.** 공개 손실을 mirror map \(\Psi\)로 쓰고, 사적 gradient를 clipping·잡음 처리한 뒤 Bregman update를 한다. Algorithm 1은 공개 최적해로 초기화한다. 선형 회귀에서는 이 update가 공개 Hessian의 역행렬을 noisy gradient에 적용하는 형태가 된다. 이론적 효용 우위에는 공개/사적 분포 및 공개 표본 수 등 Assumption 4.1의 조건이 있다. 딥러닝 실험은 정확한 내부 최소화 대신 Eq.(1)의 공개 gradient와 보호된 사적 gradient의 조합을 쓴다. §5.2는 이미 **사용자 단위** StackOverflow/DP-FedAvg 실험이며, 공개 초기화 기준선과도 비교한다. Table 1에서 warm DP-FedAvg 대비 accuracy 21.20→21.77, perplexity 85.03→80.51을 보고한다. 이것은 우리의 의료 생성 결과가 아니다. [Algorithm 1, §4, §5 및 Table 1](https://proceedings.mlr.press/v162/amid22a/amid22a.pdf)

**Bu 등.** 이 논문은 noisy Hessian을 공개 Hessian으로 교체하는 회귀 solver가 아니다. Eq.(4)–(8)은 2차 국소 근사와 clipping 방향 편향이 작다는 근사를 사용해 DP 잡음에 따른 최적화 지연을 분석한다. §4.2의 실제 전략은 공개 학습 후 DP continual pretraining으로 전환하는 방식이며, 전환 시점은 공개 통계로 정할 수 있다. §5에서는 공개 DINO 초기화 후 DP linear probing 10 epochs와 전체 파라미터 학습 20 epochs를 수행한다. §5.1의 ImageNet-11k upstream accuracy는 ε=8에서 41.5%다. Table 7의 Places365 55.7%와 iNat2021 60.0%는 **비DP downstream 학습** 결과다. 개인정보 정의는 §1.5의 한 sample 차이이며, 환자 묶음 보장을 입증한 실험으로 읽지 않는다. [§§1.5,2–5, App.D](https://proceedings.neurips.cc/paper_files/paper/2024/file/ac04e54e0a2d1927d60709019e4e7870-Paper-Conference.pdf)

## 3. 세 계층에서 비교 역할을 분리

다음 표의 “현재 적용”은 원문 실험 결과가 아니라 위 연산과 우리 고정 설계 사이의 분석이다.

| 계층 | 선행이 이미 제공하는 것 | 현재 연구에서의 공정한 역할 | 이 확인만으로 해결됐다고 할 수 없는 것 |
|---|---|---|---|
| ① 공개 보조 최적화 | PDA-DPMD의 공개 geometry 및 공개/사적 gradient 결합. Bu 등의 공개 초기화 후 보호 학습 | 같은 head·같은 목적·같은 공개 정보에서 public-only, warm-start DP-SGD, 공개 정보까지 계속 쓰는 방법을 구분. 필요하면 PDA-DPMD 원리에 따른 작은 head baseline을 적용 | 공개 곡률이나 residual 중심화를 새 DP 원리로 주장하는 것은 곤란. 반대로 모든 기존 solver를 먼저 구현·실패시켜야 현재 구성을 제안할 수 있다는 조건도 없음 |
| ② DP 생성 적응 | 이 두 논문은 공개 자료가 DP 최적화에 도움 되는 근거를 제공. 직접적인 생성 적응 비교 계열은 별도의 [DP diffusion fine-tuning](https://arxiv.org/html/2406.01355v1) | 현재 head가 denoiser 내부 고정 표현과 시간 정보를 이용해 생성 중 출력을 보정하는 구성, backbone 역전파 제거, 표현력–품질–전체 비용을 검증 | 분류/언어 성능으로 의료 영상의 생성 효용을 보장하지 않음. 작은 head의 MSE 개선으로 DP-LoRA보다 유용한 생성 적응이라고 결론 내리지 않음 |
| ③ 환자 보호 단위 | PDA-DPMD는 user-level 실험도 제공하므로 “공개 보조+사용자 단위” 자체는 선행 영역. Bu 등의 원문은 sample 단위 | 환자 내부 영상·noise·공간을 먼저 집계한 손실/gradient를 한 기여로 제한하고, 환자 add/remove 인접성·고정 분모·회계를 맞춤 | image 수를 환자 수로 바꾸는 명칭 변경만으로 기여가 되지 않음. 반면 기존 user-level 원리가 존재한다는 이유로 의료 생성의 구현·효용·비용 문제를 자동 해결된 것으로 보지도 않음 |

DP diffusion 논문 링크는 앞선 메모에서 읽은 2024 v1을 가리킨다. 이번 메모에서 최종 학회판 전체를 새로 읽었다는 뜻은 아니다.

## 4. 우리 고정 quadratic head에서 가능한 정확한 연결

우리 cached 목적을 \(L(W)=Q-2\langle B,W\rangle+\operatorname{tr}(W^T A W)+\lambda\|W\|_F^2\)로 두면,

\[
\nabla L(W)=2(AW-B)+2\lambda W,
\qquad H=2(A+\lambda I)
\]

이다. 출력 채널별 블록에 같은 Hessian이 적용된다. **고정 특징·고정 표본에서는 H가 W에 의존하지 않는다.** 따라서 공개 초기화로 현재 head의 Hessian trace 자체가 작아진다는 설명은 맞지 않는다. 공개 초기화는 시작 오차와 gradient/clipping 경로를 바꿀 수 있고, 공개 Hessian을 사용하는 mirror geometry는 update와 잡음 전달 방향을 바꿀 수 있다. 이 두 효과를 분리해야 한다. Bu 등의 깊은 모델에서 관찰된 학습 중 곡률 변화는 이 고정 회귀에 그대로 옮길 수 없다.

PDA-DPMD 원리를 사용할 경우 공개 환자 통계로 \(\Psi(W)\)를 만들고 안정화한 공개 Hessian으로 **이미 보호한** gradient update를 변환하는 구현이 가능하다. 이는 기존 원리의 적용이며, 우리의 환자 손실·clipping·Poisson sampling·공개 분모에 맞는 DP 계약이 별도로 필요하다. 기존 원문의 수치적 잡음 크기를 그대로 복사하거나, preconditioning 뒤에도 원래 clipping 민감도가 자동 유지된다고 가정해서는 안 된다. 공개 안정화 계수 역시 사적 dev 결과로 고르면 공개 설정이라는 근거가 사라진다.

또한 private-only 손실을 공개 자료로 초기화하는 것과 공개+사적 pooled 목적을 최적화하는 것은 다르다. 현재 계획의 pooled 비DP 비교를 보호 방법으로 확장한다면 공개/사적 목적 가중치와 ridge까지 동일하게 맞춘 기준선이 필요하다. 원 PDA-DPMD의 시간별 가중치까지 무조건 그대로 사용해야 한다는 뜻은 아니다.

## 5. 현재 계획에 미치는 결론

새 보고서가 “이 solver를 그대로 쓰면 solver 자체의 기여는 약하다”고 판단하는 것은 타당하다. 이를 “그러므로 전체 연구도 약하다/불가능하다”로 확대하면 근거를 벗어난다. 실제로 표준 solver가 가장 잘 작동한다면 그 solver로 현재 생성 적응 구성을 평가하는 것이 합리적이다.

남는 후보는 **공개 diffusion의 고정 공간·시간 표현을 작은 출력 보정층으로 연결하여, 환자 DP 아래에서 사적 backbone 역전파 없이 의미 있는 의료 생성 적응을 제공하는가**이다. 남는 가치는 생성 효용과 총비용에서 보여야 하며, 공개 전용 적응이 이미 얻는 이득을 구별해야 한다. 이 두 선행의 존재는 비교를 강화하는 근거이며, 새로운 DP 수학을 연구의 필수조건으로 만들지 않는다.

따라서 최신 계획의 pooled 비DP 대조와 생성 wrapper 검증을 유지한다. 이번 선행 확인 때문에 새 solver 탐색이나 추가 대규모 학습을 자동으로 시작하지 않는다. 기존 해법이 해결해 준 부분은 채택·인용하고, 현재 응용 구성의 남는 가치를 별도로 검증한다. 공개 곡률 mismatch를 인위적으로 만들어 새 문제처럼 제시하지 않는다.
