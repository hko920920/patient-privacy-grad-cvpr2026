# 방향 1의 연산 변경 후보: 동결 diffusion 특징의 환자별 잔차 통계 회귀

2026-09-16. 단계 2의 문헌 기반 설계 메모. 의료 학습·GPU·새 성능 실험은 수행하지 않았다. 편집 범위는 이 파일 하나이다. 기존 pilot/geometry v0의 수치 실패를 수정하려는 후속이 아니며, 보호 설계·효율이라는 큰 방향에서 학습 연산을 다시 선택한다.

## 1. 이번에 남기는 후보와 현재 판단

**공개 사전학습 diffusion 모델을 동결하고, 그 모델의 공간 특징으로 작은 시간 조건부 잔차 head만 학습하되, 환자별 회귀 통계를 한 번 보호하여 비공개 backbone 역전파를 없애는 방식**을 구체 후보로 남긴다.

비공개 데이터의 UNet backward를 제거하는 것은 아래 구조에서 실제로 성립한다. 같은 환자 DP 보장 아래 생성 품질까지 유지하거나 개선하는지는 아직 확인하지 않았다. 핵심 불확실성은 동결 특징의 적응 표현력과 잡음이 들어간 Gram 행렬의 안정성이다.

**한 번 보호한 통계를 반복 사용하는 것, 사적 ridge 회귀, 고정 특징의 denoising 회귀는 각각 이미 선행이 있다.** 따라서 이 메모의 후보를 지금 새로운 DP 원리나 완성된 논문 기여로 부르면 안 된다. 남은 설계 질문은 일반적인 noisy regression을 그대로 적용하는 데서 끝나지 않고, 사전학습 diffusion의 공간·시간 표현을 이용하여 **필요한 적응 능력과 보호 통계의 차원을 동시에 제한할 수 있는가**이다. 아래 명세는 그 질문을 논리적으로 판별할 수 있는 구체 연산이다.

## 2. 직접 확인한 선행과 이미 점유된 연산

| 선행·실제로 확인한 부분 | 해당 선행의 연산 | 현재 후보와의 관계 |
|---|---|---|
| [Mind the Privacy Unit](https://arxiv.org/html/2406.14322v3), Algorithm 2, §5, Appendix B.3 | user를 뽑고 여러 record gradient를 평균한 뒤 user 단위 clipping/noising. Record 선택·개수와 tight group accounting도 검토 | 환자 평균·환자 clipping·사진 개수 선택 자체는 기여가 아니다. 비교 보장은 user 수준으로 맞춰야 한다 |
| [DPDM 공식 설명](https://research.nvidia.com/labs/toronto-ai/DPDM/), Key Insights | DP-SGD에 noise multiplicity를 적용하고 강한 보호 조건의 높은 noise level 학습도 설명 | 반복 noise 또는 timestep 재배분만으로 새 기여를 주장하지 않는다. 이 메모는 반복을 조절하는 대신 비공개 backbone gradient를 제거한다 |
| [DP diffusion fine-tuning](https://arxiv.org/html/2406.01355v1), §3.1–3.2, §4.4; [ICCV 2025 공식 페이지](https://openaccess.thecvf.com/content/ICCV2025/html/Tsai_Differentially_Private_Fine-Tuning_of_Diffusion_Models_ICCV_2025_paper.html) | 공개 기반 모델에 작은 LoRA 보정과 DP-SGD/multiplicity를 적용. 읽은 상세 방법은 2024 v1이며 최종 논문 성능표와 동일하다고 가정하지 않음 | 단순 DP-LoRA/작은 trainable parameter는 이미 있다. 내부 LoRA 학습은 backbone을 통한 역전파가 필요하지만, 제안한 출력 head는 그렇지 않다 |
| [DOPE-SGD](https://proceedings.mlr.press/v202/nasr23a/nasr23a.pdf), PDF pp.2–3, §3.1 Algorithms 1–2, Propositions 2/4 | 공개 gradient 또는 과거 보호된 정보를 중심으로 사적 gradient 잔차를 clip하고 중심을 다시 더함. 효용 정리는 gradient 집중 조건을 둠 | 공개 gradient centering/residual clipping 자체를 재발견하지 않는다. 아래 residual은 gradient control variate가 아니라 출력 회귀의 정답이다 |
| [LA-LoRA](https://arxiv.org/html/2602.19926v1), §3.2 Eq.4, §4.1–4.2 | 두 LoRA factor 잡음의 교차항 문제를 분석하고 교대 factor 갱신·안정화를 사용 | 두 factor 결합을 피하는 것만으로 새 기여가 되지 않는다. 원 논문 FL 설정과 현재 중앙 user-DP도 구별해야 함 |
| [PRISM](https://arxiv.org/html/2606.00944v1), §3 Algorithm 1, §3.2 Theorems 3.3–3.4, §3.3 | low-rank update의 tangent 공간에서 clipping/noise를 정의하고 gauge 불변성과 후처리 적응을 다룸 | factor gauge·잡음 증폭·저차원 tangent 개선은 직접 선행이다. 현재 후보는 tangent DP의 새 버전이 아니다 |
| [DP-MERF](https://proceedings.mlr.press/v130/harder21a.html), 공식 초록·PDF 서론; [DP-MEPF](https://arxiv.org/abs/2205.12900), 저자 초록 | random/public perceptual 특징의 사적 평균을 한 번 보호하고 생성기의 MMD 목적에 재사용 | one-shot 통계 보호와 공개 특징 활용은 이미 있다. 이번 DP-MEPF 상세 PDF 접근은 실패했으므로 방법 전체를 새로 읽었다고 하지 않음 |
| [DP-NTK](https://backend.orbit.dtu.dk/ws/portalfiles/portal/386879759/DP_NTK_JAIR.pdf), PDF pp.5–6, §3 Algorithm 1 | 정규화된 parameter-gradient 특징 평균을 한 번 보호하고 생성기의 MMD 손실을 최적화 | one-shot 생성학습의 직접 비교 계열. 아래 방식은 parameter Jacobian 특징이 아니라 이미 계산되는 UNet activation과 조건부 denoising 목적을 사용 |
| [DP-KIP/ScatterNet](https://arxiv.org/pdf/2301.13389), PDF pp.5–6, Algorithm 1, §3.1–3.2 | 고정 kernel로 support dataset의 사적 gradient를 반복 계산·clip·noise하여 합성점을 학습 | 모든 kernel 계열을 one-shot으로 묶으면 틀린다. 이 방법은 반복 DP-SGD 경로이며 고정 특징의 계산 이점도 이미 강조 |
| [Improved DP Regression / BoostedAdaSSP](https://openreview.net/pdf/5dafc02af37718cf092472a357459a9500fabadb.pdf), §III Algorithm 1, Appendix B Algorithm 2 | AdaSSP는 XᵀX·Xᵀy 등에 잡음을 더하여 ridge를 풂. Boosted 버전은 noisy Gram을 재사용하지만 새 residual query에는 추가 보호 비용이 듦 | noisy 충분통계·역행렬·공개 bound 선택 문제는 직접 선행이다. 회귀를 denoising에 적용했다는 이유만으로 새 DP 메커니즘이 아님 |
| [Near-Optimal Private Linear Regression via Iterative Hessian Mixing](https://arxiv.org/abs/2601.07545v2), 2026-05-21 v2 저자 초록 확인 | Gaussian sketching 기반 IHM을 제안하고 AdaSSP보다 유리한 excess empirical risk 조건 및 비교를 보고 | 단순 SSP만 강한 최신 baseline이라고 부르지 않는다. 본 메모에서 IHM의 세부 연산·user 단위 adaptation은 아직 검토하지 않았음 |
| [Diffusion Random Feature Model](https://arxiv.org/html/2310.04417v2), §3 Algorithm 1, Eqs.20–24, §5 | 고정 random spatial feature와 trainable timestep weight/output weight로 diffusion을 구성 | 시간별 feature와 작은 회귀형 denoiser 발상은 이미 있다. 공개 UNet 잔차 head 또는 user-DP는 해당 방법이 아님 |
| [Where the Score Lives](https://proceedings.mlr.press/v300/finn26a.html), [저자 본문](https://arxiv.org/html/2606.08309v1), §2 Eqs.15–16, §3 Eqs.23–24, Appendix E | 고정 비선형 wavelet 특징으로 score/denoising을 ridge와 모멘트로 풂. 고해상도에서 회귀계 메모리 한계도 기록 | **denoising을 닫힌형 회귀로 바꾸는 것 자체도 선행**이다. 본 후보의 공간 공유 head는 출력 좌표·해상도에 따라 회귀 파라미터가 커지는 구성을 피하나, 이것의 성능 이점은 미검증 |
| [Differentially Private Random Feature Model](https://arxiv.org/html/2412.04785v1), §3.1 Algorithm 1 | min-norm random-feature 회귀 해에 output perturbation. 별도 데이터/특징 가정으로 민감도를 분석 | 고정 random feature+한 번의 보호라는 범주도 이미 있다. 그 가정을 의료자료 또는 user 단위에 자동 이식하지 않음 |

검색 범위는 위 방법·선택절 중심이다. 전체 최신 문헌을 망라하여 부재를 증명한 것이 아니다. 특히 고정 pretrained denoising head와 사적 충분통계를 정확히 결합한 모든 구현·미공개 연구의 부재를 주장하지 않는다.

## 3. 문제에서 연산까지

현재 DP-LoRA 계열은 파라미터 수를 줄여도 사적 image/noise별 UNet forward/backward를 반복한다. User 단위로 여러 사진을 평균하려면 그 비용이 사진과 noise 반복 수에 따라 증가한다. 반면 사전학습 모델이 이미 가진 denoising 표현을 작은 출력 보정으로 사용할 수 있다면, private backbone 갱신 없이도 일부 domain adaptation이 가능할 수 있다. 마지막 문장은 **가설**이다.

### 3.1 고정 공간·시간 특징과 출력 head

공개 고정 epsilon predictor를 `f0(z_t,t,c)`라 하자. 동일 forward에서 얻는 decoder activation을 `h_l(z_t,t,c,s)`라 한다. `s`는 latent 공간 위치이고 `c`는 조건이다.

1. 공개 자료만으로 정하거나 처음부터 고정한 선형 투영 `P_l`로 각 layer의 channel을 축소한다. 합계 channel 수를 `r`로 둔다. Layer resize와 channel 표준화도 고정한다.
2. 고정되고 norm이 제한된 timestep basis `b(t)∈R^K`를 사용한다. 예를 들어 log-SNR 구간의 4개 basis를 사용할 수 있으나, 이것을 지금 최적 설정이나 확정 실험 계약으로 정하지 않는다.
3. `phi(z_t,t,c,s) = b(t) ⊗ concat_l[P_l h_l(z_t,t,c,s)] ∈R^d`, `d=Kr`.
4. 공간에 공유되는 `W∈R^(d×o)` 하나로 `fW(s)=f0(s)+Wᵀ phi(s)`를 정의한다. SD latent의 예시는 출력 channel `o=4`이다. 별도 좌표별 W나 timestep별 독립 대형 W는 만들지 않는다.

`W=0`이면 공개 기반 모델을 정확히 복원한다. W에 관해서는 선형이지만 noisy input·조건·공간에 대해서는 동결 UNet 특징을 통해 비선형이다. 단순히 기존 LoRA의 1차 Taylor 근사를 쓴다는 가정은 필요 없다. 대신 LoRA보다 좁은 **새 함수족을 선택했다**는 표현력 대가가 있다.

phi는 생성 시점에도 얻을 수 있는 `z_t,t,c`의 함수여야 한다. Clean training image, 실제 주입한 epsilon, patient ID를 phi의 입력으로 사용하지 않는다. Epsilon은 학습 정답 e를 구성할 때만 사용한다. 학습용 조건과 생성용 조건의 정보 접근권 차이도 비교에서 숨기지 않는다.

공개 features를 이용한 선형 probing 및 공간 공유 연산 자체도 표준이다. 새로운 논문이 되려면 이 특정 잔차 표현이 낮은 보호 통계 차원에서 유용한 adaptation을 제공한다는 연결을 보여야 한다.

### 3.2 환자별 고정 회귀 통계

환자 `u`의 record를 같은 환자 내부에서 균등하게 선택하고, 선언된 원래 denoising objective의 `t,epsilon` 분포에서 정해진 횟수만 추출한다. 각 이미지의 공간 위치 평균까지 포함하여 다음을 만든다.

```text
e = epsilon - f0(z_t,t,c)
A_u = mean_(records,noise,positions) [phi phiᵀ]
B_u = mean_(records,noise,positions) [phi eᵀ]
A = (1/N0) sum_u A_u,    B = (1/N0) sum_u B_u
```

여러 공간 위치를 쓴다고 독립 환자 수가 늘어나는 것은 아니다. 환자당 기여는 평균으로 정하며, 사진이 많은 환자가 자동으로 더 큰 가중치를 갖지 않는다. User-mean 목표가 record-mean 목표와 다르다는 점도 명시한다. 모든 사진 대신 균등 표본을 쓰면 preclip 통계는 해당 환자 평균의 Monte Carlo 추정량이다. 고정·공개 random bank의 재사용 역시 finite objective를 선택하는 것이며 population expectation과 동일한 것은 아니다.

**epsilon−f0는 평균이 0인 control variate라는 주장이 아니다.** 미지의 target conditional mean에 대한 잔차 정답이다. 공개 모델을 사적 이미지에 적용한 값도 사적 통계이므로 그대로 외부 공개하지 않는다. Base가 v-prediction이면 정답과 출력 모두 해당 parameterization으로 맞추고, epsilon과 임의로 섞지 않는다.

### 3.3 반복 gradient 대신 한 번의 보호

공개 고정 척도 `a0,b0>0`와 clipping `C`를 사용한다. 대칭 행렬의 `svec`는 비대각 성분에 sqrt(2)를 곱하여 Frobenius norm을 보존한다.

```text
s_u = concat(svec(A_u)/a0, vec(B_u)/b0)
w_u = min(1, C/||s_u||2)
release = (sum_u w_u s_u + Normal(0, sigma² C² I))/N0
```

`a0,b0,C`, basis, layer, projection, normalization을 사적 통계에서 무료로 고르면 이 분석은 성립하지 않는다. 공개 고정값 또는 이미 적법하게 보호된 값만 사용한다. 환자별 통계가 다른 환자의 데이터에 의존하지 않도록 feature extractor와 전처리를 동결하며, batch normalization을 사적 batch로 새로 계산하지 않는다.

공개 고정 분모 `N0`에서 **환자 추가/제거 민감도는 C/N0**, 환자 교체에서는 상계가 **2C/N0**이다. One-shot Gaussian 회계로 sigma를 정한다. 이 주장은 새로운 정리가 아니라 bounded user contribution에 대한 표준 Gaussian 적용이다. 환자당 record 수와 무관한 보장은 마지막 joint clipping에서 나온다.

실제 환자 수가 private라면 그 수로 무보호 정규화하거나 count를 공개하지 않는다. 공개 registry 크기 같은 고정 분모를 쓰거나 DP count의 비용·수치 정책을 별도로 회계한다. User subsampling을 추가한다면 해당 샘플링을 실제 구현·회계해야 하며, 현재 one-shot q=1 구상에 증폭을 임의로 붙이지 않는다.

Decoder 출력에서 A/B를 복구하고, 대칭화·PSD projection·공개 ridge 또는 eigenvalue floor로 안정화한 뒤 W를 푼다. 이 모든 처리는 보호된 통계의 후처리이다. 새 private residual로 통계를 다시 계산하면 후처리가 아니며 추가 보호 비용이 생긴다. 사전학습 데이터에 이미 포함된 정보까지 이번 incremental user-DP가 지워 주지는 않는다.

## 4. 무엇이 정확하고 무엇이 가설인가

### 4.1 원 목적과 clipping 이후 목적을 구분

위 고정 head의 유한 표본 ridge 목적은 정확히 다음 이차식이다.

```text
L(W) = constant - 2 tr(WᵀB) + tr(WᵀAW) + lambda ||W||F²
W* = (A + lambda I)^(-1) B
```

회귀 충분통계로 풀어도 이 함수족·finite objective 안에서는 근사 최적화가 아니다. 그러나 timestep basis와 frozen 특징의 **함수 근사 오차**, 유한 noise bank의 **표본 오차**는 남는다.

여기서 제곱손실은 출력 channel 벡터의 norm을 사용한다. 구현이 channel 평균 MSE를 사용하면 공통 1/o 계수와 ridge lambda를 함께 맞춰야 같은 목적이라고 할 수 있다.

Joint clipping 후에는 `A_C=sum w_u A_u/N0`, `B_C=sum w_u B_u/N0`이다. 따라서 **한 환자에 같은 고정 weight를 적용한 재가중 이차 목적**을 푼다. 일반적으로 원 user-mean 목적에 불편하지 않다. Clip이 비활성인 경우에만 둘이 일치한다. A와 B를 따로 clipping하면 이 공통 가중 회귀 해석도 달라진다. Gaussian을 더한 뒤에는 noisy surrogate를 푸는 것이며, 이를 원 목적의 정확한 최적해로 부르지 않는다.

### 4.2 기대 개선이 가능한 조건

- 공개 denoiser가 갖춘 표현으로 목표 domain의 **조건부 잔차**를 작은 head가 충분히 설명할 수 있어야 한다. 큰 domain shift나 새로운 공간 구조의 학습은 동결 특징의 한계에 걸릴 수 있다.
- 환자 통계 joint clipping의 재가중 왜곡이 지나치지 않아야 한다. 희귀 환자·큰 residual을 가진 환자만 강하게 잘릴 가능성을 별도 확인해야 한다.
- 필요한 feature dimension이 작고 Gram이 안정적이어야 한다. 공개 whitening은 공개 분포의 조건수만 보장하며 target domain의 조건수를 보장하지 않는다.
- 한 번 만든 finite noise bank가 필요한 시간 범위의 조건부 잔차를 충분히 대표해야 한다. 강한 DP에서 high-noise 쪽을 더 쓰는 것은 이미 선행 선택이며, 목적을 바꾸는 경우 명시해야 한다.

가령 `H=A_C+lambda I`, `lambda_min(H)=kappa>0`이고 최종 안정화 후 유효 오차를 `E_A,E_B`라 하자. `||E_A||op≤kappa/2`이면 표준 역행렬 섭동 항등식으로

```text
W_hat-W_C = (H+E_A)^(-1) (E_B-E_A W_C)
||W_hat-W_C||F <= (2/kappa) (||E_B||F + ||E_A||op ||W_C||F)
```

이다. PSD projection의 영향을 포함하려면 E_A를 **실제로 후처리된 행렬의 오차**로 정의해야 한다. 원 Gaussian만의 norm과 같은 것으로 간주하지 않는다. 이 식은 noisy Gram/작은 eigenvalue/큰 residual head가 왜 불리한지 보여 주는 표준 설명이며 새 정리나 생성품질 보장이 아니다. Held-out denoising 개선도 누적 sampling 품질이나 FID 향상과 동일하지 않다.

### 4.3 시간·공간 차원을 줄이는 이유와 대가

공간 공유 head이면 보호 좌표 수는 `d(d+1)/2 + d o`로 latent 해상도에 직접 비례하지 않는다. `r=16,K=4,d=64,o=4`라는 설명용 구성은 **2,336좌표**, `d=128`이면 **8,768좌표**이다. 이는 측정된 최적 차원이 아니다.

d=64 head 자체는 256개 파라미터이므로 보호 통계 2,336좌표가 오히려 더 많다. ‘LoRA보다 작다’와 ‘동일 head의 DP-SGD보다 보호 잡음 차원이 작다’는 다른 명제이며 후자는 여기서 성립하지 않는다. One-shot 회계와 곡률 정보의 이점이 이 대가를 상쇄하는지 비교해야 한다.

해상도가 커져도 데이터 처리·UNet forward와 공간 통계 누적은 비싸진다. 통계 차원이 고정이라는 말은 연산·메모리가 모두 해상도와 무관하다는 말이 아니다. 작은 d는 통계 잡음과 inverse 비용을 줄이지만 channel/시간의 잔차 표현을 잃는다. K를 늘리면 d가 선형, Gram의 좌표 수는 대략 제곱으로 증가한다. Separate head를 시간마다 크게 두어 이 문제를 숨기지 않는다.

## 5. 비용: 보장되는 제거와 측정하지 않은 시간

사용자 N, 선택 사진 G, 사진당 noise K_n, 한 forward/backward 비용 c_F/c_B, 공간 위치 수 S라 하자. 통계 추출은 대략

```text
C_one_shot = N G K_n c_F
           + O(N G K_n S (d²+d o))
           + O(d³) + VAE/공개 특징 준비 비용
```

이다. A/B는 streaming 누적하므로 private 전체 activation을 영구 저장할 필요가 없다. 작은 W solve/후처리는 보호 예산을 더 쓰지 않는다. Inference는 기존 f0 forward와 작은 출력 head를 사용한다.

T번의 user-DP-LoRA, sampling q, noise 반복 R의 단순 비교식은 `T q N G R (c_F+c_B)`이다. 통계 계산 등 부대비용을 무시하면 one-shot이 유리한 필요 비교는 `K_n < T q R (1+c_B/c_F)` 형태이다. **실제 wall-time 손익 조건으로 확정한 식은 아니다.** d² 누적, kernel 효율, 저장 비용, 공개 준비 비용까지 측정해야 한다. 같은 epsilon에서 q=1 한 번과 q<1 반복의 noise·효용 순서는 자동으로 정해지지 않는다.

중요한 강한 비교는 **같은 동결 특징·같은 작은 head를 user-DP-SGD로 학습하는 방법**이다. 이 방법도 backbone backward가 필요 없고 같은 private feature bank를 내부에 캐시할 수 있다. 따라서 backbone backward 제거 효과는 full LoRA 비교에서만, one-shot 통계의 효과는 same-head DP-SGD 비교에서 따로 입증해야 한다. 캐시를 만들었다고 그 캐시를 반복 질의하는 DP-SGD의 회계가 사라지는 것은 아니다.

같은 캐시를 쓰는 작은 head SGD는 반복 시 UNet을 다시 부르지 않고 O(d o)의 head 연산만 수행할 수 있다. 반면 full Gram 누적은 O(d²)이다. 따라서 one-shot 방식이 이 강한 baseline보다 wall-time까지 반드시 유리한 것은 아니다.

## 6. 직접 비교와 최소 판별 계획 — 아직 실행하지 않음

| 비교 | 해소할 대안 설명 |
|---|---|
| 공개 기반 모델 그대로 W=0 | target adaptation 자체가 필요한지, 보정이 모델을 손상시키는지 |
| 동일 head의 비보호 ridge와 동일 head의 비보호 SGD | 닫힌형 구현 정합성과 함수족의 capacity ceiling |
| 동일 head의 user-DP-SGD + 같은 private feature bank | 이득이 작은 head 자체인지, one-shot 통계 때문인지 |
| 같은 feature의 AdaSSP/강한 DP ridge 변형 | 약한 noisy inverse를 세워 이긴 것은 아닌지 |
| IHM 등 최신 사적 회귀 후보의 적합성 검토 | SSP보다 나은 알려진 회귀 연산을 빠뜨린 것은 아닌지. 같은 user 보장에 실제 적용 가능한지부터 확인 |
| full user-DP-LoRA + 충분한 multiplicity/조건별 설정 | 표현력 손실을 감수한 비용·품질 tradeoff가 실제로 유용한지 |
| 적절한 ELS + tight user accounting | user clipping 한 가지 설정만 비교해서 얻은 결과인지 |
| DP-MEPF/NTK 등 | 최종 주장이 광범위한 DP 생성의 성능 우월까지 포함할 때의 가까운 생성 비교. 접근권·공개 모델·총비용을 맞춤 |

최소 검증은 다른 baseline을 모두 실패시키는 관문이 아니다. 이 설계의 핵심 연결을 차례로 확인한다.

1. **공개 또는 명시적으로 연구에 이용 가능한 작은 domain의 capacity 확인.** Fixed head에서 finite quadratic loss와 통계 해가 일치하는지, 비보호 작은 head가 residual을 학습할 수 있는지 본다. 목표 domain을 전혀 담지 못하면 DP 잡음 조절에 시간을 쓰기 전에 표현을 재검토할 근거가 된다.
2. **표현과 보호 알고리즘을 분리.** 같은 특징·같은 user objective에서 same-head DP-SGD와 one-shot 통계의 clip 왜곡·noisy Gram·비용을 비교한다. 다른 주장을 하려는 독립 설계에 자동 실행 조건을 부여하지 않는다.
3. **기여 범위 판별.** 작은 head가 단순 최종 layer 선형 회귀와 차이 없이 동작하면 표준 조합으로 보고한다. 반대로 공간·시간 residual representation이 낮은 d에서 adaptation을 유지하며 equal-user-DP/compute 비교를 개선하면 그 구조와 작동 조건을 논문의 중심으로 좁힌다.

layer/rank/time basis/scales/lambda의 선택은 공개 개발자료 또는 비용을 회계한 보호된 자료에서만 한다. 사적 selection score를 보고 공짜로 선택하지 않는다. 성능 실험에서는 noise seed와 환자 split도 구분해야 한다. 현재 의료 캐시와 과거 non-DP LoRA checkpoint를 곧바로 이 후보의 검증 결과로 재해석하지 않는다.

## 7. 탈락시킨 좁은 대안과 남은 여지

**공개 gradient를 빼고 남은 gradient만 clip한다**는 안은 DOPE-SGD와 직접 겹친다. **LoRA 한 factor를 동결하거나 gauge를 맞춰 잡음 결합을 줄인다**는 안도 FFA/LA-LoRA/PRISM 근처의 직접 선행을 넘어서는 연산을 이 메모에서 만들지 못했다. 이를 새 방향1 기여로 내세우지 않는다.

현재 후보 역시 “공개 특징 + user 평균 + noisy ridge”까지만 구현하면 표준 조합에 가깝다. 다만 막힌 연결이 명확하다. **저차원 공간 공유 잔차 표현이 충분한 생성 적응을 담는가, 그리고 그 차원에서 user clipping과 noisy Gram을 감당할 수 있는가**이다. 이 연결을 검토할 명세가 생겼다는 것이 이번 완료물이며, 새로운 방법의 우월성·의료 효능·논문 가능성이 확인됐다는 뜻은 아니다.

선형 head의 capacity가 부족할 때 작은 비선형 head+user-DP-SGD는 구현 가능한 대안이지만, 이는 one-shot 충분통계의 이점을 잃고 일반적인 frozen-feature PEFT에 가까워진다. 따라서 현재의 별도 신규성 후보로 포장하지 않고 비교·대체 구현으로만 둔다.

## 8. 읽은 범위와 현재 미확인 항목

- 위 URL의 선택된 method/algorithm/theorem/limitations 부분을 직접 확인했다. DP-MEPF는 이번에 저자 초록만 확인했으며 상세 PDF 접근 실패를 보존한다. DP diffusion fine-tuning은 2024 v1 방법과 2025 공식 게재 정보를 구분했다.
- AISTATS 2026 wavelet ridge가 가장 직접적인 추가 차단선이었다. 별도 DP random-feature 회귀의 존재도 확인했다. 이는 제안의 단순 부품을 새것으로 주장하지 않기 위한 확인이다.
- 기존 v0 CPU 배분 실패는 [완료 보고서](../TWO_TRACK_DEEPER_CHECK_20260916.md)의 범위 그대로 보존한다. 이 후보의 성능 증거로 쓰지 않았다.
- 이번에는 실험·학습·code 실행 구현을 하지 않았다. 실제 FP32/FP16 메모리, feature tap API, 처리 시간, 표본 수, clipping 빈도, 생성품질, 최신 선행 전수 신규성은 미확인이다.
- 다음 결정을 위한 가장 작은 질문은 **공개 기반 모델의 좁은 출력 잔차 함수족에서 의미 있는 adaptation이 가능한가**이다. 이는 제안된 검증 질문이지 자동으로 새 GPU 학습을 시작하라는 지시가 아니다.
