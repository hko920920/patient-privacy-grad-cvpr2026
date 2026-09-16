# Patient-Coverage Multiplicity 접근법과 독립 논문 실험계획

- 기록일: 2026-09-02
- 문서 성격: 연구 설계안. 동결 실험계약 변경이나 training 시작 승인이 아님
- working method name: Patient-Coverage Multiplicity, PCM
- working paper title:
  From Image Privacy to Patient Privacy: Coverage-Aware Multiplicity for Private Medical Diffusion

## 1. 한 문장 접근법

> 한 환자의 복수 영상을 frozen visual feature로 coverage strata로 나누고, 제한된
> gradient 계산을 서로 다른 영상과 diffusion noise/timestep 반복에 배분해 하나의
> unbiased patient gradient를 만든 뒤 환자 단위로 한 번 clipping/noising한다. 그 결과를
> 기존 image-DP checkpoint의 환자 단위 conversion과 동일 target patient privacy에서
> 비교하여 언제 재사용할 수 있고 언제 patient-DP 재학습이 필요한지 판정한다.

## 2. 왜 이 형태로 좁혔는가

단순 환자 평균, patient clipping, hierarchical averaging, user-wise DP-SGD, diffusion
noise multiplicity는 모두 선행이 있다. 이 논문의 기술적 질문은 다음 하나로 좁힌다.

> patient-wise DP diffusion에서 환자당 Q회의 gradient evaluation만 허용될 때, Q를 한
> 이미지의 여러 diffusion perturbation에 쓰는 것과 여러 patient-linked image의 visual
> coverage에 쓰는 것 중 무엇이 clipping 전 estimator의 품질과 최종 생성 utility를 더
> 개선하는가?

의료영상에서는 환자 내부 영상이 서로 독립적인 동일분포 샘플이 아니다. 현재 core인
chest X-ray에는 시간에 따른 finding·support device·노출·자세 변화가 있고, AP와 PA를
함께 쓰면 view 차이까지 섞인다. cross-domain ISIC에는 서로 다른 lesion과 anatomical
site가 섞인다. 따라서 단순 uniform record sampling보다 patient의 시각적 범위를 덜
빠뜨리는 estimator가 필요하다는 것이 vision-specific 가설이다.

## 3. 정확한 문제 정의

환자 u의 데이터는 다음과 같다.

$$
D_u = \{(x_{u,j}, y_{u,j})\}_{j=1}^{n_u}, \qquad 1 \le n_u \le K.
$$

전체 데이터는 환자 집합 D = {D_u}이고, add/remove patient adjacency를 사용한다. 즉,
인접 데이터셋은 한 환자의 모든 영상과 metadata가 통째로 추가 또는 제거된 경우다.

주 목적함수는 이미지 수가 많은 환자가 자동으로 더 큰 가중치를 받지 않는
patient-balanced diffusion objective다.

$$
\mathcal{L}(\theta)
=
\frac{1}{N}\sum_{u=1}^{N}
\frac{1}{n_u}\sum_{j=1}^{n_u}
\mathbb{E}_{t,\xi}
\left[
\ell(\theta; x_{u,j}, y_{u,j}, t, \xi)
\right].
$$

- theta는 고정된 public latent-diffusion base의 LoRA parameter다.
- t는 diffusion timestep, xi는 forward noise다.
- K는 공개되고 사전에 동결한 환자 contribution cap이다.
- patient-balanced objective와 image-balanced objective는 서로 다른 estimand이므로
  결과에서 혼용하지 않는다.

## 4. 검토한 방법 후보와 선택

| 후보 | 설명 | 장점 | 핵심 약점 | 결정 |
|---|---|---|---|---|
| A. naive patient ULS | 환자당 임의 영상 일부 평균 후 patient clip | 가장 안전한 표준 baseline | 방법 신규성이 없음 | 필수 baseline |
| B. uniform nested multiplicity | Q 안에서 서로 다른 영상을 먼저 쓰고 남은 횟수를 noise repetition에 사용 | 단순하고 unbiased, variance 분석 가능 | ULS와 DPDM의 조합으로 보일 위험 | 강한 baseline·fallback |
| C. coverage-stratified nested multiplicity | frozen visual space에서 환자 영상을 strata로 나누고 각 stratum을 대표하도록 unbiased sampling | 의료영상의 visual structure를 직접 사용, 방법 기여가 가장 선명 | embedding과 strata가 gradient variance를 실제로 줄여야 함 | 주 방법 후보 |
| D. adaptive private-gradient allocation | private gradient variance를 보며 m/r을 온라인 조절 | 잠재 utility가 큼 | 추가 privacy accounting과 leakage 위험 | 제외 |
| E. two-stage image/patient clipping | image clip 후 patient clip | 안정화 가능 | 다수 선행과 clipping bias, 기여가 흐려짐 | 보조 ablation만 고려 |

최종 선택은 C다. 단, C가 public-development diagnostic에서 B보다 실제 variance를 줄이지
못하면 C를 주 방법으로 승격하지 않는다. 이 경우 B까지는 구현하되 CVPR method paper를
강행하지 않는다.

## 5. PCM의 구체적 알고리즘

### 5.1 비공개 학습 전 고정하는 것

1. base model, exact revision, LoRA target module과 rank
2. 환자 cap K와 patient-first split
3. 환자당 gradient evaluation budget Q
4. visual feature extractor와 deterministic clustering rule
5. patient sampling probability, steps, clipping norm C, optimizer
6. accountant implementation·version·adjacency convention
7. timestep distribution

core X-ray의 exact-image interface gate를 통과한 뒤 기존 exact SD 2.1 VAE latent를 우선
feature로 사용한다. 별도 foundation encoder를 쓰면 pretraining contamination과 추가
dependency가 생기므로, radiology encoder나 DINO 계열은 사전 등록된 보조 ablation으로만
둔다. 이미 확보한 ISIC은 implementation smoke와 cross-domain confirmation에 사용할 수
있지만, X-ray 결과를 본 뒤 유리한 encoder를 고르는 통로로 사용하지 않는다.

### 5.2 환자 내부 coverage strata 만들기

환자 u의 각 영상에서 frozen feature h(x)를 구한다. Q보다 영상이 많으면 farthest-first
k-center로 m개의 anchor를 정하고 각 영상을 가장 가까운 anchor에 할당해 strata
C_{u,1}, ..., C_{u,m}을 만든다.

$$
m_u = \min(n_u, m_{\max}, Q).
$$

각 학습 step에서 stratum c마다 영상 하나를 균등하게 뽑는다.

$$
J_{u,c} \sim \mathrm{Uniform}(C_{u,c}), \qquad
w_{u,c} = \frac{|C_{u,c}|}{n_u}.
$$

그 뒤 stratum별 diffusion perturbation을 r_{u,c}회 뽑으며, 총 계산량은 환자마다
항상 Q가 되게 한다.

$$
\sum_{c=1}^{m_u} r_{u,c} = Q, \qquad r_{u,c} \ge 1.
$$

초기 구현은 각 stratum에 한 번씩 먼저 배정하고 남은 횟수를 round-robin으로 배정한다.
private gradient를 보고 r을 바꾸지 않는다. runtime이 n_u를 드러내지 않도록 모든 sampled
patient에서 정확히 Q회 forward/backward를 수행한다.

### 5.3 patient gradient

$$
g_u
=
\sum_{c=1}^{m_u}
w_{u,c}
\left[
\frac{1}{r_{u,c}}
\sum_{s=1}^{r_{u,c}}
\nabla_\theta
\ell(\theta; x_{u,J_{u,c}}, y_{u,J_{u,c}},
t_{u,c,s}, \xi_{u,c,s})
\right].
$$

각 stratum에서 uniform sampling하고 stratum size로 가중하므로, clipping 전 g_u는 환자의
전체 image-average diffusion gradient에 대한 unbiased estimator다. 단, 이후 norm clipping은
일반 DP-SGD와 같이 bias를 만든다.

### 5.4 patient clipping과 Gaussian noise

매 step에서 환자를 Poisson probability q_p로 표집한다.

$$
\bar g_u = g_u \min\left(1,\frac{C}{\|g_u\|_2}\right).
$$

$$
\tilde g
=
\frac{1}{q_pN}
\left[
\sum_{u\in U_t}\bar g_u
+
\mathcal{N}(0,\sigma^2 C^2 I)
\right].
$$

optimizer에는 tilde g만 전달한다. patient gradient norm, cluster assignment, sampled image,
raw per-patient loss는 release artifact에 포함하지 않는다. public proxy에서 연구 진단용으로
관측하더라도 그것이 confidential deployment에서 자동으로 공개 가능한 log라는 뜻은 아니다.

### 5.5 privacy statement의 정확한 범위

- 보호 단위는 add/remove patient다.
- 환자 내부 clustering, image selection, timestep/noise repetition은 최종 환자 vector를
  C로 clip하기 전 내부 계산이다.
- 따라서 Q, m, r이 커져도 한 step의 patient sensitivity bound는 C로 유지된다.
- privacy cost는 patient Poisson sampling, Gaussian noise multiplier, step 수와 accountant에
  의해 계산한다.
- 이 사실은 표준 subsampled Gaussian mechanism의 적용이며 새 privacy theorem으로
  주장하지 않는다.
- replacement adjacency를 쓰면 sensitivity convention이 달라질 수 있으므로 논문·코드·
  accountant에서 add/remove를 일치시킨다.
- model 이외의 data-dependent log나 checkpoint selection을 공개하면 별도 privacy 비용이
  생길 수 있다. confirmatory run은 공개 development 기준으로 checkpoint를 선택하거나
  final iterate만 공개한다.

## 6. variance 분석

먼저 uniform distinct-image sampling의 원리를 명시한다. 한 환자의 이미지 j에 대해

$$
\mu_{u,j}=\mathbb{E}_{t,\xi}[G_{u,j,t,\xi}], \qquad
\mu_u=\frac{1}{n_u}\sum_j \mu_{u,j}
$$

로 두고,

$$
S_u^2
=
\frac{1}{n_u-1}
\sum_j \|\mu_{u,j}-\mu_u\|_2^2
$$

를 환자 내부 image-gradient heterogeneity, V_u를 동일 이미지의 diffusion
noise/timestep에 의한 평균 conditional variance라고 둔다. m개 이미지를 without replacement로
뽑고 각 이미지에서 r개 perturbation을 평균할 때, clipping 전 estimator의 squared-error
trace는 독립성과 동일 conditional variance라는 단순화 아래

$$
\mathbb{E}\|\hat g_u-\mu_u\|_2^2
\approx
\left(1-\frac{m}{n_u}\right)\frac{S_u^2}{m}
+
\frac{V_u}{mr}.
$$

고정 계산량 Q = mr에서는 두 번째 항이 대략 V_u/Q로 유지되지만 첫 번째 항은 m이
커질수록 감소한다. 따라서 환자 내부 영상이 실제로 이질적이면, 한 영상을 Q번 반복하기보다
서로 다른 영상을 먼저 보는 것이 유리하다는 가설이 나온다.

PCM에서는 visual strata별 within-stratum variance가 작을수록 uniform distinct sampling보다
추가 이득을 얻는다. 이 이득은 embedding이 gradient geometry를 잘 반영할 때만 성립하며
보편적 theorem으로 주장하지 않는다. 논문에서는 다음을 분리한다.

1. unbiasedness: 수식으로 보일 수 있음
2. privacy: patient clipping과 표준 accountant로 보일 수 있음
3. variance 감소: 조건부 분석과 public diagnostic으로 검증
4. 최종 utility 개선: DP 학습 실험으로만 검증

## 7. 연구 질문과 사전 가설

### RQ1. privacy unit

이미지 단위 DP checkpoint의 generic/tight group conversion은 어느 K와 budget에서
실용적인 patient claim을 허용하는가?

- H1: tight conversion은 generic conversion보다 reuse 가능 영역을 넓히지만 모든 K와
  target budget에서 native patient-DP를 대체하지는 못한다.
- H1b: image-DP conversion의 환자별 bound는 해당 환자의 capped record count k_u에 따라
  달라질 수 있지만, cohort 전체에 동일한 patient claim을 하려면 worst-case K를 써야 한다.
  native patient-DP는 record count와 무관한 동일 unit guarantee를 제공한다. 평균 k로
  worst-case claim을 완화하지 않는다.

### RQ2. fixed-compute estimator

동일 Q, q_p, steps, noise, C에서 distinct-image/coverage multiplicity가 DPDM식
noise-only multiplicity보다 patient gradient 품질을 높이는가?

- H2a: within-patient heterogeneity가 큰 환자에서 uniform distinct-first가 noise-only보다
  clipping 전 estimator error와 clipping distortion을 줄인다.
- H2b: PCM은 uniform distinct-first보다 같은 Q에서 더 낮은 estimator error를 보인다.

### RQ3. 생성 utility

variance 개선이 실제 medical generation과 synthetic-trained downstream utility로 이어지는가?

- H3: PCM은 동일 patient epsilon/delta와 계산량에서 naive ULS 및 noise-only multiplicity보다
  preregistered primary medical utility metric을 개선한다.

### RQ4. empirical risk

aggregate image-level metric이 환자별 반복기록·희귀조건의 위험을 가리는가?

- H4: patient-aggregated membership/copy score의 상위 tail과 record-count/rare-condition
  strata는 aggregate image AUC만으로 설명되지 않는다.

H1부터 H4 중 방법 논문을 지탱하는 핵심은 H2와 H3다. H1만 성립하면 audit paper이고,
H4만 성립하면 risk-audit paper다.

## 8. 비교군

| ID | training | privacy unit | Q 사용 | 목적 |
|---|---|---|---|---|
| B0 | untouched base | 없음 | 해당 없음 | base contamination·domain gap |
| M0 | non-DP LoRA | 없음 | matched | utility ceiling과 memorization reference |
| M1 | image-DP LoRA | image | DPDM noise multiplicity 포함 가능 | 기존 checkpoint route |
| M1-G | M1의 generic group conversion | patient claim은 변환 | 재학습 없음 | 보수적 reuse |
| M1-T | M1의 DP-SGD-specific tight conversion | patient claim은 변환 | 재학습 없음 | 강화된 reuse |
| M2-U1 | patient ULS | patient | Q=1 | native patient-DP 최소 baseline |
| M2-NM | patient ULS + noise-only multiplicity | patient | 한 image에 Q perturbations | DPDM 결합 baseline |
| M2-UD | uniform distinct-first | patient | distinct image 우선 | nested multiplicity baseline |
| M2-PCM | coverage-stratified multiplicity | patient | visual strata 우선 | 제안 방법 |

기존 계약의 A0/A1/A2 no-noise mechanism controls도 유지한다. M2 계열의 방법 ablation에서는
q_p, steps, sigma, C, Q, optimizer, base, LoRA와 patient split을 동일하게 둔다.

M1과 M2는 sampler가 다르므로 “모든 조건이 동일하다”고 쓰지 않는다. primary fairness는
동일 target patient epsilon/delta와 동일 base/data cap이며, gradient-evaluation 수,
wall-clock, energy와 VRAM을 별도로 보고한다.

최근 image-level synthesis SOTA인 PrivImage, dp-promise, SPTI와 RPGen도 related work와
결과표의 context에 포함한다. 다만 public pretraining data, architecture와 mechanism이
서로 달라 published FID를 patient-DP 결과와 직접 같은 열에서 우열로 해석하지 않는다.
직접 실행 baseline은 다음 조건을 모두 만족할 때 한 대표 방법을 추가한다.

1. 공식 code와 privacy statement를 재현할 수 있음
2. X-ray의 동일 private split과 허용된 public data policy를 사용함
3. image-level guarantee를 동일 patient target으로 정당하게 변환할 수 있음
4. 현재 compute budget에서 최소 3 seed 또는 사전 허용한 축소 검증이 가능함

현재 공통 pretrained LDM+LoRA 구조와 가장 잘 맞는 직접 image-DP baseline은 DP-LoRA형
M1이다. SPTI처럼 training-free textual intermediary인 방법은 별도 family이며 PCM의
patient-gradient ablation을 대체하지 않는다.

## 9. privacy accounting 계획

### M1 image-DP route

1. 실제 image sampler와 일치하는 example-level accountant로 epsilon_image를 재계산한다.
2. 공개 cap K와 완전한 patient-image mapping을 확인한다.
3. generic group bound M1-G를 계산한다.
4. sampler 조건이 맞을 때 mixture-of-Gaussians/PLD 기반 tight group accountant M1-T를
   독립 구현 두 개로 교차 검증한다.
5. k_u별 converted bound profile은 불균등 보호의 diagnostic으로 보고하되, 전체 patient
   release claim은 공개 cap K의 worst-case bound로 판단한다.
6. target patient release budget을 넘으면 claim을 낮추거나 M2 retraining으로 보낸다.

### M2 direct patient-DP route

1. Poisson patient sampling probability q_p를 runtime trace와 일치시킨다.
2. 한 환자의 최종 contribution이 C 이하인지 unit test와 trace로 확인한다.
3. sigma, steps, delta_patient로 epsilon_patient를 독립 accountant 두 개에서 계산한다.
4. expected cohort normalization과 privacy accounting을 구분한다.
5. research PRNG run은 RESEARCH_ONLY로, formal confirmatory run은 secure noise path와
   model digest를 요구한다.

target epsilon/delta 값은 현재 동결 계약의 미해결 gate다. 방법 문서에서 임의로 확정하지
않는다. 후보 grid를 accountant·utility feasibility로 줄인 뒤, 원하는 ordering을 보기 전에
공통 patient target을 preregister한다.

## 10. 데이터셋 전략

### 10.1 최신 승인 hierarchy

1. primary/core: frontal chest X-ray, 첫 intake 후보는 NIH ChestXray14
2. cross-domain extension: 이미 동결·확보한 SIIM–ISIC 2020 dermoscopy
3. 금지: X-ray와 dermoscopy를 한 unconditioned generator에 혼합
4. optional third dataset: MURA 또는 다른 patient-linked radiograph dataset

이 hierarchy는 code_working/XRAY_CORE_PIVOT_DECISION.md의 승인 결정을 따른다.

### 10.2 NIH ChestXray14 core 후보

현재 상태는 exact-source/manifests/interface gate 전이며, full image corpus는 아직 받지
않았다. 예비 metadata 집계는 다음과 같다.

| scope | images | patients | 2장 이상 | 5장 이상 | 10장 이상 | 최대 |
|---|---:|---:|---:|---:|---:|---:|
| 전체 frontal | 112,120 | 30,805 | 13,302 | 5,759 | 2,545 | 184 |
| PA only | 67,310 | 28,868 | 10,688 | 3,245 | 953 | 100 |

preferred first option은 PA-only다. AP/PA acquisition shortcut을 줄이면서도 반복 영상
환자가 10,688명이라 PCM을 시험할 절대 표본은 충분하다. AP+PA를 선택하려면 view를
명시적 condition으로 넣고 모든 arm에서 view distribution을 맞춘다.

중요한 현실적 특징은 PA-only 환자 중 18,180명이 한 장만 가진다는 점이다. 따라서:

- 주 cohort는 single-record와 repeated-record patient를 모두 포함한다.
- n_u=1이면 PCM은 자동으로 m_u=1인 표준 patient update가 된다.
- method effect는 n_u=1, 2–4, 5–9, 10+ strata로 보고한다.
- repeated-patient-only training은 population을 바꾸므로 주 분석으로 쓰지 않고,
  사전 승인된 mechanism sensitivity가 아니면 수행하지 않는다.
- contribution cap K in {2, 5, 10}은 최대 기여를 제한하며 minimum record 수가 아니다.

공식 train/validation list 86,524 images/28,008 patients와 test list
25,596 images/2,797 patients는 image·patient overlap이 없다. train/validation patient에서
development, attack holdout, private train을 다시 patient-first로 나누고 official test
patient는 final evaluation으로 보존한다.

finding 조건은 radiographic finding으로만 부른다. preliminary PA-only 후보는
pneumothorax, pneumonia-or-consolidation, pleural effusion, mass-or-nodule와
No Finding reference다. Mass/Nodule은 암 확진이 아니고 No Finding은 판독의가 확정한
완전 정상 label이 아니다. multi-label 정책은 outcome을 보기 전에 동결한다.

exact main metadata mirror가 official file과 3 bytes 다르며 full official release가 약
42 GiB다. 현재 C: 여유 59.01 GiB에서 archive와 extract를 동시에 보관하면 안전하지 않다.
exact provenance, selective acquisition, 저장공간 계획이 통과되기 전에는 다운로드나
training을 시작하지 않는다.

### 10.3 ISIC cross-domain extension

- K5: private train 1,415 patients / 6,512 images
- K10: private train 1,415 patients / 10,848 images
- K2: privacy-unit sensitivity
- patient-first frozen split, exact manifest와 확보한 9,495 K5 images 재사용
- 별도 dermoscopy generator로만 학습

K5에서 1,203/1,415 patients가 4장 이상이고, K10에서 849/1,415 patients가 8장 이상이다.
K10에서는 전원이 복수 lesion이고 1,300명이 복수 anatomical site를 가진다. 따라서
X-ray의 many-singleton long tail과 ISIC의 dense repeated-record 구조가 PCM의 서로 다른
검증 조건이 된다.

ISIC은 구현 smoke를 앞당길 수 있지만 paper hierarchy에서는 extension이다. X-ray pilot이
원하는 결과를 주지 않았다는 이유로 조용히 core로 되돌리지 않는다. 전환은 failed/null
X-ray pilot을 남긴 투명한 amendment로만 가능하다.

### 10.4 optional third dataset

MURA는 공식 자료 기준 12,173 patients, 14,863 studies, 40,561 multi-view radiographs를
가져 좋은 radiograph replication 후보다. 그러나 현재 한 편의 최소 목표는 NIH core와
ISIC extension이다. MURA는 첫 두 조건의 효과와 compute가 확인된 뒤에만 access,
license와 storage gate를 연다.

공식 자료:
https://stanfordmlgroup.github.io/competitions/mura/

## 11. 단계별 실험

### Stage 0. mechanism·accountant gate

- toy patient dataset으로 add/remove adjacency unit test
- Q와 n_u에 무관하게 exactly Q gradient evaluations인지 확인
- cluster sample과 가중치가 full patient mean에 unbiased인지 Monte Carlo test
- patient gradient norm이 clip 후 C 이하인지 확인
- zero-noise일 때 수동 gradient와 일치하는지 확인
- 두 accountant의 epsilon 재계산 일치
- sampler/accountant mismatch negative tests

이 단계가 실패하면 실제 이미지 training을 시작하지 않는다.

### Stage 1. public-development estimator diagnostic

privacy noise 없이 frozen model gradient를 측정한다.

- Q in {1, 2, 4}; 자원이 허용되면 8은 진단만
- 비교: U1, NM, uniform distinct, PCM
- 환자별 reference gradient: 가능한 모든 capped image와 여러 timestep/noise의 큰 Monte
  Carlo 평균
- 지표:
  - relative gradient MSE
  - cosine similarity to reference
  - norm distribution
  - 예상 clipping rate와 clipping shrinkage
  - runtime, peak VRAM
  - record count, X-ray longitudinal/finding heterogeneity별 효과

PCM promotion gate:

- uniform distinct보다 reference-gradient error가 안정적으로 낮고
- runtime overhead가 사전 상한을 넘지 않으며
- 특정 소수 patient에만 이득이 집중되지 않아야 한다.

주 promotion gate는 exact-source X-ray public development에서 수행하고 final test는
보지 않는다. ISIC에서 같은 diagnostic을 먼저 실행해 implementation을 점검할 수 있으나,
ISIC-only 이득만으로 X-ray core의 PCM을 승격하지 않는다.

### Stage 2. X-ray K5-cap one-seed feasibility

최소 arm:

1. M0
2. M1 image-DP
3. M2-U1
4. M2-NM at selected Q
5. M2-UD at selected Q
6. M2-PCM at selected Q

목적은 논문 결론이 아니라 다음을 확인하는 것이다.

- 실제 LoRA patient gradient accumulation이 RTX 3070 8GB에서 동작하는가
- finite loss, checkpoint reload, generation이 가능한가
- 한 run의 시간·저장공간으로 confirmatory budget을 산정할 수 있는가
- DP noise 아래 generation이 완전히 붕괴하지 않는가
- PA anatomy, support device, text/marker와 finding conditioning의 명백한 shortcut이
  나타나지 않는가

RTX 3070은 smoke/pilot 장비로만 분류한다. multi-seed confirmatory를 이 장비 하나로
완료할 수 있다고 가정하지 않는다.

### Stage 3. X-ray core confirmatory

- 사전 선택된 common patient privacy target
- M2 방법 arm은 동일 q_p, T, sigma, C, Q
- 3 independent optimization seeds
- final test는 configuration freeze 후 한 번 열기
- primary comparison: M2-PCM versus M2-NM
- secondary: M2-PCM versus M2-UD, M2-U1, M1-T
- K10을 main cap 후보로, K2/K5를 unit/contribution sensitivity로 두되 exact X-ray
  cohort와 compute gate에서 outcome을 보기 전에 확정
- 3 seed와 powered attack에 필요한 외부 compute를 먼저 확보

### Stage 4. ISIC cross-domain confirmation

X-ray 결과를 보고 ISIC에 유리한 hyperparameter를 넓게 재탐색하지 않고, 미리 허용한
좁은 transfer rule과 이미 동결한 patient split/cap을 사용한다. 별도 dermoscopy model에서
최소 M0, M2-U1, M2-NM, M2-UD, M2-PCM을 비교하고, M1 checkpoint route가 준비되면
M1-G/M1-T도 포함한다.

## 12. 평가

### 12.1 생성 품질과 다양성

- KID를 주 통계 후보로 사용하고 patient bootstrap CI 보고
- FID는 보조로 보고하되 small medical test set의 bias를 명시
- DINO/의료 encoder feature distance와 precision/recall
- condition별 quality와 diversity
- nearest-neighbor montage는 정량 copy audit와 함께 사용

ImageNet feature의 FID 하나만으로 결론을 내리지 않는다.

### 12.2 medical utility

주 지표는 synthetic-only training, real patient-held-out testing이다.

- NIH core: held-out real-patient radiographic-finding AUROC/AUPRC와 sensitivity at fixed
  specificity. weak-label noise와 overlapping findings를 그대로 명시
- acquisition shortcut: PA policy, support device·marker·hardware, intensity와 anatomy에
  대한 stratified performance
- ISIC extension: melanoma finding AUROC/AUPRC와 sensitivity at fixed specificity
- optional MURA: study-level abnormality AUROC/AUPRC와 Cohen kappa
- rare/underrepresented condition과 patient record-count strata
- real-only classifier는 upper reference, random/base generation은 lower reference

real+synthetic augmentation 결과는 실제 데이터 비율과 classifier optimization이 섞이므로
secondary로 둔다.

### 12.3 patient membership

필수 scalable protocol:

1. 여러 timestep/noise query의 denoising loss 또는 calibrated likelihood proxy를 image
   score로 만든다.
2. 환자 score는 mean, max, top-k mean을 사전 고정해 집계한다.
3. member patients와 privacy-attack holdout patients를 비교한다.
4. patient ROC-AUC, TPR at low FPR, bootstrap CI와 record-count strata를 보고한다.

Nature 2026식 individual-patient AUC/eSF는 동일 환자가 여러 target model에서 member와
non-member로 반복 관측되어야 한다. 한 target model의 score tail을 individual AUC라고
부르지 않는다. 진정한 eSF는 최소 target-model 수와 power를 계산한 별도 compute gate를
통과할 때만 lightweight proxy model 또는 다수 LoRA replica로 수행한다.

### 12.4 memorization과 copying

- training image와 synthetic image의 self-supervised feature nearest neighbor
- threshold는 train-test 및 test-test real pair로 사전 calibration
- copy rate와 patient coverage rate
- 동일 patient의 여러 이미지 중 어느 하나와 가까운 경우를 patient copy로 집계
- 생성 수 증가에 따른 extraction curve
- expert review는 가능한 경우 blinded subset으로 제한

formal DP와 empirical attack은 역할이 다르다. 공격이 실패해도 formal claim의 증명이 아니고,
공격이 강하지 않아도 DP guarantee는 accountant로 판단한다.

### 12.5 compute

- trainable parameter 수
- gradient evaluations, patient updates, wall-clock
- peak allocated/reserved VRAM
- energy가 측정 가능하면 포함
- checkpoint conversion audit cost와 native retraining cost

## 13. 통계 계획

1. primary method comparison, primary privacy target, primary utility metric을 confirmatory 전
   고정한다.
2. 환자가 독립 단위이므로 CI와 bootstrap도 patient 단위로 수행한다.
3. 3 seed 평균, 표준편차, paired seed difference를 보고한다.
4. 여러 metric 중 유리한 것만 고르지 않고 전체 registered table을 공개한다.
5. PCM의 method success는 단일 FID 개선이 아니라 다음의 결합으로 판정한다.
   - estimator diagnostic 개선
   - preregistered medical utility 개선
   - 같은 patient epsilon/delta
   - 계산 상한 준수
6. privacy attack 결과는 effect size와 CI, attack power를 함께 보고한다.

## 14. go/no-go 기준

### G0. correctness

- patient adjacency, clipping, noise, sampler와 accountant가 모두 일치해야 함
- 실패하면 즉시 중단

### G1. estimator

- PCM이 uniform distinct보다 estimator error 또는 clipping distortion을 재현성 있게
  줄여야 함
- 실패하면 PCM을 제거하고 uniform nested multiplicity까지만 남김

### G2. DP utility

- M2-PCM이 같은 privacy/compute의 M2-NM보다 primary medical utility에서 유의미한 개선을
  보여야 함
- estimator만 좋아지고 생성 utility가 개선되지 않으면 CVPR method claim 중단

### G3. external validity

- X-ray core에서 효과가 나더라도 ISIC extension에서 방향이 뒤집히면 일반 method claim을
  낮추고 modality-conditional result로 보고
- X-ray와 ISIC 모두 개선되지 않아도 조건별 variance 분석이 매우 선명하면 제한적 논문은
  가능하지만 CVPR main 경쟁력은 낮아짐

### G4. reuse frontier

- M1-T와 M2 사이에 실제 reuse/claim downgrade/retrain 영역이 관찰되어야 함
- 모든 conversion이 차단되면 “conversion saves retraining” 주장은 삭제하되,
  unsafe release를 차단한 결과로 보고할 수 있음

### Reviewer pre-mortem

| 예상 비판 | 설계에서 미리 요구할 답 |
|---|---|
| ULS와 DPDM을 단순 결합했다 | M2-NM과 M2-UD를 모두 두고 PCM의 coverage stratification 자체 효과를 분리 |
| private image로 cluster를 만들면 privacy가 깨진다 | cluster는 미공개 내부 연산이고 최종 patient vector를 C로 clip함을 mechanism·unit test·threat model에 명시 |
| visual distance가 gradient geometry를 반영하지 않는다 | public-development reference-gradient MSE/cosine gate를 통과하지 못하면 PCM claim 제거 |
| estimator는 unbiased여도 clipping 뒤에는 bias가 있다 | pre-clipping theorem과 post-clipping empirical distortion을 분리해 보고 |
| X-ray 결과가 view/device shortcut이다 | PA-only 우선, device·marker·hardware와 intensity strata, radiology feature와 held-out patient utility |
| 한 장 환자가 많아 방법이 일부에만 적용된다 | all-patient cohort 유지, n_u=1에서 표준 update로 환원, record-count strata와 absolute patient counts 보고 |
| image-DP와 patient-DP 숫자가 비교 불가능하다 | common target patient epsilon/delta, k_u profile은 diagnostic, release claim은 worst-case K |
| 최신 image-DP SOTA보다 약하다 | PrivImage·dp-promise·SPTI·RPGen을 context에 포함하고 compatible representative만 동일 data/policy에서 실행 |
| patient MIA를 잘못 정의했다 | one-model aggregate protocol과 multi-target individual eSF를 명시적으로 분리 |
| public base가 NIH/ISIC을 보았을 수 있다 | B0, exact/near-duplicate overlap audit와 incremental-membership wording |
| 계산량 차이로 이득이 생겼다 | M2 ablation의 Q, q_p, steps, sigma, C를 고정하고 wall-clock·VRAM·energy를 함께 보고 |
| exact data provenance가 불명확하다 | NIH main metadata 3-byte 차이와 image source가 해결될 때까지 training gate를 닫음 |

## 15. 예상 figure와 table

### Main figures

1. patient → coverage strata → diffusion perturbations → one patient clip/noise diagram
2. fixed Q에서 image coverage와 noise repetition의 variance decomposition
3. target patient epsilon에 따른 reuse / downgrade / retrain frontier
4. utility–privacy–compute Pareto plot
5. patient record count·rare condition별 membership/copy risk tail

### Main tables

1. related-work claim matrix
2. method ablation at matched Q and patient epsilon/delta
3. NIH chest X-ray core와 ISIC extension의 generation·medical utility
4. formal accounting, attack, memorization과 compute

## 16. 논문 기여 초안

1. patient-linked medical diffusion의 privacy unit을 명시하고 patient-balanced DP LoRA
   training을 정확히 구현한다.
2. 환자 내부 image heterogeneity와 diffusion perturbation variance를 분리하고,
   fixed-compute coverage-aware multiplicity estimator를 제안한다.
3. generic/tight image-to-patient conversion과 direct patient-DP를 동일 target budget에서
   비교해 checkpoint reuse-versus-retrain frontier를 보인다.
4. 복수 의료영상 데이터셋에서 patient membership, copying, clinical utility와 compute를
   함께 평가한다.

1번만으로는 신규성이 약하다. 2번이 중심 방법 기여이고 3·4번이 그 필요성과 실용적
consequence를 입증한다.

## 17. non-claims

- patient clipping, ULS, hierarchical averaging, noise multiplicity를 최초 제안하지 않는다.
- 공개 NIH/ISIC 실험이 비공개 병원 데이터 처리 경험을 뜻하지 않는다.
- NIH weak finding label을 확정 진단, Mass/Nodule을 암, No Finding을 판독의가 확정한
  완전 정상으로 부르지 않는다.
- audit이 모델의 privacy를 향상시킨다고 하지 않는다.
- empirical MIA 실패가 privacy guarantee라고 하지 않는다.
- record-level DP epsilon을 그대로 patient epsilon이라고 부르지 않는다.
- feature coverage가 모든 데이터셋에서 variance를 낮춘다고 보장하지 않는다.
- 한 모델의 patient score tail을 individual-patient MIA AUC라고 부르지 않는다.
- filtering, watermark 또는 receipt가 DP를 대체한다고 하지 않는다.

## 18. 구현 순서

1. NIH exact metadata provenance의 3-byte 차이 해결
2. PA-only 대 view-conditioned AP+PA, finding/multi-label policy 동결
3. patient-first X-ray K2/K5/K10 candidate manifest 생성과 독립 재생성
4. source-faithful selective acquisition·storage 및 exact-image SD 2.1 interface gate
5. PCM을 포함한 동결 실험계약 변경 diff 제안
6. exact accountant·sampler contract와 patient-batch toy unit tests
7. frozen VAE feature와 per-patient strata manifest 생성
8. X-ray public-development Stage 1 diagnostic
9. G1 판단 후 PCM 승격 여부와 Q 확정
10. X-ray K5-cap one-seed DP feasibility와 runtime receipt
11. 공통 patient budget, attack power, confirmatory compute 동결
12. X-ray core three-seed confirmatory
13. ISIC cross-domain extension
14. final test, 통계, paper figures
15. 자원이 남고 필요성이 있으면 MURA third-dataset gate

이 문서는 training 명령이 아니다. 1–6번의 source·contract·accountant gate를 통과하기
전에 generator/DP training을 시작하지 않는다. 이미 확보한 ISIC으로 toy/interface 구현을
점검할 수는 있으나, 승인된 X-ray core hierarchy를 조용히 뒤집지 않는다.

## 19. 최종 판단

이 접근은 make sense 한다. 다만 성공 가능성이 가장 높은 형태는 “patient-DP를 적용했다”가
아니라 “환자 내부의 visual coverage와 diffusion stochasticity를 fixed compute에서
분리·배분했다”이다.

현재 자료는 이 질문을 시험할 구조적 근거가 충분하다. NIH PA-only 후보에도 반복 영상
환자가 10,688명이고 record-count long tail이 있어 patient unit과 PCM의 효과를 동시에
볼 수 있다. 그러나 exact metadata와 full image corpus는 아직 동결·확보되지 않았다.
반면 ISIC K5/K10, SD 2.1 LoRA와 patient-first manifest는 이미 준비되어 있어
cross-domain 구현 위험을 낮춘다. RTX 3070 8GB은 toy/interface와 one-seed pilot용이며,
X-ray multi-seed confirmatory와 powered attack은 별도 compute가 필요하다.

따라서 이 논문의 현재 평가는 다음과 같다.

- 주제·문제 중요성: 높음
- 방법·기존 인프라 정합성: 높음
- X-ray core data readiness: exact-source gate 전이므로 중간 이하
- 단순 ULS 적용의 신규성: 낮음
- PCM이 실제 개선될 때의 CVPR 경쟁력: 중상~높음
- PCM 개선이 없을 때의 CVPR method 경쟁력: 낮음~중간

즉, 지금 가장 먼저 할 일은 큰 DP training이 아니라 X-ray exact-source/view/finding/
manifest gate다. 그 다음 가장 값싼 과학적 검증이 Stage 1 reference-gradient/variance
diagnostic이다. 여기서 방법 효과가 보인 뒤에만 비싼 confirmatory training으로 넘어간다.

## 20. 2026-09-02 연구 가설 동결: 단순 평균 이후의 배분 규칙

현재 PCM은 완성된 신규 방법명이 아니라 **선행과 겹치지 않는 배분 규칙을 찾기 위한
working hypothesis**다. 아래 항목은 신규 claim으로 사용하지 않고 baseline 또는 분석
도구로만 사용한다.

- 한 환자·사용자의 여러 record gradient를 단순 평균하고 entity 단위로 clipping하는 것
- record 간 변동과 diffusion noise/timestep 변동을 총분산 법칙으로 분해하는 것
- 같은 이미지에 여러 diffusion perturbation을 적용하는 noise multiplicity
- 일반적인 clustering 또는 stratified sampling 자체

동결한 중심 질문은 다음과 같다.

> 고정된 entity-level privacy, clipping norm과 환자당 gradient evaluation budget `Q`에서,
> 서로 다른 record 수 `m`과 record당 diffusion perturbation 반복 수 `r`를 어떻게 배분해야
> pre-clipping 추정 오차뿐 아니라 post-clipping bias/distortion까지 줄일 수 있는가?

환자 `u`의 목표 기울기는 개념적으로 다음과 같이 둔다.

$$
G_u = \frac{1}{n_u}\sum_{j=1}^{n_u}
      \mathbb{E}_{t,\epsilon}[g(u,j,t,\epsilon)].
$$

첫 diagnostic에서는 reference enumeration으로 `G_u`를 근사하고, 동일 `Q`에서 다음을
비교한다.

1. `noise-only`: 한 record에 `Q`개의 diffusion perturbation
2. `uniform-distinct`: 가능한 한 서로 다른 record를 먼저 균등 선택
3. `coverage-stratified`: 사전 고정된 feature strata에서 가중 대표 선택
4. 이후에만 검토할 새로운 allocation rule

귀무가설 `H0`는 compute-matched uniform/noise-multiplicity가 충분하여 coverage-aware
allocation의 추가 이득이 없다는 것이다. 연구가설 `H1`은 record 간 이질성이 충분한
entity에서 배분 규칙이 reference-gradient MSE, clipping probability와 clipping distortion을
일관되게 줄인다는 것이다.

방법 승격 조건은 다음 네 가지를 모두 요구한다.

1. 동일 `Q`, patient sampling, privacy와 clipping contract
2. singleton에서 기존 noise-multiplicity로 자연스럽게 환원
3. uniform-distinct 대비 반복 seed의 estimator/clipping 개선
4. 작은 실제 DP pilot에서 품질·임상 utility 개선

1--3을 통과하지 못하면 PCM을 신규 방법으로 부르지 않는다. variance decomposition은 결과를
설명하는 분석이며, 논문 기여는 그 분석에서 유도되고 선행 baseline을 이기는 구체적
allocation rule과 검증이어야 한다.

## 21. 2026-09-02 단계 0 및 1차 실제-gradient smoke 결과

### 21.1 Synthetic measurement harness

5 seeds, seed당 16 entities와 entity당 64 trials로 positive/negative control을 실행했다.
`Q=4`의 pre-clipping MSE는 다음과 같았다.

| scenario | noise-only | uniform-distinct | aligned stratified |
|---|---:|---:|---:|
| no record heterogeneity | 0.00381410 | 0.00381410 | 0.00381410 |
| aligned record modes | 0.02396026 | 0.00673088 | 0.00387136 |

record effect가 없을 때 Q=2/4/8의 모든 allocation은 수치적으로 완전히 같았고, 정렬된
synthetic strata에서는 distinct와 stratified가 예상 방향으로 개선됐다. 이 PASS는 metric과
sampling harness의 내부통제만 확인하며 실제 의료영상 또는 신규 방법 증거가 아니다.

### 21.2 ISIC public-development one-patient smoke

결과를 보기 전에 eligibility, salted-hash selection, base revision, `Q=4`, `C=1.0`, rank 8,
4 perturbations/record와 feature를 프로토콜로 동결했다. 78 eligible patients 중
`IP_0687884`가 hash로 선택됐고, 5개 lesion·3개 anatomical site의 이미지 5장에 대해
1,659,904-dimensional LoRA gradient 20개를 계산했다. optimizer update와 DP noise는 없었다.

| method | pre-clipping MSE | noise-only 대비 | clip rate |
|---|---:|---:|---:|
| noise-only | 1.42465e-8 | 1.000 | 0 |
| uniform-distinct | 6.92098e-9 | 0.486 | 0 |
| VAE coverage-stratified | 9.95903e-9 | 0.699 | 0 |

실행·hash·finite gradient·weight sum·Gram PSD gate는 PASS했다. 그러나 효능 방향은 현재
coverage rule에 유리하지 않았다. 여러 record를 보는 uniform-distinct는 noise-only보다
좋았지만 VAE coverage-stratified는 uniform-distinct보다 나빴다. 또한 `C=1.0`에서 clipping이
발생하지 않아 post-clipping 가설은 시험되지 않았다.

이 한 환자 결과를 보고 환자나 feature를 교체하지 않는다. 현재 PCM coverage 후보는
**미승격** 상태다. 다음 효능 판단은 별도 사전 동결한 multi-patient public-development
diagnostic에서 모든 선택 환자와 adverse/null 결과를 유지하고, variance decomposition,
feature-gradient alignment와 사전 clip sensitivity grid를 함께 본 뒤에만 내린다.

## 22. 16-patient multi-patient diagnostic: VAE coverage 폐기

smoke 환자 `IP_0687884`를 제외하고 anatomical-site diversity가 낮은 8명과 높은 8명을
결과 확인 전 salted hash로 고정했다. 환자당 5 records와 4 perturbations, 총 320개의 실제
SD 2.1 rank-8 LoRA gradients를 계산했다.

| primary comparison | geometric MSE ratio | bootstrap 95% CI | wins |
|---|---:|---:|---:|
| VAE coverage m4 / uniform m4 | 1.0322 | [0.9674, 1.1005] | 6/16 |

low/high site subgroup 모두 3/8 wins였고, clip sensitivity `C=0.05`--`1.0`도 결론을
구하지 못했다. frozen feature distance와 gradient distance의 patient-median Spearman은
`0.2848`이었다. 반면 uniform record allocation에서 m1/m4는 `1.7294`, m2/m4는 `1.2437`로,
여러 서로 다른 record를 보는 신호 자체는 강했다.

사전 gate에 따라 `RETIRE_CURRENT_VAE_COVERAGE`로 판정한다. PCM이라는 이름을 유지하기 위해
outcome을 본 뒤 feature를 교체하지 않는다.

## 23. paired-strata oracle headroom: family 폐기

동일 Gram matrix에서 5 records의 가능한 10개 two-anchor partitions를 모두 exact
enumeration했다. gradient 결과를 본 뒤 가장 좋은 partition을 고르는 비실행 oracle조차
uniform 대비 `0.9749`, 8/16 wins였다. 사전 gate `<=0.90` 및 `>=12/16`에 미달해
`RETIRE_PAIR_STRATIFIED_M4_FAMILY`로 판정했다.

이는 “더 좋은 visual feature를 찾으면 해결된다”는 추측보다 강한 반증이다. 현재의
5-record, Q=4 pair-stratification family에는 outcome-informed upper-bound에서도 충분한
headroom이 없었다.

## 24. two-axis balance와 새 cohort confirmatory

within-perturbation variance fraction의 patient median이 약 `0.833`이었던 점에 따라, 서로
다른 records와 perturbations를 동시에 중복 없이 배치하는 새 가설을 결과 전에 고정했다.

첫 4-perturbation bank screen은 balanced/replacement `0.7440`, 16/16 wins였지만 balanced가
매 update마다 finite reference의 perturbation 네 개를 전부 보는 유리한 구조였다. 이를
확증으로 쓰지 않고 새 환자와 새 8-perturbation bank confirmatory를 만들었다.

- 이전 smoke 및 16 patients와 overlap 0인 새 low-site 8명, high-site 8명
- four equal-width timestep bins, bin마다 독립 timestep/noise 두 개
- 환자당 5 x 8 = 40 gradients, 총 640 gradients
- replacement baseline 20,480 states, bin-balanced 1,920 states exact enumeration
- 두 estimator 모두 40-cell reference에 unbiased

결과:

| comparison | geometric MSE ratio | bootstrap 95% CI | wins |
|---|---:|---:|---:|
| true-bin balanced / replacement | 0.8592 | [0.7931, 0.9220] | 16/16 |

low-site ratio는 `0.8266`, high-site는 `0.8931`이었고 각 8/8 wins였다. 독립 audit가
selection, hashes, Gram-derived metrics와 gate를 재현해 PASS했다. 이 결과는
`TWO_AXIS_BALANCE_CONFIRMED_FOR_METHOD_REVIEW`이지 training 또는 신규성 확정이 아니다.

## 25. timestep-bin 대 generic without-replacement exact ablation

positive confirmatory 결과를 본 뒤, 개선이 단지 8 perturbations 중 중복을 막은 효과인지
구분하는 protocol과 gate를 먼저 고정했다.

| decomposition | geometric MSE ratio | bootstrap 95% CI | wins |
|---|---:|---:|---:|
| distinct perturbation / replacement | 0.9328 | - | - |
| true-bin / distinct perturbation | 0.9211 | [0.8814, 0.9572] | 16/16 |
| true-bin / replacement | 0.8592 | [0.7931, 0.9220] | 16/16 |

true-bin/distinct 사전 gate는 ratio `<=0.95`, CI upper `<1`, wins `>=10/16`이었고 모두
PASS해 `TIMESTEP_BIN_INCREMENT_RETAINS_SIGNAL`로 판정했다. 8 perturbations의 105개 가능한
perfect-pair partitions를 placebo로 열거했을 때 실제 chronological grouping은 13/16
patients에서 1위였고 나머지는 4·4·11위였다.

단, 16 patients가 같은 perturbation bank를 공유하므로 patient bootstrap이 bank 불확실성을
나타내지 않는다. 전체 21 unit tests는 PASS했지만 최소 3개 새 bank replication 전에는
method promotion을 하지 않는다.

## 26. 관련연구 충돌 이후의 현재 방법 경계

추가 원문 조사에서 VDM과 DDIM의 minibatch antithetic time sampling, DPDM의 image-level
noise multiplicity, DP diffusion fine-tuning의 pre-clipping timestep multiplicity,
CleanDIFT와 HDiT의 stratified timestep sampling이 직접 확인됐다. 따라서 timestep bin
balance 자체도 신규 claim이 아니다.

현재 남기는 후보는 다음의 좁은 차이다.

> patient-level clipping의 비선형성 전에 각 privacy unit 안에서 multiple records와
> timestep strata를 함께 unbiased하게 배치한다. minibatch-global balance가 아니라
> patient-unit-local balance로 각 patient vector의 stochastic error와 clipping distortion을
> 낮춘다.

이 후보는 다음 네 gate를 모두 통과해야 한다.

1. 최소 3개 새 perturbation banks에서 재현
2. batch-global antithetic보다 patient-unit-local balance가 같은 compute에서 우세
3. exact-source NIH PA-only X-ray public-development에서도 방향 유지
4. 동일 patient privacy·compute의 DP pilot에서 generation/medical utility 개선

하나라도 핵심 gate를 통과하지 못하면 timestep-balanced PCM을 CVPR 방법 기여로 사용하지
않는다. VAE coverage는 이미 폐기됐으며 이 결과로 되살리지 않는다. 상세 판정과 문헌은
`06_ISIC_실제_gradient_진단과_현재_방법판정_2026-09-02.md`에 기록했다.

## 27. 5-new-bank perturbation replication: PASS

최초 confirmatory의 16 patients가 동일 perturbation bank를 공유한 한계를 제거하기 위해,
결과 확인 전 `repl_01`--`repl_05`, seed serialization, 5/5 방향·wins·equal-bank ratio·
hierarchical bootstrap·pairing-rank gate를 protocol로 고정했다. cohort, 80 records, model,
LoRA parameters, preprocessing와 `Q=4`는 변경하지 않았다.

새 5 banks에서 bank당 640개, 총 3,200개 실제 gradients를 계산했다. 모든 bank에서
640/640 finite, critical model hashes, Gram PSD, unbiasedness와 exact state counts가 PASS했고,
각 bank의 독립 audit와 105-partition mechanism ablation도 PASS했다.

| bank | true-bin/distinct | true-bin/replacement | wins | median grouping rank / 105 |
|---|---:|---:|---:|---:|
| repl_01 | 0.9354 | 0.8601 | 15/16 | 3 |
| repl_02 | 0.9170 | 0.8415 | 16/16 | 1 |
| repl_03 | 0.7862 | 0.6757 | 16/16 | 1 |
| repl_04 | 0.9356 | 0.8802 | 16/16 | 1 |
| repl_05 | 0.9177 | 0.8470 | 15/16 | 1 |

equal-bank true-bin/distinct ratio는 `0.8965`, bank/patient hierarchical bootstrap 95% CI는
`[0.8309, 0.9419]`였다. bank ratio 5/5가 1 미만이고 모든 bank가 10/16 wins 이상이며,
모든 frozen conditions가 PASS했다. 독립 aggregate audit가 수치와 판정을 재현했고 전체
진단 test suite 25개가 PASS했다.

따라서 결정은 `MULTIBANK_TIMESTEP_SIGNAL_REPLICATED_FOR_LOCALITY_TEST`다. 이는 단일-bank
우연 가능성을 낮추고 다음 locality ablation으로 갈 자격을 준다. 다섯 bank가 동일 환자와
checkpoint를 사용했고, optimizer/DP noise/generation이 없으며, timestep stratification
자체가 선행이라는 경계는 변하지 않는다.

다음 G2b는 동일 timestep marginal과 총 compute에서 minibatch-global balance와
patient-unit-local balance를 직접 비교한다. patient clipping 전 local placement의 추가
post-clipping 이득이 없으면 현재 후보를 신규 방법으로 승격하지 않는다.

## 28. Patient-unit-local clipping proxy: PASS

2026-09-03, 새 distinct post-clipping 결과를 계산하기 전에 B=2 same-compute 구성,
primary `C=0.20`과 5-bank gate를 별도 protocol에 고정했다.

- global-only arm: batch 전체에서 여덟 perturbation을 정확히 한 번 쓰되 환자별로는 임의의
  서로 다른 네 개를 배정한다. patient marginal은 `distinct_perturbation_m4`다.
- unit-local arm: 각 timestep bin의 두 perturbation을 두 환자에게 하나씩 상보 배정한다.
  batch 전체 사용량은 같고 patient marginal은 `true_bin_balanced_m4`다.
- 두 arm 모두 patient당 서로 다른 4/5 records, `Q=4`, 같은 timestep marginal과 compute를
  사용하며 patient vector를 만든 뒤 한 번 clipping한다.

앞선 다섯 replication banks의 저장 within-patient Gram에서 모든 patient marginal state를
exact enumeration했다. 새 gradients, optimizer update와 DP noise는 없었다.

| primary C=0.20 | 결과 |
|---|---:|
| unit-local/global-only postclip geometric MSE ratio | 0.9094 |
| bank/patient hierarchical bootstrap 95% CI | [0.8446, 0.9506] |
| direction | 5/5 banks, 78/80 patient-bank cells |
| mean clip rate, global / local | 0.1948 / 0.1379 |

모든 frozen efficacy 및 nondegenerate-clipping 조건이 PASS해
`UNIT_LOCAL_CLIPPING_PROXY_PASSES_FOR_JOINT_BATCH_TEST`로 판정했다. 별도 audit가 원 Gram,
독립 weight enumeration과 clipping 계산에서 400개 cells를 최대 상대 오차 `6.09e-16`으로
재현했고 전체 진단 test suite는 31개 PASS했다.

이 결과는 G2b 전체 완료가 아니다. 저장 자료에는 서로 다른 patient 사이 cross-Gram block이
없으므로, global/local complementary allocation이 만드는 cross-patient covariance와 실제
clipped batch-average update MSE를 계산하지 못했다. 다음 gate는 frozen patient pairs와
cross-Gram을 사용하는 actual B=2 joint-batch test다. 이 test가 실패하면 locality claim을
중단한다. 상세 기록은
`07_patient_unit_local_clipping_proxy_결과_2026-09-03.md`에 보존한다.

## 29. Actual B=2 joint-batch locality: FAIL 및 현재 family 중단

위 중단 기준을 그대로 적용했다. 결과를 보기 전에 새 `joint_01`--`joint_05` banks, 같은
16-patient cohort의 모든 120 pairs, primary `C=0.20`과 일곱 gate 조건을 동결했다. bank당
640 gradients의 full 640 x 640 Gram을 계산해 이전 proxy에 없던 cross-patient blocks를
포함했다. 총 새 gradients는 3,200개다.

primary actual joint local/global ratio는 `1.0064606`, bank bootstrap 95% CI는
`[1.0020765, 1.0117106]`였다. 1/5 banks만 ratio < 1, 전체 pair wins는 191/600,
사전 84/120 wins 조건을 충족한 bank는 0/5, leave-one-patient-out ratio < 1도 0/16이었다.
따라서 frozen decision은 `UNIT_LOCAL_JOINT_BATCH_INCREMENT_NOT_SUPPORTED`다.

새 banks의 patient-marginal ratio는 `0.9429900`으로 5/5 banks에서 다시 유리했다. 그러나
평균 cross-patient contribution은 global `-4.2473e-10`, local `-1.6055e-10`이었다. local의
within-patient variance 감소보다 global complementary 배분의 음의 covariance 상쇄가 더
커 최종 joint MSE가 global `3.7863e-9`, local `3.8114e-9`로 뒤집혔다.

따라서 26절의 좁은 unit-local 후보는 핵심 방법으로 승격하지 않고 중단한다. patient-level
DP 의료영상이라는 상위 문제는 남지만, 다음 allocation 후보는 marginal variance와
cross-unit covariance를 함께 최적화하고 공개정보만으로 실행 가능해야 한다. 기존 full Gram의
후속 탐색은 post hoc discovery로만 취급하며, 새 규칙은 별도 banks·cohort의 사전 고정
검정 없이는 증거로 재사용하지 않는다. 상세 최신 기록은
`08_actual_B2_joint_batch_locality_결과_2026-09-03.md`다.

## 30. Covariance-aware rank-template discovery: FAIL 및 allocation branch 종료

actual joint 실패 원인을 이용해 더 나은 공통 allocation의 headroom이 있는지 확인하되,
개별 template 결과를 보기 전에 post hoc discovery protocol을 별도로 고정했다. 각 bin의 두
timestep을 low/high public rank tokens로 바꾸고, 8개 중 4개와 complement를 두 patient slots에
배정한 뒤 orientation을 50:50으로 뒤집는 35개 정책을 모두 열거했다. 모든 정책은 40 gradient
cells의 기대 가중치 `1/40`, B=2와 patient당 Q=4를 유지한다.

각 held-out bank의 template을 나머지 four banks에서만 선택했다. 다섯 folds 모두 bin 0·1의
두 tokens 대 bin 2·3의 두 tokens를 나누는 `rt_0123`을 선택했고 held-out ratios는
`0.9670`, `0.9485`, `0.9886`, `0.9361`, `0.9678`이었다. equal-bank ratio `0.961428`,
bootstrap 95% CI `[0.944857, 0.976267]`, 5/5 banks와 16/16 nested patient leaveouts에서
방향은 유리했다.

하지만 사전 최소 effect `<=0.95`에 미달했고 bank별 84/120 pair wins도 4/5만 통과했다.
환자별 marginal은 `1.128109`로 악화되고 clip rate는 global `0.1614`에서 `0.3689`로
증가했다. pair-specific hindsight oracle도 `0.941876`이어서 이 family 안의 추가 headroom은
최대 약 5.8%뿐이다. 결과를 보고 기준을 낮추지 않으며 판정은
`COVARIANCE_ALLOCATION_BRANCH_NOT_SUPPORTED_BY_DISCOVERY`다.

따라서 29절의 “다음 covariance-aware 후보”까지 포함해 현재 B=2 timestep-allocation method
branch를 종료한다. fresh GPU gradient banks와 X-ray 방법 검정은 수행하지 않는다. 상위
patient-level DP 의료영상 문제는 남기되 다음 방법은 timestep 배분이 아닌 다른 mechanism
class에서 관련연구를 다시 조사한 뒤 시작한다. 상세 기록은
`09_covariance_allocation_discovery_결과_2026-09-03.md`다.
