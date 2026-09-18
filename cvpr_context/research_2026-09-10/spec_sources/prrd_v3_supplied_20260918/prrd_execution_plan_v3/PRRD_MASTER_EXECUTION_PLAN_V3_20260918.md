# PRRD：환자 관계 보존 DP 증류 — 통합 실행·논문 검증 계획 v3

작성일: 2026-09-18  
기준 저장소: `hko920920/patient-privacy-grad-cvpr2026`  
확인한 원격 기준: `e560b2c2b6a8413c842d5e691a20b19be2145237`  
상태: **실행 계획 제안. 실제 의료 모델 성능 미검증. 저장소 변경·원격 업로드·GPU 실행을 수행한 기록이 아니다.**

## 0. 이 문서가 고정하는 것

주력은 **환자 내부 대응 정보를 환자-DP로 보호하고, 합성영상과 대응관계로 전달해 원자료를 받지 않는 다른 시각 학습자에게도 추가 효용을 주는 방법**이다. 캐시 조건 대조 diffusion은 별도 후보로 보존하되 이 패키지에 섞지 않는다.

기존 v2, 사용자가 붙인 Codex의 수정, 현재 저장소 결과를 기반으로 한다. 아래의 encoder 조합·정확한 seed·최적화 상수·run 개수·진행 규칙은 **이번 계획에서 추가한 권고 실행값**이다. 기존에 승인됐거나 성능이 검증된 값으로 소급하지 않는다. 실제 checkpoint SHA, 전처리 구현, 자료 manifest 경로, 장비 실측이 채워진 뒤 실행 계약을 동결한다. 과거 private 데이터로 보고 고른 값을 단순히 `public`이라고 다시 표시하지 않는다.

성과나 채택을 미리 보장하지 않는다. 대신 논문의 중심 주장을 지지할 증거와 반박할 대조를 처음부터 같은 계획 안에 둔다.

## 1. 사실·가설·새 실행값의 구분

### 1.1 그대로 보존할 사실 [S1–S3]

- 기존 full64 head 및 고정 448-update LoRA의 사적 합성자료 효용 실패와 해당 구성의 DP 확대 중단은 유지한다.
- Rwide는 공개672명/813장에 former-selection2027명/5097장을 직접 추가한 **비DP 실자료 진단**이다. AUROC .679434, AP .103542; R1 대비 AUROC +.134825. 합성자료·DP·환자 관계의 성공이 아니다.
- 추가2027명은 원 private_train 역할이며, 이미 classifier-selection 및 Rwide 학습에서 소비됐다. 새로운 공개 baseline이나 미사용 validation이 아니다.
- Qwide의 기흉 양성 환자는114명, 양성영상230장, 양성·음성 영상 모두 가진 환자는102명이다. 공개 P의 mixed 환자는3명이다.
- V는 이미 반복 사용한 method-development이다. 새 명세나 새 bootstrap으로 독립 자료가 되지 않는다.
- Expert final532명/810장과 reserved4213명은 보존한다. 이 문서는 그 접근을 승인하지 않는다.
- 원격에는 Rwide까지 확인된다. 사용자가 붙인 새 Codex 비교·정정 본문은 출처로 사용하지만, 원격에 없는 로컬 파일을 직접 검토했다고 표기하지 않는다.

### 1.2 시험할 가설

H1. 점별 정보에 추가되는 실제 환자 대조의 2차 구조가 유용하다.
H2. 그 구조를 합성영상·대응관계로 구현해 다른 encoder에도 전달할 수 있다.
H3. 관계 통계의 추가 차원·희소집단·잡음 비용을 감수하고도 동일 환자-DP 예산의 강한 점별 방법보다 가치가 남는다.

Rwide는 H1–H3를 입증하지 않았다. 같은 환자라는 이유만으로 병변의 인과 효과·정확한 시간 변화·identity 제거를 주장하지 않는다.

## 2. 논문 주장과 평가표의 연결

| ID | 앞으로 입증할 주장 | 직접 비교 | 충분하지 않은 결과 |
|---|---|---|---|
| U | 사적 자료의 추가 정보 | B−A | C만 공개전용보다 높음 |
| R | 실제 환자 대응의 추가 가치 | C−B 및 C−D | 관계 loss/모멘트 오차만 감소 |
| T | 다른 시각 학습자로 전달 | 증류에 사용하지 않은 E_t에서 C−B, C−D | source encoder에서만 성공 |
| P | 환자-DP 아래 가치 유지 | 같은 ε,δ에서 C−강한 B | 비DP 성공/이미지가 보기 좋음 |
| N | 기존 관계 방법 대비 남는 가치 | 관계 정보를 쓸 수 있는 R_joint 및 원문 대조 | A/B/C/D 내부 제거실험만 성공 |
| C | 비용·재사용 가치 | 직접 DP predictor/summary, 가까운 증류 방법의 전체 비용 | 합성 update 구간만 빠름 |

A/B/C/D는 **정보의 역할**을 구분한다. 선행 구현·관계 baseline은 **기존 방법에 비해 무엇이 추가되는지**를 구분한다. 두 역할을 바꿔 쓰지 않는다.

## 3. 데이터 계약

| 역할 | 환자/영상 | 허용 | 금지 |
|---|---:|---|---|
| P | 672/813 | 공개 초기 영상, label-free projection, 공개 통계·실자료 baseline | Q를 여기에 합쳐 public으로 재명명 |
| Q | 2027/5097 | 보호 대상 patient-local 통계, 비DP/DP 개발 | final/validation으로 재사용, nonDP 원통계 공개 |
| V | 2026/5047 | 고정된 모든 모델 완료 후 개발 평가 | 합성 최적화 loss에 삽입, 독립 final 주장 |
| Q80 | 80/320 | 과거 결과 보존 | Q 성공을 Q80 성공으로 대체 |
| Expert | 532/810 | 전체 freeze 후 별도 승인된 final | feasibility, tuning, early stopping |
| Reserved | 4213명 | 지금은 metadata overlap 확인만 | 자동 분할·학습·추론·confirmation |

P와 Q를 합치는 **목표 통계**는 환자별 통계 합과 class별 환자 수를 먼저 합친 뒤 정규화한다. `(공개 평균+사적 평균)/2`로 임의 혼합하지 않는다. Rwide의 batch source mixture와도 동일하다고 부르지 않는다.

Manifest 필수 열: `patient_id, image_id, sha256, label_pneumothorax, original_role, current_role, prior_uses, source_path, view_if_available`. 환자별 mixed 여부는 image-level label에서 계산한다. 환자 any-positive로 음성 촬영까지 양성으로 바꾸지 않는다. `label=0`은 기흉 음성 표기이지 normal 확정이 아니다.

영상 ID·환자 ID·해시의 학습/평가 중복을 검사한다. 이전 source SHA가 맞는 파일은 검사 결과를 재사용하고, 새로 읽는 파일·converter 변경 부분만 추가 검증한다. 매 단계 전체 파일을 불필요하게 재감사하지 않는다.

## 4. 모델 및 독립 수신자 계약 — 이번 권고값

### 4.1 증류 encoder E_s

- BioViL-T의 **공식 image model**에서 single-image global embedding을 얻는다. HF text-model 예제를 image model로 혼동하지 않는다.[S8]
- 정확한 image checkpoint, code commit, embedding layer, single-image input, 전처리와 resize/crop을 실제 public-input parity로 고정한다. 아직 확인하지 않은 weight SHA/API명은 config에서 null로 둔다.
- eval(), parameters frozen. 사적 영상 feature 추출은 no-grad. **합성영상 최적화는 image에 대한 gradient를 허용**한다. frozen encoder라고 전체 경로를 no-grad로 감싸면 안 된다.
- P만으로 label-free PCA를 만들고 상위 d_s=16을 사용한다. whitening은 사용하지 않는다. 반복 eigenvalue의 basis sign/순서는 저장된 projection을 기준으로 한다.
- 변환 예: `a=Pi_s(phi_s(x)-mu_P)`; P에서 구한 q95(norm(a))를 양의 floor와 함께 scale r_s로 고정; `z=a/max(r_s,||a||,1e-12)`. 항상 ||z||≤1. 이 변환은 공개 데이터에서만 결정한다.
- d16은 시작 비용/잡음 설정이지 질환 충분성의 근거가 아니다. 같은 패키지의 real readout이 이 제한도 드러낸다.

### 4.2 핵심 수신 encoder E_t

- 이번 첫 권고: `torchvision DenseNet121 / IMAGENET1K_V1`, classifier 이전 global feature.[S9]
- 이 모델은 증류 objective/gradient, 이미지 checkpoint 선택, projection 선택에 사용하지 않는다.
- P에서만 정한 별도 PCA128와 norm bound를 사용한다. 다른 encoder에 d16을 억지로 강제할 필요가 없다. 수신자 통계는 공개/합성자료에서만 계산된다.
- 공식 ImageNet resize256/center-crop224/normalization 경로를 고정한다. 흉부영상에서 이 전처리가 최적이라는 주장은 하지 않는다. 모델 card의 기본값과 별도 의료 전처리를 조용히 섞지 않는다.
- target 성능을 보기 전에 A/B/C/D/R 전체 산출물을 고정한다. E_t를 보고 다음 버전을 튜닝하면 이후에는 `development-selected recipient`라고 기록한다. 최종 재사용 주장은 별도로 예약한 E_t2에서도 확인한다. 이번 E_t2 권고는 `torchvision ViT-B/16 / IMAGENET1K_V1`, pre-head CLS feature와 P-only PCA128이다. 이 모델은 선택용 V 성능을 열지 않고 전체 모델·계수 동결 이후 확인용 수신자로 사용한다.[S11]
- E_t는 기흉을 판독하는 외부 정답 평가기가 아니다. 합성자료로 새 readout을 학습한 뒤 V/Expert의 실제 라벨로 성능을 측정하는 **수신 학습자**다.

### 4.3 추가 호환성

기존 ImageNet ResNet18 BCE, real16+synthetic16, 400step, seeds11/23/37을 그대로 사용한다. pair index를 버리는 호환성 비교이며 성공을 핵심 claim의 필수조건으로 두지 않는다. 데이터 loader 수정이 필요하면 기존 공개-only 입력에서 kernel parity를 확인한다.

### 4.4 실제자료 참조의 접근권

E_t의 synthetic 수신 경로는 Q를 읽지 않는다. 연구자 측 `trusted_real_reference`가 E_s/E_t의 P+Q readout을 계산할 수는 있으나 이는 private real 접근 기준점이다. 이를 원자료 없는 수신자 성능으로 표시하지 않는다. 기존 Rwide .679434를 다른 encoder의 상한처럼 복사하지 않는다.

## 5. 정확한 환자별 통계

각 환자 i의 label c 영상집합 D_ic에 대해 빈 집합의 통계는0이다.

```
n_ic = 1[|D_ic|>0]
m_ic = n_ic * mean_{x in D_ic} z(x)
A_ic = n_ic * mean_{x in D_ic} z(x)z(x)^T
b_i = n_i0*n_i1
delta_i = b_i*(m_i1-m_i0)/2
C_i = delta_i delta_i^T
```

각 클래스 안에서 환자를 같은 질량으로 다루고, 환자 안에서는 해당 라벨 영상을 평균한다. 이미지가 많은 환자에게 통계 질량을 더 주지 않는다.

```
N_c = sum_i n_ic
m_c = sum_i m_ic / N_c
A_c = sum_i A_ic / N_c
M = sum_i b_i
dbar = sum_i delta_i / M
C = sum_i C_i / M
```

분모0일 때도 구현을 정의한다. 비DP의 빈 empirical-risk 항은0. DP는 공개된 denominator-floor 규칙으로 noisy surrogate를 정의한다. 질환이나 mixed 집단이 없다고 원자료 count로 실행 여부를 공개 결정하지 않는다.

**평균 방문 대조**를 보존하는 것이다. 모든 실제 방문쌍의 차이 분포·시간순서·질환의 인과 변화는 보존 대상이 아니다.

## 6. 학습 목적과 분석 범위

기본 β=1, λ=.1은 v2에서 제안한 시작값을 유지한다. β0 readout은 제거 대조다. 결과 후 좋은 값을 골라 기존 결과를 바꾸지 않는다.

```
F_D(w) = 1/4 * sum_c mean_{patients with class c}
                    mean_{images of patient/class} (w^T z -(2c-1))^2
       + beta/2 * mean_{mixed patients} (w^T delta_i -1)^2
       + lambda/2 * ||w||^2
H = (A_0+A_1)/2 + beta*C
v = (m_1-m_0)/2 + beta*dbar
w = solve(H+lambda I, v)
```

Intercept는 첫 프로토콜에서 사용하지 않는다. 이는 임상 threshold나 확률 calibration을 제공하는 모델이 아니다. image-level raw score로 AUROC/AP를 측정한다.

두 class와 mixed 모두 존재할 때 `F=.5 w^T H w-v^T w+(1+beta)/2+.5lambda||w||²`. 빈 class가 있으면 상수도 해당 항을0으로 정의해야 한다. 최적해에 관계없는 상수라도 direct loss 검산에서 조용히 무시하지 않는다.

분석할 내용:

1. 같은 주변 모멘트라도 실제 대응과 교란 대응의 C가 다를 수 있다.
2. 같은 mixed endpoint 목록의 permutation은 dbar를 바꾸지 않는다.
3. 정확한 모멘트 일치는 해당 encoder·이차 위험 전체를 일치시킨다.
4. `||w_S-w_D|| <= (||v_S-v_D||+||H_S-H_D||op ||w_D||)/lambda`.
5. 이것은 임의의 새 encoder, BCE deep training, AUROC 향상에 대한 정리가 아니다.

이 대수와 Gaussian privacy를 독립 신규 정리처럼 세지 않는다. 논문 가치는 정보 보존·보호·시각적 전이의 실제 결합 결과다.

## 7. 사적 대응 교란 D

Q의 mixed102명만 대상으로, 각 환자의 양성 centroid 목록 p_i와 음성 centroid 목록 n_i를 고정한다. 공개 mixed3명은 그대로 둔다.

- run 시작 전 bank seed에서 정한 derangement로 n_i의 대응만 바꾼다.
- pointwise A/m, 영상목록, 라벨, class별 환자 가중치, mixed count는 그대로다.
- dbar 동일성을 실제로 검사하고 C의 차이를 기록한다.
- C/D는 같은 renderer·초기 template·최적화 seed·β·영상수·계산예산을 쓴다.
- 교란은 임상적으로 참인 음성 관계가 아니라 실제 대응의 역할을 반박할 수 있는 구조 대조다.
- 이 전역 pairing 변경에 C의 patient-local DP 감도를 자동 적용하지 않는다. 첫 D는 비DP 설명 대조다.

## 8. 합성자료 형식과 최적화 — 이번 권고값

### 8.1 출력 스키마

```
artifact/
  images/*.png                 # 최종 8-bit grayscale, 224x224
  images.csv                   # synthetic_id,label,bank_type,png_sha256
  pairs.csv                    # virtual_pair_id,positive_id,negative_id
  learning_contract.json       # risk,beta,ridge,preprocess,allowed use
  provenance_public.json       # 공개 source 및 코드 hash, privacy receipt
```

virtual_pair_id는 실제 환자 ID가 아니다. 실제 환자와 합성pair를 일대일 대응시키거나 private image를 초기화하지 않는다. 합성 label은 최적화 target이며 임상 ground truth가 아니다.

### 8.2 두 bank

- A/B: 128 marginal images, class당64장.
- C/D/R_joint: 64 marginal images(class당32) + 32관계쌍(64장).
- marginal risk에는 S_m만, relation risk에는 S_r만 사용한다. 서로 다른 모집단의 통계를 같은 균등 pair bank 하나에 강제로 맞추지 않는다.
- B의128장을 C와 같은64장으로 줄이지 않는다. B가 전체 budget을 점별 표현에 쓸 수 있게 한다.

### 8.3 이미지 표현

P에서 patient-hash로 고른 공개영상에서 시작한다. 어떠한 Q 영상, Rwide 가중치, nonDP private LoRA 출력도 prior로 쓰지 않는다.

기본 renderer는 v2의 다중해상도 grayscale residual. 해상도 [8,16,32,64,112,224], 전체500 successful updates. 단계별 활성화 [1,81,161,241,321,401]. 기존 활성 residual은 계속 학습한다. 마지막500step만 주산출물이며 checkpoint는 진단용으로만 저장한다.

실행 가능한 bounded pixel 표현의 권고:

```
x = sigmoid(logit(clip(public_template,1e-4,1-1e-4)) + sum_l upsample(r_l))
pair positive residual = u_l + v_l
pair negative residual = u_l - v_l
```

이것은 이미지를[0,1]에 두는 parameterization이지 해부학적 정확성 보장이 아니다. pair는 두 독립 이미지와 같은 수의 scalar residual parameter를 갖는다. source template 다양성은 A/B와C/D가 정확히 같을 수 없으므로 unique template 수도 함께 보고한다.

AdamW lr=.01, wd=0, betas=(.9,.999), FP32, master seed별 초기 residual std=1e-3. 500step은 수렴 보장이 아니다. 이 숫자는 이번 실행값 제안이며, 모델 결과를 보기 전 공개-input throughput/gradient parity에서 기술적 수정이 필요하면 version을 올려 동결한다.

### 8.4 목적과 증강

```
J_point = 1/4 * (||mS0-mT0||²+||mS1-mT1||²
                       +||AS0-AT0||F²+||AS1-AT1||F²)
J_rel = 1/2 * (||dbarS-dbarT||²+||CS-CT||F²)
J = J_point + gamma*J_rel + 0.01*J_pixel_anchor + 1e-4*TV + 0.1*J_aug
```

gamma=1. A/B에는 relation항이 없다. matrix Frobenius mismatch를 좌표 수로 추가 나눠 사실상 무시하지 않는다. pixel prior/TV는 모든 이미지의 평균이므로 이미지 수에 따른 강도 차이를 만들지 않는다.

**정확한 위험 보존의 주대상은 무증강 모멘트**다. J_aug는 작은 synthetic augmentation 후 모멘트도 같은 목표에 가까이 두려는 추가 regularizer이며, 원 실자료의 증강 기대 위험과 동일하다고 주장하지 않는다. 권고 범위: rotation±3°, translation±1%, scale.98–1.02, brightness .98–1.02, flip 없음. 관계pair는 같은 augmentation parameter를 공유한다. 모든 arm에 동일 규칙을 적용한다.

후속으로 실자료의 finite augmented moments를 query한다면 그것은 새로운 통계 정의·privacy release이며 원자료 재접근/회계를 수정해야 한다. 조용히 섞지 않는다.

### 8.5 exact microbatch 및 배포 픽셀

전체 moment residual을 no-grad로 계산한 후, 동일 이미지·augmentation 상태에서 microbatch response를 통해 chain-rule gradient를 누적한다. 이때 class별·bank별 전체 분모를 유지하며, microbatch 안의 임시 표본 수로 다시 정규화하지 않는다. 모든 gradient 누적이 끝나기 전 parameter update를 하지 않는다. 저장된 RNG/augmentation 재사용이 필수다.

학습 중 float32 이미지와 최종 8-bit PNG는 같지 않다. **모든 평가자는 최종 PNG에서 재추출한 특징으로 학습한다.** source에서도 float moment 오차와 PNG moment 오차를 분리해 기록한다. 적은 합성 표본의 모멘트 rank 한계도 보고한다. 예를 들어 K개 특징의 중심화 covariance rank는 최대 K−1이므로, 높은 특징 차원의 모든 목표를 정확히 표현할 수 있다고 전제하지 않는다. raw float에서만 맞았다는 결과를 배포 이미지 성능으로 바꾸지 않는다. PNG converter parity는 공개 입력으로 미리 검사한다.

## 9. 내부 대조와 가까운 기존 방식

| Arm | target | 출력 | 역할 |
|---|---|---|---|
| A | P의 점별1·2차 모멘트 | 128 marginal | 공개-only |
| B | P+Q의 점별1·2차 모멘트 | 128 marginal | 동일 표현의 강한 점별 대조 |
| C | P+Q 점별 + 실제 관계 | 64 marginal+32pairs | PRRD |
| D | C의 point target + Q만 대응교란 | C와 동일 | 실제 사적 대응 대조 |
| R_joint | P+Q point + mixed endpoint joint moments | C와 동일 | 표준 관계 정합 원리 대조 |

**R_joint는 이번 계획에서 추가한 명시적 baseline**이다. A–D 합의는 유지하고, 그 제거실험을 선행 우위로 오인하지 않기 위해 관계 정보를 사용할 수 있는 대조 하나를 첫 패키지에 넣는다.

mixed 환자에서 `u_i=[m_i1;m_i0]/sqrt(2)`, `U_i=u_i u_i^T`를 정합한다. C의 delta/C는 `L=[I,-I]/sqrt(2)`로 `delta=L u`, `C=L U L^T`다. 따라서 C는 표준 joint-moment 원리와 밀접하며 새로운 통계 원리라고 주장하지 않는다. R_joint는 더 넓은 endpoint 관계를 보존하는 비교다. **CovMatch의 원형 재현이 아니라** 영상–영상/환자 조건에 명시적으로 적용한 표준 공동모멘트 baseline이다.

C와 완전히 같은 차분 통계·목적·가중치를 쓰는 구현이 있다면 동일법으로 합치고 가짜 경쟁군으로 따로 세지 않는다. R_joint와 성능이 같고 비용 차이도 없다면 `새 relational matching 알고리즘 우위`를 주장할 수 없다.

최종 선행 비교에는 최소 다음을 추가한다.

- source native feature를 이용하는 강한 LGM 계열 nonDP 점별 증류. B16만 이기고 강한 점별 방법을 이겼다고 하지 않는다.
- patient-adapted DP-MEPF/DP-KIP/Dosser 중 입력·접근권·공개모델이 맞고 재현 가능한 직접 방법. 전수 구현 의무가 아니라 가장 가까운 방식의 유효 구현을 선택한다.
- 직접 patient-DP readout 및 공개된 요약의 직접 해법.

Baseline manifest에 원형/변형, 보호 단위, 학습 목적, 추가정보, HPO 예산, 전체시간, 구현·재현 수준을 적는다. 비용 때문에 가장 가까운 대조를 생략할 경우 주장을 좁힌다. 초기 고정500step pilot을 선행의 최적 성능이라고 부르지 않는다.

## 10. 첫 통합 패키지: 정확한 크기

**이번 권고:** A/B/C/D/R_joint × 초기화3개(101,202,303).

- 15개 synthetic bank.
- bank당128장 → 총1,920장.
- bank당500step → 총7,500 synthetic optimizer updates.
- 같은 repetition index의C/D는 같은 templates·initialization/augmentationstream. D는 사전 salt의 Q permutation만 다름.
- 각 bank의 마지막 모델을 모두 고정한 뒤 평가. 한 bank 결과를 보고 다음 bank를 추가하지 않음.
- 직접 readout: E_s/E_t 각각 P점별, P+Q점별, P+Q관계의6개 real reference + 합성자료 readout.
- 합성 readout은 E_s/E_t에서A/B β0, C/D/R β0와β1 → 총48개 deterministic solves(3rep×2encoder×8).
- 기존 ResNet18:15bank×3seed=45run×400step=18,000 classifier updates. 이는 별도 compatibility 결과다.
- 실제 Q feature 추출량, 합성 two-pass encoder forward/backward, 평가 특징 추출을 따로 기록한다.

기존 공개 code/kernel 그대로 사용 가능한 부분은 재사용한다. 모듈 하나마다 다시 장기 검토를 만들지 않고 **계약→연결검사→전체패키지→결과보고**를 한 작업 단위로 묶는다.

## 11. 수신자 평가의 세 역할

### 11.1 Source fidelity, 설명 결과

직접통계의 w_D와 실제 PNG에서 재계산한 w_S를 비교한다. risk/moment gap, optimizer gap, V AUROC/AP를 기록한다. source에서잘맞는 것만으로 핵심 재사용 성공을 선언하지 않는다.

### 11.2 E_t 관계-aware readout, 주결과

수신자는 합성영상·label·virtual pairs·공개 learning contract만 받아 자기 encoder의 특징을 계산한다. C/D/R은 β1, A/B는 β0가 기본. source projection·private raw moments를 받아 E_s 분류기를 그대로 실행하는 것이 아니다.

C−B는 **자료+학습규칙 전체 효과**다. C/D/R의 같은 산출물에 β0와β1을 모두 계산해 규칙 효과를 분리한다. B에 잘못 만든 pseudo-pair를 강요해 약화하지 않는다.

### 11.3 ResNet18 BCE, 부가 호환성

모든128장을 label에 따라 사용하고 pair정보는무시한다. P real 절반+synthetic절반, 고정batch/steps/init로 비교한다. Rwide와 source접근/학습목적을 혼동하지 않는다.

E_t가 좋고 BCE가 안 좋으면 관계-aware recipient로 주장을 좁힌다. BCE실패만으로 관계자료재사용실패라고 하지않는다. E_s에서만좋으면 시각재사용핵심은미확보다.

## 12. 통계와 개발 진행 규칙

- metric: image-level AUROC primary, AP secondary. 점수는raw; threshold를V에서새로고르지않는다.
- 반복촬영: patient-cluster bootstrap. 한 환자의 모든 방문을 함께 재표집한다.
- paired:같은 draw를 모든arm/bank/learner에 사용한다. 각model의metric을계산한뒤평균한다. prediction ensemble로몰래바꾸지않는다.
- 개발구간: 기존2,000draw로조건부patientCI 계산. 3bank변동은별도표시. 세seed가여러cohort라는해석금지.
- source/target라벨은weaklabel이라는범위유지. AP는해당evalprevalence에의존하며독립임상모집단precision보장아님.

### 이번 권고 engineering 기준

E_t,사전β/λ에서:

1. C−B 평균AUROC≥.01, C−D 평균AUROC>0.
2. 각비교에서3bank중최소2개가같은양의방향.
3. C의평균AP가B보다낮지않음.
4. C가A와E_t의공개real-only기준보다좋은지별도확인. B−A도반드시보고.
5. R_joint비교로기존관계원리대비남는가치평가. C≈R이면그동등성을숨기지않고새기여의범위를낮춘다.

.01은투자판단값이지임상적유의성·최종통계유의성기준이아니다. 모든효과크기·구간·seed별수치와음성결과도보고한다. 넓은CI와failure는해로움/효과0의증명이아니다.

### 결과가 알려주는 것

- B>A,C≈B/D:사적점별증류가능성은있으나PRRD관계기여는미확보.
- C>B/D,source만성공:E_s전용표현;cross-encoder시각전달주장은미확보.
- C>B/D,E_t성공:DP개발로진입할관계전달근거.
- C≈R_joint:표준관계matching과동일설명가능. 단순새이름으로방법우위주장금지.
- riskmatch성공,image/transfer실패:선택한표현·영상실현·전이연결의범위제한.
- 원인별추가개발은금지하지않지만,어떤불확실성을구분하는지와새조건을명세하고선택이력을누적한다.

## 13. 환자-DP 명세

### 13.1 q와보호단위

C의patient-local query:

```
q_i = [n_i0,n_i1,b_i, svec(A_i0),m_i0,svec(A_i1),m_i1,svec(C_i),delta_i]
```

svec의off-diagonal은sqrt2배. ||q_i||≤3. Add/remove-one-patient sum감도≤3,replace감도≤6. 환자내모든영상·라벨·대조를함께처리한다. class별병렬합성이라고하지않는다.

B는관계항이없다. qB의norm≤sqrt6,dimension=2+2[d(d+1)/2+d]. C는3+3[d(d+1)/2+d]. d16에서B306,C459이다. R_joint는d16에서867이고norm상계3이다. 각방법은자기감도·차원에서같은전체ε를사용한다. C의추가noise를B에부과하지않는다.

### 13.2 release와정규화

`Y=sum_Q q_i+N(0,s² I)`; ε8 primary, ε4 secondary, δ1e-5 후보. analytic Gaussian으로s계산하고독립검산한다.[S7]

P의exactsum을더한후,noisycount에공개floor1을적용해normalize하는규칙을기본으로제안한다. exactprivate count로clip/denominator/output크기를선택하지않는다. 빈class/M0에서도수치가정의돼야한다. 최종DPnoise는적절한보안난수에서생성하고실제noise seed나noise vector를공개하지않는다.

원통계/캐시는내부private데이터다. 원환자ID,pair목록,학습노출원장,비DPprivatecheckpoint를합성물bundle에넣지않는다.

### 13.3 불가능한모멘트처리

**합성목표기본:** 보호된raw모멘트에그대로squaredmatching을수행한다. 불가능한목표여도최적화는정의된다. irreducible residual을보고한다.

**직접통계분류기기본:** H를symmetrize→eigenvalue0floor→lambdaI추가→solve. 최소eigenvalue/conditionnumber/repairnorm을기록한다. 이는원래clean위험의정확해가아니라안정화한noisy surrogate해다.

PSD/covariance-consistent projection은선택가능한ablation이며성능향상을보장하지않는다. 후처리를새로선택할때는noisy자료만사용한다. 통계적으로가능해도실제encoder-image range에서실현가능하다는뜻은아니다.

### 13.4 privacy범위와HPO

- 모델당(ε,δ)와공개패키지전체composition을구분한다.
- 하나의Y로3개syntheticinitialization/여러recipient를만드는것은추가Q접근없을때후처리다.
- 서로 다른 Y, encoder query, dimension grid, 독립 DP 반복, 방법을 함께 공개하면 composition을 계산한다. 서로 다른 release의 noise는 독립 보안난수로 생성한다. 합성 초기화·평가 bootstrap의 공유와 DP noise의 공유는 다르며, 공동 잡음 출력의 차분으로 보호가 사라질 수 있다.
- 연구단계의nonDP결과및선택이력을최종DP보장이소급보호하지않는다.
- Q기반nonDP HPO로고른parameter를무료공개값으로재분류하지않는다. 고정한메커니즘의보장과전체적응적연구출력의보장을구분한다.
- privacy-valid배포용은현재정한설계를**새protectedcohort접근전에고정**하거나,추가선택이DPoutput의후처리/적절히회계된선택이도록해야한다.
- 최종synthetic recipient의선택이Q의DPartifact와독립V만사용하면Q에대해서는후처리일수있지만,V의개인정보까지자동보호하는것은아니다.
- MIA가낮다는것을DP의증명으로대체하지않는다. 과거publicNIH실험의모사역할도명시한다.

## 14. DP 실험크기와반복

비DP관계·수신자효용이후,별도동결한다.

**DP1 ε8:** B/C/R_joint × 독립mechanism noise3회 × syntheticinitialization3회 =27banks,3,456장,13,500synthetic updates. A는공개-only결과를재사용. D는비DP기전대조로보존.

**DP2 ε4:** 같은27bank계획은DP1이후실행승인조건을먼저정해둔다. 자동실행하지않고실행/미실행을모두보고한다. 출력privacy비용은각release및jointledger로구분한다.

DP-noise variation과synthetic-initialization variation을분리한다. bank9개를독립patientcohort9개로세지않는다. 보고table은각noise에서3init평균,noise간분포,paired patient interval을구분한다.

동일ε에서C−B≥.01/AP비감소,대부분noise의같은방향이라는개발기준을사용하되 최종성공은효과크기·불확실성·강한대조·전이를함께판단한다. C가비DP효용을100%유지해야한다는조건은없다. 추가private효용이A위에남는지확인한다.

## 15. 선행 비교와 robustness를뒤로무한히미루지않는규칙

첫15bank패키지는구성의가치를판단한다. 양성일경우 **expert final이아닌V에서** 직접선행및사전robustness를완료한다.

우선순위:

1. 강한pointwise one-shot baseline과R_joint. source native-feature/LGM계열대조를추가하고한정d16baseline만이긴것으로끝내지않는다.
2. same-encoder직접noisy-moment분류기,단순patient-DPclassifier. 단일task에서직접방법이우세할수있음을인정한다.
3. paircoef beta [.25,1,4],source dim[8,16,32],총synthetic[64,128,256]는각각one-factor선택된제한subset으로사전에열거한다. 전조합탐색으로바꾸지않는다. 최적그리드값은개발선택으로기록한다.
4. receiver E_t2는 위에서 지정한 ViT-B/16 ImageNet1K V1으로 고정한다. E_t를기준으로계속수정했다면E_t2를구별해보고한다. checkpoint선정원칙은학습출처·API·자원이며V점수순위가아니다.
5. 사전에기흉외secondary조건으로흉수를지정할수있다. P/Q/V의label/mixed수가충분한지는metadata로먼저확인한다. 결과를본뒤유리한질환을선택하지않는다. 충분치않으면범위를한질환으로명시한다.

위범위전체를이번답변에서실행했다는뜻은아니다. 외부데이터·전문의·두번째질환이확보됐다고가정하지않고,그범위가없으면논문주장을좁힌다. 동일cohort반복만으로다기관일반화를주장하지않는다.

## 16. final은한번의동결된평가campaign

Expert532명/810장은nonDP에서먼저열지않는다. main nonDP/DP,직접선행,receivers,계수,선택이력,artifacthash,통계가모두freeze된뒤별도승인으로평가한다. Reserved4213명도자동소비하지않는다.

최종primary family권고:

1. E_t2의nonDP C−B.
2. E_t2의nonDP C−D.
3. E_t2의ε8 C−가장강한사전선정pointwiseDPbaseline.

3개primary대조는각각98.333%양측patient-cluster interval(보수적Bonferroni)과효과크기를함께보고한다. 3개의bank/model들이고정된상태에서patient표집불확실성을다루는구간이며training/randomization전체population보장은아니다. Finalbootstrap10,000draw;primaryΔAUROC,APsecondary. Exactmetricfamily는final직전에결과를보지않고동결한다.

모든810장의image별expertlabel을유지하고환자cluster로분석한다. 학습에보지않은사람들이라도예전에전문의final결과로선택했다면독립final표현을바꿔야한다. 약한label의V와expert의AP를prevalence차이없이직접비교하지않는다.

final순차추론·기술적resume은가능하다. 일부arm점수를보고뒤arm을변경하는것은불가하다. bugfix가학습/선택결론을바꾸면수정후평가의사용이력을명시하며독립성을자동복구하지않는다.

## 17. 전체비용 및 재사용가치

각방법에서다음을분리한다.

```
T_total = public preparation + private encoder forward + statistics
        + DP release + synthetic optimization + export
        + recipient training + evaluation
```

공개encoder준비공통비용은공통으로보고하되없던것처럼삭제하지않는다. warm-cache와cold-start를둘다기록한다. GPU peakallocated/reserved,CPU RAM,diskcache·artifact bytes,forward/backwardcount,재사용수1/3/5에서추가privateaccess와추가시간을보고한다.

공개된이미지bank가여러학습자에게쓰이는경우출발통계로계산한직접분류기하나와동일기능이라고취급하지않는다. 반대로한task에서직접classifier보다불리한값도숨기지않는다. C가R_joint/B와비슷한효용인경우,.01 AUROC 같은사전noninferiority margin을선택할수있으나임상동등성을뜻하지않는다.

## 18. 실제작업순서와 산출물

| 단계 | 해야할일 | 완료물 | 종료/진행기준 |
|---|---|---|---|
| W0 | 원격/로컬문서결속,모델SHA·자료역할·설계값동결 | contract.json,accessledger | 미해결필수값없음;final닫힘 |
| W1 | feature/readout/moment/synthesis연결,공개microbatchprofile | parity.json,profile.json,runmatrix | 수치·gradient·입출력검사통과 |
| W2 | 15bank nonDP 통합방법비교 | images/pairs,predictions,report | C-B/D 및 E_t가치판정 |
| W3 | 강한근접대조·제한robustness·수신자계약완료 | baselinecards,selectionledger | 비교약화/숨긴선택없음 |
| W4 | 별도동결 DP1,조건부DP2 | receipts,composition,utility-cost | 보호후추가효용/범위판정 |
| W5 | 전체freeze후expert final | finalreport,모든음성포함 | 결과를주장에그대로반영 |
| W6 | 원고와재현물정리 | paper,methodcard,releasecard | 표·수식·provenance일치 |

W1을거대한별도연구로늘리지않는다. 공개입력의소수gradient/replay검사와시간측정을묶고,통과하면같은계약으로W2를수행한다. 각음성결과뒤다음진단을자동으로붙이지않는다.

## 19. 시간계획: 추정이아니라실측에서예산산출

현재GPU모델·가용VRAM·새BioViL backwardthroughput의확정값이없으므로시간을보장하지않는다. 첫공개profile에서microbatch[1,2,4] 중메모리한도내가장큰것을선택한다. 손실/gradient가같은지확인하고 선택은성능이아닌시스템값으로만한다.

계산식:

```
T_W2 ≈ T_private_features_once
      + 15 * 500 * t_synthetic_update
      + 45 * 400 * t_RN18_step
      + T_recipient_feature_readout + T_export_verify
```

t_update는**128장전체two-pass**시간으로정의한다. microbatch1회시간을bank1회처럼쓰지않는다. publicprofile 20warmup+30timedupdate를기준으로초기ETA범위를작성한다. 검산·문서작업시간은GPU시간과분리한다.

상한을초과하면행렬/feature재사용·microbatch·캐시개선처럼목표를유지하는구현변경을먼저검토한다. 몰래bank반복을줄이거나좋은중간checkpoint만채택하지않는다. 본실험범위변경은새계약으로기록한다.

## 20. 구현모듈과테스트

```
code_working/prrd_v3/
  contracts.py              # source hashes, role guards, phase authorization
  datasets.py               # image labels; original roles; no test load in fit
  encoders.py               # source and recipients, public PCA/normalization
  patient_moments.py        # all-patient point, mixed-patient relational
  gaussian_release.py       # secure production module, separate from toy reference
  render.py                 # grayscale pyramid, virtual pairs, uint8 export
  objectives.py             # A/B/C/D/R switches; exact two-pass gradients
  fit_banks.py              # 15-run orchestration, per-run resume and hashes
  readouts.py               # direct/source/recipient; beta0/beta1
  transfer_rn18.py           # audited legacy-compatible kernel
  statistics.py             # paired patient clusters, families, AP, seed levels
  cost.py
  run.py
  verify.py
```

필수테스트는다음의작은집합이다.

- risk 직접계산대모멘트전개; emptyclass/mixed0.
- svecnorm; patient removal sensitivity; repeatedimages를늘려도bound유지.
- D의point모멘트/dbar불변,공개pair불변,C변화.
- R_joint→difference moment의정확변환. 동일연산을다른기여로포장하지않음.
- no-grad에서response계산+two-pass derivative와one-pass일치.
- source/recipient weight·BN불변,recipient가distillloss경로에없음.
- 같은initialstate resume의정합성; bank간state누출없음.
- float→PNG→decode/intensity/label/pairbinding.
- Gaussian analytic calibration및fullreleaseledger. reference_math는안전난수구현이아님.

실행이검증됐다는사실과효용표본수를혼동하지않는다. 검사항목을불필요하게늘려성과로세지않는다.

## 21. 논문구성과 그림계획

### 예상제목(성과주장없는working title)

**Patient-Relational Distillation under User-Level Differential Privacy**

### 초록의증거채움순서

문제:반복촬영환자의정보를보호된재사용시각자료로전달.  
방법:점별정보+patient-local관계요약,one-shotDP,two-bankoutput및learnercontract.  
결과:다른encoder의B/C/D효과,동일DP비교,강한관계원리대조,전체비용.  
범위:관측label의centroid대조,특정cohort/학습계약;임상counterfactual이아님.

### 원고절

1.Introduction: 구체사용자=허가된기관이protectedQ를요약,수신자는원자료없이새시각학습자학습.  
2.Relatedwork: point/statisticdistillation,relation/covariance,patient-DP—각각이미있는원리인정.  
3.Method: exactpopulationweighting,twobanks,loss,receiverlearning.  
4.Analysis: standardriskidentity와privacybound,noise/image/transfergap.  
5.Experiments: sourcefaithfulness는보조,recipient/DP/근접비교는중심.  
6.Limitations:102mixed/weaklabels/sourcebias/선택이력/조건부CI/실제임상부적합.

### 그림·표

- Fig1. Q→boundedpatientmoments→DPsummary→두imagebanks→원자료없는여러recipient.
- Fig2. 같은점별자료,다른patientpairing의C/H변화;수학예와실제효과구분.
- Fig3. receiver별C−B,C−D paired effect와CI,source/target/BCE구분.
- Fig4. epsilon–utility–총시간/VRAM 및dim/pairmass의대가.
- Table1. claim/access/baseline차이.
- Table2. A/B/C/D/R와directreadouts의전체수치.
- Table3. DP결과,각release vs 공동composition.
- Table4. 제한적ablation과모든사용이력/비용.

이목록은필요한주장에연결된산출물계획이다. 결과가없는데기여3개를완료형으로작성하지않는다.

## 22. privacy·임상·연구선택실패방지

- patient-DP는Q의학습참여에관한정의된출력분포보장이다. 공개P의초상/의료재배포권을자동해결하지않는다.
- Public-derivedtemplate재배포조건과checkpointlicense를별도기록한다. 공개NIH를모사private로쓴실험임을명시한다.
- synth/png를정밀진단·훈련용실환자정답으로광고하지않는다.
- sourcefeature정합이좋다는이유로못생긴/좋은이미지를사후선별하지않는다.
- Q/Expert와의nearestneighborprivacy점수는보조검사이며DP대신사용하지않는다. Expert를privacygallery로미리소비하지않는다.
- weaklabelabsence,치료후흉관,AP/PA등이관계신호를설명할수있다. 주원인을확정하지않고가능한metadata-stratified분석을사전규정한다. 자료에없는device/timeannotation을만들지않는다.
- sourceoverlap검토는모델card와현재실험노출을구분하며,모든사전학습corpus가완전감사됐다고하지않는다.

## 23. 먼저 확정할 12개 runtime 항목

이항목은핵심설계를다시고르는질문이아니라로컬실행계약의빈칸이다.

1.actualrepo/sourcecommit 및Codex비교문서hash.
2.P/Q/V manifest및원영상/cacheconverterhash.
3.E_s image checkpoint SHA 및실제출력layer.
4.E_s differentiablepreprocess의공식경로대응.
5.E_t ImageNet checkpoint SHA,preprocess.
6.public PCA/scale artifacthash 및d16/d128.
7.private→summary query shape·빈집합동작.
8.banktemplates/seeds101,202,303 및Dpermutation.
9.optimizer/loss/pyramid/export형식.
10.공개profile로정한microbatch·실측시간상한.
11.runmatrix/정확한계산량·resume계약.
12.DP·expert·reserved authorization=false.

실제GPU성능은로컬에서W1/W2이후얻는새결과다. 이문서의CPU검산을그결과로바꾸어보고하지않는다.

## 24. 이번에 동봉한 것

- `reference_math.py`: construction-arrayrisk/sensitivity/shuffle/joint-equivalence/Gaussiancalibration/two-pass 검산.
- `reference_checks.json`:실제CPU검산결과. 환자입력0,모델weight0,생성효용0.
- `experiment_contract_template.yaml`:권고실행값및미확정SHA가명시된template.
- `planned_run_matrix.json`:phase별bank학습단위. 어떤run도실행됐다는뜻이아님.
- `CODEX_NEXT_PACKAGE.md`:다음구현패키지인수기준.

## 25. 출처와읽은범위

[S1] 저장소 Rwide 결과,고정commit e560b2c. 해당결과문서·명세·사용이력범위.
https://github.com/hko920920/patient-privacy-grad-cvpr2026/blob/e560b2c2b6a8413c842d5e691a20b19be2145237/cvpr_context/research_2026-09-10/TRACK1_REAL_SUPPORT_DIAGNOSTIC_RESULTS_20260917.md

[S2] 기존 PRRD v2 첨부전문(PATIENT_RELATIONAL_RISK_DISTILLATION_V2_20260918.md). 방법/기여/한계의기반.

[S3] 사용자가본대화에붙인Codex최종정정. CovMatch경계,projection선택성,동일알고리즘구별,다른encoderrelation-aware사용을반영.

[S4] CovMatch. §3 방법·정의및crossarchitecture부분. image-text연구이며여기의R_joint는원형재현아님.
https://arxiv.org/html/2510.18583v1

[S5] LGM. frozenfeature/gradientmatching·pyramid·transfer구분. 이기법자체신규성주장안함.
https://arxiv.org/html/2511.16674v1

[S6] Dosser 및DP-KIP. private학습신호/증류최적화분리,DPKRR의근접선행.
https://arxiv.org/html/2508.01749v1
https://arxiv.org/abs/2301.13389

[S7] Balle & Wang 2018. analytic Gaussian calibration. 수학공식의출처이며현재코드의보안인증은아님.
https://proceedings.mlr.press/v80/balle18a.html

[S8] BioViL-T officialmodelcard. image-model이별도로제공됨과MIMIC기반학습/연구목적사용범위.
https://huggingface.co/microsoft/BiomedVLP-BioViL-T

[S9] TorchVisionDenseNet121officialweights/transformdocumentation. 로컬설치버전과weightSHA는별도고정.
https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet121.html

[S11] TorchVision ViT-B/16 official weights 및 transform. 최종 확인용 수신자의 이번 제안값 근거다.
https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html

[S10] PATH. 사용자단위내부의존성보존이라는넓은아이디어의선행;표형시계열대상.
https://arxiv.org/html/2602.02766v1

이 계획은 전수 최초성 증명이나 모든 논문 재현 완료 선언이 아니다. 실제 비교 가능한 선행의 source commit과 변형 범위는 W3 baseline card에 완성한다.
