# PRRD 종합 설계 정리 — 2026-09-18

상태: 논리적으로 구성 가능한 수정 설계의 통합 정리. 아래의 구체화는 실행 전 계약 변경 제안이며, FROZEN runtime이나 새 GPU 실행 기록이 아니다. 기존 v1, 사용자 제공 v3 원본, 기존 결과·계획 파일은 덮어쓰지 않는다.

**최신 설계 결정 — §19 우선:** 관계 개입 제한과 통계·기능 공동 정합을 다음 버전에 채택한다. §18의 J_func 지표 전용 권고는 수정 이력으로 남긴다. 기존 §6의 해는 제한 전 w1, §9의 손실은 공동 정합의 기반으로 읽는다. 계수·runtime은 미동결이며 이번 작업은 기록만 수행했다.

**후속 진행 — §20:** 단계별 실행·원고 계획을 작성하고 learner core를 구현·CPU 검산했다. kappa0.1 기반 허용량과 eta1을 새 시작값으로 지정했다. 실제 이미지 runtime은 아직 미동결이다. 위의 ‘기록만’ 문장은 §19 당시의 작업 범위다.

현재 원격 main 재확인: 0243f8b223d5df8f97f0ab62a2937304fd6b1d8a. 로컬 작업 폴더는 Git checkout이 아니므로 이 SHA를 미공개 로컬 v3 구현의 버전으로 사용하지 않는다. 로컬 소스는 별도 해시로 결속한다.

## 1. 현재 결정

- 주력은 PRRD, 환자 관계 보존 DP 증류다. Cached-condition diffusion을 병행하지 않는다.
- P+Q 전체의 점별 정보와 mixed 환자의 관계 정보를 함께 유지한다.
- 관계 전용 주제로 전환하지 않는다. 관계-only 제거 실험은 첫 패키지에 자동 추가하지 않는다.
- 관계 경로에서 ‘영상별 비선형 제한 후 대조’와 ‘대조 후 제한’을 구분한다.
- 출력은 합성영상 + synthetic label + virtual pair + 학습 계약이다.
- source에서의 위험 보존, 다른 encoder의 유용성, 환자-DP 아래 가치, 강한 기존 방식 대비 가치의 역할을 분리한다.
- 설명이 더 그럴듯해졌다는 이유로 주력을 다시 바꾸지 않는다. 실제 수학적 불일치, 직접 선행의 동일 연산, 사전에 정한 비교 결과와 자원 변경에 따라 범위를 판단한다.

연구 설계로 성립한다는 판단과 실험상 우위를 이미 확보했다는 판단은 다르다. 현재는 전자이며, 후자는 미측정이다.

## 2. 논문이 시험할 주장

P+Q의 점별 정보를 유지하면서 환자 대조를 비선형 제한 전에 구성해 보호하고, 그 정보를 합성영상과 관계로 전달하면, 강한 점별 증류 및 재보정된 관계 기준보다 원자료 없는 다른 시각 학습자에게 유용한 정보를 전달할 수 있는가?

순서 개선의 이유는 ‘어떤 정보든 보존하면 좋다’가 아니다. 개별 특징을 제한하면서 환자 대조에 필요한 크기·방향 정보가 사라질 수 있고, 차분을 먼저 하면 선택한 특징공간의 환자 공통 가산 변동이 관계 요약에 영향을 주는 경로를 없앨 수 있다는 것이다.

이는 기존 head/LoRA 실패의 원인 설명이 아니다. 그 실험들은 이 관계 통계 경로를 사용하지 않았다.

## 3. 기존 결과와 데이터 역할

| 항목 | 유지할 사실 |
|---|---|
| full64 head | 현재 구성의 사적 추가 합성 효용 관문 실패. 해당 DP 확대 중단 |
| 고정 LoRA | 같은 조건의 추가 사적 합성 효용 관문 실패 |
| 고정 분류기 진단 | 본 학습영상은 eval에서도 거의 완벽히 구별, 개발 일반화는 낮음 |
| Rwide | R1 AUROC .544610 → .679434; AP .051669 → .103542. 추가 실자료 직접 접근의 비DP 진단 |
| PRRD | 새 합성 효용·관계 효용·DP 효용 결과 없음 |

| 역할 | 환자 / 영상 | 용도와 경계 |
|---|---:|---|
| P | 672 / 813 | 공개 초기 영상·특징 변환·scale·공개 기준. 양성 환자 6, mixed 3 |
| Q | 2,027 / 5,097 | 보호 대상. 점별 정보 전체, 관계 정보 mixed 102명. 양성 환자 114 |
| V | 2,026 / 5,047 | 반복 사용된 method-development. 새 독립 평가가 아님 |
| Q80 | 80 / 320 | 과거 분기 기록 보존. 이번 Q와 동일하지 않음 |
| Expert | 532 / 810 | 전체 동결 이후의 별도 final. 현재 닫힘 |
| Reserved | 4,213명 | 현재 개발·추론·학습 금지 |

Q는 원 private_train이며 classifier-selection과 Rwide에 사용됐다. 새 public이나 unused validation으로 바꾸지 않는다. P와 Q의 class별 환자 통계 합과 count를 먼저 합쳐 정규화한다. 두 집단의 평균을 임의로 반반 섞지 않는다.

환자별 class 안의 영상은 균등 평균하고, 해당 class를 가진 환자들은 같은 질량으로 평균한다. mixed 관계는 P3+Q102=105명의 별도 모집단이다. image-level 음성 표기를 normal 확정으로 바꾸지 않는다.

## 4. 모델과 특징: 점별 경로를 유지하고 관계 경로를 명시한다

| 역할 | 설정 |
|---|---|
| Source | BioViL-T 공식 image model, projected_global_embedding, 고정/eval |
| Primary recipient | DenseNet121 IMAGENET1K_V1, classifier 이전 특징 |
| 확인용 recipient | ViT-B/16 IMAGENET1K_V1, 선택에서 분리된 후속 확인 |
| 호환성 | 기존 ImageNet ResNet18 BCE, 400step, seed 11/23/37 |

Source 차원은 16, recipient는 128이다. 모든 projection은 P만 사용한다. Receiver는 합성 loss/gradient, checkpoint 선택, 계수 선택에 사용하지 않는다.

### 4.1 기존 점별 경로

encoder e의 외부 L2 정규화 이전 출력을 h_e(x)라 하자.

현재 source wrapper는 N_s(h)=h/max(||h||,1e-12)를 사용하고, DenseNet recipient는 N_t(h)=h를 사용한다. 각 역할의 P-only 중심 mu_pt,e, PCA축 Pi_e, q95 scale s_e를 고정한다.

z_e(x) = T_{s_e}( Pi_e [N_e(h_e(x))-mu_pt,e] )
T_R(v) = v / max(R, ||v||), R>0.

Pi 표기는 열벡터 수식이며 실제 행벡터 구현은 전치 형태다. ||z_e||<=1이다. 이 점별 경로는 모든 새 비교군에서 동일하게 유지한다.

### 4.2 새 관계 경로에서 사용할 선형 특징

a_e(x) = Pi_e [h_e(x)-mu_raw,e].

Pi_e는 위와 같은 공개 축을 사용한다. 즉 source의 축이 공개 정규화 특징에서 학습됐더라도, 여기서는 그 저장된 고정 선형 축을 정규화 전 h에 적용한다. raw-PCA로 몰래 교체하지 않는다. mu_raw,e는 P-only raw feature 평균이며 환자 차분에서는 소거된다.

이 선택은 점별 경로와 공개 축을 유지하면서 관계 경로의 외부 정규화 순서만 분리하기 위한 구체화다. 원 encoder 내부의 비선형 연산을 제거했다는 뜻은 아니다.

현재 P source cache에는 외부 L2 이후의 값만 저장돼 있다. 정규화 전 norm은 거기서 복구할 수 없다. source h의 P-only 추출과 raw center/관계 scale 결속은 아직 필요하며, 이 문서에서 실행하지 않았다. 기존 normalized cache/projection을 raw feature cache라고 재명명하지 않는다.

### 4.3 공개 scale 선택 규칙 제안

E_R의 대조 scale, C의 raw 대조 scale, R_joint의 joint scale은 각각 P mixed 3명의 해당 벡터 norm에서 같은 empirical q95 규칙으로 정한다. floor는 1e-12. Q/V 성능으로 scale을 고르지 않는다.

세 환자는 많은 calibration 표본이 아니다. 실제 값과 작은 표본 범위를 기록한다. norm이 모두 0인 경우도 floor 규칙으로 정의하고, private 성능을 보고 예외를 만들지 않는다. 원 단위가 다른 벡터에 같은 숫자 R을 강제하는 대신 같은 공개 선택 절차를 적용한다.

PCA/q95의 정확한 가중·보간 규칙과 결과 해시는 runtime 동결 전에 기록한다. source와 recipient는 자기 공개 특징에서 자기 scale을 정한다. 이 E_R는 공개 q95 기반 clip 재보정이며, 모든 비영 대조를 단위 길이로 만드는 방식과 동일하지 않다. 따라서 이 기준만 비교하고 ‘모든 재정규화보다 우수하다’고 일반화하지 않는다. 완전 단위 대조 방식까지 우위를 주장하려면 그 방식도 사전 baseline 범위에 포함해야 하며, 이번 21bank 수에 실행한 것처럼 숨겨 넣지 않는다.

## 5. 세 관계 연산과 joint 기준

환자 i의 class별 z 평균을 m^z_ic, a 평균을 m^a_ic라 하고, 두 class를 가진 경우에만 관계를 계산한다.

| 연산 | 관계 벡터 | 의미 |
|---|---|---|
| E: 기존 순서 | e_i=(m^z_i1-m^z_i0)/2 | endpoint 제한 후 차분 |
| E_R: 재보정 | r_i=T_RE(e_i) | endpoint 제한 후 차분을 다시 제한 |
| C: 제안 순서 | d_i=(m^a_i1-m^a_i0)/2; r_i=T_RC(d_i) | 선형 특징의 대조 후 제한 |

셋 모두 norm<=1이며, point 통계는 동일하다. E_R의 공개 scale 규칙을 C와 동등하게 허용한다. 영상별 단위 norm과 cap 이하에서 선형인 clipping을 구분한다.

R_joint는 강한 관계 기준을 위해 다음처럼 구체화하는 변경을 제안한다.

u_raw,i=[m^a_i1;m^a_i0]/sqrt(2)
u_i=T_RJ(u_raw,i), U_i=u_i u_i^T
L=[I,-I]/sqrt(2), r_J,i=L u_i.

이는 개별 endpoint를 먼저 단위 정규화한 입력에만 제한하지 않고, 같은 raw 선형 endpoint 쌍을 환자 단위로 함께 제한해 평균·2차 모멘트를 보존하는 기준이다. ||u_i||<=1, ||L||op=1이므로 ||r_J,i||<=1이다.

중요: 이는 원 v3의 z-endpoint R_joint와 다른 target 계약이다. 원본은 보존하고, ‘raw joint-bound comparator로 수정’이라고 명시한다. 이 수정은 새로운 경쟁 알고리즘의 발명이 아니라 불리한 endpoint 처리에만 baseline을 제한하지 않기 위한 실행 전 구체화다. CovMatch 재현이라고 부르지 않는다.

차분 특징에 동일한 clipping/모멘트 정합을 적용한 표준 방법은 C와 같은 연산이다. 별도 약한 baseline으로 만들어 중복 집계하지 않는다.

## 6. 두 번째 정합성 수정: 학습 목적의 의미를 고정한다

점별 예측 점수는 모든 관계 방식에서 f_w(x)=w^T z_e(x)다.

관계 항은 각 arm에서 정의한 r_i를 사용하는 보조 학습 목적이다. C의 w^T r_i를 ‘두 정규화된 단일 영상 점수의 실제 차이’라고 부르지 않는다. 일반적으로 그 두 값은 같지 않다.

F_{e,a}(w) =
  1/4 sum_c E_patient,class E_image (w^T z_e(x)-(2c-1))^2
  + beta/2 E_mixed (w^T r_{e,a,i}-1)^2
  + lambda/2 ||w||^2.

beta=1, lambda=.1, intercept 없음. 동일 산출물의 beta=0 readout을 함께 계산한다. 관계 연산 a는 실험에서 바꾸는 명시적 요소이며, 나머지 위험 형식·계수·point score는 공통이다.

m_c=E m^z_ic, A_c=E E_image z z^T
mu_r=E r_i, C_r=E r_i r_i^T
H=(A_0+A_1)/2 + beta C_r
v=(m_1-m_0)/2 + beta mu_r
w=(H+lambda I)^(-1)v.

R_joint는 mu_r=L mu_u, C_r=L U L^T로 같은 위험을 계산한다.

각 arm 안에서 이 모멘트가 맞으면 해당 encoder와 명시한 보조 위험 전체가 맞는다. 이것은 임의의 다른 encoder 성능이나 AUROC 보장이 아니다. 다른 arm 간 비교는 관계 표현과 그것을 사용하는 전체 절차의 비교다. 합성영상만의 효과라고 단독 해석하지 않는다.

같은 산출물에서 beta0/1을 비교하면 관계 학습 규칙의 역할을 분리해 볼 수 있다. 새로운 합성 bank가 필요하지 않다.

## 7. 첫 번째 정합성 수정: 대응 교란 D

D는 C와 같은 point 통계를 쓰며, Q mixed 102명 안에서 raw negative centroid의 대응만 사전 derangement로 바꾼다. P mixed 3명은 그대로다. RC도 고정한다.

d_i^pi=(m^a_i1-m^a_pi(i),0)/2
r_i^pi=T_RC(d_i^pi).

clipping 전의 평균 d는 유지된다. clipping 후의 평균 r은 일반적으로 유지되지 않는다. 따라서 새 D에서 mu_r와 C_r, 그리고 v와 H가 모두 달라질 수 있다.

설명용 예:
p=(4,2), n=(0,2)
true d=(2,0), shuffled d=(1,1), 두 평균=1.
R=1로 제한하면 true r=(1,0), shuffled r=(1,1), 평균=.5와1.

C-D가 바꾸는 것은 실제 대응이다. 그러나 그 작용을 ‘관계 2차 모멘트만 바꾼다’고 설명하지 않는다. 기존 shuffle 코드의 제한 후 평균 불변 assert를 그대로 두거나, 억지로 평균을 맞춰 다른 대조로 바꾸면 안 된다. 제한 전 불변과 제한 후 실제 변화량을 각각 기록한다.

D는 첫 패키지에서 비DP 기전 대조다. 전역 교란에 C의 patient-local DP 감도를 복사하지 않는다.

## 8. 수학적으로 확보한 성질과 예측

- 고정 공개 선형 특징 a에서 환자 전체에 같은 가산 벡터가 들어오면 raw d와 C의 r은 변하지 않는다. 관계 경로의 불변성이다.
- point query까지 포함한 전체 출력·최종 분류기의 불변성을 주장하지 않는다.
- (M,1)/(M,-1)의 축소는 대조 재정규화로 회복할 수 있다. 이 사례만으로 E_R 우위를 주장하지 않는다.
- (11,0)/(9,0)의 endpoint 단위 정규화는 차분을 0으로 만들며, 이후 그 정규화된 특징만으로 복구할 수 없다. 모든 clipping/encoder의 실패 정리가 아니다.
- 완벽한 관계 요약도 단일 영상 AUROC를 보장하지 않는다. x+=a+1,x-=a-1, 독립 a~U[0,100]의 구성은 대조가 항상1이어도 양의 선형 점수 AUROC=.5198이다.
- 실제 정보 손실이 과제에 중요하다면 C가 E_R보다 의미 있는 성능 차이를 낼 것으로 예측한다. 성능 차이가 없으면 ‘순서가 반드시 필요하다’는 주장은 지지되지 않는다.
- 표준 patient-first gradient/moment 방법이 같은 연산이면 그 등가성을 인정한다.

같은 feature와 위험의 clean 실제/합성 H가 PSD이고 ridge가 같을 때:
||w_S-w_D|| <= (||v_S-v_D|| + ||H_S-H_D||op ||w_D||)/lambda.
이 표준 섭동식은 분석 도구이며 새로운 일반 정리로 세지 않는다.

## 9. 합성자료와 최적화

A/B는 class당64장, 총128장의 marginal bank를 쓴다.
E/E_R/C/D/R_joint는 marginal64장(class당32)+relation32쌍(64장)을 쓴다.
marginal 위험에는 marginal bank만, relation 위험에는 pair bank만 쓴다.

출력: 224x224 grayscale PNG128장, synthetic label, virtual pair index, preprocessing/projection/scale 선택 규칙·arm 연산·beta/lambda를 담은 learning_contract, 공개 provenance와 DP receipt.
실제 환자 ID·private pair 연결·원자료의 per-patient scale은 내보내지 않는다.

공개 P template만으로 logit residual pyramid를 초기화한다. 해상도8/16/32/64/112/224, 활성 step1/81/161/241/321/401. AdamW lr .01, wd0, FP32, residual std1e-3,500 successful updates. seeds101/202/303. 마지막500step 산출물만 평가한다.

J_point=1/4 sum_c (||mS_c-mT_c||²+||AS_c-AT_c||F²).
관계형 J_rel=1/2 (||muS_r-muT_r||²+||CS_r-CT_r||F²).
R_joint는 u/U의 평균·2차 모멘트 오차를 같은1/2 형식으로 사용한다.

J=J_point+J_rel+.01 pixel_anchor+1e-4 TV+.1 J_aug.
A/B에는 relation 항이 없다. 행렬 오차를 좌표 수로 추가 나눠 관계 기준을 약화하지 않는다.

증강 범위는 v3 유지: rotation±3°, translation±1%, scale .98–1.02, brightness .98–1.02, flip 없음. Pair는 같은 증강을 쓴다. 합성 증강 regularizer이지, 사적 증강 기대 위험과 동일하다는 주장이 아니다.

One-pass/two-pass는 전체 class/bank 분모를 동일하게 쓴다. 두 특징 경로 모두 같은 frozen source forward에 연결하되 이미지 gradient는 유지한다. BN/parameter는 불변이어야 한다.

평가는 PNG 재디코딩으로 한다. float matching, PNG matching, receiver 효과를 나누어 보고한다. finite-bank covariance rank 제약과 irreducible moment residual을 숨기지 않는다.

## 10. 한 번에 읽을 비교표와 실제 늘어나는 계산량

| arm | 정보·관계 target | 질문 |
|---|---|---|
| A | P point | 사적 정보 없는 기준 |
| B | P+Q point | 사적 점별 정보의 가치 |
| E | P+Q point + 기존 endpoint-first relation | 기존 PRRD 관계 경로 |
| E_R | P+Q point + 재보정 endpoint-first relation | 단순 scale 재보정으로 충분한가 |
| C | P+Q point + raw contrast-first bound | 수정 PRRD |
| D | C와 동일, Q 대응만 교란 | 실제 대응이 필요한가 |
| R_joint | P+Q point + raw joint-bound moments | 더 넓은 관계 정보를 쓰는 표준 원리 대비 가치 |

C-E_R는 같은 점별 정보·같은 feature layer·같은 공개 PCA축·같은 query 차원/감도·같은 계수에서 관계 표현의 순서를 바꾸는 비교다. R_joint는 차원이 다르며 자기 query의 같은 전체 DP 예산을 쓴다.

원 v3의15bank를 이 비교 전체를 포함한 것처럼 쓰지 않는다. 아래는 이번 통합안의 실행량 제안이며, 기존 실행 파일이나 예산을 자동 변경하지 않았다.

| 계산 | 원 v3 | 순서 대조를 포함한 통합안 |
|---|---:|---:|
| 조건 수 |5|7|
| 각 조건 초기화 |3|3|
| 합성 bank |15|21|
| PNG 수 |1,920|2,688|
| 합성 updates |7,500|10,500|
| source/recipient synthetic solves |48|72|
| RN18 호환성 run |45|63|
| RN18 updates |18,000|25,200|

72개 synthetic solves: 두 encoder 각각 A/B beta0 + E/E_R/C/D/R beta0·beta1, 이를3초기화에서 계산한다.

직접 실자료 readout은 encoder당 P point, P+Q point, E/E_R/C/R true relation4개, seed별 D3개로9개; 두 encoder18개다. 그중 P/PQ point와 C relation의6개는 기존 핵심 real reference 역할이다. Q를 읽는 recipient 실자료 readout은 연구자 측 trusted reference이며 원자료 없는 수신자 결과로 표시하지 않는다.

Receiver 평가 전에 모든 bank의 최종 hash를 고정한다. 같은 반복의 관계 arm은 가능한 동일한 template·초기 residual·augmentation stream을 쓴다. A/B와 paired arm의 고유 template 수 차이는 기록한다.

## 11. 평가와 고정된 주장별 판단

Primary는 DenseNet recipient의 image-level AUROC, AP 보조다. V patient-cluster paired bootstrap2,000draw. 모델별 metric을 계산한 뒤 평균하며 prediction ensemble로 바꾸지 않는다.

| 비교 | 입증 대상 |
|---|---|
| B-A | 사적 점별 정보의 추가 가치 |
| C-B | 관계 표현+학습 규칙을 포함한 전체 방법의 추가 가치 |
| C-D | 실제 사적 대응의 가치 |
| C-E_R | 재보정을 넘어선 순서 변경의 가치 |
| C-R_joint | joint-moment 관계 기준 대비 효용·비용 |
| 같은 PNG의 beta1-beta0 | 관계 학습 규칙의 역할 |
| source와 recipient 차이 | 통계 실현과 다른 encoder 전달의 구분 |
| RN18 BCE | pair 정보를 버린 이미지-only 호환성 |

각 관계 arm의 recipient도 자기 encoder의 P-only 변환/scale과 해당 arm의 r 연산으로 보조 위험을 계산한다. 원자료나 source raw moments를 받지 않는다. C-E_R는 전체 관계 표현 절차의 비교이며 이미지 자체만의 우위로 설명하지 않는다.

원 v3의 관계 개발 기준을 유지한다: C-B mean AUROC>=.01, C-D>0, 두 비교 각각3bank 중2개 이상 양의 방향, C의 AP가 B보다 낮지 않음. A/공개 real-only와의 차이도 모두 보고한다.

순서 개선 주장을 위한 새 사전 제안 기준은 C-E_R mean AUROC>0,3bank 중2개 이상 양의 방향, 평균 AP 비감소다. 이것은 추가 실험을 본 뒤 붙이는 조건이 아니라 새 순서 주장을 위해 실행 전에 기록하는 조건이다. 효과크기·구간도 함께 보고하며 통계적/임상적 유의성으로 부르지 않는다.

C-B/D가 좋아도 E_R와 같으면 관계 활용의 가능성과 순서의 독립 가치를 분리한다. C가 R_joint와 효용·비용 모두 비슷하면 독자적인 matching 알고리즘 우위를 주장하지 않는다. 동일 encoder에서만 성공하면 cross-encoder 주장은 미확보다. RN18만 실패하면 relation-aware reuse로 범위를 한정한다.

합성 초기화, classifier seed, 환자 bootstrap, DP noise는 서로 다른 반복이다. 결과 뒤 기준을 낮추거나 좋은 bank를 추가 선별하지 않는다. 음성 결과는 해당 고정 구성에 대한 결과이며 원인 전체의 확정이 아니다.

## 12. 환자-DP 계약

P는 공개 고정 입력, Q는 환자 한 명의 전체 기록을 보호하는 add/remove 인접성이다.

q_C,i =
[n_i0,n_i1,b_i,
 svec(A_i0),m_i0,
 svec(A_i1),m_i1,
 svec(r_i r_i^T),r_i].

빈 class/관계의 기여는0. svec offdiagonal은sqrt2배다.
점별 두 class 각각 norm²<=3, 관계 block norm²<=3이므로 결합 norm<=3.
C가 raw contrast를 먼저 만들더라도 최종 r의 norm<=1이면 이 상계는 유지된다.

| query | d16 차원 | 감도 상계 |
|---|---:|---:|
| B point |306|sqrt6|
| E/E_R/C point+relation |459|3|
| R_joint point+joint |867|3|
| relation-only 참고 |153|sqrt3|

relation-only의sqrt3를 full PRRD에 사용하지 않는다. E_R와 C는 같은 차원과 bound이므로 순서 자체의 보호 잡음 규모 우위를 주장하지 않는다.

Y=sum_Q q_i+Gaussian(0,s²I).
표준 analytic Gaussian으로 eps8,delta1e-5를 보정한다. P exact sum을 더하고 noisy count는 공개 floor1로 정규화한다. 합성은 Y/P만 사용한다.

합성은 raw noisy target에 squared matching한다. 직접 noisy readout만 H 대칭화→음의 eigenvalue0→ridge→solve를 사용한다. 보정 크기·조건수를 남기며 이미지 실현 가능성 보장으로 부르지 않는다.

Production Gaussian은 별도 secure sampler다. 테스트seed/RNG를 실제 보호 release에 쓰거나 공개하지 않는다. Private 원통계/gradient/환자별 scale/ID를 artifact로 내보내지 않는다.

하나의 Y에서3개 synthetic init와 여러 recipient를 만드는 것은 추가 private 접근 없는 후처리다. 여러 독립 Y·방법·feature-query·HPO를 공동 공개하면 composition이 필요하다. 비DP 선택과 산출물을 DP가 소급 보호하지 않는다.

기존자료를 통해 선택한 전체 연구 과정과, 고정한 메커니즘의 formal DP 보장은 구별한다. 실제 배포에서는 보호 cohort 접근 전 설계 고정 또는 선택 과정의 적절한 보호/회계가 필요하다.

## 13. 조건부 DP 패키지와 근접 선행

비DP의 관계/전달 근거를 본 뒤 DP로 진입한다. 이번 순서 주장에 E_R를 빼고 B만 DP 비교하면 재보정 반론에 답하지 못한다.

DP eps8 제안: B/E_R/C/R_joint × mechanism noise3 × synthetic init3 =36banks,4,608 PNG,18,000synthetic updates.
실제 private release는4×3=12개이며36개의 독립 privacy release가 아니다.
eps4는 같은36bank의 조건부 후속으로 두며 자동 실행하지 않는다.
D는 비DP에 남고 A는 공개 결과를 재사용한다.
원 v3의27bank DP 계획을 이36bank와 동일하다고 기록하지 않는다.

내부 arm만으로 선행 대비 우위라고 하지 않는다. Native-feature LGM 계열의 강한 pointwise 방법, 유효하게 환자 단위로 구현 가능한 DP-MEPF/DP-KIP/Dosser 중 가까운 방법, 직접 patient-DP 분류기/요약을 개발 단계에서 비교한다. 구현 범위·공개모델·환자 접근·HPO·privacy budget·전체비용을 baseline card로 고정한다.

표준 환자 차분 moment matching이 C와 정확히 같으면 동일 방법으로 인정한다. 새 DP 원리·새 moment matching 발명이라고 주장하지 않는다. 이 경우 책임질 기여는 구체적 정보 손실 분석과 보호된 시각 재사용의 실제 가치다.

## 14. Final과 비용

Expert는 nonDP/DP와 baseline, source/receiver, 계수·선택이력, artifact hash, privacy ledger, 통계 전체를 고정한 뒤 한 평가 캠페인으로 연다. 현재 접근하지 않는다. Reserved도 자동 사용하지 않는다.

원 v3 final family: 확인용 E_t2에서 nonDP C-B, nonDP C-D, eps8 C-강한 pointwise DP baseline.
순서의 우위를 최종 중심 주장에 포함하려면 C-E_R도 primary family에 명시해야 한다. 이 경우4개 대조의 Bonferroni 양측98.75% patient-cluster interval을 사용한다. 기존3개 family의98.333%를 그대로 쓰지 않는다. Final10,000draw, AUROC primary/AP secondary. 이는 사전 통계 계획 제안이며 final 결과를 보고 고르는 선택이 아니다.

총비용은 public preparation + private feature extraction/statistics + privacy + synthesis + export + recipient training/evaluation이다. Cold/warm cache, 실제 전체bank update 시간, VRAM/RAM/disk, 추가 private 접근, learner 재사용 수를 기록한다.

새 비DP 시간식:
T ~= T_private_features + 21*500*t_full128_synthesis
   + 63*400*t_RN18 + T_readout/export/verification.
추가 선행·DP 비용은 별도다. 현재 t_full128 실측이 없으므로 시간 보장은 하지 않는다.

## 15. 현재 실제 완료와 미완료

| 상태 | 실제 범위 |
|---|---|
| 완료 | 제공 v3 ZIP/해시 보존, 자료·공개 template·모델 결속 |
| 완료 | 제공 toy 대수 PASS, 새 CPU6검사 PASS |
| 완료 | P813 source/recipient 특징·PCA16/128. Q/V pixels 없음 |
| 완료 | Source frozen/eval 상태의 input gradient 존재 확인 |
| 미통과 | 실제 BioViL one-pass/two-pass gradient parity |
| 미완료 | full128 처리량, PNG model witness, RN18 runtime replay, 완성 W2 runner |
| 미실행 | 새 관계 경로용 source raw P cache/scale, Q/V 특징, synthetic bank, 새 효용 |
| 미실행 | DP, expert, reserved |
| 미정 | 새 패키지에 대한 숫자로 정한 시간 상한 |

마지막 실제 parity 최대 gradient 오차5.1300536e-5, peak.04923643, 선언 tolerance2e-7+5e-4*peak를 넘었다. Loss 오차가 작다는 이유로 PASS로 바꾸지 않는다. 원인은 아직 분리되지 않았다.

기존 profile 실패는 구현 정합성 문제이며 PRRD 효용 실패 결과가 아니다. 현재 실행 중인 프로세스는 없다.

이 통합정리에서 새 GPU/환자 영상 처리, 합성 학습, privacy release, remote push는 수행하지 않았다.

## 16. 구현 인수 기준과 원고 기여

다음 구현은 기존 패키지에 이 명세의 두 정합성 수정을 반영하는 것이다. 새 연구 주제 선택이 아니다.

- encoder API에서 raw h와 point z를 구분하고 같은 source forward를 공유한다.
- raw delta/r/E/E_R/raw-joint 경로와 P-only scale을 별도 hash로 결속한다.
- D는 clipping 전 평균 불변을 검사하고 clipping 후 평균·2차 모멘트 변화를 허용해 기록한다.
- 각 arm의 보조 위험을 직접 loss와 모멘트 해로 대조한다.
- 실제 encoder two-pass parity와 full128 profile을 통과한다.
- 수정21bank 계획의 실제 시간·계산량을 보고하며 기존15bank 승인으로 확대 실행하지 않는다.
- 전체 산출물 완료 후 정해진 평가를 한꺼번에 수행한다. DP/final은 현재 닫힌 상태를 유지한다.

기여 후보의 표현:
1) 환자 관계를 만들기 전의 비선형 제한이 소실시키는 정보와, 재보정으로 회복되는/되지 않는 조건의 분석.
2) 전체 환자 point 정보와 mixed 관계 정보를 분리해 보호하고, 두 image bank와 명시적 보조 학습 계약으로 전달하는 구성.
3) 실제 대응·재보정·joint/강한 pointwise 대안과 비교한 다른 learner의 효용, 같은 환자-DP 예산의 가치와 전체비용.

표준 수식과 Gaussian을 독립 신규 정리로 세지 않는다. 성능 결과가 없는데 기여를 완료형으로 쓰지 않는다. 세 번째 항의 실제 증거가 논문 주장의 범위를 결정한다.

## 17. 근거와 문서 연결

- [사용자 제공 v3 원본](spec_sources/prrd_v3_supplied_20260918/prrd_execution_plan_v3/PRRD_MASTER_EXECUTION_PLAN_V3_20260918.md).
- [순서 제안 검토 및 구성 반례](PRRD_CONTRAST_BEFORE_BOUND_REVIEW_20260918.md).
- [실제 공개 runtime 상태](spec_sources/prrd_v3_public_runtime_partial_record_20260918.json).
- [Rwide 실제 결과](TRACK1_REAL_SUPPORT_DIAGNOSTIC_RESULTS_20260917.md).
- [LGM](https://arxiv.org/html/2511.16674v1): frozen-feature distillation과 다른 model 전이의 선행. Gradient matching만의 encoder 과적합과 pyramid/augmentation을 구분한다.
- [CovMatch](https://arxiv.org/html/2510.18583v1): 관계 cross-covariance와 학습 목적 연결의 근접 선행이며 본 R_joint와 동일 재현이 아니다.
- [Analytic Gaussian](https://proceedings.mlr.press/v80/balle18a.html): Gaussian calibration의 표준 근거.
- [DP-MEPF 공식 구현](https://github.com/ParkLabML/DP-MEPF/blob/52e504d93d5f3c5bce8e627d1eb1ea8991dd29ea/code/dp_functions.py): norm/clip 및 layer별 제한을 구분한다.
- [Person-level mean estimation](https://arxiv.org/html/2405.20405v2): 개인별 집계 후 보호가 기존 접근에도 있음을 보여준다.

이 문서는 설계 선택과 실행 전 구체화를 종합한 기록이다. 최신 성능 결과 파일이나 원본 계약을 대체하지 않는다.

## 18. 추가 의견의 반영 범위 — 기록만, 2026-09-18

후속 변경: 아래는 당시의 판단 기록이다. J_func loss 보류는 §19에서 수정했으며, 나머지 표현·자료·평가 경계는 유지한다.

사용자 요청: 추가 의견에서 참고할 부분만 반영하고 기록한다. 새 구현·실험·실행 범위 확대는 요청하지 않았다.

### 유지하는 결정

PRRD, 전체 Q의 점별 정보, raw 관계를 사용하는 보조 학습 목적, 기존7조건/21bank 제안과 그 판정 기준을 유지한다. 이번 의견을 이유로 주력을 바꾸거나 새로운 비교군·학습 손실·성공 조건을 추가하지 않는다. 논리적으로 구현 가능한 설계라는 판단과 실제 효용·선행 대비 우위의 증거는 구분한다.

점별 특징이 z(x+)=z(x-)이면 어떤 w에서도 단일 영상 점수는 같다. raw 관계 경로가 차이를 보존하더라도 이 표현 충돌 자체를 가중치 변경으로 해소할 수 없다. 따라서 ‘추론 입력에서 이미 사라진 정보를 관계항으로 복구한다’고 주장하지 않는다. 이 한계는 기존의 다른 encoder 실제 판별 평가 범위에 포함되며 새 자동 중단 조건으로 만들지 않는다.

### 이미 정의된 수신자·기준선 규칙의 확인

- 수신자는 PNG·synthetic label·virtual pair·학습 규칙을 받아 자신의 특징공간에서 point와 relation을 계산한다. DenseNet은 자신의 raw feature와 P-only PCA128 및 공개 scale을 사용한다. Source16차원 관계벡터를 직접 전달하지 않는다.
- E_R의 현재 제안은 공개 mixed3명의 endpoint-first 대조 norm에서 정한 empirical q95와 공개 floor다. 수치·보간 규칙·해시 결속은 runtime 동결 전 항목이다. ‘공개로 정했다’는 사실을 충분한 보정·최적성의 증거로 부르지 않는다.
- R_joint는 u=T_RJ(u_raw)를 만든 뒤 r_J=L u로 정의한다. 따라서 제한된 u의 모멘트에 대해서만 mu_r=L mu_u, C_r=L U L^T가 정확하다. 이를 C의 T_RC(L u_raw)와 같다고 하거나 비선형 제한과 L이 교환된다고 가정하지 않는다.
- 통계 차원이 적다는 이유만으로 관심 있는 관계 통계의 DP 잡음이 더 작다고 주장하지 않는다. 실제 query·bound·projection·정규화에서의 잡음을 구분한다.

### 앞선 기능 보존 보완안에서 선택적으로 참고하는 부분

관계 추가에 따른 분류기 변화와 점수·순위 변화는 기존 readout의 해석 자료로 참고한다.

G=(A0+A1)/2, b=(m1-m0)/2,
w0=(G+lambda I)^(-1)b,
w_beta=(G+beta C_r+lambda I)^(-1)(b+beta mu_r)일 때,

w_beta-w0 = beta (G+beta C_r+lambda I)^(-1)(mu_r-C_r w0).

이 식은 관계 보조항이 분류기를 어떻게 바꾸는지 설명한다. 변화의 크기 자체를 AUROC 개선으로 세지 않는다.

J_func=(w_S-w_T)^T G_T (w_S-w_T)는 기능 차이를 설명하는 지표 후보로만 기록한다. Clean point 통계의 G_T에서는 해당 class-balanced 분포의 평균 점수 차이 제곱에 해당한다. Noisy 행렬을 쓸 경우에는 정의한 PSD 평가행렬과 안정화 범위를 명시해야 하며 raw indefinite 행렬 값을 자동으로 비음수 기능 오차라고 부르지 않는다.

이번에는 J_func를 합성 최적화 loss로 추가하지 않는다. Eta·학습량·optimizer를 바꾸지 않고, 별도 bank도 추가하지 않는다. 진단 지표의 활용이 새로운 성능 통과 기준이나 V를 먼저 열어 선택하는 절차를 자동으로 승인하지 않는다.

제외/보류:
- 단일 영상 점수 전체를 raw feature로 전환하는 변경.
- 기존 point query를 환자 통계 블록 clipping/가중 질량으로 교체하는 변경.
- Source readout 개선이 없으면 전체 합성을 자동 중단하는 새 규칙.
- 추가 loss·새 encoder·새 arm·추가 성공 조건.

이들은 현재 설계에 대한 작은 설명 보완을 넘어 목적·가중치·DP 또는 선택 절차를 바꾸므로 이번 기록에 실행 변경으로 반영하지 않는다.

### 현재 상태와 이번 작업의 경계

실제 BioViL W1 gradient parity는 여전히 미통과다. E_R/R_joint의 실행값·해시·runtime 대응은 아직 동결 전이다. 기존 Rwide/LoRA 효용 결과와 Expert/Reserved 보존은 변하지 않는다.

이번 작업은 이 절과 계획 JSON의 의견 반영 기록, 현재 상태·인수인계 기록만 수정한다. 실행 코드·arm·seed·학습량·loss·판정 기준은 변경하지 않는다. 새 모델/환자영상/GPU/DP 실행과 원격 업로드는 없다.

## 19. 관계 개입 제한과 통계·기능 공동 정합 채택 — 2026-09-18 후속 기록

사용자의 ‘그럼 다시 기록’ 요청에 따라 최신 권고를 다음 설계 버전의 결정으로 기록한다. **PRRD 유지 + 관계 개입 제한 + 통계·기능 공동 정합**을 채택한다. §18의 ‘J_func는 지표로만 사용하고 loss로는 넣지 않는다’는 이전 권고는 여기서 수정한다. 당시 기록·원본 계약·실험 결과는 보존한다.

이는 설계 결정의 갱신이다. 이번 작업에서 코드 구현, 계수 동결, gradient 검산 또는 새 성능 실험까지 완료한 것은 아니다. 새 모듈이나 연구 주제를 더 추가하지 않고 아래 두 변경을 구현할 범위로 삼는다.

### 19.1 관계 개입 제한

현재 점별 특징과 환자 가중치는 유지한다. Clean target의 점별 목적을 다음처럼 둔다.

\[
G_T=(A_{T0}+A_{T1})/2,\quad v_{0,T}=(m_{T1}-m_{T0})/2,\quad
M_T=G_T+\lambda I\succ0,
\]
\[
F_{0,T}(w)=\tfrac12 w^\top M_Tw-v_{0,T}^\top w+\mathrm{const},\qquad
w_0=M_T^{-1}v_{0,T}.
\]

기존 관계 보조항을 포함한 해는
\[
w_1=(M_T+\beta C_T)^{-1}(v_{0,T}+\beta\mu_{r,T})
\]
이고, 이를 그대로 쓰는 대신 \(d=w_1-w_0\)에 대해
\[
w_*=w_0+\alpha d,\qquad
\alpha=
\begin{cases}
1,&d^\top M_Td=0,\\
\min(1,\rho/\sqrt{d^\top M_Td}),&d^\top M_Td>0
\end{cases}
\]
로 정한다. \(\rho\ge0\)이며 \(d=0\)이면 \(w_*=w_0\)다.

따라서
\[
F_{0,T}(w_*)-F_{0,T}(w_0)
=\tfrac12\|w_*-w_0\|_{M_T}^2\le\tfrac12\rho^2.
\]

이는 주어진 관계 수정 방향의 크기를 제한하는 규칙이다. 전체 제약 최적화의 최적해라고 주장하지 않는다. 제한 대상은 명시한 점별 ridge 학습 목적이며, 새 환자 AUROC의 악화량은 아니다. 도움이 되는 관계 변화까지 줄일 수 있는 대가도 남는다.

### 19.2 기능 정합을 합성 loss에 포함

Target 통계에서는 위 규칙으로 \(w_*\)를 만들고, 합성자료의 통계에서도 같은 규칙을 적용해 \(w_S\)를 만든다. 합성 learner 내부의 제한에는 그 자료의 \(M_S\)를 사용하되, 두 기능을 비교하는 손실의 행렬은 **고정된 target \(M_T\)** 다.

\[
J_{\mathrm{func}}(S)=(w_S-w_*)^\top M_T(w_S-w_*),
\]
\[
L_{\mathrm{synth}}=L_{\mathrm{statistics}}+\eta J_{\mathrm{func}}+L_{\mathrm{image\ regularization}}.
\]

기존 점별·관계 모멘트 정합과 영상 prior/TV/augmentation은 유지한다. 분류기 하나만 복제하는 목표로 통계 보존을 대체하지 않는다. \(M_T\)를 합성자료에 따라 바꾸어 손실을 줄이는 경로는 허용하지 않는다.

Clean \(G_T\)에서는
\[
J_{\mathrm{func}}
=\mathbb E_{\mathrm{point},T}[(f_{w_S}(x)-f_{w_*}(x))^2]
+\lambda\|w_S-w_*\|^2
\]
이며 기대값은 기존 환자·class 가중치의 점별 분포다. 이는 같은 source 표현의 점수 기능을 맞추는 목적이다. 관계항 자체를 정규화된 두 영상의 실제 점수 차이라고 다시 정의하는 것은 아니다.

Source의 작은 선형계 solve와 제한 규칙을 거쳐 이미지까지 미분한다. \(d=0\) 처리, 제한 경계, 전체 bank 분모·동일 replay 상태를 명세하고 실제 one-pass/two-pass gradient 대응을 확인해야 한다. 이번 기록은 그 검사를 새로 수행한 결과가 아니다.

### 19.3 두 보강을 연결하는 범위

\(e_{\mathrm{dist}}=\sqrt{J_{\mathrm{func}}}\)이면 같은 \(M_T\)에 대해
\[
\|w_S-w_0\|_{M_T}
\le\|w_S-w_*\|_{M_T}+\|w_*-w_0\|_{M_T}
\le e_{\mathrm{dist}}+\rho,
\]
\[
F_{0,T}(w_S)-F_{0,T}(w_0)
\le\tfrac12(\rho+e_{\mathrm{dist}})^2.
\]

관계 개입과 합성 전달 오차를 연결한 표준 이차 목적의 성질이다. 새로운 일반화 정리·DP 원리·다른 encoder의 AUROC 보장으로 세지 않는다. 점별 표현이 같은 두 영상을 raw 관계 보조항만으로 구별할 수 없다는 §18의 한계도 그대로다.

### 19.4 동일 규칙·공정한 비교·DP

- Target, 합성자료, 수신자 모두 같은 제한 학습 규칙을 사용한다. 수신자는 자신의 특징·공개 변환으로 PNG와 가상 pair에서 통계를 다시 계산한다. Source 가중치나 비보호 사적 통계를 전달하지 않는다.
- A/B에도 자기 target에 대한 기능 정합을 제공한다. E/E_R/C/D/R_joint처럼 관계를 사용하는 비교군에는 같은 개입 제한 규칙을 적용한다. 정보·순서의 효과를 공통 최적화 보강의 효과와 혼동하지 않는다.
- 기존 PRRD / 제한만 / 기능 정합만 / 둘 다의 효과는 필요한 최소 제거 비교로 구분한다. 전체 조건을 자동으로 네 배 늘리지 않는다. 정확한 추가 run 목록·계산량은 아직 정하지 않았으며 기존21bank 숫자에 포함됐다고 간주하지 않는다.
- DP에서는 보호된 요약과 고정된 안정화 규칙만 사용한다. \(M_T\succ0\), \(w_0=M_T^{-1}v_{0,T}\), 제한 거리, 기능 손실, 상한이 같은 보정된 목적을 가리켜야 한다. Raw indefinite 행렬을 거리로 쓰지 않는다.
- DP에서 위 상한은 noisy surrogate에 대한 것이다. C의 자체 noisy point 해에 대한 제한이지, 더 작은 감도의 B release와 같은 분류기를 기준으로 한 우위 보장이 아니다. \(\alpha=0\)이어도 이미 공개한 관계 통계의 privacy 비용은 사라지지 않는다.
- 동일 보호 요약만 사용하는 계산은 후처리다. 추가 query·여러 release의 composition과 nonDP 선택 이력의 경계는 기존대로 유지한다.

### 19.5 이번에 닫은 범위와 아직 동결하지 않은 값

유지: PRRD, Q 전체 점별 정보, 현재 점별 특징·환자 가중치·query, raw 관계 보조항, PNG·pair·학습 규칙 출력, 수신자의 최적화/선택 제외, 기존 비교의 해석과 최종 평가 경계.

채택: 관계 개입 제한, 기존 통계 정합에 기능 정합을 실제 loss로 추가, target/합성/수신자 학습 규칙의 일치.

미채택 유지: raw 단일 영상 점수로의 전면 전환, point-block clipping에 따른 환자 질량 재가중, 관계-only 전환, source 결과만으로 전이를 자동 중단하는 새 규칙, 새 encoder·새 주제.

미정: \(\rho\), \(\eta\), 허용된 공개자료 또는 이미 보호된 요약을 사용하는 계수 선택 규칙, 최소 제거 비교의 정확한 run 목록·총비용. 이번 기록에서 임의 값을 채우거나 기존 성능 검증값으로 표시하지 않는다. 수신자/V/final 성능으로 계수를 골라 놓고 선택에 사용하지 않았다고 표시하지 않는다.

공개 W1의 실제 BioViL gradient parity는 아직 미통과이며 새 보강은 미구현이다. 기존21bank/조건부36bank는 이전의 미동결 계산량 제안으로 남고 이번 변경의 비용을 자동 포함하지 않는다. W2 시간 상한·runtime freeze·DP·Expert·Reserved 실행 상태는 바꾸지 않았다.

이번 작업: 문서·계획의 설계 개정 기록·상태·인수인계만 갱신. 모델/환자영상/GPU/합성 최적화/DP 실행0, 원격 업로드0. Rwide와 LoRA의 실제 결과는 그대로 보존한다.

## 20. 단계별 실행·원고 계획과 첫 core 구현 — 2026-09-18

사용자가 실제 구현을 한 단계씩 진행해 원고·재현물까지 완성할 계획을 요청했다. 최신 작업 순서는 [PRRD_PAPER_DELIVERY_PLAN_20260918.md](PRRD_PAPER_DELIVERY_PLAN_20260918.md)가 정한다. §19의 두 보강과 자료·평가 경계를 유지한다.

이번 시작 규칙: \(\rho^2=\kappa w_0^\top M w_0,\ \kappa=0.1,\ \eta=1\). 이에 따라 점별 목적 악화 허용량은 영점 분류기 대비 목적 개선량의 10%다. Target/합성/수신자는 동일 규칙과 자기 통계를 사용하고, 기능 loss만 고정 target M_T로 측정한다. \(w_0=0\)이면 관계 수정도 막히는 보수적인 선택이다. 실제 환자 성능으로 고른 수치가 아니며, 이전 §19의 계수 미정 상태 이후 새로 사전 지정한 시작값이다.

guarded_readout.py 및 test_guarded_readout.py를 추가했다. 독립 CPU 검사6개가 통과했다. 직접 목적 항등 최대 오차6.97e-16, 직접 점수 차이와 기능 loss 오차8.67e-19, toy whole/replay gradient 차이0이다. 실제 의료 encoder의 W1 PASS나 방법 효용 증거가 아니다. 기존 BioViL W1 미통과 상태는 유지한다.

새 전체 비DP 계획은 핵심7조건×3=21bank와 C의 cap-only/function-only/neither 9bank, 총30bank/3,840PNG/15,000synthetic updates다. Synthetic readout108개, trusted-real 참고20개, RN18은 핵심군만63run/25,200updates다. 기존21bank/15bank 계산량을 자동 확대 실행한 것이 아니며, runtime·실측 비용·숫자 시간 상한은 아직 결속 전이다.

논문 working manuscript와 claim–evidence ledger도 작성했다. 이번에는 새 core의 구성 배열 검사만 실행했으며 환자 pixel·실제 모델/GPU·합성 bank·DP·final은 실행하지 않았다. 기존 구현파일과 원본 계약·실제 결과를 수정하지 않았다. 다음 한 작업은 두 특징/관계 경로·기능 loss를 실제 공개 이미지에 연결하고 W1을 통과시키는 것이다.

참고: 시작 시 읽은 contracts.py와 patient_moments.py는 이전 기록 receipt의 hash와 차이가 있었다. 이번 작업에서 수정하지 않았고, 작업 시작의 읽기 내용과 현재 내용이 같음을 확인했다. 과거 receipt를 현재 코드 검증으로 재사용하지 않으며 다음 runtime은 현재 source hash로 다시 결속한다.
