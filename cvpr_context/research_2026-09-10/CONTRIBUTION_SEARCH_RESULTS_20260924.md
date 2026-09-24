# 환자-DP 합성영상: 기여 탐색 실행 결과 — 2026-09-24

**판정: 추가로 개발할 구체적인 방법 후보와 실제 양성 결과를 확보했다. CVPR 기여 확정이나 최종 독립 검증을 완료했다는 뜻은 아니다.**

추천할 후보는 **한 source의 환자-DP 요약을 공개자료로 다른 encoder의 학습 목표에 옮기고, 그 목표를 기존 합성영상의 soft label에 담는 방법**이다. 이번에는 기존 숫자에 새 이름을 붙인 것이 아니라, 연산을 구현하고 직접 대조를 실행했다.

- 기존 두 feature-DP release와 각 128장의 PNG를 그대로 사용했다.
- 사적 Q에 대한 새 접근·release·잡음 추출·이미지 최적화는 0회다.
- DINO와 공개 P에서 연결한 ResNet18 목표로 라벨을 계산했다. DenseNet은 이 후보의 라벨 학습 목적에 넣지 않았다.
- DenseNet 개발 AUROC가 두 bank에서 **0.650018 → 0.686370**, **0.596743 → 0.663404**로 개선됐다.
- 기존 2,000회 paired patient bootstrap에서 해당 차이의 구간은 각각 **[+0.002859,+0.070587]**, **[+0.027893,+0.109059]**였다.
- source-only 라벨 학습 대조도 강했다. 그보다 나은지는 첫 bank에서 불확실하고, 두 번째에서는 AUROC/AP 모두 양의 조건부 구간을 얻었다.
- 개발 V를 활용해 후보를 좁힌 **탐색 결과**다. 구간은 고정한 bank에 조건부이며 방법 탐색과 DP 잡음 전체의 불확실성을 포함하지 않는다.

## 1. 실제로 달라진 연산

기존 A2는 BioViL와 DINO의 사적 신호를 함께 보호했다. 이번 후보는 사적 query를 늘리지 않고 **이미 보호한 DINO feature summary 이후의 처리**를 바꾼다.

1. 기존 DINO K4의 class별 보호 평균을 받는다.
2. 공개 P의 같은 환자/영상에서 얻은 DINO와 ResNet18 특징으로 affine 변환을 학습한다.
3. 보호 DINO 평균을 ResNet18 특징 평균으로 변환하고 공개 P의 second moment와 결합해 ResNet18의 목표 readout을 만든다.
4. 원래 PNG 128장에 붙일 soft label을 계산한다. 그 라벨로 학습한 DINO와 ResNet18 readout이 공개 P에서 각 목표 readout의 출력을 재현하도록 한다.
5. 하나의 고정된 이미지/라벨 묶음을 DenseNet에 넘겨, 기존과 같은 frozen-feature ridge 규칙으로 평가한다.

**최종 데이터 산출물은 기존 PNG + 새 soft label이다.** 목표 classifier나 공개 P를 downstream readout의 추가 학습자료로 넘긴 결과가 아니다. 단, 기존 평가에서 공통으로 사용하던 P 기반 receiver projection은 유지한다.

이번 결과는 처음 검사한 “DP 요약에서 receiver classifier를 바로 만드는 방식”보다 합성자료라는 원래 목적에 가깝다. 원래의 hard label 0/1은 그대로 보존했으며, 새 라벨은 별도 파일에 저장했다.

## 2. 핵심 결과: 같은 이미지와 같은 DP summary에서 라벨만 변경

모든 수치는 기존 V의 **영상별 AUROC/AP**다. 환자 단위 clustering은 bootstrap에 적용한다. 환자별로 점수를 먼저 평균한 AUROC라는 뜻이 아니다.

| 방식 | DenseNet DP1 AUROC / AP | DenseNet DP2 AUROC / AP | ResNet18 DP1 AUROC / AP | ResNet18 DP2 AUROC / AP |
|---|---|---|---|---|
| 기존 feature PNG + hard label | 0.650018 / 0.080832 | 0.596743 / 0.064188 | 0.601865 / 0.072737 | 0.610507 / 0.063724 |
| DINO teacher의 직접 pseudo-label | 0.653683 / 0.074752 | 0.614055 / 0.062433 | 0.672478 / 0.076845 | 0.657153 / 0.071733 |
| DINO만으로 functional label solve | 0.671617 / 0.083324 | 0.613363 / 0.060382 | 0.696634 / 0.088181 | 0.667659 / 0.075016 |
| **DINO + 공개 전달 ResNet18의 joint label solve** | **0.686370 / 0.081264** | **0.663404 / 0.074463** | **0.698797 / 0.083631** | **0.707042 / 0.084556** |

ResNet18은 joint label 학습에 사용했으므로 그 개선은 construction-model 성능이다. DenseNet도 과거부터 사용한 개발 receiver이므로 이번 loss에 제외됐다는 이유로 untouched receiver가 되지는 않는다.

### DenseNet의 필요한 직접 대조

| 비교 | DP1 ΔAUROC [95% patient CI] | DP2 ΔAUROC [95% patient CI] |
|---|---|---|
| joint − 기존 hard label | +0.036352 [+0.002859,+0.070587] | +0.066660 [+0.027893,+0.109059] |
| joint − DINO 직접 pseudo-label | +0.032687 [+0.011241,+0.055274] | +0.049349 [+0.030680,+0.068976] |
| joint − DINO-only label solve | +0.014753 [−0.002636,+0.032335] | +0.050041 [+0.024707,+0.076099] |

DP2의 joint − DINO-only label solve AP 차이도 **+0.014081 [ +0.005663,+0.028388 ]**였다. DP1에서는 AP가 약간 낮고 구간은 0을 포함한다. 따라서 “모든 지표와 모든 반복에서 더 낫다”고 쓰지 않는다.

보조 실험인 ResNet18-only 라벨 학습은 DenseNet에서 0.713564/0.648230이었다. joint가 DP1에서는 그보다 낮고 DP2에서는 높은 점추정이다. **0.713564라는 최고점만 골라 최종 방식의 성능으로 제시하지 않는다.** 둘을 혼합해 “더 안정적”이라고 주장하지도 않는다.

[전체 AUROC 그림](contribution_search_20260924/label_compiler_auroc.png) · [대응 차이 그림](contribution_search_20260924/label_compiler_contrasts.png)

## 3. 결과를 얻은 수식과 명세

### 공개 encoder 변환

DINO의 네 조건 PCA16 특징을 연결한 64차원 벡터를 사용한다. 공개 P의 환자/class별 방문 평균을 만든 뒤, 기존 공개 clipping 기준 C0=1.9061329951133719, C1=1.5917066731966771을 적용한다. 두 class용 변환 각각은 **해당 class에 속한 소수 환자만이 아니라 공개 P의 모든 환자/class 행**에서 학습한다.

환자 전체 가중치는 균등하고, 두 class가 모두 있는 환자는 그 가중치를 두 행에 나눈다. 672명, 675개 환자/class 행이다.

행벡터 표기로 다음 affine ridge를 푼다.

\[
(T_c,b_c)=\arg\min_{T,b}
 \sum_i w_i\|x_{ic}T+b-y_i\|^2+\lambda_c\|T\|_F^2,
\quad
\lambda_c=10^{-3}\operatorname{tr}(C_{xx,c})/64.
\]

여기서 y_i는 공개 P의 receiver 특징 평균이다. 보호 class 평균을 \(\tilde\mu_c\)라 하면 전달 목표는 \(\hat\nu_c=\tilde\mu_cT_c+b_c\)다.

receiver의 목표 classifier는 공개 P의 class-balanced/patient-equal second moment \(H_{P,r}\)로

\[
\bar w_r=(H_{P,r}+0.1I)^{-1}\tfrac12(\hat\nu_1-\hat\nu_0)
\]

를 계산한다. DINO source 목표 classifier는 기존 보호 평균 차이와 공개 DINO K4 second moment를 사용하는 ridge0.1이다. clipped target과 un-clipped public DINO second moment를 결합하는 현재 정의를 그대로 기록하며, 이것이 유일하거나 최적인 prototype classifier라고 주장하지 않는다.

환자 5-fold의 일반 feature 예측 R²는 약 0.18이다. 이것을 매우 정확한 전역 encoder 정렬이라고 설명할 수 없다. 실제로 필요한 것은 class 평균/판별방향 전달이고, 그 유용성을 개발 효용으로 확인한 것이다. P의 양성 환자가 6명뿐이라는 일반화 한계도 남는다.

### 이미지/라벨로 다시 담는 연산

고정된 synthetic feature 행렬을 \(Z_{S,r}\), 공개 P feature를 \(Z_{P,r}\), n=128이라고 하자. soft label \(\ell\in[-1,1]^{128}\)로 학습할 때 공개 P에서의 예측은

\[
B_r\ell,\quad
B_r=Z_{P,r}
(Z_{S,r}^{T}Z_{S,r}/n+0.1I)^{-1}Z_{S,r}^{T}/n
\]

이다. 목표는 \(t_r=Z_{P,r}\bar w_r\)다.

DINO와 ResNet18 각각의 공개 목표 RMS로 정규화한 다음 두 loss를 1/2씩 평균한다.

\[
\min_{\ell\in[-1,1]^{128}}
\frac12\sum_{r\in\{\mathrm{DINO,ResNet18}\}}
\frac{\|B_r\ell-t_r\|_{W_P}^2}{\|t_r\|_{W_P}^2}
+\eta\|\ell-y_0\|^2.
\]

정규화한 결합 design matrix를 A라 할 때 \(\eta=10^{-3}\|A\|_F^2/128\)로 고정했다. 두 DP bank 모두 같은 규칙이며 계수 탐색은 하지 않았다. 제약이 있는 최소제곱이므로 전체 라벨 solve를 “완전한 closed form”이라고 부르지 않는다. 내부 ridge readout만 closed form이다.

학습된 \(\ell\)은 회귀용 soft target이다. \((1+\ell)/2\)도 CSV에 기록했지만 **의학적으로 보정된 질병 확률이라는 뜻은 아니다.** 기존 hard label의 부호가 바뀐 항목은 DP1 39개, DP2 36개다. 데이터 증류용 감독신호이므로 실제 임상 재라벨링과 혼동하지 않는다.

downstream은 고정된 PNG와 이 라벨만 사용해 uniform 1/128 ridge0.1을 학습한다. 라벨을 hard threshold하거나 argmax class 수로 다시 가중하면 이번 실험과 다른 방식이다.

## 4. 가까운 선행과 남는 차이

**기본 연산의 신규성은 주장하지 않는다.** 아래 선행을 회피하면 이번 후보도 이전 보고서와 같은 문제가 생긴다.

| 선행 | 이미 제공하는 것 | 이번에 추가로 검증할 좁은 차이 |
|---|---|---|
| [DP kernel mean embeddings, Balog et al. 2018](https://proceedings.mlr.press/v80/balog18a.html) | 보호 평균을 public/synthetic points의 가중 표현으로 바꾸고 통계 계산에 재사용 | 공개 affine 평균 전달은 이 원리의 직접적인 특수화에 가깝다. 변환 자체의 최초성을 주장하지 않는다. |
| [DP-NTK](https://arxiv.org/abs/2303.01687) | 한 번 보호한 유한차원 신호를 생성 최적화에서 재사용 | private query를 여러 encoder로 확장하지 않고, public cross-encoder 목표를 만들어 image/label packet의 기능을 수정하는 효과 |
| [Dosser](https://arxiv.org/html/2508.01749v1) | 신호 보호/합성 분리, 조건 재사용, 부분공간으로 잡음 효율 개선 | source query의 부분공간을 다시 설계하는 대신, 이미 고정된 release 이후 receiver 목표와 label carrier를 수정 |
| [LGM](https://arxiv.org/html/2511.16674v1) | pretrained linear-gradient matching와 다른 모델로의 전이 | source의 feature/gradient loss에만 의존하지 않고 public receiver function을 보호 요약에서 구성 |
| [KIP / Label Solve](https://arxiv.org/abs/2011.00050), [Flexible Dataset Distillation](https://arxiv.org/abs/2006.08572) | ridge 기반 dataset/label distillation, 고정 이미지의 라벨 최적화 | 원자료의 target-label supervision을 요구하지 않는, fixed patient-DP summary와 공개 연결로 만든 heterogeneous functional targets |
| [DP-KIP/ScatterNet](https://openreview.net/pdf?id=84M8xwNxrc) | private KIP; 논문은 label optimization의 추가 privacy cost를 피하려고 label을 고정한다고 설명 | 기존 보호 통계만으로 label solve의 supervision을 공급하는 실용적 경로. “DP label 학습은 일반적으로 추가 비용이 불가능하다”는 주장은 아님 |
| [DPPL, AAAI 2025](https://arxiv.org/html/2406.08039v3) | 공개 pretrained encoder에서 private class prototype을 보호해 분류, public prototype 선택 | prototype의 source-space 추론을 넘어 별도 encoder 목표로 전달하고 재사용 가능한 이미지/라벨에 반영 |
| [POST, ICML 2025](https://arxiv.org/html/2506.16196v1) | public data를 이용한 protected prompt의 모델 간 전달 | “DP 후 공개자료로 모델을 바꾼다”는 일반 아이디어는 선행이다. 여기서는 unrelated frozen vision encoders의 class 통계와 dataset carrier를 다룸 |
| [HMDC](https://arxiv.org/html/2409.14538v1) | heterogeneous models의 공동 condensation/정렬 | 추가 모델이 Q를 읽지 않고 source-DP summary 이후 public target으로만 참여 |
| [PRISM](https://arxiv.org/html/2511.09905v1) | teacher 지식과 architectural prior를 분리한 다중 모델 증류 | multi-model prior 분리 자체는 새롭지 않다. 본 후보는 private query를 고정한 상태의 receiver function 전달을 검사해야 함 |

추가로 [2026년 CIM](https://arxiv.org/html/2607.00916v1)도 정보 추출/영상 복원 과정의 손실과 relabeling의 문제를 논한다. “합성영상이 원래 신호를 완전히 전달하지 못한다”나 “soft label이 중요하다”만으로 새 원리를 주장할 수 없다.

특히 공개 affine transport를 public signed weights로 바꾸는 dual 수식을 구현해 **최대 3.3e−15 이내로 같은 결과**임을 확인했다. 따라서 이를 새로운 privacy mechanism이나 새로운 평균 추정 법칙으로 포장할 수 없다.

남는 **방법 기여 후보**는 다음 문장이다.

> 한 source에서 이미 보호된 환자 통계만 사용해 공개 heterogeneous encoder의 functional targets를 구성하고, 추가 private query와 pixel optimization 없이 합성영상의 supervision을 다시 계산하여 다른 receiver에서의 효용을 회복한다.

이번에 조사한 원문에서 이 전체 연산과 실험 설정을 동일하게 제시한 직접 선행을 확인하지 못했다. 그러나 이 부재가 최초성을 증명하지는 않는다. 새 Gaussian 메커니즘이 아니라 **기존 이론에 기반한 구체적인 dataset post-processing 방법과 그 실증적 이득**을 검증하는 후보로 취급한다.

## 5. 이번 탐색이 원인에 대해 알려준 것

**잡음을 다시 뽑지 않고 같은 DP summary/PNG에서도 효용을 회복했다.** 따라서 이번 bank의 낮은 점수를 “이미 DP에서 정보가 사라져서 어떤 후처리로도 회복할 수 없다”고 설명할 수 없다. 다만 이것이 Gaussian noise가 무해하다는 뜻은 아니다.

처음 조사한 다른 후보들도 모두 남겨두었다.

| 조사 | 실제 결과 | 판단 |
|---|---|---|
| ridge second moment/condition number | synthetic readout의 condition number는 약 4–6; 공개 second moment로 교체해도 개선이 혼합적 | covariance 불안정성만으로 현재 문제를 설명하거나 해결하지 못함 |
| 고정 bank의 평균/second moment 교차 분해 | DenseNet의 두 bank 차이는 주로 class-mean 변화에 귀속되는 대수적 분해를 얻음 | 인과적 noise 원인 증명은 아님. 다음의 mean 전달 검사를 선택하는 근거 |
| public Jacobian 기반 방향 선택 | 선형 계산에서는 좋아 보였으나 공개 영상의 nonlinear 검사에서 단순 SVD보다 나쁜 경우 | 후보로 채택하지 않음. 좋은 선형 결과만 선택하지 않음 |
| DP summary를 receiver classifier로 직접 전달 | DenseNet 0.663363/0.684300, ResNet18 0.695604/0.697544 | source summary에 유용한 전달 가능 신호가 남았다는 실제 개발 근거 |
| 기존 PNG의 soft label만 변경 | joint 방식에서 두 DenseNet bank 개선; 단순 label solve도 상당한 이득 | label carrier와 heterogeneous public supervision의 효과를 구분할 필요 |

열벡터 표기로 \(g_c(x)=T_c f_c(x)+b_c+e_c(x)\)라고 쓰면, 전달 평균의 오차는
\[
\hat\mu_{g,c}-\mu_{g,c}
=T_c(\tilde\mu_{f,c}-\mu_{f,c})-\mathbb E[e_c].
\]
보호 오차와 공개 변환의 residual/shift를 나눠 볼 수 있다. 이는 설명용 항등식이며 새로운 정리로 세지 않는다. 실제 target에는 clipping/count 복원이 있어 \(\tilde\mu-\mu\)를 독립 등방 Gaussian으로 취급해서는 안 된다. 공개 P에서 residual이 작아도 Q에서 작다는 보장은 없다.

## 6. 공정한 대조와 현재 한계

공개자료만으로 라벨을 재계산하는 대조도 실행했다. 같은 DP PNG에 대해 label target만 (a) transported PUBLIC, (b) 실제 P classifier로 바꿨다. **이미지 자체는 DP에서 왔으므로 이 대조를 전체 public-only bank라고 부르지 않는다.**

ResNet18-only로 라벨을 계산해 DenseNet에 넘겼을 때, DP target은 실제 P classifier target보다 두 bank 모두 높은 AUROC/AP 조건부 구간을 얻었다. 반대로 DenseNet-only → ResNet18에서는 실제 P 기반 라벨 보정도 강해서, DP target의 추가 차이 구간이 0을 포함했다. 전체 결과는 compiler_control_results.json에 있다.

단순 source-only label solve와 joint의 차이가 DP1에서 불확실하다는 점이 현재 가장 직접적인 미해결 사항이다. 또 다음을 구분해야 한다.

- DenseNet은 이번 joint loss에는 없지만 이미 재사용한 개발 receiver다.
- ResNet18은 construction receiver로 바뀐 별도 exploratory 실험이다. 기존 기록의 역할을 소급 변경하지 않는다.
- 여러 후보와 대조를 V에서 본 전체 탐색은 적응적이다. 각 작은 계획을 결과 전에 저장했다는 이유로 전체 연구가 사전등록된 것은 아니다.
- 두 기존 DP releases에 조건부인 비교다. 새로운 환자 cohort나 합성 초기화 반복은 아니다.
- 전체 neural network fine-tuning/CE training으로의 전이는 확인하지 않았다. 현재는 frozen-feature ridge readout이다.
- 공식 Dosser/DPPL/DP-KIP 전체 재현과 경쟁을 완료하지 않았다. 내부 source-label 대조를 논문 이름으로 부르지 않는다.
- protected summary가 없어 PNG만 있는 소비자에게는 이번 재라벨링을 그대로 적용할 수 없다. 배포자는 자신의 DP summary를 보관하고 처리해야 한다.
- 새로운 label packet을 선택한 연구 과정 전체가 개별 ε8이라는 주장을 하지 않는다. 기존 여섯 release의 joint 회계를 없애거나 소급해 줄이지 않는다.

## 7. 검증과 실제 비용

검증 결과:

- 공개 feature target 재현 최대 오차: **4.51e−17**.
- affine ridge를 별도의 augmented least-squares로 풀었을 때 최대 차이: **3.54e−14**.
- public weighted-mean dual 표현 차이: **3.22e−15**.
- 최종 label packet의 primal/dual ridge 예측 차이: **5.00e−16**.
- 저장한 2,000회 patient bootstrap 중 0/17/1999번째를 sklearn sample_weight로 검증: 최대 **1.11e−16**.
- 원래 PNG 256개 hash 일치; 새 CSV label의 float round-trip 정확히 일치.
- V labels/환자/image 순서와 공개 source/receiver image 정렬 검사 통과.

비용은 기존 feature cache 사용 기준이다. public transport 최초 계산 약 2.6초, fixed-image label solve 네 묶음 약 1.4초, DINO ordinary-label 대조를 위한 기존 PNG 조건 forward 1,024개와 계산 약 13.9초였다. paired bootstrap과 manifest/그림 생성 비용은 별도이며 해당 JSON에 기록했다. **기존 PNG를 처음 제작한 약 35분을 포함해 방법 전체가 수초라는 주장은 하지 않는다.**

문헌조사, 실패 후보 검사, 개발 데이터 분석, 코드 작성과 결과 정리를 포함한 이번 작업은 약 80분 규모다. 새 full bank 학습은 하지 않았다. 실험 스크립트별 실제 시간은 결과 JSON의 seconds를 사용한다.

## 8. 저장한 산출물과 다음 연구 결정

모든 신규 코드는 별도 폴더 contribution_search_20260924에 있다. 기존 학습기와 이전 artifact는 수정하지 않았다. 로컬 산출물을 외부에 업로드하지 않았다.

핵심 파일:

- [joint_compiler.py](contribution_search_20260924/joint_compiler.py): 최종 두 construction function의 soft-label 계산.
- [joint_compiler_results.json](contribution_search_20260924/joint_compiler_results.json): 모든 지표와 대응 구간.
- [DP1 soft labels](contribution_search_20260924/label_packets/DP1_soft_labels.csv), [DP2 soft labels](contribution_search_20260924/label_packets/DP2_soft_labels.csv): 고정 이미지와 함께 사용할 라벨.
- [research_artifact_manifest.json](contribution_search_20260924/research_artifact_manifest.json): 원래 release·target·PNG·label hash와 readout 규칙.
- [transport_validation_results.json](contribution_search_20260924/transport_validation_results.json): 직접 요약 전달 결과와 독립 검산.
- [source_label_baseline_results.json](contribution_search_20260924/source_label_baseline_results.json): 단순 relabel/Label Solve 대조.
- [compiler_control_results.json](contribution_search_20260924/compiler_control_results.json): 공개 target 대조와 양방향 receiver 검사.
- [public_probe/nonlinear_results.json](contribution_search_20260924/public_probe/nonlinear_results.json): 채택하지 않은 Jacobian 후보의 실제 nonlinear 결과.

**추천은 이 후보를 다음 방법 개발의 기준으로 고정하는 것이다.** A2의 계수 탐색, DP3, 새로운 가설의 연속 추가, Reserved 확인으로 넘어갈 시점은 아니다.

다음의 가치 있는 검증은 새 잡음을 뽑는 일이 아니라, 같은 보호 summary에서 **source-only functional label solve와 public receiver를 추가한 label solve**의 차이를 기존 독립 synthesis realization 또는 공개 별도 task에서 고정 프로토콜로 확인하는 것이다. 이후에만 추가 cohort/receiver와 공식 근접 선행 비교를 계획한다. DINO-only 제거 대조가 이득을 대부분 설명한다면 그 부분은 기존 label distillation의 효과로 인정해야 한다.

현재 확보한 결과는 “기여가 이미 완성됐다”가 아니다. **보호 query의 복잡성을 키우지 않고도, 기존 합성자료의 학습 기능을 바꿔 전이를 개선하는 실행 가능한 방법과 양성 개발 증거**다. 이전의 private-data gain/recipe-gain 재명명보다 구체적이고, 반증 가능한 다음 연구 대상이다.
