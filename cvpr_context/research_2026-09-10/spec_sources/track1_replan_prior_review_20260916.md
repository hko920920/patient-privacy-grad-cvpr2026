# 방향1 재계획: 공개 참조 이후의 사적 추가 효용과 직접 선행

2026-09-16. 현재 큰 단계2의 제한된 선행 검토·후속 계획 자문이다. 새 실험·모델 실행·코드 변경은 하지 않았다. 아래 다섯 편의 지정 부분만 직접 확인했으며, 전 분야를 망라하거나 최신 SOTA를 재현했다는 뜻이 아니다. 기존 결과·계약·상태 파일은 수정하지 않았다.

## 판단

**“공개 적응 뒤에 사적 자료가 보탤 이득이 남는가”는 정당한 중간 판별 질문이다. 그러나 “기존 보호 설계는 공개·사적 성분을 분리하지 않는다”는 넓은 문제 정의는 직접 선행과 충돌한다.** 공개 gradient 중심화, 공개 Hessian으로 noisy curvature를 대체하기, 회귀 잔차를 반복해서 보호하기가 이미 제안되어 있다.

따라서 residual 보호를 새 주제로 확정하지 말고, **공개 전용·동일한 공개 정보를 사용하는 기존 보호법·사적 자료까지 쓰는 비DP 가능성 대조를 먼저 같은 작은 함수족에서 맞추는 1회 판별**로 둔다. 모든 선행의 실패를 입증해야 아이디어를 낼 수 있다는 조건은 필요 없다. 반대로 알려진 연산을 추가해서 현재 SSP를 계속 구제하는 것도 연구 계획이 아니다.

## 직접 확인한 근접 보호 연구 다섯 편

| 원문과 실제 확인 범위 | 원문이 이미 제공하는 연산·논리 | 우리에게 남는 범위 / 재사용 |
|---|---|---|
| **Nasr et al., Effectively Using Public Data in Privacy Preserving Machine Learning (ICML 2023)**. [공식 PMLR PDF](https://proceedings.mlr.press/v202/nasr23a/nasr23a.pdf), PDF pp.3–4, §3.1, Algorithms1–2, Propositions2/4; AppendixG의 공개 중심 민감도 논리도 읽음. | DOPE-SGD는 공개 gradient를 clipping 중심으로 사용한다. 공개 중심을 뺀 사적 gradient를 clip/noise하고 중심을 되돌린다. gradient 집중 조건은 효용 논리이며 privacy와 구분한다. | 공개 gradient reference/residual clipping 자체는 신규성이 아니다. 같은 공개 정보·환자 단위로 적용한 강한 비교 방법이다. 논문의 batch/center 복원 표현을 현재 add/remove·Poisson·고정분모로 복사할 때는 인접성에 맞게 다시 확인해야 한다. |
| **Ji et al., Differentially private distributed logistic regression using private and public data (BMC Medical Genomics 2014, 7(Suppl1):S14)**. [출판사 원문](https://link.springer.com/article/10.1186/1755-8794-7-S1-S14), Methodology의 Method description/Algorithms1–2, Experiments and results, Discussion. | noisy Hessian의 역행렬 및 고유값 threshold가 큰 효용 손실을 만들 수 있음을 설명한다. 공개자료만으로 Hessian을 계산하고 공개·사적 gradient로 Newton형 갱신을 한다. public-only도 비교한다. 공개/사적 분포가 다른 경우의 한계를 논의한다. | “Gram noise를 피하려고 공개 곡률을 쓰자”도 오래된 해법이다. 우리 작은 이차문제에 옮기는 것은 합리적인 기준선이지만 새로운 원리라고 할 수 없다. 원 논문은 logistic regression이며 우리의 환자 충분통계 Gaussian 메커니즘과 동일한 구현·회계는 아니다. |
| **Wang, Revisiting differentially private linear regression: optimal and adaptive prediction & estimation in unbounded domain (2018)**. [저자 arXiv v2 PDF](https://arxiv.org/pdf/1803.02596), §2, §4/Algorithm2(PDF p.12), AppendixB.3의 안정화 선택 설명. | AdaSSP는 Gram/응답 충분통계와 최소 고유값 관련 정보를 보호하고 데이터 의존적인 안정화를 DP 안에서 처리한다. 조건수·데이터 크기·정규화가 효용을 좌우한다. | 현재 공개 grid의 joint-SSP가 모든 강한 사적 회귀를 대표하지 않는다. noisy Gram 안정화나 일회 통계 보호 자체가 신규 기여가 아니다. 원형은 행 단위 회귀이며, 환자 전체를 한 단위로 바꿀 때 보호 질의와 민감도 계약이 필요하다. |
| **Tang et al., Improved Differentially Private Regression via Gradient Boosting (SaTML 2024)**. [저자 원문 v2](https://arxiv.org/html/2303.03451v2), §3.1–3.3, Algorithm1, §4와 AppendixA.2. [공식 학회 게재 목록](https://satml.org/2024/accepted-papers/), [저자 코드](https://github.com/shuaitang/BoostedAdaSSP). | BoostedAdaSSP는 noisy Gram을 재사용하면서 현재 예측의 잔차를 clip하고 새 응답 질의를 보호한다. 새 residual 질의에는 합성 비용이 든다. 사적으로 고른 clipping bound와 데이터 독립 bound의 차이, clipping 편향을 직접 다룬다. | residual만 보호하거나 통계를 재사용하는 논리는 이미 존재한다. 현재 SSP의 열세를 전체 SSP 계열의 한계라고 하면 안 된다. 논문 코드/보호 단위 확인 후 적용 가능한 회귀 기준선이다. 원래 회귀 관측행을 우리의 픽셀·draw 독립 환자로 세면 안 된다. |
| **Tsai et al., Differentially Private Fine-Tuning of Diffusion Models / DP-LoRA**. [저자 arXiv v1](https://arxiv.org/html/2406.01355v1), §§3.1–3.2, §4.4, AppendixA/C를 확인. [ICCV 2025 공식 게재 페이지](https://openaccess.thecvf.com/content/ICCV2025/html/Tsai_Differentially_Private_Fine-Tuning_of_Diffusion_Models_ICCV_2025_paper.html). | 공개 pretrained autoencoder/LDM에 QKV·output-projection LoRA를 붙여 DP-SGD와 noise multiplicity로 미세조정한다. trainable parameter·생성 품질·학습 비용을 함께 다룬다. | 작은 보정 파라미터와 공개 사전학습 자체는 이미 있다. 현재 output head의 cache 기반 backbone-backward0와 생성 품질/총비용 절충은 비교할 가치가 있으나, 평균 denoising MSE만으로 DP-LoRA보다 낫다고 할 수 없다. 원문 per-sample DP를 patient DP로 동일시하지 않는다. |

**접근 범위 표시:** DOPE는 공식 학회 PDF, Ji는 출판사 본문, AdaSSP는 저자 PDF를 읽었다. BoostedAdaSSP는 OpenReview 재접속이 browser challenge라 저자 arXiv v2의 지정 절로 확인했고, 공식 SaTML 목록으로 게재 정보를 확인했다. DP-LoRA의 이번 상세 읽기는 2024 arXiv v1이다. CVF 최종 PDF fetch가 실패했으므로 2025 최종 수치표까지 대조했다고 하지 않는다. 저자 코드 링크는 재사용 경로를 제시한 것이며 이번에 코드를 실행·재현하지 않았다.

## 현재 관측이 정당화하는 질문

현재 전달된 full head 개발 MSE는 base .17964495, public-only32 .1740411864, private-only80 비DP .1740069294, DP-SGD .174388846, SSP .178893924다. 이 값은 한 NIH 코호트·고정 작은 선형 보정층·개발40명·평균 denoising MSE의 관측이다.

- 공개 전용이 매우 강하고 private-only 비DP와의 차이가 약 .0000343이라는 사실은 **사적 추가 이득을 먼저 분리할 이유**다. 모집단 동등성이나 사적 데이터의 무가치를 증명하지는 않는다.
- public+private 결합 목적을 고정해 푼 결과는 아직 없다. 그 비DP 해도 해당 학습 목적의 최적해이지 개발·미래 생성 품질의 엄밀한 상한은 아니다. “ceiling”보다 “고정 함수족의 비DP 가능성 대조”라고 부르는 편이 정확하다.
- SSP의 큰 floor 손실은 특정 구현에서 확인된 문제다. 공개 곡률이나 AdaSSP 등 기존 대안이 그 문제를 해결하면 그것은 기존 해법이 유효하다는 결과이며, 새 알고리즘 필요성의 증거가 아니다.
- public-only는 ε=0의 무적응 기반 모델과 다르다. 이미 공개 의료자료로 적응한 강한 모델이다. 사적 자료를 더 쓰는 방법은 이 비교점을 명시해야 한다.
- CDI/MoFit의 patient-FPR/공격 TPR은 이 보호 방법의 효용 우위를 정하는 기준이 아니다. 먼저 동일 환자 인접성·ε/δ·공개 정보 권한을 맞추고 품질과 총비용을 비교한다. 공격은 필요할 경우 별도 노출 평가 근거다.

## 제안 하나: 사적 추가 가치의 판별을 먼저 닫는 cached 비교

**질문:** 동일 공개 정보와 작은 보정 함수족 안에서, 사적 환자 자료가 제공하는 추가 효용을 정해진 patient-DP 예산으로 실제 보존할 수 있는가? 이는 residual 보호 방법 논문의 확정 주제가 아니라 중간 go/no-go 질문이다.

다음 한 번의 패키지에서 public-only, 사전에 고정한 patient-equal public+private 비DP 목적, 동일 공개 초기화/참조를 받은 강한 기존 patient-DP 기준선을 같은 개발 평가에 놓는다. Public-init DP-SGD는 최소 비교이며, 공개 gradient 중심화 또는 공개 곡률을 쓰는 기준선은 위 직접 선행의 적용으로 명명한다. 한쪽만 공개 정보나 유리한 초기값을 더 주지 않는다. 초기 중심을 이동하면서 ridge를 λ||W_public+V||²에서 λ||V||²로 바꾸면 목적도 달라지므로 별도 변경으로 밝힌다.

이 패키지는 새 backbone 실행 없이 기존 통계에서 판단 가능한 부분에 한정할 수 있다. 새 방법을 덧붙이는 일이 아니라 **보호할 추가 가치의 존재와 기존 해법으로 보존 가능한 정도를 분리**하는 일이다. 공개 곡률 mismatch를 인위적으로 만들거나 좋은 결과가 나오는 질환·사진·seed를 골라 새 문제로 포장하지 않는다. 환자 수·공개 비중·표현이 바뀌면 결론도 달라질 수 있지만, 그 가능성을 이유로 이번에 무계획 sweep을 예정하지 않는다.

재사용 가능한 기존 해법으로 차이가 모두 설명되더라도 작은 출력 head의 실질적 효율 가능성은 별개다. 이 경우 후보 중심은 새로운 residual DP 원리가 아니라 **실제 생성에서 의미 있는 품질을 유지하면서 backbone 학습 비용을 줄이는가**로 옮길 수 있다. 그 역시 현재 MSE만으로 성공이라 판정하지 않는다.

## 중단/대체 기준 하나

**사전에 정한 실용적인 추가 효용 기준에 비DP 가능성 대조도 못 미치거나, 공개 참조를 받은 기존 DP 기준선으로 잔여 이득이 이미 설명되면 “새 residual 보호 메커니즘”을 중심으로 한 확장을 멈춘다.** 이는 연구 전체 포기가 아니다. 작은 head의 실제 생성 품질–총비용 질문으로 제한해 재설계하거나, 이 결과를 음성 개발 근거로 보존한다.

반대로 추가 효용이 확인되면 먼저 그 이득을 기존 public-aware 보호법이 얼마나 보존하는지 보고, 남는 구체적 tradeoff가 있을 때 새 연산을 제안한다. 모든 선행이 실패해야만 제안할 수 있는 것은 아니다. 그러나 residual 존재만으로 방법 논문, residual 부재만으로 평가 논문이 자동 성립하지도 않는다. 평가 기여라면 중요한 보호 선택을 바꾸는 재현 가능한 발견·일반화 범위가 별도로 필요하다.

현재 자료를 이미 본 뒤 임의의 수치 gate를 발명하지 않는다. 실용 기준·집계 방식·새 평가 사용 범위는 다음 계약에서 고정하고, 개발40명을 최종 미열람 test처럼 부르지 않는다.

