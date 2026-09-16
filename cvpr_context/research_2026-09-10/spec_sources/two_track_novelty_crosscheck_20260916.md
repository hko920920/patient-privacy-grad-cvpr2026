# 두 방향의 논문 후보: 독립 신규성·논리 점검

작성일: 2026-09-16. 범위: 보호 설계 효율과 보호 평가 개선을 모두 후보로 검토한다. 졸업논문의 목표 변경을 허용한 최신 지시를 따른다. 새 GPU·학습·통계 실험은 수행하지 않았다. 선택한 1차 문헌과 기존 원문 검토 기록에 근거한 한정 검토이며, 포괄적 신규성 확정이 아니다.

## 1. 판정 요약

| 방향 | 그대로는 이미 있는 주장 | 남겨 검토할 만한 구조 | 현재 증거 수준 |
|---|---|---|---|
| 1. 보호 설계 효율 | user DP, 공개 사전학습, 여러 사진 평균, noise multiplicity, 분산에 따른 연산 배분 | 같은 patient-DP·실제 연산 예산에서 **최종 clipping된 환자 업데이트의 추정 오차**를 줄이도록 영상 선택과 noise 반복의 중첩 배분을 설계 | 조건부 설계 후보. 기존 분산 배분과 다른 결정을 해야 하는 조건 및 실제 효용 이득 미검증 |
| 2. 보호 평가 개선 | 방어에 적응한 공격, 미검출은 보호 증거가 아님, 접근권별 위험, maximin 방식의 적응적 평가 | **추정 중인 FPR 기준과 공유 계산을 함께 고려하는 제한된 공격군 내 방어 선택**: 오선택 확률 또는 같은 선택 신뢰도에 필요한 감사 비용 개선 | 조건부 설계 후보. 기존 Maximin-LUCB/Racing에 단순 환원되는지부터 검토해야 함 |

둘 다 제안 가능한 후보지만, 현재 관측된 E/U·환자 개입·clipping 사례가 이 둘의 해결 효과를 입증하지는 않는다. 모든 기존 방법의 실패를 먼저 증명할 필요는 없다. 대신 어떤 기존 설계의 어떤 구체적 선택이 잘못되거나 비효율적인지를 좁혀, 제안 변경이 그 문제를 해결하는지 보여야 한다.

## 2. 방향 1: 이미 점유된 구성요소

### 2.1 ELS/ULS와 noise multiplicity는 따로가 아니라 함께 강한 비교 상대다

- **Charles et al., ELS/ULS**: 로컬 B04 v1의 §2 pp3–4는 user-level accountant와 고정 연산 `B = M × G`에서 사용자 수 M·사용자당 영상 수 G를 비교한다. §3 p5는 gradient diversity와 noise variance를 연결한다. 여기서 clipping을 Lipschitz 상한으로 설정하여 no-op으로 둔 것은 **이론 분석의 조건**이다. 이 논문 전체가 실제 clipping을 다루지 않았다고 바꾸면 안 된다. 공개 최종 출판명은 *Learning with User-Level Differential Privacy Under Fixed Compute Budgets*, SaTML 2025이다. 이번에 최종판 전체를 다시 읽지는 않았다. [저자 공개본](https://arxiv.org/abs/2407.07737), [저자 설명](https://research.google/blog/fine-tuning-llms-with-user-level-differential-privacy/).
- **Dockhorn et al., DPDM**: §3.2 p6 Eq.7은 동일 이미지의 K개 noise/timestep loss를 평균하여 clipping 전에 저분산 gradient를 만들고, 추가 privacy 비용 없이 연산을 더 쓰는 방법이다. Appendix D.1–D.4 pp29–30은 1/K 분산, 선형 연산 증가, augmentation과의 차이를 설명한다. F.1 p33은 동일 identity의 여러 사진에도 주 실험은 per-image DP라는 한계를 이미 적는다. 따라서 ‘여러 사진을 가진 사람에겐 이미지 DP가 충분하지 않다’는 동기도 새로운 발견이 아니다. [최종 TMLR 원문](https://openreview.net/pdf?id=ZPpQk7FJXF).
- **Ghalebikesabi et al.**: B15 pp5–6은 공개 diffusion 사전학습, augmentation와 timestep multiplicity의 결합, 중간 timestep에 집중하는 비균일 학습을 사용한다. 의료 Camelyon17도 포함한다. 공개 모델·의료영상·반복 noise의 조합이나 timestep 조정만으로 새로움을 주장할 수 없다. [저자 공개본](https://arxiv.org/abs/2302.13861).
- **De et al.**의 augmentation multiplicity/공개 사전학습, **DP-LoRA**의 attention·projection 저랭크 DP fine-tuning도 기본 비교 조건이다. DP-LoRA B08 §§2–3을 재확인했다. De 원문 전체는 이번에 재독하지 않았으며 DPDM의 직접 대조와 저자 초록으로 역할을 확인했다. [De et al.](https://arxiv.org/abs/2204.13650), [DP-LoRA ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Tsai_Differentially_Private_Fine-Tuning_of_Diffusion_Models_ICCV_2025_paper.pdf).

### 2.2 ‘적응적으로 배분한다’에도 기존 해법이 있다

**DPIS**는 중요도 표집으로 sampling variance와 필요한 DP noise를 줄이고, 표집 정보의 프라이버시 비용·효율·adaptive clipping을 함께 다룬다. 초록과 §§2–3 방법 도입을 읽었다. 환자 안의 diffusion 반복 수를 정하는 문제와 동일하지는 않지만, 중요도·분산에 따라 더 계산하는 발상은 차단한다. [CCS 2022 저자 원문](https://arxiv.org/abs/2210.09634).

**ASC**는 그룹별 adaptive sampling과 clipping을 함께 바꾸어 hard group의 utility와 privacy를 조절한다. §3–5 pp3–4와 알고리즘을 확인했다. 여기서 group은 demographic group이고 adjacency는 한 그룹 내 **한 record 교체**다. 이를 patient 전체 DP 선행으로 오기하면 안 된다. 그러나 ‘어려운 집단에 더 배정하되 clipping으로 보장을 맞춘다’는 넓은 표현은 이미 겹친다. [2026 공개본](https://arxiv.org/abs/2602.10820).

**DP-SCAFFOLD** §3 p4는 noisy local gradients에서 client/server control variates를 갱신하여 이질성과 drift를 줄인다. 이 원형을 ‘공개 pretrained-gradient reference’라고 부르면 부정확하다. 공개 gradient를 쓰는 별도 변형이 있다면 그 출처를 별도로 확인해야 한다. [AISTATS 2022](https://proceedings.mlr.press/v151/noble22a/noble22a.pdf).

**Clipping geometry/bias** 자체도 선행이다. Chen et al.은 gradient 분포의 대칭성·기하로 clipping bias를 분석하고 보정한다. 이번에는 저자 초록·도입의 연구 결과 범위를 확인했고 전체 정리는 재검증하지 않았다. [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/9ecff5455677b38d19f49ce658ef0608-Paper.pdf). LOCKS의 user-private client sampling도 추가 인접 선행으로 검색되었으나 본문 미독이며, 신규성 판단의 적극적 근거로 삼지 않는다. [저자 공개본](https://arxiv.org/abs/2212.13071).

**CARV**는 동료가 추가로 찾은 더 가까운 2026 선행이다. 이번에 저자 초록을 직접 확인했다. frozen diffusion teacher의 중첩 MC에서 비싼 upstream 계산을 여러 noise draw에 재사용하고, timestep importance sampling·stratification·실제 계산비용을 함께 다룬다. 특히 single-step distillation에서는 gradient variance를 크게 줄여도 FID가 개선되지 않았다고 보고한다. 따라서 ‘중첩 noise와 계산비용을 고려한 분산 배분’과 ‘분산 감소가 성능 개선을 보장하지 않음’까지 이미 있는 결론이다. 본문 세부는 방향 1 담당 동료가 확인하며, 이 노트에서는 초록 이상의 독립 재독을 주장하지 않는다. [Variance Reduction for Expectations with Diffusion Teachers](https://arxiv.org/abs/2605.21489).

## 3. 방향 1: 남길 수 있는 구체적 후보와 치명적 반론

후보는 ‘환자마다 더 많은 noise’가 아니라, **동일한 bounded patient contribution을 더 적은 연산으로 추정하는 배분**이다. 한 환자의 실제 여러 영상과 동일 영상의 diffusion randomness를 서로 다른 변동 원인으로 다루고, 최종 `clip_C(gradient estimate)`의 오차 감소에 유효한 곳에 계산을 쓴다.

독립 표집·동일 가중치라는 단순 조건에서, 사용자 내부 gradient 평균의 covariance는 영상 간 성분/G와 영상 내부 noise 성분/(G×r)로 나눠 볼 수 있다. 이 분산 분해·Neyman식 배분은 기본 확률론이므로 기여가 아니다. 유한 영상 풀·중복·without-replacement·영상별 noise 이질성에는 이 단순식의 수정이 필요하다.

남는 차이를 보일 후보는 **raw variance를 줄이는 배분과 실제 clipping 이후의 업데이트 오차를 줄이는 배분이 달라지는 조건**이다. 예를 들어 norm이 C보다 충분히 큰 평균 gradient μ에서 clipping의 국소 Jacobian은 `C/||μ|| × (I − vvᵀ)`, `v=μ/||μ||`다. 따라서 radial variance와 tangential variance는 같은 raw variance라도 업데이트에 미치는 영향이 다르다. 이는 표준 미분 사실이며 새 정리가 아니다. 경계 근처에는 이 선형 근사가 부족하다. ‘C 근처만 더 반복’하는 규칙이 자동으로 최적이라는 결론도 나오지 않는다.

유망하려면 다음을 함께 해결해야 한다.

1. **문제 차이의 증거**: 실제 유효한 operating point에서 uniform multiplicity·raw-variance allocation·좋게 조정한 ELS/ULS가 낭비하는 계산을 예측하고, 제안 규칙이 그 선택을 바꾼다. 단순 clipping 비율이나 표준오차 감소만으로 충분하지 않다.
2. **privacy의 정확성**: 각 환자가 자신의 내부 pilot으로 r을 정하고 fresh final draws로 추정한 뒤 고정 C로 clip하는 것과, 전체 환자가 하나의 private-dependent budget을 경쟁하는 것은 다르다. 후자는 한 환자 변경으로 다른 환자들의 배정·업데이트도 변할 수 있어 ‘마지막에 각자 clip했으므로 같은 sensitivity’가 자동 성립하지 않는다. 공개/DP controller, 고정 개별 cap 등 어떤 구조를 쓰는지에 맞는 증명이 필요하다.
3. **pilot의 실제 비용과 편향**: fresh draws는 pilot에 조건부인 noise 평균의 unbiasedness를 도울 수 있지만, 최종 clipping bias를 없애지는 않는다. 영상 선택확률을 바꾸면 목표 환자 평균 자체가 바뀔 수 있다. pilot·padding·동적 batching·실제 F/B·memory/time까지 세어야 하며, private-dependent runtime/로그를 공개한다면 그것도 출력 범위에 넣어야 한다.
4. **결과의 의미**: 같은 patient ε, δ·공개 사전학습·학습가능 파라미터·연산에서 utility가 좋아지거나, 같은 utility와 보장에 필요한 연산이 줄어야 한다. 작은 gradient 모형의 이득은 이 조건을 예고할 수 있지만 대체하지 못한다.

가장 강한 반론은 ‘기존 DPDM에 표준 MC 배분을 붙였을 뿐’이다. 이에 대한 답이 없어도 후보 검토 자체를 금지할 이유는 없지만, 최종 논문 기여는 아직 성립하지 않는다. 고정 multiplicity를 잘 조정하면 차이가 사라지는 경우도 정직하게 포함해야 한다.

특히 clipping 후 오차를 새 목표로 삼더라도 Adam의 history-dependent 변환, DP Gaussian noise, 실제 연산 overhead가 그 이득을 지울 수 있다. clipping된 gradient의 방향 오차 개선만으로 학습 효율 개선을 대체하면 안 된다. 반대로 모든 경우에서 개선을 보일 필요도 없다. noise·optimizer·clip 조건 중 무엇이 병목인지를 예측하여 **개선이 생길 영역과 생기지 않을 영역을 함께 설명**하는 것이 CARV의 기존 부정 사례보다 더 나아갈 부분이다.

## 4. 방향 2: 이미 점유된 구성요소

### 4.1 공격 실패와 방어 평가의 착시는 새로운 문제 제기가 아니다

**Aerni et al., CCS 2024**는 약하거나 defense-non-adaptive한 공격, 취약한 개별 사례를 가리는 population 평가, 약한 DP 비교군 때문에 보호효과를 과대평가하는 문제를 직접 다룬다. §3 p3의 Pitfall II, §4.3 p6, §5.3 p8을 확인했다. 출력 confidence를 숨기는 방어에는 label-only 공격, SSL에는 contrastive representation 공격을 적용하고 canary도 방어에 맞춘다. 따라서 ‘방어에 따라 공격을 재보정해야 한다’, ‘평균 MIA가 낮아도 안전하지 않다’가 독립 기여는 아니다. [최종 공개본](https://arxiv.org/abs/2404.17399).

**Quantile MIA** A09 §3 pp3–4는 target model에서 공개 nonmember 분포의 example-specific reconstruction-loss threshold를 학습한다. 공격 보정 비용을 줄이고 shadow model을 피하는 것도 이미 주장한다. 원문 Algorithm 1의 underlying distribution P와 실제 학습된 분위수 정확성을 구분해야 한다. 유한 표본의 patient-group FPR 보장을 그대로 주지는 않는다. [ICML 2024](https://proceedings.mlr.press/v235/tang24g.html).

**One-run DP audit**는 독립 inclusion coins와 여러 canary를 이용해 학습 한 번으로 ε의 하한을 얻는다. §§2–3 pp2–4, §6 pp8–9를 읽었다. 점수는 임의이며, black/white access 둘 다 가능하다. 본문에서 white-box는 중간 weights 접근, black-box는 최종 weights 또는 query 접근까지 포함하므로 우리 API 구분과 용어를 그대로 동일시하면 안 된다. finite attack audit는 DP 하한이고, 약한 공격에서 낮은 하한이 나오면 보장 상한이 낮다고 증명한 것이 아니다. patient bundle을 하나의 privacy unit/coin으로 정의하는 것만으로 새 auditing 정리가 되지도 않는다. [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/9a6f6e0d6781d1cb8689192408946d73-Paper-Conference.pdf).

### 4.2 ‘공식 보장과 인터페이스에서 관측되는 위험의 간극’도 직접 선행이 있다

**Ge et al., Neurocomputing 2026**, *Quantifying observable privacy in differentially private generative models under black-box access*는 DP-SGD 경로 안정성과 숨겨진 latent randomness를 결합하여 interface-specific f-DP/GDP envelope와 post-hoc audit를 제안한다. 공개 출판사 초록·서론·§6 일부를 확인했다. 실험은 tabular DP-VAE이며, 전체 본문·가정·증명의 타당성을 이번에 확인하지 못했다. 그래도 ‘formal ε를 바꾸지 않고 observable privacy를 설명/정량화’한다는 넓은 주장은 이미 겹친다. [출판사 1차 페이지](https://www.sciencedirect.com/science/article/abs/pii/S0925231226002900), [DOI](https://doi.org/10.1016/j.neucom.2026.132893).

기존 읽은 **User Inference**는 동일 사용자의 미사용 표본, reference likelihood와 평균 집계 및 DP 방어를 이미 다룬다. **FACE-AUDITOR**는 시각 user inference와 미사용 probing 사진·reference features를 다룬다. **Hartmann**은 관측 feature가 특정 분포 누출원을 가릴 수 있음을 다룬다. **Meeus**와 의료 patient risk 연구는 counterfactual influence/환자 단위 포함-제외를 다룬다. 이런 현상을 다시 관찰하는 것만으로 평가 방법의 새로움이 생기지 않는다. [User Inference](https://aclanthology.org/2024.emnlp-main.1014/), [FACE-AUDITOR](https://www.usenix.org/conference/usenixsecurity23/presentation/chen-min), [Hartmann](https://arxiv.org/abs/2209.08541), [Meeus](https://arxiv.org/abs/2506.20481), [의료 환자 위험](https://www.nature.com/articles/s41586-026-10688-0). 이들 상세 읽기 범위는 기존 `protection_unit_eu_novelty_audit_20260915.md` 및 관련 노트 참조.

Root가 추가한 **FSCA**는 별도 동료가 직접 확인 중이다. 이 노트 작성 시 registry에서 arXiv ID/FSCA 항목은 찾지 못했다. user-level diffusion audit가 이미 존재할 수 있는 필수 대조이며, 본 노트에서 성능·가정을 확인했다고 주장하지 않는다. [저자 원문](https://arxiv.org/abs/2506.11434).

## 5. 방향 2: finite-portfolio 방어 선택의 남는 후보

최근 논의 중인 `argmin_defense max_attack TPR@FPR α`는 그 자체로 새 의사결정 형식이 아니다. **Garivier et al., COLT 2016**은 순차 표집하는 action pair에서 Maximin-LUCB와 Maximin-Racing, 고정 confidence와 표본복잡도를 이미 연구한다. 이번에는 공식 초록·문제 설명을 확인했고 증명 전체는 읽지 않았다. Root는 본문을 별도로 대조 중이다. [PMLR 원문](https://proceedings.mlr.press/v49/garivier16b.html).

남겨볼 문제는 **각 arm의 reward가 바로 관측되는 것이 아니라 추정한 ROC 기준으로 만들어진다**는 것이다. nonmember calibration 표본으로 threshold를 정하고 member 표본으로 TPR을 추정한다. calibration·attack fitting·평가 표본이 섞이면 선택된 방어와 공격의 위험 추정이 낙관적이 될 수 있다. 저 FPR에서는 member 점수 측정을 더 해도 nonmember tail 부족이 병목으로 남을 수 있다. 단순 ‘1% FPR로 맞춘다’는 표현으로 이 불확실성이 사라지지 않는다.

또한 하나의 diffusion residual/feature 계산으로 여러 공격 점수가 동시에 생기고, 같은 환자가 여러 방어/공격에서 재사용된다. 따라서 arm마다 독립적으로 비용을 지불한다는 모형보다 **공유 원시 측정 단위의 비용과 의존성**을 쓰는 것이 실제 설정에 맞을 수 있다. 다만 공유 관측 자체도 일반 bandit·실험설계에서 알려진 구조이므로 그 사실만으로 충분하지 않다.

이 후보가 논문 기여가 되려면 다음 연결이 필요하다.

1. 방어별 적응을 이미 해 둔 강한 고정 portfolio에서도, 현재 평가비용 배분과 추정 threshold 때문에 방어 선택이 실제로 달라지는 구체적 사례를 보인다. 원래 공격이 약해서 생긴 차이와 분리한다.
2. member/nonmember의 공동 불확실성과 공유 feature 비용을 고려하여 **같은 선택오류 보장에서 평가비용 감소** 또는 **같은 예산에서 잘못된 선택 감소**를 제공한다. 각 방어마다 독립 보정한 고비용 평가를 비교 기준으로 둘 수 있다.
3. 공격군·자료·접근권·utility 허용 조건을 고정하고, unresolved이면 추천을 보류한다. 이 결정은 그 유한 비교의 결론이며 실제 최강 공격 위험이나 formal ε의 우열을 인증하지 않는다.

치명적 반론: threshold를 독립 calibration으로 먼저 고정하면 기존 Maximin-LUCB/Racing에 Bernoulli 결과를 넣는 것만으로 해결될 수 있다. quantile confidence interval과 표준 union bound만 붙여 같은 보장을 얻는다면 방법 기여는 약하다. 새 알고리즘을 억지로 만들기보다, 기존 적용이 왜 계산을 낭비하거나 잘못된 선택을 하게 되는지와 그 해결이 실질적으로 다른지 확인해야 한다. **순위 반전 한 번, non-detection, abstention 기능만으로는 부족하다.**

## 6. 최종 범위와 아직 미확인인 부분

- 두 후보는 현재 E/U 또는 clipping 한 사례에 고정하지 않는다. E/U·같은 환자의 여러 사진은 난점이 커지는 한 적용 축이다.
- 방향 1은 새로운 privacy 정의나 user-DP 회계 자체보다 연산과 utility의 개선 후보다. 방향 2는 새로운 공격 자체보다 잘못된 보호 비교를 줄이는 평가·선택 후보다.
- 새로운 방어 학습을 할 경우 반복 실험·private validation·private-dependent controller의 비용도 포함해야 한다. 공개 사전학습은 기존 공개로 가정한 자료에 대한 새로운 보호를 소급 제공하지 않는다.
- 두 방향 모두 ‘보호효과’와 ‘감사 접근권/관측 부족’을 완전히 분리할 수 있다는 주장을 하지 않는다. 더 좁은 조건에서 구별하거나, 결론을 내릴 수 없는 범위를 명시하는 것이 현재 가능한 목표다.
- 이번 bounded review에서 adaptive nested Monte Carlo 전반, shared-observation bandit 전반, ROC confidence-sequence 전반을 체계적으로 전수 조사하지 않았다. 해당 연결이 최종 후보로 남으면 정확한 가장 가까운 방법·정리를 추가 대조해야 한다. 그 미완료가 이번 구체화 작업을 무효화하거나 ‘아무 후보도 없다’는 뜻은 아니다.

이번 점검에서 기존 성분의 조합만으로는 약하다는 차단선과, 각각 한 단계 더 나아가야 할 구체적 기술 대상을 정리했다. 어느 방향도 이미 새롭거나 성공했다고 확정하지 않는다.
