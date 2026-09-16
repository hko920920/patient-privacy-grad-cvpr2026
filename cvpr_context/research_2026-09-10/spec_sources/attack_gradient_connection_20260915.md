# 보호 학습 gradient와 기존 공격 점수의 연결 범위

2026-09-15. 원문·소스의 읽기 검토다. 기존 점수 자료의 재분석, 새 GPU 실행, 학습, 공격 fitting은 수행하지 않았다. 아래 소스 경로는 `research_2026-09-10` 기준이다. [현재 보호 관측 후보](../PROTECTION_OBSERVATION_CLAIM_20260915.md)와 연결하며 새 공격의 필요성이나 기여를 확정하지 않는다.

## 1. 직접 성립하는 국소 관계

고정 query의 denoising loss를 `ℓq(θ)`라 하자. 이미지, latent, conditioning, timestep, noise와 평가 연산을 고정하고 θ의 작은 변화에 대해 미분 가능하다면 다음 관계가 성립한다.

```text
Δ(−ℓq) = −〈∇θℓq, Δθ〉 + O(||Δθ||²)
SGD: Δθ = −ηv  →  Δ(−ℓq) = η〈∇θℓq, v〉 + O(η²).
```

O항은 해당 근방의 매끄러움과 유계 곡률을 전제로 한다. 이것은 한 상태의 국소 관계이며 확률적인 membership 성능 정리가 아니다. LoRA만 학습하면 gradient와 Δθ도 같은 LoRA 파라미터 공간에서 정의해야 한다.

E는 학습 영상이라는 뜻이지 query의 noise·timestep·조건까지 실제 학습 측정과 같다는 뜻은 아니다. U는 같은 환자의 미사용 영상이며, 같은 환자라는 사실만으로 `∇ℓU`와 학습 gradient의 정렬 부호·크기가 결정되지 않는다. E 평균 query gradient가 환자 평균 학습 gradient와 같아지는 것은 같은 loss와 측정 조건을 사용하거나 정당한 기대값 관계를 둔 경우다.

현재 원학습은 `code_working/u_patient_audit/train_coverage.py:38`의 AdamW와 `:53–66`의 batch MSE·전체 gradient clipping을 사용한다. `:55`는 실제 weak-label train prompt를 항상 사용하며 추가 LoRA 학습의 null-caption dropout 분기는 없다. 따라서 기존 1000-step masked 학습 결과를 단일 SGD 내적식의 직접 검증으로 재명명할 수 없다. 실제 Adam 상태·weight decay·DP noise·후속 학습 경로를 포함한 Δθ의 국소 관계는 별도로 확인할 수 있다.

## 2. 공격별 원문·실제 연산

### Denoising Loss와 CDI Multiple Loss

- 원문: [CDI X01](../pdfs/X01.pdf), PDF p3 Eq2, p4 §3.1, p5 Multiple Loss.
- 소스: `code_sources/X01/src/attacks/features_extraction/denoising_loss.py:19–29`, `scores_computation/denoising_loss.py:9–15`.
- 원시 denoising MSE는 학습 목적과 가장 직접 연결된다. 그러나 공개 CDI 구현의 개별 측정은 residual의 **L2 norm**이고 이후 반복 평균을 취한다.
- `ℓ=||r||²/D`, `e=||r||>0`이면 `∇e=D∇ℓ/(2e)`다. 단일 측정에서는 양수 배율이지만 여러 noise·영상의 평균에서는 측정별 배율이 달라지므로 평균 L2와 평균 MSE의 순위·반응이 같다고 할 수 없다.

### SecMI

- 원문: [A11](../pdfs/A11.pdf), PDF p4 Eq6, p5 Eq9–14. t-error는 deterministic forward/reverse의 posterior-estimation 근사다. Eq14의 학습 loss 연결은 수렴 조건에 관한 설명이다.
- 획득한 구현은 **CDI 저자의 SecMI 구현**이다: `code_sources/X01/src/attacks/features_extraction/secmi.py:19–35,46–59`; `scores_computation/secmi.py:22–24`. 이것을 원 SecMI 저장소 전체 재현으로 부르지 않는다.
- 최종 재구성과 기준 latent 둘 다 θ에 의존한다. 일반적인 score의 미분은 가능하지만 중간 경로의 Jacobian을 모두 포함해야 하며 단일 denoising query gradient로 치환할 수 없다. 유한 학습에서 clipping이 어느 방향으로 t-error를 바꾸는지는 원문의 수렴 논리만으로 정해지지 않는다.

### Quantile MIA

- 원문: [A09](../pdfs/A09.pdf), PDF p3 Definition 2.1/Eq8, Algorithm 1, Eq9–10; p4 §3.2–3.3.
- SecMI t-error를 쓰며 p3은 이것이 실제 training objective 자체가 아니라고 명시한다. 공개 보조 비회원 자료로 이미지별 `qα(x)`를 학습하고 `t-error ≤ qα(x)`에서 member로 판정한다.
- 이미지별 quantile은 공통 threshold 이동과 다르며 ranking을 바꿀 수 있다. 따라서 “보정은 순위를 못 고친다”는 말을 이 방법 전체에 적용하면 틀린다.
- quantile predictor를 고정하면 query t-error 변화만 분리할 수 있지만, 보호 모델마다 다시 학습하는 전체 공격에는 predictor 변화도 포함된다. 유한 자료로 추정한 quantile은 목표 FPR의 실증 검증을 대신하지 않는다.
- 환자 U 확장에서는 candidate image가 image-level nonmember인 positive 환자와 진짜 nonmember 환자를 구분해야 한다. 단순히 모든 held-out 이미지를 null reference로 쓰면 patient label을 잘못 정의하게 된다.

### PFAMI

- 원문: [X11](../pdfs/X11.pdf), PDF p5 Eq10–13, p6 Eq19.
- 소스: `code_sources/X11/attack/attack_model_PFAMI.py:103–114`의 DDPM MSE; `:249–264`의 원본/perturbation loss; `:349–363`의 상대 변화와 평균. `:363,446`은 member=0 평가와 음수 방향을 함께 사용한다. 현재 member=1 기준 상대 변화 방향과 혼동하지 않는다.
- 코드의 상대 score `F=(ℓcrop−ℓorig)/ℓorig`에 대해 다음이 성립한다.

```text
∇F = g_crop/ℓorig − ℓcrop*g_orig/ℓorig²
ΔF ≈ −η〈∇F,v〉   (SGD인 경우).
```

- 원본·crop 두 loss가 비례해서 감소하면 상대 score는 유지될 수 있다. 반대로 원본만 더 감소하면 score가 증가할 수 있다. 이는 검증 가능한 조건부 관계이지 현재 의료 U에서 상쇄가 실제 원인이었다는 결론은 아니다. 원본·crop noise와 평균 순서도 고정해야 한다.

### MoFit

- 원문: [X12](../pdfs/X12.pdf), PDF p3 Eq1–2, p5 직관·한계, p6 Eq5–8.
- 코드: `code_sources/X12/COCO/MoFit_COCO.py:220–224`의 surrogate 최적화, `:288–289,331–339`의 embedding Adam, `:349–355`의 원본 이미지 평가.
- θ를 동결하고 null 조건에서 surrogate를 최적화한 뒤, 같은 timestep·noise에서 그 surrogate에 맞는 embedding을 얻는다. 최종 MoFit score는 **원본 이미지의 conditional loss−null loss**다.
- 고정 embedding φ에서의 최종 score에는 `∇θℓφ−∇θℓnull`이 들어간다. 모델마다 embedding을 재최적화하면 `dφ*/dθ`와 앞 단계 surrogate의 영향까지 필요하다.
- embedding의 최적화 목적은 surrogate loss이며 최종 평가는 원본 loss다. 따라서 정확한 optimum을 가정하더라도 envelope theorem으로 embedding 의존 항을 삭제할 수 없다. 실제 finite-step 최적화는 추가로 알고리즘 경로에 의존한다.
- 원문의 null loss 설명은 classifier-free training을 사용한다. 현재 추가 LoRA 학습의 null-caption dropout은 0이므로 그 설명의 전달은 자동이 아니다. 이는 unconditional 경로가 전혀 변하지 않거나 원형 MoFit이 불가능하다는 뜻은 아니다.
- 고정 embedding을 새 noise에서 평가한 진단은 noise별 embedding 재최적화를 포함한 원형 공격 성능과 구분한다.

### CDI의 전체 특징과 판별기

- 원문: [X01](../pdfs/X01.pdf), PDF p5 §4.1–4.2.
- GM 소스 `features_extraction/gradient_masking.py:14–25,30–43,75–76`: loss의 **입력 latent gradient**로 mask를 선택하고 변형 입력의 loss를 계산한다. 이는 target training parameter gradient가 아니다. mask 교체점에서는 매끄러운 1차 근사가 그대로 성립하지 않을 수 있다.
- NO 소스 `features_extraction/noise_optim.py:21–37,53–84`: perturbation을 L-BFGS로 적합하고 별도 반환 feature를 계산한다. 원문의 목적값과 released-code 반환 feature 차이는 기존 source audit의 별도 명칭을 유지한다. 최적화 과정의 gradient를 보호 학습 gradient로 부르지 않는다.
- 판별기 소스 `scores_computation/cdi.py:19–27,46–55`: standardization과 logistic regression을 fitting한다. 고정 판별기의 반응도 특징 Jacobian에 가중치를 곱한 것이며 원시 loss 하나의 반응이 아니다. 모델별 U-fit을 허용하면 scaler·계수 변화까지 전체 공격에 포함된다.

## 3. U 적응을 허용한 뒤 남는 질문과 기여 경계

같은 접근권·사진 수·보조 환자 정답·계산량을 제공해 기존 공격을 각 보호 모델의 U에 적응시켜야 한다. 이때 차이가 회복되면 표준 adaptation의 성과로 인정한다. 이미지별 보정, 특징 scaling, 판별기 재학습을 전부 공통 threshold 하나로 축소하면 공정한 비교가 아니다.

남는 질문은 “기존 공격을 실패하게 만들 수 있는가”가 아니라, **동일한 유효 patient 보장 아래 E에서 내린 보호 선택이 U에서 어떤 조건까지 유지되는가**다. 필요한 보조 U 참여 정답을 제한 접근 감사자가 얻을 수 있는지는 별도 접근권·비용 문제다. auxiliary training에서 제외한 환자를 생성한 target model의 학습까지 독립이었다고 말하지 않는다.

새 공격 없이 평가·분석 기여를 주장하려면 실제 보호 연산, E/U 관측 조건, 강한 공격의 적응, 독립 평가의 불확실성, 의료 효용·비용 아래 보호 선택의 개선을 연결해야 한다. 단순 사용자 위험 잔존·AUROC와 low-FPR 차이·clipping 후 신호 잔존은 이미 User Inference 등의 선행 경계에 들어간다. 동일 환자 보장 조건의 E→U 평가 전이를 설명하는 구체적 조건이 후보이며, 미연구·새 기여를 확정한 것은 아니다.

[원래 목적](../RESEARCH_PURPOSE.md)의 7–26,40–42행, [실험 설계](../EXPERIMENT_DESIGN.md)의 148–152,187–189행, [졸논/CVPR 정렬](../THESIS_CVPR_ALIGNMENT.md)의 25–35,77행처럼 기존 강한 공격이 충분하면 보호 평가 도구로 사용한다. 경험적 공격으로 ε를 줄이거나 DP 보장을 인증하지 않으며, 새 공격 SOTA를 졸논의 필수 전제로 만들지 않는다.

## 4. 보호 관측 문서·해석적 예시의 독립 검토

`PROTECTION_OBSERVATION_CLAIM_20260915.md`와 `spec_sources/protection_observation_design_20260915/analytic_witness.py`를 읽었다. A/B 대수적 분해, Gaussian mean-shift의 공통 FPR TPR, 환자 sampling mixture, zero-moment Adam 경계에 진행을 막는 오류는 찾지 못했다. 실제 모델을 실행하거나 계산 산출물을 재생성하지 않았다.

문서 94행처럼 예시는 특정 선형 관측의 순서 반전을 보인다. 전체 Gaussian 벡터를 보는 최적 공격이나 현재 white-box에서 가능한 모든 공격의 순서 반전은 아니다. 같은 C-bounded patient contribution은 공개 정규화·인접성·sampling·noise 조건 아래 공통 DP 상계를 허용한다는 뜻이며 실제 누출의 동일성은 아니다. A는 patient-sampled 제어이지 원형 ELS가 아니다.

후속 zero-moment 실제 Adam 국소 대조는 지정 상태·초기 moment에서 조건의 존재/부재를 보는 진단이다. 여기서 차이가 사라져도 후속 moment·DP noise까지 포함한 후보 전체가 반증되지는 않으며, 차이가 남아도 전체 학습·강한 공격의 보호 선택 순위가 검증되는 것은 아니다.
