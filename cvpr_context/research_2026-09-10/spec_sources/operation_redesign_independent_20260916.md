# 연산 재설계 독립 검토: 공개 원본·복수 모델·DP-LoRA

작성: 2026-09-16. 새 GPU, 모델 실행, 학습, 통계 실험, 코드 수정은 없다. 공개 1차 문헌의 지정 절을 읽어 후보의 직접 선행을 확인했다. 연구의 두 큰 축인 보호 설계 효율과 보호 평가 개선은 유지한다.

**이번에 읽은 범위에서는, 아래 세 소재에서 기존 해법을 넘어서는 새 연산 하나를 논문 후보로 확정하지 못했다.** 단순 reference subtraction, model-sequence attack, intrinsic LoRA clipping/noise는 생각보다 훨씬 직접적인 선행이 있다. 대신 실제로 남은 수학적 범위와 가능한 변경을 구분했고, 알려진 교정 연산을 새 알고리즘으로 포장하지 않았다.

## 1. 공개 원본과 private adaptation을 같이 보는 평가

### 직접 선행 1: LoRA-Leak

[Ran et al., *LoRA-Leak: Membership Inference Attacks Against LoRA Fine-tuned Language Models*, arXiv:2507.18302](https://arxiv.org/abs/2507.18302). 직접 읽음: PDF pp.3–4, Table I, §§III–IV.A. 이번 접근 시점의 PDF는 abstract의 공격 수 설명과 Table I의 집계 표현에 차이가 있으므로 정확한 신규 공격 개수는 이 메모의 근거로 쓰지 않는다.

원문의 대상은 benign official pretrained model에서 LoRA로 조정된 모델이다. 공격자는 두 모델의 weight와 내부 신호를 볼 수 있지만 pretrain/fine-tune 데이터와 과정에 개입하지 않는다. 목표는 fine-tuning membership이며 pretraining membership이 아니다. 기존 loss뿐 아니라 perturbation, token probability, input-gradient 등의 신호를 pretrained reference로 보정한다.

**제거할 주장:** “LoRA parameter 수가 작아 안전해 보이지만 원본과 비교하면 누출이 드러난다”, “공개 원본을 공격의 무료 기준으로 사용한다”는 새 후보가 아니다. 우리 의료 diffusion에 적용하는 것은 가능한 baseline 작업이지만 적용 도메인 변경 자체가 새 연산은 아니다.

### 직접 선행 2: ICLR 2026 adaptation privacy benchmark

[Marek et al., *Benchmarking Empirical Privacy Protection for Adaptations of Large Language Models*, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/73b75e4eafbdb617822bba262cc81c76-Abstract-Conference.html). 직접 읽음: [arXiv v1](https://arxiv.org/pdf/2606.09401)의 pp.3–7, §6 pp.9–10, App.B.3–B.4 p.21.

이 연구는 exact pretrain overlap/IID-but-disjoint/OOD, Full/Head/LoRA/Prefix adaptation, DP 수준, pretrained 또는 shadow reference를 교차한다. §6은 pretraining 포함 bit a와 adaptation 포함 bit b를 따로 두며, fixed pretraining 상태에서 b만 비교하는 경우와 두 단계를 함께 바꾸는 경우를 명시적으로 나눈다. 따라서 “같은 ε라도 사전학습 분포에 따라 empirical risk가 달라진다” 또는 “두 단계의 membership을 구분해 감사한다”도 직접 점유돼 있다.

### 남는 중요 경계와 실제 변경의 수준

고정 base θ₀ 및 fixed auxiliary information S에 대해 adaptation A가 DP라면, 합법적인 post-processing T(θ₀,A(D),S)는 그 adaptation의 DP 보장을 약화시키지 않는다. 원본에 이미 정보가 있을 수 있다는 사실은 이 조건부 명제를 깨지 않는다. 반대로 audit game에서 membership bit에 따라 base나 reference의 학습 자료까지 바뀌면 같은 조건부 비교가 아니다. 이는 이미 표준 DP·causal auditing의 구분이다.

원문은 shadow를 target과 다른 data split에서 학습한다고 명시하지만, 이번에 읽은 설명만으로 **reference의 sample 포함 여부가 audit label과 독립인지**, HPO·reference 선택까지 어떤 transcript에 포함하는지 완전 복원하지 못했다. 이를 저자의 오류나 누출 원인으로 단정하지 않는다. 단일 모델의 empirical AUC 수치만으로 formal DP 위반을 주장하지도 않는다.

실제로 바꿀 수 있는 평가 연산은 다음처럼 명확하다.

```text
기존 점수 연산: s(target, reference, x)
대조 연산: 동일 x·reference·선택 규칙을 보존하고 target만
          membership과 독립인 고정 출력으로 치환한다.
          별도로 reference-only 점수의 label 판별력도 확인한다.
조건부 감사: θ₀와 auxiliary/reference transcript를 고정하고
             adaptation inclusion만 바뀌는 game을 평가한다.
```

이로써 target이 주지 않은 판별력을 target의 incremental privacy leakage로 해석하는 문제를 조사할 수 있다. 그러나 **target-free negative control와 fixed-auxiliary audit 자체는 표준적인 교정**이고, 프로젝트의 encoder-only/base control과도 겹친다. 실제 강한 평가에서 이 문제가 반복되고 교정 후 보호 선택이 실질적으로 바뀐다는 새로운 발견이 없다면, 이를 새 평가 논문 후보로 올리지 않는다. 이번에는 그런 코드·결과 검증을 실행하지 않았다.

TMI/PETS 2024는 downstream model만으로 pretraining membership이 남는 상황을 다룬다. [공식 논문 페이지](https://petsymposium.org/popets/2024/popets-2024-0075.php). 이번에는 공식 abstract를 확인했으며 PDF 본문 다운로드가 실패해 전체 원문을 재독했다고 기록하지 않는다. 비DP pretraining 위험을 새롭게 발견했다고 주장하지 않는 추가 경계로 사용한다.

## 2. 여러 checkpoint/model의 joint exposure

[Michel, Basu, Kaufmann, *Sequential Membership Inference Attacks*, arXiv:2602.16596v2](https://arxiv.org/pdf/2602.16596v2). 직접 읽음: PDF pp.1–6, §2 게임·감사, §3 isolation/unknown time/distribution shift/multiple insertion, §4 도입. 뒤쪽 실험과 증명 전부를 재검산한 것은 아니다.

이 원문은 단순 최종 snapshot 대신 model sequence를 관측한다. Gaussian mean 모형에서 insertion 직전·직후 통계가 target을 분리하는 likelihood-ratio test를 유도한다. known/unknown insertion time 및 multiple insertion을 다루고, DP-SGD에는 Gaussian-gradient 근사에 기반한 white-box 점수와 black-box loss 차분·비율을 제안한다. canary 삽입 시점의 제어와 독립 감사 반복을 통해 ε lower bound를 얻는다.

직접 앞선 [Jagielski et al., PETS 2023, *How to Combine Membership-Inference Attacks on Multiple Updated Machine Learning Models*](https://petsymposium.org/popets/2023/popets-2023-0078.pdf)는 모델 업데이트들의 공격 점수를 결합하고 update-set membership과 entry time을 구분한다. 이번에는 웹으로 제공된 §3.3 게임 및 threat-model 내용을 직접 읽었으며 PDF 전체 재독은 실패했다.

**제거할 주장:** 여러 공개 모델을 함께 보면 더 많이 누출된다, update 전후 차분으로 dilution을 피한다, participant가 들어온 시점을 찾는다, 조합된 공격으로 더 강한 DP lower bound를 얻는다는 큰 발상은 새롭지 않다. 여러 출력의 composition이나 patient grouping을 덧붙이는 것도 충분하지 않다.

**미확인 경계:** 두 특정 의료사진의 transfer가 전체 학습 단계에 퍼진 경우와 unknown patient participation을 다루는 full diffusion generator에서 위 Gaussian-gradient/controlled-insertion 구조가 정확히 성립하는지는 확인하지 않았다. 하지만 이 가정 차이는 새 inference statistic이나 release operation을 제공하지 않는다. 현재로서는 기존 sequential attack의 적용 범위를 검토하는 질문이며, 추가 snapshot 선택 heuristic을 즉석에서 만들어 새 후보로 세지 않는다.

## 3. DP-LoRA의 실제 기하 연산도 이미 바뀌어 있다

[Wang & Zhang, *PRISM: Gauge-Invariant Tangent-Space Differentially Private LoRA*, arXiv:2606.00944v1](https://arxiv.org/pdf/2606.00944v1). 직접 읽음: §§2–3 pp.3–7, Algorithm 1, App.A.22–A.27 pp.23–25, 실험 설정 pp.26–27. 여기서는 2026-05-31 preprint로 식별한다.

PRISM은 Z=ABᵀ의 factor-space DP noise가 gauge에 따라 다르게 증폭되는 문제를 다룬다. 바꾸는 연산은 intrinsic tangent gradient norm으로 global clip, tangent-space isotropic Gaussian, 저차원 noise sampler, rank-r retraction이다. 또한 DP-aware moment floor와 adaptive update를 제안한다. 따라서 “LoRA의 두 factor에 각각 noise를 넣지 말고 실제 low-rank update 공간에서 clip/noise하자”는 후보는 이미 직접 존재한다.

근접 차단선으로 [LoRA-RITE, ICLR 2025](https://openreview.net/pdf?id=VpWki1v2P8)의 transformation invariance 정의·matrix preconditioning 및 moment 유지 설명도 확인했다. [FFA-LoRA, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4e243e95c913b367775d71d7182b99d9-Abstract-Conference.html)는 공식 abstract 범위에서 freeze-A와 DP noise 증폭 문제를 확인했다. [ICCV 2025 diffusion DP-LoRA](https://openaccess.thecvf.com/content/ICCV2025/html/Tsai_Differentially_Private_Fine-Tuning_of_Diffusion_Models_ICCV_2025_paper.html)는 pretrained LDM의 rank·noise multiplicity·trainable component 비교를 이미 포함한다. 이 두 후자는 이번에 전체 proof를 새로 읽은 목록이 아니다.

### 정확히 발견한 PRISM의 범위 차이

PRISM Eq. (25) 아래와 App.A.26은 일반 가역 R에 대한 gauge invariance를 서술하지만, App.A.27 Lemma A.30이 실제 증명하는 것은 **orthogonal R**다. 특히 App.A.26의 `tr(M⁻¹)`·`tr(N⁻¹)`이 일반 GL(r) 변환에서 각각 불변이라는 문장은 성립하지 않는다. M=AᵀA, N=BᵀB일 때 R=cI로 두면

```text
A′=cA, B′=B/c, Z′=Z
M′=c²M, N′=N/c²
tr((M′)⁻¹)=c⁻² tr(M⁻¹)
tr((N′)⁻¹)=c² tr(N⁻¹).
```

따라서 두 trace가 각각 불변이라는 일반 명제에는 단순한 대수 반례가 있다. 이것은 이 메모에서 새 수치 실험을 한 결과가 아니다. 원문의 intrinsic tangent-noise mechanism과 그 post-processing의 DP 보장이 이 때문에 자동 붕괴하는 것도 아니다. **adaptive optimizer 전체가 어떤 gauge 범위에서 같은 intrinsic update를 만드는지**에 대한 명제의 범위 문제다. 실제 구현이 canonical gauge와 적절한 state transport를 강제하는지 이번에 확인하지 않았으므로 실제 학습 결과까지 틀렸다고 주장하지 않는다.

### gap에서 바뀔 연산까지 한 단계는 구체화되지만, 새 기여는 아니다

가능한 교정 연산은 factor를 intrinsic Z에서 얻은 canonical/balanced frame으로 보낸 뒤 first/second moment를 같은 frame으로 transport하고, 그 frame에서 noise covariance floor와 adaptive transform을 적용하는 것이다. 목표는 단순 orthogonal symmetry가 아니라 동일 Z·동일 intrinsic history에서 같은 update law를 만드는 것이다.

하지만 canonicalization·matrix preconditioning·moment transport는 LoRA-RITE와 Riemannian optimization에서 이미 다룬다. **PRISM의 범위를 정정하고 기존 invariant optimizer를 올바르게 결합하는 것으로 끝나면 재현·교정 작업이지 자동 paper 기여가 아니다.** 더 강한 후보가 되려면 동일 privacy와 실제 총비용 아래 알려진 intrinsic DP mechanism + invariant optimizer 결합으로는 처리하지 못하는 구체 상태/비용 문제가 있어야 하며, 그것을 바꾸는 새 연산이 필요하다. 이번에 그 추가 간격을 확보하지 못했다.

## 4. 지금 남기는 결론

| 후보 소재 | 이번 판단 | 실무적으로 남길 수 있는 것 |
|---|---|---|
| public pretrained reference로 private fine-tune 위험을 분리 | LoRA-Leak·ICLR2026 §6에 직접 점유 | reference provenance와 fixed-auxiliary 조건을 명시한 비교·negative control |
| multiple checkpoint joint attack | PETS2023·SeMI2026에 직접 점유 | 실제 공개 가능한 snapshot만 사용하는 강한 sequential baseline |
| LoRA factor clipping/noise를 intrinsic 공간으로 변경 | PRISM에 직접 점유 | diffusion DP-LoRA와 intrinsic DP/RITE 결합의 적용·재현 검토 |
| PRISM adaptive step의 full GL invariance | 본문 서술과 실제 증명의 범위 차이 확인 | 명제 범위 정정 및 canonical frame/state transport의 구현 검토. 새 방법 성공으로 세지 않음 |

새로운 소재를 억지로 하나 남기는 대신, **후보 공간을 좁히는 데 필요한 직접 선행과 구체 연산 경계**를 기록한다. 위 세 축의 중요성은 유지되지만, 이번 결과로 새 논문 기여가 확보됐다고 말할 수는 없다. 새 모델 훈련이나 의료 GPU 비교를 시작하자는 결론도 아니다. 다른 설계 담당이 제시할 후보를 이 근접 선행과 대조하는 자료로 사용할 수 있다.

미완료: PRISM 공식 구현의 canonicalization/state transport 경로, ICLR2026 benchmark의 auxiliary/reference 데이터 포함 관계와 전체 HPO transcript, SeMI의 의료 diffusion/patient 적용 proof, 해당 논문들의 전수 후속 문헌 대조. 문헌에 보이는 표현상의 간격을 실제 privacy failure로 단정하지 않았다.
