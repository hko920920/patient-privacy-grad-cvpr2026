# Biometric 후보 선행 충돌과 patient-exposure-tail 전제 결과

- 기록일: 2026-09-03 (Asia/Seoul)
- 최신 판정: **식별정보 억제 방법 후보는 직접 선행 충돌로 종료**
- 공개 진단: `PASS_PATIENT_EXPOSURE_TAIL_ENDPOINT_PREMISE`
- 현재 1순위의 성격: 신규 anonymization network가 아니라 **patient-unit generative privacy audit**
- 장기 K5 학습: 아직 미실행·미승인

## 1. 결론부터

처음 생각한 `환자 고유 특징을 찾아 억제하면서 병변은 보존하는 diffusion`은 독립 신규 방법으로
밀 수 없다. 2025--2026 선행이 identity/semantic 분리, 신원 억제, 병리 보존, diffusion-based
anonymization과 patient-level unlearning을 이미 직접 다룬다.

현재 남는 가장 방어 가능한 한 문장은 다음과 같다.

> **한 장을 privacy unit으로 삼은 CXR diffusion과 한 환자를 privacy unit으로 삼은 diffusion을
> 동일한 환자 수준 예산과 거의 같은 계산량에서 비교하고, 평균 공격 AUC가 아니라 생성물에 대한
> 환자별 biometric exposure의 분포와 극단 꼬리를 감사한다.**

이것은 현재 가장 실행 가능성이 높은 **연구 질문**이지만, 아직 CVPR급 신규 **방법**은 아니다.
핵심 구성요소가 모두 선행에 있으므로, exact intersection의 새로움과 실제로 뜻밖의 결과가 나오는지
입증해야 한다. NIH 하나와 한 seed의 단순 비교로 끝나면 MIDL/TMLR형 평가 논문에는 맞을 수 있어도
CVPR main 경쟁력은 높지 않다.

## 2. 왜 biometric suppression 방법을 종료하는가

| 1차 문헌 | 이미 점유한 핵심 | 현재 금지 claim |
|---|---|---|
| PrivDiff-Net, PMLR 2026 | latent diffusion에서 cross-attention 직교투영으로 identity cue 억제, privacy-guidance loss로 병리 보존; identity AUC 47%, diagnostic AUC 78% 보고 | `diffusion에서 identity를 지우고 pathology를 보존` 자체 |
| Semantic versus Identity, ICCV 2025 | identity blocking, medical-foundation-model semantic compensation, MDL 기반 identity/semantic 분해, adjustable privacy | identity/semantic disentanglement와 가변 de-identification 자체 |
| All-in-One MedReID, CVPR 2025 | 11개 의료영상 데이터셋의 통합 re-identification과 privacy application | 의료영상 ReID encoder 또는 identity cue 발견 자체 |
| Patient-level Machine Unlearning in LDMs, MIDL 2026 short paper | 환자 단위 latent-diffusion unlearning 세 방법과 privacy--utility 한계; 20% 이상 재식별 잔존 보고 | `환자 하나의 영향을 diffusion에서 제거` 자체 |
| Anonymous CXR latent diffusion, 2022 | 재식별 전이를 줄이는 privacy-enhancing CXR sampling | anonymous CXR synthesis 자체 |

따라서 DINO/RAD-DINO identity signal을 loss로 눌러 병변을 살리는 모델을 바로 만들면 PrivDiff-Net과
ICCV 2025의 조합에 매우 가깝다. 여기에 DP-SGD만 붙이는 것도 독립 알고리즘 기여 없이 알려진 방어를
결합한 것으로 보일 위험이 크다.

주요 출처:

- PrivDiff-Net: https://proceedings.mlr.press/v317/akhter26a.html
- Semantic versus Identity: https://openaccess.thecvf.com/content/ICCV2025/html/Tian_Semantic_versus_Identity_A_Divide-and-Conquer_Approach_towards_Adjustable_Medical_Image_ICCV_2025_paper.html
- All-in-One MedReID: https://openaccess.thecvf.com/content/CVPR2025/html/Tian_Towards_All-in-One_Medical_Image_Re-Identification_CVPR_2025_paper.html
- Patient-level diffusion unlearning: https://openreview.net/pdf?id=x83b7ybcB1
- Anonymous CXR synthesis: https://arxiv.org/abs/2211.01323

## 3. patient-unit DP 비교도 그 자체로는 새 방법이 아니다

환자별 gradient를 만들고 clip/noise하는 개념은 P3SGD가 CVPR 2019에서 이미 제시했다. CVPR 2023
DP-FedEmb는 user-level image representation 학습을 다뤘고, 최근 fixed-compute user-level DP
연구는 example-level sampling과 user-level sampling을 직접 비교했다.

더 직접적으로 Kaiser et al.의 2025 `User-Level Differential Privacy in Medical Machine Learning`은
CheXpert의 가변 환자 기록 수를 대상으로 다음을 이미 수행했다.

- naïve record-DP의 group-privacy 변환과 user-level 보호를 비교;
- 환자 record 수에 따라 Poisson sampling rate 또는 clipping bound를 조정;
- 환자별 동일 `(epsilon, delta)`를 유지하면서 CheXpert 분류 효용을 평가.

따라서 다음도 신규 claim으로 사용할 수 없다.

- 의료 데이터에서는 한 환자에게 여러 장이 있다는 발견;
- record-level DP가 환자 privacy를 충분히 표현하지 못한다는 일반론;
- group conversion보다 native patient/user DP가 효율적일 수 있다는 일반론;
- patient mean-before-clip 또는 환자별 sampling/clipping 자체.

주요 출처:

- P3SGD: https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_P3SGD_Patient_Privacy_Preserving_SGD_for_Regularizing_Deep_CNNs_in_CVPR_2019_paper.html
- DP-FedEmb: https://openaccess.thecvf.com/content/CVPR2023/html/Xu_Learning_To_Generate_Image_Embeddings_With_User-Level_Differential_Privacy_CVPR_2023_paper.html
- Medical ULDP: https://tpdp.journalprivacyconfidentiality.org/2025/pdf/kaiser.pdf
- Fixed-compute ELS/ULS: https://arxiv.org/abs/2407.07737
- DP 3D medical latent diffusion: https://arxiv.org/abs/2407.16405

## 4. 그래도 남는 정확한 교집합

아래 각각은 점유되어 있지만, 이번 검색 범위에서 이 전부를 하나의 통제 실험으로 묶은 직접 선행은
찾지 못했다. 이는 `최초`의 증명이 아니라 현재 검색 범위의 간극이다.

1. text/image diffusion의 formal DP;
2. 환자당 여러 CXR이라는 add/remove-one-patient unit;
3. ordinary image-DP, patient-budget-matched group conversion, native patient-DP의 동시 비교;
4. 동일 target patient `(epsilon, delta)`와 거의 동일한 raw-image compute;
5. CXR의 실제 biometric ReID signal;
6. 생성물의 평균 privacy metric이 아니라 **훈련 환자별 exposure survival tail**;
7. 공격·accounting·실행 artifact의 claim-aligned audit.

Nature 2026은 분류모델에서 aggregate MIA AUC가 거의 무작위여도 일부 환자는 거의 완전하게 공격될
수 있음을 보였고, 환자 점수는 record-level vulnerability의 최댓값으로 구성했다. 또한 이러한
불균등 위험이 MIA 이외 공격에도 이어지는지는 열린 문제라고 명시한다. CheXGenBench는 11개 CXR
T2I 모델의 fidelity/privacy/utility를 통합하지만 formal patient-DP unit 비교나 환자별 extreme-tail을
중심 질문으로 삼지는 않는다. 이 두 경계 사이가 현재의 가장 좁은 가능성이다.

- Nature patient audit: https://www.nature.com/articles/s41586-026-10688-0
- CheXGenBench: https://arxiv.org/abs/2505.10496
- CXR biometric ReID: https://arxiv.org/abs/2103.08562
- prompt-driven memorization: https://arxiv.org/abs/2502.07516
- 원 출발점 PFDM: https://openaccess.thecvf.com/content/CVPR2026/html/Patel_Personalized_Federated_Training_of_Diffusion_Models_with_Privacy_Guarantees_CVPR_2026_paper.html

PFDM은 client별 formal protection과 shared/personalized diffusion을 다루지만 의료 반복 환자를 privacy
unit으로 분석하지 않는다. 현 후보는 PFDM의 알고리즘을 단순 이식하는 것이 아니라, 그 논문이
가정한 client-level protection 아래 의료기관 내부의 실제 data subject가 누구인지 묻는 후속 질문이다.

## 5. 기존 K5/K10 계약과의 정확한 대조

| 항목 | 현재 계약 | 판정 |
|---|---|---|
| non-DP 기준 | `M0` | 있음 |
| ordinary image-DP | K5 `M1-I8`: image `(8,1e-5)`, patient conversion vacuous | 있음 |
| group-matched record-DP | K5 `M1-G8`: image `(1.6,1.32654e-8)` -> patient `(8,1e-5)` | 있음 |
| native patient-DP | K5 `M2-P8`: direct patient `(8,1e-5)` | 있음 |
| 계산량 통제 | M1 8 images/step, M2 expected 8.023 images/step; ratio `1.002891` | 있음 |
| K5 fidelity/weak alignment | RAD-DINO KID/PRDC, BioViL-T | 있음 |
| K5 privacy attack | 의도적으로 미실행 | 없음; K5는 feasibility만 답함 |
| K10 patient MIA | 환자별 image score의 산술평균 후 전체 AUC/TPR | 있음, 그러나 average-risk 중심 |
| K10 extraction | 2,450-query near-copy 검사 | 있음, 그러나 patient biometric identity 없음 |
| 개별 환자 exposure tail | 없음 | 새 addendum 필요 |
| Nature식 patient-specific MIA AUC | 없음; 이를 위해 inclusion이 다른 다수 target models 필요 | 현 자원에서 바로 주장 금지 |

중요한 차이는 `한 모델에서 환자별 exposure score의 survival function`과 `동일 환자가 들어간/빠진
다수 모델로 추정한 patient-specific MIA AUC`가 같지 않다는 것이다. 현 자원에서 전자는 가능하지만
후자를 흉내 내어 같은 명칭을 쓰면 안 된다.

## 6. 이번에 실제로 한 공개 전제 검정

### 6.1 결과 전 동결한 설계

- K10 `public_development`의 1,816명 중 두 장 이상인 **886명 전원** 사용;
- 환자마다 order상 첫 영상을 query, 마지막 영상을 true gallery로 고정;
- 환자당 query 1장과 gallery 1장으로 record 수가 많은 환자의 retrieval 기회 증가 제거;
- 다른 환자 negative는 `sex x target x n-bucket` 16개 cell 안에서 hash cycle derangement;
- generic DINOv2와 NIH-pretrained RAD-DINO가 모두 통과해야 함;
- pair AUC, R@1, rank>10 tail, margin p10/p90, 두 encoder margin Spearman과 5,000 patient
  bootstraps를 결과 전에 고정.

프로토콜 SHA-256:
`DF95C8A154509BF782E87D38B31D002D6FB65DB19E446F9B4E3CAD5D1792BCD2`.

### 6.2 결과

| 지표 | DINOv2 | RAD-DINO |
|---|---:|---:|
| true/matched-negative pair AUC | `0.825795` | `0.988363` |
| AUC 95% bootstrap CI | `[0.808454,0.843345]` | `[0.984242,0.991894]` |
| R@1, gallery 886명 | `0.239278` | `0.711061` |
| R@1 95% CI | `[0.211061,0.267494]` | `[0.681687,0.739278]` |
| R@5 | `0.357788` | `0.848758` |
| R@10 | `0.422122` | `0.897291` |
| rank > 10 fraction | `0.577878` | `0.102709` |
| margin p10 | `-0.012564` | `-0.157388` |
| margin p90 | `0.002356` | `0.416736` |

R@1의 정확한 random chance는 `1/886=0.001129`이므로 DINO는 chance의 약 `212x`, RAD-DINO는
약 `630x`다. 두 encoder의 886개 patient margin Spearman은 `0.505432`, 95% bootstrap CI
`[0.451656,0.555337]`였다. 사전 conjunctive gate가 모두 통과했다.

해석은 제한적이다.

- 환자별로 exposure가 쉬운 쪽과 어려운 쪽이 동시에 있으며, 두 encoder 사이에도 중간 이상의
  ordering 일치가 있다. 따라서 뒤의 생성모델에서 환자별 tail endpoint를 측정할 **도구적 전제**는
  있다.
- RAD-DINO는 NIH를 pretrain에 사용했으므로 `0.711`을 외부 일반화 수치로 해석하지 않는다.
  generic DINO도 독립적으로 gate를 통과했기 때문에 endpoint 전체가 RAD pretraining만의 산물은
  아니다.
- record 수가 큰 일부 descriptive cell의 R@1이 오히려 낮았다. 첫--마지막 order gap이 커질 수
  있으므로 이를 `기록이 많을수록 안전하다`는 인과 결론으로 사용하지 않는다.
- 실제 generator, DP noise, optimizer, private-role data를 전혀 사용하지 않았다. 따라서 어떤
  생성모델도 환자 identity를 누출했다는 결과가 아니다.

실행은 212.37초, peak CUDA `0.651451 GiB`였다. feature/model tensor는 보존하지 않았다.

Artifact SHA-256:

- runner: `4F4CD96DCC3DED272C13B2C905336D9F6B2A9BCDA332CB1E64BB9992107BA748`
- tests: `8C8A14C56B1EB22DA32BCA3697F3A35D3B3AA446D8F25E4D7F1A25CF4104B73B`
- local selection: `03B1FE65F2EACAAC9964B0ED7F124C2EAC6D610F88574D464D327304E32984EC`
- aggregate report: `8CF040117B4528170CB10EACC254457F29DD1AA4F750CB31C4A28BCA8CA2290D`

## 7. 현재 순위와 CVPR 가능성

### 1순위 — Patient-unit generative exposure-tail study

현재 가진 데이터·코드·검증 자산으로 가장 실행 가능하다. K5/K10의 formal privacy-unit matrix가 이미
준비되어 있고, 이번 결과로 biometric tail endpoint도 실제 신호와 비포화 구간을 가진다는 것이
확인됐다. 실패한 SetAdapter나 longitudinal residual을 억지로 살릴 필요도 없다.

그러나 현재 상태의 종합 평가는 다음과 같다.

- **문제 타당성:** 높음
- **실행 가능성:** 높음
- **exact intersection 신규성:** 중간
- **새 알고리즘 기여:** 낮음
- **현재 CVPR main 경쟁력:** 중간 이하

CVPR main으로 강화하려면 최소한 다음 중 두 가지 이상이 필요하다.

1. native patient-DP가 group-matched image-DP보다 같은 patient budget에서 일관된
   fidelity/pathology/exposure frontier 우위를 보이는 실제 결과;
2. 한 encoder의 최근접거리만이 아니라 독립 MedReID, denoising-loss membership, near-copy extraction의
   patient-tail 일치와 불일치를 분석;
3. NIH 외 환자 ID가 있는 두 번째 CXR 데이터셋의 완전 독립 재현;
4. 기존 ULDP를 그대로 쓰는 것을 넘어 diffusion-specific tail을 줄이는 새 mechanism;
5. accounting--execution--attack claim을 연결하는 공개 benchmark/receipt 패키지.

### 2순위 — richer-EHR longitudinal patient-private diffusion

실제 timestamp+report/EHR 자료가 확보될 때만 재개한다. 현재 로컬 NIH route는 15번 결과로 종료됐고,
EHRXDiff·DDL-CXR 선행 때문에 일반적 previous-to-future generation은 신규성이 아니다.

### 종료 — biometric suppression / patient unlearning / pooled SetAdapter

각각 직접 선행 충돌 또는 실제 held-out 실패가 있어 현 구현 방향으로 재시도하지 않는다.

## 8. 다음 한 단계

장기 K5 runner를 바로 만들기 전에, 다음을 별도 addendum으로 결과 전에 동결한다.

1. 실제 생성 image를 private-train과 attack-holdout patient gallery에 동시에 대조하는 biometric
   exposure score;
2. public-development same/different-patient pair만으로 정한 threshold와 uniqueness margin;
3. `각 query의 최근접 환자`와 `각 훈련 환자의 최대 exposure`를 분리한 eSF;
4. M0/M1-I8/M1-G8/M2-P8 paired seeds와 patient-count/label/sex descriptive tails;
5. 이 score-eSF가 Nature식 patient-specific MIA AUC가 아니라는 명칭·claim 경계;
6. dedicated MedReID 모델의 weight/data-overlap/conformance gate.

이 addendum이 고정되기 전에는 16--21시간 K5 학습을 CVPR 방향 검증이라고 부르지 않는다. K5 자체는
오직 domain adaptation과 DP collapse feasibility를 보는 선행 gate로 유지하며, 기존 frozen 계약을
결과 없이 소급 변경하지 않는다.
