# 관련·근접연구 claim matrix와 신규성 경계

- 기록일: 2026-09-02
- 조사 기준일: 2026-09-02
- 대상: CVPR 2026 PFDM을 출발점으로 한 독립 의료영상 논문
- 결론 상태: 방향은 유지하되, 방법 기여의 표현을 더 좁혀야 함

## 1. 이번 재조사의 결론

현재 1순위인 환자 단위 private medical diffusion 방향은 여전히 가장 타당하다. 그러나
추가 경계 검색에서 다음 선행이 확인되었기 때문에, 단순한 “환자별 gradient 평균 후
clipping”이나 “계층적 gradient 평균”은 신규 기여로 주장하면 안 된다.

1. P3SGD는 CVPR 2019에서 이미 환자의 모든 병리 영상을 하나의 privacy unit으로 정의하고
   환자 단위 update와 clipping/noising을 제안했다.
2. HiGradAvgDP는 subject-level DP를 위해 hierarchical gradient averaging을 제안했다.
3. User-wise DP-SGD/ULS는 한 사용자에게서 여러 record를 뽑아 평균한 뒤 사용자 단위로
   clipping/noising하는 일반 알고리즘을 정립했다.
4. DPDM은 한 이미지에 여러 diffusion noise/timestep을 적용해 sanitization 전에
   평균하는 noise multiplicity를 제안했다.
5. augmentation multiplicity는 한 이미지의 여러 augmentation gradient를 clipping 전에
   평균하는 선행이다.

따라서 남는 기술적 칸은 다음과 같이 좁혀야 한다.

> 중앙집중형 patient-linked 의료영상 latent diffusion에서, 한 환자 안의 서로 다른 영상이
> 만드는 변동과 동일 영상의 diffusion noise/timestep이 만드는 변동을 분리하고, 고정된
> 환자당 계산 예산 아래 시각적 coverage와 stochastic repetition을 배분하는 patient-level
> gradient estimator를 만든다. 이 estimator를 환자 단위로 한 번 clipping/noising하고,
> 기존 image-DP checkpoint의 환자 단위 tight conversion과 동일 target patient budget에서
> 비교하여 reuse-versus-retrain frontier를 실증한다.

이 교집합을 그대로 수행한 논문은 이번 검색 범위에서 찾지 못했다. 이는 부재의 증명이
아니므로 투고 직전 동일 검색을 다시 수행해야 한다.

## 2. claim-by-claim 선행연구 매트릭스

| 축 | 대표 선행 | privacy unit·setting | 선행이 이미 점유한 내용 | 본 연구에 남는 내용 | 허용되는 주장 |
|---|---|---|---|---|---|
| PFDM | Patel et al., CVPR 2026 | client별 data point, federated/local DP | local denoiser와 global denoiser 분리, forward diffusion noise를 이용한 privacy, personalization | 중앙 patient-linked data, patient add/remove adjacency, DP-SGD형 patient clipping, 기존 checkpoint conversion | PFDM의 의료 확장이라고 하지 않고 문제 설정과 mechanism이 다르다고 명시 |
| 환자 단위 의료 SGD | P3SGD, CVPR 2019 | 환자의 모든 pathology image | patient adjacency, patient-level large-step update, clipping/noising | diffusion objective, LoRA, 두 확률축의 gradient estimator, 생성·복제·release frontier | “최초 patient-DP” 금지; “patient-private diffusion estimator”만 조건부 주장 |
| 계층 평균 | HiGradAvgDP, NeurIPS FL Workshop 2022 / arXiv | subject가 여러 user·record에 걸친 FL | hierarchical gradient averaging과 subject-level DP | 중앙 diffusion, 환자 내부 visual coverage와 noise/timestep 배분, matched patient-budget 생성 평가 | “계층 평균 자체가 새롭다” 금지 |
| 일반 user-wise DP | Chua et al., COLM 2024; Charles et al., SaTML 2025 | 사용자별 복수 text record | user sampling, 여러 record 평균, per-user clipping/noising, ELS 대 ULS와 fixed-compute 비교 | 의료영상 생성, 영상 coverage 기반 층화, diffusion stochasticity, patient attacks | ULS를 baseline·privacy backbone으로 인용하고 새 알고리즘처럼 포장하지 않음 |
| user-level vision | DP-FedEmb, CVPR 2023 | user-partitioned image embedding | 대규모 image embedding의 per-user sensitivity control과 private aggregation | 생성 모델, 의료 환자, diffusion multiplicity, conversion frontier | “vision에서 최초 user-DP” 금지 |
| tight group conversion | Ganesh, arXiv 2024; Charles et al. | capped user contribution, image/example sampling | DP-SGD의 group privacy를 mixture-of-Gaussians/PLD로 더 타이트하게 계산 | 실제 image-DP 의료 checkpoint에 적용하고 direct patient-DP와 release decision 비교 | 새 accountant 주장 금지; 검증된 accountant의 응용·감사 기여로만 서술 |
| DP diffusion | DPDM, TMLR 2023 | 한 이미지가 unit | DP-SGD diffusion, noise multiplicity, 1/K variance 감소 | 환자 단위 adjacency와 복수 영상 변동, patient-balanced objective, nested allocation | “noise multiplicity 제안” 금지 |
| DP diffusion fine-tuning | DP-LoRA, ICCV 2025 | 한 example/image가 unit | pretrained diffusion의 LoRA DP fine-tuning, per-example clipping, noise multiplicity | patient clipping, patient-linked repeated images, patient budget 및 공격 | LoRA 효율성은 implementation choice이지 새 기여가 아님 |
| DP latent diffusion | DP-LDM 및 후속 | 한 sample이 unit | public encoder/decoder와 private attention·denoiser 학습 | 환자 unit, 의료 반복 구조, reuse frontier | “최초 DP-LDM” 금지 |
| 최근 image-DP synthesis | PrivImage·dp-promise, USENIX 2024; SPTI, NeurIPS 2025; RPGen, AAAI 2026 | private image/sample 중심 | semantic-aware public pretraining, forward-noise 활용, private textual intermediary, robust public pretraining | patient add/remove adjacency, 반복기록 coverage, direct patient-DP 및 conversion frontier | 최신 utility 기준으로 인용하되 patient grouping 없는 수치를 직접 patient-DP 성능처럼 비교하지 않음 |
| 의료 DP diffusion | Daum et al., 2024 | 의료 volume/sample 단위 | 3D cardiac MRI DP latent diffusion과 controllable generation | 한 환자의 복수 record 보호, image-to-patient conversion, native patient-DP 비교 | “최초 의료 DP diffusion” 금지 |
| 의료 LDM memorization | Dar et al., Nature Biomedical Engineering 2026 | 환자 영상 복제 탐지 | 의료 LDM의 강한 memorization·copying 위험과 self-supervised copy detection | formal patient-DP가 위험·utility에 미치는 영향, patient unit attack, direct-vs-converted 비교 | copy detector를 새로 발명했다고 하지 않고 평가 도구로 활용 |
| 환자별 privacy audit | Knolle et al., Nature 2026 | 환자별 여러 임상 record | aggregate MIA가 개인 tail risk를 가릴 수 있음, patient-level eSF, record-DP의 한계 | 생성 모델에 맞는 scalable patient MIA·copy audit와 formal patient-DP 결합 | “최초 patient audit” 금지; 생성 모델 적용 및 formal mechanism 결합으로 한정 |
| 의료 비형식 safeguard | TMI 2026 filtering 계열 | 생성 후 feature-distance filtering | 유사 synthetic sample 제거 | training-time formal patient-DP와 filtering의 보완 관계 | filtering이 DP를 제공한다고 주장 금지 |
| DP image augmentation | De et al., 2022 | image/example 단위 | 여러 augmentation을 clipping 전 평균 | 서로 다른 patient record와 diffusion perturbation의 compute allocation | “pre-clipping multiplicity 자체가 새롭다” 금지 |
| timestep sampling | Kim et al., CVPR 2025 및 diffusion training 선행 | 비공개성 중심이 아닌 일반 diffusion | timestep별 gradient variance 차이와 adaptive/non-uniform sampling | public-development에서 고정한 timestep rule의 patient-DP 결합 | adaptive timestep 자체를 새 기여로 두지 않음 |
| 종합 지침 | How to DP-fy Your Data, JAIR 2026 | 여러 privacy unit | image unit과 multiple-images-per-user 문제, ULS/group conversion, sampler-accountant 정합성 | 의료 diffusion에서 end-to-end mechanism·실험·release decision | privacy-unit 문제 발견 자체를 새롭다고 하지 않음 |

## 3. 근접도 순위

### S급: 반드시 정면 비교해야 하는 연구

1. P3SGD
   - 환자를 privacy unit으로 삼는 의료영상 선행이므로 가장 직접적인 반례다.
   - 본 논문은 P3SGD를 diffusion에 단순 이식한 것으로 보이지 않도록 해야 한다.
2. User-wise DP-SGD / ULS
   - 환자 sampling, record 평균, patient clipping의 표준 backbone이다.
   - 본 방법의 baseline은 naive ULS여야 한다.
3. DPDM
   - diffusion noise multiplicity를 이미 점유한다.
   - 동일 계산량에서 한 이미지 반복과 서로 다른 환자 영상 coverage를 직접 비교해야 한다.
4. Daum et al. 의료 DP-LDM
   - 의료영상 DP diffusion의 직접 선행이다.
5. Nature 2026 patient-level privacy audit
   - 환자별 tail-risk 평가의 직접 선행이며, aggregate MIA만 보고하면 설득력이 떨어진다.

### A급: 방법·accounting 경계를 정하는 연구

- HiGradAvgDP
- DP-FedEmb
- DP-LoRA
- PrivImage, dp-promise, SPTI, RPGen
- tight group privacy for DP-SGD
- augmentation multiplicity
- medical LDM memorization
- CVPR 2025 adaptive timestep sampling

### B급: 응용·보조 비교

- PFDM 및 personalized federated diffusion
- 의료영상 생성 후 유사도 filtering
- non-DP memorization mitigation와 unlearning
- 일반 DP synthetic-data fairness 연구

## 4. 이미 점유된 claim과 사용 가능한 claim

### 사용 금지

- 최초의 patient-level DP
- 최초의 patient-level medical image training
- 최초의 hierarchical patient gradient averaging
- 최초의 user-level DP vision
- 최초의 DP diffusion 또는 DP medical diffusion
- 최초의 noise/augmentation multiplicity
- 최초의 patient-level medical privacy audit
- forward diffusion noise만으로 일반적인 patient-DP가 성립한다는 표현

### 조건부로 사용할 수 있음

아래는 검색 결과가 아니라 실제 구현과 실험으로 입증해야 한다.

1. 중앙 patient-linked medical latent diffusion을 위한 patient-balanced LoRA fine-tuning.
2. 환자 내부의 visual-record variance와 diffusion perturbation variance를 분리한 분석.
3. fixed patient compute에서 distinct-image coverage와 noise/timestep repetition을
   compute-matched하게 비교하는 nested multiplicity.
4. frozen visual embedding으로 환자 영상을 coverage strata로 나눈 뒤 unbiased하게
   환자 gradient를 추정하는 방법.
5. 동일 target patient epsilon/delta에서 image-DP generic conversion, image-DP tight
   conversion, native patient-DP를 한 실험계에서 비교한 reuse-versus-retrain frontier.
6. 생성 품질뿐 아니라 환자 단위 membership, copy/memorization, rare-condition utility,
   runtime을 함께 제시하는 평가.

### 현재 가장 안전한 중심 claim

> 환자 단위 평균이나 clipping을 새로 제안하는 것이 아니라, patient-wise DP diffusion의
> 제한된 gradient 계산을 환자 내부 시각적 coverage와 diffusion stochasticity 사이에
> 배분하는 estimator를 제안하고, 동일 환자 privacy budget에서 그 utility와 release
> consequence를 검증한다.

## 5. 로컬 데이터가 이 질문을 실제로 지지하는가

최신 승인 결정은 frontal chest X-ray를 첫/core modality로, 동결된 ISIC을 별도
cross-domain extension으로 둔다.

### 5.1 NIH ChestXray14 core 후보

현재 exact corpus는 아직 동결되지 않았고, 아래 수치는 provenance preflight metadata의
예비 집계다.

| scope | images | patients | 2장 이상 | 5장 이상 | 10장 이상 | 최대 |
|---|---:|---:|---:|---:|---:|---:|
| 전체 frontal | 112,120 | 30,805 | 13,302 | 5,759 | 2,545 | 184 |
| PA only | 67,310 | 28,868 | 10,688 | 3,245 | 953 | 100 |

preferred PA-only 정책에서도 반복 영상 환자가 10,688명이라 PCM을 검증할 절대 표본 수는
충분하다. 동시에 PA-only 환자 중 18,180명은 한 장만 제공하므로, PCM의 이득이 모든
환자에게 발생한다고 가정하면 안 된다. 현실적인 all-patient capped cohort를 주 분석으로
두고, repeated-patient strata에서 효과가 record count와 함께 커지는지를 별도로 본다.
privacy claim은 평균 record count가 아니라 공개 cap K의 worst case를 사용한다.

공식 train/validation list와 test list는 각각 28,008명과 2,797명이며 patient overlap이
없다. 다만 main metadata mirror가 official file과 3 bytes 다르므로, exact-source gate가
해결되기 전에는 이 수치를 frozen manifest나 학습 완료 증거로 부르지 않는다.

### 5.2 ISIC cross-domain extension

이미 동결한 manifest를 직접 집계한 결과는 다음과 같다.

| 조건 | private-train 환자 | 환자당 영상 수 분포의 핵심 | 복수 구조 |
|---|---:|---|---|
| K5 | 1,415 | 1,203명이 4장 이상, 1,065명이 5장 | 전원 복수 lesion, 1,224명이 복수 anatomical site |
| K10 | 1,415 | 849명이 8장 이상, 770명이 10장 | 전원 복수 lesion, 1,300명이 복수 anatomical site |

ISIC에서는 m개의 서로 다른 영상을 택하는 실험이 거의 전체 환자에 적용되므로,
X-ray의 long-tailed contribution 구조와 다른 stress condition을 제공한다. 반대로
ISIC의 영상은 동일 병변의 multi-view가 아니라 서로 다른 lesion·site가 섞일 수 있다.
논문에서는 두 데이터셋을 모두 “multiple patient-linked images”라고 부르고, X-ray
PA-only에는 “longitudinal/repeated radiographs”, ISIC에는 “multiple lesions/sites”라는
정확한 하위 설명을 사용한다.

근거 파일:

- code_working/XRAY_CORE_PIVOT_DECISION.md
- code_working/_reports/isic2020_split_v1_001/manifest_summary.json
- code_working/_data/derived/isic2020_v2_split_v1/k5_private.csv
- code_working/_data/derived/isic2020_v2_split_v1/k10_private.csv

## 6. 신규성 위험 판정

| 구성 | 신규성 위험 | 판단 |
|---|---:|---|
| ULS + LoRA만 수행 | 매우 높음 | P3SGD + user-wise DP + DP-LoRA의 조합으로 보임 |
| ULS + DPDM noise multiplicity | 높음 | 기존 두 방법의 직접 결합으로 보임 |
| image-DP 대 patient-DP audit만 수행 | 중간~높음 | 중요한 benchmark지만 CVPR method contribution이 약함 |
| uniform distinct-first nested multiplicity + variance 분석 | 중간 | 명확하지만 단순한 조합이라는 비판 가능 |
| visual-coverage stratification + nested multiplicity + matched frontier | 중간~낮음 | 가장 강한 후보이나 실제 variance·utility 개선이 필수 |

이에 따라 권장 주 방법은 visual-coverage stratified patient estimator이고,
uniform distinct-first estimator는 반드시 포함할 강한 baseline이자 실패 시 fallback이다.

## 7. 논문 작성 시 related-work 문단의 논리

1. example-level DP diffusion은 발전했지만 image를 privacy unit으로 둔다.
2. 의료에서는 한 환자가 여러 영상을 제공하므로 patient unit이 필요하며, P3SGD와
   user-wise DP가 그 일반 원리를 이미 제공한다.
3. 그러나 diffusion에서는 각 image마다 다시 timestep/noise stochasticity가 생긴다.
4. 기존 noise multiplicity는 한 image 내부 확률만 줄이고, user-wise DP는 patient 내부
   visual heterogeneity와 diffusion stochasticity의 계산 배분을 다루지 않는다.
5. 본 연구는 이 두 변동원을 fixed compute 아래 분리·배분하고, 직접 patient-DP와
   converted image-DP의 실제 release frontier까지 평가한다.

이 순서이면 기존 연구를 축소하지 않으면서 정확한 빈칸을 설명할 수 있다.

## 8. 주요 1차 문헌

- PFDM, CVPR 2026:
  https://openaccess.thecvf.com/content/CVPR2026/papers/Patel_Personalized_Federated_Training_of_Diffusion_Models_with_Privacy_Guarantees_CVPR_2026_paper.pdf
- P3SGD, CVPR 2019:
  https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_P3SGD_Patient_Privacy_Preserving_SGD_for_Regularizing_Deep_CNNs_in_CVPR_2019_paper.html
- HiGradAvgDP:
  https://arxiv.org/abs/2206.03617
- DP-FedEmb, CVPR 2023:
  https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Learning_To_Generate_Image_Embeddings_With_User-Level_Differential_Privacy_CVPR_2023_paper.pdf
- Mind the Privacy Unit, COLM 2024:
  https://arxiv.org/abs/2406.14322
- Learning with User-Level DP under Fixed Compute Budgets:
  https://arxiv.org/abs/2407.07737
- Tight group privacy for DP-SGD:
  https://arxiv.org/abs/2401.10294
- DPDM, TMLR 2023:
  https://arxiv.org/abs/2210.09929
- DP-LoRA, ICCV 2025:
  https://openaccess.thecvf.com/content/ICCV2025/papers/Tsai_Differentially_Private_Fine-Tuning_of_Diffusion_Models_ICCV_2025_paper.pdf
- PrivImage, USENIX Security 2024:
  https://www.usenix.org/conference/usenixsecurity24/presentation/li-kecen
- dp-promise, USENIX Security 2024:
  https://www.usenix.org/conference/usenixsecurity24/presentation/wang-haichen
- SPTI, NeurIPS 2025:
  https://proceedings.neurips.cc/paper_files/paper/2025/hash/38380a17248c4dfc4f48fc4d836d9b62-Abstract-Conference.html
- RPGen, AAAI 2026:
  https://ojs.aaai.org/index.php/AAAI/article/view/38016
- Medical DP latent diffusion:
  https://arxiv.org/abs/2407.16405
- Disparate privacy risks from medical AI, Nature 2026:
  https://www.nature.com/articles/s41586-026-10688-0
- Medical LDM memorization:
  https://www.nature.com/articles/s41551-025-01468-8
- Adaptive non-uniform timestep sampling, CVPR 2025:
  https://openaccess.thecvf.com/content/CVPR2025/html/Kim_Adaptive_Non-Uniform_Timestep_Sampling_for_Accelerating_Diffusion_Model_Training_CVPR_2025_paper.html
- How to DP-fy Your Data, JAIR 2026:
  https://arxiv.org/abs/2512.03238

## 9. 최종 경계

이 조사로 1순위가 탈락한 것은 아니다. 대신 “patient aggregation을 한 diffusion”이라는
넓은 설명은 더 이상 충분하지 않다. CVPR형 독립 논문이 되려면 중심은 다음 세 항목이어야
한다.

1. visual-coverage-aware, diffusion-specific patient gradient estimator
2. fixed-compute 및 fixed-patient-privacy의 엄격한 비교
3. direct/converted guarantee가 만드는 재사용·재학습 경계

세 항목 중 첫 번째가 pilot에서 유의한 variance·clipping·utility 개선을 보이지 않으면
CVPR method paper로 강행하지 않고, audit/benchmark 논문 또는 의료영상·privacy venue로
재구성한다.

## 10. 2026-09-02 실제-gradient 결과 이후의 정정과 추가 선행

이 절은 Section 6의 “visual-coverage stratification이 가장 강한 후보”라는 사전 판단을
실험 결과에 따라 supersede한다.

### 10.1 폐기된 부분

- 16-patient ISIC SD 2.1 diagnostic에서 VAE coverage/uniform-distinct geometric MSE ratio는
  `1.0322`, bootstrap 95% CI는 `[0.9674, 1.1005]`, VAE wins는 6/16이었다.
- outcome-informed paired-strata oracle조차 uniform 대비 `0.9749`, 8/16 wins에 그쳤다.
- 따라서 frozen VAE visual coverage와 5-record pair-stratified family는 주 방법 후보에서
  폐기한다. adverse/null 결과를 지우거나 feature를 교체해 다시 이름만 유지하지 않는다.

### 10.2 새로 확인한 강한 충돌

| 선행 | 현재 후보와의 충돌 | 허용 경계 |
|---|---|---|
| VDM official code | minibatch 전체에 evenly spaced antithetic time 배치 | antithetic time 자체 claim 금지 |
| DDIM official code | `t`와 `T-t-1` timestep pairing | timestep pairing 자체 claim 금지 |
| DPDM | image별 여러 noise level/noise를 clipping 전에 평균 | pre-clipping timestep multiplicity claim 금지 |
| Ghalebikesabi et al. | image별 여러 timesteps와 augmentations를 clipping 전에 평균 | DP timestep multiplicity claim 금지 |
| CleanDIFT, CVPR 2025 | training image마다 세 timestep bins의 noise levels 사용 | equal-width stratification claim 금지 |
| HDiT, ICML 2024 | 일반 diffusion training의 stratified timestep sampling | training trick을 방법 신규성으로 주장 금지 |
| Adaptive timestep, CVPR 2025 | timestep gradient variance와 adaptive marginal sampling | timestep variance/selection 자체 claim 금지 |

### 10.3 현재 남은 좁은 후보와 증거

새 16 patients, 5 records/patient, 8 perturbations의 confirmatory에서 record 네 장과 네
timestep bins를 한 patient update 내부에서 함께 균형화한 estimator는 iid-replacement보다
geometric MSE `0.8592`, bootstrap 95% CI `[0.7931, 0.9220]`, 16/16 wins였다.

generic perturbation without-replacement와 exact 비교한 결과 true-bin/distinct ratio는
`0.9211`, CI `[0.8814, 0.9572]`, 16/16 wins였다. 실제 chronological grouping은 105개
pair partitions 중 13/16 patients에서 1위, 나머지는 4·4·11위였다.

남는 후보는 **patient vector를 clipping하기 전 privacy-unit 내부에서** record marginal과
timestep marginal을 동시에 균형화하는 배치 위치다. 하지만 이는 `ULS + DPDM + stratified
timestep`의 자명한 결합으로 보일 위험이 크다. independent perturbation banks, batch-global
대 unit-local ablation, variable-record unbiased rule, clipping 이론, X-ray와 patient-DP
utility를 통과하기 전에는 신규 방법 claim을 하지 않는다.

추가 1차 문헌:

- VDM official code:
  https://github.com/google-research/vdm/blob/main/model_vdm.py
- DDIM official code:
  https://github.com/ermongroup/ddim/blob/main/runners/diffusion.py
- DPDM full text:
  https://arxiv.org/html/2210.09929v3
- DP diffusion augmentation/timestep multiplicity:
  https://arxiv.org/html/2302.13861
- CleanDIFT:
  https://arxiv.org/abs/2412.03439
- HDiT:
  https://proceedings.mlr.press/v235/crowson24a.html

### 10.4 새 5-bank replication 이후의 상태

단일 8-bank 결과의 우연을 검사하기 위해 결과 전에 다섯 새 bank tag와 seed rule을
고정하고, 동일한 16 patients에서 bank당 640개씩 총 3,200개 실제 LoRA gradients를 추가
계산했다. bank별 true-bin/distinct ratios는 `0.9354`, `0.9170`, `0.7862`, `0.9356`,
`0.9177`이고 모든 bank가 방향 및 wins gate를 통과했다.

equal-bank geometric ratio는 `0.8965`, bank와 patient를 각각 resample한 hierarchical
bootstrap 95% CI는 `[0.8309, 0.9419]`였다. 5/5 direction의 사전 sign probability는
`1/32`이고 독립 aggregate audit가 같은 값을 재현했다.

이는 timestep-bin gradient 신호가 한 bank의 우연이라는 우려를 줄이지만 신규성 위험을
줄이지는 않는다. 다음 필수 차별화는 batch-global antithetic/stratified sampling과
patient-unit-local pre-clipping placement의 직접 비교다. 여기서 local placement의 고유한
post-clipping 이득이 없으면 `ULS + known timestep stratification` 이상의 방법 claim을 하지
않는다.

### 10.5 Patient-marginal clipping proxy 이후의 경계

2026-09-03 사전 고정한 B=2 구성에서는 batch 전체가 같은 여덟 perturbation을 한 번씩
사용하도록 맞추고, global-only는 임의 4+4 배정, unit-local은 각 bin의 두 perturbation을
1+1로 배정했다. 저장된 다섯 bank의 within-patient Gram을 exact enumeration한 결과,
primary `C=0.20`의 unit-local/global-only postclip MSE ratio는 `0.9094`, hierarchical 95% CI는
`[0.8446, 0.9506]`, 방향은 5/5 banks와 78/80 patient-bank cells에서 일치했다.

따라서 “clipping 뒤에는 pre-clipping 신호가 사라진다”는 우려는 patient marginal 수준에서는
지지되지 않았다. 그러나 이 결과로 관련연구 충돌이 해소된 것은 아니다. cross-patient Gram이
없어 실제 batch-average update와 global complementary allocation의 환자 간 covariance를
평가하지 못했기 때문이다. 현재 허용되는 표현은 **joint-batch 검정으로 진행할 empirical
headroom이 있다**까지다. “최초 patient-local timestep stratification” 또는 “global balance보다
실제 DP update가 우수하다”는 claim은 아직 금지한다.

### 10.6 Actual joint-batch 이후의 claim 경계

후속 full cross-patient Gram test에서 primary joint local/global ratio가 `1.0064606`, bank
bootstrap 95% CI가 `[1.0020765, 1.0117106]`로 나와 사전 locality gate를 실패했다. 새 banks의
patient marginal은 `0.9429900`으로 개선됐지만 global complementary allocation의 더 큰 음의
cross-patient covariance가 최종 batch MSE를 낮췄다.

따라서 이제 다음 claim도 금지한다.

- patient-unit-local timestep balance가 minibatch-global balance보다 실제 clipped update를
  개선한다는 주장
- within-patient proxy를 joint-batch 또는 DP-training utility evidence로 사용하는 주장
- 불리한 primary `C=0.20`을 버리고 secondary clipping point로 같은 가설을 구제하는 주장

관련연구와 구별될 수 있는 다음 후보가 있다면 “unit-local stratification”이 아니라,
privacy-unit marginal과 cross-unit covariance를 함께 제어하는 batch-level allocation이어야
한다. 다만 이는 VDM/DDIM의 batch antithetic sampling 및 일반 variance-reduction 문헌과 다시
충돌할 가능성이 크므로, 신규성 검색과 공개정보 기반 실행 가능성을 먼저 확인해야 한다.
이번 결과만으로 그러한 새 방법이 존재하거나 신규라고 주장하지 않는다.

### 10.7 Covariance-aware discovery 이후의 종료 경계

35개 public-rank complementary templates를 leave-one-bank-out으로 전수 조사한 결과,
low-two-bins 대 high-two-bins `rt_0123`이 5/5 held-out banks에서 선택됐고 ratio는
`0.961428`, CI `[0.944857, 0.976267]`이었다. 방향은 일관됐지만 사전 최소 5% 개선과
5/5 pair-support를 충족하지 못했다. pair-specific hindsight oracle도 `0.941876`로 절대
headroom이 작았다.

따라서 covariance-aware라는 이름으로 timestep allocation을 계속 변형하지 않는다.
특히 `rt_0123`은 patient marginal을 `1.1281`로 악화시키면서 low/high timestep batch
segregation의 음의 covariance를 이용하므로, patient-specific 신규성보다 기존 global
antithetic·stratified sampling과 더 가까운 결과다.

현재 claim matrix에서 **실증적으로 살아 있는 신규 method claim은 없음**으로 갱신한다.
상위 patient-level DP medical diffusion 문제와 image-DP 대 patient-DP audit 질문은 남지만,
이를 CVPR 방법 기여라고 부르려면 timestep 배분과 다른 mechanism class에서 선행 비중복
가설을 새로 만들고 독립 자료로 처음부터 검정해야 한다.

## 11. 2026-09-03 reset: UCAN 종료와 patient-set generation의 새 claim 경계

이 절은 1--10의 gradient/timestep-allocation 중심 추천을 supersede한다. 이전 실패를 threshold,
template 또는 새 이름으로 구제하지 않는다.

### 11.1 추가로 닫힌 correlated-noise 분기

같은 환자의 두 records에 `z,-z` corruption noise를 주는 UCAN을 NIH 공개 자료의 320 actual
full gradients로 검정했다. 실제 patient clipping 뒤 complete UCAN/shared-t IID B=2 error ratio는
`1.074199`, bootstrap CI는 `[0.965688,1.206865]`였고 patient leave-one-out win은 0/16이었다.
noise-only antithetic/ordinary IID도 `1.017928`이었다. 따라서 다음 claim을 금지한다.

- antithetic corruption noise가 patient-DP diffusion update를 개선한다는 주장
- unclipped outlier 감소를 실제 clipped DP update의 개선으로 바꾸어 말하는 주장
- 같은 결과의 shared-t secondary 신호를 사후에 주 방법으로 승격하는 주장

Blue Noise와 일반 antithetic SGD가 이미 correlated-noise/variance-reduction 경계를 차지하고
있다는 신규성 위험까지 고려해 이 branch는 종료한다.

### 11.2 새로 남기는 정확한 교집합

새 1순위는 **Patient-Set Private Diffusion (PSPD)**다. 허용되는 중심 문장은 다음과 같다.

> 가변 길이의 반복 영상을 가진 한 환자를 add/remove privacy unit으로 정의하고, 그 영상 묶음을
> permutation-equivariant하게 함께 조건화·생성하여, stable within-patient visual structure와
> record-specific finding variation을 동시에 보존하는 patient-level DP diffusion을 학습한다.

이 claim은 아래 개별 선행의 단순 재명명이 아니어야 한다.

| 선행 축 | 이미 점유된 것 | PSPD가 별도로 입증해야 할 것 |
|---|---|---|
| P3SGD·user-level DP | 환자/사용자의 여러 records를 한 privacy unit으로 clipping/noising | diffusion의 joint set-valued model과 생성 효용 |
| DPDM·DP-LoRA·medical DP-LDM | DP diffusion 학습과 parameter-efficient adaptation | repeated-patient set adjacency, set interaction, patient-set 평가 |
| M2M·multi-view diffusion | 여러 관련 이미지를 공동 또는 순차 생성 | patient add/remove DP와 의료 반복기록의 consistency/diversity trade-off |
| EHRXDiff·longitudinal CXR diffusion | 과거 기록을 이용한 시간적 CXR 생성 | real-patient linkage 없이 synthetic bundle 전체를 생성·보호하는 목표 |
| MedReID | 의료영상에 남는 환자 식별 신호 | 그 신호가 효용과 노출 위험 양쪽에 미치는 patient-set 분석 |
| PFDM·FedDP-PALD | federated/personalized 또는 prototype-DP 의료 합성 | 중앙 patient-level DP joint-set 문제; federation은 첫 논문의 필수 claim이 아님 |

### 11.3 novelty 표현의 제한

2026-09-03까지 확인한 1차 문헌에서는 `patient add/remove DP + joint synthetic medical image
set generation`을 동시에 중심 문제로 둔 직접 선행을 찾지 못했다. 그러나 이는 검색 범위의
부재 증거일 뿐이다. 더 넓은 set generation, longitudinal synthesis, subject-level DP와
anonymization 검색을 끝내기 전에는 **first**, ** 최초**, **유일**을 사용하지 않는다.

PSPD 자체도 아직 검증된 method claim이 아니다. 첫 frozen DINO data-premise probe는 사전 다섯
조건 중 네 개만 통과해 공식 `FAIL`이다. AUC `0.877852`, R@1 `0.65625`와 record-label variation은
비중복 scale-free confirmation을 실행할 근거일 뿐 generator/DP utility evidence가 아니다.
구조·생성·private gates와 중단 기준은 10·12 문서를 따른다.

### 11.4 독립 확인과 구조 smoke 뒤 허용 범위

앞선 240명을 모두 제외한 exact 2--3 record 160-patient cohort에서 결과 전 동결한 scale-free
confirmation은 5/5 PASS했다. pooled AUC `0.880605`, 네 cell AUC `0.863333--0.908472`, 네 cell의
patient-ordering wins가 모두 `0.975`, global R@1이 chance의 `119.7x`였고 record-label variation도
통과했다. 따라서 **NIH public data에 patient-set premise가 있다**는 제한된 문장은 허용한다.
generic DINO 결과이므로 그 신호를 clinical identity 또는 anatomy라고 단정하는 문장은 금지한다.

실제 SD 2.1 Q=2 mid-block SetAdapter smoke도 parameter count, zero-init, permutation equivariance,
singleton identity, cross-record connectivity, full backward와 7.5-GiB memory gates를 모두 통과했다.
따라서 **현재 자원에서 제안 구조를 실행할 수 있다**는 문장은 허용한다. 그러나 다음은 여전히
금지한다.

- SetAdapter가 independent-generation baseline보다 우수하다는 주장
- set consistency 개선이 duplicate collapse 없이 얻어진다는 주장
- 새 LoRA+SetAdapter patient gradients에서 기존 clip norm을 그대로 재사용하는 것
- patient-DP utility, attack resistance, clinical coherence 또는 CVPR-ready 주장

다음 신규 method 판정은 parameter/UNet-call matched public non-DP generation falsification에서만
나온다. 그 결과 전까지 novelty gap은 **well-motivated and executable hypothesis**이지 실증된
method contribution이 아니다. 상세 수치와 hash는 13번 문서를 따른다.

### 11.5 Held-out context 반증 이후 PSPD method claim 종료

13번의 데이터 전제와 architecture smoke 뒤, full generation보다 먼저 same-patient companion을
실제로 쓰는지 검정했다. 표준 SetAdapter는 correct/bypass `0.933078`로 generic residual은
학습했지만 correct/shuffled가 `1.000214`, correct wins가 `64/128`이었다. target 0·1 ratio도 모두
1보다 컸다. 따라서 patient context가 아니라 own-token/generic capacity로 해석했다.

이 해석을 한 번만 구조적으로 확인하기 위해 direct own-token residual과 diagonal/self value path를
모두 제거한 strict cross-record-only adapter를 결과 전에 동결했다. 같은 training patients와 draws를
쓰고 앞선 496명을 제외한 fresh validation 32명에서 재검정했으나 correct/shuffled
`1.001070`, wins `65/128`, target별 ratio `1.001550`과 `1.000723`으로 다시 실패했다.
correct/bypass는 `0.939390`이어서 companion branch 자체는 작동했지만 same-patient companion의
추가 가치는 없었다.

따라서 다음 claim을 금지한다.

- standard IID denoising에서 pooled SetAdapter가 patient-specific context를 이용한다는 주장
- DINO same-patient retrieval을 joint generator utility의 대리 증거로 쓰는 주장
- correct/shuffled가 같은데 bypass 이득이나 set consistency만으로 method를 구제하는 주장
- self path를 다른 mask·attention 위치·timestep 규칙으로 바꾸며 같은 가설을 반복하는 것
- cross-record attention을 의료 특화 신규성으로 주장하는 것

SPAD(CVPR 2024), EpiDiff(CVPR 2024), MV-Adapter(ICCV 2025), CorrAdapter와 CAMEO(CVPR 2026)는
이미 cross-view interaction, correspondence와 consistency를 다룬다. strict cross-record block은
독립 신규 방법이 아니라 이번 원인 분리용 control이었다. 두 실패의 전체 수치와 artifact hash는
14번 문서를 따른다.

### 11.6 새 longitudinal-residual 후보의 좁은 claim 경계

현재 조건부 후보는 대칭적 set denoising이 아니라 initial synthetic CXR 뒤의 directed transition을
생성하는 Patient-Private Longitudinal Residual Diffusion(PPLRD)이다. 안정 구조는 이전 합성 영상에서
stop-gradient carrier로 전달하고, 경과·임상 변화 조건에 따른 residual transition을 학습하며, 한
환자의 전체 trajectory gradient를 한 번 clip/noise하는 가설이다.

그러나 generic longitudinal generation은 비어 있지 않다.

| 선행 | 점유 claim | 현 후보의 허용 경계 |
|---|---|---|
| EHRXDiff, CHIL 2025 | previous CXR+subsequent EHR로 future CXR 생성 | future-CXR conditioning 자체 금지 |
| DDL-CXR, NeurIPS 2024 | previous CXR+irregular EHR로 individualized latent CXR | personalized/temporal fusion 자체 금지 |
| LoCI-DiffCom·BrLP, MICCAI 2024 | longitudinal MRI completion/progression diffusion | time-conditioned residual 자체 금지 |
| conditional longitudinal radiology, MICCAI 2025 | future imaging embedding·report 생성 | generic future latent prediction 금지 |
| P3SGD | patient add/remove adjacency와 환자별 gradient | patient aggregation/clipping 자체 금지 |
| DP diffusion/medical DP-LDM | provably private diffusion image generation | DP를 붙였다는 사실만으로 신규성 주장 금지 |

남을 수 있는 검정 가능한 method claim은 **동일 patient-DP 예산에서 full-image transition보다
stop-gradient anatomy carrier+change-residual parameterization이 temporal change fidelity와 stable
anatomy를 함께 개선하는가**다. NIH `Follow-up #`는 actual elapsed time이 아닌 order proxy이므로
개발 진단에만 쓰며, final claim에는 timestamp와 report/EHR이 있는 MIMIC-CXR 또는 동등 자료가
필요하다. 이 조건을 통과하기 전 PPLRD도 검증된 method가 아니다.

추가 1차 문헌:

- EHRXDiff: https://proceedings.mlr.press/v287/kyung25a.html
- DDL-CXR: https://papers.neurips.cc/paper_files/paper/2024/file/3310034c97fab48fdbcba18f90fd5364-Paper-Conference.pdf
- Conditional longitudinal radiology diffusion:
  https://papers.miccai.org/miccai-2025/0164-Paper2656.html
- LoCI-DiffCom: https://papers.miccai.org/miccai-2024/paper/1964_paper.pdf
- BrLP: https://papers.miccai.org/miccai-2024/paper/0511_paper.pdf
- SPAD:
  https://openaccess.thecvf.com/content/CVPR2024/html/Kant_SPAD_Spatially_Aware_Multi-View_Diffusers_CVPR_2024_paper.html
- EpiDiff:
  https://openaccess.thecvf.com/content/CVPR2024/html/Huang_EpiDiff_Enhancing_Multi-View_Synthesis_via_Localized_Epipolar-Constrained_Diffusion_CVPR_2024_paper.html
- CAMEO:
  https://openaccess.thecvf.com/content/CVPR2026/html/Kwon_Correspondence-Attention_Alignment_for_Multi-View_Diffusion_Models_CVPR_2026_paper.html
- CorrAdapter:
  https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Align_Images_Before_You_Generate_CVPR_2026_paper.html

### 11.7 NIH order-proxy residual premise 결과와 PPLRD-v1 종료

앞선 여섯 selection의 528 unique patients를 제외하고, exact `Follow-up #` gap-1 pair를 가진 새
288 patients를 change/no-change와 train/validation으로 균형화했다. 30-dimensional weak-label
addition/removal로 future-minus-prior feature residual을 예측하고, training-only five-fold ridge,
deranged transition, prior-copy와 finding-matched shuffled-prior control을 결과 전에 고정했다.

changed-label validation 48 pairs에서 true-residual/prior-copy는 DINOv2 `1.140915`, 95% CI
`[1.073630,1.236754]`, wins `8/48`; RAD-DINO `1.159127`, CI `[1.101906,1.245540]`, wins
`1/48`이었다. weak-label transition model은 두 feature space에서 copy보다 명확히 나빴다.
true/deranged는 각각 `0.952614`, `0.978814`였지만 두 CI 모두 1을 포함했고 DINO wins도 50%였다.
이 secondary 신호로 primary 실패를 구제하지 않는다.

반면 correct same-patient prior/finding-matched shuffled prior는 DINO `0.516105`, CI
`[0.396098,0.685204]`, wins `46/48`; RAD-DINO `0.319953`, CI `[0.266193,0.378757]`, wins
`48/48`이었다. 따라서 patient-related carrier는 강하지만 label-conditioned change direction은
학습되지 않는다는 분리 결론을 낸다.

현재 금지 claim에 다음을 추가한다.

- NIH `Follow-up #`를 실제 time interval이나 clinical visit interval로 부르는 것
- prior identity-carrier PASS를 longitudinal change-generation evidence로 쓰는 것
- NIH weak labels만으로 residual transition이 full-image transition보다 유리하다는 주장
- 결과 뒤 encoder·alpha·gap·subset을 교체해 PPLRD-v1을 구제하는 것

현 NIH weak-label/order-proxy PPLRD route는 종료한다. MIMIC-CXR처럼 timestamp+report/EHR이 있는
자료가 실제 확보되면 그것은 richer conditioning의 새 가설·새 protocol이어야 한다. 전체 수치와
hash는 15번 문서를 따른다.

### 11.8 biometric suppression 충돌과 patient-exposure-tail로의 축소

추가 1차 문헌 검토로 biometric-aware generation의 넓은 방법 claim도 닫혔다.

| 선행 | 직접 점유 | 현 경계 |
|---|---|---|
| PrivDiff-Net, PMLR 2026 | diffusion cross-attention identity suppression + pathology-preserving guidance | identity 억제 loss/모듈 자체 금지 |
| Semantic versus Identity, ICCV 2025 | adjustable identity blocking + medical-semantic compensation + MDL disentanglement | identity/semantic 분리 자체 금지 |
| Patient-level LDM unlearning, MIDL 2026 | patient 단위 unlearning과 privacy--utility failure | patient removal 자체 금지 |
| Medical ULDP, 2025 | CheXpert에서 record-count-aware sampling/clipping과 naïve group conversion 비교 | 의료 user-level DP 효율 자체 금지 |
| Nature 2026 patient audit | 환자별 MIA AUC eSF와 extreme subgroup tail | aggregate-vs-patient tail 일반론 자체 금지 |
| CheXGenBench, TMLR 2026 | CXR T2I fidelity/privacy/utility 통합 benchmark | 일반 CXR privacy benchmark 자체 금지 |

현재 허용되는 가설은 `동일 patient budget·근사 동일 compute의 M1-I8/M1-G8/M2-P8`에
**생성물의 환자별 biometric exposure survival tail**을 결합한 exact intersection뿐이다. 이것은
새 anonymization network가 아니라 generative privacy-unit study다. Kaiser et al.과 Nature가 각각
classification ULDP와 classification MIA tail을 점유하므로, diffusion output의 ReID/extraction tail과
formal unit 차이를 함께 입증해야 한다.

긴 학습 전에 endpoint 자체를 검정했다. K10 public-development에서 두 장 이상인 886명 전원을 쓰고
환자당 first-order query와 last-order gallery를 하나씩만 둔 결과, DINO/RAD-DINO pair AUC는
`0.825795/0.988363`, R@1은 `0.239278/0.711061`, rank>10 비율은
`0.577878/0.102709`였다. 환자별 margin의 cross-encoder Spearman은 `0.505432`, 95% CI
`[0.451656,0.555337]`로 사전 gate를 통과했다. 이것은 **endpoint premise**이지 generator leakage나
DP success가 아니다. 한-model exposure-score eSF를 Nature식 multi-target-model patient-specific
MIA AUC로 부르는 것도 금지한다. 전체 문헌·수치·hash·다음 경계는 16번 문서를 따른다.
