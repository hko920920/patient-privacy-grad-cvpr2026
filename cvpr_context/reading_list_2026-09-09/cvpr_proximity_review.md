# 현재 CVPR 문헌의 근접성 정정

2026-09-09. **35편은 관련 문헌 전체이며, 모두 직접 경쟁하는 근접 연구라는 설명은 부정확했다.**

35는 기존 18번의 URL 인용에서 추출한 관련 문헌 수다. 아래 등급은 그 안에서 기여 중복을 대조하기 위해 선별한 기록이며, 최신 본학회 SOTA 비교군 전체나 선행 부재 증명이 아니다. 읽기 우선순위 15편과도 다른 분류다. 후속 확인에서 CVPR 2025 CDI와 NeurIPS 2025 Tracing the Roots의 직접 대조 누락이 드러났다. 발표 형식과 비교 범위는 22번 정정을 우선한다.

현재 기여의 핵심은 환자 단위 위험, 환자 집합 membership 감사, diffusion 점수 보정과 의료 patient-DP 비교다. 이를 이미 점유한 부분과 대조하기 위한 우선 후보를 아래와 같이 골랐다. 논문 수를 줄였다고 신규성이 확인되거나 성공 가능성이 높아진 것은 아니다.

[본학회 비교 정정: 발표 형식, CDI 등 누락과 비교 우선 축](<../22_CVPR_본학회_비교군과_6편선정_정정_2026-09-09.md>)

## 핵심 주장·방법 대조 — 6편

| 문헌 | 겹치는 핵심 | 여전히 다른 범위 | PDF |
|---|---|---|---|
| [Knolle / Disparate privacy risks](https://www.nature.com/articles/s41586-026-10688-0) | 평균 지표가 가리는 환자별 membership 위험과 여러 기록을 가진 환자의 감사 | 의료 분류모델 대상. 현재 후보는 의료 diffusion의 환자 집합 감사·보정. | [PDF](https://www.nature.com/articles/s41586-026-10688-0.pdf) |
| [Kaiser / Medical user-level DP](https://tpdp.journalprivacyconfidentiality.org/2025/pdf/kaiser.pdf) | 환자마다 다른 기록 수를 고려한 의료 user-level DP와 record-DP 변환의 비교 | CheXpert 분류의 DP 학습. 현재 후보의 새 생성모델 감사법 자체를 제시한 것은 아님. | [PDF](https://tpdp.journalprivacyconfidentiality.org/2025/pdf/kaiser.pdf) |
| [CheXGenBench](https://arxiv.org/abs/2505.10496) | 합성 흉부 X-ray의 프라이버시·품질·효용 통합 평가 | 기존 기록의 대조상 training-image 중심 평가. 환자 집합의 membership 보정과 같은 patient budget의 DP 비교는 별도 검증 대상. | [PDF](https://arxiv.org/pdf/2505.10496) |
| [FACE-AUDITOR](https://www.usenix.org/conference/usenixsecurity23/presentation/chen-min) | 한 사람의 이미지들이 학습에 쓰였는지 query set으로 감사 | few-shot 얼굴인식 대상. 의료 diffusion과 보장 단위가 같은 실험은 아님. | [PDF](https://www.usenix.org/system/files/usenixsecurity23-chen-min.pdf) |
| [User-level embedding MIA](https://arxiv.org/abs/2203.02077) | 정확한 훈련 이미지가 없어도 개인 단위 membership을 추론 | metric embedding·인물 재식별 대상. 의료 생성모델의 위험 보정은 별도. | [PDF](https://arxiv.org/pdf/2203.02077) |
| [Quantile-regression MIA](https://proceedings.mlr.press/v235/tang24g.html) | 비회원 reconstruction-loss 분포를 학습해 diffusion MIA의 표본별 임계값을 보정 | 개별 표본 검정. 환자 내 기록 수·상관을 고려한 집합 검정과 구분해야 함. | [PDF](https://raw.githubusercontent.com/mlresearch/v235/main/assets/tang24g/tang24g.pdf) |

## 직접 방법·평가 비교군 — 13편

| 문헌 | 현재 후보에서의 역할 | PDF |
|---|---|---|
| Dar / Medical LDM memorization | 의료 LDM의 환자 영상 암기·복제. 실제 노출 동기와 탐지기 비교. | [PDF](https://www.nature.com/articles/s41551-025-01468-8.pdf) |
| DeepSSIM++ | 의료 생성물 near-copy 감사. 새 감사법의 독립 비교 대상. | [PDF](https://arxiv.org/pdf/2609.03615) |
| Medical MIA split bias | 분할 편향을 실제 membership 신호로 오인하는 문제. member-only 대조의 근거. | [PDF](https://pure-oai.bham.ac.uk/ws/portalfiles/portal/304478106/BaiL2026Membership.pdf) |
| DIME | 최근 diffusion MIA 기준선. 위협모델·질의 예산을 맞춰 비교해야 함. | [PDF](https://arxiv.org/pdf/2608.22824) |
| PIA / PIAN | 효율적인 diffusion membership 공격. 환자 집합 감사에 들어갈 이미지 단위 근거. | [PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/ff73af253974e6144c4eddf896d3095a-Paper-Conference.pdf) |
| SecMI | diffusion membership inference의 기초 공격 비교군. | [PDF](https://proceedings.mlr.press/v202/duan23b/duan23b.pdf) |
| CLiD | 조건부 likelihood 차이를 이용한 text-to-image MIA. 조건 접근권이 같은 경우 비교. | [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/874411a224a1934b80d499068384808b-Paper-Conference.pdf) |
| Black-box diffusion MIA | black-box 접근조건의 공격 비교군. white-box 결과와 구분. | [PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/li25k/li25k.pdf) |
| Carlini / Training-data extraction | 훈련 이미지 추출. membership 추론·유사도·실제 복제의 차이 확인. | [PDF](https://www.usenix.org/system/files/usenixsecurity23-carlini.pdf) |
| MaMI / All-in-One MedReID | 의료영상 재식별. 환자 identity 신호와 평가 encoder의 직접 선행. | [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Tian_Towards_All-in-One_Medical_Image_Re-Identification_CVPR_2025_paper.pdf) |
| InvMM | inversion 기반 diffusion memorization 척도. 별도 관점의 감사 비교. | [PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Ma_An_Inversion-based_Measure_of_Memorization_for_Diffusion_Models_ICCV_2025_paper.pdf) |
| DP-LoRA | 현재 LoRA DP diffusion 학습의 직접 기준선. rank·학습 모듈·multiplicity 조건 대조. | [PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Tsai_Differentially_Private_Fine-Tuning_of_Diffusion_Models_ICCV_2025_paper.pdf) |
| Daum / Private 3D medical LDM | 의료 DP latent diffusion은 이미 존재. 반복 2D 환자 기록과의 문제 차이. | [PDF](https://arxiv.org/pdf/2407.16405) |

## 기반·확장 관련 문헌 — 16편

| 문헌 | 현재 후보에서의 역할 | PDF |
|---|---|---|
| Dombrowski / Privacy and fairness | 의료 합성 데이터의 일반화·프라이버시·공정성 평가. | [PDF](https://raw.githubusercontent.com/mlresearch/v301/main/assets/dombrowski26a/dombrowski26a.pdf) |
| Medical LDM filtering | 유사 생성물 filtering 기반 safeguard. formal DP와 구분. | [PDF](https://pmc.ncbi.nlm.nih.gov/articles/PMC13034593/pdf/nihms-2153980.pdf) |
| Patient-level diffusion unlearning | 환자 단위 unlearning과 privacy·utility의 한계. 환자 영향 제거 자체의 선행. | [PDF](https://openreview.net/pdf?id=x83b7ybcB1) |
| PrivDiff-Net | identity 억제와 pathology 보존. 생체신호 제거 방법의 직접 선행. | [PDF](https://raw.githubusercontent.com/mlresearch/v317/main/assets/akhter26a/akhter26a.pdf) |
| Semantic versus Identity | 의료 identity·semantic 분해 및 조절 가능한 비식별화. | [PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Tian_Semantic_versus_Identity_A_Divide-and-Conquer_Approach_towards_Adjustable_Medical_Image_ICCV_2025_paper.pdf) |
| MemControl | 학습 파라미터 선택을 통한 암기 완화. 단순 선택적 fine-tuning의 선행 경계. | [PDF](https://openaccess.thecvf.com/content/WACV2025/papers/Dutt_MemControl_Mitigating_Memorization_in_Diffusion_Models_via_Automated_Parameter_Selection_WACV_2025_paper.pdf) |
| Privacy-utility memorization mitigation | diffusion 암기 완화와 privacy·utility trade-off의 인접 방법. | [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Enhancing_Privacy-Utility_Trade-offs_to_Mitigate_Memorization_in_Diffusion_Models_CVPR_2025_paper.pdf) |
| P3SGD | 환자 단위 SGD·privacy의 선행. patient clipping 자체를 새 기여로 셀 수 없음. | [PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_P3SGD_Patient_Privacy_Preserving_SGD_for_Regularizing_Deep_CNNs_in_CVPR_2019_paper.pdf) |
| DP-FedEmb | 여러 이미지를 가진 사용자 단위 DP vision embedding 학습. | [PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Learning_To_Generate_Image_Embeddings_With_User-Level_Differential_Privacy_CVPR_2023_paper.pdf) |
| Charles / Fixed-compute ELS vs ULS | 같은 계산예산에서 example/user sampling 비교와 tight accounting. | [PDF](https://arxiv.org/pdf/2407.07737) |
| DPDM | DP diffusion과 noise multiplicity의 기초. 단순 DP diffusion은 이미 존재. | [PDF](https://arxiv.org/pdf/2210.09929) |
| DP-LDM | latent diffusion의 선택적 DP fine-tuning. DP-LoRA 이전 직접 선행. | [PDF](https://arxiv.org/pdf/2305.15759) |
| PrivImage | semantic-aware public pretraining을 이용한 DP image synthesis. | [PDF](https://www.usenix.org/system/files/usenixsecurity24-li-kecen.pdf) |
| dp-promise | DP diffusion image synthesis의 강한 기존 방법. | [PDF](https://www.usenix.org/system/files/usenixsecurity24-wang-haichen.pdf) |
| RAPID | retrieval을 활용한 DP diffusion 학습. public-data 활용 조건을 맞춰 비교. | [PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/5a04615295b7512ad3ce691400e26bc7-Paper-Conference.pdf) |
| PFDM | 처음 출발한 personalized federated diffusion. hospital/client와 내부 patient 단위를 구분. | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Patel_Personalized_Federated_Training_of_Diffusion_Models_with_Privacy_Guarantees_CVPR_2026_paper.pdf) |

기반 문헌도 인용과 비교가 필요할 수 있다. 직접 경쟁성, 실험 baseline 필요성, 먼저 읽을 순서는 서로 다른 기준이다. 이번 핵심 6편의 공식 원문·초록과 기존 18번 대조표를 다시 확인했으며, 35편 전부의 새 전문 정독·재현을 수행한 것은 아니다.

[현재 CVPR 관련 문헌 35편](cvpr_patient_dp.md) · [PDF 목록](cvpr_patient_dp.html) · [분류 명세](cvpr_proximity.json)
