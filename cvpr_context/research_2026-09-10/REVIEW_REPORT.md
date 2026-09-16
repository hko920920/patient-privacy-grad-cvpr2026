# CVPR 환자 DP 의료영상: 원문 대조와 기여도 재판정

**목적 확인:** [환자 보호·보장 환산 연구와 공격의 역할](purpose.html). 원래의 보호·재사용/재학습 비용 목표와, 아래에서 검토한 새 공격법 중심 후보를 구분한다. 후자의 채택은 원래 연구의 필수 단계가 아니며 자동 확정된 것으로 취급하지 않는다.

2026-09-10 · 기존 문헌 장부 67편 + 이번에 추가한 12편 = **79편의 개별 검토 기록**. 기존 탈옥 참고 4편은 현재 범위 밖이다. [전체 검색·드롭다운 HTML](index.html) · [비교군 CSV](comparison_matrix.csv) · [검토 장부 CSV](review_ledger.csv) · [기계 판독 비교 명세](comparison_plan.json).

**현재 결론은 ‘CVPR 가능성이 검증됐다’가 아니라 ‘강한 기존 방법과 비교해 반증해야 하는 후보’다.** 기존 6편이나 35편만으로 신규성을 충분히 검토했다는 인상을 준 것은 정정한다. 아래 대조에서는 집합 inference, caption-free 공격, 환자 평균 DP, 의료 Re-ID filtering 등의 선행이 이미 존재한다. 현재 주장만으로 새 방법의 기여를 확정할 근거는 부족하다.

[환자 집합 감사법을 우선 검증할 상세 근거](PATIENT_AUDIT_RATIONALE.md)에 문제 정의, 상관·오탐의 설명용 계산, 선행 반론, 신호 부재 가능성과 방향 전환 기준을 풀어 썼다. 관계 특징의 SD-MI(AAAI 2023)와 개인화 의료 모델의 Patient Membership Inference(IJCNN 2025)를 추가 대조했다. 이 두 편은 기존 79편 장부와 별도 보완 기록이다. 상관 활용·환자의 비학습 자료 활용 자체는 신규성이 아니며, 우선 검증할 후보라는 판단을 투고 성공 예측으로 해석하지 않는다.

## 실제 검토 범위

| 범위 | 이번에 수행한 일 | 수행하지 않은 일 |
|---|---|---|
| 로컬 PDF 75편 | 원문 방법·정의·가정·실험 또는 해당 방향의 평가 설계 선택 절을 읽고 개별 비판 기록 | 75편 모두의 모든 페이지·증명 전체 정독, 모든 실험 재현 |
| A20, A21, B17 | 공식 본문 검색 노출의 방법/결과 일부 대조 | 접근 제한된 PDF 전체 확인 |
| C02 | 저자 홈페이지 초록·발표 형식 확인 | 방법 수식·결과표 검토 |
| CLiD, Tracing the Roots | 공식 공개 점수/feature packet으로 CPU 평가 재계산 | diffusion feature 추출부터의 전체 재현, 의료 환자 공격 성공 검증 |
| 공통 평가 준비 | 환자 단위 aggregation·독립 calibration·member-positive ROC 도구와 오류 검증 | private 학습·K5/K10 본학습·동결 시험집합 평가 |

PDF별 읽은 페이지, 확인 못 한 부분, 원문 링크는 개별 기록에 표시했다. 짧은 선택 절 검토와 깊은 방법·부록 대조의 차이를 숨기지 않는다. 기존 목록을 모두 다뤘다는 의미의 coverage이며, 학계의 관련 논문을 빠짐없이 망라했다는 선언이 아니다. A02는 Nature 최종본 대신 저자 preprint를 읽었다. 다운로드·코드 확보는 재현 완료로 계산하지 않는다.

## 지금 바로 기여에서 빼야 할 주장

| 단독으로는 약한 주장 | 이미 겹치는 선행 | 우리가 추가로 입증해야 할 부분 |
|---|---|---|
| 여러 이미지의 약한 membership 신호를 묶는다 | CDI의 dataset inference, SD-MIA의 10/30장 set 평가 | 상관된 가변 크기 환자 bag에서 기존 feature+표준 집계보다 필요한 새 기전 |
| training caption 없이 공격한다 | MoFit, PFAMI 및 caption을 대체하는 black-box 연구 | 같은 접근 권한·보조 정보·계산량에서의 차이 |
| 환자 gradient를 평균하고 clipping한다 | ULS/ELS, Mind the Privacy Unit, P3SGD, subject granular DP | 실제 patient 수의 tight 회계와 같은 연산량에서 독립적인 개선 |
| 환자별 최고/꼬리 위험이 평균보다 크다 | Nature patient audit, 의료 prompt tail 연구 | 기존 평가가 놓치는 위험의 정의와 재현 가능한 새로운 검출 능력 |
| CXR에서 같은 환자를 찾거나 유사한 합성물을 걸러낸다 | Packhäuser, MaMI, Anonymous CXR, CheXGenBench, DCM-DeID | 자연적인 identity 유사도와 실제 training membership의 분리 |
| DP 생성의 효용을 높인다 | DP-LoRA, DP-LDM, RAPID, PrivImage, DP-FETA/FETA-Pro, RPGen | 공개자료·전체 privacy cost·연산량·module을 맞춘 우월성 |

근거: [CDI, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Dubinski_CDI_Copyrighted_Data_Identification_in_Diffusion_Models_CVPR_2025_paper.html), [SD-MIA 개별 원문 대조](reviews/X04.md), [MoFit, ICLR 2026](https://arxiv.org/abs/2602.22689), [ULS/ELS](https://arxiv.org/abs/2407.07737), [Nature patient audit](https://www.nature.com/articles/s41586-026-10688-0), [Anonymous CXR](https://arxiv.org/abs/2211.01323). 정확한 수식·실험 조건은 각 메모를 우선한다.

특히 Nature 연구의 patient-specific AUC는 여러 target model을 학습해 개별 record 위험을 추정한 다음 환자별로 요약하는 설계다. 고정된 한 generator에 환자 사진 묶음을 넣어 얻는 patient-bag ROC와 다르다. 두 설계를 혼동해서 ‘환자 감사에는 200개 diffusion 본학습이 필수’라고 결론내리지 않는다. [A01 상세](reviews/A01.md)

## 직접 대비할 공격들

학회 수준은 중요한 근거지만, 접근 권한이 다르면 같은 표에서 숫자만 비교할 수 없다. 아래는 **13개 방법 계열의 후보 목록**이며 전부를 동일한 의무 실험으로 정한 것이 아니다. 주장하는 접근 조건에 맞춰 아래 branch를 사용한다. Mean/max/median/top-k 집계와 적절한 표준 보정을 모든 유효한 단일영상 점수에 붙인 기준선도 필요하다.

| 접근/주장 | 비교할 방법 | 필요한 공정성 조건 |
|---|---|---|
| denoiser forward 접근 | PIA/PIAN, SecMI, CLiD, PFAMI, DIME | 동일 noise/timestep·caption 정보·forward NFE; DIME는 최신 preprint로 구분 |
| 환자별 threshold/보정 | Quantile-regression MIA + 위 점수들의 동일 patient calibration | 같은 nonmember 자료, calibration/test 분리, 환자 크기별 실제 FPR |
| 묶음의 membership 기여 | CDI, 표준 bag 집계, 허용 API라면 SD-MIA | 같은 bag 크기·상관·member 비율·독립 reference·query budget |
| gradient 접근 | GSA1/2, Tracing the Roots, MoFit, CDI white-box feature | gradient를 낼 module과 backward 비용; LoRA에서의 실제 신호 |
| text-to-image 출력만 접근 | SD-MIA, NDSS fine-tuned black-box 연구의 해당 시나리오 | prompt 생성·captioner·known-member·shadow 비용 |
| image variation API 접근 | ReDiffuse | variation API 사용 가능 여부 및 완성 영상 생성 비용 |

원문별 역할과 구현 조건: [공격 13편 비교 명세](comparison_plan.json), [CLiD](reviews/A12.md), [Quantile](reviews/A09.md), [Tracing](reviews/X02.md), [GSA](reviews/X03.md), [PFAMI](reviews/X11.md), [MoFit](reviews/X12.md), [CDI](reviews/X01.md), [SD-MIA](reviews/X04.md), [NDSS](reviews/X05.md).

**LoRA에선 특히 PFAMI를 빼면 안 된다.** MoFit Table 5(c)는 LoRA fine-tuning에서 MoFit AUC 54.35%, PFAMI 77.50%를 보고한다. 동시에 1% FPR의 TPR은 각각 0%, 1%로, AUC가 남는 것과 낮은 오탐률에서 유용한 것은 별개다. 이 한 실험이 모든 LoRA의 안전을 증명하지는 않는다. MoFit의 일반 SD1.5 표는 known memorized 431개를 member로 사용하므로 전체 학습 자료의 대표 공격 성능으로 일반화할 수 없다. [원문 pp.7–10](https://arxiv.org/pdf/2602.22689)

Re-ID와 copy detection은 별도 측정 축으로 둔다. Real–real identity 판별이 높다는 사실은 도구의 감도를 보여주지만 생성모델 유출을 입증하지 않는다. MaMI/Packhäuser, copy detector, membership attack을 하나의 ‘privacy AUC’로 합치지 않는다. RAD-DINO처럼 평가 데이터 계열이 사전학습에 포함될 수 있는 encoder는 provenance와 독립 encoder 대조가 필요하다. [A03](reviews/A03.md), [A15](reviews/A15.md), [A16](reviews/A16.md)

## 학습 비교는 주장에 맞게 구성한다

졸논과 공유할 기본 구조는 같은 public backbone에서 nonprivate, record-DP, patient-DP를 비교하는 것이다. 환자 평균 clipping은 표준 ULS 기준선으로 이름을 붙인다. Record-DP의 환자 보장은 contribution cap과 tight ELS/group accountant를 포함해 평가한다. 서로 다른 privacy unit에 같은 ε 숫자를 적었다고 동일한 보호가 되지 않는다. [B04](reviews/B04.md), [B05](reviews/B05.md), [B06](reviews/B06.md)

| 주장 범위 | 기본 또는 추가 학습 비교군 |
|---|---|
| 동일 SD backbone의 학습 효용 | DP-LoRA와 DP-LDM attention-only; rank·module·public AE·multiplicity 통제 |
| retrieval 또는 학습 연산 효율 | RAPID; 공개 KB/feature extractor 비용과 private endpoint 학습 범위 포함 |
| 공개자료 domain mismatch 해결 | PrivImage와 RPGen; selection DP 비용과 AMP ablation 포함 |
| 공개자료 없이 prototype/curriculum 활용 | DPDM, DP-FETA, FETA-Pro; 통계 공개+fine-tuning 총 비용 포함 |
| API만 이용한 고해상도 DP 생성 | SPTI; 환자별 vote bound 및 caption/생성 접근 조건 포함 |
| 3D 의료 생성 | DP 3D medical LDM; volume와 전체 환자 단위의 차이 명시 |

dp-promise와 ECCV DP-SAD는 관련성이 높지만, 본문식과 실제 코드의 sensitivity/noise 연결에 미해결 쟁점이 있어 동일 ε의 인증된 patient baseline으로 즉시 취급하지 않는다. 이 검토는 논문 전체가 무효라는 판정이 아니다. [B12 쟁점](reviews/B12.md), [X10 쟁점](reviews/X10.md)

최신 main-track이라고 해서 현재 과제에 모두 직접적인 것은 아니다. PFDM은 공유 모델과 private local refinement의 경계를 갖고, FedDP-PALD는 64차원 latent와 일부 prototype release를 보호한다. SPIRE는 personalization 방법이다. 이들을 모두 환자 DP CXR generator로 분류하면 비교 조건이 틀어진다. [C01](reviews/C01.md), [C03](reviews/C03.md), [C04](reviews/C04.md)

## 공개 결과로 실제 확인한 것

| 재계산 대상 | member-positive AUC | TPR, FPR≤1% | 범위 |
|---|---:|---:|---|
| CLiD-th, 공식 loader의 첫 행 제외를 따름 | 96.1277% | 49.1797% | COCO 공개 score packet, 정확한 ROC |
| CLiD-th, 숫자 첫 행을 포함한 전체 2,500+2,500개 | 96.1289% | 51.2400% | 같은 packet, 첫 행 처리 및 FPR 격자 차이 |
| Tracing, loss-only full timestep | 73.3127% | 6.5000% | 공식 CIFAR feature에 linear classifier 재학습 |
| Tracing, 세 feature / timestep 3 간격 | 81.5826% | 13.4000% | feature 선택에 따른 비용·성능 대조 |
| Tracing, 세 feature / full timestep | 83.3055% | 16.8000% | Table 8의 83.3/16.8에 대응 |

CLiD 공개 helper는 **nonmember를 positive**로 두며, 전체 행을 사용하면 그 방향의 값은 67.52%로 논문 표와 일치한다. AUC가 같아도 양성 방향을 바꾼 저오탐률 TPR는 달라진다. 또한 원 loader는 숫자인 첫 행을 제외하므로 그 동작을 따른 버전과 전체 행 버전을 따로 남겼다. 2,499명과 2,500명의 1% FPR에서 허용되는 FP 수가 달라지는 격자 효과도 있다. 이 차이를 데이터 삭제 하나의 효과로 단정하지 않는다. [공식 supplement](https://proceedings.neurips.cc/paper_files/paper/2024/file/874411a224a1934b80d499068384808b-Supplemental-Conference.zip), [재계산 JSON](replays/A12.json), [전체 행 JSON](replays/A12_all_rows.json)

Tracing의 선형 분류기 재학습은 CPU에서 수행했다. 공급된 gradient feature를 다시 산출한 것은 아니다. 위 표는 의료 환자 실험 결과가 아니며 DP 보장 검증도 아니다. [공식 supplement](https://proceedings.neurips.cc/paper_files/paper/2025/file/901b713c3d1ecfce0d4556752eed4e02-Supplemental-Conference.zip), [설정·SHA·결과](replays/X02.json), [재계산 코드](replay_public_packets.py)

## CVPR로 남길 수 있는 질문과 중단 기준

현재 가장 일관된 질문은 **‘같은 환자의 상관된 여러 관측에서, 제한된 접근·계산량으로 membership를 얼마나 신뢰성 있게 판정할 수 있는가’**다. 의료 문제의 중요성은 충분하지만 문제의 중요성이 새 방법의 증거는 아니다.

1. **환자 bag 공격의 새 기전:** 기존 attack feature와 표준 bag 집계만으로 설명되지 않는 신호를 찾아야 한다. 가변 k를 입력으로 넣거나 quantile로 보정하는 정도는 위 선행과 표준 통계에 겹친다. 같은 환자 내 정보 중복을 다루는 통계량·feature 또는 검증 가능한 질의 배분에서 독립 기여가 나오는지가 후보 질문이다. 아직 새 알고리즘의 성능·보장을 확정하지 않았다.
2. **한정 예산의 환자 감사:** MoFit처럼 강하지만 비싼 방법과 PFAMI·CLiD·DIME 같은 방법을 동일 비용에서 비교한다. 반복 사진·중복 관측 때문에 발생하는 비효율을 줄일 수 있다면 가능성이 있으나, 기존 early stopping·timestep subsampling·표준 sequential testing보다 좋아야 한다.
3. **학습 개선으로 피벗:** ULS와 tight ELS를 같은 환자 수·연산량에 놓고도 유용한 차이가 남아야 한다. 단순 환자 평균, private prototype, timestep curriculum은 이미 선행이 강해 현재 상태에서 더 쉬운 피벗이라고 보지 않는다.

판정을 올릴 조건은 frozen patient test에서 동일 접근·label·compute budget의 최강 기준선보다 일관된 향상이 있고, patient 수/k/기관 또는 encoder 변화에서도 calibration이 유지되며, 새 구성요소 ablation으로 개선 원인이 분리되는 것이다. 낮은 FPR의 신뢰구간이 너무 넓으면 표본 수를 확보하거나 주장하는 operating point를 바꿔야 한다.

반대로 standard aggregation+동일 calibration에 개선이 사라지거나, 데이터 출처·질환·이미지 수 차이를 맞추자 공격이 무너지거나, Re-ID 신호만 남고 membership를 구분하지 못하면 **CVPR 방법 주장은 축소한다.** 그러한 결과도 졸논의 타당성·보호 단위·효용 trade-off 검토로 남길 수 있다. 워크샵 논문이라고 선행 충돌을 무시하거나, 메인 학회라고 접근이 다른 방법을 억지로 같은 표에 넣지 않는다.

## 실험 준비와 졸논 공유

공통으로 재사용할 것은 public model 초기화, 환자 분할 manifest, 이미지 전처리, 실제 adjacency/accountant 기록, 동일 checkpoint의 생성물·질환 utility다. 졸논의 DP/워터마크 실험과 CVPR의 새 attack statistic은 서로 다른 주장이다. 현재 동결 contract를 자동 변경하지 않는다. K10 본학습 완료는 원문 검토나 공개 score 재계산의 선행조건이 아니다.

실행 가능한 공통 점수 평가기는 [evaluate_patient_scores.py](evaluate_patient_scores.py)다. 입력은 한 방법·한 checkpoint의 `patient_id,image_id,split,member,score` CSV이며, 환자가 split/label 경계를 넘거나 영상이 중복되면 중단한다. score 방향을 명시적으로 선택하고 mean/max/median/top-k의 환자 점수에 calibration nonmember rank를 적용한다. Test ROC의 설명용 TPR와 calibration에서 고정된 판정의 실제 TPR/FPR를 별도로 출력한다.

Rank 보정은 fixed score와 **환자 bag의 exchangeability** 아래의 marginal error control이며, 임의의 기관 이동·조건별 FPR·모든 환자 subgroup의 동시 보장이나 DP 증명이 아니다. 최소 p-value는 1/(ncal+1)이다. 0건의 FP도 모집단 FPR 0을 뜻하지 않아 별도의 Clopper–Pearson 구간을 출력한다. 질환별/기관별 strata, uncertainty bootstrap, 공식 공격 feature 추출 adapter는 별도 구현 대상이다. 평가 도구의 6개 검증은 split 누수·중복·동점·방향·독립 calibration·유한 표본 해석을 확인했다.

공식 코드 snapshot의 commit·파일 SHA·다운로드 상태는 [code_acquisition.json](code_acquisition.json), 실행 진입점과 남은 이식 작업은 [BASELINE_PREPARATION.md](BASELINE_PREPARATION.md)에 있다. 새 대형 private 학습이나 의료 이미지의 외부 전송은 이번 문헌 분석에서 수행하지 않았다.

## 기록의 권위와 미해결 사항

기존 18–22번은 당시 판단의 이력으로 보존한다. 최신 문헌 대조와 비교군 해석은 이 문서와 개별 memo를 따른다. 실제 학습·평가 완료 여부는 기존 frozen artifact와 실행 로그가 기준이다. 이번 문헌 검토로 private 학습 완료 상태를 올리지 않는다.

남은 불확실성은 C02 전체 원문, A20/A21/B17의 접근 제한 부분, 전체 증명 검산, 공식 code를 동일 medical backbone으로 이식한 성능이다. 이들 중 한 편의 미검토 부분이 기여와 추가로 겹칠 수 있으므로 ‘선행이 없는 영역’이라고 확정하지 않는다. 관련 없는 과거 후보 D/E 계열도 개별 원문 선택 절을 확인해 제외 이유를 기록했으며, 현 연구에 79편 전부를 구현하자는 뜻은 아니다.
