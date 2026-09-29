# 공개 모델 전달과 DP 합성영상 라벨 최적화: 판단 및 다음 비교 — 2026-09-28

사용자가 상세 판단을 검토한 뒤 “오케이 이거 기록”이라고 요청하여 남긴 기록이다. 이번 작업은 문서와 현재 상태 연결만 수행하며 새 실험을 시작하지 않는다.

현재 큰 연구 단계는 2(근접연구 분석과 구체적 기여 설계)다. 2026-09-24의 실제 개발 결과를 인정하되, 독립 재현·선행 대비 독자적 가치·CVPR 기여의 완료와 구분한다.

## 현재 판단과 기여 후보

**기대할 근거가 있는 구현된 방법 후보다. 선행 대비 추가 가치와 독립 재현은 아직 미확정이다.**

> 한 source에서 이미 보호한 환자 통계를 공개자료로 다른 encoder의 학습 목표에 옮기고, 그 목표를 기존 합성영상의 soft label에 담아 다른 receiver의 효용을 개선한다.

사적 Q를 다시 조회하거나 private query 차원을 늘리지 않고 공개 모델의 요구를 반영하는 것이 설계 의도다. 새로운 Gaussian 메커니즘, 평균 추정 원리, 최초의 label solve 또는 최초의 DP 후처리를 주장하지 않는다.

## 실제 연산과 기대하는 이유

1. 공개 P에서 같은 환자/영상의 DINO K4 특징과 ResNet18 특징을 연결하는 affine ridge를 계산한다.
2. 기존 DINO feature-DP의 class별 보호 평균을 ResNet18 목표 평균으로 전달한다.
3. 전달 평균과 공개 P의 second moment로 목표 ridge classifier를 계산한다.
4. 고정된 PNG 128장에 대한 bounded soft label을 계산한다. 그 라벨로 학습한 DINO와 ResNet18 readout이 공개 P에서 목표 classifier의 출력을 재현하도록 한다.
5. 같은 이미지·라벨 묶음을 기존 frozen-feature ridge 규칙으로 DenseNet에 평가한다.

DenseNet은 최종 joint 라벨 목적함수에서 제외됐다. 그러나 이미 재사용한 개발 receiver이며 untouched receiver는 아니다. ResNet18은 이번 구성의 construction model이므로 독립 전이 성공으로 추가 계산하지 않는다.

같은 DP 요약과 PNG에서도 후처리를 바꾸면 효용이 개선됐다. 따라서 이 두 bank의 낮은 효용을 “보호 단계에서 모든 유용한 정보가 사라졌다”로 설명할 수 없다. 공개자료도 정보와 구조를 공급하므로 이를 사라진 사적 정보의 복원이라고 부르지 않는다.

공개 환자 5-fold 특징 예측 R²는 약 0.18이다. 완벽한 encoder 정렬이 아니다. 공개 P에서의 관계가 Q에 유지되는지, 128장으로 목표 분류 기능을 표현할 수 있는지는 한계다. P 양성 환자는 6명뿐이다.

## 이미 완료된 개발 데이터

원자료는 [2026-09-24 실행 보고서](CONTRIBUTION_SEARCH_RESULTS_20260924.md)와 [joint 결과 JSON](contribution_search_20260924/joint_compiler_results.json)이다. 아래는 당시 저장 결과의 재기록이며 이번에 새 평가를 수행한 것이 아니다.

| 라벨 방식 | DenseNet DP1 AUROC / AP | DenseNet DP2 AUROC / AP |
|---|---:|---:|
| 기존 hard label | 0.650018 / 0.080832 | 0.596743 / 0.064188 |
| DINO 직접 pseudo-label | 0.653683 / 0.074752 | 0.614055 / 0.062433 |
| DINO-only functional label solve | 0.671617 / 0.083324 | 0.613363 / 0.060382 |
| DINO + 공개 전달 ResNet18 joint label solve | **0.686370 / 0.081264** | **0.663404 / 0.074463** |

| Joint 대비 대조 | DP1 ΔAUROC [95% CI] | DP2 ΔAUROC [95% CI] |
|---|---:|---:|
| 기존 hard label | +0.036352 [+0.002859, +0.070587] | +0.066660 [+0.027893, +0.109059] |
| DINO 직접 pseudo-label | +0.032687 [+0.011241, +0.055274] | +0.049349 [+0.030680, +0.068976] |
| DINO-only label solve | +0.014753 [−0.002636, +0.032335] | +0.050041 [+0.024707, +0.076099] |

강한 DINO-only label solve 대비 DP1 우위는 불확실하다. DP2의 AP 차이는 +0.014081 [ +0.005663, +0.028388 ]지만 DP1 AP는 조금 낮고 구간이 0을 포함한다. 모든 지표·반복에서 우월하다고 쓰지 않는다.

ResNet18 joint AUROC는 DP1 0.698797 / DP2 0.707042다. 이 모델은 라벨 계산에 참여했다. 보조 ResNet18-only 구성의 DenseNet 0.713564 / 0.648230도 보존하며, 최고점만 골라 최종 방식의 결과로 사용하지 않는다.

평가는 영상별 AUROC/AP와 2,000회 paired patient-cluster bootstrap이다. 환자별 평균 점수의 AUROC가 아니다. 구간은 현재 bank에 조건부이고 DP 잡음 전체의 변동·적응적 방법 탐색을 포함하지 않는다. 새로운 합성 초기화나 독립 cohort 재현은 아니다.

## 근접연구와 신규성 경계

| 연구 | 이미 제공하는 부분 | 현재 후보에서 판정할 부분 |
|---|---|---|
| [DP-NTK](https://arxiv.org/abs/2303.01687), [Dosser](https://arxiv.org/html/2508.01749v1) | 보호 통계·학습신호의 재사용, 신호/잡음 효율 | 고정 release 이후 공개 receiver 목표와 라벨을 바꾸는 추가 효용 |
| [Balog et al., DP kernel mean embeddings](https://proceedings.mlr.press/v80/balog18a.html) | 보호 평균의 가중 공개/합성 자료 표현 | 평균 전달 자체의 최초성을 주장하지 않고 최종 데이터 전달 효용을 검증 |
| [KIP / Label Solve](https://arxiv.org/html/2011.00050v2), [Flexible Dataset Distillation](https://arxiv.org/abs/2006.08572) | 고정 이미지의 라벨 증류 | 강한 단일-model label solve보다 공개 heterogeneous functional targets가 유용한가 |
| [DPPL](https://arxiv.org/html/2406.08039v3) | 공개 encoder의 보호 class prototype과 분류 | 별도 encoder와 이미지/라벨 묶음으로 전달할 가치 |
| [POST](https://arxiv.org/html/2506.16196v1) | 보호된 정보를 공개자료로 다른 모델에 전달 | vision class 통계와 고정 데이터 산출물에서의 구체적 이득 |
| [LGM](https://arxiv.org/html/2511.16674v1) | pretrained linear-gradient matching과 cross-model 전이 | 보호 summary로 만든 receiver functional target을 supervision에 담는 효과 |
| [CVPR 2026: Hard Truths about Soft Labels](https://openaccess.thecvf.com/content/CVPR2026/html/Dey_Rethinking_Dataset_Distillation_Hard_Truths_about_Soft_Labels_CVPR_2026_paper.html) | soft-label 조건에서 이미지/부분집합 품질의 효과가 약해질 수 있다는 분석 | 공개 이미지에도 같은 soft-label 계산을 허용해 합성영상의 필요성을 검증 |

공개 affine 전달과 public signed-weight 표현의 수치적 동치는 2026-09-24에 확인했다. 기본 연산에 새 이름을 붙여 신규성으로 세지 않는다. 통상적인 공개자료 기반 선형 지식증류와도 대수적으로 같은 계산에 귀결되는지 먼저 비교해야 한다. 전체 연산의 최초성을 입증했거나 위 논문 전체 재현을 이겼다는 주장은 아직 없다.

## 다음 작업 우선순위와 판정 질문

반복 수·새 잡음·계수를 늘리는 것보다 기존 설명을 허용한 직접 대조를 먼저 한다. 아래는 **다음 연구 우선순위의 기록**이며 이번 기록 작업에서 실행하지 않는다.

| 순서 | 질문 | 비교와 통제 | 부정적 결과의 의미 |
|---|---|---|---|
| 1 | 현재 공개 평균 전달이 통상적 지식증류보다 필요한가? | 같은 보호 DINO 정보, P, 공개 모델 접근을 허용한다. source의 공개 P 예측을 ResNet18에 전달하는 일반적인 지식증류와 대수·연산부터 대조하고, 차이가 남으면 동일 이미지/라벨 solver로 비교한다. | 같은 연산이면 별도 방법으로 명명하지 않는다. 이득이 없으면 특수한 전달 연산의 우위를 주장하지 않는다. |
| 2 | 기존 합성영상의 제작이 필요한가? | 공개 이미지 128장에도 동일한 보호 목표와 라벨 최적화를 허용한다. 특징 변환·학습 규칙·자료 접근·전체 비용을 맞춘다. | 공개 이미지가 비슷하거나 좋으면 현재 합성영상 제작비용이 정당화되지 않았다고 인정한다. |
| 3 | 두 기존 bank 외에서도 추가 이득이 남는가? | 앞선 대조를 통과한 방법·강한 대조·판정 규칙을 먼저 고정하고 별도 합성 초기화 등으로 대응 확인한다. | 재현되지 않으면 일반적인 개선 방법으로 확대하지 않는다. |
| 4 | 개발 평가 밖에서도 성립하는가? | 비교와 방법을 동결한 후 독립 평가를 사용한다. Expert/Reserved를 먼저 열어 후보를 선택하지 않는다. | 실패하면 개발 결과의 범위로 남기며 자동으로 새 기여 이름을 붙이지 않는다. |

앞의 두 비교는 공개자료와 기존 DP 산출물의 후처리로 구성할 수 있다. 새 Q 요약이나 잡음을 먼저 만들 필요가 없다. 기존 특징 캐시와 작은 선형 계산을 재사용한다. 구현·검산·평가 시간은 별도이며, 원래 bank 합성 약 35분을 제외하고 전체 방법이 수초라고 주장하지 않는다.

구체적인 실행 범위·입출력·동일성·비용·판정은 다음 구현 묶음에서 고정한다. 숫자 성공선을 이번 기록에서 임의로 추가하지 않는다.

## 보호·평가·산출물 경계

- 2026-09-24 후보 탐색은 새 Q 접근 0, 새 보호 요약 0, 새 pixel 최적화 0이었다. 기존 두 DP 요약과 256 PNG를 재사용했다.
- 이번 2026-09-28 기록 작업 역시 새 학습·평가·Q 접근·DP release·Expert/Reserved 접근 0이다.
- 고정된 알고리즘이 공개 P와 해당 DP 산출물만 사용하는 후처리라는 전제에서 보호 보장을 이어받는다. 적응적 연구 전체가 개별 ε8이었다고 말하지 않는다.
- 이전 여섯 보호 요약을 공동 공개할 때의 기본 합성 상한 (48, 6e−5)은 삭제하거나 초기화하지 않는다.
- 현재 확인한 학습기는 frozen-feature ridge다. 신경망 전체 학습/CE 훈련 전이를 검증하지 않았다.
- soft label은 증류용 회귀 target이며 보정된 임상 질병 확률이 아니다.
- 현재 배포 형태는 기존 PNG + 새 라벨이다. 최종 receiver에 목표 classifier나 P를 별도 추가 학습자료로 넘긴 결과가 아니다. 기존 공통 P 기반 projection은 유지한다.
- 기존 결과·코드·PNG와 부정적인 대조는 보존한다. 저장된 CSV·hash·검산 내역은 원 실행 보고서를 따른다.

## 기록한 결정

현재 후보를 계속 검증할 이유는 실제 개선, 구체적 연산, 강한 내부 대조 및 판정 가능한 후속 비교가 있기 때문이다. 가장 약한 연결은 **선행 대비 추가 가치와 독립 재현**이다.

현재 단계의 완료 표시는 “구현된 긍정적 개발 후보와 다음 대조 우선순위를 기록함”이다. CVPR 기여·독립 검증·방법론 우위 완료로 표시하지 않는다.

## 2026-09-28 보완: arXiv 링크와 실제 출판 상태

사용자가 arXiv 링크를 주로 사용한 이유와 정식 출판 여부를 질문하여 확인했다. arXiv는 논문 공유 저장소이며 자체적으로 동료심사를 수행하지 않는다. 정식 게재 논문도 저자 공개본을 arXiv에 유지할 수 있으므로, 열람 URL과 출판 상태를 구분한다. [arXiv 공식 설명](https://info.arxiv.org/about/index.html)

이전 표의 arXiv 링크는 본문 열람용이었다. 아래는 학회·출판 기록 또는 저자 소속기관의 공식 기록으로 확인한 출판 상태다. 아카이브 버전과 최종 게재본의 내용이 모든 부분에서 동일하다는 뜻은 아니다.

| 논문 | 확인된 출판 상태 | 공식 확인 링크 |
|---|---|---|
| DP-NTK | Journal of Artificial Intelligence Research, 81:683–700, 2024 | [저자 소속 DTU의 게재 기록과 최종본](https://orbit.dtu.dk/en/publications/differentially-private-neural-tangent-kernels-dp-ntk-for-privacy-/), DOI 10.1613/jair.1.15985 |
| Dosser | ICCV 2025 본학회, 4838–4847 | [CVF proceedings](https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_Improving_Noise_Efficiency_in_Privacy-preserving_Dataset_Distillation_ICCV_2025_paper.html) |
| LGM | NeurIPS 2025 Main Conference Track | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5709163243753c9c9ab7f4b4e5a8766d-Abstract-Conference.html) |
| KIP / Label Solve | ICLR 2021 | [저자 기관 Google Research의 게재 기록](https://research.google/pubs/dataset-meta-learning-from-kernel-ridge-regression/) |
| DPPL | AAAI 2025 Technical Track, 39(20):20991–20999 | [AAAI proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/35395) |
| POST | ICML 2025, PMLR 267:64998–65019 | [PMLR proceedings](https://proceedings.mlr.press/v267/wang25ds.html) |
| Balog et al., DP Kernel Mean Embeddings | ICML 2018, PMLR 80:414–422 | [PMLR proceedings](https://proceedings.mlr.press/v80/balog18a.html) |
| Hard Truths about Soft Labels | CVPR 2026, 178–187 | [CVF proceedings](https://openaccess.thecvf.com/content/CVPR2026/html/Dey_Rethinking_Dataset_Distillation_Hard_Truths_about_Soft_Labels_CVPR_2026_paper.html) |
| Flexible Dataset Distillation | NeurIPS 2020 Meta-Learning Workshop 채택·발표. NeurIPS 본학회 논문으로 표기하지 않음 | [워크숍 공식 채택 목록](https://meta-learn.github.io/2020/#accepted-papers), [저자 기관 게재 기록](https://www.research.ed.ac.uk/en/publications/flexible-dataset-distillation-learn-labels-instead-of-images/) |

앞으로 근접연구 표에는 출판처·연도·본학회/저널/워크숍/미확인 프리프린트 상태를 함께 표시한다. 공식 proceedings/저널을 우선 링크하고 arXiv는 공개 본문·버전 확인용으로 병기한다. 핵심 연산과 주장은 최종 게재본을 기준으로 대조하고 다른 버전만 확인했으면 그 범위를 명시한다. 프리프린트도 관련 선행으로 검토하되 심사·게재가 확인된 연구처럼 표현하지 않는다.

이 표는 현재 전략의 근접연구 목록에 대한 확인이다. 과거 확장 검색에 등장한 모든 논문의 출판 상태까지 한꺼번에 보증하는 것은 아니다. 이번 보완은 문헌 메타데이터 정리이며 새 실험은 수행하지 않았다.

