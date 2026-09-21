# B의 증강 target 변경: 근거 검토와 한 bank 비교 명세 — 2026-09-21

판정: **부분 채택.** 현재 PRRD-C의 추가 실행과 관계 중심 기여 주장은 보류한다. 기존 B의 점별 증류·실행 기반을 재사용하고, 증강 target 변경을 한 가지 개발 후보로 채택한다. 이 변경이 현재 실패 원인이거나 가장 효과적인 해결책이라는 근거는 아직 없다. B 전이가 개선돼도 C/D는 자동 재개하지 않는다.

이번 완료 범위는 원문·공식 코드·로컬 코드 대조와 명세 작성이다. 구현 변경, 새 통계 추출, 모델 실행, 합성학습, 계수 탐색은 0이다. 아래 실행값은 결과 전에 정한 **개발안**이며 구현 검증·새 목표 hash·실행 권한이 결속된 FROZEN 실행 계약은 아니다. 큰 연구단계2는 계속 진행 중이다.

## 1. 채택하는 판단과 고치는 해석

- **채택:** 같은 B/C 목표를 더 오래 복제하는 실행, guard 해제만으로 C를 구제하는 실행, 관계 중심의 자동 확대를 하지 않는다. B/C는 실제 배포 PNG에서도 source 기능을 거의 재현했다. B source AUROC는 target 0.777791 → PNG 0.777578, C는 0.777931 → 0.778057이었다.
- **채택:** source의 좋은 기능과 다른 encoder의 유용한 영상자료는 구분한다. DenseNet B AUROC/AP는 0.567975/0.060098, 공개 실자료 기준은 0.618009/0.064941이다. 후자는 수학적 상한이 아니다.
- **채택:** 기존 데이터 역할, patient weighting, gradient·PNG·재개 검증은 유효한 범위에서 재사용한다.
- **보류:** 단일 source, 공개 음성 template, 정규화 순서, clipping 비율 중 하나가 주원인이라는 단정. 현재 비교로 원인이 분리되지 않았다.
- **수정:** 증강 실자료의 평균 통계에 맞추는 것과 **변환별 반응을 대응시키는 것**은 다르다. 아래 후보는 전자다. LGM/Dosser의 전체 구현 또는 기존 방법 대비 신규 기여로 표시하지 않는다.

근거: 기존 [예비 비교](PRRD_ABCD101_DEVELOPMENT_RESULTS_20260921.md), [source–PNG 진단](PRRD_SOURCE_TARGET_PNG_DIAGNOSTIC_20260921.md), [기전 분석](PRRD_FAILURE_MECHANISM_ANALYSIS_20260921.md). 기존 결과는 수정하지 않는다. 사용자 보고의 원격 commit을 로컬 checkout commit으로 옮겨 적지 않았다. 현재 작업 디렉터리는 Git checkout이 아니며 실제 파일 SHA로 결속했다.

## 2. 선행과 실제 코드에서 확인한 차이

| 대상 | 확인된 연산 | 이번 판단에 쓸 범위 |
|---|---|---|
| 현재 B | guarded_objectives.feature_objective에서 clean/augmented 합성 모멘트 모두 objective.moments에 정합. 기능 손실은 clean에 한 번 적용 | 증강된 합성자료에 clean target을 요구하는 것은 사실. 정의된 불변성 regularizer이며 구현 오류로 확정하지 않음 |
| LGM | 매 update 새 선형 head; real/synthetic에 같은 종류의 augmentor를 각각 호출; synthetic 여러 복사본; cross-entropy gradient cosine 정합 | 양쪽 변환 후 신호를 비교한다는 근거. 같은 난수를 두 경로에 재생하는 코드 또는 고정 평균 모멘트 정합은 아님 |
| Dosser | 논문은 변환 seed·추출기·보호된 신호를 저장하고 재사용. 공식 코드도 real 변환 seed를 저장해 synthetic에서 재사용 | 변환 조건을 보존하는 선행이 이미 있음. 아래 pooled-mean 후보와 구별 |

LGM은 단일 증류 encoder에서 다른 encoder로의 사용을 이미 연구했고, source 과적합과 이미지 표현·증강을 별도로 다뤘다. 우리 코드에도 pyramid와 약한 증강은 이미 있으므로 “없으니 추가한다”는 설명은 채택하지 않는다. 공식 pyramid에는 무작위 초기화와 zero 옵션이 있어 공개 양성영상 초기화가 필수라는 결론도 나오지 않는다. [LGM 원문](https://arxiv.org/html/2511.16674v1), [공식 linear_gm.py 고정본](https://github.com/GeorgeCazenavette/linear-gradient-matching/blob/33d004b8c16d9ab8e39aec780f9edad767f6f033/src/distillation/linear_gm.py), [pyramid.py](https://github.com/GeorgeCazenavette/linear-gradient-matching/blob/33d004b8c16d9ab8e39aec780f9edad767f6f033/src/synsets/pyramid.py).

Dosser의 논문 설명과 배포 코드는 개인정보 처리 위치까지 같은 것으로 취급하지 않는다. 확인한 코드에서는 clean real_output_sums를 저장한 뒤 train_step에서 새 잡음을 더한다. 따라서 이 코드 대조로 “한 번 보호한 신호만 후처리하므로 전체 실행의 DP가 검증됐다”고 쓰거나 해당 경로를 그대로 복사하지 않는다. 이번에는 seed·augmentation 연산만 참고한다. [Dosser §3.2](https://arxiv.org/html/2508.01749v1), [공식 dosser.py 고정본](https://github.com/humansensinglab/Dosser/blob/dff0c57f6f22f66d500a66bad4a9db757e5f714a/dosser.py), [DiffAugment 구현](https://github.com/humansensinglab/Dosser/blob/dff0c57f6f22f66d500a66bad4a9db757e5f714a/utils.py).

공식 프로젝트 링크에서 저장소를 추적했다. LGM commit은 33d004b8c16d9ab8e39aec780f9edad767f6f033, Dosser는 dff0c57f6f22f66d500a66bad4a9db757e5f714a다. 읽은 파일 5개의 URL·SHA와 로컬 근거 SHA는 spec_sources/prrd_b_aug_target_review_20260921/에 보존했다. 파일을 실행하지 않았다. CovMatch·DP-KIP·DP-NTK의 상세 비교는 기존 검토의 배경이며, 이번에 전수 재검토·재현했다고 표시하지 않는다.

## 3. 수정할 연산은 한 가지다

기존 clean target을 T0, 기존 합성 증강을 a, 증강된 실자료의 고정 평균 target을 Taug라 둔다.

- 기존: L = D(M(S),T0) + 0.1 D(M(a(S)),T0) + Jfunc + Lprior.
- 후보: L = D(M(S),T0) + 0.1 D(M(a(S)),Taug) + Jfunc + Lprior.

T0와 그로부터 계산한 목표 readout·M_T, 기능 정합 계수1, 통계 정합 계수, pixel anchor·TV, optimizer, 기존 synthetic augmentation stream은 그대로다. 합성자료의 clean 경로를 증강 통계로 바꾸거나 기능 손실을 증강 경로에도 넣지 않는다.

**중요한 한계:** D가 현재의 가중 squared norm이고 Y=M(a(S))라면,

    E ||Y−Taug||²_W
      = ||E Y−Taug||²_W + E ||Y−E Y||²_W.

따라서 평균 target을 바꿔도 합성 통계의 증강 변동을 억제하는 항은 남는다. 이 수정은 “clean 중심 대신 실자료의 증강 평균 중심을 요구한다”는 비교이며, 변환별 real 반응/공분산까지 재현하거나 불변성 압력을 제거하는 방법은 아니다. 변환별 target과 같은 변환을 매번 대응시키는 방식은 별도 설계 변경이며 이번 범위에 추가하지 않는다. 작은 기존 증강에서 Taug와 T0가 얼마나 달라지는지도 아직 추출하지 않았으므로 수정 효과의 크기를 예측하지 않는다.

## 4. 유한 증강 통계의 계산 계약

개발 후보명은 B_augmean_k4, run ID는 dev_B_augmean_k4_101이다. 이 이름은 새 논문 방법명이 아니다.

1. P 813장/672명, Q 5,097장/2,027명만 통계에 사용한다. Q는 private_train 이력을 유지한다. V는 target·계수·projection에 쓰지 않는다.
2. **K=4회/영상**을 먼저 정한다. seed helper의 입력은 ('prrd-B-realaug-v1',101,role,image_id,k), k=0,1,2,3이다. 각 seed의 CPU torch.Generator에서 5개의 uniform 값을 만들고 기존 augment_parameters와 같은 theta·brightness 공식을 쓴다. 영상별 독립 draw이며 class/환자 반복 수로 다시 seed를 고르지 않는다. private ID·seed 매핑은 내부만 보존한다.
3. 기존 범위: rotation±3°, translation±1% image extent(theta translation±0.02), scale[.98,1.02], brightness[.98,1.02], flip 없음. 기존 bilinear zero-padding, align_corners=False, 결과 [0,1] clamp를 유지한다.
4. Real 경로는 **원래 grayscale tensor → native geometry에서 증강 → 기존 source resize512/center-crop448/3채널 → 고정 BioViL → 기존 point projection/bound**다. 명부상 P+Q는 모두1024×1024다. 증강만을 위해 real을224로 먼저 축소하거나 encoder 입력448에서 대신 증강하지 않는다. Synthetic 경로는 기존224 renderer → 증강 → 같은 source 전처리다. native와 synthetic의 해상도·resampling 차이가 사라졌다는 주장은 하지 않는다. 좌표 분포·연산 순서를 고정하는 것이다.
5. clean PCA16·scale·checkpoint·T0를 재학습/재추출하지 않는다. 새 증강 특징은 기존 clean 특징만으로 계산할 수 없으므로 별도로 추출한다. FP32 encoder, no-grad/eval, microbatch 최대16; 환자 통계 누적은 FP64다.

각 이미지 x의 k번째 변환 후 bounded 특징을 z_xk라 하면:

    m_ic_aug = [1/(|D_ic| K)] Σ_(x∈D_ic) Σ_k z_xk
    A_ic_aug = [1/(|D_ic| K)] Σ_(x∈D_ic) Σ_k z_xk z_xkᵀ

그 뒤 class c가 있는 환자들을 동일 질량으로 평균한다. P/Q는 합계와 class별 환자 수를 합친 후 나눈다. 이미지/변환을 모두 펼쳐서 환자 질량을 늘리거나, 평균 특징의 outer product로 A를 대체하지 않는다. 빈 class는 기존 규칙을 유지한다. K는 경험적으로 충분하다고 확인된 값이 아니라 추출량을 제한한 첫 근사다. 같은4회에서 split-half target 차이를 설명 지표로 기록할 수 있으나 그 값이나 V 결과로 K를 자동 증가시키지 않는다.

변환 픽셀은 저장하지 않는다. 재계산을 위해 내부 z[image,view,16], view seed/theta/brightness, patient별 합계 및 P/Q 합계, target·입력 hash만 보존한다. 새 자료는 비DP private 내부 통계이며 공개 bundle에 넣지 않는다.

## 5. 최소 구현 변경 범위와 필요한 검사

이번에는 아래 변경을 **실행하지 않고 범위만 확정**했다.

| 위치 | 필요한 변경 |
|---|---|
| 새 augmented-target 준비 모듈 | 위 K4 추출·환자 집계·독립 검산·receipt. 기존 prepare.py/clean target을 덮어쓰지 않음 |
| guarded_objectives.py | BoundObjective에 별도 detached aug_moments 추가. feature_objective의 aug_stats target만 교체. 미지정 시 기존 clean target으로 동작 |
| fit_banks.py의 job 결속 | 기존 target hash와 추가 aug target hash·rule hash를 함께 로드. 목표 readout은 clean에서만 생성 |
| bank_runtime.py / 계약 결속 | immutable-target 검사와 checkpoint signature에 aug target을 포함. 잘못된/누락된 target으로 resume 금지. 옛 bank·권한 파일 재사용 금지 |
| 별도 개발 평가 wrapper | 마지막 PNG seal 뒤 기존 source/DenseNet V 특징·bootstrap draw를 재사용해 기존 B와 비교. 기존 예비 보고서를 수정하지 않음 |

필요 검사는 (a) Taug=T0일 때 기존 loss/feature gradient 복원, (b) Taug만 바꿨을 때 clean 기능 target 불변, (c) 비균등 방문 수·K·빈 class의 환자 집계 독립 계산, (d) 소수 P 입력에서 새 real 변환 위치·range·identity 및 변경 loss의 same-partition one/two-pass gradient 대응, (e) 추가 target hash 변경 시 resume 거부다. 기존 gradient 기준(loss_abs2e-6, gradient_atol2e-7 및 peak-relative5e-4)은 원 verifier의 판정식을 그대로 상속한다. 기존 전체 저장/재개/PNG campaign·전체128 profile을 다시 반복하지 않는다.

신규 코드·target이 없으므로 지금 “실행 준비 검증 완료”라고 쓰지 않는다. 구현 검증은 이후 허용된 작업 안에서 변경 경로만 수행한다.

## 6. 추가 계산량과 예상 비용

| 작업 | 계획량 |
|---|---:|
| Real 추가 증강 특징 | P 3,252 + Q 20,388 = **23,640 image-forwards**, backward0 |
| 신규 통계의 환자 단위 | P672 + Q2027; 변환 수만큼 환자 수를 늘리지 않음 |
| 수정 B 합성 | 128장×500 successful updates, source256,000 image-forwards/128,000 image-backwards |
| 최종 PNG 특징 | source128 + DenseNet128 image-forwards; 기존 V 특징 재사용 |
| 신규 비교 bank | **1개**, 기존 B는 재학습하지 않음 |

기존 B 실측은2,960.094초(49.33분)이다. 새 목표 배열 하나로 바꾸는 연산은 encoder 횟수를 늘리지 않으므로 합성 비용의 참고값으로 쓴다. 기존 Q clean 추출148.062초/5,097장을 단순 환산하면 추가23,640 forward의 참고값은약11.45분이지만, native1024 증강·I/O·새 cache 관리 비용을 측정한 값이 아니다. **49.33+11.45분을 전체 작업 보장 시간으로 쓰지 않는다.** 구현·검증 시간도 별도다. 이후 실행 시 이 근거와 미측정 비용을 포함한 ETA/시간 상한을 먼저 제시한다. 이번 작업은 GPU0이다.

저장은4회 z만으로 약1.44MiB이며 통계·매핑·receipt가 추가된다. 증강 영상 cache를 만들지 않고, 새 bank도 기존의 중복 checkpoint 정리 규칙을 적용해 불필요한 저장량을 늘리지 않는다. 기존 결과·원영상·checkpoint를 이번에 삭제하지 않는다.

## 7. 비교와 판정 — 사전 개발 기준

- seed101, 공개 template/초기 parameter,128장·500회, FP32/microbatch16, AdamW lr.01/wd0, 기존 pyramid schedule·계수·증강 stream을 유지한다. 기존 학습된 B에서 fine-tune하지 않고 같은 초기 상태에서 시작한다.
- B는 beta0/ridge.1/eta1이며 관계 개입 제한은 적용할 수정 방향이 없다. 관계 경로·새 encoder·native feature 확장·강한 crop·template 변경을 함께 넣지 않는다.
- 마지막500-step PNG 전체를 고정한 뒤 DenseNet121 P-only PCA128의 기존 V AUROC/AP를 한 번 계산한다. 기존2,000 patient-cluster paired draw로 B_aug−B_old 및 B_aug−P_real 차이를 함께 보고한다. Source target/PNG fidelity도 설명 결과로 함께 남긴다.
- 이번 제안의 **후속 투자 신호**는 ΔAUROC(B_aug−B_old)≥.01이고 AP가 감소하지 않는 경우로 정한다. 임상/유의성/새 기여 기준은 아니며 단일seed 효과와 CI 전체를 보고한다. CI가0을 포함하면 그 불확실성을 유지한다. 이 기준을 넘더라도 추가 실행 권한은 자동 발생하지 않는다.
- 공개 real 기준0.618009/0.064941 대비 격차를 별도로 표시한다. 수정 B가 좋아져도 같은 수정 A와 비교하기 전에는 private 추가 효용을 주장하지 않는다.
- DenseNet과 V는 이미 이번 수정안을 선택하는 개발에 사용됐다. “선택에 쓰지 않은 수신자/독립 확인”으로 부르지 않는다. ViT 확인용 수신자는 계속 열지 않는다.
- 신호가 좋으면 별도 범위에서 같은 방식의 A와 반복성을 판단한다. 혼합/음성 결과면 그 결과를 보존하며 seed·학습량·증강 강도 확대를 자동 실행하지 않는다. 작은 평균-target 수정의 실패가 LGM/Dosser 또는 점별 증류 전체의 불가능성을 뜻하지도 않는다.
- **C/D 재개에는 관계가 실제 과제에 유용하다는 별도 근거가 필요하다. B 전이 개선은 그 근거를 대신하지 않는다.**

이번 명세만으로 새 학습/추출/평가를 시작하지 않는다. DP·Expert·Reserved와 추가 비교군은 계속 닫혀 있다. 이후 DP 연결 시 clean+augmented query·감도·release/composition을 다시 정의해야 한다. K를 늘려도 환자 수나 privacy budget이 늘어난 것으로 취급하지 않는다.

## 8. 완료 상태

근거가 확인된 것은 현재 코드의 target 재사용, 선행의 실제 증강 경로, source fidelity와 recipient utility의 차이다. 채택한 것은 그에 근거한 **한 가지 개발 비교 명세**다. 전이 개선·private 효용·환자 관계 가치·DP 가치·논문 기여는 새로 입증되지 않았다.

로컬 생산 코드와 기존 보고서/clean target은 unchanged이며 해시 검증 결과를 같은 spec_sources 폴더에 남긴다. 이번에는 공개 선행 파일 조회·문서/상태 기록만 수행했다.

