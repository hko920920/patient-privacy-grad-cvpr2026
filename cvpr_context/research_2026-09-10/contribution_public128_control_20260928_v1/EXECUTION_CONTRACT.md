# 3번 실행 계약 — 공개영상 128장과 기존 합성영상 비교

2026-09-28 / Stage2 / 사용자가 “진행”으로 승인한 범위

## 질문과 실행표

같은 보호 목표·라벨 계산·영상 수를 허용할 때, 기존 합성영상이 공개영상 128장보다 추가 개발 효용을 주는가?

기존 DP1/DP2 각각에서 **합성128/공개128 × 공개 전달 RN-only/Joint(DINO+RN)**를 계산한다. 총 8개의 128변수 bounded label solve다. 새 영상 학습, 새 source/receiver, 새 잡음/요약, hyperparameter 탐색, Expert/Reserved는 없다.

1·2번은 역사적 hard-anchor 결과로 보존한다. 이번 비교는 공정한 영상 비교를 위해 모든 셀에 **0 중심 라벨 regularizer**를 적용한다. 이 차이를 숨기거나 이전 점수를 이번 합성군의 결과로 대체하지 않는다. RN-only도 같은 기회를 준다.

## 공통 연산

- 기존 feature-DP 요약 두 개와 1번에서 검산·봉인한 DINO/RN 목표 계수를 재사용한다.
- P의 class별 환자 균등·환자/class 안 방문 균등 가중치는 그대로다.
- 각 모델의 ridge operator: (ZᵀZ/128 + 0.1 I)⁻¹ Zᵀ/128.
- 공개 P 예측을 목표로 하며, 모델별 weighted target RMS로 정규화한다. Joint는 각 1/2, RN-only는 1의 비중이다.
- 라벨 범위 [-1,1], objective ||Aℓ−t||² + η||ℓ||², η=10⁻³||A||²_F/128.
- η는 **같은 산출 규칙**이며 영상/모델별 operator가 달라 수치 자체는 다를 수 있다.
- scipy lsq_linear, float64, tol=10⁻¹², max_iter=500. 목표 RMS≤10⁻¹²이면 실패로 남긴다.
- ridge primal/dual 및 독립 projected-gradient KKT 검사. 허용오차 각각10⁻¹⁰/10⁻⁷. 수렴 실패를 결과에 맞춘 계수 변경으로 해결하지 않는다.

## 공개128 선정과 이미지 경로

기존 전체 계획의 선정 규칙을 유지한다.

1. P의 기존 813장/672명 공개 명단과 DINO/RN 특징만 사용한다.
2. 환자별 영상 하나를 salt “public-carrier-20260928-v1” + patient/image ID의 SHA256 순서로 고른다. 양성 방문이 있는 6명은 자신의 양성 영상 중에서 고른다.
3. 672명 후보의 DINO64/RN128 특징을 각 block의 후보 평균 squared norm의 제곱근으로 맞추고, 같은 비중으로 연결한다.
4. 양성6명에서 시작해 최소 squared distance가 가장 큰 환자를 122명 추가한다. 동점은 SHA256 순서다.
5. 선택 순서와 원본 영상 해시를 DP 목표 접근 전에 봉인한다. DenseNet 특징과 V는 선정에 쓰지 않는다.

공개영상은 원본 1024×1024를 참조하고 새 resize/export를 만들지 않는다. DINO는 조건 변환 후 기존 tensor 전처리, RN/DenseNet은 기존 tensor short256/center224 규칙을 사용했던 캐시다. 합성영상은 기존 PNG의 해당 규칙 캐시를 사용한다. 모델·projection·조건은 그대로이며 원래 이미지 해상도 차이는 영상 구성의 일부다.

동일한 공개128을 두 release와 두 recipe에 사용한다. 64:64 class 구성이나 임의의 hard label을 강제하지 않는다. 최종 soft label은 DP 목표로부터 계산하므로 **공개128+DP 라벨은 완전한 public-only 기준이 아니다.**

## 평가·판정

모든 8개 라벨과 이미지 명세를 봉인한 뒤 V 평가를 수행한다.

- DenseNet 영상 AUROC 주지표, AP 필수 보조지표.
- ResNet18은 construction 모델의 참고 평가이며 독립 전이로 합산하지 않는다.
- 기존 V와 2,000회 동일 paired patient-cluster bootstrap 사용.
- recipe별·release별 **합성−공개128** 네 직접 차이를 보고한다. 회차별 두 차이의 평균으로 조건부 평균 CI를 계산한다.
- 두 release의 지표 평균은 앙상블이나 잡음 모집단 기대성능이 아니다.
- 개발 투자 기준은 두 AUROC 점차이>0, 평균ΔAUROC≥0.01, 평균ΔAP≥0, 평균AUROC 조건부CI하한>0. AP의 명확한 악화도 함께 표시한다.
- RN-only와 Joint를 따로 판정한다. 유리한 recipe만 골라 전체 통과로 합치지 않는다.
- 차이가 불명확하면 추가 영상 제작 가치 미확인이다. 동등성이나 공개자료의 보편적 충분성을 주장하지 않는다.
- 어떤 결과도 자동으로 새 기여 또는 논문 성공으로 바꾸지 않는다.

## 비용·보존

작업 시작 2026-09-28 16:14:01 KST. 예상 준비·평가 포함 약1시간 전후, 수치 phase당600초/전체90분 한도. 예상 저장≤250MiB, 최대500MiB. 캐시가 유효한 현재 경로는 모델 forward0, label solve8, 새로운 이미지 bank0이다.

새 Q 접근·DP release0. 기존 여섯 release 공동 공개 기본 상한(48,6×10⁻⁵)은 그대로이며 이번 후처리가 과거 전체 연구를 DP로 보호한다는 뜻은 아니다. 원본·코드·계약·기존 결과를 보존하고, 환자 예측을 외부에 자동 업로드하지 않는다.

완료 산출물: 공개128 manifest/selection seal, 8개 라벨과 seal, 기존 형식의 예측·bootstrap·결과, `PUBLIC_TRANSFER_PUBLIC128_CONTROL_RESULTS_20260928.md`.

완료 후 자동 추가 seed/encoder/라벨 설정/독립 평가를 실행하지 않는다.

## 성능 평가 전 수치 구현 보정

원래 기본 TRF는 DP2/SYN/RN-only에서 자체 종료 기준은 통과했으나 독립 projected-gradient 1e−7 기준에 대해 1.916e−7이었다. 같은 문제를 active-set BVLS로 풀면 2.22e−16이며 목적함수도 5.61e−13만큼 낮아졌다. 모든 8개 셀에 BVLS를 동일하게 적용한다. 목적함수·라벨 범위·계수·허용오차·반복 상한은 바꾸지 않는다. 머신 정밀도로 ±1 밖에 있는 라벨만 box로 되돌리고, 보정폭≤1e−12를 검사한다. 원본 코드·계약·부분 라벨·실패와 검산을 attempts/00_trf_projected_gradient_failure에 보존했다. 이 변경 전에 V 평가는 없었다.
