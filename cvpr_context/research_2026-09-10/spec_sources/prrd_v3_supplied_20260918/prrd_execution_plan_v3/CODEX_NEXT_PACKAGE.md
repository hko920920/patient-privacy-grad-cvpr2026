# Codex 다음 작업: PRRD 통합 비교 구현·실행 계약

## 요청 범위

`PRRD_MASTER_EXECUTION_PLAN_V3_20260918.md`와 사용자가 승인한 최신 로컬 비교 문서를 함께 읽고, 차이가 있으면 기존 결과를 바꾸지 않는 별도 계약 변경으로 기록한다. 이번 계획의 주력은 PRRD이며, cached-condition diffusion을 병행하지 않는다.

**첫 작업은 아래 W0/W1을 하나의 패키지로 구현하는 것이다.** 공개 입력의 기술 검사에 통과하면 실제 비용과 계산량을 보고하고, 승인된 시간 범위에서만 고정 W2를 실행한다. 합성자료나 새로운 환자 성능을 이미 검증했다고 보고하지 않는다.

## W0: 로컬에 결속할 항목

1. 저장소의 실제 commit, 최신 Codex 비교 문서, P/Q/V manifest를 읽고 hash를 기록한다.
2. P=672명/813장, Q=former-selection2027명/5097장, V=2026명/5047장과 사용 이력을 확인한다. Q는 public이 아니며 Q80과도 다르다.
3. Expert532/reserved4213 접근 금지는 loader의 실제 allow-list/deny-list로 구현한다. 경고문만 적는 것으로 대신하지 않는다.
4. BioViL-T **image** checkpoint·embedding layer·공식 전처리를 결속한다. text weight를 잘못 로드하지 않는다.
5. DenseNet121 ImageNet1K V1 수신 encoder의 checkpoint와 공식 전처리를 별도로 결속한다.
6. 공개 P만으로 source PCA16와 recipient PCA128 및 scale을 생성하고 저장한다. Q/V에서 projection을 선택하지 않는다.
7. master의 이번 권고 실행값을 실제 YAML에 복사한 뒤 null인 hash·경로·microbatch·hardware 항목을 채운다. template의 status를 근거 없이 FROZEN으로 바꾸지 않는다.

## W1: 한정된 기술 검사

- 동봉 `reference_math.py`를 먼저 실행하고 출력이 `PASS_TOY_ALGEBRA_ONLY`임을 기록한다. 이것은 환자 코드 검증이 아니다.
- 환자별 class 평균/2차 모멘트, mixed delta/C, 서로 다른 모집단의 정규화를 구현한다.
- C/D는 공개 pair를 고정하고 Q 내부 대응만 바꾼다. 영상 목록·pointwise 통계·dbar가 같고 관계 C가 달라지는지 확인한다.
- R_joint는 표준 endpoint joint-moment 비교로 구현한다. 원형 CovMatch 재현이라고 표시하지 않는다. C와 동일한 차분 연산을 다른 이름의 baseline으로 만들지 않는다.
- source encoder는 frozen/eval이어도 **synthetic input gradient가 존재**해야 한다. 두-pass global-moment gradient와 one-pass를 공개 toy/input에서 대조한다.
- microbatch를 바꿔도 global class/bank 분모가 바뀌지 않아야 한다. 각 microbatch 안의 임시 class 수로 평균해서는 안 된다.
- 공개 pixel→preprocess→feature, float synthetic→8-bit PNG→재추출의 경로를 확인한다. 최종 평가에는 배포 PNG만 사용한다.
- 기존 ResNet18 loader 연동은 공개-only 입력에서 이전 kernel과 대응 검사한다. 과거 잘못된 98-update 경로는 계속 차단한다.
- 공개 입력으로 전체128장 two-pass update의 warmup/실측을 수행하고 시간·VRAM·IO를 보고한다. 예측값을 잘 보이게 만드는 최적화에는 수신 encoder를 사용하지 않는다.

## W2: 사전 승인된 비DP 방법 비교

기본 계획은 A/B/C/D/R_joint ×3개 초기화(101,202,303), 총15bank×128장=1,920장이다. 각각500step, 총7,500 synthetic updates다. 최종500step의 모든 bank를 저장한 후 한 번에 평가한다.

평가:

- source encoder: 직접 통계 분류기와 배포 PNG의 source-risk 일치.
- primary recipient: DenseNet121에서 관계-aware readout, C−B/C−D/C−R_joint.
- 같은 C/D/R 산출물의 β0/β1 readout으로 학습 규칙 효과를 구분.
- 기존 ResNet18:45run×400step=18,000 updates, 이미지-only 호환성.
- 기존 Rwide 숫자를 다른 encoder의 성능 상한으로 가져오지 않는다.
- patient-cluster 2,000 paired draw, bank별/learner별/metric별 결과 전체를 저장한다.

연결 검사가 통과해도 DP/expert/reserved를 자동으로 열지 않는다. 시간 상한을 초과하면 정확한 예상 비용과 아직 실행하지 않은 단위를 보고한다. 일부 결과만 보고 나머지 arm의 설정을 바꾸지 않는다.

## 결과 보고의 순서

1. 실제 바뀐 코드/입력/계산량과 완료 단위.
2. A/B/C/D/R 전체의 source·recipient·compatibility 수치.
3. C−B와 C−D가 다른 수신 encoder에서 어느 정도인지.
4. 반복 간 변동, patient 조건부 구간, AP, public-real 기준.
5. matching 오차·PNG quantization 손실·전체 비용.
6. 어떤 주장에 근거가 생겼고 어떤 주장은 미확인인지.
7. 기존 실패·역할 원장·expert/reserved 보존 여부.

`실행 정합성 PASS`와 `방법 효용 PASS`를 분리한다. 좋은 결과만 보고 합성 bank를 추가하거나, 실패 원인을 관계·분류기·라벨 중 하나로 자동 확정하지 않는다.

## DP 구현 시 반드시 추가할 사항

동봉 코드는 수학 참고이며 secure Gaussian sampler가 아니다. 실제 DP는 별도 모듈로 구현한다.

- B/C/R은 각자의 감도와 통계 차원에 전체 ε를 할당한다.
- **서로 다른 release의 noise는 독립적인 보안난수로 생성한다.** 공통 test bootstrap/합성초기화 seed를 공유하는 것과 DP noise를 공유하는 것은 다르다. noise를 공유한 여러 출력을 공개하면 차분으로 보호가 사라질 수 있다.
- 실제 DP noise seed, RNG state, 원통계, private patient ID/pairs는 공개하지 않는다.
- 같은 release에서 여러 합성 초기화를 만드는 것만 후처리다. 여러 release의 공동 공개는 composition을 기록한다.
- noisy raw 목표에 matching하는 것과 직접 분류기의 PSD 안정화를 구분한다. 보정이 의료영상 실현 가능성을 보장한다고 쓰지 않는다.
- nonDP Q-based HPO/선택을 무료 공개 상수로 바꾸지 않는다. 고정 메커니즘 보장과 전체 연구 산출물 보장 범위를 구분한다.
