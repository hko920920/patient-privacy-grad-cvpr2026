# ISIC SD 2.1 two-axis balanced perturbation diagnostic protocol v1

동결 시각: 2026-09-02, paired-strata oracle 결과 확인 후, two-axis 결과 확인 전

## 목적

16-patient diagnostic에서 `uniform_m4_r1`은 모든 환자에서 m1/m2보다 낮은 MSE를 보였고,
총분산 중 between-record fraction의 중앙값은 0.167이었다. 현재 VAE paired-strata 규칙과
그 oracle 상한은 중단 판정을 받았다. 다음 screen은 record coverage를 uniform distinct-first로
유지하면서, 지배적인 within-perturbation 변동을 줄이기 위해 환자 update마다 4개 동결
timestep/noise perturbation을 중복 없이 한 번씩 쓰는 two-axis balance를 평가한다.

이것은 아직 신규 방법 또는 선행 비중복 claim이 아니다. finite-bank diagnostic이며 optimizer
update, DP noise, generator training, 환자/feature 교체를 하지 않는다.

## 고정 입력과 estimator

- 동일 16 patients, 환자당 5 records x 4 perturbations의 기존 Gram matrix
- `Q=4`, finite reference는 20 gradients의 균등 평균
- source perturbation 0--3은 각각 서로 다른 equal-width timestep bin과 frozen noise seed를 가짐
- comparator는 perturbation을 복원추출하는 기존 `uniform_m4_r1`

balanced methods는 다음과 같다.

1. `balanced_m1_r4`: record 하나를 균등 선택하고 perturbation 0--3을 각각 한 번 사용
2. `balanced_m2_r2`: 서로 다른 records 둘을 선택하고 네 perturbation을 2개씩 중복 없이 배분
3. `balanced_m4_r1`: 서로 다른 records 넷을 선택하고 네 perturbation의 random permutation을
   record마다 하나씩 배정

모든 record choice와 고유 perturbation assignment를 정확 열거한다. 각 gradient의 weight는
`1/4`이고, 전체 randomization에 대한 각 record-perturbation cell의 expected weight는 `1/20`이다.

## 사전 고정 gate

primary comparison은 `balanced_m4_r1 / uniform_m4_r1` patient-paired pre-clipping MSE다.
다음 세 조건을 모두 만족할 때만 `CONTINUE_TWO_AXIS_BALANCE`로 판정한다.

1. geometric-mean ratio `<= 0.90`
2. 10,000 patient bootstrap 95% upper bound `< 1.00`
3. balanced m4 win `>= 12/16`

하나라도 실패하면 `RETIRE_CURRENT_TWO_AXIS_BALANCE`로 판정한다. 통과하더라도 같은 환자와
finite bank에서 얻은 development screen일 뿐이다. 독립 perturbation bank 또는 독립 환자에서
confirmatory 반복을 먼저 통과하고, timestep stratification 및 variance-reduction 선행과의
비중복성을 재검증한 뒤에만 방법 후보로 유지한다.

## Secondary와 경계

- m1/m2/m4에서 replacement 대비 balanced perturbation 효과
- frozen `C in {0.05, 0.10, 0.20, 0.50, 1.00}` post-clipping MSE와 clip rate
- site group별 paired ratio

이 screen은 DP guarantee, 학습 utility, 생성 품질, 임상 utility 또는 CVPR 신규성을 입증하지
않는다. perturbation index가 timestep과 noise seed를 함께 바꾸므로 첫 screen만으로 어느 요소가
효과를 냈는지 분리해 주장하지 않는다.

