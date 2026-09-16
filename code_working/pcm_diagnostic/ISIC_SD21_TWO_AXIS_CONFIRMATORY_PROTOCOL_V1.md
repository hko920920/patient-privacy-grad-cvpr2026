# ISIC SD 2.1 two-axis balance confirmatory protocol v1

동결 시각: 2026-09-02, four-bank development screen 결과 확인 후, confirmatory 결과 확인 전

## 확인할 주장과 제한

첫 two-axis screen은 동일 환자 update에서 서로 다른 records 네 개와 네 perturbation을 각각
한 번씩 쓰면 replacement baseline보다 finite-reference gradient MSE가 감소한다는 신호를
보였다. 그러나 perturbation population 자체가 네 개뿐이어서 balanced estimator가 매 update에
그 population을 전부 관측하는 유리한 구조였다.

이 confirmatory diagnostic은 (1) 이전 결과에 사용되지 않은 환자 16명과 (2) timestep bin당
서로 다른 timestep/noise 두 개, 총 8개의 독립 bank를 사용한다. balanced estimator도 각 bin의
두 perturbation 중 하나만 보므로 within-bin uncertainty가 남는다. optimizer update, DP noise와
generator training은 하지 않는다.

## 환자 선택

- source: SIIM-ISIC 2020 K5 `public_development`
- eligibility: 정확히 5 images와 5 distinct lesions
- exclusion: one-patient smoke `IP_0687884`와 multi-patient v1의 16명 전부
- low-site: non-empty anatomical site 1--2개
- high-site: non-empty anatomical site 3개 이상
- 각 group에서
  `sha256("pcm-isic-sd21-two-axis-confirm-v1|" + group + "|" + patient_id)`가
  작은 8명, 총 16명
- target, diagnosis, image feature와 gradient는 selection에 사용하지 않음

## 모델과 8-perturbation bank

- 동일 hash-pinned SD 2.1 revision `0094d483a120f3f33dafbd187ea4aa60d10de75c`
- 256 pixels, deterministic center crop, frozen VAE mode, rank-8 attention LoRA
- 4 equal-width timestep bins, bin당 replicate 2개
- timestep과 noise seed는 `stable_seed`의 confirmatory 전용 tag, bin과 replicate로 고정
- 같은 perturbation index는 records 사이에서 common-random-number timestep/noise 사용
- 환자당 5 x 8 = 40 gradients, 총 640 gradients
- raw gradient는 저장하지 않고 40 x 40 patient Gram matrix만 LOCAL_ONLY 저장
- finite reference는 40 gradients의 균등 평균

## 고정 비교

두 방법 모두 `Q=4`이고 네 records를 비복원으로 균등 선택한다.

1. `replacement_uniform_m4`: 각 record의 perturbation을 8-bank에서 독립 복원추출
2. `bin_balanced_m4`: 네 timestep bins를 records에 random permutation으로 배정하고, 각 bin의
   두 perturbation 중 하나를 균등 선택

가능한 상태를 정확 열거한다. replacement는 `5 x 8^4 = 20,480`, bin-balanced는
`5 x 4! x 2^4 = 1,920` states다. 두 estimator 모두 40개 cell에 대한 expected weight가
`1/40`이어야 한다.

## 사전 고정 confirmatory gate

`bin_balanced_m4 / replacement_uniform_m4` patient-paired pre-clipping MSE에서 다음을 모두
만족해야 `TWO_AXIS_BALANCE_CONFIRMED_FOR_METHOD_REVIEW`다.

1. geometric-mean ratio `<= 0.90`
2. 10,000 patient bootstrap 95% upper bound `< 1.00`
3. balanced win `>= 12/16`

하나라도 실패하면 `RETIRE_CURRENT_TWO_AXIS_BALANCE`다. 통과는 CVPR 신규 방법 확정이 아니라,
정확한 joint record/timestep-bin sampling 규칙에 대한 근접 선행 재검색과 X-ray gradient
diagnostic으로 진행할 자격만 준다.

## Secondary와 해석 금지선

- low/high site group별 paired 결과
- `C in {0.05, 0.10, 0.20, 0.50, 1.00}` exact post-clipping sensitivity
- bin-balanced의 clip rate와 displacement

본 결과만으로 DP guarantee, 학습 또는 생성 utility, 임상 성능, 실제 병원 일반화, timestep과
noise 효과의 완전한 분리, 신규성 또는 CVPR acceptability를 주장하지 않는다.

