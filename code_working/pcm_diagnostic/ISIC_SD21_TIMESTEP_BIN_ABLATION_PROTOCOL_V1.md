# ISIC SD 2.1 timestep-bin versus generic without-replacement ablation v1

동결 시각: 2026-09-02, confirmatory 결과와 antithetic/noise-multiplicity 선행 확인 후,
ablation 결과 확인 전

## 목적

8-perturbation confirmatory test에서 `bin_balanced_m4`가 i.i.d. replacement보다 낮은
finite-reference gradient MSE를 보였다. 그러나 이 개선은 실제 timestep stratification이 아니라
8개 중 서로 다른 perturbation 네 개를 복원 없이 고른 finite-population 효과일 수 있다.

본 ablation은 동일한 새 16 patients와 저장된 40 x 40 Gram matrix만 사용해 두 효과를 분리한다.
새 gradient, optimizer update, DP noise 또는 training을 하지 않는다.

## 고정 비교

모두 서로 다른 records 네 개, `Q=4`, cell weight `1/4`를 사용하며 40-cell finite reference에
대해 unbiased하다.

1. `replacement_uniform_m4`: 8 perturbations에서 record마다 독립 복원추출
2. `distinct_perturbation_m4`: 8 perturbations 중 서로 다른 네 개를 균등 비복원 선택하고
   네 records에 random permutation으로 배정
3. `true_bin_balanced_m4`: 실제 equal-width timestep bin 네 개에서 각각 한 perturbation을
   선택하고 records에 random permutation으로 배정

`distinct_perturbation_m4`는 `5 x C(8,4) x 4! = 8,400` states이고,
`true_bin_balanced_m4`는 1,920 states다.

Secondary placebo로 8 perturbations를 네 unordered pairs로 나누는 가능한 105 partitions를 모두
열거한다. 각 partition에서 pair마다 하나를 선택한다. 실제 chronological bin pairing의 환자별
rank와 105-partition 평균 대비 결과를 보고한다. 이는 post-confirmatory mechanism ablation이며
새 primary efficacy claim이 아니다.

## 사전 고정 incremental-signal gate

primary는 `true_bin_balanced_m4 / distinct_perturbation_m4` patient-paired pre-clipping MSE다.
다음을 모두 만족해야 `TIMESTEP_BIN_INCREMENT_RETAINS_SIGNAL`이다.

1. geometric-mean ratio `<= 0.95`
2. 10,000 patient bootstrap 95% upper bound `< 1.00`
3. true-bin win `>= 10/16`

하나라도 실패하면 `TIMESTEP_BIN_INCREMENT_NOT_SUPPORTED`로 판정한다. 이 경우 generic
without-replacement 결과는 알려진 finite-population/antithetic sampling의 강한 baseline으로만
유지하고, timestep-bin 균형 자체를 신규 방법 기여로 주장하지 않는다.

통과하더라도 VDM의 antithetic time sampling, DDIM 구현, DPDM noise multiplicity와 DP
augmentation/timestep multiplicity가 직접 선행이므로, privacy-unit-local placement의 차별성이
이론·DP training utility·X-ray에서 추가로 성립하기 전에는 신규성을 주장하지 않는다.

