# ISIC SD 2.1 multi-patient allocation diagnostic protocol v1

동결 시각: 2026-09-02, multi-patient gradient 결과 확인 전

## 목적

한 환자 smoke에서 실행 경로는 통과했지만 frozen-VAE coverage가 uniform-distinct보다
나빴다. 이 프로토콜은 smoke 환자를 제외한 독립적인 16명에서 현재 VAE coverage 규칙을
계속 연구할 가치가 있는지 판정한다. 이 단계도 optimizer update, DP noise와 generator
training을 하지 않으며 최종 의료·utility·privacy 결과가 아니다.

## 결과 독립 cohort

- source: SIIM-ISIC 2020 K5 manifest `public_development`
- eligibility: 정확히 5 images, 5 distinct lesions
- smoke exclusion: `IP_0687884`
- low-site stratum: non-empty anatomical site 1--2개
- high-site stratum: non-empty anatomical site 3개 이상
- 각 stratum에서
  `sha256("pcm-isic-sd21-multipatient-v1|" + stratum + "|" + patient_id)`가 작은 8명
- 총 16명이며 target/diagnosis/영상/gradient를 selection에 사용하지 않는다.
- 선택된 모든 환자와 null/adverse 결과를 최종 표에 유지한다.

## 모델·gradient bank

- base: `Manojb/stable-diffusion-2-1-base`
- revision: `0094d483a120f3f33dafbd187ea4aa60d10de75c`
- exact critical fp16 hash 3개 재검증
- resolution 256, deterministic center crop
- frozen VAE posterior mode latent, rank-8 attention LoRA
- optimizer update 없음; 모든 환자는 동일한 초기 LoRA state에서 평가
- record당 4 perturbations
- timestep은 4개 equal-width bin에서 하나씩 deterministic sampling
- 같은 perturbation index는 record 간 common-random-number timestep/noise를 사용
- 환자당 20 gradients, 전체 320 gradients
- raw gradient는 저장하지 않고 per-patient Gram matrix만 LOCAL_ONLY로 저장

## 고정 계산량 비교

환자당 `Q=4` gradient evaluations를 고정하고 finite gradient bank에 대한 estimator를 정확
열거한다.

1. `uniform_m1_r4`: record 1개, perturbation 4회
2. `uniform_m2_r2`: 서로 다른 record 2개, record당 perturbation 2회
3. `uniform_m4_r1`: 서로 다른 record 4개, perturbation 1회
4. `vae_stratified_m2_r2`: VAE feature strata 2개, stratum당 record 1개와 perturbation 2회
5. `vae_stratified_m4_r1`: VAE feature strata 4개, stratum당 record 1개와 perturbation 1회

unequal-size strata는 `|stratum| / n_u`로 가중하여 finite reference에 대해 unbiased하게
한다. 모든 가능한 record choice와 4-bank perturbation sequence를 열거하므로 estimator
Monte Carlo seed 오차는 없다.

## 분석

- primary: patient별 pre-clipping MSE
- secondary: cosine error, estimator norm
- total variance를 between-record와 within-record perturbation component로 exact 분해
- VAE feature pairwise distance와 record-mean gradient distance의 Spearman correlation
- clipping sensitivity grid `C in {0.05, 0.10, 0.20, 0.50, 1.00}` 전체 보고
- primary comparison:
  `vae_stratified_m4_r1 / uniform_m4_r1` patient-paired MSE ratio
- 10,000회 patient bootstrap으로 geometric-mean ratio의 95% interval 계산

## 현재 VAE coverage 계속 조건

다음 네 조건을 모두 만족할 때만 `CONTINUE_VAE_COVERAGE`다.

1. paired geometric-mean MSE ratio `<= 0.95`
2. bootstrap 95% upper bound `< 1.00`
3. 16명 중 coverage win `>= 10`명
4. median VAE-feature/gradient-distance Spearman correlation `> 0.10`

하나라도 실패하면 `RETIRE_CURRENT_VAE_COVERAGE`로 판정한다. 이 판정은 patient-level DP와
multi-record 문제 전체를 폐기한다는 뜻이 아니다. frozen VAE clustering을 신규 방법으로
밀지 않고, uniform-distinct를 강한 baseline으로 유지하면서 다른 선행 비중복 allocation
규칙을 탐색한다는 뜻이다.

## 실행 PASS와 해석 경계

실행 PASS는 exact hashes, 320 finite gradients, 16 nonempty strata, exact estimator
unbiasedness, Gram PSD와 variance decomposition identity가 통과했다는 뜻이다. 효능 gate와
분리해서 보고한다. 이 결과만으로 DP guarantee, 생성 품질, 임상 utility 또는 CVPR 신규성을
주장하지 않는다.
