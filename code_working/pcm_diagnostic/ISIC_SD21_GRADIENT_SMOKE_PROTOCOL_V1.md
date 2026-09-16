# ISIC SD 2.1 real-gradient smoke protocol v1

동결 시각: 2026-09-02, 결과 확인 전

## 목적과 금지 해석

이 단계는 실제 의료영상, frozen SD 2.1과 LoRA parameter gradient를 연결하는 첫 실행
smoke다. 한 환자만 사용하므로 PCM의 효과, 일반화, DP utility 또는 신규성을 판정하지 않는다.
결과를 본 뒤 환자, `Q`, clip norm 또는 feature를 교체하여 유리한 사례를 고르지 않는다.

## 동결 입력

- 데이터: SIIM-ISIC 2020 K5 manifest의 `public_development` partition
- eligibility: K5에 5장, 5개 distinct lesion, 3개 이상 non-empty anatomical site
- 선택: `sha256("pcm-isic-sd21-gradient-smoke-v1|" + patient_id)`가 가장 작은 환자 1명
- 이미지 크기: 256, deterministic center crop
- base: `Manojb/stable-diffusion-2-1-base`
- revision: `0094d483a120f3f33dafbd187ea4aa60d10de75c`
- artifact role: hash-pinned third-party mirror; official Stability AI repository라고 부르지 않음
- LoRA: rank 8, attention `to_q`, `to_k`, `to_v`, `to_out.0`
- LoRA state: 고정 seed의 초기 상태에서 optimizer update 없이 gradient만 평가
- VAE latent: posterior mode, scaling factor 적용
- perturbation: scheduler의 uniform timestep과 Gaussian noise
- reference bank: 각 record당 4개 perturbation, 총 20개 gradient
- patient evaluation budget: `Q=4`
- clipping norm: `C=1.0`
- Monte Carlo: 5 seeds, seed당 1,024 estimator trials

## frozen feature와 strata

각 이미지의 frozen SD 2.1 VAE latent를 `8 x 8`로 adaptive-average-pool하고 L2
normalize한다. farthest-first anchor 4개로 strata를 만들고 각 record를 가장 가까운 anchor에
할당한다. 이 feature는 결과 gradient를 보지 않고 생성한다.

## 비교 방법

1. `noise_only`: record 한 장을 균등 선택하고 perturbation 4개 평균
2. `uniform_distinct`: 서로 다른 record 4장을 균등 비복원 선택하고 perturbation 하나씩
3. `coverage_stratified`: 4개 VAE-feature strata에서 하나씩 선택하고 stratum 크기로 가중

모든 estimator는 finite reference bank에 대해 가중치 합이 1인 unbiased estimator다. 원본
gradient vector는 보고서에 저장하지 않고, 정확한 norm·거리·clipping metric 계산에 필요한
Gram matrix만 LOCAL_ONLY artifact로 저장한다.

## 측정값

- reference-gradient pre-clipping MSE
- clipped-reference post-clipping MSE
- clip rate
- clipping displacement
- cosine error
- gradient-bank norm, Gram symmetry와 PSD tolerance
- wall-clock과 peak CUDA memory

## smoke PASS 조건

- exact model critical hash 3개 PASS
- eligible patient의 deterministic selection과 5개 image load PASS
- gradient 20개가 모두 finite이며 parameter ordering·dimension 동일
- strata가 비어 있지 않고 method별 weight sum이 수치 오차 내에서 1
- Gram matrix가 symmetry·PSD tolerance를 통과
- artifact와 interpretation limit가 기록됨

세 방법 중 어느 것이 이기는지는 smoke PASS 조건이 아니다. 효능 비교는 이 코드가 통과한
뒤 별도로 동결할 multi-patient public-development diagnostic에서만 수행한다.
