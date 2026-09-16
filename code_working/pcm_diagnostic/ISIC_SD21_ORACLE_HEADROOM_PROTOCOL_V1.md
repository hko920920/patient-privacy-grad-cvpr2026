# ISIC SD 2.1 paired-strata oracle headroom protocol v1

동결 시각: 2026-09-02, 16-patient VAE coverage gate 결과 확인 후, oracle 결과 확인 전

## 목적과 경계

16-patient diagnostic은 서로 다른 record를 먼저 보는 `uniform_m4_r1`의 이점은 확인했지만
현재 frozen VAE coverage 규칙은 중단하도록 판정했다. 이 후속 분석은 이미 계산한 환자별
20 x 20 gradient Gram matrix만 사용해, `m=4` paired-strata 배분 family 자체에 의미 있는
상한(headroom)이 있는지 판정한다.

이 분석의 oracle은 reference gradient를 보고 가장 좋은 pair를 고르므로 실제 학습에서 사용할
수 없고 성능 claim도 아니다. 새 gradient, optimizer update, DP noise, generator training,
feature 교체 또는 환자 교체를 하지 않는다.

## 고정 입력

- source report: `pcm_isic_sd21_multipatient_v1_001`
- source patients: 결과와 무관하게 salted hash로 이미 선택된 동일 16명
- source gradient bank: 환자당 5 records x 4 perturbations의 Gram matrix
- reference: 20 gradients의 균등 평균
- compute budget: `Q=4`
- comparator: `uniform_m4_r1`

## 정확 열거

5 records를 4개 strata로 나누는 경우에는 정확히 한 stratum만 2 records이고 나머지는
singleton이다. 환자마다 가능한 record pair 10개를 모두 열거한다. 각 pair partition에서
2-record stratum은 한 record를 균등 선택해 weight `2/5`를 주고, singleton 세 개에는 각각
weight `1/5`를 준다. 각 선택 record에는 4-bank perturbation 하나를 균등 선택한다. 따라서
각 partition estimator는 finite reference에 대해 unbiased하다.

환자별로 다음을 계산한다.

1. `uniform_m4_r1` MSE
2. frozen VAE가 만든 pair의 MSE와 10개 중 rank
3. 10개 pair MSE의 평균인 uniform-random-pair MSE
4. reference gradient를 본 뒤 최소 MSE pair를 택한 oracle MSE
5. 최대 MSE pair인 anti-oracle MSE

## 사전 고정 headroom gate

다음 두 조건을 모두 만족할 때만 `CONTINUE_PAIR_PROXY_SEARCH`로 판정한다.

1. patient-paired `oracle / uniform_m4` MSE geometric-mean ratio `<= 0.90`
2. oracle이 uniform m4보다 좋은 환자 `>= 12/16`

하나라도 실패하면 `RETIRE_PAIR_STRATIFIED_M4_FAMILY`로 판정한다. 통과하더라도 oracle 자체를
방법으로 승격하지 않는다. 통과의 뜻은 독립 development split에서 사전 계산 가능한 저비용
proxy를 찾을 여지가 있다는 것뿐이다. 후속 proxy는 target 결과를 보지 않고 별도 동결하며,
uniform m4 대비 confirmatory 개선을 다시 요구한다.

## 실행 검증

- source execution/audit가 모두 PASS
- 16개 Gram matrix가 각각 20 x 20이고 finite/symmetric
- 환자마다 가능한 pair가 정확히 10개
- frozen VAE partition에는 size-2 stratum이 정확히 하나
- 재계산한 frozen VAE MSE와 source MSE가 허용오차 안에서 일치
- 모든 pair estimator의 expected weight가 20개 gradient에 대해 `1/20`

