# ISIC SD 2.1 timestep-balance multi-bank replication protocol v1

동결 시각: 2026-09-02, original 8-bank confirmatory와 true-bin mechanism ablation 결과를
확인한 후, 아래 5개 새 bank 결과를 확인하기 전

## 목적

original confirmatory의 16 patients가 같은 하나의 8-perturbation bank를 공유했기 때문에,
patient bootstrap은 perturbation-bank uncertainty를 포함하지 않는다. 본 실험은 cohort,
records, model, LoRA parameterization, image preprocessing와 `Q=4`를 바꾸지 않고 독립
timestep/noise banks만 다섯 번 새로 생성해 true timestep-bin 신호가 한 bank의 우연인지
검증한다.

## 고정 cohort와 모델

- original confirmatory의 동일한 salted-hash 16 patients와 동일한 5 records/patient
- SD 2.1 revision `0094d483a120f3f33dafbd187ea4aa60d10de75c`
- 256 pixels, frozen VAE mode, rank-8 attention LoRA
- optimizer update, DP noise, generator training 없음
- bank마다 5 x 8 x 16 = 640 gradients, 새 5 banks 총 3,200 gradients

## 새 banks

bank tags는 결과 전에 다음으로 고정한다.

1. `repl_01`
2. `repl_02`
3. `repl_03`
4. `repl_04`
5. `repl_05`

각 bank는 four equal-width timestep bins마다 두 개의 timestep/noise를 갖는다. seeds는
`stable_seed("two_axis_replication_timestep", bank_tag, bin, replicate)`와
`stable_seed("two_axis_replication_noise", bank_tag, bin, replicate)`로 고정한다. 다섯 bank와
original bank의 noise seeds는 겹치지 않아야 한다.

## bank별 exact 비교

저장한 40 x 40 patient Gram matrix에서 다음 세 estimator를 exact enumeration한다.

1. `replacement_uniform_m4`: 네 distinct records, perturbation iid with replacement
2. `distinct_perturbation_m4`: 네 distinct records와 네 distinct perturbations, bin 무시
3. `true_bin_balanced_m4`: four true bins에서 각각 하나, records에 random permutation

각 bank에서는 105개 가능한 perfect-pair partitions를 모두 열거해 true chronological
grouping rank도 계산한다.

## 사전 고정 multi-bank gate

primary는 bank별 patient-paired pre-clipping
`true_bin_balanced_m4 / distinct_perturbation_m4` geometric MSE ratio다. 다음을 모두 만족할
때만 `MULTIBANK_TIMESTEP_SIGNAL_REPLICATED_FOR_LOCALITY_TEST`로 판정한다.

1. 새 5/5 banks에서 bank-level geometric ratio `< 1.00`
2. 새 5/5 banks에서 true-bin wins `>= 10/16`
3. 다섯 bank-level ratios의 equal-bank geometric mean `<= 0.95`
4. bank와 patient를 각각 복원추출하는 10,000-replicate hierarchical bootstrap의
   95% upper bound `< 1.00`
5. 5개 중 최소 4개 bank에서 true grouping의 patient-median rank `<= 27/105`

1번의 5/5 방향 일치는 median-null 아래 사전 고정 one-sided sign probability `1/32`와 함께
보고하되, 이를 전체 방법의 통계적 증명으로 과장하지 않는다. original positive bank는
위 gate 계산에 넣지 않고 context로만 보고한다.

하나라도 실패하면 `MULTIBANK_TIMESTEP_SIGNAL_NOT_REPLICATED`로 판정하고, 현재 timestep-bin
method 후보를 locality/X-ray/DP pilot로 진행하지 않는다. adverse bank를 삭제하거나 seed를
추가해 5/5가 될 때까지 반복하지 않는다.

## 해석 경계

PASS는 perturbation-bank 우연에 대한 local-gradient replication일 뿐이다. timestep
stratification은 VDM, DDIM, CleanDIFT, HDiT 등의 선행이고 timestep multiplicity는 DPDM 및
후속 DP diffusion 선행이다. PASS하더라도 batch-global 대 patient-unit-local 차별성,
variable-record rule, clipping theory, X-ray와 실제 patient-DP utility 전에는 신규성,
training utility, privacy, clinical 또는 CVPR claim을 하지 않는다.
