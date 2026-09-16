# ISIC 실제-gradient 진단과 현재 방법 판정

- 기록일: 2026-09-02
- 최신 갱신: 2026-09-03 covariance-allocation discovery
- 출발 논문: CVPR 2026 PFDM
- 실험 범위: ISIC public-development, Stable Diffusion 2.1 rank-8 LoRA
- 현재 판정: **상위 patient-level DP 의료영상 연구 문제는 유지, VAE coverage 규칙과
  단순 unit-local timestep balance 및 후속 covariance-aware rank-template branch는 폐기.
  actual joint와 사전 discovery headroom gate가 차례로 실패**

## 1. 한 문장 결론

> 한 환자 안에서는 record와 diffusion timestep 구간의 균형화가 gradient 오차를 줄였지만,
> 실제 두 환자의 clipped contribution을 평균하면 global complementary 배분의 오차 상쇄가
> 더 커 local 방법의 joint MSE가 `0.646%` 증가했다. 따라서 이 단순 규칙은 신규 방법으로
> 진행하지 않는다. covariance를 포함해 35개 배분을 다시 탐색해도 실행 가능한 공통규칙의
> 개선은 `3.86%`, pair oracle 한계는 `5.81%`여서 현재 timestep-allocation branch를 종료한다.

## 2. 결과를 본 순서와 판정

모든 수치는 optimizer update와 DP noise 없이 저장한 patient별 finite-reference gradient에
대한 정확 열거 MSE다. 결과를 본 뒤 환자나 feature를 유리하게 바꾸지 않았다.

| 단계 | 사전 고정 질문 | 핵심 결과 | 판정 |
|---|---|---:|---|
| 1-patient smoke | 구현이 실제 SD 2.1 LoRA에서 동작하는가 | uniform-distinct/noise-only `0.486`; VAE는 uniform보다 나쁨 | 실행 PASS, 효능 판정 보류 |
| 16-patient VAE diagnostic | VAE coverage가 uniform distinct보다 좋은가 | VAE/uniform `1.0322`, 95% CI `[0.9674, 1.1005]`, 6/16 wins | **VAE coverage 폐기** |
| paired-strata oracle | 5 records를 두 쌍으로 나누는 family에 충분한 headroom이 있는가 | outcome-informed oracle/uniform `0.9749`, 8/16 wins | **pair-stratified family 폐기** |
| 첫 4-bank screen | record와 perturbation을 동시에 비복원 균형화하면 좋은가 | balanced/replacement `0.7440`, 16/16 wins | favorable finite-bank screen만 통과 |
| 새 환자·8-bank confirmatory | 새 16명과 더 큰 perturbation bank에서도 유지되는가 | true-bin/replacement `0.8592`, CI `[0.7931, 0.9220]`, 16/16 wins | method review로 진행 |
| mechanism ablation | 이득이 단순 perturbation 중복 방지보다 큰가 | true-bin/distinct `0.9211`, CI `[0.8814, 0.9572]`, 16/16 wins | 고정 incremental gate PASS |
| 5-new-bank replication | 한 perturbation bank의 우연인가 | equal-bank true-bin/distinct `0.8965`, hierarchical CI `[0.8309, 0.9419]`, 5/5 banks 방향 일치 | **multi-bank gate PASS** |
| unit-local clipping proxy | 같은 batch-wide timestep multiset에서 patient 내부 균형이 clipping 뒤에도 유리한가 | `C=0.20` local/global `0.9094`, hierarchical CI `[0.8446, 0.9506]`, 78/80 cells | **proxy PASS; actual joint는 후속 FAIL** |
| actual B=2 joint batch | cross-patient covariance까지 포함한 실제 clipped batch update가 좋은가 | joint local/global `1.0065`, bank CI `[1.0021, 1.0117]`, 1/5 banks 및 191/600 pair wins | **locality increment 폐기** |

16-patient VAE diagnostic은 320개, 새 confirmatory는 640개, multi-bank replication은
새로 3,200개의 finite LoRA gradient를 실제로 계산했다. confirmatory, 각 bank와 multi-bank
aggregate의 별도 audit가 판정을 재현했다. 2026-09-03 clipping proxy 추가 뒤에는 전체
31개 tests가 PASS했으며 raw-Gram 독립 감사도 400개 patient-bank-C 셀과 판정을 재현했다.
이후 새 다섯 joint banks에서 3,200 gradients와 full cross-patient Gram을 추가 계산했다.
actual joint 분석의 별도 감사가 최대 상대오차 `2.50e-14`로 실패 판정을 재현했고 전체
37개 tests가 PASS했다.

## 3. mechanism ablation의 정확한 뜻

새 confirmatory에서는 환자마다 5 records와 8 perturbations, 즉 40개 gradient cell을
reference로 사용했다. `Q=4`에서 다음 세 estimator를 exact enumeration했다.

1. `replacement_uniform_m4`: 네 개의 서로 다른 record에 perturbation을 독립 복원 배정
2. `distinct_perturbation_m4`: 8개 중 서로 다른 perturbation 네 개를 고르되 시간구간은 무시
3. `true_bin_balanced_m4`: 네 equal-width timestep bin에서 하나씩 고르고 record에 무작위 배정

분해 결과는 다음과 같다.

- generic distinct/replacement: `0.9328`, 즉 약 6.7% 감소
- true-bin/distinct: `0.9211`, 즉 generic 중복 방지 위에서 약 7.9% 추가 감소
- true-bin/replacement: `0.8592`, 즉 합쳐서 약 14.1% 감소
- true chronological pairing의 105개 perfect-pair partition 중 환자별 rank:
  13/16명 1위, 나머지 4·4·11위, 중앙값 1위, 전원 상위 사분위
- true-bin은 distinct보다 16/16명에서 작았지만 개별 ratio 범위는
  `0.7598`--`0.9984`로, 일부 환자에서는 효과가 매우 작았다.

105개 partition 평균은 generic distinct estimator와 수치적으로 동일하다. 따라서 true-bin
대 distinct 비교는 arbitrary pairing의 평균이 아니라 실제 chronological grouping이 주는
추가 신호를 분리한다.

## 4. 반드시 붙는 한계

1. 최초 confirmatory의 단일-bank 한계는 결과 확인 전 고정한 새 5 banks로 보강했다. 그러나
   여섯 banks 모두 동일 16-patient cohort와 동일 checkpoint/parameter point를 사용했다.
2. ISIC 한 modality, SD 2.1 한 checkpoint, rank-8 LoRA 한 지점의 local gradient 진단이다.
3. finite-bank reference MSE 개선이지 optimizer convergence, 생성 품질, 의료 utility 또는
   DP utility 개선이 아니다.
4. patient-marginal post-clipping proxy는 `C=0.20`에서 PASS했지만 cross-patient covariance를
   포함한 actual batch update, 실제 DP training의 clipping/noise와 optimizer trajectory를
   대신하지 않는다.
5. public data 결과이므로 실제 병원 workflow나 임상적 유효성을 입증하지 않는다.

## 5. 근접연구 재조사: 무엇이 이미 알려져 있는가

### 5.1 직접 충돌

| 선행 | 이미 점유한 내용 | 현재 후보에 남는 차이 |
|---|---|---|
| VDM, NeurIPS 2021 | minibatch 전체에 evenly shifted time을 배치하는 antithetic time sampling | patient clipping 내부의 unit-local 배치는 다루지 않음 |
| DDIM 공식 구현 | minibatch timestep을 `t`와 `T-t-1`로 짝지음 | 같은 경계 |
| DPDM, TMLR 2023 | 한 image의 여러 `(noise level, noise)` gradient를 clipping 전에 평균; noise multiplicity | 여러 patient-linked records와 stratified bin의 결합은 직접 평가하지 않음 |
| Ghalebikesabi et al. | image별 여러 timestep 및 augmentation gradient를 clipping 전에 평균 | timestep multiplicity 자체는 이미 점유 |
| CleanDIFT, CVPR 2025 | 한 training image에 대해 timestep range를 세 구간으로 나눠 구간별 noise level 사용 | 비-DP feature alignment이고 patient unit이 아님 |
| HDiT, ICML 2024 | 일반 diffusion training에서 stratified timestep sampling 사용 | stratification은 일반 training trick임을 확인 |
| Adaptive non-uniform timestep, CVPR 2025 | timestep별 gradient variance 차이와 adaptive marginal sampling | timestep variance 또는 timestep selection 자체를 신규 claim으로 둘 수 없음 |
| ULS/fixed-compute user DP | 여러 user records를 평균해 per-user clipping, fixed compute 비교 | patient record와 diffusion time의 내부 배치만 남음 |

따라서 다음 주장은 모두 금지한다.

- 최초 antithetic 또는 stratified timestep sampling
- 최초 diffusion noise/timestep multiplicity
- 최초 patient/user record 평균 및 clipping
- 최초 fixed-compute multiplicity
- variance decomposition 자체를 방법 기여로 주장

### 5.2 남을 수 있는 좁은 교집합

현재 검색에서 정확히 같은 완성형을 찾지는 못했지만, 이는 부재의 증명이 아니다. 남을 수
있는 교집합은 다음과 같다.

> 여러 record를 가진 하나의 privacy unit마다 record marginal과 timestep marginal을
> 동시에 unbiased하게 균형화하고, 그 unit vector를 만든 뒤 한 번 clipping하는 estimator를
> 설계한다. minibatch-global timestep balance와 달리 clipping의 비선형성 전에 각 환자의
> stochastic error를 줄이는 것이 목적이다.

이 차이는 `ULS + DPDM + stratified timestep`의 자명한 결합으로 평가될 위험이 크다. 논문
기여가 되려면 단순 배치 변경이 아니라 다음이 필요하다.

1. variable record count에서도 unbiased한 exact rule
2. 같은 계산량에서 iid unit-local, generic without-replacement, batch-global antithetic,
   unit-local stratified의 정면 비교
3. per-unit clipping 때문에 global balance와 local balance가 달라지는 이론 또는 bound
4. independent perturbation banks에서의 재현성
5. X-ray core와 ISIC extension에서 실제 patient-DP utility 개선

## 6. 잠정 알고리즘 형태

이름은 아직 고정하지 않는다. `Q` gradient evaluations를 가진 sampled patient `u`에 대해
다음 형태만 후보로 유지한다.

1. `n_u >= Q`이면 서로 다른 records `Q`개를 uniform without replacement로 고른다.
2. `n_u < Q`이면 모든 record에 query를 가능한 균등하게 배정하고, 나머지 query의 record는
   무작위화하여 각 record의 기대 weight를 동일하게 한다.
3. 목표 timestep distribution을 equal-probability strata `Q`개로 나누고 각 구간에서
   timestep과 noise를 하나씩 독립 추출한다.
4. timestep strata를 selected record slots에 무작위 permutation으로 배정한다.
5. `Q` gradients를 평균한 뒤 patient vector를 `C`로 한 번 clip한다.
6. sampled patient vectors를 집계하고 patient-level DP noise를 한 번 더한다.

1--4의 내부 난수화는 최종 patient vector가 한 번 clipping되는 한 privacy accountant의
sampled patient unit을 바꾸지 않는 후보 설계다. 그러나 exact sampler/accountant 일치와
implementation proof는 별도 gate에서 검증해야 한다.

## 7. 다음 고정 gate

### G2a. independent perturbation-bank replication — PASS

- 같은 frozen cohort와 model에서 결과 전에 `repl_01`--`repl_05`를 commit했다.
- 새 5 banks, bank당 640개, 총 3,200개 gradients가 finite였고 bank별 execution·audit·
  105-partition ablation이 모두 PASS했다.
- true-bin/distinct bank ratios는 `0.9354`, `0.9170`, `0.7862`, `0.9356`, `0.9177`이고
  wins는 각각 15, 16, 16, 16, 15/16이었다.
- equal-bank ratio는 `0.8965`, bank/patient hierarchical bootstrap 95% CI는
  `[0.8309, 0.9419]`였다. 5/5 direction의 사전 one-sided sign probability는 `1/32`다.
- 판정은 `MULTIBANK_TIMESTEP_SIGNAL_REPLICATED_FOR_LOCALITY_TEST`다. 이는 G2b로 갈
  자격이지 방법 승격이나 신규성 확정이 아니다.

### G2b. locality ablation — marginal PASS, actual joint FAIL

- `B=2`, `Q=4`에서 global-only는 여덟 perturbation을 두 환자에게 임의의 4+4로, local은
  각 bin의 두 perturbation을 1+1로 상보 배정하는 same-compute 구성을 고정했다.
- 저장 within-patient Gram의 exact marginal proxy에서 `C=0.20` local/global postclip MSE는
  `0.9094`, hierarchical CI `[0.8446, 0.9506]`, 5/5 banks, 78/80 cells 방향 일치였다.
- global/local 평균 clip rate는 `0.1948`/`0.1379`; 모든 frozen gate와 독립 audit가 PASS했다.
- 후속 새 다섯 full cross-patient Gram banks의 actual B=2 test에서 joint local/global은
  `1.0065`, bank CI `[1.0021, 1.0117]`, 1/5 banks 및 191/600 pair wins였다.
- marginal local/global은 새 banks에서도 `0.9430`으로 재현됐지만 global의 음의 cross-patient
  covariance 상쇄가 더 컸다. 사전 기준에 따라 이 patient-specific locality claim을 중단한다.

### G2c. X-ray public-development gradient diagnostic

- primary는 동결한 NIH PA-only patient manifest와 official exact PNG
- finding/device/marker 및 record-count strata 보고
- ISIC-only 결과로 X-ray core를 승격하지 않음

### G3. patient-DP one-seed utility pilot

- 동일 patient epsilon/delta, patient sampling, `Q`, `C`, steps와 trainable parameters
- ULS, iid noise multiplicity, generic distinct, batch-global balance, unit-local balance 비교
- estimator가 좋아도 생성·의료 utility가 개선되지 않으면 CVPR method claim 중단

## 8. 현재 추천 순위의 변화

- **연구 프로그램 1순위:** 유지. patient-private medical diffusion과 image-DP
  checkpoint reuse/retrain audit는 문제 중요성·로컬 준비도·결과의 판정 가능성이 가장 높다.
- **기존 VAE-PCM:** 폐기. multi-patient gate와 oracle headroom이 지지하지 않았다.
- **unit-local timestep-balanced estimator:** actual joint gate 실패로 방법 후보에서 중단한다.
- **다음 방법 후보:** 아직 없음. marginal variance와 cross-unit covariance를 함께 낮추는
  공개정보 기반 규칙의 headroom을 post hoc으로 먼저 진단해야 한다.
- **현재 CVPR 적합성:** 상위 주제는 적합하지만 현재 locality 규칙으로는 method paper가
  성립하지 않는다. 새 규칙의 독립 검정 또는 방법 중심이 아닌 별도 연구 질문이 필요하다.

## 9. 1차 문헌

- PFDM, CVPR 2026:
  https://openaccess.thecvf.com/content/CVPR2026/html/Patel_Personalized_Federated_Training_of_Diffusion_Models_with_Privacy_Guarantees_CVPR_2026_paper.html
- VDM, NeurIPS 2021:
  https://research.google/pubs/variational-diffusion-models/
- VDM official antithetic implementation:
  https://github.com/google-research/vdm/blob/main/model_vdm.py
- DDIM official antithetic implementation:
  https://github.com/ermongroup/ddim/blob/main/runners/diffusion.py
- DPDM:
  https://arxiv.org/html/2210.09929v3
- Differentially Private Diffusion Models Generate Useful Synthetic Images:
  https://arxiv.org/html/2302.13861
- CleanDIFT, CVPR 2025:
  https://arxiv.org/abs/2412.03439
- HDiT, ICML 2024:
  https://proceedings.mlr.press/v235/crowson24a.html
- Adaptive non-uniform timestep sampling, CVPR 2025:
  https://openaccess.thecvf.com/content/CVPR2025/html/Kim_Adaptive_Non-Uniform_Timestep_Sampling_for_Accelerating_Diffusion_Model_Training_CVPR_2025_paper.html
- Learning with User-Level DP under Fixed Compute Budgets:
  https://arxiv.org/abs/2407.07737
- P3SGD, CVPR 2019:
  https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_P3SGD_Patient_Privacy_Preserving_SGD_for_Regularizing_Deep_CNNs_in_CVPR_2019_paper.html

## 10. 산출물 위치

- multi-patient report: `code_working/_reports/pcm_isic_sd21_multipatient_v1_001/`
- new-cohort confirmatory report:
  `code_working/_reports/pcm_isic_sd21_two_axis_confirm_v1_001/`
- mechanism ablation summary:
  `code_working/_reports/pcm_isic_sd21_two_axis_confirm_v1_001/timestep_bin_ablation.md`
- five-bank aggregate:
  `code_working/_reports/pcm_isic_sd21_timestep_multibank_v1_001/`
- patient-unit-local clipping proxy:
  `code_working/_reports/pcm_isic_sd21_unit_local_clipping_proxy_v1_001/`
- actual B=2 cross-patient analysis:
  `code_working/_reports/pcm_isic_sd21_joint_batch_locality_v1_001/`
- covariance-allocation discovery:
  `code_working/_reports/pcm_isic_sd21_covariance_allocation_discovery_v1_001/`
- 최신 상세 판정:
  `CVPR 주제 탐색/09_covariance_allocation_discovery_결과_2026-09-03.md`
- protocols and code: `code_working/pcm_diagnostic/`

이 기록은 generator 학습 완료, patient-DP 보장, clinical validation, 신규성 확정 또는 CVPR
채택 가능성의 보장을 뜻하지 않는다.

## 11. 2026-09-03 actual B=2 결과에 따른 최종 갱신

이 문서에서 unit-local 후보를 조건부로 유지했던 이전 표현은 actual joint-batch 결과로
superseded된다. 새 다섯 banks와 120 patient pairs 전체에서 primary local/global ratio는
`1.0064606`, bank bootstrap 95% CI는 `[1.0020765, 1.0117106]`였다. ratio < 1인 bank는
1/5, 사전 84/120 pair-wins 조건을 충족한 bank는 0/5, leave-one-patient-out wins는 0/16이다.

동시에 새 bank의 patient-marginal ratio는 `0.9429900`으로 5/5 banks에서 1 미만이었다.
즉 marginal 개선은 실제로 재현됐지만, global arm의 더 큰 음의 cross-patient error
covariance가 이를 상쇄하고도 남았다. 따라서 사전 판정
`UNIT_LOCAL_JOINT_BATCH_INCREMENT_NOT_SUPPORTED`를 받아들이며 단순 unit-local 규칙은
핵심 방법 claim에서 중단한다.

상세 수치·분해·무결성 기록의 최신 authority는
`08_actual_B2_joint_batch_locality_결과_2026-09-03.md`다.

## 12. Covariance-allocation discovery 이후의 최종 판정

35개 orientation-symmetric public rank templates를 결과 전에 정한 leave-one-bank-out
절차로 전수 조사했다. 모든 held-out folds가 같은 low-two-bins 대 high-two-bins
`rt_0123`을 선택했고 ratio는 `0.961428`, bank CI `[0.944857, 0.976267]`, 5/5 bank
방향이었다. 그러나 사전 최소 effect `<=0.95`와 5/5 pair-wins gate 중 두 조건을 실패했다.

이 규칙은 patient marginal을 약 12.8% 악화하고 clip rate를 `0.1614`에서 `0.3689`로
높이는 대신 cross-patient cancellation을 키운다. 또한 본질적으로 low/high timestep의
batch-global antithetic 분리라 기존 선행과 차별화도 약하다. pair-specific hindsight oracle도
`0.941876`에 그쳐 더 복잡한 정책의 절대 headroom이 작다.

따라서 unit-local뿐 아니라 현재 정의한 B=2 covariance-aware timestep-allocation family를
모두 방법 후보에서 중단한다. 상위 patient-level DP 의료영상 연구 문제만 유지한다. 최신
authority는 `09_covariance_allocation_discovery_결과_2026-09-03.md`다.
