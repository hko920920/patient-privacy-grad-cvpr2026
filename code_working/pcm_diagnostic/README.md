# PCM allocation diagnostic

이 폴더는 Patient-Coverage Multiplicity(PCM)를 완성된 방법으로 가정하지 않고,
고정 gradient-evaluation budget에서 record 선택과 diffusion perturbation 반복을 비교하기
위한 작은 진단 코드다.

## 단계 0: synthetic harness

`run_synthetic_allocation_diagnostic.py`는 entity `u`, record `j`, perturbation `k`의
gradient를 다음 세 층으로 구성한다.

```text
gradient = entity mean + record effect + perturbation noise
```

동일한 총 계산량 `Q = m * r`에서 다음 allocation을 모두 비교한다.

- `m`: 서로 다른 record 수
- `r`: record당 독립 perturbation 반복 수
- `uniform`: record를 균등 비복원 추출
- `stratified`: 사전 고정된 equal-size coverage strata에서 추출

측정값은 reference patient gradient에 대한 pre-clipping MSE, clipped-reference에 대한
post-clipping MSE, clip rate와 clip displacement다.

세 synthetic scenario를 함께 사용한다.

1. `no_record_heterogeneity`: record effect가 없는 negative control
2. `aligned_strata`: selection strata와 실제 record modes가 정렬된 positive control
3. `misaligned_strata`: feature strata가 gradient structure와 정렬되지 않은 control

이 단계의 PASS는 코드와 metric의 내부 통제가 맞는다는 뜻일 뿐이다. PCM의 실제 의료영상
효과, DP utility, privacy guarantee 또는 신규성을 입증하지 않는다.

## 실행

프로젝트의 base-gate 가상환경을 사용한다.

```powershell
..\base_gate\.venv\Scripts\python.exe .\run_synthetic_allocation_diagnostic.py `
  --output-dir ..\_reports\pcm_synthetic_allocation_v1_001

..\base_gate\.venv\Scripts\python.exe -m unittest -v `
  .\test_synthetic_allocation_diagnostic.py
```

단계 0가 통과한 뒤에만 실제 ISIC image latent와 SD 2.1 LoRA gradient를 사용하는 단계 1
diagnostic을 추가한다.

## 실제-gradient 단계와 현재 판정

### One-patient smoke

`run_isic_sd21_gradient_smoke.py`는 ISIC public-development의 hash-selected 한 환자,
5 records x 4 perturbations에서 실제 SD 2.1 rank-8 LoRA gradient를 계산한다. 실행 gate는
PASS했지만 VAE coverage는 uniform distinct보다 나빴다.

### 16-patient VAE diagnostic 및 oracle

```powershell
..\base_gate\.venv\Scripts\python.exe .\run_isic_sd21_multipatient_diagnostic.py <frozen args>
..\base_gate\.venv\Scripts\python.exe .\audit_isic_sd21_multipatient_results.py <frozen args>
..\base_gate\.venv\Scripts\python.exe .\run_isic_sd21_oracle_headroom.py <frozen args>
```

- report: `../_reports/pcm_isic_sd21_multipatient_v1_001/`
- 320 finite gradients, execution/audit PASS
- VAE/uniform geometric MSE ratio `1.0322`, CI `[0.9674, 1.1005]`, 6/16 wins
- paired oracle/uniform `0.9749`, 8/16 wins
- decisions: `RETIRE_CURRENT_VAE_COVERAGE`, `RETIRE_PAIR_STRATIFIED_M4_FAMILY`

### New-cohort 8-perturbation two-axis confirmatory

```powershell
..\base_gate\.venv\Scripts\python.exe .\run_isic_sd21_two_axis_confirmatory.py <frozen args>
..\base_gate\.venv\Scripts\python.exe .\audit_isic_sd21_two_axis_confirmatory.py <frozen args>
```

- report: `../_reports/pcm_isic_sd21_two_axis_confirm_v1_001/`
- 이전 17 patients와 overlap 0인 새 16 patients
- 5 records x 8 perturbations, 640 finite gradients
- true-bin/replacement geometric ratio `0.8592`, CI `[0.7931, 0.9220]`, 16/16 wins
- decision: `TWO_AXIS_BALANCE_CONFIRMED_FOR_METHOD_REVIEW`

### True timestep-bin mechanism ablation

```powershell
..\base_gate\.venv\Scripts\python.exe .\run_isic_sd21_timestep_bin_ablation.py `
  --report-dir ..\_reports\pcm_isic_sd21_two_axis_confirm_v1_001
```

- 새 gradient 없이 저장 Gram matrix만 exact enumeration
- generic distinct/replacement `0.9328`
- true-bin/distinct `0.9211`, CI `[0.8814, 0.9572]`, 16/16 wins
- true chronological pairing rank: 13/16 patients 1위, 나머지 4·4·11위
- decision: `TIMESTEP_BIN_INCREMENT_RETAINS_SIGNAL`

전체 CPU test suite:

```powershell
..\base_gate\.venv\Scripts\python.exe -m unittest discover -s . -p 'test_*.py' -v
```

### Five-new-bank replication

`ISIC_SD21_TIMESTEP_MULTIBANK_REPLICATION_PROTOCOL_V1.md`는 `repl_01`--`repl_05`의
timestep/noise seed와 multi-bank gate를 결과 전에 고정한다. 각 bank는
`run_isic_sd21_two_axis_confirmatory.py`의 `--bank-tag`와 `--protocol-file`을 사용해 실행한
뒤 confirmatory audit와 timestep-bin ablation을 적용한다.

- reports: `../_reports/pcm_isic_sd21_timestep_repl_01/`--`repl_05/`
- aggregate: `../_reports/pcm_isic_sd21_timestep_multibank_v1_001/`
- new gradients: 5 x 640 = 3,200, 모두 finite
- true-bin/distinct bank ratios: `0.9354`, `0.9170`, `0.7862`, `0.9356`, `0.9177`
- equal-bank ratio `0.8965`, hierarchical CI `[0.8309, 0.9419]`
- decision: `MULTIBANK_TIMESTEP_SIGNAL_REPLICATED_FOR_LOCALITY_TEST`
- independent aggregate audit: PASS

2026-09-02 기준 25 tests PASS다. 이 결과는 finite-bank gradient diagnostic이다. VAE
coverage는 폐기됐고, timestep balance는 VDM/DDIM/DPDM/CleanDIFT/HDiT의 직접 선행 때문에
그 자체가 신규 기여가 아니다. independent perturbation-bank gate는 PASS했으며,
batch-global 대 unit-local, X-ray와 실제 patient-DP utility gate 전에는
training·privacy·novelty claim을 하지 않는다.

### Patient-unit-local clipping proxy

2026-09-03에는 새 comparator 값을 보기 전에
`ISIC_SD21_UNIT_LOCAL_CLIPPING_PROXY_PROTOCOL_V1.md`로 B=2 same-compute 구성, primary
`C=0.20`과 gate를 고정했다. `global_only_distinct_m4`는 batch 전체의 여덟 perturbation을
4+4로 나누고, `unit_local_true_bin_m4`는 각 bin의 두 perturbation을 두 환자에게 하나씩
배정한다. 두 경우 모두 batch 전체에서는 여덟 perturbation을 한 번씩 쓴다.

```powershell
..\base_gate\.venv\Scripts\python.exe .\run_isic_sd21_unit_local_clipping_proxy.py `
  --report-dir ..\_reports\pcm_isic_sd21_timestep_repl_01 `
  --report-dir ..\_reports\pcm_isic_sd21_timestep_repl_02 `
  --report-dir ..\_reports\pcm_isic_sd21_timestep_repl_03 `
  --report-dir ..\_reports\pcm_isic_sd21_timestep_repl_04 `
  --report-dir ..\_reports\pcm_isic_sd21_timestep_repl_05 `
  --output-dir ..\_reports\pcm_isic_sd21_unit_local_clipping_proxy_v1_001

..\base_gate\.venv\Scripts\python.exe .\audit_isic_sd21_unit_local_clipping_proxy.py `
  --report-dir ..\_reports\pcm_isic_sd21_unit_local_clipping_proxy_v1_001
```

- primary local/global postclip ratio `0.9094`
- hierarchical CI `[0.8446, 0.9506]`
- 5/5 banks, 78/80 patient-bank cells 방향 일치
- global/local mean clip rate `0.1948`/`0.1379`
- decision: `UNIT_LOCAL_CLIPPING_PROXY_PASSES_FOR_JOINT_BATCH_TEST`
- independent raw-Gram audit PASS; 전체 31 tests PASS

report는 `../_reports/pcm_isic_sd21_unit_local_clipping_proxy_v1_001/`이다. 이는 within-patient
marginal proxy다. cross-patient Gram이 없으므로 actual joint-batch update, DP training,
generation, privacy 또는 신규성 claim은 하지 않는다.

### Actual B=2 joint-batch locality

proxy 다음에는 새 `joint_01`--`joint_05` banks에서 full cross-patient Gram을 생성해 모든
120 patient pairs를 exact 비교했다.

```powershell
..\base_gate\.venv\Scripts\python.exe .\run_isic_sd21_joint_crossgram.py `
  --manifest ..\_data\derived\isic2020_v2_split_v1\k5_private.csv `
  --image-root ..\_data\raw\isic2020_k5_v1\images `
  --cohort-source-report ..\_reports\pcm_isic_sd21_timestep_repl_01 `
  --previous-bank-report ..\_reports\pcm_isic_sd21_timestep_repl_01 `
  --previous-bank-report ..\_reports\pcm_isic_sd21_timestep_repl_02 `
  --previous-bank-report ..\_reports\pcm_isic_sd21_timestep_repl_03 `
  --previous-bank-report ..\_reports\pcm_isic_sd21_timestep_repl_04 `
  --previous-bank-report ..\_reports\pcm_isic_sd21_timestep_repl_05 `
  --output-dir ..\_reports\pcm_isic_sd21_joint_crossgram_v1_001 `
  --resume

..\base_gate\.venv\Scripts\python.exe .\run_isic_sd21_joint_batch_locality.py `
  --crossgram-dir ..\_reports\pcm_isic_sd21_joint_crossgram_v1_001 `
  --output-dir ..\_reports\pcm_isic_sd21_joint_batch_locality_v1_001

..\base_gate\.venv\Scripts\python.exe .\audit_isic_sd21_joint_batch_locality.py `
  --report-dir ..\_reports\pcm_isic_sd21_joint_batch_locality_v1_001
```

- new gradients: 3,200, all finite
- primary joint local/global ratio: `1.0064606`
- five-bank bootstrap CI: `[1.0020765, 1.0117106]`
- banks below 1: `1/5`; pair wins: `191/600`; leave-one-patient-out below 1: `0/16`
- replicated marginal local/global ratio: `0.9429900`, `5/5` banks
- decision: `UNIT_LOCAL_JOINT_BATCH_INCREMENT_NOT_SUPPORTED`
- independent full-Gram recomputation audit: PASS
- full test suite: `37/37 PASS`

local은 patient marginal을 줄였지만 global arm의 더 큰 음의 cross-patient error covariance를
잃어 실제 batch update는 악화됐다. 따라서 이 단순 locality family를 방법 claim에서
중단한다. report는 `../_reports/pcm_isic_sd21_joint_batch_locality_v1_001/`, 상세 연구 기록은
`../../CVPR 주제 탐색/08_actual_B2_joint_batch_locality_결과_2026-09-03.md`다.

### Covariance-allocation post hoc discovery

actual local failure 뒤 개별 결과를 보기 전에
`ISIC_SD21_COVARIANCE_ALLOCATION_DISCOVERY_PROTOCOL_V1.md`로 35개 public rank-template
전수탐색과 leave-one-bank-out gate를 고정했다.

```powershell
..\base_gate\.venv\Scripts\python.exe `
  .\run_isic_sd21_covariance_allocation_discovery.py `
  --crossgram-dir ..\_reports\pcm_isic_sd21_joint_crossgram_v1_001 `
  --joint-report-dir ..\_reports\pcm_isic_sd21_joint_batch_locality_v1_001 `
  --output-dir ..\_reports\pcm_isic_sd21_covariance_allocation_discovery_v1_001

..\base_gate\.venv\Scripts\python.exe `
  .\audit_isic_sd21_covariance_allocation_discovery.py `
  --report-dir ..\_reports\pcm_isic_sd21_covariance_allocation_discovery_v1_001
```

- 모든 five held-out folds가 `rt_0123`을 선택
- held-out bank ratios: `0.9670`, `0.9485`, `0.9886`, `0.9361`, `0.9678`
- equal-bank ratio `0.961428`, CI `[0.944857, 0.976267]`
- 5/5 direction, 4/5 banks만 84/120 pair wins
- pair-specific hindsight oracle `0.941876`
- patient marginal `1.128109`, selected/global clip rate `0.3689/0.1614`
- decision: `COVARIANCE_ALLOCATION_BRANCH_NOT_SUPPORTED_BY_DISCOVERY`
- independent raw-Gram audit PASS; 전체 test suite `44/44 PASS`

사전 5% effect와 pair-support gate를 실패했으므로 fresh gradient confirmation은 실행하지
않는다. report는
`../_reports/pcm_isic_sd21_covariance_allocation_discovery_v1_001/`, 상세 기록은
`../../CVPR 주제 탐색/09_covariance_allocation_discovery_결과_2026-09-03.md`다.
