# Patient-unit-local clipping proxy 결과

- 기록일: 2026-09-03
- 선행 단계: 5-new-bank timestep replication PASS
- 당시 판정: **G2b 저비용 proxy PASS**
- 결정 코드: `UNIT_LOCAL_CLIPPING_PROXY_PASSES_FOR_JOINT_BATCH_TEST`
- 후속 상태: **actual joint-batch G2b FAIL로 이 proxy의 방법 승격 효력 종료**
- actual 결과: `08_actual_B2_joint_batch_locality_결과_2026-09-03.md`
- 최신 방법 판정: `09_covariance_allocation_discovery_결과_2026-09-03.md`

## 1. 한 문장 결론

> minibatch 전체에서 여덟 perturbation을 한 번씩 쓰는 조건을 같게 두더라도, timestep
> 균형을 batch 전체에서만 맞추는 대신 각 patient vector 안에서 먼저 맞춘 뒤 clipping하면
> `C=0.20`의 patient별 post-clipping finite-bank MSE가 약 9.1% 낮아지는 재현 신호가
> 확인됐지만, 서로 다른 환자의 cross-gradient까지 포함한 실제 batch update 검정 전에는
> locality 기여가 확정된 것이 아니다.

## 2. 무엇을 같은 조건에서 비교했는가

개념적 minibatch 크기는 `B=2`, 환자당 계산량은 `Q=4`다. 한 bank에는 네 timestep bin마다
두 개씩 총 여덟 perturbation이 있다. 두 arm 모두 batch 전체에서는 이 여덟 개를 정확히
한 번씩 사용하고, 각 환자는 서로 다른 4/5 records를 사용한다.

1. **Global-only balance:** 임의의 네 perturbation을 첫 환자에게 주고 나머지 네 개를 둘째
   환자에게 준다. 한 환자의 marginal은 `distinct_perturbation_m4`다.
2. **Patient-unit-local balance:** 각 bin의 두 perturbation을 두 환자에게 하나씩 상보 배정한다.
   각 환자는 네 bin에서 하나씩 받으며 marginal은 `true_bin_balanced_m4`다.

각 환자 vector는 네 gradient의 평균으로 만든 후 한 번 L2 clipping한다. reference는 해당
환자의 5 records x 8 perturbations, 총 40개 gradient 균등 평균을 동일한 `C`로 clipping한
벡터다. 두 marginal estimator 모두 40개 cell에 unbiased하며 exact state 수는 각각
8,400과 1,920이다.

## 3. 사전 고정 사항

새 distinct post-clipping 값을 계산하기 전에
`ISIC_SD21_UNIT_LOCAL_CLIPPING_PROXY_PROTOCOL_V1.md`에 다음을 고정했다.

- 결과 전에 고정됐던 `repl_01`--`repl_05`만 사용
- primary `C=0.20`; 기존 동결 grid의 나머지 값은 secondary
- 5/5 bank ratio `<1`, 5/5 bank에서 wins `>=10/16`
- 전체 equal-bank geometric ratio `<=0.95`
- bank와 patient hierarchical bootstrap 95% upper `<1`
- 두 arm의 전체 평균 clip rate가 각각 `[0.05,0.95]`

`C=0.20`은 기존 grid 중 clipping이 모두 켜지거나 사실상 꺼지지 않은 중간 지점이라
선택했다. 다만 같은 Gram 자료의 앞선 결과를 알고 선택했으므로 완전 독립 confirmatory
검정이라고 표현하지 않는다.

## 4. Primary 결과

5 banks x 16 patients의 `C=0.20` 결과다.

| bank | local/global postclip MSE | local/global preclip MSE | local wins | global clip rate | local clip rate |
|---|---:|---:|---:|---:|---:|
| repl_01 | 0.9467 | 0.9354 | 14/16 | 0.1965 | 0.1494 |
| repl_02 | 0.9380 | 0.9170 | 16/16 | 0.2235 | 0.1639 |
| repl_03 | 0.8028 | 0.7862 | 16/16 | 0.0963 | 0.0539 |
| repl_04 | 0.9241 | 0.9356 | 16/16 | 0.1860 | 0.1379 |
| repl_05 | 0.9442 | 0.9177 | 16/16 | 0.2719 | 0.1845 |

Aggregate:

- equal-bank geometric ratio: `0.9093935`
- hierarchical bootstrap 95% CI: `[0.8446018, 0.9505718]`
- bank direction: `5/5`
- bank별 wins gate: `5/5`
- patient-bank cell wins: `78/80`
- global-only 평균 clip rate: `0.1948289`
- unit-local 평균 clip rate: `0.1379362`
- 사전 고정 조건: 모두 PASS

즉 primary operating point에서 patient-unit-local marginal은 global-only marginal보다
post-clipping MSE가 약 `1 - 0.9094 = 9.06%` 낮았고, clipping되는 estimator state의 비율도
약 5.69 percentage points 낮았다.

## 5. Secondary C grid

| C | local/global ratio | hierarchical 95% CI | wins / 80 | global clip rate | local clip rate |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.9443 | [0.9155, 0.9630] | 75 | 0.9591 | 1.0000 |
| 0.10 | 0.9703 | [0.9459, 0.9870] | 57 | 0.8073 | 0.9560 |
| 0.20 | 0.9094 | [0.8446, 0.9506] | 78 | 0.1948 | 0.1379 |
| 0.50 | 0.8948 | [0.8272, 0.9407] | 78 | 0.0516 | 0.0501 |
| 1.00 | 0.8959 | [0.8284, 0.9418] | 79 | 0.0177 | 0.0177 |

모든 `C`에서 aggregate 방향은 유지됐지만 이를 다섯 번의 독립 primary test로 해석하지
않는다. 특히 `C=0.05`는 local states가 전부 clipping되고, `C=1.00`은 clipping이 거의 없어
locality의 비선형 효과를 분리하기 어렵다. 사전 선택한 `C=0.20`만 primary다.

## 6. 계산 및 감사 무결성

- 새 gradient를 생성하지 않고 앞서 계산한 3,200 finite gradients의 within-patient Gram을
  사용했다.
- 70개 global complementary allocation과 16개 local complementary allocation의 batch-wide
  perturbation 완전 사용 및 local bin balance를 검사했다.
- 모든 estimator의 row sum과 40-cell expected weight가 exact했다.
- 앞선 pre-clipping 결과 재현 최대 상대 오차: `2.34e-14`
- 독립 감사가 자체 weight enumeration과 별도 clipping 계산으로 400개 셀을 원 Gram부터
  재계산했다.
- 독립 감사 최대 cell 상대 오차: `6.09e-16`; 판정 재현 PASS
- 전체 `pcm_diagnostic` test suite: `31/31 PASS`

## 7. 이 결과가 말하는 것과 말하지 않는 것

### 말할 수 있는 것

- pre-clipping timestep-bin 신호가 중간 clipping point 뒤에도 소멸하지 않았다.
- 동일한 batch-wide perturbation 사용량 아래에서 각 patient 내부를 균형화하는 marginal이
  global-only marginal보다 안정적일 가능성이 5 banks에서 일관됐다.
- 따라서 현재 후보를 바로 폐기하지 않고 실제 joint-batch test로 진행할 근거가 생겼다.

### 아직 말할 수 없는 것

- 실제 batch-average update MSE가 줄었다고 말할 수 없다. 저장 Gram에는 서로 다른 patient의
  gradient 사이 cross block이 없어서 complementary allocation이 만드는 환자 간 covariance를
  계산하지 못했다.
- timestep stratification, multiple-record averaging, patient clipping 자체는 신규가 아니다.
- optimizer convergence, DP noise 아래 SNR, 생성 품질, 의료 utility 또는 privacy guarantee를
  검증하지 않았다.
- 같은 16 patients, SD 2.1 checkpoint, rank-8 LoRA 한 지점의 ISIC 결과다.
- 이 PASS만으로 독립 방법 논문이나 CVPR-ready 결과가 된 것은 아니다.

## 8. 다음 정확한 gate

다음은 proxy가 아니라 **실제 B=2 joint-batch locality test**다.

1. 결과 전에 16 patients의 8개 pair와 bank·seed·`C=0.20` gate를 고정한다.
2. 각 pair에서 두 환자의 40+40 gradients 사이 cross-Gram block을 계산한다.
3. global complementary allocation과 bin별 local complementary allocation에서 patient별
   vector를 먼저 clipping한 뒤 두 clipped vectors를 평균한다.
4. 두 환자의 clipped full-reference 평균에 대한 expected batch-update MSE를 비교한다.
5. 같은 compute, 같은 batch 전체 timestep multiset, 같은 patient/record marginal을 유지한다.
6. joint MSE 이득이 없으면 locality claim을 중단한다. 통과해도 variable-record rule·이론·
   X-ray·실제 patient-DP utility가 추가로 필요하다.

## 9. 산출물

- protocol: `code_working/pcm_diagnostic/ISIC_SD21_UNIT_LOCAL_CLIPPING_PROXY_PROTOCOL_V1.md`
- analysis: `code_working/pcm_diagnostic/run_isic_sd21_unit_local_clipping_proxy.py`
- independent audit: `code_working/pcm_diagnostic/audit_isic_sd21_unit_local_clipping_proxy.py`
- unit tests: `code_working/pcm_diagnostic/test_isic_sd21_unit_local_clipping_proxy.py`
- report: `code_working/_reports/pcm_isic_sd21_unit_local_clipping_proxy_v1_001/`

## 10. 후속 actual joint-batch 결과

이 문서의 proxy PASS는 보존하지만 이후 새 다섯 banks의 full cross-patient Gram을 사용한
actual B=2 test가 완료됐다. primary joint local/global ratio는 `1.0064606`, bank bootstrap
95% CI는 `[1.0020765, 1.0117106]`이고, 사전 판정은
`UNIT_LOCAL_JOINT_BATCH_INCREMENT_NOT_SUPPORTED`다.

patient marginal은 새 banks에서도 `0.9429900`으로 개선됐지만 global complementary arm의
더 큰 음의 cross-patient covariance가 실제 batch에서 이를 상쇄했다. 따라서 현재 방법
후보의 상태는 **VAE coverage 폐기, timestep balance 자체의 신규성 부정, 단순 patient-unit-
local clipping placement도 방법 후보에서 중단**이다. 이 문서의 “다음 gate” 표현은 역사적
기록이며 최신 판단에는 08 문서를 사용한다.
