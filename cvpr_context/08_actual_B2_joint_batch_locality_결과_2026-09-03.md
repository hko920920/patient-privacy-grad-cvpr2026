# Actual B=2 joint-batch locality 결과

- 기록일: 2026-09-03
- 데이터·모델: ISIC public-development 16 patients, Stable Diffusion 2.1 rank-8 LoRA
- 실행 판정: **PASS**
- 효능 판정: **`UNIT_LOCAL_JOINT_BATCH_INCREMENT_NOT_SUPPORTED`**
- 현재 방법 결정: **단순 patient-unit-local timestep 배분을 핵심 방법 claim에서 중단**

## 1. 한 문장 결론

> 환자별로 따로 보면 timestep-bin local balance가 clipping 뒤 오차를 줄였지만, 실제 두
> 환자를 함께 평균하면 global complementary 배분의 환자 간 오차 상쇄가 더 커서 최종
> update 오차가 오히려 `0.646%` 증가했다.

따라서 앞선 patient-marginal proxy의 PASS를 actual batch-update 개선으로 확장할 수 없다.
이는 연구 프로그램 전체나 patient-level DP 의료영상 주제의 실패가 아니라, 현재의 **단순
unit-local 배분 규칙**을 독립 방법으로 승격하지 않는다는 판정이다.

## 2. 결과 전에 고정한 비교

새 결과를 보기 전에 `ISIC_SD21_JOINT_BATCH_LOCALITY_PROTOCOL_V1.md`에 다음을 고정했다.

- 앞선 cohort와 같은 16 patients, 가능한 unordered patient pairs `C(16,2)=120` 전부 사용
- 기존 bank를 재사용하지 않은 `joint_01`--`joint_05` 다섯 perturbation banks
- patient당 5 records x 8 perturbations = 40 gradients
- 총 새 gradients: `5 x 16 x 40 = 3,200`
- bank당 640 x 640 full Gram을 계산해 모든 cross-patient blocks 포함
- raw gradient vectors는 저장하지 않음
- 두 arm 모두 patient당 서로 다른 4/5 records와 `Q=4`, batch 전체에서 같은 여덟
  perturbations를 정확히 한 번 사용
- global-only: 여덟 perturbations를 임의의 상보적 4+4로 배정
- unit-local: 각 timestep bin의 두 perturbations를 두 patients에게 1+1로 배정
- patient vector를 먼저 `C=0.20`으로 clipping한 뒤 두 patients를 평균
- target: 두 patient의 clipped 40-gradient full references 평균
- 각 bank의 120 pair MSE를 먼저 산술평균한 ratio와 다섯 bank의 equal-bank geometric mean

patient pair당 exact state 수는 global `1,008,000`, local `230,400`이다. 직접 모든 vector를
저장하는 대신 full Gram의 exact first/second moments로 같은 expectation을 계산했다.

## 3. Primary 결과

| 지표 | 관측값 | 사전 조건 | 판정 |
|---|---:|---:|---|
| equal-bank joint local/global ratio | `1.0064606` | `<=0.95` | FAIL |
| bank-bootstrap 95% CI | `[1.0020765, 1.0117106]` | upper `<1` | FAIL |
| ratio가 1 미만인 banks | `1/5` | `5/5` | FAIL |
| bank당 84/120 pair wins 충족 | `0/5` | `5/5` | FAIL |
| 전체 pair wins | `191/600` | 보조 기록 | 불리 |
| leave-one-patient-out ratio < 1 | `0/16` | `16/16` | FAIL |
| 새-bank marginal local/global ratio | `0.9429900` | `<=0.95`, 5/5 | PASS |
| global/local 평균 clip rate | `0.16136 / 0.11259` | 둘 다 `[.05,.95]` | PASS |

Bank별 actual joint ratio는 `0.999862`, `1.016319`, `1.002915`, `1.008449`,
`1.004838`이다. `joint_01`만 1보다 `0.014%` 낮았고 나머지 네 banks에서는 local이 더
나빴다. 어느 한 patient가 결과를 만든 것도 아니다. 각 patient를 하나씩 제외한 16개
ratios가 모두 `1.0051`--`1.0076`이었다.

독립 감사가 원 full-Gram에서 primary 600 pair cells와 80 marginal cells를 별도 구현으로
재계산했다. pair metric 최대 상대오차는 `2.50e-14`, marginal은 `4.51e-16`이며 같은 실패
판정을 재현했다.

## 4. Proxy와 actual 결과가 달라진 이유

두 clipped patient estimator의 오차를 `e_u`, `e_v`라 하면 B=2 update MSE는 다음 두 부분을
모두 포함한다.

```text
E ||(e_u + e_v)/2||^2
= 1/4 [E||e_u||^2 + E||e_v||^2]
  + 1/2 E[e_u^T e_v]
```

다섯 banks의 산술평균 분해는 다음과 같다.

| 평균 성분 | global-only | unit-local | local의 변화 |
|---|---:|---:|---:|
| patient-marginal MSE | `8.4221e-9` | `7.9439e-9` | `-5.68%` |
| joint 안의 within contribution | `4.2110e-9` | `3.9720e-9` | `-2.3908e-10` |
| cross-patient contribution | `-4.2473e-10` | `-1.6055e-10` | `+2.6418e-10` |
| 최종 joint MSE | `3.7863e-9` | `3.8114e-9` | `+2.5099e-11` |

local balance는 각 환자의 marginal variance를 줄였다. 그러나 global complementary
allocation은 두 환자의 오차에 더 큰 음의 cross term, 즉 batch에서 상쇄되는 오차를 만들었다.
local의 marginal 절감량보다 잃어버린 cross-patient 상쇄량이 조금 더 커 최종 부호가
뒤집혔다. 앞선 proxy에는 이 cross block이 없었기 때문에 이 현상을 측정할 수 없었다.

이 결과로 얻은 일반적 설계 교훈은 **privacy unit별 variance만 최소화해서는 batch update
variance가 최소화되지 않는다**는 것이다. 다음 후보가 있다면 목적함수에 unit marginal과
cross-unit covariance를 함께 넣어야 한다. 단, gradient 결과에 의존해 private batch 배정을
고르는 규칙은 privacy·sensitivity·추가 계산 문제를 만들 수 있으므로 곧바로 방법으로
채택하지 않는다.

## 5. Secondary clipping grid의 해석

| C | joint local/global | bank-bootstrap 95% CI | banks < 1 | global/local clip rate |
|---:|---:|---:|---:|---:|
| 0.05 | 0.9779 | [0.9679, 0.9861] | 5/5 | 0.9785 / 1.0000 |
| 0.10 | 0.9896 | [0.9799, 0.9987] | 3/5 | 0.8032 / 0.9572 |
| 0.20 | 1.0065 | [1.0021, 1.0117] | 1/5 | 0.1614 / 0.1126 |
| 0.50 | 0.9985 | [0.9961, 1.0017] | 4/5 | 0.0332 / 0.0331 |
| 1.00 | 0.9992 | [0.9981, 1.0003] | 4/5 | 0.0163 / 0.0163 |

`C=0.05`와 `0.10`의 유리한 수치를 보고 primary를 바꾸지 않는다. `C=0.05`는 두 arm이
거의 또는 전부 clipping되고 `C=0.10`은 local clip rate가 사전 nondegenerate 범위의 상한을
벗어난다. `C=0.50`과 `1.00`은 clipping이 거의 없다. 이 grid는 기전 단서일 뿐 current
locality claim을 구제하는 confirmatory evidence가 아니다.

## 6. 실행 무결성과 수치 허용오차 기록

- 다섯 banks의 3,200 gradients는 모두 finite였고 full Gram은 finite·symmetric·PSD
  tolerance를 통과했다.
- 첫 `joint_01` 검증은 outcome analysis 전에 지나치게 엄격한 float32 reduction-order
  tolerance 때문에 중단됐다.
  - direct-dot 최대 상대오차 `1.8031e-6`, 최초 한계 `1e-6`
  - stored norm과 Gram diagonal 최대 상대오차 `5.9442e-5`, 최초 한계 `1e-5`
  - Gram 최소 eigenvalue는 양수 `1.1096e-7`
- 결과를 계산하거나 보지 않은 상태에서 허용오차를 direct-dot `1e-5`, diagonal `1e-4`로
  수정했다. raw artifact hash는 유지했고 `joint_01_crossgram.json`의
  `validation_history`에 최초 실패와 수정 이유를 보존했다.
- 수정 한계에서도 가장 큰 direct-dot 오차는 `joint_05`의 `7.2233e-6`였으며 다섯 banks가
  모두 통과했다.
- 분석 독립 감사 PASS, 전체 `pcm_diagnostic` unit tests `37/37 PASS`다.

이는 계산 검증의 PASS이지 효능 PASS가 아니다. 효능은 사전 gate에 따라 FAIL이다.

## 7. 연구 방향에 미치는 결정

### 중단하는 것

- `unit-local_true_bin_m4`를 global batch balance보다 우월한 핵심 신규 방법으로 주장
- patient-marginal proxy `0.9094`를 실제 batch-update 이득으로 표현
- 불리한 patient, pair, bank를 빼거나 secondary `C`로 primary를 교체해 같은 가설을 재판정
- 이 규칙을 곧바로 NIH X-ray DP training에 넣어 대규모 계산을 수행

### 남기는 것

- 의료영상의 반복-record patient-level DP라는 상위 연구 문제
- 여러 record를 쓰는 patient contribution과 image-DP 대 native patient-DP의 비교·감사
- 실제 SD 2.1에서 재현된 timestep-bin marginal variance 진단 결과 자체
- batch objective가 marginal variance와 cross-unit covariance를 함께 고려해야 한다는
  새로운 분석 질문

### 다음 후보의 승격 조건

다음 규칙을 탐색한다면 먼저 기존 full Gram만 사용하는 **post hoc discovery 분석**으로
joint variance를 줄일 수 있는 고정·공개정보 기반 allocation family가 실제로 존재하는지
확인한다. 발견된 규칙은 이 다섯 banks에서 효능을 확정하지 않고, 별도 새 banks·새 cohort와
사전 고정 gate에서 다시 검정해야 한다. privacy-dependent gradient 선택 없이 구현 가능하고,
선행 batch antithetic·stratified sampling과 구별되며, X-ray와 실제 DP utility까지 통과할 때만
독립 방법 후보로 복귀한다.

## 8. 산출물과 SHA-256

- protocol: `code_working/pcm_diagnostic/ISIC_SD21_JOINT_BATCH_LOCALITY_PROTOCOL_V1.md`
  - `A688FD4A7F893192B013ED12980876C66B100390C06E7E537D9840286BF9A595`
- bank specification: `code_working/pcm_diagnostic/ISIC_SD21_JOINT_BATCH_BANKS_V1.csv`
  - `ECF583521CEBC760E5FF4BCA2AB20B5F16FA0ABC4B89436448EF1560089D981A`
- cross-Gram generator:
  `code_working/pcm_diagnostic/run_isic_sd21_joint_crossgram.py`
  - `6B3D429F01D634B7DF746A8258ECA0CCC46EDA338AD91A1D664BE33F5DAE1E6E`
- exact analysis: `code_working/pcm_diagnostic/run_isic_sd21_joint_batch_locality.py`
  - `3F392A2082532964368EA6E12DAEEE651A611B90F77167B45FA78E39CD3E7CE5`
- independent audit: `code_working/pcm_diagnostic/audit_isic_sd21_joint_batch_locality.py`
  - `C80CBA813C7BD335CEE10DB8F1C922CA9C943649AB88DB4AEC7CA022CB6E5BB3`
- tests: `code_working/pcm_diagnostic/test_isic_sd21_joint_batch_locality.py`
  - `15B144BA8DD865A6BBD330C0BC7F1C9379230EEABD69CB0EF44B55F3FCA3B360`
- cross-Gram report:
  `code_working/_reports/pcm_isic_sd21_joint_crossgram_v1_001/`
  - `crossgram_summary.json`:
    `9208B089AD1C93EEB41630820AE54DCD9A4F37369885775CA366AC43AEA19463`
- analysis report:
  `code_working/_reports/pcm_isic_sd21_joint_batch_locality_v1_001/`
  - `summary.json`:
    `2A06856BFC37CEAE2D2194DE350D01C45C310B57D0D6BF0ECBFAC0A150A37286`
  - `secondary_audit.json`:
    `43E1BABAD30DC91B2CEC0D92DC25666BEC465CD4156BE6EDC01F1F151E798137`

## 9. 주장 경계

이번 결과는 frozen 16-patient ISIC cohort, 다섯 finite perturbation banks, SD 2.1 한
checkpoint와 LoRA parameter point에서 DP noise 이전의 B=2 gradient diagnostic이다. optimizer
training, Gaussian DP noise, accountant, 생성 품질, 의료 utility, 임상 타당성, 다른 batch size,
X-ray 일반화, 방법 신규성 또는 CVPR 적합성을 입증하지 않는다.

## 10. 후속 covariance-allocation discovery

이 문서가 제안한 cheap post hoc headroom 분석을 결과 전에 별도 protocol로 고정해 수행했다.
35개 공정한 complementary rank templates를 전수 조사하고 leave-one-bank-out으로 선택한 결과,
다섯 folds가 모두 low-two-bins 대 high-two-bins 분할 `rt_0123`을 골랐다. held-out
equal-bank ratio는 `0.961428`, bank CI는 `[0.944857, 0.976267]`로 5/5 banks에서 방향은
유리했다.

그러나 사전 최소 ratio `<=0.95`에 못 미쳤고 joint_03의 pair wins가 76/120이어서
5/5 pair-support gate도 실패했다. pair-specific hindsight oracle조차 `0.941876`이므로 이
35-template family의 절대 headroom도 작다. 판정은
`COVARIANCE_ALLOCATION_BRANCH_NOT_SUPPORTED_BY_DISCOVERY`이며 fresh bank 검정으로 진행하지
않는다. 최신 authority는 `09_covariance_allocation_discovery_결과_2026-09-03.md`다.
