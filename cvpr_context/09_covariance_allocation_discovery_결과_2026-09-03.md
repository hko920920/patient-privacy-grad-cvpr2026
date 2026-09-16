# Covariance-aware allocation discovery 결과

- 기록일: 2026-09-03
- 선행 결과: actual B=2 unit-local locality FAIL
- 분석 지위: **이미 관측한 full Gram을 사용한 post hoc discovery**
- 실행·독립 감사: **PASS**
- 사전 discovery 판정:
  **`COVARIANCE_ALLOCATION_BRANCH_NOT_SUPPORTED_BY_DISCOVERY`**
- 결정: **현재 B=2 timestep-allocation method branch 종료; fresh gradient confirmation 미진행**

## 1. 한 문장 결론

> 모든 환자와 bank에 공통으로 적용 가능한 low-timestep-half 대 high-timestep-half 상보
> 배분이 joint MSE를 `3.86%` 줄이는 일관된 신호는 보였지만, 사전 요구한 `5%`에 못 미쳤고
> unimplementable pair-specific oracle조차 `5.81%`가 한계여서 독립 방법으로 개발할 충분한
> headroom이 없었다.

이 결과를 “완전한 null”로 표현하지는 않는다. 방향은 다섯 held-out banks와 16개 nested
patient leaveouts에서 일관됐다. 그러나 효과 크기, 한 bank의 pair support, 환자별 clipping
악화와 직접 선행 충돌을 함께 보면 새 gradient banks와 X-ray DP training에 투자할 기준은
통과하지 못했다.

## 2. 결과 전에 고정한 discovery 범위

actual joint 결과는 이미 알려진 상태였지만 35개 개별 template 결과를 열기 전에
`ISIC_SD21_COVARIANCE_ALLOCATION_DISCOVERY_PROTOCOL_V1.md`를 작성하고 SHA-256
`45930814495B582FF859B65B8B732294FBCDED78488A594DAB0EE776D3C38445`로 고정했다.

각 four timestep bins 안의 두 perturbations를 timestep 오름차순으로 low/high rank token으로
바꾸었다. 여덟 tokens 중 네 개를 첫 patient slot에 주고 complement를 둘째 slot에 주되 두
patient의 slot orientation을 50:50으로 뒤집었다. 가능한 정책은 정확히
`C(8,4)/2=35`개다.

- `1111`: 각 bin에서 하나씩, 8 templates
- `2110`: one full, one empty, two single bins, 24 templates
- `2200`: two full, two empty bins, 3 templates

모든 template은 orientation randomization과 uniform distinct-record mapping에 의해 40개
patient gradient cells의 기대 가중치가 정확히 `1/40`이다. 따라서 B=2, patient당 Q=4,
batch당 여덟 perturbations, record·timestep marginal과 `C=0.20`을 동일하게 유지했다.

## 3. 과적합을 막은 평가

각 held-out bank마다 나머지 four banks의 120-pair 평균 ratio만 사용해 가장 좋은 rank
template을 하나 선택하고, 해당 template을 held-out bank에서 평가했다. held-out 결과나
patient identity로 template을 고르지 않았다.

| Held-out bank | 선택 template | 구조 | joint/global | pair wins |
|---|---|---|---:|---:|
| joint_01 | rt_0123 | 2200 | 0.966972 | 102/120 |
| joint_02 | rt_0123 | 2200 | 0.948496 | 113/120 |
| joint_03 | rt_0123 | 2200 | 0.988557 | 76/120 |
| joint_04 | rt_0123 | 2200 | 0.936119 | 111/120 |
| joint_05 | rt_0123 | 2200 | 0.967836 | 103/120 |

다섯 folds가 모두 같은 `rt_0123`을 선택했다. rank tokens `0,1,2,3`은 bin 0과 bin 1의 두
perturbations를 한 patient slot에, bin 2와 bin 3의 두 perturbations를 다른 slot에 주는
정책이다. orientation을 무작위 반전하므로 어느 환자가 low/high half를 받는지는 공정하다.

## 4. 사전 discovery gate 결과

| 조건 | 관측값 | 요구 | 판정 |
|---|---:|---:|---|
| held-out bank ratios < 1 | 5/5 | 5/5 | PASS |
| equal-bank held-out ratio | 0.961428 | <=0.95 | **FAIL** |
| bank bootstrap 95% CI | [0.944857, 0.976267] | upper <1 | PASS |
| bank별 pair wins >=84/120 | 4/5 | 5/5 | **FAIL** |
| nested bank+patient leaveout ratio <1 | 16/16 | 16/16 | PASS |
| selected/global clip rate | 0.36891 / 0.16136 | 둘 다 [.05,.95] | PASS |
| source·enumeration integrity | 8/8 | 8/8 | PASS |

두 조건이 실패했으므로 사전 판정은
`COVARIANCE_ALLOCATION_BRANCH_NOT_SUPPORTED_BY_DISCOVERY`다. `3.86%` 개선을 보고
threshold를 낮추거나 joint_03을 제외하지 않는다.

## 5. 실제로 무엇을 바꾼 규칙인가

다섯 banks의 산술평균으로 보면 다음과 같다.

| 성분 | global-uniform | rt_0123 | 변화 |
|---|---:|---:|---:|
| patient-marginal MSE | 8.4221e-9 | 9.4956e-9 | +12.75% |
| joint within contribution | 4.2110e-9 | 4.7478e-9 | +12.75% |
| cross-patient contribution | -4.2473e-10 | -1.1100e-9 | 더 큰 음의 상쇄 |
| joint MSE | 3.7863e-9 | 3.6378e-9 | 약 -3.92% |
| patient-state clip rate | 0.1614 | 0.3689 | +20.75%p |

`rt_0123`은 local balance를 회복한 규칙이 아니다. 각 환자에게 low 또는 high timestep
half만 몰아주어 patient estimator와 clipping을 더 불안정하게 만든 뒤, 두 patient errors가
batch에서 반대 방향으로 상쇄되도록 한 규칙이다. equal-bank marginal ratio는
`1.128109`로 약 12.8% 악화됐다.

이 negative covariance는 B=2에서 joint MSE를 낮췄지만 다음 이유로 method seed가 약하다.

1. effect가 사전 최소치 5%보다 작다.
2. joint_03에서는 ratio `0.9886`, 76/120 wins로 거의 null이다.
3. clip rate가 두 배 이상 올라 실제 optimizer trajectory와 다른 batch size에서 불리할 수 있다.
4. low-half/high-half batch segregation은 새로운 patient-local mechanism이라기보다 알려진
   global antithetic/stratified timestep 배분과 더 가까워졌다.
5. patient-specific 또는 clipping-specific 독립 기여였던 local 가설과 반대 방향이다.

## 6. Headroom upper bounds

| 분석 | equal-bank ratio | 의미 |
|---|---:|---|
| pair-specific hindsight oracle | 0.941876 | 각 bank·patient pair 결과를 보고 최선 선택; 실행 불가 |
| bank-shared hindsight oracle | 0.956060 | 각 bank 결과를 보고 공통 template 선택; held-out 아님 |
| pooled fixed hindsight best | 0.961428 | 전체 결과를 본 뒤 고른 rt_0123 |
| leave-one-bank-out rt_0123 | 0.961428 | primary discovery score |

가장 낙관적인 pair-specific oracle도 개선 한계가 약 `5.81%`다. 실제 공통규칙의 `3.86%`와
차이가 작아, 더 복잡한 선택기를 만들어도 이 35-template family 안에서 얻을 추가 이득이
매우 제한적이다. private pair gradient를 보고 oracle을 구현하는 것은 privacy·compute 측면에서
허용되지 않는다.

Shape 전체를 균등 혼합한 결과도 `1111=1.00646`, `2110=0.99918`, `2200=0.98930`이다.
좋은 결과는 일반적인 shape 효과가 아니라 세 `2200` partitions 중 low-half/high-half 하나에
집중된다.

## 7. 독립 감사와 테스트

분석 구현을 import하지 않는 별도 감사기가 frozen raw full Grams에서 다음을 다시 계산했다.

- 5 x 120 x 35 = 21,000 pair-template cells
- 5 x 16 x 35 = 2,800 patient-template cells
- 175 bank-template aggregates
- five leave-one-bank-out selections
- 80 nested bank-and-patient leaveout folds와 16 aggregates
- bootstrap, gate와 최종 decision

감사 최대 상대오차는 pair cells `2.29e-13`, patient cells `4.47e-16`, LOBO
`2.34e-16`이며 모든 checks가 PASS했다. 전체 `pcm_diagnostic` test suite는 기존 37개에
신규 7개를 더해 `44/44 PASS`다.

## 8. 연구 결정

### 종료

- unit-local, rank-template와 covariance-aware B=2 timestep allocation을 현재 논문의 핵심
  신규 방법으로 개발하는 가지
- 같은 five banks에서 template·C·threshold를 추가 탐색해 결론을 구제하는 작업
- `rt_0123`만 들고 새 GPU gradients 또는 X-ray DP training을 시작하는 작업

### 유지

- 반복 의료영상 환자를 privacy unit으로 보는 상위 patient-level DP diffusion 문제
- 확보한 NIH X-ray data, exact preprocessing, DP mechanism/accountant와 attack protocol
- image-DP checkpoint의 patient-level 의미와 native patient-DP 재학습 비교라는 audit 질문
- 이번 negative result와 joint covariance decomposition 자체

다음 방법 가설은 timestep 배분의 변형이 아니라 다른 mechanism class에서 관련연구를 다시
확인한 뒤 세워야 한다. 가장 현실적인 다음 결정은 `(A)` method novelty를 새로 찾기 전에
현재 인프라로 image-DP 대 patient-DP의 실제 utility/privacy mismatch를 측정하거나,
`(B)` medical multi-record DP에서 clipping/aggregation 자체의 미점유 문제를 새로 검색하는
것이다. 어느 쪽이든 이번 five-bank 결과를 positive confirmatory evidence로 재사용하지 않는다.

## 9. 산출물과 SHA-256

- protocol:
  `code_working/pcm_diagnostic/ISIC_SD21_COVARIANCE_ALLOCATION_DISCOVERY_PROTOCOL_V1.md`
  - `45930814495B582FF859B65B8B732294FBCDED78488A594DAB0EE776D3C38445`
- analysis:
  `code_working/pcm_diagnostic/run_isic_sd21_covariance_allocation_discovery.py`
  - `34113735B6987CF54E8FC9590BE9C0AD33DD07C2B598399B4E52AB3B7E4CE3E0`
- independent audit:
  `code_working/pcm_diagnostic/audit_isic_sd21_covariance_allocation_discovery.py`
  - `9D25E66D6E0A8953984DD547147D99EB9E858F4DF0C2E425DFC6264DD9EB0D9C`
- tests:
  `code_working/pcm_diagnostic/test_isic_sd21_covariance_allocation_discovery.py`
  - `D66AABE98CE39B6C6BE6DDA050BF10471B144C6BA44729AA24DAD62EBE1C8D56`
- report:
  `code_working/_reports/pcm_isic_sd21_covariance_allocation_discovery_v1_001/`
  - `summary.json`:
    `3801301320366053DDE0FC7AB6339EF918E0F26F4C6CD42F2E519E37620EE6E8`
  - `secondary_audit.json`:
    `636D7D45E696B88425C8539A729EC1A9A3729088ADABF9A80EBD14D16BC358BF`

## 10. 주장 경계

이번 분석은 이미 본 ISIC full Grams의 post hoc discovery다. `rt_0123`의 다섯-bank 방향
일치는 새 자료의 재현이 아니고, DP training·privacy·generation·medical utility·B>2·X-ray
일반화 또는 신규성을 입증하지 않는다. discovery gate가 실패했으므로 fresh confirmation을
실행할 근거로 사용하지 않는다.
