# ISIC SD 2.1 covariance-aware allocation discovery protocol v1

동결 시각: 2026-09-03. 이 문서는 actual B=2 unit-local primary 결과
`1.0064606 [1.0020765, 1.0117106]`을 확인한 **뒤**, 그러나 35개 개별 complementary
allocation template의 결과를 계산하거나 열람하기 **전**에 작성한 post hoc discovery
protocol이다.

## 지위와 목적

앞선 patient-unit-local rule은 patient-marginal MSE를 낮췄지만 global complementary
allocation의 더 큰 음의 cross-patient error covariance를 잃어 actual joint MSE가 증가했다.
이번 분석은 기존 full cross-Gram 안에 다음 단계로 갈 만한 **data-independent common
allocation headroom**이 있는지만 검사한다.

- 이 자료에서 고른 규칙은 confirmatory evidence가 아니다.
- 새 gradient를 계산하지 않는다.
- 유리한 template이 발견돼도 별도 새 perturbation banks와 가능하면 새 cohort에서 사전
  고정한 검정을 통과하기 전에는 방법 효능을 주장하지 않는다.
- 이번 discovery가 실패하면 현재 timestep-allocation method branch를 종료한다.

## 동결 source

- source directory:
  `code_working/_reports/pcm_isic_sd21_joint_crossgram_v1_001/`
- `crossgram_summary.json` SHA-256:
  `9208B089AD1C93EEB41630820AE54DCD9A4F37369885775CA366AC43AEA19463`
- actual joint source directory:
  `code_working/_reports/pcm_isic_sd21_joint_batch_locality_v1_001/`
- actual joint `summary.json` SHA-256:
  `2A06856BFC37CEAE2D2194DE350D01C45C310B57D0D6BF0ECBFAC0A150A37286`
- banks: `joint_01`--`joint_05`
- cohort: 같은 frozen 16 ISIC public-development patients
- all unordered patient pairs: `C(16,2)=120`
- bank당 640 x 640 full Gram, 16 patients x 5 records x 8 perturbations
- model dimension: 1,659,904
- primary clipping norm: `C=0.20`

source의 actual joint 결과, marginal/cross decomposition과 protocol은 이미 알려져 있으므로
이번 분석은 명시적으로 post hoc이다. 아직 알려지지 않은 것은 개별 template별 결과다.

## Rank-token 표현

각 bank는 four equal-width timestep bins마다 두 perturbations를 갖는다. 각 bin 안에서 실제
timestep을 오름차순 정렬하고 동률이면 perturbation index로 정렬해 다음 여덟 public random
tokens로 바꾼다.

```text
(bin0_low, bin0_high,
 bin1_low, bin1_high,
 bin2_low, bin2_high,
 bin3_low, bin3_high)
```

template은 여덟 rank tokens 중 네 개를 첫 patient slot에 주고 complement 네 개를 둘째
slot에 주는 partition이다. patient를 두 slots에 50:50 무작위 orientation하므로 template과
그 complement는 같은 정책이다. `C(8,4)/2=35`개 orientation-symmetric templates를 모두
사용한다. canonical representative는 선택 token tuple과 complement tuple 중
lexicographically 작은 tuple이다.

이 표현은 noise seed나 gradient 값이 아니라 sampled timestep의 공개 rank만 사용한다.
template 선택 과정에서도 held-out bank의 gradient 결과는 사용하지 않는다.

## 동일 계산량과 unbiased marginal

각 patient는 서로 다른 4/5 records를 균등 비복원 선택하고, 배정받은 네 perturbations를
선택 records에 모든 `4!` permutations로 균등 배정한다. fixed oriented subset당
`C(5,4) x 4! = 120` record states다.

각 complementary template의 orientation을 50:50으로 뒤집으면 모든 perturbation token은
각 patient에 확률 `1/2`로 들어간다. record choice와 mapping까지 평균하면 40 gradient cells의
expected weight가 각각 `1/40`이다. 따라서 모든 35 templates가 같은 pre-clipping patient
reference에 대해 unbiased이고 다음이 같다.

- B=2 patients
- patient당 Q=4 gradients
- batch당 여덟 perturbations 정확히 한 번 사용
- distinct-record marginal
- timestep/noise marginal
- clipping norm과 clipping 위치

## Template별 estimand

앞선 actual B=2 protocol과 같은 target을 사용한다. patient `u`의 40-gradient 평균을
`mu_u`라 하면 pair target은 다음이다.

```text
b_uv = [clip_C(mu_u) + clip_C(mu_v)] / 2
```

각 template에서 patient contribution을 먼저 clip하고 평균한 estimator와 `b_uv` 사이의
expected squared L2 error를 parameter dimension으로 나눈 joint MSE를 exact Gram moments로
계산한다. orientation 두 방향과 두 patients의 record states를 모두 적분한다.

bank `b`, template `k`의 값은 120 pair joint MSE의 산술평균이다. comparator는 기존
global-only와 같은 35 templates의 uniform mixture다.

```text
R_bk = mean_pair MSE(template k) / mean_pair MSE(global-uniform)
```

## 사전 고정 discovery analyses

### 1. Descriptive headroom bounds

- pair-specific oracle: bank와 pair마다 최저 MSE template 선택
- bank-shared hindsight oracle: bank마다 120 pairs 평균이 최저인 template 선택
- pooled fixed hindsight: 다섯 banks의 equal-bank geometric ratio가 최저인 한 template

이 셋은 held-out selection이 아니므로 효능 evidence로 쓰지 않는다. 특히 pair-specific
oracle은 private gradient/identity-dependent이며 실행 방법 후보가 아니다.

### 2. Leave-one-bank-out rank-template selection

각 held-out bank `h`마다 다음을 독립 수행한다.

1. 나머지 four training banks에서 template별 `R_bk`의 equal-bank geometric mean을 계산한다.
2. 값이 가장 작은 한 canonical rank template을 선택한다. exact tie는 template ID의
   lexicographic order로 푼다.
3. 선택된 template을 held-out bank의 120 pairs에 적용한다.

이렇게 얻은 다섯 held-out ratios만 primary discovery score에 사용한다. all-five pooled best,
held-out bank 자체에서 고른 best 또는 pair-specific best로 primary를 대체하지 않는다.

### 3. Nested bank-and-patient leaveout

특정 patient가 template 선택과 결과를 동시에 지배하는지 보기 위해 patient `u`마다:

1. 각 held-out bank에서 나머지 four banks의 `u`가 포함되지 않은 105 pairs로 template 선택
2. held-out bank의 같은 105 pairs에서 평가
3. five held-out ratios의 equal-bank geometric mean 계산

16개 nested leaveout ratios를 전부 보고한다.

### 4. Structure summaries

templates를 patient 한 slot의 per-bin token counts로 분류한다.

- `1111`: bin마다 하나, 기존 unit-local family, 8 templates
- `2110`: one full bin, one empty bin, two single bins, 24 templates
- `2200`: two full bins와 two empty bins, 3 templates

shape별 uniform mixture와 template별 marginal/within/cross contribution을 descriptive로
보고한다. 결과를 보고 shape를 새 primary로 바꾸지 않는다.

## Primary discovery gate

아래를 **모두** 만족할 때만
`RANK_TEMPLATE_HEADROOM_FOUND_FOR_FRESH_CONFIRMATION`으로 판정한다.

1. five leave-one-bank-out held-out ratios가 `5/5 < 1.00`
2. five held-out ratios의 equal-bank geometric mean `<= 0.95`
3. five held-out ratios를 복원추출하는 10,000-replicate bank bootstrap, seed
   `26090306`의 95% upper bound `< 1.00`
4. 각 held-out bank에서 selected template의 pair wins `>= 84/120`
5. nested bank-and-patient leaveout ratios가 `16/16 < 1.00`
6. held-out selected policies의 patient-state mean clip rate와 global-uniform mean clip rate가
   각각 `[0.05,0.95]`
7. source hashes, 35-template completeness, complement/orientation, exact expected cell weights,
   finite positive MSE, 기존 global/local primary 수치 재현 및 독립 감사가 모두 PASS

하나라도 실패하면 `COVARIANCE_ALLOCATION_BRANCH_NOT_SUPPORTED_BY_DISCOVERY`다. gate를
통과해도 이번 five-bank 자료에서 method가 확인된 것이 아니라 fresh confirmation을 실행할
최소 headroom만 있다는 뜻이다.

## Fresh-confirmation 조건

discovery gate가 PASS할 때만 all-five discovery banks로 한 final rank template을 선택해
결과와 hash를 고정한다. 그 뒤 최소 다섯 완전히 새로운 perturbation banks에서 다음을
사전 고정해 검정한다.

- selected fixed rank template 대 global-uniform
- 같은 cohort 반복과 별도 cohort 중 최소 하나, 가능하면 둘 다
- primary C, pair set, bank seeds와 gate를 결과 전에 동결
- discovery data로 threshold, template 또는 comparator를 다시 수정하지 않음

fresh confirmation이 PASS하기 전에는 X-ray DP trainer의 핵심 방법으로 넣지 않는다.

## 해석 경계

이 분석은 finite-bank, B=2, ISIC, SD 2.1 한 checkpoint/LoRA point, DP noise 이전의 clipped
gradient diagnostic이다. public calibration으로 template을 고를 수 있는 가능성만 검사한다.
optimizer convergence, B>2, variable record counts, Gaussian DP noise, privacy accounting,
X-ray transfer, generation/medical utility, 임상 타당성, 신규성 또는 CVPR 적합성을 입증하지
않는다. VDM/DDIM antithetic sampling, optimal experimental design, covariance reduction과
user-level DP batching에 대한 추가 원문 검색도 방법 claim 전에 별도로 필요하다.
