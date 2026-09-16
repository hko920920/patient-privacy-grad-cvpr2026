# ISIC SD 2.1 actual B=2 joint-batch locality protocol v1

사전 고정 시각: 2026-09-03. 이 문서는 `joint_01`--`joint_05`의 새 gradient를 계산하거나
cross-patient 결과를 보기 전에 bank, cohort, estimand, exact state space와 판정 기준을
고정한다.

## 목적

앞선 patient-marginal clipping proxy는 각 환자 내부의 timestep-bin 균형이 clipping 뒤에도
유리하다는 신호를 보였지만, 저장 Gram에 서로 다른 환자 사이 cross block이 없어 두 clipped
patient vectors를 평균한 실제 B=2 update 오차를 계산하지 못했다. 이번 실험은 새 perturbation
banks에서 cross-patient Gram을 직접 계산하여 이 빈칸을 닫는다.

## 동결 cohort와 model

- 앞선 confirmatory와 동일한 salted-hash 16 ISIC public-development patients
- patient당 정확히 5 images와 5 distinct lesions
- 가능한 unordered patient pairs `C(16,2)=120` 전부 사용; 유리한 pair 선택 금지
- SD 2.1 revision `0094d483a120f3f33dafbd187ea4aa60d10de75c`, 256 pixels
- frozen VAE mode, rank-8 attention LoRA, 1,659,904 trainable parameters
- patient당 5 records x 8 perturbations = 40 gradients, bank당 640 gradients
- optimizer update, DP noise와 generator training 없음

## 새 perturbation banks

기존 `original_confirmatory`와 `repl_01`--`repl_05`를 재사용하지 않는다. 결과 전에 다음
tags를 고정한다.

1. `joint_01`
2. `joint_02`
3. `joint_03`
4. `joint_04`
5. `joint_05`

각 bank는 four equal-width timestep bins마다 두 timestep/noise perturbations를 갖는다.
정확한 timestep과 noise seeds는 `ISIC_SD21_JOINT_BATCH_BANKS_V1.csv`, SHA-256
`ECF583521CEBC760E5FF4BCA2AB20B5F16FA0ABC4B89436448EF1560089D981A`로 동결한다. 40개 noise
seeds는 서로 달라야 하고 이전 replication seeds와 겹치면 실행 FAIL이다.

총 새 gradient 수는 `5 banks x 16 patients x 5 records x 8 perturbations = 3,200`이다.
각 bank에서 patient 순으로 쌓은 640 x 640 full Gram을 계산하되 raw gradients는 저장하지
않는다. full Gram의 off-diagonal patient blocks가 이번 신규 자료다.

## B=2 same-compute arms

두 patient 모두 서로 다른 4/5 records를 선택하고 record와 perturbation을 무작위 일대일
배정하여 네 gradients를 평균한 뒤, 각 patient vector를 L2 norm `C`로 한 번 clipping한다.
마지막에 두 clipped patient vectors를 평균한다.

1. `global_only_distinct_m4`
   - 여덟 perturbations 중 네 개를 첫 patient에 균등 배정하고 둘째 patient에는 complement를
     배정한다.
   - batch 전체에서는 여덟 perturbations를 정확히 한 번 사용하지만 patient별 bin balance는
     보장하지 않는다.
2. `unit_local_true_bin_m4`
   - 각 bin의 두 perturbations를 두 patients에게 하나씩 상보 배정한다.
   - batch 전체 사용량은 같고 각 patient도 네 bins에서 하나씩 받는다.

고정 subset에서 한 patient의 record-choice/mapping states는 `C(5,4) x 4! = 120`이다.
따라서 한 patient pair의 exact joint states는 global `C(8,4) x 120^2 = 1,008,000`, local
`2^4 x 120^2 = 230,400`이다. record randomization은 두 patients 사이 조건부 독립이다.
모든 states를 직접 펼치는 대신 각 subset의 정확한 first/second moments와 cross-Gram을
사용해 동일 expectation을 계산한다.

## target과 primary estimand

patient `u`의 40-gradient 균등 평균을 `mu_u`라 하고 `clip_C(mu_u)`를 이상적인 finite-bank
patient contribution으로 둔다. pair `(u,v)`의 target은

```text
b_uv = [clip_C(mu_u) + clip_C(mu_v)] / 2
```

이다. 각 arm에서 patient contributions를 먼저 clip한 후 평균한 estimator와 `b_uv` 사이의
expected squared L2 error를 parameter dimension으로 나눈 값을 joint MSE로 쓴다.

primary clipping norm은 앞서 고정한 nondegenerate point `C=0.20`을 그대로 유지한다.
bank `b`의 primary ratio는 120 pairs의 MSE를 각각 산술평균한 뒤 계산한다.

```text
R_b = mean_pair joint_MSE(unit-local) / mean_pair joint_MSE(global-only)
```

다섯 `R_b`의 geometric mean이 aggregate primary다. 이는 pair별 ratio의 geometric mean이
아니며, frozen 16-patient finite cohort에서 무작위 B=2 batch의 expected squared error를 직접
비교하기 위한 ratio-of-means다.

## 사전 고정 gate

실행 무결성과 독립 감사가 PASS하고 아래를 모두 만족할 때만
`JOINT_BATCH_LOCALITY_SIGNAL_CONFIRMED_FOR_XRAY_METHOD_GATE`로 판정한다.

1. 새 다섯 banks 모두 `R_b < 1.00`
2. 다섯 bank ratios의 equal-bank geometric mean `<= 0.95`
3. five banks를 복원추출하는 10,000-replicate bank bootstrap, seed `26090305`의
   95% upper bound `< 1.00`
4. 각 bank에서 unit-local pair wins `>= 84/120`
5. 한 patient를 제외한 뒤 나머지 15 patients의 105 pairs와 다섯 banks를 합친
   leave-one-patient-out ratio가 `16/16` 모두 `< 1.00`
6. 새 banks의 within-patient post-clipping marginal도 5/5 bank ratios `<1`이고
   equal-bank ratio `<=0.95`
7. primary에서 두 arms의 전체 평균 patient-state clip rate가 각각 `[0.05,0.95]`

한 조건이라도 실패하면 `UNIT_LOCAL_JOINT_BATCH_INCREMENT_NOT_SUPPORTED`로 판정한다.
불리한 patient, pair 또는 bank를 제외하거나 primary `C`를 바꾸어 재판정하지 않는다.
`C=(0.05,0.10,0.50,1.00)`은 secondary sensitivity이며 primary gate를 대체하지 않는다.

## uncertainty와 해석 경계

120 pairs는 frozen 16-patient cohort의 완전 열거이며 서로 독립인 120 samples가 아니다.
따라서 pair-level iid confidence interval은 사용하지 않는다. primary interval은 새로 고정한
다섯 perturbation banks의 변동만 나타내며, leave-one-patient-out은 특정 환자 지배 여부를
검사한다. five-of-five bank direction의 median-null one-sided sign probability `1/32`를 함께
보고하되 전체 연구의 통계적 확증으로 과장하지 않는다.

PASS가 뜻하는 것은 새 실제 LoRA gradients에서 same-compute global-only balance보다
patient-unit-local balance의 clipped B=2 finite-bank update error가 낮다는 진단 신호다.
다음은 여전히 범위 밖이다.

- Gaussian noise가 포함된 patient-DP SGD training utility와 convergence
- formal accountant/sampler proof 또는 새로운 privacy guarantee
- X-ray, 비의료 multi-record domain과 다른 checkpoint로의 일반화
- timestep stratification, patient averaging 또는 clipping 자체의 신규성
- 생성 품질, 의료 utility, 임상 타당성 또는 CVPR-ready claim

PASS 뒤에는 variable-record-count rule과 clipping analysis를 정식화하고 exact-source X-ray
gradient gate로 진행한다. FAIL이면 unit-locality를 핵심 방법 claim으로 진행하지 않는다.
