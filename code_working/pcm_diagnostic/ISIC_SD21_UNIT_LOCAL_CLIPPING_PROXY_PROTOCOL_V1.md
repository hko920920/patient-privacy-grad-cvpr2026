# ISIC SD 2.1 patient-unit-local clipping proxy protocol v1

사전 고정 시각: 2026-09-03. 이 문서는 아래 다섯 replication bank의
`distinct_perturbation_m4` post-clipping 결과를 계산하기 전에 분석 대상, 비교 구성,
primary clipping norm과 판정 기준을 고정한다.

## 목적

앞선 5-new-bank 검정은 한 환자의 네 gradient evaluation 안에서 네 timestep bin을
균형화하면, 단순히 서로 다른 네 perturbation을 쓰는 것보다 pre-clipping gradient
추정 MSE가 반복된다는 사실을 확인했다. 그러나 timestep stratification 자체는 기존
방법이므로, 다음 질문은 동일한 전역 timestep 사용량을 **minibatch 전체에서만**
맞추는 경우와 **각 patient privacy unit 내부에서** 맞춘 뒤 patient vector를 clipping하는
경우가 비선형 clipping 뒤에도 구별되는가이다.

이번 검정은 그 질문의 저비용 선별용 proxy다. 저장된 patient별 Gram matrix에는 서로
다른 patient 사이의 cross-Gram block이 없으므로 실제 minibatch 평균 update의 MSE나
cross-patient covariance는 계산하지 않는다.

## 고정 자료와 분석 단위

- 결과를 보지 않고 seed와 gate를 고정했던 `repl_01`--`repl_05`만 사용한다.
- 각 bank는 같은 16 patients, patient당 5 records, four bins x two perturbations를 갖는다.
- gradient model, image preprocessing, LoRA parameterization과 `Q=4`는 원 분석과 같다.
- 각 patient-bank의 저장된 40 x 40 LoRA-gradient Gram matrix만 사용한다.
- 새 gradient, optimizer update, DP noise, generator training은 수행하지 않는다.

## 동일 계산량의 두 B=2 구성

두 patient가 있는 개념적 minibatch와 bank당 여덟 perturbation을 생각한다. 각 patient는
서로 다른 4/5 records를 선택하고 각 record에 perturbation 하나를 배정하여 네 gradient를
평균한다. 두 arm 모두 minibatch 전체에서는 여덟 perturbation을 정확히 한 번씩 쓰므로
총 계산량과 전역 perturbation 사용량이 같다.

1. `global_only_distinct_m4`: 여덟 perturbation 중 임의의 네 개를 첫 patient에, 나머지를
   둘째 patient에 배정한다. 한 patient의 정확한 marginal estimator는
   `distinct_perturbation_m4`다.
2. `unit_local_true_bin_m4`: 각 bin의 두 perturbation을 두 patient에게 하나씩 상보적으로
   배정한다. 각 patient는 네 bin에서 하나씩 받으며, 정확한 marginal estimator는
   `true_bin_balanced_m4`다.

각 arm의 patient marginal은 exact enumeration한다. 이 상보적 B=2 구성은 comparator의
의미를 명확히 하기 위한 것으로, 이번 자료만으로 두 patient가 평균된 실제 batch update의
오차를 계산했다는 뜻은 아니다.

## estimand

각 estimator state에서 patient vector를 먼저 만든 뒤 L2 norm `C`로 한 번 clipping한다.
비교 reference는 같은 40개 finite-bank gradient의 균등 평균을 동일한 `C`로 clipping한
벡터다. 각 patient-bank-cell의 metric은 parameter당 expected squared error다.

primary ratio는 `C=0.20`에서

```text
postclip MSE(unit_local_true_bin_m4)
------------------------------------------------
postclip MSE(global_only_distinct_m4)
```

이다. `C=0.20`은 기존에 고정된 `(0.05, 0.10, 0.20, 0.50, 1.00)` grid 가운데 원 분석에서
clipping이 전부 켜지거나 사실상 꺼지지 않은 중간 operating point였기 때문에 선택했다.
이는 새 distinct comparator의 post-clipping 값을 보기 전 선택이지만, 같은 Gram 자료에서
고른 것이므로 완전한 독립 confirmatory test로 과장하지 않는다. 나머지 `C`는 secondary다.

## 사전 고정 gate

실행 무결성이 PASS이고 아래 조건을 모두 만족할 때만
`UNIT_LOCAL_CLIPPING_PROXY_PASSES_FOR_JOINT_BATCH_TEST`로 판정한다.

1. 다섯 bank 모두 bank-level geometric primary ratio `< 1.00`
2. 다섯 bank 모두 unit-local wins `>= 10/16`
3. 5 x 16 cell의 equal-bank geometric primary ratio `<= 0.95`
4. bank와 patient를 각각 복원추출하는 10,000-replicate hierarchical bootstrap의
   95% upper bound `< 1.00`
5. primary `C=0.20`에서 두 arm 각각의 전체 평균 clip rate가 `[0.05, 0.95]`

하나라도 실패하면 `UNIT_LOCAL_CLIPPING_INCREMENT_NOT_SUPPORTED`로 판정한다. 불리한
bank를 제외하거나 primary `C`를 바꾸어 gate를 다시 계산하지 않는다.

## 해석 경계와 다음 단계

PASS는 저장된 실제 SD 2.1 gradient에서 unit-local 배치가 global-only marginal보다
patient별 post-clipping finite-bank MSE를 줄였다는 선별 신호만 뜻한다. 다음을 뜻하지 않는다.

- actual batch-average update 또는 cross-patient covariance 개선
- DP-SGD training utility, privacy guarantee 또는 더 좋은 생성 품질
- 의료적·임상적 타당성
- timestep stratification, multiple-record averaging 또는 patient clipping의 신규성
- 독립 논문 완성 또는 CVPR 적합성 확정

PASS 뒤의 다음 gate는 cross-patient Gram block을 저장하는 실제 B=2 joint-batch 실험이며,
그 뒤에 variable-record unbiased rule, clipping analysis, X-ray 확장과 patient-DP training을
검토한다. FAIL이면 현재 unit-local increment를 핵심 방법 claim으로 진행하지 않는다.
