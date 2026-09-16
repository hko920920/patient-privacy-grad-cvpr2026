# CDI reference-set 검정의 정확한 범위와 현재 실행 준비

2026-09-15. 연구 2번의 누락 비교 준비다. 새 GPU·공격 fitting·실제 p-value 계산은 하지 않았다. 원형 코드 commit은 dcd62258b0b3fde05d52aaecfade3b5f4c09507a다.

핵심 결론: 기존 fit80/selection40로 환자당 한 점수를 사용하는 **분리된 control/test 집합의 Welch 검정**은 실행할 수 있다. 이는 환자군20명 대 reference20명의 점수 분포 비교다. 환자 한 명의 두 사진으로 membership을 확정하는 검정, 원형의 image-level5fold/1000subset 전체 재현, 또는 실제 patient-FPR 확인과는 다르다.

## 직접 확인한 원문·코드

| 근거 | 실제 내용 |
|---|---|
| [본문 PDF p5 §4](../pdfs/X01.pdf#page=5) / [추출문:245](../texts/X01.txt#L245) | 의심되는 공개 집합 P와 같은 분포의 미공개 reference U를 가정한다. Pctrl/Uctrl로 scorer를 학습하고 분리된 Ptest/Utest에 적용한다. 여기서 논문의 U는 reference 집합이며 우리 연구의 unseen-same-patient U와 다른 기호다. |
| [본문 p5 §4.2, p6 §4.3](../pdfs/X01.pdf#page=6) | LR 점수의 평균에 대해 단측 Welch 검정을 설명한다. P의 평균이 reference보다 높다는 방향이다. 논문이 logits라는 표현을 쓰지만 점수 함수 범위는[0,1]이며 실제 코드는 probability를 사용한다. |
| [공개 scorer:27](../code_sources/X01/src/attacks/scores_computation/cdi.py#L27) | predict_proba(data)[:,1]을 반환한다. t-test는 비선형 변환에 불변이 아니므로 raw decision_function/logit으로 바꿔서는 안 된다. |
| [실험 scorer:51–90](../code_sources/X01/experiments/cdi_perf.py#L51) | P/R에 같은 index의5fold를 적용한다(shuffle=True,random_state=0). 각fold의 train에만 StandardScaler를 fit하고, LogisticRegression(random_state=0,max_iter=1000,solver=liblinear,C 기본값1)을 학습한다. test probability를 합쳐 OOF 점수를 만든다. split 단위는 배열 행이며 환자 ID를 사용하지 않는다. |
| [라벨 생성:18–20](../code_sources/X01/src/attacks/utils.py#L18) | 입력으로 넘긴 suspect/member 집합을1, reference/nonmember 집합을0으로 붙인다. 함수가 원장을 확인하여 실제 membership을 알아내는 것은 아니다. |
| [공개 검정:79–97](../code_sources/X01/src/evaluation/evaluate.py#L79) | ttest_ind(...,equal_var=False)의 **기본 양측 p-value**를 반환한다. 원하는 방향인지 여부는 별도 bool이다. 본문의 단측 명세와 구별해야 한다. |
| [반복 실험:139–160](../code_sources/X01/experiments/cdi_perf.py#L139), [반복수:90](../code_sources/X01/experiments/utils.py#L90) | 크기별 부분집합을 매번 뽑고 scorer를 다시 fit한다. 반복수1000, main의 NumPy seed42. p값과 방향 bool을 별도 열로 보존한다. |
| [부록 PDF p13 §E](../pdfs/X01_supp.pdf#page=13) | 반복 p값의 산술평균은 여러 data-owner 사례의 성능 보고를 위한 요약이라고 설명한다. 독립된1000개 환자 관측 또는 한 사례의 통합 p값으로 자동 해석할 수 없다. 같은 페이지 §D.3은 member/reference 분포 불일치만으로 공격이 좋아 보일 수 있음을 경고한다. |

원형 OOF는 각 관측을 자신의 scorer fitting에서 제외한다. 그러나 서로 다른 OOF 점수는 겹치는 training folds를 공유하므로, 합친 점수 전체의 통계적 독립성이 그 사실만으로 증명되지는 않는다. 이는 코드에 대한 통계적 해석이며 저자의 주장과 구별한다. 현재의 fit80/selection40 분리는 새로운 test-scorer fitting을 하지 않아 이 공유 OOF 문제를 추가하지 않는다.

## 환자 설정에서 유지해야 하는 구분

1. **학습용·검정용 reference를 환자 단위로 분리한다.** 현재 fit80에는 target마다 참여40/비참여40명이 있고 selection40에는20/20명이 있다. 각 환자의 두 영상이 scorer train/test로 갈라지지 않는다.

2. **사진80장을 독립 표본80개로 세지 않는다.** selection 환자40명의 영상2장씩은 내부 상관이 있다. 고정 image LR의 두 probability를 먼저 평균한 환자 점수로 검정한다. 양측 집합의 표본 수는 각각20이다.

3. **우리 U의 image nonmembership은 patient nonmembership이 아니다.** 참여 환자가 가진 U 영상은 실제 학습 영상이 아니므로 원형의 exact-image membership 가설에서는 nonmember다. 환자 U 위험을 평가할 때 reference는 비참여 환자의 U 영상에서 만든 환자 점수여야 한다. 참여 환자의 다른 미학습 사진을 privacy-negative reference로 놓으면 연구 질문 자체가 달라진다.

4. **reference가 같은 분포라는 것은 별도 가정이다.** 같은 병원/데이터셋과 동일한 환자 수, 알려진 원장 라벨만으로 분포 일치가 증명되지는 않는다. A/B 환자군이나 E/U 영상 구성 차이가 평균 점수 차이를 만들 수 있다.

5. **한 환자에 대한 통계적 결론을 만들지 않는다.** 이 검정은 여러 환자군의 점수 평균 차이다. 단일 환자를 한 표본으로 평균내면 Welch의 집합 분산을 추정할 수 없고, 두 사진을 독립2표본처럼 취급하는 것도 정당화되지 않는다.

## 준비한 고정 실행

[사전 계약](cdi_reference_set_test_contract_20260915.json)은 cdi_image_mean_fixed 하나를 원형 실험의 fixed C1 image scorer와 가깝다는 이유로 미리 선택한다. 결과가 좋았던 방법을 고르는 선택이 아니다. Efit→Eeval, Efit→Ueval, Ufit→Eeval, Ufit→Ueval을 두 target에서 계산하므로 총8개 집합 검정이다.

새 fitting은 없다. 검산된 EU 분석의 저장 환자점수를 그대로 사용한다. 각 셀에서 참여20명 대 비참여 reference20명의 환자 probability 평균에 대해 다음을 모두 보존한다.

- 공개 evaluator의 변경하지 않은 asserts/get_p_value AST: 양측p + 방향 bool.
- 본문 단측 명세에 해당하는 SciPy alternative=greater.
- math.fsum 평균·불편분산에서 독립 재계산한 Welch t, Satterthwaite 자유도 및 scipy.stats.t.sf.
- 두 집합 환자 ID·원점수·표본수와 산술 차이. 분산0/비유한 검정은 판단 불가능 상태로 남긴다.

Runner는 run_cdi_reference_set_test.py다. 실행 전 protocol에 source/input/contract SHA를 기록하며, EU 독립 검산 PASS_EU_SAVED_PREDICTIONS_FIT_AND_STATISTICS가 있어야 진행한다. 실제 점수는 아직 읽지 않았다.

선택40명은 이미 반복 사용한 개발자료이고8개 검정은 서로 상관된 다중 비교다. p값을 confirmatory error control, 실측 patient-FPR 또는 인과적 participation 증거로 보고하지 않는다. threshold·반복 subset·p-value 평균·새 방법·표본수 확대·outcome-based stop rule은 없다. 이 준비는 빠졌던 reference-set 점수 검정을 분리해 수행하는 범위이며 2번 전체의 완료가 아니다.
