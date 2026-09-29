# 2-4B 공개 입력 prototype·대조 결과

2026-09-29. 상위2 / 연구 단계2. 사용자 승인 ‘예상60~90분 오케이 진행해’에 따른 공개 구현 묶음이다.

## 결론

**구현·보호 상계·source 보존 검산 통과. 그러나 사전 고정한 공개 대조에서 새 공간의 추가 실용 이득은 미확보.**

같은 cap에서 새 방식의 retained-target 기대 MSE는 **4.652877**, 전체 방식은 **4.652889**로 거의 같았다. 동일 rank 공개 PCA의 **3.134940**가 새 방식보다 **32.62% 낮았다**. 각 방식에 같은 규칙으로 자기 cap을 허용해도 PCA가 **34.77% 낮았다**.

**따라서 현 형태의 후보를 Q 추가 release나 V 효용 실험으로 자동 확대하지 않는다.** 이번 수치는 공개 P에서의 목표 복원 오차이며 AUROC가 아니다. P의 양성 환자가6명인 조건의 결과로, 실제 Q의 우열·새 후보의 모든 가능성·연구 전체의 실패를 판정하지 않는다. 기존 HR의 실제 개발 양성 결과는 그대로 보존한다.

## 완료한 작업과 접근 범위

- 고정 public128, 공개 P672환자/813영상/675환자-class 행. 음성 class 환자669명·양성6명·두 class에 등장하는 환자3명.
- 환자/class 방문 평균, 공개 whitening, source 보존 nullspace, receiver 도달 공간, patient joint query, 공유 noisy count 복원, 라벨 계산과 양의 공통 배율까지 구현.
- 전체/호환공간/PCA **세 방식**을 같은 cap 및 같은 공개 규칙의 자기 cap으로 비교. 전체 방식의 두 cap 행은 동일하므로 독립 실험처럼 세지 않는다.
- 하나의 고정 공개 모의 잡음512회. 6설정×512 라벨 벡터는 CPU 수치 검산용이며 실제 합성영상 bank3072개나 실제 DP release512개가 아니다.
- 기존 DINO Label Solve와 Gaussian calibration 함수를 재사용했다. source 초기 라벨 y₀도 **P만으로 한 번 계산**했다. 기존 사적 DP 요약·라벨 값은 읽지 않았다.
- **Q/V/DenseNet/Expert/Reserved 접근0, 새 private release0, encoder forward0, pixel 합성0, 임상 효용 평가0.**
- 기록·구현·분석·독립 검산까지 실측 **20.51분**. 핵심 수치 계산 0.532초, 독립 검산 1.059초. 캐시와 폐쇄형 선형계산을 사용해60~90분 예상보다 일찍 마쳤다. 이 시간을 향후 Q/encoder 추출·합성학습 비용으로 복사하지 않는다.

## 결과 전에 고정한 조건

contract.json은 방법별 결과 계산 전에 기록했으며 SHA256은 4872667e175d9f4173ceca544e2876e4ea8e2054c0ce72100fe0055b58c54484다.

1. source DINO64, 추가 construction model ResNet18 128, 고정 공개 영상128, ridge0.1.
2. source 보존64차원, receiver 도달64차원. PCA도64차원이며 동일한 공개 모델·특징·H 변환·최종 라벨 학습을 허용.
3. 공개 task curvature H는 class별 환자 균등 가중. PCA covariance와 공통 공개 중심은 환자별 균등이며 한 환자의 질량을 그 환자의 관측 class에 분배.
4. **2-4A에서 남겨 둔 cap 실행 정의를 이번 계약에서 고정했다.** 모든 방식에 동일한 공개 중심 이동을 허용하고, class별 norm의95% quantile을 사용한다. 주 대조는 전체 공간에서 산출한 같은 cap, 보조 대조는 각 방식의 같은95% 규칙이다. cap/rank/예산 탐색은 없다.
5. 공통 공개 중심은 a의 환자균등 평균이다. 각 class 평균 복원 후 같은 중심을 더하면 class contrast에서 상쇄된다. 이는 공개 기준값을 대조에도 허용한 표준 affine 처리이며 새로운 기여가 아니다. 2-4A의 설명용 uncentered pooled cap을 채택했다고 하지 않는다.
6. 양성 class cap은 공개6명에서 정해진다. 이를 Q의 꼬리 분포나 충분한 임상 calibration이라고 해석하지 않는다.
7. 분모는 동일 noisy count를 모든 방식이 공유하고 max(1,count)를 사용한다. 분자에 추가 norm-consistency projection을 하지 않는 고정 decoder다. 기존 production 집계 전체를 그대로 재현했다고 하지 않는다.
8. 현재는 공개 모집단의 numerator/count 전달 자체를 비교했다. 실제 P+DP(Q)의 합치기, source head와 noisy count의 상관, Q의 분포는 이 공개 스트레스 검산으로 검증하지 않았다. 실제 실행 계약이 필요하다.
9. fixed y₀·모의 count에 조건부인 비교다. source 보호 자체를 다시 평가하는 실험이 아니다.
10. 같은 정보를 사용하는 일반 equality-constrained Label Solve와 같은 U를 쓰는 표준 workload 측정은 **동치**로 검산하고 별도 방법 우위로 세지 않았다.

## 오차 비교

아래 오차는 라벨을 양의 배율로 정규화하기 **전**의 H 기하 target 오차다. 정규화 후 head 크기·확률 calibration이 같다는 뜻은 아니다. 값이 낮을수록 좋다.

### 주 대조: 같은 class cap

음성 cap=2.780369, 양성 cap=2.050054.

| 방식 | 분자 query 차원 | 집계 bias² | 잡음 분산 | 고정 분모 기대 MSE | 잡음 분모 조건부 기대 MSE |
|---|---:|---:|---:|---:|---:|
| 전체→제한→투영 | 256 | 1.25210386e-05 | 4.652877 | 4.652889 | 10.604141 |
| 새 후보: 호환공간→제한 | 128 | 5.05605125e-10 | 4.652877 | 4.652877 | 10.605091 |
| 동일 rank 공개 PCA→제한 | 128 | 0.00875480793 | 3.126185 | 3.134940 | 7.152347 |

새 공간을 먼저 택하면 clipping bias는 실제로 작아졌다. 전체 방식의 class별 제한 행은34/1, 새 방식은1/0이다. 그러나 **같은 cap의 남는 Gaussian 분산은 동일**하다. 작은 bias 감소가 이 공개 조건에서 지배적인 잡음 분산을 줄이지 못했다.

PCA는 일부 필요한 성분도 버려 bias²가0.008755로 커진다. 대신 도달 공간에 남는 잡음 covariance trace가 감소해 총 MSE가 더 낮았다. U와 PCA의 겹침 trace는 **43.000458/64**다. PCA의 모든 오차를 noise라고 세지 않고 투영 bias까지 포함한 결과다.

즉 여기서는 ‘label이 사용할 수 있는 성분을 모두 보존’하는 선택보다 일부 성분을 버리는 표준 대조의 **bias–variance 절충**이 나았다. 이것은 공개 유한 target의 계산 결과이며, 사적 receiver 효용에 관한 일반 원리나 새 논문 기여가 아니다.

### 보조 대조: 각 방식의 동일한 공개 cap 규칙

| 방식 | 분자 query 차원 | 집계 bias² | 잡음 분산 | 고정 분모 기대 MSE | 잡음 분모 조건부 기대 MSE |
|---|---:|---:|---:|---:|---:|
| 전체→제한→투영 | 256 | 1.25210386e-05 | 4.652877 | 4.652889 | 10.604141 |
| 새 후보: 호환공간→제한 | 128 | 3.18847779e-05 | 4.080524 | 4.080556 | 9.312043 |
| 동일 rank 공개 PCA→제한 | 128 | 0.00882203568 | 2.652848 | 2.661670 | 6.083571 |

자기 cap을 허용하면 새 방식은 전체 방식보다 MSE가 **12.30% 낮아졌다**. 그러나 PCA보다 여전히 높다. 이 부분적 개선으로 주 대조의 판정을 바꾸지 않는다.

### 잡음 분모

공유 모의 count512회에서 양성 분모 floor는9회, 음성은0회 활성화됐다. 표의 마지막 열은 **각 모의 분모를 고정한 상태에서 분자 잡음을 해석적으로 적분한 MSE의 평균**이다. 분모 분포 전체에 대한 정확한 기댓값이 아니다.

실제512개 공통 Gaussian 분자까지 적용한 MSE는:
- 같은 cap: 전체 10.602043, 새 방식 10.602822, PCA 7.046582.
- 자기 cap: 전체 10.602043, 새 방식 9.308381, PCA 5.992347.

PCA가 낮다는 방향은 유지됐다. 같은 cap에서 새 방식의 개별 clipping 오차가 작아진다는 정리는 noisy ratio의 전체 오차까지 낮춘다고 보장하지 않는다. 실제로 잡음 분모를 포함한 전체/새 방식의 작은 차이는 반대 방향이다. 어떤 잡음에서도 안정적이라는 주장이나 환자 bootstrap의 임상 CI는 만들지 않았다.

## source 보존과 일반 Label Solve 대조

- 독립 KKT solver와 새 nullspace solver의 모든 라벨 차이 최대 **8.36e-12**.
- y=y₀+Nv에서 source head를 보존하고, ℓ=αy에서 source 점수 함수가 α배가 되는 것을 ridge0.01/0.1/1과 모든 공개 라벨 사례에서 확인. 공개 점수 비례 오차 최대 **2.56e-15**, 유의미한 순위 역전0.
- generic workload에 같은 U를 주면 환자 query가 같은 계산이다. 차이 최대 **1.67e-16**. 이것을 서로 다른 신규 알고리즘의 비교라고 부르지 않는다.
- clipping/noise가 없는 공통 cap 대조에서 전체 신호를 뒤에 투영한 target과 먼저 U로 투영한 target 차이 **5.33e-15**. 남는 noise covariance도 같았다.
- noiseless 공개 target에서 source를 보존하면서 receiver H오차를 **0.109594→0.009005**로 줄일 수 있었다(91.78% 감소). 설계의 표현 능력은 존재한다. 실제 보호 오차에서 PCA보다 유용하다는 뜻은 아니다.
- source 제약 없는 일반 Label Solve는 noiseless target을 약 1.2e-30까지 맞췄으나 source head가 변했다. source를 보존해야 하는 방식의 경쟁 대조로 거짓 포장하지 않는다. 양쪽에 같은 양의 공통 정규화를 허용했다.
- noiseless 새 라벨의 α=0.149644. 모의 잡음 하에서는 α가 더 작다. 보존되는 것은 frozen signed-label ridge의 순위이며 calibrated probability·고정 threshold·fine-tuning·CE 학습은 아니다.

## 보호 수식·구현 검산

공개로 고정한 변환/center/cap에서 환자의 모든 방문·두 class를 하나로 묶는다.

t_i = concat(clip_C0(x_i0)/C0, clip_C1(x_i1)/C1) / sqrt(2)

한 class가 없으면0. 각 block norm≤1이므로 전체 norm≤1이며 환자 add/remove sum sensitivity≤1이다. class0/class1에 동시에 등장하는 환자를 두 독립 개인처럼 세지 않았다. 마지막 unit-radius 처리는 float64 roundoff 여유다.

검산:
- 실제 공개 입력의 환자 삭제4032개 경우(6설정×672), 최대 L2변화 **0.810588≤1**.
- 양쪽 class가 포화되는 극단 입력·class 부재·음수 noisy count·잘못된 shape/NaN/cap9개 처리.
- Gaussian calibrator는 기존 것을 재사용하고 별도 privacy profile 식으로 δ를 계산했다: **4.99999998874e-06≤5×10⁻⁶**.
- 행별 projection-before-clipping 오차 부등식 확인. 합계 오차 우위를 정리로 과장하지 않음.
- 원 입력3개, 2-4A 및 기존 효용 보고서7개의 SHA256 모두 보존.

이번 **공개 모의 수치 예산**은 source/count ε4,δ5×10⁻⁶ 및 추가 numerator ε4,δ5×10⁻⁶이다. 향후 적법한 두 메커니즘을 조합한다면 basic 상한 ε8,δ10⁻⁵가 되는 예시다. σ=1.115937039, 공유 count의 SD=2.231874077. 실제 source 보호 전체를 실행한 것은 아니며 **실제 Q의 다음 release 계약을 승인·생성한 기록도 아니다**.

역사적 ε8 source에 추가 ε4 query를 붙여 전체 ε8이라 쓰지 않는다. 실제 Q release0이므로 기존 privacy ledger의 실제 누적값도 변하지 않았다. 공통 모의 난수는 비교 오차를 줄이기 위한 public coupling이며, 이처럼 상관된 여러 실 mechanism을 공동 공개해도 같은 보장이 된다는 주장은 하지 않는다. 기록한 PCG64 seed는 공개 모의 입력 전용이며 실제 private noise seed로 재사용하지 않는다.

## 판정과 다음 단계

**2-4B 완료: 구현 검산 PASS / 사전 공개 비교의 실용 이득 기준 미충족.**

이번에 확인한 긍정은 ‘source 보존 라벨 공간이 실제로 작동하고 clipping 순서가 그 성분의 오차를 줄인다’는 것이다. 확인하지 못한 핵심은 ‘그 선택을 표준 PCA/workload 처리보다 우선할 실용적인 이유’다. 작은 행별 오차 감소만으로 Q의 추가 개인정보 비용을 지출할 근거를 만들지 않는다.

반대로 P는 양성6명이고 분모 잡음이 크다. 따라서 이 비교로 Q에서의 실패나 설계의 보편적 무효를 입증했다고 하지 않는다. 다만 **현재 가진 근거에서 큰 실행을 권하지 않는다는 판단**은 가능하다. cap·공간 수·encoder를 바꾸어 유리한 결과를 찾는 자동 후속은 없다.

- **2-4A:** 후보 설계·공개 기하 완료, 그대로 보존.
- **2-4B:** 공개 입력 구현·대조·독립 검산·기록 완료.
- **2-1R3:** 미실행·권고 철회 유지.
- **상위2:** 논문 기여 검증 미완료.
- **상위3:** 미착수, Expert/Reserved 보존.
- **실제 최신 임상 개발 효용 결과:** 계속2-2R2.
- **다음 세부 실행:** 예약하지 않음. 2-4C나 Q 실행을 자동 생성하지 않는다. 다른 후속을 제안하려면 이번 표준 대조가 놓치는 구체적인 차이와 검증 예측을 먼저 제시해야 한다. 그 전에는 근거 없는 실행 규모·소요시간을 만들어 안내하지 않는다.

## 재현 산출물

- contribution_signal_prototype_20260929_v1/contract.json: 결과 전 고정한 범위와 수식·모의 수치 설정.
- prototype.py: P-only 입력 whitelist, query·decoder·라벨·해석적 오차 및 고정 모의 계산.
- verify_independent.py: 원 입력의 별도 집계·KKT·privacy profile 검산.
- results.json, verification.json, comparison.csv: 전체 결과.
- public_arrays.npz: 공개 수치 label/target/query/noise 배열. private summary가 아님.
- execution_seal.json, completion.json: 코드·계약·결과 결속.
- state_before.json: 이번 시작 전 연구 상태.

정식 선행 경계와 source 보존 목적은 [2-4A 설계](<PUBLIC_TRANSFER_2_4A_SIGNAL_DESIGN_20260929.md>)를 따른다. 이번 PCA 대조는 GEP/Dosser식 공개 부분공간 선택을 허용한 연산 대조이며, 해당 논문 전체를 재현하거나 이겼다는 주장을 하지 않는다.
