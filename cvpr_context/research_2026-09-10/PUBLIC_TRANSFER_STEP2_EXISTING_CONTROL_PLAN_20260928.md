# 다음 계획: 2번을 기존 단독 대조로 정리

2026-09-28 / 큰 단계2 / **계획만 기록, 새 실험 미실행**

## 결정

1번의 일반 teacher-score KD 비교는 긍정적인 개발 결과로 완료했다. 다음은 2번이다.

다만 DINO-only, publicly transported ResNet18-only, DINO+ResNet18 joint의 **두 기존 DP release 결과·라벨·예측·bootstrap이 이미 존재한다.** 따라서 같은 대조를 다시 학습하거나 라벨을 새로 만들지 않는다.

다음 작업은 저장된 산출물의 비교 규칙을 확인하고, 기존 대응 차이와 두 차이의 평균을 정리하는 **분석 한 묶음**으로 제한한다. 예상10~20분, 새 Q/DP/합성/모델 forward0이다.

## 두 질문을 구분한다

- **2A:** DINO만 맞추는 것보다 공개 RN 목표를 함께 반영하는 것이 유용한가?
- **2B:** 그렇게 얻은 이득 때문에 DINO와 RN을 *둘 다* 맞춰야 하는가, 아니면 전달된 RN 목표만으로도 되는가?

2A가 양성이어도 2B가 자동으로 성립하지 않는다. 2B는 이미 있는 RN-only 대조를 재사용하며 새로운 방법을 추가하는 일이 아니다. 여기서 RN-only도 사적 RN query가 아니라 기존 DINO-DP 요약을 공개 P로 전달한 목표다.

## 현재 확보된 수치 — 새 계산 결과가 아님

| 기존 release | DINO-only DenseNet AUROC | RN-only DenseNet AUROC | Joint DenseNet AUROC |
|---|---:|---:|---:|
| DP1 | 0.671617 | 0.713564 | 0.686370 |
| DP2 | 0.613363 | 0.648230 | 0.663404 |

기존 Joint−DINO AUROC 구간은 DP1 [-0.002636,+0.032335], DP2 [+0.024707,+0.076099].
기존 Joint−RN 구간은 DP1 [-0.053644,-0.001882], DP2 [-0.013283,+0.044077].

따라서 2A에는 긍정적인 단서가 있지만, 2B의 일반적 joint 우위는 현재 미확인이다. RN-only가 DP1에서 좋았던 결과도 반드시 함께 보고한다. 어느 release에서든 가장 좋은 숫자만 선택하지 않는다.

## 정규화 차이 때문에 다시 학습할 필요가 없는 이유

source_label_baselines.py의 단독 목적은 다음 형태다.

\[
L_D(\ell)=\|\sqrt W(B_D\ell-t_D)\|^2+
\left(10^{-3}\|\sqrt WB_D\|_F^2/128\right)\|\ell-\ell_{\rm hard}\|^2.
\]

단독 목표를 target RMS s로 나누는 공통 표현을 적용하면 data term과 같은 공식의 regularizer 모두 1/s²배가 된다. 모델 무게를0.5로 쓰더라도 단독인 경우 전체에 같은 양의 상수가 곱해진다. 따라서 bounds와 hard-anchor가 같다면 최적해는 바뀌지 않는다.

RN-only도 같은 구조다. 현재 joint는 두 normalized 단독 목적을 같은 비중으로 합친 구조다. 단독군이 target RMS 정규화를 생략했다는 이유만으로 새 라벨 대조가 필요하다고 보지 않는다.

이번 계획에서 소스 코드를 읽고 대수적으로 확인했다. 실제 배열을 다시 풀거나 새 성능을 계산하지 않았다. 이후 분석에서는 과거 input/label/평가 해시·설정이 같은지 확인하되, 이미 통과한 전체 pipeline 검산을 재실행하지 않는다.

## 다음 분석의 정확한 범위

입력:
- contribution_search_20260924/joint_compiler_results.json
- joint_compiler_predictions_private.npz / joint_compiler_bootstrap.npz
- joint_compiler_soft_labels.npz / source_label_baseline_soft_labels.npz / compiler_control_soft_labels.npz
- 기존 계약·연산 코드·2,000회 환자 bootstrap schedule의 provenance

작업:
1. 원래 두 PNG/release·source/RN targets·hard-anchor·bounds·ridge0.1·환자/class 무게·평가 순서의 동일성을 기존 기록에서 확인한다.
2. 기존 release별 Joint−DINO, Joint−RN의 AUROC/AP와 구간을 재사용한다.
3. 같은 환자 resampling index에서 두 release 차이를 평균한 조건부 CI를 추가 정리한다. 가능하면 저장된 bootstrap arrays만 사용해 불필요한 metric 재실행을 피한다.
4. DINO/RN/J의 모든 점추정과 두 직접 비교를 함께 보고한다. KD 결과는 1번의 배경 참고로만 두며 재계산하지 않는다.

금지:
- 새 label solve, 새 PNG, 새 seed/noise, Q 재접근, encoder/계수 탐색.
- zero-centered regularizer로 변경. 현재 비교는 과거 hard-anchor 그대로다.
- 3번 공개128 또는 Expert/Reserved 자동 착수.

## 해석·판정

DenseNet 영상 AUROC가 주 지표, AP는 필수 보조 지표다. 구간은 현재 고정 banks에 대한 paired patient-cluster uncertainty이며 DP 잡음 분포 전체에 대한 구간이 아니다.

2A와 2B를 각각 양성/혼합/불확실로 보고한다. 결합의 필요성을 주장하려면 DINO뿐 아니라 RN-only와의 비교도 뒷받침해야 한다. CI가0을 포함하는 것을 동등성이나 “추가 모델이 쓸모없음”으로 바꾸지 않는다.

전체 계획에 명시했던 개발 투자 기준(양의 두 AUROC 점차이, 평균≥.01, 평균AP≥0, 평균조건부CI>0)은 필요한 대비별로 공개한다. 기존 결과를 이미 본 사후 분석이므로 사전등록된 독립 확인이라고 부르지 않는다. 실패한 기준을 바꾸거나 여러 기준 중 좋은 것만 고르지 않는다.

- 2A만 지지되면 “공개 RN 목표가 DINO-only보다 도움이 될 수 있다”까지만 남긴다.
- 2B가 혼합이면 joint가 모든 단독 방식보다 낫다는 주장은 하지 않는다.
- 2번이 혼합이어도 1번의 MT−KD 양성 결과를 삭제하지 않는다. 서로 다른 질문이다.
- 어떤 결과든 자동으로 새 논문 기여 이름을 붙이지 않는다.

## 이후 3번

2번을 정리한 뒤에는 “동일한 라벨 계산을 허용한 공개영상128장으로도 되는가?”가 남는다. 3번에는 carrier에 공정한 공통 라벨 regularizer와 공개128 선정·전처리 계약이 필요하다. 그때 별도 범위를 확정하며 이번 분석에 끼워 넣지 않는다.

연구 진행은 **1번 완료 → 2번 기존 대조 분석 → 3번 공개 carrier 비교 → 필요한 반복·독립 확인**이다. 결과가 좋아질 때까지 새로운 합성 seed나 receiver를 늘리는 일정이 아니다.

예정 산출물: PUBLIC_TRANSFER_STEP2_EXISTING_CONTROL_RESULTS_20260928.md와 작은 analysis JSON.
현재 실제 최신 실행 결과는 [1번 결과](PUBLIC_TRANSFER_KD_CONTROL_RESULTS_20260928.md)이며 이 문서는 다음 분석 계획이다.

