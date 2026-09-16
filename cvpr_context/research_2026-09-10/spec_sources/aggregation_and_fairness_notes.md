# 2번: 환자 집계와 보조자료 공정성 명세 메모

2026-09-14. 현재 단계는 **2. 가까운 기존 방법의 실패 조건·원인 확인**이다. [고정 연구틀](../RESEARCH_FRAMEWORK.md)과 [현재 상태](../research_state.json)를 읽고 작성했다. 이 문서는 원자료의 역할·기존 실행 범위 확인과 후속 비교 명세에 대한 조언이다. 새 feature 계산, GPU, fitting, 통계 재평가는 수행하지 않았다. 3번 설계나 4번 핵심 가설 검증의 진입 선언이 아니다.

## 1. 실제 보유 범위: cohort와 계산된 공격값은 다르다

cohort/summary.json 및 evaluation_images.csv에서 fit80, selection40, calibration140, test140을 확인했다. 아래 경로는 code_working/_reports/cvpr_u_pilot_v1_001 기준이다. 이미지/latent/encoder cache가 존재한다는 사실을 모든 target attack feature가 계산됐다는 뜻으로 쓰지 않는다.

| 기존 자산 | 실제 관측 범위 | 재사용과 해석 |
|---|---|---|
| SecMI-stat patient_scores | fit8 및 selection40, E/U, 두 target; mean/max | baseline_screen_20260914/secmi_v1/patient_scores.json. fit80 전체 response matrix가 아니다. |
| 원 R smoke와 부가 loss/gradient | U fit8×두 target; E fit1(14393)×두 target | probe_smoke/training_coverage_v2/U_step_1000_n8/results.json 및 E_step_1000_n1/results.json. U8의 원 FP16 response를 안정된 FP32 endpoint 점수와 혼용하지 않는다. |
| R 검증 및 FP32 endpoint | 동일 U fit8×두 target | verification_20260914/traced_U8_v1 및 endpoint_precision_U8_v1. 반복 검증은 새 환자를 추가한 자료가 아니다. |
| FP32 기본 loss 대조 | fit8, E/U, 두 target, 두 prompt 조건 | verification_20260914/basic_controls/patient_scores.json. 조건별 값이 있다는 사실이 특정 prompt 선택을 정당화하지 않는다. |
| PFAMI-style 본 비교 | selection40, E/U, 두 target와 base | baseline_screen_20260914/pfami_v1. fit matrix를 제공하지 않는다. |
| PFAMI benchmark | fit 환자14393 한 명, E/U, 두 target와 base | pfami_benchmark_v2/image_scores.json의 12개 영상·모델 기록. 원래 계약대로 전처리·속도 검증용이며 본 성능 표에서 제외했다. v1/v2를 별도 두 환자처럼 세지 않는다. |
| 공개 encoder-only 학습 대조 | **U fit80/selection40 전체**, 두 membership 배정 | encoder_control/scores.json·report.json. DINO mean, 절대 차이, cosine을 fit80 전용 scaler와 logistic regression(C=.1)으로 학습한 기존 대조다. target response의 학습형 결합이나 CDI 전체 구현은 아니다. |

target response의 공통 fit8 ID는 11632, 12511, 14393, 22454, 7038, 9216, 9429, 9925이다. encoder-only 코드는 u_patient_audit/encoder_control.py:18–45에서 U만 사용하고 fit80만 scaler/classifier에 넣는다. 기존 기록의 selection AUC는 두 배정에서 .5475이며 fit AUC1.0은 평가 성공 근거가 아니다. 이는 기존 기록을 읽은 값이고 재계산하지 않았다. “학습형 대안 전부 미실행”이 아니라 **target-response의 강한 특징 결합·CDI/MoFit 환자 확장 비교가 미완료**라고 적어야 한다.

## 2. 역할을 바꾸지 않는 비교 계약

- **fit80:** 계수·scaler·feature 선택·정규화 정책·학습형 score의 방향을 결정할 수 있는 개발 fitting 구역이다. 설정 선택이 필요하면 fit 안의 환자 단위 분할을 명시한다. 같은 환자의 사진·E/U·두 target 관측을 다른 fold의 독립 사례처럼 나누지 않는다.
- **selection40:** 이미 여러 방법 개발과 분석에 사용했다. 고정된 방법을 비교하고 개발 선택을 기록할 수 있으나 untouched test, 확증 성공, 독립 재현으로 부르지 않는다. selection을 fitting에 몰래 편입하거나 결과에 맞춰 부호를 뒤집지 않는다.
- **calibration140:** 후속에 방법이 고정된 뒤 환자 단위 threshold를 정하는 역할이다. 이번 명세 작성은 이 구역의 실행/튜닝 개방이 아니다. 방법 계수 선택과 threshold 보정을 구분한다.
- **test140:** 최종 고정 방법·threshold의 실제 환자 FPR/TPR 평가용으로 보존한다. 방법을 고르는 입력이나 약한 selection 결과를 구제할 자료가 아니다.
- 각 방법은 동일한 모집단·환자 역할·m=2·E/U·대상 checkpoint를 사용한다. 두 target의 환자 역할은 유지한다. 한 target의 공격 feature에 다른 target 점수, A/B 배정, 실제 학습 manifest, 정답 정렬 delta를 추가하지 않는다. 정답은 허용된 fit label과 평가에만 사용한다.
- fixed statistic의 member 방향은 원문/명시적 정의대로 고정한다. 별도 sign/scale fitting을 허용한다면 fit만 사용하고 모든 비교군에 같은 기회를 주며, 이를 원형 고정 통계와 구별한다. selection/calibration/test 전체로 standardization하거나 feature 선택하지 않는다.

## 3. E/U fitting의 의미를 먼저 구분한다

권장 비교 명세는 **E/U와 target별로 fitting 역할을 구분**하고, 각 조건에서 동일 fit 환자·보조정보 기회를 주는 것이다. 조건마다 임의로 다른 방법을 골라 최고값만 제시하지 않는다.

- U fitting의 label은 patient membership이다. U 영상 자체는 모두 image nonmember이므로 image-membership=0을 그대로 학습 label로 쓰면 환자 문제를 구현한 것이 아니다.
- E 환자/영상 label로 학습한 per-image scorer를 고정해 U로 옮기는 비교는 **E→U transfer** 질문이다. U에서 알려진 patient label로 scorer/aggregator를 fitting한 **U 조건에 맞춘 대안**과 같은 설정으로 부르지 않는다.
- CDI 원형의 P/비공개 U 집합 출처 label, image membership, 우리 U의 patient membership을 구분한다. U patient label로 특징 결합을 학습하면 유효한 patient adaptation이지만 원 논문 전체 재현은 아니다.
- E/U를 합쳐 fitting하면 unknown-overlap 문제라는 다른 조건이 섞인다. 이번 주 U 비교의 기본값으로 암묵적으로 합치지 않는다.
- E가 정상 작동의 유용한 진단인 것은 맞지만, **E 통과를 U 가능성의 논리적 필수조건으로 추가하지 않는다.**

## 4. 표준 대안을 약화하지 않는 최소 명세

| 대안 | 반드시 명시할 것 | 기여 해석 |
|---|---|---|
| per-image scalar의 mean/max | 같은 두 영상·같은 scalar·같은 방향; 각 patient score에 별도 공정한 threshold 기회 | 표준 환자 집계. m=2에서 top-1=max, top-2=mean이므로 중복 baseline을 늘리지 않는다. |
| 기존 특징의 학습형 결합 | 어떤 per-image 특징을 어떤 대칭 집계로 바꾸는지, fit scaler/classifier/규제, 환자별 가중치 | 기존 특징 결합 대안이다. 단순 평균보다 낫다는 사실만으로 새 공격 기여가 생기지 않는다. |
| CDI-style 환자 확장 | 26차원 특징 중 사용/미사용 항목, 학습형 scorer, 보조자료의 출처·label·reference, patient pooling | SecMI 한 점수나 작은 특징 일부를 full CDI로 부르지 않는다. 별도 fit per-image scorer→patient pooling은 강한 적용 후보지만 전체 절차·비용을 기록해야 한다. |
| MoFit의 환자 적용 | 원형 surrogate/embedding 최적화와 최종 평가; image score 집계인지 실제 다른 사진으로 전이하는 변형인지 | 기존 R을 MoFit 대신으로 세지 않는다. 원형 핵심을 생략한 저예산 변형의 실패를 MoFit 전체 실패라고 하지 않는다. |
| 공개 encoder-only 및 표준 관계 | 기존 U encoder 대조의 실제 범위, score-only/관계 feature의 입력·학습 역할 | target-independent 편향 대조와 표준 대안이다. 실패가 모든 교란 제거를 증명하지 않는다. |
| 기존 특징 vs 같은 특징+후보 R | 동일 fit 환자·분할·학습기·규제 선택 예산; R 이외의 정보 차이 제거 | 후속 추가 이득의 직접 대조이나 지금 fitting된 결과는 없고 새 설계의 근거도 아직 아니다. |

현재 fit response가 8명뿐인 상태에서 수십 차원 학습 결합을 돌리고 “강한 대안을 충분히 비교했다”고 판단하지 않는다. 최소 표본 수나 합격선을 여기서 임의로 만들지 않는다. 필요한 공통 feature 범위와 fitting 복잡도·정보·비용을 root의 inventory와 첫 실행 계약에서 먼저 고정한다.

## 5. 정보·비용·환자 FPR의 공정성

- MoFit/CDI가 별도 reference 모델·공개 보조자료·알려진 labels·최적화·추가 fitting을 필요로 하면 숨기지 않는다. 우리 후보에 같은 기회를 주거나, 정보가 다른 비교를 별도 구역으로 명시한다. 공개 base를 disjoint 의료 reference generator라고 부르지 않는다.
- 같은 입력 2장을 사용했다고 비용이 같지는 않다. target/base forward, backward, VAE/encoder/검색, reference 모델 및 scorer 학습을 구분한다. 공유 cache의 실제 재사용 절감과 각 방법을 독립 실행하는 비용을 함께 구분한다.
- 같은 비용 예산 비교와 원형에 가까운 실행 비교를 구분한다. baseline만 약하게 잘라 놓고 원형 SOTA보다 우월하다고 하지 않는다. 더 비싼 기존 방법을 비용 때문에 제외할 때는 남은 비교 공백을 적는다.
- FPR은 **환자 단위**로 보정·평가한다. image threshold를 그대로 가져와 patient mean/max와 비교하지 않는다. 각 방법에 동일 calibration 환자·목표 FPR을 주되 method-specific threshold를 허용한다. 실측 test FPR와 불확실성을 같이 보고한다.
- 140 calibration/test 환자가 1% FPR를 정밀하게 검증한다고 전제하지 않는다. 실제 nonmember 수·검정 해상도를 별도로 확인해야 한다. 이 메모는 새 FPR 목표나 합격선을 정하지 않는다.
- 두 target은 독립 seed 반복이 아니다. bootstrap/분할에서 환자 짝을 보존하고, 사진 수·timestep 수·모델 수로 독립 환자 수를 부풀리지 않는다.

## 6. 이번 메모의 종료 범위

**완료:** 기존 fitting 자산의 실제 범위와 공정 비교에서 보존할 역할·표준 대안을 확인했다. **미완료:** MoFit/CDI의 실제 환자 확장, 충분한 target-response 학습형 결합 비교, 기존 실패 원인의 확인이다. 다음 한 작업을 공정 비교 명세와 실제 첫 실행 하나의 범위·비용 고정으로 두는 것은 타당하다. 이 문서가 새 GPU/feature 계산/fitting을 실행했거나 연구 2번을 마쳤다는 뜻은 아니다.

근거: [기존 방법 조건](../METHOD_PREMISE_AUDIT_2026-09-14.md), [SecMI 실행](../U_BASELINE_SCREEN.md), [PFAMI 실행](../PFAMI_COMPARISON_RESULTS.md), [점수 분해](../PAIRED_SCORE_AUDIT.md), [원 실행계획](../U_EXECUTION_PLAN.md), 위 표의 원시 JSON 및 encoder_control.py.

