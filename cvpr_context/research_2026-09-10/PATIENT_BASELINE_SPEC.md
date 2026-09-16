# 2번 — 가까운 방법의 환자 확장과 공정한 비교 명세

2026-09-14 작성, 2026-09-15 KST 실행 상태 갱신. [고정 연구틀](RESEARCH_FRAMEWORK.md)의 **2번 진행 중**이다. 목표는 감사자가 보유한 사진과 실제 학습 사진이 다를 때 환자 참여·위험을 평가하는 것이다. 일부 SecMI/PFAMI 진단의 약한 결과를 전체 방법의 실패나 연구 불가능으로 확대하지 않는다.

**최신 지정 확장:** [두 환자 기여 제거 교차 대조](STAGE2_CROSS_PATIENT_INTERVENTION_20260915.md)를 완료했다. 원형/p제거/q제거의 U41명·E2명,258개 영상 특징 중90개를 새로 계산했다. 기존 U79 고정 C 주4개와 원fit80/모든 보조 방법을 그대로 적용했고 새 fitting0이다. mean+max52는 U에서 R_p/R_q 모두 양수지만 q의 h는 공통39명 IQR 안이다. 강한 기존 방법이 잡는 대응 반응을 새 공격 필요성으로 주장하지 않으며 같은 U-fit scorer의 E 결과와 scalar의 E 양성도 함께 보존한다. 이 교차 대조는 내부 원인 진단이며 외부 감사자의 허용 입력을 늘리는 실험이 아니다.

이 명세에 따른 **CDI U 환자 비교·순위 분해·실제 base 대조·평균/영상 간 차이 항 진단과 독립 검산은 완료했다.** [실제 비교 결과](CDI_U_COMPARISON_RESULTS.md)에 모든 지정 비교를 보고한다. 기존 meanmax의 차이 항이 순위를 개선하는 관측은 남았지만 각 자기 target 차이 구간은0을 포함한다. 환자 참여 효과나 단순 base 편향이라는 원인은 미확인이고, 2번 전체는 계속 진행 중이다.

## 1. 비교 대상과 이름

| 실행할 대상 | 보존할 핵심 | 환자 적용·해석 |
|---|---|---|
| MoFit-derived patient pooling | null-conditioned 이미지 surrogate 최적화 1000회 → 전체 text embedding 최적화 300회 → 원본 영상에서 최종 점수와 auxiliary | 각 사진을 독립 공격한 후 mean/max. 현재 R의 8D fitting으로 대체하지 않는다. 의료 LoRA 공개 전용 main이 없어 원형 의료 실험 전체 재현이라고 부르지 않는다. |
| CDI-26 image scorer → patient pooling | DL1 + SecMI1 + PIA/PIAN2 + GM10 + ML10 + NO2, fit-only scaling/LR | 별도 fit 환자에서 scorer를 학습한 뒤 환자 2장 점수를 mean/max. 원형 집합별 reference 검정 전체와 구분한다. |
| CDI-26 patient feature aggregation | 같은 26D 영상 특징의 평균 또는 평균+최대 → 같은 학습기 | 기존 특징의 강한 표준 결합. 원형 CDI 자체와 구분하지만 가까운 대안으로 반드시 공정하게 다룬다. |
| 고정 영상 점수 → mean/max | 같은 영상·noise·모델·점수 방향 | 기본 집계 효과 대조. m=2에서 top1=max, top2=mean, median=mean이다. |
| 공개 encoder-only | DINO 평균·절대 차이·cosine → fit80 scaler/LR | U fit80/selection40 실행 결과를 이번 CDI 비교에도 재사용했다. selection AUC .5475/.5475. 단일 encoder 대조가 모든 target 비의존적 설명을 배제하지는 않는다. |
| 기존 특징과 표준 관계의 결합 | 같은 fit 정보와 encoder 접근권·계산비용 | 새 구조를 주장하기 전 대안 설명을 제거한다. 모든 관계 특징을 새 기여로 보지 않는다. |

사진 두 장의 특징별 mean+max는 mean+absolute difference와 정보상 서로 변환 가능하다. 다만 StandardScaler·L2 정규화 LR의 최종 예측까지 항상 같다는 뜻은 아니다. 같은 정보를 별개의 새 관계 정보로 부풀리지 않는다. 대칭적인 두 사진 유사도만으로 동일한 두 가중치를 주는 정규화 weighted mean 역시 단순 mean으로 돌아갈 수 있다. 실제 연산이 달라지는지 먼저 확인한다.

CDI 원형의 U_ref(미사용 reference 집합)와 우리 U(동일 환자의 보유 사진이 학습에 직접 들어가지 않음)는 다르다. 우리 U의 member 환자를 nonmember reference로 쓰지 않는다. 한 환자 사진 2장으로 원형 5fold를 만들 수 없다는 이유로 CDI를 제외하지 않는다. 원문 Appendix U의 별도 control fitting 경로와 같이 다른 fit 환자에서 scorer를 학습할 수 있다.

## 2. 실제 보유한 자료와 아직 없는 자료

cohort는 fit80 / selection40 / calibration140 / test140이다. 모델1의 A, 모델2의 B가 참여한다. E는 train_candidate 역할 사진이며 해당 target에서 참여 환자일 때만 실제 학습 사진이다(nonmember의 E는 학습 제외). U는 환자 참여 여부와 무관하게 학습 제외 사진이다. 두 모델은 독립 학습 seed 반복이 아니다.

| 자산 | 실제 범위 |
|---|---|
| 고정 latent·text hidden·encoder cache | 잠긴 cohort 입력. 공격 특징 전체가 계산됐다는 뜻이 아니다. |
| SecMI-stat | fit8 + selection40, E/U, 두 모델 |
| 기본 FP32 loss·R 관련 진단 | 대체로 fit8. 시간/noise 정의가 다른 PFAMI 값과 같은 feature로 합치지 않는다. |
| PFAMI | selection40; fit14393의 별도 benchmark |
| encoder-only 학습 | U fit80/selection40 완료 |
| CDI 26D kernel | fit14393 E2/U2, model1의 4장 추출·독립 검산 완료. 환자 성능 비교는 이 4장으로 하지 않았다. |
| CDI U 전체 특징·학습형 비교 | fit80+selection40, 두 target의 480개 records 추출·18개 지정 방법과 기존 DINO 대조·독립 저장 특징/통계 검산 완료. 기존 kernel U2개 재사용, 478개 새 계산. [결과](CDI_U_COMPARISON_RESULTS.md). |
| MoFit 원형 기반 의료 환자 공격 | 동일 환자 E2/U2 × 두 target의8 records·네 점수 mean/max 실행·검산 완료. full 실행기3,725.633초. 의료 fusion과 환자군 성능 비교는 미완료다. [실제 결과](MOFIT_MEDICAL_RESULTS.md). |

따라서 fit8의 수십 차원 학습이나 서로 다른 loss 정의의 캐시 결합으로 “강한 기존 방법과 비교했다”고 하지 않는다. 정확히 같은 ID·checkpoint·dtype·입력·noise·정의가 맞는 기존 SecMI 등만 재사용한다.

## 3. fitting과 접근권

기본 접근은 현재 고정된 **white-box 의료 SD2.1 LoRA, 후보 영상과 환자 그룹, 공개 base**다. 일반 생성 API black-box 결과라고 부르지 않는다. 학습된 공격은 알려진 보조 참여 정답을 추가로 사용하므로, 그런 정답이 없는 환자 단독 설정과 별도 조건으로 표시한다.

두 경로를 섞지 않고 비교한다.

1. **E→U 전이:** fit E의 image/member label로 영상 scorer를 학습하고 고정하여 selection E와 U를 평가한다.
2. **U 조건에 맞춘 강한 대안:** fit U에 patient participation label을 붙여 학습하고 selection U를 평가한다. U 영상 자체의 image membership은 모두 0이지만 환자 label은 아니다.

각 target의 scorer는 별도로 학습한다. 다른 target 점수, A/B 정답 정렬 delta, 실제 train manifest를 공격 feature로 넣지 않는다. 두 모델 관측을 함께 쓰는 분석은 analyst-only 진단이다.

CDI 작동 경로의 기본은 원형 experiments/cdi_perf.py와 대응하는 StandardScaler + LogisticRegression(C=1, solver=liblinear, max_iter=1000, random_state=0)이다. scaler는 fit 데이터에서만 맞춘다. 이미지 단위 fitting은 같은 환자의 두 사진을 같은 fold에 두고 환자당 동일 가중치를 준다.

강한 보정 변형은 모든 같은 입력의 학습형 비교군에 동일하게 fit80의 환자 단위 5fold, C∈{.01,.1,1,10,100}, 평균 validation log-loss 최소(동률이면 작은 C)를 허용한다. 같은 환자의 E/U·두 모델 관측은 같은 fold ID를 유지한다. 이후 fit80 전체로 재학습한다. 원형 고정 C와 이 튜닝 변형은 구별하고 selection 점수로 부호·C를 뒤집지 않는다. 실제 fitting 전에 fold ID·feature 열·순서·설정이 결속된 실행 계약을 저장한다.

selection40은 이미 여러 개발 판단에 사용했다. 여기서 비교한 결과를 untouched test나 확증 성공이라고 부르지 않는다. calibration140은 고정된 방법별 환자 threshold에, test140은 최종 평가에 남겨 둔다. calibration/test 각각의 nonmember는 모델당70명뿐이라 1% FPR를 정밀 검증할 수 있다고 전제하지 않는다(fit40, selection20 nonmember).

CDI reference의 환자군 확장은 fit80과 환자가 겹치지 않는 selection member20명 대 nonmember reference20명으로 별도 고정해8개를 완료했다. 원형 함수의 양측p와 원문의 단측 Welch를 모두 저장·대조했다. reference64가 있다는 사실만으로 그 분포·수량이 적절하다고 확정하지 않았으며, 원형 전체5fold/1000subset 실험 재현이나 환자 한 명 판정으로 부르지 않는다. [E/U와 집합 검정 결과](CDI_EU_COMPARISON_RESULTS.md).

## 4. 연산·비용과 구현 차이

### CDI 26D

열 순서: DL(t100,5회 L2 평균), 양의 SecMI L2, PIA(t200,p5), PIAN, GM(t0:100:900) 10개, ML(같은10t) 10개, NO 최종 residual L2와 perturbation L2.

| 모듈 | 이미지·target당 UNet 비용 |
|---|---:|
| DL | 5F / 0B |
| SecMI | 12F / 0B |
| PIA와 PIAN 공동 추출 | 3F / 0B |
| GM | 20F / 10B |
| ML | 10F / 0B |
| NO | (q+1)F / qB |
| 전체 | **(51+q)F / (10+q)B**, q=실제 optimizer objective 평가 수 |

F는 forward-example, B는 입력 미분 backward다. NO maxiter=5는 q=5가 아니다. CPU 원형 실행에서 SciPy jac=True의 MemoizeJac 캐시로 nfev와 실제 objective 호출 수가 다른 경우도 확인했다. q는 실제 trace/모델 계산 횟수이며 nfev/njev·캐시 차이를 별도 저장한다. ML yaml의 반복5는 실제 extractor에서 사용하지 않는다.

원형 코드 그대로의 첫 경로를 보존한다. GM은 gradient를 z_t에서 구한 후 clean z0를 masking하고 (epsilon−z0)와 예측의 차이를 계산한다. NO는 objective에 z_t+delta를 넣지만 최종 feature에서는 원 noise 인자로 한 번 더 noising한다. 논문 수식 의도에 따른 수정은 별도 변형이며 원형 실행으로 숨기지 않는다.

Generic wrapper의 random placeholder를 사용하지 않고 현재 고정 latent·alpha·실제 epsilon backend를 연결한다. CDI scorer에는 missing numpy import와 dataset 문자열 indexing 문제가 있어 정상 작동하는 원문 experiments 경로의 .data/.label 인터페이스를 따른다. 동결 원본은 수정하지 않는다. 이후 원형 방법의 한계를 주장하려면 GM/NO의 논문 의도와 코드 차이가 결과를 약하게 만든 설명도 분리해야 한다. literal 코드 진단의 약한 결과만으로 원형 방법의 실패를 확정하지 않는다.

### MoFit

COCO 기본은1000+300회이며 surrogate VAE mean, embedding 단계의 고정 posterior sample, 전체77×1024 embedding을 유지했다. 의료 BLIP revision f8a8378a9013e8e6330a930dc0b405798d326c56의 실제4장 caption·CLIP hidden·CPU 난수를 고정했다. source의 별도 의료 schedule/fusion 조건은 공개되지 않아 현재 NIH P256/SD2.1 적용과 원 논문 의료 실험 재현을 구별한다. [실제 실행·검산](MOFIT_MEDICAL_RESULTS.md).

공개 literal stage1/2는3200F/1300B,VAE encoder1002F/1000B이며 외부 최종 평가가 별도다. 이번 실제 실행은 CFG2·매회monitor를 보존하고 endpoint 진단3F·VAE1F를 추가해 이미지당3203F/1300B다. 모델1 E/U 두 장은923.982초, 최대 CUDA 할당5,490,651,648bytes였다. 첫 점의 CFG1/2 손실·gradient가 근접한 것은 확인했지만 전체 궤적 동일성을 입증하지 않았으므로 빠른 변형으로 바꾸지 않았다. scalar 공개 replay의 gamma .55는 COCO fixture에서 선택된 값이며 의료 최적값으로 가져오지 않는다.

후속으로 같은 환자 E2/U2 × 두 target의 8 records와 네 고정 점수의 mean/max를 완료·검산했다. full 실행기 합계는 3,725.633초, 25,624F/10,400B다. U max가 다른 사진의 모델 차이를 남기지 않는 사례를 확인했지만 표준 평균으로 결과가 달라졌다. 이 사례를 강한 기존 대안에도 남는 한계나 새 설계 근거로 승격하지 않는다. 한 환자 산술 진단이며 의료 Eq9 fusion과 환자군 성능 비교는 아직 완료하지 않았다.

독립 실행 비용에는 보조 모델·VLM·VAE·text/encoder·reference 추출·공격 fitting을 포함한다. 이미 계산한 특징을 여러 pooling에 재사용한 실제 절감도 별도로 기록한다. baseline만 임의로 잘라 놓고 원형 SOTA 우위를 주장하지 않는다.

## 5. 2번에서 답해야 할 질문과 필요한 증거

| 질문 | 이번부터 확보할 관측 | 판정의 한계 |
|---|---|---|
| 구현이 원형 핵심을 보존하는가? | 소스 연산, noise/alpha/부호, gradient·중간값, 기존 SecMI와의 동일 입력 대조 | 검산 PASS는 membership 성공이 아니다. |
| 알려진 양성 조건에서도 원형이 작동하는가? | 공개 점수 해석 대조와, 이후 필요한 matching checkpoint/image의 추론 확인을 구분 | 공개 점수 재생은 전체 inference 재현이 아니다. |
| 약한 결과가 영상 score 자체 때문인가, U 전이 때문인가? | 같은 특징·scorer의 E와 U, E-fit 전이와 U-fit 대안 비교 | E 약함을 U 불가능의 필수 전제로 삼지 않는다. |
| 단순 집계가 어떤 정보를 잃는가? | 같은 원특징에서 image-score pooling과 patient-feature fitting 비교 | 표준 결합이 해결하면 그 효과 자체를 새 기여로 주장하지 않는다. |
| 모델이 바뀌어도 남는 쉬운 특성 때문인가? | encoder-only와 target 반응 대조, paired 분석의 설명 범위, 노출/분포 조건 | 높은 순위 상관만으로 원인을 환자 bias라고 확정하지 않는다. |
| 구체적인 실패 조건·원인이 반복되는가? | 방법이 정상인 조건과 실패 조건의 차이를 사전 명시해 검증 | 연산/학습 조건을 임의 변경해 좋은 값만 선택하지 않는다. |

공개 MoFit COCO 저장 점수 대조는 실제 완료했다. 공개 fixture에서 선택한 gamma=.55에서 grid AUC .941948, ASR .883, 원래1% 경계 규칙의 TPR .468이다. 독립 rank-AUC .94394, 정확 empirical FPR≤1% TPR .488이다. 논문 수치와 가깝지만 헤더 iterations 불일치·행 ID 부재·동일 fixture에서 tuning한 한계가 남는다. [실제 replay와 근거](replays/mofit_coco_fixture_20260914/report.md)를 참고한다. 이는 우리 의료 환자에서의 성공 관측이 아니다.

## 6. 완료된 kernel·U 비교·순위 분해·base 대조와 후속 원인 점검

**한 환자 kernel은 완료했다.** 기존 fit14393, model1 step1000, E2/U2 총4장에서 CDI literal 26D를 계산하고 저장된 원시 tensor를 독립 CPU float64 경로로 검산했다. 실제 비용은 232F/68B, 측정 구간25.238초, 실행기 전체29.724초였다. 104개 특징과 같은 입력의 기존 SecMI 중간값 일치를 확인했다. 이는 저장 산술 검산이며 새 GPU에서 실제 UNet Jacobian 전체를 다시 계산한 것은 아니다. [완료 결과와 원시 근거](CDI_KERNEL_RESULTS.md)에 구체적 범위를 기록했다.

**U fit80+selection40, 두 target, 환자당2장으로 고정한 비교를 완료했다.** 총480개 image-target records 중 같은 계약으로 계산한 model1의 kernel U2개를 재사용하고478개를 새로 계산했다. 새로운 target 학습은 없다. 추가학습 참여가 환자 label이고 U 사진 자체는 모두 해당 추가학습에서 제외됐다. base의 사전학습 참여 여부를 정답으로 삼는 실험은 아니다.

GPU 실행은 2026-09-14 15:39 UTC(09-15 00:39 KST)에 시작했고, 실행기 전체 시간은 **2,916.393초(48분36초)**였다. 신규 계산은27,942F/8,344B, 재사용을 포함한 독립 실행 비용은28,058F/8,378B다. 사전 안내한 추출45–55분은 과거 예상으로 남기고 현재 시간을 대신하지 않는다. 코드·분석 계약은 실행 전에 동결했으며, 전체 추출·아래 CPU 비교·저장 특징과 통계 검산이 완료됐다. 검산 PASS는 독립 GPU Jacobian 재실행이나 원인 입증을 뜻하지 않는다.

| 완료한 고정 CPU 비교 | 수 | 조건 |
|---|---:|---|
| CDI26 영상 LR → 환자 mean/max, 환자 mean26 LR, 환자 mean+max52 LR | 8 | 네 family 각각 C=1 고정형과 fit80 5fold로 C를 고르는 강한 변형. 영상 scorer의 CV 점수도 환자 확률로 집계한 뒤 log-loss를 계산한다. |
| DL/SecMI/PIA/PIAN → 환자 mean/max | 8 | 사전 지정한 음의 member 방향을 영상 점수에 먼저 적용한 후 집계한다. selection 결과로 방향을 뒤집지 않는다. |
| NO objective endpoint를 추가한 27D → 환자 mean+max54 LR | 2 | 고정 C/튜닝 C 각각 비교한다. 공개 코드 final feature를 교체하지 않고, 실제 평가 좌표에 결속된 optimizer.fun을 추가하는 비원형 진단이다. |
| 기존 DINO encoder-only | 별도1 | 이미 계산된 fit80/selection40 점수를 그대로 재사용한다. 새로운 fitting이나 target 정보 추가는 없다. |

주대조는 **튜닝한 환자 mean+max52 대 튜닝한 CDI 영상 mean pooling**, 그리고 **튜닝한 CDI 영상 mean pooling 대 고정 SecMI mean**이다. 고정 C의 대응 대조와 NO 추가 변형은 별도로 보고한다. 모든 방법을 표시하며 selection에서 가장 잘 나온 것만 고르지 않는다. 공통 selection 환자 재표집2000회(seed260914)를 모든 방법과 두 target에 함께 사용해 AUC와 paired 차이의 탐색적 구간을 계산한다. fitting 불확실성을 재표집한 확증 구간이나 다중비교 보정 결과가 아니다.

각 target에서 따로 학습한 LR가 서로 반대되는 환자군 연관성을 학습할 수도 있으므로, AUC 개선만으로 target 학습 참여가 만든 신호를 확정하지 않는다. 이를 구별하는 보조 분석으로 **같은 fitted scorer를 고정한 교차-target 진단**을 사전 선언했다. fit-target의 scaler/LR/C를 그대로 두고 같은 selection 환자의 두 target 특징에 적용해 `참여 target 점수 − 비참여 target 점수`의 평균·양수 수·공통 bootstrap 구간을 기록한다. 다른 target에서 재학습하지 않는다. 이는 두 모델을 공격 입력으로 사용하는 새 방법이 아닌 분석자 진단이며, 환자군 전체가 교환된 두 모델 비교이므로 개인 한 명의 인과 효과로 해석하지 않는다.

selection40은 이미 사용한 개발 자료다. 이번에는 calibration/test를 열거나1% 환자 FPR 성능을 주장하지 않았다. 후속 E/U 네 셀·환자군 Welch8개까지 완료했지만 MoFit 환자군 성능 비교와 개인 참여에 특이적인 실패 원인 확인은 남아 있다. **지정 비교의 완료와 2번 전체 완료를 구별한다.**

[완료 결과](CDI_U_COMPARISON_RESULTS.md)의 primary CDI−SecMI ΔAUC는 +.005/−.0125, primary meanmax52−image-mean Δ는 +.0825/+.0575이며 각각 두 target의 구간이0을 포함한다. 고정 C의 표준 meanmax52 개선과 NO27 모델2의 양성 관측도 그대로 보고했다. 같은 고정 scorer의 참여방향 평균 변화 구간은 모두0을 포함하지만, 평균 이동만으로 AUC 개선 원인을 cohort bias로 확정할 수는 없다.

**동일 scorer의 공통 점수/target 변화 순위 분해와 실제 base U80장 대조까지 완료했다.** 표준 meanmax52fixed의 base A/B AUC는 .555/.570,자기 target은 .6325/.670이며 각 차이의 구간은0을 포함한다. 모델2fixed의 상대 집계 이득 변화+.170 [.0275,.3225]도 함께 보고한다. 참여 효과나 base 편향이라는 원인 확정은 아니다. base는 이번 추가학습 membership이 모두0이므로 A/B 배정 구분 지표로만 읽고, scorer가 보유한 target-fit 정보와 base의 사전학습 unknown을 명시한다.

평균/half-range 후속 분해도4개scorer·3출처·36AUC의 독립 검산까지 완료했다. 차이 항을 제거하면 자기 target fixed AUC는 .5450/.5775로 내려가지만 두 차이 구간 모두0을 포함한다. 반대 target에서도 원래 환자군 구분이 남았으므로 그 점수를 보편적인 member-positive 신호로 읽지 않는다. 다음은 이 구분과 실제 참여를 나눌 대조의 필요성·예측·비용을 구체화하는 2번 작업이다.

동결 근거: [추출 계약](../../code_working/_reports/cvpr_u_pilot_v1_001/baseline_screen_20260914/cdi_u_cohort_contract_v1.json), [분석 계약](spec_sources/cdi_u_analysis_contract.json), [분석기](../../code_working/u_patient_audit/analyze_cdi_u_cohort.py).

- 추출 계약 SHA256: `a479363e9c094fbbff37af5d0d6c6e087d8dd191082baa73824318b0e164ab45`
- 분석 계약 SHA256: `6f0afa62cf2bbf26790370965c4716f3afc927e0ab2412f8c4e6befbe02efca3`
- 분석기 SHA256: `752be6cdf1db57b2309bd897fee8f43f53f1807d23fa92cb07973b4720d14268`

근거 메모: [MoFit](spec_sources/mofit_patient_spec_notes.md), [CDI](spec_sources/cdi_patient_spec_notes.md), [집계·공정성](spec_sources/aggregation_and_fairness_notes.md), [독립 kernel 검산 요구](spec_sources/cdi_kernel_review_requirements.md).
