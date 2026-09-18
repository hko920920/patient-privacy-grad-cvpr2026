# PRRD 단계별 구현·논문 완성 계획 — 2026-09-18

**2026-09-18 C128 비용 측정 갱신:** [공개 C bank128장 실측](PRRD_C128_PUBLIC_PROFILE_RESULTS_20260918.md)을 완료했다. Microbatch4, warm-up5/측정10에서 전체 update 평균7.8364초, allocated peak1732.42MiB. 이번은4단계 중 이 측정 하나만이며 계수·본실험 규모·예산은 동결하지 않았다. 현재 상태는 §13이 우선한다.

**2026-09-18 3단계 후속 갱신:** [관계 개입 제한·기능 정합 연결](PRRD_STEP3_GUARDED_OBJECTIVE_RESULTS_20260918.md)을 완료했다. 실제 BioViL19조건이 기존 gradient/loss 기준을 통과했다. 계수는 수치 검사 전용이며 본실험 선택이 아니다. 이번에는3단계에서 멈췄고 전체128장 profile·계수/예산 동결·본 합성은 진행하지 않았다. 현재 진행 상태는 §12가 우선하며 이전 단계 문구는 이력이다.

**2026-09-18 2단계 후속 갱신:** [공개 두 특징 경로](PRRD_STEP2_FEATURE_PATHS_RESULTS_20260918.md)와 환자 통계 연결을 완료했다. 기존 점별 출력은 정확히 유지했고 raw 특징은 새로 추출했다. 이번에는 여기서 멈췄으며 다음은 사용자3단계의 학습 규칙 연결이다. 계수·30bank는 제안 상태이고 전체 runtime/W1은 미동결이다. 현재 진행 상태는 §11이 우선한다.

**2026-09-18 후속 갱신:** 사용자의 1단계인 기존 목적 W1 gradient 수리를 완료했다. [수리 결과](PRRD_W1_GRADIENT_REPAIR_RESULTS_20260918.md)의 공개 18조건이 기존 기준을 통과했다. 새 방법의 전체 W1은 아직이며 이번에는 여기서 멈췄다. 아래 최초 계획의 미통과 표시는 당시 상태이며 현재 순서는 §10이 우선한다.

상태: 실행 계획 작성 및 첫 learner core 구현 완료. 전체 runtime 동결·합성 효용·DP 성능·논문 완성을 보고하는 문서가 아니다.

목표는 PRRD의 보존 정보, 실제 학습 규칙, 공개 가능한 합성자료, 다른 수신자, 환자-DP 평가를 같은 근거 체계로 연결한 원고와 재현물을 완성하는 것이다. 성능 우위나 학회 채택을 미리 확정하지 않는다. 결과와 다른 주장을 만들지 않는다.

## 1. 작업 원칙

- 주력·데이터 역할을 다시 고르지 않는다. PRRD, 전체 Q 점별 정보, mixed 관계 정보, 현재 source/recipient를 유지한다.
- 근거 없이 실험을 추가하지 않는다. 아래 단계에서 정한 산출물과 미해결 구현 항목을 하나씩 닫는다.
- 구현 오류 수정과 방법 변경을 구분한다. 오류를 고친 뒤 같은 검사를 다시 수행할 수 있지만 허용오차·seed·단계를 결과에 맞춰 완화하지 않는다.
- 계획 → 구현 → 의미 있는 검사 → 증거 저장 → 상태 갱신을 한 단위로 완료한다. 검사항목 수를 연구 성과로 세지 않는다.
- CPU core PASS, 실제 encoder W1 PASS, nonDP 효용, DP 효용을 따로 기록한다.
- Source 성능만으로 다른 encoder 전달을 자동 실패 처리하지 않는다. V는 합성 loss·gradient·checkpoint 선택에 쓰지 않는다.
- Expert532·Reserved4213은 현재 닫혀 있다. Q/V와 기존 사용 이력도 그대로 유지한다.
- 하나의 단계가 끝나면 같은 범위의 다음 구현을 이어간다. 큰 실행의 숫자 시간 상한은 기존 계약대로 실제 profile 후 결속하며, 일반 구현 수정마다 다시 허락을 묻는 절차는 만들지 않는다.

## 2. 이번에 정한 계수 규칙

기존 \(\beta=1,\lambda=0.1\)을 유지한다. 관계를 사용하지 않는 readout은 \(\beta=0\)이다.

관계 허용량은 절대값을 모델마다 임의로 넣는 대신 다음 공개된 규칙으로 정한다.

\[
\rho^2=\kappa\,w_0^\top M w_0,\qquad \kappa=0.1,\qquad M=G+\lambda I.
\]

점별 목적에서 \(F_0(0)-F_0(w_0)=\tfrac12w_0^\top M w_0\)이므로,
\[
F_0(w_*)-F_0(w_0)\le
\kappa\{F_0(0)-F_0(w_0)\}.
\]

즉 영점 분류기 대비 점별 목적 개선량의 10%를 관계 반영의 최대 희생량으로 둔다. Source/합성/수신자는 같은 \(\kappa\)와 자신의 점별 목적을 사용한다. \(w_0=0\)이면 허용량도 0이고, 관계 수정이 막히는 보수적인 대가를 명시한다. \(\rho\) 자체가 모든 learner에서 같은 숫자일 필요는 없다.

기능 정합 가중치는 \(\eta=1\)로 둔다. 기존 통계 손실과 영상 prior/TV/augmentation 계수는 유지한다. 기능 손실은 고정 target \(M_T\)와 target guarded weight를 사용하며, 무증강 학습기능에 한 번 적용한다. 기존 증강 정규화는 모멘트 정합으로 유지해 기능 항을 조용히 중복 가중하지 않는다.

이 수치는 **이번 구현용 사전 시작값**이다. 환자 성능을 보고 선택한 값이나 최적값이 아니다. 새로운 scale/HPO 탐색을 자동 추가하지 않는다. DP에서는 이 규칙을 보호된 요약에 적용하고, 비DP Q 결과로 선택한 값을 무료 공개 상수로 재분류하지 않는다. 전체 이미지 runtime은 아직 미검증이므로 실행 계약 상태는 FROZEN이 아니다.

## 3. 현재 실제 상태와 코드 차이

| 항목 | 현재 상태 | 다음 반영 |
|---|---|---|
| 공개 BioViL/Dense 특징·PCA16/128 | 기존 공개 추출 완료 | 원 해시 보존, 필요한 raw source만 별도 결속 |
| 실제 BioViL one-pass/two-pass | 이전 허용오차 미통과 | 원 실패 보존, 최초 차이가 생기는 계산 위치 수정 |
| 관계 통계 코드 | 기존 endpoint delta, clipping 전 평균 불변 assertion | E/E_R/C/D/R_joint별 raw/point 경로와 bound 후 의미로 수정 |
| renderer·계약 arm 목록 | 기존5조건 | 문서의7조건과 별도 제거 비교에 맞게 갱신 |
| 개입 제한·기능 손실 | 이번 독립 core 구현 완료 | 실제 bank objective/receiver에 같은 함수 연결 |
| run orchestration/readout/통계 | 전체 연결 미완료 | resume·hash·모든 bank 동결 후 평가 구현 |
| DP release | 미구현·미실행 | nonDP 판단 후 별도 sampler/accounting 계약 |
| 원고 | 이번 working manuscript·claim ledger 착수 | 매 단계 실제 표/그림과 함께 갱신 |

기존 코드와 원본 입력·run 목록을 그대로 두고 새 core 파일만 추가했다. 이번 모듈을 import하지 않는 이전 runner가 새 방법을 실행한다고 표시하지 않는다.

## 4. 단계와 완료 조건

| 단계 | 실제로 할 일 | 완료 증거 | 현재 |
|---|---|---|---|
| S0 설계·작업 범위 결속 | §19를 기준으로 계수 규칙·역할·비교·미구현 항목 기록 | 이 계획, machine-readable plan, 코드/입력 해시 | 완료 |
| S1 learner core | guarded solve, fixed-target 기능 loss, zero/분기 처리 | 독립 NumPy·유한차분·replay 검사6개 | CPU 범위 완료 |
| S2 공개 이미지 연결 | raw/point 두 경로, arm별 관계/renderer/objective, 전체 bank gradient, PNG·resume·legacy RN18 연결 | 실제 모델 W1 보고서와 전체128장 처리량 | 다음 작업 |
| S3 비DP 통합 비교 | 고정 core와 최소 제거 비교를 전부 생성·동결한 뒤 일괄 평가 | 모든 bank·prediction·patient interval·비용표 | 미실행 |
| S4 가까운 강한 대조 | native-feature 점별 증류·유효한 직접 관계 대조·직접 predictor 구현 범위 확정 후 비교 | baseline card, 동일 접근권/예산 표, 모든 결과 | 미실행 |
| S5 환자-DP | bounded query·보안 난수·noisy surrogate·release/composition·동일 수신자 연결 | privacy receipt/ledger 및 DP utility-cost 표 | 조건부·미실행 |
| S6 동결·최종 캠페인 | 모든 방법/선택 이력/수신자/통계를 고정하고 허용된 final 평가 | freeze manifest와 최종 결과 | 현재 접근 없음 |
| S7 원고·재현물 완성 | 본문·수식·표·그림·부록·방법/배포 카드·실행 명령 정합 | 실제 결과를 인용하는 전체 원고, 누락 없는 재현 목록 | S0부터 병행 |

S0/S1의 완료는 전체 연구 단계2 완료가 아니다. 논문 완성까지를 한 번에 실행 승인된 GPU 작업으로 해석하지 않는다.

### S2: 바로 다음 한 작업

1. 동일한 encoder 실행에서 point와 pre-wrapper raw 특징을 얻고, 기존 point projection은 보존한다.
2. P-only raw 중심/관계 scale을 명시된 공개 mixed3 규칙으로 결속한다. 새로운 PCA나 V 성능 선택을 끼워 넣지 않는다.
3. E/E_R/C/D/R_joint 통계와 합성 bank 경로를 구현한다. D의 point 불변과 clipping 전 평균 불변만 assert하고, clipping 후 평균·2차 변화를 기록한다.
4. S1의 learner를 objective에 연결한다. 무증강 통계 + \(\eta J_{\mathrm{func}}\) + 기존 영상/증강 항의 gradient를 한 번에 계산하고 같은 전역 response를 replay한다.
5. 공개 소수 입력에서 feature 값 → moment response → image gradient → renderer gradient 순서로 기존 불일치의 최초 위치를 찾는다. 오차 기준을 통과할 때까지 느슨하게 바꾸는 방식은 쓰지 않는다.
6. 같은 상태 resume, PNG decode/label/pair 결속, frozen parameter/BN, 수신자 배제, RN18 공개-only 연결을 한정된 검사로 묶는다.
7. 통과 후 전체128장 two-pass update를 20 warmup/30 timed로 측정한다. pure microbatch 시간으로 전체 bank ETA를 만들지 않는다.

산출물: 버전 결속된 runtime contract, 실제 W1 PASS/FAIL, profile, 아래 S3 명령과 예상 시간. 작업 예상은 구현·수정 2–4시간이며 GPU 실측 예측이 아니다. 구체적인 미해결 오류가 남으면 그 단위와 원인을 기록하고 같은 단계에서 수정한다. 연구 주제를 바꾸는 계기로 사용하지 않는다.

## 5. S3의 고정 범위: core와 보강 효과를 분리

### 핵심 비교

A/B/E/E_R/C/D/R_joint × 초기화101/202/303 =21bank.

A/B에도 기능 정합을 적용한다. 관계를 쓰는 E/E_R/C/D/R_joint에는 같은 개입 제한과 기능 정합을 적용한다. 통계·자료 차이만으로 핵심 비교를 해석한다.

### 두 보강의 최소 제거 비교

C에서만 추가로 세 variant를 둔다.

| variant | 관계 제한 | 기능 정합 |
|---|---|---|
| C_main | on | on |
| C_no_guard | off | on |
| C_no_func | on | off |
| C_original_rule | off | off |

C_main은 핵심21bank에 이미 포함된다. 나머지3variant×3초기화 =9bank만 추가한다. 관계 구성·point 정보·template·augmentation·500step은 동일하며, 각 variant는 자신의 학습 규칙으로 target/synthetic/recipient를 계산한다.

**새 S3 제안 총량:30bank,3,840PNG,15,000synthetic successful updates.** 이전21bank 또는 원v3의15bank를 승인된30bank로 소급하지 않는다. 이 확장은 보강 효과를 분리하기 위해 이번에 사전 열거한 것이다.

Source/primary recipient의 \(\beta0/\beta1\) readout 구성은 총108개다: 기존 핵심72 + 추가9bank×2encoder×2beta. 이는 학습 결과의 개수이며 내부 선형계 solve 호출 수와 같지 않다. 실자료 참고는 핵심18개에 C의 제한 전 분류기2encoder를 추가한20개이고, 동일 통계를 재사용한다. RN18 호환성은 핵심21bank×3seed=63run×400=25,200updates만 유지한다. 추가 제거 비교9bank에 RN18을 자동 추가하지 않는다.

Q raw features/statistics는 허용된 내부 저장소에서 한 번 추출해 해당 targets를 만든다. 실제 환자 pixel/통계를 합성물에 공개하지 않는다. 모든30bank 최종 산출물을 고정한 뒤 수신자 V 평가를 연다. Source real readout은 설명 자료이며 새 source-only 중단 규칙이 아니다.

개발 patient-cluster paired bootstrap은 기존2,000draw, AUROC primary/AP secondary다. 지표별 model 평균과 합성 seed·환자 sampling 변동을 구분하고 좋은 seed만 고르지 않는다.

## 6. 판정과 수정의 경계

기존 중심 비교와 공정성 조건을 유지한다.

- C−B: 관계 표현·학습 규칙 전체의 추가 가치.
- C−D: 실제 사적 환자 대응의 가치.
- C−E_R: 단순 재보정 이상의 순서 변경 가치.
- C−R_joint: 가까운 표준 관계 정합 대비 가치·비용.
- C의2×2 제거 비교: 관계 개입 제한과 기능 정합 각각의 역할.
- DP C−강한 pointwise DP: 관계 보호 비용을 감수한 이득.
- E_t/E_t2: source에서 만든 시각 자료의 다른 encoder 사용 가치.

기존 engineering 기준 C−B≥.01, C−D>0, 각2/3 방향 일치, AP 비감소와 별도 순서 기준은 유지한다. 새 cap/기능 loss가 잘 맞았다는 이유로 이를 대신하지 않는다.

문제 유형별 다음 행동도 고정한다.

| 관측 | 행동 |
|---|---|
| 구현/gradient/입출력 오류 | 같은 명세의 오류 수정 후 해당 검사, 원 실패 보존 |
| 실제 recipient에서 관계 이득 없음 | 계획된 전체 결과·대조·오차를 보고하고 자동 DP 확대나 새 loss/seed 추가를 하지 않음 |
| source만 성공 | source fidelity로 범위를 제한, 시각 전이 성공을 쓰지 않음 |
| C≈E_R | 관계 효과와 순서 효과를 구분하고 순서 우위를 주장하지 않음 |
| C≈R_joint | 표준 관계 정합과 같은 설명을 인정하고 효율/보호/적용 범위만 실제 결과로 판단 |
| cap이 관계 효과를 거의 없앰 | alpha·상한·제거 비교를 함께 보고; 결과 뒤 허용량을 임의 확대하지 않음 |
| core 양성 | S4의 강한 대조와 S5의 사전 고정 DP로 진행 |
| DP에서 이득 소실 | 비DP 결과와 보호 후 한계를 모두 원고에 남김 |

추가 방법 개발이 필요하다면 이번 고정 결과와 구분된 계약으로만 한다. 이 계획의 음성 결과를 지우거나 자동적인 진단 연쇄로 이어가지 않는다. 결과가 핵심 주장을 지지하지 않아도 수행한 연구의 전체 분석·원고·재현물 정리는 끝까지 완료하며, 우위 논문이나 채택 가능성으로 포장하지 않는다.

## 7. 강한 선행·DP·final의 진입 조건

S4에서는 가장 가까운 직접 원리를 설명할 비교를 고정한다. 전체 논문을 전수 구현하는 일이 아니다.

- native source feature를 쓰는 강한 점별 증류: 가능한 LGM 계열 구현의 실제 objective/source/license/변형 범위를 baseline card에 고정한다. B16만을 최강 점별 방법이라고 부르지 않는다.
- R_joint 및 표준 차분 모멘트: C와 동일한 경우 별도 경쟁군을 만들지 않고 등가성으로 보고한다.
- 직접 point/관계 readout과 환자-DP classifier: 단일 task에서 이미지를 만드는 비용을 숨기지 않는다.
- DP-MEPF/DP-KIP/Dosser 중 접근권·공개모델·환자 보호단위에 맞는 직접 구현 하나를 원문/코드 대조 후 고정한다. 선택 기준은 실행 가능성과 근접성이고 V 성능 순위가 아니다.

S5 기본: B/E_R/C/R_joint × 독립 noise3 × init3 =36bank,12개의 사적 release, \(\varepsilon8,\delta10^{-5}\). 같은 release의3init는 후처리다. \(\varepsilon4\)는 자동 실행하지 않는다. 강한 외부 baseline의 추가 run과 전체 공개 composition은 DP 계약에 별도로 포함한다. 기존36개만으로 외부 baseline까지 했다고 표시하지 않는다.

독립 보안 난수, noisy normalization, 고정 SPD repair와 guarded objective 일치, exact-count/비보호 HPO 경계, 배포 금지 private metadata를 구현한다. 기록만으로 secure DP 구현 완료를 선언하지 않는다.

S6는 모든 방법·HPO/선택·수신자·bank hash·release·metric family가 고정된 뒤다. Expert/Reserved를 feasibility·tuning·privacy gallery로 먼저 열지 않는다. 계획된 별도 final 접근 계약을 유지한다.

## 8. 시간·재현성·원고 완성의 정의

S3 예상 계산시간은 다음으로 산출한다.

\[
T_{S3}=T_{\mathrm{allowed\ private\ features}}
 +30\times500\times t_{\mathrm{full128\ synthesis}}
 +63\times400\times t_{\mathrm{RN18}}
 +T_{\mathrm{readout,export,bootstrap}}.
\]

Cold/warm 준비, public/raw feature, private summary, 합성, PNG, recipient, 통계 시간을 분리한다. 실제 profile 전 임의의 총 GPU시간을 확정하지 않는다. 숫자 시간 상한을 결속한 뒤 manifest에 열거된 작업만 실행한다. 중단/resume은 동일 상태를 보존하며 checkpoint 고르기로 바꾸지 않는다.

원고 작업은 마지막에 시작하지 않는다.

- S0/S1: 문제·방법·정리 범위·기여 가설·평가표 구조 작성.
- S2: 실제 구현·전처리·계산량·재현 명령으로 Method를 수정.
- S3: 실제 recipient 효과·제거 비교·source 오차·PNG 손실을 표/그림으로 연결.
- S4/S5: 가장 강한 반론, DP composition·utility·전체 비용 반영.
- S6: final과 개발 결과의 역할을 구분해 Results/Limitations 확정.
- S7: 빈 결과칸 제거, 모든 수치의 artifact 연결, 모든 claim에 해당 증거 연결, 코드/환경/방법·배포 카드 정리.

완료는 ‘방법을 구현했다’가 아니라 **결과에 맞는 전체 원고 + 표·그림 + 부록 + 재현 명령·설정 + privacy/selection ledger**다. 학회 제출·공개는 완성된 구체적 산출물 기준으로 진행한다.

## 9. 이번 실제 완료

- 최신 설계와 현재 코드의 차이를 대조했다.
- guarded_readout.py와 test_guarded_readout.py를 추가했다.
- CPU 검사6개 PASS, 약0.600초. 직접 quadratic 항등 오차 \(6.97\times10^{-16}\), 기능 loss 대 직접 점수 오차 \(8.67\times10^{-19}\), toy full/replay gradient 최대 차이0.
- 유한차분으로 active cap/solve gradient를 확인했고, zero radius/point, invalid SPD, target metric 고정, microbatch1/3/5/16을 포함했다.
- 실제 의료 encoder/input·환자정보·DP sampler를 사용한 검사가 아니다. W1 실제 모델 미통과 상태를 PASS로 바꾸지 않았다.
- 새 GPU·환자영상·합성 최적화·DP·final 실행0. 기존 코드 파일·원본 계약·Rwide/LoRA 결과 보존.
- 다음 한 작업은 S2의 raw/point/arm/functional 경로 실제 연결과 공개 모델 W1 수정이다.

연결 파일: [기준 설계 §19](PRRD_CONSOLIDATED_DESIGN_20260918.md), [기계 판독 계획](spec_sources/prrd_paper_delivery_plan_20260918.json), [working manuscript](paper_prrd/manuscript.md), [claim ledger](paper_prrd/claims_evidence.md).


## 10. 사용자 8단계 계획에 따른 첫 수리 완료

기존 목적과 실패 입력을 유지한 실제 BioViL gradient 수리만 수행했다. 결과는 [별도 보고서](PRRD_W1_GRADIENT_REPAIR_RESULTS_20260918.md)와 [기록 JSON](spec_sources/prrd_w1_gradient_repair_record_20260918.json)에 결속했다. 과거 실패를 지우지 않고 reference의 renderer·augmentation·encoder 분할을 replay와 일치시켰다. 전체 S2 또는 새 설계 W1을 통과 처리하지 않는다.

후속 순서는 기존 목적 수리(완료) → raw/point·환자통계 경로(다음) → guarded learner와 기능 loss 연결 → 전체 128장 profile/실행 계약 결속이다. 이번 작업은 1단계 완료 후 중단했으며, 새 목적을 동시에 넣지 않았다.

첨부의 절대 rho=.1 및 profile5+10은 현재 상대 반경·profile20+30과 다른 권고값이다. 이번에는 계수/처리량 계획을 바꾸지 않았다. 30bank 및 수신자/DP/final 실행 승인 범위도 늘리지 않았다. 새 환자 성능은 없고 실제 Rwide/LoRA 결과는 보존한다.


## 11. 공개 점별·raw 관계 경로 연결 완료

사용자가 지정한2단계만 구현·검증했다. BioViL projected_global_embedding을 wrapper의 외부 L2 전에 추출하고, 기존 point PCA16·전처리·bound는 그대로 유지했다. P813장의 정규화 cache·point 출력·환자 point 통계가 기존과 정확히 같았다. 같은 공개 PCA축의 raw 선형 특징에서 환자 대조를 만든 뒤 제한하는 경로와 저장 특징의 독립 검산이 통과했다.

[실제 결과](PRRD_STEP2_FEATURE_PATHS_RESULTS_20260918.md)와 [기록 JSON](spec_sources/prrd_step2_feature_paths_record_20260918.json)을 기준으로 한다. Raw 중심은 label-free 환자 균등 P 평균이며, 관계 scale은 공개 mixed3명의 inverse-CDF q95/no interpolation이다. 이 수치 결속은 learner의 rho/eta 선택과 다르다.

전체 S2 가운데 기존 gradient 수리와 source 두 특징 경로까지 완료했다. 다음은 사용자3단계의 guarded learner·기능 손실 연결이며 이번에는 수행하지 않았다. 수신자 실검증·전체 bank gradient/profile·Q/V 본 비교는 남아 있다. 기존 계수/30bank 값은 제안으로 보존하며 실행 계약 동결로 표시하지 않는다.

## 12. 관계 개입 제한·통계 및 기능 공동 정합 연결 완료

사용자3단계만 수행했다. 목표와 합성자료가 같은 learner policy를 사용하되 각자의 점별 행렬로 관계 개입을 제한하며, 기능 정합은 고정 target M_T를 사용한다. 기존 통계 정합/영상 규제를 유지하고 무증강 경로에만 기능loss를한번적용했다. 옛 objective/learner/encoder와2단계public통계는보존하고 새목적은별도API로연결했다.

[결과보고서](PRRD_STEP3_GUARDED_OBJECTIVE_RESULTS_20260918.md)와 [실행기록](spec_sources/prrd_step3_guarded_objective_record_20260918.json)이 근거다. 새구성배열6+기존core6, 실제BioViL19조건을검증했다. PixelVJP차이0, rendererparametergradient상대L2최대2.27e-7, loss최대차이1.14e-13. 기존허용오차를변경하지않았다. 기능loss단독으로도이미지gradient가유한·비영이다.

사용자단계별상태는1단계기존gradient수리완료→2단계두특징/환자통계완료→3단계새목적연결완료→4단계계수·full128profile·실행량동결예정이다. 이문서의상위S2는남은profile/export/resume/runner/수신자연결범위를포함하므로전체완료로바꾸지않는다.

검사에서beta1/ridge.1/eta1/kappa.1및edge용rho0/100을썼지만본실험계수를선정한것은아니다. P고유4장/8decode,460F/308Bimagepresentations,optimizer0. 수신자/Q·V성능/DP/final을실행하지않았다.30bank는기존제안상태다. 여기서중단했으며다음단계를자동실행하거나예약하지않는다.

## 13. 공개 C128 전체 update 비용 측정 완료

사용자는4단계 전체 대신 공개자료의 C128 profile 하나를 지정했다. [사전 계약 및 실측 결과](PRRD_C128_PUBLIC_PROFILE_RESULTS_20260918.md), [기록 JSON](spec_sources/prrd_c128_public_profile_record_20260918.json)을 기준으로 한다. Microbatch4, warm-up5+측정10은 이번 측정 전에 고정한 별도 실행 범위이며, 이전 전체 profile20/30 제안을 실행했다고 표시하지 않는다.

Beta1/ridge.1/kappa.1/eta1은3단계 기본 수치 검사용 설정을 유지했다. 검증된 새 two-pass 목적에서 모든128장 forward/global statistics/function/backward와 optimizer 갱신을 매번 포함했다. 평균7.8364초/update, allocated1732.42MiB/reserved1852MiB를 측정했고15회 모두 finite 및 실제 갱신·고정 source/target 불변을 확인했다.

본실험 계수나30bank 규모·전체시간 상한은 아직 미확정이다. 이번 측정용 상태는 본 bank로 export하거나 재사용하지 않았다. 다른arm/수신자/DP/final을 진행하지 않았고 여기서 멈췄다. 다음 판단은 이 비용을 바탕으로 한 본 계수·규모·예산이며 자동 실행은 없다.

## 14. 첫 비DP 실행 계약 고정 — 실행은 미승인

사용자 후속 요청에 따라 [단일 실행 계약](PRRD_FIRST_NONDP_EXECUTION_CONTRACT_20260918.md)에서 첫 시작값과30bank 범위를 확정했다. Beta1(A/B0),ridge0.1,kappa0.1의 상대 반경,eta1,128장/500회,seeds101/202/303을 사용한다. 이전 ‘미확정/검사용’ 문장은 각 단계 당시 상태이며, 이번 선택은 성능 최적값 검증이 아닌 prospective recipe 결정이다.

C128실측의 단순 환산은 C500=65.30분,30bank=32.65시간이다. 전체 비용이나 승인된 시간 상한이 아니다. 기존 실측만 사용했으며 다른 profile은 추가하지 않았다.

30개 ID·code/public/target-rule SHA 및 bank별 checkpoint/resume/최종 PNG 규칙은 위 계약을 따른다. recipe는 고정됐으나 image runtime 전체가 준비된 것은 아니다. main runner와 동일 상태 resume/export 연결, 허용 bank/숫자 시간 상한, 허용된 준비 후 Q feature/target 실제 hash가 남았다. 사용자3단계 gradient와 C128 profile의 PASS 범위를 확대하지 않는다.

이번에는 계약·상태 기록만 수행했다. 모든 실행 승인false, DP/Expert/Reserved 잠금,profile bank 재사용 금지와 기존 실제 효용 결과를 유지한다. source가 약하면 모든 합성을 자동 중단하는 새 조건도 추가하지 않는다.
