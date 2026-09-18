# PRRD 첫 비DP 실행 계약 — 2026-09-18

상태: **시작값·방법 범위 고정 / 실행 미승인 / 본 실행기 연결 미완료**.
계약 ID: PRRD-NONDP-20260918-v1. 연구의 큰 단계는 2를 유지한다.

이 문서는 사용자에게 전달받은 809ff05 비용 보고 검토와 현재 로컬 코드·저장 산출물을 바탕으로 첫 비교의 실행값을 고정한다. 이번에는 원격 commit을 새로 조회하지 않았다. 로컬은 Git checkout이 아니므로 파일 SHA-256을 실제 코드의 기준으로 삼는다. 본문과 부록을 합쳐 하나의 계약이며, 이전 제안의 이력과 실제 결과를 덮어쓰지 않는다.

**새 학습·profile·픽셀 읽기·수신자·DP·Expert/Reserved 실행은 모두 0이다.** 계수는 성능이 검증된 최적값이 아니라 첫 비교 전에 선택한 시작값이다. 과거 수치 검사·profile 당시의 ‘검사용 설정’이라는 기록도 그대로 보존한다.

## 1. 확정값과 승인 범위

| 항목 | 첫 실행값 |
|---|---|
| 관계 arm beta | 1 |
| A/B beta | 0; 관계 통계·관계 loss 없음 |
| ridge lambda | 0.1, intercept 없음 |
| 관계 개입 제한 | rho² = 0.1 × w0ᵀ M w0, kappa=0.1 |
| 기능 정합 eta | 1; 지정된 no_func 제거실험만 0 |
| 영상/업데이트 | 128장 / 500 successful updates |
| 초기화 | 101, 202, 303 |
| 기본 비교 | A/B/E/E_R/C/D/R_joint, 각 3회 = 21bank |
| C 제거 비교 | 제한만·기능만·둘 다 없음, 각 3회 = 추가 9bank |
| 총 계획량 | 30bank, 3,840장, 15,000 합성 optimizer updates |
| microbatch | 4; 기존 C128 실측과 동일 |
| precision | encoder·이미지 FP32, 작은 통계·readout FP64, AMP 없음 |
| optimizer | AdamW lr=0.01, wd=0, betas=(0.9,0.999), eps=1e-8, foreach=False |
| 채택할 산출물 | 각 bank의 마지막 500-step; best checkpoint 선택 없음 |

고정된 설계 규모와 실행 권한은 다르다. 현재 authorized_run_ids=[], execution_authorized=false, runtime_cap_seconds=null이다. 30bank 전체 및 그 일부도 이번 작업에서 실행하거나 예약하지 않는다. 약 32.7시간은 승인된 예산이 아니다. 후속 실행 허용 범위와 숫자 시간 상한은 별도 기록으로 결속한다.

## 2. 학습 규칙과 목적

점별 목적의 M=(A0+A1)/2+lambda I, b=(m1-m0)/2에서 w0=solve(M,b)를 구한다. 관계 arm은 w1=solve(M+beta C,b+beta mu), d=w1-w0를 계산한다.

guard 활성 시 rho²=kappa·w0ᵀM w0, alpha=min(1,sqrt(rho²/(dᵀM d))), w*=w0+alpha d다. d=0은 w0, rho=0은 관계 수정량 0으로 처리한다. 실제 구현의 안전한 zero 분기를 유지한다. guard 미사용 제거실험은 w1을 사용한다. A/B는 관계항이 없어 w1=w0이며 guard 플래그가 결과를 바꾸지 않는다.

제한하는 것은 명시된 점별 ridge 목적의 악화량이다. F0(w*)−F0(w0)≤rho²/2이며, 상대 반경은 F0(0)−F0(w0)의 10%에 해당한다. 고정 rho=0.1로 바꾸지 않는다. AUROC 악화 상한이나 새로운 환자에 대한 성능 보장이 아니다.

목표·합성·수신 readout은 같은 정책을 각자의 통계에 적용한다. 합성 learner 내부 제한은 M_S, 기능 차이는 **고정된 M_T와 w_T**로 계산한다.

- J_stat: point 평균/2차 모멘트 제곱오차의 계수 1/4; 관계 평균/2차 모멘트는 1/2.
- R_joint는 관계 정합에서 joint u/U를 사용하고, readout의 mu/C는 이미 제한된 u/U의 선형 투영으로 얻는다.
- J_func=(w_S−w_T)ᵀ M_T (w_S−w_T).
- 전체 loss = J_stat(clean) + 0.1 J_stat(augmented) + eta J_func(clean) + 0.01 pixel_anchor + 1e-4 TV.
- 기능 정합은 무증강 경로에 한 번만 적용한다. 목표 통계·M_T·w_T는 detached 사본으로 고정한다.
- 모든 arm은 같은 기능 정합을 받으며 관계 baseline도 같은 제한 규칙을 받는다. 명시된 C 제거실험만 예외다.

모든 moment 분모는 전체 bank/class/pair 기준이다. microbatch별 분모 재계산이나 prior 중복 누적을 하지 않는다. 전체 response 계산과 replay에서 같은 renderer 분할·증강 상태를 사용하고, 누적이 끝난 뒤 한 번만 optimizer를 갱신한다.

## 3. 자료·특징·목표 생성 규칙

기존 역할/사용 이력과 cohort 수를 유지한다. 이번에는 기존 명부 파일의 hash만 재확인하며 환자 픽셀을 열지 않는다.

| 역할 | 환자/영상 | 계약상 위치 |
|---|---:|---|
| P | 672 / 813, mixed 3 | 공개 projection·scale·초기 template·공개 목표 |
| Q | 2,027 / 5,097, mixed 102 | private_train, 기존 selection/Rwide 사용 이력 유지 |
| V | 2,026 / 5,047 | 반복 사용한 개발 평가; 독립 final 아님 |
| Expert / Reserved | 532 / 810; Reserved 4,213명 | 잠금 유지, 본 계약으로 접근 불가 |

Q80 과거 결과 및 head/LoRA 실패·Rwide 실자료 결과는 별개로 보존한다. Q를 public으로 재분류하지 않는다. 본계약은 private 통계나 환자 식별자를 공개할 권한을 부여하지 않는다.

Source는 결속된 BioViL-T 공식 image checkpoint의 projected_global_embedding이다. 전처리는 기존 differentiable short-side512/center448/grayscale3 경로다. 기존 point 특징 z는 외부 L2→기존 공개 중심/PCA16→공개 norm bound를 그대로 사용한다. Raw 관계 특징 a는 **외부 L2 전 h − 공개 환자 균등 raw 중심**을 같은 기존 공개 PCA축에 투영한다. 정규화 cache를 raw cache로 재명명하지 않는다. Source는 eval/frozen이며 합성 input gradient는 유지한다.

환자 안의 class별 영상 평균/2차 모멘트를 먼저 계산하고, class를 가진 환자를 같은 질량으로 평균한다. P+Q는 환자별 합과 class count를 합친 뒤 정규화한다. 공개 평균과 사적 평균을 절반씩 섞지 않는다. 빈 class/mixed 항은 기존 코드의 0 통계 규칙을 사용한다.

| arm | 점별 목표 | 관계 목표 |
|---|---|---|
| A | P 전체 | 없음 |
| B | P+Q 전체 | 없음 |
| E | B와 동일 | bounded point endpoint의 환자 centroid 차분 / 2 |
| E_R | B와 동일 | E 차분을 공개 scale로 다시 제한 |
| C | B와 동일 | raw affine centroid 차분 / 2, 이후 공개 scale로 제한 |
| D | B와 동일 | Q mixed의 negative centroid 대응만 derangement; 이후 C와 같은 제한 |
| R_joint | B와 동일 | raw endpoint u=[positive;negative]/sqrt(2)를 공동 제한; 그 평균/2차 정합 |

공개 inverse empirical-CDF q95/no interpolation scale을 그대로 결속한다: E_R=0.29153658488143597, C=0.2650091310071505, R_joint=1.2582928166602765. 공개 mixed 3명에 근거한 고정 규칙이며 최적 보정이 입증된 것은 아니다.

D는 공개 pair 3명을 그대로 두고 Q mixed 102명 내부에서만 대응을 바꾼다. patient_contributions의 정렬된 Q patient_id 순서에서 mixed를 추린 순서와 저장 permutation을 결속한다. 제한 전 평균 차분은 불변이지만 제한 후 평균과 2차 모멘트는 모두 바뀔 수 있다. 옛 mean-invariant shuffle_target을 새 D에 재사용하지 않는다.

새 목표는 relation_paths의 patient_paths/relation_target 경로와 결속된 환자별 point 집계로 만든다. Q 특징은 허용된 후 한 번 추출해 모든 arm에 재사용한다. 지금 고정하는 것은 **목표 생성 규칙의 hash**다. 아직 생성하지 않은 Q feature/target hash는 null이며, 별도 허용 이후 첫 관련 bank 전에 실제 파일·순서·해시를 기록해야 한다.

관계 항은 normalized point score의 실제 차이와 동일하다고 주장하지 않는 보조 학습 목적이다. R_joint도 C와 비선형 제한 순서가 다르므로 제한 전 선형 항등식만으로 동일법이라고 판정하지 않는다.

## 4. Renderer·초기화·배포

A/B는 class당 64장씩 128 marginal, 관계 arm은 class당 32 marginal + 32 virtual pairs다. 공통 배열의 negative index=0..63, positive=64..127이며, 관계 arm marginal=0..31와64..95, pair negative=32..63와positive=96..127이다.

기존 seed별 templates_P_private.json의 **공개 P 초기영상 목록**과 permutation_private.json을 결속한다. 원 Q 이미지나 private checkpoint는 초기화에 사용하지 않는다. seed별 선택된 공개 64 template가 128 slot으로 반복되며, 같은 seed의 arm·C 제거실험은 같은 template 선택을 사용한다. C/D·C 제거실험은 동일 renderer 초기화와 augmentation stream을 사용하되 optimizer 상태를 공유하지 않는다.

Renderer는 bounded sigmoid(logit(public template)+pyramid residual), 해상도 8/16/32/64/112/224, 활성 update 1/81/161/241/321/401이다. residual std=1e-3, seed 함수는 결속된 contracts.seed와 render.py를 따른다. pair residual은 u±v다. 기존 활성 residual은 계속 학습한다.

증강은 rotation±3°, translation±1%, scale/brightness 0.98–1.02, flip 없음이다. pair는 같은 파라미터를 공유한다. seed('prrd-augmentation', bank_seed, successful_step)로 생성해 두 pass에 재사용한다. 처음 step은 1이다. Profile에서 all-level active/nonce500부터 시작한 상태를 가져오지 않는다.

마지막 500-step에서 224×224 grayscale uint8 PNG를 내보내고, label·bank_type·virtual_pair_id·PNG hash·learning contract를 저장한다. float→PNG의 수치 차이를 별도로 기록하며, 모든 성능 readout은 실제 배포 PNG를 다시 읽는다. private patient ID/pair/원통계를 배포 manifest에 넣지 않는다. 측정용 C128의 parameter·optimizer·출력은 **재사용 금지**이며, 원 공개 template 목록·코드만 재사용할 수 있다.

## 5. 고정 실행 목록

모든 행은 **NOT_STARTED / NOT_AUTHORIZED**다. 순서는 아래 목록이고 각 ID가 독립적인 중단·재개 단위다. ‘original_rule’은 현재 raw C에서 두 보강만 끈 경우이며 옛 DINO v1 전체 재현이 아니다. 기본 C는 두 보강을 모두 사용하므로 제거실험에서 중복 실행하지 않는다.

| # | bank ID | beta | guard | eta | marginal / pairs |
|---:|---|---:|---|---:|---:|
| 1 | main_A_101 | 0 | N/A (point only) | 1 | 128 / 0 |
| 2 | main_A_202 | 0 | N/A (point only) | 1 | 128 / 0 |
| 3 | main_A_303 | 0 | N/A (point only) | 1 | 128 / 0 |
| 4 | main_B_101 | 0 | N/A (point only) | 1 | 128 / 0 |
| 5 | main_B_202 | 0 | N/A (point only) | 1 | 128 / 0 |
| 6 | main_B_303 | 0 | N/A (point only) | 1 | 128 / 0 |
| 7 | main_E_101 | 1 | ON | 1 | 64 / 32 |
| 8 | main_E_202 | 1 | ON | 1 | 64 / 32 |
| 9 | main_E_303 | 1 | ON | 1 | 64 / 32 |
| 10 | main_E_R_101 | 1 | ON | 1 | 64 / 32 |
| 11 | main_E_R_202 | 1 | ON | 1 | 64 / 32 |
| 12 | main_E_R_303 | 1 | ON | 1 | 64 / 32 |
| 13 | main_C_101 | 1 | ON | 1 | 64 / 32 |
| 14 | main_C_202 | 1 | ON | 1 | 64 / 32 |
| 15 | main_C_303 | 1 | ON | 1 | 64 / 32 |
| 16 | main_D_101 | 1 | ON | 1 | 64 / 32 |
| 17 | main_D_202 | 1 | ON | 1 | 64 / 32 |
| 18 | main_D_303 | 1 | ON | 1 | 64 / 32 |
| 19 | main_R_joint_101 | 1 | ON | 1 | 64 / 32 |
| 20 | main_R_joint_202 | 1 | ON | 1 | 64 / 32 |
| 21 | main_R_joint_303 | 1 | ON | 1 | 64 / 32 |
| 22 | ablation_C_no_guard_101 | 1 | OFF | 1 | 64 / 32 |
| 23 | ablation_C_no_guard_202 | 1 | OFF | 1 | 64 / 32 |
| 24 | ablation_C_no_guard_303 | 1 | OFF | 1 | 64 / 32 |
| 25 | ablation_C_no_func_101 | 1 | ON | 0 | 64 / 32 |
| 26 | ablation_C_no_func_202 | 1 | ON | 0 | 64 / 32 |
| 27 | ablation_C_no_func_303 | 1 | ON | 0 | 64 / 32 |
| 28 | ablation_C_original_rule_101 | 1 | OFF | 0 | 64 / 32 |
| 29 | ablation_C_original_rule_202 | 1 | OFF | 0 | 64 / 32 |
| 30 | ablation_C_original_rule_303 | 1 | OFF | 0 | 64 / 32 |

## 6. 중단·재개와 중간 결과 사용 금지

실행을 허용할 때 계약 hash, 허용 bank ID, 허용 자료 역할/단계, 숫자 wall-time 상한을 먼저 기록한다. 30개 중 일부를 허용해도 나머지는 미실행 상태로 남기며 완료 범위를 부풀리지 않는다.

한 bank는 독립 renderer·optimizer·난수 상태로 시작한다. 성공한 전체 update 경계마다 완료 step을 기록하고, checkpoint는 매 25 successful updates 및 정상 중단 경계에서 원자적으로 저장하는 규칙으로 고정한다. 저장 대상은 모든 residual/optimizer/RNG 상태, 다음 step, 활성 pyramid, 코드/contract/target/template hash다. 중간 microbatch 상태를 완료 checkpoint로 표시하지 않는다.

시간 상한에 도달하기 전에 현재 완료 update를 저장하고 멈춘다. 비정상 중단 시 마지막 유효 checkpoint에서 동일한 augmentation stream으로 재개한다. 기존 실패/미완료 update는 남기고 성공 step에 포함하지 않는다. 해시가 다른 상태를 덮어쓰거나 이어 붙이지 않는다. 완료 bank는 봉인하고 미완료 ID만 이어 간다.

Resume·PNG·main runner는 아래 §9의 **미구현 연결**이다. 이 규칙을 적었다는 이유로 실제 재개가 검증됐다고 표시하지 않는다. 후속 구현은 동일 상태 공개 입력 검사와 코드 hash 추가 receipt로 연결한다. 새로운 profile은 필수로 요구하지 않는다.

모든 30bank가 봉인되기 전 V/수신자 효용을 열지 않는다. 중간 loss·이미지 모양·source/recipient 점수로 계수, arm, seed, 학습량, checkpoint를 선택하지 않는다. 허용 subset이 끝나면 저장 상태와 실행량만 보고하며 비교 결과를 먼저 열지 않는다. source readout이 약하다는 이유로 모든 합성을 자동 중단하는 새 기준을 넣지 않는다. OOM/비유한 수치/코드 오류는 실패를 기록하고 같은 계약의 기술 수정만 허용하며 목적·계수·범위 변경은 별도 계약 버전으로 남긴다.

## 7. 추후 평가 범위 — 이번 실행 아님

기존 계획의 source/primary readout 108개, trusted-real reference 20개는 readout 구성 수이며 선형계 solve 호출 수가 아니다. Core21bank의 ResNet18은 3seed×400update = 63run/25,200updates다. 추가 9 제거 bank의 RN18은 자동 추가하지 않는다. Primary DenseNet121, source 및 관계 bank의 beta0/beta1, paired patient-cluster bootstrap 2,000draw를 유지한다.

수신자는 PNG·label·virtual pairs·공개 학습 규칙을 받아 자기 공개 변환과 통계를 사용한다. Source의 private/raw 통계를 받지 않는다. Recipient의 raw 특징·공개 관계 scale·readout 구현 결속은 평가 전 필요한 연결이며, 본 계약 작성에서 수신자를 실행하지 않는다. DenseNet/ViT를 합성 loss·checkpoint·계수 선택에 쓰지 않는다. ViT 확인용 평가, DP, Expert/Reserved는 본 합성 실행 승인에 포함되지 않는다.

C−B, C−D, C−E_R, C−R_joint 및 C 보강 제거 비교를 모두 보고한다. 기존 개발 기준(C−B 평균 AUROC≥0.01, C−D>0, 각각 3개 중 2개 양수, B 대비 평균 AP 비감소)을 유지한다. 효과·불확실성·negative 결과를 함께 남기며 수학/실행 PASS를 효용 PASS로 바꾸지 않는다.

## 8. 실측 근거와 예상 비용

| quantity | C-speed reference |
|---|---:|
| Full128 update (measured mean) | 7.8364 s |
| C bank, 500 updates | 65.30 min |
| Core 21 banks, 10,500 updates | 22.86 h |
| Additional 9 banks, 4,500 updates | 9.80 h |
| All 30 banks, 15,000 updates | 32.65 h |
| Preparation + export + all evaluation | NOT MEASURED |

이 값은 기존 C128 profile의 5 warm-up + 10 timed update에서 얻은 동기화된 실제 전체 loss/two-pass backward/optimizer 시간이다. 이번 추가 profile은 없다. Peak allocated=1,732.42 MiB, reserved=1,852 MiB, RTX 3070 8GB다.

C 속도를 모든 arm·step에 동일하게 적용한 참고 추정이다. Profile은 전체 pyramid가 활성 상태였고 본 계획은 점진 활성이다. 다른 arm·제거실험의 처리량, Q 특징/목표 준비, cold start, checkpoint/PNG IO, recipient/RN18/통계 시간은 측정하지 않았으므로 총시간 또는 상한으로 사용할 수 없다. 전체 예산은 이 미측정 부분과 실제 허용 범위를 포함해 별도로 정해야 하며, 비용을 맞추려고 500회나 3반복을 조용히 줄이지 않는다.

## 9. 실행을 막는 필수 항목

1. **실행 허용 범위와 시간**: authorized_run_ids 및 numeric runtime cap 미정. Q 접근도 이번에는 불허이며 필요한 준비 단계의 허용을 따로 결속한다.
2. **실행기 연결**: 현재 prrd_v3/run.py·fit_banks.py는 없고, 검증/profile API를 main orchestration, 동일 상태 resume, 마지막 PNG/manifest 봉인에 연결하지 않았다. 옛 require_w2_authority와 존재하지 않는 contract.yaml을 실행 준비 완료로 오인하지 않는다. 새 runner의 hash와 같은 상태 재개/PNG 검증 receipt가 필요하다. 이번 문서는 실행 명령이 이미 존재한다고 제시하지 않는다.
3. **실제 사적 입력 산출물 결속**: Q raw/point feature 및 B/E/E_R/C/D/R_joint 목표 파일의 실제 hash는 미생성 상태다. 규칙은 고정됐으며, 허용된 Q 준비 후 새 성능 선택 없이 생성·검산·해시 결속해야 한다. A는 이미 존재하는 P 목표를 사용할 수 있으나 1·2 조건은 동일하게 적용된다.

추가 연구 방향 선정, 계수 탐색, 다른 arm profile, 선행 전수 구현은 위 첫 합성 launch의 필수 조건으로 추가하지 않는다. 평가 연결·DP·final은 각 단계의 별도 범위다. 여기서 멈추며 자동 실행을 예약하지 않는다.

## 10. 결속 부록

아래 SHA-256은 현재 로컬 파일의 실제 bytes 기준이다. 실행 코드와 공개 특징/모델/메타데이터만 읽어 대조했다. 원영상 파일의 hash는 기존 명부 기록을 재사용하며 이번에 픽셀을 재검사하지 않았다. W0의 오래된 remote commit은 과거 출처이고, 사용자가 알려준 809ff05를 현재 로컬 commit처럼 표시하지 않는다.

target_generation_rule_sha256은 아래 target_generation_rule 객체를 JSON(sort_keys=True, separators=(',', ':'), ensure_ascii=True)로 직렬화한 hash다. 이 객체에는 실제 함수 파일 및 입력 binding이 포함된다. 아직 없는 Q 산출물 hash와 구별한다.

```json
{
  "schema": "prrd.first-nondp.contract-bindings/v1",
  "contract_id": "PRRD-NONDP-20260918-v1",
  "status": "STARTING_RECIPE_FIXED_EXECUTION_NOT_AUTHORIZED",
  "local_git_commit": "NOT_A_GIT_CHECKOUT",
  "user_reported_remote_commit": "809ff05",
  "remote_commit_reverified_this_task": false,
  "source_sha256": {
    "CODE/prrd_v3/__init__.py": "88b8a1a908246fec87b198f0e6512b977c66eecf5bd5f3ab34f60c1a0995905f",
    "CODE/prrd_v3/contracts.py": "bd5679b3269fd02de618d581d0649177f003b5a27496931bfaa2e8bdca70f376",
    "CODE/prrd_v3/datasets.py": "dbaa0ce9067933d148f07cd62f5d8e7692429329c35553038e5932bb2983742e",
    "CODE/prrd_v3/encoders.py": "d4f432be4e7bbbe7cdb11928c88f2b4190c4692b19d353e7b0bcec0ea4310c4a",
    "CODE/prrd_v3/guarded_objectives.py": "3a69015a1f610432a3f32010601134ba0659a75c29fc5e23658b42322fa97b71",
    "CODE/prrd_v3/guarded_readout.py": "cd934c2098dc9e823673effdf21942986b68070e37ef98778aa3f3f1d5eb84c2",
    "CODE/prrd_v3/objectives.py": "3b7c109a8fc525902e2433576c6f1aa399a0017a0629675081843e334a84cb88",
    "CODE/prrd_v3/patient_moments.py": "a548092ac02d007a95ec08ca9e0b52014e3094056ac02095e0550cd5bcc7d616",
    "CODE/prrd_v3/prepare.py": "cfd46b6ea31edae9b144dfaf0d96d7a2d6f3fdd2ab0961e15aaf3d8582ed0004",
    "CODE/prrd_v3/profile.py": "91967621e004c2edb21c309c7b307fc3b649228e52f7fe2edf6ce42c823ceee4",
    "CODE/prrd_v3/profile_guarded_c128.py": "35aabec39d46b8da44916e8fdae36915bf2ed1cfc6cada358c683037391657e9",
    "CODE/prrd_v3/public_feature_paths.py": "cb12acacec3fe0670c788d422cd1112ef0d8ec6d6ba0826dd9c6b60f563ff52a",
    "CODE/prrd_v3/public_setup.py": "0c6778dc07837344c144cbc2339454a340ea094ef4b6f0ec513ab13a9b94adb0",
    "CODE/prrd_v3/relation_paths.py": "2f320dcf5a8248892dc30f2bf49f621c945d910ad3ea80ec52fe1fa6a0ae5826",
    "CODE/prrd_v3/render.py": "7c405f2509561ed5662edf97451f54cc04c587b688e0c2c6ff0ead55ddc5e6a0",
    "CODE/prrd_v3/resampling.py": "11c83201079b52363a3bd27556fb4fc7cb90418c84ff8d64a598e370320221cd",
    "CODE/prrd_v3/test_core.py": "7168b3a299a67f73a206a59c59fe91b2dd0a1f3d1ebfe99f82a48716cfd31457",
    "CODE/prrd_v3/test_guarded_objectives.py": "a1e07435b3e10ed587362d602fccc5881167d43e17016cdf55e69dc5daccc21c",
    "CODE/prrd_v3/test_guarded_readout.py": "15bc3d2fca153f3869488e31a38f4c95969567c30613867c3a03e4fc38de0d2d",
    "CODE/prrd_v3/test_relation_paths.py": "497f46d1bd27b4493f77bb7c347b19d46ae10d9b253ccf4fb5a6f905b26f933c",
    "CODE/prrd_v3/transfer_rn18.py": "8ec3a1c9da3582eb5b7b46685848ea0bb00cb42af664ccd9a9f768f7ded84a96",
    "CODE/prrd_v3/verify_guarded_image_objective.py": "c091220e673e62cee5614a60ae162d2b7183c27766b45f2acdf88d29d142cee3",
    "CODE/prrd_v3/verify_public_feature_paths.py": "aed1d8547fda7bf60a94d84554f5bee54c2b360ff4735571f470d97faa9954d8",
    "CODE/prrd_v3/w1_boundary_debug.py": "e7aedbea3bb0b503ed18e689411d7cbce53382f59dfb3d804f28d364dd5b518f",
    "CODE/prrd_v3/w1_repair_verify.py": "1d7f7153509fea78209f4c3c255025b1cbdac4536875d5bad012e4a04b69b52a"
  },
  "dependency_sha256": {
    "CODE/relation_distillation/moments.py": "a498d5e86203355adbe968a66c9051a42cb156736711d951558dfb7dadfe4491",
    "C:/Users/SOGANG/anaconda3/Lib/site-packages/health_multimodal/image/__init__.py": "9a5b4a4d2a1256d76534b9dda181d5885c2453712d843b2a256a89075771e910",
    "C:/Users/SOGANG/anaconda3/Lib/site-packages/health_multimodal/image/data/__init__.py": "6134c3f67e930a3088f6f902f7c9cda7e2152ee218457a7b7a73949ee6abafca",
    "C:/Users/SOGANG/anaconda3/Lib/site-packages/health_multimodal/image/data/io.py": "0e7362e87a266cd1cd4ee2027c4ef4204ceabffef90886643cbef15b20a012a7",
    "C:/Users/SOGANG/anaconda3/Lib/site-packages/health_multimodal/image/data/transforms.py": "0be6fd628bd9ac0bc64656314dc358fd9def7599fee1aa0a5a9bf288194bc76d",
    "C:/Users/SOGANG/anaconda3/Lib/site-packages/health_multimodal/image/inference_engine.py": "e8add193733bb472cbdd6fd64539467695bdd85c91fb1533f953197c1221c23c",
    "C:/Users/SOGANG/anaconda3/Lib/site-packages/health_multimodal/image/model/__init__.py": "4f27cf5f331168b71a9f30cbbfca23277ae9953422c7af95721b6c2370e5a754",
    "C:/Users/SOGANG/anaconda3/Lib/site-packages/health_multimodal/image/model/encoder.py": "c094e56097d6addb6c61fb987ea4c60a0e359ebe6c5e2112eb1860e54034b351",
    "C:/Users/SOGANG/anaconda3/Lib/site-packages/health_multimodal/image/model/model.py": "44b91fce7783057532eace456e8592eaea8c75cd05ae92b3e7333f16e6c357fc",
    "C:/Users/SOGANG/anaconda3/Lib/site-packages/health_multimodal/image/model/modules.py": "c6dc024995633f83af525436af59babb7330838e4acb668919e2965c1368ff99",
    "C:/Users/SOGANG/anaconda3/Lib/site-packages/health_multimodal/image/model/pretrained.py": "f188f0dea45b23d60b7d46aa09533e08e8e3f49d4ab69bc6729aec4a88779033",
    "C:/Users/SOGANG/anaconda3/Lib/site-packages/health_multimodal/image/model/resnet.py": "d7e7d9418ddc5e20614f3eb3ec3c08a98c836bb9bcf26aa9a41798233986c151",
    "C:/Users/SOGANG/anaconda3/Lib/site-packages/health_multimodal/image/model/transformer.py": "9b404c8399a290ed248214da5b9838564f70948870e0a18f5110e57db66cf5cb",
    "C:/Users/SOGANG/anaconda3/Lib/site-packages/health_multimodal/image/model/types.py": "7515c35376491a8d05883b34b51f740c4d007e878fb8d84d5b6952ae178f7d71",
    "C:/Users/SOGANG/anaconda3/Lib/site-packages/health_multimodal/image/utils.py": "ed8b4b7e3a27f15f1e272e760103894407e519a0a48823b9fc5fab6972551a83"
  },
  "input_and_evidence_sha256": {
    "CODE/_reports/prrd_step2_paths_20260918_v1/public_paths/patient_paths_P.npz": "caa0004c4e943b7351fe6250ac6535aec3fd61917fb7749c2cd515caeec22398",
    "CODE/_reports/prrd_step2_paths_20260918_v1/public_paths/source_P_dual_features.npz": "0a9ed69056b449c32ffd805ce50687f5863eaa40be9e1d8d24096dfe5595f40a",
    "CODE/_reports/prrd_step2_paths_20260918_v1/public_paths/source_P_pre_external_norm_features.npz": "d7aa2721f1e8136991dcf1da9d4bca908180c0ff34718856e57ff76c1e73a931",
    "CODE/_reports/prrd_step2_paths_20260918_v1/public_paths/source_P_relation_projection.npz": "e7b31033c8fdc35b0d67d36a58ac3360c13e773e08ce28eac3d02ad289defd2b",
    "CODE/_reports/prrd_step2_paths_20260918_v1/public_paths/source_P_relation_targets.npz": "a9f21994a7db51da4c9994046299b1131b379a1ecf46d8b2a86ead1bed3997ae",
    "CODE/_reports/real_support_plan_20260917_v1/expanded_training_manifest_private.csv": "2dceb96c90a7013d38ea9dd274511f2a55978d8ec83f76f6b7aad0ef9b6d4c91",
    "CODE/_reports/real_support_plan_20260917_v1/method_development_unchanged_private.csv": "ba70d37000c42312266fddb9ac1568fc68318d1d80aa4c1a9f05691834461ab3",
    "CODE/_reports/prrd_v3_20260918_w01/source_P_projection.npz": "a6cdeed9dac2c9f348cf8fd99138a741cfc6e8770f2952b89fd3deb1f7d247fa",
    "CODE/_reports/prrd_v3_20260918_w01/source_P_features.npz": "3f7eb6a96e115c762ef5219fb2111a7be63cd92e74dcc5b84e62ef0ba96c8362",
    "C:/Users/SOGANG/anaconda3/Lib/site-packages/health_multimodal/image/model/model.py": "44b91fce7783057532eace456e8592eaea8c75cd05ae92b3e7333f16e6c357fc",
    "C:/Users/SOGANG/.cache/huggingface/hub/models--microsoft--BiomedVLP-BioViL-T/snapshots/692f09e9be1bfe5fdd5f3efdd0e1eca7d2c10b23/biovil_t_image_model_proj_size_128.pt": "b2399d73dc2a68b9f3a1950e864ae0ecd24093fb07aa459d7e65807ebdc0fb77",
    "CODE/_reports/prrd_step2_paths_20260918_v1/public_paths/source_P_relation_scales.json": "0aa9bc3ed8a1957b399b1bcd5b7b8e7e88aad867980750ca5424afb9a9e4d0f3",
    "CODE/_reports/prrd_v3_20260918_w01/templates_P_private.json": "80733480bc19ed4e01883f2e822ae8d294f05f8da9a62663197c7b35fadc32ab",
    "CODE/_reports/prrd_v3_20260918_w01/permutation_private.json": "8693928b1fbf1990e7b6b2c47de8d1b82edde8d7b3f69fa9fa2b139f19fd52bb",
    "CODE/_reports/prrd_v3_20260918_w01/w0_bindings.json": "6d253bea4f75eada535cdc3856115db76e8d4c8f6169dd4f34a4faef1cafd3b2",
    "CODE/_reports/prrd_step2_paths_20260918_v1/public_paths/feature_path_contract.json": "44a7369a22706c600a07bfd8e6f7bfed6aff57a9c08f0659b86af8d3e36bea7d",
    "CODE/_reports/prrd_c128_profile_20260918_v1/run/profile_contract.json": "6499045c70b7175c7dd97edc01125933752826a3c1cd4a3307708481cde8772e",
    "CODE/_reports/prrd_c128_profile_20260918_v1/run/full128_profile.json": "5dd557a5307745b50e89c588f8ebf1c02e15240fd68178bd98a642c1f6a5e30e",
    "RESEARCH/PRRD_CONSOLIDATED_DESIGN_20260918.md": "afd00131e41593a45a29b69c43d67322f431b739309f3a7eb41c8921222139c9",
    "RESEARCH/PRRD_STEP3_GUARDED_OBJECTIVE_RESULTS_20260918.md": "35485cc7dc055ab6f126bcb08ed9b7b2c25b73432d9a8b5050be4221eddbc08d",
    "RESEARCH/PRRD_C128_PUBLIC_PROFILE_RESULTS_20260918.md": "356baa48317802f4f82acf38a7488399cc0b528abf7ffd27cc874bbb2166740e",
    "RESEARCH/spec_sources/prrd_non_dp_guarded_functional_runs_20260918.json": "44bb94d64516f821cf3379595e6cc93336f88163045df4804eacdec02ac6bdd4"
  },
  "target_generation_rule": {
    "schema": "prrd.target-generation-rule/v1",
    "code_sha256": {
      "CODE/prrd_v3/contracts.py": "bd5679b3269fd02de618d581d0649177f003b5a27496931bfaa2e8bdca70f376",
      "CODE/prrd_v3/datasets.py": "dbaa0ce9067933d148f07cd62f5d8e7692429329c35553038e5932bb2983742e",
      "CODE/prrd_v3/encoders.py": "d4f432be4e7bbbe7cdb11928c88f2b4190c4692b19d353e7b0bcec0ea4310c4a",
      "CODE/prrd_v3/guarded_objectives.py": "3a69015a1f610432a3f32010601134ba0659a75c29fc5e23658b42322fa97b71",
      "CODE/prrd_v3/guarded_readout.py": "cd934c2098dc9e823673effdf21942986b68070e37ef98778aa3f3f1d5eb84c2",
      "CODE/prrd_v3/patient_moments.py": "a548092ac02d007a95ec08ca9e0b52014e3094056ac02095e0550cd5bcc7d616",
      "CODE/prrd_v3/relation_paths.py": "2f320dcf5a8248892dc30f2bf49f621c945d910ad3ea80ec52fe1fa6a0ae5826",
      "CODE/relation_distillation/moments.py": "a498d5e86203355adbe968a66c9051a42cb156736711d951558dfb7dadfe4491"
    },
    "input_sha256": {
      "CODE/_reports/prrd_step2_paths_20260918_v1/public_paths/patient_paths_P.npz": "caa0004c4e943b7351fe6250ac6535aec3fd61917fb7749c2cd515caeec22398",
      "CODE/_reports/prrd_step2_paths_20260918_v1/public_paths/source_P_relation_projection.npz": "e7b31033c8fdc35b0d67d36a58ac3360c13e773e08ce28eac3d02ad289defd2b",
      "CODE/_reports/real_support_plan_20260917_v1/expanded_training_manifest_private.csv": "2dceb96c90a7013d38ea9dd274511f2a55978d8ec83f76f6b7aad0ef9b6d4c91",
      "CODE/_reports/prrd_v3_20260918_w01/source_P_projection.npz": "a6cdeed9dac2c9f348cf8fd99138a741cfc6e8770f2952b89fd3deb1f7d247fa",
      "CODE/_reports/prrd_step2_paths_20260918_v1/public_paths/source_P_relation_scales.json": "0aa9bc3ed8a1957b399b1bcd5b7b8e7e88aad867980750ca5424afb9a9e4d0f3",
      "CODE/_reports/prrd_v3_20260918_w01/templates_P_private.json": "80733480bc19ed4e01883f2e822ae8d294f05f8da9a62663197c7b35fadc32ab",
      "CODE/_reports/prrd_v3_20260918_w01/permutation_private.json": "8693928b1fbf1990e7b6b2c47de8d1b82edde8d7b3f69fa9fa2b139f19fd52bb"
    },
    "point_path": "Existing normalized BioViL wrapper h -> stored P patient-uniform center/PCA16 -> stored P q95 bound; unchanged",
    "raw_relation_path": "BioViL projected_global_embedding BEFORE wrapper L2 -> P patient-uniform raw center -> SAME stored point PCA axes; no pre-contrast cap",
    "patient_order": "patient_contributions sorted patient_id strings; D permutation indexes Q mixed subset in this order",
    "point_weighting": "Within patient/class mean z and zzT; equal present-patient mass by class; merge P+Q sums and counts BEFORE normalization",
    "populations": {
      "A": "P point",
      "B": "P+Q point",
      "E/E_R/C/D/R_joint": "P+Q point; P+Q mixed relation"
    },
    "empty": "Absent empirical class/mixed contribution is zero, existing code max(count,1) denominators",
    "relation_rules": {
      "E": "(mean z_positive-mean z_negative)/2",
      "E_R": "bound(E, public_E_R_scale)",
      "C": "bound((mean a_positive-mean a_negative)/2, public_C_scale)",
      "D": "Q-only negative-centroid derangement then C; P pairing unchanged",
      "R_joint": "bound(concat(mean a_positive,mean a_negative)/sqrt(2),public_joint_scale); r=L*bounded_u; match u/U; derive readout mu/C after bounding"
    },
    "public_scales": {
      "E_R": 0.29153658488143597,
      "C": 0.2650091310071505,
      "R_joint": 1.2582928166602765
    },
    "public_scale_estimator": "mixed P3 inverse empirical-CDF q95, no interpolation; unchanged",
    "D_seeds": [
      101,
      202,
      303
    ],
    "synthetic_target_fixed": "Detached copies; fixed target readout and M_T; synthetic uses own M_S for guard",
    "source_bound_definition": "Only external wrapper normalization is bypassed for relation a; internal encoder nonlinearities remain",
    "no_private_target_created_in_this_task": true
  },
  "target_generation_rule_sha256": "6713ab1811be9d6dece26caf33d2a9ca17e84fc1177cb964723ca8b987bcdb31",
  "Q_feature_sha256": null,
  "private_target_sha256": null,
  "learning_policy": {
    "beta_relation": 1,
    "beta_A_B": 0,
    "ridge": 0.1,
    "rho": null,
    "kappa": 0.1,
    "eta_default": 1,
    "intercept": false,
    "status": "PROSPECTIVE_START_VALUES_NOT_EFFICACY_OPTIMA",
    "fixed_target_metric": true,
    "functional_clean_only": true
  },
  "runtime_recipe": {
    "microbatch": 4,
    "dtype_encoder_image": "float32",
    "dtype_moments_readout": "float64",
    "AMP": false,
    "optimizer": {
      "name": "AdamW",
      "lr": 0.01,
      "weight_decay": 0.0,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-08,
      "foreach": false
    },
    "progressive_renderer_levels": [
      8,
      16,
      32,
      64,
      112,
      224
    ],
    "activate_at_successful_step": [
      1,
      81,
      161,
      241,
      321,
      401
    ],
    "hardware_reference": {
      "gpu": "NVIDIA GeForce RTX 3070",
      "vram_bytes": 8589410304,
      "platform": "Windows-10-10.0.26200-SP0",
      "torch": "2.6.0+cu124",
      "torchvision": "0.21.0+cpu"
    },
    "checkpoint_every_successful_updates": 25,
    "final_step": 500
  },
  "run_matrix_source_sha256": "44bb94d64516f821cf3379595e6cc93336f88163045df4804eacdec02ac6bdd4",
  "runs": [
    {
      "id": "main_A_101",
      "arm": "A",
      "seed": 101,
      "beta": 0,
      "guard": true,
      "eta": 1,
      "images": 128,
      "updates": 500,
      "execution_authorized": false
    },
    {
      "id": "main_A_202",
      "arm": "A",
      "seed": 202,
      "beta": 0,
      "guard": true,
      "eta": 1,
      "images": 128,
      "updates": 500,
      "execution_authorized": false
    },
    {
      "id": "main_A_303",
      "arm": "A",
      "seed": 303,
      "beta": 0,
      "guard": true,
      "eta": 1,
      "images": 128,
      "updates": 500,
      "execution_authorized": false
    },
    {
      "id": "main_B_101",
      "arm": "B",
      "seed": 101,
      "beta": 0,
      "guard": true,
      "eta": 1,
      "images": 128,
      "updates": 500,
      "execution_authorized": false
    },
    {
      "id": "main_B_202",
      "arm": "B",
      "seed": 202,
      "beta": 0,
      "guard": true,
      "eta": 1,
      "images": 128,
      "updates": 500,
      "execution_authorized": false
    },
    {
      "id": "main_B_303",
      "arm": "B",
      "seed": 303,
      "beta": 0,
      "guard": true,
      "eta": 1,
      "images": 128,
      "updates": 500,
      "execution_authorized": false
    },
    {
      "id": "main_E_101",
      "arm": "E",
      "seed": 101,
      "beta": 1,
      "guard": true,
      "eta": 1,
      "images": 128,
      "updates": 500,
      "execution_authorized": false
    },
    {
      "id": "main_E_202",
      "arm": "E",
      "seed": 202,
      "beta": 1,
      "guard": true,
      "eta": 1,
      "images": 128,
      "updates": 500,
      "execution_authorized": false
    },
    {
      "id": "main_E_303",
      "arm": "E",
      "seed": 303,
      "beta": 1,
      "guard": true,
      "eta": 1,
      "images": 128,
      "updates": 500,
      "execution_authorized": false
    },
    {
      "id": "main_E_R_101",
      "arm": "E_R",
      "seed": 101,
      "beta": 1,
      "guard": true,
      "eta": 1,
      "images": 128,
      "updates": 500,
      "execution_authorized": false
    },
    {
      "id": "main_E_R_202",
      "arm": "E_R",
      "seed": 202,
      "beta": 1,
      "guard": true,
      "eta": 1,
      "images": 128,
      "updates": 500,
      "execution_authorized": false
    },
    {
      "id": "main_E_R_303",
      "arm": "E_R",
      "seed": 303,
      "beta": 1,
      "guard": true,
      "eta": 1,
      "images": 128,
      "updates": 500,
      "execution_authorized": false
    },
    {
      "id": "main_C_101",
      "arm": "C",
      "seed": 101,
      "beta": 1,
      "guard": true,
      "eta": 1,
      "images": 128,
      "updates": 500,
      "execution_authorized": false
    },
    {
      "id": "main_C_202",
      "arm": "C",
      "seed": 202,
      "beta": 1,
      "guard": true,
      "eta": 1,
      "images": 128,
      "updates": 500,
      "execution_authorized": false
    },
    {
      "id": "main_C_303",
      "arm": "C",
      "seed": 303,
      "beta": 1,
      "guard": true,
      "eta": 1,
      "images": 128,
      "updates": 500,
      "execution_authorized": false
    },
    {
      "id": "main_D_101",
      "arm": "D",
      "seed": 101,
      "beta": 1,
      "guard": true,
      "eta": 1,
      "images": 128,
      "updates": 500,
      "execution_authorized": false
    },
    {
      "id": "main_D_202",
      "arm": "D",
      "seed": 202,
      "beta": 1,
      "guard": true,
      "eta": 1,
      "images": 128,
      "updates": 500,
      "execution_authorized": false
    },
    {
      "id": "main_D_303",
      "arm": "D",
      "seed": 303,
      "beta": 1,
      "guard": true,
      "eta": 1,
      "images": 128,
      "updates": 500,
      "execution_authorized": false
    },
    {
      "id": "main_R_joint_101",
      "arm": "R_joint",
      "seed": 101,
      "beta": 1,
      "guard": true,
      "eta": 1,
      "images": 128,
      "updates": 500,
      "execution_authorized": false
    },
    {
      "id": "main_R_joint_202",
      "arm": "R_joint",
      "seed": 202,
      "beta": 1,
      "guard": true,
      "eta": 1,
      "images": 128,
      "updates": 500,
      "execution_authorized": false
    },
    {
      "id": "main_R_joint_303",
      "arm": "R_joint",
      "seed": 303,
      "beta": 1,
      "guard": true,
      "eta": 1,
      "images": 128,
      "updates": 500,
      "execution_authorized": false
    },
    {
      "id": "ablation_C_no_guard_101",
      "arm": "C",
      "seed": 101,
      "beta": 1,
      "guard": false,
      "eta": 1,
      "images": 128,
      "updates": 500,
      "execution_authorized": false
    },
    {
      "id": "ablation_C_no_guard_202",
      "arm": "C",
      "seed": 202,
      "beta": 1,
      "guard": false,
      "eta": 1,
      "images": 128,
      "updates": 500,
      "execution_authorized": false
    },
    {
      "id": "ablation_C_no_guard_303",
      "arm": "C",
      "seed": 303,
      "beta": 1,
      "guard": false,
      "eta": 1,
      "images": 128,
      "updates": 500,
      "execution_authorized": false
    },
    {
      "id": "ablation_C_no_func_101",
      "arm": "C",
      "seed": 101,
      "beta": 1,
      "guard": true,
      "eta": 0,
      "images": 128,
      "updates": 500,
      "execution_authorized": false
    },
    {
      "id": "ablation_C_no_func_202",
      "arm": "C",
      "seed": 202,
      "beta": 1,
      "guard": true,
      "eta": 0,
      "images": 128,
      "updates": 500,
      "execution_authorized": false
    },
    {
      "id": "ablation_C_no_func_303",
      "arm": "C",
      "seed": 303,
      "beta": 1,
      "guard": true,
      "eta": 0,
      "images": 128,
      "updates": 500,
      "execution_authorized": false
    },
    {
      "id": "ablation_C_original_rule_101",
      "arm": "C",
      "seed": 101,
      "beta": 1,
      "guard": false,
      "eta": 0,
      "images": 128,
      "updates": 500,
      "execution_authorized": false
    },
    {
      "id": "ablation_C_original_rule_202",
      "arm": "C",
      "seed": 202,
      "beta": 1,
      "guard": false,
      "eta": 0,
      "images": 128,
      "updates": 500,
      "execution_authorized": false
    },
    {
      "id": "ablation_C_original_rule_303",
      "arm": "C",
      "seed": 303,
      "beta": 1,
      "guard": false,
      "eta": 0,
      "images": 128,
      "updates": 500,
      "execution_authorized": false
    }
  ],
  "authorization": {
    "contract_writing_only": true,
    "execution_authorized": false,
    "authorized_run_ids": [],
    "runtime_cap_seconds": null,
    "Q_V_pixels": false,
    "recipient_execution": false,
    "DP": false,
    "expert_reserved": false,
    "new_profile": false
  },
  "profile_artifact_reuse_allowed": false,
  "runtime_ready": false,
  "required_before_first_related_bank": [
    "Authorized bank IDs, preparation roles and numeric time cap",
    "Main runner/checkpoint-resume/final PNG wiring with same-state public verification and implementation hash receipt",
    "Actual Q feature and applicable target hashes after authorized preparation; target rules already fixed"
  ],
  "new_work_counts": {
    "GPU_processes": 0,
    "optimizer_updates": 0,
    "pixel_decodes": 0,
    "recipient_runs": 0,
    "DP_releases": 0,
    "remote_upload": 0
  },
  "cost_reference": {
    "profile_seconds_per_full128_update": 7.836363809998147,
    "proxy_C500_minutes": 65.30303174998456,
    "proxy_30banks_hours": 32.65151587499228,
    "proxy_core21_hours": 22.856061112494597,
    "proxy_ablation9_hours": 9.795454762497684,
    "total_pipeline_hours": null,
    "numeric_authorized_cap_seconds": null,
    "is_all_arm_measurement": false,
    "is_upper_bound": false,
    "peak_bytes": {
      "allocated_bytes": 1816572928,
      "reserved_bytes": 1941962752
    },
    "profile_sha256": "5dd557a5307745b50e89c588f8ebf1c02e15240fd68178bd98a642c1f6a5e30e"
  },
  "unchanged_actual_result": "TRACK1_REAL_SUPPORT_DIAGNOSTIC_RESULTS_20260917.md",
  "unchanged_last_efficacy_result": "TRACK1_LORA_TRANSFER_COMPARISON_RESULTS_20260917.md"
}
```

이 문서 작성은 코드·GPU 실행·환자 성능 검증이 아니다. 이전 단계의 실제 결과와 실패 기록, Expert/Reserved 경계, 현재 실제 효용 보고서 포인터를 유지한다.
