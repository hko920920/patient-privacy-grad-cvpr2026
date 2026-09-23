# A1 조건별 목표 준비·실행 연결 결과 — 2026-09-22

판정: **요청한 A1 목표 준비와 단일 encoder 실행 연결을 완료했다.** 실제 P/Q 조건별 목표가 생성·독립 검산됐고, 새 목적을 사용한 저장/재개/PNG 연결과 공개128장 비용 측정도 통과했다. 정식500회 학습·V 성능·A2·DP·Expert/Reserved는 실행하지 않았다.

[이번 실행 계약](RECEIVER_A1_TARGET_RUNTIME_CONTRACT_20260922.md), [개발 방향](RECEIVER_DISTILLATION_DEVELOPMENT_PLAN_20260922.md)에 따른 큰 연구단계2의 준비 작업이다. 기존 관계C/augmean의 음성 결과를 변경하지 않고, 이번 성공을 전이 개선으로 해석하지 않는다.

## 1. 실제 목표 준비

| 역할 | 환자 | 원영상 | 조건별 특징 계산 | class-present 환자0 / 1 |
|---|---:|---:|---:|---:|
| P 공개 |672|813|3,252|669 / 6|
| Q 보호 대상 |2,027|5,097|20,388|2015 / 114|
| 합계 |2,699|5,910|23,640|2684 / 120|

한 환자가 두 class에 속할 수 있다. 기존 class별 환자 균등 가중치를 유지하고, 각 환자의 같은 class 방문을 먼저 평균했다. P/Q의 class 합과 환자 수를 합친 다음 두 class를 각1/2로 결합한다. 조건 수나 방문 수가 환자 질량을 늘리지 않는다.

특징은 **[4조건, 영상,16좌표]**, 최종 gradient는 **[4조건,34좌표]**다. 각 조건의 2×16 weight gradient와2 bias gradient를 별도로 저장한다. 네 조건을 평균 target 하나로 합치지 않았고 이전 K4 augmean cache를 재사용하지 않았다.

고정 head·affine/brightness tensor, 조건 순서, native1024 변환→512 resize/448 crop, BioViL checkpoint, P-only PCA16/q95 scale을 파일·내용 hash로 결속했다. 기존 clean 특징/목표/목표 분류기와 production PRRD33파일은 hash가 유지됐다. Q 특징·target·환자 명부는 비DP 내부 자료이며 보호된 release가 아니다.

저장된 특징을 별도 NumPy CE 식과 환자별 직접 가중 집계로 다시 계산했다. Production 집계 함수를 재사용하지 않았다. 최대 절대 오차 **4.54747350886e-13**, 사전 기준1e-12 이내다. 환자/class 수와 영상/환자/label 순서도 일치했고 실제 runtime loader의 target은 exact였다. 독립 encoder 전체 재실행이나 의료 성능 검사는 아니다.

## 2. 새 실행 연결과 검증

기존 renderer·원자적 파일 저장·파일 잠금·checkpoint의 parameter/Adam/RNG/next step/pyramid 복원·PNG rounding과검증을 재사용했다. 새 조건별 target에 맞는 adapter와 출력 학습 계약만 별도 구현했다. 과거 guarded learner를 새 목적이라고 재명명하지 않았다.

- 신규 target I/O: `receiver_distillation/target_io.py`.
- P/Q 조건 추출과 독립 검산: `prepare_a1.py`.
- 실행 adapter: `runtime.py`. 실제 실행 함수는 technical/main에서 공통이며, 이번 CLI는 기술 작업만 허용한다.
- 검산: `test_target_runtime.py`, `verify_a1_runtime.py`.

| 변경 경로 검사 | 결과 |
|---|---|
| CPU 실제 P/Q/pooled target 로드 및 변조 거부 | 6개 PASS |
| Target 파일 변경·조건 순서 변경·평균 target 대체 | 거부 |
| 다른 target signature checkpoint 재개 | 거부 |
| 준비 작업으로 MAIN 실행 요청 | 거부 |
| 공개P4 연속3update 대2+새 프로세스1update | 전체 상태 digest exact |
| Parameter/optimizer/RNG/next step | exact |
| 재개 직전 의도적으로 바꾼 RNG | checkpoint의 값으로 exact 복원 |
| 연속/재개 loss 차이 | 0.0 |
| 두 경로 PNG·label·hash·학습 계약 | exact |
| 중간 checkpoint의 완료 bank 오인 | 없음 |
| Encoder weights/buffers·target | 불변 |

연속/분할/재개는 서로 다른3개 프로세스였다. 기존 pyramid 경계83/84 검사를 반복하지 않고 검증된 복원 코드를 재사용했다. 이번 작은3step 검사는 활성 level1에서 새 목적의 재개를 검증한 것이며, 공개128 비용 검사는6개 level을 모두 활성화했다.

기술 산출물은 TECHNICAL_TEST_ONLY이며 본학습 재사용을 금지했다. Full128 비용 검사는 PNG/bank를 만들지 않는다. 이10개의 공개 기술 update를 A1 본실험 결과로 세지 않는다.

## 3. 본학습 비용의 실제 근거

공개 P 목표와 기존 공개 template, K4·128장·microbatch16으로 **renderer→전체조건 특징→전역signal loss→two-pass backward→Adam update→finite 검사**를 측정했다. Warmup1회 뒤3회를 측정했다.

| 항목 | 값 |
|---|---:|
| 준비 update | 12.605초 |
| 측정 update 3회 | 11.788, 11.790, 11.766초 |
| 전체128장 update 평균 | **11.781초** |
| Peak allocated | 5.599GiB |
| Peak reserved | 6.203GiB |
| 500회 합성 단순 환산 | **98.2분 (약1시간38분)** |
| 계획용 변동 범위 | 79–123분 |

계획용 범위는 통계적 신뢰구간이나 승인된 시간 상한이 아니다. 전체 level에서4회만 측정했으며 실제500회 스케줄은1개 level부터 시작한다. 따라서 이는 planning estimate다. 모델 로딩·주기적 checkpoint I/O·최종PNG 저장·수신자 평가 시간은 별도다. P/Q 목표 추출은 이번에 완료되어 같은 target을 유지하면 반복하지 않는다.

A1 한 update는1024 image-forward/512 image-backward,500회는512,000F/256,000B다. 기존 B의2경로 대비 encoder image-evaluation 수가2배이며, 시간 자체는 새 경로로 실측했다. 기존 B49분을 그대로 새 비용으로 붙이지 않았다.

## 4. 실제 실행량과 저장

- 목표 준비:23,640F/0B/optimizer0. P49.02초, Q297.41초; 준비 프로그램 전체 **5.89분**.
- 실행 연결: 공개4장3+3 update, 공개128장4 update; 합계4,288F/2,144B, 기술 optimizer10회.
- 전체 모델 작업: **27,928F/2,144B**. 이 중 Q는 목표 추출에서만 접근했다.
- CPU 변경 경로 검사 0.93초; 새 프로세스 검증·full128 측정 포함 **93.50초**.
- 최초 예상25–45분; 구현·기록 포함 이 보고서 작성 시점 **약17.5분**.
- 이번 새 저장량 약16.4MiB; 변환 픽셀 저장·기존 파일 삭제 없음. 디스크 여유 약10.67GiB.
- A2·V 평가·수신 모델·DP·Expert/Reserved0. 첫 시도에서 준비와 실행 검증 모두 통과했고 허용오차 변경은 없었다.

## 5. 준비된 파일과 남은 항목

- 실제 target: `code_working/_reports/receiver_a1_targets_runtime_20260922_v1/targets/private/A1_condition_targets.npz`.
- Target SHA256: `4901a5122e552d2bd3af0ec96f7ced13e92dd2527497a5b8ae06d5f86f42cf58`.
- 조건 tuple/전처리 결속: 같은 private 폴더의 `condition_rule.json` 및 `target_handoff.json`.
- 실행 연결과비용: `runtime_attempt1/verification.json`, `cost_report.json`.
- 실제 pooled target·공개 template·코드 hash가 들어 있는 본학습 제안 job: `runtime_attempt1/main_proposal_job.json`.

본학습용 시작 recipe는128장·500회·seed101·microbatch16, 기존 AdamW lr0.01/weight_decay0을 유지한 값이다. 조건별 CE cosine과 기존 anchor0.01/TV0.0001을 사용하며 기능/관계 loss는 없다. 이것은 시작값이지 성능으로 선택한 최적값이 아니다. 새로운 main 실행에서는 제안 job의 hash와 허용 bank·숫자 시간 상한을 묶어 공통 `execute` 함수를 호출해야 한다. 이번 준비 CLI로500회가 자동 시작되지는 않는다.

**현재 target 준비나 실행 연결의 기술 차단 항목은 없다.** 남은 것은 정식 A1 실행 범위와 수치 시간 상한을 잠그는 일, V를 열기 전에 개발 판단 기준을 수치로 고정하는 일이다. A2용 두 번째 encoder의 adapter/P-only 변환은 별도 다음 범위이며 이번 A1 준비의 완료를 A2 준비 완료로 확대하지 않는다.

관계C·augmean 자동 재개, 계수 탐색, 추가seed, DP/final은 여전히 자동 실행하지 않는다. DenseNet은 개발 receiver이고, 이번 단계는 전이 개선을 확인한 결과가 아니다.

근거: [목표 준비](../../code_working/_reports/receiver_a1_targets_runtime_20260922_v1/targets/result.json), [독립 검산](../../code_working/_reports/receiver_a1_targets_runtime_20260922_v1/targets/independent_verification.json), [실행 연결](../../code_working/_reports/receiver_a1_targets_runtime_20260922_v1/runtime_attempt1/verification.json), [비용 실측](../../code_working/_reports/receiver_a1_targets_runtime_20260922_v1/runtime_attempt1/cost_report.json).
