# B 증강 target 첫 구현·공개 연결 검사 — 2026-09-21

판정: **첫 구현 단계는 통과했다.** 별도 증강 통계를 loss에 연결했고 실제 BioViL 이미지 gradient까지 확인했다. 전이 효용을 측정하거나 수정 B가 기존 B보다 좋다고 확인한 결과는 아니다.

사용자의 “하나씩 진행”에 따라 [검토 명세](PRRD_B_AUGMENTED_TARGET_REVIEW_SPEC_20260921.md)의 첫 구현 묶음만 수행했다. 관계 중심 C는 계속 보류한다. P/Q 전체 추출·정식 합성학습·수신자 성능 평가·계수 변경·DP·Expert/Reserved는 실행하지 않았다.

## 1. 실제로 바뀐 코드

| 파일 | 변경 |
|---|---|
| code_working/prrd_v3/guarded_objectives.py | 별도 detached aug_moments를 선택적으로 결속. 증강 통계 loss의 target만 교체. 미지정하면 이전 clean target 사용 |
| code_working/prrd_v3/augmented_targets.py (신규) | 고정 K4 변환, native1024 → 증강 → 기존 source point 경로, 환자·class별 view/visit 평균과 raw 2차 모멘트, P/Q 합계·분모 결합 |
| code_working/prrd_v3/fit_banks.py | 추가 target 파일/rule SHA 검증·로드, 새 job/spec에 내용 hash 결속 |
| code_working/prrd_v3/bank_runtime.py | 증강 목표의 내용/출처를 checkpoint signature 및 불변성 검사에 포함 |
| test_augmented_targets.py / verify_augmented_target_public.py (신규) | 변경 경로 CPU 검사와 공개4장 실제 모델 검사 |

Clean 목표 통계·목표 분류기·M_T·기능 정합의 위치, renderer·증강 분포·encoder·공개 projection·학습 계수는 바꾸지 않았다. 기존 생산 파일30개 중 변경은3개이고 나머지는 동일하다. 기존 결과 보고서·clean target·명부·옛 실행 계약도 hash로 보존을 확인했다. 변경 코드 전체를 과거 frozen 계약의 동일 코드라고 소급하지 않는다.

## 2. CPU 검산

변경 경로7개와 기존 목적 회귀6개, 총13개가 통과했다. 의료영상이나 encoder를 사용하는 CPU 성능 실험은 아니다.

- Taug=T0이면 이전 손실과 feature gradient가 정확히 같다.
- Taug만 바꾸면 augmented 경로 gradient는 달라지고 clean 통계·기능 손실·clean gradient·목표 readout은 정확히 유지된다.
- 외부 target 배열을 바꿔도 복사한 고정 target은 바뀌지 않는다.
- 환자별 방문 수가 달라도 같은 환자 질량을 유지한다. view/visit 반복으로 환자 수를 늘리지 않으며 P/Q 평균을 단순 반씩 섞지 않는다.
- raw 2차 모멘트는 mean(zzᵀ)다. mean(z)의 outer product로 대체하지 않았다. 빈 class는0으로 정의했다.
- 독립 직접 합산과 최대 차이는1.39e-17이었다.
- 증강 target 파일·rule·내용 변경/누락과 다른 target signature의 checkpoint 재개를 거부한다. 재개 거부는 작은 CPU 파일 검사이며 기존 프로세스 재개 campaign을 반복하지 않았다.

## 3. 실제 BioViL 공개 검사

P 안에서 사전 hash 규칙으로 음성2/양성2, 서로 다른4명의4장을 정했다. 이들은 기술 검사 입력이며 본실험 초기화/target으로 재사용하지 않는다. Clean target은 기존 P 통계, 증강 target은 이4장의 K4 통계다. 전체 P의 증강 통계를 추출했다는 뜻이 아니다.

| 검사 | 결과 |
|---|---:|
| 공개 K4 환자 통계 독립 계산 최대 오차 | 1.39e-17 |
| Same-partition one-pass/two-pass 조건 | microbatch1,4 모두 통과 |
| 최대 parameter gradient 오차 | 5.22e-8 |
| 최대 parameter gradient 상대 L2 오차 | 1.68e-7 |
| 이미지 gradient 오차 | 두 조건 모두0 |
| 최대 loss 오차 | 1.14e-13 |
| 증강 통계항만의 이미지 gradient norm | 2.660683, finite |
| Encoder 가중치·buffer·projection / 고정 target | 불변 |
| Optimizer 업데이트 | 0 |

Renderer의 step500 설정은 전체 pyramid level을 활성화해 derivative를 검사한 것이다. 500회 학습한 것이 아니다. 기존 gradient 기준(loss_abs2e-6, atol2e-7+peak-relative5e-4)을 유지했다. 서로 다른 microbatch 사이의 bitwise 동일성을 요구하거나 주장하지 않는다.

## 4. 최초 검사 실패와 수치 계약 정정

첫 공개 검사(v1)는 gradient 비교 전에 “native1024 항등 변환이면 원픽셀과2e-6 이내여야 한다”는 새 검사에서 멈췄다. 이 실패와 당시 verifier·계약·추출 결과를 보존했다.

모델을 호출하지 않은 별도 P 검사에서 다음을 확인했다.

- 항등 affine_grid의 FP32 좌표와 정확한 pixel-center 좌표 차이: 5.96e-8.
- 기존 sampler와 PyTorch grid_sample 모두 원픽셀 대비 최대8.76e-6 차이를 보였다.
- 두 sampler끼리의 차이는5.96e-8이며, 정확한 center grid를 쓰면 항등 픽셀 차이는0이었다.
- CPU/GPU 기존 전처리 차이는1.79e-7이었다.

따라서 기존 연산을 바꾸거나 gradient 허용오차를 높이지 않았다. v2에서는 **기존 FP32 affine_grid + bilinear zero-padding grid_sample과의 일치**를2e-6 기준으로 검사하고 이상적 항등 오차는 별도 보고하도록 검사 계약을 정정했다. 항등 변환의 reference 차이5.96e-8, 실제 sampled 변환의 reference 차이2.38e-7, identity 경로 feature reference 차이1.69e-7로 통과했다.

이는 첫 기준을 그대로 통과한 결과가 아니다. 실패가 실제 분모/변환/모델 오류인지 조사한 뒤, 부동소수점 기준 연산과 이상적 항등을 혼동한 검사를 수정한 기록이다. 원 알고리즘과 기존 loss/gradient 허용오차는 유지됐다.

## 5. 이번 실제 실행량

- v1: source24 image-forwards/0 image-backwards 후 중단.
- 원인 확인: encoder0회. 같은 공개4장 픽셀만 읽었다.
- v2: source76 image-forwards/36 image-backwards,34 forward 호출/21 backward 호출. 측정된 검사 본문9.51초, RTX3070 peak allocated2.79GiB.
- 합계: source100 image-forwards/36 image-backwards. 공개 고유4장·4명, 재시도/진단 포함20번 decode.
- 새 optimizer update0, 본 bank0, Q/V 픽셀0, 수신자0, DP0, Expert/Reserved0, 삭제/upload0.

9.51초는 v2 소수 입력 검사의 측정 시간이며 full128 update 처리량이나 전체 개발 작업 시간으로 환산하지 않는다. 통과한 기술 검사를 환자 성능 표본 수로 세지 않는다.

## 6. 완료 범위와 다음 단계

첫 단계의 질문인 **“증강된 실자료 target을 별도로 받아 의도한 loss와 이미지 gradient를 계산하는가?”**에는 근거가 생겼다. **“그렇게 만든 영상이 다른 encoder에 더 유용한가?”**는 여전히 미검증이다.

다음 작업은 새 추출/실행 계약의 코드·rule·입력을 결속하고, P/Q 전체 K4 통계를 생성·독립 검산해 target hash를 고정하는 것이다. 아직 실제 full target이 없고 새 main job/평가 wrapper의 캠페인 결속도 끝나지 않았으므로, 옛 권한 파일로500회 학습을 실행하지 않는다. 기존 저장·재개·PNG의 전체 검사를 다시 관문으로 추가하지 않는다. 새 B가 좋아져도 C/D를 자동 재개하지 않는 원칙은 유지한다.

내부 산출물: code_working/_reports/prrd_b_aug_implementation_20260921_v1/
- cpu_changed_path.json, cpu_existing_objective.json
- public_probe_v1/failure.json, verifier_at_failure.py
- native_identity_diagnosis.json, numerical_amendment_v2.json
- public_probe_v2/public_contract.json, native_path_check.json, verification.json
- source_change_receipt.json, implementation.diff, 최종 report_receipt.json

