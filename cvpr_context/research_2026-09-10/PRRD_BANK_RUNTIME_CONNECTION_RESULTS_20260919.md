# PRRD 실행기·저장·프로세스 재개·PNG 연결 결과 — 2026-09-19

**요청 범위에서 통과했다.** 고정된 guarded objective를 본 실행기에 연결하고, 공개 4장으로 연속 실행과 실제 프로세스 종료 후 재개를 비교했다. 최종 parameter·optimizer·난수·step·pyramid 상태 및 loss trace의 차이는 모두 0이었다. 배포 PNG의 픽셀·label·virtual pair·hash도 일치했다.

이번은 실행 정합성 결과다. 정식 128장×500회 학습, 환자 효용, 다른 encoder 전이, DP 결과가 아니다. 새 profile도 수행하지 않았다. 사용자가 전달한 원격 b306c5a는 출처로 기록하며 이번에는 원격을 새로 조회하지 않았다.

## 1. 변경과 보존

기준은 [첫 비DP 계약](PRRD_FIRST_NONDP_EXECUTION_CONTRACT_20260918.md)이다. beta1(A/B0), ridge0.1, 상대 rho²=0.1·w0ᵀMw0, eta1 및 30bank 목록은 바꾸지 않았다. 기존 25개 소스 파일과 계약·공개 특징·목표 생성 규칙의 hash는 그대로다.

추가 파일:

| 파일 | 역할 |
|---|---|
| bank_runtime.py | 전체 update 실행, checkpoint·상태 복원, 완료 봉인, PNG 검증 |
| fit_banks.py | 결속된 공개 template·목표·source를 동일 실행 경로에 연결 |
| run.py | bank 단위 CLI, 별도 본실행 권한·예산·입력 receipt 확인 |
| test_bank_runtime.py | 구성 배열 기반 저장/복원·오염/덮어쓰기 방지 검사 |
| verify_bank_runtime.py | 공개 소수 입력의 별도 프로세스 비교 및 독립 저장 결과 대조 |

사용법과 경계는 [BANK_RUNTIME.md](../../code_working/prrd_v3/BANK_RUNTIME.md)에 기록했다. 본실행 명령은 구현됐으나 승인된 bank·시간·사적 목표 receipt가 없으면 모델/픽셀 실행 전에 거부한다. 이 검사를 통과했다는 사실로 30bank가 자동 승인되지는 않는다.

## 2. 검사 전에 고정한 조건

[사전 검증 계획](../../code_working/_reports/prrd_bank_runtime_20260919_v1/predeclared_verification_plan.json)을 GPU 실행 전에 저장했다.

- 기존 C128 seed101 명부의 첫 공개 P template 4장, C arm, microbatch4.
- 기존 P-only C 목표와 source projection·scale·전처리 사용.
- 연속 84회와 82회 저장·종료 + 새 프로세스 2회, 실제 update 총168회.
- 기존 pyramid 일정 1/81/161/241/321/401 유지. 이번에는 81회 활성 전환을 지나 active2 상태를 저장·복원.
- 복원 직후 parameter/optimizer/RNG/step/active flags는 exact를 요구.
- 재개 후 parameter·optimizer는 atol1e-7/rtol1e-6, loss trace는 atol1e-8/rtol1e-6.
- PNG pixels·labels·pairs·SHA와 정수 상태는 exact. Encoder와 target은 불변.
- 검사 산출물은 TECHNICAL_TEST_ONLY이며 본실험에 재사용 금지.

허용오차를 사후에 확대하지 않았다. 실제 결과는 선언한 허용오차보다 강한 bitwise 일치였다.

## 3. 실제 프로세스 재개 결과

| 경로 | PID | 실제 successful updates | 결과 |
|---|---:|---:|---|
| 연속 | 41108 | 84 | 종료, 검사 전용 final 저장 |
| 중간 저장 | 8764 | 82 | checkpoint 저장 후 종료, final/PNG 미생성 |
| 새 프로세스 재개 | 34716 | 2 | 83·84 실행 후 검사 전용 final 저장 |

세 프로세스는 각각 종료했으며 서로 다른 PID를 확인했다. 재개 프로세스는 시작 RNG를 일부러 다르게 만든 뒤 checkpoint에서 복원했다.

| 대조 항목 | 결과 |
|---|---|
| checkpoint 복원 직후 전체 상태 | exact |
| 최종 parameter·optimizer 및 중첩 상태 | tensor16개/scalar1283개 전부 exact |
| 최대 절대 / 상대 L2 차이 | 0 / 0 |
| 84개 step 전체 loss/진단 trace 차이 | 0 |
| 다음 step | 양쪽 모두85 |
| 최종 pyramid / requires_grad | active2; true,true,false,false,false,false |
| 해상도별 optimizer step | 84,4,0,0,0,0 |
| 누락·중복 successful step | 없음; 양쪽 모두1..84 |
| 고정 encoder weight·buffer·projection / target | 불변 |

코드 자체가 갱신을 생략한 것이 아닌지도 확인했다. 초기와 최종 합성 parameter가 달랐으며, 실제 모델 forward/backward와 optimizer update를 계수했다.

- 실제 source forward: 672 calls / 2,688 image presentations.
- 실제 backward: 336 calls / 1,344 image presentations.
- 공개 P: 고유4장, 세 프로세스 합12decode.
- 자식 프로세스 wall time 합82.25초. 이는 검증 실행량 기록이며 새 throughput profile이나 본학습 ETA가 아니다.

## 4. PNG와 완료 상태

연속·재개 경로 모두 마지막84회에서만4장 PNG를 내보냈다. 이는 검사 전용 final이며, main bank의 마지막500회 조건을 변경하지 않는다.

- 224×224, grayscale L, uint8 및 독립 np.rint(clip(float)*255) 계산과 decoded pixels 일치.
- 4장 모두 연속/재개 pixel·SHA 동일, images.csv의 label·bank_type·파일 binding 일치.
- marginal2장과 virtual pair1쌍(2장)의 연결 일치.
- 82회 중간 상태에는 artifact 및 COMPLETED가 없음을 재개 전에 확인.
- 임시 export를 검증한 뒤 디렉터리를 게시하고 COMPLETED를 마지막에 기록.
- 이미 완료된 bank는 재실행·덮어쓰기 거부. 게시 직후 완료 표시 전 중단은 같은 bytes를 확인해 복구.
- 생성된 총8 PNG는 검사 증거이며 본실험 candidate가 아니다.

## 5. 저장·거부 검사와 한계

CPU 구성 배열 검사6개도 통과했다. 이는 의료 모델 성능 검사가 아니다.

계약 불일치·checkpoint byte 오염, 중간 checkpoint의 final export, 완료 bank/기존 파일 덮어쓰기, manifest 변조를 거부했다. A/C의 PNG 레이아웃과 게시 후 완료 표시 전 복구도 검사했다. 본학습 승인이 false이면 모델 생성 전에 거부되는 것을 확인했다.

Checkpoint는 model/optimizer/Python·NumPy·Torch CPU/CUDA RNG·completed/next step·pyramid/gradient 활성 상태를 포함한다. 파일과 hash receipt를 먼저 저장하고 latest pointer를 마지막에 교체한다. OS process lock은 프로세스 종료 시 해제된다. 원자적 저장을 완료하지 못한 임시 파일을 완료 checkpoint로 취급하지 않는다.

숫자 시간 상한은 전체 update 사이에서 확인한다. 비정상 종료로 예산 정산이 끝나지 않으면 미정산 예약을 임의로 초기화하지 않는다. 동일 bank의 checkpoint는 보존되며, 예산 정산 또는 추가 허용 범위를 결속한 뒤 재개한다. OS 수준의 엄격한 실시간 강제 종료 보장은 아니다.

검증 범위는 공개 C4의 실제 BioViL 경로와 CPU A/C export 경로다. 30bank 전체·500회·private target 생성·수신 평가가 실행됐다고 확대하지 않는다.

## 6. 남은 실행 차단 항목

1. **허용 bank ID와 숫자 시간 상한**. 현재 허용 목록은 비어 있고 본학습은 미승인이다.
2. **별도 Q 준비 권한과 실제 산출물 hash**. 목표 생성 규칙은 이미 고정됐지만, 실제 Q raw/point feature와 해당 arm/seed 목표 receipt는 아직 없다.
3. 후속 본실행 권한 파일에 위 범위와 이번 코드/검증 receipt를 결속한다. 이는 허용 범위의 파일 결속이며 새 연구 관문이나 추가 profile이 아니다.

실행기 연결은 이번 범위에서 완료했다. Recipient 평가 구현/결속·DP·Expert/Reserved는 각 후속 단계에 남으며 이번에는 열지 않았다.

## 7. 증거와 현재 상태

- [실제 비교 결과](../../code_working/_reports/prrd_bank_runtime_20260919_v1/verification.json)
- [코드 추가 및 원 코드 보존 receipt](../../code_working/_reports/prrd_bank_runtime_20260919_v1/implementation_receipt.json)
- [검사 전 결속한 입력·코드](../../code_working/_reports/prrd_bank_runtime_20260919_v1/verification_job.json)
- [CPU 파일 검사](../../code_working/_reports/prrd_bank_runtime_20260919_v1/cpu_checks.json)
- [단계 기록 JSON](spec_sources/prrd_bank_runtime_connection_record_20260919.json)

새 main bank0, profile0, Q/V pixels0, recipient0, DP0, Expert/Reserved0, 원격 업로드0. 기존 head·LoRA 실패, Rwide 실자료 결과와 데이터 역할/선택 이력을 보존했다. 연구 큰 단계2 및 기존 실제 효용 포인터를 유지한다. 요청한 연결 검증을 마치고 중단했으며 다음 실행을 예약하지 않았다.

