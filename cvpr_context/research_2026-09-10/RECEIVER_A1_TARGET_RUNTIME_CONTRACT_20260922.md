# A1 조건별 목표 준비와 실행 연결 계약 — 2026-09-22

범위: 큰 연구단계2에서 A1의 실제 P/Q 목표 준비, 변경된 단일 encoder 실행·저장·재개 연결, 본학습 비용 산정까지 한 묶음으로 수행한다. 사용자 요청에 따라 정식500회 합성·A2·계수 탐색·V 성능·DP·Expert/Reserved는 실행하지 않는다.

예상: P/Q5910장×K4=23640 image-forward, backward0인 목표 추출6–12분; 구현·검산·기록 포함25–45분. 기존 B49분을 A1 학습시간으로 복사하지 않는다.

## 고정 입력과 목표

- P672명/813장 공개, Q2027명/5097장 보호 대상. 전체2699명. Class-present 환자수P669/6, Q2015/114, pooled2684/120. Mixed 환자는 두 class에 속할 수 있다.
- 기존 BioViL checkpoint, point 전처리·P-only PCA16·q95 scale 유지. 기존 clean 자료/목표/분류기와 과거 실패 산출물 보존.
- 첫 구현의 fixed condition/head K4, public seed101, encoder ID biovil, CE weight/bias gradient 그대로 사용한다. Native1024에서 각 공유 상대좌표 변환 후 기존512 resize/448 crop을 거친다.
- 각 조건을 별도 축으로 보존한 특징[K,N,16], target[K,34]를 저장한다. 조건별 환자/class sum과 count를 저장하며, pooled target은 P/Q 합·count 결합 후 class0/1을 각1/2로 가중한다. 조건끼리 평균 target으로 합치지 않는다.
- 특징은FP32, 통계/선형 head/CE derivative는FP64. Target은 사적 내부 비DP 자료다. Private image/patient ID를 공개 보고서나 PNG에 넣지 않는다.
- 추출 image microbatch16, image decode는 각P/Q 원영상1회. 변환 픽셀 저장 없음. 별도 NumPy CE 계산과 환자별 직접 가중 집계로 저장 특징을 검산한다. Target/class sum 허용오차1e-12를 실행 전에 고정한다.
- 실제 head·변환 tensor, checkpoint/projection/전처리 코드/명부/target의 hash를 결속한다. 파일 hash 외에 condition 순서·내용도 loader에서 검증한다. 이전 augmean 파일은 이번 target으로 허용하지 않는다.

## 실행 연결의 한정된 검사

기존 renderer, 원자적 저장, 파일 잠금, checkpoint의 parameter/Adam 상태/RNG/next step/pyramid 복원, PNG rounding·검증 함수를 유효한 범위에서 재사용한다. 새 objective용 adapter/learning metadata만 별도로 구현하며, 과거 guarded objective를 새 gradient 목적이라고 재명명하지 않는다.

- CPU: 실제 target handoff 로드, 다른 target/조건 hash와 잘못된 checkpoint 재개 거부, 조건 축 보존. 작은 구성 배열의 변경 경로만 검사한다.
- 실제 공개P4: 연속3update와2update 저장→새 프로세스 재개1update 비교. 두 경로 합계6개의 기술 update. 원래83/84 pyramid 경계 검사를 반복하지 않고 검증된 복원 코드를 재사용한다. 전체 parameter/optimizer/RNG/next step의 digest는 exact, loss차이≤2e-6, 최종PNG·label·hash는 exact가 기준이다.
- 공개P128: microbatch16, 모든 pyramid 수준 활성화 위치(step500), warmup1+measurement3의 전체K4 two-pass→finite gradient→optimizer update를 측정한다. 이는 실제500회 학습이 아니라4개의 기술 update다. P target만 사용하고 Q 기반 성능 후보를 만들지 않는다.
- 기술 검사 optimizer는 기존 실행기의 AdamW(lr0.01, weight_decay0, betas0.9/0.999, eps1e-8, foreach=False)를 사용한다. 계수 탐색은 없고, 이 값이 새 목적에서 성능 최적임을 주장하지 않는다.
- 본학습안은128장·500회·seed101·microbatch16을 유지하는 첫 recipe다. 실제 실행기와 ready job에 모든 수치/target/code를 포함하되, 본학습 실행·개발 판정 기준·수치 시간 상한은 이번 검사로 승인된 것으로 취급하지 않는다.

## 완료와 해석

완료물은 실제 조건별P/Q/pooled 목표 파일, 독립 검산, 새 hash에 묶인 단일 encoder 실행 연결, 공개 technical resume/PNG·전체update 비용, 남은 본학습 항목이다. 검산 성공은 receiver 전이·의료 효용이나 DP 보장이 아니다. DenseNet/V는 적응적 개발용이고 새로운 unseen/final을 열지 않는다.

이번 source당 update 계산은128장×K4×two-pass로1024 image-forward와512 image-backward다. 500회는512000F/256000B이며, 목표추출23640F 및 최종저장·평가 비용은 별도다. 기존 B의 clean+aug2경로보다 encoder image-evaluation 수가2배라는 계산량 비교이지, 시간도 정확히2배라는 보장이 아니다.

