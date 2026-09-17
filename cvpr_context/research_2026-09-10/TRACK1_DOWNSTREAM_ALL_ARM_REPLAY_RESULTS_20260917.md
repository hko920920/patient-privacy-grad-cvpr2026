# 수정된 7군 실제 학습 연결:14update 재검증 통과

2026-09-17 · 큰 단계2/방향1. **판정은 좋다. 수정된7군의 one-step integration replay가 실제 GPU 학습과 독립 검산을 모두 통과했다.** 이제 수정 kernel을 본 개발 실험에 사용할 수 있다.400/800step 수렴이나 사적 효용·DP 성능을 확인한 결과는 아니다.

[실행 전 명세](TRACK1_DOWNSTREAM_ALL_ARM_REPLAY_PROTOCOL_20260917.md)에 따라 기존 profile 입력만 사용했다. 새 생성0회,7군×1step×2회=14optimizer update였다. [이전 profile](TRACK1_DOWNSTREAM_PROFILE_RESULTS_20260917.md)의100회와 별도이며, 잘못된98회는 계속 무효다. 그98회의 loss·예측·weight 차이·arm 시간 우열을 이번 판단에 사용하지 않았다.

## 실행 진입점도 차단했다

구형 `python -m downstream_utility.run classifier`는 이제 즉시 `Legacy classifier path is invalid` 오류로 종료한다. 단순 README 주의문으로 남기지 않았다. 별도 프로세스에서 종료코드1을 확인했으며 모델 실행은 없었다.

새 [run_v2.py](../../code_working/downstream_utility/run_v2.py)는 `prepare`, `classifier_corrected`, `verify_corrected`를 제공한다. 이번 전용 runner는 기존 bank를 재사용하므로 생성 단계는 없다. `downstream_utility.data`·`downstream_utility.train`은 import hook으로 차단하고, 이미 로드돼 있어도 중단한다. 실제 금지 import와 preloaded-module 검사를 통과했다. Data_v2는 이제 구형 data의 helper까지 import하지 않는다.

실제 실행 module의 `__file__`, corrected data/train/runner SHA, 여섯 namespace를 계약과 각 run에 저장했다. 여섯 키는 `real/public`, `real/private`, `synthetic/backbone`, `synthetic/public`, `synthetic/private_only`, `synthetic/pooled`다.

현재 파일을 안전하게 바꾸기 전에 과거 실행 source 전체를 해시 사본으로 보존했다. 이전 계약의 source 경로 중 현재 수정된 것은 `historical_source_resolution`으로 당시 사본에 연결한다. 과거 계약·결과를 새 코드에 맞춰 덮어쓰지 않았다.

## 실제 입력과 업데이트 검증

| Arm | 실제 앞16장 | 실제 뒤16장 | 두 반복 |
|---|---|---|---|
| R0 | 공개 real | 공개 real | exact |
| R1 | 공개 real | 공개 real | exact |
| S0 | 공개 real | backbone synthetic | exact |
| S1 | 공개 real | public-head synthetic | exact |
| S2 | 공개 real | private-only synthetic | exact |
| S3 | 공개 real | pooled synthetic | exact |
| Dreal | 공개 real | private real | exact |

모든 실행은 같은 ImageNet ResNet18 초기 state `05a226d52a9dfa194eb33852ad5fd03d967c35801f49f565a630e309ca91d729`, seed11, 새 AdamW state0개, lr1e-4/weight_decay1e-4, FP32로 시작했다. 각 batch32장은 positive16/negative16이며 각 source 절반도8/8이었다.

14회 모두 finite loss·gradient와 AdamW step1을 확인했고, 매 실행에서 **62개 trainable parameter tensor가 실제로 바뀌었다.** BatchNorm buffer 변화만 보고 학습됐다고 판단하지 않았다.

선택된 고유 파일61개는 공개 real22장·private real15장·각 synthetic method6장씩이었다. 역할 문자열만 보지 않고 다음을 확인했다.

- 실제 raw/PNG file SHA → 독립 decode/224 letterbox pixel SHA → 실제 array key/index.
- Image ID·patient/cell ID·label·generated method → batch trace.
- 저장한 실제 GPU 입력 tensor → 독립 raw부터 GPU affine/정규화 재계산.
- 모든 arm의 앞 public16 동일, R0/R1의32개 ID 동일.
- R0에는 augmentation이 없고 R1에는 적용돼 실제 입력이 다름. R1/S0–S3/Dreal의 공통 real16 전처리 결과는 exact.
- S0–S3의 뒤16은 같은 cell/요청 label이며, 각 위치의 실제 method pixel SHA는4개 모두 다름.
- Dreal 뒤16은 실제 private 영상이며 synthetic alias가 아님.

두 반복의 선택 ID·role·patient/cell·label·array index·원본/pixel SHA·실제 전처리 입력·logit·loss·최종 state는7군 모두 exact였다. 독립 검산의 BCE 최대 오차는3.875×10^-8로 고정2×10^-6 이하였고, raw→GPU 전처리는7군 모두 exact였다. 독립 검산은 생산 data/train 함수를 import하지 않았으며 표준 PIL/torchvision primitive는 공유했다.

## 기술적 중단과 resume

첫 R0 update의 trace·입력·최종 state를 정상 저장한 뒤, 진행 로그 출력의 `json` import 누락으로 프로세스가 종료됐다. 학습 계산 실패가 아니었다. 원 코드를 사본으로 보존하고 로그 import와 **이미 저장된 R0_0 한 건만 건너뛰는 resume**을 별도 amendment로 고정했다.

기존1회의6개 산출물 SHA와 source/array 계획·초기값을 확인하고, 나머지13회만 실행했다. 반복 첫 결과와 재시작 후 R0 두 번째 결과도 exact였다. 총14회를 넘기거나 처음부터 다시14회를 돌리지 않았다. 모델·optimizer·자료·seed·허용오차는 바뀌지 않았다. 별도의 진입점 차단 테스트에서는 Windows stderr decoding 문제를 binary capture로 고쳤으며 그 테스트의 모델 실행은0이었다.

## 시간·메모리

단위는 초이며 각 칸은 반복0/1이다. 이번 값은 one-step 기록과 입력 저장 비용을 포함하므로 장기 처리량 또는 방법 간 비용 우위를 뜻하지 않는다.

| Arm | one-step 총시간 | 전처리 | forward/backward | optimizer |
|---|---:|---:|---:|---:|
| R0 | 2.456 /2.647 | .008 /.007 | 2.400 /2.570 | .013 /.024 |
| R1 | .184 /.199 | .042 /.055 | .082 /.078 | .021 /.020 |
| S0 | .186 /.173 | .044 /.045 | .080 /.076 | .017 /.015 |
| S1 | .174 /.172 | .033 /.034 | .084 /.081 | .016 /.018 |
| S2 | .242 /.218 | .038 /.039 | .091 /.080 | .028 /.017 |
| S3 | .456 /.254 | .034 /.034 | .108 /.125 | .017 /.014 |
| Dreal | .412 /.204 | .041 /.036 | .104 /.080 | .015 /.016 |

R0은 두 프로세스의 첫 학습이라 초기 GPU 연산 비용이 포함됐다. “R0이 synthetic arm보다 느리다”는 결론은 부적절하다. 나머지 편차에도 짧은 실행·파일 저장·순서 효과가 있다. Peak allocated는 최대0.837GiB, reserved는 최대1.172GiB였다. 독립 검산은4.030초였다.

본512장 생성은 이전 유효 profile의 약22.4분 연산(+상세 trace 저장 약3.6분)을 계획값으로 유지한다. Classifier21run·공개 calibration·분석까지 전체90–150분은 아직 잠정 ETA다. 본실험에서 각 arm 첫10–20step의 실측으로 다시 계산한다. 무효98회의 arm 비교로 비용 우위를 주장하지 않는다.

## 연구 상태와 다음 실행

| 항목 | 상태 |
|---|---|
| 생성 sampling·실자료 binding | 이전 통과 유지 |
| Legacy classifier 우발 실행 방지 | 이번 실제 차단 검사 통과 |
| 수정7군 one-step 학습 연결 | 이번14update 통과 |
| 400/800step 장기 수렴·moment 누적 | 미검증 |
| 512장 본 bank·공개 classifier calibration·21run | 미실행 |
| S2/S3의 downstream 사적 추가 효용 | 미판정 |
| 새 의료 patient-DP·DP-LoRA | 미실행 |
| Expert final·reserved | pixel/prediction 접근0, 보존 |

다음은 [통합 명세](TRACK1_DOWNSTREAM_MASTER_PROTOCOL_20260917.md)의 본 비DP 개발 실행이다. 이번 수정 source SHA를 본 계약에 결속하고512장 bank·공개 calibration·21개 classifier run을 수행한다. 현재 `run_v2`는 한정 재검증용이므로 본512장/21run orchestration을 별도 명시적으로 확장해야 하며, 구형 classifier 분기를 재사용하지 않는다.

진행 판단은 method-development에서 S2 또는 S3가 S1과 R1 모두보다 평균 AUROC .01 이상 높고,3개 classifier seed 중 최소2개에서 같은 방향이며 AP를 낮추지 않는지다. 이 조건을 보기 전 DP로 확대하지 않는다. Expert final은 비DP·DP 및 모든 분석을 동결한 뒤에만 연다. 기존192장과 두 자동평가기 실패는 바꾸지 않는다.

## 근거 파일

- [새 계약](../../code_working/_reports/downstream_all_arm_replay_20260917_v1/contract.json), [안전 진입점 실제 차단](../../code_working/_reports/downstream_all_arm_replay_20260917_v1/entrypoint_safety.json).
- [실제14회 결과](../../code_working/_reports/downstream_all_arm_replay_20260917_v1/result.json), [독립 검산](../../code_working/_reports/downstream_all_arm_replay_20260917_v1/verification.json), [1+13회 resume 기록](../../code_working/_reports/downstream_all_arm_replay_20260917_v1/logging_fix_amendment.json).

이번 결과는 로컬 실행·검산이다. 원격 main의 현재 SHA나 push 완료를 별도로 확인했다는 주장은 하지 않는다.
