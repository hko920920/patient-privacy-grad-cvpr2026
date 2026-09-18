# PRRD W1 gradient 불일치 수리 결과 — 2026-09-18

**판정: 실행 연결 수리에는 긍정적이다. PRRD의 환자 판별·전이·DP 효용은 이번에 평가하지 않았다.** 기존 실제 BioViL 경로의 gradient 검사를 같은 허용오차로 통과했다. 현재 완료 범위는 사용자가 제시한 1단계다. 새 관계 경로·개입 제한·기능 손실 연결, 전체128장 profile 및 W2는 미실행이다.

## 1. 고정한 범위

기존 목적을 유지하고 원 실패 seed101/A/step500을 재현했다. 공개 template의 slot `[0,32,64,96]`, 전처리, renderer 초기 상태, augmentation, checkpoint/PCA, dtype과 코드 해시를 저장했다. 별도 seed202 공개 fixture를 추가해 같은 절차로 검증했다. 각 fixture는4개 slot/2개 고유 공개영상이며 두 fixture 합계4개 고유영상이다. 독립 환자 효용 표본이나128장 본 bank가 아니다.

- [고정 재현 입력](../../code_working/_reports/prrd_w1_repair_20260918/original_fixture/w1_reproducer.json)
- [수정 전 경계별 오차](../../code_working/_reports/prrd_w1_repair_20260918/original_fixture/w1_boundary_errors.json)
- [실제 모델 재검증 결과](../../code_working/_reports/prrd_w1_repair_20260918/verification_v1/w1_repair_verification.json)
- [수정 전 source 사본과 해시](../../code_working/_reports/prrd_w1_repair_20260918/source_before_repair_manifest.json)

Pixel allow-list는 P만 허용했다. 기존 P 특징 cache와 projection을 재사용했다. 명부 metadata 확인과 실제 pixel 접근을 구분하며, Q/V pixel·새 성능·본 bank·DP release·expert/reserved 접근은0이다. 별도 수신 모델 추론도 수행하지 않았다.

## 2. 원인과 최소 수정

기존 비교용 reference는 renderer와 augmentation을 전체 batch로 계산하고 encoder만 microbatch로 나눴다. 실제 replay는 renderer부터 microbatch로 계산했다. 두 reference가 실제로 같은 픽셀을 모델에 넣는다는 전제가 깨져 있었다.

고정 fixture에서 최초 차이는 renderer의 첫 residual resize였다. 전체4장 대 microbatch1의 resize 차이는 최대 `3.4924596548080444e-10`, 최종 sigmoid 픽셀 차이는 `5.960464477539063e-08`이었다. 기존 `resampling.resize`의 batched FP32 행렬곱 실행 크기가 달라졌고, 차이는 encoder 이전에 이미 생겼다. 특정 activation이 바뀌었다는 별도 원인 추정까지 확인한 것은 아니다.

수정은 retained-graph one-pass도 renderer→augmentation→encoder 전체를 같은 microbatch 순서로 계산하도록 한 것이다. 모든 특징을 이어 붙인 후 **전체 bank 모멘트 손실을 한 번 계산하고 한 번 backward**한다. Two-pass는 기존처럼 전역 response를 계산하고 재계산한 각 chunk로 역전파한다. Microbatch별 모멘트 손실 평균으로 바꾼 것이 아니다. Prior/TV도 전체 이미지 수에 대한 가중치를 유지한다.

수정 후 같은 분할의 reference와 replay first pass는 원 fixture의 픽셀·증강·전처리·특징·feature response에서 bitwise 일치했다. 동일 입력과 동일 upstream gradient를 준 encoder의 픽셀 VJP, renderer VJP도 해당 검사에서 정확히 같았다. 전체 bank와 다른 microbatch 구성이 모든 단계에서 bitwise 같다고 주장하지 않는다.

| 코드 | 실제 변경 |
|---|---|
| `prrd_v3/objectives.py` | `one_pass(..., microbatch=...)`에 전체 연산 분할을 적용. AST 대조상 변경 함수는 one_pass뿐 |
| `prrd_v3/profile.py` | encoder만 나누던 reference 대신 위 reference를 사용. loss 차이도 같은 분할끼리 보고 |
| `prrd_v3/w1_boundary_debug.py` | 고정 재현 입력·경계별 비교·수정 전 실패 보존 |
| `prrd_v3/w1_repair_verify.py` | 다른 공개 fixture, 기존3목적,3microbatch, 독립 FP64 모멘트/Adam 및 실제 VJP 검증 |

Production `two_pass`, 모멘트/영상 손실, renderer, resampling, encoder, 모델 가중치 및 새 guarded core는 변경하지 않았다. 따라서 연구 방법의 효과를 높인 변경이 아니라 gradient 검사의 기준 경로를 바로잡은 수리다.

## 3. 전후 결과와 허용오차

기존 기준 `max_abs <= 2e-7 + 5e-4 × reference_peak`, loss 차이 `<= 2e-6`을 유지했다. 결과를 보고 기준을 늘리지 않았다.

| 검사 | 최대 gradient 절대차 | 상대 L2 | 판정 |
|---|---:|---:|---|
| 원 실패 재현 A/seed101/micro1 | 5.1300768973e-5 | 2.5549043481e-4 | 기존 기준 FAIL |
| 같은 실행의 전체 분할 reference | 1.4901161194e-8 | 1.4164407498e-7 | 기존 기준 PASS |
| 수정 코드 별도 재검증 A/seed101/micro1 | 1.1175870895e-8 | 1.3500979647e-7 | 기존 기준 PASS |
| 공개2fixture × 기존 A/C/R_joint × micro1/2/4 전체18조건의 최대 | 1.3224780560e-7 | 2.1091048701e-7 | 18/18 PASS |

전체18조건 loss 차이 최대 `1.1368683772161603e-13`. 독립 NumPy FP64 모멘트 값/gradient 대조의 최대 차이는 각각 `1.1102230246251565e-16`이다. Source parameter와 buffer 전체 해시는 실행 전후 동일했다. 여기의 C/R_joint는 **기존 endpoint 목적**이며 새 raw 관계·guard/function 목적의 실제 모델 통과를 뜻하지 않는다.

## 4. 한 번의 optimizer 갱신

각 조건의 두 경로에서 같은 초기 상태와 fresh AdamW(lr.01, wd0, betas.9/.999, eps1e-8, foreach=False)를 사용했다. 총36회의4-image probe update이며 본 합성 bank 학습이 아니다.

갱신 후 parameter 최대 차이는 `3.87274194508791e-05`, 갱신 후 배열 기준 상대 L2 차이 최대는 `1.0752055599678217e-05`였다. 이 차이를0이나 bitwise 일치로 표시하지 않는다.

각 경로에서 관측한 gradient를 독립 FP64 첫 Adam 갱신식 `p-lr*g/(abs(g)+eps)`에 넣어 대조했으며, 실제 parameter와의 최대 차이는 `3.1135125345971293e-09`였다. 따라서 작은 gradient 잔차가 Adam에서 서로 다른 갱신을 만드는 것도 관측·기록했다. 갱신 간 차이를 새로운 임의 허용오차로 통과시킨 것은 아니며, 기존 gradient/loss 기준 통과와 Adam 계산식 검증을 구분한다. 장기 trajectory 일치나 모든 입력에서의 안정성을 이번 검사로 주장하지 않는다.

## 5. 실제 실행량과 회귀검사

- 최초 경계 진단: 고정 seed101 공개4slot, micro1/2/4 비교, 진단 loop 약6.000초. Optimizer update0.
- 수리 재검증: 공개2fixture/18조건, 약25.967초(모델 준비 및 검사 포함). Instrumented BioViL forward292호출/472image presentations, backward184호출/304image presentations, probe Adam update36회.
- 재검증 pixel decode8slot/고유 공개파일4개. 최초 진단의4slot을 합하면 두 도구의 decode는12회이며 고유 공개영상은4개다. P672/813 전체를 새로 추출한 것은 아니다.
- 기존 CPU 회귀검사 `python -B -X utf8 -m prrd_v3.test_core`: 6개 PASS,2.977초. Pillow `mode` deprecation warning1건은 기존 경로의 경고이며 실패가 아니다.
- 위 시간은 검증 실행 시간이다. 구현·문서 작업 전체 시간이나 full128 처리량·W2 ETA로 쓰지 않는다.

## 6. 재현 명령과 중단 위치

`code_working`에서 동일 환경의 Python으로 아래 명령을 실행한다. 결과 디렉터리는 기존 결과를 덮어쓰지 않는 새 경로여야 한다.

```powershell
base_gate/.venv/Scripts/python.exe -B -X utf8 -m prrd_v3.w1_repair_verify --original _reports/prrd_w1_repair_20260918/original_fixture --output _reports/prrd_w1_repair_20260918/verification_replay_new
base_gate/.venv/Scripts/python.exe -B -X utf8 -m prrd_v3.test_core
```

현재 status는 `PASS_EXISTING_OBJECTIVE_ACTUAL_MODEL_GRADIENT_REPAIR_ONLY`다. 전체 `W1_passed`, `runtime_ready`는 아직 false다. 다음 정해진 작업은 사용자의2단계인 **점별 특징 유지 + raw 관계 경로·환자 통계 연결**이며 이번에는 착수하지 않았다. 이후 새 학습 규칙 연결, 전체128장 profile을 거쳐 같은 계약의 비교로 진행한다.

첨부 계획의 고정 `rho=.1`과 profile5+10은 로컬 계획의 상대 반경 `rho²=.1*w0ᵀMw0`, profile20+30과 다른 제안이다. 이번 수리는 이 값을 사용하거나 변경하지 않았으며 자동 채택하지 않는다. 30bank 제안·DP·final 접근 계약도 확대하지 않았다. 첨부의 remote `1663328`은 사용자 제공 정보이고 이번에는 remote 확인·upload를 하지 않았다. 실행 근거는 저장된 로컬 source/checkpoint/input 해시다.

기존 실패·실자료 Rwide 결과·private head/LoRA 실패·자료 역할은 보존했다. 본 결과는 논문 효용 표의 양성 결과로 계산하지 않는다.
