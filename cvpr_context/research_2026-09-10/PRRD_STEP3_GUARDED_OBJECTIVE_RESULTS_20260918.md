# PRRD 3단계: 관계 개입 제한·기능 정합의 실제 이미지 gradient 연결

작성일: 2026-09-18. 상태: **요청된 3단계 완료 — 새 목적의 실제 이미지 gradient 검증 통과. 효용 실험은 수행하지 않음.**

## 1. 연결한 연산

2단계의 point/raw 특징 및 공개 통계 파일을 SHA로 다시 결속하고, 기존 `guarded_readout.py`를 그대로 재사용했다. 모델·점별 표현·PCA·환자 가중치·관계 scale은 변경하지 않았다. 사용자가 전달한 원격 `0b06417`은 문맥상의 기준이며 이번 실행에서 원격 commit을 별도로 조회하거나 업로드하지 않았다.

새 [guarded_objectives.py](../../code_working/prrd_v3/guarded_objectives.py)는 다음을 연결한다.

1. 동일 BioViL forward에서 얻는 점별 `z`와 정규화 전 선형 관계 특징 `a`.
2. A/B의 전체 점별 bank 또는 E/E_R/C/D/R_joint의 marginal/pair bank별 전역 모멘트.
3. 목표 통계와 합성 통계에 동일한 `LearnerPolicy`를 적용한 guarded readout. 각자의 점별 행렬로 각자의 관계 수정량을 제한한다.
4. 고정된 목표 \(M_T\)와 목표 \(w_T\)를 사용한 기능 손실 \((w_S-w_T)^T M_T(w_S-w_T)\).
5. 통계 정합·기능 정합·기존 영상 규제를 거쳐 이미지와 renderer parameter까지 이어지는 미분.

전체 목적은 기존 clean 통계 정합 + `0.1 × augmented 통계 정합` + `eta × clean 기능 정합` + 기존 pixel-anchor/TV다. 기능 손실은 **무증강 경로에 한 번만** 적용한다. Synthetic learner의 제한에는 \(M_S\), 기능 비교에는 고정 \(M_T\)를 사용한다. Target 통계/행렬/가중치는 복사·detach 후 고정한다.

R_joint는 이미 공동 제한된 endpoint의 joint moments를 맞추며, 그 모멘트의 선형 변환으로 관계 learner의 평균·2차 모멘트를 얻는다. 비선형 제한과 선형 차분의 순서를 바꾸지 않았다. D의 사적 대응 교란은 목표 통계 생성 단계의 역할이며, 이번 P-only 검사에서는 C와 같은 공개 target이다. 사적 대응 효과를 검사했다고 해석하지 않는다.

두-pass에서는 clean/augmented의 전체 `z/a`를 모은 뒤 작은 모멘트·solve graph에서 전역 응답 gradient를 계산한다. Replay는 `z`와 `a` 양쪽 응답을 같은 encoder forward에 전달한다. Microbatch 내 표본 수로 모멘트를 다시 평균하지 않고, 모든 gradient 누적이 끝나기 전 parameter update도 없다.

기존 [objectives.py](../../code_working/prrd_v3/objectives.py)와 learner core는 원본 SHA 그대로다. [render.py](../../code_working/prrd_v3/render.py)는 E/E_R에도 기존 C/D/R_joint와 같은 paired residual·공유 증강 규칙을 적용하도록 relation arm 목록 한 줄만 확장했다. 옛 A/B/C/D/R_joint 동작은 유지한다. 새 목적은 별도 API이며, 기존 full-run runner를 자동 전환하지 않았다.

## 2. 수치 검사와 실제 모델 검증

검사 코드: [작은 배열 검사](../../code_working/prrd_v3/test_guarded_objectives.py), [공개 실제 모델 검사](../../code_working/prrd_v3/verify_guarded_image_objective.py).

| 검사 | 결과 |
|---|---:|
| 새 통합 목적의 구성 배열 검사 | 6/6 통과, 1.189초 |
| 기존 guarded learner core 회귀 검사 | 6/6 통과, 0.592초 |
| 독립 NumPy 전역 모멘트 최대 차이 | 5.55e-17 |
| 독립 NumPy 기능 손실 최대 차이 | 2.78e-17 |
| 구성 배열 one-pass/replay gradient 최대 차이 | 1.39e-17 |
| 실제 BioViL 전체 목적 비교 | 19/19 조건 통과 |
| 실제 전체 loss 최대 차이 | 1.14e-13 |
| 실제 encoder 입력 픽셀 gradient 최대 차이 | 0, 모든 조건 exact |
| 실제 renderer parameter gradient 최대 절대 차이 | 2.89e-6 |
| 실제 renderer parameter gradient 최대 상대 L2 차이 | 2.27e-7 |

작은 배열은 7개 arm, 독립 NumPy readout, 제한 활성/미활성/반경0·수정량0, 고정 target 복사, \(M_S/M_T\) 구별, 결합 점별 목적 상한, 활성·미활성 C/R_joint의 FP64 유한차분, 불균등 microbatch1/5/12 전역분모를 확인했다. 정확한 비미분 경계에 smooth gradcheck를 강요하지 않았다. Eta를 켜고 끈 gradient 차이를 계산해 augmented 경로에 추가 기능 gradient가 **정확히 0**임도 확인했다.

실제 모델 검사는 공개 4장 입력 두 묶음(seed101/202)을 사용했다. 각 묶음은 virtual slot4개/고유 이미지2개이며, 두 묶음 전체 고유 공개 이미지4개다. Seed101의7arm×micro1/3=14조건, seed202의A/C/R_joint×micro2=3조건, seed101 C의반경0/100×micro2=2조건으로 총19개다. Renderer는 모든 pyramid level의 미분을 확인하기 위해 step500 활성 상태를 설정했지만 **500회 학습한 것이 아니며 optimizer update는 0회**다.

검사 계수는 beta1/ridge.1/eta1, 기본 kappa.1과 edge case용 절대 rho0/100이다. 모두 수치 검사 전용이며 본실험 계수나 최적값으로 선정하지 않았다. 2단계의 P mixed3 기반 relation scale은 그대로 재사용했다.

기존 통과 기준을 유지했다.

```text
abs(loss_two_pass - loss_one_pass) <= 2e-6
gradient_max_abs <= 2e-7 + 5e-4 * reference_gradient_peak
```

최대 절대 차이2.89e-6은 rho100의 제한 미활성 stress case다. 이 조건의 reference peak는4.0012293, 기존 허용치는0.00200081이었다. 모든 조건 중 `오차/해당 허용치`의 최대값은0.002046이다. 단일 고정 절대오차2e-7만을 기준으로 통과했다고 쓰지 않는다. 픽셀 VJP는 clean/augmented 각 encoder 입력의 identity view에서 분리 측정했으며, renderer parameter까지 누적할 때의 작은 FP32 차이는 별도로 위에 기록했다.

통계·영상 규제에서만 gradient가 나온 것이 아닌지도 확인했다. 실제 공개 이미지에 대해 **기능 손실 단독** gradient norm57.6821, 최대 절대값2.94508로 유한·비영이었다. 이 값의 크기는 학습 안정성이나 효용을 뜻하지 않는다. 모델 parameter·BN buffer·공개 projection·목표 행렬 및 가중치는 검사 전후 그대로였으며, source parameter gradient가 생기지 않았다.

## 3. 실제 실행량과 보존 범위

실행 산출물은 [stage3 output](../../code_working/_reports/prrd_step3_objective_20260918_v1/)에 저장했다. [이미지 gradient 결과](../../code_working/_reports/prrd_step3_objective_20260918_v1/public_gradient/image_gradient_verification.json), [사전 수치 계약](../../code_working/_reports/prrd_step3_objective_20260918_v1/public_gradient/public_gradient_contract.json), 입력 fixture와 각 조건의 결과를 보존한다.

- 새 GPU 검증 프로세스1회, actual model forward316/backward212호출; image presentation460F/308B.
- 공개 P 이미지 decode8회/고유4장. P-only allow-list 적용; Q/V pixel0, expert/reserved0.
- 실제 모델 검사29.812초, RTX3070 peak allocated2863.30MiB/reserved3046MiB. 이는 **4장 gradient 검사 비용**이며 full128 처리량·ETA로 환산하지 않았다.
- 기존 공개 P813 통계·projection을 재사용했고 전체 P raw feature 재추출은 하지 않았다.
- Optimizer update0, 본 합성 bank0, 성능 평가0, 수신 모델 실행0, DP release0, remote upload0.
- 기존 head/LoRA 실패·Rwide 직접 실자료 결과·W1 최초 실패와 수리·2단계 기록은 보존했다.

재현 명령은 `code_working`에서 실행한다. Output/report 경로는 기존 결과를 덮어쓰지 않는 새 경로로 지정한다.

```powershell
& ./base_gate/.venv/Scripts/python.exe -B -X utf8 -m prrd_v3.test_guarded_objectives --report <new_array_report.json>
& ./base_gate/.venv/Scripts/python.exe -B -X utf8 -m prrd_v3.test_guarded_readout --report <new_core_report.json>
& ./base_gate/.venv/Scripts/python.exe -B -X utf8 -m prrd_v3.verify_guarded_image_objective --output <new_gradient_directory> --step2 _reports/prrd_step2_paths_20260918_v1/public_paths
```

## 4. 완료 범위와 남은 항목

**이번 완료는 ‘새 학습 목적이 실제 이미지 gradient까지 정확히 연결됨’이다.** 요청된 연결 검사에 남은 실패는 없다.

전체128장 profile·본실험 계수/예산 동결·full-run orchestration·수신자 실제 실행·새 목적의 full-bank export/resume 검증·환자 효용 비교는 이번에 수행하지 않았다. 따라서 전체 W1/runtime freeze는 아직 false이고,30bank는 제안 상태다. 다른 모델 전이·기흉 AUROC/AP·DP 아래 효용은 이 결과로 입증되지 않는다.

3단계 결과와 진행 계획을 기록하고 멈췄다. 이후 순서는 기존4단계이지만 이번 실행에 포함하지 않았다.
