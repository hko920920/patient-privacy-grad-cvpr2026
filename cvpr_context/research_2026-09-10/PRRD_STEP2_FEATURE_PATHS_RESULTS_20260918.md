# PRRD 2단계: 점별·raw 관계 경로 연결 결과 — 2026-09-18

**판정: 요청한 공개 특징·환자 통계 연결은 통과했다. 관계정보의 판별 효용·합성 전이·DP 성능은 평가하지 않았다.** 기존 점별 특징을 유지한 채 실제 정규화 이전 특징에서 환자 대조를 만드는 경로를 구현하고, 저장 특징의 독립 재계산을 완료했다. 이번 작업은2단계에서 멈춘다.

## 1. 구현한 경로

Source는 기존 BioViL-T image checkpoint와 전처리를 그대로 사용한다. Raw 위치는 설치된 `health_multimodal.image.model.model.ImageModel.forward_post_encoder`의 다음 반환값이다.

```python
projected_global_embedding = torch.mean(projected_patch_embeddings, dim=(2, 3))
```

128차원 `h`를 PRRD wrapper의 외부 `F.normalize` 전에 받는다. 전체 공개 추출에서 모델 출력과 projected patch의 공간 평균이 같은 tensor 값인지 확인했다. 모델 내부의 비선형 연산 이전 특징이라는 뜻은 아니다.

기존 `Encoder.raw()`는 source에서 이미 L2 정규화한 값을 반환한다. 그 의미와 동작을 보존하고, 새 `pre_normalized`, `raw_and_point`, `forward_paths` API를 추가했다. `forward_paths`는 한 번의 encoder forward로 두 경로를 만든다.

```text
동일한 BioViL 출력 h
  ├─ 기존 L2 → 기존 공개 중심/PCA16 → 기존 norm cap → 점별 z
  └─ 새 공개 raw 중심 → 동일한 저장 PCA16 축          → 선형 관계 a

환자별 a의 양성·음성 평균 → (평균양성−평균음성)/2 → 공개 scale로 제한
```

수식은 `a=(h−mu_raw)@basis`, `d_i=(mean(a_i,+)−mean(a_i,−))/2`, `r_i=d_i/max(R_C,||d_i||)`다. 기존 점별 PCA를 raw 특징에서 다시 학습하지 않았다. Raw 중심은 P의 환자별 평균을 같은 질량으로 평균하며 label은 사용하지 않았다. 전체 환자의 점별 모집단과 mixed 환자의 관계 모집단은 분리해 정규화한다.

새 `relation_paths.py`는 E/E_R/C/D/R_joint의 endpoint·대조·제한 및 환자 통계를 명시적으로 계산한다. D는 Q mixed에만 permutation을 받는 인터페이스이며, 실제 이번 입력에는 Q가 없어 P의 C/D가 동일하다. Q 대응 교란과 제한 후 평균 변화는 구성 배열에서만 검사했다. 기존 endpoint 목적의 `patient_moments.py`와 합성 objective는 보존했다. 새 관계 코드가 아직 전체 bank 학습 loss에 연결됐다는 뜻은 아니다.

## 2. 기존 점별 출력 유지

공개 P672명/813장을 기존과 같은 순서·feature microbatch4로 새로 인코딩했다. 이후 전체 P projection은 기존 target 계산과 같은813행 연산으로 비교했다.

| 확인 | 결과 |
|---|---|
| 새 `normalize(h)` 대 기존 저장 normalized cache | 813장 전체 bitwise 일치 |
| 새 점별 z 대 기존 cache의 같은 projection 출력 | 813장 전체 bitwise 일치 |
| 환자별 class 평균·2차 모멘트 및 전체 점별 통계 | 기존 경로와 정확히 일치 |
| P의 class별 환자 수·mixed 수 | 음성669 / 양성6 / mixed3, 기존과 동일 |
| 고정 공개 이미지2장의 기존·새 점별 출력과 입력 gradient | bitwise 일치 |
| 모델 parameter·buffer·eval 상태 | 실행 전후 유지, parameter gradient 없음 |
| 기존 point cache와 PCA 파일 | 원 해시 유지 |

점별 특징을 raw 점수로 전환하거나 환자 가중치를 바꾸지 않았다. 위 일치는 같은 연산 분할의 비교다. 서로 다른 batch 크기의 모든 FP32 연산이 bitwise 같다는 주장은 아니다.

## 3. 별도 raw 저장과 관계 계산 검산

기존 normalized cache를 raw로 재명명하지 않았다. 실제 새로 추출한 h, 공개 raw 중심과 기존 PCA 결속, z/a, 환자 통계 및 관계 target을 새 디렉터리에 저장했다.

- [추출·연산 계약](../../code_working/_reports/prrd_step2_paths_20260918_v1/public_paths/feature_path_contract.json)
- [실제 모델 경로 결과](../../code_working/_reports/prrd_step2_paths_20260918_v1/public_paths/public_feature_paths_result.json)
- [독립 저장 특징 검산](../../code_working/_reports/prrd_step2_paths_20260918_v1/public_paths/independent_feature_statistics_verification.json)
- [공개 관계 scale 및 선택 규칙](../../code_working/_reports/prrd_step2_paths_20260918_v1/public_paths/source_P_relation_scales.json)

새 raw h의 norm 최소/중앙/최대는 약 `0.435682 / 0.769831 / 2.883406`이다. 정규화된 단위 벡터 cache와 구별된다.

독립 verifier는 생산 코드의 encoder·환자 통계 함수를 import하지 않고 저장된 h/z/a를 NumPy로 재계산했다. 기존 FP32 상수를 FP64에서 계산한 특징 차이와, 저장 특징에서 FP64로 계산한 통계 차이를 구분했다.

| 비교 | 최대 절대차 |
|---|---:|
| 독립 FP64 계산 대 저장 점별 z | 1.2575902786e-7 |
| 독립 FP64 계산 대 저장 선형 a | 2.3933404503e-7 |
| raw 중심·환자 평균·관계 모멘트·scale 재계산 전체 | 1.3877787808e-16 |

실행 전 고정한 특징 기준 `atol2e-6, rtol1e-5`, 통계 기준 `atol1e-12`를 통과했다. 기존 출력 대 새 출력의 정확 일치 검사와 혼용하지 않았다.

공개 mixed3명의 norm에 동일 질량 empirical inverse-CDF q95, 보간 없음, floor1e-12를 적용했다. 현재 표본3개에서는 q95가 최대값이다.

| 관계 경로 | 결속한 공개 scale |
|---|---:|
| E_R: 기존 endpoint 대조 재보정 | 0.29153658488143597 |
| C: raw 차분 후 제한 | 0.2650091310071505 |
| R_joint: raw endpoint 공동 제한 | 1.2582928166602765 |

이는 관계 특징의 단위를 정한 값이다. 학습자의 개입 허용량 rho나 기능 loss 계수 eta를 정한 것이 아니다. 공개 mixed3명에서 정했다는 사실을 충분한 보정·최적성·효용 근거로 부르지 않는다.

구성 배열6검사에서는 기존 point 질량/통계, 환자 반복 수, 빈 class/mixed0, P-only scale, raw 차분의 공통 이동 성질, 제한 후 joint 선형 사상, Q-only 교란의 제한 전/후 구분, Torch endpoint gradient를 확인했다. 기존 CPU 회귀6검사도 통과했다.

## 4. 실제 이미지 gradient와 실행량

공개 mixed 환자1명의 고정 양성·음성 이미지에서 단순 선형 probe의 gradient를 확인했다. 실제 point gradient norm은 `6.0917473`, raw 관계 probe gradient norm은 `5.5160184`, 최대 절대값은 `0.4289000`이다. 유한하고0이 아니다. 새 관계 학습 목적·guard·기능 loss를 학습시킨 결과가 아니다.

- 실제 공개 pixel decode815회: P813장 추출과 고정 witness2장. 고유 파일813개.
- 계측 source forward206호출/817image presentations, backward3호출/6image presentations. 두 경로를 위해 전체 cohort를 두 번 인코딩하지 않았다.
- Raw 특징 추출25.686초, 실제 경로·통계·witness 전체31.338초. 구성 배열6검사0.449초, 기존 CPU6검사2.868초. 구현·기록 시간이나 전체128장 합성 처리량은 이 값과 별개다.
- 합성 optimizer update0, classifier solve/학습/성능 평가0, 수신 모델 실행0.
- P-only pixel allow-list를 유지했다. Q/V는 기존 명부 metadata 검증 외 pixel·특징 추출·평가 없음. DP·expert·reserved 접근0.

## 5. 완료 범위와 남은 작업

완료 status는 `PASS_PUBLIC_RAW_POINT_PATHS_AND_PATIENT_STATISTICS_ONLY`다. 1단계의 기존 gradient 수리와 이번2단계의 source 특징 경로 검증은 각각 보존한다.

다음은 이미 정해진3단계, 즉 같은 guarded learner와 기능 정합을 실제 목표·합성 통계·수신 규칙에 연결하는 작업이다. 새 학습 규칙 전체의 실제 모델 one-pass/two-pass 검사는 그 연결 후 수행해야 한다. 수신 encoder의 실제 raw 경로 검증, 전체128장 profile, 본 실험은 이번 완료 범위에 포함하지 않는다.

rho/eta 및30bank는 제안 상태로 남겨두었으며 이번에 선택·동결·실행하지 않았다. 기존 자료 역할, Rwide와 head/LoRA 결과,1단계 실패 원본, expert/reserved 경계와 원격 저장소는 보존했다. 연구 방향을 다시 선택하거나 새 효용 중단 조건을 추가하지 않았다.

재현 명령은 `code_working`에서 실행한다. 출력 경로는 기존 결과를 덮어쓰지 않는 새 경로여야 한다.

```powershell
base_gate/.venv/Scripts/python.exe -B -X utf8 -m prrd_v3.public_feature_paths --output _reports/prrd_step2_paths_replay_new
base_gate/.venv/Scripts/python.exe -B -X utf8 -m prrd_v3.verify_public_feature_paths --directory _reports/prrd_step2_paths_replay_new
base_gate/.venv/Scripts/python.exe -B -X utf8 -m prrd_v3.test_relation_paths
```
