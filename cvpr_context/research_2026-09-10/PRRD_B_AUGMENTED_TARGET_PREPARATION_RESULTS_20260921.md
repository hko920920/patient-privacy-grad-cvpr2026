# B 증강 target 준비 결과 — 2026-09-21

상태: **P/Q 전체 K=4 추출·독립 검산·실제 target 파일 결속 완료.** 큰 연구단계2의 데이터 준비 결과이며 합성자료 전이 성능 결과가 아니다.

## 1. 이번에 완료한 범위

사용자의 이번 요청에 따라 고정한 영상당4회 변환을 P/Q 전체에 적용했다. 검증된 `prrd_v3/augmented_targets.py`와 기존 BioViL point 경로를 사용했다. Production `prrd_v3/*.py` 33개 파일은 이번에 수정하지 않았다. 새 준비 실행기와 독립 검산기는 이번 산출물 디렉터리에만 추가했다.

|자료|기존 환자 수|원영상 수|실제 변환 특징 수|음성/양성 class를 가진 환자 수|
|---|---:|---:|---:|---:|
|P|672|813|3,252|669 / 6|
|Q|2,027|5,097|20,388|2015 / 114|
|합계|2,699|5,910|23,640|2,684 / 120|

같은 환자는 두 class에 모두 포함될 수 있다. 23,640은 변환 특징 수이며 새로운 환자 수가 아니다. Q는 기존 private_train/former-selection이며 기존 선택·Rwide·pilot 사용 이력을 유지한다. 이번 증강 특징/변환 매핑/환자별 통계/target은 비DP 내부 산출물이다.

## 2. 고정 연산과 변하지 않은 입력

- 실제 native 1024×1024 grayscale 픽셀에 기존 범위의 affine/brightness 변환을 적용한 뒤 기존 source 전처리(512 resize,448 crop)와 FP32 BioViL point 특징/PCA16/scale을 사용했다. View microbatch16이다.
- 각 영상의4개 특징 평균을 구하고, 같은 환자·class 내 영상들을 평균한 뒤 class를 가진 환자들을 같은 질량으로 합쳤다. P와Q의 환자 합·count를 더한 뒤 정규화했다.
- 2차 모멘트는 개별 특징 외적들의 평균이다. 평균 특징의 외적으로 대체하지 않았다. 집계는FP64, 저장 특징은FP32다.
- 기존 clean P/Q 특징·PCA/scale·clean B target·목표 readout 기록을 해시로 보존했다. Clean/readout/기능 정합 코드는 그대로이며 이번에 목표 분류기를 새로 선택하지 않았다.
- 이 target은 **실자료의 유한 증강 평균**이다. 개별 변환에 대한 real/synthetic 반응을 일대일 대응시키는 목적이 아니다. K·증강 범위·가중치를 결과에 따라 바꾸지 않았다.

## 3. 독립 검산

저장 특징과 원 manifest를 다시 읽는 별도 CPU 코드가 환자→class→영상→변환 순서로 재계산했다. Production 집계 함수를 재사용하지 않았다. 모든 변환 seed/affine/brightness 매핑도 고정 규칙에서 독립 재생성해 exact 일치를 확인했다.

- 환자 통계·P/Q 합·pooled target 전체의 최대 절대 차이: **7.1054273576e-15**. 사전 기준1e-12 통과.
- 환자별 질량과 pooled class count: 기존 clean B의2,684/120과 동일.
- 실제 production augmented-target loader로 다시 읽은 m/A: exact 일치.
- Frozen encoder 가중치·buffer: 전후 hash 동일. Source backward0, optimizer update0.
- 최초 실패/재시도 없이 이번 전체 준비 실행 완료. 이전 구현 단계의 항등 검사 실패 기록은 원래 보고서에 그대로 남긴다.

이는 저장 특징의 집계·바인딩 검증이며, 독립 encoder 재현이나 의료 효용 검사라는 뜻은 아니다.

## 4. 기존 clean B target과의 차이

|class|평균 벡터 L2 차이|평균 상대 L2 차이|2차 모멘트 Frobenius 차이|2차 모멘트 상대 차이|
|---|---:|---:|---:|---:|
|0|0.028479937|0.353126354|0.00735845798|0.0379974752|
|1|0.0239323447|0.0487441323|0.0161969318|0.0397583921|

상대 차이의 분모는 각 clean target의 동일 norm이다. 두 class의 평균·2차 모멘트 제곱오차 합에1/4을 곱한 gap은 **0.000425087859396**다. 이것은 target 변경량이며 개선량·전이 성공 기준이 아니다. 이 차이를 이용한 계수/변환 선택은 하지 않았다.

## 5. 실제 비용과 저장

- P 추출·저장: 50.397초. Q: 298.524초.
- 전체 추출·저장: **348.921초 (5.82분)**.
- 독립 CPU 검산: 70.455초. 초기 결속·모델 로딩·최종 해시 재확인을 포함한 준비 프로그램: **426.353초 (7.11분)**.
- 코드 연결·보고서 기록까지 이번 작업 경과: 약 11.6분. 최초 안내15–25분은 준비 작업 전체의 예상 범위였다.
- 실제 source image-forward23,640, backward0, optimizer0. Pixel decode P813/Q5097, 새 원영상 SHA 확인5,910. V/recipient/C/D/DP/Expert/Reserved 실행0.
- GPU peak allocated 1.856GiB, reserved 2.604GiB.
- 준비 프로그램 완료 시 새 저장량 9.01MiB, 디스크 여유 12.10GiB. 변환 픽셀은 저장하지 않았다. 기존 파일 삭제 없음.

## 6. 완성 파일과 결속

- Target: `code_working/_reports/prrd_b_aug_targets_20260921_v1/private/B_augmean_k4_target.npz`
- Target SHA256: `538f3fcd3474c6f3b91a3503ce174846636b82a3e9f71aac433423b6219c8054`
- Rule SHA256: `d852cb9e9f2f55a95f915b93d43cf98efa7cc9a18f025e7129525abcc02c92f4`
- Loader 연결 값: 같은 `private/target_handoff.json`의 clean/aug 파일·prefix·hash. 이는 실행 승인 문서가 아니다.
- 내부 source 특징/변환 매핑: `private/P_augmented_point_features.npz`, `private/Q_augmented_point_features.npz` 및 각 chunk.
- 내부 환자 통계: `private/patient_augmented_point_statistics.npz`.
- 실행/검산: `contract.json`, `extraction_receipt.json`, `independent_verification.json`, `target_difference.json`, `result.json`.

## 7. 남은 범위

**목표 파일 준비의 차단 항목은 없다.** 다음 수정 B 한 bank의 job에 이 새 target hash를 결속하고 해당 실행/평가 범위를 정하는 일이 남는다. 이번 요청 범위에는 합성학습·V 평가가 없으므로 실행하지 않았다. 기존seed101/128장/500회와 비교 범위를 확대하지 않으며 C/D·추가seed·DP·final은 자동 재개하지 않는다.

현재 완료는 “새 증강 목표를 실제 데이터로 만들고 검산했다”까지다. 기존 source 재현/전이 음성 결과는 바뀌지 않았고, 수정 B의 유용성은 아직 미평가다.
