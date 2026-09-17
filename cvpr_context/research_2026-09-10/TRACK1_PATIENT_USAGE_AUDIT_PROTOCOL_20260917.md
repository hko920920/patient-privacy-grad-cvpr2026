# 실제 환자 사용 이력 감사 명세

2026-09-17 KST. 큰 단계2·방향1. 예상20–30분. 기존 자료 역할 이름과 실제 학습·선택·평가 소비를 구분해서 별도 CVPR 확인자료가 가능한지 판단한다. 새 모델 추론·생성·학습·DP는 하지 않는다.

## 판단 규칙

- A: 실제 optimizer/head fitting/backbone 학습에 사용. 최종 생성 reference에서 제외한다. 버린 dry-run 모델의 학습도 별도 표시해 보수적으로 제외한다.
- B: 개발 loss, 조건 가설, 모델·threshold 선택에 사용. 최종 확인에서 제외한다.
- C: evaluator preflight, metric/reference, 공격·시각/모델 진단에 사용. 목적을 명시한 evaluator 개발에는 재사용할 수 있지만 새 최종 확인으로 세지 않는다.
- D: 역할과 metadata/file-intake 기록만 존재하고 조사한 실행 기록에서 모델·결과 사용을 찾지 못함. 새 연구 분할의 후보이며 절대적인 미사용 증명으로 표현하지 않는다.
- E: original thesis final/test/census, CVPR locked calibration/test. 기존 역할을 보존하고 후보에서 제외한다.
- U: 모델 소비 가능성이 있는데 기록이 부족해 환자별 판정이 불가능함. 후보로 통과시키지 않는다. 복수사용 flag를 보존하고 표의 단일등급은 E→A→B→C→U→D 순서로 요약한다.

Metadata 집계·공식 분할 생성·파일 hash/decode만 수행한 것은 모델 결과 사용과 구분한다. 조건 후보의 발견에 실제 사용한 public32/private80/development40 및 공개 backbone의 label 맥락은 가설 이력으로 표시한다. 전체 source population의 역할별 metadata 수를 센 것만으로 전 환자의 모델 결과를 소비했다고 간주하지 않는다.

## 확인할 증거

K5 private dry-run의 실제 patient/image runtime log와 공개 report hash, full-run gate와 실행 폴더, 공개 smoke/resume fixture, M1/M2 학습 명부·실행/coverage, backbone trace, head extraction/fit 입력, 개발 loss 및 reference, 이전 selected diagnostics, 공격 결과 파일을 대조한다. 단순 planned manifest나 population size를 실제 학습 인원으로 세지 않는다. 광범위한 report JSON/CSV/JSONL 검색은 추가 사용 가능성 누락 탐지용으로 수행하며, 의미가 불명확한 hit는 보수적으로 표시한다.

각 환자에 original_thesis_role, actual training/head/loss/reference/evaluator/visual/selection flags, locked 여부, 조건별 실제 보유영상 수, 개발/확인 후보 여부, 근거 파일 목록을 기록한다. Original private_train 후보를 우선 조사하며 original privacy_attack_holdout은 자동 전환하지 않는다. 기존 frozen 명부·모델·원본·192장 실패 판정은 바꾸지 않는다.

## 자료 구성 가능성만 사전 계산

충분한 D 환자가 있으면 그 전체 후보를 outcome-independent SHA256 순서로 둘로 균등 분할한 **가상 target-development/reference와 별도 confirmation 후보 명부**를 저장한다. Salt는 `cvpr-target-usage-audit-20260917-v1`, 키는 `SHA256(salt+'|patient|'+numeric_patient_id)`이다. 원래 patient 역할을 수정하거나 이 명부를 실행 승인으로 해석하지 않는다. 새 private train은 이번에 만들지 않는다.

환자당 조건별 영상은 같은 salt에 image ID를 넣은 hash가 가장 작은 해당조건 영상1장으로 고정 가능성을 계산한다. Multi-label 환자는 같은 split 안에서 여러 조건에 나타날 수 있으며 조건 간 독립표본으로 세지 않는다. 확인 split은 이후 evaluator 선택/threshold/tuning에 사용할 수 없다. 정상은 `No Finding` 단독label, 흉수 control은 두 target label과 겹치지 않는 영상으로 정의한다. 조건당20명은 비교 규모 참고선이며 검정력 보장이 아니다.

평가기 검증은 기존 소비된 공개자료를 별도 후보로 조사하되, 잠긴 환자는 제외하고 새 target-development/confirmation과 환자 분리를 유지한다. 이번에는 classifier를 로드하거나 threshold를 정하지 않는다.

자료·사용 이력이 충분하면 다음은 평가기 validation과 targeted 비DP 실행 명세다. 부족/불명확하면 외부 target-domain 구성 또는 현방향 보류를 명시한다. 어느 경우에도 patient-DP로 자동 진입하지 않는다.
