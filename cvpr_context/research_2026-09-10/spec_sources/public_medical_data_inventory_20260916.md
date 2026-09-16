# 공개 의료 backbone용 로컬 자료 감사

**판정: 준비 측면에서 좋다. 공개32명 반복학습에 한정할 이유가 없었다.** 기존 public_development의1,816명·5,206개 PA 명부 중4,831장이 로컬에 있다. 현재 역할을 그대로 유지하며 후보를 환자 단위로 제외했다.

fit80, selection40, locked calibration140, locked test140, public quality32, auxiliary reference64, background256 및 기존448 참조에서 현재 역할을 제거한 reference259의 **합집합911명**을 제외했다. 역할끼리 중복되므로 각 수를 단순히 더해 빼지 않았다. 남은905명·1,016장은 모두 로컬 파일이 있다.

데이터 담당이 원 명부·제외·과거 진단 이력을 조사하고 builder를 작성했다. 담당 턴이 도구 사용 한도로 종료되어 root가 builder 실행과 실제 파일 해시 대조, 저장 명부 재구성, 별도 실행계획 검산을 이어서 완료했다. 담당이 완료하지 못한 검산을 수행한 것으로 기록하지 않는다.

## 최종 분할

환자 순서는 `SHA256('public-medical-backbone-20260916-v1|patient|'+patient_id)` 오름차순이다. 조사한 진단명부 이력이 없는 환자에서 confirmation128, selection128을 먼저 분리하고, 남은 전체의 앞640명을 train, 나머지9명을 reserve로 정했다. 이전 단순순서안은 명부 공개나 학습 전 계산안이며, 이력 확인에 따라 성능을 보기 전에 변경한 과정을 JSON에 보존했다.

| 역할 | 환자 | 영상 | 조사한 기존 진단 이력 환자 |
|---|---:|---:|---:|
| 학습 |640|749|117|
| 선택용 예약 |128|128|0|
| 확인용 예약 |128|128|0|
| 예비 |9|11|1|

최종 조사 이력 환자는118명이다. 중간 보고의117명은 K5 fixture까지 합치기 전 값이다. history0는 과거 `selected_*.csv`, 공개 triplet, 코드에 정의된 K5 fixture와 겹치지 않는다는 뜻이며, 그 이미지가 연구 전체에서 한 번도 열리거나 처리되지 않았다는 보장은 아니다. 모든 자료는 이미 취득·전처리 출처 감사 이력이 있다.

학습 영상의 약한 라벨은 No Finding424장, Effusion71장, Cardiomegaly23장이다. 중복 질환 라벨이 가능하며 이 숫자를 수동 진단 정답으로 취급하지 않는다. 후보905명의 영상 수는822명1장·55명2장·28명3장이다. 과거 여러 영상을 가진 환자 중심의 CVPR cohort와 구성 차이가 있다.

## 수행한 검증과 한계

- 원 환자 역할 변경0, 새 영상 다운로드0, 새 GPU/모델 실행0, 새 PNG decode0.
- 후보1,016개 실제 파일의 SHA256을 원 inventory 및 source audit와 대조했다.
- 환자 간 exact byte 중복과 기존 audit의 pixel hash 중복이 없음을 확인했다. 새로운 perceptual near-duplicate 검사는 아니다.
- 저장 patients/images/history CSV를 원 자료에서 다시 구성해 byte 단위로 비교했다. 이 재구성은 같은 builder를 재실행한 검증이며 독립 구현이라고 부르지 않는다.
- root의 별도 검산은 저장 환자/영상 역할, 예정된 학습순서·노출량·생성입력을 독립 계산했다. 품질·통계적 표본 수가 아니라 명세 일관성 검사다.
- 기존2304장 cache의 모든 환자 역할을 제외했으므로 이번749장 학습은 새 public-only VAE cache가 필요하다. 사적 역할이 섞인 옛 cache를 열어 필터링하는 실행을 기본으로 삼지 않는다.
- 이번 첫 형태 확인에서 실제 평가 단위는 새 prompt/seed 생성 입력이다.128명씩의 원본 자료는 후속 held-out/참조 평가를 위한 예약이며, 아직 이 환자들을 평가했다고 말하지 않는다.

[원 명부 및 입력 해시](public_medical_backbone_plan_20260916_v1/exclusion_calculation.json) · [저장 명부 재구성 결과](public_medical_backbone_plan_20260916_v1/verification.json) · [별도 학습·생성 명세 검산](public_medical_backbone_plan_20260916_v1/execution_spec_verification.json).
