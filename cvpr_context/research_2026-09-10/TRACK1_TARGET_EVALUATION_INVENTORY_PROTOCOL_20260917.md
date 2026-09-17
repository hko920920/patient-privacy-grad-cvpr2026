# 폐기종·기흉 평가자료 inventory — 실행 범위

2026-09-17 KST. 큰 단계2·방향1. 예상15–25분. 자료·평가기의 준비 여부만 확인하며 새 생성, 모델 추론, head 학습, DP는 실행하지 않는다.

기존192장에서는 pooled의 추가 생성 효용이 없었다. 이후 저장 통계에서 나온 폐기종·기흉 후보는 사후 가설이며, 기존 실패를 취소하지 않는다. 이번 질문은 **그 가설을 새 환자 reference와 조건 민감 평가로 공정하게 시험할 수 있는가**이다.

## 자료 경계

- 전체 local NIH content inventory와 공식 metadata/split을 대조한다. patient ID로 교집합을 계산하며 영상 수와 환자 수를 구분한다.
- 현재 public backbone640, head public32/private80/development40, 모든 기존 CVPR evaluation/auxiliary 역할, 과거 생성 reference40 및 evaluator preflight448, 기록된 진단 환자를 각각 구분한다. M1/M2 훈련 명부 포함 여부도 확인한다.
- 공개 backbone의 selection/confirmation/reserve는 미사용 여부와 기존 예약 역할을 별도 표시한다. 숫자가 부족하다는 이유로 자동 재배정하지 않는다.
- original private_train, privacy_attack_holdout, final_test와 official test census, CVPR locked calibration/test는 이번 개발 reference로 전환하지 않는다. 이들의 이미지/모델 점수는 열지 않는다.
- 아직 받지 않은 official train_val의 PA metadata도 현재 모든 배정 환자를 제외해 수만 조사한다. metadata 존재와 local file 확보를 구별한다.
- 현재 학습·CVPR 역할을 제외한 public 후보 파일의 존재·bytes·SHA256·PNG decode와 기존 exact duplicate 기록을 확인한다. 임상 판독/질환 유효성 검증은 아니다.
- Emphysema/Pneumothorax를 둘 다 유지하고 No Finding/Effusion을 대조 조건으로 계산한다. multi-label overlap과 전환 가능한지의 제약을 함께 보고한다. 결과가 유리한 조건만 남기지 않는다.
- 조건당20명은 이전 파일럿과 규모를 맞추는 참고선이며 충분한 검정력의 기준이 아니다. 새 reference나 성공 gate는 이번에 채택하지 않는다.

## 평가기 경계

기존 RAD-DINO/BioViL-T와 로컬 질환 분류기의 설치·checkpoint·유효 출력 라벨·학습 데이터 출처·전처리 규칙을 확인한다. 출력18개 중 해당 라벨 학습이 없는 분류기의 빈 slot을 유효 질환 점수로 사용하지 않는다. 실제 해당 조건에서의 discrimination 검증이 없으면 ready가 아닌 candidate로 표시한다. Raw text cosine만으로 성공 판정하지 않는다.

## 산출물과 결정

source hash에 결속한 inventory, 평가기 readiness, 별도 재집계 검산, 결과 MD/HTML 및 상태 갱신. 기존 생성물·실패 판정·역할 명부 불변을 확인한다. 자료가 부족하면 현재 데이터에서 targeted generation과 DP 확대를 보류하고 구체적인 데이터/설정 변경 필요성을 남긴다. 자료가 충분해도 fresh latent와 네 방법(backbone/public/private-only/pooled), 두 target 및 두 control을 포함하는 새 사전 명세와 평가기 점검이 먼저다.
