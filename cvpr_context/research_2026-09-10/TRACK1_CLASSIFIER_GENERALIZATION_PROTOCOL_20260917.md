# 고정 분류기의 학습자료–개발자료 일반화 진단

2026-09-17. 큰 단계2의 제한된 진단이다. 기존 full64 및 LoRA의 효용 관문 실패를 유지한다. **새 추론 전에 본 명세와 실행 계약을 고정하며, 방법 성공 판정이나 재튜닝은 하지 않는다.**

질문은 저장된 분류기가 기존 평가 방식에서도 이미 학습한 영상을 구별하는가다. 학습자료 성능이 높고 개발 성능만 낮다면 일반화 차이를 확인한다. 학습자료부터 낮다면 학습/추론 입력·정규화·모드 차이 등을 후속 후보로 좁힐 수 있다. 어느 결과도 특정 원인의 인과적 증명은 아니다.

## 고정 모델과 자료

- R1, Dreal, L_public, L_pooled 각각 seed11/23/37의 기존 400-step checkpoint, 총12개.
- 각 checkpoint의 실제 400개 학습 trace에 한 번 이상 등장한 고유 영상만 주 분석에 사용한다. 명부에는 있으나 배치에 등장하지 않은 영상은 학습자료 성능에 포함하지 않는다.
- R1: real/public. Dreal: real/public 및 real/private. 두 LoRA arm: real/public 및 해당 synthetic. 각 source를 별도로 집계하며 혼합 성능으로 대체하지 않는다.
- 실영상은 기존 weak 기흉 label, 합성영상은 요청 label이다. 합성자료에서의 높은 분류 성능은 임상적 질환 정확성을 증명하지 않는다.
- 기존 개발 예측5,047장/2,026명은 저장 파일을 재집계한다. 평가 경로 검사를 위해 저장된 순서의 첫64장만 각 모델에서 다시 추론한다. 이64장은 결과에 따라 고르지 않는다.
- 실제 학습 고유영상1,389장과 개발 경로 검사64장만 픽셀 허용 목록에 넣는다. 전문가 final532명과 reserved4,213명은 접근하지 않는다.

## 추론

동일 ImageNet ResNet18 구조에 checkpoint를 strict load한다. 224 letterbox: grayscale, 종횡비 유지, PIL bilinear, 검은 padding, 3채널 복제, /255, 기존 ImageNet mean/std를 적용한다. 증강을 끄고 FP32, batch32, eval(), inference_mode(), 기존 deterministic 설정을 유지한다.

모든 parameter와 buffer를 실행 전후 비교한다. BatchNorm running_mean, running_var, num_batches_tracked도 포함한다. optimizer, backward, train(), 통계 재보정, checkpoint/threshold 선택을 실행하지 않는다. 기존 source와 산출물도 변경하지 않는다.

개발 첫64장 재추론 logit은 기존 같은 batch32 결과와 절대차1e-5 이내여야 한다. 사전 허용치를 넘으면 기술적 실패로 기록하고 원인을 점검하며 이를 일반화 결과로 해석하지 않는다. 모든 입력은 실제 raw/PNG SHA 및 독립 letterbox 배열과 기존 학습 pixel SHA를 결속한다. 구형 data/train import는 차단한다.

## 집계

각 모델·source에 대해 실제 본 고유영상의 영상별 AUROC, AP, 비가중 BCE를 계산한다. 양/음성별 BCE와 같은 비중의 평균 BCE, label별 logit 분위수, 양성 비율, 고유 환자/영상 수를 함께 저장한다. 합성 cell은 실제 환자로 세지 않는다.

추가 기술 통계는 실제 학습 노출 횟수로 가중한 BCE와 노출 횟수 분포다. 학습 마지막50step의 평균 BCE, 첫/마지막50step의 source별 BCE를 기존 로그에서 계산한다. 이는 당시 가중치·증강·train mode·balanced sampling에서 측정된 값이므로 현재 고유영상 eval BCE와 같아야 하는 수치가 아니다.

개발 AUROC/AP/BCE는 기존 raw logit에서 재계산한다. seed별 결과를 먼저 계산하고 3seed 평균을 보고한다. 학습 AP와 개발 AP는 양성 비율이 다르므로 직접적인 향상률로 해석하지 않는다. 공개 학습 양성은6장뿐이므로 높은 학습 AUROC 자체의 일반화 근거도 약하다.

새 검정·threshold·성공 gate·bootstrap·모델 선택은 추가하지 않는다. 이 진단은 고정 checkpoint의 동작을 기술하며 추가 효용의 독립 확인이 아니다.

## 실행과 검산

새 inference 수는 trace로 확정한 각 모델의 고유 학습영상 수의 합 및64×12 경로 검사로 제한한다. 원본 파일 decode는 허용 목록1,453개만 한다. 새 생성0, optimizer update0, DP0.

별도 검산기는 생산 metric 함수를 import하지 않고 pairwise AUROC, tie-aware AP, 안정적인 BCE와 trace 노출 수를 재계산한다. source, patient/image/label 대응, 전체 state 및 BatchNorm 불변, 기존 예측 경로 대응, 과거 실패 결과 불변을 확인한다.

완료 후 관측된 일반화 차이와 미확인 원인을 구분해 기록한다. 자동 재학습, BatchNorm 재보정, 새 adapter, DP, final 개방으로 이어가지 않는다. 필요한 후속 대조는 결과의 범위 안에서 별도 결정한다.
