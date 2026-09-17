# 수정 classifier 7군 one-step 연결 재검증

2026-09-17 · 큰 단계2/방향1. 사용자 검토에 따라 별도14update를 실행한다. 예상10–15분. 이전100update와 구분하고 실패98회는 계속 무효로 보존한다.

## 실행 전 고정

- 새 생성 없이 기존28장 profile bank, 검증된 real224 cache와 manifest를 사용한다. Expert532명과 reserved4213명 pixel·prediction은 접근하지 않는다.
- R0/R1은 public32, S0–S3은 public16+해당 synthetic16, Dreal은 public16+private16. 각 절반8positive/8negative, 총32장이다.
- 같은 ImageNet ResNet18 초기값·seed11, FP32, AdamW lr1e-4/weight_decay1e-4, 새 optimizer state에서 각 arm1step씩2회, 총14update만 실행한다. 새 학습률·증강·seed·방법 선택은 하지 않는다.
- 구형 `run.py classifier`는 즉시 실패시킨다. 과거 실행 source는 변경 전에 hash별 사본으로 보존한다. `run_v2.py`만 사용하고 구형 `downstream_utility.data`·`downstream_utility.train` import는 실행 전후 검사와 import hook으로 차단한다. Data_v2는 공용 전처리 함수까지 독립 보유해 구형 data 모듈을 import하지 않는다.
- 새 계약에 corrected runner·data_v2·train_v2·독립 verifier의 SHA와 실제 module file,6개 namespace, 자료·모델 SHA를 결속한다.

## 통과 조건

모든 arm과 반복에서 초기 state SHA·fresh AdamW·고정 lr/weight decay·batch/class balance·finite loss/gradient·실제 trainable parameter 변화가 확인돼야 한다. 역할명뿐 아니라 실제 raw/PNG SHA, decoded224 pixel SHA, array index, image/patient/cell ID, label, generated method를 결속한다. 선택된 raw만 다시 열고 final/reserved는 제외한다.

앞 real16은 모든 arm에서 같고 R0/R1의32개 ID는 같아야 한다. R0에는 증강을 하지 않으며 다른 arm의 공통 real half에는 같은 augmentation을 적용한다. S0–S3은 같은 synthetic cell과 요청 label을 사용하되 실제 pixel은 method와 일치해야 한다.

각 arm의 두 반복은 선택 ID·역할·label·실제 전처리 tensor·logit·loss·최종 state가 exact여야 한다. 독립 검산기는 생산 data/train 함수를 import하지 않고 raw decode→letterbox→공식 torchvision GPU affine/정규화를 재계산해 실제 저장 입력과 exact 대조한다. 표준 torchvision primitive는 공유하지만 생산 wrapper는 공유하지 않는다. BCE는 별도 float64 logaddexp 계산, 고정 atol2e-6이다. 결과를 본 뒤 tolerance를 넓히지 않는다.

첫step 총시간·preprocess·forward/backward·optimizer·allocated/reserved VRAM을 분리한다. cold/warm 순서 영향이 있으므로1step 값으로 장기 학습 시간이나 arm 비용 우위를 확정하지 않는다.

통과의 이름은 `corrected all-arm one-step integration replay`다.400/800step 수렴·장기 AdamW 거동·AUROC/AP·private 효용·DP·CVPR 기여까지 검증한 것이 아니다. 통과하면 본512장/21run 개발 실험의 실행 기반으로 사용할 수 있다. 본실험 시작 후 각 arm 첫10–20step에서 ETA를 다시 계산한다. Expert final은 비DP·DP와 전체 분석 동결 이후에만 사용한다.
